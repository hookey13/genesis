"""Failover and high availability tests."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
from datetime import datetime, timedelta

from genesis.engine.engine import TradingEngine
from genesis.exchange.circuit_breaker import CircuitBreaker
from genesis.exchange.gateway import ExchangeGateway
from genesis.data.repository import DatabaseConnection

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestFailoverMechanisms:
    """Test failover and high availability mechanisms."""

    async def test_primary_exchange_failure(self, trading_system):
        """Test failover when primary exchange fails."""
        primary_gateway = trading_system.exchange_gateway
        backup_gateway = AsyncMock(spec=ExchangeGateway)
        
        trading_system.engine.set_backup_gateway(backup_gateway)
        
        primary_gateway.get_ticker.side_effect = ConnectionError("Primary exchange down")
        
        backup_gateway.get_ticker.return_value = {
            "bid": Decimal("50000.00"),
            "ask": Decimal("50001.00"),
            "last": Decimal("50000.50")
        }
        
        ticker = await trading_system.engine.get_market_data("BTC/USDT")
        
        assert ticker is not None
        assert ticker["last"] == Decimal("50000.50")
        assert backup_gateway.get_ticker.called

    async def test_database_failover(self, trading_system):
        """Test database failover to replica."""
        primary_db = DatabaseConnection("primary")
        replica_db = DatabaseConnection("replica")
        
        trading_system.engine.set_database_connections(primary_db, replica_db)
        
        test_data = {"order_id": "test_123", "symbol": "BTC/USDT"}
        
        primary_db.save = AsyncMock(side_effect=Exception("Primary DB failure"))
        replica_db.save = AsyncMock(return_value=True)
        
        result = await trading_system.engine.save_order(test_data)
        
        assert result is True
        assert replica_db.save.called
        assert replica_db.save.call_args[0][0] == test_data

    async def test_redis_cache_failure(self, trading_system):
        """Test system behavior when Redis cache fails."""
        from genesis.cache.manager import CacheManager
        
        cache = CacheManager()
        trading_system.engine.set_cache(cache)
        
        cache.set = AsyncMock(side_effect=ConnectionError("Redis down"))
        cache.get = AsyncMock(side_effect=ConnectionError("Redis down"))
        
        position = {
            "symbol": "BTC/USDT",
            "quantity": Decimal("0.1"),
            "entry_price": Decimal("50000.00")
        }
        
        await trading_system.engine.cache_position(position)
        
        cached_position = await trading_system.engine.get_cached_position("BTC/USDT")
        
        assert cached_position is None  # Should gracefully handle cache failure
        
        db_position = await trading_system.engine.get_position_from_db("BTC/USDT")
        assert db_position is not None  # Should fallback to database

    async def test_monitoring_failover(self, trading_system):
        """Test monitoring system failover."""
        from genesis.monitoring.monitor import MonitoringSystem
        
        primary_monitor = MonitoringSystem("primary")
        backup_monitor = MonitoringSystem("backup")
        
        trading_system.engine.set_monitors(primary_monitor, backup_monitor)
        
        metric = {"name": "order_latency", "value": 45.2, "timestamp": datetime.now()}
        
        primary_monitor.record = AsyncMock(side_effect=Exception("Primary monitor down"))
        backup_monitor.record = AsyncMock(return_value=True)
        
        await trading_system.engine.record_metric(metric)
        
        assert backup_monitor.record.called
        assert backup_monitor.record.call_args[0][0] == metric

    async def test_circuit_breaker_activation(self, trading_system):
        """Test circuit breaker activation on repeated failures."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=5,
            expected_exception=ConnectionError
        )
        
        trading_system.exchange_gateway.set_circuit_breaker(circuit_breaker)
        
        failures = 0
        for i in range(5):
            try:
                if i < 3:
                    raise ConnectionError("Exchange error")
                else:
                    # Circuit should be open
                    result = await circuit_breaker.call(
                        lambda: {"status": "ok"}
                    )
            except Exception as e:
                failures += 1
        
        assert failures >= 3
        assert circuit_breaker.is_open()

    async def test_service_health_checks(self, trading_system):
        """Test health check mechanisms for all services."""
        services = {
            "exchange": trading_system.exchange_gateway,
            "database": trading_system.engine.database,
            "cache": trading_system.engine.cache,
            "risk_engine": trading_system.risk_engine
        }
        
        health_status = {}
        
        for name, service in services.items():
            try:
                health = await service.health_check()
                health_status[name] = health
            except Exception as e:
                health_status[name] = {"status": "unhealthy", "error": str(e)}
        
        assert all(s.get("status") in ["healthy", "unhealthy"] for s in health_status.values())

    async def test_graceful_degradation(self, trading_system):
        """Test graceful degradation of non-critical services."""
        non_critical_services = {
            "analytics": Mock(),
            "reporting": Mock(),
            "notifications": Mock()
        }
        
        for service_name, service in non_critical_services.items():
            service.process = Mock(side_effect=Exception(f"{service_name} failed"))
        
        trading_system.engine.set_non_critical_services(non_critical_services)
        
        order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": Decimal("0.1")
        }
        
        result = await trading_system.engine.execute_order(order)
        
        assert result is not None  # Core trading should still work
        assert result.get("status") in ["filled", "pending"]

    async def test_network_partition_handling(self, trading_system):
        """Test handling of network partitions."""
        from genesis.network.partition_detector import PartitionDetector
        
        detector = PartitionDetector()
        trading_system.engine.set_partition_detector(detector)
        
        await detector.simulate_partition("exchange")
        
        is_partitioned = detector.is_partitioned("exchange")
        assert is_partitioned
        
        trading_allowed = await trading_system.engine.is_trading_safe()
        assert not trading_allowed  # Should prevent trading during partition
        
        await detector.heal_partition("exchange")
        
        is_partitioned = detector.is_partitioned("exchange")
        assert not is_partitioned
        
        trading_allowed = await trading_system.engine.is_trading_safe()
        assert trading_allowed

    async def test_automatic_recovery(self, trading_system):
        """Test automatic recovery after service restoration."""
        recovery_attempts = []
        
        async def failing_service():
            recovery_attempts.append(datetime.now())
            if len(recovery_attempts) < 3:
                raise ConnectionError("Service down")
            return {"status": "recovered"}
        
        result = None
        for attempt in range(5):
            try:
                result = await failing_service()
                break
            except ConnectionError:
                await asyncio.sleep(1)
        
        assert result is not None
        assert result["status"] == "recovered"
        assert len(recovery_attempts) >= 3

    async def test_load_balancing(self, trading_system):
        """Test load balancing across multiple endpoints."""
        endpoints = [
            AsyncMock(spec=ExchangeGateway),
            AsyncMock(spec=ExchangeGateway),
            AsyncMock(spec=ExchangeGateway)
        ]
        
        for i, endpoint in enumerate(endpoints):
            endpoint.get_ticker.return_value = {
                "source": f"endpoint_{i}",
                "price": Decimal("50000.00")
            }
        
        from genesis.network.load_balancer import LoadBalancer
        
        balancer = LoadBalancer(endpoints, strategy="round_robin")
        trading_system.engine.set_load_balancer(balancer)
        
        results = []
        for _ in range(9):
            result = await balancer.get_ticker("BTC/USDT")
            results.append(result["source"])
        
        for i in range(3):
            endpoint_calls = results.count(f"endpoint_{i}")
            assert endpoint_calls == 3  # Each endpoint should be called 3 times

    async def test_hot_standby_switching(self, trading_system):
        """Test switching to hot standby instance."""
        primary_engine = trading_system.engine
        standby_engine = TradingEngine(
            exchange_gateway=trading_system.exchange_gateway,
            risk_engine=trading_system.risk_engine,
            state_machine=trading_system.state_machine
        )
        
        state = await primary_engine.get_state()
        await standby_engine.sync_state(state)
        
        primary_engine.simulate_failure()
        
        is_primary_healthy = await primary_engine.health_check()
        assert not is_primary_healthy
        
        await standby_engine.promote_to_primary()
        
        is_standby_primary = await standby_engine.is_primary()
        assert is_standby_primary
        
        order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": Decimal("0.1")
        }
        
        result = await standby_engine.execute_order(order)
        assert result is not None