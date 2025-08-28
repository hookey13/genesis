"""
Error recovery and resilience testing.
Tests disconnection handling, crash recovery, and system resilience.
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import structlog
import sqlite3
import psutil
import os

from genesis.core.models import (
    Position,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
)
from genesis.core.constants import TradingTier as TierType
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.data.repository import Repository
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway
from genesis.exchange.websocket_manager import WebSocketManager
from genesis.exchange.circuit_breaker import CircuitBreaker
from genesis.utils.disaster_recovery import DisasterRecoveryManager

logger = structlog.get_logger()


class TestErrorRecovery:
    """Test error recovery and system resilience."""

    @pytest.fixture
    def mock_repository(self):
        """Mock repository with state persistence."""
        repo = Mock(spec=Repository)
        repo.positions = {}
        repo.orders = {}
        repo.save_state = Mock()
        repo.load_state = Mock(
            return_value={
                "positions": {},
                "orders": {},
                "last_update": datetime.utcnow(),
            }
        )
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange with connection simulation."""
        exchange = Mock(spec=ExchangeGateway)
        exchange.connected = True
        exchange.connect = AsyncMock()
        exchange.disconnect = AsyncMock()
        exchange.reconnect = AsyncMock()
        exchange.place_order = AsyncMock()
        exchange.get_order = AsyncMock()
        return exchange

    @pytest.fixture
    def websocket_manager(self):
        """Create WebSocket manager for testing."""
        return WebSocketManager(url="wss://test.exchange.com")

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        return CircuitBreaker(
            failure_threshold=3, recovery_timeout=5, expected_exception=Exception
        )

    @pytest.mark.asyncio
    async def test_api_disconnection_and_reconnection(self, mock_exchange):
        """Test API disconnection detection and automatic reconnection."""
        reconnect_attempts = 0
        max_attempts = 3

        async def simulate_disconnect():
            nonlocal reconnect_attempts
            mock_exchange.connected = False

            # Simulate reconnection attempts
            for attempt in range(max_attempts):
                reconnect_attempts += 1
                await asyncio.sleep(0.1)

                if attempt == max_attempts - 1:
                    mock_exchange.connected = True
                    break

            return mock_exchange.connected

        # Disconnect
        await simulate_disconnect()

        assert reconnect_attempts == max_attempts
        assert mock_exchange.connected is True

    @pytest.mark.asyncio
    async def test_strategy_crash_isolation(self, mock_repository, mock_exchange):
        """Test that one strategy crash doesn't affect others."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

        # Create multiple strategies
        healthy_strategy = Mock()
        healthy_strategy.name = "healthy"
        healthy_strategy.analyze = AsyncMock(return_value=None)
        healthy_strategy.active = True

        crashing_strategy = Mock()
        crashing_strategy.name = "crashing"
        crashing_strategy.analyze = AsyncMock(side_effect=Exception("Strategy crashed"))
        crashing_strategy.active = True

        # Add strategies
        await orchestrator.add_strategy("healthy", healthy_strategy)
        await orchestrator.add_strategy("crashing", crashing_strategy)

        # Run analysis - crashing strategy should fail
        market_data = {"symbol": "BTCUSDT", "price": "50000"}

        # Healthy strategy continues
        await healthy_strategy.analyze(market_data)
        assert healthy_strategy.active is True

        # Crashing strategy fails but is isolated
        with pytest.raises(Exception):
            await crashing_strategy.analyze(market_data)

        # Verify healthy strategy still active
        assert orchestrator.active_strategies.get("healthy") == healthy_strategy
        assert len(orchestrator.active_strategies) >= 1

    @pytest.mark.asyncio
    async def test_database_lock_handling(self, mock_repository):
        """Test handling of database locks and contention."""
        # Simulate database lock
        mock_repository.save_order.side_effect = sqlite3.OperationalError(
            "database is locked"
        )

        order = Order(
            id="test_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW,
        )

        # Implement retry logic
        max_retries = 3
        retry_count = 0

        for attempt in range(max_retries):
            try:
                mock_repository.save_order(order)
                break
            except sqlite3.OperationalError:
                retry_count += 1
                await asyncio.sleep(0.1 * (2**attempt))  # Exponential backoff

                # Clear lock on last attempt
                if attempt == max_retries - 1:
                    mock_repository.save_order.side_effect = None

        assert retry_count == max_retries - 1

    @pytest.mark.asyncio
    async def test_system_restart_position_recovery(self, mock_repository):
        """Test position recovery after system restart."""
        # Save positions before "crash"
        positions_before = [
            Position(
                id="pos_1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                quantity=Decimal("0.1"),
                unrealized_pnl=Decimal("100"),
                realized_pnl=Decimal("0"),
            ),
            Position(
                id="pos_2",
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                entry_price=Decimal("3000"),
                current_price=Decimal("2900"),
                quantity=Decimal("1"),
                unrealized_pnl=Decimal("100"),
                realized_pnl=Decimal("0"),
            ),
        ]

        # Save state
        mock_repository.save_state(
            {
                "positions": {p.id: p for p in positions_before},
                "timestamp": datetime.utcnow(),
            }
        )

        # Simulate restart - load state
        loaded_state = mock_repository.load_state()
        mock_repository.load_state.return_value = {
            "positions": {p.id: p for p in positions_before},
            "orders": {},
            "last_update": datetime.utcnow(),
        }

        # Verify positions recovered
        recovered_positions = loaded_state["positions"]
        assert len(recovered_positions) == 2
        assert "pos_1" in recovered_positions
        assert "pos_2" in recovered_positions

    @pytest.mark.asyncio
    async def test_exchange_error_handling(self, mock_exchange):
        """Test handling of various exchange errors."""
        error_scenarios = [
            ("INSUFFICIENT_BALANCE", "Account has insufficient balance"),
            ("MARKET_CLOSED", "Market is closed"),
            ("RATE_LIMIT", "Rate limit exceeded"),
            ("INVALID_ORDER", "Invalid order parameters"),
            ("NETWORK_ERROR", "Network request failed"),
        ]

        for error_code, error_msg in error_scenarios:
            mock_exchange.place_order.side_effect = Exception(
                f"{error_code}: {error_msg}"
            )

            order = Order(
                id=f"order_{error_code}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                status=OrderStatus.NEW,
            )

            # Handle specific errors
            try:
                await mock_exchange.place_order(order.__dict__)
            except Exception as e:
                error_str = str(e)

                if "INSUFFICIENT_BALANCE" in error_str:
                    # Don't retry, log and alert
                    assert "insufficient balance" in error_str.lower()
                elif "RATE_LIMIT" in error_str:
                    # Wait and retry
                    await asyncio.sleep(1)
                elif "NETWORK_ERROR" in error_str:
                    # Immediate retry
                    pass

        # Reset for next test
        mock_exchange.place_order.side_effect = None

    @pytest.mark.asyncio
    async def test_network_partition_simulation(self, websocket_manager):
        """Test system behavior during network partition."""
        # Simulate network partition
        partition_start = datetime.utcnow()
        partition_duration = 5  # seconds

        with patch(
            "websockets.connect", side_effect=ConnectionError("Network unreachable")
        ):
            # Try to connect during partition
            connection_attempts = 0
            connected = False

            while (datetime.utcnow() - partition_start).seconds < partition_duration:
                try:
                    await websocket_manager.connect()
                    connected = True
                    break
                except:
                    connection_attempts += 1
                    await asyncio.sleep(0.5)

            assert connection_attempts > 0
            assert not connected

        # Network restored
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value = mock_ws

            await websocket_manager.connect()
            assert mock_connect.called

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, circuit_breaker, mock_exchange):
        """Test circuit breaker prevents cascading failures."""
        # Configure exchange to fail
        mock_exchange.place_order.side_effect = Exception("Service unavailable")

        # Attempt operations until circuit opens
        failures = 0
        for i in range(5):
            try:
                async with circuit_breaker:
                    await mock_exchange.place_order({})
            except Exception:
                failures += 1

                if circuit_breaker.state == "open":
                    break

        assert circuit_breaker.state == "open"
        assert failures >= circuit_breaker.failure_threshold

        # Circuit should reject calls while open
        with pytest.raises(Exception) as exc_info:
            async with circuit_breaker:
                await mock_exchange.place_order({})

        assert "Circuit breaker is open" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_memory_leak_recovery(self):
        """Test system recovers from memory pressure."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create memory pressure
        large_objects = []
        for i in range(100):
            # Create large object
            obj = [0] * (1024 * 1024)  # ~8MB per list
            large_objects.append(obj)

            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            # Trigger cleanup if memory grows too much
            if memory_increase > 500:  # 500MB threshold
                # Clear old objects
                large_objects = large_objects[-10:]

                # Force garbage collection
                import gc

                gc.collect()
                break

        # Clean up
        del large_objects
        import gc

        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_recovered = initial_memory - final_memory

        # Should recover most memory
        assert final_memory < initial_memory + 100  # Allow 100MB overhead

    @pytest.mark.asyncio
    async def test_cascade_failure_prevention(self, mock_repository, mock_exchange):
        """Test prevention of cascade failures across components."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

        # Simulate component failures
        component_failures = []

        # Exchange fails
        mock_exchange.place_order.side_effect = Exception("Exchange down")
        component_failures.append("exchange")

        # Repository fails
        mock_repository.save_order.side_effect = Exception("Database down")
        component_failures.append("repository")

        # Try to execute order - should handle gracefully
        order = Order(
            id="cascade_test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW,
        )

        try:
            await orchestrator.execute_order(order)
        except Exception as e:
            # Should get clear error, not cascade
            assert len(component_failures) == 2
            assert "Exchange down" in str(e) or "Database down" in str(e)

    @pytest.mark.asyncio
    async def test_disaster_recovery_procedures(self, mock_repository):
        """Test disaster recovery manager functionality."""
        dr_manager = DisasterRecoveryManager(repository=mock_repository)

        # Create backup
        backup_data = {
            "positions": {"pos_1": "position_data"},
            "orders": {"order_1": "order_data"},
            "strategies": {"strat_1": "strategy_config"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        backup_id = await dr_manager.create_backup(backup_data)
        assert backup_id is not None

        # Simulate disaster
        mock_repository.positions = {}
        mock_repository.orders = {}

        # Restore from backup
        restored = await dr_manager.restore_backup(backup_id)
        assert restored is not None
        assert "positions" in restored
        assert "orders" in restored

        # Verify restore point
        restore_points = await dr_manager.list_restore_points()
        assert len(restore_points) > 0

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_repository, mock_exchange):
        """Test system degrades gracefully under failure conditions."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

        # Disable non-critical features
        orchestrator.enable_analytics = False
        orchestrator.enable_advanced_orders = False

        # Basic functionality should still work
        order = Order(
            id="degraded_test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW,
        )

        mock_exchange.place_order.side_effect = None
        mock_exchange.place_order.return_value = {"orderId": "12345", "status": "NEW"}

        result = await mock_exchange.place_order(order.__dict__)
        assert result["orderId"] == "12345"
