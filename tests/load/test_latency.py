"""Latency testing suite for critical paths."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
import time
from statistics import mean, median, stdev
from datetime import datetime

from genesis.engine.engine import TradingEngine
from genesis.core.models import Order, OrderType, OrderSide

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestLatencyMetrics:
    """Test latency across critical system paths."""

    async def test_signal_to_execution_latency(self, trading_system):
        """Test latency from signal generation to order execution."""
        latencies = []
        
        for _ in range(100):
            signal_time = time.perf_counter_ns()
            
            signal = {
                "strategy": "test_strategy",
                "symbol": "BTC/USDT",
                "side": OrderSide.BUY,
                "quantity": Decimal("0.001"),
                "signal_strength": Decimal("0.8"),
                "timestamp": signal_time
            }
            
            order_result = await trading_system.engine.process_signal(signal)
            
            execution_time = time.perf_counter_ns()
            latency_ms = (execution_time - signal_time) / 1_000_000
            latencies.append(latency_ms)
        
        metrics = {
            "mean": mean(latencies),
            "median": median(latencies),
            "stdev": stdev(latencies) if len(latencies) > 1 else 0,
            "min": min(latencies),
            "max": max(latencies),
            "p50": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99": sorted(latencies)[int(len(latencies) * 0.99)]
        }
        
        assert metrics["mean"] < 50, f"Mean latency {metrics['mean']:.2f}ms exceeds 50ms"
        assert metrics["p95"] < 100, f"P95 latency {metrics['p95']:.2f}ms exceeds 100ms"
        assert metrics["p99"] < 150, f"P99 latency {metrics['p99']:.2f}ms exceeds 150ms"
        
        logger.info(f"Signal to execution latency: {metrics}")

    async def test_market_data_processing_latency(self, trading_system):
        """Test latency of market data processing pipeline."""
        latencies = []
        
        for i in range(1000):
            market_data = {
                "symbol": "BTC/USDT",
                "bid": Decimal("50000") + Decimal(i % 100),
                "ask": Decimal("50001") + Decimal(i % 100),
                "timestamp": time.perf_counter_ns()
            }
            
            start = time.perf_counter_ns()
            await trading_system.engine.process_market_data({"BTC/USDT": market_data})
            end = time.perf_counter_ns()
            
            latency_us = (end - start) / 1000  # microseconds
            latencies.append(latency_us)
        
        avg_latency_us = mean(latencies)
        max_latency_us = max(latencies)
        
        assert avg_latency_us < 5000, f"Avg latency {avg_latency_us:.2f}μs exceeds 5ms"
        assert max_latency_us < 10000, f"Max latency {max_latency_us:.2f}μs exceeds 10ms"
        
        logger.info(f"Market data processing - Avg: {avg_latency_us:.2f}μs, Max: {max_latency_us:.2f}μs")

    async def test_risk_validation_latency(self, trading_system):
        """Test latency of risk engine validation."""
        latencies = []
        
        for i in range(500):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001") * (i % 10 + 1)
            )
            
            start = time.perf_counter_ns()
            is_valid = await trading_system.risk_engine.validate_order(order)
            end = time.perf_counter_ns()
            
            latency_us = (end - start) / 1000
            latencies.append(latency_us)
        
        avg_latency = mean(latencies)
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        assert avg_latency < 1000, f"Avg validation latency {avg_latency:.2f}μs exceeds 1ms"
        assert p99_latency < 5000, f"P99 validation latency {p99_latency:.2f}μs exceeds 5ms"
        
        logger.info(f"Risk validation - Avg: {avg_latency:.2f}μs, P99: {p99_latency:.2f}μs")

    async def test_database_write_latency(self, trading_system):
        """Test database write operation latency."""
        from genesis.data.repository import TradeRepository
        
        repo = TradeRepository()
        latencies = []
        
        for i in range(100):
            trade = {
                "order_id": f"perf_test_{i}",
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": Decimal("0.001"),
                "price": Decimal("50000.00"),
                "timestamp": datetime.now()
            }
            
            start = time.perf_counter_ns()
            await repo.save_trade(trade)
            end = time.perf_counter_ns()
            
            latency_ms = (end - start) / 1_000_000
            latencies.append(latency_ms)
        
        avg_latency = mean(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 10, f"Avg DB write latency {avg_latency:.2f}ms exceeds 10ms"
        assert max_latency < 50, f"Max DB write latency {max_latency:.2f}ms exceeds 50ms"
        
        logger.info(f"Database writes - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")

    async def test_cache_operation_latency(self, trading_system):
        """Test cache read/write latency."""
        from genesis.cache.manager import CacheManager
        
        cache = CacheManager()
        read_latencies = []
        write_latencies = []
        
        # Write operations
        for i in range(1000):
            key = f"test_key_{i}"
            value = {"data": f"value_{i}", "number": i}
            
            start = time.perf_counter_ns()
            await cache.set(key, value)
            end = time.perf_counter_ns()
            
            write_latency_us = (end - start) / 1000
            write_latencies.append(write_latency_us)
        
        # Read operations
        for i in range(1000):
            key = f"test_key_{i}"
            
            start = time.perf_counter_ns()
            value = await cache.get(key)
            end = time.perf_counter_ns()
            
            read_latency_us = (end - start) / 1000
            read_latencies.append(read_latency_us)
        
        avg_write = mean(write_latencies)
        avg_read = mean(read_latencies)
        
        assert avg_write < 100, f"Avg cache write {avg_write:.2f}μs exceeds 100μs"
        assert avg_read < 50, f"Avg cache read {avg_read:.2f}μs exceeds 50μs"
        
        logger.info(f"Cache operations - Write: {avg_write:.2f}μs, Read: {avg_read:.2f}μs")

    async def test_strategy_decision_latency(self, trading_system):
        """Test strategy decision-making latency."""
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        
        strategy = SimpleArbitrageStrategy(
            symbol="BTC/USDT",
            min_spread=Decimal("0.001")
        )
        
        latencies = []
        
        for i in range(500):
            market_data = {
                "BTC/USDT": {
                    "bid": Decimal("50000") - Decimal(i % 10),
                    "ask": Decimal("50010") + Decimal(i % 10),
                    "volume": Decimal("100")
                }
            }
            
            start = time.perf_counter_ns()
            signal = await strategy.analyze(market_data)
            end = time.perf_counter_ns()
            
            latency_ms = (end - start) / 1_000_000
            latencies.append(latency_ms)
        
        avg_latency = mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        assert avg_latency < 10, f"Avg strategy latency {avg_latency:.2f}ms exceeds 10ms"
        assert p95_latency < 20, f"P95 strategy latency {p95_latency:.2f}ms exceeds 20ms"
        
        logger.info(f"Strategy decisions - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

    async def test_websocket_roundtrip_latency(self, trading_system):
        """Test WebSocket message roundtrip latency."""
        from genesis.exchange.websocket_manager import WebSocketManager
        
        ws_manager = WebSocketManager("wss://test.example.com")
        latencies = []
        
        for i in range(100):
            ping_message = {
                "type": "ping",
                "id": i,
                "timestamp": time.perf_counter_ns()
            }
            
            start = time.perf_counter_ns()
            pong = await ws_manager.ping(ping_message)
            end = time.perf_counter_ns()
            
            if pong:
                latency_ms = (end - start) / 1_000_000
                latencies.append(latency_ms)
        
        if latencies:
            avg_latency = mean(latencies)
            max_latency = max(latencies)
            
            assert avg_latency < 10, f"Avg WS latency {avg_latency:.2f}ms exceeds 10ms"
            assert max_latency < 50, f"Max WS latency {max_latency:.2f}ms exceeds 50ms"
            
            logger.info(f"WebSocket roundtrip - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")

    async def test_concurrent_operation_latency(self, trading_system):
        """Test latency under concurrent operations."""
        async def measure_operation(operation_type):
            latencies = []
            
            for _ in range(100):
                start = time.perf_counter_ns()
                
                if operation_type == "order":
                    order = Order(
                        symbol="BTC/USDT",
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=Decimal("0.001")
                    )
                    await trading_system.engine.execute_order(order)
                elif operation_type == "market_data":
                    await trading_system.engine.process_market_data({
                        "BTC/USDT": {"bid": Decimal("50000"), "ask": Decimal("50001")}
                    })
                elif operation_type == "risk_check":
                    await trading_system.risk_engine.check_limits()
                
                end = time.perf_counter_ns()
                latency_ms = (end - start) / 1_000_000
                latencies.append(latency_ms)
            
            return latencies
        
        # Run operations concurrently
        results = await asyncio.gather(
            measure_operation("order"),
            measure_operation("market_data"),
            measure_operation("risk_check")
        )
        
        for i, operation_type in enumerate(["order", "market_data", "risk_check"]):
            latencies = results[i]
            avg_latency = mean(latencies)
            
            logger.info(f"Concurrent {operation_type} - Avg: {avg_latency:.2f}ms")
            
            if operation_type == "order":
                assert avg_latency < 100, f"Order latency {avg_latency:.2f}ms too high under load"
            elif operation_type == "market_data":
                assert avg_latency < 20, f"Market data latency {avg_latency:.2f}ms too high under load"
            elif operation_type == "risk_check":
                assert avg_latency < 10, f"Risk check latency {avg_latency:.2f}ms too high under load"