"""Performance and load testing suite."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from genesis.engine.engine import TradingEngine
from genesis.core.models import Order, OrderType, OrderSide

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestPerformanceUnderLoad:
    """Test system performance under various load conditions."""

    async def test_1000_orders_per_second_throughput(self, trading_system):
        """Test handling 1000+ orders per second."""
        orders_processed = 0
        errors = 0
        
        async def submit_order():
            nonlocal orders_processed, errors
            try:
                order = Order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY if orders_processed % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=Decimal("0.001")
                )
                await trading_system.engine.execute_order(order)
                orders_processed += 1
            except Exception as e:
                errors += 1
                logger.error(f"Order submission failed: {e}")
        
        start_time = time.perf_counter()
        
        tasks = [submit_order() for _ in range(1000)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        throughput = orders_processed / duration
        error_rate = errors / 1000
        
        assert throughput >= 1000, f"Throughput {throughput:.2f} orders/sec is below target"
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1% threshold"
        
        logger.info(f"Achieved throughput: {throughput:.2f} orders/sec")

    async def test_end_to_end_latency(self, trading_system):
        """Test end-to-end latency under load."""
        latencies = []
        
        for _ in range(100):
            start = time.perf_counter_ns()
            
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )
            
            await trading_system.engine.execute_order(order)
            
            end = time.perf_counter_ns()
            latency_ms = (end - start) / 1_000_000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms exceeds 50ms"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms exceeds 100ms"
        assert p99_latency < 200, f"P99 latency {p99_latency:.2f}ms exceeds 200ms"
        
        logger.info(f"Latencies - Avg: {avg_latency:.2f}ms, P50: {p50_latency:.2f}ms, "
                   f"P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")

    async def test_memory_usage_under_load(self, trading_system):
        """Test memory usage remains bounded under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        peak_memory = initial_memory
        
        for batch in range(10):
            orders = []
            for i in range(1000):
                order = Order(
                    symbol=f"PAIR{i % 10}/USDT",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("0.001"),
                    price=Decimal("50000") + Decimal(i)
                )
                orders.append(order)
            
            tasks = [trading_system.engine.execute_order(o) for o in orders]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            await asyncio.sleep(0.1)  # Allow garbage collection
        
        memory_increase = peak_memory - initial_memory
        
        assert memory_increase < 2000, f"Memory increase {memory_increase:.2f}MB exceeds 2GB limit"
        
        logger.info(f"Memory usage - Initial: {initial_memory:.2f}MB, "
                   f"Peak: {peak_memory:.2f}MB, Increase: {memory_increase:.2f}MB")

    async def test_cpu_utilization_with_strategies(self, trading_system):
        """Test CPU utilization with multiple active strategies."""
        process = psutil.Process(os.getpid())
        
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        
        strategies = []
        for i in range(10):
            strategy = SimpleArbitrageStrategy(
                symbol=f"PAIR{i}/USDT",
                min_spread=Decimal("0.001")
            )
            strategies.append(strategy)
            await trading_system.engine.register_strategy(strategy)
        
        cpu_samples = []
        
        async def monitor_cpu():
            for _ in range(60):
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(1)
        
        monitor_task = asyncio.create_task(monitor_cpu())
        
        for _ in range(60):
            market_data = {
                f"PAIR{i}/USDT": {
                    "bid": Decimal("100") + Decimal(i),
                    "ask": Decimal("101") + Decimal(i)
                }
                for i in range(10)
            }
            await trading_system.engine.process_market_data(market_data)
            await asyncio.sleep(1)
        
        await monitor_task
        
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        
        assert avg_cpu < 50, f"Average CPU {avg_cpu:.2f}% exceeds 50% limit"
        assert max_cpu < 80, f"Peak CPU {max_cpu:.2f}% exceeds 80% limit"
        
        logger.info(f"CPU usage - Avg: {avg_cpu:.2f}%, Max: {max_cpu:.2f}%")

    async def test_concurrent_market_data_processing(self, trading_system):
        """Test concurrent processing of market data updates."""
        update_count = 0
        error_count = 0
        
        async def process_update(symbol, price):
            nonlocal update_count, error_count
            try:
                market_data = {
                    symbol: {
                        "bid": price - Decimal("1"),
                        "ask": price + Decimal("1"),
                        "last": price
                    }
                }
                await trading_system.engine.process_market_data(market_data)
                update_count += 1
            except Exception:
                error_count += 1
        
        symbols = [f"PAIR{i}/USDT" for i in range(100)]
        
        start_time = time.perf_counter()
        
        tasks = []
        for _ in range(10):
            for i, symbol in enumerate(symbols):
                price = Decimal("100") + Decimal(i)
                tasks.append(process_update(symbol, price))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        updates_per_second = update_count / duration
        
        assert updates_per_second > 500, f"Update rate {updates_per_second:.2f}/sec below target"
        assert error_count < update_count * 0.01, "Error rate exceeds 1%"
        
        logger.info(f"Market data processing rate: {updates_per_second:.2f} updates/sec")

    async def test_database_query_performance(self, trading_system):
        """Test database query performance under load."""
        from genesis.data.repository import TradeRepository
        
        repo = TradeRepository()
        
        # Insert test data
        for i in range(10000):
            trade = {
                "order_id": f"order_{i}",
                "symbol": f"PAIR{i % 10}/USDT",
                "side": "buy" if i % 2 == 0 else "sell",
                "quantity": Decimal("0.001"),
                "price": Decimal("50000") + Decimal(i),
                "timestamp": datetime.now() - timedelta(minutes=i)
            }
            await repo.save_trade(trade)
        
        query_times = []
        
        for _ in range(100):
            start = time.perf_counter()
            
            trades = await repo.get_recent_trades(limit=100)
            
            end = time.perf_counter()
            query_time_ms = (end - start) * 1000
            query_times.append(query_time_ms)
        
        avg_query_time = sum(query_times) / len(query_times)
        max_query_time = max(query_times)
        
        assert avg_query_time < 10, f"Avg query time {avg_query_time:.2f}ms exceeds 10ms"
        assert max_query_time < 50, f"Max query time {max_query_time:.2f}ms exceeds 50ms"
        
        logger.info(f"Database query performance - Avg: {avg_query_time:.2f}ms, "
                   f"Max: {max_query_time:.2f}ms")

    async def test_websocket_message_processing(self, trading_system):
        """Test WebSocket message processing rate."""
        from genesis.exchange.websocket_manager import WebSocketManager
        
        ws_manager = WebSocketManager("wss://test.example.com")
        messages_processed = 0
        
        async def process_message(message):
            nonlocal messages_processed
            await ws_manager.process_message(message)
            messages_processed += 1
        
        messages = [
            {
                "type": "trade",
                "symbol": f"PAIR{i % 10}/USDT",
                "price": Decimal("50000") + Decimal(i),
                "quantity": Decimal("0.001"),
                "timestamp": time.time()
            }
            for i in range(10000)
        ]
        
        start_time = time.perf_counter()
        
        tasks = [process_message(msg) for msg in messages]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        messages_per_second = messages_processed / duration
        processing_time_per_message = duration / messages_processed * 1000
        
        assert messages_per_second > 1000, f"Message rate {messages_per_second:.2f}/sec below target"
        assert processing_time_per_message < 5, f"Processing time {processing_time_per_message:.2f}ms exceeds 5ms"
        
        logger.info(f"WebSocket processing - Rate: {messages_per_second:.2f} msg/sec, "
                   f"Time per message: {processing_time_per_message:.2f}ms")

    async def test_strategy_calculation_performance(self, trading_system):
        """Test strategy calculation performance."""
        from genesis.strategies.hunter.mean_reversion import MeanReversionStrategy
        
        strategy = MeanReversionStrategy(
            symbol="BTC/USDT",
            lookback_period=100,
            entry_threshold=Decimal("2.0")
        )
        
        calculation_times = []
        
        for i in range(1000):
            market_data = {
                "BTC/USDT": {
                    "bid": Decimal("50000") + Decimal(i % 100),
                    "ask": Decimal("50001") + Decimal(i % 100),
                    "volume": Decimal("100")
                }
            }
            
            start = time.perf_counter()
            signal = await strategy.analyze(market_data)
            end = time.perf_counter()
            
            calc_time_ms = (end - start) * 1000
            calculation_times.append(calc_time_ms)
        
        avg_calc_time = sum(calculation_times) / len(calculation_times)
        max_calc_time = max(calculation_times)
        
        assert avg_calc_time < 100, f"Avg calculation time {avg_calc_time:.2f}ms exceeds 100ms"
        assert max_calc_time < 200, f"Max calculation time {max_calc_time:.2f}ms exceeds 200ms"
        
        logger.info(f"Strategy calculation - Avg: {avg_calc_time:.2f}ms, Max: {max_calc_time:.2f}ms")