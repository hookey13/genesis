"""
Load testing suite for trading loop performance validation.

Tests high-frequency event scenarios, memory stability, and latency degradation.
"""

import asyncio
import gc
import time
from decimal import Decimal

import psutil
import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.trading_loop import TradingLoop


class LoadTestMetrics:
    """Collect and analyze load test metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.events_processed = 0
        self.memory_samples: list[float] = []
        self.latency_samples: list[float] = []
        self.errors = 0
        self.process = psutil.Process()

    def start(self):
        """Start metrics collection."""
        self.start_time = time.perf_counter()
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB

    def record_event(self, latency_ms: float):
        """Record processed event metrics."""
        self.events_processed += 1
        self.latency_samples.append(latency_ms)

    def record_memory(self):
        """Sample current memory usage."""
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB

    def stop(self):
        """Stop metrics collection and calculate results."""
        self.end_time = time.perf_counter()

    def get_results(self) -> dict:
        """Get test results summary."""
        duration = self.end_time - self.start_time if self.end_time else 0

        # Calculate latency percentiles
        sorted_latencies = sorted(self.latency_samples) if self.latency_samples else [0]
        n = len(sorted_latencies)

        # Calculate memory statistics
        memory_delta = self.memory_samples[-1] - self.memory_samples[0] if len(self.memory_samples) > 1 else 0

        return {
            "duration_seconds": duration,
            "events_processed": self.events_processed,
            "events_per_second": self.events_processed / duration if duration > 0 else 0,
            "error_rate": self.errors / self.events_processed if self.events_processed > 0 else 0,
            "latency_p50_ms": sorted_latencies[int(n * 0.5)] if n > 0 else 0,
            "latency_p95_ms": sorted_latencies[int(n * 0.95)] if n > 0 else 0,
            "latency_p99_ms": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
            "latency_max_ms": sorted_latencies[-1] if n > 0 else 0,
            "memory_start_mb": self.memory_samples[0] if self.memory_samples else 0,
            "memory_end_mb": self.memory_samples[-1] if self.memory_samples else 0,
            "memory_delta_mb": memory_delta,
            "memory_leak_detected": memory_delta > 100,  # More than 100MB growth indicates potential leak
        }


@pytest.fixture
async def load_test_setup():
    """Set up components for load testing."""
    # Create real components (not mocks) for load testing
    event_bus = EventBus()
    await event_bus.start()

    # Create risk engine with test account
    from genesis.core.models import Account, TradingTier
    account = Account(
        account_id="load_test",
        balance_usdt=Decimal("10000"),
        tier=TradingTier.SNIPER
    )

    risk_engine = RiskEngine(account)

    # Create mock exchange gateway
    from unittest.mock import AsyncMock, MagicMock
    exchange_gateway = MagicMock()
    exchange_gateway.validate_connection = AsyncMock(return_value=True)
    exchange_gateway.execute_order = AsyncMock(return_value={
        "success": True,
        "exchange_order_id": "TEST123",
        "fill_price": Decimal("50000"),
        "latency_ms": 10
    })

    # Create trading loop
    trading_loop = TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=exchange_gateway
    )

    await trading_loop.startup()

    yield {
        "event_bus": event_bus,
        "trading_loop": trading_loop,
        "exchange_gateway": exchange_gateway
    }

    await trading_loop.shutdown()
    await event_bus.stop()


class TestHighFrequencyEvents:
    """Test trading loop under high-frequency event load."""

    async def test_thousand_events_per_second(self, load_test_setup):
        """Verify system handles 1000+ events/second."""
        metrics = LoadTestMetrics()
        event_bus = load_test_setup["event_bus"]
        trading_loop = load_test_setup["trading_loop"]

        # Target: 1000 events/second for 10 seconds
        target_rate = 1000
        duration_seconds = 10
        total_events = target_rate * duration_seconds

        metrics.start()

        # Generate events at target rate
        start = time.perf_counter()
        events_sent = 0

        for i in range(total_events):
            # Create price update event
            event = Event(
                event_type=EventType.MARKET_DATA_UPDATED,
                event_data={
                    "symbol": "BTC/USDT",
                    "price": str(50000 + (i % 1000)),  # Vary price slightly
                    "volume": "100",
                    "timestamp": time.time()
                }
            )

            event_start = time.perf_counter()
            await event_bus.publish(event, EventPriority.NORMAL)
            event_latency = (time.perf_counter() - event_start) * 1000
            metrics.record_event(event_latency)

            events_sent += 1

            # Rate limiting to maintain target rate
            elapsed = time.perf_counter() - start
            expected_events = int(elapsed * target_rate)
            if events_sent > expected_events:
                sleep_time = (events_sent - expected_events) / target_rate
                await asyncio.sleep(sleep_time)

            # Sample memory every 100 events
            if i % 100 == 0:
                metrics.record_memory()

        # Allow processing to complete
        await asyncio.sleep(1)

        metrics.stop()
        results = metrics.get_results()

        # Assertions
        assert results["events_per_second"] >= 900, f"Failed to achieve target rate: {results['events_per_second']:.2f} events/sec"
        assert results["latency_p99_ms"] < 100, f"P99 latency too high: {results['latency_p99_ms']:.2f}ms"
        assert results["error_rate"] < 0.01, f"Error rate too high: {results['error_rate']:.2%}"
        assert not results["memory_leak_detected"], f"Memory leak detected: {results['memory_delta_mb']:.2f}MB growth"

        print("\n✅ High-frequency test passed:")
        print(f"   - Rate achieved: {results['events_per_second']:.2f} events/sec")
        print(f"   - P50 latency: {results['latency_p50_ms']:.2f}ms")
        print(f"   - P99 latency: {results['latency_p99_ms']:.2f}ms")
        print(f"   - Memory delta: {results['memory_delta_mb']:.2f}MB")

    async def test_burst_load_handling(self, load_test_setup):
        """Test handling of sudden traffic bursts."""
        metrics = LoadTestMetrics()
        event_bus = load_test_setup["event_bus"]

        metrics.start()

        # Send burst of 5000 events as fast as possible
        burst_size = 5000
        burst_start = time.perf_counter()

        tasks = []
        for i in range(burst_size):
            event = Event(
                event_type=EventType.ARBITRAGE_SIGNAL,
                event_data={
                    "strategy_id": f"burst_test_{i}",
                    "pair1_symbol": "BTC/USDT",
                    "signal_type": "ENTRY" if i % 2 == 0 else "EXIT",
                    "confidence_score": "0.8",
                    "zscore": "2.5"
                }
            )
            tasks.append(event_bus.publish(event, EventPriority.HIGH))

        # Send all events concurrently
        await asyncio.gather(*tasks)

        burst_duration = time.perf_counter() - burst_start
        burst_rate = burst_size / burst_duration

        # Allow processing to complete
        await asyncio.sleep(2)

        metrics.stop()

        # Assertions
        assert burst_rate > 2000, f"Burst handling too slow: {burst_rate:.2f} events/sec"
        assert event_bus.queue.qsize() < 1000, f"Queue backup too large: {event_bus.queue.qsize()} events"

        print("\n✅ Burst load test passed:")
        print(f"   - Burst rate handled: {burst_rate:.2f} events/sec")
        print(f"   - Queue depth: {event_bus.queue.qsize()} events")


class TestMemoryStability:
    """Test for memory leaks and stability under load."""

    async def test_sustained_load_memory(self, load_test_setup):
        """Ensure no memory leaks under sustained load."""
        metrics = LoadTestMetrics()
        event_bus = load_test_setup["event_bus"]
        trading_loop = load_test_setup["trading_loop"]

        metrics.start()
        initial_memory = metrics.memory_samples[0]

        # Run for 60 seconds at moderate load
        duration_seconds = 60
        events_per_second = 100

        start = time.perf_counter()
        events_sent = 0

        while time.perf_counter() - start < duration_seconds:
            # Create various event types to exercise different code paths
            event_types = [
                (EventType.MARKET_DATA_UPDATED, {
                    "symbol": "BTC/USDT",
                    "price": "50000",
                    "volume": "100"
                }),
                (EventType.ARBITRAGE_SIGNAL, {
                    "strategy_id": "test",
                    "pair1_symbol": "BTC/USDT",
                    "signal_type": "ENTRY",
                    "confidence_score": "0.8"
                }),
                (EventType.ORDER_FILLED, {
                    "order_id": f"ORDER_{events_sent}",
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "quantity": "0.1",
                    "price": "50000"
                })
            ]

            for event_type, event_data in event_types:
                event = Event(event_type=event_type, event_data=event_data)
                await event_bus.publish(event, EventPriority.NORMAL)
                events_sent += 1

            # Sample memory every second
            if int(time.perf_counter() - start) > len(metrics.memory_samples) - 1:
                metrics.record_memory()
                gc.collect()  # Force garbage collection to get accurate readings

            # Rate limiting
            await asyncio.sleep(1 / events_per_second * 3)  # 3 events per iteration

        metrics.stop()
        results = metrics.get_results()

        # Check for memory stability
        memory_growth_rate = results["memory_delta_mb"] / duration_seconds

        # Assertions
        assert results["memory_delta_mb"] < 50, f"Excessive memory growth: {results['memory_delta_mb']:.2f}MB"
        assert memory_growth_rate < 1, f"Memory growth rate too high: {memory_growth_rate:.2f}MB/sec"

        # Check if memory stabilizes (last 10 samples should be relatively stable)
        if len(metrics.memory_samples) > 10:
            recent_samples = metrics.memory_samples[-10:]
            recent_variation = max(recent_samples) - min(recent_samples)
            assert recent_variation < 10, f"Memory not stabilizing: {recent_variation:.2f}MB variation"

        print("\n✅ Memory stability test passed:")
        print(f"   - Initial memory: {initial_memory:.2f}MB")
        print(f"   - Final memory: {metrics.memory_samples[-1]:.2f}MB")
        print(f"   - Growth rate: {memory_growth_rate:.4f}MB/sec")
        print(f"   - Total events: {events_sent}")

    async def test_position_accumulation_memory(self, load_test_setup):
        """Test memory usage with many open positions."""
        trading_loop = load_test_setup["trading_loop"]
        metrics = LoadTestMetrics()

        metrics.start()

        # Create many positions
        num_positions = 1000

        from datetime import datetime

        from genesis.core.models import Position, PositionSide

        for i in range(num_positions):
            position = Position(
                account_id="load_test",
                symbol=f"PAIR{i}/USDT",
                side=PositionSide.LONG if i % 2 == 0 else PositionSide.SHORT,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                dollar_value=Decimal("5000"),
                created_at=datetime.now()
            )
            trading_loop.positions[position.position_id] = position

            if i % 100 == 0:
                metrics.record_memory()

        metrics.stop()

        # Check memory usage is reasonable
        memory_per_position = metrics.memory_samples[-1] / num_positions if num_positions > 0 else 0
        assert memory_per_position < 1, f"Excessive memory per position: {memory_per_position:.4f}MB"

        print("\n✅ Position accumulation test passed:")
        print(f"   - Positions created: {num_positions}")
        print(f"   - Memory per position: {memory_per_position:.4f}MB")
        print(f"   - Total memory used: {metrics.memory_samples[-1]:.2f}MB")


class TestLatencyDegradation:
    """Test latency characteristics under various load conditions."""

    async def test_latency_under_load(self, load_test_setup):
        """Measure latency degradation as load increases."""
        event_bus = load_test_setup["event_bus"]

        load_levels = [10, 50, 100, 500, 1000]  # Events per second
        results_by_load = {}

        for target_rate in load_levels:
            metrics = LoadTestMetrics()
            metrics.start()

            # Test each load level for 5 seconds
            duration = 5
            total_events = target_rate * duration

            start = time.perf_counter()
            for i in range(total_events):
                event = Event(
                    event_type=EventType.MARKET_DATA_UPDATED,
                    event_data={
                        "symbol": "BTC/USDT",
                        "price": str(50000 + i),
                        "volume": "100"
                    }
                )

                event_start = time.perf_counter()
                await event_bus.publish(event, EventPriority.NORMAL)
                latency = (time.perf_counter() - event_start) * 1000
                metrics.record_event(latency)

                # Rate limiting
                elapsed = time.perf_counter() - start
                expected_events = int(elapsed * target_rate)
                if i + 1 > expected_events:
                    await asyncio.sleep((i + 1 - expected_events) / target_rate)

            metrics.stop()
            results_by_load[target_rate] = metrics.get_results()

            # Allow system to settle between tests
            await asyncio.sleep(1)

        # Analyze latency degradation
        print("\n✅ Latency degradation analysis:")
        print(f"{'Load (evt/s)':<15} {'P50 (ms)':<10} {'P95 (ms)':<10} {'P99 (ms)':<10}")
        print("-" * 45)

        baseline_p99 = results_by_load[10]["latency_p99_ms"]

        for rate, results in results_by_load.items():
            degradation = (results["latency_p99_ms"] / baseline_p99 - 1) * 100 if baseline_p99 > 0 else 0
            print(f"{rate:<15} {results['latency_p50_ms']:<10.2f} {results['latency_p95_ms']:<10.2f} {results['latency_p99_ms']:<10.2f}")

            # Assert reasonable degradation (less than 10x at highest load)
            assert results["latency_p99_ms"] < baseline_p99 * 10, f"Excessive latency degradation at {rate} evt/s"

        # Check that system maintains sub-100ms P99 even at 1000 evt/s
        assert results_by_load[1000]["latency_p99_ms"] < 100, "P99 latency exceeds 100ms at maximum load"
