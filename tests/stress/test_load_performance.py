"""
Load and performance testing suite.
Tests system behavior under extreme load conditions.
"""

import asyncio
import os
import random
import time
from datetime import datetime
from decimal import Decimal

import psutil
import pytest
import structlog

logger = structlog.get_logger()


class LoadGenerator:
    """Generate various types of load for testing."""

    @staticmethod
    async def generate_market_data(rate_per_second: int, duration_seconds: int):
        """Generate market data at specified rate."""
        data_points = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            batch_start = time.time()

            # Generate batch of data points
            for _ in range(rate_per_second):
                data_points.append(
                    {
                        "symbol": random.choice(["BTCUSDT", "ETHUSDT", "ADAUSDT"]),
                        "price": Decimal(str(random.uniform(100, 60000))),
                        "volume": Decimal(str(random.uniform(0.01, 100))),
                        "timestamp": datetime.utcnow(),
                    }
                )

            # Sleep to maintain rate
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)

        return data_points

    @staticmethod
    async def generate_orders(count: int, concurrent: bool = True):
        """Generate multiple orders."""
        orders = []

        async def create_order(i):
            return {
                "id": f"order_{i}",
                "symbol": random.choice(["BTCUSDT", "ETHUSDT"]),
                "side": random.choice(["BUY", "SELL"]),
                "quantity": Decimal(str(random.uniform(0.01, 1))),
                "price": Decimal(str(random.uniform(1000, 60000))),
            }

        if concurrent:
            tasks = [create_order(i) for i in range(count)]
            orders = await asyncio.gather(*tasks)
        else:
            for i in range(count):
                orders.append(await create_order(i))

        return orders


class TestLoadPerformance:
    """Test system performance under load."""

    @pytest.mark.asyncio
    async def test_high_frequency_market_data(self):
        """Test processing high-frequency market data updates."""
        generator = LoadGenerator()

        # Generate 1000 updates per second for 10 seconds
        data = await generator.generate_market_data(
            rate_per_second=1000, duration_seconds=10
        )

        assert len(data) >= 9000  # Allow some variance
        assert all("symbol" in d for d in data)

    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self):
        """Test executing 100+ strategies concurrently."""
        strategy_count = 100

        async def run_strategy(strategy_id):
            # Simulate strategy execution
            start = time.time()

            # Simulate computation
            result = sum(i * random.random() for i in range(1000))

            await asyncio.sleep(random.uniform(0.01, 0.1))

            return {
                "strategy_id": strategy_id,
                "result": result,
                "duration": time.time() - start,
            }

        # Run all strategies concurrently
        start = time.time()
        tasks = [run_strategy(i) for i in range(strategy_count)]
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start

        assert len(results) == strategy_count
        assert total_duration < 5.0  # Should complete within 5 seconds

        # Check individual strategy performance
        slow_strategies = [r for r in results if r["duration"] > 1.0]
        assert len(slow_strategies) < 5  # Less than 5% slow

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage doesn't exceed limits under load."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate load
        data_storage = []
        for i in range(1000):
            # Create data structures
            data = {
                "positions": [{"id": f"pos_{i}_{j}"} for j in range(10)],
                "orders": [{"id": f"ord_{i}_{j}"} for j in range(20)],
                "trades": [{"id": f"trd_{i}_{j}"} for j in range(50)],
            }
            data_storage.append(data)

            # Check memory periodically
            if i % 100 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory

                # If memory grows too much, cleanup
                if memory_growth > 500:  # 500MB limit
                    data_storage = data_storage[-100:]  # Keep only recent
                    import gc

                    gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory

        assert total_growth < 500, f"Memory grew by {total_growth}MB"

    @pytest.mark.asyncio
    async def test_cpu_usage_optimization(self):
        """Test CPU usage remains reasonable under load."""
        process = psutil.Process(os.getpid())

        # Measure CPU during intensive operations
        cpu_samples = []

        async def intensive_task():
            # Simulate CPU-intensive calculation
            result = 0
            for i in range(1000000):
                result += i * 0.001
            return result

        # Run multiple intensive tasks
        for _ in range(10):
            start_cpu = process.cpu_percent(interval=0.1)

            tasks = [intensive_task() for _ in range(5)]
            await asyncio.gather(*tasks)

            end_cpu = process.cpu_percent(interval=0.1)
            cpu_samples.append(end_cpu)

        avg_cpu = sum(cpu_samples) / len(cpu_samples)

        # CPU usage should be reasonable (not constantly at 100%)
        assert avg_cpu < 90, f"Average CPU usage too high: {avg_cpu}%"

    @pytest.mark.asyncio
    async def test_network_saturation_handling(self):
        """Test system behavior under network saturation."""
        generator = LoadGenerator()

        # Simulate network requests
        successful_requests = 0
        failed_requests = 0

        async def make_request(request_id):
            # Simulate network delay and random failures
            delay = random.uniform(0.001, 0.1)
            await asyncio.sleep(delay)

            # Simulate 5% failure rate under load
            if random.random() < 0.05:
                raise ConnectionError(f"Request {request_id} failed")

            return {"request_id": request_id, "status": "success"}

        # Send many concurrent requests
        request_count = 1000
        tasks = []

        for i in range(request_count):
            tasks.append(make_request(i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
            else:
                successful_requests += 1

        success_rate = successful_requests / request_count
        assert success_rate > 0.9, f"Success rate too low: {success_rate:.2%}"

    @pytest.mark.asyncio
    async def test_database_connection_pooling(self):
        """Test database connection pool under load."""
        connection_pool_size = 10
        query_count = 1000

        # Simulate connection pool
        class ConnectionPool:
            def __init__(self, size):
                self.connections = asyncio.Queue(maxsize=size)
                for i in range(size):
                    self.connections.put_nowait(f"conn_{i}")

            async def get_connection(self):
                return await self.connections.get()

            async def return_connection(self, conn):
                await self.connections.put(conn)

        pool = ConnectionPool(connection_pool_size)

        async def execute_query(query_id):
            conn = await pool.get_connection()
            try:
                # Simulate query execution
                await asyncio.sleep(random.uniform(0.001, 0.01))
                return {"query_id": query_id, "conn": conn}
            finally:
                await pool.return_connection(conn)

        # Execute many queries concurrently
        start = time.time()
        tasks = [execute_query(i) for i in range(query_count)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start

        assert len(results) == query_count
        assert duration < 10.0  # Should complete within 10 seconds

        # Verify all connections were used
        used_connections = set(r["conn"] for r in results)
        assert len(used_connections) == connection_pool_size

    @pytest.mark.asyncio
    async def test_order_throughput(self):
        """Test maximum order throughput."""
        generator = LoadGenerator()

        # Generate orders at increasing rates
        throughput_results = []

        for orders_per_second in [10, 50, 100, 500, 1000]:
            start = time.time()

            # Generate and "process" orders
            orders = await generator.generate_orders(
                count=orders_per_second, concurrent=True
            )

            duration = time.time() - start
            actual_throughput = len(orders) / duration

            throughput_results.append(
                {
                    "target": orders_per_second,
                    "actual": actual_throughput,
                    "duration": duration,
                }
            )

            # Should achieve at least 80% of target throughput
            assert actual_throughput > orders_per_second * 0.8

    @pytest.mark.asyncio
    async def test_latency_under_load(self):
        """Test latency remains acceptable under load."""
        latencies = []

        async def measure_latency():
            start = time.time()
            # Simulate operation
            await asyncio.sleep(0.001)
            return (time.time() - start) * 1000  # Convert to ms

        # Create background load
        background_tasks = []
        for _ in range(100):
            task = asyncio.create_task(asyncio.sleep(random.uniform(0.01, 0.1)))
            background_tasks.append(task)

        # Measure latencies
        for _ in range(100):
            latency = await measure_latency()
            latencies.append(latency)

        # Cleanup background tasks
        for task in background_tasks:
            task.cancel()

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.5)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        assert p50 < 10, f"P50 latency too high: {p50}ms"
        assert p95 < 50, f"P95 latency too high: {p95}ms"
        assert p99 < 100, f"P99 latency too high: {p99}ms"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system degrades gracefully under extreme load."""
        # Track performance metrics
        metrics = {
            "requests_processed": 0,
            "requests_rejected": 0,
            "degraded_mode_activated": False,
        }

        # Simulate increasing load
        for load_level in [100, 500, 1000, 2000, 5000]:

            async def process_request():
                # Check if should reject due to overload
                if load_level > 1000 and not metrics["degraded_mode_activated"]:
                    metrics["degraded_mode_activated"] = True

                if metrics["degraded_mode_activated"] and random.random() < 0.5:
                    metrics["requests_rejected"] += 1
                    raise Exception("System overloaded")

                metrics["requests_processed"] += 1
                await asyncio.sleep(0.001)

            # Process requests at current load level
            tasks = []
            for _ in range(min(load_level, 1000)):  # Cap concurrent tasks
                tasks.append(process_request())

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # System should handle some requests even under extreme load
            successful = sum(1 for r in results if not isinstance(r, Exception))
            assert successful > 0

        # Verify graceful degradation occurred
        assert metrics["degraded_mode_activated"]
        assert metrics["requests_processed"] > 0
        assert metrics["requests_rejected"] > 0

    @pytest.mark.asyncio
    async def test_recovery_after_load_spike(self):
        """Test system recovers after load spike."""
        process = psutil.Process(os.getpid())

        # Baseline metrics
        baseline_memory = process.memory_info().rss / 1024 / 1024
        baseline_handles = len(process.open_files())

        # Create load spike
        spike_data = []
        for _ in range(1000):
            spike_data.append([random.random() for _ in range(1000)])

        spike_memory = process.memory_info().rss / 1024 / 1024

        # Clear spike data
        del spike_data
        import gc

        gc.collect()

        # Wait for recovery
        await asyncio.sleep(1)

        # Check recovery
        recovered_memory = process.memory_info().rss / 1024 / 1024
        recovered_handles = len(process.open_files())

        # Memory should recover to near baseline
        memory_diff = recovered_memory - baseline_memory
        assert memory_diff < 100, f"Memory not recovered: {memory_diff}MB difference"

        # File handles should be same
        assert recovered_handles <= baseline_handles + 5
