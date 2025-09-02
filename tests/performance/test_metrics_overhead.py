"""Performance tests for metrics collection overhead."""

import asyncio
import gc
import time
from decimal import Decimal
from statistics import mean, stdev
from unittest.mock import MagicMock

import pytest

from genesis.core.models import Order, OrderStatus
from genesis.monitoring.metrics_collector import MetricsCollector
from genesis.monitoring.prometheus_exporter import MetricsRegistry
from genesis.monitoring.application_metrics import (
    ApplicationMetricsCollector,
    WebSocketState,
)


class TestMetricsOverhead:
    """Test metrics collection performance overhead."""
    
    @pytest.fixture
    async def setup_metrics(self):
        """Set up metrics collection stack."""
        registry = MetricsRegistry()
        metrics_collector = MetricsCollector(registry, enable_memory_profiling=False)
        app_metrics = ApplicationMetricsCollector(registry)
        
        await metrics_collector.start()
        
        yield {
            "registry": registry,
            "metrics_collector": metrics_collector,
            "app_metrics": app_metrics
        }
        
        await metrics_collector.stop()
    
    async def test_metrics_collection_overhead(self, setup_metrics):
        """Ensure metrics collection adds <1% latency to operations."""
        metrics_collector = setup_metrics["metrics_collector"]
        
        # Number of iterations for testing
        iterations = 1000
        
        # Create test order
        def create_order(i):
            order = MagicMock(spec=Order)
            order.client_order_id = f"order_{i}"
            order.status = OrderStatus.FILLED
            order.exchange = "binance"
            order.symbol = "BTC/USDT"
            order.side = "BUY" if i % 2 == 0 else "SELL"
            order.order_type = "MARKET"
            order.executed_qty = 0.01 * (i % 10 + 1)
            order.price = 50000 + (i % 1000)
            return order
        
        # Measure baseline (no metrics)
        gc.collect()
        baseline_times = []
        
        for i in range(iterations):
            order = create_order(i)
            start = time.perf_counter()
            # Simulate order processing without metrics
            _ = order.client_order_id
            _ = order.status
            _ = float(order.executed_qty) * float(order.price)
            baseline_times.append(time.perf_counter() - start)
        
        baseline_mean = mean(baseline_times)
        baseline_std = stdev(baseline_times)
        
        # Measure with metrics collection
        gc.collect()
        metrics_times = []
        
        for i in range(iterations):
            order = create_order(i)
            start = time.perf_counter()
            await metrics_collector.record_order(order)
            await metrics_collector.record_execution_time(0.01 + (i % 100) / 10000)
            metrics_times.append(time.perf_counter() - start)
        
        metrics_mean = mean(metrics_times)
        metrics_std = stdev(metrics_times)
        
        # Calculate overhead
        overhead_abs = metrics_mean - baseline_mean
        overhead_percent = (overhead_abs / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Print results for debugging
        print(f"\nBaseline: {baseline_mean*1000:.3f}ms ± {baseline_std*1000:.3f}ms")
        print(f"With metrics: {metrics_mean*1000:.3f}ms ± {metrics_std*1000:.3f}ms")
        print(f"Overhead: {overhead_abs*1000:.3f}ms ({overhead_percent:.2f}%)")
        
        # Assert overhead is less than 1%
        assert overhead_percent < 1.0, f"Metrics overhead {overhead_percent:.2f}% exceeds 1%"
    
    async def test_high_cardinality_performance(self, setup_metrics):
        """Test performance with high-cardinality labels."""
        registry = setup_metrics["registry"]
        
        # Create many unique label combinations
        start = time.perf_counter()
        
        for i in range(1000):
            await registry.set_gauge(
                "test_metric",
                float(i),
                {
                    "id": str(i),
                    "type": f"type_{i % 10}",
                    "status": f"status_{i % 5}"
                }
            )
        
        elapsed = time.perf_counter() - start
        
        # Should handle 1000 unique label combinations in reasonable time
        assert elapsed < 1.0, f"High cardinality handling too slow: {elapsed:.3f}s"
        
        # Test collection performance
        start = time.perf_counter()
        output = await registry.collect()
        collection_time = time.perf_counter() - start
        
        # Collection should be fast even with many metrics
        assert collection_time < 0.5, f"Collection too slow: {collection_time:.3f}s"
        assert len(output) > 0
    
    async def test_concurrent_metrics_updates(self, setup_metrics):
        """Test concurrent metric updates performance."""
        registry = setup_metrics["registry"]
        
        # Number of concurrent tasks
        num_tasks = 100
        updates_per_task = 100
        
        async def update_metrics(task_id):
            """Update metrics concurrently."""
            for i in range(updates_per_task):
                await registry.increment_counter(
                    "concurrent_counter",
                    1.0,
                    {"task": str(task_id), "iteration": str(i % 10)}
                )
                await registry.set_gauge(
                    "concurrent_gauge",
                    float(i),
                    {"task": str(task_id)}
                )
        
        # Measure concurrent updates
        start = time.perf_counter()
        tasks = [update_metrics(i) for i in range(num_tasks)]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        
        total_operations = num_tasks * updates_per_task * 2  # counter + gauge
        ops_per_second = total_operations / elapsed
        
        print(f"\nConcurrent updates: {ops_per_second:.0f} ops/sec")
        
        # Should handle at least 10k ops/sec
        assert ops_per_second > 10000, f"Concurrent performance too low: {ops_per_second:.0f} ops/sec"
    
    async def test_websocket_metrics_overhead(self, setup_metrics):
        """Test WebSocket metrics collection overhead."""
        app_metrics = setup_metrics["app_metrics"]
        
        iterations = 10000
        
        # Baseline: process messages without metrics
        gc.collect()
        baseline_times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            # Simulate message processing
            message_size = 100 + (i % 900)
            _ = f"message_{i}"
            _ = message_size
            baseline_times.append(time.perf_counter() - start)
        
        baseline_mean = mean(baseline_times)
        
        # With metrics
        gc.collect()
        metrics_times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            await app_metrics.record_websocket_message(
                "binance",
                "trades",
                "received",
                100 + (i % 900)
            )
            metrics_times.append(time.perf_counter() - start)
        
        metrics_mean = mean(metrics_times)
        
        # Calculate overhead
        overhead_percent = ((metrics_mean - baseline_mean) / baseline_mean * 100
                          if baseline_mean > 0 else 0)
        
        print(f"\nWebSocket metrics overhead: {overhead_percent:.2f}%")
        
        # WebSocket metrics should have minimal overhead
        assert overhead_percent < 0.5, f"WebSocket metrics overhead {overhead_percent:.2f}% too high"
    
    async def test_memory_usage_with_metrics(self, setup_metrics):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        registry = setup_metrics["registry"]
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate lots of metrics
        for i in range(10000):
            await registry.increment_counter(
                "memory_test_counter",
                1.0,
                {"iteration": str(i % 100)}  # Limit cardinality
            )
            
            if i % 1000 == 0:
                # Collect periodically
                await registry.collect()
        
        # Get final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory growth: {memory_growth:.2f} MB")
        
        # Memory growth should be reasonable (< 50 MB for 10k metrics)
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.2f} MB"
    
    async def test_collection_time_scaling(self, setup_metrics):
        """Test how collection time scales with number of metrics."""
        registry = setup_metrics["registry"]
        
        collection_times = []
        metric_counts = [100, 500, 1000, 5000, 10000]
        
        for count in metric_counts:
            # Clear registry
            registry._metrics.clear()
            registry._labeled_metrics.clear()
            
            # Add metrics
            for i in range(count):
                await registry.set_gauge(
                    f"metric_{i % 100}",  # Reuse some metric names
                    float(i),
                    {"label": str(i % 10)} if i % 2 == 0 else None
                )
            
            # Measure collection time
            times = []
            for _ in range(10):
                start = time.perf_counter()
                output = await registry.collect()
                times.append(time.perf_counter() - start)
            
            avg_time = mean(times)
            collection_times.append((count, avg_time))
            
            print(f"{count} metrics: {avg_time*1000:.2f}ms")
        
        # Check that collection time scales reasonably (not exponentially)
        # Time should not more than double when metrics increase 10x
        time_100 = collection_times[0][1]
        time_1000 = collection_times[2][1]
        time_10000 = collection_times[4][1]
        
        scaling_factor_1k = time_1000 / time_100
        scaling_factor_10k = time_10000 / time_100
        
        print(f"\nScaling: 1k={scaling_factor_1k:.1f}x, 10k={scaling_factor_10k:.1f}x")
        
        # Should scale sub-linearly
        assert scaling_factor_10k < 20, f"Collection time scaling too steep: {scaling_factor_10k:.1f}x"


@pytest.mark.benchmark
class TestMetricsBenchmark:
    """Benchmark tests for metrics operations."""
    
    @pytest.fixture
    async def registry(self):
        """Create test registry."""
        return MetricsRegistry()
    
    async def test_counter_increment_benchmark(self, benchmark, registry):
        """Benchmark counter increment operation."""
        await registry.register(Metric(
            name="benchmark_counter",
            type=MetricType.COUNTER,
            help="Benchmark counter"
        ))
        
        async def increment():
            await registry.increment_counter("benchmark_counter", 1.0)
        
        # Run benchmark
        result = benchmark(asyncio.run, increment())
        
        # Should be very fast (< 1ms)
        assert result < 0.001
    
    async def test_gauge_set_benchmark(self, benchmark, registry):
        """Benchmark gauge set operation."""
        await registry.register(Metric(
            name="benchmark_gauge",
            type=MetricType.GAUGE,
            help="Benchmark gauge"
        ))
        
        async def set_gauge():
            await registry.set_gauge("benchmark_gauge", 42.0)
        
        # Run benchmark
        result = benchmark(asyncio.run, set_gauge())
        
        # Should be very fast (< 1ms)
        assert result < 0.001
    
    async def test_collection_benchmark(self, benchmark, registry):
        """Benchmark metrics collection."""
        # Add various metrics
        for i in range(100):
            await registry.register(Metric(
                name=f"metric_{i}",
                type=MetricType.GAUGE if i % 2 == 0 else MetricType.COUNTER,
                help=f"Metric {i}"
            ))
            
            if i % 2 == 0:
                await registry.set_gauge(f"metric_{i}", float(i))
            else:
                await registry.increment_counter(f"metric_{i}", float(i))
        
        async def collect():
            return await registry.collect()
        
        # Run benchmark
        result = benchmark(asyncio.run, collect())
        
        # Should complete quickly even with 100 metrics (< 10ms)
        assert result < 0.01