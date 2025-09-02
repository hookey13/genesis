"""Load test to verify tracing overhead is <1% on system performance."""

import asyncio
import time
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import List, Tuple

import pytest
from unittest.mock import Mock, patch

from genesis.monitoring.opentelemetry_tracing import (
    OpenTelemetryTracer,
    get_opentelemetry_tracer,
)


@dataclass
class PerformanceMetrics:
    """Performance test metrics."""
    operation_name: str
    iterations: int
    baseline_times: List[float]
    traced_times: List[float]
    
    @property
    def baseline_mean(self) -> float:
        """Mean baseline execution time."""
        return mean(self.baseline_times)
    
    @property
    def traced_mean(self) -> float:
        """Mean traced execution time."""
        return mean(self.traced_times)
    
    @property
    def overhead_ms(self) -> float:
        """Overhead in milliseconds."""
        return (self.traced_mean - self.baseline_mean) * 1000
    
    @property
    def overhead_percent(self) -> float:
        """Overhead as percentage."""
        if self.baseline_mean == 0:
            return 0
        return ((self.traced_mean - self.baseline_mean) / self.baseline_mean) * 100
    
    @property
    def baseline_p99(self) -> float:
        """99th percentile baseline time."""
        sorted_times = sorted(self.baseline_times)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def traced_p99(self) -> float:
        """99th percentile traced time."""
        sorted_times = sorted(self.traced_times)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_summary(self) -> str:
        """Get performance summary."""
        return (
            f"{self.operation_name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Baseline mean: {self.baseline_mean * 1000:.3f}ms\n"
            f"  Traced mean: {self.traced_mean * 1000:.3f}ms\n"
            f"  Overhead: {self.overhead_ms:.3f}ms ({self.overhead_percent:.2f}%)\n"
            f"  Baseline P99: {self.baseline_p99 * 1000:.3f}ms\n"
            f"  Traced P99: {self.traced_p99 * 1000:.3f}ms"
        )


class TestTracingOverhead:
    """Test tracing overhead on system performance."""
    
    @pytest.fixture
    def tracer(self):
        """Create test tracer with production settings."""
        return OpenTelemetryTracer(
            service_name="load-test",
            otlp_endpoint=None,  # No actual export to isolate overhead
            sampling_rate=0.01,  # Production sampling rate
            export_to_console=False,
            production_mode=True
        )
    
    async def simulate_order_execution(self, delay: float = 0.01) -> dict:
        """Simulate order execution operation."""
        await asyncio.sleep(delay)
        return {"order_id": "test_123", "status": "success"}
    
    async def simulate_risk_check(self, delay: float = 0.005) -> bool:
        """Simulate risk check operation."""
        await asyncio.sleep(delay)
        return True
    
    async def simulate_market_data_processing(self, delay: float = 0.002) -> dict:
        """Simulate market data processing."""
        await asyncio.sleep(delay)
        return {"processed": True}
    
    async def simulate_database_query(self, delay: float = 0.02) -> list:
        """Simulate database query."""
        await asyncio.sleep(delay)
        return [{"id": 1, "data": "test"}]
    
    async def measure_baseline_performance(
        self,
        operation,
        iterations: int = 1000
    ) -> List[float]:
        """Measure baseline performance without tracing."""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            await operation()
            times.append(time.perf_counter() - start)
        
        return times
    
    async def measure_traced_performance(
        self,
        operation,
        tracer: OpenTelemetryTracer,
        operation_name: str,
        iterations: int = 1000
    ) -> List[float]:
        """Measure performance with tracing enabled."""
        times = []
        
        # Decorate operation with tracing
        traced_operation = tracer.track_performance(operation_name)(operation)
        
        for _ in range(iterations):
            start = time.perf_counter()
            await traced_operation()
            times.append(time.perf_counter() - start)
        
        return times
    
    @pytest.mark.asyncio
    async def test_order_execution_overhead(self, tracer):
        """Test overhead for order execution operations."""
        iterations = 1000
        
        # Measure baseline
        baseline_times = await self.measure_baseline_performance(
            self.simulate_order_execution,
            iterations
        )
        
        # Measure with tracing
        traced_times = await self.measure_traced_performance(
            self.simulate_order_execution,
            tracer,
            "order_execution",
            iterations
        )
        
        metrics = PerformanceMetrics(
            operation_name="Order Execution",
            iterations=iterations,
            baseline_times=baseline_times,
            traced_times=traced_times
        )
        
        print("\n" + metrics.get_summary())
        
        # Assert overhead is less than 1%
        assert metrics.overhead_percent < 1.0, f"Overhead {metrics.overhead_percent:.2f}% exceeds 1% limit"
    
    @pytest.mark.asyncio
    async def test_risk_check_overhead(self, tracer):
        """Test overhead for risk check operations."""
        iterations = 1000
        
        # Measure baseline
        baseline_times = await self.measure_baseline_performance(
            self.simulate_risk_check,
            iterations
        )
        
        # Measure with tracing
        traced_times = await self.measure_traced_performance(
            self.simulate_risk_check,
            tracer,
            "risk_check",
            iterations
        )
        
        metrics = PerformanceMetrics(
            operation_name="Risk Check",
            iterations=iterations,
            baseline_times=baseline_times,
            traced_times=traced_times
        )
        
        print("\n" + metrics.get_summary())
        
        # Assert overhead is less than 1%
        assert metrics.overhead_percent < 1.0, f"Overhead {metrics.overhead_percent:.2f}% exceeds 1% limit"
    
    @pytest.mark.asyncio
    async def test_market_data_overhead(self, tracer):
        """Test overhead for market data processing."""
        iterations = 2000  # More iterations for faster operations
        
        # Measure baseline
        baseline_times = await self.measure_baseline_performance(
            self.simulate_market_data_processing,
            iterations
        )
        
        # Measure with tracing
        traced_times = await self.measure_traced_performance(
            self.simulate_market_data_processing,
            tracer,
            "market_data_processing",
            iterations
        )
        
        metrics = PerformanceMetrics(
            operation_name="Market Data Processing",
            iterations=iterations,
            baseline_times=baseline_times,
            traced_times=traced_times
        )
        
        print("\n" + metrics.get_summary())
        
        # Assert overhead is less than 1%
        assert metrics.overhead_percent < 1.0, f"Overhead {metrics.overhead_percent:.2f}% exceeds 1% limit"
    
    @pytest.mark.asyncio
    async def test_database_query_overhead(self, tracer):
        """Test overhead for database operations."""
        iterations = 500  # Fewer iterations for slower operations
        
        # Measure baseline
        baseline_times = await self.measure_baseline_performance(
            self.simulate_database_query,
            iterations
        )
        
        # Measure with tracing
        traced_times = await self.measure_traced_performance(
            self.simulate_database_query,
            tracer,
            "database_query",
            iterations
        )
        
        metrics = PerformanceMetrics(
            operation_name="Database Query",
            iterations=iterations,
            baseline_times=baseline_times,
            traced_times=traced_times
        )
        
        print("\n" + metrics.get_summary())
        
        # Assert overhead is less than 1%
        assert metrics.overhead_percent < 1.0, f"Overhead {metrics.overhead_percent:.2f}% exceeds 1% limit"
    
    @pytest.mark.asyncio
    async def test_mixed_workload_overhead(self, tracer):
        """Test overhead with mixed workload simulating real usage."""
        iterations = 500
        
        async def mixed_workload():
            """Simulate mixed workload."""
            await self.simulate_order_execution()
            await self.simulate_risk_check()
            await self.simulate_market_data_processing()
            await self.simulate_database_query()
        
        # Measure baseline
        baseline_times = await self.measure_baseline_performance(
            mixed_workload,
            iterations
        )
        
        # Measure with tracing
        traced_times = await self.measure_traced_performance(
            mixed_workload,
            tracer,
            "mixed_workload",
            iterations
        )
        
        metrics = PerformanceMetrics(
            operation_name="Mixed Workload",
            iterations=iterations,
            baseline_times=baseline_times,
            traced_times=traced_times
        )
        
        print("\n" + metrics.get_summary())
        
        # Assert overhead is less than 1%
        assert metrics.overhead_percent < 1.0, f"Overhead {metrics.overhead_percent:.2f}% exceeds 1% limit"
    
    @pytest.mark.asyncio
    async def test_high_frequency_operations(self, tracer):
        """Test overhead for high-frequency operations."""
        iterations = 10000
        
        async def high_freq_operation():
            """Simulate very fast operation."""
            # Just a simple calculation, no I/O
            result = sum(range(100))
            return result
        
        # Measure baseline
        baseline_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await high_freq_operation()
            baseline_times.append(time.perf_counter() - start)
        
        # Measure with tracing
        traced_operation = tracer.track_performance("high_freq_op")(high_freq_operation)
        traced_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await traced_operation()
            traced_times.append(time.perf_counter() - start)
        
        metrics = PerformanceMetrics(
            operation_name="High Frequency Operation",
            iterations=iterations,
            baseline_times=baseline_times,
            traced_times=traced_times
        )
        
        print("\n" + metrics.get_summary())
        
        # For very fast operations, allow slightly higher overhead but still under 2%
        assert metrics.overhead_percent < 2.0, f"Overhead {metrics.overhead_percent:.2f}% exceeds 2% limit for high-freq ops"
    
    def test_sampling_effectiveness(self, tracer):
        """Test that sampling reduces overhead effectively."""
        # With 1% sampling rate, only 1 in 100 spans should be sampled
        sampled_count = 0
        total_count = 10000
        
        for i in range(total_count):
            # Create span manually to check sampling
            with tracer.create_span(f"test_span_{i}") as span:
                if span.is_recording():
                    sampled_count += 1
        
        sampling_ratio = sampled_count / total_count
        print(f"\nSampling ratio: {sampling_ratio:.4f} (expected ~0.01)")
        
        # Allow some variance but should be close to 1%
        assert 0.005 < sampling_ratio < 0.02, f"Sampling ratio {sampling_ratio} not within expected range"
    
    @pytest.mark.asyncio
    async def test_memory_overhead(self, tracer):
        """Test memory overhead of tracing."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        
        # Create many spans to test memory usage
        spans = []
        for i in range(1000):
            span = tracer.create_span(f"memory_test_{i}")
            spans.append(span)
        
        # Get approximate memory usage
        total_size = sum(sys.getsizeof(span) for span in spans)
        avg_size = total_size / len(spans)
        
        print(f"\nAverage span memory: {avg_size:.0f} bytes")
        
        # Each span should use less than 10KB
        assert avg_size < 10240, f"Span memory usage {avg_size} bytes exceeds 10KB limit"
        
        # Clean up
        for span in spans:
            span.end()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_overhead(self, tracer):
        """Test overhead with concurrent operations."""
        concurrent_count = 100
        iterations_per_task = 10
        
        async def concurrent_operation(task_id: int):
            """Operation to run concurrently."""
            for _ in range(iterations_per_task):
                await self.simulate_order_execution(0.001)
        
        # Baseline: Run operations concurrently without tracing
        start = time.perf_counter()
        tasks = [concurrent_operation(i) for i in range(concurrent_count)]
        await asyncio.gather(*tasks)
        baseline_time = time.perf_counter() - start
        
        # With tracing: Run operations concurrently with tracing
        traced_op = tracer.track_performance("concurrent_op")(concurrent_operation)
        start = time.perf_counter()
        tasks = [traced_op(i) for i in range(concurrent_count)]
        await asyncio.gather(*tasks)
        traced_time = time.perf_counter() - start
        
        overhead_percent = ((traced_time - baseline_time) / baseline_time) * 100
        
        print(f"\nConcurrent Operations:")
        print(f"  Tasks: {concurrent_count}")
        print(f"  Baseline: {baseline_time:.3f}s")
        print(f"  Traced: {traced_time:.3f}s")
        print(f"  Overhead: {overhead_percent:.2f}%")
        
        # Assert overhead is less than 1%
        assert overhead_percent < 1.0, f"Concurrent overhead {overhead_percent:.2f}% exceeds 1% limit"


def generate_performance_report(metrics_list: List[PerformanceMetrics]) -> str:
    """Generate performance test report."""
    report = []
    report.append("=" * 60)
    report.append("TRACING OVERHEAD PERFORMANCE REPORT")
    report.append("=" * 60)
    
    for metrics in metrics_list:
        report.append(metrics.get_summary())
        report.append("")
    
    # Overall summary
    total_overhead = mean([m.overhead_percent for m in metrics_list])
    max_overhead = max([m.overhead_percent for m in metrics_list])
    
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"Average overhead: {total_overhead:.3f}%")
    report.append(f"Maximum overhead: {max_overhead:.3f}%")
    report.append(f"Target: <1%")
    report.append(f"Status: {'PASS' if max_overhead < 1.0 else 'FAIL'}")
    report.append("=" * 60)
    
    return "\n".join(report)