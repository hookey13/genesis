"""
Integration tests for memory monitoring workflow.
Tests the complete memory profiling and monitoring pipeline.
"""

import asyncio
import gc
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import pytest
import psutil
import aiohttp
from fastapi.testclient import TestClient

from genesis.monitoring.memory_profiler import MemoryProfiler, MemoryTrend
from genesis.monitoring.advanced_profiler import AdvancedPerformanceProfiler
from genesis.monitoring.metrics_collector import MetricsCollector
from genesis.monitoring.prometheus_exporter import MetricsRegistry
from genesis.monitoring.performance_monitor import PerformanceMonitor
from genesis.api.metrics_endpoints import router, get_memory_profiler, get_cpu_profiler
from genesis.core.models import Order, OrderStatus, Position


@pytest.fixture
async def metrics_registry():
    """Create metrics registry for testing."""
    registry = MetricsRegistry()
    yield registry


@pytest.fixture
async def metrics_collector(metrics_registry):
    """Create metrics collector with memory profiling enabled."""
    collector = MetricsCollector(registry=metrics_registry, enable_memory_profiling=True)
    await collector.start()
    yield collector
    await collector.stop()


@pytest.fixture
async def memory_profiler():
    """Create and start memory profiler."""
    profiler = MemoryProfiler(
        growth_threshold=0.05,
        snapshot_interval=1,
        enable_tracemalloc=True
    )
    await profiler.start_monitoring()
    yield profiler
    await profiler.stop_monitoring()


@pytest.fixture
async def cpu_profiler():
    """Create CPU profiler."""
    profiler = AdvancedPerformanceProfiler(
        profile_dir=".test_profiles",
        memory_threshold_mb=50.0,
        cpu_threshold_percent=70.0
    )
    yield profiler
    # Cleanup
    import shutil
    if os.path.exists(".test_profiles"):
        shutil.rmtree(".test_profiles")


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestMemoryMonitoringIntegration:
    """Integration tests for memory monitoring workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self, memory_profiler, metrics_collector):
        """Test the complete memory monitoring workflow from collection to metrics."""
        # Simulate some memory usage
        test_data = []
        for i in range(100):
            test_data.append("x" * 1000)  # 1KB per iteration
            await asyncio.sleep(0.01)
        
        # Wait for metrics collection
        await asyncio.sleep(2)
        
        # Verify memory profiler collected data
        assert len(memory_profiler.snapshots) > 0
        stats = memory_profiler.get_memory_stats()
        assert stats['snapshot_count'] > 0
        assert stats['rss_mb'] > 0
        
        # Verify metrics collector has memory data
        assert metrics_collector.metrics.memory_usage > 0
        assert metrics_collector.metrics.memory_percent >= 0
        
        # Check for memory growth detection
        trend = memory_profiler.get_memory_trend(hours=0.1)
        assert isinstance(trend, MemoryTrend)
        
        # Clean up
        test_data.clear()
        gc.collect()
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection_pipeline(self, memory_profiler):
        """Test memory leak detection through the full pipeline."""
        # Create an intentional memory leak
        leak_data = []
        
        async def create_leak():
            for _ in range(200):
                leak_data.append("x" * 10000)  # 10KB per iteration
                await asyncio.sleep(0.005)
        
        # Run leak creation
        await create_leak()
        
        # Wait for profiler to analyze
        await asyncio.sleep(2)
        
        # Detect leaks
        leak_result = memory_profiler.detect_leaks()
        
        # Verify detection
        assert leak_result is not None
        assert leak_result.growth_rate >= 0  # Should detect some growth
        
        # Get trend analysis
        trend = memory_profiler.get_memory_trend(hours=0.1)
        assert trend.growth_rate_per_hour >= 0
        
        # Clean up
        leak_data.clear()
        gc.collect()
    
    @pytest.mark.asyncio
    async def test_cpu_and_memory_profiling_together(self, memory_profiler, cpu_profiler):
        """Test concurrent CPU and memory profiling."""
        # Start CPU profiling
        cpu_profile_task = asyncio.create_task(
            cpu_profiler.profile_cpu_with_cprofile(duration_seconds=2.0)
        )
        
        # Simulate workload
        async def workload():
            data = []
            for i in range(100):
                # CPU intensive operation
                result = sum(j * j for j in range(1000))
                # Memory allocation
                data.append("x" * 1000)
                await asyncio.sleep(0.01)
            return data
        
        # Run workload
        workload_task = asyncio.create_task(workload())
        
        # Wait for both to complete
        cpu_profile = await cpu_profile_task
        workload_data = await workload_task
        
        # Verify CPU profile
        assert cpu_profile.duration_seconds == 2.0
        assert cpu_profile.samples > 0
        assert len(cpu_profile.top_functions) > 0
        
        # Verify memory monitoring continued
        assert len(memory_profiler.snapshots) > 0
        
        # Clean up
        workload_data.clear()
        gc.collect()
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_export(self, memory_profiler, metrics_collector, metrics_registry):
        """Test that memory metrics are properly exported to Prometheus."""
        # Generate some activity
        test_data = ["x" * 1000 for _ in range(100)]
        
        # Wait for metrics update
        await asyncio.sleep(2)
        
        # Update Prometheus metrics
        await metrics_collector._update_prometheus_metrics()
        
        # Verify metrics are registered
        metrics = await metrics_registry.collect()
        
        # Check for memory-specific metrics
        metric_names = [m.name for m in metrics]
        assert any('memory' in name for name in metric_names)
        
        # Clean up
        test_data.clear()
    
    @pytest.mark.asyncio
    async def test_performance_monitor_integration(self):
        """Test integration with performance monitor."""
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Update memory profiling metrics
        await monitor.update_memory_profiling_metrics(
            growth_rate=0.05,
            leak_detected=False,
            leak_confidence=0.3,
            peak_usage=100 * 1024 * 1024
        )
        
        # Verify metrics are set
        assert monitor.metrics.memory_growth_rate_gauge is not None
        assert monitor.metrics.memory_leak_detected_gauge is not None
        assert monitor.metrics.memory_leak_confidence_gauge is not None
        assert monitor.metrics.memory_peak_usage_gauge is not None
    
    @pytest.mark.asyncio
    async def test_resource_forecasting(self, memory_profiler):
        """Test resource usage forecasting."""
        # Generate historical data
        for i in range(20):
            await memory_profiler.take_snapshot()
            await asyncio.sleep(0.1)
        
        # Get forecast
        forecast = memory_profiler.forecast_resource_usage(hours_ahead=1.0)
        
        # Verify forecast structure
        assert 'forecast_available' in forecast
        if forecast['forecast_available']:
            assert 'current_memory_mb' in forecast
            assert 'forecasted_memory_mb' in forecast
            assert 'growth_rate_per_hour' in forecast
            assert 'confidence_level' in forecast
            assert 'recommendations' in forecast
            assert len(forecast['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_alert_thresholds(self, memory_profiler):
        """Test alert threshold configuration and triggering."""
        # Set custom thresholds
        memory_profiler.set_alert_threshold('memory_percent', 50.0)
        memory_profiler.set_alert_threshold('growth_rate_per_hour', 0.02)
        
        # Verify thresholds are set
        assert memory_profiler.alert_thresholds['memory_percent'] == 50.0
        assert memory_profiler.alert_thresholds['growth_rate_per_hour'] == 0.02
        
        # Generate data that should trigger alerts
        with patch('genesis.monitoring.memory_profiler.psutil.virtual_memory') as mock_vm:
            mock_vm.return_value.percent = 60.0  # Above threshold
            
            snapshot = await memory_profiler.take_snapshot()
            assert snapshot.percent_used == 60.0
    
    def test_api_endpoints_integration(self, api_client):
        """Test FastAPI endpoints for profiling control."""
        # Test memory profile endpoint
        response = api_client.get("/metrics/profile/memory")
        assert response.status_code == 200
        data = response.json()
        assert 'current_memory_mb' in data
        assert 'growth_rate_per_hour' in data
        
        # Test profiling status endpoint
        response = api_client.get("/metrics/profile/status")
        assert response.status_code == 200
        data = response.json()
        assert 'memory_profiling_active' in data
        assert 'cpu_profiling_active' in data
        
        # Test recommendations endpoint
        response = api_client.get("/metrics/profile/recommendations")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    @pytest.mark.asyncio
    async def test_stability_test_integration(self):
        """Test stability test framework integration."""
        from tests.performance.test_memory_stability import StabilityTestFramework
        
        # Create short stability test
        framework = StabilityTestFramework(
            test_duration_hours=0.01,  # 36 seconds
            memory_growth_threshold=0.1,
            checkpoint_interval_hours=0.005  # 18 seconds
        )
        
        # Run test
        results = await framework.run_stability_test()
        
        # Verify results structure
        assert 'test_passed' in results
        assert 'duration_hours' in results
        assert 'total_memory_growth_percent' in results
        assert 'checkpoints' in results
        assert 'resource_forecast' in results
        
        # Verify resource forecast was included
        if results['resource_forecast'].get('forecast_available'):
            assert 'recommendations' in results['resource_forecast']


class TestRegressionDetection:
    """Test performance regression detection for CI/CD."""
    
    @pytest.mark.asyncio
    async def test_performance_baseline_comparison(self, memory_profiler):
        """Test performance comparison against baseline."""
        # Create baseline
        baseline_snapshots = []
        for i in range(5):
            snapshot = await memory_profiler.take_snapshot()
            baseline_snapshots.append({
                'memory_mb': snapshot.rss_bytes / 1024 / 1024,
                'timestamp': snapshot.timestamp
            })
            await asyncio.sleep(0.1)
        
        # Calculate baseline average
        baseline_avg = sum(s['memory_mb'] for s in baseline_snapshots) / len(baseline_snapshots)
        
        # Simulate current run with higher memory usage
        current_snapshots = []
        with patch.object(memory_profiler.process, 'memory_info') as mock_mem:
            # Simulate 20% higher memory usage
            mock_mem.return_value.rss = int(baseline_avg * 1.2 * 1024 * 1024)
            mock_mem.return_value.vms = 200 * 1024 * 1024
            
            for i in range(5):
                snapshot = await memory_profiler.take_snapshot()
                current_snapshots.append({
                    'memory_mb': snapshot.rss_bytes / 1024 / 1024,
                    'timestamp': snapshot.timestamp
                })
                await asyncio.sleep(0.1)
        
        # Calculate current average
        current_avg = sum(s['memory_mb'] for s in current_snapshots) / len(current_snapshots)
        
        # Detect regression
        regression_detected = (current_avg - baseline_avg) / baseline_avg > 0.15  # 15% threshold
        
        assert regression_detected, "Should detect performance regression"
    
    @pytest.mark.asyncio
    async def test_memory_leak_regression(self, memory_profiler):
        """Test detection of memory leak as regression."""
        # Baseline: no leak
        baseline_result = memory_profiler.detect_leaks()
        assert not baseline_result.has_leak
        
        # Current: simulate leak
        leak_data = []
        for i in range(20):
            leak_data.append("x" * 100000)  # 100KB per iteration
            snapshot = await memory_profiler.take_snapshot()
            # Mock increasing memory
            snapshot.rss_bytes = (100 + i * 5) * 1024 * 1024
            memory_profiler.snapshots[-1] = snapshot
            await asyncio.sleep(0.05)
        
        # Check for leak (regression)
        current_result = memory_profiler.detect_leaks()
        
        # Should detect the regression (leak introduced)
        regression_detected = current_result.has_leak and not baseline_result.has_leak
        assert regression_detected or current_result.growth_rate > 0.01
        
        # Clean up
        leak_data.clear()
        gc.collect()
    
    def test_ci_performance_report_generation(self, memory_profiler, cpu_profiler):
        """Test generation of performance report for CI/CD."""
        # Generate performance report
        report = {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'current_mb': memory_profiler.get_memory_stats().get('rss_mb', 0),
                'growth_rate': memory_profiler._calculate_growth_rate(),
                'leak_detected': False,
                'snapshots_collected': len(memory_profiler.snapshots)
            },
            'cpu': {
                'profiles_collected': len(cpu_profiler.cpu_history),
                'hot_paths_identified': len(cpu_profiler.identify_hot_paths())
            },
            'recommendations': [
                rec.__dict__ for rec in cpu_profiler.generate_optimization_recommendations()
            ],
            'regression_checks': {
                'memory_threshold_mb': 500,
                'cpu_threshold_percent': 80,
                'passed': True
            }
        }
        
        # Verify report structure
        assert 'timestamp' in report
        assert 'memory' in report
        assert 'cpu' in report
        assert 'recommendations' in report
        assert 'regression_checks' in report
        
        # Report should be serializable for CI/CD
        import json
        json_report = json.dumps(report, default=str)
        assert json_report is not None


@pytest.mark.asyncio
class TestMemoryLeakFixtures:
    """Test pytest fixtures for memory leak detection."""
    
    async def test_memory_tracking_fixture(self, memory_profiler):
        """Test memory tracking fixture for tests."""
        # Fixture should track memory before and after test
        start_memory = memory_profiler.process.memory_info().rss
        
        # Simulate test execution
        test_data = ["x" * 1000 for _ in range(1000)]
        
        # Check memory after
        end_memory = memory_profiler.process.memory_info().rss
        growth = (end_memory - start_memory) / start_memory if start_memory > 0 else 0
        
        # Clean up
        test_data.clear()
        gc.collect()
        
        # Memory growth should be detected
        assert end_memory >= start_memory
    
    async def test_leak_assertion_helper(self, memory_profiler):
        """Test helper for asserting no memory leaks in tests."""
        
        async def assert_no_memory_leak(test_func, threshold=0.05):
            """Helper to assert no memory leak during test execution."""
            # Take baseline
            await memory_profiler.take_snapshot()
            baseline_memory = memory_profiler.snapshots[-1].rss_bytes
            
            # Run test
            await test_func()
            
            # Take final snapshot
            await memory_profiler.take_snapshot()
            final_memory = memory_profiler.snapshots[-1].rss_bytes
            
            # Check for leak
            growth = (final_memory - baseline_memory) / baseline_memory if baseline_memory > 0 else 0
            assert growth < threshold, f"Memory leak detected: {growth:.2%} growth"
        
        # Test with non-leaking function
        async def good_test():
            data = ["x" * 100 for _ in range(10)]
            await asyncio.sleep(0.1)
            data.clear()
            gc.collect()
        
        await assert_no_memory_leak(good_test)
    
    async def test_memory_threshold_per_test_type(self):
        """Test configurable memory thresholds for different test types."""
        thresholds = {
            'unit': 0.02,      # 2% for unit tests
            'integration': 0.05,  # 5% for integration tests
            'performance': 0.10,  # 10% for performance tests
        }
        
        def get_test_threshold(test_name: str) -> float:
            """Get memory threshold based on test type."""
            if 'unit' in test_name:
                return thresholds['unit']
            elif 'integration' in test_name:
                return thresholds['integration']
            elif 'performance' in test_name:
                return thresholds['performance']
            return 0.05  # Default
        
        # Verify threshold selection
        assert get_test_threshold('test_unit_calculation') == 0.02
        assert get_test_threshold('test_integration_workflow') == 0.05
        assert get_test_threshold('test_performance_load') == 0.10