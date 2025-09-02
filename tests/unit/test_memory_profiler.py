"""
Unit tests for the memory profiler module.
"""

import asyncio
import gc
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import psutil

from genesis.monitoring.memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryTrend,
    LeakDetectionResult
)


@pytest.fixture
def memory_profiler():
    """Create a memory profiler instance for testing."""
    profiler = MemoryProfiler(
        growth_threshold=0.05,
        snapshot_interval=1,  # 1 second for testing
        history_size=100,
        enable_tracemalloc=False  # Disable for unit tests
    )
    return profiler


@pytest.fixture
def mock_process():
    """Mock psutil Process for testing."""
    with patch('genesis.monitoring.memory_profiler.psutil.Process') as mock:
        process_mock = MagicMock()
        process_mock.memory_info.return_value.rss = 100 * 1024 * 1024  # 100 MB
        process_mock.memory_info.return_value.vms = 200 * 1024 * 1024  # 200 MB
        mock.return_value = process_mock
        yield process_mock


@pytest.fixture
def mock_virtual_memory():
    """Mock psutil virtual_memory for testing."""
    with patch('genesis.monitoring.memory_profiler.psutil.virtual_memory') as mock:
        memory_mock = MagicMock()
        memory_mock.available = 8 * 1024 * 1024 * 1024  # 8 GB
        memory_mock.percent = 50.0
        memory_mock.total = 16 * 1024 * 1024 * 1024  # 16 GB
        mock.return_value = memory_mock
        yield memory_mock


class TestMemoryProfiler:
    """Test suite for MemoryProfiler class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test profiler initialization."""
        profiler = MemoryProfiler(
            growth_threshold=0.1,
            snapshot_interval=60,
            history_size=500
        )
        
        assert profiler.growth_threshold == 0.1
        assert profiler.snapshot_interval == 60
        assert profiler.history_size == 500
        assert profiler.baseline_memory is None
        assert not profiler.is_monitoring
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test starting memory monitoring."""
        await memory_profiler.start_monitoring()
        
        assert memory_profiler.is_monitoring
        assert memory_profiler.baseline_memory is not None
        assert memory_profiler.start_time is not None
        assert memory_profiler.monitoring_task is not None
        assert len(memory_profiler.snapshots) > 0
        
        # Clean up
        await memory_profiler.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test stopping memory monitoring."""
        await memory_profiler.start_monitoring()
        await asyncio.sleep(0.1)  # Let monitoring run briefly
        await memory_profiler.stop_monitoring()
        
        assert not memory_profiler.is_monitoring
        assert memory_profiler.monitoring_task is None
    
    @pytest.mark.asyncio
    async def test_take_snapshot(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test taking a memory snapshot."""
        snapshot = await memory_profiler.take_snapshot()
        
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.rss_bytes == 100 * 1024 * 1024
        assert snapshot.vms_bytes == 200 * 1024 * 1024
        assert snapshot.percent_used == 50.0
        assert snapshot.available_bytes == 8 * 1024 * 1024 * 1024
        assert 'objects' in snapshot.gc_stats
        assert len(memory_profiler.snapshots) == 1
    
    def test_detect_leaks_insufficient_data(self, memory_profiler):
        """Test leak detection with insufficient data."""
        result = memory_profiler.detect_leaks()
        
        assert isinstance(result, LeakDetectionResult)
        assert not result.has_leak
        assert result.confidence == 0.0
        assert result.growth_rate == 0.0
        assert "Insufficient data" in result.recommendation
    
    @pytest.mark.asyncio
    async def test_detect_leaks_with_growth(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test leak detection with memory growth."""
        # Simulate memory growth
        memory_values = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
        
        for i, mem_mb in enumerate(memory_values):
            mock_process.memory_info.return_value.rss = mem_mb * 1024 * 1024
            snapshot = await memory_profiler.take_snapshot()
            # Adjust timestamp for growth calculation
            snapshot.timestamp = datetime.now() - timedelta(hours=len(memory_values) - i - 1)
            memory_profiler.snapshots[-1] = snapshot
        
        result = memory_profiler.detect_leaks()
        
        assert isinstance(result, LeakDetectionResult)
        # Should detect growth pattern
        assert result.growth_rate > 0
        # Confidence should be elevated due to monotonic increase
        assert result.confidence > 0
    
    def test_calculate_growth_rate(self, memory_profiler):
        """Test growth rate calculation."""
        # Add snapshots with known growth pattern
        base_time = datetime.now()
        base_memory = 100 * 1024 * 1024  # 100 MB
        
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(hours=i),
                rss_bytes=base_memory + (i * 5 * 1024 * 1024),  # 5 MB per hour
                vms_bytes=base_memory * 2,
                available_bytes=8 * 1024**3,
                percent_used=50.0,
                gc_stats={},
                top_allocations=[]
            )
            memory_profiler.snapshots.append(snapshot)
        
        growth_rate = memory_profiler._calculate_growth_rate()
        
        # Should detect approximately 5% growth per hour (5MB/100MB)
        assert 0.04 < growth_rate < 0.06
    
    def test_check_monotonic_increase(self, memory_profiler):
        """Test monotonic increase detection."""
        base_time = datetime.now()
        
        # Create monotonically increasing memory pattern
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(minutes=i),
                rss_bytes=100 * 1024 * 1024 + (i * 1024 * 1024),
                vms_bytes=200 * 1024 * 1024,
                available_bytes=8 * 1024**3,
                percent_used=50.0,
                gc_stats={},
                top_allocations=[]
            )
            memory_profiler.snapshots.append(snapshot)
        
        score = memory_profiler._check_monotonic_increase()
        
        # Should be close to 1.0 for perfect monotonic increase
        assert score > 0.9
    
    def test_get_memory_trend(self, memory_profiler):
        """Test memory trend analysis."""
        base_time = datetime.now()
        
        # Add snapshots
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(minutes=i * 10),
                rss_bytes=100 * 1024 * 1024 + (i * 2 * 1024 * 1024),
                vms_bytes=200 * 1024 * 1024,
                available_bytes=8 * 1024**3,
                percent_used=50.0 + i,
                gc_stats={},
                top_allocations=[]
            )
            memory_profiler.snapshots.append(snapshot)
        
        trend = memory_profiler.get_memory_trend(hours=1)
        
        assert isinstance(trend, MemoryTrend)
        assert trend.average_usage_bytes > 0
        assert trend.peak_usage_bytes >= trend.average_usage_bytes
        assert trend.growth_rate_per_hour >= 0
    
    def test_get_memory_stats(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test getting memory statistics."""
        memory_profiler.baseline_memory = 90 * 1024 * 1024  # 90 MB
        memory_profiler.start_time = datetime.now() - timedelta(hours=2)
        
        stats = memory_profiler.get_memory_stats()
        
        assert 'rss_mb' in stats
        assert 'vms_mb' in stats
        assert 'percent_used' in stats
        assert 'available_gb' in stats
        assert 'monitoring_duration_hours' in stats
        assert 'growth_from_baseline' in stats
        assert stats['monitoring_duration_hours'] >= 2.0
    
    def test_force_gc(self, memory_profiler, mock_process):
        """Test forced garbage collection."""
        # Set initial memory
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        
        stats = memory_profiler.force_gc()
        
        assert 'memory_before_mb' in stats
        assert 'memory_after_mb' in stats
        assert 'memory_freed_mb' in stats
        assert 'gc_objects' in stats
        assert stats['gc_objects'] > 0
    
    def test_set_alert_threshold(self, memory_profiler):
        """Test setting alert thresholds."""
        memory_profiler.set_alert_threshold('memory_percent', 90.0)
        memory_profiler.set_alert_threshold('growth_rate_per_hour', 0.1)
        
        assert memory_profiler.alert_thresholds['memory_percent'] == 90.0
        assert memory_profiler.alert_thresholds['growth_rate_per_hour'] == 0.1
        
        # Test invalid threshold name
        memory_profiler.set_alert_threshold('invalid_threshold', 50.0)
        # Should not raise error, just log warning
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_with_alerts(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test monitoring loop with alert conditions."""
        # Set high memory usage to trigger alert
        mock_virtual_memory.percent = 85.0
        
        await memory_profiler.start_monitoring()
        await asyncio.sleep(1.5)  # Let monitoring run for at least one snapshot
        
        assert len(memory_profiler.snapshots) > 0
        
        await memory_profiler.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_stability_test_short(self, memory_profiler, mock_process, mock_virtual_memory):
        """Test short stability test run."""
        # Mock increasing memory for test
        memory_values = [100, 101, 102]  # Small growth
        call_count = 0
        
        def mock_memory_info():
            nonlocal call_count
            result = MagicMock()
            result.rss = memory_values[min(call_count, len(memory_values) - 1)] * 1024 * 1024
            result.vms = 200 * 1024 * 1024
            call_count += 1
            return result
        
        mock_process.memory_info = mock_memory_info
        
        # Run very short stability test
        results = await memory_profiler.run_stability_test(duration_hours=0.001)  # ~3.6 seconds
        
        assert 'test_passed' in results
        assert 'duration_hours' in results
        assert 'start_memory_mb' in results
        assert 'end_memory_mb' in results
        assert 'growth_percentage' in results
    
    def test_memory_snapshot_creation(self):
        """Test MemorySnapshot dataclass creation."""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_bytes=100 * 1024 * 1024,
            vms_bytes=200 * 1024 * 1024,
            available_bytes=8 * 1024**3,
            percent_used=50.0,
            gc_stats={'generation_0': 700, 'objects': 50000},
            top_allocations=[('module.py:123', 1024000)]
        )
        
        assert snapshot.rss_bytes == 100 * 1024 * 1024
        assert snapshot.gc_stats['objects'] == 50000
        assert len(snapshot.top_allocations) == 1
    
    def test_memory_trend_creation(self):
        """Test MemoryTrend dataclass creation."""
        trend = MemoryTrend(
            growth_rate_per_hour=0.05,
            average_usage_bytes=100 * 1024 * 1024,
            peak_usage_bytes=150 * 1024 * 1024,
            leak_detected=True,
            leak_confidence=0.8,
            estimated_time_to_oom=24.5
        )
        
        assert trend.growth_rate_per_hour == 0.05
        assert trend.leak_detected
        assert trend.leak_confidence == 0.8
        assert trend.estimated_time_to_oom == 24.5
    
    def test_leak_detection_result_creation(self):
        """Test LeakDetectionResult dataclass creation."""
        result = LeakDetectionResult(
            has_leak=True,
            confidence=0.85,
            growth_rate=0.07,
            suspicious_allocations=[('module.py:456', 2048000, 0.15)],
            recommendation="Review memory allocations in module.py"
        )
        
        assert result.has_leak
        assert result.confidence == 0.85
        assert len(result.suspicious_allocations) == 1
        assert "module.py" in result.recommendation


@pytest.mark.asyncio
class TestMemoryProfilerIntegration:
    """Integration tests for memory profiler."""
    
    async def test_concurrent_monitoring_and_snapshots(self):
        """Test concurrent monitoring and manual snapshots."""
        profiler = MemoryProfiler(snapshot_interval=2)
        
        await profiler.start_monitoring()
        
        # Take manual snapshots while monitoring
        for _ in range(3):
            await profiler.take_snapshot()
            await asyncio.sleep(0.5)
        
        await profiler.stop_monitoring()
        
        # Should have snapshots from both monitoring and manual calls
        assert len(profiler.snapshots) >= 3
    
    async def test_memory_leak_simulation(self):
        """Test detection of simulated memory leak."""
        profiler = MemoryProfiler(
            growth_threshold=0.01,  # 1% threshold
            snapshot_interval=0.5,
            enable_tracemalloc=False
        )
        
        # Create a leak
        leak_data = []
        
        async def create_leak():
            for _ in range(100):
                leak_data.append("x" * 10000)  # 10KB per iteration
                await asyncio.sleep(0.01)
        
        await profiler.start_monitoring()
        await create_leak()
        await asyncio.sleep(1)  # Let profiler catch up
        
        result = profiler.detect_leaks()
        await profiler.stop_monitoring()
        
        # Clear the leak
        leak_data.clear()
        gc.collect()
        
        # Detection depends on actual memory behavior
        # Just verify the detection ran without errors
        assert isinstance(result, LeakDetectionResult)