"""Unit tests for memory profiler."""

import pytest
from tests.stress.memory_profiler import MemoryProfiler, MemoryMetrics, MemorySnapshot
from datetime import datetime


@pytest.mark.asyncio
async def test_memory_profiler_init():
    """Test memory profiler initialization."""
    profiler = MemoryProfiler(snapshot_interval_seconds=60)
    assert profiler.snapshot_interval == 60
    assert not profiler.running
    assert not profiler.tracemalloc_started


@pytest.mark.asyncio
async def test_memory_snapshot():
    """Test memory snapshot creation."""
    profiler = MemoryProfiler()
    snapshot = await profiler.take_snapshot()
    
    assert isinstance(snapshot, MemorySnapshot)
    assert snapshot.rss_mb > 0
    assert snapshot.vms_mb > 0
    assert isinstance(snapshot.gc_stats, dict)
    assert isinstance(snapshot.object_counts, dict)


@pytest.mark.asyncio
async def test_memory_metrics():
    """Test memory metrics tracking."""
    metrics = MemoryMetrics()
    
    # Add snapshots
    snapshot1 = MemorySnapshot(
        timestamp=datetime.now(),
        rss_mb=100,
        vms_mb=200,
        heap_mb=50,
        top_allocations=[],
        gc_stats={},
        object_counts={"dict": 1000}
    )
    
    metrics.add_snapshot(snapshot1)
    
    assert len(metrics.snapshots) == 1
    assert metrics.peak_memory_mb == 100
    assert metrics.average_memory_mb == 100


@pytest.mark.asyncio
async def test_leak_detection():
    """Test memory leak detection."""
    metrics = MemoryMetrics()
    
    # Simulate growing memory
    for i in range(15):
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=100 + i * 10,  # Growing memory
            vms_mb=200,
            heap_mb=50,
            top_allocations=[],
            gc_stats={},
            object_counts={"dict": 1000 + i * 100}
        )
        metrics.add_snapshot(snapshot)
    
    leaks = metrics.detect_leaks()
    
    assert len(leaks) > 0
    assert any(leak["type"] == "continuous_growth" for leak in leaks)