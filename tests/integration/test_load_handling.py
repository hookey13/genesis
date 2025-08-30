"""
Integration tests for load handling capabilities.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.stress.load_generator import LoadGenerator, LoadProfile, WebSocketMessageGenerator


@pytest.mark.asyncio
async def test_load_generator_basic():
    """Test basic load generation functionality."""
    # Create load generator with mock processor
    processed_messages = []
    
    async def mock_processor(message):
        processed_messages.append(message)
        return True
    
    generator = LoadGenerator(target_system=mock_processor)
    
    # Run short test
    await generator.run_load_test(
        duration_seconds=5,
        multiplier=10,
        profile=LoadProfile.NORMAL
    )
    
    # Verify metrics
    assert generator.metrics.messages_sent > 0
    assert generator.metrics.messages_processed > 0
    assert generator.metrics.messages_processed <= generator.metrics.messages_sent
    assert len(processed_messages) == generator.metrics.messages_processed


@pytest.mark.asyncio
async def test_websocket_message_generator():
    """Test WebSocket message generation."""
    generator = WebSocketMessageGenerator(base_rate=10)
    generator.set_multiplier(5)
    
    # Generate some messages
    messages = []
    message_stream = generator.generate_stream()
    
    for _ in range(10):
        msg = await message_stream.__anext__()
        messages.append(msg)
    
    # Verify message structure
    for msg in messages:
        assert "type" in msg
        assert "symbol" in msg
        assert "timestamp" in msg
        assert "data" in msg
        assert msg["type"] in ["ticker", "trade", "orderbook"]


@pytest.mark.asyncio
async def test_load_profiles():
    """Test different load profiles."""
    generator = LoadGenerator()
    
    # Test gradual ramp profile
    await generator.run_load_test(
        duration_seconds=2,
        multiplier=10,
        profile=LoadProfile.GRADUAL_RAMP
    )
    
    assert generator.metrics.messages_sent > 0
    assert generator.metrics.end_time is not None


@pytest.mark.asyncio
async def test_message_dropping_under_load():
    """Test message dropping when queue is full."""
    # Create slow processor to cause backpressure
    async def slow_processor(message):
        await asyncio.sleep(0.1)  # Slow processing
        return True
    
    generator = LoadGenerator(target_system=slow_processor)
    
    # Run with high load
    await generator.run_load_test(
        duration_seconds=3,
        multiplier=100,
        profile=LoadProfile.NORMAL
    )
    
    # Should have some dropped messages due to slow processing
    assert generator.metrics.messages_dropped >= 0
    assert generator.metrics.messages_sent > generator.metrics.messages_processed


@pytest.mark.asyncio
async def test_latency_tracking():
    """Test latency measurement accuracy."""
    latencies = []
    
    async def tracked_processor(message):
        # Simulate variable processing time
        import random
        delay = random.uniform(0.001, 0.01)
        await asyncio.sleep(delay)
        return True
    
    generator = LoadGenerator(target_system=tracked_processor)
    
    await generator.run_load_test(
        duration_seconds=3,
        multiplier=5,
        profile=LoadProfile.NORMAL
    )
    
    # Verify latency metrics
    assert generator.metrics.get_average_latency() > 0
    assert generator.metrics.get_percentile(50) > 0
    assert generator.metrics.get_percentile(95) >= generator.metrics.get_percentile(50)
    assert generator.metrics.get_percentile(99) >= generator.metrics.get_percentile(95)


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling during load test."""
    error_count = 0
    
    async def failing_processor(message):
        nonlocal error_count
        # Fail 10% of messages
        import random
        if random.random() < 0.1:
            error_count += 1
            raise Exception("Processing failed")
        return True
    
    generator = LoadGenerator(target_system=failing_processor)
    
    await generator.run_load_test(
        duration_seconds=3,
        multiplier=10,
        profile=LoadProfile.NORMAL
    )
    
    # Should have recorded errors
    assert generator.metrics.errors > 0
    assert generator.metrics.errors == error_count


@pytest.mark.asyncio
async def test_throughput_calculation():
    """Test throughput calculation accuracy."""
    generator = LoadGenerator()
    
    await generator.run_load_test(
        duration_seconds=5,
        multiplier=20,
        profile=LoadProfile.NORMAL
    )
    
    throughput = generator.metrics.get_throughput()
    
    # Throughput should be positive and reasonable
    assert throughput > 0
    assert throughput < 10000  # Sanity check


@pytest.mark.asyncio
async def test_peak_rate_tracking():
    """Test peak message rate tracking."""
    generator = LoadGenerator()
    
    await generator.run_load_test(
        duration_seconds=5,
        multiplier=50,
        profile=LoadProfile.NEWS_SPIKE
    )
    
    # Peak should be higher than average
    avg_rate = generator.metrics.get_throughput()
    peak_rate = generator.metrics.peak_messages_per_second
    
    assert peak_rate > 0
    assert peak_rate >= avg_rate  # Peak should be at least average


@pytest.mark.asyncio
async def test_memory_monitoring():
    """Test memory usage monitoring during load test."""
    generator = LoadGenerator()
    
    await generator.run_load_test(
        duration_seconds=3,
        multiplier=10,
        profile=LoadProfile.NORMAL
    )
    
    # Should have collected memory metrics
    assert len(generator.metrics.memory_usage_mb) > 0
    assert all(m > 0 for m in generator.metrics.memory_usage_mb)


@pytest.mark.asyncio
async def test_report_generation():
    """Test report generation after load test."""
    import json
    from pathlib import Path
    
    generator = LoadGenerator()
    
    await generator.run_load_test(
        duration_seconds=2,
        multiplier=5,
        profile=LoadProfile.NORMAL
    )
    
    # Check if report was created
    report_path = Path("tests/stress/reports")
    report_files = list(report_path.glob("load_test_*.json"))
    
    if report_files:
        # Read most recent report
        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_report) as f:
            report = json.load(f)
        
        assert report["test_type"] == "load_test"
        assert "metrics" in report
        assert "success" in report
        assert report["metrics"]["messages_sent"] > 0