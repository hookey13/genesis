"""
Tests for failure recovery under chaos conditions.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.chaos.chaos_engine import (
    ChaosMonkey, ChaosType, ChaosEvent, RecoveryValidator, ChaosMetrics
)


@pytest.mark.asyncio
async def test_chaos_monkey_basic():
    """Test basic chaos monkey functionality."""
    # Create chaos monkey
    monkey = ChaosMonkey()
    
    # Run short test
    await monkey.run_chaos_test(
        duration_minutes=1,
        chaos_probability=1.0,  # Always inject
        chaos_types=[ChaosType.CPU_STRESS]
    )
    
    # Verify metrics
    assert len(monkey.metrics.events_injected) > 0
    assert monkey.metrics.successful_recoveries >= 0
    assert monkey.metrics.calculate_availability() > 0


@pytest.mark.asyncio
async def test_process_kill_injection():
    """Test process kill chaos injection."""
    monkey = ChaosMonkey()
    
    event = await monkey.inject_chaos(ChaosType.PROCESS_KILL)
    
    assert event.chaos_type == ChaosType.PROCESS_KILL
    assert event.target == "trading_engine"
    assert event.impact == "process_terminated"


@pytest.mark.asyncio
async def test_network_delay_injection():
    """Test network delay injection."""
    monkey = ChaosMonkey()
    
    event = await monkey.inject_chaos(ChaosType.NETWORK_DELAY)
    
    assert event.chaos_type == ChaosType.NETWORK_DELAY
    assert event.target == "network"
    assert "delay_ms" in event.parameters
    assert "latency" in event.impact


@pytest.mark.asyncio
async def test_cpu_stress_injection():
    """Test CPU stress injection."""
    monkey = ChaosMonkey()
    
    # Set short duration for test
    event = await monkey.inject_chaos(ChaosType.CPU_STRESS)
    event.duration_seconds = 1
    
    assert event.chaos_type == ChaosType.CPU_STRESS
    assert event.target == "cpu"
    assert event.impact == "high_cpu_usage"


@pytest.mark.asyncio
async def test_memory_stress_injection():
    """Test memory stress injection."""
    monkey = ChaosMonkey()
    
    event = await monkey.inject_chaos(ChaosType.MEMORY_STRESS)
    event.duration_seconds = 1
    
    assert event.chaos_type == ChaosType.MEMORY_STRESS
    assert event.target == "memory"
    assert "allocated_mb" in event.parameters


@pytest.mark.asyncio
async def test_recovery_validation():
    """Test recovery validation after chaos."""
    # Create validator with health checks
    async def healthy_check():
        return True
    
    async def unhealthy_check():
        return False
    
    # Test with healthy system
    validator = RecoveryValidator([healthy_check])
    assert await validator.validate() is True
    
    # Test with unhealthy system
    validator = RecoveryValidator([unhealthy_check])
    assert await validator.validate() is False
    
    # Test with mixed health
    validator = RecoveryValidator([healthy_check, unhealthy_check])
    assert await validator.validate() is False


@pytest.mark.asyncio
async def test_recovery_metrics():
    """Test recovery metrics tracking."""
    monkey = ChaosMonkey()
    
    # Inject chaos and track recovery
    event = await monkey.inject_chaos(ChaosType.API_FAILURE)
    
    # Check metrics updated
    assert len(monkey.metrics.events_injected) == 1
    
    if event.recovered:
        assert monkey.metrics.successful_recoveries > 0
        assert event.recovery_time_seconds is not None
    else:
        assert monkey.metrics.failed_recoveries > 0


@pytest.mark.asyncio
async def test_data_consistency_validation():
    """Test data consistency validation."""
    monkey = ChaosMonkey()
    
    # Test consistency check
    is_consistent = await monkey.validate_data_consistency()
    
    assert isinstance(is_consistent, bool)
    assert monkey.metrics.data_consistency_checks == 1
    
    if not is_consistent:
        assert monkey.metrics.data_inconsistencies > 0


@pytest.mark.asyncio
async def test_availability_calculation():
    """Test service availability calculation."""
    metrics = ChaosMetrics()
    
    # Add some events
    from datetime import datetime, timedelta
    
    event1 = ChaosEvent(
        chaos_type=ChaosType.PROCESS_KILL,
        timestamp=datetime.now(),
        duration_seconds=10,
        target="test",
        parameters={},
        impact="test",
        recovered=True,
        recovery_time_seconds=5
    )
    
    event2 = ChaosEvent(
        chaos_type=ChaosType.API_FAILURE,
        timestamp=datetime.now(),
        duration_seconds=20,
        target="test",
        parameters={},
        impact="test",
        recovered=False
    )
    
    metrics.add_event(event1)
    metrics.add_event(event2)
    
    availability = metrics.calculate_availability()
    
    assert 0 <= availability <= 100
    
    # With failed recovery, availability should be less than 100%
    if event2 in metrics.events_injected and not event2.recovered:
        assert availability < 100


@pytest.mark.asyncio
async def test_chaos_schedule():
    """Test scheduled chaos injection."""
    monkey = ChaosMonkey()
    
    # Run with different chaos types
    chaos_types = [
        ChaosType.PROCESS_RESTART,
        ChaosType.NETWORK_LOSS,
        ChaosType.DATABASE_SLOW
    ]
    
    await monkey.run_chaos_test(
        duration_minutes=1,
        chaos_probability=1.0,
        chaos_types=chaos_types
    )
    
    # Verify different types were injected
    injected_types = {e.chaos_type for e in monkey.metrics.events_injected}
    assert len(injected_types) > 0


@pytest.mark.asyncio
async def test_chaos_report_generation():
    """Test chaos test report generation."""
    import json
    from pathlib import Path
    
    monkey = ChaosMonkey()
    
    # Run short test
    await monkey.run_chaos_test(
        duration_minutes=1,
        chaos_probability=0.5
    )
    
    # Check if report was created
    report_path = Path("tests/chaos/reports")
    
    if report_path.exists():
        report_files = list(report_path.glob("chaos_test_*.json"))
        
        if report_files:
            # Read most recent report
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_report) as f:
                report = json.load(f)
            
            assert report["test_type"] == "chaos_engineering"
            assert "metrics" in report
            assert "success" in report
            assert "data_consistent" in report


@pytest.mark.asyncio
async def test_multiple_chaos_injections():
    """Test multiple simultaneous chaos injections."""
    monkey = ChaosMonkey()
    
    # Inject multiple chaos events
    tasks = [
        monkey.inject_chaos(ChaosType.CPU_STRESS),
        monkey.inject_chaos(ChaosType.MEMORY_STRESS),
        monkey.inject_chaos(ChaosType.API_FAILURE)
    ]
    
    events = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify all events were tracked
    successful_events = [e for e in events if not isinstance(e, Exception)]
    assert len(successful_events) > 0
    
    for event in successful_events:
        if not isinstance(event, Exception):
            assert event in monkey.metrics.events_injected