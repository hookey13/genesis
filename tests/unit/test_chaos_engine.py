"""
Unit tests for chaos engineering framework.
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from tests.chaos.chaos_engine import (
    ChaosMonkey,
    ChaosType,
    ChaosEvent,
    ChaosMetrics,
    RecoveryValidator
)


class TestChaosType:
    """Tests for ChaosType enum."""
    
    def test_chaos_types_exist(self):
        """Test that all required chaos types are defined."""
        expected_types = [
            "PROCESS_KILL",
            "NETWORK_DELAY",
            "NETWORK_LOSS",
            "CPU_STRESS",
            "MEMORY_STRESS",
            "DISK_STRESS",
            "DATABASE_SLOW",
            "API_FAILURE",
            "NETWORK_PARTITION",
            "PROCESS_RESTART"
        ]
        
        for chaos_type in expected_types:
            assert hasattr(ChaosType, chaos_type)
            assert getattr(ChaosType, chaos_type).value == chaos_type.lower()


class TestChaosEvent:
    """Tests for ChaosEvent dataclass."""
    
    def test_event_creation(self):
        """Test creating a chaos event."""
        event = ChaosEvent(
            chaos_type=ChaosType.PROCESS_KILL,
            timestamp=datetime.now(),
            duration_seconds=5.0,
            target="test_service",
            parameters={"signal": "SIGKILL"},
            impact="process_terminated"
        )
        
        assert event.chaos_type == ChaosType.PROCESS_KILL
        assert event.duration_seconds == 5.0
        assert event.target == "test_service"
        assert event.parameters == {"signal": "SIGKILL"}
        assert event.impact == "process_terminated"
    
    def test_event_defaults(self):
        """Test event default values."""
        event = ChaosEvent(
            chaos_type=ChaosType.NETWORK_DELAY,
            timestamp=datetime.now(),
            duration_seconds=10.0,
            target="network",
            parameters={},
            impact="unknown"
        )
        
        assert event.recovered is False  # Default not recovered
        assert event.recovery_time_seconds is None  # Default no recovery time


class TestChaosMetrics:
    """Tests for ChaosMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ChaosMetrics()
        
        assert metrics.start_time is not None
        assert metrics.end_time is None
        assert metrics.events_injected == []
        assert metrics.failures_detected == 0
        assert metrics.successful_recoveries == 0
        assert metrics.failed_recoveries == 0
        assert metrics.mean_recovery_time == 0
        assert metrics.max_recovery_time == 0
        assert metrics.service_availability == 100.0
        assert metrics.data_consistency_checks == 0
        assert metrics.data_inconsistencies == 0
    
    def test_add_event(self):
        """Test adding event to metrics."""
        metrics = ChaosMetrics()
        
        event = ChaosEvent(
            chaos_type=ChaosType.CPU_STRESS,
            timestamp=datetime.now(),
            duration_seconds=10,
            target="cpu",
            parameters={},
            impact="stress",
            recovered=True,
            recovery_time_seconds=5
        )
        
        metrics.add_event(event)
        
        assert len(metrics.events_injected) == 1
        assert metrics.mean_recovery_time == 5
        assert metrics.max_recovery_time == 5
    
    def test_calculate_availability(self):
        """Test availability calculation."""
        metrics = ChaosMetrics()
        
        # Add an event with downtime
        event = ChaosEvent(
            chaos_type=ChaosType.PROCESS_KILL,
            timestamp=datetime.now(),
            duration_seconds=30,
            target="service",
            parameters={},
            impact="down",
            recovered=False
        )
        metrics.add_event(event)
        
        # Availability should be less than 100%
        availability = metrics.calculate_availability()
        assert availability < 100.0
        assert availability >= 0.0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ChaosMetrics()
        metrics.failures_detected = 2
        metrics.successful_recoveries = 5
        metrics.failed_recoveries = 1
        
        report = metrics.to_dict()
        
        assert report["failures_detected"] == 2
        assert report["successful_recoveries"] == 5
        assert report["failed_recoveries"] == 1
        assert "service_availability_percent" in report
        assert "total_events" in report


class TestChaosMonkeyInjection:
    """Tests for ChaosMonkey injection methods."""
    
    @pytest.mark.asyncio
    async def test_inject_process_kill(self):
        """Test process kill injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.PROCESS_KILL,
            timestamp=datetime.now(),
            duration_seconds=5,
            target="",
            parameters={},
            impact="unknown"
        )
        
        # Inject process kill
        await monkey._inject_process_kill(event)
        
        assert event.target == "trading_engine"
        assert event.parameters == {"signal": "SIGKILL"}
        assert event.impact == "process_terminated"
    
    @pytest.mark.asyncio
    async def test_inject_network_delay(self):
        """Test network delay injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.NETWORK_DELAY,
            timestamp=datetime.now(),
            duration_seconds=5,
            target="",
            parameters={},
            impact="unknown"
        )
        
        await monkey._inject_network_delay(event)
        
        assert event.target == "network"
        assert "delay_ms" in event.parameters
        assert "latency" in event.impact
    
    @pytest.mark.asyncio
    async def test_inject_network_loss(self):
        """Test network packet loss injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.NETWORK_LOSS,
            timestamp=datetime.now(),
            duration_seconds=5,
            target="",
            parameters={},
            impact="unknown"
        )
        
        await monkey._inject_network_loss(event)
        
        assert event.target == "network"
        assert "loss_percent" in event.parameters
        assert "packet_loss" in event.impact
    
    @pytest.mark.asyncio
    async def test_inject_cpu_stress(self):
        """Test CPU stress injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.CPU_STRESS,
            timestamp=datetime.now(),
            duration_seconds=1,
            target="",
            parameters={},
            impact="unknown"
        )
        
        await monkey._inject_cpu_stress(event)
        
        assert event.target == "cpu"
        assert event.parameters == {"cores": 2, "load_percent": 80}
        assert event.impact == "high_cpu_usage"
    
    @pytest.mark.asyncio
    async def test_inject_memory_stress(self):
        """Test memory stress injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.MEMORY_STRESS,
            timestamp=datetime.now(),
            duration_seconds=1,
            target="",
            parameters={},
            impact="unknown"
        )
        
        await monkey._inject_memory_stress(event)
        
        assert event.target == "memory"
        assert event.parameters == {"allocated_mb": 100}
        assert "100MB" in event.impact
    
    @pytest.mark.asyncio
    async def test_inject_database_slow(self):
        """Test database slowdown injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.DATABASE_SLOW,
            timestamp=datetime.now(),
            duration_seconds=5,
            target="",
            parameters={},
            impact="unknown"
        )
        
        await monkey._inject_database_slow(event)
        
        assert event.target == "database"
        assert event.parameters == {"slowdown_factor": 10}
        assert "10x_query_slowdown" in event.impact
    
    @pytest.mark.asyncio
    async def test_inject_api_failure(self):
        """Test API failure injection."""
        monkey = ChaosMonkey()
        event = ChaosEvent(
            chaos_type=ChaosType.API_FAILURE,
            timestamp=datetime.now(),
            duration_seconds=5,
            target="",
            parameters={},
            impact="unknown"
        )
        
        await monkey._inject_api_failure(event)
        
        assert event.target == "exchange_api"
        assert event.parameters == {"failure_rate": 0.5}
        assert "50%" in event.impact


class TestRecoveryValidator:
    """Tests for RecoveryValidator class."""
    
    @pytest.mark.asyncio
    async def test_validate_recovery(self):
        """Test recovery validation."""
        # Create health checks
        async def health_check_pass():
            return True
        
        async def health_check_fail():
            return False
        
        # Test with passing checks
        validator = RecoveryValidator([health_check_pass])
        result = await validator.validate()
        assert result is True
        
        # Test with failing checks
        validator = RecoveryValidator([health_check_fail])
        result = await validator.validate()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_with_exception(self):
        """Test recovery validation with exception."""
        async def health_check_error():
            raise Exception("Health check error")
        
        validator = RecoveryValidator([health_check_error])
        result = await validator.validate()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_empty_checks(self):
        """Test recovery validation with no checks."""
        validator = RecoveryValidator()
        result = await validator.validate()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_multiple_checks(self):
        """Test recovery validation with multiple checks."""
        async def check1():
            return True
        
        async def check2():
            return True
        
        async def check3():
            return False
        
        # All pass
        validator = RecoveryValidator([check1, check2])
        result = await validator.validate()
        assert result is True
        
        # One fails
        validator = RecoveryValidator([check1, check2, check3])
        result = await validator.validate()
        assert result is False


class TestChaosMonkey:
    """Tests for main ChaosMonkey class."""
    
    @pytest.mark.asyncio
    async def test_monkey_initialization(self):
        """Test chaos monkey initialization."""
        monkey = ChaosMonkey()
        
        assert monkey.target_system == {}
        assert monkey.recovery_validator is not None
        assert monkey.metrics is not None
        assert monkey.running is False
        assert monkey.chaos_schedule == []
    
    @pytest.mark.asyncio
    async def test_inject_chaos(self):
        """Test injecting chaos events."""
        monkey = ChaosMonkey()
        
        # Inject chaos
        event = await monkey.inject_chaos(ChaosType.NETWORK_DELAY)
        
        assert event is not None
        assert event.chaos_type == ChaosType.NETWORK_DELAY
        assert event.target == "network"
        assert len(monkey.metrics.events_injected) == 1
    
    @pytest.mark.asyncio
    async def test_inject_chaos_with_recovery(self):
        """Test chaos injection with recovery."""
        # Create recovery validator
        async def health_check():
            return True
        
        monkey = ChaosMonkey(recovery_validator=health_check)
        
        # Inject chaos with short duration
        event = await monkey.inject_chaos(ChaosType.CPU_STRESS)
        
        assert event is not None
        assert event.chaos_type == ChaosType.CPU_STRESS
        # Recovery status will be set after duration
    
    @pytest.mark.asyncio
    async def test_data_consistency_validation(self):
        """Test data consistency validation."""
        monkey = ChaosMonkey()
        
        # Test validation
        result = await monkey.validate_data_consistency()
        
        assert result is True
        assert monkey.metrics.data_consistency_checks == 1
        assert monkey.metrics.data_inconsistencies == 0
    
    @pytest.mark.asyncio
    async def test_generate_report(self):
        """Test report generation."""
        monkey = ChaosMonkey()
        
        # Add some events
        event = ChaosEvent(
            chaos_type=ChaosType.CPU_STRESS,
            timestamp=datetime.now(),
            duration_seconds=10,
            target="cpu",
            parameters={},
            impact="stress",
            recovered=True,
            recovery_time_seconds=5
        )
        monkey.metrics.add_event(event)
        
        # Generate report
        await monkey.generate_report()
        
        # Check report was created
        report_dir = Path("tests/chaos/reports")
        assert report_dir.exists()
    
    @pytest.mark.asyncio
    async def test_run_chaos_test_short(self):
        """Test running a short chaos test."""
        # Create recovery validator
        async def health_check():
            return True
        
        monkey = ChaosMonkey(recovery_validator=health_check)
        
        # Run very short test (1 minute, low probability)
        # Note: Using asyncio.wait_for to ensure test doesn't hang
        try:
            await asyncio.wait_for(
                monkey.run_chaos_test(
                    duration_minutes=0.1,  # 6 seconds
                    chaos_probability=0.0,  # No chaos for quick test
                    chaos_types=[ChaosType.NETWORK_DELAY]
                ),
                timeout=10
            )
        except asyncio.TimeoutError:
            pass  # Expected for very short test
        
        # Check that metrics were recorded
        assert monkey.metrics.end_time is not None
        assert monkey.running is False


@pytest.mark.asyncio
async def test_full_chaos_scenario():
    """Test a full chaos engineering scenario."""
    # Create recovery validator
    async def health_check():
        return True
    
    validator = RecoveryValidator([health_check])
    
    # Create chaos monkey
    monkey = ChaosMonkey(recovery_validator=validator.validate)
    
    # Inject multiple chaos types
    chaos_types = [
        ChaosType.NETWORK_DELAY,
        ChaosType.CPU_STRESS,
        ChaosType.MEMORY_STRESS
    ]
    
    for chaos_type in chaos_types:
        event = await monkey.inject_chaos(chaos_type)
        assert event is not None
        assert event.chaos_type == chaos_type
    
    # Validate metrics
    assert len(monkey.metrics.events_injected) == 3
    assert monkey.metrics.failures_detected == 0
    
    # Test data consistency
    result = await monkey.validate_data_consistency()
    assert result is True
    
    # Generate final report
    await monkey.generate_report()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])