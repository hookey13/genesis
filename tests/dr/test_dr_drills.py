"""Tests for DR drills and testing framework."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from genesis.dr.dr_testing import (
    DRTestFramework,
    DRTestConfig,
    DRTestType,
    DRTestResult,
    DRTestScenario,
    DRTestExecution
)
from genesis.dr.failover_manager import FailoverManager, FailoverConfig
from genesis.dr.recovery_validator import (
    RecoveryValidator,
    ValidationStatus,
    ValidationCheck,
    RecoveryValidationResult
)
from genesis.core.exceptions import GenesisException


@pytest.fixture
def dr_test_config():
    """Create test DR configuration."""
    return DRTestConfig(
        enabled=True,
        schedule="0 0 1 * *",
        test_environment="dr-test",
        test_types=[
            DRTestType.FAILOVER,
            DRTestType.DATA_INTEGRITY,
            DRTestType.FAILBACK
        ],
        chaos_testing_enabled=False,
        auto_rollback=True,
        max_test_duration_minutes=60,
        parallel_tests=False,
        test_data_sample_size=1000
    )


@pytest.fixture
def failover_manager():
    """Create mock failover manager."""
    config = FailoverConfig(
        primary_region="us-east-1",
        failover_regions=["us-west-2", "eu-west-1"]
    )
    return FailoverManager(config)


@pytest.fixture
def dr_test_framework(dr_test_config, failover_manager):
    """Create test DR framework."""
    return DRTestFramework(dr_test_config, failover_manager)


@pytest.mark.asyncio
async def test_dr_drill_execution(dr_test_framework):
    """Test execution of comprehensive DR drill."""
    # Mock test scenario execution
    dr_test_framework._execute_test_scenario = AsyncMock(
        return_value=DRTestExecution(
            test_id="test_001",
            scenario=dr_test_framework.test_scenarios[0],
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5),
            status=DRTestResult.PASSED
        )
    )
    
    dr_test_framework._generate_dr_report = AsyncMock(
        return_value="Test Report"
    )
    dr_test_framework._send_test_notifications = AsyncMock()
    
    # Run DR drill
    result = await dr_test_framework.run_dr_drill()
    
    # Verify drill execution
    assert result["drill_id"].startswith("drill_")
    assert result["overall_status"] == DRTestResult.PASSED.value
    assert len(result["tests"]) > 0
    assert "duration" in result


@pytest.mark.asyncio
async def test_monthly_dr_drill_scheduling(dr_test_framework):
    """Test monthly DR drill scheduling."""
    # Mock the drill execution
    dr_test_framework.run_dr_drill = AsyncMock(
        return_value={"status": "completed"}
    )
    
    # Start scheduler
    await dr_test_framework.start_scheduled_testing()
    assert dr_test_framework._scheduler_task is not None
    
    # Stop scheduler
    await dr_test_framework.stop_scheduled_testing()
    assert dr_test_framework._scheduler_task is None


@pytest.mark.asyncio
async def test_failover_scenario_execution(dr_test_framework):
    """Test execution of failover test scenario."""
    scenario = DRTestScenario(
        name="primary_region_failure",
        test_type=DRTestType.FAILOVER,
        description="Test primary region failure",
        steps=[
            {"action": "simulate_region_failure", "region": "primary"},
            {"action": "wait_for_detection", "timeout": 1},
            {"action": "verify_auto_failover", "expected": True}
        ],
        expected_outcomes={
            "rto_achieved": True,
            "rpo_achieved": True
        },
        rollback_steps=[
            {"action": "restore_primary_region"}
        ],
        timeout_seconds=10
    )
    
    # Mock step execution
    dr_test_framework._execute_test_step = AsyncMock()
    dr_test_framework._validate_outcomes = AsyncMock(return_value=True)
    dr_test_framework._execute_rollback = AsyncMock()
    
    # Execute scenario
    execution = await dr_test_framework._execute_test_scenario(scenario)
    
    # Verify execution
    assert execution.status == DRTestResult.PASSED
    assert execution.scenario.name == "primary_region_failure"
    assert execution.end_time is not None


@pytest.mark.asyncio
async def test_data_integrity_validation(dr_test_framework):
    """Test data integrity validation during DR drill."""
    # Mock data operations
    dr_test_framework._insert_test_data = AsyncMock(
        return_value=["test_id_1", "test_id_2", "test_id_3"]
    )
    dr_test_framework._compare_data_across_regions = AsyncMock()
    dr_test_framework._cleanup_test_data = AsyncMock()
    
    # Create test execution
    execution = DRTestExecution(
        test_id="test_data_001",
        scenario=MagicMock(),
        start_time=datetime.utcnow()
    )
    
    # Test data insertion
    await dr_test_framework._execute_test_step(
        {"action": "insert_test_data", "count": 3},
        execution
    )
    assert len(execution.artifacts["test_data_ids"]) == 3
    
    # Test data comparison
    await dr_test_framework._execute_test_step(
        {"action": "compare_data_regions"},
        execution
    )
    dr_test_framework._compare_data_across_regions.assert_called_once()
    
    # Test data cleanup
    await dr_test_framework._execute_test_step(
        {"action": "cleanup_test_data"},
        execution
    )
    dr_test_framework._cleanup_test_data.assert_called_once()


@pytest.mark.asyncio
async def test_network_partition_scenario(dr_test_framework):
    """Test network partition scenario."""
    scenario = DRTestScenario(
        name="network_partition",
        test_type=DRTestType.NETWORK_PARTITION,
        description="Test network partition",
        steps=[
            {"action": "create_network_partition", "duration": 60},
            {"action": "verify_split_brain_prevention"},
            {"action": "heal_network_partition"}
        ],
        expected_outcomes={
            "split_brain_prevented": True
        },
        rollback_steps=[],
        timeout_seconds=120
    )
    
    # Mock validations
    dr_test_framework._execute_test_step = AsyncMock()
    dr_test_framework._validate_outcomes = AsyncMock(return_value=True)
    
    # Execute scenario
    execution = await dr_test_framework._execute_test_scenario(scenario)
    
    assert execution.status == DRTestResult.PASSED
    assert execution.scenario.test_type == DRTestType.NETWORK_PARTITION


@pytest.mark.asyncio
async def test_automatic_rollback_on_failure(dr_test_framework):
    """Test automatic rollback when test fails."""
    scenario = DRTestScenario(
        name="test_with_rollback",
        test_type=DRTestType.FAILOVER,
        description="Test with rollback",
        steps=[
            {"action": "simulate_failure"}
        ],
        expected_outcomes={},
        rollback_steps=[
            {"action": "restore_state"}
        ]
    )
    
    # Mock failure and rollback
    dr_test_framework._execute_test_step = AsyncMock(
        side_effect=[Exception("Test failed"), None]
    )
    
    # Execute scenario
    execution = await dr_test_framework._execute_test_scenario(scenario)
    
    # Verify rollback was executed
    assert execution.status == DRTestResult.FAILED
    assert execution.rollback_executed == True
    assert "Test failed" in execution.errors


@pytest.mark.asyncio
async def test_parallel_test_execution(dr_test_framework):
    """Test parallel execution of multiple test scenarios."""
    dr_test_framework.config.parallel_tests = True
    
    # Mock scenario execution
    async def mock_execute(scenario):
        await asyncio.sleep(0.1)  # Simulate execution time
        return DRTestExecution(
            test_id=f"test_{scenario.name}",
            scenario=scenario,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            status=DRTestResult.PASSED
        )
    
    dr_test_framework._execute_test_scenario = mock_execute
    dr_test_framework._generate_dr_report = AsyncMock(return_value="Report")
    dr_test_framework._send_test_notifications = AsyncMock()
    
    # Run drill with parallel execution
    start_time = datetime.utcnow()
    result = await dr_test_framework.run_dr_drill()
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Verify parallel execution (should be faster than sequential)
    assert result["overall_status"] == DRTestResult.PASSED.value
    assert duration < 1.0  # All tests should run in parallel


@pytest.mark.asyncio
async def test_chaos_testing(dr_test_framework):
    """Test chaos engineering functionality."""
    dr_test_framework.config.chaos_testing_enabled = True
    
    # Mock scenario execution
    dr_test_framework._execute_test_scenario = AsyncMock(
        return_value=DRTestExecution(
            test_id="chaos_test",
            scenario=MagicMock(),
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            status=DRTestResult.PASSED
        )
    )
    
    # Run chaos test
    result = await dr_test_framework.run_chaos_test()
    
    assert result["test_type"] == "chaos"
    assert result["scenarios_executed"] > 0


@pytest.mark.asyncio
async def test_recovery_validation():
    """Test recovery validation after failover."""
    validator = RecoveryValidator()
    
    # Mock validation checks
    validator._execute_validation_check = AsyncMock()
    validator._calculate_validation_metrics = AsyncMock(
        return_value={
            "total_checks": 8,
            "passed_checks": 7,
            "failed_checks": 0,
            "warning_checks": 1,
            "success_rate": 87.5
        }
    )
    
    # Validate recovery
    result = await validator.validate_recovery(
        source_region="us-east-1",
        target_region="us-west-2",
        recovery_type="failover"
    )
    
    assert result.validation_id.startswith("val_")
    assert result.overall_status in [
        ValidationStatus.PASSED,
        ValidationStatus.WARNING,
        ValidationStatus.FAILED
    ]
    assert len(result.checks) > 0
    assert result.end_time is not None


@pytest.mark.asyncio
async def test_post_incident_analysis():
    """Test post-incident analysis generation."""
    validator = RecoveryValidator()
    
    # Mock analysis components
    validator._reconstruct_timeline = AsyncMock(
        return_value=[
            {"time": "T-5m", "event": "Failure detected"},
            {"time": "T+0", "event": "Failover completed"}
        ]
    )
    validator._analyze_root_cause = AsyncMock(
        return_value={
            "primary_cause": "Network failure",
            "contributing_factors": [],
            "preventable": False
        }
    )
    validator._assess_impact = AsyncMock(
        return_value={
            "downtime_seconds": 240,
            "data_loss": False
        }
    )
    validator._evaluate_recovery = AsyncMock(
        return_value={
            "rto_met": True,
            "rpo_met": True
        }
    )
    validator._identify_lessons = AsyncMock(
        return_value=["Failover successful"]
    )
    validator._generate_action_items = AsyncMock(
        return_value=[
            {"action": "Review procedures", "priority": "medium"}
        ]
    )
    
    # Generate analysis
    analysis = await validator.generate_post_incident_analysis("failover_001")
    
    assert analysis["failover_id"] == "failover_001"
    assert "timeline" in analysis
    assert "root_cause" in analysis
    assert "impact_assessment" in analysis
    assert "recovery_effectiveness" in analysis
    assert "lessons_learned" in analysis
    assert "action_items" in analysis


@pytest.mark.asyncio
async def test_test_history_tracking(dr_test_framework):
    """Test that DR test history is properly tracked."""
    # Create test executions
    execution1 = DRTestExecution(
        test_id="test_001",
        scenario=MagicMock(name="scenario1", test_type=DRTestType.FAILOVER),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        status=DRTestResult.PASSED
    )
    
    execution2 = DRTestExecution(
        test_id="test_002",
        scenario=MagicMock(name="scenario2", test_type=DRTestType.DATA_INTEGRITY),
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        status=DRTestResult.FAILED,
        errors=["Data mismatch"]
    )
    
    # Add to history
    dr_test_framework.test_history.append(execution1)
    dr_test_framework.test_history.append(execution2)
    
    # Get history
    history = dr_test_framework.get_test_history(limit=2)
    
    assert len(history) == 2
    assert history[0]["test_id"] == "test_001"
    assert history[0]["status"] == "passed"
    assert history[1]["test_id"] == "test_002"
    assert history[1]["status"] == "failed"
    assert "Data mismatch" in history[1]["errors"]


@pytest.mark.asyncio
async def test_validation_criteria_checking():
    """Test validation criteria are properly checked."""
    validator = RecoveryValidator()
    
    # Create validation check
    check = ValidationCheck(
        name="data_integrity",
        category="data",
        description="Test data integrity",
        critical=True
    )
    
    # Mock data validation
    validator._calculate_data_checksums = AsyncMock(
        return_value={
            "users": "checksum1",
            "orders": "checksum2"
        }
    )
    
    # Execute validation
    await validator._validate_data_integrity(
        check,
        "us-east-1",
        "us-west-2"
    )
    
    # Check should have results
    assert check.end_time is not None
    assert check.status in [
        ValidationStatus.PASSED,
        ValidationStatus.FAILED,
        ValidationStatus.WARNING
    ]


@pytest.mark.asyncio
async def test_dr_report_generation(dr_test_framework):
    """Test DR report generation."""
    test_results = {
        "drill_id": "drill_20240101_120000",
        "start_time": datetime.utcnow(),
        "end_time": datetime.utcnow() + timedelta(minutes=10),
        "duration": 600,
        "overall_status": "passed",
        "tests": [
            {
                "scenario": "primary_region_failure",
                "status": "passed",
                "duration": 300,
                "metrics": {"rto": 240},
                "errors": []
            },
            {
                "scenario": "data_integrity",
                "status": "warning",
                "duration": 150,
                "metrics": {"records_validated": 1000},
                "errors": []
            }
        ]
    }
    
    # Generate report
    report = await dr_test_framework._generate_dr_report(test_results)
    
    # Verify report content
    assert "DR Drill Report" in report
    assert "drill_20240101_120000" in report
    assert "primary_region_failure" in report
    assert "data_integrity" in report
    assert "Status: passed" in report
    assert "Status: warning" in report


@pytest.mark.asyncio
async def test_outcome_validation(dr_test_framework):
    """Test validation of test outcomes against expectations."""
    execution = DRTestExecution(
        test_id="test_001",
        scenario=MagicMock(),
        start_time=datetime.utcnow()
    )
    
    # Set metrics
    execution.metrics = {
        "rto_achieved": True,
        "rpo_achieved": True,
        "data_loss": False
    }
    
    # Define expected outcomes
    expected = {
        "rto_achieved": True,
        "rpo_achieved": True,
        "data_loss": False
    }
    
    # Validate outcomes
    result = await dr_test_framework._validate_outcomes(expected, execution)
    assert result == True
    
    # Test with mismatched outcomes
    execution.metrics["data_loss"] = True
    result = await dr_test_framework._validate_outcomes(expected, execution)
    assert result == False


@pytest.mark.asyncio
async def test_service_validation_in_recovery(dr_test_framework):
    """Test service availability validation during recovery."""
    # Mock service checks
    dr_test_framework._check_service_health = AsyncMock(
        return_value=(True, 50.0)  # (is_healthy, response_time_ms)
    )
    
    # Create validation step
    execution = DRTestExecution(
        test_id="test_001",
        scenario=MagicMock(),
        start_time=datetime.utcnow()
    )
    
    # Validate services
    await dr_test_framework._execute_test_step(
        {"action": "validate_services", "region": "us-west-2"},
        execution
    )
    
    # Verify service validation was performed
    dr_test_framework.failover_manager._check_region_health.assert_called()