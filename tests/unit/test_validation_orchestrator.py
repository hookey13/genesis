"""Unit tests for ValidationOrchestrator."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.validation import (
    ValidationOrchestrator,
    ValidationReport,
    TestValidator,
    StabilityTester,
    SecurityScanner,
    PerformanceValidator,
    DisasterRecoveryValidator,
    PaperTradingValidator,
    ComplianceValidator,
    OperationalValidator
)


@pytest.fixture
def orchestrator(tmp_path):
    """Create ValidationOrchestrator instance."""
    return ValidationOrchestrator(genesis_root=tmp_path)


@pytest.fixture
def mock_validator_result():
    """Create mock validator result."""
    return {
        "validator": "test",
        "timestamp": datetime.utcnow().isoformat(),
        "passed": True,
        "score": 95.0,
        "checks": {
            "check1": {"passed": True, "details": ["Test passed"]},
            "check2": {"passed": True, "details": ["Another test passed"]}
        },
        "summary": "All tests passed",
        "details": ["Detail 1", "Detail 2"]
    }


@pytest.fixture
def mock_failing_validator_result():
    """Create mock failing validator result."""
    return {
        "validator": "test",
        "timestamp": datetime.utcnow().isoformat(),
        "passed": False,
        "score": 45.0,
        "checks": {
            "check1": {"passed": False, "details": ["Test failed"], "error": "Error message"},
            "check2": {"passed": True, "details": ["Test passed"]}
        },
        "summary": "Some tests failed",
        "errors": ["Error 1", "Error 2"],
        "warnings": ["Warning 1"]
    }


class TestValidationReport:
    """Test ValidationReport class."""
    
    def test_initialization(self):
        """Test ValidationReport initialization."""
        report = ValidationReport("test_validator")
        
        assert report.validator_name == "test_validator"
        assert report.status == "pending"
        assert report.score == 0.0
        assert report.passed is False
        assert report.errors == []
        assert report.warnings == []
        assert isinstance(report.timestamp, datetime)
        
    def test_to_dict(self):
        """Test ValidationReport.to_dict method."""
        report = ValidationReport("test_validator")
        report.status = "completed"
        report.score = 95.0
        report.passed = True
        report.details = {"test": "data"}
        report.errors = ["error1"]
        report.warnings = ["warning1"]
        
        result = report.to_dict()
        
        assert result["validator_name"] == "test_validator"
        assert result["status"] == "completed"
        assert result["score"] == 95.0
        assert result["passed"] is True
        assert result["details"] == {"test": "data"}
        assert result["errors"] == ["error1"]
        assert result["warnings"] == ["warning1"]
        assert "timestamp" in result


class TestValidationOrchestrator:
    """Test ValidationOrchestrator class."""
    
    def test_initialization(self, tmp_path):
        """Test ValidationOrchestrator initialization."""
        orchestrator = ValidationOrchestrator(genesis_root=tmp_path)
        
        assert orchestrator.genesis_root == tmp_path
        assert len(orchestrator.validators) == 8
        assert "test_coverage" in orchestrator.validators
        assert "stability" in orchestrator.validators
        assert "security" in orchestrator.validators
        assert "performance" in orchestrator.validators
        assert "disaster_recovery" in orchestrator.validators
        assert "paper_trading" in orchestrator.validators
        assert "compliance" in orchestrator.validators
        assert "operational" in orchestrator.validators
        
    @pytest.mark.asyncio
    async def test_run_all_validators_parallel(self, orchestrator, mock_validator_result):
        """Test running all validators in parallel."""
        # Mock all validators
        for name, validator in orchestrator.validators.items():
            validator.validate = AsyncMock(return_value=mock_validator_result)
            
        result = await orchestrator.run_all_validators(parallel=True)
        
        assert result["overall_passed"] is True
        assert result["overall_score"] == 95.0
        assert len(result["validators"]) == 8
        
        for validator_name, validator_result in result["validators"].items():
            assert validator_result["passed"] is True
            assert validator_result["score"] == 95.0
            
    @pytest.mark.asyncio
    async def test_run_all_validators_sequential(self, orchestrator, mock_validator_result):
        """Test running all validators sequentially."""
        # Mock all validators
        for name, validator in orchestrator.validators.items():
            validator.validate = AsyncMock(return_value=mock_validator_result)
            
        result = await orchestrator.run_all_validators(parallel=False)
        
        assert result["overall_passed"] is True
        assert result["overall_score"] == 95.0
        assert len(result["validators"]) == 8
        
    @pytest.mark.asyncio
    async def test_run_all_validators_with_failures(self, orchestrator, mock_validator_result, mock_failing_validator_result):
        """Test running validators with some failures."""
        # Mock validators with mixed results
        for i, (name, validator) in enumerate(orchestrator.validators.items()):
            if i % 2 == 0:
                validator.validate = AsyncMock(return_value=mock_validator_result)
            else:
                validator.validate = AsyncMock(return_value=mock_failing_validator_result)
                
        result = await orchestrator.run_all_validators(parallel=True)
        
        assert result["overall_passed"] is False
        assert result["overall_score"] == 70.0  # (4 * 95 + 4 * 45) / 8
        
    @pytest.mark.asyncio
    async def test_run_all_validators_with_exception(self, orchestrator, mock_validator_result):
        """Test handling validator exceptions."""
        # Mock validators with one throwing exception
        for i, (name, validator) in enumerate(orchestrator.validators.items()):
            if i == 0:
                validator.validate = AsyncMock(side_effect=Exception("Test error"))
            else:
                validator.validate = AsyncMock(return_value=mock_validator_result)
                
        result = await orchestrator.run_all_validators(parallel=True)
        
        assert result["overall_passed"] is False
        # First validator has score 0 due to exception
        expected_score = (0 + 7 * 95.0) / 8
        assert result["overall_score"] == expected_score
        
    @pytest.mark.asyncio
    async def test_run_critical_validators(self, orchestrator, mock_validator_result):
        """Test running only critical validators."""
        # Mock all validators
        for name, validator in orchestrator.validators.items():
            validator.validate = AsyncMock(return_value=mock_validator_result)
            
        result = await orchestrator.run_critical_validators()
        
        assert result["type"] == "critical"
        assert len(result["validators"]) == 3  # test_coverage, security, performance
        assert "test_coverage" in result["validators"]
        assert "security" in result["validators"]
        assert "performance" in result["validators"]
        
    @pytest.mark.asyncio
    async def test_run_validator_with_timeout(self, orchestrator):
        """Test validator timeout handling."""
        # Create a validator that takes too long
        async def slow_validate():
            await asyncio.sleep(100)
            return {}
            
        validator = MagicMock()
        validator.validate = slow_validate
        
        with pytest.raises(asyncio.TimeoutError):
            await orchestrator._run_validator("test", validator)
            
    def test_create_report(self, orchestrator, mock_validator_result):
        """Test creating ValidationReport from results."""
        report = orchestrator._create_report("test", mock_validator_result)
        
        assert report.validator_name == "test"
        assert report.status == "completed"
        assert report.passed is True
        assert report.score == 95.0
        assert report.details == ["Detail 1", "Detail 2"]
        
    def test_create_report_with_errors(self, orchestrator, mock_failing_validator_result):
        """Test creating ValidationReport with errors."""
        report = orchestrator._create_report("test", mock_failing_validator_result)
        
        assert report.validator_name == "test"
        assert report.status == "completed"
        assert report.passed is False
        assert report.score == 45.0
        assert len(report.errors) == 3  # 2 from errors + 1 from check1
        assert len(report.warnings) == 1
        
    def test_create_error_report(self, orchestrator):
        """Test creating error ValidationReport."""
        report = orchestrator._create_error_report("test", "Test error message")
        
        assert report.validator_name == "test"
        assert report.status == "failed"
        assert report.passed is False
        assert report.score == 0
        assert report.errors == ["Test error message"]
        
    def test_generate_summary_all_passed(self, orchestrator):
        """Test generating summary when all validators pass."""
        # Create passing reports
        for name in ["test1", "test2"]:
            report = ValidationReport(name)
            report.passed = True
            report.score = 95.0
            orchestrator.results[name] = report
            
        summary = orchestrator._generate_summary()
        
        assert "✅ All validators passed!" in summary
        assert "Overall Score: 95.0%" in summary
        assert "Passed: 2/2" in summary
        
    def test_generate_summary_with_failures(self, orchestrator):
        """Test generating summary with failures."""
        # Create mixed reports
        report1 = ValidationReport("test1")
        report1.passed = True
        report1.score = 95.0
        orchestrator.results["test1"] = report1
        
        report2 = ValidationReport("test2")
        report2.passed = False
        report2.score = 45.0
        report2.errors = ["Error 1", "Error 2"]
        orchestrator.results["test2"] = report2
        
        summary = orchestrator._generate_summary()
        
        assert "❌ 1 validator(s) failed:" in summary
        assert "test2: score=45.0%" in summary
        assert "Error 1, Error 2" in summary
        assert "Overall Score: 70.0%" in summary
        assert "Passed: 1/2" in summary
        
    @pytest.mark.asyncio
    async def test_save_report(self, orchestrator, tmp_path):
        """Test saving validation report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_passed": True,
            "overall_score": 95.0,
            "validators": {}
        }
        
        output_path = await orchestrator.save_report(report)
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        assert "validation_report_" in output_path.name
        
        # Verify content
        with open(output_path) as f:
            saved_report = json.load(f)
            
        assert saved_report["overall_passed"] is True
        assert saved_report["overall_score"] == 95.0
        
    @pytest.mark.asyncio
    async def test_save_report_custom_path(self, orchestrator, tmp_path):
        """Test saving validation report to custom path."""
        report = {"test": "data"}
        custom_path = tmp_path / "custom_report.json"
        
        output_path = await orchestrator.save_report(report, custom_path)
        
        assert output_path == custom_path
        assert output_path.exists()
        
        with open(output_path) as f:
            saved_report = json.load(f)
            
        assert saved_report == {"test": "data"}


class TestValidatorIntegration:
    """Test validator integration."""
    
    @pytest.mark.asyncio
    async def test_validator_initialization(self, tmp_path):
        """Test that all validators can be initialized."""
        orchestrator = ValidationOrchestrator(genesis_root=tmp_path)
        
        assert isinstance(orchestrator.validators["test_coverage"], TestValidator)
        assert isinstance(orchestrator.validators["stability"], StabilityTester)
        assert isinstance(orchestrator.validators["security"], SecurityScanner)
        assert isinstance(orchestrator.validators["performance"], PerformanceValidator)
        assert isinstance(orchestrator.validators["disaster_recovery"], DisasterRecoveryValidator)
        assert isinstance(orchestrator.validators["paper_trading"], PaperTradingValidator)
        assert isinstance(orchestrator.validators["compliance"], ComplianceValidator)
        assert isinstance(orchestrator.validators["operational"], OperationalValidator)
        
    @pytest.mark.asyncio
    async def test_orchestrator_end_to_end(self, tmp_path):
        """Test orchestrator end-to-end with mock validators."""
        orchestrator = ValidationOrchestrator(genesis_root=tmp_path)
        
        # Mock all validators with quick responses
        mock_result = {
            "validator": "mock",
            "timestamp": datetime.utcnow().isoformat(),
            "passed": True,
            "score": 100.0,
            "checks": {},
            "summary": "Mock validation passed"
        }
        
        for validator in orchestrator.validators.values():
            validator.validate = AsyncMock(return_value=mock_result)
            
        # Run validation
        result = await orchestrator.run_all_validators(parallel=True)
        
        # Verify results
        assert result["overall_passed"] is True
        assert result["overall_score"] == 100.0
        assert "summary" in result
        assert "✅ All validators passed!" in result["summary"]
        
        # Save report
        report_path = await orchestrator.save_report(result)
        assert report_path.exists()