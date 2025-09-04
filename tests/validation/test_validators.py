"""Unit tests for production validators."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.validators import (
    BaseValidator,
    ValidationStatus,
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    ValidationOrchestrator
)
from scripts.validators.strategy_validator import StrategyValidator
from scripts.validators.risk_validator import RiskValidator
from scripts.validators.database_validator import DatabaseValidator
from scripts.validators.monitoring_validator import MonitoringValidator
from scripts.validators.security_validator import SecurityValidator


class TestBaseValidator:
    """Test base validator functionality."""
    
    def test_validation_issue_creation(self):
        """Test ValidationIssue creation."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error",
            details={"key": "value"},
            recommendation="Fix the error"
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Test error"
        assert issue.details["key"] == "value"
        assert issue.recommendation == "Fix the error"
    
    def test_validation_result_status(self):
        """Test ValidationResult status tracking."""
        result = ValidationResult(validator_name="test")
        
        assert result.status == ValidationStatus.PENDING
        assert result.passed is False
        assert result.failed is False
        
        result.status = ValidationStatus.PASSED
        assert result.passed is True
        assert result.failed is False
        
        result.status = ValidationStatus.FAILED
        assert result.passed is False
        assert result.failed is True
    
    def test_add_issue_updates_status(self):
        """Test that adding issues updates result status."""
        result = ValidationResult(validator_name="test")
        result.status = ValidationStatus.PASSED
        
        # Adding info issue shouldn't change passed status
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message="Info message"
        ))
        assert result.status == ValidationStatus.PASSED
        
        # Adding warning should change to warning
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message="Warning message"
        ))
        assert result.status == ValidationStatus.WARNING
        
        # Adding error should change to failed
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Error message"
        ))
        assert result.status == ValidationStatus.FAILED
        
        # Critical should also be failed
        result = ValidationResult(validator_name="test2")
        result.add_issue(ValidationIssue(
            severity=ValidationSeverity.CRITICAL,
            message="Critical message"
        ))
        assert result.status == ValidationStatus.FAILED


class TestStrategyValidator:
    """Test strategy validator."""
    
    @pytest.mark.asyncio
    async def test_strategy_validator_initialization(self):
        """Test strategy validator can be initialized."""
        validator = StrategyValidator()
        assert validator.name == "strategy"
        assert "strategies" in validator.description.lower()
    
    @pytest.mark.asyncio
    async def test_check_directory_structure(self):
        """Test directory structure validation."""
        validator = StrategyValidator()
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            
            # Mock glob to return strategy files
            with patch("pathlib.Path.glob") as mock_glob:
                mock_glob.return_value = [
                    Path("strategy1.py"),
                    Path("strategy2.py")
                ]
                
                await validator._check_directory_structure()
                
                # Check that issues were added for found directories
                assert any(
                    "Strategy base directory exists" in issue.message
                    for issue in validator.result.issues
                )
    
    @pytest.mark.asyncio
    async def test_validate_strategy_configs(self):
        """Test strategy configuration validation."""
        validator = StrategyValidator()
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                with patch("builtins.open", create=True) as mock_open:
                    await validator._validate_strategy_configs()
                    
                    # Should create default configs
                    assert mock_open.called


class TestRiskValidator:
    """Test risk validator."""
    
    @pytest.mark.asyncio
    async def test_risk_validator_initialization(self):
        """Test risk validator can be initialized."""
        validator = RiskValidator()
        assert validator.name == "risk"
        assert "risk" in validator.description.lower()
    
    @pytest.mark.asyncio
    async def test_validate_risk_config(self):
        """Test risk configuration validation."""
        validator = RiskValidator()
        
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            
            with patch("builtins.open", create=True) as mock_open:
                await validator._validate_risk_config()
                
                # Should create default config
                assert mock_open.called
                
                # Check that info issue was added
                assert any(
                    "Created default risk configuration" in issue.message
                    for issue in validator.result.issues
                )


class TestDatabaseValidator:
    """Test database validator."""
    
    @pytest.mark.asyncio
    async def test_database_validator_initialization(self):
        """Test database validator can be initialized."""
        validator = DatabaseValidator()
        assert validator.name == "database"
        assert "database" in validator.description.lower()
    
    @pytest.mark.asyncio
    async def test_test_connectivity(self):
        """Test database connectivity check."""
        validator = DatabaseValidator()
        
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ["3.39.0"]
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            await validator._test_connectivity()
            
            # Should report successful connection
            assert any(
                "SQLite connected" in issue.message
                for issue in validator.result.issues
            )


class TestMonitoringValidator:
    """Test monitoring validator."""
    
    @pytest.mark.asyncio
    async def test_monitoring_validator_initialization(self):
        """Test monitoring validator can be initialized."""
        validator = MonitoringValidator()
        assert validator.name == "monitoring"
        assert "monitoring" in validator.description.lower()
    
    @pytest.mark.asyncio
    async def test_validate_logging(self):
        """Test logging infrastructure validation."""
        validator = MonitoringValidator()
        
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("pathlib.Path.touch") as mock_touch:
                await validator._validate_logging()
                
                # Should create log directory
                assert mock_mkdir.called
                
                # Should check for log files
                assert any(
                    "Log file ready" in issue.message
                    for issue in validator.result.issues
                )


class TestSecurityValidator:
    """Test security validator."""
    
    @pytest.mark.asyncio
    async def test_security_validator_initialization(self):
        """Test security validator can be initialized."""
        validator = SecurityValidator()
        assert validator.name == "security"
        assert "security" in validator.description.lower()
    
    @pytest.mark.asyncio
    async def test_check_api_key_encryption(self):
        """Test API key encryption check."""
        validator = SecurityValidator()
        
        with patch("pathlib.Path.exists") as mock_exists:
            # Simulate .env file exists
            mock_exists.side_effect = lambda: True
            
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = """
                BINANCE_API_KEY=test_key_123
                BINANCE_API_SECRET=test_secret_456
                """
                
                await validator._check_api_key_encryption()
                
                # Should check for .env in gitignore
                assert mock_open.called


class TestValidationOrchestrator:
    """Test validation orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator can be initialized."""
        orchestrator = ValidationOrchestrator()
        assert isinstance(orchestrator.validators, dict)
        assert isinstance(orchestrator.results, dict)
    
    @pytest.mark.asyncio
    async def test_register_validator(self):
        """Test validator registration."""
        orchestrator = ValidationOrchestrator()
        validator = StrategyValidator()
        
        orchestrator.register_validator(validator)
        assert "strategy" in orchestrator.validators
        assert orchestrator.validators["strategy"] == validator
    
    @pytest.mark.asyncio
    async def test_validate_sequential(self):
        """Test sequential validation."""
        orchestrator = ValidationOrchestrator()
        
        # Create mock validator
        mock_validator = Mock(spec=BaseValidator)
        mock_validator.name = "test"
        mock_result = ValidationResult(validator_name="test")
        mock_result.status = ValidationStatus.PASSED
        mock_validator.validate.return_value = mock_result
        
        orchestrator.register_validator(mock_validator)
        
        results = await orchestrator.validate(
            mode="standard",
            validators=["test"],
            parallel=False
        )
        
        assert "test" in results
        assert results["test"].status == ValidationStatus.PASSED
        mock_validator.validate.assert_called_once_with("standard")
    
    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test summary generation."""
        orchestrator = ValidationOrchestrator()
        
        # Add test results
        result1 = ValidationResult(validator_name="test1")
        result1.status = ValidationStatus.PASSED
        
        result2 = ValidationResult(validator_name="test2")
        result2.status = ValidationStatus.FAILED
        
        orchestrator.results = {
            "test1": result1,
            "test2": result2
        }
        
        summary = orchestrator.get_summary()
        
        assert summary["total_validators"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["overall_status"] == "FAILED"


@pytest.mark.asyncio
async def test_full_validation_flow():
    """Test complete validation flow."""
    orchestrator = ValidationOrchestrator()
    
    # Register real validators (will mostly skip due to missing dependencies)
    orchestrator.register_validator(StrategyValidator())
    orchestrator.register_validator(RiskValidator())
    orchestrator.register_validator(DatabaseValidator())
    
    # Run quick validation
    results = await orchestrator.validate(mode="quick", parallel=True)
    
    # Should have results for each validator
    assert len(results) >= 3
    
    # Get summary
    summary = orchestrator.get_summary()
    assert "total_validators" in summary
    assert "overall_status" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])