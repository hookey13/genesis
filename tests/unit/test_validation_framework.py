"""Unit tests for the validation framework."""

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml

from genesis.validation.base import (
    CheckStatus,
    ValidationCheck,
    ValidationContext,
    ValidationEvidence,
    ValidationMetadata,
    ValidationResult,
    Validator
)
from genesis.validation.config import (
    PipelineConfig,
    PipelineStage,
    ValidationConfig,
    ValidatorConfig
)
from genesis.validation.exceptions import (
    ValidationConfigError,
    ValidationDependencyError,
    ValidationOverrideError,
    ValidationPipelineError,
    ValidationRetryExhausted,
    ValidationTimeout
)
from genesis.validation.orchestrator import ValidationOrchestrator
from genesis.validation.report import ReportGenerator, ValidationReport


class TestValidationBase:
    """Test base validation classes."""
    
    def test_check_status_enum(self):
        """Test CheckStatus enum values."""
        assert CheckStatus.PASSED.value == "passed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.WARNING.value == "warning"
        assert CheckStatus.SKIPPED.value == "skipped"
        assert CheckStatus.ERROR.value == "error"
        assert CheckStatus.IN_PROGRESS.value == "in_progress"
    
    def test_validation_metadata(self):
        """Test ValidationMetadata dataclass."""
        metadata = ValidationMetadata(
            version="1.0.0",
            environment="test",
            run_id="test-123",
            started_at=datetime.utcnow()
        )
        
        assert metadata.version == "1.0.0"
        assert metadata.environment == "test"
        assert metadata.run_id == "test-123"
        assert metadata.completed_at is None
        assert metadata.duration_ms is None
    
    def test_validation_context(self):
        """Test ValidationContext dataclass."""
        context = ValidationContext(
            genesis_root="/test/path",
            environment="test",
            run_mode="quick",
            dry_run=True,
            force_continue=False
        )
        
        assert context.genesis_root == "/test/path"
        assert context.environment == "test"
        assert context.run_mode == "quick"
        assert context.dry_run is True
        assert context.force_continue is False
    
    def test_validation_check(self):
        """Test ValidationCheck dataclass."""
        evidence = ValidationEvidence()
        check = ValidationCheck(
            id="test-check",
            name="Test Check",
            description="A test check",
            category="test",
            status=CheckStatus.PASSED,
            details="Check passed successfully",
            is_blocking=False,
            evidence=evidence,
            duration_ms=100.5,
            timestamp=datetime.utcnow()
        )
        
        assert check.id == "test-check"
        assert check.status == CheckStatus.PASSED
        assert check.is_blocking is False
        assert check.duration_ms == 100.5
        
        # Test to_dict conversion
        check_dict = check.to_dict()
        assert check_dict["id"] == "test-check"
        assert check_dict["status"] == "passed"
        assert check_dict["is_blocking"] is False
    
    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        metadata = ValidationMetadata(
            version="1.0.0",
            environment="test",
            run_id="test-123",
            started_at=datetime.utcnow()
        )
        
        evidence = ValidationEvidence()
        check1 = ValidationCheck(
            id="check1",
            name="Check 1",
            description="First check",
            category="test",
            status=CheckStatus.PASSED,
            details="Passed",
            is_blocking=False,
            evidence=evidence,
            duration_ms=50,
            timestamp=datetime.utcnow()
        )
        
        check2 = ValidationCheck(
            id="check2",
            name="Check 2",
            description="Second check",
            category="test",
            status=CheckStatus.FAILED,
            details="Failed",
            is_blocking=True,
            evidence=evidence,
            duration_ms=75,
            timestamp=datetime.utcnow()
        )
        
        result = ValidationResult(
            validator_id="test-validator",
            validator_name="Test Validator",
            status=CheckStatus.FAILED,
            message="Validation failed",
            checks=[check1, check2],
            evidence=evidence,
            metadata=metadata
        )
        
        # Test count updates
        result.update_counts()
        assert result.passed_checks == 1
        assert result.failed_checks == 1
        assert result.warning_checks == 0
        
        # Test score calculation
        score = result.calculate_score()
        assert score == Decimal("50.0")  # 1 passed out of 2
        
        # Test blocking failures detection
        assert result.has_blocking_failures() is True
    
    def test_validator_abstract_class(self):
        """Test Validator abstract base class."""
        
        class TestValidator(Validator):
            async def run_validation(self, context: ValidationContext) -> ValidationResult:
                return ValidationResult(
                    validator_id=self.validator_id,
                    validator_name=self.name,
                    status=CheckStatus.PASSED,
                    message="Test passed",
                    checks=[],
                    evidence=ValidationEvidence(),
                    metadata=context.metadata
                )
        
        validator = TestValidator("test-id", "Test Validator", "A test validator")
        
        assert validator.validator_id == "test-id"
        assert validator.name == "Test Validator"
        assert validator.description == "A test validator"
        assert validator.timeout_seconds == 60
        
        # Test dependency management
        validator.add_dependency("dep1")
        validator.add_dependency("dep2")
        assert "dep1" in validator.dependencies
        assert "dep2" in validator.dependencies
        
        validator.remove_dependency("dep1")
        assert "dep1" not in validator.dependencies
        assert "dep2" in validator.dependencies
        
        # Test configuration methods
        validator.set_critical(True)
        assert validator.is_critical is True
        
        validator.set_timeout(120)
        assert validator.timeout_seconds == 120
        
        validator.set_retry_policy(3, 10)
        assert validator.retry_count == 3
        assert validator.retry_delay_seconds == 10


# DependencyGraph tests removed - not in current implementation


class TestValidationOrchestrator:
    """Test validation orchestrator."""
    
    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator instance."""
        return ValidationOrchestrator(genesis_root=tmp_path)
    
    @pytest.fixture
    def mock_validator(self):
        """Create mock validator."""
        
        class MockValidator(Validator):
            async def run_validation(self, context: ValidationContext) -> ValidationResult:
                return ValidationResult(
                    validator_id=self.validator_id,
                    validator_name=self.name,
                    status=CheckStatus.PASSED,
                    message="Mock validation passed",
                    checks=[],
                    evidence=ValidationEvidence(),
                    metadata=context.metadata or ValidationMetadata(
                        version="1.0.0",
                        environment="test",
                        run_id="test",
                        started_at=datetime.utcnow()
                    )
                )
        
        return MockValidator("mock-validator", "Mock Validator", "A mock validator")
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.genesis_root is not None
        assert orchestrator.validators is not None
        assert orchestrator.pipeline_config is not None
    
    @pytest.mark.asyncio
    async def test_run_pipeline(self, orchestrator):
        """Test running a validation pipeline."""
        # The current orchestrator has predefined validators
        report = await orchestrator.run_pipeline("quick")
        
        assert report is not None
        assert report.pipeline_name == "quick"
        assert report.overall_score >= 0
    
    @pytest.mark.asyncio
    async def test_run_full_validation(self, orchestrator):
        """Test running full validation."""
        report = await orchestrator.run_full_validation()
        
        assert report is not None
        assert report.pipeline_name == "comprehensive"


class TestValidationConfig:
    """Test validation configuration management."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create config instance."""
        config_path = tmp_path / "validation_pipeline.yaml"
        return ValidationConfig(config_path)
    
    def test_load_defaults(self, config):
        """Test loading default configuration."""
        assert len(config.validators) > 0
        assert len(config.pipelines) > 0
        
        # Check default validators
        assert "test_coverage" in config.validators
        assert "security_scan" in config.validators
        
        # Check default pipelines
        assert "quick" in config.pipelines
        assert "standard" in config.pipelines
        assert "comprehensive" in config.pipelines
        assert "go_live" in config.pipelines
    
    def test_get_validator_config(self, config):
        """Test getting validator configuration."""
        validator_config = config.get_validator_config("test_coverage")
        
        assert validator_config is not None
        assert validator_config.id == "test_coverage"
        assert validator_config.name == "Test Coverage"
        assert validator_config.is_critical is True
    
    def test_get_pipeline_config(self, config):
        """Test getting pipeline configuration."""
        pipeline_config = config.get_pipeline_config("quick")
        
        assert pipeline_config is not None
        assert pipeline_config.name == "quick"
        assert pipeline_config.mode == "quick"
        assert pipeline_config.required_score == 70.0
    
    def test_save_config(self, config, tmp_path):
        """Test saving configuration."""
        output_path = tmp_path / "test_config.yaml"
        config.save_config(output_path)
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert "global" in data
        assert "validators" in data
        assert "pipelines" in data
    
    def test_validate_pipeline(self, config):
        """Test pipeline validation."""
        # Valid pipeline
        errors = config.validate_pipeline("quick")
        assert len(errors) == 0
        
        # Invalid pipeline
        errors = config.validate_pipeline("nonexistent")
        assert len(errors) > 0
        assert "not found" in errors[0]


class TestReportGenerator:
    """Test report generation."""
    
    @pytest.fixture
    def generator(self, tmp_path):
        """Create report generator instance."""
        return ReportGenerator(genesis_root=tmp_path)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample validation results."""
        metadata = ValidationMetadata(
            version="1.0.0",
            environment="test",
            run_id="test-123",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=5000
        )
        
        evidence = ValidationEvidence()
        
        check1 = ValidationCheck(
            id="check1",
            name="Check 1",
            description="First check",
            category="test",
            status=CheckStatus.PASSED,
            details="All good",
            is_blocking=False,
            evidence=evidence,
            duration_ms=100,
            timestamp=datetime.utcnow()
        )
        
        check2 = ValidationCheck(
            id="check2",
            name="Check 2",
            description="Second check",
            category="test",
            status=CheckStatus.FAILED,
            details="Something wrong",
            is_blocking=True,
            evidence=evidence,
            duration_ms=200,
            timestamp=datetime.utcnow(),
            remediation="Fix the issue"
        )
        
        result1 = ValidationResult(
            validator_id="validator1",
            validator_name="Validator 1",
            status=CheckStatus.PASSED,
            message="All checks passed",
            checks=[check1],
            evidence=evidence,
            metadata=metadata,
            score=Decimal("100.0")
        )
        result1.update_counts()
        
        result2 = ValidationResult(
            validator_id="validator2",
            validator_name="Validator 2",
            status=CheckStatus.FAILED,
            message="Some checks failed",
            checks=[check2],
            evidence=evidence,
            metadata=metadata,
            score=Decimal("0.0")
        )
        result2.update_counts()
        
        return {"validator1": result1, "validator2": result2}
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample report metadata."""
        return {
            "run_id": "test-123",
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": 5.0,
            "environment": "test",
            "mode": "quick",
            "overall_status": "failed",
            "overall_score": 50.0,
            "validators_run": 2,
            "validators_passed": 1,
            "validators_failed": 1,
            "validators_warning": 0,
            "validators_skipped": 0
        }
    
    def test_generate_json_report(self, generator, sample_results, sample_metadata):
        """Test JSON report generation."""
        json_report = generator.generate_json_report(sample_results, sample_metadata)
        
        assert json_report is not None
        assert '"validator1"' in json_report
        assert '"validator2"' in json_report
        assert '"passed"' in json_report
        assert '"failed"' in json_report
    
    def test_generate_yaml_report(self, generator, sample_results, sample_metadata):
        """Test YAML report generation."""
        yaml_report = generator.generate_yaml_report(sample_results, sample_metadata)
        
        assert yaml_report is not None
        assert "validator1" in yaml_report
        assert "validator2" in yaml_report
        assert "passed" in yaml_report
        assert "failed" in yaml_report
    
    def test_generate_markdown_report(self, generator, sample_results, sample_metadata):
        """Test Markdown report generation."""
        md_report = generator.generate_markdown_report(sample_results, sample_metadata)
        
        assert md_report is not None
        assert "# Validation Report" in md_report
        assert "## Summary" in md_report
        assert "## Detailed Results" in md_report
        assert "Validator 1" in md_report
        assert "Validator 2" in md_report
    
    @pytest.mark.asyncio
    async def test_save_report(self, generator, sample_results, sample_metadata, tmp_path):
        """Test saving report to file."""
        # Test JSON format
        json_path = await generator.save_report(
            sample_results,
            sample_metadata,
            format="json"
        )
        
        assert json_path.exists()
        assert json_path.suffix == ".json"
        
        # Test YAML format
        yaml_path = await generator.save_report(
            sample_results,
            sample_metadata,
            format="yaml"
        )
        
        assert yaml_path.exists()
        assert yaml_path.suffix == ".yaml"
        
        # Test Markdown format
        md_path = await generator.save_report(
            sample_results,
            sample_metadata,
            format="markdown"
        )
        
        assert md_path.exists()
        assert md_path.suffix == ".markdown"


class TestValidationExceptions:
    """Test custom validation exceptions."""
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationConfigError(
            "Invalid configuration",
            config_key="test_key",
            config_value="bad_value",
            expected_type="string"
        )
        
        assert str(error) == "Invalid configuration"
        assert error.config_key == "test_key"
        assert error.config_value == "bad_value"
        assert error.expected_type == "string"
    
    def test_validation_timeout(self):
        """Test ValidationTimeout exception."""
        error = ValidationTimeout(
            "Validation timed out",
            validator_id="test-validator",
            timeout_seconds=60
        )
        
        assert str(error) == "Validation timed out"
        assert error.validator_id == "test-validator"
        assert error.timeout_seconds == 60
    
    def test_validation_dependency_error(self):
        """Test ValidationDependencyError exception."""
        error = ValidationDependencyError(
            "Missing dependencies",
            validator_id="test-validator",
            missing_dependencies=["dep1", "dep2"],
            failed_dependencies=["dep3"]
        )
        
        assert str(error) == "Missing dependencies"
        assert error.validator_id == "test-validator"
        assert error.missing_dependencies == ["dep1", "dep2"]
        assert error.failed_dependencies == ["dep3"]
    
    def test_validation_pipeline_error(self):
        """Test ValidationPipelineError exception."""
        error = ValidationPipelineError(
            "Pipeline failed",
            pipeline_name="test-pipeline",
            stage="validation",
            failed_validators=["v1", "v2"]
        )
        
        assert str(error) == "Pipeline failed"
        assert error.pipeline_name == "test-pipeline"
        assert error.stage == "validation"
        assert error.failed_validators == ["v1", "v2"]
    
    def test_validation_retry_exhausted(self):
        """Test ValidationRetryExhausted exception."""
        last_error = Exception("Last attempt failed")
        error = ValidationRetryExhausted(
            "All retries exhausted",
            validator_id="test-validator",
            attempts=3,
            last_error=last_error
        )
        
        assert str(error) == "All retries exhausted"
        assert error.validator_id == "test-validator"
        assert error.attempts == 3
        assert error.last_error == last_error