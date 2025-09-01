"""Integration tests for full validation pipeline."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from genesis.validation.decision import DecisionEngine, DeploymentTarget
from genesis.validation.history import ValidationHistory
from genesis.validation.orchestrator import ValidationOrchestrator
from genesis.validation.report_generator import ReportGenerator


@pytest.mark.integration
class TestFullValidationPipeline:
    """Test complete validation pipeline integration."""

    @pytest.fixture
    async def setup_environment(self, tmp_path):
        """Set up test environment with all components."""
        # Create directory structure
        (tmp_path / "config").mkdir()
        (tmp_path / "docs" / "reports").mkdir(parents=True)
        (tmp_path / ".genesis").mkdir()
        (tmp_path / "genesis" / "monitoring").mkdir(parents=True)
        (tmp_path / "scripts").mkdir()
        
        # Create mock configuration files
        config_file = tmp_path / "config" / "validation_pipeline.yaml"
        config_file.write_text("""
validation_pipeline:
  test_quick:
    description: "Quick test pipeline"
    validators:
      - test_coverage
      - security
    timeout_minutes: 1
    required_score: 70
    
  test_comprehensive:
    description: "Comprehensive test pipeline"
    validators: all
    timeout_minutes: 5
    required_score: 85
    blocking_on_failure: true
""")
        
        # Create mock metrics collector
        metrics_file = tmp_path / "genesis" / "monitoring" / "metrics_collector.py"
        metrics_file.write_text("# Mock metrics collector")
        
        # Set environment
        os.environ["GENESIS_ROOT"] = str(tmp_path)
        
        return {
            "root": tmp_path,
            "orchestrator": ValidationOrchestrator(genesis_root=tmp_path),
            "engine": DecisionEngine(genesis_root=tmp_path),
            "history": ValidationHistory(genesis_root=tmp_path),
            "generator": ReportGenerator(genesis_root=tmp_path)
        }

    @pytest.mark.asyncio
    async def test_quick_validation_pipeline(self, setup_environment):
        """Test running quick validation pipeline."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        
        # Mock validators
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 85.0,
            "checks": {
                "basic_check": {
                    "passed": True,
                    "message": "All good"
                }
            }
        }
        
        orchestrator.validators = {
            "test_coverage": mock_validator,
            "security": mock_validator
        }
        
        # Run quick pipeline
        report = await orchestrator.run_pipeline("test_quick")
        
        assert report.pipeline_name == "test_quick"
        assert report.overall_score >= 70
        assert report.ready
        assert len(report.results) == 2

    @pytest.mark.asyncio
    async def test_comprehensive_validation_pipeline(self, setup_environment):
        """Test running comprehensive validation pipeline."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        
        # Mock different validators with varying scores
        validators = {}
        scores = [95, 88, 92, 75, 90, 85, 93, 87]
        
        for i, (name, score) in enumerate(zip(orchestrator.validators.keys(), scores)):
            mock = AsyncMock()
            mock.validate.return_value = {
                "passed": score >= 80,
                "score": float(score),
                "checks": {
                    f"check_{i}": {
                        "passed": score >= 80,
                        "message": f"Score: {score}"
                    }
                }
            }
            validators[name] = mock
        
        orchestrator.validators = validators
        
        # Run comprehensive pipeline
        report = await orchestrator.run_pipeline("test_comprehensive")
        
        assert report.pipeline_name == "test_comprehensive"
        assert len(report.results) == len(validators)
        assert report.overall_score > 0
        
        # Check if blocking issues are detected
        if report.overall_score < 85:
            assert not report.ready

    @pytest.mark.asyncio
    async def test_validation_with_history_tracking(self, setup_environment):
        """Test validation with history tracking."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        history = env["history"]
        
        # Mock validator
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 90.0,
            "checks": {}
        }
        
        orchestrator.validators = {"test_coverage": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["test_coverage"]
        
        # Run validation
        report = await orchestrator.run_pipeline("test_quick")
        
        # Save to history
        history_id = history.save_report(report)
        assert history_id > 0
        
        # Retrieve from history
        latest = history.get_latest()
        assert latest is not None
        assert latest.pipeline_name == "test_quick"
        assert latest.overall_score == 90.0

    @pytest.mark.asyncio
    async def test_validation_to_decision_flow(self, setup_environment):
        """Test flow from validation to go-live decision."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        engine = env["engine"]
        
        # Mock successful validation
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 96.0,
            "checks": {
                "production_ready": {
                    "passed": True,
                    "message": "System ready for production"
                }
            }
        }
        
        orchestrator.validators = {"test_coverage": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["test_coverage"]
        
        # Run validation
        report = await orchestrator.run_pipeline("test_quick")
        
        # Make go-live decision
        decision = engine.make_decision(report, DeploymentTarget.PRODUCTION)
        
        assert decision.ready
        assert decision.deployment_allowed
        assert decision.score >= 95.0

    @pytest.mark.asyncio
    async def test_validation_report_generation(self, setup_environment):
        """Test generating reports from validation results."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        generator = env["generator"]
        
        # Mock validator
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 88.0,
            "checks": {
                "test": {"passed": True, "message": "OK"}
            }
        }
        
        orchestrator.validators = {"test_coverage": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["test_coverage"]
        
        # Run validation
        report = await orchestrator.run_pipeline("test_quick")
        
        # Generate reports
        markdown = generator.generate_markdown(report)
        assert "Go-Live Readiness Report" in markdown
        assert "88.0%" in markdown
        
        json_report = generator.generate_json(report)
        data = json.loads(json_report)
        assert data["overall_score"] == 88.0

    @pytest.mark.asyncio
    async def test_failed_validation_handling(self, setup_environment):
        """Test handling of failed validations."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        engine = env["engine"]
        
        # Mock failing validator
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": False,
            "score": 45.0,
            "errors": ["Critical security issue found"],
            "checks": {
                "security_scan": {
                    "passed": False,
                    "message": "Vulnerabilities detected",
                    "error": "CVE-2024-1234 found"
                }
            }
        }
        
        orchestrator.validators = {"security": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["security"]
        
        # Run validation
        report = await orchestrator.run_pipeline("test_quick")
        
        assert not report.ready
        assert len(report.blocking_issues) > 0
        
        # Make decision
        decision = engine.make_decision(report, DeploymentTarget.PRODUCTION)
        
        assert not decision.ready
        assert not decision.deployment_allowed
        assert len(decision.blocking_issues) > 0

    @pytest.mark.asyncio
    async def test_validation_with_mixed_results(self, setup_environment):
        """Test validation with mixed pass/fail results."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        
        # Create validators with different results
        passing_validator = AsyncMock()
        passing_validator.validate.return_value = {
            "passed": True,
            "score": 95.0,
            "checks": {"check1": {"passed": True, "message": "Good"}}
        }
        
        warning_validator = AsyncMock()
        warning_validator.validate.return_value = {
            "passed": True,
            "score": 78.0,
            "warnings": ["Performance could be better"],
            "checks": {
                "check2": {
                    "passed": True,
                    "message": "Passed with warnings",
                    "warning": "Consider optimization"
                }
            }
        }
        
        failing_validator = AsyncMock()
        failing_validator.validate.return_value = {
            "passed": False,
            "score": 55.0,
            "errors": ["Test coverage too low"],
            "checks": {
                "coverage": {
                    "passed": False,
                    "message": "Coverage below threshold",
                    "error": "Only 55% coverage"
                }
            }
        }
        
        orchestrator.validators = {
            "security": passing_validator,
            "performance": warning_validator,
            "test_coverage": failing_validator
        }
        
        # Run comprehensive pipeline
        report = await orchestrator.run_pipeline("test_comprehensive")
        
        # Check aggregated results
        assert not report.overall_passed  # One validator failed
        assert report.overall_score == pytest.approx(76.0, rel=1.0)  # Average of scores
        assert len(report.blocking_issues) > 0  # Failed validator creates blocking issue

    @pytest.mark.asyncio
    async def test_validation_history_comparison(self, setup_environment):
        """Test comparing validation runs over time."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        history = env["history"]
        
        # First run with lower score
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 75.0,
            "checks": {}
        }
        
        orchestrator.validators = {"test": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["test"]
        
        report1 = await orchestrator.run_pipeline("test_quick")
        id1 = history.save_report(report1)
        
        # Second run with improved score
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 90.0,
            "checks": {}
        }
        
        report2 = await orchestrator.run_pipeline("test_quick")
        id2 = history.save_report(report2)
        
        # Compare reports
        comparison = history.compare_reports(id1, id2)
        
        assert comparison["score_diff"] == 15.0
        assert len(comparison["validators"]["improved"]) > 0

    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_validation_with_safety_checks(self, mock_run, setup_environment):
        """Test validation with deployment safety checks."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        engine = env["engine"]
        
        # Mock successful validation
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 97.0,
            "checks": {}
        }
        
        orchestrator.validators = {"test": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["test"]
        
        # Mock git status (clean repo)
        mock_run.return_value = Mock(stdout="", returncode=0)
        
        # Run validation
        report = await orchestrator.run_pipeline("test_quick")
        decision = engine.make_decision(report, DeploymentTarget.PRODUCTION)
        
        # Perform safety checks
        safety_checks = engine.perform_safety_checks(decision)
        
        assert len(safety_checks) > 0
        
        # Verify git status was checked
        git_check = next((c for c in safety_checks if c["name"] == "git_status"), None)
        assert git_check is not None
        assert git_check["passed"]

    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self, setup_environment):
        """Test complete validation workflow from start to report."""
        env = await setup_environment
        orchestrator = env["orchestrator"]
        engine = env["engine"]
        history = env["history"]
        generator = env["generator"]
        
        # Setup mock validators
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 92.0,
            "checks": {
                "integration_test": {
                    "passed": True,
                    "message": "All integration tests pass"
                }
            }
        }
        
        orchestrator.validators = {"test": mock_validator}
        orchestrator.pipeline_config["test_quick"]["validators"] = ["test"]
        
        # 1. Run validation
        report = await orchestrator.run_pipeline("test_quick")
        assert report is not None
        
        # 2. Save to history
        history_id = history.save_report(report)
        assert history_id > 0
        
        # 3. Make go-live decision
        decision = engine.make_decision(report, DeploymentTarget.STAGING)
        assert decision.ready  # Score is above staging threshold
        
        # 4. Generate reports
        paths = generator.save_all_formats(report, decision)
        assert all(p.exists() for p in paths.values())
        
        # 5. Verify history
        latest = history.get_latest()
        assert latest.id == history_id
        assert latest.ready