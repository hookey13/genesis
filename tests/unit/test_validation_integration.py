"""Unit tests for validation integration components."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from genesis.validation.decision import (
    DecisionEngine,
    DeploymentTarget,
    GoLiveDecision,
    Override,
    UnauthorizedError,
)
from genesis.validation.history import ValidationHistory, ValidationHistoryEntry
from genesis.validation.orchestrator import (
    ValidationCheck,
    ValidationOrchestrator,
    ValidationReport,
    ValidationResult,
)
from genesis.validation.report_generator import ReportGenerator


class TestValidationOrchestrator:
    """Test validation orchestrator functionality."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator instance with test config."""
        # Create test pipeline config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "validation_pipeline.yaml"
        config_file.write_text("""
validation_pipeline:
  test:
    description: "Test pipeline"
    validators:
      - test_coverage
      - security
    timeout_minutes: 1
""")
        return ValidationOrchestrator(genesis_root=tmp_path)

    @pytest.mark.asyncio
    async def test_run_pipeline(self, orchestrator):
        """Test running a validation pipeline."""
        # Mock validators
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 95.0,
            "checks": {
                "test_check": {
                    "passed": True,
                    "message": "Check passed"
                }
            }
        }
        
        orchestrator.validators = {
            "test_coverage": mock_validator,
            "security": mock_validator
        }
        
        # Run pipeline
        report = await orchestrator.run_pipeline("test")
        
        assert report.pipeline_name == "test"
        assert len(report.results) == 2
        assert report.overall_score == 95.0
        assert report.overall_passed

    @pytest.mark.asyncio
    async def test_run_pipeline_with_failure(self, orchestrator):
        """Test pipeline with failing validators."""
        # Mock failing validator
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": False,
            "score": 45.0,
            "errors": ["Test error"],
            "checks": {
                "failing_check": {
                    "passed": False,
                    "message": "Check failed",
                    "error": "Critical error"
                }
            }
        }
        
        orchestrator.validators = {"test_coverage": mock_validator}
        orchestrator.pipeline_config["test"]["validators"] = ["test_coverage"]
        
        # Run pipeline
        report = await orchestrator.run_pipeline("test")
        
        assert not report.overall_passed
        assert report.overall_score == 45.0
        assert len(report.blocking_issues) > 0
        assert not report.ready

    def test_load_pipeline_config(self, tmp_path):
        """Test loading pipeline configuration."""
        orchestrator = ValidationOrchestrator(genesis_root=tmp_path)
        
        # Should use default config when file doesn't exist
        assert "quick" in orchestrator.pipeline_config
        assert "standard" in orchestrator.pipeline_config
        assert "go_live" in orchestrator.pipeline_config

    @pytest.mark.asyncio
    async def test_validator_timeout(self, orchestrator):
        """Test validator timeout handling."""
        # Mock slow validator
        async def slow_validate():
            await asyncio.sleep(10)
            return {"passed": True, "score": 100}
        
        mock_validator = Mock()
        mock_validator.validate = slow_validate
        
        orchestrator.validators = {"slow": mock_validator}
        
        # Run with timeout
        result = await orchestrator._run_validator("slow", "test", mock_validator)
        
        assert not result.passed
        assert result.score == 0
        assert "timed out" in result.errors[0].lower()


class TestDecisionEngine:
    """Test go-live decision engine."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create decision engine instance."""
        return DecisionEngine(genesis_root=tmp_path)

    @pytest.fixture
    def sample_report(self):
        """Create sample validation report."""
        results = [
            ValidationResult(
                validator_name="test",
                category="technical",
                passed=True,
                score=98.0,
                checks=[],
                duration_seconds=1.0
            )
        ]
        
        return ValidationReport(
            pipeline_name="test",
            timestamp=datetime.utcnow(),
            duration_seconds=10.0,
            overall_passed=True,
            overall_score=98.0,
            results=results,
            blocking_issues=[],
            ready=True
        )

    def test_make_decision_ready(self, engine, sample_report):
        """Test making a go-live decision when ready."""
        decision = engine.make_decision(sample_report, DeploymentTarget.PRODUCTION)
        
        assert decision.ready
        assert decision.score == 98.0
        assert decision.deployment_allowed
        assert len(decision.blocking_issues) == 0

    def test_make_decision_not_ready(self, engine, sample_report):
        """Test making decision when not ready."""
        # Modify report to fail
        sample_report.overall_score = 70.0
        sample_report.ready = False
        
        decision = engine.make_decision(sample_report, DeploymentTarget.PRODUCTION)
        
        assert not decision.ready
        assert not decision.deployment_allowed

    def test_apply_override_authorized(self, engine, sample_report):
        """Test applying authorized override."""
        decision = engine.make_decision(sample_report)
        
        # Apply override with correct credentials
        decision = engine.apply_override(
            decision,
            "Emergency deployment needed",
            "admin",
            "12345"  # Correct password for admin
        )
        
        assert decision.ready
        assert decision.deployment_allowed
        assert decision.override is not None
        assert decision.override.authorized_by == "admin"

    def test_apply_override_unauthorized(self, engine, sample_report):
        """Test unauthorized override attempt."""
        decision = engine.make_decision(sample_report)
        
        # Try with wrong password
        with pytest.raises(UnauthorizedError):
            engine.apply_override(
                decision,
                "Trying to override",
                "admin",
                "wrong_password"
            )

    @patch('subprocess.run')
    def test_trigger_deployment_dry_run(self, mock_run, engine, sample_report):
        """Test dry run deployment."""
        decision = engine.make_decision(sample_report)
        
        result = engine.trigger_deployment(decision, dry_run=True)
        
        assert result["success"]
        assert result["dry_run"]
        mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_perform_safety_checks(self, mock_run, engine, sample_report):
        """Test safety checks before deployment."""
        decision = engine.make_decision(sample_report)
        
        # Mock git status
        mock_run.return_value = Mock(stdout="", returncode=0)
        
        checks = engine.perform_safety_checks(decision)
        
        assert len(checks) > 0
        
        # Find git status check
        git_check = next((c for c in checks if c["name"] == "git_status"), None)
        assert git_check is not None
        assert git_check["passed"]


class TestValidationHistory:
    """Test validation history tracking."""

    @pytest.fixture
    def history(self, tmp_path):
        """Create history instance."""
        return ValidationHistory(genesis_root=tmp_path)

    @pytest.fixture
    def sample_report(self):
        """Create sample validation report."""
        results = [
            ValidationResult(
                validator_name="test",
                category="technical",
                passed=True,
                score=95.0,
                checks=[],
                duration_seconds=1.0
            )
        ]
        
        return ValidationReport(
            pipeline_name="test",
            timestamp=datetime.utcnow(),
            duration_seconds=10.0,
            overall_passed=True,
            overall_score=95.0,
            results=results,
            ready=True
        )

    def test_save_report(self, history, sample_report):
        """Test saving validation report to history."""
        history_id = history.save_report(sample_report)
        
        assert history_id is not None
        assert history_id > 0

    def test_get_latest(self, history, sample_report):
        """Test getting latest validation entry."""
        # Save a report
        history.save_report(sample_report)
        
        # Get latest
        latest = history.get_latest()
        
        assert latest is not None
        assert latest.pipeline_name == "test"
        assert latest.overall_score == 95.0

    def test_get_history(self, history, sample_report):
        """Test getting validation history."""
        # Save multiple reports
        for i in range(3):
            sample_report.overall_score = 90 + i
            history.save_report(sample_report)
        
        # Get history
        entries = history.get_history(limit=5)
        
        assert len(entries) == 3
        assert entries[0].overall_score == 92  # Latest first

    def test_compare_reports(self, history, sample_report):
        """Test comparing two validation reports."""
        # Save two reports with different scores
        sample_report.overall_score = 85.0
        id1 = history.save_report(sample_report)
        
        sample_report.overall_score = 95.0
        id2 = history.save_report(sample_report)
        
        # Compare
        comparison = history.compare_reports(id1, id2)
        
        assert comparison["score_diff"] == 10.0
        assert not comparison["ready_changed"]  # Both are ready

    def test_cleanup_old_entries(self, history, sample_report):
        """Test cleaning up old history entries."""
        # Save old report
        old_report = sample_report
        old_report.timestamp = datetime.utcnow() - timedelta(days=40)
        history.save_report(old_report)
        
        # Save recent report
        sample_report.timestamp = datetime.utcnow()
        history.save_report(sample_report)
        
        # Cleanup old entries
        history.cleanup_old_entries(days_to_keep=30)
        
        # Check only recent entry remains
        entries = history.get_history()
        assert len(entries) == 1


class TestReportGenerator:
    """Test report generation."""

    @pytest.fixture
    def generator(self, tmp_path):
        """Create report generator instance."""
        return ReportGenerator(genesis_root=tmp_path)

    @pytest.fixture
    def sample_report(self):
        """Create sample validation report."""
        results = [
            ValidationResult(
                validator_name="test_coverage",
                category="technical",
                passed=True,
                score=95.0,
                checks=[
                    ValidationCheck(
                        name="coverage",
                        passed=True,
                        message="Coverage is sufficient"
                    )
                ],
                duration_seconds=1.0
            ),
            ValidationResult(
                validator_name="security",
                category="security",
                passed=False,
                score=75.0,
                checks=[
                    ValidationCheck(
                        name="secrets",
                        passed=False,
                        message="Found hardcoded secrets",
                        severity="critical"
                    )
                ],
                duration_seconds=2.0,
                errors=["Hardcoded API key found"]
            )
        ]
        
        return ValidationReport(
            pipeline_name="test",
            timestamp=datetime.utcnow(),
            duration_seconds=10.0,
            overall_passed=False,
            overall_score=85.0,
            results=results,
            blocking_issues=[
                ValidationCheck(
                    name="secrets",
                    passed=False,
                    message="Found hardcoded secrets",
                    severity="critical"
                )
            ],
            ready=False
        )

    def test_generate_markdown(self, generator, sample_report):
        """Test markdown report generation."""
        markdown = generator.generate_markdown(sample_report)
        
        assert "# Go-Live Readiness Report" in markdown
        assert "NOT READY FOR DEPLOYMENT" in markdown
        assert "85.0%" in markdown
        assert "Blocking Issues" in markdown
        assert "secrets" in markdown

    def test_generate_json(self, generator, sample_report):
        """Test JSON report generation."""
        json_str = generator.generate_json(sample_report)
        data = json.loads(json_str)
        
        assert data["pipeline_name"] == "test"
        assert data["overall_score"] == 85.0
        assert not data["ready"]
        assert len(data["results"]) == 2
        assert "recommendations" in data

    def test_generate_html(self, generator, sample_report):
        """Test HTML report generation."""
        html = generator.generate_html(sample_report)
        
        assert "<!DOCTYPE html>" in html
        assert "Go-Live Readiness Report" in html
        assert "NOT READY" in html
        assert "85.0%" in html

    def test_save_all_formats(self, generator, sample_report):
        """Test saving reports in all formats."""
        paths = generator.save_all_formats(sample_report)
        
        assert "markdown" in paths
        assert "json" in paths
        assert "html" in paths
        assert "pdf" in paths  # Actually saves as .txt for now
        
        # Check files exist
        for format_name, path in paths.items():
            assert path.exists()

    def test_generate_recommendations(self, generator, sample_report):
        """Test recommendation generation."""
        recommendations = generator._generate_recommendations(sample_report)
        
        assert len(recommendations) > 0
        
        # Should recommend improving score
        assert any("85.0%" in r for r in recommendations)
        
        # Should mention blocking issues
        assert any("blocking" in r.lower() for r in recommendations)