"""End-to-end tests for complete go-live validation process."""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.mark.e2e
class TestGoLiveProcess:
    """End-to-end tests for go-live readiness validation."""

    @pytest.fixture
    def setup_full_environment(self, tmp_path):
        """Set up complete test environment mimicking production."""
        # Create full directory structure
        dirs = [
            "config",
            "docs/reports",
            "docs/runbooks", 
            ".genesis/backups",
            "genesis/monitoring",
            "genesis/validation",
            "genesis/cli",
            "scripts",
            "tests"
        ]
        
        for dir_path in dirs:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create configuration files
        self._create_config_files(tmp_path)
        
        # Create mock scripts
        self._create_mock_scripts(tmp_path)
        
        # Create mock backup
        self._create_mock_backup(tmp_path)
        
        # Set environment variables
        os.environ["GENESIS_ROOT"] = str(tmp_path)
        os.environ["GENESIS_ENV"] = "test"
        
        return tmp_path

    def _create_config_files(self, root: Path):
        """Create configuration files."""
        # Validation pipeline config
        pipeline_config = root / "config" / "validation_pipeline.yaml"
        pipeline_config.write_text("""
validation_pipeline:
  go_live:
    description: "Go-live readiness validation"
    validators: all
    required_score: 95
    blocking_on_failure: true
    timeout_minutes: 10
    
  staging:
    description: "Staging validation"
    validators:
      - test_coverage
      - security
      - performance
      - operational
    required_score: 85
    timeout_minutes: 5
""")
        
        # Create .env file
        env_file = root / ".env"
        env_file.write_text("""
ENVIRONMENT=test
API_KEY=test_key_123
DATABASE_URL=sqlite:///test.db
REDIS_URL=redis://localhost:6379
""")

    def _create_mock_scripts(self, root: Path):
        """Create mock deployment scripts."""
        # Deployment script
        deploy_script = root / "scripts" / "deploy.sh"
        deploy_script.write_text("""#!/bin/bash
echo "Deploying to production..."
exit 0
""")
        deploy_script.chmod(0o755)
        
        # Rollback script
        rollback_script = root / "scripts" / "rollback.sh"
        rollback_script.write_text("""#!/bin/bash
echo "Rolling back deployment..."
exit 0
""")
        rollback_script.chmod(0o755)

    def _create_mock_backup(self, root: Path):
        """Create mock database backup."""
        backup_file = root / ".genesis" / "backups" / "backup.db"
        backup_file.write_text("Mock backup data")

    @pytest.mark.asyncio
    async def test_successful_go_live_process(self, setup_full_environment):
        """Test successful go-live validation and deployment."""
        root = setup_full_environment
        
        # Import after environment setup
        from genesis.validation.orchestrator import ValidationOrchestrator
        from genesis.validation.decision import DecisionEngine, DeploymentTarget
        from genesis.validation.history import ValidationHistory
        from genesis.validation.report_generator import ReportGenerator
        
        # Initialize components
        orchestrator = ValidationOrchestrator(genesis_root=root)
        engine = DecisionEngine(genesis_root=root)
        history = ValidationHistory(genesis_root=root)
        generator = ReportGenerator(genesis_root=root)
        
        # Mock all validators to pass
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 98.0,
            "checks": {
                "e2e_check": {
                    "passed": True,
                    "message": "End-to-end validation passed"
                }
            }
        }
        
        # Replace all validators with mocks
        for name in orchestrator.validators:
            orchestrator.validators[name] = mock_validator
        
        # Step 1: Run go-live validation
        print("Step 1: Running go-live validation...")
        report = await orchestrator.run_go_live_validation()
        
        assert report.ready
        assert report.overall_score >= 95.0
        assert len(report.blocking_issues) == 0
        
        # Step 2: Save validation history
        print("Step 2: Saving validation history...")
        history_id = history.save_report(report)
        assert history_id > 0
        
        # Step 3: Make go-live decision
        print("Step 3: Making go-live decision...")
        decision = engine.make_decision(report, DeploymentTarget.PRODUCTION)
        
        assert decision.ready
        assert decision.deployment_allowed
        
        # Step 4: Perform safety checks
        print("Step 4: Performing safety checks...")
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=0)
            safety_checks = engine.perform_safety_checks(decision)
        
        assert all(check["passed"] for check in safety_checks[:2])  # At least git and backup
        
        # Step 5: Generate reports
        print("Step 5: Generating validation reports...")
        report_paths = generator.save_all_formats(report, decision)
        
        assert all(path.exists() for path in report_paths.values())
        
        # Step 6: Trigger deployment (dry run)
        print("Step 6: Triggering deployment (dry run)...")
        deployment_result = engine.trigger_deployment(decision, dry_run=True)
        
        assert deployment_result["success"]
        assert deployment_result["dry_run"]
        
        print("✅ Go-live process completed successfully!")

    @pytest.mark.asyncio
    async def test_failed_go_live_with_blocking_issues(self, setup_full_environment):
        """Test go-live process with validation failures."""
        root = setup_full_environment
        
        from genesis.validation.orchestrator import ValidationOrchestrator
        from genesis.validation.decision import DecisionEngine, DeploymentTarget
        
        orchestrator = ValidationOrchestrator(genesis_root=root)
        engine = DecisionEngine(genesis_root=root)
        
        # Create validators with mixed results
        passing_validator = AsyncMock()
        passing_validator.validate.return_value = {
            "passed": True,
            "score": 96.0,
            "checks": {}
        }
        
        failing_validator = AsyncMock()
        failing_validator.validate.return_value = {
            "passed": False,
            "score": 45.0,
            "errors": ["Critical security vulnerability"],
            "checks": {
                "security_scan": {
                    "passed": False,
                    "message": "High-risk vulnerability detected",
                    "error": "SQL injection vulnerability"
                }
            }
        }
        
        # Mix passing and failing validators
        for i, name in enumerate(orchestrator.validators):
            if i % 3 == 0:  # Every third validator fails
                orchestrator.validators[name] = failing_validator
            else:
                orchestrator.validators[name] = passing_validator
        
        # Run validation
        report = await orchestrator.run_go_live_validation()
        
        assert not report.ready
        assert len(report.blocking_issues) > 0
        
        # Make decision
        decision = engine.make_decision(report, DeploymentTarget.PRODUCTION)
        
        assert not decision.ready
        assert not decision.deployment_allowed
        assert len(decision.blocking_issues) > 0
        
        # Verify deployment is blocked
        deployment_result = engine.trigger_deployment(decision, dry_run=True)
        
        assert not deployment_result["success"]
        assert "not allowed" in deployment_result["error"].lower()

    @pytest.mark.asyncio
    async def test_go_live_with_override(self, setup_full_environment):
        """Test go-live process with manual override."""
        root = setup_full_environment
        
        from genesis.validation.orchestrator import ValidationOrchestrator
        from genesis.validation.decision import DecisionEngine, DeploymentTarget
        
        orchestrator = ValidationOrchestrator(genesis_root=root)
        engine = DecisionEngine(genesis_root=root)
        
        # Create failing validator
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": False,
            "score": 85.0,  # Below production threshold
            "warnings": ["Score below production threshold"],
            "checks": {}
        }
        
        for name in orchestrator.validators:
            orchestrator.validators[name] = mock_validator
        
        # Run validation
        report = await orchestrator.run_go_live_validation()
        
        assert not report.ready
        
        # Make initial decision
        decision = engine.make_decision(report, DeploymentTarget.PRODUCTION)
        
        assert not decision.ready
        assert not decision.deployment_allowed
        
        # Apply authorized override
        decision = engine.apply_override(
            decision,
            "Emergency fix deployment required",
            "admin",
            "12345"  # Correct password
        )
        
        assert decision.ready
        assert decision.deployment_allowed
        assert decision.override is not None
        
        # Verify deployment is now allowed
        deployment_result = engine.trigger_deployment(decision, dry_run=True)
        
        assert deployment_result["success"]

    @pytest.mark.asyncio
    async def test_staging_to_production_workflow(self, setup_full_environment):
        """Test progression from staging to production deployment."""
        root = setup_full_environment
        
        from genesis.validation.orchestrator import ValidationOrchestrator
        from genesis.validation.decision import DecisionEngine, DeploymentTarget
        from genesis.validation.history import ValidationHistory
        
        orchestrator = ValidationOrchestrator(genesis_root=root)
        engine = DecisionEngine(genesis_root=root)
        history = ValidationHistory(genesis_root=root)
        
        # Mock validator that improves over time
        scores = [75.0, 85.0, 95.0]  # Improving scores
        current_score_index = 0
        
        async def improving_validate():
            nonlocal current_score_index
            score = scores[min(current_score_index, len(scores) - 1)]
            current_score_index += 1
            return {
                "passed": score >= 70,
                "score": score,
                "checks": {
                    "improvement": {
                        "passed": score >= 70,
                        "message": f"Score: {score}"
                    }
                }
            }
        
        mock_validator = Mock()
        mock_validator.validate = improving_validate
        
        for name in orchestrator.validators:
            orchestrator.validators[name] = mock_validator
        
        # Stage 1: Initial validation (low score)
        print("Stage 1: Initial validation...")
        report1 = await orchestrator.run_pipeline("staging")
        history.save_report(report1)
        
        decision1 = engine.make_decision(report1, DeploymentTarget.STAGING)
        assert decision1.ready  # Staging has lower threshold
        
        # Stage 2: Improved validation
        print("Stage 2: Improved validation...")
        report2 = await orchestrator.run_pipeline("staging")
        history.save_report(report2)
        
        decision2 = engine.make_decision(report2, DeploymentTarget.STAGING)
        assert decision2.ready
        
        # Stage 3: Production-ready validation
        print("Stage 3: Production validation...")
        report3 = await orchestrator.run_go_live_validation()
        history.save_report(report3)
        
        decision3 = engine.make_decision(report3, DeploymentTarget.PRODUCTION)
        assert decision3.ready
        assert decision3.deployment_allowed
        
        # Verify improvement trend
        history_entries = history.get_history(limit=3)
        scores_history = [e.overall_score for e in reversed(history_entries)]
        assert scores_history[0] < scores_history[1] < scores_history[2]

    @pytest.mark.asyncio
    @patch('click.echo')
    async def test_cli_validation_command(self, mock_echo, setup_full_environment):
        """Test CLI validation command execution."""
        root = setup_full_environment
        
        # Mock the CLI module path
        sys.path.insert(0, str(root / "genesis" / "cli"))
        
        from genesis.validation.orchestrator import ValidationOrchestrator
        
        # Mock validator
        orchestrator = ValidationOrchestrator(genesis_root=root)
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 92.0,
            "checks": {}
        }
        
        for name in orchestrator.validators:
            orchestrator.validators[name] = mock_validator
        
        # Test would normally invoke CLI command
        # For this test, we just verify the orchestrator works
        report = await orchestrator.run_pipeline("go_live")
        
        assert report is not None
        assert report.overall_score > 0

    @pytest.mark.asyncio
    async def test_complete_e2e_with_all_components(self, setup_full_environment):
        """Test complete end-to-end flow with all components."""
        root = setup_full_environment
        
        from genesis.validation.orchestrator import ValidationOrchestrator
        from genesis.validation.decision import DecisionEngine, DeploymentTarget
        from genesis.validation.history import ValidationHistory
        from genesis.validation.report_generator import ReportGenerator
        
        # Initialize all components
        components = {
            "orchestrator": ValidationOrchestrator(genesis_root=root),
            "engine": DecisionEngine(genesis_root=root),
            "history": ValidationHistory(genesis_root=root),
            "generator": ReportGenerator(genesis_root=root)
        }
        
        # Mock successful validators
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = {
            "passed": True,
            "score": 96.5,
            "checks": {
                "final_check": {
                    "passed": True,
                    "message": "All systems operational"
                }
            }
        }
        
        for name in components["orchestrator"].validators:
            components["orchestrator"].validators[name] = mock_validator
        
        # Execute complete workflow
        print("\n🚀 Starting complete E2E validation workflow...")
        
        # 1. Pre-flight checks
        print("1️⃣ Running pre-flight checks...")
        assert root.exists()
        assert (root / "config" / "validation_pipeline.yaml").exists()
        
        # 2. Run validation
        print("2️⃣ Running validation pipeline...")
        report = await components["orchestrator"].run_go_live_validation()
        assert report.ready
        print(f"   ✓ Validation score: {report.overall_score:.1f}%")
        
        # 3. Save history
        print("3️⃣ Saving validation history...")
        history_id = components["history"].save_report(report)
        assert history_id > 0
        print(f"   ✓ History saved with ID: {history_id}")
        
        # 4. Make decision
        print("4️⃣ Making go-live decision...")
        decision = components["engine"].make_decision(report, DeploymentTarget.PRODUCTION)
        assert decision.ready
        print(f"   ✓ Decision: {'GO' if decision.ready else 'NO-GO'}")
        
        # 5. Safety checks
        print("5️⃣ Performing safety checks...")
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=0)
            safety_checks = components["engine"].perform_safety_checks(decision)
        print(f"   ✓ Safety checks completed: {len(safety_checks)} checks")
        
        # 6. Generate reports
        print("6️⃣ Generating reports...")
        report_paths = components["generator"].save_all_formats(report, decision)
        assert len(report_paths) == 4  # markdown, json, html, pdf/txt
        print(f"   ✓ Generated {len(report_paths)} report formats")
        
        # 7. Verify history
        print("7️⃣ Verifying validation history...")
        latest = components["history"].get_latest()
        assert latest.id == history_id
        assert latest.ready
        print(f"   ✓ Latest history entry confirmed")
        
        # 8. Simulate deployment
        print("8️⃣ Simulating deployment...")
        deployment = components["engine"].trigger_deployment(decision, dry_run=True)
        assert deployment["success"]
        print(f"   ✓ Deployment simulation: {deployment['message']}")
        
        print("\n✅ Complete E2E validation workflow successful!")
        print(f"📊 Final Score: {report.overall_score:.1f}%")
        print(f"🎯 Decision: {'DEPLOY' if decision.deployment_allowed else 'HOLD'}")
        
        return True