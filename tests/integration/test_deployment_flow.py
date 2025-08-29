"""
Integration tests for deployment flow.
Tests blue-green deployment, rollback, and audit trail.
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import pytest
import structlog

logger = structlog.get_logger(__name__)


class TestDeploymentFlow:
    """Test deployment flow including blue-green and rollback."""
    
    @pytest.fixture
    def scripts_dir(self):
        """Provide scripts directory path."""
        return Path(__file__).parent.parent.parent / "scripts"
    
    @pytest.fixture
    def deploy_script(self, scripts_dir):
        """Provide deploy script path."""
        return scripts_dir / "deploy.sh"
    
    @pytest.fixture
    def rollback_script(self, scripts_dir):
        """Provide rollback script path."""
        return scripts_dir / "rollback.sh"
    
    def test_deployment_scripts_exist(self, deploy_script, rollback_script):
        """Test that deployment scripts exist."""
        assert deploy_script.exists(), f"Deploy script not found at {deploy_script}"
        assert rollback_script.exists(), f"Rollback script not found at {rollback_script}"
    
    def test_deploy_script_syntax(self, deploy_script):
        """Test deploy script has valid bash syntax."""
        try:
            result = subprocess.run(
                ["bash", "-n", str(deploy_script)],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0, f"Deploy script syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.skip("Bash syntax check timed out")
        except FileNotFoundError:
            pytest.skip("Bash not available")
    
    def test_rollback_script_syntax(self, rollback_script):
        """Test rollback script has valid bash syntax."""
        try:
            result = subprocess.run(
                ["bash", "-n", str(rollback_script)],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0, f"Rollback script syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            pytest.skip("Bash syntax check timed out")
        except FileNotFoundError:
            pytest.skip("Bash not available")
    
    def test_deploy_script_functions(self, deploy_script):
        """Test that deploy script has required functions."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Required functions for blue-green deployment
            required_functions = [
                'check_prerequisites',
                'get_current_color',
                'get_target_color',
                'deploy_new_version',
                'wait_for_health',
                'run_smoke_tests',
                'drain_connections',
                'switch_traffic',
                'rollback',
                'create_deployment_record'
            ]
            
            for func in required_functions:
                assert f"{func}()" in content, f"Missing function: {func}"
    
    def test_rollback_script_functions(self, rollback_script):
        """Test that rollback script has required functions."""
        with open(rollback_script, 'r') as f:
            content = f.read()
            
            # Required functions for rollback
            required_functions = [
                'get_current_color',
                'get_previous_deployment',
                'emergency_rollback'
            ]
            
            for func in required_functions:
                assert f"{func}()" in content, f"Missing function: {func}"
    
    def test_deployment_logging(self, deploy_script):
        """Test that deployment script has proper logging."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for logging functions
            assert 'log_info' in content, "Missing log_info function"
            assert 'log_warn' in content, "Missing log_warn function"
            assert 'log_error' in content, "Missing log_error function"
            
            # Check for deployment log file
            assert 'DEPLOYMENT_LOG' in content, "Missing deployment log configuration"
    
    def test_health_check_integration(self, deploy_script):
        """Test that deploy script integrates with health checks."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for health check usage
            assert 'genesis.api.health' in content, "Should use health check module"
            assert 'readiness' in content, "Should check readiness"
            assert 'HEALTH_CHECK_TIMEOUT' in content, "Should have health check timeout"
    
    def test_blue_green_logic(self, deploy_script):
        """Test blue-green deployment logic."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for blue-green color handling
            assert 'blue' in content and 'green' in content, "Should handle blue and green"
            assert 'get_current_color' in content, "Should determine current color"
            assert 'get_target_color' in content, "Should determine target color"
            
            # Check for zero-downtime features
            assert 'drain_connections' in content, "Should drain connections"
            assert 'switch_traffic' in content, "Should switch traffic"
    
    def test_rollback_mechanism(self, deploy_script, rollback_script):
        """Test rollback mechanism is properly configured."""
        # Check deploy script has rollback
        with open(deploy_script, 'r') as f:
            deploy_content = f.read()
            assert 'rollback()' in deploy_content, "Deploy should have rollback function"
            assert 'save_current_state' in deploy_content, "Should save state for rollback"
            
        # Check rollback script
        with open(rollback_script, 'r') as f:
            rollback_content = f.read()
            assert 'emergency_rollback' in rollback_content, "Should have emergency rollback"
            assert 'get_previous_deployment' in rollback_content, "Should find previous deployment"
    
    def test_deployment_record_creation(self, deploy_script):
        """Test that deployment records are created."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for deployment record function
            assert 'create_deployment_record' in content, "Should create deployment records"
            
            # Check record contains required fields
            assert '"timestamp"' in content, "Record should have timestamp"
            assert '"version"' in content, "Record should have version"
            assert '"status"' in content, "Record should have status"
            assert '"git_commit"' in content, "Record should have git commit"
            assert '"deployer"' in content, "Record should have deployer"
    
    def test_dry_run_support(self, deploy_script):
        """Test that deploy script supports dry run mode."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for dry run support
            assert 'dry_run' in content, "Should support dry run mode"
            assert 'DRY RUN' in content, "Should indicate dry run mode"
    
    def test_error_handling(self, deploy_script):
        """Test error handling in deployment script."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for error handling
            assert 'set -euo pipefail' in content, "Should use strict error handling"
            assert 'if !' in content, "Should check command success"
            assert 'exit 1' in content, "Should exit on error"
            
            # Check for rollback on failure
            assert 'rollback' in content and 'if' in content, "Should rollback on failure"
    
    def test_smoke_tests_configuration(self, deploy_script):
        """Test smoke tests are configured."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for smoke tests
            assert 'run_smoke_tests' in content, "Should run smoke tests"
            assert 'genesis.api.health detailed' in content, "Should run detailed health check"
            assert 'Repository' in content, "Should check database connectivity"
    
    def test_connection_draining(self, deploy_script):
        """Test connection draining for zero-downtime."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for connection draining
            assert 'drain_connections' in content, "Should drain connections"
            assert 'DRAIN_TIMEOUT' in content, "Should have drain timeout"
            assert 'SIGTERM' in content, "Should send SIGTERM for graceful shutdown"
    
    def test_git_tagging(self, deploy_script):
        """Test that deployments are tagged in git."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for git tagging
            assert 'git tag' in content, "Should create git tags"
            assert 'deployed-' in content, "Should use deployment tag prefix"
    
    def test_environment_validation(self, deploy_script):
        """Test environment validation."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for environment file validation
            assert '.env.production' in content, "Should check for production env file"
            assert 'check_prerequisites' in content, "Should check prerequisites"
            
            # Check Docker validation
            assert 'command -v docker' in content, "Should check Docker installation"
            assert 'docker-compose' in content or 'docker compose' in content, \
                "Should check Docker Compose"
    
    def test_database_backup(self, deploy_script):
        """Test database backup during deployment."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for database backup
            assert 'genesis.db.backup' in content, "Should backup database"
            assert 'cp' in content and '.genesis/data' in content, \
                "Should copy database for backup"
    
    def test_deployment_verification(self, deploy_script):
        """Test post-deployment verification."""
        with open(deploy_script, 'r') as f:
            content = f.read()
            
            # Check for verification steps
            assert 'wait_for_health' in content, "Should wait for health"
            assert 'run_smoke_tests' in content, "Should run smoke tests"
            assert 'Deployment completed successfully' in content, \
                "Should confirm successful deployment"
    
    def test_rollback_record_creation(self, rollback_script):
        """Test that rollback creates audit records."""
        with open(rollback_script, 'r') as f:
            content = f.read()
            
            # Check for rollback record
            assert 'rollback_' in content, "Should create rollback record"
            assert '"type": "rollback"' in content, "Should mark as rollback"
            assert '"reason"' in content, "Should include rollback reason"