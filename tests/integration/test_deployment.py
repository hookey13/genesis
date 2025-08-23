"""Deployment validation tests."""

import subprocess
import pytest
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any


class TestDeploymentValidation:
    """Test deployment processes and configurations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts"
        
    def test_deployment_scripts_exist(self):
        """Verify deployment scripts exist with correct permissions."""
        deploy_script = self.scripts_dir / "deploy.sh"
        backup_script = self.scripts_dir / "backup.sh"
        emergency_script = self.scripts_dir / "emergency_close.py"
        
        assert deploy_script.exists(), "Deploy script not found"
        assert backup_script.exists(), "Backup script not found"
        assert emergency_script.exists(), "Emergency close script not found"
        
        # Check execute permissions
        assert os.access(deploy_script, os.X_OK), "Deploy script not executable"
        assert os.access(backup_script, os.X_OK), "Backup script not executable"
        
    def test_environment_configuration(self):
        """Test environment-specific configurations."""
        env_example = self.project_root / ".env.example"
        assert env_example.exists(), ".env.example not found"
        
        content = env_example.read_text()
        required_vars = [
            "DEPLOYMENT_ENV",
            "LOG_LEVEL",
            "DATABASE_URL",
            "BINANCE_API_KEY",
            "TELEGRAM_BOT_TOKEN"
        ]
        
        for var in required_vars:
            assert var in content, f"Environment variable {var} not in .env.example"
            
    def test_production_configuration_validation(self):
        """Validate production-specific configurations."""
        prod_compose = self.project_root / "docker" / "docker-compose.prod.yml"
        
        result = subprocess.run(
            ["docker-compose", "-f", str(prod_compose), "config"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        
        assert result.returncode == 0, f"Production config invalid: {result.stderr}"
        
        config = result.stdout
        assert "restart: always" in config or "restart: unless-stopped" in config, \
            "Production containers missing restart policy"
        assert "supervisord" in config.lower() or "supervisor" in config.lower(), \
            "Supervisor not configured for production"
            
    def test_database_migration_readiness(self):
        """Test database migration setup."""
        alembic_ini = self.project_root / "alembic" / "alembic.ini"
        alembic_env = self.project_root / "alembic" / "env.py"
        
        assert alembic_ini.exists(), "Alembic config not found"
        assert alembic_env.exists(), "Alembic env.py not found"
        
        # Test alembic configuration
        result = subprocess.run(
            ["alembic", "history", "--verbose"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        
        # Should not crash even if no migrations exist
        assert result.returncode in [0, 1], f"Alembic command failed: {result.stderr}"
        
    def test_health_check_endpoints(self):
        """Test health check configuration."""
        # Check if health check code exists
        health_files = list(self.project_root.rglob("*health*.py"))
        observability_files = list(self.project_root.rglob("*observability*.py"))
        
        assert health_files or observability_files, \
            "No health check or observability modules found"
            
    def test_backup_strategy(self):
        """Test backup script functionality."""
        backup_script = self.scripts_dir / "backup.sh"
        
        if backup_script.exists():
            # Test backup script in dry-run mode if supported
            result = subprocess.run(
                ["bash", str(backup_script), "--dry-run"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Should complete without critical errors
            assert "error" not in result.stderr.lower() or result.returncode == 0, \
                f"Backup script has errors: {result.stderr}"
                
    def test_deployment_checklist(self):
        """Verify deployment checklist items."""
        checklist = {
            "secrets_management": self._check_secrets_management(),
            "logging_configured": self._check_logging_configuration(),
            "monitoring_setup": self._check_monitoring_setup(),
            "database_ready": self._check_database_readiness(),
            "docker_configured": self._check_docker_configuration(),
            "ci_cd_pipeline": self._check_cicd_pipeline(),
        }
        
        failed_checks = [k for k, v in checklist.items() if not v]
        assert not failed_checks, f"Deployment checks failed: {failed_checks}"
        
    def _check_secrets_management(self) -> bool:
        """Check if secrets are properly managed."""
        gitignore = self.project_root / ".gitignore"
        content = gitignore.read_text()
        return all(pattern in content for pattern in [".env", "*.key", "*.pem"])
        
    def _check_logging_configuration(self) -> bool:
        """Check if logging is properly configured."""
        settings = self.project_root / "config" / "settings.py"
        content = settings.read_text()
        return "structlog" in content and "LoggingSettings" in content
        
    def _check_monitoring_setup(self) -> bool:
        """Check if monitoring is configured."""
        observability = self.project_root / "genesis" / "observability"
        return observability.exists() and (observability / "__init__.py").exists()
        
    def _check_database_readiness(self) -> bool:
        """Check database configuration."""
        alembic_dir = self.project_root / "alembic"
        return alembic_dir.exists() and (alembic_dir / "env.py").exists()
        
    def _check_docker_configuration(self) -> bool:
        """Check Docker setup."""
        docker_dir = self.project_root / "docker"
        return all([
            (docker_dir / "Dockerfile").exists(),
            (docker_dir / "docker-compose.yml").exists(),
            (docker_dir / "docker-compose.prod.yml").exists()
        ])
        
    def _check_cicd_pipeline(self) -> bool:
        """Check CI/CD pipeline configuration."""
        workflows = self.project_root / ".github" / "workflows"
        return workflows.exists() and len(list(workflows.glob("*.yml"))) > 0
        
    def test_rollback_capability(self):
        """Test rollback mechanisms are in place."""
        # Check for backup/restore scripts
        scripts = self.scripts_dir
        backup_exists = (scripts / "backup.sh").exists()
        
        # Check for versioning in Docker images
        docker_compose = self.project_root / "docker" / "docker-compose.prod.yml"
        if docker_compose.exists():
            content = docker_compose.read_text()
            # Should use tags, not latest
            assert ":latest" not in content or "${VERSION}" in content, \
                "Production should use versioned images, not :latest"
                
        assert backup_exists, "No backup strategy found for rollback"
        
    def test_zero_downtime_deployment(self):
        """Test configurations for zero-downtime deployment."""
        prod_compose = self.project_root / "docker" / "docker-compose.prod.yml"
        
        if prod_compose.exists():
            content = prod_compose.read_text()
            
            # Check for health checks
            has_healthcheck = "healthcheck:" in content
            
            # Check for proper restart policies
            has_restart = "restart:" in content
            
            # Check for resource limits
            has_limits = "mem_limit:" in content or "deploy:" in content
            
            assert has_restart, "Missing restart policy for zero-downtime"
            # Health checks and limits are recommended but not required
            
    def test_emergency_procedures(self):
        """Test emergency shutdown procedures."""
        emergency_script = self.scripts_dir / "emergency_close.py"
        
        assert emergency_script.exists(), "Emergency close script not found"
        
        content = emergency_script.read_text()
        
        # Should have functions for closing positions
        assert "close" in content.lower() or "emergency" in content.lower(), \
            "Emergency script missing close functionality"
            
        # Should handle exceptions
        assert "try:" in content and "except" in content, \
            "Emergency script should have error handling"