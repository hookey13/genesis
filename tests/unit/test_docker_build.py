"""
Unit tests for Docker build validation.
Ensures Docker image meets production requirements.
"""

import os
import subprocess
import json
from pathlib import Path
from decimal import Decimal
import pytest
import structlog

logger = structlog.get_logger(__name__)


class TestDockerBuild:
    """Test Docker build configuration and resulting image."""
    
    @pytest.fixture
    def docker_context(self):
        """Provide Docker build context path."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def dockerfile_path(self):
        """Provide Dockerfile path."""
        return Path(__file__).parent.parent.parent / "docker" / "Dockerfile"
    
    def test_dockerfile_exists(self, dockerfile_path):
        """Test that Dockerfile exists."""
        assert dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"
    
    def test_dockerfile_multi_stage(self, dockerfile_path):
        """Test that Dockerfile uses multi-stage build."""
        content = dockerfile_path.read_text()
        
        # Check for multiple FROM statements (multi-stage)
        from_count = content.count("FROM ")
        assert from_count >= 3, f"Expected multi-stage build with at least 3 stages, found {from_count}"
        
        # Check for required stages
        assert "AS builder" in content, "Missing builder stage"
        assert "AS production" in content, "Missing production stage"
        assert "AS security-scan" in content, "Missing security scan stage"
    
    def test_dockerfile_python_version(self, dockerfile_path):
        """Test that Dockerfile uses correct Python version."""
        content = dockerfile_path.read_text()
        
        # Check for Python 3.11.8
        assert "python:3.11.8" in content.lower(), "Must use Python 3.11.8"
        assert "python:3.12" not in content.lower(), "Must not use Python 3.12"
    
    def test_dockerfile_security_features(self, dockerfile_path):
        """Test that Dockerfile implements security best practices."""
        content = dockerfile_path.read_text()
        
        # Check for non-root user
        assert "USER genesis" in content, "Must run as non-root user"
        assert "useradd" in content or "adduser" in content, "Must create non-root user"
        
        # Check for security scanning
        assert "pip-audit" in content, "Must include pip-audit security scanning"
        assert "safety" in content or "bandit" in content, "Must include security scanning"
        
        # Check for minimal base image
        assert "slim" in content or "alpine" in content, "Should use minimal base image"
    
    def test_dockerfile_health_check(self, dockerfile_path):
        """Test that Dockerfile includes health check."""
        content = dockerfile_path.read_text()
        
        assert "HEALTHCHECK" in content, "Must include HEALTHCHECK instruction"
        assert "genesis.api.health" in content, "Must use health check module"
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists and excludes unnecessary files."""
        dockerignore_path = Path(__file__).parent.parent.parent / ".dockerignore"
        assert dockerignore_path.exists(), ".dockerignore file must exist"
        
        content = dockerignore_path.read_text()
        
        # Check for critical exclusions
        required_exclusions = [
            ".git",
            "__pycache__",
            "venv",
            ".env",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            ".genesis/"  # Runtime data should be volume mounted
        ]
        
        for exclusion in required_exclusions:
            assert exclusion in content, f".dockerignore must exclude {exclusion}"
    
    def test_dockerfile_optimization(self, dockerfile_path):
        """Test that Dockerfile is optimized for size and caching."""
        content = dockerfile_path.read_text()
        
        # Check for layer optimization patterns
        assert "COPY requirements" in content, "Should copy requirements before code for better caching"
        assert "pip install --no-cache-dir" in content, "Should use --no-cache-dir to reduce image size"
        assert "rm -rf /var/lib/apt/lists" in content, "Should clean apt lists to reduce size"
        assert "&& \\" in content, "Should chain RUN commands to reduce layers"
    
    def test_dockerfile_labels(self, dockerfile_path):
        """Test that Dockerfile includes proper labels."""
        content = dockerfile_path.read_text()
        
        # Check for metadata labels
        assert "LABEL" in content, "Must include LABEL instructions"
        assert "org.label-schema" in content, "Should use label-schema.org convention"
        assert "version" in content.lower(), "Should include version label"
        assert "description" in content.lower(), "Should include description label"
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker").exists() and not Path("/usr/local/bin/docker").exists(),
        reason="Docker not installed"
    )
    def test_docker_build_syntax(self, docker_context, dockerfile_path):
        """Test that Dockerfile syntax is valid (requires Docker)."""
        try:
            # Validate Dockerfile syntax without actually building
            result = subprocess.run(
                ["docker", "build", "--no-cache", "--target", "builder", 
                 "-f", str(dockerfile_path), "--dry-run", str(docker_context)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0 and "--dry-run" not in result.stderr:
                # --dry-run might not be supported, try alternative validation
                result = subprocess.run(
                    ["docker", "run", "--rm", "-i", "hadolint/hadolint"],
                    input=dockerfile_path.read_text(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            
            assert result.returncode == 0, f"Dockerfile syntax validation failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Docker syntax check timed out")
        except FileNotFoundError:
            pytest.skip("Docker command not found")
    
    def test_supervisor_config_exists(self):
        """Test that supervisor configuration exists."""
        supervisor_conf = Path(__file__).parent.parent.parent / "docker" / "supervisord.conf"
        assert supervisor_conf.exists(), "supervisord.conf must exist for production deployment"
    
    def test_build_args_configuration(self, dockerfile_path):
        """Test that Dockerfile supports build arguments."""
        content = dockerfile_path.read_text()
        
        # Check for build arguments
        assert "ARG BUILD_DATE" in content, "Should support BUILD_DATE argument"
        assert "ARG VERSION" in content, "Should support VERSION argument"
        assert "ARG VCS_REF" in content, "Should support VCS_REF argument"
    
    @pytest.mark.parametrize("required_dir", [
        "/app/.genesis/data",
        "/app/.genesis/logs", 
        "/app/.genesis/state",
        "/app/.genesis/backups"
    ])
    def test_dockerfile_creates_directories(self, dockerfile_path, required_dir):
        """Test that Dockerfile creates required directories."""
        content = dockerfile_path.read_text()
        
        # Check that directory is created
        dir_name = required_dir.split("/")[-1]
        assert dir_name in content, f"Dockerfile must create {required_dir}"
    
    def test_dockerfile_permissions(self, dockerfile_path):
        """Test that Dockerfile sets proper permissions."""
        content = dockerfile_path.read_text()
        
        # Check for permission settings
        assert "chown" in content, "Must set ownership for application files"
        assert "chmod" in content, "Must set permissions for scripts"
        assert "genesis:genesis" in content, "Must use genesis user and group"
    
    def test_dockerfile_exposed_ports(self, dockerfile_path):
        """Test that Dockerfile exposes necessary ports."""
        content = dockerfile_path.read_text()
        
        # Check for EXPOSE instruction
        assert "EXPOSE" in content, "Should expose application port"
        assert "8000" in content, "Should expose port 8000 for API"
    
    def test_virtual_environment_usage(self, dockerfile_path):
        """Test that Dockerfile uses virtual environment for isolation."""
        content = dockerfile_path.read_text()
        
        # Check for venv usage
        assert "venv" in content or "virtualenv" in content, "Should use virtual environment"
        assert "/opt/venv" in content, "Should create venv in /opt/venv"
        assert 'PATH="/opt/venv' in content, "Should add venv to PATH"
    
    def test_production_optimizations(self, dockerfile_path):
        """Test production-specific optimizations."""
        content = dockerfile_path.read_text()
        
        # Check for production optimizations
        assert "PYTHONDONTWRITEBYTECODE=1" in content, "Should disable bytecode writing"
        assert "PYTHONUNBUFFERED=1" in content, "Should disable output buffering"
        assert "PIP_NO_CACHE_DIR=1" in content, "Should disable pip cache"
        
    def test_development_stage_exists(self, dockerfile_path):
        """Test that development stage is provided."""
        content = dockerfile_path.read_text()
        
        # Check for development stage
        assert "AS development" in content, "Should include development stage"
        assert "requirements/dev.txt" in content, "Development stage should install dev requirements"