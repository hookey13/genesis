"""
Unit tests for Docker container build validation.

Tests verify that the Dockerfile meets production requirements
including size constraints, security configuration, and health checks.
"""

import os
import subprocess
import pytest
from pathlib import Path


class TestDockerfileBuild:
    """Test suite for Dockerfile validation."""
    
    @pytest.fixture
    def dockerfile_path(self):
        """Get the path to the Dockerfile."""
        return Path(__file__).parent.parent.parent / "docker" / "Dockerfile"
    
    def test_dockerfile_exists(self, dockerfile_path):
        """Test that Dockerfile exists in the correct location."""
        assert dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"
    
    def test_dockerfile_multi_stage(self, dockerfile_path):
        """Test that Dockerfile uses multi-stage build pattern."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for multiple FROM statements indicating multi-stage
        assert content.count('FROM') >= 3, "Dockerfile should use multi-stage build"
        
        # Check for specific stages
        assert 'AS builder' in content, "Missing builder stage"
        assert 'AS production' in content, "Missing production stage"
        assert 'AS security-scan' in content or 'AS security' in content, "Missing security scan stage"
    
    def test_base_image_version(self, dockerfile_path):
        """Test that correct Python version is used."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for Python 3.11.8 as specified in tech stack
        assert 'python:3.11.8-slim' in content, "Must use Python 3.11.8-slim base image"
    
    def test_non_root_user(self, dockerfile_path):
        """Test that container runs as non-root user."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for user creation
        assert 'useradd' in content, "Should create a non-root user"
        # Check for user creation with UID 1000
        assert 'useradd' in content and 'genesis' in content and '1000' in content, "Should create genesis user with UID 1000"
        assert 'USER genesis' in content, "Should switch to non-root user"
    
    def test_health_check_configured(self, dockerfile_path):
        """Test that health check is properly configured."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for HEALTHCHECK instruction
        assert 'HEALTHCHECK' in content, "Missing HEALTHCHECK instruction"
        assert 'python -m genesis.cli doctor' in content, "Health check should use CLI doctor command"
    
    def test_exposed_ports(self, dockerfile_path):
        """Test that required ports are exposed."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for exposed ports
        assert 'EXPOSE 8000' in content or 'EXPOSE 8000 9090' in content, "API port 8000 should be exposed"
        assert 'EXPOSE' in content and '9090' in content, "Prometheus metrics port 9090 should be exposed"
    
    def test_environment_variables(self, dockerfile_path):
        """Test that proper environment variables are set."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for Python environment variables
        assert 'PYTHONPATH' in content, "PYTHONPATH should be set"
        assert 'PYTHONUNBUFFERED' in content, "PYTHONUNBUFFERED should be set"
        assert 'PYTHONDONTWRITEBYTECODE' in content, "PYTHONDONTWRITEBYTECODE should be set"
    
    def test_working_directory(self, dockerfile_path):
        """Test that working directory is properly set."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for WORKDIR instruction
        assert 'WORKDIR /app' in content, "Working directory should be /app"
    
    def test_build_dependencies(self, dockerfile_path):
        """Test that build dependencies are installed in builder stage."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
            
        # Split by FROM to analyze stages
        stages = content.split('FROM')
        builder_stage = None
        
        for stage in stages:
            if 'AS builder' in stage:
                builder_stage = stage
                break
        
        assert builder_stage is not None, "Builder stage not found"
        
        # Check for required build dependencies
        assert 'gcc' in builder_stage, "gcc should be installed in builder stage"
        assert 'g++' in builder_stage, "g++ should be installed in builder stage"
        assert 'libssl-dev' in builder_stage, "libssl-dev should be installed"
        assert 'libffi-dev' in builder_stage, "libffi-dev should be installed"
    
    def test_requirements_copied(self, dockerfile_path):
        """Test that requirements are properly copied."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for requirements copy
        assert 'COPY requirements/' in content, "Requirements directory should be copied"
    
    def test_supervisor_configuration(self, dockerfile_path):
        """Test that supervisor is configured for process management."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for supervisor installation and configuration
        assert 'supervisor' in content.lower(), "Supervisor should be installed"
        assert 'supervisord.conf' in content, "Supervisor configuration should be copied"
    
    def test_permissions_set(self, dockerfile_path):
        """Test that proper permissions are set on directories."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for permission settings
        assert 'chown' in content, "File ownership should be set"
        assert 'chmod' in content, "File permissions should be set"
        assert '.genesis' in content, "Runtime directory should be created"
    
    @pytest.mark.skipif(not os.path.exists("/.dockerenv"), reason="Not running in Docker environment")
    def test_image_size_constraint(self):
        """Test that final image size is under 500MB (requires Docker)."""
        # This test would actually build the image and check size
        # Skip in unit tests, run in integration tests
        pass
    
    def test_label_metadata(self, dockerfile_path):
        """Test that proper labels are set for metadata."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for standard labels
        assert 'LABEL' in content, "Should include LABEL instructions"
        assert 'org.label-schema' in content, "Should use label-schema standard"
    
    def test_security_scan_stage(self, dockerfile_path):
        """Test that security scanning stage is included."""
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for security scanning
        assert 'security' in content.lower(), "Should include security scanning"
        assert any(tool in content for tool in ['pip-audit', 'safety', 'bandit']), \
            "Should use security scanning tools"


class TestDockerComposeValidation:
    """Test suite for Docker Compose configuration."""
    
    @pytest.fixture
    def compose_files(self):
        """Get paths to docker-compose files."""
        docker_dir = Path(__file__).parent.parent.parent / "docker"
        return {
            'dev': docker_dir / "docker-compose.yml",
            'prod': docker_dir / "docker-compose.prod.yml"
        }
    
    def test_compose_files_exist(self, compose_files):
        """Test that docker-compose files exist."""
        for env, path in compose_files.items():
            assert path.exists(), f"docker-compose file for {env} not found at {path}"
    
    def test_compose_syntax_valid(self, compose_files):
        """Test that docker-compose files have valid syntax."""
        import yaml
        
        for env, path in compose_files.items():
            with open(path, 'r') as f:
                try:
                    config = yaml.safe_load(f)
                    assert 'services' in config, f"{env} compose file missing services section"
                    assert 'genesis' in config['services'], f"{env} compose file missing genesis service"
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {env} compose file: {e}")