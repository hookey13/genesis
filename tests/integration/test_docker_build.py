"""Integration tests for Docker build and deployment configurations."""

import subprocess
import pytest
import os
from pathlib import Path


class TestDockerBuild:
    """Test Docker build process and configurations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.docker_dir = self.project_root / "docker"
        
    def test_dockerfile_exists(self):
        """Verify Dockerfile exists and is valid."""
        dockerfile = self.docker_dir / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"
        
        # Verify essential Docker instructions
        content = dockerfile.read_text()
        assert "FROM python:3.11.8" in content, "Incorrect Python version"
        assert "WORKDIR" in content, "No WORKDIR specified"
        assert "COPY requirements" in content, "Requirements not copied"
        assert "RUN pip install" in content, "Dependencies not installed"
        
    def test_docker_compose_files_exist(self):
        """Verify docker-compose files exist."""
        dev_compose = self.docker_dir / "docker-compose.yml"
        prod_compose = self.docker_dir / "docker-compose.prod.yml"
        
        assert dev_compose.exists(), "Development docker-compose not found"
        assert prod_compose.exists(), "Production docker-compose not found"
        
    def test_docker_build_development(self):
        """Test Docker image build for development."""
        result = subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile", "-t", "genesis-test:dev", "."],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        
        # Verify image was created
        result = subprocess.run(
            ["docker", "images", "genesis-test:dev", "--format", "{{.Repository}}"],
            capture_output=True,
            text=True
        )
        assert "genesis-test" in result.stdout, "Docker image not created"
        
    def test_docker_compose_validation_dev(self):
        """Validate development docker-compose configuration."""
        result = subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.yml", "config"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Docker-compose config invalid: {result.stderr}"
        assert "genesis" in result.stdout, "Service 'genesis' not defined"
        
    def test_docker_compose_validation_prod(self):
        """Validate production docker-compose configuration."""
        result = subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.prod.yml", "config"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Docker-compose config invalid: {result.stderr}"
        assert "restart:" in result.stdout, "Restart policy not configured"
        
    def test_supervisor_config_exists(self):
        """Verify supervisor configuration for production."""
        supervisor_conf = self.docker_dir / "supervisord.conf"
        assert supervisor_conf.exists(), "Supervisor config not found"
        
        content = supervisor_conf.read_text()
        assert "[supervisord]" in content, "Invalid supervisor config"
        assert "nodaemon=true" in content, "Supervisor not in foreground mode"
        
    def test_docker_volume_mounts(self):
        """Verify volume mounts are properly configured."""
        result = subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.yml", "config"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        assert "./genesis:/app/genesis" in result.stdout or "volumes:" in result.stdout, \
            "Volume mounts not configured"
        
    def test_docker_environment_variables(self):
        """Test environment variable handling in Docker."""
        result = subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.yml", "config"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        assert "environment:" in result.stdout or "env_file:" in result.stdout, \
            "Environment variables not configured"
        
    @pytest.mark.slow
    def test_docker_container_startup(self):
        """Test container startup and health check."""
        # Build image first
        subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile", "-t", "genesis-test:integration", "."],
            cwd=self.project_root,
            capture_output=True
        )
        
        # Run container with test command
        result = subprocess.run(
            ["docker", "run", "--rm", "genesis-test:integration", "python", "-c", 
             "import genesis; print('Genesis package imported successfully')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0, f"Container startup failed: {result.stderr}"
        assert "Genesis package imported successfully" in result.stdout
        
    def teardown_method(self):
        """Clean up test Docker images."""
        subprocess.run(
            ["docker", "rmi", "genesis-test:dev", "genesis-test:integration"],
            capture_output=True
        )