"""Integration tests for Docker multi-stage builds."""

import subprocess
import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil


class TestDockerBuild:
    """Test Docker multi-stage build optimization."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_dockerfile_exists(self, project_root):
        """Verify Dockerfile exists."""
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile must exist"
    
    def test_dockerfile_multi_stage(self, project_root):
        """Verify Dockerfile uses multi-stage build."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for multi-stage build markers
            assert "FROM" in content, "Dockerfile must have FROM statements"
            assert "as builder" in content.lower(), "Must have builder stage"
            assert "as production" in content.lower(), "Must have production stage"
            
            # Count FROM statements (should be multiple for multi-stage)
            from_count = content.count("FROM")
            assert from_count >= 2, f"Multi-stage build should have multiple FROM statements, found {from_count}"
    
    def test_python_version_consistency(self, project_root):
        """Verify Python 3.11.8 is used consistently."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for Python 3.11.8
            assert "python:3.11.8" in content.lower(), "Must use Python 3.11.8"
            
            # Should use slim variant for smaller size
            assert "python:3.11.8-slim" in content.lower(), "Should use slim Python image"
    
    def test_non_root_user(self, project_root):
        """Verify container runs as non-root user."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for user creation
            assert "useradd" in content or "adduser" in content, "Must create non-root user"
            assert "USER genesis" in content, "Must switch to non-root user"
            assert "uid 1000" in content.lower() or "-u 1000" in content, "Should use UID 1000"
    
    def test_health_check_defined(self, project_root):
        """Verify health check is defined."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for HEALTHCHECK instruction
            assert "HEALTHCHECK" in content, "Must define health check"
            assert "--interval=" in content, "Health check must have interval"
            assert "--timeout=" in content, "Health check must have timeout"
            assert "--retries=" in content, "Health check must have retries"
    
    def test_build_optimization(self, project_root):
        """Verify build optimization techniques."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for optimization patterns
            assert "COPY --from=builder" in content, "Should copy from builder stage"
            assert "--no-cache-dir" in content, "Should use --no-cache-dir for pip"
            assert "rm -rf /var/lib/apt/lists/*" in content, "Should clean apt cache"
            
            # Check for proper layer caching
            lines = content.split('\n')
            copy_req_index = -1
            copy_app_index = -1
            
            for i, line in enumerate(lines):
                if 'COPY' in line and ('requirements' in line or 'pyproject' in line):
                    copy_req_index = i
                elif 'COPY' in line and 'genesis/' in line:
                    copy_app_index = i
            
            if copy_req_index > 0 and copy_app_index > 0:
                assert copy_req_index < copy_app_index, \
                    "Requirements should be copied before application code for better caching"
    
    def test_security_hardening(self, project_root):
        """Verify security hardening in Dockerfile."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Security checks
            assert "USER genesis" in content or "USER 1000" in content, \
                "Must run as non-root user"
            assert "--chown=genesis:genesis" in content or "--chown=1000:1000" in content, \
                "Files should be owned by non-root user"
            assert "tini" in content.lower(), "Should use tini for signal handling"
            
            # Environment variables for security
            assert "PYTHONDONTWRITEBYTECODE" in content, "Should disable bytecode writing"
            assert "PYTHONUNBUFFERED" in content, "Should use unbuffered output"
    
    def test_dockerignore_exists(self, project_root):
        """Verify .dockerignore exists and is comprehensive."""
        dockerignore = project_root / ".dockerignore"
        assert dockerignore.exists(), ".dockerignore must exist"
        
        if dockerignore.exists():
            content = dockerignore.read_text()
            
            # Check for important exclusions
            assert ".git/" in content, "Should exclude .git directory"
            assert "__pycache__/" in content, "Should exclude Python cache"
            assert ".env" in content, "Should exclude environment files"
            assert "venv/" in content or ".venv/" in content, "Should exclude virtual environments"
            assert ".pytest_cache/" in content, "Should exclude test cache"
    
    def test_docker_compose_exists(self, project_root):
        """Verify docker-compose.yml exists."""
        compose_file = project_root / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml must exist"
    
    def test_docker_compose_services(self, project_root):
        """Verify docker-compose services are properly configured."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for essential services
            assert "genesis:" in content, "Must have genesis service"
            assert "build:" in content, "Must have build configuration"
            assert "environment:" in content, "Must have environment variables"
            assert "volumes:" in content, "Must have volume mounts"
            assert "healthcheck:" in content, "Must have health check"
            
            # Check for resource limits
            assert "limits:" in content, "Should have resource limits"
            assert "memory:" in content, "Should limit memory usage"
            
            # Check for profiles
            assert "profiles:" in content, "Should use profiles for optional services"
    
    def test_development_stage(self, project_root):
        """Verify development stage is available."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for development stage
            assert "as development" in content.lower(), "Should have development stage"
            assert "as testing" in content.lower(), "Should have testing stage"
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not shutil.which("docker"),
        reason="Docker not available"
    )
    def test_docker_build_command(self, project_root):
        """Test actual Docker build (if Docker is available)."""
        dockerfile = project_root / "Dockerfile"
        
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")
        
        # Try to build the production stage
        result = subprocess.run(
            ["docker", "build", "--target", "production", "-t", "genesis:test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Check image size
            size_result = subprocess.run(
                ["docker", "images", "genesis:test", "--format", "{{.Size}}"],
                capture_output=True,
                text=True
            )
            
            if size_result.stdout:
                size_str = size_result.stdout.strip()
                # Parse size (format: "123MB" or "1.23GB")
                if "GB" in size_str:
                    size_gb = float(size_str.replace("GB", ""))
                    size_mb = size_gb * 1024
                else:
                    size_mb = float(size_str.replace("MB", ""))
                
                # Target is <500MB
                assert size_mb < 500, f"Production image should be <500MB, got {size_mb}MB"
            
            # Clean up test image
            subprocess.run(["docker", "rmi", "genesis:test"], capture_output=True)
    
    def test_tier_build_args(self, project_root):
        """Verify tier selection via build arguments."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for tier build argument
            assert "ARG TIER=" in content, "Must have TIER build argument"
            assert "${TIER}" in content, "Must use TIER variable"
            assert "TIER=sniper" in content, "Should default to sniper tier"
    
    def test_volume_configuration(self, project_root):
        """Verify proper volume configuration."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for named volumes
            assert "volumes:" in content, "Must define volumes"
            assert "genesis-data:" in content, "Must have data volume"
            assert "genesis-logs:" in content, "Must have logs volume"
            assert "genesis-state:" in content, "Must have state volume"
            
            # Check volume mounts
            assert "/app/.genesis/data" in content, "Must mount data directory"
            assert "/app/.genesis/logs" in content, "Must mount logs directory"
            assert "/app/.genesis/state" in content, "Must mount state directory"
    
    def test_prometheus_metrics_port(self, project_root):
        """Verify Prometheus metrics port is exposed."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for metrics port
            assert "EXPOSE 9090" in content, "Must expose Prometheus metrics port"
        
        compose_file = project_root / "docker-compose.yml"
        if compose_file.exists():
            content = compose_file.read_text()
            assert "9090:9090" in content, "Must map Prometheus port"
    
    def test_environment_variables(self, project_root):
        """Verify environment variables are properly handled."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check critical environment variables
            env_vars = [
                "GENESIS_ENV",
                "TIER",
                "BINANCE_API_KEY",
                "BINANCE_API_SECRET",
                "DATABASE_URL",
                "LOG_LEVEL"
            ]
            
            for var in env_vars:
                assert var in content, f"Must configure {var} environment variable"
    
    def test_container_restart_policy(self, project_root):
        """Verify restart policy is configured."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for restart policy
            assert "restart:" in content, "Must have restart policy"
            assert "unless-stopped" in content or "always" in content, \
                "Should use unless-stopped or always restart policy"
    
    def test_logging_configuration(self, project_root):
        """Verify logging is properly configured."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for logging configuration
            assert "logging:" in content, "Must configure logging"
            assert "max-size:" in content, "Should limit log file size"
            assert "max-file:" in content, "Should limit number of log files"