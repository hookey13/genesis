"""Integration tests for container security scanning."""

import subprocess
import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import yaml


class TestContainerSecurity:
    """Test container security scanning and compliance."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_container_security_workflow_exists(self, project_root):
        """Verify container security workflow exists."""
        workflow_file = project_root / ".github" / "workflows" / "container-security.yml"
        assert workflow_file.exists(), "container-security.yml workflow must exist"
    
    def test_container_security_workflow_configuration(self, project_root):
        """Verify container security workflow is properly configured."""
        workflow_file = project_root / ".github" / "workflows" / "container-security.yml"
        
        if workflow_file.exists():
            content = workflow_file.read_text()
            
            # Check for security scanners
            assert "trivy" in content.lower(), "Must use Trivy scanner"
            assert "snyk" in content.lower() or "grype" in content.lower(), \
                "Should use additional scanners"
            assert "hadolint" in content.lower(), "Must lint Dockerfile"
            
            # Check for SBOM generation
            assert "sbom" in content.lower(), "Should generate SBOM"
            
            # Check for secret scanning
            assert "trufflehog" in content.lower() or "secrets" in content.lower(), \
                "Should scan for secrets"
    
    def test_dockerfile_security_best_practices(self, project_root):
        """Verify Dockerfile follows security best practices."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Security best practices
            assert "USER" in content and "USER root" not in content[content.find("USER"):], \
                "Must not switch back to root after setting USER"
            
            # Check for apt-get best practices
            if "apt-get update" in content:
                assert "apt-get update && apt-get install" in content.replace('\n', ' '), \
                    "apt-get update and install should be in same RUN to avoid cache issues"
                assert "rm -rf /var/lib/apt/lists/*" in content, \
                    "Should clean apt cache after installation"
            
            # Check for COPY vs ADD
            add_count = content.count("ADD")
            copy_count = content.count("COPY")
            assert copy_count > add_count, "Prefer COPY over ADD for local files"
            
            # Check for explicit versions
            if "FROM" in content:
                from_lines = [l for l in content.split('\n') if l.strip().startswith("FROM")]
                for line in from_lines:
                    assert ":" in line and "latest" not in line.lower(), \
                        f"Must use specific image tags, not latest: {line}"
    
    def test_no_hardcoded_secrets(self, project_root):
        """Verify no hardcoded secrets in Dockerfile or compose files."""
        files_to_check = [
            project_root / "Dockerfile",
            project_root / "docker-compose.yml",
        ]
        
        secret_patterns = [
            "password=",
            "secret=",
            "api_key=",
            "token=",
            "private_key",
            "BEGIN RSA",
            "BEGIN PRIVATE",
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                content = file_path.read_text().lower()
                for pattern in secret_patterns:
                    # Allow environment variable references
                    if pattern.lower() in content:
                        # Check if it's an environment variable reference
                        lines_with_pattern = [
                            l for l in content.split('\n') 
                            if pattern.lower() in l.lower()
                        ]
                        for line in lines_with_pattern:
                            assert "${" in line or "$(" in line or "ENV" in line or "ARG" in line, \
                                f"Potential hardcoded secret in {file_path.name}: {line[:50]}"
    
    def test_container_user_configuration(self, project_root):
        """Verify container runs with proper user configuration."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for user creation and configuration
            assert "useradd" in content or "adduser" in content, \
                "Must create a non-root user"
            assert "USER" in content, "Must switch to non-root user"
            
            # Check for proper permissions
            assert "chown" in content, "Should set proper file ownership"
            assert "chmod" not in content or "777" not in content, \
                "Should not use overly permissive file permissions"
    
    def test_minimal_base_image(self, project_root):
        """Verify use of minimal base images."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for minimal base images
            acceptable_bases = ["slim", "alpine", "distroless", "scratch"]
            from_lines = [l for l in content.split('\n') if l.strip().startswith("FROM")]
            
            for line in from_lines:
                has_minimal = any(base in line.lower() for base in acceptable_bases)
                assert has_minimal or "as builder" in line.lower(), \
                    f"Should use minimal base image: {line}"
    
    @pytest.mark.skipif(
        not shutil.which("trivy"),
        reason="Trivy not installed"
    )
    def test_trivy_scan_local(self, project_root):
        """Run Trivy scan locally if available."""
        dockerfile = project_root / "Dockerfile"
        
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")
        
        # Build test image
        build_result = subprocess.run(
            ["docker", "build", "--target", "production", "-t", "genesis:security-test", "."],
            cwd=project_root,
            capture_output=True,
            timeout=300
        )
        
        if build_result.returncode == 0:
            # Run Trivy scan
            scan_result = subprocess.run(
                ["trivy", "image", "--severity", "CRITICAL,HIGH", "--format", "json", "genesis:security-test"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if scan_result.stdout:
                results = json.loads(scan_result.stdout)
                
                # Check for critical vulnerabilities
                critical_vulns = []
                high_vulns = []
                
                if "Results" in results:
                    for result in results["Results"]:
                        if "Vulnerabilities" in result:
                            for vuln in result["Vulnerabilities"]:
                                if vuln.get("Severity") == "CRITICAL":
                                    critical_vulns.append(vuln)
                                elif vuln.get("Severity") == "HIGH":
                                    high_vulns.append(vuln)
                
                # Assert no critical vulnerabilities
                assert len(critical_vulns) == 0, \
                    f"Found {len(critical_vulns)} CRITICAL vulnerabilities"
                
                # Allow maximum 3 high vulnerabilities
                assert len(high_vulns) <= 3, \
                    f"Found {len(high_vulns)} HIGH vulnerabilities (max 3 allowed)"
            
            # Clean up
            subprocess.run(["docker", "rmi", "genesis:security-test"], capture_output=True)
    
    @pytest.mark.skipif(
        not shutil.which("hadolint"),
        reason="Hadolint not installed"
    )
    def test_hadolint_dockerfile_linting(self, project_root):
        """Run Hadolint on Dockerfile if available."""
        dockerfile = project_root / "Dockerfile"
        
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")
        
        # Run Hadolint
        result = subprocess.run(
            ["hadolint", "--ignore", "DL3008", "--ignore", "DL3009", str(dockerfile)],
            capture_output=True,
            text=True
        )
        
        # Check for errors (exit code 0 = no errors, 1 = errors found)
        if result.returncode != 0:
            # Parse output for specific issues
            issues = result.stdout.split('\n')
            critical_issues = [i for i in issues if "error" in i.lower()]
            
            assert len(critical_issues) == 0, \
                f"Hadolint found critical issues: {critical_issues}"
    
    def test_compose_security_configuration(self, project_root):
        """Verify docker-compose security configuration."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for security configurations
            assert "cap_drop:" in content or "CAP_DROP" in content, \
                "Should drop unnecessary capabilities"
            
            assert "read_only:" in content or "privileged: false" in content, \
                "Should use security constraints"
            
            # Check for proper secret handling
            if "secrets:" in content:
                assert "external: true" in content, \
                    "Secrets should be external, not embedded"
    
    def test_health_check_configuration(self, project_root):
        """Verify health check is properly configured."""
        dockerfile = project_root / "Dockerfile"
        compose_file = project_root / "docker-compose.yml"
        
        health_check_found = False
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            if "HEALTHCHECK" in content:
                health_check_found = True
                # Verify health check doesn't use curl with --insecure
                assert "--insecure" not in content, \
                    "Health check should not use insecure options"
        
        if compose_file.exists():
            content = compose_file.read_text()
            if "healthcheck:" in content:
                health_check_found = True
        
        assert health_check_found, "Health check must be configured"
    
    def test_network_security(self, project_root):
        """Verify network security configuration."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for network configuration
            if "networks:" in content:
                # Should use custom networks, not default
                assert "driver: bridge" in content, \
                    "Should use custom bridge network"
                
                # Check for network isolation
                if "external:" in content:
                    assert "external: false" in content or "internal: true" in content, \
                        "Should properly isolate networks"
    
    def test_resource_limits(self, project_root):
        """Verify resource limits are set."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check for resource limits
            assert "limits:" in content or "mem_limit:" in content, \
                "Should set memory limits"
            
            assert "cpus:" in content or "cpu_shares:" in content, \
                "Should set CPU limits"
    
    def test_logging_security(self, project_root):
        """Verify logging doesn't expose sensitive data."""
        compose_file = project_root / "docker-compose.yml"
        
        if compose_file.exists():
            content = compose_file.read_text()
            
            # Check logging configuration
            if "logging:" in content:
                assert "max-size:" in content, \
                    "Should limit log size to prevent disk exhaustion"
                assert "max-file:" in content, \
                    "Should limit number of log files"
    
    def test_sbom_generation_capability(self, project_root):
        """Verify SBOM can be generated for the container."""
        workflow_file = project_root / ".github" / "workflows" / "container-security.yml"
        
        if workflow_file.exists():
            content = workflow_file.read_text()
            
            # Check for SBOM generation
            assert "sbom" in content.lower(), \
                "Should generate Software Bill of Materials"
            assert "spdx" in content.lower() or "cyclonedx" in content.lower(), \
                "Should use standard SBOM format"
    
    def test_image_signing_preparation(self, project_root):
        """Verify preparation for image signing."""
        workflow_file = project_root / ".github" / "workflows" / "container-security.yml"
        
        if workflow_file.exists():
            content = workflow_file.read_text()
            
            # Check for registry and metadata
            assert "registry" in content.lower(), \
                "Should configure container registry"
            assert "metadata" in content.lower() or "labels" in content.lower(), \
                "Should add metadata/labels to images"