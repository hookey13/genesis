"""Unit tests for security scanner validator."""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

import pytest

from genesis.validation.security_scanner import SecurityScanner


@pytest.fixture
def security_scanner():
    """Create a security scanner instance."""
    return SecurityScanner()


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for external commands."""
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def temp_code_dir(tmp_path):
    """Create temporary code directory structure."""
    code_dir = tmp_path / "genesis"
    code_dir.mkdir()
    
    # Create sample Python files
    (code_dir / "main.py").write_text("""
import os
from decimal import Decimal

def process_order(amount):
    return Decimal(amount) * Decimal("0.001")
""")
    
    (code_dir / "config.py").write_text("""
import os

DATABASE_URL = os.getenv("DATABASE_URL")
""")
    
    return tmp_path


class TestSecurityScanner:
    """Test security scanner functionality."""
    
    @pytest.mark.asyncio
    async def test_validate_no_vulnerabilities(self, security_scanner, mock_subprocess_run):
        """Test validation when no vulnerabilities are found."""
        # Mock pip-audit - no vulnerabilities
        pip_audit_result = Mock()
        pip_audit_result.returncode = 0
        pip_audit_result.stdout = "[]"
        pip_audit_result.stderr = ""
        
        # Mock bandit - no issues
        bandit_result = Mock()
        bandit_result.returncode = 0
        bandit_result.stdout = json.dumps({
            "metrics": {},
            "results": []
        })
        bandit_result.stderr = ""
        
        mock_subprocess_run.side_effect = [pip_audit_result, bandit_result]
        
        # Mock the hardcoded secrets check to return no secrets
        with patch.object(security_scanner, '_check_hardcoded_secrets') as mock_secrets:
            mock_secrets.return_value = {
                "secrets_found": False,
                "count": 0,
                "locations": []
            }
            
            # Mock API key management to be secure
            with patch.object(security_scanner, '_check_api_key_management') as mock_api:
                mock_api.return_value = {
                    "secure": True,
                    "issues": []
                }
                
                # Mock file permissions to be secure
                with patch.object(security_scanner, '_check_file_permissions') as mock_perms:
                    mock_perms.return_value = {
                        "secure": True,
                        "issues": []
                    }
                    
                    result = await security_scanner.validate()
        
        assert result["passed"] is True
        assert result["details"]["critical_issues"] == 0
        assert result["details"]["dependency_vulnerabilities"] == 0
        assert result["details"]["code_security_issues"] == 0
        assert result["details"]["hardcoded_secrets"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_with_vulnerabilities(self, security_scanner, mock_subprocess_run):
        """Test validation when vulnerabilities are found."""
        # Mock pip-audit - critical vulnerability
        pip_audit_result = Mock()
        pip_audit_result.returncode = 1
        pip_audit_result.stdout = json.dumps([{
            "name": "vulnerable-package",
            "version": "1.0.0",
            "id": "CVE-2024-1234",
            "fix_versions": [{"severity": "CRITICAL"}],
            "description": "Critical vulnerability"
        }])
        pip_audit_result.stderr = ""
        
        # Mock bandit - high severity issue
        bandit_result = Mock()
        bandit_result.returncode = 0
        bandit_result.stdout = json.dumps({
            "metrics": {},
            "results": [{
                "issue_severity": "HIGH",
                "issue_confidence": "HIGH",
                "issue_text": "Hardcoded password",
                "filename": "config.py",
                "line_number": 10,
                "test_id": "B105"
            }]
        })
        bandit_result.stderr = ""
        
        mock_subprocess_run.side_effect = [pip_audit_result, bandit_result]
        
        result = await security_scanner.validate()
        
        assert result["passed"] is False
        assert result["details"]["critical_issues"] > 0
        assert result["details"]["dependency_vulnerabilities"] == 1
        assert result["details"]["code_security_issues"] == 1
    
    @pytest.mark.asyncio
    async def test_pip_audit_not_installed(self, security_scanner):
        """Test handling when pip-audit is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = await security_scanner._run_pip_audit()
            
            assert "pip-audit not installed" in result.get("error", "")
            assert result["total_vulnerabilities"] == 0
    
    @pytest.mark.asyncio
    async def test_bandit_not_installed(self, security_scanner):
        """Test handling when bandit is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = await security_scanner._run_bandit()
            
            assert "bandit not installed" in result.get("error", "")
            assert result["total_issues"] == 0
    
    @pytest.mark.asyncio
    async def test_check_hardcoded_secrets(self, security_scanner):
        """Test detection of hardcoded secrets."""
        # Create a mock Path that returns our test files
        mock_files = []
        
        class MockPath:
            def __init__(self, path_str):
                self.path_str = path_str
                
            def exists(self):
                return True
                
            def rglob(self, pattern):
                if pattern == "*.py":
                    # Return mock file paths
                    return mock_files
                return []
                
        class MockFile:
            def __init__(self, path, content):
                self.path = path
                self.content = content
                
            def read_text(self, encoding='utf-8', errors='ignore'):
                return self.content
                
            def __str__(self):
                return self.path
        
        # Create mock files with secrets
        test_content = """
API_KEY = "sk_live_abc123xyz"
PASSWORD = "hardcoded_password"
BINANCE_KEY = "binance_secret_key"
"""
        mock_files = [MockFile("genesis/bad_config.py", test_content)]
        
        with patch('genesis.validation.security_scanner.Path', MockPath):
            result = await security_scanner._check_hardcoded_secrets()
        
        assert result["secrets_found"] is True
        assert result["count"] >= 2  # At least 2 secrets (API_KEY and PASSWORD patterns)
        assert len(result["locations"]) >= 2
    
    def test_check_api_key_management_secure(self, security_scanner, tmp_path):
        """Test API key management check when secure."""
        # Create .env.example
        (tmp_path / ".env.example").write_text("API_KEY=your_api_key_here")
        
        # Create .gitignore with .env
        (tmp_path / ".gitignore").write_text(".env\n*.pyc\n")
        
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = security_scanner._check_api_key_management()
            
            assert result["secure"] is True
            assert len(result["issues"]) == 0
    
    def test_check_api_key_management_insecure(self, security_scanner, tmp_path):
        """Test API key management check when insecure."""
        # Create .env without .gitignore entry
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=secret")
        gitignore_file = tmp_path / ".gitignore"
        gitignore_file.write_text("*.pyc\n")
        
        # Patch the Path constructor to return our temp paths
        def mock_path(path_str):
            if path_str == ".env":
                return env_file
            elif path_str == ".gitignore":
                return gitignore_file
            elif path_str == ".env.example":
                return tmp_path / ".env.example"
            elif path_str == "config":
                return tmp_path / "config"
            else:
                return Path(path_str)
        
        with patch("genesis.validation.security_scanner.Path", side_effect=mock_path):
            result = security_scanner._check_api_key_management()
            
            assert result["secure"] is False
            assert ".env file not in .gitignore" in result["issues"]
    
    def test_check_file_permissions(self, security_scanner, tmp_path):
        """Test file permissions check."""
        # Create sensitive file
        sensitive_file = tmp_path / ".env"
        sensitive_file.write_text("SECRET=value")
        
        with patch("pathlib.Path", side_effect=lambda x: tmp_path / x if x == ".env" else Path(x)):
            result = security_scanner._check_file_permissions()
            
            # Result depends on OS permissions
            assert "secure" in result
            assert "issues" in result
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, security_scanner):
        """Test generation of security recommendations."""
        pip_audit_results = {
            "critical_count": 2,
            "high_count": 3,
            "moderate_count": 1,
            "low_count": 0
        }
        
        bandit_results = {
            "high_severity_count": 1,
            "medium_severity_count": 10,
            "low_severity_count": 5
        }
        
        secrets_results = {
            "secrets_found": True,
            "count": 3
        }
        
        api_key_results = {
            "secure": False,
            "issues": ["Missing .env.example"]
        }
        
        permissions_results = {
            "secure": False,
            "issues": [".env is world-readable"]
        }
        
        recommendations = security_scanner._generate_recommendations(
            pip_audit_results,
            bandit_results,
            secrets_results,
            api_key_results,
            permissions_results
        )
        
        assert len(recommendations) > 0
        assert any("critical dependency vulnerabilities" in r for r in recommendations)
        assert any("hardcoded secrets" in r for r in recommendations)
        assert any("Fix: Missing .env.example" in r for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_validate_with_timeout(self, security_scanner, mock_subprocess_run):
        """Test handling of command timeouts."""
        # Mock timeout exception
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("pip-audit", 120)
        
        result = await security_scanner._run_pip_audit()
        
        assert "timed out" in result.get("error", "")
        assert result["total_vulnerabilities"] == 0
    
    @pytest.mark.asyncio
    async def test_parse_pip_audit_text_fallback(self, security_scanner):
        """Test fallback text parsing for pip-audit."""
        text_output = """
Found vulnerability CVE-2024-1234 in package xyz
Found vulnerability CVE-2024-5678 in package abc
"""
        
        result = security_scanner._parse_pip_audit_text(text_output)
        
        assert result["total_vulnerabilities"] == 2
        assert len(result["vulnerabilities"]) == 2
    
    @pytest.mark.asyncio
    async def test_parse_bandit_text_fallback(self, security_scanner):
        """Test fallback text parsing for bandit."""
        text_output = """
Issue: Hardcoded password found
Severity: High
Issue: Weak cryptographic key
Severity: Medium
"""
        
        result = security_scanner._parse_bandit_text(text_output)
        
        assert result["total_issues"] == 2
        assert result["high_severity_count"] == 1
        assert result["medium_severity_count"] == 1


@pytest.mark.asyncio
async def test_security_scanner_integration(security_scanner, temp_code_dir):
    """Integration test for full security validation."""
    with patch("genesis.validation.security_scanner.Path") as mock_path:
        # Mock code paths to use temp directory
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.rglob.return_value = [
            temp_code_dir / "genesis" / "main.py",
            temp_code_dir / "genesis" / "config.py"
        ]
        
        # Mock external commands to succeed
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "[]"
            
            result = await security_scanner.validate()
            
            assert "passed" in result
            assert "details" in result
            assert "vulnerabilities" in result
            assert "recommendations" in result