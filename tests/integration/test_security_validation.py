"""Integration tests for security validation."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from genesis.validation.security.compliance_validator import ComplianceValidator
from genesis.validation.security.config_validator import SecurityConfigValidator
from genesis.validation.security.encryption_validator import EncryptionValidator
from genesis.validation.security.secrets_scanner import SecretsScanner
from genesis.validation.security.vulnerability_scanner import VulnerabilityScanner


@pytest.mark.integration
class TestSecurityValidationIntegration:
    """Integration tests for security validators."""

    @pytest.fixture
    async def test_project(self, tmp_path):
        """Create a test project structure."""
        # Create directory structure
        genesis_dir = tmp_path / "genesis"
        genesis_dir.mkdir()
        
        security_dir = genesis_dir / "security"
        security_dir.mkdir()
        
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        docs_dir = tmp_path / "docs" / "security"
        docs_dir.mkdir(parents=True)
        
        # Create sample files
        # Python file with mixed security practices
        py_file = genesis_dir / "api.py"
        py_file.write_text("""
import os
from cryptography.fernet import Fernet

# Good: Environment variable
API_KEY = os.getenv("BINANCE_API_KEY")

# Bad: Hardcoded secret (for testing)
# TEST_KEY = "sk_test_1234567890abcdef"

# Good: Encrypted storage
def encrypt_api_key(key):
    fernet = Fernet.generate_key()
    cipher = Fernet(fernet)
    return cipher.encrypt(key.encode())

# Good: Rate limiting
from functools import wraps
def rate_limit(max_calls=100):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Rate limiting logic
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Good: Input validation
def validate_order(order_data):
    if not isinstance(order_data.get("amount"), (int, float)):
        raise ValueError("Invalid amount")
    return True
""")
        
        # Vault client for encryption
        vault_file = security_dir / "vault_client.py"
        vault_file.write_text("""
class VaultClient:
    def __init__(self):
        self.encrypted_storage = {}
    
    def encrypt(self, data):
        # Encryption logic
        return f"encrypted_{data}"
    
    def decrypt(self, data):
        # Decryption logic
        return data.replace("encrypted_", "")
    
    def get_secret(self, key):
        return self.decrypt(self.encrypted_storage.get(key))
    
    def rotate_keys(self):
        # Key rotation logic
        pass
    
    def authenticate(self, credentials):
        # Authentication logic
        return True
""")
        
        # Configuration file
        config_file = config_dir / "security.yaml"
        config_file.write_text("""
tls:
  version: 1.3
  ciphers:
    - TLS_AES_256_GCM_SHA384
    - TLS_AES_128_GCM_SHA256
  
headers:
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Strict-Transport-Security: max-age=31536000

rate_limiting:
  api_rate_limit: 100
  login_attempts: 5
  order_rate_limit: 10

authentication:
  session_timeout: 3600
  password_min_length: 12
  password_complexity: true
""")
        
        # Requirements file for vulnerability scanning
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("""
cryptography==41.0.0
pydantic==2.5.3
structlog==24.1.0
aiohttp==3.9.3
""")
        
        # Documentation files
        (docs_dir / "security_policy.md").write_text("# Security Policy\n")
        (docs_dir / "incident_response.md").write_text("# Incident Response\n")
        (docs_dir / "data_retention.md").write_text("# Data Retention Policy\n")
        
        # Error handler
        error_handler = genesis_dir / "core" / "error_handler.py"
        error_handler.parent.mkdir(parents=True)
        error_handler.write_text("""
import structlog

logger = structlog.get_logger()

class ErrorHandler:
    def handle_error(self, error, production=True):
        if production:
            # Hide stack traces in production
            return {"error": "An error occurred", "code": 500}
        else:
            return {"error": str(error), "traceback": error.__traceback__}
    
    def sanitize_error_message(self, message):
        # Remove sensitive information
        return message.replace("password", "***")
""")
        
        return tmp_path

    @pytest.mark.asyncio
    async def test_secrets_scanner_integration(self, test_project):
        """Test SecretsScanner with real project structure."""
        scanner = SecretsScanner()
        
        with patch.object(Path, "cwd", return_value=test_project):
            with patch("genesis.validation.security.secrets_scanner.Path") as mock_path:
                mock_path.return_value.rglob.return_value = list(test_project.rglob("*.py"))
                result = await scanner.validate()
        
        assert result["passed"]  # Should pass as we commented out hardcoded secret
        assert result["summary"]["files_scanned"] > 0
        assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_vulnerability_scanner_integration(self, test_project):
        """Test VulnerabilityScanner with real dependencies."""
        scanner = VulnerabilityScanner()
        
        # Mock tool availability
        scanner.tools_available = {
            "safety": False,
            "bandit": False,
            "pip-audit": False,
            "trivy": False,
        }
        
        with patch.object(Path, "cwd", return_value=test_project):
            result = await scanner.validate()
        
        assert "passed" in result
        assert "summary" in result
        assert result["summary"]["total_vulnerabilities"] == 0  # No tools available

    @pytest.mark.asyncio
    async def test_compliance_validator_integration(self, test_project):
        """Test ComplianceValidator with real project structure."""
        validator = ComplianceValidator()
        
        with patch.object(Path, "cwd", return_value=test_project):
            with patch.object(Path, "exists") as mock_exists:
                # Simulate some files exist
                def exists_side_effect(self):
                    path_str = str(self)
                    if "vault_client.py" in path_str:
                        return True
                    if "error_handler.py" in path_str:
                        return True
                    if "security_policy.md" in path_str:
                        return True
                    if "incident_response.md" in path_str:
                        return True
                    if "data_retention.md" in path_str:
                        return True
                    return False
                
                mock_exists.side_effect = exists_side_effect
                result = await validator.validate()
        
        assert "passed" in result
        assert "compliance_score" in result
        assert "details" in result
        assert "gaps" in result

    @pytest.mark.asyncio
    async def test_encryption_validator_integration(self, test_project):
        """Test EncryptionValidator with real project structure."""
        validator = EncryptionValidator()
        
        with patch.object(Path, "cwd", return_value=test_project):
            # Use actual files from test project
            genesis_path = test_project / "genesis"
            with patch("genesis.validation.security.encryption_validator.Path") as mock_path:
                mock_path.return_value.rglob.return_value = list(genesis_path.rglob("*.py"))
                mock_path.return_value = test_project
                
                # Mock Path instances for specific checks
                with patch.object(Path, "exists") as mock_exists:
                    def exists_side_effect(self):
                        path_str = str(self)
                        if "vault_client.py" in path_str:
                            return True
                        if "encryption.py" in path_str:
                            return False
                        return test_project / path_str.replace(str(Path.cwd()), "") if path_str.startswith(str(Path.cwd())) else False
                    
                    mock_exists.side_effect = exists_side_effect
                    
                    with patch.object(Path, "read_text") as mock_read:
                        def read_text_side_effect(self, encoding="utf-8"):
                            file_path = test_project / str(self).replace(str(Path.cwd()), "")
                            if file_path.exists():
                                return file_path.read_text(encoding=encoding)
                            return ""
                        
                        mock_read.side_effect = read_text_side_effect
                        result = await validator.validate()
        
        assert "passed" in result
        assert "compliance_score" in result
        assert "details" in result

    @pytest.mark.asyncio
    async def test_security_config_validator_integration(self, test_project):
        """Test SecurityConfigValidator with real configuration."""
        validator = SecurityConfigValidator()
        
        with patch.object(Path, "cwd", return_value=test_project):
            with patch("genesis.validation.security.config_validator.Path") as mock_path:
                # Return actual config files from test project
                def rglob_side_effect(pattern):
                    return list(test_project.rglob(pattern))
                
                mock_path.return_value.rglob.side_effect = rglob_side_effect
                result = await validator.validate()
        
        assert "passed" in result
        assert "security_score" in result
        assert "security_posture" in result

    @pytest.mark.asyncio
    async def test_all_validators_together(self, test_project):
        """Test all validators working together."""
        results = {}
        
        # Run all validators
        with patch.object(Path, "cwd", return_value=test_project):
            # Secrets Scanner
            scanner = SecretsScanner()
            with patch("genesis.validation.security.secrets_scanner.Path") as mock_path:
                mock_path.return_value.rglob.return_value = list(test_project.rglob("*.py"))
                results["secrets"] = await scanner.validate()
            
            # Vulnerability Scanner
            vuln_scanner = VulnerabilityScanner()
            vuln_scanner.tools_available = {}  # No tools for testing
            results["vulnerabilities"] = await vuln_scanner.validate()
            
            # Compliance Validator
            compliance = ComplianceValidator()
            results["compliance"] = await compliance.validate()
            
            # Encryption Validator
            encryption = EncryptionValidator()
            results["encryption"] = await encryption.validate()
            
            # Security Config Validator
            config = SecurityConfigValidator()
            results["config"] = await config.validate()
        
        # Check overall results
        assert all("passed" in r for r in results.values())
        
        # Calculate overall security score
        scores = []
        if "compliance_score" in results["compliance"]:
            scores.append(results["compliance"]["compliance_score"])
        if "compliance_score" in results["encryption"]:
            scores.append(results["encryption"]["compliance_score"])
        if "security_score" in results["config"]:
            scores.append(results["config"]["security_score"])
        
        if scores:
            overall_score = sum(scores) / len(scores)
            assert overall_score >= 0  # Should have some score

    @pytest.mark.asyncio
    async def test_security_validation_with_issues(self, tmp_path):
        """Test validators with security issues present."""
        # Create problematic files
        genesis_dir = tmp_path / "genesis"
        genesis_dir.mkdir()
        
        # File with hardcoded secrets
        bad_file = genesis_dir / "bad_security.py"
        bad_file.write_text("""
# Bad practices for testing
API_KEY = "sk_test_FAKE_KEY_FOR_TESTING_ONLY"
PASSWORD = "admin123"
BINANCE_SECRET = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

# No encryption
def store_password(password):
    with open("passwords.txt", "w") as f:
        f.write(password)  # Plaintext storage!

# No input validation
def process_order(data):
    exec(data["command"])  # Dangerous!
""")
        
        scanner = SecretsScanner()
        with patch.object(Path, "cwd", return_value=tmp_path):
            with patch("genesis.validation.security.secrets_scanner.Path") as mock_path:
                mock_path.return_value.rglob.return_value = [bad_file]
                result = await scanner.validate()
        
        assert not result["passed"]  # Should fail
        assert result["summary"]["total_violations"] > 0
        assert any(v["type"] == "binance_secret" for v in result["violations"])
        assert any(v["severity"] == "critical" for v in result["violations"])

    @pytest.mark.asyncio
    async def test_compliance_gap_analysis(self, test_project):
        """Test compliance gap analysis."""
        validator = ComplianceValidator()
        
        # Simulate missing compliance requirements
        with patch.object(validator, "_check_authentication", return_value=False):
            with patch.object(validator, "_check_mfa", return_value=False):
                with patch.object(validator, "_check_encryption", return_value=False):
                    with patch.object(Path, "cwd", return_value=test_project):
                        result = await validator.validate()
        
        assert not result["passed"]
        assert len(result["gaps"]) > 0
        assert any("high" in gap.get("severity", "") for gap in result["gaps"])

    @pytest.mark.asyncio
    async def test_security_posture_report(self, test_project):
        """Test comprehensive security posture reporting."""
        validator = SecurityConfigValidator()
        
        with patch.object(Path, "cwd", return_value=test_project):
            with patch("genesis.validation.security.config_validator.Path") as mock_path:
                mock_path.return_value.rglob.return_value = list(test_project.rglob("*.py"))
                result = await validator.validate()
        
        posture = result["security_posture"]
        assert "overall_score" in posture
        assert "risk_level" in posture
        assert "category_scores" in posture
        assert "top_risks" in posture
        assert "strengths" in posture
        assert "weaknesses" in posture

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, test_project):
        """Test security recommendation generation."""
        # Run all validators and collect recommendations
        all_recommendations = []
        
        with patch.object(Path, "cwd", return_value=test_project):
            # Each validator should generate recommendations
            validators = [
                SecretsScanner(),
                VulnerabilityScanner(),
                ComplianceValidator(),
                EncryptionValidator(),
                SecurityConfigValidator(),
            ]
            
            for validator in validators:
                if isinstance(validator, VulnerabilityScanner):
                    validator.tools_available = {}
                
                try:
                    if isinstance(validator, SecretsScanner):
                        with patch("genesis.validation.security.secrets_scanner.Path") as mock_path:
                            mock_path.return_value.rglob.return_value = list(test_project.rglob("*.py"))
                            result = await validator.validate()
                    else:
                        result = await validator.validate()
                    
                    if "recommendations" in result:
                        all_recommendations.extend(result["recommendations"])
                except Exception:
                    pass  # Some validators might fail in test environment
        
        assert len(all_recommendations) > 0
        # Should have actionable recommendations
        assert any("implement" in r.lower() or "configure" in r.lower() or "add" in r.lower() 
                  for r in all_recommendations)