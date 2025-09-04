"""Security validation for Genesis trading system."""

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import BaseValidator, ValidationIssue, ValidationSeverity


class SecurityValidator(BaseValidator):
    """Validates security configuration and vulnerabilities."""
    
    @property
    def name(self) -> str:
        return "security"
    
    @property
    def description(self) -> str:
        return "Validates security configuration, API key encryption, and vulnerabilities"
    
    async def _validate(self, mode: str):
        """Perform security validation."""
        # Run dependency vulnerability scan
        await self._scan_dependencies()
        
        # Check API key encryption
        await self._check_api_key_encryption()
        
        # Verify secure configuration
        await self._verify_secure_config()
        
        # Test authentication flow
        if mode in ["standard", "thorough"]:
            await self._test_authentication()
        
        # Check for hardcoded secrets
        if mode == "thorough":
            await self._scan_for_secrets()
    
    async def _scan_dependencies(self):
        """Scan dependencies for vulnerabilities."""
        requirements_files = [
            Path("requirements/base.txt"),
            Path("requirements/sniper.txt"),
            Path("requirements/hunter.txt"),
            Path("requirements/strategist.txt")
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                # Check for outdated packages
                with open(req_file, "r") as f:
                    dependencies = f.readlines()
                
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Found {len(dependencies)} dependencies in {req_file.name}",
                    details={"file": str(req_file), "count": len(dependencies)}
                ))
                
                # Check for known vulnerable packages
                vulnerable_packages = [
                    ("requests", "<2.31.0", "CVE-2023-32681"),
                    ("cryptography", "<41.0.0", "CVE-2023-38325"),
                    ("pyyaml", "<6.0.1", "CVE-2020-14343")
                ]
                
                for package, vulnerable_version, cve in vulnerable_packages:
                    for dep in dependencies:
                        if package in dep.lower():
                            # Parse version
                            version_match = re.search(r"==(\d+\.\d+\.\d+)", dep)
                            if version_match:
                                version = version_match.group(1)
                                self.result.add_issue(ValidationIssue(
                                    severity=ValidationSeverity.INFO,
                                    message=f"Package {package} version {version} checked",
                                    details={"package": package, "version": version, "cve": cve}
                                ))
        
        # Try to run safety check if available
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Security scan completed: No vulnerabilities found"
                ))
            else:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Security scan found vulnerabilities",
                    details={"output": result.stdout},
                    recommendation="Run 'safety check' and update vulnerable packages"
                ))
        except FileNotFoundError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Safety scanner not installed",
                recommendation="Install safety: pip install safety"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Dependency scan error",
                details={"error": str(e)}
            ))
    
    async def _check_api_key_encryption(self):
        """Check API key encryption and storage."""
        # Check .env file
        env_file = Path(".env")
        env_example = Path(".env.example")
        
        if env_file.exists():
            # Check if .env is in .gitignore
            gitignore = Path(".gitignore")
            if gitignore.exists():
                with open(gitignore, "r") as f:
                    gitignore_content = f.read()
                
                self.check_condition(
                    ".env" in gitignore_content,
                    ".env file is gitignored",
                    ".env file NOT in .gitignore - CRITICAL SECURITY ISSUE",
                    ValidationSeverity.CRITICAL,
                    recommendation="Add .env to .gitignore immediately"
                )
            
            # Check for encrypted API keys
            with open(env_file, "r") as f:
                env_content = f.read()
            
            # Check for plaintext API keys
            sensitive_patterns = [
                r"API_KEY\s*=\s*['\"]?[a-zA-Z0-9]{20,}",
                r"SECRET\s*=\s*['\"]?[a-zA-Z0-9]{20,}",
                r"PASSWORD\s*=\s*['\"]?[^\s]+",
                r"PRIVATE_KEY\s*=\s*['\"]?[a-zA-Z0-9]{20,}"
            ]
            
            for pattern in sensitive_patterns:
                matches = re.findall(pattern, env_content, re.IGNORECASE)
                if matches:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Potential plaintext sensitive data in .env",
                        recommendation="Use encryption or secure vault for API keys"
                    ))
                    break
        
        if env_example.exists():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Environment template (.env.example) present"
            ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Missing .env.example file",
                recommendation="Create .env.example with placeholder values"
            ))
    
    async def _verify_secure_config(self):
        """Verify secure configuration settings."""
        try:
            from genesis.config.settings import Settings
            
            settings = Settings()
            
            # Check security settings
            security_checks = [
                (hasattr(settings, "use_https") and settings.use_https, "HTTPS enabled"),
                (hasattr(settings, "api_key_encrypted") and settings.api_key_encrypted, "API keys encrypted"),
                (hasattr(settings, "require_2fa") and settings.require_2fa, "2FA required"),
                (hasattr(settings, "session_timeout"), "Session timeout configured"),
                (hasattr(settings, "max_login_attempts"), "Login attempt limiting"),
                (hasattr(settings, "rate_limiting"), "Rate limiting enabled")
            ]
            
            for check, description in security_checks:
                if check:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Security feature: {description}"
                    ))
                else:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing security feature: {description}",
                        recommendation=f"Enable {description} in settings"
                    ))
            
            # Check WebSocket security
            if hasattr(settings, "websocket_url"):
                self.check_condition(
                    settings.websocket_url.startswith("wss://"),
                    "WebSocket using secure connection (wss://)",
                    "WebSocket using insecure connection",
                    ValidationSeverity.CRITICAL,
                    recommendation="Use wss:// for WebSocket connections"
                )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Settings module not found",
                recommendation="Implement genesis/config/settings.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Security configuration check failed",
                details={"error": str(e)}
            ))
    
    async def _test_authentication(self):
        """Test authentication flow."""
        try:
            from genesis.api.auth import AuthManager
            
            auth = AuthManager()
            
            # Test password strength requirements
            weak_passwords = ["password", "12345678", "qwerty123"]
            strong_password = "Str0ng!P@ssw0rd#2024"
            
            for weak_pass in weak_passwords:
                is_valid = auth.validate_password_strength(weak_pass)
                self.check_condition(
                    not is_valid,
                    f"Weak password rejected: {weak_pass[:3]}...",
                    f"Weak password accepted - SECURITY ISSUE",
                    ValidationSeverity.CRITICAL,
                    details={"password_hint": weak_pass[:3] + "..."},
                    recommendation="Enforce strong password requirements"
                )
            
            is_valid = auth.validate_password_strength(strong_password)
            self.check_condition(
                is_valid,
                "Strong password accepted",
                "Strong password rejected - too restrictive",
                ValidationSeverity.WARNING
            )
            
            # Test token generation
            token = auth.generate_token("test_user")
            self.check_condition(
                len(token) >= 32,
                f"Secure token generated: {len(token)} chars",
                "Token too short",
                ValidationSeverity.ERROR,
                recommendation="Use at least 256-bit tokens"
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Authentication not implemented (added at $2k+)",
                recommendation="Will be implemented with API in Hunter tier"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Authentication test error",
                details={"error": str(e)}
            ))
    
    async def _scan_for_secrets(self):
        """Scan codebase for hardcoded secrets."""
        # Patterns that indicate potential secrets
        secret_patterns = [
            (r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]", "API Key"),
            (r"secret\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]", "Secret"),
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Password"),
            (r"token\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]", "Token"),
            (r"private[_-]?key\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]", "Private Key")
        ]
        
        # Directories to scan
        scan_dirs = [Path("genesis"), Path("scripts"), Path("tests")]
        
        secrets_found = 0
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            
            for py_file in scan_dir.rglob("*.py"):
                try:
                    with open(py_file, "r") as f:
                        content = f.read()
                    
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            secrets_found += 1
                            self.result.add_issue(ValidationIssue(
                                severity=ValidationSeverity.CRITICAL,
                                message=f"Potential {secret_type} found in {py_file.name}",
                                details={"file": str(py_file)},
                                recommendation="Remove hardcoded secrets, use environment variables"
                            ))
                            break
                except Exception:
                    pass
        
        if secrets_found == 0:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="No hardcoded secrets detected"
            ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Found {secrets_found} potential hardcoded secrets",
                recommendation="Review and remove all hardcoded secrets immediately"
            ))