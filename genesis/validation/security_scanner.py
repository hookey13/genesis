"""Security scanning integration for vulnerability detection."""

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class SecurityScanner:
    """Runs security scans for vulnerabilities and issues."""
    
    def __init__(self):
        self.sensitive_patterns = [
            r"(?i)(api[_-]?key|apikey|api_secret|secret_key)[\s]*=[\s]*['\"][\w]+['\"]",
            r"(?i)(password|passwd|pwd)[\s]*=[\s]*['\"][\w]+['\"]",
            r"(?i)(token|auth|bearer)[\s]*=[\s]*['\"][\w]+['\"]",
            r"(?i)binance[\s]*=[\s]*['\"][\w]+['\"]",
            r"sk_live_[\w]+",  # Stripe keys
            r"ghp_[\w]+",  # GitHub tokens
            r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",  # Private keys
        ]
        self.code_paths = ["genesis/", "scripts/"]
        self.exclude_patterns = [
            "*.pyc",
            "__pycache__",
            ".git",
            "tests/",
            "docs/",
        ]
        
    async def validate(self) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        try:
            # Run pip-audit for dependency vulnerabilities
            pip_audit_results = await self._run_pip_audit()
            
            # Run Bandit for code security issues
            bandit_results = await self._run_bandit()
            
            # Check for hardcoded secrets
            secrets_results = await self._check_hardcoded_secrets()
            
            # Check API key management
            api_key_results = self._check_api_key_management()
            
            # Check file permissions
            permissions_results = self._check_file_permissions()
            
            # Aggregate results
            critical_issues = (
                pip_audit_results["critical_count"]
                + bandit_results["high_severity_count"]
                + (1 if secrets_results["secrets_found"] else 0)
                + (1 if not api_key_results["secure"] else 0)
            )
            
            high_issues = (
                pip_audit_results["high_count"]
                + bandit_results["medium_severity_count"]
            )
            
            medium_issues = (
                pip_audit_results["moderate_count"]
                + bandit_results["low_severity_count"]
            )
            
            passed = critical_issues == 0
            
            return {
                "passed": passed,
                "details": {
                    "critical_issues": critical_issues,
                    "high_issues": high_issues,
                    "medium_issues": medium_issues,
                    "dependency_vulnerabilities": pip_audit_results["total_vulnerabilities"],
                    "code_security_issues": bandit_results["total_issues"],
                    "hardcoded_secrets": secrets_results["count"],
                    "api_key_secure": api_key_results["secure"],
                    "file_permissions_secure": permissions_results["secure"],
                },
                "vulnerabilities": {
                    "dependencies": pip_audit_results["vulnerabilities"],
                    "code": bandit_results["issues"],
                    "secrets": secrets_results["locations"],
                },
                "recommendations": self._generate_recommendations(
                    pip_audit_results,
                    bandit_results,
                    secrets_results,
                    api_key_results,
                    permissions_results,
                ),
            }
            
        except Exception as e:
            logger.error("Security validation failed", error=str(e))
            return {
                "passed": False,
                "error": str(e),
                "details": {},
            }
    
    async def _run_pip_audit(self) -> Dict[str, Any]:
        """Run pip-audit for dependency vulnerability scanning."""
        try:
            # Run pip-audit
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--desc"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                return {
                    "total_vulnerabilities": 0,
                    "critical_count": 0,
                    "high_count": 0,
                    "moderate_count": 0,
                    "low_count": 0,
                    "vulnerabilities": [],
                }
            
            # Parse JSON output
            try:
                audit_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Fallback to text parsing
                return self._parse_pip_audit_text(result.stdout)
            
            # Count vulnerabilities by severity
            vulnerabilities = []
            critical_count = 0
            high_count = 0
            moderate_count = 0
            low_count = 0
            
            for vuln in audit_data:
                severity = vuln.get("fix_versions", [{}])[0].get("severity", "UNKNOWN")
                
                if severity == "CRITICAL":
                    critical_count += 1
                elif severity == "HIGH":
                    high_count += 1
                elif severity in ["MODERATE", "MEDIUM"]:
                    moderate_count += 1
                else:
                    low_count += 1
                
                vulnerabilities.append({
                    "package": vuln.get("name", ""),
                    "version": vuln.get("version", ""),
                    "vulnerability": vuln.get("id", ""),
                    "severity": severity,
                    "description": vuln.get("description", ""),
                })
            
            return {
                "total_vulnerabilities": len(vulnerabilities),
                "critical_count": critical_count,
                "high_count": high_count,
                "moderate_count": moderate_count,
                "low_count": low_count,
                "vulnerabilities": vulnerabilities[:10],  # Limit to top 10
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "pip-audit timed out",
                "total_vulnerabilities": 0,
                "critical_count": 0,
                "high_count": 0,
                "moderate_count": 0,
                "low_count": 0,
                "vulnerabilities": [],
            }
        except FileNotFoundError:
            logger.warning("pip-audit not installed")
            return {
                "error": "pip-audit not installed - run: pip install pip-audit",
                "total_vulnerabilities": 0,
                "critical_count": 0,
                "high_count": 0,
                "moderate_count": 0,
                "low_count": 0,
                "vulnerabilities": [],
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_vulnerabilities": 0,
                "critical_count": 0,
                "high_count": 0,
                "moderate_count": 0,
                "low_count": 0,
                "vulnerabilities": [],
            }
    
    def _parse_pip_audit_text(self, output: str) -> Dict[str, Any]:
        """Parse pip-audit text output as fallback."""
        # Simple text parsing
        vulnerabilities = []
        lines = output.split("\n")
        
        for line in lines:
            if "vulnerability" in line.lower() or "cve" in line.lower():
                vulnerabilities.append({
                    "description": line.strip(),
                    "severity": "UNKNOWN",
                })
        
        return {
            "total_vulnerabilities": len(vulnerabilities),
            "critical_count": 0,
            "high_count": 0,
            "moderate_count": 0,
            "low_count": len(vulnerabilities),
            "vulnerabilities": vulnerabilities[:10],
        }
    
    async def _run_bandit(self) -> Dict[str, Any]:
        """Run Bandit for Python code security analysis."""
        try:
            # Run bandit
            result = subprocess.run(
                [
                    "bandit",
                    "-r",
                    "genesis/",
                    "-f",
                    "json",
                    "-ll",  # Only medium and high severity
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            # Parse JSON output
            try:
                bandit_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                return self._parse_bandit_text(result.stdout)
            
            # Extract metrics
            metrics = bandit_data.get("metrics", {})
            results = bandit_data.get("results", [])
            
            # Count by severity
            high_count = 0
            medium_count = 0
            low_count = 0
            issues = []
            
            for issue in results:
                severity = issue.get("issue_severity", "").upper()
                
                if severity == "HIGH":
                    high_count += 1
                elif severity == "MEDIUM":
                    medium_count += 1
                else:
                    low_count += 1
                
                issues.append({
                    "severity": severity,
                    "confidence": issue.get("issue_confidence", ""),
                    "description": issue.get("issue_text", ""),
                    "file": issue.get("filename", ""),
                    "line": issue.get("line_number", 0),
                    "test_id": issue.get("test_id", ""),
                })
            
            return {
                "total_issues": len(results),
                "high_severity_count": high_count,
                "medium_severity_count": medium_count,
                "low_severity_count": low_count,
                "issues": issues[:10],  # Limit to top 10
            }
            
        except subprocess.TimeoutExpired:
            return {
                "error": "bandit timed out",
                "total_issues": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0,
                "issues": [],
            }
        except FileNotFoundError:
            logger.warning("bandit not installed")
            return {
                "error": "bandit not installed - run: pip install bandit",
                "total_issues": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0,
                "issues": [],
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_issues": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0,
                "issues": [],
            }
    
    def _parse_bandit_text(self, output: str) -> Dict[str, Any]:
        """Parse bandit text output as fallback."""
        issues = []
        high_count = 0
        medium_count = 0
        low_count = 0
        
        lines = output.split("\n")
        for line in lines:
            if "Severity: High" in line:
                high_count += 1
            elif "Severity: Medium" in line:
                medium_count += 1
            elif "Severity: Low" in line:
                low_count += 1
            
            if "Issue:" in line:
                issues.append({
                    "description": line.replace("Issue:", "").strip(),
                    "severity": "UNKNOWN",
                })
        
        return {
            "total_issues": len(issues),
            "high_severity_count": high_count,
            "medium_severity_count": medium_count,
            "low_severity_count": low_count,
            "issues": issues[:10],
        }
    
    async def _check_hardcoded_secrets(self) -> Dict[str, Any]:
        """Check for hardcoded secrets in code."""
        secrets_found = []
        
        try:
            for code_path in self.code_paths:
                path = Path(code_path)
                if not path.exists():
                    continue
                
                # Search Python files
                for py_file in path.rglob("*.py"):
                    # Skip test files
                    if "test" in str(py_file).lower():
                        continue
                    
                    try:
                        content = py_file.read_text()
                        
                        # Check each sensitive pattern
                        for pattern in self.sensitive_patterns:
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                # Find line number
                                line_num = content[:match.start()].count("\n") + 1
                                
                                secrets_found.append({
                                    "file": str(py_file),
                                    "line": line_num,
                                    "pattern": pattern[:20] + "...",  # Truncate pattern
                                    "match": match.group()[:50],  # Truncate match
                                })
                    except Exception as e:
                        logger.error(f"Error checking file {py_file}", error=str(e))
            
            return {
                "secrets_found": len(secrets_found) > 0,
                "count": len(secrets_found),
                "locations": secrets_found[:10],  # Limit to top 10
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "secrets_found": False,
                "count": 0,
                "locations": [],
            }
    
    def _check_api_key_management(self) -> Dict[str, Any]:
        """Check if API keys are properly managed."""
        issues = []
        
        # Check for .env file
        env_file = Path(".env")
        env_example = Path(".env.example")
        
        if not env_example.exists():
            issues.append("Missing .env.example file")
        
        if env_file.exists():
            # Check if .env is in .gitignore
            gitignore = Path(".gitignore")
            if gitignore.exists():
                gitignore_content = gitignore.read_text()
                if ".env" not in gitignore_content:
                    issues.append(".env file not in .gitignore")
            else:
                issues.append("No .gitignore file found")
        
        # Check config files for hardcoded keys
        config_dir = Path("config")
        if config_dir.exists():
            for config_file in config_dir.glob("*.py"):
                try:
                    content = config_file.read_text()
                    if re.search(r"(?i)(api_key|secret)[\s]*=[\s]*['\"][\w]+['\"]", content):
                        issues.append(f"Possible hardcoded key in {config_file}")
                except Exception:
                    pass
        
        return {
            "secure": len(issues) == 0,
            "issues": issues,
        }
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions for sensitive files."""
        issues = []
        
        sensitive_files = [
            ".env",
            ".genesis/state/tier_state.json",
            "config/settings.py",
        ]
        
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                # Check if file is world-readable (Unix-like systems)
                try:
                    import stat
                    mode = path.stat().st_mode
                    if mode & stat.S_IROTH:
                        issues.append(f"{file_path} is world-readable")
                except Exception:
                    pass  # Windows doesn't have same permission model
        
        return {
            "secure": len(issues) == 0,
            "issues": issues,
        }
    
    def _generate_recommendations(
        self,
        pip_audit_results: Dict,
        bandit_results: Dict,
        secrets_results: Dict,
        api_key_results: Dict,
        permissions_results: Dict,
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Dependency vulnerabilities
        if pip_audit_results["critical_count"] > 0:
            recommendations.append(
                f"Fix {pip_audit_results['critical_count']} critical dependency vulnerabilities immediately"
            )
        if pip_audit_results["high_count"] > 0:
            recommendations.append(
                f"Update {pip_audit_results['high_count']} packages with high severity vulnerabilities"
            )
        
        # Code security issues
        if bandit_results["high_severity_count"] > 0:
            recommendations.append(
                f"Address {bandit_results['high_severity_count']} high severity code security issues"
            )
        if bandit_results["medium_severity_count"] > 5:
            recommendations.append(
                f"Review {bandit_results['medium_severity_count']} medium severity issues"
            )
        
        # Hardcoded secrets
        if secrets_results["secrets_found"]:
            recommendations.append(
                f"Remove {secrets_results['count']} hardcoded secrets from code"
            )
            recommendations.append(
                "Move all secrets to environment variables"
            )
        
        # API key management
        if not api_key_results["secure"]:
            for issue in api_key_results["issues"]:
                recommendations.append(f"Fix: {issue}")
        
        # File permissions
        if not permissions_results["secure"]:
            recommendations.append(
                "Restrict file permissions on sensitive files"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Security scan passed - continue monitoring")
        
        return recommendations