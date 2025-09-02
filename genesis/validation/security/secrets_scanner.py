"""Secret detection and scanning for hardcoded credentials."""

import asyncio
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from genesis.validation.base import (
    CheckStatus,
    ValidationCheck,
    ValidationContext,
    ValidationEvidence,
    ValidationResult,
    Validator,
)

logger = structlog.get_logger(__name__)


class SecretsScanner(Validator):
    """Scans for hardcoded secrets in codebase."""

    SECRET_PATTERNS = {
        "api_key": r"(?i)(api[_\-]?key|apikey)[\"']?\s*[:=]\s*[\"']([A-Za-z0-9+/]{20,})[\"']",
        "aws_key": r"(?i)(aws[_\-]?access[_\-]?key[_\-]?id|aws[_\-]?secret[_\-]?access[_\-]?key)",
        "private_key": r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
        "password": r"(?i)(password|passwd|pwd)[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']",
        "token": r"(?i)(auth[_\-]?token|access[_\-]?token|bearer)[\"']?\s*[:=]\s*[\"']([A-Za-z0-9+/=]{20,})[\"']",
        "binance_key": r"(?i)(binance[_\-]?api[_\-]?key)[\"']?\s*[:=]\s*[\"']([A-Za-z0-9]{64})[\"']",
        "binance_secret": r"(?i)(binance[_\-]?api[_\-]?secret)[\"']?\s*[:=]\s*[\"']([A-Za-z0-9]{64})[\"']",
        "github_token": r"ghp_[A-Za-z0-9]{36}",
        "stripe_key": r"sk_live_[A-Za-z0-9]{24,}",
        "jwt_secret": r"(?i)(jwt[_\-]?secret)[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']",
        "database_url": r"(?i)(database[_\-]?url|db[_\-]?url)[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']",
        "ssh_key": r"ssh-rsa\s+[A-Za-z0-9+/]+[=]{0,2}",
    }

    ALLOWED_PATTERNS = {
        "env_var": r"os\.getenv\([\"']([A-Z_]+)[\"']\)",
        "env_access": r"os\.environ\[[\"']([A-Z_]+)[\"']\]",
        "config_ref": r"config\.(get|settings)\.([A-Z_]+)",
        "vault_ref": r"vault\.get_secret\([\"']([^\"']+)[\"']\)",
        "dotenv": r"load_dotenv\(\)",
        "example": r"\.example$|_example\.|example_",
        "test_fixture": r"tests?/fixtures?/|test_data/",
    }

    EXCLUDE_PATHS = [
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.egg-info",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
    ]

    def __init__(self):
        """Initialize the secrets scanner."""
        super().__init__(
            validator_id="SEC-001",
            name="Secrets Scanner",
            description="Scans for hardcoded secrets and credentials in codebase"
        )
        self.violations: list[dict[str, Any]] = []
        self.files_scanned = 0
        self.gitleaks_available = self._check_gitleaks()
        self.is_critical = True
        self.timeout_seconds = 120

    def _check_gitleaks(self) -> bool:
        """Check if gitleaks is available."""
        try:
            result = subprocess.run(
                ["gitleaks", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def run_validation(self, context: ValidationContext) -> ValidationResult:
        """Run comprehensive secret scanning validation.
        
        Args:
            context: Validation context with configuration
            
        Returns:
            ValidationResult with checks and evidence
        """
        logger.info("Starting secret scanning validation")

        # Reset counters
        self.violations = []
        self.files_scanned = 0

        # Scan Python files
        python_violations = await self._scan_python_files()

        # Scan configuration files
        config_violations = await self._scan_config_files()

        # Scan with gitleaks if available
        git_violations = []
        if self.gitleaks_available:
            git_violations = await self._run_gitleaks()

        # Check environment variable usage
        env_var_issues = await self._check_env_var_usage()

        # Aggregate all violations
        all_violations = python_violations + config_violations + git_violations

        # Categorize by severity
        critical_count = sum(1 for v in all_violations if v.get("severity") == "critical")
        high_count = sum(1 for v in all_violations if v.get("severity") == "high")
        medium_count = sum(1 for v in all_violations if v.get("severity") == "medium")
        low_count = sum(1 for v in all_violations if v.get("severity") == "low")

        passed = critical_count == 0 and high_count == 0
        status = CheckStatus.PASSED if passed else CheckStatus.FAILED

        # Create validation checks from violations
        checks = []
        for violation in all_violations:
            check = ValidationCheck(
                id=f"SEC-001-{len(checks)+1:03d}",
                name=f"Secret Detection: {violation.get('type', 'unknown')}",
                description=f"Found potential {violation.get('type', 'secret')} in {violation.get('file', 'unknown')}",
                category="security",
                status=CheckStatus.FAILED,
                details=f"Line {violation.get('line', 'unknown')}: {violation.get('context', 'No context')}[:100]",
                is_blocking=violation.get('severity') in ['critical', 'high'],
                evidence=ValidationEvidence(
                    raw_data={"violation": violation}
                ),
                duration_ms=0,
                timestamp=datetime.now(),
                severity=violation.get('severity', 'medium'),
                remediation="Remove hardcoded secret and use environment variables or secret management service"
            )
            checks.append(check)

        # Add env var issues as checks
        for issue in env_var_issues:
            check = ValidationCheck(
                id=f"SEC-001-{len(checks)+1:03d}",
                name="Environment Variable Usage",
                description=issue.get('issue', 'Environment variable issue'),
                category="security",
                status=CheckStatus.WARNING,
                details=f"{issue.get('file', 'unknown')}:{issue.get('line', '')} - {issue.get('context', '')}[:100]",
                is_blocking=False,
                evidence=ValidationEvidence(
                    raw_data={"issue": issue}
                ),
                duration_ms=0,
                timestamp=datetime.now(),
                severity="low",
                remediation=issue.get('recommendation', 'Review environment variable usage')
            )
            checks.append(check)

        # Create overall evidence
        evidence = ValidationEvidence(
            metrics={
                "files_scanned": self.files_scanned,
                "total_violations": len(all_violations),
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
                "env_var_issues": len(env_var_issues),
            },
            raw_data={
                "violations": all_violations,
                "env_var_issues": env_var_issues,
                "recommendations": self._generate_recommendations(all_violations, env_var_issues),
                "gitleaks_scan": self.gitleaks_available,
            }
        )

        message = f"Scanned {self.files_scanned} files. Found {len(all_violations)} potential secrets."
        if not passed:
            message += f" Critical: {critical_count}, High: {high_count}"

        return ValidationResult(
            validator_id=self.validator_id,
            validator_name=self.name,
            status=status,
            message=message,
            checks=checks,
            evidence=evidence,
            metadata=context.metadata,
            passed_checks=self.files_scanned - len(all_violations) if self.files_scanned > len(all_violations) else 0,
            failed_checks=len(all_violations),
            warning_checks=len(env_var_issues),
            is_blocking=not passed
        )

    async def validate(self) -> dict[str, Any]:
        """Legacy validate method for backward compatibility.
        
        Returns:
            Legacy format validation results
        """
        # Create a minimal context for backward compatibility
        from datetime import datetime

        from genesis.validation.base import ValidationContext, ValidationMetadata

        context = ValidationContext(
            genesis_root=".",
            environment="production",
            run_mode="standard",
            dry_run=False,
            force_continue=False,
            metadata=ValidationMetadata(
                version="1.0.0",
                environment="production",
                run_id="legacy-run",
                started_at=datetime.now()
            )
        )

        result = await self.run_validation(context)

        # Convert back to legacy format
        return {
            "passed": result.status == CheckStatus.PASSED,
            "summary": result.evidence.metrics,
            "violations": result.evidence.raw_data.get("violations", []),
            "env_var_issues": result.evidence.raw_data.get("env_var_issues", []),
            "recommendations": result.evidence.raw_data.get("recommendations", []),
            "gitleaks_scan": result.evidence.raw_data.get("gitleaks_scan", False),
        }

    async def _scan_python_files(self) -> list[dict[str, Any]]:
        """Scan Python files for hardcoded secrets with parallel processing.
        
        Returns:
            List of violations found
        """
        violations = []
        genesis_path = Path("genesis")

        # Collect all Python files to scan
        files_to_scan = []
        for py_file in genesis_path.rglob("*.py"):
            # Skip excluded paths
            if any(exclude in str(py_file) for exclude in self.EXCLUDE_PATHS):
                continue
            files_to_scan.append(py_file)

        # Process files in parallel batches for better performance
        batch_size = 10  # Process 10 files concurrently
        for i in range(0, len(files_to_scan), batch_size):
            batch = files_to_scan[i:i + batch_size]
            tasks = [self._scan_file(file_path) for file_path in batch]
            batch_results = await asyncio.gather(*tasks)

            for file_violations in batch_results:
                violations.extend(file_violations)
                self.files_scanned += 1

        return violations

    async def _scan_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Scan a single file for secrets.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of violations found in file
        """
        violations = []

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                # Skip comments and docstrings
                if line.strip().startswith("#") or '"""' in line or "'''" in line:
                    continue

                # Check if line contains allowed patterns
                if self._is_allowed_pattern(line):
                    continue

                # Check for secret patterns
                for secret_type, pattern in self.SECRET_PATTERNS.items():
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        violation = {
                            "file": str(file_path),
                            "line": line_num,
                            "type": secret_type,
                            "severity": self._get_severity(secret_type),
                            "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
                            "context": line.strip()[:100],
                        }
                        violations.append(violation)
                        logger.warning(
                            "Potential secret found",
                            file=str(file_path),
                            line=line_num,
                            type=secret_type,
                        )

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")

        return violations

    def _is_allowed_pattern(self, line: str) -> bool:
        """Check if line contains allowed patterns.
        
        Args:
            line: Line to check
            
        Returns:
            True if line contains allowed pattern
        """
        for pattern_name, pattern in self.ALLOWED_PATTERNS.items():
            if re.search(pattern, line):
                return True
        return False

    def _get_severity(self, secret_type: str) -> str:
        """Get severity level for secret type.
        
        Args:
            secret_type: Type of secret
            
        Returns:
            Severity level
        """
        critical_types = ["private_key", "binance_secret", "aws_key", "ssh_key"]
        high_types = ["api_key", "binance_key", "database_url", "jwt_secret"]
        medium_types = ["token", "github_token", "stripe_key"]

        if secret_type in critical_types:
            return "critical"
        elif secret_type in high_types:
            return "high"
        elif secret_type in medium_types:
            return "medium"
        else:
            return "low"

    async def _scan_config_files(self) -> list[dict[str, Any]]:
        """Scan configuration files for secrets.
        
        Returns:
            List of violations found
        """
        violations = []
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.conf", "*.cfg"]

        for pattern in config_patterns:
            for config_file in Path(".").rglob(pattern):
                # Skip excluded paths
                if any(exclude in str(config_file) for exclude in self.EXCLUDE_PATHS):
                    continue

                self.files_scanned += 1
                file_violations = await self._scan_config_file(config_file)
                violations.extend(file_violations)

        return violations

    async def _scan_config_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Scan configuration file for secrets.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            List of violations found
        """
        violations = []

        try:
            content = file_path.read_text(encoding="utf-8")

            # Check for common secret patterns in config files
            secret_config_patterns = {
                "api_key_config": r"[\"']?api[_\-]?key[\"']?\s*:\s*[\"']([^\"']+)[\"']",
                "password_config": r"[\"']?password[\"']?\s*:\s*[\"']([^\"']+)[\"']",
                "token_config": r"[\"']?token[\"']?\s*:\s*[\"']([^\"']+)[\"']",
                "secret_config": r"[\"']?secret[\"']?\s*:\s*[\"']([^\"']+)[\"']",
            }

            for secret_type, pattern in secret_config_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip if it's a placeholder or environment variable reference
                    value = match.group(1) if match.groups() else match.group(0)
                    if not self._is_placeholder(value):
                        violation = {
                            "file": str(file_path),
                            "type": secret_type,
                            "severity": "high",
                            "match": match.group(0)[:50],
                        }
                        violations.append(violation)

        except Exception as e:
            logger.error(f"Error scanning config file {file_path}: {e}")

        return violations

    def _is_placeholder(self, value: str) -> bool:
        """Check if value is a placeholder.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is a placeholder
        """
        placeholders = [
            "your-api-key",
            "your-secret",
            "your-password",
            "xxx",
            "XXX",
            "...",
            "****",
            "${",
            "{{",
            "%(",
            "REPLACE_ME",
            "CHANGEME",
            "TODO",
        ]

        return any(placeholder in value for placeholder in placeholders)

    async def _run_gitleaks(self) -> list[dict[str, Any]]:
        """Run gitleaks to scan git history.
        
        Returns:
            List of violations found by gitleaks
        """
        violations = []

        try:
            # Create temporary config for gitleaks
            gitleaks_config = {
                "title": "Genesis Gitleaks Config",
                "allowlist": {
                    "paths": [
                        "tests/",
                        "docs/",
                        "*.md",
                        ".example",
                    ]
                }
            }

            config_path = Path(".gitleaks.toml")
            if not config_path.exists():
                # Use default config
                cmd = ["gitleaks", "detect", "--no-git", "--verbose", "--report-format", "json"]
            else:
                cmd = ["gitleaks", "detect", "--config", str(config_path), "--no-git", "--verbose", "--report-format", "json"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                try:
                    findings = json.loads(result.stdout)
                    for finding in findings:
                        violation = {
                            "file": finding.get("file", ""),
                            "line": finding.get("startLine", 0),
                            "type": finding.get("ruleID", "unknown"),
                            "severity": "high",
                            "match": finding.get("match", "")[:50],
                            "commit": finding.get("commit", "")[:8],
                            "source": "gitleaks",
                        }
                        violations.append(violation)
                except json.JSONDecodeError:
                    logger.error("Failed to parse gitleaks output")

        except subprocess.TimeoutExpired:
            logger.warning("Gitleaks scan timed out")
        except Exception as e:
            logger.error(f"Error running gitleaks: {e}")

        return violations

    async def _check_env_var_usage(self) -> list[dict[str, Any]]:
        """Check for proper environment variable usage.
        
        Returns:
            List of environment variable usage issues
        """
        issues = []

        # Check for direct os.environ access without defaults
        direct_access_pattern = r"os\.environ\[[\"']([^\"']+)[\"']\](?!\s*\.get)"

        for py_file in Path("genesis").rglob("*.py"):
            if any(exclude in str(py_file) for exclude in self.EXCLUDE_PATHS):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                lines = content.splitlines()

                for line_num, line in enumerate(lines, 1):
                    if re.search(direct_access_pattern, line):
                        issues.append({
                            "file": str(py_file),
                            "line": line_num,
                            "issue": "Direct os.environ access without default",
                            "recommendation": "Use os.getenv() with default value",
                            "context": line.strip()[:100],
                        })

            except Exception as e:
                logger.error(f"Error checking env vars in {py_file}: {e}")

        # Check for missing .env.example
        if not Path(".env.example").exists():
            issues.append({
                "file": ".env.example",
                "issue": "Missing .env.example file",
                "recommendation": "Create .env.example with all required environment variables",
            })

        return issues

    def _generate_recommendations(
        self,
        violations: list[dict[str, Any]],
        env_var_issues: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on findings.
        
        Args:
            violations: List of secret violations
            env_var_issues: List of environment variable issues
            
        Returns:
            List of recommendations
        """
        recommendations = []

        if violations:
            recommendations.append("Remove all hardcoded secrets from source code")
            recommendations.append("Use environment variables or secrets management service")

            # Check for specific types
            if any(v["type"] == "private_key" for v in violations):
                recommendations.append("Never commit private keys - use key management service")

            if any(v["type"] in ["binance_key", "binance_secret"] for v in violations):
                recommendations.append("Store exchange API keys in encrypted vault or environment variables")

            if any(v.get("source") == "gitleaks" for v in violations):
                recommendations.append("Review git history and rotate any exposed secrets")
                recommendations.append("Consider using git-filter-branch to remove secrets from history")

        if env_var_issues:
            recommendations.append("Use os.getenv() with defaults instead of direct os.environ access")
            recommendations.append("Document all required environment variables in .env.example")

        if not self.gitleaks_available:
            recommendations.append("Install gitleaks for comprehensive git history scanning")

        if not recommendations:
            recommendations.append("No security issues detected - maintain good practices")

        return recommendations
