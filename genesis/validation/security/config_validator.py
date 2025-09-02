"""Security configuration validation."""

from pathlib import Path
from typing import Any

import structlog

from genesis.validation.base import Validator

logger = structlog.get_logger(__name__)


class SecurityConfigValidator(Validator):
    """Validates security configuration and policies."""

    SECURITY_REQUIREMENTS = {
        "headers": {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
        },
        "rate_limiting": {
            "api_rate_limit": 100,  # requests per minute
            "login_attempts": 5,     # max login attempts
            "order_rate_limit": 10,  # orders per second
        },
        "authentication": {
            "session_timeout": 3600,  # seconds
            "token_expiry": 86400,    # seconds
            "password_min_length": 12,
            "password_complexity": True,
            "mfa_required": False,  # Optional for MVP
        },
        "network": {
            "allowed_origins": ["https://localhost", "https://127.0.0.1"],
            "firewall_rules": True,
            "ip_whitelist": False,  # Optional
            "ddos_protection": True,
        },
    }

    def __init__(self):
        """Initialize security configuration validator."""
        super().__init__(
            validator_id="SEC-005",
            name="Security Config Validator",
            description="Validates security configuration and policies"
        )
        self.config_issues = []
        self.security_score = 0
        self.is_critical = False  # Not as critical as other security validators
        self.timeout_seconds = 60

    async def validate(self) -> dict[str, Any]:
        """Run comprehensive security configuration validation.
        
        Returns:
            Security configuration validation results
        """
        logger.info("Starting security configuration validation")

        # Validate security headers
        headers_results = await self._validate_security_headers()

        # Validate authentication mechanisms
        auth_results = await self._validate_authentication()

        # Validate rate limiting
        rate_limit_results = await self._validate_rate_limiting()

        # Validate firewall and network policies
        network_results = await self._validate_network_policies()

        # Validate logging and monitoring
        logging_results = await self._validate_logging_config()

        # Validate error handling
        error_handling_results = await self._validate_error_handling()

        # Calculate security posture score
        all_results = [
            headers_results,
            auth_results,
            rate_limit_results,
            network_results,
            logging_results,
            error_handling_results,
        ]

        total_passed = sum(r["passed"] for r in all_results)
        total_checks = sum(r["total_checks"] for r in all_results)

        self.security_score = (total_passed / total_checks * 100) if total_checks > 0 else 0
        passed = self.security_score >= 80  # 80% security configuration required

        return {
            "passed": passed,
            "security_score": self.security_score,
            "summary": {
                "total_checks": total_checks,
                "passed_checks": total_passed,
                "failed_checks": total_checks - total_passed,
                "headers_configured": headers_results["configured"],
                "auth_secure": auth_results["secure"],
                "rate_limiting_enabled": rate_limit_results["enabled"],
                "network_secure": network_results["secure"],
            },
            "details": {
                "security_headers": headers_results,
                "authentication": auth_results,
                "rate_limiting": rate_limit_results,
                "network_policies": network_results,
                "logging": logging_results,
                "error_handling": error_handling_results,
            },
            "issues": self.config_issues,
            "recommendations": self._generate_recommendations(all_results),
            "security_posture": self._generate_security_posture_report(all_results),
        }

    async def _validate_security_headers(self) -> dict[str, Any]:
        """Validate security headers configuration.
        
        Returns:
            Security headers validation results
        """
        checks = {}
        found_headers = {}

        # Initialize all header checks as False
        for header in self.SECURITY_REQUIREMENTS["headers"]:
            checks[header] = False

        # Check for security headers in configuration files
        config_patterns = ["*.py", "*.yaml", "*.yml", "*.conf", "*.toml"]

        for pattern in config_patterns:
            for config_file in Path(".").rglob(pattern):
                if "test" in str(config_file).lower():
                    continue

                try:
                    content = config_file.read_text()

                    for header, expected_value in self.SECURITY_REQUIREMENTS["headers"].items():
                        if header in content:
                            checks[header] = True
                            found_headers[header] = str(config_file)

                            # Check if value matches expected
                            if expected_value not in content:
                                self.config_issues.append({
                                    "type": "incorrect_header_value",
                                    "header": header,
                                    "expected": expected_value,
                                    "file": str(config_file),
                                })

                except Exception as e:
                    logger.error(f"Error checking file {config_file}: {e}")

        # Check for FastAPI/Flask security middleware
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for security middleware
                if "SecurityMiddleware" in content or "secure_headers" in content:
                    checks["middleware_configured"] = True

                # Check for CORS configuration
                if "CORS" in content or "cors" in content:
                    checks["cors_configured"] = True

            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "configured": passed >= len(self.SECURITY_REQUIREMENTS["headers"]) * 0.6,
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "found_headers": found_headers,
            "missing_headers": [h for h, v in checks.items() if not v and h in self.SECURITY_REQUIREMENTS["headers"]],
        }

    async def _validate_authentication(self) -> dict[str, Any]:
        """Validate authentication mechanisms.
        
        Returns:
            Authentication validation results
        """
        checks = {
            "auth_implementation": False,
            "password_policy": False,
            "session_management": False,
            "token_validation": False,
            "brute_force_protection": False,
            "secure_password_reset": False,
        }

        auth_requirements = self.SECURITY_REQUIREMENTS["authentication"]

        # Check for authentication implementation
        auth_files = [
            Path("genesis/security/authentication.py"),
            Path("genesis/api/auth.py"),
            Path("genesis/auth"),
        ]

        for auth_path in auth_files:
            if auth_path.exists():
                checks["auth_implementation"] = True

                if auth_path.is_file():
                    try:
                        content = auth_path.read_text()

                        # Check password policy
                        if f"min_length.*{auth_requirements['password_min_length']}" in content or "password_validator" in content or "validate_password" in content:
                            checks["password_policy"] = True

                        # Check session management
                        if "session" in content.lower() or "jwt" in content:
                            checks["session_management"] = True

                        # Check token validation
                        if "verify_token" in content or "validate_token" in content:
                            checks["token_validation"] = True

                        # Check brute force protection
                        if "rate_limit" in content or "login_attempts" in content:
                            checks["brute_force_protection"] = True

                        # Check password reset
                        if "reset_password" in content or "forgot_password" in content:
                            if "token" in content or "secure" in content:
                                checks["secure_password_reset"] = True

                    except Exception as e:
                        logger.error(f"Error checking auth file {auth_path}: {e}")

        # Check for JWT or session configuration
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for JWT implementation
                if "jwt" in content.lower() or "pyjwt" in content:
                    checks["token_validation"] = True

                # Check for session timeout
                if f"timeout.*{auth_requirements['session_timeout']}" in content:
                    checks["session_management"] = True

            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "secure": passed >= 4,  # At least 4 of 6 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "requirements": auth_requirements,
        }

    async def _validate_rate_limiting(self) -> dict[str, Any]:
        """Validate rate limiting configuration.
        
        Returns:
            Rate limiting validation results
        """
        checks = {
            "rate_limiter_implemented": False,
            "api_rate_limit": False,
            "login_rate_limit": False,
            "order_rate_limit": False,
            "circuit_breaker": False,
            "ddos_protection": False,
        }

        rate_requirements = self.SECURITY_REQUIREMENTS["rate_limiting"]

        # Check for rate limiter implementation
        rate_limiter_files = [
            Path("genesis/core/rate_limiter.py"),
            Path("genesis/security/rate_limiter.py"),
            Path("genesis/api/rate_limit.py"),
        ]

        for rate_file in rate_limiter_files:
            if rate_file.exists():
                checks["rate_limiter_implemented"] = True

                try:
                    content = rate_file.read_text()

                    # Check API rate limit
                    if str(rate_requirements["api_rate_limit"]) in content or "api_limit" in content:
                        checks["api_rate_limit"] = True

                    # Check login rate limit
                    if str(rate_requirements["login_attempts"]) in content or "login_limit" in content:
                        checks["login_rate_limit"] = True

                    # Check order rate limit
                    if str(rate_requirements["order_rate_limit"]) in content or "order_limit" in content:
                        checks["order_rate_limit"] = True

                except Exception as e:
                    logger.error(f"Error checking rate limiter {rate_file}: {e}")

        # Check for circuit breaker
        circuit_breaker = Path("genesis/exchange/circuit_breaker.py")
        if circuit_breaker.exists():
            checks["circuit_breaker"] = True

        # Check for DDoS protection
        for config_file in Path(".").rglob("*.yaml"):
            try:
                content = config_file.read_text()
                if "ddos" in content.lower() or "cloudflare" in content.lower():
                    checks["ddos_protection"] = True
                    break
            except Exception:
                pass

        # Check for rate limiting decorators
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()
                if "@rate_limit" in content or "RateLimiter" in content:
                    checks["rate_limiter_implemented"] = True
                if "throttle" in content.lower():
                    checks["api_rate_limit"] = True
            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "enabled": passed >= 3,  # At least 3 of 6 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "limits": rate_requirements,
        }

    async def _validate_network_policies(self) -> dict[str, Any]:
        """Validate firewall and network policies.
        
        Returns:
            Network policy validation results
        """
        checks = {
            "cors_configured": False,
            "allowed_origins": False,
            "firewall_rules": False,
            "ip_restrictions": False,
            "ssl_only": False,
            "port_restrictions": False,
        }

        network_requirements = self.SECURITY_REQUIREMENTS["network"]

        # Check for CORS configuration
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check CORS
                if "CORS" in content or "cors" in content:
                    checks["cors_configured"] = True

                    # Check allowed origins
                    for origin in network_requirements["allowed_origins"]:
                        if origin in content:
                            checks["allowed_origins"] = True
                            break

                # Check SSL enforcement
                if "https://" in content or "ssl=True" in content:
                    checks["ssl_only"] = True

                # Check IP restrictions
                if "ip_whitelist" in content or "allowed_ips" in content:
                    checks["ip_restrictions"] = True

            except Exception:
                pass

        # Check for firewall configuration
        firewall_configs = [
            Path("config/firewall.yaml"),
            Path("config/security.yaml"),
            Path(".firewall"),
            Path("iptables.rules"),
        ]

        for fw_config in firewall_configs:
            if fw_config.exists():
                checks["firewall_rules"] = True

                try:
                    content = fw_config.read_text()
                    if "port" in content or "PORT" in content:
                        checks["port_restrictions"] = True
                except Exception:
                    pass

        # Check Docker/Kubernetes network policies
        for yaml_file in Path(".").rglob("*.yaml"):
            if "docker" in str(yaml_file) or "k8s" in str(yaml_file):
                try:
                    content = yaml_file.read_text()
                    if "NetworkPolicy" in content or "ports:" in content:
                        checks["port_restrictions"] = True
                except Exception:
                    pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "secure": passed >= 3,  # At least 3 of 6 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "requirements": network_requirements,
        }

    async def _validate_logging_config(self) -> dict[str, Any]:
        """Validate logging and monitoring configuration.
        
        Returns:
            Logging configuration validation results
        """
        checks = {
            "structured_logging": False,
            "security_events_logged": False,
            "audit_trail": False,
            "log_rotation": False,
            "sensitive_data_masked": False,
            "centralized_logging": False,
        }

        # Check for structured logging
        logger_file = Path("genesis/utils/logger.py")
        if logger_file.exists():
            try:
                content = logger_file.read_text()

                if "structlog" in content or "json" in content:
                    checks["structured_logging"] = True

                # Check for sensitive data masking
                if "mask" in content or "redact" in content or "sanitize" in content:
                    checks["sensitive_data_masked"] = True

            except Exception:
                pass

        # Check for security event logging
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for security events
                if "logger" in content and any(event in content for event in ["auth", "login", "permission", "access"]):
                    checks["security_events_logged"] = True

                # Check for audit trail
                if "audit" in content.lower():
                    checks["audit_trail"] = True

            except Exception:
                pass

        # Check for log rotation
        logrotate_config = Path("/etc/logrotate.d/genesis")
        if logrotate_config.exists():
            checks["log_rotation"] = True
        else:
            # Check in application config
            for config_file in Path(".").rglob("*.yaml"):
                try:
                    content = config_file.read_text()
                    if "rotate" in content or "max_size" in content or "max_files" in content:
                        checks["log_rotation"] = True
                        break
                except Exception:
                    pass

        # Check for centralized logging
        for config_file in Path(".").rglob("*.yaml"):
            try:
                content = config_file.read_text()
                if any(service in content for service in ["elasticsearch", "logstash", "fluentd", "datadog", "sentry"]):
                    checks["centralized_logging"] = True
                    break
            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "configured": passed >= 4,  # At least 4 of 6 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
        }

    async def _validate_error_handling(self) -> dict[str, Any]:
        """Validate error handling configuration.
        
        Returns:
            Error handling validation results
        """
        checks = {
            "error_handler_implemented": False,
            "generic_error_messages": False,
            "stack_traces_hidden": False,
            "error_logging": False,
            "graceful_degradation": False,
            "input_validation": False,
        }

        # Check for error handler
        error_handler = Path("genesis/core/error_handler.py")
        if error_handler.exists():
            checks["error_handler_implemented"] = True

            try:
                content = error_handler.read_text()

                # Check for generic error messages
                if "generic" in content or "sanitize" in content:
                    checks["generic_error_messages"] = True

                # Check for stack trace handling
                if "traceback" in content and ("hide" in content or "production" in content):
                    checks["stack_traces_hidden"] = True

                # Check for error logging
                if "logger" in content:
                    checks["error_logging"] = True

            except Exception:
                pass

        # Check for graceful degradation
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for fallback mechanisms
                if "fallback" in content or "circuit_breaker" in content:
                    checks["graceful_degradation"] = True

                # Check for input validation
                if "validate" in content or "validator" in content or "schema" in content:
                    checks["input_validation"] = True

            except Exception:
                pass

        # Check for global exception handlers
        for py_file in Path("genesis").rglob("*app*.py"):
            try:
                content = py_file.read_text()
                if "@app.exception_handler" in content or "exception_handler" in content:
                    checks["error_handler_implemented"] = True
            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "secure": passed >= 4,  # At least 4 of 6 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
        }

    def _generate_recommendations(self, results: list[dict[str, Any]]) -> list[str]:
        """Generate security configuration recommendations.
        
        Args:
            results: List of validation results
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # Headers recommendations
        headers_results = results[0]
        if not headers_results["configured"]:
            if headers_results["missing_headers"]:
                recommendations.append(f"Configure missing security headers: {', '.join(headers_results['missing_headers'][:3])}")

        # Authentication recommendations
        auth_results = results[1]
        if not auth_results["secure"]:
            missing_auth = [k for k, v in auth_results["checks"].items() if not v]
            if "brute_force_protection" in missing_auth:
                recommendations.append("Implement brute force protection with rate limiting")
            if "password_policy" in missing_auth:
                recommendations.append(f"Enforce password policy (min {auth_results['requirements']['password_min_length']} chars)")

        # Rate limiting recommendations
        rate_results = results[2]
        if not rate_results["enabled"]:
            recommendations.append("Implement rate limiting for API endpoints")
            if not rate_results["checks"]["circuit_breaker"]:
                recommendations.append("Add circuit breaker for external service calls")

        # Network recommendations
        network_results = results[3]
        if not network_results["secure"]:
            if not network_results["checks"]["cors_configured"]:
                recommendations.append("Configure CORS with specific allowed origins")
            if not network_results["checks"]["ssl_only"]:
                recommendations.append("Enforce HTTPS/SSL for all connections")

        # Logging recommendations
        logging_results = results[4]
        if not logging_results["configured"]:
            if not logging_results["checks"]["structured_logging"]:
                recommendations.append("Implement structured logging (JSON format)")
            if not logging_results["checks"]["sensitive_data_masked"]:
                recommendations.append("Mask sensitive data in logs (passwords, API keys)")

        # Error handling recommendations
        error_results = results[5]
        if not error_results["secure"]:
            if not error_results["checks"]["stack_traces_hidden"]:
                recommendations.append("Hide stack traces in production error responses")
            if not error_results["checks"]["input_validation"]:
                recommendations.append("Implement comprehensive input validation")

        # General recommendations
        if self.security_score < 90:
            recommendations.append(f"Improve security score from {self.security_score:.1f}% to 90%+")

        if self.config_issues:
            critical_issues = [i for i in self.config_issues if i.get("severity") == "critical"]
            if critical_issues:
                recommendations.append(f"Address {len(critical_issues)} critical configuration issues")

        if not recommendations:
            recommendations.append("Maintain security configuration with regular audits")

        return recommendations

    def _generate_security_posture_report(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate overall security posture report.
        
        Args:
            results: List of validation results
            
        Returns:
            Security posture report
        """
        # Calculate category scores
        category_scores = {
            "headers": (results[0]["passed"] / results[0]["total_checks"] * 100) if results[0]["total_checks"] > 0 else 0,
            "authentication": (results[1]["passed"] / results[1]["total_checks"] * 100) if results[1]["total_checks"] > 0 else 0,
            "rate_limiting": (results[2]["passed"] / results[2]["total_checks"] * 100) if results[2]["total_checks"] > 0 else 0,
            "network": (results[3]["passed"] / results[3]["total_checks"] * 100) if results[3]["total_checks"] > 0 else 0,
            "logging": (results[4]["passed"] / results[4]["total_checks"] * 100) if results[4]["total_checks"] > 0 else 0,
            "error_handling": (results[5]["passed"] / results[5]["total_checks"] * 100) if results[5]["total_checks"] > 0 else 0,
        }

        # Determine risk level
        if self.security_score >= 90:
            risk_level = "low"
        elif self.security_score >= 70:
            risk_level = "medium"
        elif self.security_score >= 50:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Identify top risks
        top_risks = []
        for category, score in category_scores.items():
            if score < 50:
                top_risks.append({
                    "category": category,
                    "score": score,
                    "risk": "high" if score < 30 else "medium",
                })

        top_risks.sort(key=lambda x: x["score"])

        return {
            "overall_score": self.security_score,
            "risk_level": risk_level,
            "category_scores": category_scores,
            "top_risks": top_risks[:3],  # Top 3 risks
            "strengths": [cat for cat, score in category_scores.items() if score >= 80],
            "weaknesses": [cat for cat, score in category_scores.items() if score < 50],
            "compliance_ready": self.security_score >= 80,
        }
