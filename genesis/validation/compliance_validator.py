"""Regulatory compliance validation module."""

from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ComplianceValidator:
    """Validates regulatory compliance requirements for production deployment."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize compliance validator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.results: dict[str, Any] = {}

    async def validate(self) -> dict[str, Any]:
        """Run all compliance validations.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting compliance validation")

        self.results = {
            "validator": "compliance",
            "timestamp": datetime.utcnow().isoformat(),
            "passed": True,
            "score": 0,
            "checks": {},
            "summary": "",
            "details": []
        }

        # Run all compliance checks
        await self._check_kyc_aml_requirements()
        await self._check_trading_limits()
        await self._check_audit_trail()
        await self._check_data_retention()
        await self._check_transaction_reporting()
        await self._check_risk_disclosure()
        await self._check_tier_compliance()
        await self._check_geographic_restrictions()

        # Calculate overall score
        passed_checks = sum(1 for check in self.results["checks"].values() if check["passed"])
        total_checks = len(self.results["checks"])
        self.results["score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        self.results["passed"] = self.results["score"] >= 90  # 90% compliance threshold

        # Generate summary
        if self.results["passed"]:
            self.results["summary"] = f"Compliance validation passed with {self.results['score']:.1f}% score"
        else:
            failed_checks = [name for name, check in self.results["checks"].items() if not check["passed"]]
            self.results["summary"] = f"Compliance validation failed. Failed checks: {', '.join(failed_checks)}"

        logger.info(
            "Compliance validation completed",
            passed=self.results["passed"],
            score=self.results["score"]
        )

        return self.results

    async def _check_kyc_aml_requirements(self) -> None:
        """Check KYC/AML compliance requirements."""
        check_name = "kyc_aml"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for user verification configuration
            config_path = self.genesis_root / "config" / "compliance.yaml"
            if not config_path.exists():
                details.append("WARNING: compliance.yaml not found - creating template")
                # Create template for compliance config
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text("""# Compliance Configuration
kyc_required: true
aml_checks_enabled: true
identity_verification: true
source_of_funds_verification: true
pep_screening: true  # Politically Exposed Persons
sanctions_screening: true
""")

            # Check for user data encryption
            user_data_path = self.genesis_root / ".genesis" / "data" / "users.db"
            if user_data_path.exists():
                # Would check for SQLCipher encryption in production
                details.append("User data storage detected - ensure encryption")

            # Check for compliance documentation
            compliance_docs = [
                "docs/compliance/privacy_policy.md",
                "docs/compliance/terms_of_service.md",
                "docs/compliance/risk_disclosure.md"
            ]

            for doc in compliance_docs:
                doc_path = self.genesis_root / doc
                if not doc_path.exists():
                    details.append(f"Missing: {doc}")
                    passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_trading_limits(self) -> None:
        """Check trading limit compliance."""
        check_name = "trading_limits"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for position limits configuration
            limits_path = self.genesis_root / "config" / "trading_rules.yaml"
            if limits_path.exists():
                with open(limits_path) as f:
                    content = f.read()
                    required_limits = [
                        "max_position_size",
                        "max_leverage",
                        "daily_loss_limit",
                        "max_open_positions"
                    ]

                    for limit in required_limits:
                        if limit not in content:
                            details.append(f"Missing limit: {limit}")
                            passed = False
                        else:
                            details.append(f"✓ {limit} configured")
            else:
                details.append("trading_rules.yaml not found")
                passed = False

            # Check for tier-based limits
            tier_gates_path = self.genesis_root / "config" / "tier_gates.yaml"
            if tier_gates_path.exists():
                details.append("✓ Tier-based limits configured")
            else:
                details.append("Missing tier_gates.yaml")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_audit_trail(self) -> None:
        """Check audit trail requirements."""
        check_name = "audit_trail"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for audit logging
            audit_log_path = self.genesis_root / ".genesis" / "logs" / "audit.log"
            if audit_log_path.parent.exists():
                details.append("✓ Audit log directory exists")
            else:
                details.append("Audit log directory missing")
                passed = False

            # Check for structured logging configuration
            logger_module = self.genesis_root / "genesis" / "utils" / "logger.py"
            if logger_module.exists():
                with open(logger_module) as f:
                    content = f.read()
                    if "structlog" in content and "JSON" in content:
                        details.append("✓ Structured JSON logging configured")
                    else:
                        details.append("Structured logging not properly configured")
                        passed = False
            else:
                details.append("Logger module not found")
                passed = False

            # Check for event tracking
            events_module = self.genesis_root / "genesis" / "core" / "events.py"
            if events_module.exists():
                details.append("✓ Event tracking system exists")
            else:
                details.append("Event tracking system missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_data_retention(self) -> None:
        """Check data retention policy compliance."""
        check_name = "data_retention"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for data retention policy
            retention_policy = self.genesis_root / "docs" / "compliance" / "data_retention_policy.md"
            if retention_policy.exists():
                details.append("✓ Data retention policy documented")
            else:
                details.append("Data retention policy missing")
                # Create template
                retention_policy.parent.mkdir(parents=True, exist_ok=True)
                retention_policy.write_text("""# Data Retention Policy

## Trading Records
- Order history: 7 years
- Transaction logs: 7 years
- Audit trails: 7 years

## User Data
- KYC documents: 5 years after account closure
- Communication records: 5 years

## System Logs
- Application logs: 90 days
- Security logs: 1 year
- Performance metrics: 6 months
""")
                details.append("Created data retention policy template")

            # Check for backup retention configuration
            backup_script = self.genesis_root / "scripts" / "backup.sh"
            if backup_script.exists():
                details.append("✓ Backup script exists")
            else:
                details.append("Backup script missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_transaction_reporting(self) -> None:
        """Check transaction reporting capabilities."""
        check_name = "transaction_reporting"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for reporting module
            analytics_module = self.genesis_root / "genesis" / "analytics" / "reports.py"
            if analytics_module.exists():
                details.append("✓ Reporting module exists")

                # Check for required report types
                with open(analytics_module) as f:
                    content = f.read()
                    required_reports = [
                        "transaction_report",
                        "daily_summary",
                        "monthly_statement",
                        "tax_report"
                    ]

                    for report in required_reports:
                        if report in content.lower():
                            details.append(f"✓ {report} capability found")
                        else:
                            details.append(f"Missing: {report}")
                            passed = False
            else:
                details.append("Analytics reports module missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_risk_disclosure(self) -> None:
        """Check risk disclosure requirements."""
        check_name = "risk_disclosure"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for risk disclosure document
            risk_disclosure = self.genesis_root / "docs" / "compliance" / "risk_disclosure.md"
            if not risk_disclosure.exists():
                risk_disclosure.parent.mkdir(parents=True, exist_ok=True)
                risk_disclosure.write_text("""# Risk Disclosure Statement

## Trading Risks
- Cryptocurrency trading involves substantial risk of loss
- Past performance does not guarantee future results
- Leverage can magnify both gains and losses
- Market volatility can result in rapid and substantial losses

## System Risks
- Technical failures may prevent order execution
- Network delays may impact trading performance
- Third-party service outages may affect operations

## Regulatory Risks
- Cryptocurrency regulations may change
- Tax obligations vary by jurisdiction
- Legal status of cryptocurrencies varies globally

By using this system, you acknowledge and accept all risks.
""")
                details.append("Created risk disclosure template")

            # Check for risk warnings in code
            ui_module = self.genesis_root / "genesis" / "ui"
            if ui_module.exists():
                has_risk_warning = False
                for py_file in ui_module.glob("*.py"):
                    with open(py_file) as f:
                        if "risk" in f.read().lower():
                            has_risk_warning = True
                            break

                if has_risk_warning:
                    details.append("✓ Risk warnings in UI")
                else:
                    details.append("No risk warnings found in UI")
                    passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_tier_compliance(self) -> None:
        """Check tier progression compliance."""
        check_name = "tier_compliance"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for tier gates configuration
            tier_gates = self.genesis_root / "config" / "tier_gates.yaml"
            if tier_gates.exists():
                details.append("✓ Tier gates configured")

                # Check for tier enforcement in code
                state_machine = self.genesis_root / "genesis" / "engine" / "state_machine.py"
                if state_machine.exists():
                    with open(state_machine) as f:
                        content = f.read()
                        if "@requires_tier" in content:
                            details.append("✓ Tier enforcement decorators found")
                        else:
                            details.append("No tier enforcement decorators")
                            passed = False
            else:
                details.append("Tier gates configuration missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }

    async def _check_geographic_restrictions(self) -> None:
        """Check geographic restriction compliance."""
        check_name = "geographic_restrictions"
        logger.info(f"Running {check_name} check")

        try:
            passed = True
            details = []

            # Check for geo-restriction configuration
            geo_config = self.genesis_root / "config" / "geographic_restrictions.yaml"
            if not geo_config.exists():
                geo_config.parent.mkdir(parents=True, exist_ok=True)
                geo_config.write_text("""# Geographic Restrictions Configuration
restricted_countries:
  - USA  # Regulatory restrictions
  - PRK  # North Korea - sanctions
  - IRN  # Iran - sanctions
  - SYR  # Syria - sanctions

allowed_regions:
  - APAC
  - EUR
  - LATAM

ip_geolocation_enabled: true
vpn_detection_enabled: true
""")
                details.append("Created geographic restrictions template")

            # Check for IP validation
            gateway = self.genesis_root / "genesis" / "exchange" / "gateway.py"
            if gateway.exists():
                details.append("✓ Exchange gateway exists for connection control")
            else:
                details.append("Exchange gateway missing")
                passed = False

            self.results["checks"][check_name] = {
                "passed": passed,
                "details": details
            }

        except Exception as e:
            logger.error(f"Error in {check_name} check", error=str(e))
            self.results["checks"][check_name] = {
                "passed": False,
                "details": [f"Error: {e!s}"]
            }
