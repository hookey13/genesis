"""Production readiness validation framework."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from genesis.validation.compliance_validator import ComplianceValidator
from genesis.validation.dr_validator import DisasterRecoveryValidator
from genesis.validation.operational_validator import OperationalValidator
from genesis.validation.paper_trading_validator import PaperTradingValidator
from genesis.validation.performance_validator import PerformanceValidator
from genesis.validation.security_scanner import SecurityScanner
from genesis.validation.stability_tester import StabilityTester
from genesis.validation.test_validator import TestValidator

logger = structlog.get_logger(__name__)


class ValidationReport:
    """Standardized validation report data model."""

    def __init__(self, validator_name: str):
        """Initialize validation report.
        
        Args:
            validator_name: Name of the validator
        """
        self.validator_name = validator_name
        self.timestamp = datetime.utcnow()
        self.status = "pending"
        self.score = 0.0
        self.details: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passed = False

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary.
        
        Returns:
            Dictionary representation of report
        """
        return {
            "validator_name": self.validator_name,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "score": self.score,
            "details": self.details,
            "errors": self.errors,
            "warnings": self.warnings,
            "passed": self.passed
        }


class ValidationOrchestrator:
    """Coordinates all validators and generates comprehensive reports."""

    def __init__(
        self,
        genesis_root: Path | None = None,
        default_timeout: int = 60,
        long_running_timeout: int = 300
    ):
        """Initialize validation orchestrator.
        
        Args:
            genesis_root: Root directory of Genesis project
            default_timeout: Default timeout in seconds for validators (default: 60)
            long_running_timeout: Timeout for long-running validators (default: 300)
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.default_timeout = default_timeout
        self.long_running_timeout = long_running_timeout

        # Initialize all validators
        self.validators = {
            "test_coverage": TestValidator(),
            "stability": StabilityTester(),
            "security": SecurityScanner(),
            "performance": PerformanceValidator(),
            "disaster_recovery": DisasterRecoveryValidator(),
            "paper_trading": PaperTradingValidator(),
            "compliance": ComplianceValidator(self.genesis_root),
            "operational": OperationalValidator(self.genesis_root)
        }

        self.results: dict[str, ValidationReport] = {}

    async def run_all_validators(self, parallel: bool = True) -> dict[str, Any]:
        """Run all validators and generate comprehensive report.
        
        Args:
            parallel: Whether to run validators in parallel
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting validation orchestration", parallel=parallel)
        start_time = datetime.utcnow()

        if parallel:
            # Run validators in parallel
            tasks = []
            for name, validator in self.validators.items():
                tasks.append(self._run_validator(name, validator))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(self.validators.keys(), results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Validator {name} failed", error=str(result))
                    self.results[name] = self._create_error_report(name, str(result))
                else:
                    self.results[name] = self._create_report(name, result)
        else:
            # Run validators sequentially
            for name, validator in self.validators.items():
                try:
                    result = await self._run_validator(name, validator)
                    self.results[name] = self._create_report(name, result)
                except Exception as e:
                    logger.error(f"Validator {name} failed", error=str(e))
                    self.results[name] = self._create_error_report(name, str(e))

        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Generate overall summary
        overall_passed = all(r.passed for r in self.results.values())
        overall_score = sum(r.score for r in self.results.values()) / len(self.results) if self.results else 0

        summary = {
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "overall_passed": overall_passed,
            "overall_score": overall_score,
            "validators": {name: report.to_dict() for name, report in self.results.items()},
            "summary": self._generate_summary()
        }

        logger.info(
            "Validation orchestration completed",
            duration=duration,
            passed=overall_passed,
            score=overall_score
        )

        return summary

    async def run_critical_validators(self) -> dict[str, Any]:
        """Run only critical validators for quick validation.
        
        Returns:
            Critical validation results
        """
        critical_validators = ["test_coverage", "security", "performance"]
        logger.info("Running critical validators", validators=critical_validators)

        results = {}
        for name in critical_validators:
            if name in self.validators:
                try:
                    validator = self.validators[name]
                    result = await self._run_validator(name, validator)
                    results[name] = self._create_report(name, result)
                except Exception as e:
                    logger.error(f"Critical validator {name} failed", error=str(e))
                    results[name] = self._create_error_report(name, str(e))

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "critical",
            "validators": {name: report.to_dict() for name, report in results.items()}
        }

    async def _run_validator(self, name: str, validator: Any) -> dict[str, Any]:
        """Run a single validator with error handling and logging.
        
        Args:
            name: Validator name
            validator: Validator instance
            
        Returns:
            Validator results
        """
        logger.info(f"Running validator: {name}")

        try:
            # Add timeout for long-running validators
            if name in ["stability", "paper_trading"]:
                # These validators might take longer
                result = await asyncio.wait_for(
                    validator.validate(),
                    timeout=self.long_running_timeout
                )
            else:
                result = await asyncio.wait_for(
                    validator.validate(),
                    timeout=self.default_timeout
                )

            logger.info(f"Validator {name} completed", passed=result.get("passed", False))
            return result

        except TimeoutError:
            logger.error(f"Validator {name} timed out")
            raise
        except Exception as e:
            logger.error(f"Validator {name} failed", error=str(e))
            raise

    def _create_report(self, name: str, result: dict[str, Any]) -> ValidationReport:
        """Create a ValidationReport from validator results.
        
        Args:
            name: Validator name
            result: Raw validator results
            
        Returns:
            ValidationReport instance
        """
        report = ValidationReport(name)
        report.status = "completed"
        report.passed = result.get("passed", False)
        report.score = result.get("score", 0)
        report.details = result.get("details", {})

        # Extract errors and warnings
        if "errors" in result:
            report.errors = result["errors"]
        if "warnings" in result:
            report.warnings = result["warnings"]

        # Extract from checks if present
        if "checks" in result:
            for check_name, check_result in result["checks"].items():
                if not check_result.get("passed", True):
                    if "error" in check_result:
                        report.errors.append(f"{check_name}: {check_result['error']}")
                    if "warning" in check_result:
                        report.warnings.append(f"{check_name}: {check_result['warning']}")

        return report

    def _create_error_report(self, name: str, error: str) -> ValidationReport:
        """Create an error ValidationReport.
        
        Args:
            name: Validator name
            error: Error message
            
        Returns:
            ValidationReport instance with error
        """
        report = ValidationReport(name)
        report.status = "failed"
        report.passed = False
        report.score = 0
        report.errors = [error]
        return report

    def _generate_summary(self) -> str:
        """Generate a human-readable summary of validation results.
        
        Returns:
            Summary string
        """
        passed_validators = [name for name, report in self.results.items() if report.passed]
        failed_validators = [name for name, report in self.results.items() if not report.passed]

        summary_parts = []

        if not failed_validators:
            summary_parts.append("✅ All validators passed!")
        else:
            summary_parts.append(f"❌ {len(failed_validators)} validator(s) failed:")
            for name in failed_validators:
                report = self.results[name]
                summary_parts.append(f"  - {name}: score={report.score:.1f}%")
                if report.errors:
                    summary_parts.append(f"    Errors: {', '.join(report.errors[:3])}")

        summary_parts.append(f"\n📊 Overall Score: {sum(r.score for r in self.results.values()) / len(self.results):.1f}%")
        summary_parts.append(f"✓ Passed: {len(passed_validators)}/{len(self.results)}")

        return "\n".join(summary_parts)

    async def save_report(self, report: dict[str, Any], output_path: Path | None = None) -> Path:
        """Save validation report to file.
        
        Args:
            report: Validation report
            output_path: Optional output path
            
        Returns:
            Path to saved report
        """
        if output_path is None:
            reports_dir = self.genesis_root / "docs" / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"validation_report_{timestamp}.json"

        import json
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Validation report saved", path=str(output_path))
        return output_path


__all__ = [
    "ComplianceValidator",
    "DisasterRecoveryValidator",
    "OperationalValidator",
    "PaperTradingValidator",
    "PerformanceValidator",
    "SecurityScanner",
    "StabilityTester",
    "TestValidator",
    "ValidationOrchestrator",
    "ValidationReport"
]
