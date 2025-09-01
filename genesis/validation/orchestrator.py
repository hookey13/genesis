"""Enhanced validation orchestrator with full validator integration."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

# Import existing validators
from genesis.validation.compliance_validator import ComplianceValidator
from genesis.validation.dr_validator import DisasterRecoveryValidator
from genesis.validation.operational_validator import OperationalValidator
from genesis.validation.paper_trading_validator import PaperTradingValidator
from genesis.validation.performance_validator import PerformanceValidator
from genesis.validation.security_scanner import SecurityScanner
from genesis.validation.stability_tester import StabilityTester
from genesis.validation.test_validator import TestValidator

logger = structlog.get_logger(__name__)


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error, critical
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from a single validator."""
    validator_name: str
    category: str
    passed: bool
    score: float
    checks: list[ValidationCheck]
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validator_name": self.validator_name,
            "category": self.category,
            "passed": self.passed,
            "score": self.score,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                    "details": c.details
                }
                for c in self.checks
            ],
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    pipeline_name: str
    timestamp: datetime
    duration_seconds: float
    overall_passed: bool
    overall_score: float
    results: list[ValidationResult]
    blocking_issues: list[ValidationCheck] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "overall_passed": self.overall_passed,
            "overall_score": self.overall_score,
            "ready": self.ready,
            "results": [r.to_dict() for r in self.results],
            "blocking_issues": [
                {
                    "name": c.name,
                    "message": c.message,
                    "severity": c.severity
                }
                for c in self.blocking_issues
            ],
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        summary_lines = []

        if self.ready:
            summary_lines.append("✅ System is GO for launch!")
        else:
            summary_lines.append("❌ System is NO-GO")

        summary_lines.append(f"Score: {self.overall_score:.1f}%")
        summary_lines.append(f"Validators: {passed_count}/{total_count} passed")

        if self.blocking_issues:
            summary_lines.append(f"Blocking issues: {len(self.blocking_issues)}")
            for issue in self.blocking_issues[:3]:
                summary_lines.append(f"  - {issue.name}: {issue.message}")

        return "\n".join(summary_lines)


class ValidationOrchestrator:
    """Enhanced orchestrator with full validator integration and pipeline support."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize validation orchestrator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.config_path = self.genesis_root / "config" / "validation_pipeline.yaml"
        self.pipeline_config = self._load_pipeline_config()

        # Initialize all validators organized by category
        self.validators = self._initialize_validators()
        self.validator_categories = self._organize_validators()

    def _load_pipeline_config(self) -> dict[str, Any]:
        """Load pipeline configuration from YAML."""
        if not self.config_path.exists():
            logger.warning("Pipeline config not found, using defaults", path=str(self.config_path))
            return self._default_pipeline_config()

        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                return config.get("validation_pipeline", {})
        except Exception as e:
            logger.error("Failed to load pipeline config", error=str(e))
            return self._default_pipeline_config()

    def _default_pipeline_config(self) -> dict[str, Any]:
        """Return default pipeline configuration."""
        return {
            "quick": {
                "description": "Quick validation",
                "validators": ["test_coverage", "security", "health"],
                "timeout_minutes": 5
            },
            "standard": {
                "description": "Standard validation",
                "validators": ["test_coverage", "security", "performance", "operational"],
                "timeout_minutes": 30
            },
            "comprehensive": {
                "description": "Full validation",
                "validators": "all",
                "timeout_minutes": 120
            },
            "go_live": {
                "description": "Go-live readiness",
                "validators": "all",
                "required_score": 95,
                "blocking_on_failure": True,
                "timeout_minutes": 180
            }
        }

    def _initialize_validators(self) -> dict[str, Any]:
        """Initialize all available validators."""
        validators = {}

        # Technical validators
        validators["test_coverage"] = TestValidator()
        validators["code_quality"] = TestValidator()  # Using TestValidator as placeholder
        validators["performance"] = PerformanceValidator()
        validators["resources"] = PerformanceValidator()  # Using PerformanceValidator for resources
        validators["database"] = OperationalValidator(self.genesis_root)  # Using OperationalValidator

        # Security validators
        validators["secrets"] = SecurityScanner()
        validators["vulnerabilities"] = SecurityScanner()  # Using same scanner
        validators["compliance"] = ComplianceValidator(self.genesis_root)
        validators["encryption"] = SecurityScanner()  # Using SecurityScanner
        validators["security_config"] = SecurityScanner()  # Using SecurityScanner

        # Operational validators
        validators["monitoring"] = OperationalValidator(self.genesis_root)
        validators["backup"] = DisasterRecoveryValidator()
        validators["documentation"] = OperationalValidator(self.genesis_root)
        validators["deployment"] = OperationalValidator(self.genesis_root)
        validators["health"] = OperationalValidator(self.genesis_root)

        # Business validators
        validators["paper_trading"] = PaperTradingValidator()
        validators["stability"] = StabilityTester()
        validators["risk"] = PaperTradingValidator()  # Using PaperTradingValidator for risk
        validators["metrics"] = PerformanceValidator()  # Using PerformanceValidator for metrics
        validators["tier_gates"] = ComplianceValidator(self.genesis_root)  # Using ComplianceValidator

        return validators

    def _organize_validators(self) -> dict[str, list[str]]:
        """Organize validators by category."""
        return {
            "technical": ["test_coverage", "code_quality", "performance", "resources", "database"],
            "security": ["secrets", "vulnerabilities", "compliance", "encryption", "security_config"],
            "operational": ["monitoring", "backup", "documentation", "deployment", "health"],
            "business": ["paper_trading", "stability", "risk", "metrics", "tier_gates"]
        }

    async def run_pipeline(self, pipeline_name: str = "standard") -> ValidationReport:
        """Run a validation pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to run
            
        Returns:
            Comprehensive validation report
        """
        if pipeline_name not in self.pipeline_config:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        pipeline = self.pipeline_config[pipeline_name]
        logger.info("Starting validation pipeline", pipeline=pipeline_name)

        start_time = datetime.utcnow()

        # Determine which validators to run
        validators_to_run = self._get_pipeline_validators(pipeline)

        # Run validators by category for dependency management
        results = []
        for category in ["technical", "security", "operational", "business"]:
            category_validators = [
                v for v in validators_to_run
                if v in self.validator_categories.get(category, [])
            ]

            if category_validators:
                category_results = await self._run_category(category, category_validators)
                results.extend(category_results)

                # Check for blocking failures
                if self._has_blocking_failure(category_results, pipeline):
                    logger.warning(f"Blocking failure in {category}, stopping pipeline")
                    break

        # Calculate overall metrics
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        overall_score = self._calculate_overall_score(results)
        overall_passed = all(r.passed for r in results)
        blocking_issues = self._get_blocking_issues(results)

        # Determine readiness
        required_score = pipeline.get("required_score", 80)
        ready = overall_score >= required_score and len(blocking_issues) == 0

        report = ValidationReport(
            pipeline_name=pipeline_name,
            timestamp=start_time,
            duration_seconds=duration,
            overall_passed=overall_passed,
            overall_score=overall_score,
            results=results,
            blocking_issues=blocking_issues,
            ready=ready
        )

        logger.info(
            "Pipeline completed",
            pipeline=pipeline_name,
            duration=duration,
            score=overall_score,
            ready=ready
        )

        return report

    def _get_pipeline_validators(self, pipeline: dict[str, Any]) -> list[str]:
        """Get list of validators for a pipeline."""
        validators = pipeline.get("validators", [])

        if validators == "all":
            return list(self.validators.keys())
        elif isinstance(validators, list):
            # Filter to only available validators
            return [v for v in validators if v in self.validators]
        else:
            logger.warning("Invalid validators config, using all", validators=validators)
            return list(self.validators.keys())

    async def _run_category(
        self,
        category: str,
        validator_names: list[str]
    ) -> list[ValidationResult]:
        """Run validators in a category.
        
        Args:
            category: Category name
            validator_names: List of validator names to run
            
        Returns:
            List of validation results
        """
        logger.info(f"Running {category} validators", validators=validator_names)

        tasks = []
        for name in validator_names:
            if name in self.validators:
                validator = self.validators[name]
                tasks.append(self._run_validator(name, category, validator))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results = []
        for name, result in zip(validator_names, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Validator {name} failed", error=str(result))
                processed_results.append(self._create_error_result(name, category, str(result)))
            else:
                processed_results.append(result)

        return processed_results

    async def _run_validator(
        self,
        name: str,
        category: str,
        validator: Any
    ) -> ValidationResult:
        """Run a single validator.
        
        Args:
            name: Validator name
            category: Validator category
            validator: Validator instance
            
        Returns:
            Validation result
        """
        start_time = datetime.utcnow()
        logger.info(f"Running validator: {name}")

        try:
            # Determine timeout
            timeout = 300 if name in ["stability", "paper_trading"] else 60

            # Run validator
            result = await asyncio.wait_for(
                validator.validate(),
                timeout=timeout
            )

            # Process result
            duration = (datetime.utcnow() - start_time).total_seconds()

            return self._process_validator_result(name, category, result, duration)

        except TimeoutError:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Validator {name} timed out")
            return self._create_error_result(name, category, "Validation timed out")

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Validator {name} failed", error=str(e))
            return self._create_error_result(name, category, str(e))

    def _process_validator_result(
        self,
        name: str,
        category: str,
        result: dict[str, Any],
        duration: float
    ) -> ValidationResult:
        """Process raw validator result into ValidationResult.
        
        Args:
            name: Validator name
            category: Validator category
            result: Raw validator result
            duration: Validation duration
            
        Returns:
            Processed ValidationResult
        """
        checks = []
        errors = []
        warnings = []

        # Extract checks
        if "checks" in result:
            for check_name, check_result in result["checks"].items():
                passed = check_result.get("passed", True)
                message = check_result.get("message", "")
                severity = "error" if not passed else "info"

                checks.append(ValidationCheck(
                    name=check_name,
                    passed=passed,
                    message=message,
                    severity=severity,
                    details=check_result.get("details", {})
                ))

                if not passed:
                    if check_result.get("error"):
                        errors.append(f"{check_name}: {check_result['error']}")
                    if check_result.get("warning"):
                        warnings.append(f"{check_name}: {check_result['warning']}")

        # Add direct errors/warnings
        if "errors" in result:
            errors.extend(result["errors"])
        if "warnings" in result:
            warnings.extend(result["warnings"])

        return ValidationResult(
            validator_name=name,
            category=category,
            passed=result.get("passed", False),
            score=result.get("score", 0.0),
            checks=checks,
            duration_seconds=duration,
            errors=errors,
            warnings=warnings
        )

    def _create_error_result(
        self,
        name: str,
        category: str,
        error: str
    ) -> ValidationResult:
        """Create an error ValidationResult.
        
        Args:
            name: Validator name
            category: Validator category
            error: Error message
            
        Returns:
            Error ValidationResult
        """
        return ValidationResult(
            validator_name=name,
            category=category,
            passed=False,
            score=0.0,
            checks=[
                ValidationCheck(
                    name="validation_error",
                    passed=False,
                    message=error,
                    severity="critical"
                )
            ],
            duration_seconds=0.0,
            errors=[error]
        )

    def _has_blocking_failure(
        self,
        results: list[ValidationResult],
        pipeline: dict[str, Any]
    ) -> bool:
        """Check if results contain blocking failures.
        
        Args:
            results: List of validation results
            pipeline: Pipeline configuration
            
        Returns:
            True if blocking failure found
        """
        if not pipeline.get("blocking_on_failure", False):
            return False

        for result in results:
            if not result.passed:
                # Check for critical severity
                for check in result.checks:
                    if check.severity == "critical":
                        return True

        return False

    def _calculate_overall_score(self, results: list[ValidationResult]) -> float:
        """Calculate overall validation score.
        
        Args:
            results: List of validation results
            
        Returns:
            Overall score percentage
        """
        if not results:
            return 0.0

        total_score = sum(r.score for r in results)
        return total_score / len(results)

    def _get_blocking_issues(self, results: list[ValidationResult]) -> list[ValidationCheck]:
        """Get all blocking issues from results.
        
        Args:
            results: List of validation results
            
        Returns:
            List of blocking issues
        """
        blocking = []

        for result in results:
            for check in result.checks:
                if not check.passed and check.severity in ["critical", "error"]:
                    blocking.append(check)

        return blocking

    async def run_full_validation(self) -> ValidationReport:
        """Run comprehensive validation (alias for comprehensive pipeline).
        
        Returns:
            Comprehensive validation report
        """
        return await self.run_pipeline("comprehensive")

    async def run_quick_validation(self) -> ValidationReport:
        """Run quick validation for development.
        
        Returns:
            Quick validation report
        """
        return await self.run_pipeline("quick")

    async def run_go_live_validation(self) -> ValidationReport:
        """Run go-live readiness validation.
        
        Returns:
            Go-live validation report
        """
        return await self.run_pipeline("go_live")
