"""Production validation framework for Genesis trading system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import time
import traceback


class ValidationStatus(Enum):
    """Status of validation check."""
    
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "recommendation": self.recommendation,
            "code": self.code
        }


@dataclass
class ValidationMetrics:
    """Performance metrics for validation execution."""
    
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warning: int = 0
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    def finalize(self):
        """Calculate final metrics."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "checks_performed": self.checks_performed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warning": self.checks_warning,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    validator_name: str
    status: ValidationStatus = ValidationStatus.PENDING
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """Check if validation failed."""
        return self.status in [ValidationStatus.FAILED, ValidationStatus.ERROR]
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return self.status == ValidationStatus.WARNING or any(
            issue.severity == ValidationSeverity.WARNING for issue in self.issues
        )
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
    
    @property
    def error_issues(self) -> List[ValidationIssue]:
        """Get error issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue and update status."""
        self.issues.append(issue)
        
        # Update status based on issue severity
        if issue.severity == ValidationSeverity.CRITICAL:
            self.status = ValidationStatus.FAILED
        elif issue.severity == ValidationSeverity.ERROR and self.status != ValidationStatus.FAILED:
            self.status = ValidationStatus.FAILED
        elif issue.severity == ValidationSeverity.WARNING and self.status == ValidationStatus.PASSED:
            self.status = ValidationStatus.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validator_name": self.validator_name,
            "status": self.status.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
            "exception": self.exception
        }


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize validator with configuration."""
        self.config = config or {}
        self.result = ValidationResult(validator_name=self.name)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get validator name."""
        pass
    
    @property
    def description(self) -> str:
        """Get validator description."""
        return f"Validates {self.name.replace('_', ' ')}"
    
    async def validate(self, mode: str = "standard") -> ValidationResult:
        """Run validation checks."""
        self.result.status = ValidationStatus.RUNNING
        self.result.metrics.start_time = datetime.now()
        
        try:
            # Run pre-validation setup
            await self._setup()
            
            # Run actual validation
            await self._validate(mode)
            
            # Finalize result
            self._finalize_result()
            
        except Exception as e:
            self.result.status = ValidationStatus.ERROR
            self.result.exception = traceback.format_exc()
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Validator crashed: {str(e)}",
                details={"exception": str(e), "traceback": self.result.exception}
            ))
        finally:
            # Run cleanup
            await self._cleanup()
            self.result.metrics.finalize()
        
        return self.result
    
    @abstractmethod
    async def _validate(self, mode: str):
        """Perform actual validation logic."""
        pass
    
    async def _setup(self):
        """Setup before validation."""
        pass
    
    async def _cleanup(self):
        """Cleanup after validation."""
        pass
    
    def _finalize_result(self):
        """Finalize validation result."""
        # Set final status if not already set
        if self.result.status == ValidationStatus.RUNNING:
            if self.result.critical_issues or self.result.error_issues:
                self.result.status = ValidationStatus.FAILED
            elif self.result.has_warnings:
                self.result.status = ValidationStatus.WARNING
            else:
                self.result.status = ValidationStatus.PASSED
        
        # Update metrics
        self.result.metrics.checks_performed = len(self.result.issues)
        self.result.metrics.checks_failed = len(self.result.error_issues) + len(self.result.critical_issues)
        self.result.metrics.checks_warning = len([i for i in self.result.issues if i.severity == ValidationSeverity.WARNING])
        self.result.metrics.checks_passed = self.result.metrics.checks_performed - self.result.metrics.checks_failed - self.result.metrics.checks_warning
    
    def check_condition(
        self,
        condition: bool,
        success_message: str,
        failure_message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        recommendation: Optional[str] = None
    ):
        """Helper to check a condition and add appropriate issue."""
        if condition:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=success_message,
                details=details
            ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=severity,
                message=failure_message,
                details=details,
                recommendation=recommendation
            ))
    
    def check_threshold(
        self,
        value: Union[int, float, Decimal],
        threshold: Union[int, float, Decimal],
        comparison: str,
        metric_name: str,
        unit: str = "",
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        """Check if a value meets a threshold."""
        comparisons = {
            "<": lambda v, t: v < t,
            "<=": lambda v, t: v <= t,
            ">": lambda v, t: v > t,
            ">=": lambda v, t: v >= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t
        }
        
        if comparison not in comparisons:
            raise ValueError(f"Invalid comparison operator: {comparison}")
        
        passed = comparisons[comparison](value, threshold)
        
        if passed:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"{metric_name}: {value}{unit} (threshold: {comparison} {threshold}{unit})",
                details={"value": float(value), "threshold": float(threshold), "comparison": comparison}
            ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=severity,
                message=f"{metric_name} failed: {value}{unit} (required: {comparison} {threshold}{unit})",
                details={"value": float(value), "threshold": float(threshold), "comparison": comparison},
                recommendation=f"Ensure {metric_name} is {comparison} {threshold}{unit}"
            ))


class ValidationOrchestrator:
    """Orchestrates execution of multiple validators."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize orchestrator with configuration."""
        self.config = self._load_config(config_path)
        self.validators: Dict[str, BaseValidator] = {}
        self.results: Dict[str, ValidationResult] = {}
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        if not config_path:
            config_path = Path("scripts/config/validation_criteria.yaml")
        
        if config_path.exists():
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}
    
    def register_validator(self, validator: BaseValidator):
        """Register a validator."""
        self.validators[validator.name] = validator
    
    async def validate(
        self,
        mode: str = "standard",
        validators: Optional[List[str]] = None,
        parallel: bool = True,
        fail_fast: bool = False
    ) -> Dict[str, ValidationResult]:
        """Run validation suite."""
        # Determine which validators to run
        mode_config = self.config.get("validation_modes", {}).get(mode, {})
        
        if validators:
            validators_to_run = validators
        elif mode_config.get("validators") == "all":
            validators_to_run = list(self.validators.keys())
        else:
            validators_to_run = mode_config.get("validators", list(self.validators.keys()))
        
        # Override parallel setting from config if specified
        if "parallel" in mode_config:
            parallel = mode_config["parallel"]
        
        if "fail_fast" in mode_config:
            fail_fast = mode_config["fail_fast"]
        
        # Run validators
        if parallel:
            await self._run_parallel(validators_to_run, mode, fail_fast)
        else:
            await self._run_sequential(validators_to_run, mode, fail_fast)
        
        return self.results
    
    async def _run_parallel(self, validators: List[str], mode: str, fail_fast: bool):
        """Run validators in parallel."""
        tasks = []
        for name in validators:
            if name in self.validators:
                task = asyncio.create_task(self.validators[name].validate(mode))
                tasks.append((name, task))
        
        for name, task in tasks:
            try:
                result = await task
                self.results[name] = result
                
                if fail_fast and result.failed:
                    # Cancel remaining tasks
                    for _, t in tasks:
                        if not t.done():
                            t.cancel()
                    break
            except asyncio.CancelledError:
                pass
    
    async def _run_sequential(self, validators: List[str], mode: str, fail_fast: bool):
        """Run validators sequentially."""
        for name in validators:
            if name in self.validators:
                result = await self.validators[name].validate(mode)
                self.results[name] = result
                
                if fail_fast and result.failed:
                    break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total_passed = sum(1 for r in self.results.values() if r.passed)
        total_failed = sum(1 for r in self.results.values() if r.failed)
        total_warning = sum(1 for r in self.results.values() if r.has_warnings)
        
        return {
            "total_validators": len(self.results),
            "passed": total_passed,
            "failed": total_failed,
            "warnings": total_warning,
            "overall_status": "PASSED" if total_failed == 0 else "FAILED",
            "timestamp": datetime.now().isoformat(),
            "results": {name: result.to_dict() for name, result in self.results.items()}
        }