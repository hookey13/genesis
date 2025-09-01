"""Base classes and interfaces for the validation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class CheckStatus(Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"
    IN_PROGRESS = "in_progress"


@dataclass
class ValidationMetadata:
    """Metadata for a validation run."""

    version: str
    environment: str
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: float | None = None
    machine_info: dict[str, Any] = field(default_factory=dict)
    additional_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Context passed to validators during execution."""

    genesis_root: str
    environment: str
    run_mode: str  # 'quick', 'standard', 'comprehensive'
    dry_run: bool
    force_continue: bool
    override_config: dict[str, Any] = field(default_factory=dict)
    shared_data: dict[str, Any] = field(default_factory=dict)
    metadata: ValidationMetadata | None = None


@dataclass
class ValidationEvidence:
    """Evidence collected during validation."""

    screenshots: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationCheck:
    """Individual validation check result."""

    id: str
    name: str
    description: str
    category: str
    status: CheckStatus
    details: str
    is_blocking: bool
    evidence: ValidationEvidence
    duration_ms: float
    timestamp: datetime
    error_message: str | None = None
    remediation: str | None = None
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "status": self.status.value,
            "details": self.details,
            "is_blocking": self.is_blocking,
            "evidence": {
                "screenshots": self.evidence.screenshots,
                "logs": self.evidence.logs,
                "metrics": self.evidence.metrics,
                "artifacts": self.evidence.artifacts,
                "raw_data": self.evidence.raw_data
            },
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "remediation": self.remediation,
            "severity": self.severity,
            "tags": self.tags
        }


@dataclass
class ValidationResult:
    """Result of a validation run."""

    validator_id: str
    validator_name: str
    status: CheckStatus
    message: str
    checks: list[ValidationCheck]
    evidence: ValidationEvidence
    metadata: ValidationMetadata
    score: Decimal = Decimal("0.0")
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    skipped_checks: int = 0
    error_checks: int = 0
    is_blocking: bool = False
    dependencies: list[str] = field(default_factory=list)

    def calculate_score(self) -> Decimal:
        """Calculate overall score based on check results."""
        if not self.checks:
            return Decimal("0.0")

        total_checks = len(self.checks)
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASSED)
        warnings = sum(1 for c in self.checks if c.status == CheckStatus.WARNING)

        # Warnings count as 0.5 points
        score = (Decimal(passed) + Decimal(warnings) * Decimal("0.5")) / Decimal(total_checks)
        return score * Decimal("100")

    def update_counts(self) -> None:
        """Update check counts based on current checks."""
        self.passed_checks = sum(1 for c in self.checks if c.status == CheckStatus.PASSED)
        self.failed_checks = sum(1 for c in self.checks if c.status == CheckStatus.FAILED)
        self.warning_checks = sum(1 for c in self.checks if c.status == CheckStatus.WARNING)
        self.skipped_checks = sum(1 for c in self.checks if c.status == CheckStatus.SKIPPED)
        self.error_checks = sum(1 for c in self.checks if c.status == CheckStatus.ERROR)
        self.score = self.calculate_score()

    def has_blocking_failures(self) -> bool:
        """Check if there are any blocking failures."""
        return any(
            c.is_blocking and c.status in [CheckStatus.FAILED, CheckStatus.ERROR]
            for c in self.checks
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "validator_id": self.validator_id,
            "validator_name": self.validator_name,
            "status": self.status.value,
            "message": self.message,
            "checks": [c.to_dict() for c in self.checks],
            "evidence": {
                "screenshots": self.evidence.screenshots,
                "logs": self.evidence.logs,
                "metrics": self.evidence.metrics,
                "artifacts": self.evidence.artifacts,
                "raw_data": self.evidence.raw_data
            },
            "metadata": {
                "version": self.metadata.version,
                "environment": self.metadata.environment,
                "run_id": self.metadata.run_id,
                "started_at": self.metadata.started_at.isoformat(),
                "completed_at": self.metadata.completed_at.isoformat() if self.metadata.completed_at else None,
                "duration_ms": self.metadata.duration_ms,
                "machine_info": self.metadata.machine_info,
                "additional_info": self.metadata.additional_info
            },
            "score": str(self.score),
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "skipped_checks": self.skipped_checks,
            "error_checks": self.error_checks,
            "is_blocking": self.is_blocking,
            "dependencies": self.dependencies
        }


class Validator(ABC):
    """Abstract base class for all validators."""

    def __init__(self, validator_id: str, name: str, description: str):
        """Initialize validator.
        
        Args:
            validator_id: Unique identifier for the validator
            name: Human-readable name
            description: Description of what the validator checks
        """
        self.validator_id = validator_id
        self.name = name
        self.description = description
        self.dependencies: set[str] = set()
        self.is_critical = False
        self.timeout_seconds = 60
        self.retry_count = 0
        self.retry_delay_seconds = 5

    @abstractmethod
    async def run_validation(self, context: ValidationContext) -> ValidationResult:
        """Run validation and return result.
        
        Args:
            context: Validation context with configuration and shared data
            
        Returns:
            ValidationResult with checks and evidence
        """
        pass

    async def pre_validation(self, context: ValidationContext) -> None:
        """Hook called before validation starts.
        
        Args:
            context: Validation context
        """
        pass

    async def post_validation(self, context: ValidationContext, result: ValidationResult) -> None:
        """Hook called after validation completes.
        
        Args:
            context: Validation context
            result: Validation result
        """
        pass

    def add_dependency(self, validator_id: str) -> None:
        """Add a dependency on another validator.
        
        Args:
            validator_id: ID of the validator this one depends on
        """
        self.dependencies.add(validator_id)

    def remove_dependency(self, validator_id: str) -> None:
        """Remove a dependency.
        
        Args:
            validator_id: ID of the validator to remove dependency on
        """
        self.dependencies.discard(validator_id)

    def set_critical(self, is_critical: bool = True) -> None:
        """Mark validator as critical (blocking).
        
        Args:
            is_critical: Whether this validator is critical
        """
        self.is_critical = is_critical

    def set_timeout(self, timeout_seconds: int) -> None:
        """Set validation timeout.
        
        Args:
            timeout_seconds: Timeout in seconds
        """
        self.timeout_seconds = timeout_seconds

    def set_retry_policy(self, retry_count: int, retry_delay_seconds: int = 5) -> None:
        """Set retry policy for transient failures.
        
        Args:
            retry_count: Number of retries
            retry_delay_seconds: Delay between retries in seconds
        """
        self.retry_count = retry_count
        self.retry_delay_seconds = retry_delay_seconds
