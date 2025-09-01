"""Custom exceptions for the validation framework."""

from typing import Any


class ValidationError(Exception):
    """Base exception for validation errors."""

    def __init__(
        self,
        message: str,
        validator_id: str | None = None,
        details: dict[str, Any] | None = None
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            validator_id: ID of the validator that raised the error
            details: Additional error details
        """
        super().__init__(message)
        self.validator_id = validator_id
        self.details = details or {}


class ValidationTimeout(ValidationError):
    """Raised when a validation operation times out."""

    def __init__(
        self,
        message: str,
        validator_id: str | None = None,
        timeout_seconds: int | None = None
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message
            validator_id: ID of the validator that timed out
            timeout_seconds: Timeout duration in seconds
        """
        details = {"timeout_seconds": timeout_seconds} if timeout_seconds else {}
        super().__init__(message, validator_id, details)
        self.timeout_seconds = timeout_seconds


class ValidationDependencyError(ValidationError):
    """Raised when a validation dependency is not met."""

    def __init__(
        self,
        message: str,
        validator_id: str | None = None,
        missing_dependencies: list[str] | None = None,
        failed_dependencies: list[str] | None = None
    ):
        """Initialize dependency error.
        
        Args:
            message: Error message
            validator_id: ID of the validator with dependency issues
            missing_dependencies: List of missing validator IDs
            failed_dependencies: List of failed validator IDs
        """
        details = {
            "missing_dependencies": missing_dependencies or [],
            "failed_dependencies": failed_dependencies or []
        }
        super().__init__(message, validator_id, details)
        self.missing_dependencies = missing_dependencies or []
        self.failed_dependencies = failed_dependencies or []


class ValidationConfigError(ValidationError):
    """Raised when there's an issue with validation configuration."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any = None,
        expected_type: str | None = None
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
            expected_type: Expected type or format
        """
        details = {
            "config_key": config_key,
            "config_value": config_value,
            "expected_type": expected_type
        }
        super().__init__(message, None, details)
        self.config_key = config_key
        self.config_value = config_value
        self.expected_type = expected_type


class ValidationOverrideError(ValidationError):
    """Raised when there's an issue with validation override."""

    def __init__(
        self,
        message: str,
        validator_id: str | None = None,
        authorization_level: str | None = None,
        required_level: str | None = None
    ):
        """Initialize override error.
        
        Args:
            message: Error message
            validator_id: ID of the validator being overridden
            authorization_level: Current authorization level
            required_level: Required authorization level
        """
        details = {
            "authorization_level": authorization_level,
            "required_level": required_level
        }
        super().__init__(message, validator_id, details)
        self.authorization_level = authorization_level
        self.required_level = required_level


class ValidationPipelineError(ValidationError):
    """Raised when there's an issue with the validation pipeline."""

    def __init__(
        self,
        message: str,
        pipeline_name: str | None = None,
        stage: str | None = None,
        failed_validators: list[str] | None = None
    ):
        """Initialize pipeline error.
        
        Args:
            message: Error message
            pipeline_name: Name of the pipeline
            stage: Pipeline stage where error occurred
            failed_validators: List of validators that failed
        """
        details = {
            "pipeline_name": pipeline_name,
            "stage": stage,
            "failed_validators": failed_validators or []
        }
        super().__init__(message, None, details)
        self.pipeline_name = pipeline_name
        self.stage = stage
        self.failed_validators = failed_validators or []


class ValidationRetryExhausted(ValidationError):
    """Raised when validation retries are exhausted."""

    def __init__(
        self,
        message: str,
        validator_id: str | None = None,
        attempts: int = 0,
        last_error: Exception | None = None
    ):
        """Initialize retry exhausted error.
        
        Args:
            message: Error message
            validator_id: ID of the validator that exhausted retries
            attempts: Number of attempts made
            last_error: Last error encountered
        """
        details = {
            "attempts": attempts,
            "last_error": str(last_error) if last_error else None
        }
        super().__init__(message, validator_id, details)
        self.attempts = attempts
        self.last_error = last_error
