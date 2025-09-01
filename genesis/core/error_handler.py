"""
Global error handler infrastructure for comprehensive error management.

Provides centralized error handling with correlation IDs, severity levels,
categorization, and structured logging integration.
"""

import asyncio
import traceback
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import structlog


class ErrorSeverity(Enum):
    """Error severity levels for prioritization and alerting."""
    
    CRITICAL = "critical"  # Money at risk, immediate action required
    HIGH = "high"  # Service degraded, requires attention
    MEDIUM = "medium"  # Recoverable error, monitoring needed
    LOW = "low"  # Informational, no action required


class ErrorCategory(Enum):
    """Error categories for routing and handling strategies."""
    
    EXCHANGE = "exchange"  # Exchange API errors
    NETWORK = "network"  # Network connectivity issues
    DATABASE = "database"  # Database operations failures
    VALIDATION = "validation"  # Input validation errors
    BUSINESS = "business"  # Business logic violations
    SYSTEM = "system"  # System-level errors
    RATE_LIMIT = "rate_limit"  # Rate limiting errors
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker errors
    BACKPRESSURE = "backpressure"  # Backpressure-related errors


class ErrorContext:
    """Container for error context information."""
    
    def __init__(
        self,
        correlation_id: str,
        error: Exception,
        severity: ErrorSeverity,
        category: ErrorCategory,
        component: str,
        function: str,
        line_number: int,
        additional_context: Optional[Dict[str, Any]] = None,
    ):
        self.correlation_id = correlation_id
        self.error = error
        self.severity = severity
        self.category = category
        self.component = component
        self.function = function
        self.line_number = line_number
        self.additional_context = additional_context or {}
        self.timestamp = datetime.utcnow()
        self.error_code = self._generate_error_code()
        
    def _generate_error_code(self) -> str:
        """Generate GENESIS-XXXX error code based on category and severity."""
        category_codes = {
            ErrorCategory.EXCHANGE: "1",
            ErrorCategory.NETWORK: "2",
            ErrorCategory.DATABASE: "3",
            ErrorCategory.VALIDATION: "4",
            ErrorCategory.BUSINESS: "5",
            ErrorCategory.SYSTEM: "6",
        }
        severity_codes = {
            ErrorSeverity.CRITICAL: "9",
            ErrorSeverity.HIGH: "7",
            ErrorSeverity.MEDIUM: "5",
            ErrorSeverity.LOW: "3",
        }
        
        # Generate unique error code: GENESIS-CSNN
        # C = Category, S = Severity, NN = hash of error type
        error_hash = str(abs(hash(type(self.error).__name__)))[-2:]
        code = f"GENESIS-{category_codes[self.category]}{severity_codes[self.severity]}{error_hash}"
        return code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            "correlation_id": self.correlation_id,
            "error_code": self.error_code,
            "error_type": type(self.error).__name__,
            "error_message": str(self.error),
            "severity": self.severity.value,
            "category": self.category.value,
            "component": self.component,
            "function": self.function,
            "line_number": self.line_number,
            "timestamp": self.timestamp.isoformat(),
            "additional_context": self.additional_context,
            "traceback": traceback.format_exc(),
        }


class RemediationStep:
    """Represents a remediation step for error recovery."""
    
    def __init__(
        self,
        action: str,
        description: str,
        automated: bool = False,
        retry_after_seconds: Optional[int] = None,
    ):
        self.action = action
        self.description = description
        self.automated = automated
        self.retry_after_seconds = retry_after_seconds


class GlobalErrorHandler:
    """
    Central error handling system with correlation tracking and remediation.
    
    Provides unified error handling across all components with:
    - Correlation ID generation and tracking
    - Severity-based alerting and escalation
    - Structured logging integration
    - Remediation step suggestions
    - Error context capture
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self._error_handlers: Dict[Type[Exception], Any] = {}
        self._remediation_registry: Dict[str, List[RemediationStep]] = {}
        self._error_counts: Dict[str, int] = {}
        self._critical_error_callbacks: List[Any] = []
        self._initialize_remediation_registry()
        
    def _initialize_remediation_registry(self):
        """Initialize default remediation steps for common errors."""
        # Exchange errors
        self._remediation_registry["RateLimitError"] = [
            RemediationStep(
                "wait_and_retry",
                "Wait for rate limit window to reset",
                automated=True,
                retry_after_seconds=60,
            ),
            RemediationStep(
                "reduce_request_frequency",
                "Reduce API request frequency in configuration",
                automated=False,
            ),
        ]
        
        # Network errors
        self._remediation_registry["ConnectionTimeout"] = [
            RemediationStep(
                "retry_with_backoff",
                "Retry connection with exponential backoff",
                automated=True,
                retry_after_seconds=5,
            ),
            RemediationStep(
                "check_network",
                "Verify network connectivity and firewall rules",
                automated=False,
            ),
        ]
        
        # Database errors
        self._remediation_registry["DatabaseLocked"] = [
            RemediationStep(
                "retry_transaction",
                "Retry database transaction after brief delay",
                automated=True,
                retry_after_seconds=1,
            ),
            RemediationStep(
                "check_concurrent_access",
                "Review concurrent database access patterns",
                automated=False,
            ),
        ]
        
        # Business logic errors
        self._remediation_registry["TierViolation"] = [
            RemediationStep(
                "adjust_parameters",
                "Adjust trading parameters to match tier limits",
                automated=False,
            ),
            RemediationStep(
                "upgrade_tier",
                "Complete tier requirements to unlock feature",
                automated=False,
            ),
        ]
        
        self._remediation_registry["RiskLimitExceeded"] = [
            RemediationStep(
                "reduce_position",
                "Reduce position size to comply with risk limits",
                automated=False,
            ),
            RemediationStep(
                "wait_for_cooldown",
                "Wait for risk cooldown period",
                automated=True,
                retry_after_seconds=300,
            ),
        ]
        
        # Circuit breaker errors
        self._remediation_registry["CircuitBreakerError"] = [
            RemediationStep(
                "wait_for_recovery",
                "Wait for circuit breaker recovery timeout",
                automated=True,
                retry_after_seconds=60,
            ),
            RemediationStep(
                "use_fallback",
                "Use fallback mechanism if available",
                automated=True,
            ),
            RemediationStep(
                "check_service_health",
                "Check health of downstream service",
                automated=False,
            ),
        ]
        
        # Rate limiting errors
        self._remediation_registry["RateLimitError"] = [
            RemediationStep(
                "backoff_and_retry",
                "Apply exponential backoff and retry",
                automated=True,
                retry_after_seconds=2,
            ),
            RemediationStep(
                "use_priority_queue",
                "Use priority queue for critical operations",
                automated=True,
            ),
            RemediationStep(
                "adjust_rate_limits",
                "Adjust rate limit configuration",
                automated=False,
            ),
        ]
        
        # Backpressure errors
        self._remediation_registry["BackpressureError"] = [
            RemediationStep(
                "shed_low_priority",
                "Shed low priority events/requests",
                automated=True,
            ),
            RemediationStep(
                "increase_capacity",
                "Increase queue capacity if possible",
                automated=False,
            ),
            RemediationStep(
                "slow_down_producers",
                "Reduce event production rate",
                automated=True,
            ),
        ]
        
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID for tracking operations."""
        return str(uuid.uuid4())
    
    def handle_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: str = "unknown",
        function: str = "unknown",
        line_number: int = 0,
        correlation_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """
        Handle an error with full context capture and logging.
        
        Args:
            error: The exception to handle
            severity: Error severity level
            category: Error category
            component: Component where error occurred
            function: Function where error occurred
            line_number: Line number where error occurred
            correlation_id: Correlation ID for tracking
            additional_context: Additional context information
            
        Returns:
            ErrorContext object with full error information
        """
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()
            
        # Create error context
        context = ErrorContext(
            correlation_id=correlation_id,
            error=error,
            severity=severity,
            category=category,
            component=component,
            function=function,
            line_number=line_number,
            additional_context=additional_context,
        )
        
        # Log error based on severity
        log_method = self._get_log_method(severity)
        log_method(
            "Error handled",
            **context.to_dict()
        )
        
        # Update error counts
        error_key = f"{category.value}:{type(error).__name__}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Handle critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(context)
            
        # Get remediation steps
        remediation_steps = self.get_remediation_steps(error)
        if remediation_steps:
            self.logger.info(
                "Remediation steps available",
                correlation_id=correlation_id,
                steps=[
                    {
                        "action": step.action,
                        "description": step.description,
                        "automated": step.automated,
                    }
                    for step in remediation_steps
                ],
            )
            
        return context
    
    def _get_log_method(self, severity: ErrorSeverity):
        """Get appropriate log method based on severity."""
        severity_to_log = {
            ErrorSeverity.CRITICAL: self.logger.critical,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.LOW: self.logger.info,
        }
        return severity_to_log[severity]
    
    def _handle_critical_error(self, context: ErrorContext):
        """Handle critical errors with immediate escalation."""
        self.logger.critical(
            "CRITICAL ERROR - Immediate action required",
            correlation_id=context.correlation_id,
            error_code=context.error_code,
            component=context.component,
        )
        
        # Execute critical error callbacks
        for callback in self._critical_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(context))
                else:
                    callback(context)
            except Exception as e:
                self.logger.error(
                    "Failed to execute critical error callback",
                    error=str(e),
                    correlation_id=context.correlation_id,
                )
    
    def register_critical_error_callback(self, callback):
        """Register callback for critical errors."""
        self._critical_error_callbacks.append(callback)
        
    def register_error_handler(
        self,
        error_type: Type[Exception],
        handler,
    ):
        """Register custom handler for specific error type."""
        self._error_handlers[error_type] = handler
        
    def register_remediation_steps(
        self,
        error_name: str,
        steps: List[RemediationStep],
    ):
        """Register remediation steps for an error type."""
        self._remediation_registry[error_name] = steps
        
    def get_remediation_steps(
        self,
        error: Exception,
    ) -> Optional[List[RemediationStep]]:
        """Get remediation steps for an error."""
        error_name = type(error).__name__
        return self._remediation_registry.get(error_name)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        total_errors = sum(self._error_counts.values())
        
        # Calculate error rates by category
        category_stats = {}
        for key, count in self._error_counts.items():
            category = key.split(":")[0]
            if category not in category_stats:
                category_stats[category] = 0
            category_stats[category] += count
            
        return {
            "total_errors": total_errors,
            "error_counts": self._error_counts,
            "category_statistics": category_stats,
            "error_rate": total_errors,  # Would be calculated over time window
        }
    
    def clear_error_counts(self):
        """Clear error count statistics."""
        self._error_counts.clear()
        
    async def handle_async_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: str = "unknown",
        function: str = "unknown",
        line_number: int = 0,
        correlation_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """
        Async version of handle_error for use in async contexts.
        
        Args:
            error: The exception to handle
            severity: Error severity level
            category: Error category
            component: Component where error occurred
            function: Function where error occurred
            line_number: Line number where error occurred
            correlation_id: Correlation ID for tracking
            additional_context: Additional context information
            
        Returns:
            ErrorContext object with full error information
        """
        return self.handle_error(
            error=error,
            severity=severity,
            category=category,
            component=component,
            function=function,
            line_number=line_number,
            correlation_id=correlation_id,
            additional_context=additional_context,
        )


# Global instance for singleton pattern
_global_error_handler: Optional[GlobalErrorHandler] = None


def get_error_handler() -> GlobalErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = GlobalErrorHandler()
    return _global_error_handler