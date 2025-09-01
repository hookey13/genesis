"""
Integration module for error handling components with existing system.

This module provides the glue code to integrate all error handling components
(circuit breakers, retry logic, DLQ, recovery procedures, etc.) with the
existing exchange gateway and other critical components.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type

import structlog

from genesis.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker_registry,
)
from genesis.core.correlation import (
    CorrelationContext,
    get_correlation_context,
)
from genesis.core.dead_letter_queue import DeadLetterQueue
from genesis.core.error_handler import (
    ErrorCategory,
    ErrorSeverity,
    get_error_handler,
)
from genesis.core.exceptions import (
    NetworkError,
    ConnectionTimeout,
    RateLimitError,
    OrderError,
    map_binance_error_to_domain,
)
from genesis.core.feature_flags import (
    DegradationLevel,
    FeatureManager,
)
from genesis.core.recovery_manager import RecoveryManager
from genesis.monitoring.error_budget import ErrorBudget
from genesis.utils.decorators import with_retry


class ErrorHandlingIntegration:
    """
    Central integration point for all error handling components.
    
    Provides:
    - Unified error handling pipeline
    - Component registration and configuration
    - Cross-component coordination
    - Monitoring and alerting integration
    """
    
    def __init__(
        self,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        
        # Initialize all error handling components
        self.error_handler = get_error_handler()
        self.correlation_context = get_correlation_context()
        self.circuit_registry = get_circuit_breaker_registry()
        self.recovery_manager = RecoveryManager()
        self.dlq = DeadLetterQueue(name="main")
        self.feature_manager = FeatureManager()
        self.error_budget = ErrorBudget()
        
        # Configure components
        self._configure_components()
        
        # Register default handlers
        self._register_default_handlers()
        
        self.logger.info("Error handling integration initialized")
    
    def _configure_components(self):
        """Configure error handling components."""
        
        # Configure circuit breakers for critical services
        self._configure_circuit_breakers()
        
        # Register DLQ retry handlers
        self._configure_dlq_handlers()
        
        # Set up error budget alerts
        self._configure_error_budget_alerts()
        
        # Configure feature degradation thresholds
        self._configure_feature_degradation()
    
    def _configure_circuit_breakers(self):
        """Configure circuit breakers for various services."""
        
        # Binance API circuit breaker
        binance_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=3,
            time_window=30.0,
            excluded_exceptions=(ValidationError,),  # Don't trip on validation
        )
        self.circuit_registry.get_or_create("binance_api", binance_config)
        
        # Database circuit breaker
        db_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            time_window=10.0,
        )
        self.circuit_registry.get_or_create("database", db_config)
        
        # WebSocket circuit breaker
        ws_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            success_threshold=1,
            time_window=60.0,
        )
        self.circuit_registry.get_or_create("websocket", ws_config)
        
        self.logger.info(
            "Configured circuit breakers",
            breakers=["binance_api", "database", "websocket"],
        )
    
    def _configure_dlq_handlers(self):
        """Configure Dead Letter Queue retry handlers."""
        
        # Order retry handler
        async def retry_order(payload: Dict[str, Any]):
            """Retry failed order."""
            self.logger.info("Retrying order from DLQ", payload=payload)
            # Would call actual order execution here
            pass
        
        self.dlq.register_retry_handler("order_execution", retry_order)
        
        # Market data retry handler
        async def retry_market_data(payload: Dict[str, Any]):
            """Retry market data fetch."""
            self.logger.info("Retrying market data fetch", payload=payload)
            # Would call actual market data fetch here
            pass
        
        self.dlq.register_retry_handler("market_data", retry_market_data)
    
    def _configure_error_budget_alerts(self):
        """Configure error budget alerting."""
        
        def budget_alert_handler(alert_data: Dict[str, Any]):
            """Handle error budget alerts."""
            alert_type = alert_data.get("type")
            slo_name = alert_data.get("slo_name")
            
            if alert_type == "budget_exhausted":
                # Trigger degradation when budget exhausted
                self.logger.critical(
                    "Error budget exhausted, triggering degradation",
                    slo=slo_name,
                )
                
                # Move to degraded mode
                if alert_data.get("critical"):
                    self.feature_manager.set_degradation_level(
                        DegradationLevel.CRITICAL
                    )
                else:
                    self.feature_manager.set_degradation_level(
                        DegradationLevel.MAJOR
                    )
            
            elif alert_type == "threshold_exceeded":
                # Warning when approaching limit
                self.logger.warning(
                    "Error budget threshold exceeded",
                    slo=slo_name,
                    consumption_rate=alert_data.get("consumption_rate"),
                )
        
        self.error_budget.register_alert_callback(budget_alert_handler)
    
    def _configure_feature_degradation(self):
        """Configure feature degradation based on error rates."""
        
        # Set error thresholds for auto-degradation
        features = [
            ("multi_pair_trading", 0.05),  # 5% error rate
            ("advanced_analytics", 0.1),   # 10% error rate
            ("ui_dashboard", 0.15),        # 15% error rate
        ]
        
        for feature_name, threshold in features:
            feature = self.feature_manager._features.get(feature_name)
            if feature:
                feature.error_threshold = threshold
    
    def _register_default_handlers(self):
        """Register default error handlers."""
        
        # Critical error callback
        def handle_critical_error(context):
            """Handle critical errors."""
            self.logger.critical(
                "CRITICAL ERROR - System intervention required",
                error_code=context.error_code,
                component=context.component,
            )
            
            # Could trigger pager duty, send alerts, etc.
        
        self.error_handler.register_critical_error_callback(handle_critical_error)
    
    def wrap_exchange_call(self, func: Callable) -> Callable:
        """
        Wrap exchange API calls with full error handling.
        
        Applies:
        - Correlation ID tracking
        - Circuit breaker protection
        - Retry logic with exponential backoff
        - Error tracking and recovery
        - Dead letter queue for failures
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create correlation ID
            correlation_id = CorrelationContext.get_current()
            if not correlation_id:
                correlation_id = CorrelationContext.generate_correlation_id()
                CorrelationContext.set_current(correlation_id)
            
            # Get circuit breaker
            breaker = self.circuit_registry.get_or_create("binance_api")
            
            try:
                # Execute through circuit breaker
                result = await breaker.call(func, *args, **kwargs)
                
                # Record success
                self.error_budget.record_success("order_execution")
                self.feature_manager.record_feature_success("order_execution")
                
                return result
                
            except Exception as e:
                # Map Binance errors to domain exceptions
                if hasattr(e, "code") and isinstance(e.code, int):
                    domain_error = map_binance_error_to_domain(e.code, str(e))
                else:
                    domain_error = e
                
                # Determine error severity and category
                severity = self._determine_severity(domain_error)
                category = self._determine_category(domain_error)
                
                # Handle error through global handler
                error_context = self.error_handler.handle_error(
                    error=domain_error,
                    severity=severity,
                    category=category,
                    component="exchange_gateway",
                    function=func.__name__,
                    line_number=0,
                    correlation_id=correlation_id,
                    additional_context={
                        "args": str(args)[:100],  # Truncate for logging
                        "kwargs": str(kwargs)[:100],
                    },
                )
                
                # Record error in budget
                self.error_budget.record_error(
                    "order_execution",
                    category.value,
                    severity,
                )
                
                # Record feature error
                self.feature_manager.record_feature_error("order_execution")
                
                # Attempt automatic recovery
                recovery_attempt = await self.recovery_manager.attempt_recovery(
                    error=domain_error,
                    context={
                        "function": func.__name__,
                        "args": args,
                        "kwargs": kwargs,
                    },
                    correlation_id=correlation_id,
                )
                
                if recovery_attempt and recovery_attempt.status == "succeeded":
                    # Retry after successful recovery
                    return await func(*args, **kwargs)
                
                # Add to DLQ if critical
                if severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
                    await self.dlq.add(
                        operation_type="exchange_call",
                        payload={
                            "function": func.__name__,
                            "args": str(args),
                            "kwargs": str(kwargs),
                        },
                        error=domain_error,
                        correlation_id=correlation_id,
                    )
                
                # Re-raise the domain error
                raise domain_error
        
        # Apply retry decorator
        wrapped = with_retry(
            max_attempts=3,
            retryable_exceptions=(NetworkError, ConnectionTimeout, RateLimitError),
        )(wrapper)
        
        return wrapped
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(error, (RiskLimitExceeded, TiltInterventionRequired)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (OrderError, RateLimitError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (NetworkError, DatabaseLocked)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _determine_category(self, error: Exception) -> ErrorCategory:
        """Determine error category based on exception type."""
        if isinstance(error, (OrderError, ExchangeError)):
            return ErrorCategory.EXCHANGE
        elif isinstance(error, (NetworkError, ConnectionTimeout)):
            return ErrorCategory.NETWORK
        elif isinstance(error, DatabaseError):
            return ErrorCategory.DATABASE
        elif isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (RiskLimitExceeded, TierViolation)):
            return ErrorCategory.BUSINESS
        else:
            return ErrorCategory.SYSTEM
    
    async def start_background_services(self):
        """Start background services for error handling."""
        
        # Start DLQ retry worker
        await self.dlq.start_retry_worker(interval=60)
        
        # Start recovery manager (if it had background tasks)
        # await self.recovery_manager.start()
        
        self.logger.info("Started error handling background services")
    
    async def stop_background_services(self):
        """Stop background services."""
        
        # Stop DLQ retry worker
        await self.dlq.stop_retry_worker()
        
        self.logger.info("Stopped error handling background services")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status.
        
        Returns:
            Dictionary with health metrics from all components
        """
        return {
            "circuit_breakers": self.circuit_registry.get_statistics(),
            "error_budget": self.error_budget.get_statistics(),
            "recovery_manager": self.recovery_manager.get_statistics(),
            "dlq": self.dlq.get_statistics(),
            "feature_flags": self.feature_manager.get_statistics(),
            "error_counts": self.error_handler.get_error_statistics(),
        }
    
    def create_gateway_wrapper(self, gateway):
        """
        Create a wrapped version of the exchange gateway.
        
        Args:
            gateway: Original gateway instance
            
        Returns:
            Wrapped gateway with error handling
        """
        # Wrap critical gateway methods
        critical_methods = [
            "place_order",
            "cancel_order",
            "get_balance",
            "get_ticker",
            "get_order_book",
        ]
        
        for method_name in critical_methods:
            if hasattr(gateway, method_name):
                original_method = getattr(gateway, method_name)
                wrapped_method = self.wrap_exchange_call(original_method)
                setattr(gateway, method_name, wrapped_method)
        
        self.logger.info(
            "Wrapped gateway methods with error handling",
            methods=critical_methods,
        )
        
        return gateway


# Global integration instance
_integration: Optional[ErrorHandlingIntegration] = None


def get_error_integration() -> ErrorHandlingIntegration:
    """Get or create global error handling integration."""
    global _integration
    if _integration is None:
        _integration = ErrorHandlingIntegration()
    return _integration


def integrate_with_gateway(gateway) -> Any:
    """
    Integrate error handling with exchange gateway.
    
    Args:
        gateway: Exchange gateway instance
        
    Returns:
        Gateway with integrated error handling
    """
    integration = get_error_integration()
    return integration.create_gateway_wrapper(gateway)


# Import guards for circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.core.exceptions import (
        ValidationError,
        RiskLimitExceeded,
        TierViolation,
        TiltInterventionRequired,
        DatabaseError,
        ExchangeError,
        DatabaseLocked,
    )