"""
Correlation ID tracking system for distributed tracing.

Provides correlation ID generation, propagation, and context management
across all system components for end-to-end request tracking.
"""

import asyncio
import contextvars
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable

import structlog


# Context variable for correlation ID
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id",
    default=None,
)


class CorrelationContext:
    """
    Manages correlation ID context for request tracking.
    
    Provides:
    - Correlation ID generation
    - Context propagation across async boundaries
    - Structured logging integration
    - Request tracking across components
    """
    
    def __init__(
        self,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self._middleware_stack: list[Callable] = []
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def get_current() -> Optional[str]:
        """Get current correlation ID from context."""
        return correlation_id_var.get()
    
    @staticmethod
    def set_current(correlation_id: str):
        """Set correlation ID in current context."""
        correlation_id_var.set(correlation_id)
    
    @staticmethod
    @contextmanager
    def with_correlation_id(correlation_id: Optional[str] = None):
        """
        Context manager for correlation ID scope.
        
        Args:
            correlation_id: Specific ID to use (None = generate new)
            
        Example:
            with CorrelationContext.with_correlation_id() as cid:
                # All operations here will have correlation_id=cid
                await some_operation()
        """
        if correlation_id is None:
            correlation_id = CorrelationContext.generate_correlation_id()
        
        token = correlation_id_var.set(correlation_id)
        try:
            yield correlation_id
        finally:
            correlation_id_var.reset(token)
    
    @staticmethod
    def inject_to_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """
        Inject correlation ID into HTTP headers.
        
        Args:
            headers: Existing headers dictionary
            
        Returns:
            Headers with correlation ID added
        """
        correlation_id = CorrelationContext.get_current()
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers
    
    @staticmethod
    def extract_from_headers(headers: Dict[str, str]) -> Optional[str]:
        """
        Extract correlation ID from HTTP headers.
        
        Args:
            headers: Headers dictionary
            
        Returns:
            Correlation ID if found
        """
        return headers.get("X-Correlation-ID") or headers.get("x-correlation-id")
    
    def configure_logging(self):
        """Configure structlog to include correlation ID."""
        def add_correlation_id(logger, method_name, event_dict):
            """Add correlation ID to log entries."""
            correlation_id = self.get_current()
            if correlation_id:
                event_dict["correlation_id"] = correlation_id
            return event_dict
        
        structlog.configure(
            processors=[
                add_correlation_id,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
        )
    
    def wrap_async_function(self, func: Callable) -> Callable:
        """
        Wrap async function to propagate correlation ID.
        
        Args:
            func: Async function to wrap
            
        Returns:
            Wrapped function with correlation propagation
        """
        async def wrapper(*args, **kwargs):
            # Get correlation ID from current context
            correlation_id = self.get_current()
            
            if correlation_id:
                # Propagate to new context
                correlation_id_var.set(correlation_id)
            else:
                # Generate new ID if none exists
                correlation_id = self.generate_correlation_id()
                correlation_id_var.set(correlation_id)
            
            try:
                return await func(*args, **kwargs)
            finally:
                # Context automatically cleaned up
                pass
        
        return wrapper
    
    def create_task_with_context(
        self,
        coro,
        *,
        name: Optional[str] = None,
    ) -> asyncio.Task:
        """
        Create asyncio task with correlation context.
        
        Args:
            coro: Coroutine to run
            name: Optional task name
            
        Returns:
            Task with correlation context
        """
        # Capture current correlation ID
        correlation_id = self.get_current()
        
        async def wrapped():
            # Set correlation ID in new task context
            if correlation_id:
                correlation_id_var.set(correlation_id)
            return await coro
        
        return asyncio.create_task(wrapped(), name=name)


class CorrelationMiddleware:
    """
    Middleware for automatic correlation ID management.
    
    Can be used with web frameworks, message queues, etc.
    """
    
    def __init__(
        self,
        context: Optional[CorrelationContext] = None,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.context = context or CorrelationContext()
        self.logger = logger or structlog.get_logger(__name__)
    
    async def __call__(self, request: Any, handler: Callable) -> Any:
        """
        Process request with correlation ID.
        
        Args:
            request: Incoming request object
            handler: Next handler in chain
            
        Returns:
            Response from handler
        """
        # Extract or generate correlation ID
        correlation_id = None
        
        # Try to extract from headers if request has them
        if hasattr(request, "headers"):
            correlation_id = CorrelationContext.extract_from_headers(
                dict(request.headers)
            )
        
        if not correlation_id:
            correlation_id = CorrelationContext.generate_correlation_id()
        
        # Set in context
        with CorrelationContext.with_correlation_id(correlation_id):
            self.logger.info(
                "Processing request",
                correlation_id=correlation_id,
                path=getattr(request, "path", None),
            )
            
            try:
                # Process request
                response = await handler(request)
                
                # Add correlation ID to response headers if possible
                if hasattr(response, "headers"):
                    response.headers["X-Correlation-ID"] = correlation_id
                
                return response
                
            except Exception as e:
                self.logger.error(
                    "Request processing failed",
                    correlation_id=correlation_id,
                    error=str(e),
                )
                raise


class CorrelationTracker:
    """
    Tracks correlation IDs across system operations.
    
    Provides visibility into request flows and operation chains.
    """
    
    def __init__(
        self,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self._active_correlations: Dict[str, Dict[str, Any]] = {}
        self._correlation_history: list[Dict[str, Any]] = []
    
    def start_operation(
        self,
        operation_name: str,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start tracking an operation.
        
        Args:
            operation_name: Name of the operation
            correlation_id: Correlation ID (None = generate)
            metadata: Additional metadata
            
        Returns:
            Correlation ID for the operation
        """
        if correlation_id is None:
            correlation_id = CorrelationContext.generate_correlation_id()
        
        operation = {
            "correlation_id": correlation_id,
            "operation": operation_name,
            "started_at": asyncio.get_event_loop().time(),
            "metadata": metadata or {},
            "events": [],
        }
        
        self._active_correlations[correlation_id] = operation
        
        self.logger.info(
            "Started operation tracking",
            correlation_id=correlation_id,
            operation=operation_name,
        )
        
        return correlation_id
    
    def add_event(
        self,
        correlation_id: str,
        event_name: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Add event to correlation tracking.
        
        Args:
            correlation_id: Correlation ID
            event_name: Name of the event
            data: Event data
        """
        if correlation_id in self._active_correlations:
            event = {
                "name": event_name,
                "timestamp": asyncio.get_event_loop().time(),
                "data": data or {},
            }
            self._active_correlations[correlation_id]["events"].append(event)
    
    def end_operation(
        self,
        correlation_id: str,
        status: str = "success",
        result: Optional[Any] = None,
    ):
        """
        End tracking an operation.
        
        Args:
            correlation_id: Correlation ID
            status: Operation status
            result: Operation result
        """
        if correlation_id not in self._active_correlations:
            return
        
        operation = self._active_correlations.pop(correlation_id)
        operation["ended_at"] = asyncio.get_event_loop().time()
        operation["duration"] = operation["ended_at"] - operation["started_at"]
        operation["status"] = status
        operation["result"] = result
        
        # Move to history
        self._correlation_history.append(operation)
        
        # Keep only recent history
        if len(self._correlation_history) > 1000:
            self._correlation_history = self._correlation_history[-1000:]
        
        self.logger.info(
            "Ended operation tracking",
            correlation_id=correlation_id,
            operation=operation["operation"],
            duration=operation["duration"],
            status=status,
        )
    
    def get_operation(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get operation details by correlation ID."""
        # Check active operations
        if correlation_id in self._active_correlations:
            return self._active_correlations[correlation_id].copy()
        
        # Check history
        for op in reversed(self._correlation_history):
            if op["correlation_id"] == correlation_id:
                return op.copy()
        
        return None
    
    def get_active_operations(self) -> list[Dict[str, Any]]:
        """Get all active operations."""
        return list(self._active_correlations.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        total_completed = len(self._correlation_history)
        
        if total_completed == 0:
            return {
                "active_operations": len(self._active_correlations),
                "completed_operations": 0,
                "avg_duration": 0.0,
                "success_rate": 0.0,
            }
        
        successful = sum(
            1 for op in self._correlation_history
            if op.get("status") == "success"
        )
        
        durations = [
            op["duration"] for op in self._correlation_history
            if "duration" in op
        ]
        
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "active_operations": len(self._active_correlations),
            "completed_operations": total_completed,
            "avg_duration": avg_duration,
            "success_rate": successful / total_completed,
        }


# Global correlation context instance
_global_context: Optional[CorrelationContext] = None


def get_correlation_context() -> CorrelationContext:
    """Get or create global correlation context."""
    global _global_context
    if _global_context is None:
        _global_context = CorrelationContext()
        _global_context.configure_logging()
    return _global_context