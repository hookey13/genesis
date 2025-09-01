"""
Circuit breaker pattern implementation for preventing cascading failures.

Implements three states: CLOSED (normal), OPEN (failing), HALF_OPEN (recovery).
Protects against cascading failures by temporarily blocking calls to failing services.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union

import structlog

from genesis.core.exceptions import NetworkError
from genesis.core.error_handler import GlobalErrorHandler, ErrorSeverity, ErrorCategory

# Degradation strategies
class DegradationStrategy(Enum):
    """Strategies for graceful degradation when circuit is open."""
    
    FAIL_FAST = "fail_fast"  # Return error immediately
    FALLBACK = "fallback"  # Use fallback value/function
    CACHE = "cache"  # Return cached result if available
    QUEUE = "queue"  # Queue request for later
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Circuit tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5  # Failures needed to open circuit
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 2  # Successes needed to close from half-open
    time_window: float = 30.0  # Time window for failure counting
    excluded_exceptions: tuple = field(default_factory=tuple)  # Don't count these
    degradation_strategy: DegradationStrategy = DegradationStrategy.FAIL_FAST
    max_queued_requests: int = 100  # Max requests to queue when using QUEUE strategy
    cache_ttl: float = 300.0  # Cache TTL in seconds when using CACHE strategy


class CircuitBreakerError(NetworkError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, service: str, state: CircuitState, retry_after: float):
        super().__init__(
            f"Circuit breaker is {state.value} for {service}",
            code="GENESIS-2901",
        )
        self.service = service
        self.state = state
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    
    Transitions:
    - CLOSED -> OPEN: When failure threshold exceeded
    - OPEN -> HALF_OPEN: After recovery timeout
    - HALF_OPEN -> CLOSED: When success threshold met
    - HALF_OPEN -> OPEN: When any failure occurs
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        logger: Optional[structlog.BoundLogger] = None,
        error_handler: Optional[GlobalErrorHandler] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logger or structlog.get_logger(__name__)
        self.error_handler = error_handler
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._circuit_opened_at: Optional[datetime] = None
        
        # Failure tracking within time window
        self._recent_failures: list[datetime] = []
        
        # Lock for thread-safe state changes
        self._lock = asyncio.Lock()
        
        # Degradation support
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queued_requests if config else 100)
        self._fallback_function: Optional[Callable] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def _clean_old_failures(self):
        """Remove failures outside the time window."""
        if not self._recent_failures:
            return
            
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.time_window)
        self._recent_failures = [
            failure_time for failure_time in self._recent_failures
            if failure_time > cutoff_time
        ]
        self._failure_count = len(self._recent_failures)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._circuit_opened_at is None:
            return False
            
        elapsed = (datetime.utcnow() - self._circuit_opened_at).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure tracking on success in closed state
                self._recent_failures = []
                self._failure_count = 0
    
    async def _record_failure(self, exception: Exception):
        """Record a failed call."""
        # Check if exception should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return
            
        async with self._lock:
            now = datetime.utcnow()
            
            if self._state == CircuitState.CLOSED:
                # Track failure in time window
                self._recent_failures.append(now)
                self._clean_old_failures()
                
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state reopens circuit
                self._transition_to_open()
            
            self._last_failure_time = now
    
    def _transition_to_open(self):
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._circuit_opened_at = datetime.utcnow()
        self._success_count = 0
        
        self.logger.warning(
            "Circuit breaker opened",
            circuit=self.name,
            failure_count=self._failure_count,
            threshold=self.config.failure_threshold,
        )
        
        # Report to error handler if available
        if self.error_handler:
            error = CircuitBreakerError(
                service=self.name,
                state=self._state,
                retry_after=self.config.recovery_timeout
            )
            self.error_handler.handle_error(
                error=error,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.CIRCUIT_BREAKER,
                component="circuit_breaker",
                function="_transition_to_open",
                additional_context={
                    "circuit_name": self.name,
                    "failure_count": self._failure_count,
                    "threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout
                }
            )
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._failure_count = 0
        self._recent_failures = []
        
        self.logger.info(
            "Circuit breaker half-open, testing recovery",
            circuit=self.name,
        )
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._circuit_opened_at = None
        self._failure_count = 0
        self._success_count = 0
        self._recent_failures = []
        
        self.logger.info(
            "Circuit breaker closed, service recovered",
            circuit=self.name,
        )
        
        # Schedule processing of queued requests if applicable
        if self.config.degradation_strategy == DegradationStrategy.QUEUE:
            asyncio.create_task(self.process_queued_requests())
    
    def set_fallback(self, fallback_func: Callable):
        """Set fallback function for degradation."""
        self._fallback_function = fallback_func
    
    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        func_name = getattr(func, '__name__', str(func))
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{func_name}:{args_str}:{kwargs_str}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self.config.cache_ttl:
                return result
            else:
                del self._cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache result with timestamp."""
        self._cache[cache_key] = (result, datetime.utcnow())
        
        # Clean old cache entries
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if (now - timestamp).total_seconds() > self.config.cache_ttl
        ]
        for key in expired_keys:
            del self._cache[key]
    
    async def _handle_degradation(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle request based on degradation strategy."""
        strategy = self.config.degradation_strategy
        
        if strategy == DegradationStrategy.FAIL_FAST:
            retry_after = self.config.recovery_timeout
            if self._circuit_opened_at:
                elapsed = (datetime.utcnow() - self._circuit_opened_at).total_seconds()
                retry_after = max(0, self.config.recovery_timeout - elapsed)
            
            raise CircuitBreakerError(
                service=self.name,
                state=self._state,
                retry_after=retry_after,
            )
        
        elif strategy == DegradationStrategy.FALLBACK:
            if self._fallback_function:
                self.logger.info(
                    "Using fallback function",
                    circuit=self.name,
                )
                if asyncio.iscoroutinefunction(self._fallback_function):
                    return await self._fallback_function(*args, **kwargs)
                return self._fallback_function(*args, **kwargs)
            else:
                raise CircuitBreakerError(
                    service=self.name,
                    state=self._state,
                    retry_after=0,
                )
        
        elif strategy == DegradationStrategy.CACHE:
            cache_key = self._get_cache_key(func, args, kwargs)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result is not None:
                self.logger.info(
                    "Returning cached result",
                    circuit=self.name,
                    cache_key=cache_key,
                )
                return cached_result
            else:
                raise CircuitBreakerError(
                    service=self.name,
                    state=self._state,
                    retry_after=0,
                )
        
        elif strategy == DegradationStrategy.QUEUE:
            if not self._request_queue.full():
                request_info = {
                    'func': func,
                    'args': args,
                    'kwargs': kwargs,
                    'future': asyncio.Future()
                }
                await self._request_queue.put(request_info)
                self.logger.info(
                    "Request queued for later execution",
                    circuit=self.name,
                    queue_size=self._request_queue.qsize(),
                )
                return await request_info['future']
            else:
                raise CircuitBreakerError(
                    service=self.name,
                    state=self._state,
                    retry_after=0,
                )
        
        elif strategy == DegradationStrategy.RETRY_WITH_BACKOFF:
            retry_count = kwargs.pop('_circuit_retry_count', 0)
            max_retries = 3
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                self.logger.info(
                    "Retrying with backoff",
                    circuit=self.name,
                    retry_count=retry_count,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)
                kwargs['_circuit_retry_count'] = retry_count + 1
                return await self.call(func, *args, **kwargs)
            else:
                raise CircuitBreakerError(
                    service=self.name,
                    state=self._state,
                    retry_after=0,
                )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func execution
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from func execution
        """
        # Check if circuit should transition from open to half-open
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            async with self._lock:
                if self._state == CircuitState.OPEN:  # Double-check with lock
                    self._transition_to_half_open()
        
        # Handle degradation if circuit is open
        if self._state == CircuitState.OPEN:
            return await self._handle_degradation(func, args, kwargs)
        
        # Execute the function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._record_success()
            
            # Cache result if using cache strategy
            if self.config.degradation_strategy == DegradationStrategy.CACHE:
                cache_key = self._get_cache_key(func, args, kwargs)
                self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            await self._record_failure(e)
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        status = {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "time_window": self.config.time_window,
            },
        }
        
        if self._circuit_opened_at:
            elapsed = (datetime.utcnow() - self._circuit_opened_at).total_seconds()
            status["circuit_open_duration"] = elapsed
            status["retry_after"] = max(0, self.config.recovery_timeout - elapsed)
        
        if self._last_failure_time:
            status["last_failure"] = self._last_failure_time.isoformat()
        
        return status
    
    async def process_queued_requests(self):
        """Process queued requests when circuit recovers."""
        processed = 0
        while not self._request_queue.empty() and self._state == CircuitState.CLOSED:
            try:
                request_info = await self._request_queue.get()
                func = request_info['func']
                args = request_info['args']
                kwargs = request_info['kwargs']
                future = request_info['future']
                
                try:
                    result = await self.call(func, *args, **kwargs)
                    future.set_result(result)
                    processed += 1
                except Exception as e:
                    future.set_exception(e)
                    
            except asyncio.QueueEmpty:
                break
        
        if processed > 0:
            self.logger.info(
                "Processed queued requests",
                circuit=self.name,
                count=processed,
            )
    
    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self._transition_to_closed()
            self.logger.info(
                "Circuit breaker manually reset",
                circuit=self.name,
            )
        
        # Process any queued requests
        if self.config.degradation_strategy == DegradationStrategy.QUEUE:
            await self.process_queued_requests()
    
    async def trip(self):
        """Manually trip circuit breaker to open state."""
        async with self._lock:
            self._transition_to_open()
            self.logger.warning(
                "Circuit breaker manually tripped",
                circuit=self.name,
            )


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides centralized management of circuit breakers for different services
    or endpoints, allowing for monitoring and control of all breakers.
    """
    
    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        self.logger = logger or structlog.get_logger(__name__)
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for new breaker (if creating)
            
        Returns:
            Circuit breaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
                logger=self.logger,
            )
            
            self.logger.info(
                "Created new circuit breaker",
                circuit=name,
            )
        
        return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        if name in self._breakers:
            del self._breakers[name]
            self.logger.info(
                "Removed circuit breaker",
                circuit=name,
            )
            return True
        return False
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
    
    async def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            await breaker.reset()
        
        self.logger.info(
            "Reset all circuit breakers",
            count=len(self._breakers),
        )
    
    def get_open_circuits(self) -> list[str]:
        """Get names of all open circuits."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about all circuit breakers."""
        total = len(self._breakers)
        open_count = sum(1 for b in self._breakers.values() if b.is_open)
        half_open_count = sum(1 for b in self._breakers.values() if b.is_half_open)
        closed_count = sum(1 for b in self._breakers.values() if b.is_closed)
        
        return {
            "total_breakers": total,
            "open": open_count,
            "half_open": half_open_count,
            "closed": closed_count,
            "health_percentage": (closed_count / total * 100) if total > 0 else 100,
        }


# Global registry instance
_circuit_breaker_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create the global circuit breaker registry."""
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry