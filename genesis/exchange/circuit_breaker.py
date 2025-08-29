"""
Circuit breaker pattern implementation for fault tolerance.

Protects the system from cascading failures by temporarily blocking
requests to failing services.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class FailureRecord:
    """Record of a failure event."""

    timestamp: float
    error: str
    endpoint: str | None = None


class CircuitBreaker:
    """
    Circuit breaker for managing service failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests are blocked
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        failure_window_seconds: int = 30,
        recovery_timeout_seconds: int = 60,
        success_threshold: int = 2,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: int = 30,
    ):
        """
        Initialize the circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures to trip the circuit
            failure_window_seconds: Time window for counting failures
            recovery_timeout_seconds: Time to wait before testing recovery
            success_threshold: Successful calls needed to close circuit
            backoff_multiplier: Multiplier for exponential backoff
            max_backoff_seconds: Maximum backoff time
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_window_seconds = failure_window_seconds
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.success_threshold = success_threshold
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_seconds = max_backoff_seconds

        # State
        self.state = CircuitState.CLOSED
        self.failures: list[FailureRecord] = []
        self.consecutive_successes = 0
        self.last_failure_time: float | None = None
        self.state_changed_at: float = time.time()
        self.current_backoff = 1.0

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.state_changes = []

        logger.info(
            f"CircuitBreaker {name} initialized",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout_seconds,
        )

    def _clean_old_failures(self) -> None:
        """Remove failures outside the current window."""
        current_time = time.time()
        cutoff_time = current_time - self.failure_window_seconds

        self.failures = [f for f in self.failures if f.timestamp >= cutoff_time]

    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit state."""
        old_state = self.state
        self.state = new_state
        self.state_changed_at = time.time()

        # Record state change
        self.state_changes.append(
            {"timestamp": self.state_changed_at, "from": old_state, "to": new_state}
        )

        # Reset backoff when closing
        if new_state == CircuitState.CLOSED:
            self.current_backoff = 1.0
            self.consecutive_successes = 0

        logger.info(
            f"CircuitBreaker {self.name} state changed",
            from_state=old_state,
            to_state=new_state,
            failures_count=len(self.failures),
        )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.state != CircuitState.OPEN:
            return False

        time_since_change = time.time() - self.state_changed_at
        return time_since_change >= self.recovery_timeout_seconds

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        self.total_calls += 1

        # Check if circuit should transition to half-open
        if self._should_attempt_reset():
            self._change_state(CircuitState.HALF_OPEN)

        # Check circuit state
        if self.state == CircuitState.OPEN:
            self.rejected_calls += 1
            raise CircuitOpenError(
                f"Circuit breaker {self.name} is OPEN",
                recovery_timeout=self.recovery_timeout_seconds,
                time_since_failure=(
                    time.time() - self.last_failure_time
                    if self.last_failure_time
                    else 0
                ),
            )

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Record success
            self._on_success()
            return result

        except Exception as e:
            # Record failure
            self._on_failure(str(e), kwargs.get("endpoint"))
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        self.successful_calls += 1

        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes += 1

            if self.consecutive_successes >= self.success_threshold:
                # Circuit has recovered
                self._change_state(CircuitState.CLOSED)
                logger.info(
                    f"CircuitBreaker {self.name} recovered",
                    consecutive_successes=self.consecutive_successes,
                )

        # Clear old failures on success in closed state
        if self.state == CircuitState.CLOSED:
            self._clean_old_failures()

    def _on_failure(self, error: str, endpoint: str | None = None) -> None:
        """Handle failed call."""
        self.failed_calls += 1
        self.last_failure_time = time.time()

        # Record failure
        self.failures.append(
            FailureRecord(
                timestamp=self.last_failure_time, error=error, endpoint=endpoint
            )
        )

        # Clean old failures
        self._clean_old_failures()

        # Check if we should trip the circuit
        if self.state == CircuitState.HALF_OPEN:
            # Single failure in half-open trips the circuit
            self._change_state(CircuitState.OPEN)
            # Apply exponential backoff
            self.current_backoff = min(
                self.current_backoff * self.backoff_multiplier, self.max_backoff_seconds
            )
            self.recovery_timeout_seconds = int(self.current_backoff)

            logger.warning(
                f"CircuitBreaker {self.name} tripped again",
                backoff_seconds=self.recovery_timeout_seconds,
            )

        elif self.state == CircuitState.CLOSED:
            # Check if threshold is exceeded
            if len(self.failures) >= self.failure_threshold:
                self._change_state(CircuitState.OPEN)
                logger.warning(
                    f"CircuitBreaker {self.name} tripped",
                    failures_count=len(self.failures),
                    failure_threshold=self.failure_threshold,
                )

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state

    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._change_state(CircuitState.CLOSED)
        self.failures.clear()
        self.consecutive_successes = 0
        self.last_failure_time = None
        self.current_backoff = 1.0

        logger.info(f"CircuitBreaker {self.name} manually reset")

    def get_statistics(self) -> dict:
        """Get circuit breaker statistics."""
        self._clean_old_failures()

        success_rate = (
            (self.successful_calls / self.total_calls * 100)
            if self.total_calls > 0
            else 0
        )

        return {
            "name": self.name,
            "state": self.state,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": success_rate,
            "current_failures": len(self.failures),
            "failure_threshold": self.failure_threshold,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": self.last_failure_time,
            "state_changed_at": self.state_changed_at,
            "current_backoff": self.current_backoff,
            "state_changes": self.state_changes[-10:],  # Last 10 state changes
        }


class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, message: str, recovery_timeout: int, time_since_failure: float):
        """
        Initialize the exception.

        Args:
            message: Error message
            recovery_timeout: Seconds until recovery attempt
            time_since_failure: Seconds since last failure
        """
        super().__init__(message)
        self.recovery_timeout = recovery_timeout
        self.time_since_failure = time_since_failure


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""

    def __init__(self):
        """Initialize the manager."""
        self.breakers: dict[str, CircuitBreaker] = {}

        # Create default breakers
        self._setup_default_breakers()

    def _setup_default_breakers(self) -> None:
        """Set up default circuit breakers."""
        # API circuit breaker
        self.breakers["api"] = CircuitBreaker(
            name="api",
            failure_threshold=5,
            failure_window_seconds=30,
            recovery_timeout_seconds=60,
        )

        # WebSocket circuit breaker
        self.breakers["websocket"] = CircuitBreaker(
            name="websocket",
            failure_threshold=3,
            failure_window_seconds=20,
            recovery_timeout_seconds=30,
        )

        # Order execution circuit breaker
        self.breakers["orders"] = CircuitBreaker(
            name="orders",
            failure_threshold=3,
            failure_window_seconds=10,
            recovery_timeout_seconds=20,
        )

    def get_breaker(self, name: str) -> CircuitBreaker:
        """
        Get a circuit breaker by name.

        Args:
            name: Breaker name

        Returns:
            Circuit breaker instance
        """
        if name not in self.breakers:
            # Create a new breaker with default settings
            self.breakers[name] = CircuitBreaker(name=name)

        return self.breakers[name]

    def get_all_states(self) -> dict[str, str]:
        """Get states of all circuit breakers."""
        return {name: breaker.state for name, breaker in self.breakers.items()}

    def get_all_statistics(self) -> dict[str, dict]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_statistics() for name, breaker in self.breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()

        logger.info("All circuit breakers reset")

    def check_health(self) -> dict[str, bool]:
        """
        Check health of all circuit breakers.

        Returns:
            Dictionary mapping breaker names to health status
        """
        return {name: breaker.is_closed() for name, breaker in self.breakers.items()}
