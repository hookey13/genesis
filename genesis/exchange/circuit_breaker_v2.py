"""
Enhanced circuit breaker matching specification requirements.

Trip conditions:
- 5xx error rate > 50% in last minute
- WebSocket disconnect > 30 seconds
- Clock skew > 5 seconds
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorRecord:
    """Record of an error event."""

    timestamp: float
    status_code: int | None = None
    error_type: str | None = None
    endpoint: str | None = None


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with specification-compliant trip conditions.

    Trip conditions:
    1. 5xx error rate > 50% in sliding window (1 minute)
    2. WebSocket disconnection > 30 seconds
    3. Clock skew > 5 seconds
    """

    def __init__(
        self,
        name: str = "exchange",
        error_rate_threshold: float = 0.5,  # 50% error rate
        error_window_seconds: int = 60,  # 1 minute window
        ws_disconnect_threshold: int = 30,  # 30 seconds
        clock_skew_threshold: int = 5000,  # 5 seconds in ms
        recovery_timeout: int = 30,  # 30 seconds cooldown
        half_open_success_threshold: int = 3,
        event_bus: EventBus | None = None,
    ):
        """Initialize enhanced circuit breaker."""
        self.name = name
        self.error_rate_threshold = error_rate_threshold
        self.error_window_seconds = error_window_seconds
        self.ws_disconnect_threshold = ws_disconnect_threshold
        self.clock_skew_threshold = clock_skew_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success_threshold = half_open_success_threshold
        self.event_bus = event_bus or EventBus()

        # State tracking
        self.state = CircuitState.CLOSED
        self.state_changed_at = time.time()

        # Error tracking (sliding window)
        self.error_records: deque[ErrorRecord] = deque()
        self.success_records: deque[float] = deque()

        # WebSocket tracking
        self.ws_disconnected_at: float | None = None
        self.ws_connected = True

        # Clock skew tracking
        self.last_clock_skew_ms = 0

        # Half-open state tracking
        self.half_open_successes = 0

        # Statistics
        self.total_requests = 0
        self.blocked_requests = 0
        self.trip_count = 0

        logger.info(
            f"Enhanced CircuitBreaker {name} initialized",
            error_threshold=error_rate_threshold,
            ws_threshold=ws_disconnect_threshold,
            clock_threshold=clock_skew_threshold,
        )

    def _clean_old_records(self):
        """Remove records outside sliding window."""
        current_time = time.time()
        cutoff_time = current_time - self.error_window_seconds

        # Clean error records
        while self.error_records and self.error_records[0].timestamp < cutoff_time:
            self.error_records.popleft()

        # Clean success records
        while self.success_records and self.success_records[0] < cutoff_time:
            self.success_records.popleft()

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate in sliding window."""
        self._clean_old_records()

        total = len(self.error_records) + len(self.success_records)
        if total == 0:
            return 0.0

        # Count 5xx errors specifically
        error_count = sum(
            1
            for record in self.error_records
            if record.status_code and 500 <= record.status_code < 600
        )

        return error_count / total

    def _check_trip_conditions(self) -> tuple[bool, str]:
        """
        Check if circuit should trip.

        Returns:
            Tuple of (should_trip, reason)
        """
        # Condition 1: 5xx error rate
        error_rate = self._calculate_error_rate()
        if error_rate > self.error_rate_threshold:
            return (
                True,
                f"5xx error rate {error_rate:.1%} exceeds {self.error_rate_threshold:.1%}",
            )

        # Condition 2: WebSocket disconnection duration
        if not self.ws_connected and self.ws_disconnected_at:
            disconnect_duration = time.time() - self.ws_disconnected_at
            if disconnect_duration > self.ws_disconnect_threshold:
                return True, f"WebSocket disconnected for {disconnect_duration:.1f}s"

        # Condition 3: Clock skew
        if self.last_clock_skew_ms > self.clock_skew_threshold:
            return (
                True,
                f"Clock skew {self.last_clock_skew_ms}ms exceeds {self.clock_skew_threshold}ms",
            )

        return False, ""

    def _transition_to(self, new_state: CircuitState, reason: str = ""):
        """Transition to new state and emit event."""
        old_state = self.state
        self.state = new_state
        self.state_changed_at = time.time()

        if new_state == CircuitState.OPEN:
            self.trip_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.half_open_successes = 0

        logger.info(
            f"Circuit breaker {self.name}: {old_state} -> {new_state}", reason=reason
        )

        # Emit event
        if self.event_bus:
            event_type = (
                EventType.CIRCUIT_BREAKER_OPEN
                if new_state == CircuitState.OPEN
                else EventType.CIRCUIT_BREAKER_CLOSED
            )
            event = Event(
                event_type=event_type,
                aggregate_id=self.name,
                event_data={
                    "state": new_state.value,
                    "reason": reason,
                    "error_count": len(self.error_records),
                    "error_rate": self._calculate_error_rate(),
                    "cooldown_seconds": (
                        self.recovery_timeout if new_state == CircuitState.OPEN else None
                    ),
                },
            )
            # Only create task if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.event_bus.publish(
                        event,
                        priority=EventPriority.CRITICAL if new_state == CircuitState.OPEN else EventPriority.HIGH,
                    )
                )
            except RuntimeError:
                # No running loop (e.g., in synchronous tests)
                # Event publishing is best-effort, continue without it
                pass

    def record_success(self, endpoint: str | None = None):
        """Record successful request."""
        self.success_records.append(time.time())
        self.total_requests += 1

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_success_threshold:
                self._transition_to(CircuitState.CLOSED, "Recovery confirmed")

    def record_error(
        self,
        error: Exception,
        status_code: int | None = None,
        endpoint: str | None = None,
    ):
        """Record error and check if circuit should trip."""
        self.error_records.append(
            ErrorRecord(
                timestamp=time.time(),
                status_code=status_code,
                error_type=type(error).__name__,
                endpoint=endpoint,
            )
        )
        self.total_requests += 1

        # Reset half-open successes on error
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes = 0
            self._transition_to(CircuitState.OPEN, "Error during recovery")

        # Check trip conditions if closed
        elif self.state == CircuitState.CLOSED:
            should_trip, reason = self._check_trip_conditions()
            if should_trip:
                self._transition_to(CircuitState.OPEN, reason)

    def update_ws_status(self, connected: bool):
        """Update WebSocket connection status."""
        self.ws_connected = connected

        if connected:
            self.ws_disconnected_at = None
        else:
            if self.ws_disconnected_at is None:
                self.ws_disconnected_at = time.time()

        # Check trip conditions
        if self.state == CircuitState.CLOSED:
            should_trip, reason = self._check_trip_conditions()
            if should_trip:
                self._transition_to(CircuitState.OPEN, reason)

    def update_clock_skew(self, skew_ms: int):
        """Update clock skew measurement."""
        self.last_clock_skew_ms = abs(skew_ms)

        # Check trip conditions
        if self.state == CircuitState.CLOSED:
            should_trip, reason = self._check_trip_conditions()
            if should_trip:
                self._transition_to(CircuitState.OPEN, reason)

    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.state_changed_at > self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN, "Recovery timeout reached")
                return False  # Allow test request
            return True

        # Half-open allows requests
        return False

    def call(self, func: Callable) -> Callable:
        """
        Decorator for protecting function calls.

        Usage:
            @circuit_breaker.call
            async def protected_function():
                ...
        """

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if circuit is open
            if self.is_open():
                self.blocked_requests += 1
                raise Exception(f"Circuit breaker {self.name} is OPEN")

            try:
                # Execute function
                result = await func(*args, **kwargs)
                self.record_success()
                return result

            except Exception as e:
                # Extract status code if available
                status_code = None
                if hasattr(e, "response") and hasattr(e.response, "status"):
                    status_code = e.response.status
                elif hasattr(e, "status_code"):
                    status_code = e.status_code

                self.record_error(e, status_code)
                raise

        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        self._clean_old_records()

        return {
            "name": self.name,
            "state": self.state.value,
            "state_duration": time.time() - self.state_changed_at,
            "error_rate": self._calculate_error_rate(),
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "trip_count": self.trip_count,
            "ws_connected": self.ws_connected,
            "clock_skew_ms": self.last_clock_skew_ms,
            "errors_in_window": len(self.error_records),
            "successes_in_window": len(self.success_records),
        }

    def reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.state_changed_at = time.time()
        self.error_records.clear()
        self.success_records.clear()
        self.half_open_successes = 0
        logger.info(f"Circuit breaker {self.name} reset")
