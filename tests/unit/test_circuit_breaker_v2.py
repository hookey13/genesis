"""
Unit tests for Enhanced Circuit Breaker V2.

Tests circuit breaker states, transitions, trip conditions, and recovery mechanisms.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus
from genesis.exchange.circuit_breaker_v2 import (
    CircuitState,
    EnhancedCircuitBreaker,
    ErrorRecord,
)


@pytest.fixture
async def event_bus():
    """Create and start an event bus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


class TestEnhancedCircuitBreaker:
    """Test enhanced circuit breaker functionality."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        breaker = EnhancedCircuitBreaker(
            name="test",
            error_rate_threshold=0.5,
            error_window_seconds=60,
            ws_disconnect_threshold=30,
            clock_skew_threshold=5000,
            recovery_timeout=60,
        )

        assert breaker.name == "test"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.error_rate_threshold == 0.5
        assert breaker.ws_disconnect_threshold == 30
        assert breaker.clock_skew_threshold == 5000
        assert breaker.recovery_timeout == 60

    def test_error_rate_calculation(self):
        """Test error rate calculation with sliding window."""
        breaker = EnhancedCircuitBreaker(
            error_window_seconds=60,
            error_rate_threshold=0.5,
        )

        # Add some errors and successes
        current_time = time.time()

        # Add 5xx errors
        for i in range(3):
            breaker.error_records.append(
                ErrorRecord(timestamp=current_time - i, status_code=500)
            )

        # Add non-5xx error (should not count)
        breaker.error_records.append(
            ErrorRecord(timestamp=current_time - 10, status_code=400)
        )

        # Add successes
        for i in range(3):
            breaker.success_records.append(current_time - i)

        # Calculate error rate (3 5xx errors out of 7 total requests)
        error_rate = breaker._calculate_error_rate()
        assert error_rate == pytest.approx(3 / 7, rel=0.01)

    def test_trip_condition_5xx_error_rate(self):
        """Test circuit trips on high 5xx error rate (>50% in 1 minute)."""
        breaker = EnhancedCircuitBreaker(
            error_rate_threshold=0.5,
            error_window_seconds=60,
        )

        current_time = time.time()

        # Add 6 5xx errors
        for i in range(6):
            breaker.error_records.append(
                ErrorRecord(timestamp=current_time - i, status_code=503)
            )

        # Add 4 successes (60% error rate)
        for i in range(4):
            breaker.success_records.append(current_time - i)

        # Check trip conditions
        should_trip, reason = breaker._check_trip_conditions()
        assert should_trip is True
        assert "5xx error rate" in reason
        assert "60.0%" in reason or "0.6" in reason

    def test_trip_condition_websocket_disconnect(self):
        """Test circuit trips on WebSocket disconnect >30 seconds."""
        breaker = EnhancedCircuitBreaker(
            ws_disconnect_threshold=30,
        )

        # Simulate WebSocket disconnection
        breaker.ws_connected = False
        breaker.ws_disconnected_at = time.time() - 31  # 31 seconds ago

        # Check trip conditions
        should_trip, reason = breaker._check_trip_conditions()
        assert should_trip is True
        assert "WebSocket disconnected" in reason

    def test_trip_condition_clock_skew(self):
        """Test circuit trips on clock skew >5 seconds."""
        breaker = EnhancedCircuitBreaker(
            clock_skew_threshold=5000,  # 5 seconds in ms
        )

        # Set clock skew
        breaker.last_clock_skew_ms = 5001  # Just over threshold

        # Check trip conditions
        should_trip, reason = breaker._check_trip_conditions()
        assert should_trip is True
        assert "Clock skew" in reason
        assert "5001ms" in reason

    def test_state_transitions(self):
        """Test state transitions between CLOSED, OPEN, and HALF_OPEN."""
        breaker = EnhancedCircuitBreaker(
            recovery_timeout=1,  # Short timeout for testing
            half_open_success_threshold=3,
        )

        # Start in CLOSED state
        assert breaker.state == CircuitState.CLOSED

        # Transition to OPEN
        breaker._transition_to(CircuitState.OPEN, "Test trip")
        assert breaker.state == CircuitState.OPEN
        assert breaker.trip_count == 1

        # Check is_open
        assert breaker.is_open() is True

        # Wait for recovery timeout
        time.sleep(1.1)

        # Should transition to HALF_OPEN on next check
        assert breaker.is_open() is False
        assert breaker.state == CircuitState.HALF_OPEN

        # Record successes to close circuit
        for _ in range(3):
            breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_half_open_recovery(self):
        """Test recovery from HALF_OPEN state with successful requests."""
        breaker = EnhancedCircuitBreaker(
            half_open_success_threshold=3,
        )

        # Start in HALF_OPEN state
        breaker.state = CircuitState.HALF_OPEN
        breaker.half_open_successes = 0

        # Record successes
        breaker.record_success()
        assert breaker.half_open_successes == 1
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.half_open_successes == 2
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.half_open_successes == 3
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure(self):
        """Test failure during HALF_OPEN state returns to OPEN."""
        breaker = EnhancedCircuitBreaker()

        # Start in HALF_OPEN state
        breaker.state = CircuitState.HALF_OPEN
        breaker.half_open_successes = 2

        # Record error
        breaker.record_error(Exception("Test error"), status_code=500)

        # Should return to OPEN
        assert breaker.state == CircuitState.OPEN
        assert breaker.half_open_successes == 0

    def test_websocket_status_updates(self):
        """Test WebSocket status updates trigger circuit checks."""
        breaker = EnhancedCircuitBreaker(
            ws_disconnect_threshold=30,
        )

        # Disconnect WebSocket
        breaker.update_ws_status(False)
        assert breaker.ws_connected is False
        assert breaker.ws_disconnected_at is not None

        # Wait for threshold
        time.sleep(0.1)
        breaker.ws_disconnected_at = time.time() - 31

        # Update status should check conditions
        breaker.update_ws_status(False)
        assert breaker.state == CircuitState.OPEN

    def test_clock_skew_updates(self):
        """Test clock skew updates trigger circuit checks."""
        breaker = EnhancedCircuitBreaker(
            clock_skew_threshold=5000,
        )

        # Update with high skew
        breaker.update_clock_skew(6000)

        # Should trip circuit
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_decorator_protection(self):
        """Test circuit breaker decorator protects function calls."""
        breaker = EnhancedCircuitBreaker()

        # Create protected function
        call_count = 0

        @breaker.call
        async def protected_function():
            nonlocal call_count
            call_count += 1
            return "success"

        # Should work when closed
        result = await protected_function()
        assert result == "success"
        assert call_count == 1

        # Open circuit
        breaker.state = CircuitState.OPEN
        breaker.state_changed_at = time.time()

        # Should block when open
        with pytest.raises(Exception) as exc_info:
            await protected_function()
        assert "Circuit breaker exchange is OPEN" in str(exc_info.value)
        assert call_count == 1  # Not called
        assert breaker.blocked_requests == 1

    @pytest.mark.asyncio
    async def test_decorator_error_handling(self):
        """Test decorator handles errors and updates circuit state."""
        breaker = EnhancedCircuitBreaker(
            error_rate_threshold=0.5,
        )

        @breaker.call
        async def failing_function():
            error = Exception("API Error")
            error.status_code = 500
            raise error

        # Call should fail and record error
        with pytest.raises(Exception):
            await failing_function()

        assert len(breaker.error_records) == 1
        assert breaker.error_records[0].status_code == 500

    def test_statistics(self):
        """Test circuit breaker statistics."""
        breaker = EnhancedCircuitBreaker()

        # Add some data
        breaker.record_success()
        breaker.record_error(Exception("Test"), status_code=500)
        breaker.total_requests = 10
        breaker.blocked_requests = 2
        breaker.trip_count = 1

        stats = breaker.get_stats()

        assert stats["name"] == "exchange"
        assert stats["state"] == "closed"
        assert stats["total_requests"] == 10
        assert stats["blocked_requests"] == 2
        assert stats["trip_count"] == 1
        assert "error_rate" in stats
        assert "state_duration" in stats

    def test_reset(self):
        """Test circuit breaker reset."""
        breaker = EnhancedCircuitBreaker()

        # Add some state
        breaker.state = CircuitState.OPEN
        breaker.record_error(Exception("Test"), status_code=500)
        breaker.record_success()

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert len(breaker.error_records) == 0
        assert len(breaker.success_records) == 0
        assert breaker.half_open_successes == 0

    @pytest.mark.asyncio
    async def test_event_bus_integration(self, event_bus):
        """Test circuit breaker publishes events to event bus."""
        breaker = EnhancedCircuitBreaker(event_bus=event_bus)

        # Track events
        captured_events = []

        async def event_handler(event: Event):
            captured_events.append(event)

        # Subscribe to circuit breaker events
        event_bus.subscribe(
            event_handler,
            {EventType.CIRCUIT_BREAKER_OPEN, EventType.CIRCUIT_BREAKER_CLOSED},
        )

        # Trigger state change
        breaker._transition_to(CircuitState.OPEN, "Test trip")

        # Allow event to propagate through the event bus
        await asyncio.sleep(0.5)

        # Verify event was published
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == EventType.CIRCUIT_BREAKER_OPEN
        assert event.aggregate_id == "exchange"
        assert event.event_data["state"] == "open"
        assert event.event_data["reason"] == "Test trip"

    def test_sliding_window_cleanup(self):
        """Test old records are cleaned from sliding window."""
        breaker = EnhancedCircuitBreaker(
            error_window_seconds=60,
        )

        current_time = time.time()

        # Add old records (outside window)
        for i in range(5):
            breaker.error_records.append(
                ErrorRecord(timestamp=current_time - 70 - i, status_code=500)
            )
            breaker.success_records.append(current_time - 70 - i)

        # Add recent records (inside window)
        for i in range(3):
            breaker.error_records.append(
                ErrorRecord(timestamp=current_time - i, status_code=500)
            )
            breaker.success_records.append(current_time - i)

        # Clean old records
        breaker._clean_old_records()

        # Only recent records should remain
        assert len(breaker.error_records) == 3
        assert len(breaker.success_records) == 3

    def test_recovery_timeout_60s(self):
        """Test that recovery timeout is 60 seconds after opening."""
        breaker = EnhancedCircuitBreaker(
            recovery_timeout=60,  # 60 seconds as per requirements
        )

        # Open circuit
        breaker._transition_to(CircuitState.OPEN, "Test")

        # Should be open immediately
        assert breaker.is_open() is True

        # Mock time to simulate passage
        with patch("time.time") as mock_time:
            # Set current time to 59 seconds after opening
            mock_time.return_value = breaker.state_changed_at + 59
            assert breaker.is_open() is True  # Still open

            # Set current time to 61 seconds after opening
            mock_time.return_value = breaker.state_changed_at + 61
            assert breaker.is_open() is False  # Should transition to half-open
            assert breaker.state == CircuitState.HALF_OPEN
