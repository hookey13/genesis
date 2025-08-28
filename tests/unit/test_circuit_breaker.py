"""
Unit tests for CircuitBreaker.
"""

import asyncio
import time

import pytest

from genesis.exchange.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitState,
    FailureRecord,
)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=5,
            failure_window_seconds=30,
            recovery_timeout_seconds=60,
        )

        assert breaker.name == "test"
        assert breaker.failure_threshold == 5
        assert breaker.failure_window_seconds == 30
        assert breaker.recovery_timeout_seconds == 60
        assert breaker.state == CircuitState.CLOSED
        assert len(breaker.failures) == 0

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker."""

        async def successful_func():
            return "success"

        result = await circuit_breaker.call(successful_func)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.successful_calls == 1
        assert circuit_breaker.failed_calls == 0

    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker):
        """Test failed function call through circuit breaker."""

        async def failing_func():
            raise Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.failed_calls == 1
        assert len(circuit_breaker.failures) == 1
        assert (
            circuit_breaker.state == CircuitState.CLOSED
        )  # Still closed (threshold not reached)

    @pytest.mark.asyncio
    async def test_circuit_trip(self, circuit_breaker):
        """Test circuit tripping after threshold failures."""

        async def failing_func():
            raise Exception("Test error")

        # Fail 3 times (threshold)
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        # Circuit should be open now
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failed_calls == 3

        # Next call should be rejected
        try:
            await circuit_breaker.call(failing_func)
            assert False, "Expected CircuitOpenError to be raised"
        except Exception as e:
            # Handle case where clean_imports fixture causes exception class mismatch
            if type(
                e
            ).__name__ == "CircuitOpenError" and "Circuit breaker test is OPEN" in str(
                e
            ):
                assert circuit_breaker.rejected_calls == 1
            else:
                raise AssertionError(
                    f"Expected CircuitOpenError but got {type(e).__name__}: {e}"
                )

    @pytest.mark.asyncio
    async def test_half_open_state(self, circuit_breaker):
        """Test half-open state transition."""
        circuit_breaker.recovery_timeout_seconds = 0.1  # Short timeout for testing

        async def failing_func():
            raise Exception("Test error")

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Next call should transition to half-open
        async def successful_func():
            return "success"

        result = await circuit_breaker.call(successful_func)
        assert result == "success"

        # After one success in half-open, need more for full recovery
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Another success should close the circuit
        result = await circuit_breaker.call(successful_func)
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure(self, circuit_breaker):
        """Test failure in half-open state."""
        circuit_breaker.recovery_timeout_seconds = 0.1  # Short timeout for testing

        async def failing_func():
            raise Exception("Test error")

        # Trip the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Fail in half-open state
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_func)

        # Should be open again with exponential backoff
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.current_backoff == 2.0  # Exponential backoff

    def test_clean_old_failures(self, circuit_breaker):
        """Test cleaning of old failure records."""
        current_time = time.time()

        # Add old failure (outside window)
        circuit_breaker.failures.append(
            FailureRecord(
                timestamp=current_time - 20, error="Old error"  # Outside 10s window
            )
        )

        # Add recent failure
        circuit_breaker.failures.append(
            FailureRecord(
                timestamp=current_time - 5, error="Recent error"  # Inside window
            )
        )

        circuit_breaker._clean_old_failures()

        assert len(circuit_breaker.failures) == 1
        assert circuit_breaker.failures[0].error == "Recent error"

    def test_manual_reset(self, circuit_breaker):
        """Test manual circuit reset."""
        # Set circuit to open state
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.failures = [FailureRecord(timestamp=time.time(), error="Error")]
        circuit_breaker.consecutive_successes = 1
        circuit_breaker.current_backoff = 4.0

        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert len(circuit_breaker.failures) == 0
        assert circuit_breaker.consecutive_successes == 0
        assert circuit_breaker.current_backoff == 1.0

    def test_get_statistics(self, circuit_breaker):
        """Test statistics retrieval."""
        circuit_breaker.total_calls = 100
        circuit_breaker.successful_calls = 95
        circuit_breaker.failed_calls = 5
        circuit_breaker.rejected_calls = 10

        stats = circuit_breaker.get_statistics()

        assert stats["name"] == "test"
        assert stats["state"] == CircuitState.CLOSED
        assert stats["total_calls"] == 100
        assert stats["successful_calls"] == 95
        assert stats["failed_calls"] == 5
        assert stats["rejected_calls"] == 10
        assert stats["success_rate"] == 95.0

    @pytest.mark.asyncio
    async def test_sync_function_call(self, circuit_breaker):
        """Test calling synchronous function through circuit breaker."""

        def sync_func():
            return "sync_result"

        result = await circuit_breaker.call(sync_func)
        assert result == "sync_result"


class TestCircuitBreakerManager:
    """Test suite for CircuitBreakerManager."""

    def test_initialization(self):
        """Test manager initialization with default breakers."""
        manager = CircuitBreakerManager()

        assert "api" in manager.breakers
        assert "websocket" in manager.breakers
        assert "orders" in manager.breakers

    def test_get_breaker(self):
        """Test getting circuit breaker by name."""
        manager = CircuitBreakerManager()

        # Get existing breaker
        api_breaker = manager.get_breaker("api")
        assert api_breaker.name == "api"
        assert api_breaker.failure_threshold == 5

        # Get non-existing breaker (creates new one)
        new_breaker = manager.get_breaker("custom")
        assert new_breaker.name == "custom"
        assert "custom" in manager.breakers

    def test_get_all_states(self):
        """Test getting all breaker states."""
        manager = CircuitBreakerManager()

        states = manager.get_all_states()

        assert states["api"] == CircuitState.CLOSED
        assert states["websocket"] == CircuitState.CLOSED
        assert states["orders"] == CircuitState.CLOSED

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        manager = CircuitBreakerManager()

        # Open one breaker
        manager.breakers["api"].state = CircuitState.OPEN

        manager.reset_all()

        # All should be closed
        states = manager.get_all_states()
        assert all(state == CircuitState.CLOSED for state in states.values())

    def test_check_health(self):
        """Test health check."""
        manager = CircuitBreakerManager()

        # All closed = healthy
        health = manager.check_health()
        assert all(health.values())

        # Open one breaker
        manager.breakers["api"].state = CircuitState.OPEN

        health = manager.check_health()
        assert health["api"] is False
        assert health["websocket"] is True
        assert health["orders"] is True
