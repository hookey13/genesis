"""
Integration tests for circuit breaker pattern implementation.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    get_circuit_breaker_registry,
)
from genesis.core.exceptions import NetworkError, ValidationError


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2,
            time_window=5.0,
        )
    
    @pytest.fixture
    def breaker(self, config):
        """Create circuit breaker instance."""
        return CircuitBreaker("test_service", config)
    
    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, breaker):
        """Test circuit breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
        assert not breaker.is_half_open
    
    @pytest.mark.asyncio
    async def test_successful_calls_in_closed_state(self, breaker):
        """Test successful calls pass through in closed state."""
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_sync_function_support(self, breaker):
        """Test circuit breaker works with sync functions."""
        def sync_func(value):
            return value * 2
        
        result = await breaker.call(sync_func, 5)
        assert result == 10
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failure_counting(self, breaker):
        """Test failures are counted correctly."""
        async def failing_func():
            raise NetworkError("Connection failed")
        
        # First two failures shouldn't open circuit
        for i in range(2):
            with pytest.raises(NetworkError):
                await breaker.call(failing_func)
            assert breaker.state == CircuitState.CLOSED
        
        # Third failure should open circuit
        with pytest.raises(NetworkError):
            await breaker.call(failing_func)
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_open_blocks_calls(self, breaker):
        """Test open circuit blocks calls."""
        async def failing_func():
            raise NetworkError("Connection failed")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(NetworkError):
                await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Subsequent calls should be blocked
        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.call(failing_func)
        
        assert exc_info.value.service == "test_service"
        assert exc_info.value.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_transition_after_timeout(self, breaker):
        """Test circuit transitions to half-open after recovery timeout."""
        async def failing_func():
            raise NetworkError("Connection failed")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(NetworkError):
                await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Mock time passage
        with patch.object(breaker, '_should_attempt_reset', return_value=True):
            # Next call should transition to half-open
            async def success_func():
                return "recovered"
            
            result = await breaker.call(success_func)
            assert result == "recovered"
            # Should be half-open after first success
            assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_half_open_to_closed_transition(self, breaker):
        """Test circuit closes after enough successes in half-open state."""
        # Force to half-open state
        breaker._state = CircuitState.HALF_OPEN
        
        async def success_func():
            return "success"
        
        # First success keeps it half-open
        await breaker.call(success_func)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Second success should close it
        await breaker.call(success_func)
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, breaker):
        """Test failure in half-open state reopens circuit."""
        # Force to half-open state
        breaker._state = CircuitState.HALF_OPEN
        
        async def failing_func():
            raise NetworkError("Still failing")
        
        with pytest.raises(NetworkError):
            await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_excluded_exceptions_not_counted(self, breaker):
        """Test excluded exceptions don't trigger circuit."""
        breaker.config.excluded_exceptions = (ValidationError,)
        
        async def validation_error_func():
            raise ValidationError("Invalid input")
        
        # These shouldn't count as failures
        for _ in range(5):
            with pytest.raises(ValidationError):
                await breaker.call(validation_error_func)
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_time_window_failure_tracking(self, breaker):
        """Test failures outside time window are not counted."""
        async def failing_func():
            raise NetworkError("Connection failed")
        
        # Record two failures
        for _ in range(2):
            with pytest.raises(NetworkError):
                await breaker.call(failing_func)
        
        # Mock time passage beyond window
        old_failures = breaker._recent_failures
        breaker._recent_failures = [
            datetime.utcnow() - timedelta(seconds=10)
            for _ in old_failures
        ]
        
        # Clean old failures
        breaker._clean_old_failures()
        
        # Now a single failure shouldn't open circuit
        with pytest.raises(NetworkError):
            await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_manual_reset(self, breaker):
        """Test manual reset of circuit breaker."""
        # Open the circuit
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 5
        
        await breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
    
    @pytest.mark.asyncio
    async def test_manual_trip(self, breaker):
        """Test manual trip of circuit breaker."""
        assert breaker.state == CircuitState.CLOSED
        
        await breaker.trip()
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_get_status(self, breaker):
        """Test getting circuit breaker status."""
        status = breaker.get_status()
        
        assert status["name"] == "test_service"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["config"]["failure_threshold"] == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_calls_thread_safety(self, breaker):
        """Test circuit breaker is thread-safe with concurrent calls."""
        call_count = 0
        
        async def counted_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise NetworkError("Failing")
            return "success"
        
        # Simulate concurrent calls
        tasks = []
        for _ in range(10):
            tasks.append(asyncio.create_task(
                breaker.call(counted_func)
            ))
        
        # Some will fail, some will be blocked
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Circuit should be open after threshold
        assert breaker.state == CircuitState.OPEN
        
        # Check we have both NetworkErrors and CircuitBreakerErrors
        network_errors = sum(1 for r in results if isinstance(r, NetworkError))
        circuit_errors = sum(1 for r in results if isinstance(r, CircuitBreakerError))
        
        assert network_errors >= 3  # At least threshold failures
        assert circuit_errors > 0  # Some calls were blocked


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        return CircuitBreakerRegistry()
    
    def test_get_or_create_new_breaker(self, registry):
        """Test creating new circuit breaker."""
        breaker = registry.get_or_create("service1")
        
        assert breaker is not None
        assert breaker.name == "service1"
        assert "service1" in registry._breakers
    
    def test_get_or_create_existing_breaker(self, registry):
        """Test getting existing circuit breaker."""
        breaker1 = registry.get_or_create("service1")
        breaker2 = registry.get_or_create("service1")
        
        assert breaker1 is breaker2
    
    def test_get_breaker(self, registry):
        """Test getting breaker by name."""
        registry.get_or_create("service1")
        
        breaker = registry.get("service1")
        assert breaker is not None
        assert breaker.name == "service1"
        
        none_breaker = registry.get("nonexistent")
        assert none_breaker is None
    
    def test_remove_breaker(self, registry):
        """Test removing breaker from registry."""
        registry.get_or_create("service1")
        
        removed = registry.remove("service1")
        assert removed is True
        assert registry.get("service1") is None
        
        removed_again = registry.remove("service1")
        assert removed_again is False
    
    def test_get_all_status(self, registry):
        """Test getting status of all breakers."""
        registry.get_or_create("service1")
        registry.get_or_create("service2")
        
        status = registry.get_all_status()
        
        assert len(status) == 2
        assert "service1" in status
        assert "service2" in status
        assert status["service1"]["state"] == "closed"
    
    @pytest.mark.asyncio
    async def test_reset_all(self, registry):
        """Test resetting all circuit breakers."""
        breaker1 = registry.get_or_create("service1")
        breaker2 = registry.get_or_create("service2")
        
        # Open both circuits
        breaker1._state = CircuitState.OPEN
        breaker2._state = CircuitState.OPEN
        
        await registry.reset_all()
        
        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED
    
    def test_get_open_circuits(self, registry):
        """Test getting list of open circuits."""
        breaker1 = registry.get_or_create("service1")
        breaker2 = registry.get_or_create("service2")
        breaker3 = registry.get_or_create("service3")
        
        breaker1._state = CircuitState.OPEN
        breaker3._state = CircuitState.OPEN
        
        open_circuits = registry.get_open_circuits()
        
        assert len(open_circuits) == 2
        assert "service1" in open_circuits
        assert "service3" in open_circuits
        assert "service2" not in open_circuits
    
    def test_get_statistics(self, registry):
        """Test getting registry statistics."""
        breaker1 = registry.get_or_create("service1")
        breaker2 = registry.get_or_create("service2")
        breaker3 = registry.get_or_create("service3")
        
        breaker1._state = CircuitState.OPEN
        breaker2._state = CircuitState.HALF_OPEN
        
        stats = registry.get_statistics()
        
        assert stats["total_breakers"] == 3
        assert stats["open"] == 1
        assert stats["half_open"] == 1
        assert stats["closed"] == 1
        assert stats["health_percentage"] == pytest.approx(33.33, rel=0.01)


class TestGlobalRegistry:
    """Test global registry singleton."""
    
    def test_get_circuit_breaker_registry_singleton(self):
        """Test global registry returns singleton."""
        registry1 = get_circuit_breaker_registry()
        registry2 = get_circuit_breaker_registry()
        
        assert registry1 is registry2
    
    def test_singleton_persists_state(self):
        """Test singleton maintains state."""
        registry1 = get_circuit_breaker_registry()
        registry1.get_or_create("global_service")
        
        registry2 = get_circuit_breaker_registry()
        breaker = registry2.get("global_service")
        
        assert breaker is not None
        assert breaker.name == "global_service"