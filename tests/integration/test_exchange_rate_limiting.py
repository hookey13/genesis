"""Integration tests for exchange rate limiting and circuit breaker."""

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from genesis.core.rate_limiter import RateLimiter, RateLimitConfig, Priority
from genesis.core.circuit_breaker import (
    CircuitBreaker, 
    CircuitBreakerConfig, 
    CircuitBreakerError,
    DegradationStrategy
)
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderRequest


class TestExchangeRateLimiting:
    """Test rate limiting integration with exchange gateway."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_normal_requests(self):
        """Test rate limiting for normal priority requests."""
        config = RateLimitConfig(
            requests_per_second=5,
            burst_size=10,
            window_size_seconds=60
        )
        limiter = RateLimiter(config)
        
        # Should allow burst of 10 requests
        for i in range(10):
            result = await limiter.acquire(tokens=1, wait=False)
            assert result is True
        
        # 11th request should be rejected
        result = await limiter.acquire(tokens=1, wait=False)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_critical_priority_bypass(self):
        """Test that critical priority can bypass normal limits."""
        config = RateLimitConfig(
            requests_per_second=5,
            burst_size=10,
            window_size_seconds=60,
            critical_reserve_percent=Decimal("0.2")
        )
        limiter = RateLimiter(config)
        
        # Use up normal capacity
        for i in range(10):
            await limiter.acquire(tokens=1, priority=Priority.NORMAL)
        
        # Normal request should fail
        result = await limiter.acquire(tokens=1, priority=Priority.NORMAL, wait=False)
        assert result is False
        
        # Critical request should still succeed (using reserve)
        result = await limiter.acquire(tokens=1, priority=Priority.CRITICAL, wait=False)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self):
        """Test adaptive rate limiting based on response headers."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        # Simulate high usage response
        headers = {
            "X-MBX-USED-WEIGHT-1M": "1000"  # 83% of 1200 limit
        }
        await limiter.update_from_headers(headers)
        
        # Refill rate should be reduced
        assert limiter.token_bucket.refill_rate < 10
        
        # Simulate low usage response
        headers = {
            "X-MBX-USED-WEIGHT-1M": "300"  # 25% of 1200 limit
        }
        await limiter.update_from_headers(headers)
        
        # Refill rate should increase
        assert limiter.token_bucket.refill_rate > 5
    
    @pytest.mark.asyncio
    async def test_request_coalescing(self):
        """Test request coalescing for identical requests."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        call_count = 0
        
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        # Make 5 identical requests simultaneously
        tasks = [
            limiter.coalesce_request("test_key", mock_api_call())
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Should only call API once
        assert call_count == 1
        
        # All should get same result
        assert all(r["result"] == "success" for r in results)


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with exchange gateway."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            time_window=10.0
        )
        breaker = CircuitBreaker("test", config)
        
        async def failing_call():
            raise Exception("API Error")
        
        # First 3 failures should go through
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_call)
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await breaker.call(failing_call)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery to half-open and closed."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            success_threshold=2,
            time_window=10.0
        )
        breaker = CircuitBreaker("test", config)
        
        # Open the circuit
        async def failing_call():
            raise Exception("API Error")
        
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_call)
        
        assert breaker.is_open
        
        # Wait for recovery timeout
        await asyncio.sleep(0.6)
        
        # Successful call should transition to half-open
        async def successful_call():
            return "success"
        
        result = await breaker.call(successful_call)
        assert result == "success"
        assert breaker.is_half_open
        
        # One more success should close the circuit
        result = await breaker.call(successful_call)
        assert result == "success"
        assert breaker.is_closed
    
    @pytest.mark.asyncio
    async def test_fallback_degradation(self):
        """Test fallback degradation strategy."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=60.0,
            degradation_strategy=DegradationStrategy.FALLBACK
        )
        breaker = CircuitBreaker("test", config)
        
        # Set fallback function
        async def fallback_func(*args, **kwargs):
            return {"fallback": True}
        
        breaker.set_fallback(fallback_func)
        
        # Open the circuit
        async def failing_call():
            raise Exception("API Error")
        
        with pytest.raises(Exception):
            await breaker.call(failing_call)
        
        assert breaker.is_open
        
        # Next call should use fallback
        result = await breaker.call(failing_call)
        assert result["fallback"] is True
    
    @pytest.mark.asyncio
    async def test_cache_degradation(self):
        """Test cache degradation strategy."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=60.0,
            degradation_strategy=DegradationStrategy.CACHE,
            cache_ttl=5.0
        )
        breaker = CircuitBreaker("test", config)
        
        # First successful call should be cached
        async def api_call(param):
            if breaker.is_closed:
                return {"result": param}
            raise Exception("API Error")
        
        # Make successful call
        result = await breaker.call(api_call, "test_value")
        assert result["result"] == "test_value"
        
        # Force circuit open
        async def failing_call():
            raise Exception("API Error")
        
        with pytest.raises(Exception):
            await breaker.call(failing_call)
        
        assert breaker.is_open
        
        # Same call should return cached result
        result = await breaker.call(api_call, "test_value")
        assert result["result"] == "test_value"


class TestGatewayIntegration:
    """Test complete integration with BinanceGateway."""
    
    @pytest.mark.asyncio
    async def test_gateway_rate_limiting(self):
        """Test gateway applies rate limiting correctly."""
        gateway = BinanceGateway(mock_mode=True)
        
        # Mock the exchange
        gateway.exchange = AsyncMock()
        gateway.exchange.create_order = AsyncMock(return_value={
            "id": "12345",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "status": "open",
            "price": 50000,
            "amount": 0.001,
            "timestamp": time.time() * 1000
        })
        
        # Create order request
        order = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
            client_order_id="test_order_1"
        )
        
        # Place order (should apply rate limiting)
        gateway.mock_mode = False  # Disable mock mode to test real flow
        result = await gateway.place_order(order)
        
        assert result.order_id == "12345"
        assert gateway.rate_limiter.metrics["requests_allowed"] > 0
    
    @pytest.mark.asyncio
    async def test_gateway_circuit_breaker(self):
        """Test gateway circuit breaker protects against failures."""
        gateway = BinanceGateway(mock_mode=False)
        
        # Mock exchange to fail
        gateway.exchange = AsyncMock()
        gateway.exchange.fetch_balance = AsyncMock(
            side_effect=Exception("Connection error")
        )
        
        # Configure circuit breaker for quick testing
        gateway.circuit_breaker.config.failure_threshold = 2
        gateway.circuit_breaker.config.recovery_timeout = 0.5
        
        # First 2 calls should fail normally
        for i in range(2):
            with pytest.raises(Exception):
                await gateway.get_account_balance()
        
        # Circuit should be open now, next call should fail fast
        with pytest.raises(CircuitBreakerError):
            await gateway.get_account_balance()
    
    @pytest.mark.asyncio
    async def test_emergency_close_priority(self):
        """Test emergency close uses CRITICAL priority."""
        gateway = BinanceGateway(mock_mode=False)
        
        # Mock exchange
        gateway.exchange = AsyncMock()
        gateway.exchange.fetch_balance = AsyncMock(return_value={
            "info": {
                "balances": {
                    "BTC": {"free": "0.1", "locked": "0"},
                    "ETH": {"free": "1.5", "locked": "0"},
                    "USDT": {"free": "1000", "locked": "0"}
                }
            }
        })
        
        gateway.exchange.create_order = AsyncMock(return_value={
            "id": "emergency_1",
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "market",
            "status": "filled",
            "amount": 0.1,
            "timestamp": time.time() * 1000
        })
        
        # Use up normal rate limit capacity
        for i in range(20):
            await gateway.rate_limiter.acquire(tokens=1, priority=Priority.NORMAL)
        
        # Emergency close should still work (using CRITICAL priority)
        orders = await gateway.emergency_close_all_positions()
        
        # Should have closed BTC and ETH positions
        assert len(orders) == 2
        assert any(o.symbol == "BTC/USDT" for o in orders)
        assert any(o.symbol == "ETH/USDT" for o in orders)


class TestBackpressureIntegration:
    """Test backpressure handling in the system."""
    
    @pytest.mark.asyncio
    async def test_event_bus_backpressure(self):
        """Test event bus applies backpressure correctly."""
        from genesis.engine.event_bus import EventBus
        from genesis.core.events import Event, EventType, EventPriority
        
        # Create event bus with small queue for testing
        bus = EventBus(max_queue_size=10)
        await bus.start()
        
        # Register backpressure callback
        backpressure_triggered = False
        
        def on_backpressure(active: bool):
            nonlocal backpressure_triggered
            backpressure_triggered = active
        
        bus.register_backpressure_callback(on_backpressure)
        bus.backpressure_threshold = 0.5  # Lower threshold for testing
        
        # Publish events to trigger backpressure
        for i in range(25):
            event = Event(
                event_type=EventType.ORDER_PLACED,
                aggregate_id=f"order_{i}",
                data={"order_id": i}
            )
            await bus.publish(event, EventPriority.NORMAL)
        
        # Backpressure should be triggered
        assert backpressure_triggered
        
        # Get metrics
        metrics = bus.get_backpressure_metrics()
        assert metrics["backpressure_active"] is True
        assert metrics["queue_utilization_percent"] > 50
        
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_backpressure_event_shedding(self):
        """Test low priority events are shed under backpressure."""
        from genesis.engine.event_bus import EventBus
        from genesis.core.events import Event, EventType, EventPriority
        
        bus = EventBus(max_queue_size=5)
        bus.shed_low_priority = True
        bus.backpressure_threshold = 0.5
        await bus.start()
        
        # Fill queue with normal priority events
        for i in range(10):
            event = Event(
                event_type=EventType.MARKET_DATA,
                aggregate_id=f"data_{i}",
                data={"price": i}
            )
            await bus.publish(event, EventPriority.NORMAL)
        
        # Low priority event should be shed
        initial_dropped = bus.events_dropped
        
        low_event = Event(
            event_type=EventType.MARKET_DATA,
            aggregate_id="low_priority",
            data={"price": 999}
        )
        await bus.publish(low_event, EventPriority.LOW)
        
        # Should have dropped the low priority event
        assert bus.events_dropped > initial_dropped
        
        await bus.stop()