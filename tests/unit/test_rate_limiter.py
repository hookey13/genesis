"""Unit tests for rate limiter implementation."""

import asyncio
import time
from decimal import Decimal
import pytest
from unittest.mock import Mock, AsyncMock, patch

from genesis.core.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    SlidingWindowLimiter,
    Priority,
    DistributedRateLimiter
)


class TestRateLimitConfig:
    """Test rate limit configuration."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20,
            window_size_seconds=60,
            critical_reserve_percent=Decimal("0.2")
        )
        assert config.requests_per_second == 10
        assert config.burst_size == 20
        assert config.window_size_seconds == 60
        assert config.critical_reserve_percent == Decimal("0.2")
    
    def test_invalid_requests_per_second(self):
        """Test invalid requests per second."""
        with pytest.raises(ValueError, match="requests_per_second must be positive"):
            RateLimitConfig(
                requests_per_second=0,
                burst_size=20
            )
    
    def test_invalid_burst_size(self):
        """Test burst size less than requests per second."""
        with pytest.raises(ValueError, match="burst_size must be >= requests_per_second"):
            RateLimitConfig(
                requests_per_second=10,
                burst_size=5
            )
    
    def test_invalid_critical_reserve(self):
        """Test invalid critical reserve percentage."""
        with pytest.raises(ValueError, match="critical_reserve_percent must be between"):
            RateLimitConfig(
                requests_per_second=10,
                burst_size=20,
                critical_reserve_percent=Decimal("1.5")
            )


class TestTokenBucket:
    """Test token bucket implementation."""
    
    @pytest.mark.asyncio
    async def test_initial_tokens(self):
        """Test initial token state."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        assert bucket.tokens == 10.0
        assert bucket.capacity == 10
        assert bucket.refill_rate == 5.0
    
    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        """Test acquiring tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        
        # Acquire tokens within capacity
        result = await bucket.acquire(5)
        assert result is True
        assert bucket.tokens == 5.0
        
        # Acquire more tokens
        result = await bucket.acquire(3)
        assert result is True
        assert bucket.tokens == 2.0
        
        # Try to acquire more than available
        result = await bucket.acquire(5)
        assert result is False
        assert bucket.tokens == 2.0
    
    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time."""
        with patch('time.monotonic') as mock_time:
            mock_time.return_value = 0
            bucket = TokenBucket(capacity=10, refill_rate=5.0)
            
            # Use all tokens
            await bucket.acquire(10)
            assert bucket.tokens == 0.0
            
            # Advance time by 1 second (should add 5 tokens)
            mock_time.return_value = 1.0
            result = await bucket.acquire(4)
            assert result is True
            assert bucket.tokens == pytest.approx(1.0, rel=0.1)
    
    @pytest.mark.asyncio
    async def test_critical_priority_overdraft(self):
        """Test critical priority can overdraft."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        
        # Use all tokens
        await bucket.acquire(10)
        assert bucket.tokens == 0.0
        
        # Critical request can overdraft
        result = await bucket.acquire(1, priority=Priority.CRITICAL)
        assert result is True
        assert bucket.tokens == -1.0
        
        # But not beyond 10% of capacity
        result = await bucket.acquire(1, priority=Priority.CRITICAL)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_wait_for_tokens(self):
        """Test calculating wait time for tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        
        # No wait if tokens available
        wait_time = await bucket.wait_for_tokens(5)
        assert wait_time == 0.0
        
        # Use all tokens
        await bucket.acquire(10)
        
        # Calculate wait time for 5 tokens (1 second at 5 tokens/sec)
        wait_time = await bucket.wait_for_tokens(5)
        assert wait_time == pytest.approx(1.0, rel=0.1)


class TestSlidingWindowLimiter:
    """Test sliding window limiter."""
    
    @pytest.mark.asyncio
    async def test_window_limits(self):
        """Test sliding window limits."""
        limiter = SlidingWindowLimiter(window_size_seconds=10, max_requests=5)
        
        # Add requests within limit
        for _ in range(5):
            result = await limiter.check_and_add()
            assert result is True
        
        # Exceed limit
        result = await limiter.check_and_add()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_window_sliding(self):
        """Test sliding window behavior."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 0
            limiter = SlidingWindowLimiter(window_size_seconds=10, max_requests=5)
            
            # Add 5 requests at time 0
            for _ in range(5):
                await limiter.check_and_add()
            
            # Can't add more
            result = await limiter.check_and_add()
            assert result is False
            
            # Move time forward by 11 seconds (outside window)
            mock_time.return_value = 11
            
            # Now can add again
            result = await limiter.check_and_add()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_critical_priority_excess(self):
        """Test critical priority can exceed limit."""
        limiter = SlidingWindowLimiter(window_size_seconds=10, max_requests=5)
        
        # Fill to limit
        for _ in range(5):
            await limiter.check_and_add()
        
        # Critical can exceed by 10%
        result = await limiter.check_and_add(priority=Priority.CRITICAL)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_window_usage(self):
        """Test getting window usage."""
        limiter = SlidingWindowLimiter(window_size_seconds=10, max_requests=5)
        
        # Initial state
        used, total = await limiter.get_window_usage()
        assert used == 0
        assert total == 5
        
        # Add some requests
        for _ in range(3):
            await limiter.check_and_add()
        
        used, total = await limiter.get_window_usage()
        assert used == 3
        assert total == 5


class TestRateLimiter:
    """Test main rate limiter."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test rate limiter initialization."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        assert limiter.config == config
        assert limiter.token_bucket.capacity == 20
        assert limiter.token_bucket.refill_rate == 10
        assert len(limiter.priority_queues) == len(Priority)
    
    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful acquire."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        result = await limiter.acquire(tokens=1)
        assert result is True
        assert limiter.metrics["requests_allowed"] == 1
        assert limiter.metrics["requests_rejected"] == 0
    
    @pytest.mark.asyncio
    async def test_acquire_rejection(self):
        """Test acquire rejection when limits exceeded."""
        config = RateLimitConfig(
            requests_per_second=1,
            burst_size=2
        )
        limiter = RateLimiter(config)
        
        # Use up capacity
        await limiter.acquire(tokens=2, wait=False)
        
        # Next request should be rejected
        result = await limiter.acquire(tokens=1, wait=False)
        assert result is False
        assert limiter.metrics["requests_rejected"] == 1
    
    @pytest.mark.asyncio
    async def test_release_tokens(self):
        """Test releasing tokens back to bucket."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        # Use some tokens
        await limiter.acquire(tokens=5)
        assert limiter.token_bucket.tokens == 15.0
        
        # Release tokens
        await limiter.release(tokens=3)
        assert limiter.token_bucket.tokens == 18.0
    
    @pytest.mark.asyncio
    async def test_coalesce_request(self):
        """Test request coalescing."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        # Create a mock coroutine
        call_count = 0
        
        async def mock_request():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return f"result_{call_count}"
        
        # Make multiple coalesced requests
        tasks = [
            limiter.coalesce_request("key1", mock_request())
            for _ in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should get same result (only called once)
        assert call_count == 1
        assert all(r == "result_1" for r in results)
    
    @pytest.mark.asyncio
    async def test_update_from_headers(self):
        """Test updating rate limits from response headers."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        # Test high usage - should slow down
        headers = {
            "X-MBX-USED-WEIGHT-1M": "1000",  # 83% of 1200 limit
        }
        await limiter.update_from_headers(headers)
        assert limiter.token_bucket.refill_rate < 10
        
        # Test low usage - should speed up
        headers = {
            "X-MBX-USED-WEIGHT-1M": "400",  # 33% of 1200 limit
        }
        limiter.token_bucket.refill_rate = 5  # Reset to lower rate
        await limiter.update_from_headers(headers)
        assert limiter.token_bucket.refill_rate > 5
    
    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting metrics."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        limiter = RateLimiter(config)
        
        # Make some requests
        await limiter.acquire(tokens=1)
        await limiter.acquire(tokens=20, wait=False)  # Will be rejected
        
        metrics = limiter.get_metrics()
        assert metrics["requests_allowed"] == 1
        assert metrics["requests_rejected"] == 1
        assert metrics["token_bucket_capacity"] == 20


class TestDistributedRateLimiter:
    """Test distributed rate limiter with Redis."""
    
    @pytest.mark.asyncio
    async def test_fallback_to_local(self):
        """Test fallback to local rate limiting when Redis unavailable."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        
        # No Redis client
        limiter = DistributedRateLimiter(config, redis_client=None)
        
        result = await limiter.acquire_distributed("test_key", tokens=1)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_redis_rate_limiting(self):
        """Test rate limiting with Redis."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.eval.return_value = 1  # Allow request
        
        limiter = DistributedRateLimiter(config, redis_client=mock_redis)
        
        result = await limiter.acquire_distributed("test_key", tokens=1)
        assert result is True
        
        # Verify Redis was called with correct script
        mock_redis.eval.assert_called_once()
        args = mock_redis.eval.call_args
        assert "ratelimit:test_key" in args[1]["keys"]
    
    @pytest.mark.asyncio
    async def test_redis_failure_fallback(self):
        """Test fallback when Redis fails."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20
        )
        
        # Mock Redis client that fails
        mock_redis = AsyncMock()
        mock_redis.eval.side_effect = Exception("Redis connection failed")
        
        limiter = DistributedRateLimiter(config, redis_client=mock_redis)
        
        # Should fall back to local rate limiting
        result = await limiter.acquire_distributed("test_key", tokens=1)
        assert result is True  # Local rate limiter allows it