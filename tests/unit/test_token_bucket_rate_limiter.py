"""
Unit tests for token bucket rate limiter implementation.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from genesis.exchange.token_bucket_rate_limiter import (
    TokenBucket,
    TokenBucketRateLimiter,
)


class TestTokenBucket:
    """Test the TokenBucket class."""
    
    def test_initial_capacity(self):
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(
            capacity=100,
            refill_rate=10.0,
            tokens=100,
            last_refill=time.time()
        )
        assert bucket.tokens == 100
        assert bucket.capacity == 100
    
    def test_token_consumption(self):
        """Test consuming tokens from bucket."""
        bucket = TokenBucket(
            capacity=100,
            refill_rate=10.0,
            tokens=100,
            last_refill=time.time()
        )
        
        # Consume 30 tokens
        result = bucket.consume(30)
        assert result is True
        assert bucket.tokens == 70
        
        # Try to consume more than available
        result = bucket.consume(80)
        assert result is False
        assert bucket.tokens == 70  # Unchanged
    
    def test_token_refill(self):
        """Test token refill based on elapsed time."""
        start_time = time.time()
        bucket = TokenBucket(
            capacity=100,
            refill_rate=10.0,  # 10 tokens per second
            tokens=50,
            last_refill=start_time
        )
        
        # Simulate 2 seconds passing
        with patch("time.time", return_value=start_time + 2):
            bucket.refill()
            # Should have added 20 tokens (10 per second * 2 seconds)
            assert bucket.tokens == 70
    
    def test_refill_caps_at_capacity(self):
        """Test that refill doesn't exceed capacity."""
        start_time = time.time()
        bucket = TokenBucket(
            capacity=100,
            refill_rate=10.0,
            tokens=90,
            last_refill=start_time
        )
        
        # Simulate 5 seconds passing (would add 50 tokens)
        with patch("time.time", return_value=start_time + 5):
            bucket.refill()
            # Should be capped at capacity
            assert bucket.tokens == 100
    
    def test_time_until_tokens_available(self):
        """Test calculating wait time for tokens."""
        bucket = TokenBucket(
            capacity=100,
            refill_rate=10.0,
            tokens=30,
            last_refill=time.time()
        )
        
        # No wait for available tokens
        wait_time = bucket.time_until_tokens_available(20)
        assert wait_time == 0
        
        # Calculate wait for unavailable tokens
        wait_time = bucket.time_until_tokens_available(50)
        # Need 20 more tokens, at 10/second = 2 seconds
        assert wait_time == 2.0


class TestTokenBucketRateLimiter:
    """Test the TokenBucketRateLimiter class."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter instance."""
        return TokenBucketRateLimiter(
            bucket_capacity=1200,
            refill_rate=20.0,
            burst_reserve=100
        )
    
    def test_initialization(self, rate_limiter):
        """Test rate limiter initialization."""
        assert rate_limiter.bucket.capacity == 1200
        assert rate_limiter.bucket.refill_rate == 20.0
        assert rate_limiter.burst_reserve == 100
        assert rate_limiter.bucket.tokens == 1200
    
    def test_endpoint_weight_calculation(self, rate_limiter):
        """Test weight calculation for different endpoints."""
        # Simple endpoint
        weight = rate_limiter._get_endpoint_weight("GET", "/api/v3/time")
        assert weight == 1
        
        # Heavy endpoint
        weight = rate_limiter._get_endpoint_weight("GET", "/api/v3/account")
        assert weight == 10
        
        # Conditional weight based on limit
        weight = rate_limiter._get_endpoint_weight(
            "GET", "/api/v3/depth", {"limit": 100}
        )
        assert weight == 1
        
        weight = rate_limiter._get_endpoint_weight(
            "GET", "/api/v3/depth", {"limit": 500}
        )
        assert weight == 5
        
        weight = rate_limiter._get_endpoint_weight(
            "GET", "/api/v3/depth", {"limit": 1500}
        )
        assert weight == 50
    
    def test_endpoint_weight_with_symbol(self, rate_limiter):
        """Test weight calculation for symbol-dependent endpoints."""
        # With symbol
        weight = rate_limiter._get_endpoint_weight(
            "GET", "/api/v3/ticker/24hr", {"symbol": "BTCUSDT"}
        )
        assert weight == 1
        
        # Without symbol
        weight = rate_limiter._get_endpoint_weight(
            "GET", "/api/v3/ticker/24hr", {}
        )
        assert weight == 40
    
    async def test_check_and_wait_with_available_tokens(self, rate_limiter):
        """Test request proceeds when tokens available."""
        # Should not wait since tokens are available
        start_time = time.time()
        await rate_limiter.check_and_wait("GET", "/api/v3/time")
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1  # Should be nearly instant
        assert rate_limiter.bucket.tokens < 1200  # Tokens consumed
        assert rate_limiter.total_requests == 1
    
    async def test_check_and_wait_with_insufficient_tokens(self, rate_limiter):
        """Test request waits when tokens insufficient."""
        # Consume most tokens
        rate_limiter.bucket.tokens = 5
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Request needs 10 tokens but only 5 available
            await rate_limiter.check_and_wait("GET", "/api/v3/account")
            
            # Should have waited for tokens
            mock_sleep.assert_called_once()
            # Wait time should be approximately (10-5)/20 = 0.25 seconds
            wait_time = mock_sleep.call_args[0][0]
            assert 0.2 <= wait_time <= 0.3
    
    async def test_burst_capacity_activation(self, rate_limiter):
        """Test burst capacity is used for important operations."""
        # Set tokens low but within burst reserve
        rate_limiter.bucket.tokens = 50
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Small request that fits in burst reserve
            await rate_limiter.check_and_wait("GET", "/api/v3/time")
            
            # Should not wait due to burst capacity
            mock_sleep.assert_not_called()
            assert rate_limiter.burst_activations == 1
    
    async def test_order_rate_limiting(self, rate_limiter):
        """Test order-specific rate limiting."""
        # Place multiple orders quickly
        for _ in range(5):
            await rate_limiter.check_and_wait("POST", "/api/v3/order")
        
        assert rate_limiter.total_orders_placed == 5
        assert rate_limiter.order_bucket_10s.tokens < 50
        assert rate_limiter.order_bucket_daily.tokens < 160000
    
    async def test_order_rate_limit_waiting(self, rate_limiter):
        """Test waiting when order rate limit reached."""
        # Exhaust 10-second order bucket
        rate_limiter.order_bucket_10s.tokens = 0
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await rate_limiter.check_and_wait("POST", "/api/v3/order")
            
            # Should have waited for order tokens
            mock_sleep.assert_called()
            # At 5 tokens/second refill, should wait 0.2 seconds for 1 token
            wait_time = mock_sleep.call_args_list[0][0][0]
            assert 0.15 <= wait_time <= 0.25
    
    def test_update_weight_from_headers(self, rate_limiter):
        """Test updating weights based on API response headers."""
        rate_limiter.last_request_weight = 10
        rate_limiter.bucket.tokens = 1190  # Used 10 tokens
        
        # Simulate header indicating more weight was actually used
        headers = {
            "X-MBX-USED-WEIGHT": "15",
            "X-MBX-USED-WEIGHT-1M": "15"
        }
        
        rate_limiter.update_weight_from_headers(headers)
        
        # Should have consumed additional tokens
        assert rate_limiter.bucket.tokens < 1190
    
    def test_statistics(self, rate_limiter):
        """Test statistics gathering."""
        rate_limiter.total_requests = 100
        rate_limiter.total_weight_consumed = 500
        rate_limiter.burst_activations = 5
        rate_limiter.total_orders_placed = 20
        
        stats = rate_limiter.get_statistics()
        
        assert stats["total_requests"] == 100
        assert stats["total_weight_consumed"] == 500
        assert stats["burst_activations"] == 5
        assert stats["order_buckets"]["total_orders_placed"] == 20
        assert "main_bucket" in stats
        assert "order_buckets" in stats
    
    def test_reset(self, rate_limiter):
        """Test resetting all buckets."""
        # Consume some tokens
        rate_limiter.bucket.tokens = 500
        rate_limiter.order_bucket_10s.tokens = 20
        rate_limiter.order_bucket_daily.tokens = 100000
        
        # Reset
        rate_limiter.reset()
        
        # All buckets should be at full capacity
        assert rate_limiter.bucket.tokens == 1200
        assert rate_limiter.order_bucket_10s.tokens == 50
        assert rate_limiter.order_bucket_daily.tokens == 160000
    
    def test_utilization_calculation(self, rate_limiter):
        """Test utilization percentage calculation."""
        rate_limiter.bucket.tokens = 900  # Used 300 tokens
        
        utilization = rate_limiter.get_current_utilization()
        assert utilization == 25.0  # 300/1200 = 25%
        
        rate_limiter.bucket.tokens = 0  # All tokens used
        utilization = rate_limiter.get_current_utilization()
        assert utilization == 100.0