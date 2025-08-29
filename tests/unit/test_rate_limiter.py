"""
Unit tests for RateLimiter.
"""

import asyncio
import time

import pytest

from genesis.exchange.rate_limiter import RateLimiter, WeightWindow


class TestRateLimiter:
    """Test suite for RateLimiter."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_weight=1200, window_seconds=60, threshold_percent=80)

        assert limiter.max_weight == 1200
        assert limiter.window_seconds == 60
        assert limiter.threshold_percent == 80
        assert limiter.threshold_weight == 960  # 80% of 1200
        assert limiter.current_weight == 0
        assert limiter.max_orders_10s == 50
        assert limiter.max_orders_daily == 160000
        assert limiter.order_threshold_10s == 40  # 80% of 50
        assert limiter.order_threshold_daily == 128000  # 80% of 160000

    @pytest.mark.asyncio
    async def test_simple_request(self, rate_limiter):
        """Test a simple request within limits."""
        await rate_limiter.check_and_wait(
            "GET", "/api/v3/ticker/price", {"symbol": "BTCUSDT"}
        )

        assert rate_limiter.current_weight == 1  # Ticker with symbol = 1 weight
        assert rate_limiter.total_requests == 1
        assert rate_limiter.get_current_utilization() < 1  # Less than 1%

    @pytest.mark.asyncio
    async def test_heavy_request(self, rate_limiter):
        """Test a heavy request."""
        await rate_limiter.check_and_wait("GET", "/api/v3/account")

        assert rate_limiter.current_weight == 10  # Account endpoint = 10 weight
        assert rate_limiter.total_requests == 1

    @pytest.mark.asyncio
    async def test_conditional_weight_limit(self, rate_limiter):
        """Test conditional weight based on limit parameter."""
        # Small limit
        await rate_limiter.check_and_wait("GET", "/api/v3/depth", {"limit": 50})
        assert rate_limiter.current_weight == 1

        rate_limiter.reset_window()

        # Medium limit
        await rate_limiter.check_and_wait("GET", "/api/v3/depth", {"limit": 200})
        assert rate_limiter.current_weight == 5

        rate_limiter.reset_window()

        # Large limit
        await rate_limiter.check_and_wait("GET", "/api/v3/depth", {"limit": 600})
        assert rate_limiter.current_weight == 10

    @pytest.mark.asyncio
    async def test_threshold_warning(self, rate_limiter):
        """Test threshold warning and backoff."""
        # Set a low threshold for testing
        rate_limiter.max_weight = 100
        rate_limiter.threshold_weight = 80

        # Add weight up to threshold
        rate_limiter.current_weight = 75

        # This should trigger backoff
        start_time = time.time()
        await rate_limiter.check_and_wait("GET", "/api/v3/account")  # 10 weight
        elapsed = time.time() - start_time

        # Should have waited due to threshold
        assert elapsed >= 1.0  # Default backoff is 2^1 = 2 seconds, but might be faster
        assert rate_limiter.threshold_hits == 1
        assert rate_limiter.consecutive_threshold_hits == 1

    def test_clean_old_weights(self, rate_limiter):
        """Test cleaning of old weight entries."""
        current_time = time.time()

        # Add old weight (outside window)
        old_weight = WeightWindow(
            timestamp=current_time - 70,  # 70 seconds ago (outside 60s window)
            weight=10,
        )
        rate_limiter.weight_history.append(old_weight)
        rate_limiter.current_weight = 10

        # Add recent weight
        recent_weight = WeightWindow(
            timestamp=current_time - 30, weight=5  # 30 seconds ago (inside window)
        )
        rate_limiter.weight_history.append(recent_weight)
        rate_limiter.current_weight = 15

        # Clean old weights
        rate_limiter._clean_old_weights()

        # Old weight should be removed
        assert len(rate_limiter.weight_history) == 1
        assert rate_limiter.current_weight == 5

    def test_get_remaining_weight(self, rate_limiter):
        """Test getting remaining weight capacity."""
        rate_limiter.current_weight = 300

        remaining = rate_limiter.get_remaining_weight()
        assert remaining == 900  # 1200 - 300

    def test_get_utilization(self, rate_limiter):
        """Test utilization calculation."""
        rate_limiter.current_weight = 600

        utilization = rate_limiter.get_current_utilization()
        assert utilization == 50.0  # 600/1200 * 100

    def test_reset_window(self, rate_limiter):
        """Test window reset."""
        rate_limiter.current_weight = 500
        rate_limiter.consecutive_threshold_hits = 3
        rate_limiter.backoff_until = time.time() + 10

        rate_limiter.reset_window()

        assert rate_limiter.current_weight == 0
        assert rate_limiter.consecutive_threshold_hits == 0
        assert rate_limiter.backoff_until is None
        assert len(rate_limiter.weight_history) == 0

    def test_get_statistics(self, rate_limiter):
        """Test statistics retrieval."""
        rate_limiter.total_requests = 100
        rate_limiter.total_weight_consumed = 500
        rate_limiter.current_weight = 200
        rate_limiter.threshold_hits = 5

        stats = rate_limiter.get_statistics()

        assert stats["total_requests"] == 100
        assert stats["total_weight_consumed"] == 500
        assert stats["current_weight"] == 200
        assert stats["threshold_hits"] == 5
        assert stats["current_utilization_percent"] == pytest.approx(16.67, rel=0.01)
        assert stats["remaining_weight"] == 1000

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, rate_limiter):
        """Test handling concurrent requests."""
        # Create multiple concurrent requests
        tasks = [
            rate_limiter.check_and_wait(
                "GET", "/api/v3/ticker/price", {"symbol": "BTCUSDT"}
            ),
            rate_limiter.check_and_wait(
                "GET", "/api/v3/ticker/price", {"symbol": "ETHUSDT"}
            ),
            rate_limiter.check_and_wait(
                "GET", "/api/v3/ticker/price", {"symbol": "BNBUSDT"}
            ),
        ]

        await asyncio.gather(*tasks)

        assert rate_limiter.current_weight == 3  # 3 requests, 1 weight each
        assert rate_limiter.total_requests == 3

    def test_endpoint_weight_mapping(self, rate_limiter):
        """Test endpoint weight mapping."""
        # Test various endpoints
        assert rate_limiter._get_endpoint_weight("GET", "/api/v3/ping") == 1
        assert rate_limiter._get_endpoint_weight("GET", "/api/v3/time") == 1
        assert rate_limiter._get_endpoint_weight("GET", "/api/v3/account") == 10
        assert rate_limiter._get_endpoint_weight("POST", "/api/v3/order") == 1
        assert rate_limiter._get_endpoint_weight("DELETE", "/api/v3/order") == 1
        assert rate_limiter._get_endpoint_weight("GET", "/api/v3/exchangeInfo") == 10

        # Test unknown endpoint (default to 1)
        assert rate_limiter._get_endpoint_weight("GET", "/unknown/endpoint") == 1

    @pytest.mark.asyncio
    async def test_order_rate_limit_10s(self, rate_limiter):
        """Test order-specific rate limiting for 10-second window."""
        # Place orders up to threshold (40 orders, 80% of 50)
        for _ in range(39):
            await rate_limiter.check_and_wait("POST", "/api/v3/order")

        stats = rate_limiter.get_statistics()
        assert stats["orders"]["current_10s"] == 39
        assert stats["orders"]["remaining_10s"] == 11

        # 40th order should trigger warning but still go through
        await rate_limiter.check_and_wait("POST", "/api/v3/order")

        stats = rate_limiter.get_statistics()
        assert stats["orders"]["current_10s"] == 40
        assert stats["orders"]["total_placed"] == 40

    @pytest.mark.asyncio
    async def test_order_rate_limit_10s_wait(self, rate_limiter):
        """Test that rate limiter waits when 10s order limit is reached."""
        # Fill up to threshold
        for _ in range(40):
            await rate_limiter.check_and_wait("POST", "/api/v3/order")

        # Next order should cause a wait
        start_time = time.time()
        await rate_limiter.check_and_wait("POST", "/api/v3/order")
        _ = time.time() - start_time  # Elapsed time (checking wait occurred)

        # Should have waited, but not too long (mocked time in tests)
        assert rate_limiter.total_orders_placed == 41

    @pytest.mark.asyncio
    async def test_order_statistics(self, rate_limiter):
        """Test order statistics in get_statistics."""
        # Place some orders
        for _ in range(5):
            await rate_limiter.check_and_wait("POST", "/api/v3/order")

        stats = rate_limiter.get_statistics()

        assert "orders" in stats
        assert stats["orders"]["total_placed"] == 5
        assert stats["orders"]["current_10s"] == 5
        assert stats["orders"]["current_daily"] == 5
        assert stats["orders"]["remaining_10s"] == 45
        assert stats["orders"]["remaining_daily"] == 159995
        assert stats["orders"]["utilization_10s_percent"] == 10.0  # 5/50 * 100
        assert stats["orders"]["utilization_daily_percent"] < 0.01  # Very small
