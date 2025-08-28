"""
Performance benchmarks for Binance API integration.

Tests to verify that critical operations meet the <100ms latency requirement.
"""

import asyncio
import time
from decimal import Decimal
from unittest.mock import patch

import pytest

from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderRequest


class TestLatencyBenchmarks:
    """Performance benchmarks for API operations."""

    @pytest.mark.benchmark(group="gateway")
    def test_gateway_initialization_latency(self, benchmark, mock_settings):
        """Benchmark gateway initialization time."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):

            def init_gateway():
                gateway = BinanceGateway(mock_mode=True)
                return gateway

            result = benchmark(init_gateway)
            assert result is not None
            # Should initialize in under 100ms
            assert benchmark.stats["mean"] < 0.1

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="orders")
    async def test_place_order_latency(self, benchmark, mock_settings):
        """Benchmark order placement latency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            request = OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                type="limit",
                quantity=Decimal("0.001"),
                price=Decimal("50000"),
            )

            async def place_order():
                return await gateway.place_order(request)

            # Run benchmark
            result = await benchmark.pedantic(place_order, rounds=100, iterations=1)

            assert result is not None
            # Order placement should complete in under 100ms
            assert benchmark.stats["mean"] < 0.1

            await gateway.close()

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="market_data")
    async def test_fetch_order_book_latency(self, benchmark, mock_settings):
        """Benchmark order book fetch latency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            async def fetch_order_book():
                return await gateway.get_order_book("BTC/USDT", limit=20)

            # Run benchmark
            result = await benchmark.pedantic(
                fetch_order_book, rounds=100, iterations=1
            )

            assert result is not None
            # Order book fetch should complete in under 100ms
            assert benchmark.stats["mean"] < 0.1

            await gateway.close()

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="market_data")
    async def test_fetch_ticker_latency(self, benchmark, mock_settings):
        """Benchmark ticker fetch latency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            async def fetch_ticker():
                return await gateway.get_ticker("BTC/USDT")

            # Run benchmark
            result = await benchmark.pedantic(fetch_ticker, rounds=100, iterations=1)

            assert result is not None
            # Ticker fetch should complete in under 100ms
            assert benchmark.stats["mean"] < 0.1

            await gateway.close()

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="account")
    async def test_fetch_balance_latency(self, benchmark, mock_settings):
        """Benchmark account balance fetch latency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            async def fetch_balance():
                return await gateway.get_account_balance()

            # Run benchmark
            result = await benchmark.pedantic(fetch_balance, rounds=100, iterations=1)

            assert result is not None
            # Balance fetch should complete in under 100ms
            assert benchmark.stats["mean"] < 0.1

            await gateway.close()

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="orders")
    async def test_cancel_order_latency(self, benchmark, mock_settings):
        """Benchmark order cancellation latency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            async def cancel_order():
                return await gateway.cancel_order("mock_order_001", "BTC/USDT")

            # Run benchmark
            result = await benchmark.pedantic(cancel_order, rounds=100, iterations=1)

            assert result is not None
            # Order cancellation should complete in under 100ms
            assert benchmark.stats["mean"] < 0.1

            await gateway.close()

    @pytest.mark.benchmark(group="rate_limiter")
    def test_rate_limiter_check_latency(self, benchmark):
        """Benchmark rate limiter check latency."""
        from genesis.exchange.rate_limiter import RateLimiter

        limiter = RateLimiter(max_weight=1200, window_seconds=60)

        def check_weight():
            # Simulate checking weight for a typical request
            remaining = limiter.get_remaining_weight()
            # Calculate utilization manually since get_utilization doesn't exist
            used_weight = limiter.max_weight - remaining
            utilization = (
                used_weight / limiter.max_weight if limiter.max_weight > 0 else 0
            )
            return remaining, utilization

        result = benchmark(check_weight)
        assert result is not None
        # Rate limiter check should be near-instant (< 1ms)
        assert benchmark.stats["mean"] < 0.001

    @pytest.mark.benchmark(group="circuit_breaker")
    def test_circuit_breaker_check_latency(self, benchmark):
        """Benchmark circuit breaker state check latency."""
        from genesis.exchange.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(
            name="test",
            failure_threshold=5,
            failure_window_seconds=30,
            recovery_timeout_seconds=60,
        )

        def check_state():
            return breaker.get_state()

        result = benchmark(check_state)
        assert result is not None
        # Circuit breaker state check should be near-instant (< 1ms)
        assert benchmark.stats["mean"] < 0.001

    @pytest.mark.asyncio
    @pytest.mark.benchmark(group="end_to_end")
    async def test_end_to_end_order_flow_latency(self, benchmark, mock_settings):
        """Benchmark complete order flow from placement to status check."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            async def complete_order_flow():
                # Place order
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="limit",
                    quantity=Decimal("0.001"),
                    price=Decimal("50000"),
                    client_order_id="test_order_001",
                )
                order = await gateway.place_order(request)

                # Check status
                status = await gateway.get_order_status(order.order_id, "BTC/USDT")

                # Cancel if still open
                if status.status == "open":
                    await gateway.cancel_order(order.order_id, "BTC/USDT")

                return status

            # Run benchmark
            result = await benchmark.pedantic(
                complete_order_flow, rounds=50, iterations=1
            )

            assert result is not None
            # Complete flow should finish in under 300ms (3 operations)
            assert benchmark.stats["mean"] < 0.3

            await gateway.close()


class TestLatencyUnderLoad:
    """Test latency under various load conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_latency(self, mock_settings):
        """Test latency with multiple concurrent requests."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            async def make_request(i):
                start = time.perf_counter()
                result = await gateway.get_ticker("BTC/USDT")
                elapsed = time.perf_counter() - start
                return elapsed

            # Run 50 concurrent requests
            tasks = [make_request(i) for i in range(50)]
            latencies = await asyncio.gather(*tasks)

            # Calculate statistics
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            # Average latency should stay under 100ms even under load
            assert avg_latency < 0.1
            # No single request should exceed 200ms
            assert max_latency < 0.2

            await gateway.close()

    @pytest.mark.asyncio
    async def test_rate_limited_requests(self, mock_settings):
        """Test that rate limiting doesn't excessively impact latency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            # Pre-fill rate limiter to 70% capacity
            gateway.rate_limiter.current_weight = 840  # 70% of 1200

            start = time.perf_counter()
            # This should still execute without delay
            result = await gateway.get_ticker("BTC/USDT")
            elapsed = time.perf_counter() - start

            assert result is not None
            # Should complete quickly even at 70% capacity
            assert elapsed < 0.1

            await gateway.close()


@pytest.fixture
def mock_exchange_with_latency():
    """Create a mock exchange with simulated network latency."""
    from genesis.exchange.mock_exchange import MockExchange

    exchange = MockExchange()
    exchange.latency_ms = 20  # Simulate 20ms network latency
    return exchange
