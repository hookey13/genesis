"""Integration tests for exchange API failure handling."""

import pytest
from tests.mocks.failing_exchange import (
    FailingExchange, FailureMode,
    test_rate_limit_handling, test_order_rejection_handling
)


@pytest.mark.asyncio
async def test_failing_exchange_init():
    """Test failing exchange initialization."""
    exchange = FailingExchange()
    assert exchange.failure_mode == FailureMode.NONE
    assert exchange.failure_rate == 0.0


@pytest.mark.asyncio
async def test_rate_limit_failure():
    """Test rate limit failure injection."""
    result = await test_rate_limit_handling()
    assert result is True  # Should handle rate limits gracefully


@pytest.mark.asyncio
async def test_order_rejection():
    """Test order rejection handling."""
    result = await test_order_rejection_handling()
    assert result is True  # Should handle rejections properly


@pytest.mark.asyncio
async def test_timeout_failure():
    """Test timeout failure handling."""
    exchange = FailingExchange()
    exchange.set_failure_mode(FailureMode.TIMEOUT, 1.0)
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            exchange.fetch_ticker("BTC/USDT"),
            timeout=1.0
        )