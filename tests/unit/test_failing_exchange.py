"""
Unit tests for failing exchange mock.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from decimal import Decimal
from enum import Enum
from tests.mocks.failing_exchange import FailingExchange, FailureMode


class TestFailureMode:
    """Tests for FailureMode enum."""
    
    def test_failure_modes_exist(self):
        """Test that failure modes are defined."""
        expected_modes = [
            "RATE_LIMIT",
            "CONNECTION_ERROR",
            "TIMEOUT",
            "INVALID_ORDER",
            "INSUFFICIENT_BALANCE"
        ]
        
        for mode in expected_modes:
            assert hasattr(FailureMode, mode)
            assert isinstance(getattr(FailureMode, mode), FailureMode)


class TestFailingExchange:
    """Tests for FailingExchange mock class."""
    
    @pytest.mark.asyncio
    async def test_exchange_initialization(self):
        """Test failing exchange initialization."""
        exchange = FailingExchange()
        
        assert exchange is not None
        assert hasattr(exchange, 'failure_mode')
        assert hasattr(exchange, 'failure_rate')
        assert exchange.failure_rate >= 0
        assert exchange.failure_rate <= 1
    
    @pytest.mark.asyncio
    async def test_set_failure_mode(self):
        """Test setting failure mode."""
        exchange = FailingExchange()
        
        # Set rate limit failure
        exchange.set_failure_mode(FailureMode.RATE_LIMIT, probability=0.5)
        
        assert exchange.failure_mode == FailureMode.RATE_LIMIT
        assert exchange.failure_rate == 0.5
    
    @pytest.mark.asyncio
    async def test_rate_limit_failure(self):
        """Test rate limit failure mode."""
        exchange = FailingExchange()
        exchange.set_failure_mode(FailureMode.RATE_LIMIT, probability=1.0)
        
        # Should fail with rate limit error
        with pytest.raises(Exception) as exc_info:
            await exchange.fetch_balance()
        
        assert "rate limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test connection error failure."""
        exchange = FailingExchange()
        exchange.set_failure_mode(FailureMode.CONNECTION_ERROR, probability=1.0)
        
        # Should fail with connection error
        with pytest.raises(Exception) as exc_info:
            await exchange.fetch_ticker("BTC/USDT")
        
        assert "connection" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_timeout_failure(self):
        """Test timeout failure."""
        exchange = FailingExchange()
        exchange.set_failure_mode(FailureMode.TIMEOUT, probability=1.0)
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await exchange.fetch_order_book("BTC/USDT")
    
    @pytest.mark.asyncio
    async def test_invalid_order_failure(self):
        """Test invalid order failure."""
        exchange = FailingExchange()
        exchange.set_failure_mode(FailureMode.INVALID_ORDER, probability=1.0)
        
        # Should reject order
        with pytest.raises(Exception) as exc_info:
            await exchange.create_order(
                symbol="BTC/USDT",
                type="limit",
                side="buy",
                amount=0.01,
                price=50000
            )
        
        assert "invalid" in str(exc_info.value).lower() or "rejected" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_failure(self):
        """Test insufficient balance failure."""
        exchange = FailingExchange()
        exchange.set_failure_mode(FailureMode.INSUFFICIENT_BALANCE, probability=1.0)
        
        # Should fail with insufficient balance
        with pytest.raises(Exception) as exc_info:
            await exchange.create_order(
                symbol="BTC/USDT",
                type="market",
                side="buy",
                amount=100  # Large amount
            )
        
        assert "insufficient" in str(exc_info.value).lower() or "balance" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_random_failures(self):
        """Test random failure generation."""
        exchange = FailingExchange()
        exchange.set_failure_mode(FailureMode.CONNECTION_ERROR, probability=0.5)
        
        # Make multiple requests
        successes = 0
        failures = 0
        
        for _ in range(20):
            try:
                await exchange.fetch_ticker("BTC/USDT")
                successes += 1
            except Exception:
                failures += 1
        
        # Should have some of each (statistically)
        assert successes > 0
        assert failures > 0
    
    @pytest.mark.asyncio
    async def test_no_failure_mode(self):
        """Test exchange with no failures."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        # Should work normally
        balance = await exchange.fetch_balance()
        assert balance is not None
        assert isinstance(balance, dict)
        
        ticker = await exchange.fetch_ticker("BTC/USDT")
        assert ticker is not None
        assert isinstance(ticker, dict)
    
    @pytest.mark.asyncio
    async def test_fetch_balance(self):
        """Test fetching balance."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        balance = await exchange.fetch_balance()
        
        assert 'USDT' in balance
        assert 'BTC' in balance
        assert 'free' in balance['USDT']
        assert 'used' in balance['USDT']
        assert 'total' in balance['USDT']
    
    @pytest.mark.asyncio
    async def test_fetch_ticker(self):
        """Test fetching ticker."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        ticker = await exchange.fetch_ticker("BTC/USDT")
        
        assert 'symbol' in ticker
        assert 'bid' in ticker
        assert 'ask' in ticker
        assert 'last' in ticker
        assert ticker['symbol'] == 'BTC/USDT'
    
    @pytest.mark.asyncio
    async def test_fetch_order_book(self):
        """Test fetching order book."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        order_book = await exchange.fetch_order_book("BTC/USDT")
        
        assert 'bids' in order_book
        assert 'asks' in order_book
        assert len(order_book['bids']) > 0
        assert len(order_book['asks']) > 0
    
    @pytest.mark.asyncio
    async def test_create_order(self):
        """Test creating an order."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        order = await exchange.create_order(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            amount=0.01,
            price=50000
        )
        
        assert 'id' in order
        assert 'symbol' in order
        assert 'type' in order
        assert 'side' in order
        assert 'amount' in order
        assert 'price' in order
        assert 'status' in order
        assert order['symbol'] == 'BTC/USDT'
        assert order['type'] == 'limit'
        assert order['side'] == 'buy'
    
    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """Test canceling an order."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        # Create an order first
        order = await exchange.create_order(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            amount=0.01,
            price=50000
        )
        
        # Cancel it
        result = await exchange.cancel_order(order['id'], "BTC/USDT")
        
        assert result is not None
        assert 'id' in result
        assert result['id'] == order['id']
        assert result['status'] == 'canceled'
    
    @pytest.mark.asyncio
    async def test_fetch_orders(self):
        """Test fetching orders."""
        exchange = FailingExchange()
        exchange.set_failure_mode(None, probability=0)
        
        # Create some orders
        await exchange.create_order(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            amount=0.01,
            price=50000
        )
        
        # Fetch orders
        orders = await exchange.fetch_orders("BTC/USDT")
        
        assert isinstance(orders, list)
        assert len(orders) > 0
        assert all('id' in order for order in orders)
        assert all('symbol' in order for order in orders)


@pytest.mark.asyncio
async def test_complex_failure_scenario():
    """Test complex failure scenario with mode changes."""
    exchange = FailingExchange()
    
    # Start with no failures
    exchange.set_failure_mode(None, probability=0)
    balance = await exchange.fetch_balance()
    assert balance is not None
    
    # Switch to rate limiting
    exchange.set_failure_mode(FailureMode.RATE_LIMIT, probability=1.0)
    with pytest.raises(Exception) as exc_info:
        await exchange.fetch_balance()
    assert "rate limit" in str(exc_info.value).lower()
    
    # Switch to connection errors
    exchange.set_failure_mode(FailureMode.CONNECTION_ERROR, probability=1.0)
    with pytest.raises(Exception) as exc_info:
        await exchange.fetch_ticker("BTC/USDT")
    assert "connection" in str(exc_info.value).lower()
    
    # Back to normal
    exchange.set_failure_mode(None, probability=0)
    ticker = await exchange.fetch_ticker("BTC/USDT")
    assert ticker is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])