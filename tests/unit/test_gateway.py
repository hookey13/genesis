"""
Unit tests for BinanceGateway.
"""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from genesis.exchange.gateway import (
    BinanceGateway,
    OrderRequest,
    OrderResponse,
    AccountBalance,
    MarketTicker,
    OrderBook
)


class TestBinanceGateway:
    """Test suite for BinanceGateway."""
    
    @pytest.mark.asyncio
    async def test_gateway_initialization_mock_mode(self, mock_settings):
        """Test gateway initialization in mock mode."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            assert gateway.mock_mode is True
            assert gateway._initialized is False
            
            await gateway.initialize()
            assert gateway._initialized is True
            assert gateway.mock_exchange is not None
            
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_gateway_initialization_real_mode(self, mock_settings, mock_ccxt_exchange):
        """Test gateway initialization in real mode."""
        mock_settings.development.use_mock_exchange = False
        
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("genesis.exchange.gateway.ccxt.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway(mock_mode=False)
                assert gateway.mock_mode is False
                
                await gateway.initialize()
                assert gateway._initialized is True
                assert gateway.exchange is not None
                mock_ccxt_exchange.load_markets.assert_called_once()
                
                await gateway.close()
                mock_ccxt_exchange.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_account_balance(self, gateway):
        """Test fetching account balance."""
        balances = await gateway.get_account_balance()
        
        assert isinstance(balances, dict)
        assert "USDT" in balances
        assert isinstance(balances["USDT"], AccountBalance)
        assert balances["USDT"].free == Decimal("10000")
        assert balances["USDT"].used == Decimal("0")
        assert balances["USDT"].total == Decimal("10000")
    
    @pytest.mark.asyncio
    async def test_place_order_market(self, gateway):
        """Test placing a market order."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            quantity=Decimal("0.001")
        )
        
        response = await gateway.place_order(request)
        
        assert isinstance(response, OrderResponse)
        assert response.symbol == "BTC/USDT"
        assert response.side == "buy"
        assert response.type == "market"
        assert response.quantity == Decimal("0.001")
    
    @pytest.mark.asyncio
    async def test_place_order_limit(self, gateway):
        """Test placing a limit order."""
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="limit",
            quantity=Decimal("0.001"),
            price=Decimal("49000"),
            client_order_id="test_order_001"
        )
        
        response = await gateway.place_order(request)
        
        assert isinstance(response, OrderResponse)
        assert response.symbol == "BTC/USDT"
        assert response.side == "buy"
        assert response.type == "limit"
        assert response.price == Decimal("49000")
        assert response.client_order_id == "test_order_001"
    
    @pytest.mark.asyncio
    async def test_place_order_validation_error(self, gateway):
        """Test order validation error."""
        # Limit order without price should fail validation
        with pytest.raises(ValueError, match="Price required for limit orders"):
            OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                type="limit",
                quantity=Decimal("0.001")
            )
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, gateway):
        """Test cancelling an order."""
        result = await gateway.cancel_order("order_123", "BTC/USDT")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_order_status(self, gateway):
        """Test fetching order status."""
        response = await gateway.get_order_status("order_123", "BTC/USDT")
        
        assert isinstance(response, OrderResponse)
        assert response.order_id == "order_123"
        assert response.status == "filled"
    
    @pytest.mark.asyncio
    async def test_get_order_book(self, gateway):
        """Test fetching order book."""
        orderbook = await gateway.get_order_book("BTC/USDT", limit=5)
        
        assert isinstance(orderbook, OrderBook)
        assert orderbook.symbol == "BTC/USDT"
        assert len(orderbook.bids) > 0
        assert len(orderbook.asks) > 0
        assert isinstance(orderbook.bids[0][0], Decimal)  # Price
        assert isinstance(orderbook.bids[0][1], Decimal)  # Quantity
    
    @pytest.mark.asyncio
    async def test_get_klines(self, gateway):
        """Test fetching kline data."""
        klines = await gateway.get_klines("BTC/USDT", "1m", limit=10)
        
        assert isinstance(klines, list)
        assert len(klines) == 10
        assert "timestamp" in klines[0]
        assert "open" in klines[0]
        assert "high" in klines[0]
        assert "low" in klines[0]
        assert "close" in klines[0]
        assert "volume" in klines[0]
        assert isinstance(klines[0]["open"], Decimal)
    
    @pytest.mark.asyncio
    async def test_get_ticker(self, gateway):
        """Test fetching ticker data."""
        ticker = await gateway.get_ticker("BTC/USDT")
        
        assert isinstance(ticker, MarketTicker)
        assert ticker.symbol == "BTC/USDT"
        assert isinstance(ticker.bid, Decimal)
        assert isinstance(ticker.ask, Decimal)
        assert isinstance(ticker.last, Decimal)
        assert isinstance(ticker.volume, Decimal)
    
    @pytest.mark.asyncio
    async def test_get_server_time(self, gateway):
        """Test fetching server time."""
        server_time = await gateway.get_server_time()
        
        assert isinstance(server_time, int)
        assert server_time > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self, mock_settings):
        """Test that rate limiter is called for API requests."""
        mock_settings.development.use_mock_exchange = False
        
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            # Spy on rate limiter
            with patch.object(gateway.rate_limiter, "check_and_wait") as mock_check:
                mock_check.return_value = None
                
                # Make a request
                await gateway.get_account_balance()
                
                # Verify rate limiter was called
                mock_check.assert_called_once_with("GET", "/api/v3/account")
            
            await gateway.close()