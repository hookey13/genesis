"""
Unit tests for BinanceGateway.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import MarketTicker, OrderBook, OrderRequest, OrderResponse


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
    async def test_gateway_initialization_real_mode(self):
        """Test gateway initialization in real mode."""
        # Create mock settings with mock exchange disabled
        mock_settings = MagicMock()
        mock_settings.exchange = MagicMock()
        mock_settings.exchange.binance_api_key = MagicMock()
        mock_settings.exchange.binance_api_key.get_secret_value = MagicMock(
            return_value="test_api_key"
        )
        mock_settings.exchange.binance_api_secret = MagicMock()
        mock_settings.exchange.binance_api_secret.get_secret_value = MagicMock(
            return_value="test_api_secret"
        )
        mock_settings.exchange.binance_testnet = True
        mock_settings.exchange.exchange_rate_limit = 1200
        mock_settings.development = MagicMock()
        mock_settings.development.use_mock_exchange = False

        # Create mock ccxt exchange
        mock_ccxt_exchange = AsyncMock()
        mock_ccxt_exchange.load_markets = AsyncMock(return_value=True)
        mock_ccxt_exchange.close = AsyncMock(return_value=None)

        # Patch environment variable and get_settings
        with patch.dict("os.environ", {"USE_MOCK_EXCHANGE": "false"}):
            with patch(
                "genesis.exchange.gateway.get_settings", return_value=mock_settings
            ):
                with patch(
                    "genesis.exchange.gateway.ccxt.binance",
                    return_value=mock_ccxt_exchange,
                ):
                    gateway = BinanceGateway(mock_mode=False)
                    assert gateway.mock_mode is False

                    await gateway.initialize()
                    assert gateway._initialized is True
                    assert gateway.exchange is not None
                    mock_ccxt_exchange.load_markets.assert_called_once()

                    await gateway.close()
                    mock_ccxt_exchange.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_account_balance(self, mock_settings):
        """Test fetching account balance."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                balances = await gateway.get_account_balance()

                assert isinstance(balances, dict)
                assert "USDT" in balances
                assert type(balances["USDT"]).__name__ == "AccountBalance"
                assert balances["USDT"].free == Decimal("10000")
                assert balances["USDT"].locked == Decimal("0")
                assert balances["USDT"].total == Decimal("10000")
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_place_order_market(self, mock_settings):
        """Test placing a market order."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="market",
                    quantity=Decimal("0.001"),
                )

                response = await gateway.place_order(request)

                assert isinstance(response, OrderResponse)
                assert response.symbol == "BTC/USDT"
                assert response.side == "buy"
                assert response.type == "market"
                assert response.quantity == Decimal("0.001")
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_place_order_limit(self, mock_settings):
        """Test placing a limit order."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="limit",
                    quantity=Decimal("0.001"),
                    price=Decimal("49000"),
                    client_order_id="test_order_001",
                )

                response = await gateway.place_order(request)

                assert isinstance(response, OrderResponse)
                assert response.symbol == "BTC/USDT"
                assert response.side == "buy"
                assert response.type == "limit"
                assert response.price == Decimal("49000")
                assert response.client_order_id == "test_order_001"
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_place_order_validation_error(self):
        """Test order validation error."""
        # Limit order without price should fail validation
        with pytest.raises(ValueError, match="Price required for limit orders"):
            OrderRequest(
                symbol="BTC/USDT", side="buy", type="limit", quantity=Decimal("0.001")
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_settings):
        """Test cancelling an order."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                result = await gateway.cancel_order("order_123", "BTC/USDT")
                assert result is True
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_get_order_status(self, mock_settings):
        """Test fetching order status."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                response = await gateway.get_order_status("order_123", "BTC/USDT")

                assert isinstance(response, OrderResponse)
                assert response.order_id == "order_123"
                assert response.status == "filled"
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_get_order_book(self, mock_settings):
        """Test fetching order book."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                orderbook = await gateway.get_order_book("BTC/USDT", limit=5)

                assert isinstance(orderbook, OrderBook)
                assert orderbook.symbol == "BTC/USDT"
                assert len(orderbook.bids) > 0
                assert len(orderbook.asks) > 0
                assert isinstance(orderbook.bids[0][0], Decimal)  # Price
                assert isinstance(orderbook.bids[0][1], Decimal)  # Quantity
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_get_klines(self, mock_settings):
        """Test fetching kline data."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
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
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_settings):
        """Test fetching ticker data."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                ticker = await gateway.get_ticker("BTC/USDT")

                assert isinstance(ticker, MarketTicker)
                assert ticker.symbol == "BTC/USDT"
                assert isinstance(ticker.bid_price, Decimal)
                assert isinstance(ticker.ask_price, Decimal)
                assert isinstance(ticker.last_price, Decimal)
                assert isinstance(ticker.volume_24h, Decimal)
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_get_server_time(self, mock_settings):
        """Test fetching server time."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            try:
                server_time = await gateway.get_server_time()

                assert isinstance(server_time, int)
                assert server_time > 0
            finally:
                await gateway.close()

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self):
        """Test that rate limiter is called for API requests."""
        # Create mock settings with mock exchange disabled
        mock_settings = MagicMock()
        mock_settings.exchange = MagicMock()
        mock_settings.exchange.binance_api_key = MagicMock()
        mock_settings.exchange.binance_api_key.get_secret_value = MagicMock(
            return_value="test_api_key"
        )
        mock_settings.exchange.binance_api_secret = MagicMock()
        mock_settings.exchange.binance_api_secret.get_secret_value = MagicMock(
            return_value="test_api_secret"
        )
        mock_settings.exchange.binance_testnet = True
        mock_settings.exchange.exchange_rate_limit = 1200
        mock_settings.development = MagicMock()
        mock_settings.development.use_mock_exchange = False

        # Create mock ccxt exchange for real mode
        mock_ccxt_exchange = AsyncMock()
        mock_ccxt_exchange.load_markets = AsyncMock(return_value=True)
        mock_ccxt_exchange.fetch_balance = AsyncMock(
            return_value={
                "info": {"balances": {"USDT": {"free": "10000", "locked": "0"}}}
            }
        )
        mock_ccxt_exchange.close = AsyncMock(return_value=None)

        # Patch environment variable and get_settings
        with patch.dict("os.environ", {"USE_MOCK_EXCHANGE": "false"}):
            with patch(
                "genesis.exchange.gateway.get_settings", return_value=mock_settings
            ):
                with patch(
                    "genesis.exchange.gateway.ccxt.binance",
                    return_value=mock_ccxt_exchange,
                ):
                    gateway = BinanceGateway(mock_mode=False)
                    await gateway.initialize()

                    try:
                        # Spy on rate limiter
                        with patch.object(
                            gateway.rate_limiter, "check_and_wait"
                        ) as mock_check:
                            mock_check.return_value = None

                            # Make a request
                            await gateway.get_account_balance()

                            # Verify rate limiter was called
                            mock_check.assert_called_once_with("GET", "/api/v3/account")
                    finally:
                        await gateway.close()
