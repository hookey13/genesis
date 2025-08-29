"""
Integration tests for REST API order placement.

Tests order placement, status checking, and cancellation through the
Binance API gateway with proper authentication and testnet support.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderRequest


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.exchange.binance_testnet = True
    settings.exchange.binance_api_key.get_secret_value.return_value = "test_api_key"
    settings.exchange.binance_api_secret.get_secret_value.return_value = "test_secret"
    settings.exchange.exchange_rate_limit = 1200
    settings.development.use_mock_exchange = False
    return settings


@pytest.fixture
def mock_ccxt_exchange():
    """Mock ccxt exchange instance."""
    exchange = AsyncMock()
    exchange.load_markets = AsyncMock()
    exchange.fetch_balance = AsyncMock(return_value={
        "info": {
            "balances": {
                "USDT": {"free": "10000", "locked": "0"},
                "BTC": {"free": "0.1", "locked": "0"},
            }
        }
    })
    exchange.create_order = AsyncMock()
    exchange.cancel_order = AsyncMock()
    exchange.fetch_order = AsyncMock()
    exchange.fetch_open_orders = AsyncMock(return_value=[])
    exchange.fetch_order_book = AsyncMock()
    exchange.fetch_ticker = AsyncMock()
    exchange.fetch_recent_trades = AsyncMock(return_value=[])
    exchange.markets = {"BTC/USDT": {"active": True}}
    exchange.close = AsyncMock()
    return exchange


class TestRestApiOrders:
    """Test REST API order operations."""

    @pytest.mark.asyncio
    async def test_gateway_initialization(self, mock_settings):
        """Test gateway initialization with CCXT wrapper."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance") as mock_binance:
                mock_exchange = AsyncMock()
                mock_exchange.load_markets = AsyncMock()
                mock_exchange.markets = {"BTC/USDT": {"active": True}}
                mock_exchange.close = AsyncMock()
                mock_binance.return_value = mock_exchange

                gateway = BinanceGateway()
                await gateway.initialize()

                # Verify configuration
                assert gateway._initialized is True
                assert gateway.exchange is not None

                # Verify testnet configuration
                config_call = mock_binance.call_args[0][0]
                assert config_call["hostname"] == "testnet.binance.vision"
                assert "testnet.binance.vision" in config_call["urls"]["api"]["public"]

                # Verify authentication
                assert config_call["apiKey"] == "test_api_key"
                assert config_call["secret"] == "test_secret"

                # Verify HMAC SHA256 will be used (configured in ccxt)
                assert config_call["options"]["recvWindow"] == 5000

                # Verify rate limiting enabled
                assert config_call["enableRateLimit"] is True
                assert config_call["rateLimit"] == 1200

                await gateway.close()

    @pytest.mark.asyncio
    async def test_place_market_order(self, mock_settings, mock_ccxt_exchange):
        """Test placing a market order."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Configure mock order response
                mock_ccxt_exchange.create_order.return_value = {
                    "id": "12345",
                    "clientOrderId": "client_001",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "type": "market",
                    "status": "closed",
                    "price": 50000,
                    "amount": 0.001,
                    "filled": 0.001,
                    "timestamp": datetime.now().timestamp() * 1000,
                }

                # Create order request
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="market",
                    quantity=Decimal("0.001"),
                    client_order_id="client_001",
                )

                # Place order
                response = await gateway.place_order(request)

                # Verify order placement
                assert response.order_id == "12345"
                assert response.client_order_id == "client_001"
                assert response.symbol == "BTC/USDT"
                assert response.side == "buy"
                assert response.type == "market"
                assert response.status == "closed"
                assert response.filled_quantity == Decimal("0.001")

                # Verify ccxt was called correctly
                mock_ccxt_exchange.create_order.assert_called_once_with(
                    symbol="BTC/USDT",
                    type="market",
                    side="buy",
                    amount=0.001,
                    price=None,
                    params={"clientOrderId": "client_001"},
                )

    @pytest.mark.asyncio
    async def test_place_limit_order(self, mock_settings, mock_ccxt_exchange):
        """Test placing a limit order."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Configure mock order response
                mock_ccxt_exchange.create_order.return_value = {
                    "id": "12346",
                    "clientOrderId": "client_002",
                    "symbol": "BTC/USDT",
                    "side": "sell",
                    "type": "limit",
                    "status": "open",
                    "price": 51000,
                    "amount": 0.001,
                    "filled": 0,
                    "timestamp": datetime.now().timestamp() * 1000,
                }

                # Create order request
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="sell",
                    type="limit",
                    price=Decimal("51000"),
                    quantity=Decimal("0.001"),
                    client_order_id="client_002",
                )

                # Place order
                response = await gateway.place_order(request)

                # Verify order placement
                assert response.order_id == "12346"
                assert response.status == "open"
                assert response.price == Decimal("51000")
                assert response.filled_quantity == Decimal("0")

                # Verify ccxt was called with price
                mock_ccxt_exchange.create_order.assert_called_once_with(
                    symbol="BTC/USDT",
                    type="limit",
                    side="sell",
                    amount=0.001,
                    price=51000.0,
                    params={"clientOrderId": "client_002"},
                )

    @pytest.mark.asyncio
    async def test_check_order_status(self, mock_settings, mock_ccxt_exchange):
        """Test checking order status."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Configure mock status response
                mock_ccxt_exchange.fetch_order.return_value = {
                    "id": "12345",
                    "clientOrderId": "client_001",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "type": "limit",
                    "status": "closed",
                    "price": 50000,
                    "amount": 0.001,
                    "filled": 0.001,
                    "timestamp": datetime.now().timestamp() * 1000,
                    "lastUpdateTimestamp": datetime.now().timestamp() * 1000,
                }

                # Check order status
                response = await gateway.get_order_status("12345", "BTC/USDT")

                # Verify status
                assert response.order_id == "12345"
                assert response.status == "closed"
                assert response.filled_quantity == Decimal("0.001")
                assert response.updated_at is not None

                # Verify ccxt was called correctly
                mock_ccxt_exchange.fetch_order.assert_called_once_with("12345", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_cancel_order(self, mock_settings, mock_ccxt_exchange):
        """Test cancelling an order."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Configure mock cancel response
                mock_ccxt_exchange.cancel_order.return_value = {
                    "id": "12345",
                    "status": "canceled",
                }

                # Cancel order
                result = await gateway.cancel_order("12345", "BTC/USDT")

                # Verify cancellation
                assert result is True

                # Verify ccxt was called
                mock_ccxt_exchange.cancel_order.assert_called_once_with("12345", "BTC/USDT")

    @pytest.mark.asyncio
    async def test_testnet_configuration(self, mock_settings):
        """Test testnet environment configuration."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance") as mock_binance:
                mock_exchange = AsyncMock()
                mock_exchange.load_markets = AsyncMock()
                mock_exchange.markets = {"BTC/USDT": {"active": True}}
                mock_exchange.close = AsyncMock()
                mock_binance.return_value = mock_exchange

                # Test with testnet enabled
                mock_settings.exchange.binance_testnet = True
                gateway = BinanceGateway()
                await gateway.initialize()

                config_call = mock_binance.call_args[0][0]
                assert config_call["hostname"] == "testnet.binance.vision"
                assert "testnet.binance.vision" in config_call["urls"]["api"]["public"]

                await gateway.close()

                # Test with testnet disabled (production)
                mock_settings.exchange.binance_testnet = False
                gateway = BinanceGateway()
                await gateway.initialize()

                config_call = mock_binance.call_args[0][0]
                assert "hostname" not in config_call  # No hostname override for production
                assert "urls" not in config_call  # No URL override for production

                await gateway.close()

    @pytest.mark.asyncio
    async def test_decimal_precision(self, mock_settings, mock_ccxt_exchange):
        """Test that Decimal is used for all monetary calculations."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Test with precise decimal values
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="limit",
                    price=Decimal("50000.123456789"),
                    quantity=Decimal("0.000123456789"),
                    client_order_id="precise_order",
                )

                mock_ccxt_exchange.create_order.return_value = {
                    "id": "12347",
                    "clientOrderId": "precise_order",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "type": "limit",
                    "status": "open",
                    "price": 50000.123456789,
                    "amount": 0.000123456789,
                    "filled": 0,
                    "timestamp": datetime.now().timestamp() * 1000,
                }

                response = await gateway.place_order(request)

                # Verify Decimal types are preserved
                assert isinstance(request.price, Decimal)
                assert isinstance(request.quantity, Decimal)

                # Verify ccxt receives float conversion
                mock_ccxt_exchange.create_order.assert_called_once()
                call_args = mock_ccxt_exchange.create_order.call_args
                assert isinstance(call_args[1]["price"], float)
                assert isinstance(call_args[1]["amount"], float)

    @pytest.mark.asyncio
    async def test_idempotency_key(self, mock_settings, mock_ccxt_exchange):
        """Test that every order has a client_order_id for idempotency."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Test order WITH client_order_id
                request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="market",
                    quantity=Decimal("0.001"),
                    client_order_id="my_idempotent_key",
                )

                mock_ccxt_exchange.create_order.return_value = {
                    "id": "12348",
                    "clientOrderId": "my_idempotent_key",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "type": "market",
                    "status": "closed",
                    "price": 50000,
                    "amount": 0.001,
                    "filled": 0.001,
                    "timestamp": datetime.now().timestamp() * 1000,
                }

                await gateway.place_order(request)

                # Verify client_order_id was passed to params
                call_args = mock_ccxt_exchange.create_order.call_args
                assert "clientOrderId" in call_args[1]["params"]
                assert call_args[1]["params"]["clientOrderId"] == "my_idempotent_key"

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self, mock_settings, mock_ccxt_exchange):
        """Test that rate limiter is applied to API calls."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange):
                gateway = BinanceGateway()

                # Spy on rate limiter
                with patch.object(gateway.rate_limiter, "check_and_wait", new=AsyncMock()) as mock_rate_check:
                    # Place an order
                    request = OrderRequest(
                        symbol="BTC/USDT",
                        side="buy",
                        type="market",
                        quantity=Decimal("0.001"),
                    )

                    mock_ccxt_exchange.create_order.return_value = {
                        "id": "12349",
                        "symbol": "BTC/USDT",
                        "side": "buy",
                        "type": "market",
                        "status": "closed",
                        "price": 50000,
                        "amount": 0.001,
                        "filled": 0.001,
                        "timestamp": datetime.now().timestamp() * 1000,
                    }

                    await gateway.place_order(request)

                    # Verify rate limiter was called
                    mock_rate_check.assert_called_once_with("POST", "/api/v3/order")

                    # Check order status
                    await gateway.get_order_status("12349", "BTC/USDT")

                    # Verify rate limiter was called for status check
                    assert mock_rate_check.call_count == 2
                    mock_rate_check.assert_called_with("GET", "/api/v3/order")

    @pytest.mark.asyncio
    async def test_mock_mode(self, mock_settings):
        """Test gateway in mock mode for testing."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            # Enable mock mode
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()

            # Place order in mock mode
            request = OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                type="limit",
                price=Decimal("50000"),
                quantity=Decimal("0.001"),
                client_order_id="mock_test",
            )

            response = await gateway.place_order(request)

            # Verify mock response
            assert response.order_id == "mock_order_001"
            assert response.client_order_id == "mock_test"
            assert response.status == "open"

            # Cancel order in mock mode
            result = await gateway.cancel_order("mock_order_001", "BTC/USDT")
            assert result is True

            # Check status in mock mode
            status = await gateway.get_order_status("mock_order_001", "BTC/USDT")
            assert status.status == "filled"  # Mock always returns filled

            await gateway.close()
