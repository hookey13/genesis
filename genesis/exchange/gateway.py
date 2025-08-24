"""
Binance API Gateway wrapper for Project GENESIS.

This module provides a high-level interface to the Binance exchange through
the ccxt library, with built-in connection pooling, credential management,
and request/response validation.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import ccxt.async_support as ccxt
from pydantic import BaseModel, Field, field_validator
import structlog

from config.settings import get_settings
from genesis.exchange.rate_limiter import RateLimiter
from genesis.exchange.mock_exchange import MockExchange


logger = structlog.get_logger(__name__)


class OrderRequest(BaseModel):
    """Validated order placement request."""
    
    symbol: str = Field(..., description="Trading pair (e.g., BTC/USDT)")
    side: str = Field(..., pattern="^(buy|sell)$")
    type: str = Field(..., pattern="^(market|limit|stop_limit)$")
    quantity: Decimal = Field(..., gt=0)
    price: Optional[Decimal] = Field(None, gt=0)
    stop_price: Optional[Decimal] = Field(None, gt=0)
    client_order_id: Optional[str] = Field(None, min_length=1, max_length=36)
    
    @field_validator("quantity", "price", "stop_price", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v
    
    @field_validator("price")
    @classmethod
    def validate_price_for_limit(cls, v, info):
        """Ensure price is provided for limit orders."""
        if info.data.get("type") == "limit" and v is None:
            raise ValueError("Price required for limit orders")
        return v


class OrderResponse(BaseModel):
    """Validated order response from exchange."""
    
    order_id: str
    exchange_order_id: str
    symbol: str
    side: str
    type: str
    status: str
    price: Optional[Decimal]
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    timestamp: int
    client_order_id: Optional[str] = None
    
    @field_validator("quantity", "filled_quantity", "price", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v


class AccountBalance(BaseModel):
    """Account balance information."""
    
    asset: str
    free: Decimal
    used: Decimal
    total: Decimal
    
    @field_validator("free", "used", "total", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))


class MarketTicker(BaseModel):
    """Market ticker data."""
    
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: int
    
    @field_validator("bid", "ask", "last", "volume", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))


class OrderBook(BaseModel):
    """Order book data."""
    
    symbol: str
    bids: List[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    asks: List[tuple[Decimal, Decimal]]
    timestamp: int
    
    @field_validator("bids", "asks", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert price/quantity to Decimal."""
        return [(Decimal(str(price)), Decimal(str(qty))) for price, qty in v]


class BinanceGateway:
    """
    High-level gateway for Binance exchange interaction.
    
    Provides a unified interface for all exchange operations with built-in
    validation, error handling, and connection management.
    """
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize the Binance gateway.
        
        Args:
            mock_mode: If True, use mock exchange for testing
        """
        self.settings = get_settings()
        self.mock_mode = mock_mode or self.settings.development.use_mock_exchange
        self.exchange: Optional[ccxt.Exchange] = None
        self.mock_exchange: Optional[MockExchange] = None
        self.rate_limiter = RateLimiter()
        self._initialized = False
        
        logger.info(
            "Initializing BinanceGateway",
            mock_mode=self.mock_mode,
            testnet=self.settings.exchange.binance_testnet
        )
    
    async def initialize(self) -> None:
        """Initialize the exchange connection."""
        if self._initialized:
            return
        
        try:
            if self.mock_mode:
                logger.info("Initializing in mock mode")
                self.mock_exchange = MockExchange()
                self._initialized = True
                return
            
            # Configure exchange
            config = {
                "apiKey": self.settings.exchange.binance_api_key.get_secret_value(),
                "secret": self.settings.exchange.binance_api_secret.get_secret_value(),
                "enableRateLimit": True,
                "rateLimit": self.settings.exchange.exchange_rate_limit,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                    "recvWindow": 5000,
                },
                "timeout": 30000,  # 30 seconds total timeout
                "session": True,  # Enable connection pooling
            }
            
            # Use testnet if configured
            if self.settings.exchange.binance_testnet:
                config["hostname"] = "testnet.binance.vision"
                config["urls"] = {
                    "api": {
                        "public": "https://testnet.binance.vision/api",
                        "private": "https://testnet.binance.vision/api",
                    }
                }
            
            # Create exchange instance
            self.exchange = ccxt.binance(config)
            
            # Load markets
            await self.exchange.load_markets()
            
            self._initialized = True
            logger.info(
                "BinanceGateway initialized successfully",
                markets_loaded=len(self.exchange.markets)
            )
            
        except Exception as e:
            logger.error("Failed to initialize BinanceGateway", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self._initialized = False
            logger.info("BinanceGateway closed")
    
    async def get_account_balance(self) -> Dict[str, AccountBalance]:
        """
        Fetch account balance information.
        
        Returns:
            Dictionary mapping asset symbols to balance information
        """
        await self.initialize()
        
        try:
            if self.mock_mode and self.mock_exchange:
                return await self.mock_exchange.fetch_balance()
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/account")
            
            balance = await self.exchange.fetch_balance()
            
            result = {}
            for asset, info in balance["info"]["balances"].items():
                if info["free"] != "0" or info["locked"] != "0":
                    result[asset] = AccountBalance(
                        asset=asset,
                        free=info["free"],
                        used=info["locked"],
                        total=Decimal(info["free"]) + Decimal(info["locked"])
                    )
            
            logger.info("Fetched account balance", assets=list(result.keys()))
            return result
            
        except Exception as e:
            logger.error("Failed to fetch account balance", error=str(e))
            raise
    
    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """
        Place an order on the exchange.
        
        Args:
            request: Validated order request
            
        Returns:
            Order response with exchange details
        """
        await self.initialize()
        
        try:
            params = {}
            if request.client_order_id:
                params["clientOrderId"] = request.client_order_id
            
            if request.stop_price:
                params["stopPrice"] = float(request.stop_price)
            
            logger.info(
                "Placing order",
                symbol=request.symbol,
                side=request.side,
                type=request.type,
                quantity=str(request.quantity)
            )
            
            if self.mock_mode:
                # Return mock order response
                import time
                return OrderResponse(
                    order_id="mock_order_001",
                    exchange_order_id="MOCK001",
                    symbol=request.symbol,
                    side=request.side,
                    type=request.type,
                    status="open",
                    price=request.price,
                    quantity=request.quantity,
                    filled_quantity=Decimal("0"),
                    timestamp=int(time.time() * 1000),
                    client_order_id=request.client_order_id
                )
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("POST", "/api/v3/order")
            
            # Place the order
            order = await self.exchange.create_order(
                symbol=request.symbol,
                type=request.type,
                side=request.side,
                amount=float(request.quantity),
                price=float(request.price) if request.price else None,
                params=params
            )
            
            # Convert to response model
            response = OrderResponse(
                order_id=order["id"],
                exchange_order_id=order["info"]["orderId"],
                symbol=order["symbol"],
                side=order["side"],
                type=order["type"],
                status=order["status"],
                price=order["price"],
                quantity=order["amount"],
                filled_quantity=order["filled"],
                timestamp=order["timestamp"],
                client_order_id=order.get("clientOrderId")
            )
            
            logger.info(
                "Order placed successfully",
                order_id=response.order_id,
                status=response.status
            )
            
            return response
            
        except Exception as e:
            logger.error("Failed to place order", error=str(e))
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair
            
        Returns:
            True if cancellation successful
        """
        await self.initialize()
        
        try:
            logger.info("Cancelling order", order_id=order_id, symbol=symbol)
            
            if self.mock_mode:
                return True
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("DELETE", "/api/v3/order")
            
            result = await self.exchange.cancel_order(order_id, symbol)
            
            logger.info("Order cancelled successfully", order_id=order_id)
            return result["status"] == "canceled"
            
        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            raise
    
    async def get_order_status(self, order_id: str, symbol: str) -> OrderResponse:
        """
        Get the status of an existing order.
        
        Args:
            order_id: Exchange order ID
            symbol: Trading pair
            
        Returns:
            Order response with current status
        """
        await self.initialize()
        
        try:
            if self.mock_mode:
                import time
                return OrderResponse(
                    order_id=order_id,
                    exchange_order_id=order_id,
                    symbol=symbol,
                    side="buy",
                    type="limit",
                    status="filled",
                    price=Decimal("50000"),
                    quantity=Decimal("0.001"),
                    filled_quantity=Decimal("0.001"),
                    timestamp=int(time.time() * 1000)
                )
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/order")
            
            order = await self.exchange.fetch_order(order_id, symbol)
            
            return OrderResponse(
                order_id=order["id"],
                exchange_order_id=order["info"]["orderId"],
                symbol=order["symbol"],
                side=order["side"],
                type=order["type"],
                status=order["status"],
                price=order["price"],
                quantity=order["amount"],
                filled_quantity=order["filled"],
                timestamp=order["timestamp"],
                client_order_id=order.get("clientOrderId")
            )
            
        except Exception as e:
            logger.error("Failed to get order status", order_id=order_id, error=str(e))
            raise
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Fetch the order book for a symbol.
        
        Args:
            symbol: Trading pair
            limit: Number of price levels to fetch
            
        Returns:
            Order book data
        """
        await self.initialize()
        
        try:
            if self.mock_mode:
                import time
                return OrderBook(
                    symbol=symbol,
                    bids=[(Decimal("50000"), Decimal("1.5")), (Decimal("49999"), Decimal("2.0"))],
                    asks=[(Decimal("50001"), Decimal("1.2")), (Decimal("50002"), Decimal("1.8"))],
                    timestamp=int(time.time() * 1000)
                )
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/depth", {"limit": limit})
            
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            
            return OrderBook(
                symbol=symbol,
                bids=orderbook["bids"][:limit],
                asks=orderbook["asks"][:limit],
                timestamp=orderbook["timestamp"]
            )
            
        except Exception as e:
            logger.error("Failed to fetch order book", symbol=symbol, error=str(e))
            raise
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical kline/candlestick data.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1m, 5m, 1h, etc.)
            limit: Number of klines to fetch
            
        Returns:
            List of kline data
        """
        await self.initialize()
        
        try:
            if self.mock_mode:
                import time
                now = int(time.time() * 1000)
                return [
                    {
                        "timestamp": now - (i * 60000),
                        "open": Decimal("50000") + Decimal(i),
                        "high": Decimal("50100") + Decimal(i),
                        "low": Decimal("49900") + Decimal(i),
                        "close": Decimal("50050") + Decimal(i),
                        "volume": Decimal("100")
                    }
                    for i in range(limit)
                ]
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/klines")
            
            klines = await self.exchange.fetch_ohlcv(symbol, interval, limit=limit)
            
            return [
                {
                    "timestamp": k[0],
                    "open": Decimal(str(k[1])),
                    "high": Decimal(str(k[2])),
                    "low": Decimal(str(k[3])),
                    "close": Decimal(str(k[4])),
                    "volume": Decimal(str(k[5]))
                }
                for k in klines
            ]
            
        except Exception as e:
            logger.error("Failed to fetch klines", symbol=symbol, error=str(e))
            raise
    
    async def get_ticker(self, symbol: str) -> MarketTicker:
        """
        Fetch 24hr ticker statistics.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Market ticker data
        """
        await self.initialize()
        
        try:
            if self.mock_mode:
                import time
                return MarketTicker(
                    symbol=symbol,
                    bid=Decimal("50000"),
                    ask=Decimal("50001"),
                    last=Decimal("50000.5"),
                    volume=Decimal("1500"),
                    timestamp=int(time.time() * 1000)
                )
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/ticker/24hr", {"symbol": symbol})
            
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return MarketTicker(
                symbol=ticker["symbol"],
                bid=ticker["bid"],
                ask=ticker["ask"],
                last=ticker["last"],
                volume=ticker["baseVolume"],
                timestamp=ticker["timestamp"]
            )
            
        except Exception as e:
            logger.error("Failed to fetch ticker", symbol=symbol, error=str(e))
            raise
    
    async def get_server_time(self) -> int:
        """
        Get the current server time.
        
        Returns:
            Server timestamp in milliseconds
        """
        await self.initialize()
        
        try:
            if self.mock_mode:
                import time
                return int(time.time() * 1000)
            
            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/time")
            
            # Use ccxt's built-in method
            return await self.exchange.fetch_time()
            
        except Exception as e:
            logger.error("Failed to fetch server time", error=str(e))
            raise