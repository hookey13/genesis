"""
Binance API Gateway wrapper for Project GENESIS.

This module provides a high-level interface to the Binance exchange through
the ccxt library, with built-in connection pooling, credential management,
and request/response validation.
"""

from decimal import Decimal
from typing import TYPE_CHECKING, Any
import asyncio
import aiohttp
import time

import ccxt.async_support as ccxt
import structlog

from config.settings import get_settings
from genesis.exchange.models import (
    AccountBalance,
    MarketTicker,
    OrderBook,
    OrderRequest,
    OrderResponse,
)
from genesis.exchange.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from genesis.exchange.mock_exchange import MockExchange


logger = structlog.get_logger(__name__)


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
        self.exchange: ccxt.Exchange | None = None
        self.mock_exchange: MockExchange | None = None
        self.rate_limiter = RateLimiter()
        self._initialized = False
        
        # Connection pool configuration
        self._session: aiohttp.ClientSession | None = None
        self._connection_pool_size = 10  # Max connections per host
        self._connection_timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=5,  # Connection timeout
            sock_connect=5,  # Socket connection timeout
            sock_read=25  # Socket read timeout
        )
        self._keep_alive_timeout = 30
        self._connection_metrics = {
            "active_connections": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "connection_reuses": 0,
            "last_health_check": 0
        }
        self._retry_config = {
            "max_retries": 3,
            "base_delay": 1,  # seconds
            "max_delay": 30,  # seconds
            "exponential_base": 2
        }

        logger.info(
            "Initializing BinanceGateway",
            mock_mode=self.mock_mode,
            testnet=self.settings.exchange.binance_testnet,
            connection_pool_size=self._connection_pool_size,
        )

    async def initialize(self) -> None:
        """Initialize the exchange connection."""
        if self._initialized:
            return

        try:
            if self.mock_mode:
                logger.info("Initializing in mock mode")
                # Import here to avoid circular import
                from genesis.exchange.mock_exchange import MockExchange

                self.mock_exchange = MockExchange()
                self._initialized = True
                return

            # Create persistent session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self._connection_pool_size,
                limit_per_host=self._connection_pool_size,
                ttl_dns_cache=300,  # DNS cache for 5 minutes
                keepalive_timeout=self._keep_alive_timeout,
                force_close=False,  # Reuse connections
                enable_cleanup_closed=True
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self._connection_timeout,
                headers={
                    "Connection": "keep-alive",
                    "Keep-Alive": f"timeout={self._keep_alive_timeout}"
                }
            )
            
            # Configure exchange with persistent session
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
                "session": self._session,  # Use our persistent session
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
                markets_loaded=len(self.exchange.markets),
            )

        except Exception as e:
            logger.error("Failed to initialize BinanceGateway", error=str(e))
            raise

    async def close(self) -> None:
        """Close the exchange connection and cleanup resources."""
        try:
            if self.exchange:
                await self.exchange.close()
            
            # Close persistent session
            if self._session:
                await self._session.close()
                self._session = None
            
            self._initialized = False
            
            logger.info(
                "BinanceGateway closed",
                total_requests=self._connection_metrics["total_requests"],
                connection_reuses=self._connection_metrics["connection_reuses"]
            )
        except Exception as e:
            logger.error("Error closing BinanceGateway", error=str(e))

    async def get_account_balance(self) -> dict[str, AccountBalance]:
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
                        locked=info["locked"],
                        total=Decimal(info["free"]) + Decimal(info["locked"]),
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
                params["stopPrice"] = str(request.stop_price)  # Binance API accepts string format

            # Handle advanced order types
            order_type = request.type
            if request.type in ["FOK", "IOC"]:
                # FOK and IOC are handled via timeInForce parameter
                params["timeInForce"] = request.type
                order_type = "LIMIT"  # Both require limit orders
            elif request.type == "POST_ONLY":
                # Binance uses LIMIT_MAKER for post-only orders
                order_type = "LIMIT_MAKER"
            elif request.type == "LIMIT_MAKER":
                # Direct LIMIT_MAKER order type
                order_type = "LIMIT_MAKER"

            logger.info(
                "Placing order",
                symbol=request.symbol,
                side=request.side,
                type=request.type,
                quantity=str(request.quantity),
            )

            if self.mock_mode:
                # Return mock order response
                from datetime import datetime

                return OrderResponse(
                    order_id="mock_order_001",
                    client_order_id=request.client_order_id,
                    symbol=request.symbol,
                    side=request.side,
                    type=request.type,
                    status="open",
                    price=request.price,
                    quantity=request.quantity,
                    filled_quantity=Decimal("0"),
                    created_at=datetime.now(),
                    updated_at=None,
                )

            # Apply rate limiting
            await self.rate_limiter.check_and_wait("POST", "/api/v3/order")

            # Place the order
            order = await self.exchange.create_order(
                symbol=request.symbol,
                type=order_type,  # Use the mapped order type
                side=request.side,
                amount=str(request.quantity),  # CCXT handles string to appropriate format
                price=str(request.price) if request.price else None,
                params=params,
            )

            # Convert to response model
            from datetime import datetime

            response = OrderResponse(
                order_id=order["id"],
                client_order_id=order.get("clientOrderId"),
                symbol=order["symbol"],
                side=order["side"],
                type=order["type"],
                status=order["status"],
                price=order["price"],
                quantity=order["amount"],
                filled_quantity=order["filled"],
                created_at=datetime.fromtimestamp(order["timestamp"] / 1000),
                updated_at=None,
            )

            logger.info(
                "Order placed successfully",
                order_id=response.order_id,
                status=response.status,
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

    async def get_open_orders(
        self, symbol: str | None = None
    ) -> list[OrderResponse]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol to filter orders

        Returns:
            List of open orders
        """
        await self.initialize()

        try:
            if self.mock_mode:
                return []

            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/openOrders")

            orders = await self.exchange.fetch_open_orders(symbol)

            from datetime import datetime

            result = []
            for order in orders:
                result.append(
                    OrderResponse(
                        order_id=order["id"],
                        client_order_id=order.get("clientOrderId"),
                        symbol=order["symbol"],
                        side=order["side"],
                        type=order["type"],
                        status=order["status"],
                        price=order["price"],
                        quantity=order["amount"],
                        filled_quantity=order["filled"],
                        created_at=datetime.fromtimestamp(order["timestamp"] / 1000),
                        updated_at=(
                            datetime.fromtimestamp(order["lastUpdateTimestamp"] / 1000)
                            if order.get("lastUpdateTimestamp")
                            else None
                        ),
                    )
                )

            return result

        except Exception as e:
            logger.error("Failed to get open orders", symbol=symbol, error=str(e))
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
                from datetime import datetime

                return OrderResponse(
                    order_id=order_id,
                    client_order_id=None,
                    symbol=symbol,
                    side="buy",
                    type="limit",
                    status="filled",
                    price=Decimal("50000"),
                    quantity=Decimal("0.001"),
                    filled_quantity=Decimal("0.001"),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )

            # Apply rate limiting
            await self.rate_limiter.check_and_wait("GET", "/api/v3/order")

            order = await self.exchange.fetch_order(order_id, symbol)

            from datetime import datetime

            return OrderResponse(
                order_id=order["id"],
                client_order_id=order.get("clientOrderId"),
                symbol=order["symbol"],
                side=order["side"],
                type=order["type"],
                status=order["status"],
                price=order["price"],
                quantity=order["amount"],
                filled_quantity=order["filled"],
                created_at=datetime.fromtimestamp(order["timestamp"] / 1000),
                updated_at=(
                    datetime.fromtimestamp(order["lastUpdateTimestamp"] / 1000)
                    if order.get("lastUpdateTimestamp")
                    else None
                ),
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
                from datetime import datetime

                return OrderBook(
                    symbol=symbol,
                    bids=[
                        [Decimal("50000"), Decimal("1.5")],
                        [Decimal("49999"), Decimal("2.0")],
                    ],
                    asks=[
                        [Decimal("50001"), Decimal("1.2")],
                        [Decimal("50002"), Decimal("1.8")],
                    ],
                    timestamp=datetime.now(),
                )

            # Apply rate limiting
            await self.rate_limiter.check_and_wait(
                "GET", "/api/v3/depth", {"limit": limit}
            )

            orderbook = await self.exchange.fetch_order_book(symbol, limit)

            from datetime import datetime

            return OrderBook(
                symbol=symbol,
                bids=orderbook["bids"][:limit],
                asks=orderbook["asks"][:limit],
                timestamp=datetime.fromtimestamp(orderbook["timestamp"] / 1000),
            )

        except Exception as e:
            logger.error("Failed to fetch order book", symbol=symbol, error=str(e))
            raise

    async def get_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[dict[str, Any]]:
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
                        "volume": Decimal("100"),
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
                    "volume": Decimal(str(k[5])),
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
                return MarketTicker(
                    symbol=symbol,
                    last_price=Decimal("50000.5"),
                    bid_price=Decimal("50000"),
                    ask_price=Decimal("50001"),
                    volume_24h=Decimal("1500"),
                    quote_volume_24h=Decimal("75000750"),
                    price_change_percent=Decimal("2.5"),
                    high_24h=Decimal("51000"),
                    low_24h=Decimal("49000"),
                )

            # Apply rate limiting
            await self.rate_limiter.check_and_wait(
                "GET", "/api/v3/ticker/24hr", {"symbol": symbol}
            )

            ticker = await self.exchange.fetch_ticker(symbol)

            return MarketTicker(
                symbol=ticker["symbol"],
                last_price=ticker["last"],
                bid_price=ticker["bid"],
                ask_price=ticker["ask"],
                volume_24h=ticker["baseVolume"],
                quote_volume_24h=ticker["quoteVolume"],
                price_change_percent=ticker["percentage"],
                high_24h=ticker["high"],
                low_24h=ticker["low"],
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

    async def place_post_only_order(
        self, request: OrderRequest, max_retries: int = 3
    ) -> OrderResponse:
        """
        Place a post-only order with retry logic.

        Post-only orders are rejected if they would immediately match.
        This method retries with adjusted prices to ensure maker execution.

        Args:
            request: Order request (must be a limit order)
            max_retries: Maximum number of retry attempts

        Returns:
            Order response with exchange details
        """
        await self.initialize()

        if request.type not in ["LIMIT", "POST_ONLY", "LIMIT_MAKER"]:
            raise ValueError("Post-only orders must be limit orders")

        original_price = request.price
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Set order type to post-only
                request.type = "POST_ONLY"

                # Try to place the order
                response = await self.place_order(request)

                logger.info(
                    "Post-only order placed successfully",
                    order_id=response.order_id,
                    retry_count=retry_count,
                )
                return response

            except Exception as e:
                error_msg = str(e).lower()

                # Check if order was rejected for crossing the spread
                if "would match" in error_msg or "post only" in error_msg:
                    retry_count += 1

                    if retry_count >= max_retries:
                        logger.warning(
                            "Post-only order failed after max retries",
                            symbol=request.symbol,
                            original_price=str(original_price),
                            retries=retry_count,
                        )
                        raise

                    # Adjust price to avoid crossing spread
                    # For buy orders, decrease price slightly
                    # For sell orders, increase price slightly
                    adjustment = Decimal("0.0001")  # 0.01% adjustment

                    if request.side.upper() == "BUY":
                        request.price = request.price * (Decimal("1") - adjustment)
                    else:
                        request.price = request.price * (Decimal("1") + adjustment)

                    logger.info(
                        "Retrying post-only order with adjusted price",
                        symbol=request.symbol,
                        side=request.side,
                        new_price=str(request.price),
                        retry=retry_count,
                    )

                    # Small delay before retry
                    import asyncio

                    await asyncio.sleep(0.5)

                else:
                    # Different error, re-raise
                    raise

    async def validate_connection(self) -> bool:
        """
        Validate the exchange connection is working.
        
        Returns:
            True if connection is valid and working
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Try to get server time as a simple connectivity check
            server_time = await self.get_server_time()

            if server_time and server_time > 0:
                logger.info("Exchange connection validated successfully", server_time=server_time)
                return True
            else:
                logger.error("Exchange connection validation failed - invalid server time")
                return False

        except Exception as e:
            logger.error("Exchange connection validation failed", error=str(e))
            return False

    async def execute_order(self, order) -> dict[str, Any]:
        """
        Execute an order through the exchange.
        
        Args:
            order: Order model to execute
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Convert order model to OrderRequest
            request = OrderRequest(
                symbol=order.symbol,
                side=order.side.value,
                type=order.type.value,
                quantity=order.quantity,
                price=order.price,
                client_order_id=order.client_order_id
            )

            # Track execution time
            import time
            start_time = time.time()

            # Place the order
            response = await self.place_order(request)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            if response.status in ["FILLED", "PARTIALLY_FILLED"]:
                return {
                    "success": True,
                    "exchange_order_id": response.order_id,
                    "fill_price": response.price or response.average_price,
                    "filled_quantity": response.filled_quantity,
                    "status": response.status,
                    "latency_ms": latency_ms
                }
            else:
                return {
                    "success": False,
                    "error": f"Order not filled: {response.status}",
                    "exchange_order_id": response.order_id,
                    "status": response.status,
                    "latency_ms": latency_ms
                }

        except Exception as e:
            logger.error("Order execution failed", error=str(e), order_id=order.order_id)
            return {
                "success": False,
                "error": str(e)
            }


    async def _execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Function result
        """
        last_error = None
        
        for attempt in range(self._retry_config["max_retries"]):
            try:
                # Track metrics
                self._connection_metrics["total_requests"] += 1
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Track successful reuse if this is a retry
                if attempt > 0:
                    self._connection_metrics["connection_reuses"] += 1
                    
                return result
                
            except Exception as e:
                last_error = e
                self._connection_metrics["failed_requests"] += 1
                
                # Check if we should retry
                error_str = str(e).lower()
                if any(err in error_str for err in ["timeout", "connection", "network", "refused"]):
                    if attempt < self._retry_config["max_retries"] - 1:
                        # Calculate backoff delay
                        delay = min(
                            self._retry_config["base_delay"] * (self._retry_config["exponential_base"] ** attempt),
                            self._retry_config["max_delay"]
                        )
                        
                        logger.warning(
                            "Request failed, retrying with backoff",
                            attempt=attempt + 1,
                            delay=delay,
                            error=str(e)
                        )
                        
                        await asyncio.sleep(delay)
                        continue
                
                # Non-retryable error or max retries reached
                raise
        
        # Max retries exceeded
        logger.error(
            "Max retries exceeded",
            retries=self._retry_config["max_retries"],
            last_error=str(last_error)
        )
        raise last_error

    def get_connection_metrics(self) -> dict:
        """
        Get current connection pool metrics.
        
        Returns:
            Dictionary with connection pool statistics
        """
        metrics = self._connection_metrics.copy()
        
        # Add session-specific metrics if available
        if self._session and self._session.connector:
            connector = self._session.connector
            metrics.update({
                "active_connections": len(connector._acquired),
                "available_connections": len(connector._available),
                "connection_limit": connector.limit,
                "connection_limit_per_host": connector.limit_per_host,
            })
        
        return metrics

    async def monitor_connection_health(self) -> bool:
        """
        Monitor and log connection pool health.
        
        Returns:
            True if connection pool is healthy
        """
        try:
            current_time = time.time()
            
            # Only check every 60 seconds
            if current_time - self._connection_metrics["last_health_check"] < 60:
                return True
            
            self._connection_metrics["last_health_check"] = current_time
            
            # Get current metrics
            metrics = self.get_connection_metrics()
            
            # Check connection pool health
            if self._session and self._session.connector:
                connector = self._session.connector
                
                # Calculate usage percentage
                usage_pct = (metrics.get("active_connections", 0) / 
                           self._connection_pool_size) * 100
                
                # Log metrics
                logger.info(
                    "Connection pool health check",
                    active=metrics.get("active_connections", 0),
                    available=metrics.get("available_connections", 0),
                    usage_pct=f"{usage_pct:.1f}%",
                    total_requests=metrics["total_requests"],
                    failed_requests=metrics["failed_requests"],
                    reuses=metrics["connection_reuses"]
                )
                
                # Warn if pool is nearly exhausted
                if usage_pct > 80:
                    logger.warning(
                        "Connection pool usage high",
                        usage_pct=f"{usage_pct:.1f}%"
                    )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error("Connection health check failed", error=str(e))
            return False

# Alias for compatibility
ExchangeGateway = BinanceGateway
