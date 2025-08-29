"""
Exchange client interface matching specification requirements.

Provides unified interface for REST and WebSocket operations with
proper event emission, idempotency, and reconciliation support.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import ccxt.async_support as ccxt

from genesis.exchange.circuit_breaker import CircuitBreaker
from genesis.exchange.events import (
    ClockSkewEvent,
    EventBus,
    ExchangeHeartbeat,
    OrderAck,
    OrderCancel,
    OrderReject,
    ReconciliationEvent,
)
from genesis.exchange.exceptions import (
    ExchangeError,
    OrderNotFoundError,
    RateLimitError,
)
from genesis.exchange.exceptions import (
    MaintenanceError as ExchangeNotAvailable,
)
from genesis.exchange.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class ExchangeClient:
    """
    Unified exchange client interface with event-driven architecture.

    Provides REST API operations with proper idempotency, rate limiting,
    circuit breaking, and event emission.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        event_bus: EventBus | None = None,
    ):
        """Initialize exchange client."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.event_bus = event_bus or EventBus()

        # Initialize CCXT exchange
        self.exchange = self._init_exchange()

        # Components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=30, expected_exception=ExchangeError
        )
        self.rate_limiter = RateLimiter(max_requests=1200, time_window=60)

        # State tracking
        self._pending_orders: dict[str, dict] = {}
        self._idempotency_cache: dict[str, dict] = {}
        self._last_reconciliation = None
        self._clock_drift_ms = 0

    def _init_exchange(self) -> ccxt.binance:
        """Initialize CCXT exchange instance."""
        exchange_class = ccxt.binance

        config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "rateLimit": 50,  # 50ms between requests
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
                "recvWindow": 5000,
            },
        }

        if self.testnet:
            config["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }

        return exchange_class(config)

    async def initialize(self):
        """Initialize exchange connection and perform startup checks."""
        try:
            # Load markets
            await self.exchange.load_markets()

            # Check time sync
            await self.check_time_sync()

            # Perform reconciliation
            await self.reconcile_positions()

            logger.info("Exchange client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise ExchangeNotAvailable(f"Exchange initialization failed: {e}")

    async def check_time_sync(self) -> int:
        """
        Check time synchronization with exchange.

        Returns:
            Clock drift in milliseconds
        """
        try:
            server_time = await self.exchange.fetch_time()
            local_time = int(time.time() * 1000)
            drift_ms = abs(local_time - server_time)

            self._clock_drift_ms = drift_ms

            if drift_ms > 1000:  # More than 1 second
                event = ClockSkewEvent(
                    timestamp=datetime.now(UTC),
                    sequence=0,
                    local_time=datetime.fromtimestamp(
                        local_time / 1000, tz=UTC
                    ),
                    server_time=datetime.fromtimestamp(
                        server_time / 1000, tz=UTC
                    ),
                    skew_ms=drift_ms,
                    threshold_ms=1000,
                    action_taken="WARNING" if drift_ms < 5000 else "HALT_TRADING",
                )
                self.event_bus.publish(event)

                if drift_ms > 5000:  # More than 5 seconds
                    raise ExchangeError(f"Clock skew too high: {drift_ms}ms")

            return drift_ms

        except Exception as e:
            logger.error(f"Time sync check failed: {e}")
            return 0

    def _generate_client_order_id(self, prefix: str = "twap") -> str:
        """Generate unique client order ID with prefix."""
        return f"{prefix}_{uuid4().hex[:16]}"

    def _check_idempotency(self, client_order_id: str) -> dict | None:
        """Check if order was already submitted."""
        return self._idempotency_cache.get(client_order_id)

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "GTC",
    ) -> dict:
        """
        Place order with idempotency and event emission.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            order_type: "limit" or "market"
            quantity: Order quantity
            price: Limit price (required for limit orders)
            client_order_id: Optional client order ID
            time_in_force: Time in force (GTC, IOC, FOK)

        Returns:
            Order response dictionary
        """
        # Generate or validate client order ID
        if not client_order_id:
            client_order_id = self._generate_client_order_id()

        # Check idempotency
        cached_result = self._check_idempotency(client_order_id)
        if cached_result:
            logger.info(f"Order {client_order_id} already exists (idempotent)")
            return cached_result

        # Validate inputs
        if order_type == "limit" and price is None:
            raise ValueError("Price required for limit orders")

        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitError("Rate limit exceeded")

        # Circuit breaker
        @self.circuit_breaker
        async def _place_order():
            params = {"newClientOrderId": client_order_id, "timeInForce": time_in_force}

            if order_type == "market":
                result = await self.exchange.create_market_order(
                    symbol, side, float(quantity), params=params
                )
            else:
                result = await self.exchange.create_limit_order(
                    symbol, side, float(quantity), float(price), params=params
                )

            return result

        try:
            # Place order
            order_result = await _place_order()

            # Cache for idempotency
            self._idempotency_cache[client_order_id] = order_result
            self._pending_orders[client_order_id] = order_result

            # Emit OrderAck event
            event = OrderAck(
                timestamp=datetime.now(UTC),
                sequence=0,
                client_order_id=client_order_id,
                exchange_order_id=str(order_result["id"]),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                time_in_force=time_in_force,
                status=order_result["status"].upper(),
            )
            self.event_bus.publish(event)

            logger.info(f"Order placed: {client_order_id} -> {order_result['id']}")
            return order_result

        except Exception as e:
            # Emit OrderReject event
            event = OrderReject(
                timestamp=datetime.now(UTC),
                sequence=0,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                reject_reason=str(e),
            )
            self.event_bus.publish(event)

            logger.error(f"Order placement failed: {e}")
            raise

    async def cancel_order(
        self,
        symbol: str,
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
    ) -> dict:
        """
        Cancel order with post-cancel verification.

        Args:
            symbol: Trading pair
            client_order_id: Client order ID
            exchange_order_id: Exchange order ID

        Returns:
            Cancellation result
        """
        if not client_order_id and not exchange_order_id:
            raise ValueError("Either client_order_id or exchange_order_id required")

        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitError("Rate limit exceeded")

        @self.circuit_breaker
        async def _cancel_order():
            if client_order_id:
                # Try to find exchange order ID from cache
                pending = self._pending_orders.get(client_order_id)
                if pending:
                    exchange_order_id = pending.get("id")

            if not exchange_order_id:
                # Query open orders to find it
                open_orders = await self.exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    if order.get("clientOrderId") == client_order_id:
                        exchange_order_id = order["id"]
                        break

            if not exchange_order_id:
                raise OrderNotFoundError(f"Order not found: {client_order_id}")

            return await self.exchange.cancel_order(exchange_order_id, symbol)

        try:
            result = await _cancel_order()

            # Remove from pending
            if client_order_id in self._pending_orders:
                del self._pending_orders[client_order_id]

            # Verify cancellation with polling (up to 3 attempts)
            for attempt in range(3):
                await asyncio.sleep(0.5 * (attempt + 1))
                try:
                    order = await self.get_order(
                        symbol, client_order_id, exchange_order_id
                    )
                    if order["status"] in ["canceled", "expired", "rejected"]:
                        break
                except OrderNotFoundError:
                    break  # Order not found means it was canceled

            # Emit OrderCancel event
            event = OrderCancel(
                timestamp=datetime.now(UTC),
                sequence=0,
                client_order_id=client_order_id or "",
                exchange_order_id=exchange_order_id or "",
                symbol=symbol,
                reason="User requested",
                canceled_qty=Decimal(str(result.get("amount", 0))),
                executed_qty=Decimal(str(result.get("filled", 0))),
            )
            self.event_bus.publish(event)

            logger.info(f"Order canceled: {client_order_id or exchange_order_id}")
            return result

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            raise

    async def get_order(
        self,
        symbol: str,
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
    ) -> dict:
        """Get order status."""
        if not client_order_id and not exchange_order_id:
            raise ValueError("Either client_order_id or exchange_order_id required")

        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitError("Rate limit exceeded")

        @self.circuit_breaker
        async def _get_order():
            if exchange_order_id:
                return await self.exchange.fetch_order(exchange_order_id, symbol)

            # Try to find by client order ID
            orders = await self.exchange.fetch_orders(symbol, limit=100)
            for order in orders:
                if order.get("clientOrderId") == client_order_id:
                    return order

            raise OrderNotFoundError(f"Order not found: {client_order_id}")

        return await _get_order()

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Get all open orders."""
        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitError("Rate limit exceeded")

        @self.circuit_breaker
        async def _get_open_orders():
            return await self.exchange.fetch_open_orders(symbol)

        return await _get_open_orders()

    async def get_balance(self) -> dict[str, dict]:
        """Get account balances."""
        # Rate limiting
        if not await self.rate_limiter.acquire():
            raise RateLimitError("Rate limit exceeded")

        @self.circuit_breaker
        async def _get_balance():
            balance = await self.exchange.fetch_balance()
            return {
                asset: {
                    "free": Decimal(str(bal["free"])),
                    "used": Decimal(str(bal["used"])),
                    "total": Decimal(str(bal["total"])),
                }
                for asset, bal in balance.items()
                if bal["total"] > 0
            }

        return await _get_balance()

    async def reconcile_positions(self) -> dict:
        """
        Reconcile local positions with exchange.

        Returns:
            Reconciliation report
        """
        start_time = time.time()

        # Emit reconciliation start event
        start_event = ReconciliationEvent(
            timestamp=datetime.now(UTC), sequence=0, phase="START"
        )
        self.event_bus.publish(start_event)

        try:
            # Get exchange state
            open_orders = await self.get_open_orders()
            balances = await self.get_balance()

            # Update pending orders
            exchange_order_ids = {order["id"] for order in open_orders}
            local_order_ids = set(self._pending_orders.keys())

            # Find discrepancies
            missing_on_exchange = local_order_ids - exchange_order_ids
            missing_locally = exchange_order_ids - local_order_ids

            corrections = 0

            # Remove stale local orders
            for order_id in missing_on_exchange:
                del self._pending_orders[order_id]
                corrections += 1

            # Add missing local orders
            for order in open_orders:
                client_id = order.get("clientOrderId")
                if client_id and client_id not in self._pending_orders:
                    self._pending_orders[client_id] = order
                    corrections += 1

            duration_ms = int((time.time() - start_time) * 1000)

            # Emit reconciliation complete event
            complete_event = ReconciliationEvent(
                timestamp=datetime.now(UTC),
                sequence=0,
                phase="COMPLETE",
                orders_reconciled=len(open_orders),
                positions_reconciled=len(balances),
                discrepancies_found=len(missing_on_exchange) + len(missing_locally),
                corrections_made=corrections,
                duration_ms=duration_ms,
            )
            self.event_bus.publish(complete_event)

            self._last_reconciliation = datetime.now(UTC)

            logger.info(f"Reconciliation complete: {corrections} corrections made")

            return {
                "open_orders": len(open_orders),
                "balances": balances,
                "corrections": corrections,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            raise

    async def emit_heartbeat(self) -> ExchangeHeartbeat:
        """Emit exchange heartbeat event."""
        try:
            # Check connectivity
            await self.check_time_sync()

            # Get current state
            open_orders = len(self._pending_orders)
            rate_limit_remaining = self.rate_limiter.available_tokens

            heartbeat = ExchangeHeartbeat(
                timestamp=datetime.now(UTC),
                sequence=0,
                exchange="binance",
                ws_connected=False,  # Will be set by WSManager
                rest_responsive=True,
                latency_ms=self._clock_drift_ms,
                open_orders=open_orders,
                rate_limit_remaining=rate_limit_remaining,
                listen_key_valid=False,  # Will be set by WSManager
            )

            self.event_bus.publish(heartbeat)
            return heartbeat

        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            raise

    async def close(self):
        """Close exchange connections."""
        try:
            await self.exchange.close()
            logger.info("Exchange client closed")
        except Exception as e:
            logger.error(f"Error closing exchange: {e}")
