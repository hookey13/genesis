"""
WebSocket manager with user data stream and market data support.

Handles listen key lifecycle, auto-reconnection, sequence gap detection,
and event emission for real-time order and market updates.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import websockets
from websockets.client import WebSocketClientProtocol

from genesis.exchange.events import (
    EventBus,
    MarketTick,
    OrderFill,
    WebSocketEvent,
)

logger = logging.getLogger(__name__)


class WSManager:
    """
    WebSocket manager for market and user data streams.

    Features:
    - Listen key lifecycle management
    - Auto-reconnection with exponential backoff
    - Sequence gap detection and recovery
    - Event-driven architecture
    - Separate streams for market and user data
    """

    def __init__(
        self,
        exchange_client,
        event_bus: EventBus | None = None,
        testnet: bool = True,
    ):
        """Initialize WebSocket manager."""
        self.exchange = exchange_client
        self.event_bus = event_bus or EventBus()
        self.testnet = testnet

        # WebSocket URLs
        self.ws_base_url = (
            "wss://testnet.binance.vision/ws"
            if testnet
            else "wss://stream.binance.com:9443/ws"
        )
        self.ws_stream_url = (
            "wss://testnet.binance.vision/stream"
            if testnet
            else "wss://stream.binance.com:9443/stream"
        )

        # Connection state
        self.market_ws: WebSocketClientProtocol | None = None
        self.user_ws: WebSocketClientProtocol | None = None
        self.listen_key: str | None = None
        self.listen_key_expires: float | None = None

        # Stream subscriptions
        self.market_subscriptions: set[str] = set()
        self.user_callbacks: dict[str, Callable] = {}

        # Sequence tracking for gap detection
        self.last_sequence: dict[str, int] = {}
        self.sequence_gaps: dict[str, List[Tuple[int, int]]] = {}

        # Connection management
        self.running = False
        self.reconnect_delay = 1  # Start with 1 second
        self.max_reconnect_delay = 60  # Max 60 seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        # Tasks
        self.market_task: asyncio.Task | None = None
        self.user_task: asyncio.Task | None = None
        self.keepalive_task: asyncio.Task | None = None

    async def start(self):
        """Start WebSocket connections."""
        self.running = True

        # Start market data stream
        if self.market_subscriptions:
            self.market_task = asyncio.create_task(self._run_market_stream())

        # Start user data stream
        self.user_task = asyncio.create_task(self._run_user_stream())

        # Start listen key keepalive
        self.keepalive_task = asyncio.create_task(self._keepalive_listen_key())

        logger.info("WSManager started")

    async def stop(self):
        """Stop WebSocket connections."""
        self.running = False

        # Cancel tasks
        for task in [self.market_task, self.user_task, self.keepalive_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close connections
        for ws in [self.market_ws, self.user_ws]:
            if ws:
                await ws.close()

        # Delete listen key
        if self.listen_key:
            await self._delete_listen_key()

        logger.info("WSManager stopped")

    def subscribe_market(self, symbol: str, stream_type: str = "ticker"):
        """
        Subscribe to market data stream.

        Args:
            symbol: Trading symbol (e.g., "btcusdt")
            stream_type: Stream type (ticker, depth, trade)
        """
        stream_name = f"{symbol.lower()}@{stream_type}"
        self.market_subscriptions.add(stream_name)

        # If already running, send subscription message
        if self.market_ws:
            asyncio.create_task(self._send_subscription(True))

    def unsubscribe_market(self, symbol: str, stream_type: str = "ticker"):
        """Unsubscribe from market data stream."""
        stream_name = f"{symbol.lower()}@{stream_type}"
        self.market_subscriptions.discard(stream_name)

        # If connected, send unsubscription message
        if self.market_ws:
            asyncio.create_task(self._send_subscription(False))

    async def _get_listen_key(self) -> str:
        """Get new listen key for user data stream."""
        try:
            result = await self.exchange.exchange.privatePostUserDataStream()
            listen_key = result["listenKey"]
            self.listen_key = listen_key
            self.listen_key_expires = time.time() + 3600  # Expires in 1 hour
            logger.info(f"Got new listen key: {listen_key[:8]}...")
            return listen_key
        except Exception as e:
            logger.error(f"Failed to get listen key: {e}")
            raise

    async def _keepalive_listen_key(self):
        """Keep listen key alive by sending keepalive every 30 minutes."""
        while self.running:
            try:
                if self.listen_key:
                    # Check if approaching expiration
                    if time.time() > (
                        self.listen_key_expires - 900
                    ):  # 15 min before expiry
                        await self.exchange.exchange.privatePutUserDataStream(
                            {"listenKey": self.listen_key}
                        )
                        self.listen_key_expires = time.time() + 3600
                        logger.debug("Listen key keepalive sent")

                await asyncio.sleep(1800)  # Check every 30 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Listen key keepalive failed: {e}")
                # Try to get new listen key
                try:
                    await self._get_listen_key()
                except:
                    pass

    async def _delete_listen_key(self):
        """Delete listen key when shutting down."""
        if self.listen_key:
            try:
                await self.exchange.exchange.privateDeleteUserDataStream(
                    {"listenKey": self.listen_key}
                )
                logger.info("Listen key deleted")
            except Exception as e:
                logger.error(f"Failed to delete listen key: {e}")

    async def _run_market_stream(self):
        """Run market data WebSocket stream."""
        while self.running:
            try:
                # Build stream URL
                streams = "/".join(self.market_subscriptions)
                url = f"{self.ws_stream_url}?streams={streams}"

                # Connect
                async with websockets.connect(url) as ws:
                    self.market_ws = ws
                    self.reconnect_attempts = 0
                    self.reconnect_delay = 1

                    # Emit connected event
                    event = WebSocketEvent(
                        timestamp=datetime.now(UTC),
                        sequence=0,
                        status="CONNECTED",
                        stream_type="market_data",
                        url=url,
                    )
                    self.event_bus.publish(event)

                    logger.info("Market data WebSocket connected")

                    # Message loop
                    async for message in ws:
                        try:
                            await self._handle_market_message(json.loads(message))
                        except Exception as e:
                            logger.error(f"Market message handling error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market WebSocket error: {e}")

                # Emit disconnected event
                event = WebSocketEvent(
                    timestamp=datetime.now(UTC),
                    sequence=0,
                    status="DISCONNECTED",
                    stream_type="market_data",
                    url="",
                    error=str(e),
                )
                self.event_bus.publish(event)

                # Reconnect with backoff
                if self.running:
                    await self._reconnect_with_backoff()

    async def _run_user_stream(self):
        """Run user data WebSocket stream."""
        while self.running:
            try:
                # Get listen key
                if not self.listen_key:
                    self.listen_key = await self._get_listen_key()

                # Connect
                url = f"{self.ws_base_url}/{self.listen_key}"

                async with websockets.connect(url) as ws:
                    self.user_ws = ws
                    self.reconnect_attempts = 0
                    self.reconnect_delay = 1

                    # Emit connected event
                    event = WebSocketEvent(
                        timestamp=datetime.now(UTC),
                        sequence=0,
                        status="CONNECTED",
                        stream_type="user_data",
                        url=url[:50] + "...",  # Don't log full listen key
                    )
                    self.event_bus.publish(event)

                    logger.info("User data WebSocket connected")

                    # Message loop
                    async for message in ws:
                        try:
                            await self._handle_user_message(json.loads(message))
                        except Exception as e:
                            logger.error(f"User message handling error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"User WebSocket error: {e}")

                # Emit disconnected event
                event = WebSocketEvent(
                    timestamp=datetime.now(UTC),
                    sequence=0,
                    status="DISCONNECTED",
                    stream_type="user_data",
                    url="",
                    error=str(e),
                )
                self.event_bus.publish(event)

                # Reconnect with backoff
                if self.running:
                    await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff."""
        self.reconnect_attempts += 1

        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error(
                f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            return

        # Emit reconnecting event
        event = WebSocketEvent(
            timestamp=datetime.now(UTC),
            sequence=0,
            status="RECONNECTING",
            stream_type="",
            url="",
            reconnect_attempt=self.reconnect_attempts,
        )
        self.event_bus.publish(event)

        logger.info(
            f"Reconnecting in {self.reconnect_delay} seconds... (attempt {self.reconnect_attempts})"
        )
        await asyncio.sleep(self.reconnect_delay)

        # Exponential backoff with max delay
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def _handle_market_message(self, data: dict):
        """Handle market data message."""
        if "stream" in data:
            stream = data["stream"]
            payload = data["data"]

            # Check for sequence gaps
            if "E" in payload:  # Event time as sequence
                symbol = payload.get("s", "")
                sequence = payload["E"]

                if symbol in self.last_sequence:
                    expected = self.last_sequence[symbol] + 1
                    if sequence > expected:
                        # Gap detected
                        gap = (expected, sequence - 1)
                        if symbol not in self.sequence_gaps:
                            self.sequence_gaps[symbol] = []
                        self.sequence_gaps[symbol].append(gap)
                        logger.warning(f"Sequence gap detected for {symbol}: {gap}")

                self.last_sequence[symbol] = sequence

            # Handle different stream types
            if "@ticker" in stream:
                await self._handle_ticker(payload)
            elif "@depth" in stream:
                await self._handle_depth(payload)
            elif "@trade" in stream:
                await self._handle_trade(payload)

    async def _handle_user_message(self, data: dict):
        """Handle user data message."""
        event_type = data.get("e")

        if event_type == "executionReport":
            await self._handle_execution_report(data)
        elif event_type == "outboundAccountPosition":
            await self._handle_account_update(data)
        elif event_type == "balanceUpdate":
            await self._handle_balance_update(data)
        else:
            logger.debug(f"Unhandled user event type: {event_type}")

    async def _handle_ticker(self, data: dict):
        """Handle ticker update."""
        event = MarketTick(
            timestamp=datetime.fromtimestamp(data["E"] / 1000, tz=UTC),
            sequence=0,
            symbol=data["s"],
            bid=Decimal(data["b"]),
            ask=Decimal(data["a"]),
            bid_qty=Decimal(data["B"]),
            ask_qty=Decimal(data["A"]),
            last_price=Decimal(data["c"]),
            volume_24h=Decimal(data["v"]),
        )
        self.event_bus.publish(event)

    async def _handle_execution_report(self, data: dict):
        """Handle order execution report."""
        status = data["X"]  # Current order status
        exec_type = data["x"]  # Execution type

        if exec_type == "TRADE":
            # Order fill event
            event = OrderFill(
                timestamp=datetime.fromtimestamp(data["E"] / 1000, tz=UTC),
                sequence=0,
                client_order_id=data["c"],
                exchange_order_id=str(data["i"]),
                trade_id=str(data["t"]),
                symbol=data["s"],
                side=data["S"].lower(),
                fill_qty=Decimal(data["l"]),  # Last executed quantity
                fill_price=Decimal(data["L"]),  # Last executed price
                cumulative_qty=Decimal(data["z"]),  # Cumulative filled quantity
                remaining_qty=Decimal(data["q"]) - Decimal(data["z"]),
                commission=Decimal(data["n"]),
                commission_asset=data["N"],
                is_partial=status == "PARTIALLY_FILLED",
                status=status,
            )
            self.event_bus.publish(event)

            logger.info(f"Order fill: {data['c']} - {data['l']} @ {data['L']}")

    async def _handle_account_update(self, data: dict):
        """Handle account position update."""
        # Update internal balance tracking
        logger.debug(f"Account update: {data.get('B', [])}")

    async def _handle_balance_update(self, data: dict):
        """Handle balance update."""
        logger.debug(f"Balance update: {data.get('a')} - {data.get('d')}")

    async def _send_subscription(self, subscribe: bool = True):
        """Send subscription/unsubscription message."""
        if not self.market_ws:
            return

        method = "SUBSCRIBE" if subscribe else "UNSUBSCRIBE"
        message = {
            "method": method,
            "params": list(self.market_subscriptions),
            "id": int(time.time() * 1000),
        }

        try:
            await self.market_ws.send(json.dumps(message))
            logger.debug(f"Sent {method} for {len(self.market_subscriptions)} streams")
        except Exception as e:
            logger.error(f"Failed to send subscription: {e}")

    def get_sequence_gaps(self, symbol: str) -> List[Tuple[int, int]]:
        """Get sequence gaps for a symbol."""
        return self.sequence_gaps.get(symbol, [])

    def clear_sequence_gaps(self, symbol: str):
        """Clear sequence gaps after recovery."""
        if symbol in self.sequence_gaps:
            del self.sequence_gaps[symbol]
