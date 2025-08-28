"""Order Book Management System for real-time market depth tracking."""

import asyncio
import json
import ssl
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import structlog
import websockets
from websockets.exceptions import ConnectionClosed

from genesis.core.events import Event
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: Decimal
    quantity: Decimal
    order_count: int = 1

    @property
    def notional(self) -> Decimal:
        """Calculate notional value of this level."""
        return self.price * self.quantity


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot at a point in time."""

    symbol: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    sequence_number: int = 0

    @property
    def best_bid(self) -> Decimal | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / Decimal("2")
        return None

    @property
    def spread(self) -> Decimal | None:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> int | None:
        """Calculate spread in basis points."""
        if self.spread and self.mid_price:
            return int((self.spread / self.mid_price) * Decimal("10000"))
        return None

    def get_bid_volume(self, levels: int = 20) -> Decimal:
        """Get total bid volume up to specified levels."""
        return sum(level.quantity for level in self.bids[:levels])

    def get_ask_volume(self, levels: int = 20) -> Decimal:
        """Get total ask volume up to specified levels."""
        return sum(level.quantity for level in self.asks[:levels])

    def get_imbalance_ratio(self, levels: int = 5) -> Decimal | None:
        """Calculate order book imbalance ratio."""
        bid_vol = self.get_bid_volume(levels)
        ask_vol = self.get_ask_volume(levels)

        if bid_vol + ask_vol == 0:
            return None

        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def get_weighted_mid_price(self, levels: int = 3) -> Decimal | None:
        """Calculate volume-weighted mid price."""
        if not self.bids or not self.asks:
            return None

        bid_weight = sum(b.price * b.quantity for b in self.bids[:levels])
        bid_vol = sum(b.quantity for b in self.bids[:levels])

        ask_weight = sum(a.price * a.quantity for a in self.asks[:levels])
        ask_vol = sum(a.quantity for a in self.asks[:levels])

        if bid_vol + ask_vol == 0:
            return None

        weighted_bid = bid_weight / bid_vol if bid_vol > 0 else Decimal("0")
        weighted_ask = ask_weight / ask_vol if ask_vol > 0 else Decimal("0")

        return (weighted_bid * ask_vol + weighted_ask * bid_vol) / (bid_vol + ask_vol)


class OrderBookManager:
    """Manages real-time order book data from exchange websockets."""

    def __init__(
        self,
        event_bus: EventBus,
        depth_levels: int = 20,
        update_frequency_ms: int = 100,
    ):
        """Initialize order book manager.

        Args:
            event_bus: Event bus for publishing updates
            depth_levels: Number of price levels to maintain
            update_frequency_ms: Update frequency in milliseconds
        """
        self.event_bus = event_bus
        self.depth_levels = depth_levels
        self.update_frequency_ms = update_frequency_ms
        self.order_books: dict[str, OrderBookSnapshot] = {}
        self.websocket_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self.running = False
        self.heartbeat_task: asyncio.Task | None = None
        self.connection_tasks: list[asyncio.Task] = []
        self.reconnect_delay = 2  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 60  # Maximum reconnect delay

    async def start(self, symbols: list[str]) -> None:
        """Start order book management for specified symbols.

        Args:
            symbols: List of trading symbols to track
        """
        self.running = True

        # Start heartbeat task
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Connect to websockets for each symbol
        for symbol in symbols:
            task = asyncio.create_task(self._connect_and_subscribe(symbol))
            self.connection_tasks.append(task)

        logger.info(
            "order_book_manager_started",
            symbols=symbols,
            depth_levels=self.depth_levels,
        )

    async def stop(self) -> None:
        """Stop order book management and close connections."""
        self.running = False

        # Cancel heartbeat task
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        # Cancel all connection tasks
        for task in self.connection_tasks:
            task.cancel()

        # Close all websocket connections
        for _symbol, ws in self.websocket_connections.items():
            await ws.close()

        logger.info("order_book_manager_stopped")

    async def _connect_and_subscribe(self, symbol: str) -> None:
        """Connect to websocket and subscribe to order book updates.

        Args:
            symbol: Trading symbol
        """
        while self.running:
            try:
                # Construct websocket URL for Binance depth stream
                stream_name = f"{symbol.lower()}@depth{self.depth_levels}@{self.update_frequency_ms}ms"
                ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"

                # Create SSL context for secure connection
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED

                # Add connection headers for authentication (if API key is needed)
                headers = {"User-Agent": "Genesis-Trading-Bot/1.0"}

                async with websockets.connect(
                    ws_url,
                    ssl=ssl_context,
                    extra_headers=headers,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,  # Wait 10 seconds for pong
                ) as websocket:
                    self.websocket_connections[symbol] = websocket
                    self.reconnect_delay = (
                        2  # Reset reconnect delay on successful connection
                    )

                    logger.info("websocket_connected", symbol=symbol, url=ws_url)

                    await self._process_order_book_updates(symbol, websocket)

            except ConnectionClosed as e:
                logger.warning(
                    "websocket_disconnected",
                    symbol=symbol,
                    code=e.code,
                    reason=e.reason,
                )

            except Exception as e:
                logger.error("websocket_error", symbol=symbol, error=str(e))

            # Exponential backoff for reconnection
            if self.running:
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(
                    self.reconnect_delay * 2, self.max_reconnect_delay
                )

    async def _process_order_book_updates(
        self, symbol: str, websocket: websockets.WebSocketClientProtocol
    ) -> None:
        """Process order book updates from websocket.

        Args:
            symbol: Trading symbol
            websocket: Websocket connection
        """
        async for message in websocket:
            if not self.running:
                break

            try:
                data = json.loads(message)

                # Parse order book update
                snapshot = self._parse_depth_update(symbol, data)

                # Store snapshot
                self.order_books[symbol] = snapshot

                # Publish order book update event
                await self.event_bus.publish(
                    Event(
                        type="order_book_updated",
                        data={
                            "symbol": symbol,
                            "mid_price": (
                                float(snapshot.mid_price)
                                if snapshot.mid_price
                                else None
                            ),
                            "spread_bps": snapshot.spread_bps,
                            "imbalance_ratio": (
                                float(snapshot.get_imbalance_ratio())
                                if snapshot.get_imbalance_ratio()
                                else None
                            ),
                            "best_bid": (
                                float(snapshot.best_bid) if snapshot.best_bid else None
                            ),
                            "best_ask": (
                                float(snapshot.best_ask) if snapshot.best_ask else None
                            ),
                            "timestamp": snapshot.timestamp.isoformat(),
                        },
                    )
                )

                # Check for significant imbalances
                imbalance = snapshot.get_imbalance_ratio()
                if imbalance and abs(imbalance) > Decimal("0.3"):
                    await self.event_bus.publish(
                        Event(
                            type="order_book_imbalance_detected",
                            data={
                                "symbol": symbol,
                                "imbalance_ratio": float(imbalance),
                                "direction": "buy" if imbalance > 0 else "sell",
                                "timestamp": snapshot.timestamp.isoformat(),
                            },
                        )
                    )

            except json.JSONDecodeError as e:
                logger.error("json_decode_error", symbol=symbol, error=str(e))
            except Exception as e:
                logger.error("order_book_processing_error", symbol=symbol, error=str(e))

    def _parse_depth_update(self, symbol: str, data: dict) -> OrderBookSnapshot:
        """Parse depth update from Binance websocket.

        Args:
            symbol: Trading symbol
            data: Raw websocket message

        Returns:
            Parsed order book snapshot
        """
        snapshot = OrderBookSnapshot(
            symbol=symbol, sequence_number=data.get("lastUpdateId", 0)
        )

        # Parse bids
        for bid in data.get("bids", []):
            snapshot.bids.append(
                OrderBookLevel(price=Decimal(bid[0]), quantity=Decimal(bid[1]))
            )

        # Parse asks
        for ask in data.get("asks", []):
            snapshot.asks.append(
                OrderBookLevel(price=Decimal(ask[0]), quantity=Decimal(ask[1]))
            )

        # Sort bids descending, asks ascending
        snapshot.bids.sort(key=lambda x: x.price, reverse=True)
        snapshot.asks.sort(key=lambda x: x.price)

        return snapshot

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to keep websocket connections alive."""
        while self.running:
            try:
                # Send ping to all active connections
                for _symbol, ws in self.websocket_connections.items():
                    if ws and not ws.closed:
                        pong = await ws.ping()
                        await asyncio.wait_for(pong, timeout=10)

                # Wait 30 seconds before next heartbeat
                await asyncio.sleep(30)

            except TimeoutError:
                logger.warning("heartbeat_timeout")
            except Exception as e:
                logger.error("heartbeat_error", error=str(e))

    def get_order_book(self, symbol: str) -> OrderBookSnapshot | None:
        """Get current order book for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current order book snapshot or None
        """
        return self.order_books.get(symbol)

    def get_mid_price(self, symbol: str) -> Decimal | None:
        """Get current mid price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current mid price or None
        """
        book = self.get_order_book(symbol)
        return book.mid_price if book else None

    def get_spread(self, symbol: str) -> Decimal | None:
        """Get current spread for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current spread or None
        """
        book = self.get_order_book(symbol)
        return book.spread if book else None

    def get_imbalance(self, symbol: str, levels: int = 5) -> Decimal | None:
        """Get current order book imbalance.

        Args:
            symbol: Trading symbol
            levels: Number of levels to consider

        Returns:
            Imbalance ratio or None
        """
        book = self.get_order_book(symbol)
        return book.get_imbalance_ratio(levels) if book else None

    def calculate_price_impact(
        self, symbol: str, side: str, quantity: Decimal
    ) -> tuple[Decimal, Decimal] | None:
        """Calculate expected price impact for an order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity

        Returns:
            Tuple of (average_price, price_impact) or None
        """
        book = self.get_order_book(symbol)
        if not book:
            return None

        levels = book.asks if side.lower() == "buy" else book.bids
        if not levels:
            return None

        remaining = quantity
        total_cost = Decimal("0")

        for level in levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            remaining -= fill_qty

        if remaining > 0:
            # Not enough liquidity
            return None

        average_price = total_cost / quantity
        mid_price = book.mid_price

        if not mid_price:
            return None

        price_impact = abs(average_price - mid_price) / mid_price

        return (average_price, price_impact)
