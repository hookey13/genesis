"""
Market data feed integration for Project GENESIS.

Manages real-time market data ingestion from WebSocket streams,
data validation, caching, and distribution to strategies.
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import Trade
from genesis.engine.event_bus import EventBus
from genesis.exchange.websocket_manager import WebSocketManager

logger = structlog.get_logger(__name__)


class Ticker(BaseModel):
    """Market ticker data."""
    
    symbol: str
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal
    last_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    timestamp: datetime


class OrderBook(BaseModel):
    """Order book data."""
    
    symbol: str
    bids: list[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    asks: list[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    timestamp: datetime
    last_update_id: int = 0


class MarketData(BaseModel):
    """Combined market data."""
    
    symbol: str
    ticker: Ticker | None = None
    orderbook: OrderBook | None = None
    recent_trades: list[Trade] = Field(default_factory=list)
    timestamp: datetime


@dataclass
class MarketDataCache:
    """Cache for market data with TTL management."""

    tickers: dict[str, Ticker] = field(default_factory=dict)
    order_books: dict[str, OrderBook] = field(default_factory=dict)
    recent_trades: dict[str, deque[Trade]] = field(default_factory=dict)

    # Cache timestamps
    ticker_timestamps: dict[str, float] = field(default_factory=dict)
    orderbook_timestamps: dict[str, float] = field(default_factory=dict)

    # TTL in seconds
    ticker_ttl: int = 5
    orderbook_ttl: int = 2
    trade_history_size: int = 100

    def update_ticker(self, symbol: str, ticker: Ticker) -> None:
        """Update ticker cache."""
        self.tickers[symbol] = ticker
        self.ticker_timestamps[symbol] = time.time()

    def update_orderbook(self, symbol: str, orderbook: OrderBook) -> None:
        """Update order book cache."""
        self.order_books[symbol] = orderbook
        self.orderbook_timestamps[symbol] = time.time()

    def add_trade(self, symbol: str, trade: Trade) -> None:
        """Add trade to history."""
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = deque(maxlen=self.trade_history_size)
        self.recent_trades[symbol].append(trade)

    def get_ticker(self, symbol: str) -> Ticker | None:
        """Get ticker if not stale."""
        if symbol in self.tickers:
            age = time.time() - self.ticker_timestamps.get(symbol, 0)
            if age < self.ticker_ttl:
                return self.tickers[symbol]
        return None

    def get_orderbook(self, symbol: str) -> OrderBook | None:
        """Get order book if not stale."""
        if symbol in self.order_books:
            age = time.time() - self.orderbook_timestamps.get(symbol, 0)
            if age < self.orderbook_ttl:
                return self.order_books[symbol]
        return None

    def clear_stale(self) -> None:
        """Clear stale cache entries."""
        now = time.time()

        # Clear stale tickers
        stale_tickers = [
            symbol for symbol, ts in self.ticker_timestamps.items()
            if now - ts > self.ticker_ttl * 2
        ]
        for symbol in stale_tickers:
            self.tickers.pop(symbol, None)
            self.ticker_timestamps.pop(symbol, None)

        # Clear stale order books
        stale_orderbooks = [
            symbol for symbol, ts in self.orderbook_timestamps.items()
            if now - ts > self.orderbook_ttl * 2
        ]
        for symbol in stale_orderbooks:
            self.order_books.pop(symbol, None)
            self.orderbook_timestamps.pop(symbol, None)


@dataclass
class FeedMetrics:
    """Metrics for market data feed performance."""

    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0

    ticker_updates: int = 0
    orderbook_updates: int = 0
    trade_updates: int = 0

    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0

    connection_errors: int = 0
    validation_errors: int = 0

    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))


class MarketDataFeed:
    """
    Market data feed manager.

    Handles:
    - WebSocket stream subscription and management
    - Data validation and normalization
    - Caching and distribution
    - Error handling and recovery
    """

    def __init__(
        self,
        websocket_manager: WebSocketManager,
        event_bus: EventBus | None = None,
        symbols: list[str] | None = None,
        enable_orderbook: bool = True,
        enable_trades: bool = True,
    ):
        """
        Initialize market data feed.

        Args:
            websocket_manager: WebSocket connection manager
            event_bus: Optional event bus for publishing data
            symbols: List of symbols to subscribe to
            enable_orderbook: Enable order book stream
            enable_trades: Enable trades stream
        """
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.symbols = symbols or []
        self.enable_orderbook = enable_orderbook
        self.enable_trades = enable_trades

        self.cache = MarketDataCache()
        self.metrics = FeedMetrics()

        self.running = False
        self.processing_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=10000)

        # Processing tasks
        self.reader_task: asyncio.Task | None = None
        self.processor_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None

        # Callbacks for data updates
        self.ticker_callbacks: list[Callable[[str, Ticker], None]] = []
        self.orderbook_callbacks: list[Callable[[str, OrderBook], None]] = []
        self.trade_callbacks: list[Callable[[str, Trade], None]] = []

        logger.info(
            "MarketDataFeed initialized",
            symbols=self.symbols,
            orderbook_enabled=self.enable_orderbook,
            trades_enabled=self.enable_trades
        )

    def subscribe_ticker(self, callback: Callable[[str, Ticker], None]) -> None:
        """Subscribe to ticker updates."""
        self.ticker_callbacks.append(callback)

    def subscribe_orderbook(self, callback: Callable[[str, OrderBook], None]) -> None:
        """Subscribe to order book updates."""
        self.orderbook_callbacks.append(callback)

    def subscribe_trade(self, callback: Callable[[str, Trade], None]) -> None:
        """Subscribe to trade updates."""
        self.trade_callbacks.append(callback)

    async def start(self) -> None:
        """Start the market data feed."""
        if self.running:
            logger.warning("Market data feed already running")
            return

        self.running = True
        logger.info("Starting market data feed")

        # Subscribe to WebSocket streams
        await self._subscribe_to_streams()

        # Start processing tasks
        self.reader_task = asyncio.create_task(self._reader_loop())
        self.processor_task = asyncio.create_task(self._processor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Market data feed started")

    async def stop(self) -> None:
        """Stop the market data feed."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping market data feed")

        # Cancel processing tasks
        tasks = [self.reader_task, self.processor_task, self.cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()

        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)

        # Unsubscribe from streams
        await self._unsubscribe_from_streams()

        logger.info("Market data feed stopped")

    async def _subscribe_to_streams(self) -> None:
        """Subscribe to WebSocket streams."""
        streams = []

        for symbol in self.symbols:
            # Ticker stream
            streams.append(f"{symbol.lower()}@ticker")

            # Order book stream
            if self.enable_orderbook:
                streams.append(f"{symbol.lower()}@depth20@100ms")

            # Trades stream
            if self.enable_trades:
                streams.append(f"{symbol.lower()}@trade")

        if streams:
            await self.websocket_manager.subscribe(streams)
            logger.info("Subscribed to market data streams", stream_count=len(streams))

    async def _unsubscribe_from_streams(self) -> None:
        """Unsubscribe from WebSocket streams."""
        streams = []

        for symbol in self.symbols:
            streams.append(f"{symbol.lower()}@ticker")
            if self.enable_orderbook:
                streams.append(f"{symbol.lower()}@depth20@100ms")
            if self.enable_trades:
                streams.append(f"{symbol.lower()}@trade")

        if streams:
            await self.websocket_manager.unsubscribe(streams)
            logger.info("Unsubscribed from market data streams")

    async def _reader_loop(self) -> None:
        """Read messages from WebSocket and queue for processing."""
        while self.running:
            try:
                # Get message from WebSocket
                message = await self.websocket_manager.receive()

                if message:
                    self.metrics.messages_received += 1

                    # Queue for processing
                    try:
                        await asyncio.wait_for(
                            self.processing_queue.put(message),
                            timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        self.metrics.messages_dropped += 1
                        logger.warning("Processing queue full, dropping message")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics.connection_errors += 1
                logger.error("Market data reader error", error=str(e))
                await asyncio.sleep(1)

    async def _processor_loop(self) -> None:
        """Process queued market data messages."""
        while self.running:
            try:
                # Get message from queue
                message = await self.processing_queue.get()

                start_time = time.time()

                # Process the message
                await self._process_message(message)

                # Update metrics
                processing_time = (time.time() - start_time) * 1000
                self._update_processing_metrics(processing_time)

                self.metrics.messages_processed += 1

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Market data processing error", error=str(e))

    async def _process_message(self, message: dict) -> None:
        """
        Process a market data message.

        Args:
            message: Raw message from WebSocket
        """
        try:
            # Determine message type
            if "stream" not in message:
                return

            stream = message["stream"]
            data = message.get("data", {})

            if "@ticker" in stream:
                await self._process_ticker(data)
            elif "@depth" in stream:
                await self._process_orderbook(data)
            elif "@trade" in stream:
                await self._process_trade(data)
            else:
                logger.debug("Unknown stream type", stream=stream)

        except Exception as e:
            self.metrics.validation_errors += 1
            logger.error("Message processing failed", error=str(e))

    async def _process_ticker(self, data: dict) -> None:
        """Process ticker data."""
        try:
            symbol = data.get("s", "")

            ticker = Ticker(
                symbol=symbol,
                bid_price=Decimal(data.get("b", "0")),
                bid_quantity=Decimal(data.get("B", "0")),
                ask_price=Decimal(data.get("a", "0")),
                ask_quantity=Decimal(data.get("A", "0")),
                last_price=Decimal(data.get("c", "0")),
                volume=Decimal(data.get("v", "0")),
                quote_volume=Decimal(data.get("q", "0")),
                open_price=Decimal(data.get("o", "0")),
                high_price=Decimal(data.get("h", "0")),
                low_price=Decimal(data.get("l", "0")),
                timestamp=datetime.fromtimestamp(data.get("E", 0) / 1000, UTC)
            )

            # Update cache
            self.cache.update_ticker(symbol, ticker)
            self.metrics.ticker_updates += 1

            # Call callbacks
            for callback in self.ticker_callbacks:
                try:
                    callback(symbol, ticker)
                except Exception as e:
                    logger.error("Ticker callback failed", error=str(e))

            # Publish to event bus
            if self.event_bus:
                event = Event(
                    type=EventType.TICKER_UPDATE,
                    data={
                        "symbol": symbol,
                        "ticker": ticker
                    },
                    priority=EventPriority.NORMAL
                )
                await self.event_bus.publish(event)

        except Exception as e:
            logger.error("Ticker processing failed", error=str(e))

    async def _process_orderbook(self, data: dict) -> None:
        """Process order book data."""
        try:
            symbol = data.get("s", "")

            # Parse bids and asks
            bids = [
                (Decimal(price), Decimal(qty))
                for price, qty in data.get("bids", [])
            ]
            asks = [
                (Decimal(price), Decimal(qty))
                for price, qty in data.get("asks", [])
            ]

            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.fromtimestamp(data.get("E", 0) / 1000, UTC),
                last_update_id=data.get("u", 0)
            )

            # Update cache
            self.cache.update_orderbook(symbol, orderbook)
            self.metrics.orderbook_updates += 1

            # Call callbacks
            for callback in self.orderbook_callbacks:
                try:
                    callback(symbol, orderbook)
                except Exception as e:
                    logger.error("OrderBook callback failed", error=str(e))

            # Publish to event bus
            if self.event_bus:
                event = Event(
                    type=EventType.ORDERBOOK_UPDATE,
                    data={
                        "symbol": symbol,
                        "orderbook": orderbook
                    },
                    priority=EventPriority.NORMAL
                )
                await self.event_bus.publish(event)

        except Exception as e:
            logger.error("OrderBook processing failed", error=str(e))

    async def _process_trade(self, data: dict) -> None:
        """Process trade data."""
        try:
            symbol = data.get("s", "")

            trade = Trade(
                trade_id=str(data.get("t", "")),
                symbol=symbol,
                price=Decimal(data.get("p", "0")),
                quantity=Decimal(data.get("q", "0")),
                timestamp=datetime.fromtimestamp(data.get("T", 0) / 1000, UTC),
                is_buyer_maker=data.get("m", False)
            )

            # Update cache
            self.cache.add_trade(symbol, trade)
            self.metrics.trade_updates += 1

            # Call callbacks
            for callback in self.trade_callbacks:
                try:
                    callback(symbol, trade)
                except Exception as e:
                    logger.error("Trade callback failed", error=str(e))

            # Publish to event bus
            if self.event_bus:
                event = Event(
                    type=EventType.TRADE_UPDATE,
                    data={
                        "symbol": symbol,
                        "trade": trade
                    },
                    priority=EventPriority.LOW
                )
                await self.event_bus.publish(event)

        except Exception as e:
            logger.error("Trade processing failed", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Periodically clean up stale cache entries."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Clean up every 30 seconds
                self.cache.clear_stale()
                logger.debug("Cache cleanup completed")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Cache cleanup error", error=str(e))

    def _update_processing_metrics(self, processing_time_ms: float) -> None:
        """Update processing time metrics."""
        if self.metrics.messages_processed == 0:
            self.metrics.avg_processing_time_ms = processing_time_ms
        else:
            # Running average
            count = self.metrics.messages_processed
            self.metrics.avg_processing_time_ms = (
                (self.metrics.avg_processing_time_ms * count + processing_time_ms) /
                (count + 1)
            )

        self.metrics.max_processing_time_ms = max(
            self.metrics.max_processing_time_ms,
            processing_time_ms
        )

    async def get_latest_data(self, symbol: str | None = None) -> MarketData | dict[str, MarketData]:
        """
        Get latest market data.

        Args:
            symbol: Optional symbol to get data for

        Returns:
            Market data for symbol or all symbols
        """
        if symbol:
            ticker = self.cache.get_ticker(symbol)
            orderbook = self.cache.get_orderbook(symbol)
            recent_trades = list(self.cache.recent_trades.get(symbol, []))[-10:]

            if ticker:
                return MarketData(
                    symbol=symbol,
                    ticker=ticker,
                    orderbook=orderbook,
                    recent_trades=recent_trades,
                    timestamp=datetime.now(UTC)
                )
            return None
        else:
            # Return data for all symbols
            result = {}
            for sym in self.symbols:
                data = await self.get_latest_data(sym)
                if data:
                    result[sym] = data
            return result

    def get_metrics(self) -> FeedMetrics:
        """
        Get current feed metrics.

        Returns:
            Current metrics snapshot
        """
        self.metrics.last_update = datetime.now(UTC)
        return self.metrics

    async def add_symbol(self, symbol: str) -> None:
        """
        Add a new symbol to the feed.

        Args:
            symbol: Symbol to add
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)

            # Subscribe to streams for new symbol
            streams = [f"{symbol.lower()}@ticker"]
            if self.enable_orderbook:
                streams.append(f"{symbol.lower()}@depth20@100ms")
            if self.enable_trades:
                streams.append(f"{symbol.lower()}@trade")

            await self.websocket_manager.subscribe(streams)
            logger.info("Added symbol to feed", symbol=symbol)

    async def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from the feed.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)

            # Unsubscribe from streams
            streams = [f"{symbol.lower()}@ticker"]
            if self.enable_orderbook:
                streams.append(f"{symbol.lower()}@depth20@100ms")
            if self.enable_trades:
                streams.append(f"{symbol.lower()}@trade")

            await self.websocket_manager.unsubscribe(streams)

            # Clear cache
            self.cache.tickers.pop(symbol, None)
            self.cache.order_books.pop(symbol, None)
            self.cache.recent_trades.pop(symbol, None)

            logger.info("Removed symbol from feed", symbol=symbol)
