"""
Market Data Service for real-time price feeds and order book management.

Manages market data streams, order book depth, spread calculations,
and volume profile analysis for multiple trading pairs.
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import uuid4

import structlog

from genesis.analytics.spread_analyzer import SpreadAnalyzer
from genesis.analytics.spread_tracker import SpreadTracker
from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.websocket_manager import WebSocketManager

logger = structlog.get_logger(__name__)


class MarketState(str, Enum):
    """Market state classification."""

    DEAD = "DEAD"  # Very low volume/activity
    NORMAL = "NORMAL"  # Regular trading conditions
    VOLATILE = "VOLATILE"  # High volatility detected
    PANIC = "PANIC"  # Extreme market conditions
    MAINTENANCE = "MAINTENANCE"  # Exchange maintenance


@dataclass
class Tick:
    """Single price tick."""

    symbol: str
    price: Decimal
    quantity: Decimal
    timestamp: float
    is_buyer_maker: bool


@dataclass
class Candle:
    """OHLCV candle data."""

    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime
    trades: int = 0


@dataclass
class OrderBookLevel:
    """Single order book level."""

    price: Decimal
    quantity: Decimal

    def total_value(self) -> Decimal:
        """Calculate total value at this level."""
        return self.price * self.quantity


@dataclass
class OrderBook:
    """Order book with bid/ask levels."""

    symbol: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    last_update_id: int = 0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def best_bid(self) -> Decimal | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Decimal | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    def spread(self) -> Decimal | None:
        """Calculate spread."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask - bid
        return None

    def spread_basis_points(self) -> int | None:
        """Calculate spread in basis points."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            mid = (bid + ask) / Decimal("2")
            spread = ask - bid
            return int((spread / mid) * Decimal("10000"))
        return None

    def mid_price(self) -> Decimal | None:
        """Calculate mid price."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return (bid + ask) / Decimal("2")
        return None


@dataclass
class VolumeProfile:
    """Volume profile by time of day."""

    symbol: str
    hour_volumes: dict[int, Decimal] = field(default_factory=dict)
    rolling_24h_volume: Decimal = Decimal("0")
    average_hourly_volume: Decimal = Decimal("0")
    last_update: datetime = field(default_factory=datetime.now)

    def add_volume(self, hour: int, volume: Decimal):
        """Add volume for specific hour."""
        if hour not in self.hour_volumes:
            self.hour_volumes[hour] = Decimal("0")
        self.hour_volumes[hour] += volume
        self._recalculate_averages()

    def _recalculate_averages(self):
        """Recalculate volume averages."""
        if self.hour_volumes:
            total = sum(self.hour_volumes.values())
            self.rolling_24h_volume = total
            self.average_hourly_volume = total / Decimal(str(len(self.hour_volumes)))

    def is_volume_anomaly(
        self, current_volume: Decimal, threshold: Decimal = Decimal("2")
    ) -> bool:
        """Check if current volume is anomalous."""
        if self.average_hourly_volume > 0:
            ratio = current_volume / self.average_hourly_volume
            return ratio > threshold or ratio < (Decimal("1") / threshold)
        return False


class MarketDataService:
    """
    Service for managing real-time market data.

    Provides interfaces for subscribing to market data streams,
    accessing current prices and order books, and analyzing market conditions.
    """

    def __init__(
        self,
        websocket_manager: WebSocketManager | None = None,
        gateway: BinanceGateway | None = None,
        event_bus: EventBus | None = None,
        repository: Optional["Repository"] = None,
    ):
        """
        Initialize the Market Data Service.

        Args:
            websocket_manager: WebSocket manager for streams
            gateway: Exchange gateway for REST API
            event_bus: Event bus for publishing events
            repository: Repository for data persistence
        """
        self.websocket_manager = websocket_manager or WebSocketManager(gateway)
        self.gateway = gateway
        self.event_bus = event_bus
        self.repository = repository

        # Data storage
        self.current_prices: dict[str, Decimal] = {}
        self.order_books: dict[str, OrderBook] = {}
        self.candles: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ticks: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_profiles: dict[str, VolumeProfile] = {}
        self.market_states: dict[str, MarketState] = {}
        self.spread_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Spread analytics integration
        self.spread_analyzer = SpreadAnalyzer(max_history_size=1000)
        self.spread_tracker = SpreadTracker(self.spread_analyzer, self.event_bus)

        # Aggregation state
        self.candle_aggregators: dict[str, dict] = {}
        self.last_candle_time: dict[str, datetime] = {}

        # Subscriptions
        self.active_subscriptions: set[str] = set()

        logger.info("MarketDataService initialized")

    async def start(self) -> None:
        """Start the market data service."""
        logger.info("Starting MarketDataService")

        # Start WebSocket manager
        await self.websocket_manager.start()

        # Start spread tracker
        await self.spread_tracker.start()

        # Subscribe to WebSocket callbacks
        self.websocket_manager.subscribe("trade", self._handle_trade)
        self.websocket_manager.subscribe("depth", self._handle_depth)
        self.websocket_manager.subscribe("kline", self._handle_kline)
        self.websocket_manager.subscribe("ticker", self._handle_ticker)

        # Start aggregation tasks
        asyncio.create_task(self._candle_aggregation_task())
        asyncio.create_task(self._spread_monitoring_task())
        asyncio.create_task(self._volume_analysis_task())
        asyncio.create_task(self._spread_persistence_task())

    async def stop(self) -> None:
        """Stop the market data service."""
        logger.info("Stopping MarketDataService")
        await self.websocket_manager.stop()

    async def subscribe_market_data(self, symbol: str) -> AsyncIterator[Tick]:
        """
        Subscribe to real-time market data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Yields:
            Tick objects as they arrive
        """
        self.active_subscriptions.add(symbol)

        # Get tick queue for this symbol
        tick_queue = self.ticks[symbol]

        # Yield existing ticks
        for tick in tick_queue:
            yield tick

        # Continue yielding new ticks
        while symbol in self.active_subscriptions:
            # Wait for new tick
            await asyncio.sleep(0.1)

            # Check for new ticks
            if tick_queue and len(tick_queue) > 0:
                # Get latest tick
                latest_tick = tick_queue[-1]
                yield latest_tick

    def get_current_price(self, symbol: str) -> Decimal | None:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price or None if not available
        """
        return self.current_prices.get(symbol)

    def get_order_book(self, symbol: str, depth: int = 5) -> OrderBook | None:
        """
        Get order book for a symbol.

        Args:
            symbol: Trading symbol
            depth: Number of levels to return (max 5)

        Returns:
            OrderBook or None if not available
        """
        book = self.order_books.get(symbol)
        if book and depth < 5:
            # Limit depth
            limited_book = OrderBook(symbol=symbol)
            limited_book.bids = book.bids[:depth]
            limited_book.asks = book.asks[:depth]
            limited_book.last_update_id = book.last_update_id
            limited_book.timestamp = book.timestamp
            return limited_book
        return book

    def calculate_spread(self, symbol: str) -> Decimal | None:
        """
        Calculate spread in basis points.

        Args:
            symbol: Trading symbol

        Returns:
            Spread in basis points or None
        """
        book = self.order_books.get(symbol)
        if book:
            return book.spread_basis_points()
        return None

    def classify_market_state(self, symbol: str) -> MarketState:
        """
        Classify current market state for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Market state classification
        """
        # Get current data
        candles = self.candles.get(symbol, deque())
        volume_profile = self.volume_profiles.get(symbol)
        spread = self.calculate_spread(symbol)

        if not candles or len(candles) < 20:
            return MarketState.DEAD

        # Calculate volatility (ATR)
        recent_candles = list(candles)[-20:]
        high_low_ranges = [float(c.high - c.low) for c in recent_candles]
        atr = sum(high_low_ranges) / len(high_low_ranges)
        avg_price = sum(float(c.close) for c in recent_candles) / len(recent_candles)
        volatility_ratio = atr / avg_price if avg_price > 0 else 0

        # Check volume
        if volume_profile:
            is_low_volume = volume_profile.rolling_24h_volume < Decimal(
                "10000"
            )  # Threshold
            is_volume_anomaly = volume_profile.is_volume_anomaly(
                volume_profile.rolling_24h_volume
            )
        else:
            is_low_volume = True
            is_volume_anomaly = False

        # Classify state
        if is_low_volume:
            state = MarketState.DEAD
        elif volatility_ratio > 0.05:  # 5% volatility
            state = MarketState.PANIC
        elif volatility_ratio > 0.02 or is_volume_anomaly:  # 2% volatility
            state = MarketState.VOLATILE
        else:
            state = MarketState.NORMAL

        # Cache and return
        self.market_states[symbol] = state
        return state

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade stream data."""
        try:
            trade_data = data.get("data", {})
            symbol = trade_data.get("s", "").upper()

            if not symbol:
                return

            # Create tick
            tick = Tick(
                symbol=symbol,
                price=Decimal(trade_data.get("p", "0")),
                quantity=Decimal(trade_data.get("q", "0")),
                timestamp=trade_data.get("T", 0) / 1000.0,
                is_buyer_maker=trade_data.get("m", False),
            )

            # Update current price
            self.current_prices[symbol] = tick.price

            # Store tick
            self.ticks[symbol].append(tick)

            # Update candle aggregator
            await self._update_candle_aggregator(symbol, tick)

            # Publish event
            if self.event_bus:
                event = Event(
                    event_type=EventType.MARKET_DATA_UPDATED,
                    aggregate_id=symbol,
                    event_data={
                        "symbol": symbol,
                        "price": str(tick.price),
                        "quantity": str(tick.quantity),
                        "timestamp": tick.timestamp,
                    },
                )
                await self.event_bus.publish(event, EventPriority.NORMAL)

        except Exception as e:
            logger.error("Error handling trade data", error=str(e))

    async def _handle_depth(self, data: dict) -> None:
        """Handle depth stream data."""
        try:
            depth_data = data.get("data", {})
            symbol = data.get("stream", "").split("@")[0].upper()

            if not symbol:
                return

            # Create order book
            order_book = OrderBook(symbol=symbol)
            order_book.last_update_id = depth_data.get("lastUpdateId", 0)

            # Parse bids
            for bid in depth_data.get("bids", [])[:5]:
                order_book.bids.append(
                    OrderBookLevel(price=Decimal(bid[0]), quantity=Decimal(bid[1]))
                )

            # Parse asks
            for ask in depth_data.get("asks", [])[:5]:
                order_book.asks.append(
                    OrderBookLevel(price=Decimal(ask[0]), quantity=Decimal(ask[1]))
                )

            # Store order book
            self.order_books[symbol] = order_book

            # Track spread with our analytics
            if order_book.best_bid() and order_book.best_ask():
                # Update spread tracker
                orderbook_dict = {
                    "bids": [
                        [str(level.price), str(level.quantity)]
                        for level in order_book.bids
                    ],
                    "asks": [
                        [str(level.price), str(level.quantity)]
                        for level in order_book.asks
                    ],
                }

                await self.spread_tracker.track_pair_spread(symbol, orderbook_dict)

            # Track spread history
            spread_bp = order_book.spread_basis_points()
            if spread_bp is not None:
                self.spread_history[symbol].append(
                    {"timestamp": datetime.now(), "spread_bp": spread_bp}
                )

                # Check for spread compression
                if spread_bp < 10:  # Less than 10 basis points
                    if self.event_bus:
                        event = Event(
                            event_type=EventType.SPREAD_ALERT,
                            aggregate_id=symbol,
                            event_data={
                                "symbol": symbol,
                                "spread_bp": spread_bp,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        await self.event_bus.publish(event, EventPriority.HIGH)

        except Exception as e:
            logger.error("Error handling depth data", error=str(e))

    async def _handle_kline(self, data: dict) -> None:
        """Handle kline stream data."""
        try:
            kline_data = data.get("data", {}).get("k", {})
            symbol = kline_data.get("s", "").upper()

            if not symbol or not kline_data.get(
                "x", False
            ):  # Only process closed candles
                return

            # Create candle
            candle = Candle(
                symbol=symbol,
                open=Decimal(kline_data.get("o", "0")),
                high=Decimal(kline_data.get("h", "0")),
                low=Decimal(kline_data.get("l", "0")),
                close=Decimal(kline_data.get("c", "0")),
                volume=Decimal(kline_data.get("v", "0")),
                timestamp=datetime.fromtimestamp(kline_data.get("t", 0) / 1000),
                trades=kline_data.get("n", 0),
            )

            # Store candle
            self.candles[symbol].append(candle)

            # Update volume profile
            hour = candle.timestamp.hour
            if symbol not in self.volume_profiles:
                self.volume_profiles[symbol] = VolumeProfile(symbol=symbol)
            self.volume_profiles[symbol].add_volume(hour, candle.volume)

        except Exception as e:
            logger.error("Error handling kline data", error=str(e))

    async def _handle_ticker(self, data: dict) -> None:
        """Handle ticker stream data."""
        try:
            ticker_data = data.get("data", {})
            symbol = ticker_data.get("s", "").upper()

            if not symbol:
                return

            # Update 24h volume
            volume_24h = Decimal(ticker_data.get("v", "0"))

            if symbol in self.volume_profiles:
                self.volume_profiles[symbol].rolling_24h_volume = volume_24h

        except Exception as e:
            logger.error("Error handling ticker data", error=str(e))

    async def _update_candle_aggregator(self, symbol: str, tick: Tick) -> None:
        """Update candle aggregator with new tick."""
        current_time = datetime.fromtimestamp(tick.timestamp)
        minute_start = current_time.replace(second=0, microsecond=0)

        if symbol not in self.candle_aggregators:
            self.candle_aggregators[symbol] = {
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": tick.quantity,
                "trades": 1,
                "minute": minute_start,
            }
        else:
            agg = self.candle_aggregators[symbol]

            # Check if we're in a new minute
            if agg["minute"] != minute_start:
                # Create candle from aggregator
                candle = Candle(
                    symbol=symbol,
                    open=agg["open"],
                    high=agg["high"],
                    low=agg["low"],
                    close=agg["close"],
                    volume=agg["volume"],
                    timestamp=agg["minute"],
                    trades=agg["trades"],
                )

                # Store candle
                self.candles[symbol].append(candle)

                # Reset aggregator
                self.candle_aggregators[symbol] = {
                    "open": tick.price,
                    "high": tick.price,
                    "low": tick.price,
                    "close": tick.price,
                    "volume": tick.quantity,
                    "trades": 1,
                    "minute": minute_start,
                }
            else:
                # Update aggregator
                agg["high"] = max(agg["high"], tick.price)
                agg["low"] = min(agg["low"], tick.price)
                agg["close"] = tick.price
                agg["volume"] += tick.quantity
                agg["trades"] += 1

    async def _candle_aggregation_task(self) -> None:
        """Background task for candle aggregation."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Force close any open candles
                current_time = datetime.now()
                minute_start = current_time.replace(second=0, microsecond=0)

                for symbol, agg in self.candle_aggregators.items():
                    if agg["minute"] < minute_start:
                        # Create candle
                        candle = Candle(
                            symbol=symbol,
                            open=agg["open"],
                            high=agg["high"],
                            low=agg["low"],
                            close=agg["close"],
                            volume=agg["volume"],
                            timestamp=agg["minute"],
                            trades=agg["trades"],
                        )

                        # Store candle
                        self.candles[symbol].append(candle)

                        # Clear aggregator
                        del self.candle_aggregators[symbol]

            except Exception as e:
                logger.error("Error in candle aggregation task", error=str(e))

    async def _spread_monitoring_task(self) -> None:
        """Background task for spread monitoring."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                for symbol, history in self.spread_history.items():
                    if not history:
                        continue

                    # Calculate spread persistence
                    recent = list(history)[-20:]  # Last 20 readings
                    if len(recent) >= 10:
                        spreads = [r["spread_bp"] for r in recent]
                        avg_spread = sum(spreads) / len(spreads)
                        min_spread = min(spreads)
                        max_spread = max(spreads)

                        # Check for persistent tight spreads
                        if avg_spread < 15 and max_spread - min_spread < 5:
                            logger.info(
                                "Persistent tight spread detected",
                                symbol=symbol,
                                avg_spread=avg_spread,
                                range=max_spread - min_spread,
                            )

            except Exception as e:
                logger.error("Error in spread monitoring task", error=str(e))

    async def _volume_analysis_task(self) -> None:
        """Background task for volume analysis and persistence."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                for symbol, profile in self.volume_profiles.items():
                    current_hour = datetime.now().hour
                    current_volume = profile.hour_volumes.get(
                        current_hour, Decimal("0")
                    )

                    # Check for anomalies
                    if profile.is_volume_anomaly(current_volume):
                        if self.event_bus:
                            event = Event(
                                event_type=EventType.VOLUME_ANOMALY,
                                aggregate_id=symbol,
                                event_data={
                                    "symbol": symbol,
                                    "current_volume": str(current_volume),
                                    "average_volume": str(
                                        profile.average_hourly_volume
                                    ),
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                            await self.event_bus.publish(event, EventPriority.NORMAL)

                    # Persist volume profile to database
                    if self.repository:
                        await self._persist_volume_profile(symbol, profile)

            except Exception as e:
                logger.error("Error in volume analysis task", error=str(e))

    async def _persist_volume_profile(
        self, symbol: str, profile: VolumeProfile
    ) -> None:
        """
        Persist volume profile to database.

        Args:
            symbol: Trading symbol
            profile: Volume profile to persist
        """
        try:
            from genesis.data.sqlite_repo import SQLiteRepository

            if not isinstance(self.repository, SQLiteRepository):
                return

            # Prepare volume profile data for each hour
            current_date = datetime.now().date()

            for hour, volume in profile.hour_volumes.items():
                profile_id = str(uuid4())

                # Calculate average trade size if possible
                trade_count = len(
                    [
                        t
                        for t in self.ticks.get(symbol, [])
                        if datetime.fromtimestamp(t.timestamp).hour == hour
                    ]
                )
                avg_trade_size = str(volume / Decimal(str(max(trade_count, 1))))

                # Insert or update volume profile
                await self.repository.connection.execute(
                    """
                    INSERT OR REPLACE INTO volume_profiles
                    (profile_id, symbol, hour, volume, trade_count, average_trade_size, date, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        profile_id,
                        symbol,
                        hour,
                        str(volume),
                        trade_count,
                        avg_trade_size,
                        current_date.isoformat(),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )

            await self.repository.connection.commit()
            logger.debug(f"Persisted volume profile for {symbol}")

        except Exception as e:
            logger.error(
                "Failed to persist volume profile", symbol=symbol, error=str(e)
            )

    def get_statistics(self) -> dict:
        """Get service statistics."""
        return {
            "active_subscriptions": len(self.active_subscriptions),
            "tracked_symbols": len(self.current_prices),
            "order_books": len(self.order_books),
            "candle_buffers": {
                symbol: len(candles) for symbol, candles in self.candles.items()
            },
            "tick_buffers": {
                symbol: len(ticks) for symbol, ticks in self.ticks.items()
            },
            "market_states": dict(self.market_states),
            "websocket_stats": self.websocket_manager.get_statistics(),
        }

    async def get_price_history(self, symbol: str, window_days: int) -> list[Decimal]:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Trading symbol
            window_days: Number of days of history to retrieve

        Returns:
            List of historical prices
        """
        try:
            # Check cache first
            if symbol in self.candles and len(self.candles[symbol]) > 0:
                candles = list(self.candles[symbol])
                prices = [c.close for c in candles]

                # If we have enough cached data, return it
                candles_per_day = 24 * 60  # Assuming 1-minute candles
                required_candles = window_days * candles_per_day

                if len(prices) >= required_candles:
                    return prices[-required_candles:]

            # Fetch from exchange if not enough cached data
            if self.gateway:
                klines = await self.gateway.get_historical_klines(
                    symbol=symbol, interval="1m", limit=min(1000, window_days * 24 * 60)
                )

                prices = []
                for kline in klines:
                    # kline format: [open_time, open, high, low, close, volume, ...]
                    close_price = Decimal(str(kline[4]))
                    prices.append(close_price)

                logger.info(f"Retrieved {len(prices)} historical prices for {symbol}")
                return prices

            # Return what we have from cache
            if symbol in self.candles:
                return [c.close for c in self.candles[symbol]]

            return []

        except Exception as e:
            logger.error(f"Failed to get price history for {symbol}: {e}")
            return []

    async def subscribe_multi_pair(self, symbols: list[str]) -> None:
        """
        Subscribe to multiple trading pairs for correlation monitoring.

        Args:
            symbols: List of trading symbols to subscribe to
        """
        for symbol in symbols:
            if symbol not in self.active_subscriptions:
                await self.subscribe(symbol)

        logger.info(f"Subscribed to {len(symbols)} pairs for multi-pair monitoring")

    def get_correlated_pairs_data(
        self, pairs: list[tuple[str, str]]
    ) -> dict[str, list[Decimal]]:
        """
        Get current price data for correlated pairs.

        Args:
            pairs: List of trading pair tuples

        Returns:
            Dictionary mapping pair keys to price lists
        """
        result = {}

        for pair1, pair2 in pairs:
            # Get prices for each pair
            if pair1 in self.candles and pair2 in self.candles:
                prices1 = [c.close for c in self.candles[pair1]]
                prices2 = [c.close for c in self.candles[pair2]]

                # Ensure equal length
                min_len = min(len(prices1), len(prices2))
                if min_len > 0:
                    result[pair1] = prices1[-min_len:]
                    result[pair2] = prices2[-min_len:]

        return result

    async def start_correlation_monitoring(self, pairs: list[tuple[str, str]]) -> None:
        """
        Start monitoring correlations between specified pairs.

        Args:
            pairs: List of pair tuples to monitor
        """
        # Subscribe to all unique symbols
        unique_symbols = set()
        for pair1, pair2 in pairs:
            unique_symbols.add(pair1)
            unique_symbols.add(pair2)

        await self.subscribe_multi_pair(list(unique_symbols))

        logger.info(
            f"Started correlation monitoring for {len(pairs)} pair combinations"
        )

    async def get_all_trading_pairs(self) -> list[str]:
        """
        Get all available trading pairs from the exchange.

        Returns:
            List of trading pair symbols
        """
        try:
            if self.gateway:
                # Get exchange info from gateway
                exchange_info = await self.gateway.get_exchange_info()

                # Filter for active USDT pairs
                pairs = []
                for symbol_info in exchange_info.get("symbols", []):
                    if (
                        symbol_info.get("status") == "TRADING"
                        and symbol_info.get("quoteAsset") == "USDT"
                    ):
                        pairs.append(symbol_info.get("symbol"))

                logger.info(f"Retrieved {len(pairs)} active USDT trading pairs")
                return pairs

            return []

        except Exception as e:
            logger.error(f"Failed to get trading pairs: {e}")
            return []

    async def get_order_book_snapshot(
        self, symbol: str, limit: int = 10
    ) -> OrderBook | None:
        """
        Get order book snapshot with specified depth.

        Args:
            symbol: Trading pair symbol
            limit: Number of levels to fetch (default 10)

        Returns:
            OrderBook snapshot or None if error
        """
        try:
            if self.gateway:
                # Fetch order book from exchange
                depth_data = await self.gateway.get_order_book(symbol, limit)

                # Create order book object
                order_book = OrderBook(symbol=symbol)
                order_book.timestamp = datetime.now().timestamp()

                # Parse bids
                for bid in depth_data.get("bids", []):
                    order_book.bids.append(
                        OrderBookLevel(price=Decimal(bid[0]), quantity=Decimal(bid[1]))
                    )

                # Parse asks
                for ask in depth_data.get("asks", []):
                    order_book.asks.append(
                        OrderBookLevel(price=Decimal(ask[0]), quantity=Decimal(ask[1]))
                    )

                # Cache it
                self.order_books[symbol] = order_book

                return order_book

            # Return cached version if available
            return self.order_books.get(symbol)

        except Exception as e:
            logger.error(f"Failed to get order book snapshot for {symbol}: {e}")
            return None

    async def get_historical_candles(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[Candle]:
        """
        Get historical candles for volatility calculation.

        Args:
            symbol: Trading symbol
            interval: Candle interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of candles to fetch

        Returns:
            List of Candle objects
        """
        try:
            if not self.gateway:
                logger.warning("No gateway available for historical data")
                return []

            # Fetch from exchange
            klines = await self.gateway.get_klines(symbol, interval, limit)

            candles = []
            for kline in klines:
                candle = Candle(
                    symbol=symbol,
                    open=Decimal(str(kline[1])),
                    high=Decimal(str(kline[2])),
                    low=Decimal(str(kline[3])),
                    close=Decimal(str(kline[4])),
                    volume=Decimal(str(kline[5])),
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    trades=int(kline[8]) if len(kline) > 8 else 0,
                )
                candles.append(candle)

            # Cache recent candles
            if symbol not in self.candles:
                self.candles[symbol] = deque(maxlen=1000)

            for candle in candles[-100:]:  # Cache last 100
                self.candles[symbol].append(candle)

            return candles

        except Exception as e:
            logger.error(f"Failed to get historical candles for {symbol}: {e}")
            return []

    async def get_volatility_data(
        self, symbol: str, period: int = 14
    ) -> dict[str, Decimal]:
        """
        Get volatility data for market state classification.

        Args:
            symbol: Trading symbol
            period: Period for calculations

        Returns:
            Dict with volatility metrics
        """
        try:
            # Get historical candles
            candles = await self.get_historical_candles(symbol, "1h", period * 2)

            if len(candles) < period:
                logger.warning(
                    f"Insufficient data for volatility calculation: {len(candles)} < {period}"
                )
                return {}

            # Extract price data
            high_prices = [c.high for c in candles]
            low_prices = [c.low for c in candles]
            close_prices = [c.close for c in candles]

            # Calculate ATR
            true_ranges = []
            for i in range(1, len(candles)):
                high_low = candles[i].high - candles[i].low
                high_close = abs(candles[i].high - candles[i - 1].close)
                low_close = abs(candles[i].low - candles[i - 1].close)
                true_range = max(high_low, high_close, low_close)
                true_ranges.append(true_range)

            # Simple ATR calculation
            atr = (
                sum(true_ranges[-period:]) / Decimal(period)
                if true_ranges
                else Decimal(0)
            )

            # Calculate realized volatility (simplified)
            returns = []
            for i in range(1, len(close_prices)):
                if close_prices[i - 1] > 0:
                    ret = (close_prices[i] - close_prices[i - 1]) / close_prices[i - 1]
                    returns.append(ret)

            if returns:
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                realized_vol = variance.sqrt() if variance > 0 else Decimal(0)
            else:
                realized_vol = Decimal(0)

            return {
                "atr": atr,
                "realized_volatility": realized_vol,
                "current_price": close_prices[-1] if close_prices else Decimal(0),
            }

        except Exception as e:
            logger.error(f"Failed to calculate volatility for {symbol}: {e}")
            return {}

    async def update_market_state(
        self, symbol: str, state: MarketState, reason: str = ""
    ) -> None:
        """
        Update market state for a symbol.

        Args:
            symbol: Trading symbol
            state: New market state
            reason: Reason for state change
        """
        old_state = self.market_states.get(symbol)
        self.market_states[symbol] = state

        # Publish event if state changed
        if old_state != state and self.event_bus:
            event = Event(
                id=str(uuid4()),
                type=EventType.MARKET_STATE_CHANGE,
                timestamp=datetime.now(),
                data={
                    "symbol": symbol,
                    "old_state": old_state.value if old_state else None,
                    "new_state": state.value,
                    "reason": reason,
                },
                priority=(
                    EventPriority.HIGH
                    if state == MarketState.PANIC
                    else EventPriority.MEDIUM
                ),
            )
            await self.event_bus.publish(event)

        logger.info(
            f"Market state updated for {symbol}: {old_state} -> {state}, reason: {reason}"
        )

    async def get_24h_ticker(self, symbol: str) -> dict[str, Decimal]:
        """
        Get 24-hour ticker statistics.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with ticker data including volume
        """
        try:
            if not self.gateway:
                return {}

            ticker = await self.gateway.get_24h_ticker(symbol)

            return {
                "volume": Decimal(str(ticker.get("volume", 0))),
                "quote_volume": Decimal(str(ticker.get("quoteVolume", 0))),
                "price_change_percent": Decimal(
                    str(ticker.get("priceChangePercent", 0))
                ),
                "high": Decimal(str(ticker.get("highPrice", 0))),
                "low": Decimal(str(ticker.get("lowPrice", 0))),
                "weighted_avg_price": Decimal(str(ticker.get("weightedAvgPrice", 0))),
            }

        except Exception as e:
            logger.error(f"Failed to get 24h ticker for {symbol}: {e}")
            return {}

    async def _spread_persistence_task(self) -> None:
        """Background task for persisting spread data to database."""
        while True:
            try:
                await asyncio.sleep(60)  # Persist every minute

                if not self.repository:
                    continue

                # Get all current spread metrics
                all_metrics = self.spread_analyzer.get_all_metrics()

                for symbol, metrics in all_metrics.items():
                    # Calculate order imbalance if we have orderbook
                    order_imbalance = Decimal("1.0")
                    if symbol in self.order_books:
                        orderbook_dict = {
                            "bids": [
                                [str(level.price), str(level.quantity)]
                                for level in self.order_books[symbol].bids
                            ],
                            "asks": [
                                [str(level.price), str(level.quantity)]
                                for level in self.order_books[symbol].asks
                            ],
                        }
                        imbalance = self.spread_analyzer.calculate_order_imbalance(
                            orderbook_dict
                        )
                        order_imbalance = imbalance.ratio

                    # Prepare spread data for persistence
                    spread_data = {
                        "symbol": symbol,
                        "spread_bps": metrics.spread_bps,
                        "bid_price": metrics.bid_price,
                        "ask_price": metrics.ask_price,
                        "bid_volume": metrics.bid_volume,
                        "ask_volume": metrics.ask_volume,
                        "order_imbalance": order_imbalance,
                        "timestamp": metrics.timestamp,
                    }

                    # Save to database
                    await self.repository.save_spread_history(spread_data)

                logger.debug(f"Persisted spread data for {len(all_metrics)} symbols")

            except Exception as e:
                logger.error("Error in spread persistence task", error=str(e))

    def get_spread_analytics(self, symbol: str) -> dict:
        """
        Get comprehensive spread analytics for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with spread analytics
        """
        metrics = self.spread_analyzer.get_spread_metrics(symbol)
        patterns = self.spread_tracker.identify_spread_patterns(symbol)
        compression = self.spread_analyzer.detect_spread_compression(symbol)

        return {
            "current_metrics": metrics,
            "patterns": patterns,
            "compression_event": compression,
            "compression_duration": self.spread_analyzer.get_compression_duration(
                symbol
            ),
        }
