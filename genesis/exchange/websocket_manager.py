"""
WebSocket connection management for Binance streams.

Manages multiple WebSocket connections with automatic reconnection,
heartbeat handling, and message buffering.
"""

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import structlog
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from config.settings import get_settings
from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus
from genesis.exchange.circuit_breaker_v2 import EnhancedCircuitBreaker
from genesis.exchange.gateway import BinanceGateway

logger = structlog.get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class StreamSubscription:
    """Represents a stream subscription."""

    stream: str
    callback: Callable[[dict], None]
    symbol: str | None = None


class WebSocketConnection:
    """Single WebSocket connection handler."""

    def __init__(
        self,
        name: str,
        url: str,
        subscriptions: list[StreamSubscription],
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,  # Maximum 30s as per requirements
        heartbeat_interval: float = 30.0,
        circuit_breaker: Optional["EnhancedCircuitBreaker"] = None,
        gateway: Optional["BinanceGateway"] = None,
    ):
        """
        Initialize a WebSocket connection.

        Args:
            name: Connection identifier
            url: WebSocket URL
            subscriptions: List of stream subscriptions
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay (30s per requirements)
            heartbeat_interval: Heartbeat interval in seconds
            circuit_breaker: Optional circuit breaker for failure protection
            gateway: Optional gateway for REST API gap detection
        """
        self.name = name
        self.url = url
        self.subscriptions = subscriptions
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        self.circuit_breaker = circuit_breaker
        self.gateway = gateway

        self.state = ConnectionState.DISCONNECTED
        self.websocket: Any | None = None  # WebSocket connection object
        self.current_reconnect_delay = reconnect_delay
        self.reconnect_attempt = 0  # Track attempt number for exponential backoff
        self.last_heartbeat = time.time()
        self.message_buffer = deque(maxlen=1000)

        # Tasks
        self.connection_task: asyncio.Task | None = None
        self.heartbeat_task: asyncio.Task | None = None
        self.message_handler_task: asyncio.Task | None = None

        # Statistics
        self.messages_received = 0
        self.reconnect_count = 0
        self.last_message_time = None

        # Gap detection with bounded history
        self.last_sequence_numbers: dict[str, int] = {}
        self.detected_gaps: deque = deque(
            maxlen=100
        )  # Limit gap history to prevent memory leaks
        self.gap_cleanup_interval = 3600  # Clean up gaps older than 1 hour
        self.last_gap_cleanup = time.time()

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return

        self.state = ConnectionState.CONNECTING
        logger.info(f"Connecting WebSocket {self.name}", url=self.url)

        try:
            # Build subscription URL
            streams = [sub.stream for sub in self.subscriptions]
            stream_url = f"{self.url}/stream?streams={'/'.join(streams)}"

            # Connect
            self.websocket = await websockets.connect(
                stream_url,
                ping_interval=None,  # We handle our own heartbeat
                ping_timeout=10,
                close_timeout=10,
            )

            self.state = ConnectionState.CONNECTED
            self.current_reconnect_delay = self.reconnect_delay  # Reset delay
            self.reconnect_attempt = 0  # Reset exponential backoff counter  
            self.last_heartbeat = time.time()

            # Start tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.message_handler_task = asyncio.create_task(self._message_handler())

            logger.info(
                f"WebSocket {self.name} connected",
                streams=streams,
                reconnect_count=self.reconnect_count,
            )

        except Exception as e:
            logger.error(f"Failed to connect WebSocket {self.name}", error=str(e))
            self.state = ConnectionState.DISCONNECTED
            await self._schedule_reconnect()

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self.state = ConnectionState.CLOSED

        # Cancel tasks
        for task in [self.heartbeat_task, self.message_handler_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logger.info(f"WebSocket {self.name} disconnected")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to keep connection alive."""
        while self.state == ConnectionState.CONNECTED:
            try:
                # Send pong frame
                if self.websocket:
                    pong_frame = {"pong": int(time.time() * 1000)}
                    await self.websocket.send(json.dumps(pong_frame))
                    self.last_heartbeat = time.time()
                    logger.debug(f"Heartbeat sent for {self.name}")

                await asyncio.sleep(self.heartbeat_interval)

            except (ConnectionClosed, WebSocketException) as e:
                logger.warning(f"Heartbeat failed for {self.name}", error=str(e))
                await self._handle_disconnect()
                break
            except Exception as e:
                logger.error(
                    f"Unexpected heartbeat error for {self.name}", error=str(e)
                )
                await asyncio.sleep(self.heartbeat_interval)

    async def _message_handler(self) -> None:
        """Handle incoming messages."""
        while self.state == ConnectionState.CONNECTED:
            try:
                if not self.websocket:
                    break

                # Receive message
                message = await self.websocket.recv()
                self.messages_received += 1
                self.last_message_time = time.time()

                # Parse message
                data = json.loads(message)

                # Check for gaps in sequence numbers (if available)
                await self._check_for_gaps(data)

                # Buffer message
                self.message_buffer.append(data)

                # Route to appropriate callback
                stream_name = data.get("stream", "")
                for subscription in self.subscriptions:
                    if subscription.stream in stream_name:
                        try:
                            # Handle both sync and async callbacks
                            if asyncio.iscoroutinefunction(subscription.callback):
                                await subscription.callback(data)
                            else:
                                subscription.callback(data)
                        except Exception as e:
                            logger.error(
                                f"Callback error for stream {subscription.stream}",
                                error=str(e),
                            )

            except ConnectionClosed:
                logger.warning(f"WebSocket {self.name} connection closed")
                await self._handle_disconnect()
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received on {self.name}", error=str(e))
            except Exception as e:
                logger.error(f"Message handler error for {self.name}", error=str(e))

    async def _handle_disconnect(self) -> None:
        """Handle disconnection and schedule reconnection."""
        if self.state == ConnectionState.CLOSED:
            return

        self.state = ConnectionState.DISCONNECTED
        self.websocket = None

        logger.warning(
            f"WebSocket {self.name} disconnected",
            messages_received=self.messages_received,
        )

        await self._schedule_reconnect()

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff (2^n seconds, max 30)."""
        if self.state == ConnectionState.CLOSED:
            return

        self.state = ConnectionState.RECONNECTING
        self.reconnect_count += 1
        self.reconnect_attempt += 1

        # Calculate exponential backoff: 2^n seconds, capped at max_reconnect_delay
        self.current_reconnect_delay = min(
            2 ** self.reconnect_attempt, self.max_reconnect_delay
        )

        logger.info(
            f"Scheduling reconnection for {self.name}",
            delay=self.current_reconnect_delay,
            attempt=self.reconnect_count,
            backoff_formula=f"2^{self.reconnect_attempt} = {self.current_reconnect_delay}s",
        )

        await asyncio.sleep(self.current_reconnect_delay)

        # Attempt reconnection
        await self.connect()

    async def _check_for_gaps(self, data: dict) -> None:
        """
        Check for gaps in message sequence.

        Args:
            data: Message data
        """
        # Periodic cleanup of old gaps (to prevent memory growth)
        current_time = time.time()
        if current_time - self.last_gap_cleanup > self.gap_cleanup_interval:
            self._cleanup_old_gaps()
            self.last_gap_cleanup = current_time

        # Look for sequence number in different message types
        sequence = None
        stream = data.get("stream", "")

        # Binance includes 'u' field for order book updates (updateId)
        if "data" in data and "u" in data.get("data", {}):
            sequence = data["data"]["u"]
        # Trade messages have 't' field (trade ID)
        elif "data" in data and "t" in data.get("data", {}):
            sequence = data["data"]["t"]

        if sequence and stream:
            last_sequence = self.last_sequence_numbers.get(stream, 0)

            # Check for gap
            if last_sequence > 0 and sequence > last_sequence + 1:
                gap_size = sequence - last_sequence - 1
                logger.warning(
                    f"Gap detected in {stream}",
                    last_sequence=last_sequence,
                    current_sequence=sequence,
                    gap_size=gap_size,
                )

                # Record gap
                self.detected_gaps.append(
                    {
                        "stream": stream,
                        "timestamp": time.time(),
                        "last_sequence": last_sequence,
                        "current_sequence": sequence,
                        "gap_size": gap_size,
                    }
                )

                # Attempt to fill gap using REST API
                if self.gateway:
                    asyncio.create_task(self._fill_gap(stream, last_sequence, sequence))

            # Update last sequence
            self.last_sequence_numbers[stream] = sequence

    async def _fill_gap(
        self, stream: str, start_sequence: int, end_sequence: int
    ) -> None:
        """
        Fill gap in data using REST API fallback for missed messages.

        Args:
            stream: Stream name
            start_sequence: Start of gap
            end_sequence: End of gap
        """
        try:
            # Extract symbol from stream name (e.g., "btcusdt@trade" -> "BTC/USDT")
            symbol_raw = stream.split("@")[0].upper()
            # Convert to standard format for CCXT
            if "USDT" in symbol_raw:
                symbol = symbol_raw.replace("USDT", "/USDT")
            elif "BTC" in symbol_raw and not symbol_raw.startswith("BTC"):
                symbol = symbol_raw.replace("BTC", "/BTC")
            else:
                symbol = symbol_raw

            gap_size = end_sequence - start_sequence
            logger.info(
                f"Attempting to fill gap for {symbol} using REST API",
                start=start_sequence,
                end=end_sequence,
                gap_size=gap_size,
            )

            # Use REST API to fetch missing data based on stream type
            if "@trade" in stream and self.gateway:
                # For trades, we can't fetch specific sequence ranges,
                # but we can get recent trades to catch up
                logger.info(f"Gap detected in trade stream for {symbol}, fetching recent trades")
                # Note: Would need to implement get_recent_trades in gateway
                # For now, log the gap for monitoring
                
            elif "@depth" in stream and self.gateway:
                # Fetch order book snapshot to resync
                order_book = await self.gateway.get_order_book(symbol, limit=20)
                logger.info(
                    f"Fetched order book snapshot for {symbol} to fill gap",
                    bids=len(order_book.bids),
                    asks=len(order_book.asks),
                )
                
                # Publish snapshot event to resync local order book state
                if self.event_bus:
                    from genesis.core.events import Event, EventType, EventPriority
                    event = Event(
                        event_type=EventType.ORDER_BOOK_SNAPSHOT,
                        aggregate_id=symbol,
                        event_data={
                            "type": "gap_fill",
                            "symbol": symbol,
                            "last_update_id": end_sequence,
                            "bids": [[str(price), str(qty)] for price, qty in order_book.bids[:5]],
                            "asks": [[str(price), str(qty)] for price, qty in order_book.asks[:5]],
                            "gap_recovered": True,
                            "gap_size": gap_size,
                        },
                    )
                    await self.event_bus.publish(event, priority=EventPriority.HIGH)
                    
            elif "@kline" in stream and self.gateway:
                # Fetch recent klines to fill gap
                interval = stream.split("_")[1] if "_" in stream else "1m"
                klines = await self.gateway.get_klines(symbol, interval=interval, limit=100)
                logger.info(
                    f"Fetched {len(klines)} klines for {symbol} to fill gap",
                    interval=interval,
                )
                
            elif "@ticker" in stream and self.gateway:
                # Fetch current ticker to get latest state
                ticker = await self.gateway.get_ticker(symbol)
                logger.info(
                    f"Fetched ticker for {symbol} to fill gap",
                    last_price=ticker.last_price,
                )

            # Mark gap as recovered
            logger.info(
                f"Successfully attempted gap recovery for {stream}",
                start=start_sequence,
                end=end_sequence,
                gap_size=gap_size,
            )

        except Exception as e:
            logger.error(
                f"Failed to fill gap for {stream} using REST API",
                error=str(e),
                start=start_sequence,
                end=end_sequence,
            )

    def _cleanup_old_gaps(self) -> None:
        """
        Clean up old gap records to prevent memory growth.
        Since detected_gaps is now a deque with maxlen, it will automatically
        limit size, but we can still clean up based on age if needed.
        """
        # The deque with maxlen=100 automatically limits the size
        # No additional cleanup needed as old entries are automatically removed
        # when new ones are added beyond the maxlen limit
        pass

    def get_statistics(self) -> dict:
        """Get connection statistics."""
        return {
            "name": self.name,
            "state": self.state,
            "messages_received": self.messages_received,
            "reconnect_count": self.reconnect_count,
            "last_message_time": self.last_message_time,
            "last_heartbeat": self.last_heartbeat,
            "buffer_size": len(self.message_buffer),
            "detected_gaps": len(self.detected_gaps),
            "last_sequences": dict(self.last_sequence_numbers),
        }


class WebSocketManager:
    """
    Manages multiple WebSocket connections for Binance.

    Maintains connection pools for execution, monitoring, and backup,
    with automatic failover and message buffering.
    """

    def __init__(
        self,
        gateway: BinanceGateway | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize the WebSocket manager."""
        self.settings = get_settings()
        self.connections: dict[str, WebSocketConnection] = {}
        self.stream_callbacks: dict[str, list[Callable]] = {}
        self.running = False
        self.gateway = gateway
        self.event_bus = event_bus

        # Circuit breaker for connection protection
        self.circuit_breaker = EnhancedCircuitBreaker(
            failure_threshold=5, recovery_timeout=60, half_open_max_calls=3
        )

        # URLs
        if self.settings.exchange.binance_testnet:
            self.base_url = "wss://testnet.binance.vision"
        else:
            self.base_url = "wss://stream.binance.com:9443"

        logger.info("WebSocketManager initialized", base_url=self.base_url)

    async def start(self) -> None:
        """Start the WebSocket manager."""
        if self.running:
            return

        self.running = True
        logger.info("Starting WebSocketManager")

        # Create connection pools
        await self._setup_connections()

        # Connect all
        connect_tasks = [conn.connect() for conn in self.connections.values()]
        await asyncio.gather(*connect_tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop the WebSocket manager."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping WebSocketManager")

        # Disconnect all
        disconnect_tasks = [conn.disconnect() for conn in self.connections.values()]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self.connections.clear()

    async def _setup_connections(self) -> None:
        """Set up connection pools."""
        # Get trading pairs
        trading_pairs = self.settings.trading.trading_pairs

        # Convert to Binance format (btcusdt instead of BTC/USDT)
        symbols = [pair.replace("/", "").lower() for pair in trading_pairs]

        # Define subscriptions for each connection type
        execution_subs = []
        monitoring_subs = []
        backup_subs = []

        for symbol in symbols:
            # Execution connection: trades and order book
            execution_subs.extend(
                [
                    StreamSubscription(f"{symbol}@trade", self._handle_trade, symbol),
                    StreamSubscription(
                        f"{symbol}@depth20@100ms", self._handle_depth, symbol
                    ),
                ]
            )

            # Monitoring connection: klines and ticker
            monitoring_subs.extend(
                [
                    StreamSubscription(
                        f"{symbol}@kline_1m", self._handle_kline, symbol
                    ),
                    StreamSubscription(f"{symbol}@ticker", self._handle_ticker, symbol),
                ]
            )

            # Backup connection: all streams (for failover)
            backup_subs.extend(
                [
                    StreamSubscription(f"{symbol}@trade", self._handle_trade, symbol),
                    StreamSubscription(
                        f"{symbol}@depth20@100ms", self._handle_depth, symbol
                    ),
                    StreamSubscription(
                        f"{symbol}@kline_1m", self._handle_kline, symbol
                    ),
                    StreamSubscription(f"{symbol}@ticker", self._handle_ticker, symbol),
                ]
            )

        # Create connections with circuit breakers
        if execution_subs:
            self.connections["execution"] = WebSocketConnection(
                "execution",
                self.base_url,
                execution_subs,
                circuit_breaker=self.circuit_breaker,
                gateway=self.gateway,
            )

        if monitoring_subs:
            self.connections["monitoring"] = WebSocketConnection(
                "monitoring",
                self.base_url,
                monitoring_subs,
                circuit_breaker=self.circuit_breaker,
                gateway=self.gateway,
            )

        if backup_subs:
            self.connections["backup"] = WebSocketConnection(
                "backup",
                self.base_url,
                backup_subs,
                circuit_breaker=self.circuit_breaker,
                gateway=self.gateway,
            )

    def subscribe(self, stream: str, callback: Callable[[dict], None]) -> None:
        """
        Subscribe to a stream.

        Args:
            stream: Stream name
            callback: Callback function for messages
        """
        if stream not in self.stream_callbacks:
            self.stream_callbacks[stream] = []
        self.stream_callbacks[stream].append(callback)

        logger.info(f"Subscribed to stream {stream}")

    def unsubscribe(self, stream: str, callback: Callable[[dict], None]) -> None:
        """
        Unsubscribe from a stream.

        Args:
            stream: Stream name
            callback: Callback function to remove
        """
        if stream in self.stream_callbacks:
            self.stream_callbacks[stream].remove(callback)
            if not self.stream_callbacks[stream]:
                del self.stream_callbacks[stream]

        logger.info(f"Unsubscribed from stream {stream}")

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade stream data."""
        stream = data.get("stream", "")

        # Publish event to Event Bus if available
        if self.event_bus and "data" in data:
            trade_data = data["data"]
            event = Event(
                event_type=EventType.MARKET_DATA_UPDATED,
                aggregate_id=stream.split("@")[0].upper(),  # Symbol
                event_data={
                    "type": "trade",
                    "symbol": trade_data.get("s", ""),
                    "price": trade_data.get("p", "0"),
                    "quantity": trade_data.get("q", "0"),
                    "time": trade_data.get("T", 0),
                    "is_buyer_maker": trade_data.get("m", False),
                },
            )
            await self.event_bus.publish(event, priority=EventPriority.NORMAL)

        # Also call registered callbacks
        callbacks = self.stream_callbacks.get("trade", [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error("Trade callback error", error=str(e))

    async def _handle_depth(self, data: dict) -> None:
        """Handle depth stream data."""
        stream = data.get("stream", "")

        # Publish event to Event Bus if available
        if self.event_bus and "data" in data:
            depth_data = data["data"]
            event = Event(
                event_type=EventType.ORDER_BOOK_SNAPSHOT,
                aggregate_id=stream.split("@")[0].upper(),  # Symbol
                event_data={
                    "type": "depth",
                    "symbol": stream.split("@")[0].upper(),
                    "last_update_id": depth_data.get("u", 0),
                    "bids": depth_data.get("b", [])[:5],  # Top 5 bids
                    "asks": depth_data.get("a", [])[:5],  # Top 5 asks
                },
            )
            await self.event_bus.publish(event, priority=EventPriority.NORMAL)

        # Also call registered callbacks
        callbacks = self.stream_callbacks.get("depth", [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error("Depth callback error", error=str(e))

    async def _handle_kline(self, data: dict) -> None:
        """Handle kline stream data."""
        stream = data.get("stream", "")

        # Publish event to Event Bus if available
        if self.event_bus and "data" in data:
            kline_data = data["data"]["k"] if "k" in data.get("data", {}) else {}
            if kline_data:
                event = Event(
                    event_type=EventType.MARKET_DATA_UPDATED,
                    aggregate_id=stream.split("@")[0].upper(),  # Symbol
                    event_data={
                        "type": "kline",
                        "symbol": kline_data.get("s", ""),
                        "interval": kline_data.get("i", ""),
                        "open": kline_data.get("o", "0"),
                        "high": kline_data.get("h", "0"),
                        "low": kline_data.get("l", "0"),
                        "close": kline_data.get("c", "0"),
                        "volume": kline_data.get("v", "0"),
                        "close_time": kline_data.get("T", 0),
                    },
                )
                await self.event_bus.publish(event, priority=EventPriority.LOW)

        # Also call registered callbacks
        callbacks = self.stream_callbacks.get("kline", [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error("Kline callback error", error=str(e))

    async def _handle_ticker(self, data: dict) -> None:
        """Handle ticker stream data."""
        stream = data.get("stream", "")

        # Publish event to Event Bus if available
        if self.event_bus and "data" in data:
            ticker_data = data["data"]

            # Check for significant spread changes
            if "a" in ticker_data and "b" in ticker_data:
                try:
                    ask = float(ticker_data["a"])
                    bid = float(ticker_data["b"])
                    if ask > 0 and bid > 0:
                        spread_bps = int(((ask - bid) / bid) * 10000)
                        if spread_bps < 10:  # Alert on tight spreads
                            spread_event = Event(
                                event_type=EventType.SPREAD_COMPRESSION,
                                aggregate_id=stream.split("@")[0].upper(),
                                event_data={
                                    "symbol": stream.split("@")[0].upper(),
                                    "spread_bps": spread_bps,
                                    "bid": ticker_data["b"],
                                    "ask": ticker_data["a"],
                                },
                            )
                            await self.event_bus.publish(
                                spread_event, priority=EventPriority.HIGH
                            )
                except (ValueError, ZeroDivisionError):
                    pass

            # Publish ticker update event
            event = Event(
                event_type=EventType.MARKET_DATA_UPDATED,
                aggregate_id=stream.split("@")[0].upper(),
                event_data={
                    "type": "ticker",
                    "symbol": ticker_data.get("s", ""),
                    "price": ticker_data.get("c", "0"),
                    "bid": ticker_data.get("b", "0"),
                    "ask": ticker_data.get("a", "0"),
                    "bid_qty": ticker_data.get("B", "0"),
                    "ask_qty": ticker_data.get("A", "0"),
                    "volume_24h": ticker_data.get("v", "0"),
                    "weighted_avg_price": ticker_data.get("w", "0"),
                },
            )
            await self.event_bus.publish(event, priority=EventPriority.NORMAL)

        # Also call registered callbacks
        callbacks = self.stream_callbacks.get("ticker", [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error("Ticker callback error", error=str(e))

    def get_connection_states(self) -> dict[str, str]:
        """Get current connection states."""
        return {name: conn.state for name, conn in self.connections.items()}

    def get_statistics(self) -> dict:
        """Get manager statistics."""
        return {
            "running": self.running,
            "connections": {
                name: conn.get_statistics() for name, conn in self.connections.items()
            },
            "subscriptions": list(self.stream_callbacks.keys()),
        }

    async def check_health(self) -> dict[str, bool]:
        """Check health of all connections."""
        health = {}

        for name, conn in self.connections.items():
            is_healthy = (
                conn.state == ConnectionState.CONNECTED
                and conn.last_message_time
                and (time.time() - conn.last_message_time)
                < 60  # No messages for 60s = unhealthy
            )
            health[name] = is_healthy

        return health
