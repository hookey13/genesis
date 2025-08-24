"""
WebSocket connection management for Binance streams.

Manages multiple WebSocket connections with automatic reconnection,
heartbeat handling, and message buffering.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from config.settings import get_settings


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
    callback: Callable[[Dict], None]
    symbol: Optional[str] = None


class WebSocketConnection:
    """Single WebSocket connection handler."""
    
    def __init__(
        self,
        name: str,
        url: str,
        subscriptions: List[StreamSubscription],
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        heartbeat_interval: float = 30.0
    ):
        """
        Initialize a WebSocket connection.
        
        Args:
            name: Connection identifier
            url: WebSocket URL
            subscriptions: List of stream subscriptions
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.name = name
        self.url = url
        self.subscriptions = subscriptions
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.current_reconnect_delay = reconnect_delay
        self.last_heartbeat = time.time()
        self.message_buffer = deque(maxlen=1000)
        
        # Tasks
        self.connection_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_handler_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.messages_received = 0
        self.reconnect_count = 0
        self.last_message_time = None
    
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
                close_timeout=10
            )
            
            self.state = ConnectionState.CONNECTED
            self.current_reconnect_delay = self.reconnect_delay  # Reset delay
            self.last_heartbeat = time.time()
            
            # Start tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.message_handler_task = asyncio.create_task(self._message_handler())
            
            logger.info(
                f"WebSocket {self.name} connected",
                streams=streams,
                reconnect_count=self.reconnect_count
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
                logger.error(f"Unexpected heartbeat error for {self.name}", error=str(e))
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
                
                # Buffer message
                self.message_buffer.append(data)
                
                # Route to appropriate callback
                stream_name = data.get("stream", "")
                for subscription in self.subscriptions:
                    if subscription.stream in stream_name:
                        try:
                            await asyncio.create_task(
                                asyncio.coroutine(subscription.callback)(data)
                            )
                        except Exception as e:
                            logger.error(
                                f"Callback error for stream {subscription.stream}",
                                error=str(e)
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
            messages_received=self.messages_received
        )
        
        await self._schedule_reconnect()
    
    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self.state == ConnectionState.CLOSED:
            return
        
        self.state = ConnectionState.RECONNECTING
        self.reconnect_count += 1
        
        logger.info(
            f"Scheduling reconnection for {self.name}",
            delay=self.current_reconnect_delay,
            attempt=self.reconnect_count
        )
        
        await asyncio.sleep(self.current_reconnect_delay)
        
        # Exponential backoff
        self.current_reconnect_delay = min(
            self.current_reconnect_delay * 2,
            self.max_reconnect_delay
        )
        
        # Attempt reconnection
        await self.connect()
    
    def get_statistics(self) -> Dict:
        """Get connection statistics."""
        return {
            "name": self.name,
            "state": self.state,
            "messages_received": self.messages_received,
            "reconnect_count": self.reconnect_count,
            "last_message_time": self.last_message_time,
            "last_heartbeat": self.last_heartbeat,
            "buffer_size": len(self.message_buffer)
        }


class WebSocketManager:
    """
    Manages multiple WebSocket connections for Binance.
    
    Maintains connection pools for execution, monitoring, and backup,
    with automatic failover and message buffering.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.settings = get_settings()
        self.connections: Dict[str, WebSocketConnection] = {}
        self.stream_callbacks: Dict[str, List[Callable]] = {}
        self.running = False
        
        # URLs
        if self.settings.exchange.binance_testnet:
            self.base_url = "wss://testnet.binance.vision"
        else:
            self.base_url = "wss://stream.binance.com:9443"
        
        logger.info(
            "WebSocketManager initialized",
            base_url=self.base_url
        )
    
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
            execution_subs.extend([
                StreamSubscription(f"{symbol}@trade", self._handle_trade, symbol),
                StreamSubscription(f"{symbol}@depth20@100ms", self._handle_depth, symbol)
            ])
            
            # Monitoring connection: klines and ticker
            monitoring_subs.extend([
                StreamSubscription(f"{symbol}@kline_1m", self._handle_kline, symbol),
                StreamSubscription(f"{symbol}@ticker", self._handle_ticker, symbol)
            ])
            
            # Backup connection: all streams (for failover)
            backup_subs.extend([
                StreamSubscription(f"{symbol}@trade", self._handle_trade, symbol),
                StreamSubscription(f"{symbol}@depth20@100ms", self._handle_depth, symbol),
                StreamSubscription(f"{symbol}@kline_1m", self._handle_kline, symbol),
                StreamSubscription(f"{symbol}@ticker", self._handle_ticker, symbol)
            ])
        
        # Create connections
        if execution_subs:
            self.connections["execution"] = WebSocketConnection(
                "execution",
                self.base_url,
                execution_subs
            )
        
        if monitoring_subs:
            self.connections["monitoring"] = WebSocketConnection(
                "monitoring",
                self.base_url,
                monitoring_subs
            )
        
        if backup_subs:
            self.connections["backup"] = WebSocketConnection(
                "backup",
                self.base_url,
                backup_subs
            )
    
    def subscribe(self, stream: str, callback: Callable[[Dict], None]) -> None:
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
    
    def unsubscribe(self, stream: str, callback: Callable[[Dict], None]) -> None:
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
    
    async def _handle_trade(self, data: Dict) -> None:
        """Handle trade stream data."""
        stream = data.get("stream", "")
        callbacks = self.stream_callbacks.get("trade", [])
        
        for callback in callbacks:
            try:
                await asyncio.create_task(asyncio.coroutine(callback)(data))
            except Exception as e:
                logger.error(f"Trade callback error", error=str(e))
    
    async def _handle_depth(self, data: Dict) -> None:
        """Handle depth stream data."""
        stream = data.get("stream", "")
        callbacks = self.stream_callbacks.get("depth", [])
        
        for callback in callbacks:
            try:
                await asyncio.create_task(asyncio.coroutine(callback)(data))
            except Exception as e:
                logger.error(f"Depth callback error", error=str(e))
    
    async def _handle_kline(self, data: Dict) -> None:
        """Handle kline stream data."""
        stream = data.get("stream", "")
        callbacks = self.stream_callbacks.get("kline", [])
        
        for callback in callbacks:
            try:
                await asyncio.create_task(asyncio.coroutine(callback)(data))
            except Exception as e:
                logger.error(f"Kline callback error", error=str(e))
    
    async def _handle_ticker(self, data: Dict) -> None:
        """Handle ticker stream data."""
        stream = data.get("stream", "")
        callbacks = self.stream_callbacks.get("ticker", [])
        
        for callback in callbacks:
            try:
                await asyncio.create_task(asyncio.coroutine(callback)(data))
            except Exception as e:
                logger.error(f"Ticker callback error", error=str(e))
    
    def get_connection_states(self) -> Dict[str, str]:
        """Get current connection states."""
        return {
            name: conn.state
            for name, conn in self.connections.items()
        }
    
    def get_statistics(self) -> Dict:
        """Get manager statistics."""
        return {
            "running": self.running,
            "connections": {
                name: conn.get_statistics()
                for name, conn in self.connections.items()
            },
            "subscriptions": list(self.stream_callbacks.keys())
        }
    
    async def check_health(self) -> Dict[str, bool]:
        """Check health of all connections."""
        health = {}
        
        for name, conn in self.connections.items():
            is_healthy = (
                conn.state == ConnectionState.CONNECTED and
                conn.last_message_time and
                (time.time() - conn.last_message_time) < 60  # No messages for 60s = unhealthy
            )
            health[name] = is_healthy
        
        return health