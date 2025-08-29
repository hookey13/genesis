"""
Unit tests for WebSocket Manager.

Tests WebSocket connection management, reconnection logic, gap detection,
and circuit breaker integration.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.exchange.circuit_breaker import CircuitBreaker
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.websocket_manager import (
    ConnectionState,
    StreamSubscription,
    WebSocketConnection,
    WebSocketManager,
)


@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker."""
    breaker = MagicMock(spec=CircuitBreaker)
    breaker.is_open.return_value = False
    breaker.is_closed.return_value = True
    return breaker


@pytest.fixture
def mock_gateway():
    """Create mock gateway."""
    gateway = AsyncMock(spec=BinanceGateway)
    gateway.get_recent_trades = AsyncMock(return_value=[])
    gateway.get_order_book = AsyncMock(return_value={})
    return gateway


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws


class TestStreamSubscription:
    """Test StreamSubscription dataclass."""

    def test_subscription_creation(self):
        """Test creating a stream subscription."""
        callback = MagicMock()
        sub = StreamSubscription(
            stream="btcusdt@trade", callback=callback, symbol="BTCUSDT"
        )

        assert sub.stream == "btcusdt@trade"
        assert sub.callback == callback
        assert sub.symbol == "BTCUSDT"


class TestWebSocketConnection:
    """Test WebSocketConnection functionality."""

    def test_connection_initialization(self, mock_circuit_breaker, mock_gateway):
        """Test WebSocket connection initialization."""
        subs = [StreamSubscription("btcusdt@trade", MagicMock())]

        conn = WebSocketConnection(
            name="test",
            url="wss://test.com",
            subscriptions=subs,
            circuit_breaker=mock_circuit_breaker,
            gateway=mock_gateway,
        )

        assert conn.name == "test"
        assert conn.url == "wss://test.com"
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.max_reconnect_delay == 30.0  # Updated to 30s per requirements
        assert conn.heartbeat_interval == 30.0
        assert len(conn.subscriptions) == 1

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_websocket, mock_circuit_breaker):
        """Test successful connection."""
        subs = [StreamSubscription("btcusdt@trade", MagicMock())]

        conn = WebSocketConnection(
            name="test",
            url="wss://test.com",
            subscriptions=subs,
            circuit_breaker=mock_circuit_breaker,
        )

        # Patch websockets.connect to return the mock directly
        async def mock_connect(*args, **kwargs):
            return mock_websocket

        with patch("websockets.connect", new=mock_connect):
            await conn.connect()

            assert conn.state == ConnectionState.CONNECTED
            assert conn.websocket is not None
            assert conn.heartbeat_task is not None
            assert conn.message_handler_task is not None

    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_circuit_breaker):
        """Test connection failure."""
        subs = [StreamSubscription("btcusdt@trade", MagicMock())]

        conn = WebSocketConnection(
            name="test",
            url="wss://test.com",
            subscriptions=subs,
            circuit_breaker=mock_circuit_breaker,
        )

        with patch("websockets.connect", side_effect=Exception("Connection failed")):
            with patch.object(
                conn, "_schedule_reconnect", new_callable=AsyncMock
            ) as mock_reconnect:
                await conn.connect()

                assert conn.state == ConnectionState.DISCONNECTED
                assert conn.websocket is None
                mock_reconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_websocket):
        """Test disconnection."""
        subs = [StreamSubscription("btcusdt@trade", MagicMock())]

        conn = WebSocketConnection(
            name="test", url="wss://test.com", subscriptions=subs
        )

        # Set up connected state
        conn.state = ConnectionState.CONNECTED
        conn.websocket = mock_websocket
        conn.heartbeat_task = asyncio.create_task(asyncio.sleep(0))
        conn.message_handler_task = asyncio.create_task(asyncio.sleep(0))

        await conn.disconnect()

        assert conn.state == ConnectionState.CLOSED
        assert conn.websocket is None
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_heartbeat_loop(self, mock_websocket):
        """Test heartbeat sending."""
        subs = []
        conn = WebSocketConnection(
            name="test",
            url="wss://test.com",
            subscriptions=subs,
            heartbeat_interval=0.1,
        )

        conn.state = ConnectionState.CONNECTED
        conn.websocket = mock_websocket

        # Run heartbeat for a short time
        heartbeat_task = asyncio.create_task(conn._heartbeat_loop())
        await asyncio.sleep(0.15)
        conn.state = ConnectionState.DISCONNECTED
        await heartbeat_task

        # Check heartbeat was sent
        mock_websocket.send.assert_called()
        call_args = mock_websocket.send.call_args[0][0]
        data = json.loads(call_args)
        assert "pong" in data

    @pytest.mark.asyncio
    async def test_message_handler(self, mock_websocket):
        """Test message handling."""
        callback = AsyncMock()
        subs = [StreamSubscription("btcusdt@trade", callback)]

        conn = WebSocketConnection(
            name="test", url="wss://test.com", subscriptions=subs
        )

        conn.state = ConnectionState.CONNECTED
        conn.websocket = mock_websocket

        # Mock incoming message
        message = json.dumps({"stream": "btcusdt@trade", "data": {"price": "50000"}})
        mock_websocket.recv.return_value = message

        # Run message handler briefly
        handler_task = asyncio.create_task(conn._message_handler())
        await asyncio.sleep(0.1)
        conn.state = ConnectionState.DISCONNECTED

        # Cancel task
        handler_task.cancel()
        try:
            await handler_task
        except asyncio.CancelledError:
            pass

        # Check callback was called
        callback.assert_called()
        assert conn.messages_received == 1

    @pytest.mark.asyncio
    async def test_gap_detection(self, mock_gateway):
        """Test gap detection in messages."""
        subs = []
        conn = WebSocketConnection(
            name="test", url="wss://test.com", subscriptions=subs, gateway=mock_gateway
        )

        # First message with sequence
        data1 = {"stream": "btcusdt@depth", "data": {"u": 100}}
        await conn._check_for_gaps(data1)
        assert conn.last_sequence_numbers["btcusdt@depth"] == 100

        # Second message with gap
        data2 = {"stream": "btcusdt@depth", "data": {"u": 105}}
        await conn._check_for_gaps(data2)

        # Check gap was detected
        assert len(conn.detected_gaps) == 1
        gap = conn.detected_gaps[0]
        assert gap["stream"] == "btcusdt@depth"
        assert gap["gap_size"] == 4

        # Verify REST API was called to fill gap
        mock_gateway.get_order_book.assert_called()

    @pytest.mark.asyncio
    async def test_reconnection_with_backoff(self):
        """Test reconnection with exponential backoff."""
        subs = []
        conn = WebSocketConnection(
            name="test",
            url="wss://test.com",
            subscriptions=subs,
            reconnect_delay=1.0,
            max_reconnect_delay=30.0,  # Updated to 30s per requirements
        )

        # Mock connect to always fail
        with patch.object(conn, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = [None, None]  # Fail twice

            # First reconnection
            await conn._schedule_reconnect()
            assert conn.current_reconnect_delay == 2.0  # Doubled
            assert conn.reconnect_count == 1

            # Second reconnection
            await conn._schedule_reconnect()
            assert conn.current_reconnect_delay == 4.0  # Doubled again
            assert conn.reconnect_count == 2

    def test_get_statistics(self):
        """Test getting connection statistics."""
        subs = []
        conn = WebSocketConnection(
            name="test", url="wss://test.com", subscriptions=subs
        )

        conn.messages_received = 100
        conn.reconnect_count = 2
        conn.last_message_time = 1234567890

        stats = conn.get_statistics()

        assert stats["name"] == "test"
        assert stats["state"] == ConnectionState.DISCONNECTED
        assert stats["messages_received"] == 100
        assert stats["reconnect_count"] == 2
        assert stats["last_message_time"] == 1234567890
        assert stats["detected_gaps"] == 0


class TestWebSocketManager:
    """Test WebSocketManager functionality."""

    def test_manager_initialization(self):
        """Test WebSocket manager initialization."""
        manager = WebSocketManager()

        assert manager.running is False
        assert len(manager.connections) == 0
        assert len(manager.stream_callbacks) == 0

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the manager."""
        manager = WebSocketManager()

        with patch.object(
            manager, "_setup_connections", new_callable=AsyncMock
        ) as mock_setup:
            # Add mock connections
            mock_conn = AsyncMock()
            mock_conn.connect = AsyncMock()
            mock_conn.disconnect = AsyncMock()
            manager.connections["test"] = mock_conn

            await manager.start()
            assert manager.running is True
            mock_setup.assert_called_once()
            mock_conn.connect.assert_called()

            await manager.stop()
            assert manager.running is False
            mock_conn.disconnect.assert_called()

    def test_subscribe_unsubscribe(self):
        """Test subscribing and unsubscribing to streams."""
        manager = WebSocketManager()
        callback = MagicMock()

        # Subscribe
        manager.subscribe("trade", callback)
        assert "trade" in manager.stream_callbacks
        assert callback in manager.stream_callbacks["trade"]

        # Unsubscribe
        manager.unsubscribe("trade", callback)
        assert "trade" not in manager.stream_callbacks

    @pytest.mark.asyncio
    async def test_handle_trade(self):
        """Test handling trade data."""
        manager = WebSocketManager()
        callback = AsyncMock()

        manager.subscribe("trade", callback)

        data = {"stream": "btcusdt@trade", "data": {"price": "50000"}}
        await manager._handle_trade(data)

        callback.assert_called_with(data)

    @pytest.mark.asyncio
    async def test_handle_depth(self):
        """Test handling depth data."""
        manager = WebSocketManager()
        callback = AsyncMock()

        manager.subscribe("depth", callback)

        data = {"stream": "btcusdt@depth", "data": {"bids": [], "asks": []}}
        await manager._handle_depth(data)

        callback.assert_called_with(data)

    @pytest.mark.asyncio
    async def test_handle_kline(self):
        """Test handling kline data."""
        manager = WebSocketManager()
        callback = AsyncMock()

        manager.subscribe("kline", callback)

        data = {"stream": "btcusdt@kline_1m", "data": {"k": {}}}
        await manager._handle_kline(data)

        callback.assert_called_with(data)

    @pytest.mark.asyncio
    async def test_handle_ticker(self):
        """Test handling ticker data."""
        manager = WebSocketManager()
        callback = AsyncMock()

        manager.subscribe("ticker", callback)

        data = {"stream": "btcusdt@ticker", "data": {"c": "50000"}}
        await manager._handle_ticker(data)

        callback.assert_called_with(data)

    def test_get_connection_states(self):
        """Test getting connection states."""
        manager = WebSocketManager()

        # Add mock connections
        mock_conn1 = MagicMock()
        mock_conn1.state = ConnectionState.CONNECTED
        mock_conn2 = MagicMock()
        mock_conn2.state = ConnectionState.DISCONNECTED

        manager.connections["conn1"] = mock_conn1
        manager.connections["conn2"] = mock_conn2

        states = manager.get_connection_states()

        assert states["conn1"] == ConnectionState.CONNECTED
        assert states["conn2"] == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Test health checking."""
        manager = WebSocketManager()

        # Add mock connections
        mock_conn1 = MagicMock()
        mock_conn1.state = ConnectionState.CONNECTED
        mock_conn1.last_message_time = time.time()  # Recent message

        mock_conn2 = MagicMock()
        mock_conn2.state = ConnectionState.CONNECTED
        mock_conn2.last_message_time = time.time() - 120  # Old message

        manager.connections["healthy"] = mock_conn1
        manager.connections["unhealthy"] = mock_conn2

        health = await manager.check_health()

        assert health["healthy"] is True
        assert health["unhealthy"] is False

    def test_get_statistics(self):
        """Test getting manager statistics."""
        manager = WebSocketManager()
        manager.running = True

        # Add mock connection
        mock_conn = MagicMock()
        mock_conn.get_statistics.return_value = {"messages_received": 100}
        manager.connections["test"] = mock_conn

        # Add subscription
        manager.stream_callbacks["trade"] = [MagicMock()]

        stats = manager.get_statistics()

        assert stats["running"] is True
        assert "connections" in stats
        assert stats["connections"]["test"]["messages_received"] == 100
        assert "trade" in stats["subscriptions"]

    @pytest.mark.asyncio
    async def test_setup_connections(self):
        """Test setting up connection pools."""
        manager = WebSocketManager()

        # Mock settings
        with patch("genesis.exchange.websocket_manager.get_settings") as mock_settings:
            mock_settings.return_value.trading.trading_pairs = ["BTC/USDT", "ETH/USDT"]
            mock_settings.return_value.exchange.binance_testnet = False

            await manager._setup_connections()

            # Check connections were created
            assert "execution" in manager.connections
            assert "monitoring" in manager.connections
            assert "backup" in manager.connections

            # Check subscriptions were set up correctly
            exec_conn = manager.connections["execution"]
            assert len(exec_conn.subscriptions) == 4  # 2 symbols * 2 stream types

            monitor_conn = manager.connections["monitoring"]
            assert len(monitor_conn.subscriptions) == 4  # 2 symbols * 2 stream types

            backup_conn = manager.connections["backup"]
            assert len(backup_conn.subscriptions) == 8  # 2 symbols * 4 stream types

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        gateway = mock_gateway()
        manager = WebSocketManager(gateway=gateway)

        # Check circuit breaker manager was created
        assert manager.circuit_breaker_manager is not None

        # Check breakers are created for connections
        with patch("genesis.exchange.websocket_manager.get_settings") as mock_settings:
            mock_settings.return_value.trading.trading_pairs = ["BTC/USDT"]
            mock_settings.return_value.exchange.binance_testnet = False

            await manager._setup_connections()

            # Verify circuit breakers were assigned
            exec_conn = manager.connections["execution"]
            assert exec_conn.circuit_breaker is not None
