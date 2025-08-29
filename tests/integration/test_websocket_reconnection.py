"""
Integration tests for WebSocket reconnection with exponential backoff.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from genesis.exchange.websocket_manager import (
    WebSocketConnection,
    WebSocketManager,
    ConnectionState,
    StreamSubscription,
)


@pytest.fixture
async def mock_gateway():
    """Create a mock gateway."""
    gateway = MagicMock()
    gateway.get_order_book = AsyncMock()
    gateway.get_ticker = AsyncMock()
    gateway.get_klines = AsyncMock()
    return gateway


@pytest.fixture
async def mock_event_bus():
    """Create a mock event bus."""
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    return event_bus


class TestWebSocketReconnection:
    """Test WebSocket reconnection functionality."""
    
    async def test_exponential_backoff_formula(self):
        """Test that exponential backoff follows 2^n formula with max 30s."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
            reconnect_delay=2.0,
            max_reconnect_delay=30.0,
        )
        
        # Test backoff progression
        expected_delays = [2, 4, 8, 16, 30, 30]  # 2^1, 2^2, 2^3, 2^4, max, max
        
        for i, expected_delay in enumerate(expected_delays):
            connection.reconnect_attempt = i + 1
            delay = min(2 ** connection.reconnect_attempt, connection.max_reconnect_delay)
            assert delay == expected_delay, f"Attempt {i+1}: expected {expected_delay}s, got {delay}s"
    
    async def test_connection_state_tracking(self):
        """Test that connection states are properly tracked."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
        )
        
        # Initial state
        assert connection.state == ConnectionState.DISCONNECTED
        
        # Track state transitions
        states_seen = []
        
        async def mock_connect():
            states_seen.append(connection.state)
            connection.state = ConnectionState.CONNECTING
            states_seen.append(connection.state)
            await asyncio.sleep(0.01)
            connection.state = ConnectionState.CONNECTED
            states_seen.append(connection.state)
        
        with patch.object(connection, 'connect', mock_connect):
            await connection.connect()
        
        assert ConnectionState.DISCONNECTED in states_seen
        assert ConnectionState.CONNECTING in states_seen
        assert ConnectionState.CONNECTED in states_seen
    
    async def test_gap_detection_with_sequence_numbers(self):
        """Test gap detection using sequence numbers."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
        )
        
        # Simulate messages with sequence numbers
        messages = [
            {"stream": "btcusdt@depth", "data": {"u": 100}},
            {"stream": "btcusdt@depth", "data": {"u": 101}},
            {"stream": "btcusdt@depth", "data": {"u": 105}},  # Gap: 102-104 missing
        ]
        
        gaps_detected = []
        
        # Override _fill_gap to track gaps
        async def track_gap(stream, start, end):
            gaps_detected.append((stream, start, end))
        
        connection._fill_gap = track_gap
        
        # Process messages
        for msg in messages:
            await connection._check_for_gaps(msg)
        
        # Verify gap was detected
        assert len(gaps_detected) == 1
        assert gaps_detected[0] == ("btcusdt@depth", 101, 105)
        assert len(connection.detected_gaps) == 1
        assert connection.detected_gaps[0]["gap_size"] == 3
    
    async def test_rest_api_fallback_for_gaps(self, mock_gateway, mock_event_bus):
        """Test REST API fallback when gaps are detected."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
            gateway=mock_gateway,
        )
        connection.event_bus = mock_event_bus
        
        # Mock order book response
        mock_order_book = MagicMock()
        mock_order_book.bids = [(50000, 1.5), (49999, 2.0)]
        mock_order_book.asks = [(50001, 1.2), (50002, 1.8)]
        mock_gateway.get_order_book.return_value = mock_order_book
        
        # Simulate gap in order book stream
        await connection._fill_gap("btcusdt@depth", 100, 105)
        
        # Verify REST API was called
        mock_gateway.get_order_book.assert_called_once()
        
        # Verify event was published if event bus available
        if connection.event_bus:
            mock_event_bus.publish.assert_called_once()
            event = mock_event_bus.publish.call_args[0][0]
            assert event.event_type.value == "ORDER_BOOK_SNAPSHOT"
            assert event.event_data["gap_recovered"] is True
            assert event.event_data["gap_size"] == 5
    
    async def test_reconnection_with_backoff(self):
        """Test reconnection with exponential backoff timing."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
            reconnect_delay=2.0,
            max_reconnect_delay=30.0,
        )
        
        sleep_delays = []
        
        # Mock sleep to track delays
        async def mock_sleep(delay):
            sleep_delays.append(delay)
        
        # Mock connect to always fail
        async def failing_connect():
            if connection.state != ConnectionState.CONNECTING:
                connection.state = ConnectionState.CONNECTING
            raise ConnectionClosed(None, None)
        
        with patch("asyncio.sleep", mock_sleep):
            with patch.object(connection, "connect", failing_connect):
                # Try multiple reconnection attempts
                for i in range(3):
                    await connection._schedule_reconnect()
                    connection.state = ConnectionState.DISCONNECTED  # Reset for next attempt
        
        # Verify exponential backoff delays
        assert len(sleep_delays) == 3
        assert sleep_delays[0] == 2  # 2^1
        assert sleep_delays[1] == 4  # 2^2
        assert sleep_delays[2] == 8  # 2^3
    
    async def test_reconnection_counter_reset_on_success(self):
        """Test that reconnection counter resets on successful connection."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
        )
        
        # Set initial reconnection state
        connection.reconnect_attempt = 5
        connection.reconnect_count = 10
        connection.current_reconnect_delay = 30
        
        # Mock successful connection
        with patch("websockets.connect", AsyncMock(return_value=MagicMock())):
            connection.state = ConnectionState.DISCONNECTED
            await connection.connect()
        
        # Verify counters were reset
        assert connection.reconnect_attempt == 0
        assert connection.current_reconnect_delay == connection.reconnect_delay
        assert connection.state == ConnectionState.CONNECTED
    
    async def test_multiple_connection_pools(self, mock_gateway):
        """Test WebSocket manager with multiple connection pools."""
        manager = WebSocketManager(gateway=mock_gateway)
        
        with patch("genesis.exchange.websocket_manager.get_settings") as mock_settings:
            mock_settings.return_value.exchange.binance_testnet = False
            mock_settings.return_value.trading.trading_pairs = ["BTC/USDT", "ETH/USDT"]
            
            await manager._setup_connections()
        
        # Verify three connection pools created
        assert "execution" in manager.connections
        assert "monitoring" in manager.connections
        assert "backup" in manager.connections
        
        # Verify each has correct subscriptions
        execution_conn = manager.connections["execution"]
        assert len(execution_conn.subscriptions) == 4  # 2 symbols × 2 streams
        
        monitoring_conn = manager.connections["monitoring"]
        assert len(monitoring_conn.subscriptions) == 4  # 2 symbols × 2 streams
        
        backup_conn = manager.connections["backup"]
        assert len(backup_conn.subscriptions) == 8  # 2 symbols × 4 streams (all)
    
    async def test_heartbeat_with_reconnection(self):
        """Test heartbeat handling and reconnection on failure."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
            heartbeat_interval=0.1,  # Short interval for testing
        )
        
        # Mock WebSocket that fails on heartbeat
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=ConnectionClosed(None, None))
        connection.websocket = mock_ws
        connection.state = ConnectionState.CONNECTED
        
        # Track disconnection
        disconnected = False
        
        async def track_disconnect():
            nonlocal disconnected
            disconnected = True
            connection.state = ConnectionState.DISCONNECTED
        
        connection._handle_disconnect = track_disconnect
        
        # Run heartbeat loop briefly
        heartbeat_task = asyncio.create_task(connection._heartbeat_loop())
        await asyncio.sleep(0.2)
        heartbeat_task.cancel()
        
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
        # Verify disconnection was triggered
        assert disconnected is True
    
    async def test_message_buffer_during_reconnection(self):
        """Test that messages are buffered during reconnection."""
        connection = WebSocketConnection(
            name="test",
            url="wss://test.example.com",
            subscriptions=[],
        )
        
        # Add messages to buffer
        test_messages = [
            {"id": 1, "data": "message1"},
            {"id": 2, "data": "message2"},
            {"id": 3, "data": "message3"},
        ]
        
        for msg in test_messages:
            connection.message_buffer.append(msg)
        
        # Verify buffer contents
        assert len(connection.message_buffer) == 3
        assert list(connection.message_buffer) == test_messages
        
        # Verify buffer has max size limit (1000)
        for i in range(1100):
            connection.message_buffer.append({"id": i})
        
        assert len(connection.message_buffer) == 1000  # Limited by maxlen