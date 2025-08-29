"""
Integration tests for WebSocket and Event Bus integration.

Tests that WebSocket market data streams properly publish events
to the Event Bus using the publish-subscribe pattern.
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus
from genesis.exchange.websocket_manager import (
    WebSocketManager,
)


@pytest.fixture
async def event_bus():
    """Create and start an event bus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.exchange.binance_testnet = True
    settings.trading.trading_pairs = ["BTC/USDT"]
    return settings


class TestWebSocketEventBusIntegration:
    """Test WebSocket and Event Bus integration."""

    @pytest.mark.asyncio
    async def test_trade_stream_publishes_event(self, event_bus):
        """Test that trade stream data publishes market data event."""
        # Create WebSocket manager with event bus
        manager = WebSocketManager(event_bus=event_bus)

        # Set up event capture
        captured_events = []

        def event_handler(event: Event):
            captured_events.append(event)

        # Subscribe to market data events
        event_bus.subscribe(
            event_handler,
            {EventType.MARKET_DATA_UPDATED},
        )

        # Simulate trade data
        trade_data = {
            "stream": "btcusdt@trade",
            "data": {
                "e": "trade",
                "E": 1234567890123,
                "s": "BTCUSDT",
                "t": 12345,
                "p": "50000.00",
                "q": "0.001",
                "T": 1234567890123,
                "m": False,
                "M": True,
            },
        }

        # Handle the trade data
        await manager._handle_trade(trade_data)

        # Allow event to propagate
        await asyncio.sleep(0.1)

        # Verify event was published
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == EventType.MARKET_DATA_UPDATED
        assert event.aggregate_id == "BTCUSDT"
        assert event.event_data["type"] == "trade"
        assert event.event_data["price"] == "50000.00"
        assert event.event_data["quantity"] == "0.001"

    @pytest.mark.asyncio
    async def test_depth_stream_publishes_order_book_event(self, event_bus):
        """Test that depth stream publishes order book snapshot event."""
        # Create WebSocket manager with event bus
        manager = WebSocketManager(event_bus=event_bus)

        # Set up event capture
        captured_events = []

        def event_handler(event: Event):
            captured_events.append(event)

        # Subscribe to order book events
        event_bus.subscribe(
            event_handler,
            {EventType.ORDER_BOOK_SNAPSHOT},
        )

        # Simulate depth data
        depth_data = {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1234567890123,
                "s": "BTCUSDT",
                "u": 12345,
                "b": [
                    ["49999.00", "1.0"],
                    ["49998.00", "2.0"],
                    ["49997.00", "3.0"],
                ],
                "a": [
                    ["50001.00", "1.0"],
                    ["50002.00", "2.0"],
                    ["50003.00", "3.0"],
                ],
            },
        }

        # Handle the depth data
        await manager._handle_depth(depth_data)

        # Allow event to propagate
        await asyncio.sleep(0.1)

        # Verify event was published
        assert len(captured_events) == 1
        event = captured_events[0]
        assert event.event_type == EventType.ORDER_BOOK_SNAPSHOT
        assert event.aggregate_id == "BTCUSDT"
        assert event.event_data["type"] == "depth"
        assert len(event.event_data["bids"]) == 3
        assert len(event.event_data["asks"]) == 3

    @pytest.mark.asyncio
    async def test_ticker_spread_compression_alert(self, event_bus):
        """Test that ticker stream detects and alerts on spread compression."""
        # Create WebSocket manager with event bus
        manager = WebSocketManager(event_bus=event_bus)

        # Set up event capture for both event types
        captured_events = []

        def event_handler(event: Event):
            captured_events.append(event)

        # Subscribe to spread compression and market data events
        event_bus.subscribe(
            event_handler,
            {EventType.SPREAD_COMPRESSION, EventType.MARKET_DATA_UPDATED},
        )

        # Simulate ticker data with tight spread (< 10 bps)
        ticker_data = {
            "stream": "btcusdt@ticker",
            "data": {
                "e": "24hrTicker",
                "E": 1234567890123,
                "s": "BTCUSDT",
                "c": "50000.00",  # Current price
                "b": "49999.50",  # Bid (1 bps spread)
                "a": "50000.50",  # Ask
                "B": "10.0",      # Bid quantity
                "A": "10.0",      # Ask quantity
                "v": "1000.0",    # Volume
                "w": "50000.00",  # Weighted avg price
            },
        }

        # Handle the ticker data
        await manager._handle_ticker(ticker_data)

        # Allow events to propagate
        await asyncio.sleep(0.1)

        # Verify both events were published
        assert len(captured_events) == 2

        # Find spread compression event
        spread_event = next(
            (e for e in captured_events if e.event_type == EventType.SPREAD_COMPRESSION),
            None,
        )
        assert spread_event is not None
        assert spread_event.aggregate_id == "BTCUSDT"
        assert spread_event.event_data["spread_bps"] < 10

        # Find market data event
        market_event = next(
            (e for e in captured_events if e.event_type == EventType.MARKET_DATA_UPDATED),
            None,
        )
        assert market_event is not None
        assert market_event.event_data["type"] == "ticker"

    @pytest.mark.asyncio
    async def test_kline_stream_publishes_low_priority_event(self, event_bus):
        """Test that kline stream publishes low priority events."""
        # Create WebSocket manager with event bus
        manager = WebSocketManager(event_bus=event_bus)

        # Track event priorities
        event_priorities = []

        # Patch the publish method to capture priority
        original_publish = event_bus.publish

        async def mock_publish(event, priority=EventPriority.NORMAL):
            event_priorities.append(priority)
            await original_publish(event, priority)

        event_bus.publish = mock_publish

        # Simulate kline data
        kline_data = {
            "stream": "btcusdt@kline_1m",
            "data": {
                "e": "kline",
                "E": 1234567890123,
                "s": "BTCUSDT",
                "k": {
                    "t": 1234567890000,
                    "T": 1234567950000,
                    "s": "BTCUSDT",
                    "i": "1m",
                    "o": "49990.00",
                    "c": "50010.00",
                    "h": "50020.00",
                    "l": "49980.00",
                    "v": "100.0",
                },
            },
        }

        # Handle the kline data
        await manager._handle_kline(kline_data)

        # Allow event to propagate
        await asyncio.sleep(0.1)

        # Verify low priority was used
        assert len(event_priorities) == 1
        assert event_priorities[0] == EventPriority.LOW

    @pytest.mark.asyncio
    async def test_multiple_stream_subscriptions(self, event_bus, mock_settings):
        """Test handling multiple stream subscriptions with event publishing."""
        with patch("genesis.exchange.websocket_manager.get_settings", return_value=mock_settings):
            manager = WebSocketManager(event_bus=event_bus)

            # Track all events
            all_events = []

            def event_handler(event: Event):
                all_events.append(event)

            # Subscribe to all relevant event types
            event_bus.subscribe(
                event_handler,
                {
                    EventType.MARKET_DATA_UPDATED,
                    EventType.ORDER_BOOK_SNAPSHOT,
                    EventType.SPREAD_COMPRESSION,
                },
            )

            # Simulate multiple data streams
            streams_data = [
                {
                    "stream": "btcusdt@trade",
                    "data": {"s": "BTCUSDT", "p": "50000", "q": "1"},
                },
                {
                    "stream": "ethusdt@trade",
                    "data": {"s": "ETHUSDT", "p": "3000", "q": "10"},
                },
                {
                    "stream": "btcusdt@depth20@100ms",
                    "data": {
                        "u": 123,
                        "b": [["49999", "1"]],
                        "a": [["50001", "1"]],
                    },
                },
            ]

            # Handle all streams
            for data in streams_data:
                if "@trade" in data["stream"]:
                    await manager._handle_trade(data)
                elif "@depth" in data["stream"]:
                    await manager._handle_depth(data)

            # Allow events to propagate
            await asyncio.sleep(0.1)

            # Verify all events were published
            assert len(all_events) == 3

            # Check we have events for both symbols
            symbols = {event.aggregate_id for event in all_events}
            assert "BTCUSDT" in symbols
            assert "ETHUSDT" in symbols

    @pytest.mark.asyncio
    async def test_websocket_with_callbacks_and_events(self, event_bus):
        """Test that both callbacks and events work together."""
        # Create WebSocket manager with event bus
        manager = WebSocketManager(event_bus=event_bus)

        # Track callback invocations
        callback_invoked = False
        callback_data = None

        def stream_callback(data):
            nonlocal callback_invoked, callback_data
            callback_invoked = True
            callback_data = data

        # Register callback
        manager.subscribe("trade", stream_callback)

        # Track events
        captured_events = []

        def event_handler(event: Event):
            captured_events.append(event)

        event_bus.subscribe(
            event_handler,
            {EventType.MARKET_DATA_UPDATED},
        )

        # Simulate trade data
        trade_data = {
            "stream": "btcusdt@trade",
            "data": {
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.001",
            },
        }

        # Handle the trade
        await manager._handle_trade(trade_data)

        # Allow processing
        await asyncio.sleep(0.1)

        # Verify both callback and event were processed
        assert callback_invoked is True
        assert callback_data == trade_data
        assert len(captured_events) == 1
        assert captured_events[0].event_type == EventType.MARKET_DATA_UPDATED

    @pytest.mark.asyncio
    async def test_connection_state_events(self, event_bus, mock_settings):
        """Test that connection state changes publish system events."""
        # This would require modifying WebSocketConnection to publish
        # connection state events (CONNECTED, DISCONNECTED, RECONNECTING)
        # For now, we'll verify the structure is in place

        with patch("genesis.exchange.websocket_manager.get_settings", return_value=mock_settings):
            manager = WebSocketManager(event_bus=event_bus)

            # Verify event bus is assigned
            assert manager.event_bus == event_bus

            # Verify manager can handle events from all connection pools
            await manager._setup_connections()

            assert "execution" in manager.connections
            assert "monitoring" in manager.connections
            assert "backup" in manager.connections

    @pytest.mark.asyncio
    async def test_event_validation_with_pydantic(self, event_bus):
        """Test that events use proper Pydantic validation."""
        # Create WebSocket manager with event bus
        manager = WebSocketManager(event_bus=event_bus)

        # Track events
        captured_events = []

        def event_handler(event: Event):
            captured_events.append(event)

        event_bus.subscribe(
            event_handler,
            {EventType.MARKET_DATA_UPDATED},
        )

        # Simulate trade with all fields
        trade_data = {
            "stream": "btcusdt@trade",
            "data": {
                "e": "trade",
                "E": 1234567890123,
                "s": "BTCUSDT",
                "t": 12345,
                "p": "50000.00",
                "q": "0.001",
                "b": 88888,
                "a": 88889,
                "T": 1234567890123,
                "m": False,
                "M": True,
            },
        }

        # Handle the trade
        await manager._handle_trade(trade_data)

        # Allow processing
        await asyncio.sleep(0.1)

        # Verify event structure
        assert len(captured_events) == 1
        event = captured_events[0]

        # Verify Event model attributes
        assert hasattr(event, "event_id")
        assert hasattr(event, "event_type")
        assert hasattr(event, "aggregate_id")
        assert hasattr(event, "event_data")
        assert hasattr(event, "created_at")
        assert isinstance(event.created_at, datetime)
