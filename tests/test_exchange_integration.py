"""
Integration tests for exchange layer components.

Tests the complete exchange layer including ExchangeClient, WSManager,
enhanced circuit breaker, and event emission.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import websockets

from genesis.exchange.client import ExchangeClient
from genesis.exchange.ws_manager import WSManager
from genesis.exchange.circuit_breaker_v2 import EnhancedCircuitBreaker, CircuitState
from genesis.exchange.events import (
    EventBus,
    EventType,
    OrderAck,
    OrderFill,
    OrderCancel,
    MarketTick,
    WebSocketEvent,
    CircuitBreakerEvent,
    ReconciliationEvent,
    ClockSkewEvent,
)
from genesis.exchange.exceptions import (
    ExchangeError,
    OrderNotFoundError,
    RateLimitError,
    InsufficientBalanceError,
)


class TestExchangeClient:
    """Test ExchangeClient with event emission and idempotency."""

    @pytest.fixture
    async def client(self):
        """Create test client with mocked CCXT."""
        client = ExchangeClient(
            api_key="test_key", api_secret="test_secret", testnet=True
        )

        # Mock CCXT exchange
        client.exchange = AsyncMock()
        client.exchange.load_markets = AsyncMock()
        client.exchange.fetch_time = AsyncMock(return_value=int(time.time() * 1000))
        client.exchange.fetch_balance = AsyncMock(
            return_value={
                "BTC": {"free": 1.5, "used": 0.5, "total": 2.0},
                "USDT": {"free": 10000, "used": 5000, "total": 15000},
            }
        )

        yield client

        if hasattr(client.exchange, "close"):
            await client.exchange.close()

    @pytest.mark.asyncio
    async def test_initialize(self, client):
        """Test client initialization and startup checks."""
        client.exchange.fetch_open_orders = AsyncMock(return_value=[])

        await client.initialize()

        client.exchange.load_markets.assert_called_once()
        client.exchange.fetch_time.assert_called()
        assert client._last_reconciliation is not None

    @pytest.mark.asyncio
    async def test_place_order_with_idempotency(self, client):
        """Test order placement with idempotency check."""
        events = []
        client.event_bus.subscribe(EventType.ORDER_ACK, events.append)

        # Mock order response
        order_response = {
            "id": "12345",
            "status": "new",
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "price": 50000,
            "amount": 0.1,
            "filled": 0,
            "remaining": 0.1,
            "clientOrderId": "twap_test123",
        }
        client.exchange.create_limit_order = AsyncMock(return_value=order_response)

        # First call
        result1 = await client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            client_order_id="twap_test123",
        )

        # Second call with same client_order_id (should return cached)
        result2 = await client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            client_order_id="twap_test123",
        )

        # Should only call exchange once
        client.exchange.create_limit_order.assert_called_once()
        assert result1 == result2
        assert len(events) == 1  # Only one event emitted
        assert events[0].client_order_id == "twap_test123"

    @pytest.mark.asyncio
    async def test_cancel_order_with_verification(self, client):
        """Test order cancellation with post-cancel verification."""
        events = []
        client.event_bus.subscribe(EventType.ORDER_CANCEL, events.append)

        # Mock responses
        client.exchange.cancel_order = AsyncMock(
            return_value={
                "id": "12345",
                "status": "canceled",
                "amount": 0.1,
                "filled": 0.02,
            }
        )

        client.exchange.fetch_order = AsyncMock(
            return_value={
                "id": "12345",
                "status": "canceled",
                "clientOrderId": "twap_test123",
            }
        )

        result = await client.cancel_order(
            symbol="BTC/USDT", client_order_id="twap_test123", exchange_order_id="12345"
        )

        # Verify post-cancel check was performed
        client.exchange.fetch_order.assert_called()
        assert len(events) == 1
        assert events[0].exchange_order_id == "12345"
        assert events[0].status == "CANCELED"

    @pytest.mark.asyncio
    async def test_reconciliation(self, client):
        """Test position reconciliation with event emission."""
        events = []
        client.event_bus.subscribe(EventType.RECONCILIATION_START, events.append)
        client.event_bus.subscribe(EventType.RECONCILIATION_COMPLETE, events.append)

        # Mock open orders
        client.exchange.fetch_open_orders = AsyncMock(
            return_value=[
                {
                    "id": "12345",
                    "clientOrderId": "twap_test123",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "status": "open",
                }
            ]
        )

        # Add local order that doesn't exist on exchange
        client._pending_orders["twap_old_order"] = {"id": "99999"}

        report = await client.reconcile_positions()

        # Check reconciliation
        assert "twap_test123" in client._pending_orders
        assert "twap_old_order" not in client._pending_orders  # Removed stale
        assert report["corrections"] > 0

        # Check events
        assert len(events) == 2
        assert events[0].phase == "START"
        assert events[1].phase == "COMPLETE"
        assert events[1].corrections_made > 0

    @pytest.mark.asyncio
    async def test_clock_skew_detection(self, client):
        """Test clock skew detection and event emission."""
        events = []
        client.event_bus.subscribe(EventType.CLOCK_SKEW_DETECTED, events.append)

        # Mock significant clock skew (6 seconds)
        server_time = int((time.time() - 6) * 1000)
        client.exchange.fetch_time = AsyncMock(return_value=server_time)

        with pytest.raises(ExchangeError, match="Clock skew too high"):
            await client.check_time_sync()

        assert len(events) == 1
        assert events[0].skew_ms > 5000
        assert events[0].action_taken == "HALT_TRADING"


class TestWSManager:
    """Test WebSocket manager with reconnection and sequence tracking."""

    @pytest.fixture
    async def ws_manager(self):
        """Create test WebSocket manager."""
        exchange_client = MagicMock()
        exchange_client.exchange = AsyncMock()

        manager = WSManager(exchange_client=exchange_client, testnet=True)

        yield manager

        if manager.running:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_listen_key_lifecycle(self, ws_manager):
        """Test listen key creation and keepalive."""
        # Mock listen key operations
        ws_manager.exchange.exchange.privatePostUserDataStream = AsyncMock(
            return_value={"listenKey": "test_listen_key_123"}
        )
        ws_manager.exchange.exchange.privatePutUserDataStream = AsyncMock()
        ws_manager.exchange.exchange.privateDeleteUserDataStream = AsyncMock()

        # Get listen key
        listen_key = await ws_manager._get_listen_key()
        assert listen_key == "test_listen_key_123"
        assert ws_manager.listen_key == "test_listen_key_123"
        assert ws_manager.listen_key_expires > time.time()

        # Test keepalive near expiration
        ws_manager.listen_key_expires = time.time() + 800  # 13 minutes from now
        ws_manager.running = True

        # Run one keepalive iteration
        keepalive_task = asyncio.create_task(ws_manager._keepalive_listen_key())
        await asyncio.sleep(0.1)
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass

        # Verify keepalive was sent
        ws_manager.exchange.exchange.privatePutUserDataStream.assert_called()

    @pytest.mark.asyncio
    async def test_market_subscription(self, ws_manager):
        """Test market data subscription management."""
        # Subscribe to multiple streams
        ws_manager.subscribe_market("BTCUSDT", "ticker")
        ws_manager.subscribe_market("ETHUSDT", "depth")

        assert "btcusdt@ticker" in ws_manager.market_subscriptions
        assert "ethusdt@depth" in ws_manager.market_subscriptions

        # Unsubscribe
        ws_manager.unsubscribe_market("BTCUSDT", "ticker")
        assert "btcusdt@ticker" not in ws_manager.market_subscriptions
        assert "ethusdt@depth" in ws_manager.market_subscriptions

    @pytest.mark.asyncio
    async def test_sequence_gap_detection(self, ws_manager):
        """Test sequence gap detection in market messages."""
        # Set initial sequence
        ws_manager.last_sequence["BTCUSDT"] = 1000

        # Simulate message with gap
        message = {
            "stream": "btcusdt@ticker",
            "data": {
                "s": "BTCUSDT",
                "E": 1005,  # Gap from 1001-1004
                "b": "50000",
                "a": "50001",
                "B": "10",
                "A": "10",
                "c": "50000",
                "v": "1000",
            },
        }

        await ws_manager._handle_market_message(message)

        # Check gap was recorded
        gaps = ws_manager.get_sequence_gaps("BTCUSDT")
        assert len(gaps) == 1
        assert gaps[0] == (1001, 1004)
        assert ws_manager.last_sequence["BTCUSDT"] == 1005

    @pytest.mark.asyncio
    async def test_reconnection_backoff(self, ws_manager):
        """Test reconnection with exponential backoff."""
        ws_manager.running = True
        initial_delay = ws_manager.reconnect_delay

        # First reconnection
        await ws_manager._reconnect_with_backoff()
        assert ws_manager.reconnect_attempts == 1
        assert ws_manager.reconnect_delay == initial_delay * 2

        # Second reconnection
        await ws_manager._reconnect_with_backoff()
        assert ws_manager.reconnect_attempts == 2
        assert ws_manager.reconnect_delay == initial_delay * 4

        # Check max delay limit
        ws_manager.reconnect_delay = 100
        await ws_manager._reconnect_with_backoff()
        assert ws_manager.reconnect_delay <= ws_manager.max_reconnect_delay


class TestEnhancedCircuitBreaker:
    """Test enhanced circuit breaker with spec-compliant conditions."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create test circuit breaker."""
        return EnhancedCircuitBreaker(
            name="test",
            error_rate_threshold=0.5,
            error_window_seconds=60,
            ws_disconnect_threshold=30,
            clock_skew_threshold=5000,
            recovery_timeout=30,
        )

    def test_5xx_error_rate_trip(self, circuit_breaker):
        """Test circuit trips on 5xx error rate > 50%."""
        assert circuit_breaker.state == CircuitState.CLOSED

        # Add successful requests
        for _ in range(5):
            circuit_breaker.record_success()

        # Add 5xx errors to exceed threshold
        for i in range(6):
            error = Exception("Server error")
            circuit_breaker.record_error(error, status_code=500)

        # Should trip (6 errors / 11 total > 50%)
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.trip_count == 1

    def test_websocket_disconnect_trip(self, circuit_breaker):
        """Test circuit trips on WebSocket disconnect > 30 seconds."""
        assert circuit_breaker.state == CircuitState.CLOSED

        # Simulate WebSocket disconnect
        circuit_breaker.update_ws_status(connected=False)
        assert circuit_breaker.state == CircuitState.CLOSED  # Not tripped yet

        # Simulate time passing (31 seconds)
        circuit_breaker.ws_disconnected_at = time.time() - 31

        # Check trip condition
        should_trip, reason = circuit_breaker._check_trip_conditions()
        assert should_trip
        assert "WebSocket disconnected" in reason

        # Trigger trip check
        circuit_breaker.update_ws_status(connected=False)
        assert circuit_breaker.state == CircuitState.OPEN

    def test_clock_skew_trip(self, circuit_breaker):
        """Test circuit trips on clock skew > 5 seconds."""
        assert circuit_breaker.state == CircuitState.CLOSED

        # Update clock skew beyond threshold
        circuit_breaker.update_clock_skew(6000)  # 6 seconds

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.trip_count == 1

    def test_recovery_and_half_open(self, circuit_breaker):
        """Test recovery timeout and half-open state."""
        # Trip the circuit
        circuit_breaker._transition_to(CircuitState.OPEN, "Manual trip")
        assert circuit_breaker.is_open()

        # Before recovery timeout
        assert circuit_breaker.is_open()

        # Simulate recovery timeout passing
        circuit_breaker.state_changed_at = time.time() - 31

        # Should transition to half-open
        assert not circuit_breaker.is_open()
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Successful requests in half-open
        for _ in range(3):
            circuit_breaker.record_success()

        # Should close after threshold successes
        assert circuit_breaker.state == CircuitState.CLOSED

    def test_decorator_usage(self, circuit_breaker):
        """Test circuit breaker as decorator."""
        call_count = 0

        @circuit_breaker.call
        async def protected_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Error")
            return "success"

        async def run_test():
            # First calls fail but don't trip
            with pytest.raises(Exception):
                await protected_function()
            with pytest.raises(Exception):
                await protected_function()

            # Trip the circuit manually
            circuit_breaker._transition_to(CircuitState.OPEN, "Manual")

            # Now calls are blocked
            with pytest.raises(Exception, match="Circuit breaker .* is OPEN"):
                await protected_function()

            assert circuit_breaker.blocked_requests == 1

        asyncio.run(run_test())

    def test_event_emission(self, circuit_breaker):
        """Test circuit breaker event emission."""
        events = []
        circuit_breaker.event_bus.subscribe(
            EventType.CIRCUIT_BREAKER_OPEN, events.append
        )
        circuit_breaker.event_bus.subscribe(
            EventType.CIRCUIT_BREAKER_CLOSE, events.append
        )

        # Trip circuit
        circuit_breaker._transition_to(CircuitState.OPEN, "Test trip")
        assert len(events) == 1
        assert events[0].state == "OPEN"
        assert events[0].reason == "Test trip"

        # Close circuit
        circuit_breaker._transition_to(CircuitState.CLOSED, "Test recovery")
        assert len(events) == 2
        assert events[1].state == "CLOSED"


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_order_lifecycle_with_events(self):
        """Test complete order lifecycle with event flow."""
        event_bus = EventBus()
        events = []

        # Subscribe to all order events
        for event_type in [
            EventType.ORDER_ACK,
            EventType.ORDER_FILL,
            EventType.ORDER_CANCEL,
        ]:
            event_bus.subscribe(event_type, events.append)

        # Create client
        client = ExchangeClient(
            api_key="test", api_secret="test", testnet=True, event_bus=event_bus
        )

        # Mock exchange
        client.exchange = AsyncMock()
        client.exchange.create_limit_order = AsyncMock(
            return_value={
                "id": "12345",
                "status": "new",
                "clientOrderId": "twap_abc123",
            }
        )

        # Place order
        await client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            client_order_id="twap_abc123",
        )

        assert len(events) == 1
        assert events[0].event_type == EventType.ORDER_ACK
        assert events[0].client_order_id == "twap_abc123"

    @pytest.mark.asyncio
    async def test_websocket_mock_server(self):
        """Test WebSocket connection with mock server."""
        received_messages = []

        async def mock_handler(websocket, path):
            """Mock WebSocket server handler."""
            # Send test message
            await websocket.send(
                json.dumps(
                    {
                        "e": "executionReport",
                        "E": int(time.time() * 1000),
                        "X": "FILLED",
                        "x": "TRADE",
                        "c": "twap_test123",
                        "i": 12345,
                        "t": 67890,
                        "s": "BTCUSDT",
                        "S": "BUY",
                        "q": "0.1",
                        "l": "0.1",
                        "L": "50000",
                        "z": "0.1",
                        "n": "0.0001",
                        "N": "BTC",
                    }
                )
            )

            # Receive messages
            async for message in websocket:
                received_messages.append(json.loads(message))

        # Start mock server
        async with websockets.serve(mock_handler, "localhost", 0) as server:
            port = server.sockets[0].getsockname()[1]

            # Create WebSocket connection
            async with websockets.connect(f"ws://localhost:{port}") as ws:
                # Send subscription
                await ws.send(
                    json.dumps(
                        {"method": "SUBSCRIBE", "params": ["btcusdt@ticker"], "id": 1}
                    )
                )

                # Receive execution report
                message = await ws.recv()
                data = json.loads(message)
                assert data["e"] == "executionReport"
                assert data["c"] == "twap_test123"

        assert len(received_messages) == 1
        assert received_messages[0]["method"] == "SUBSCRIBE"


@pytest.mark.asyncio
async def test_complete_flow():
    """Test complete exchange flow from order to fill."""
    event_bus = EventBus()
    all_events = []

    # Subscribe to all events
    for event_type in EventType:
        event_bus.subscribe(event_type, all_events.append)

    # Create components
    client = ExchangeClient("test", "test", testnet=True, event_bus=event_bus)
    circuit_breaker = EnhancedCircuitBreaker(event_bus=event_bus)

    # Mock exchange
    client.exchange = AsyncMock()
    client.exchange.load_markets = AsyncMock()
    client.exchange.fetch_time = AsyncMock(return_value=int(time.time() * 1000))
    client.exchange.fetch_open_orders = AsyncMock(return_value=[])
    client.exchange.fetch_balance = AsyncMock(
        return_value={
            "BTC": {"free": 1, "used": 0, "total": 1},
            "USDT": {"free": 50000, "used": 0, "total": 50000},
        }
    )
    client.exchange.create_limit_order = AsyncMock(
        return_value={"id": "12345", "status": "new", "clientOrderId": "twap_flow_test"}
    )

    # Initialize
    await client.initialize()

    # Place order
    order = await client.place_order(
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        quantity=Decimal("0.1"),
        price=Decimal("45000"),
        client_order_id="twap_flow_test",
    )

    # Verify events
    reconciliation_events = [
        e
        for e in all_events
        if e.event_type
        in [EventType.RECONCILIATION_START, EventType.RECONCILIATION_COMPLETE]
    ]
    order_events = [e for e in all_events if e.event_type == EventType.ORDER_ACK]

    assert len(reconciliation_events) == 2  # Start and complete
    assert len(order_events) == 1
    assert order_events[0].client_order_id == "twap_flow_test"

    # Test circuit breaker integration
    circuit_breaker.record_success()
    assert circuit_breaker.get_stats()["total_requests"] == 1
    assert circuit_breaker.state == CircuitState.CLOSED
