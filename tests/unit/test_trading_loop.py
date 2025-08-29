"""
Unit tests for the trading loop core module.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
)
from genesis.engine.event_bus import EventBus
from genesis.engine.trading_loop import TradingLoop


@pytest.fixture
async def event_bus():
    """Create event bus fixture."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def risk_engine():
    """Create mock risk engine."""
    engine = MagicMock()
    engine.validate_configuration = MagicMock(return_value=True)
    engine.validate_signal = AsyncMock(return_value={
        "approved": True,
        "position_size": Decimal("100"),
        "reason": None
    })
    # Add tier limits for stop loss calculation
    engine.tier_limits = {"stop_loss_percent": Decimal("2.0")}
    return engine


@pytest.fixture
def exchange_gateway():
    """Create mock exchange gateway."""
    gateway = MagicMock()
    gateway.validate_connection = AsyncMock(return_value=True)
    gateway.execute_order = AsyncMock(return_value={
        "success": True,
        "exchange_order_id": "EX123456",
        "fill_price": Decimal("50000"),
        "latency_ms": 50
    })
    return gateway


@pytest.fixture
async def trading_loop(event_bus, risk_engine, exchange_gateway):
    """Create trading loop fixture."""
    loop = TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=exchange_gateway
    )
    yield loop
    if loop.running:
        await loop.shutdown()


class TestTradingLoopStartup:
    """Test trading loop startup and validation."""

    async def test_successful_startup(self, trading_loop):
        """Test successful startup sequence."""
        result = await trading_loop.startup()

        assert result is True
        assert trading_loop.startup_validated is True
        assert trading_loop.event_bus.running is True

    async def test_startup_with_exchange_failure(self, trading_loop):
        """Test startup with exchange validation failure."""
        trading_loop.exchange_gateway.validate_connection = AsyncMock(return_value=False)

        result = await trading_loop.startup()

        assert result is False
        assert trading_loop.startup_validated is False

    async def test_startup_with_risk_engine_failure(self, trading_loop):
        """Test startup with risk engine validation failure."""
        trading_loop.risk_engine.validate_configuration = MagicMock(return_value=False)

        result = await trading_loop.startup()

        assert result is False
        assert trading_loop.startup_validated is False

    async def test_startup_publishes_event(self, trading_loop):
        """Test that startup publishes system startup event."""
        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append((event, priority))

        trading_loop.event_bus.publish = capture_event

        await trading_loop.startup()

        assert len(published_events) == 1
        event, priority = published_events[0]
        assert event.event_type == EventType.SYSTEM_STARTUP
        assert priority == EventPriority.HIGH


class TestPriceUpdateHandling:
    """Test price update event handling."""

    async def test_price_update_updates_positions(self, trading_loop):
        """Test that price updates update position P&L."""
        await trading_loop.startup()

        # Create a position
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000")
        )
        trading_loop.positions[position.position_id] = position

        # Send price update
        event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": "BTC/USDT",
                "price": "51000"
            }
        )

        await trading_loop._handle_price_update(event)

        # Check position was updated
        assert position.current_price == Decimal("51000")
        assert position.pnl_dollars == Decimal("1000")  # 51000 - 50000
        assert position.pnl_percent == Decimal("2")  # 2% gain

    async def test_stop_loss_trigger_on_price_update(self, trading_loop):
        """Test stop loss triggers on price update."""
        await trading_loop.startup()

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        # Create a long position with stop loss
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000"),
            stop_loss=Decimal("48000")
        )
        trading_loop.positions[position.position_id] = position

        # Send price update below stop loss
        event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": "BTC/USDT",
                "price": "47500"
            }
        )

        await trading_loop._handle_price_update(event)

        # Check stop loss was triggered
        stop_loss_events = [e for e in published_events if e.event_type == EventType.STOP_LOSS_TRIGGERED]
        assert len(stop_loss_events) == 1
        assert stop_loss_events[0].aggregate_id == position.position_id

    async def test_short_position_stop_loss(self, trading_loop):
        """Test stop loss for short positions."""
        await trading_loop.startup()

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        # Create a short position with stop loss
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000"),
            stop_loss=Decimal("52000")
        )
        trading_loop.positions[position.position_id] = position

        # Send price update above stop loss
        event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": "BTC/USDT",
                "price": "52500"
            }
        )

        await trading_loop._handle_price_update(event)

        # Check stop loss was triggered
        stop_loss_events = [e for e in published_events if e.event_type == EventType.STOP_LOSS_TRIGGERED]
        assert len(stop_loss_events) == 1


class TestSignalHandling:
    """Test trading signal handling."""

    async def test_signal_creates_order(self, trading_loop):
        """Test that approved signals create orders."""
        await trading_loop.startup()

        event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "test_strategy",
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": "0.8"
            }
        )

        await trading_loop._handle_trading_signal(event)

        # Check order was executed
        assert trading_loop.exchange_gateway.execute_order.called
        assert trading_loop.signals_generated == 1

    async def test_rejected_signal_no_order(self, trading_loop):
        """Test that rejected signals don't create orders."""
        await trading_loop.startup()

        # Configure risk engine to reject
        trading_loop.risk_engine.validate_signal = AsyncMock(return_value={
            "approved": False,
            "reason": "Insufficient balance"
        })

        event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "test_strategy",
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": "0.8"
            }
        )

        await trading_loop._handle_trading_signal(event)

        # Check no order was executed
        assert not trading_loop.exchange_gateway.execute_order.called
        assert trading_loop.signals_generated == 1

    async def test_zero_position_size_no_order(self, trading_loop):
        """Test that zero position size doesn't create order."""
        await trading_loop.startup()

        # Configure risk engine to return zero size
        trading_loop.risk_engine.validate_signal = AsyncMock(return_value={
            "approved": True,
            "position_size": Decimal("0")
        })

        event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "test_strategy",
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": "0.8"
            }
        )

        await trading_loop._handle_trading_signal(event)

        # Check no order was executed
        assert not trading_loop.exchange_gateway.execute_order.called


class TestOrderExecution:
    """Test order execution flow."""

    async def test_successful_order_execution(self, trading_loop):
        """Test successful order execution."""
        await trading_loop.startup()

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        order = Order(
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1")
        )

        await trading_loop._execute_order(order)

        # Check order was executed
        assert trading_loop.exchange_gateway.execute_order.called
        assert order.status == OrderStatus.FILLED
        assert order.exchange_order_id == "EX123456"
        assert trading_loop.orders_executed == 1

        # Check event was published
        filled_events = [e for e in published_events if e.event_type == EventType.ORDER_FILLED]
        assert len(filled_events) == 1

    async def test_failed_order_execution(self, trading_loop):
        """Test failed order execution."""
        await trading_loop.startup()

        # Configure gateway to fail
        trading_loop.exchange_gateway.execute_order = AsyncMock(return_value={
            "success": False,
            "error": "Insufficient balance"
        })

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        order = Order(
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1")
        )

        await trading_loop._execute_order(order)

        # Check order failed
        assert order.status == OrderStatus.FAILED
        assert trading_loop.orders_executed == 0

        # Check event was published
        failed_events = [e for e in published_events if e.event_type == EventType.ORDER_FAILED]
        assert len(failed_events) == 1


class TestPositionManagement:
    """Test position lifecycle management."""

    async def test_new_position_creation(self, trading_loop):
        """Test creating new position from filled order."""
        await trading_loop.startup()

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        event = Event(
            event_type=EventType.ORDER_FILLED,
            event_data={
                "order_id": "ORDER123",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "1",
                "price": "50000"
            }
        )

        await trading_loop._handle_order_filled(event)

        # Check position was created
        assert len(trading_loop.positions) == 1
        position = list(trading_loop.positions.values())[0]
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        assert position.quantity == Decimal("1")
        assert position.entry_price == Decimal("50000")
        # Stop loss should be 2% below entry price (from config)
        assert position.stop_loss == Decimal("49000")  # 2% stop loss from config
        assert trading_loop.positions_opened == 1

        # Check event was published
        opened_events = [e for e in published_events if e.event_type == EventType.POSITION_OPENED]
        assert len(opened_events) == 1

    async def test_position_averaging(self, trading_loop):
        """Test averaging into existing position."""
        await trading_loop.startup()

        # Create initial position
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000")
        )
        trading_loop.positions[position.position_id] = position

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        # Fill another buy order
        event = Event(
            event_type=EventType.ORDER_FILLED,
            event_data={
                "order_id": "ORDER124",
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "1",
                "price": "52000"
            }
        )

        await trading_loop._handle_order_filled(event)

        # Check position was averaged
        assert len(trading_loop.positions) == 1
        assert position.quantity == Decimal("2")
        assert position.entry_price == Decimal("51000")  # (50000 + 52000) / 2
        assert position.dollar_value == Decimal("102000")

        # Check event was published
        updated_events = [e for e in published_events if e.event_type == EventType.POSITION_UPDATED]
        assert len(updated_events) == 1

    async def test_position_closure_on_stop_loss(self, trading_loop):
        """Test position closure when stop loss is triggered."""
        await trading_loop.startup()

        # Create position
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000"),
            stop_loss=Decimal("48000"),
            pnl_dollars=Decimal("-2000")
        )
        trading_loop.positions[position.position_id] = position

        # Track published events
        published_events = []

        async def capture_event(event, priority):
            published_events.append(event)

        trading_loop.event_bus.publish = capture_event

        event = Event(
            event_type=EventType.STOP_LOSS_TRIGGERED,
            event_data={
                "position_id": position.position_id,
                "symbol": "BTC/USDT",
                "stop_loss": "48000",
                "current_pnl": "-2000"
            }
        )

        await trading_loop._handle_stop_loss(event)

        # Check position was closed
        assert position.position_id not in trading_loop.positions
        assert trading_loop.positions_closed == 1
        assert position.close_reason == "stop_loss"

        # Check event was published
        closed_events = [e for e in published_events if e.event_type == EventType.POSITION_CLOSED]
        assert len(closed_events) == 1
        assert closed_events[0].event_data["close_reason"] == "stop_loss"


class TestStatistics:
    """Test statistics and metrics tracking."""

    async def test_statistics_tracking(self, trading_loop):
        """Test that statistics are properly tracked."""
        await trading_loop.startup()

        # Process some events
        trading_loop.events_processed = 10
        trading_loop.signals_generated = 5
        trading_loop.orders_executed = 3
        trading_loop.positions_opened = 2
        trading_loop.positions_closed = 1

        # Add active position and pending order
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000")
        )
        trading_loop.positions[position.position_id] = position

        order = Order(
            symbol="ETH/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("10")
        )
        trading_loop.pending_orders[order.order_id] = order

        stats = trading_loop.get_statistics()

        assert stats["events_processed"] == 10
        assert stats["signals_generated"] == 5
        assert stats["orders_executed"] == 3
        assert stats["positions_opened"] == 2
        assert stats["positions_closed"] == 1
        assert stats["active_positions"] == 1
        assert stats["pending_orders"] == 1


class TestShutdown:
    """Test graceful shutdown."""

    async def test_graceful_shutdown(self, trading_loop):
        """Test graceful shutdown cleans up properly."""
        await trading_loop.startup()

        # Add some positions and orders
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("50000")
        )
        trading_loop.positions[position.position_id] = position

        order = Order(
            symbol="ETH/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("10")
        )
        trading_loop.pending_orders[order.order_id] = order

        await trading_loop.shutdown()

        assert trading_loop.running is False
        assert not trading_loop.event_bus.running

        # In production, would verify positions closed and orders cancelled
