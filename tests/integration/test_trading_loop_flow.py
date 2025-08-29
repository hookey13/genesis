"""
Integration tests for trading loop order flow and event routing.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import (
    Account,
    OrderSide,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway


@pytest.fixture
async def account():
    """Create test account."""
    return Account(
        account_id="test_account",
        balance_usdt=Decimal("10000"),
        tier=TradingTier.SNIPER
    )


@pytest.fixture
async def session(account):
    """Create test trading session."""
    return TradingSession(
        account_id=account.account_id,
        starting_balance=account.balance_usdt,
        current_balance=account.balance_usdt,
        daily_loss_limit=Decimal("500")
    )


@pytest.fixture
async def event_bus():
    """Create event bus fixture."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def risk_engine(account, session):
    """Create risk engine."""
    return RiskEngine(account=account, session=session, use_kelly_sizing=False)


@pytest.fixture
def exchange_gateway():
    """Create mock exchange gateway."""
    gateway = MagicMock(spec=ExchangeGateway)
    gateway.validate_connection = AsyncMock(return_value=True)
    gateway.execute_order = AsyncMock()
    return gateway


@pytest.fixture
async def trading_loop(event_bus, risk_engine, exchange_gateway):
    """Create trading loop with dependencies."""
    loop = TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=exchange_gateway
    )
    await loop.startup()
    yield loop
    await loop.shutdown()


class TestCompleteOrderFlow:
    """Test complete order flow from signal to position."""

    async def test_signal_to_position_flow(self, trading_loop, exchange_gateway):
        """Test complete flow from signal generation to position creation."""
        # Configure exchange to return successful order
        exchange_gateway.execute_order.return_value = {
            "success": True,
            "exchange_order_id": "EX123456",
            "fill_price": Decimal("50000"),
            "filled_quantity": Decimal("0.1"),
            "status": "FILLED",
            "latency_ms": 50
        }

        # Track events
        captured_events = []

        def capture_event(event):
            captured_events.append(event)

        # Subscribe to all events (global subscription with no specific event type)
        trading_loop.event_bus.subscribe(
            capture_event,  # Callback as first argument
            priority=EventPriority.LOW
        )

        # 1. Publish arbitrage signal
        signal_event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "test_strategy",
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": "0.8",
                "zscore": "2.5",
                "threshold_sigma": "2.0"
            }
        )

        await trading_loop.event_bus.publish(signal_event, EventPriority.HIGH)

        # Allow time for event processing
        await asyncio.sleep(0.5)

        # Verify order was executed
        assert exchange_gateway.execute_order.called

        # Verify position was created
        assert len(trading_loop.positions) == 1
        position = list(trading_loop.positions.values())[0]
        assert position.symbol == "BTC/USDT"
        assert position.side == PositionSide.LONG
        # Risk engine calculates 0.2 BTC: $10k balance, 5% risk = $500,
        # stop loss at 5% = $2500 price risk, $500/$2500 = 0.2 BTC
        assert position.quantity == Decimal("0.2")

        # Verify events were published
        event_types = [e.event_type for e in captured_events]
        assert EventType.ORDER_FILLED in event_types
        assert EventType.POSITION_OPENED in event_types

    async def test_price_update_triggers_stop_loss(self, trading_loop, exchange_gateway):
        """Test that price updates can trigger stop losses."""
        # Create a position with stop loss
        position = Position(
            account_id="test",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            dollar_value=Decimal("5000"),
            stop_loss=Decimal("48000")
        )
        trading_loop.positions[position.position_id] = position

        # Configure exchange for stop loss order
        exchange_gateway.execute_order.return_value = {
            "success": True,
            "exchange_order_id": "EX123457",
            "fill_price": Decimal("47500"),
            "filled_quantity": Decimal("0.1"),
            "status": "FILLED",
            "latency_ms": 45
        }

        # Track events
        captured_events = []

        def capture_event(event):
            captured_events.append(event)

        trading_loop.event_bus.subscribe(
            EventType.STOP_LOSS_TRIGGERED,
            capture_event
        )

        trading_loop.event_bus.subscribe(
            EventType.POSITION_CLOSED,
            capture_event
        )

        # Publish price update below stop loss
        price_event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": "BTC/USDT",
                "price": "47500"
            }
        )

        await trading_loop.event_bus.publish(price_event, EventPriority.HIGH)

        # Allow time for event processing
        await asyncio.sleep(0.5)

        # Verify stop loss was triggered
        assert exchange_gateway.execute_order.called
        order_call = exchange_gateway.execute_order.call_args[0][0]
        assert order_call.side == OrderSide.SELL
        assert order_call.quantity == Decimal("0.1")

        # Verify events
        event_types = [e.event_type for e in captured_events]
        assert EventType.STOP_LOSS_TRIGGERED in event_types
        assert EventType.POSITION_CLOSED in event_types

        # Verify position was closed
        assert position.position_id not in trading_loop.positions
