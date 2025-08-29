"""
Unit tests for risk engine integration with trading loop.

Tests position sizing, risk validation, stop-loss calculations,
and proper integration between components.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.exceptions import (
    DailyLossLimitReached,
    MinimumPositionSize,
    RiskLimitExceeded,
)
from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway


class TestRiskIntegration:
    """Test risk engine integration with trading loop."""

    @pytest.fixture
    def account(self):
        """Create test account."""
        return Account(
            account_id="test_account",
            tier=TradingTier.SNIPER,
            balance_usdt=Decimal("500"),
            locked_features=[]
        )

    @pytest.fixture
    def session(self):
        """Create test trading session."""
        return TradingSession(
            account_id="test_account",
            daily_loss_limit=Decimal("50"),  # 10% of $500
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            starting_balance=Decimal("500"),
            current_balance=Decimal("500")
        )

    @pytest.fixture
    def risk_engine(self, account, session):
        """Create risk engine with test account and session."""
        return RiskEngine(
            account=account,
            session=session,
            use_kelly_sizing=False  # Disable for SNIPER tier
        )

    @pytest.fixture
    def event_bus(self):
        """Create event bus."""
        return EventBus()

    @pytest.fixture
    def exchange_gateway(self):
        """Create mock exchange gateway."""
        gateway = MagicMock(spec=ExchangeGateway)
        gateway.validate_connection = AsyncMock(return_value=True)
        gateway.execute_order = AsyncMock(return_value={
            "success": True,
            "exchange_order_id": "12345",
            "fill_price": Decimal("50000"),
            "latency_ms": 10
        })
        return gateway

    @pytest.fixture
    def trading_loop(self, event_bus, risk_engine, exchange_gateway):
        """Create trading loop with dependencies."""
        return TradingLoop(
            event_bus=event_bus,
            risk_engine=risk_engine,
            exchange_gateway=exchange_gateway
        )

    @pytest.mark.asyncio
    async def test_position_sizing_respects_5_percent_rule(self, trading_loop, risk_engine):
        """Test that position sizing respects the 5% risk rule."""
        # Calculate position size for BTC at $50,000
        entry_price = Decimal("50000")
        position_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price
        )

        # With $500 balance and 5% risk = $25 risk amount
        # With 2% stop loss from $50,000 = $1,000 price risk per unit
        # Position size = $25 / $1,000 = 0.025 BTC
        # But position value = 0.025 * $50,000 = $1,250 > $500 balance
        # So it should be limited to balance / price = $500 / $50,000 = 0.01 BTC
        expected_size = Decimal("0.01")

        assert position_size == expected_size

    @pytest.mark.asyncio
    async def test_orders_rejected_when_balance_insufficient(self, trading_loop, risk_engine):
        """Test that orders are rejected when balance is insufficient."""
        # Set balance to $5 (below minimum position size)
        risk_engine.account.balance_usdt = Decimal("5")

        with pytest.raises(MinimumPositionSize):
            risk_engine.calculate_position_size(
                symbol="BTC/USDT",
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_stop_loss_orders_created_at_correct_price(self, trading_loop, risk_engine):
        """Test that stop-loss orders are created at the correct price (2% for SNIPER)."""
        entry_price = Decimal("50000")

        # Test long position stop loss
        long_stop_loss = risk_engine.calculate_stop_loss(
            entry_price=entry_price,
            side=PositionSide.LONG
        )
        expected_long_sl = Decimal("49000")  # 2% below entry
        assert long_stop_loss == expected_long_sl

        # Test short position stop loss
        short_stop_loss = risk_engine.calculate_stop_loss(
            entry_price=entry_price,
            side=PositionSide.SHORT
        )
        expected_short_sl = Decimal("51000")  # 2% above entry
        assert short_stop_loss == expected_short_sl

    @pytest.mark.asyncio
    async def test_maximum_position_limit_enforced(self, trading_loop, risk_engine):
        """Test that maximum position limit is enforced ($100 for SNIPER)."""
        # SNIPER tier has max_position_size of $100
        # Try to validate an order that would exceed this
        with pytest.raises(RiskLimitExceeded):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.003"),  # 0.003 * $50,000 = $150 > $100 limit
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_daily_loss_limit_prevents_new_trades(self, trading_loop, risk_engine):
        """Test that daily loss limit prevents new trades after 10% loss."""
        # Set session to have reached daily loss limit
        # Daily loss limit is $50 for the test session
        risk_engine.session.realized_pnl = Decimal("-51")  # More than $50 loss

        # Session should now report daily limit reached
        assert risk_engine.session.is_daily_limit_reached() is True

        with pytest.raises(DailyLossLimitReached):
            # Use a smaller position that won't trigger position risk limit (5%)
            # 0.0004 * 50000 = $20 which is 4% of $500
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.0004"),
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_portfolio_exposure_correctly_calculated(self, trading_loop, risk_engine):
        """Test that portfolio exposure is correctly calculated across positions."""
        # Create multiple positions
        positions = [
            Position(
                position_id="pos1",
                account_id="test_account",
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.001"),
                dollar_value=Decimal("50")
            ),
            Position(
                position_id="pos2",
                account_id="test_account",
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("3000"),
                quantity=Decimal("0.01"),
                dollar_value=Decimal("30")
            )
        ]

        # Add positions to risk engine
        for position in positions:
            risk_engine.add_position(position)

        # Validate portfolio risk
        portfolio_risk = risk_engine.validate_portfolio_risk(positions)

        assert portfolio_risk["portfolio_exposure"] == Decimal("80")  # $50 + $30
        assert portfolio_risk["approved"] is True  # Within 90% limit of $500

    @pytest.mark.asyncio
    async def test_risk_events_properly_logged_with_correlation_ids(self, trading_loop):
        """Test that risk events are properly logged with correlation IDs."""
        await trading_loop.startup()

        # Create a trading signal event
        signal_event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "test_strategy",
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": 0.8,
                "entry_price": "50000"
            }
        )

        # Spy on event publishing
        published_events = []
        original_publish = trading_loop.event_bus.publish

        async def capture_publish(event, priority=EventPriority.NORMAL):
            published_events.append(event)
            return await original_publish(event, priority)

        trading_loop.event_bus.publish = capture_publish

        # Handle the signal
        await trading_loop._handle_trading_signal(signal_event)

        # Check that risk validation events were published
        risk_events = [e for e in published_events if e.event_type in [
            EventType.RISK_CHECK_PASSED,
            EventType.RISK_CHECK_FAILED
        ]]

        # Should have at least one risk event
        assert len(risk_events) > 0

        # Check correlation ID is present
        for event in risk_events:
            assert event.correlation_id is not None
            assert len(event.correlation_id) > 0

    @pytest.mark.asyncio
    async def test_signal_to_order_flow_with_risk_validation(self, trading_loop):
        """Test complete signal to order flow with risk validation."""
        await trading_loop.startup()

        # Track order execution
        order_executed = False

        async def mock_execute_order(order):
            nonlocal order_executed
            order_executed = True
            return {
                "success": True,
                "exchange_order_id": "12345",
                "fill_price": Decimal("50000"),
                "latency_ms": 10
            }

        trading_loop.exchange_gateway.execute_order = mock_execute_order

        # Create valid signal
        signal_event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "test_strategy",
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": 0.8,
                "entry_price": "50000"
            }
        )

        # Process signal
        await trading_loop._handle_trading_signal(signal_event)

        # Verify order was executed after risk validation
        assert order_executed is True
        assert trading_loop.signals_generated == 1

    @pytest.mark.asyncio
    async def test_stop_loss_triggered_on_price_update(self, trading_loop):
        """Test that stop loss is triggered when price hits stop level."""
        await trading_loop.startup()

        # Create a position with stop loss
        position = Position(
            position_id="test_pos",
            account_id="test_account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001"),
            dollar_value=Decimal("50"),
            stop_loss=Decimal("49000")  # 2% below entry
        )

        trading_loop.positions[position.position_id] = position

        # Track stop loss trigger
        stop_loss_triggered = False

        async def mock_trigger_stop_loss(pos):
            nonlocal stop_loss_triggered
            stop_loss_triggered = True

        trading_loop._trigger_stop_loss = mock_trigger_stop_loss

        # Send price update that triggers stop loss
        price_event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": "BTC/USDT",
                "price": "48500"  # Below stop loss
            }
        )

        await trading_loop._handle_price_update(price_event)

        # Verify stop loss was triggered
        assert stop_loss_triggered is True

    @pytest.mark.asyncio
    async def test_risk_validation_with_portfolio_limits(self, trading_loop, risk_engine):
        """Test risk validation considers portfolio-wide limits."""
        # Create three existing positions to reach SNIPER limit
        for i in range(3):
            position = Position(
                position_id=f"pos_{i}",
                account_id="test_account",
                symbol=f"COIN{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("1000"),
                quantity=Decimal("0.01"),
                dollar_value=Decimal("10")
            )
            risk_engine.add_position(position)

        # SNIPER tier allows max 3 positions, we have 3 already
        assert len(risk_engine.positions) == 3

        # Try to add a 4th position - should fail
        with pytest.raises(RiskLimitExceeded) as exc_info:
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000")
            )

        assert "Maximum positions" in str(exc_info.value)
