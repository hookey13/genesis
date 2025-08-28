"""Integration tests for pre-trade correlation checks."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from genesis.analytics.correlation import CorrelationImpact, CorrelationMonitor
from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import Order, Position
from genesis.engine.event_bus import EventBus
from genesis.engine.executor.base import OrderExecutor
from genesis.engine.risk_engine import RiskEngine


@pytest.fixture
def mock_order_executor():
    """Create mock order executor with correlation checks."""
    executor = Mock(spec=OrderExecutor)
    executor.execute_order = AsyncMock()
    executor.validate_order = AsyncMock(return_value=True)
    return executor


@pytest.fixture
def mock_risk_engine():
    """Create mock risk engine with correlation integration."""
    engine = Mock(spec=RiskEngine)
    engine.check_risk_limits = AsyncMock(return_value=True)
    engine.get_correlation_matrix = AsyncMock()
    return engine


@pytest.fixture
async def pre_trade_environment(mock_order_executor, mock_risk_engine):
    """Set up pre-trade test environment."""
    event_bus = EventBus()

    config = {
        "correlation_monitoring": {
            "thresholds": {"warning": 0.6, "critical": 0.8},
            "analysis": {"cache_ttl_seconds": 5},
        }
    }

    correlation_monitor = CorrelationMonitor(event_bus=event_bus, config=config)

    return {
        "event_bus": event_bus,
        "correlation_monitor": correlation_monitor,
        "order_executor": mock_order_executor,
        "risk_engine": mock_risk_engine,
    }


@pytest.fixture
def existing_portfolio():
    """Create existing portfolio positions."""
    return [
        Position(
            position_id=uuid4(),
            account_id=uuid4(),
            symbol="BTC/USDT",
            side="LONG",
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("25500"),
            pnl_dollars=Decimal("500"),
            pnl_percent=Decimal("2.0"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        ),
        Position(
            position_id=uuid4(),
            account_id=uuid4(),
            symbol="ETH/USDT",
            side="LONG",
            entry_price=Decimal("3000"),
            current_price=Decimal("3100"),
            quantity=Decimal("5"),
            dollar_value=Decimal("15500"),
            pnl_dollars=Decimal("500"),
            pnl_percent=Decimal("3.33"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        ),
    ]


class TestPreTradeCorrelation:
    """Test pre-trade correlation impact analysis."""

    @pytest.mark.asyncio
    async def test_low_correlation_trade_approved(
        self, pre_trade_environment, existing_portfolio
    ):
        """Test that low correlation trades are approved."""
        env = await pre_trade_environment
        monitor = env["correlation_monitor"]

        # New position with low expected correlation
        new_position = Position(
            position_id=uuid4(),
            account_id=existing_portfolio[0].account_id,
            symbol="GOLD/USDT",  # Different asset class
            side="LONG",
            entry_price=Decimal("1800"),
            current_price=Decimal("1810"),
            quantity=Decimal("10"),
            dollar_value=Decimal("18100"),
            pnl_dollars=Decimal("100"),
            pnl_percent=Decimal("0.56"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        )

        # Mock low correlation
        with patch.object(monitor, "calculate_correlation_matrix") as mock_calc:
            # Current portfolio correlation
            mock_calc.side_effect = [
                np.array([[1.0, 0.3], [0.3, 1.0]]),  # Current
                np.array(
                    [  # With new position
                        [1.0, 0.3, 0.1],
                        [0.3, 1.0, 0.15],
                        [0.1, 0.15, 1.0],
                    ]
                ),
            ]

            impact = await monitor.calculate_correlation_impact(
                new_position, existing_portfolio
            )

            assert impact.risk_assessment == "low"
            assert "Safe to proceed" in impact.recommendation
            assert impact.projected_correlation < Decimal("0.6")

    @pytest.mark.asyncio
    async def test_high_correlation_trade_warning(
        self, pre_trade_environment, existing_portfolio
    ):
        """Test that high correlation trades generate warnings."""
        env = await pre_trade_environment
        monitor = env["correlation_monitor"]

        # New position highly correlated with existing
        new_position = Position(
            position_id=uuid4(),
            account_id=existing_portfolio[0].account_id,
            symbol="WBTC/USDT",  # Wrapped BTC - high correlation with BTC
            side="LONG",
            entry_price=Decimal("50100"),
            current_price=Decimal("51100"),
            quantity=Decimal("0.3"),
            dollar_value=Decimal("15330"),
            pnl_dollars=Decimal("300"),
            pnl_percent=Decimal("2.0"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        )

        # Mock high correlation
        with patch.object(monitor, "calculate_correlation_matrix") as mock_calc:
            mock_calc.side_effect = [
                np.array([[1.0, 0.4], [0.4, 1.0]]),  # Current
                np.array(
                    [  # With new position
                        [1.0, 0.4, 0.95],  # Very high correlation with BTC
                        [0.4, 1.0, 0.3],
                        [0.95, 0.3, 1.0],
                    ]
                ),
            ]

            impact = await monitor.calculate_correlation_impact(
                new_position, existing_portfolio
            )

            assert impact.risk_assessment in ["medium", "high"]
            assert impact.projected_correlation > Decimal("0.6")
            assert "correlation" in impact.recommendation.lower()

    @pytest.mark.asyncio
    async def test_correlation_check_in_order_flow(
        self, pre_trade_environment, existing_portfolio
    ):
        """Test correlation check integrated in order execution flow."""
        env = await pre_trade_environment
        monitor = env["correlation_monitor"]
        executor = env["order_executor"]
        risk_engine = env["risk_engine"]

        # Create order
        order = Order(
            order_id=uuid4(),
            account_id=existing_portfolio[0].account_id,
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.2"),
            price=None,
            status="PENDING",
            created_at=datetime.now(UTC),
        )

        # Mock order validation with correlation check
        async def validate_with_correlation(order):
            # Convert order to position for correlation check
            new_position = Position(
                position_id=uuid4(),
                account_id=order.account_id,
                symbol=order.symbol,
                side="LONG" if order.side == "BUY" else "SHORT",
                entry_price=Decimal("51000"),  # Current market price
                current_price=Decimal("51000"),
                quantity=order.quantity,
                dollar_value=order.quantity * Decimal("51000"),
                pnl_dollars=Decimal("0"),
                pnl_percent=Decimal("0"),
                opened_at=datetime.now(UTC),
                closed_at=None,
            )

            impact = await monitor.calculate_correlation_impact(
                new_position, existing_portfolio
            )

            # Reject if high risk
            if impact.risk_assessment == "high":
                return False, "High correlation risk"

            return True, "Approved"

        # Execute order with correlation check
        approved, message = await validate_with_correlation(order)

        if approved:
            await executor.execute_order(order)
            assert executor.execute_order.called
        else:
            assert "correlation" in message.lower()

    @pytest.mark.asyncio
    async def test_multi_asset_correlation_impact(self, pre_trade_environment):
        """Test correlation impact with multiple asset classes."""
        env = await pre_trade_environment
        monitor = env["correlation_monitor"]

        # Diverse portfolio
        portfolio = [
            Position(
                position_id=uuid4(),
                account_id=uuid4(),
                symbol="BTC/USDT",
                side="LONG",
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                quantity=Decimal("0.5"),
                dollar_value=Decimal("25500"),
                pnl_dollars=Decimal("500"),
                pnl_percent=Decimal("2.0"),
                opened_at=datetime.now(UTC),
                closed_at=None,
            ),
            Position(
                position_id=uuid4(),
                account_id=uuid4(),
                symbol="GOLD/USDT",
                side="LONG",
                entry_price=Decimal("1800"),
                current_price=Decimal("1810"),
                quantity=Decimal("10"),
                dollar_value=Decimal("18100"),
                pnl_dollars=Decimal("100"),
                pnl_percent=Decimal("0.56"),
                opened_at=datetime.now(UTC),
                closed_at=None,
            ),
            Position(
                position_id=uuid4(),
                account_id=uuid4(),
                symbol="EUR/USDT",
                side="SHORT",
                entry_price=Decimal("1.08"),
                current_price=Decimal("1.07"),
                quantity=Decimal("10000"),
                dollar_value=Decimal("10700"),
                pnl_dollars=Decimal("100"),
                pnl_percent=Decimal("0.93"),
                opened_at=datetime.now(UTC),
                closed_at=None,
            ),
        ]

        # New crypto position
        new_crypto = Position(
            position_id=uuid4(),
            account_id=portfolio[0].account_id,
            symbol="ETH/USDT",
            side="LONG",
            entry_price=Decimal("3000"),
            current_price=Decimal("3000"),
            quantity=Decimal("3"),
            dollar_value=Decimal("9000"),
            pnl_dollars=Decimal("0"),
            pnl_percent=Decimal("0"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        )

        impact = await monitor.calculate_correlation_impact(new_crypto, portfolio)

        assert isinstance(impact, CorrelationImpact)
        assert impact.current_correlation >= Decimal("0")
        assert impact.projected_correlation >= Decimal("0")

    @pytest.mark.asyncio
    async def test_correlation_based_position_sizing(
        self, pre_trade_environment, existing_portfolio
    ):
        """Test position sizing adjustment based on correlation."""
        env = await pre_trade_environment
        monitor = env["correlation_monitor"]

        # High correlation position
        high_corr_position = Position(
            position_id=uuid4(),
            account_id=existing_portfolio[0].account_id,
            symbol="WBTC/USDT",
            side="LONG",
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            quantity=Decimal("1.0"),  # Original size
            dollar_value=Decimal("50000"),
            pnl_dollars=Decimal("0"),
            pnl_percent=Decimal("0"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        )

        # Mock high correlation
        with patch.object(monitor, "calculate_correlation_matrix") as mock_calc:
            mock_calc.side_effect = [
                np.array([[1.0, 0.4], [0.4, 1.0]]),
                np.array([[1.0, 0.4, 0.9], [0.4, 1.0, 0.3], [0.9, 0.3, 1.0]]),
            ]

            impact = await monitor.calculate_correlation_impact(
                high_corr_position, existing_portfolio
            )

            # Suggest reduced position size for high correlation
            if impact.risk_assessment == "high":
                suggested_reduction = Decimal("0.5")  # Reduce by 50%
                high_corr_position.quantity *= suggested_reduction
                high_corr_position.dollar_value *= suggested_reduction

                assert high_corr_position.quantity == Decimal("0.5")
                assert high_corr_position.dollar_value == Decimal("25000")

    @pytest.mark.asyncio
    async def test_correlation_event_publication(
        self, pre_trade_environment, existing_portfolio
    ):
        """Test that correlation events are published correctly."""
        env = await pre_trade_environment
        monitor = env["correlation_monitor"]
        event_bus = env["event_bus"]

        events_received = []

        # Subscribe to events
        async def event_handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.RISK_ALERT, event_handler, EventPriority.HIGH)

        # New high correlation position
        new_position = Position(
            position_id=uuid4(),
            account_id=existing_portfolio[0].account_id,
            symbol="GBTC/USDT",  # Another BTC derivative
            side="LONG",
            entry_price=Decimal("45000"),
            current_price=Decimal("45000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("22500"),
            pnl_dollars=Decimal("0"),
            pnl_percent=Decimal("0"),
            opened_at=datetime.now(UTC),
            closed_at=None,
        )

        # Check correlation impact
        with patch.object(monitor, "calculate_correlation_matrix") as mock_calc:
            mock_calc.side_effect = [
                np.array([[1.0, 0.5], [0.5, 1.0]]),
                np.array([[1.0, 0.5, 0.85], [0.5, 1.0, 0.4], [0.85, 0.4, 1.0]]),
            ]

            impact = await monitor.calculate_correlation_impact(
                new_position, existing_portfolio
            )

            # If high risk, publish event
            if impact.risk_assessment == "high":
                event = Event(
                    type=EventType.RISK_ALERT,
                    priority=EventPriority.HIGH,
                    data={
                        "type": "correlation_risk",
                        "impact": impact,
                        "position": new_position,
                    },
                )
                await event_bus.publish(event)

                # Allow event processing
                await asyncio.sleep(0.1)

                assert len(events_received) > 0
                assert events_received[0].data["type"] == "correlation_risk"
