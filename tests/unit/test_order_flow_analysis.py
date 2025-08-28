"""Unit tests for Order Flow Analysis."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

import pytest

from genesis.analytics.order_flow_analysis import (
    OrderFlowMetrics,
    TradeFlow,
    FlowWindow,
    OrderFlowAnalyzer,
)
from genesis.exchange.order_book_manager import OrderBookSnapshot, OrderBookLevel
from genesis.engine.event_bus import EventBus
from genesis.core.events import Event


class TestOrderFlowMetrics:
    """Test OrderFlowMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating order flow metrics."""
        metrics = OrderFlowMetrics(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ofi=Decimal("0.5"),
            volume_ratio=Decimal("2.0"),
            aggression_ratio=Decimal("1.5"),
            net_flow=Decimal("100"),
            flow_velocity=Decimal("10"),
            pressure_score=Decimal("50"),
            confidence=Decimal("0.8"),
        )

        assert metrics.symbol == "BTCUSDT"
        assert metrics.ofi == Decimal("0.5")
        assert metrics.pressure_score == Decimal("50")

    def test_bullish_bearish_detection(self):
        """Test bullish/bearish detection methods."""
        # Bullish metrics
        metrics = OrderFlowMetrics(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ofi=Decimal("0.5"),
            volume_ratio=Decimal("2.0"),
            aggression_ratio=Decimal("1.5"),
            net_flow=Decimal("100"),
            flow_velocity=Decimal("10"),
            pressure_score=Decimal("50"),
            confidence=Decimal("0.8"),
        )
        assert metrics.is_bullish()
        assert not metrics.is_bearish()

        # Bearish metrics
        metrics.pressure_score = Decimal("-50")
        assert not metrics.is_bullish()
        assert metrics.is_bearish()

        # Neutral
        metrics.pressure_score = Decimal("10")
        assert not metrics.is_bullish()
        assert not metrics.is_bearish()

    def test_significance_check(self):
        """Test significance checking."""
        metrics = OrderFlowMetrics(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ofi=Decimal("0.4"),
            volume_ratio=Decimal("2.0"),
            aggression_ratio=Decimal("1.5"),
            net_flow=Decimal("100"),
            flow_velocity=Decimal("10"),
            pressure_score=Decimal("50"),
            confidence=Decimal("0.8"),
        )

        # Significant: high OFI and confidence
        assert metrics.is_significant()

        # Not significant: low OFI
        metrics.ofi = Decimal("0.2")
        assert not metrics.is_significant()

        # Not significant: low confidence
        metrics.ofi = Decimal("0.4")
        metrics.confidence = Decimal("0.5")
        assert not metrics.is_significant()


class TestTradeFlow:
    """Test TradeFlow dataclass."""

    def test_trade_flow_creation(self):
        """Test creating trade flow record."""
        trade = TradeFlow(
            timestamp=datetime.now(timezone.utc),
            price=Decimal("50000"),
            quantity=Decimal("1.5"),
            side="buy",
            is_aggressive=True,
        )

        assert trade.price == Decimal("50000")
        assert trade.quantity == Decimal("1.5")
        assert trade.notional == Decimal("75000")
        assert trade.side == "buy"
        assert trade.is_aggressive


class TestFlowWindow:
    """Test FlowWindow class."""

    def test_add_trade_and_expiry(self):
        """Test adding trades and automatic expiry."""
        window = FlowWindow(window_size=timedelta(minutes=5))

        # Add recent trade
        recent_trade = TradeFlow(
            timestamp=datetime.now(timezone.utc),
            price=Decimal("50000"),
            quantity=Decimal("1"),
            side="buy",
            is_aggressive=True,
        )
        window.add_trade(recent_trade)
        assert len(window.trades) == 1

        # Add old trade (should be removed)
        old_trade = TradeFlow(
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
            price=Decimal("49000"),
            quantity=Decimal("2"),
            side="sell",
            is_aggressive=False,
        )
        window.trades.appendleft(old_trade)

        # Add another recent trade to trigger cleanup
        window.add_trade(recent_trade)
        assert len(window.trades) == 2  # Old trade removed

    def test_volume_calculations(self):
        """Test volume calculation methods."""
        window = FlowWindow(window_size=timedelta(minutes=5))

        # Add various trades
        trades = [
            TradeFlow(
                datetime.now(timezone.utc), Decimal("50000"), Decimal("1"), "buy", True
            ),
            TradeFlow(
                datetime.now(timezone.utc), Decimal("50001"), Decimal("2"), "buy", False
            ),
            TradeFlow(
                datetime.now(timezone.utc),
                Decimal("49999"),
                Decimal("1.5"),
                "sell",
                True,
            ),
            TradeFlow(
                datetime.now(timezone.utc),
                Decimal("49998"),
                Decimal("0.5"),
                "sell",
                False,
            ),
        ]

        for trade in trades:
            window.add_trade(trade)

        assert window.get_buy_volume() == Decimal("3")
        assert window.get_sell_volume() == Decimal("2")
        assert window.get_aggressive_volume() == Decimal("2.5")
        assert window.get_passive_volume() == Decimal("2.5")


@pytest.mark.asyncio
class TestOrderFlowAnalyzer:
    """Test OrderFlowAnalyzer class."""

    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        return AsyncMock(spec=EventBus)

    @pytest.fixture
    def analyzer(self, event_bus):
        """Create order flow analyzer instance."""
        return OrderFlowAnalyzer(
            event_bus=event_bus, window_minutes=5, sensitivity=Decimal("0.3")
        )

    async def test_analyze_trade(self, analyzer, event_bus):
        """Test analyzing individual trades."""
        # Analyze buy trade (taker buy)
        metrics = await analyzer.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            quantity=Decimal("1"),
            is_buyer_maker=False,
        )

        assert metrics is not None
        assert metrics.symbol == "BTCUSDT"
        assert metrics.net_flow == Decimal("1")  # One buy trade

        # Analyze sell trade (taker sell)
        metrics = await analyzer.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("49999"),
            quantity=Decimal("0.5"),
            is_buyer_maker=True,
        )

        assert metrics.net_flow == Decimal("0.5")  # 1 buy - 0.5 sell

    async def test_imbalance_detection(self, analyzer, event_bus):
        """Test detecting significant imbalances."""
        # Create strong buy imbalance
        for _ in range(10):
            await analyzer.analyze_trade(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                quantity=Decimal("1"),
                is_buyer_maker=False,
            )

        # Add small sell to trigger imbalance
        metrics = await analyzer.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("49999"),
            quantity=Decimal("0.1"),
            is_buyer_maker=True,
        )

        assert metrics.ofi > Decimal("0.9")  # Strong buy imbalance
        assert metrics.is_significant()

        # Check event was published
        event_bus.publish.assert_called()
        call_args = event_bus.publish.call_args[0][0]
        assert call_args.type == "order_flow_imbalance"
        assert call_args.data["direction"] == "buy"

    def test_calculate_metrics(self, analyzer):
        """Test metric calculations."""
        window = FlowWindow(window_size=timedelta(minutes=5))

        # Add balanced trades
        trades = [
            TradeFlow(
                datetime.now(timezone.utc), Decimal("50000"), Decimal("1"), "buy", True
            ),
            TradeFlow(
                datetime.now(timezone.utc), Decimal("49999"), Decimal("1"), "sell", True
            ),
        ]

        for trade in trades:
            window.add_trade(trade)

        metrics = analyzer._calculate_metrics("BTCUSDT", window)

        assert metrics.ofi == Decimal("0")  # Balanced
        assert metrics.volume_ratio == Decimal("1")  # Equal volumes
        assert metrics.net_flow == Decimal("0")

    def test_pressure_score_calculation(self, analyzer):
        """Test pressure score calculation."""
        # Strong buy pressure
        score = analyzer._calculate_pressure_score(
            ofi=Decimal("0.8"),
            volume_ratio=Decimal("4"),
            aggression_ratio=Decimal("2"),
            flow_velocity=Decimal("5"),
        )
        assert score > Decimal("20")  # Bullish

        # Strong sell pressure
        score = analyzer._calculate_pressure_score(
            ofi=Decimal("-0.8"),
            volume_ratio=Decimal("0.25"),
            aggression_ratio=Decimal("0.5"),
            flow_velocity=Decimal("-5"),
        )
        assert score < Decimal("-20")  # Bearish

        # Neutral
        score = analyzer._calculate_pressure_score(
            ofi=Decimal("0"),
            volume_ratio=Decimal("1"),
            aggression_ratio=Decimal("1"),
            flow_velocity=Decimal("0"),
        )
        assert abs(score) < Decimal("20")

    def test_confidence_calculation(self, analyzer):
        """Test confidence score calculation."""
        window = FlowWindow(window_size=timedelta(minutes=5))

        # Low confidence: few trades, low volume
        confidence = analyzer._calculate_confidence(
            symbol="BTCUSDT", total_volume=Decimal("10"), window=window
        )
        assert confidence < Decimal("0.7")

        # Add many trades for higher confidence
        for _ in range(150):
            window.trades.append(
                TradeFlow(
                    datetime.now(timezone.utc),
                    Decimal("50000"),
                    Decimal("1"),
                    "buy",
                    True,
                )
            )

        confidence = analyzer._calculate_confidence(
            symbol="BTCUSDT", total_volume=Decimal("150"), window=window
        )
        assert confidence > Decimal("0.8")

    async def test_flow_trend_detection(self, analyzer):
        """Test detecting flow trends."""
        # Generate bullish trend
        for i in range(15):
            # Increasing buy pressure
            for _ in range(i + 1):
                await analyzer.analyze_trade(
                    symbol="BTCUSDT",
                    price=Decimal("50000"),
                    quantity=Decimal("1"),
                    is_buyer_maker=False,
                )
            # Small sells
            await analyzer.analyze_trade(
                symbol="BTCUSDT",
                price=Decimal("49999"),
                quantity=Decimal("0.1"),
                is_buyer_maker=True,
            )

        trend = analyzer.get_flow_trend("BTCUSDT")
        assert trend == "bullish"

    def test_cumulative_flow(self, analyzer):
        """Test getting cumulative flow."""
        # No data
        assert analyzer.get_cumulative_flow("BTCUSDT") is None

        # Create flow window
        window = FlowWindow(window_size=timedelta(minutes=5))
        trades = [
            TradeFlow(
                datetime.now(timezone.utc), Decimal("50000"), Decimal("3"), "buy", True
            ),
            TradeFlow(
                datetime.now(timezone.utc), Decimal("49999"), Decimal("1"), "sell", True
            ),
        ]
        for trade in trades:
            window.add_trade(trade)

        analyzer.flow_windows["BTCUSDT"] = window

        cumulative = analyzer.get_cumulative_flow("BTCUSDT")
        assert cumulative == Decimal("2")  # 3 buy - 1 sell

    def test_flow_divergence_detection(self, analyzer):
        """Test detecting price-flow divergence."""
        # Setup flow history for bullish flow
        analyzer.metrics_history["BTCUSDT"] = deque(
            [
                OrderFlowMetrics(
                    symbol="BTCUSDT",
                    timestamp=datetime.now(timezone.utc),
                    ofi=Decimal("0.5"),
                    volume_ratio=Decimal("2"),
                    aggression_ratio=Decimal("1.5"),
                    net_flow=Decimal("100"),
                    flow_velocity=Decimal("10"),
                    pressure_score=Decimal("50"),
                    confidence=Decimal("0.8"),
                )
                for _ in range(10)
            ]
        )

        # Bullish divergence: price down but flow bullish
        divergence = analyzer.detect_flow_divergence("BTCUSDT", "down")
        assert divergence == "bullish"

        # No divergence: price and flow aligned
        divergence = analyzer.detect_flow_divergence("BTCUSDT", "up")
        assert divergence is None

        # Setup bearish flow
        for metrics in analyzer.metrics_history["BTCUSDT"]:
            metrics.pressure_score = Decimal("-50")

        # Bearish divergence: price up but flow bearish
        divergence = analyzer.detect_flow_divergence("BTCUSDT", "up")
        assert divergence == "bearish"

    def test_update_order_book(self, analyzer):
        """Test updating order book snapshots."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1"))],
        )

        analyzer.update_order_book(snapshot)
        assert analyzer.last_snapshots["BTCUSDT"] == snapshot

    async def test_flow_velocity_calculation(self, analyzer):
        """Test flow velocity calculation."""
        # First trade - no velocity yet
        await analyzer.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            quantity=Decimal("1"),
            is_buyer_maker=False,
        )

        # Wait a moment for time difference
        await asyncio.sleep(0.1)

        # Second trade - should have velocity
        metrics = await analyzer.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("50001"),
            quantity=Decimal("2"),
            is_buyer_maker=False,
        )

        # Velocity should be positive (increasing buy flow)
        assert metrics.flow_velocity != Decimal("0")
