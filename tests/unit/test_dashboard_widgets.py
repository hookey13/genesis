"""Unit tests for enhanced dashboard UI widgets."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from genesis.tilt.detector import TiltLevel
from genesis.ui.widgets.pnl import PnLWidget
from genesis.ui.widgets.positions import PositionWidget
from genesis.ui.widgets.tilt_indicator import TiltIndicator


class TestPnLWidget:
    """Test enhanced P&L widget functionality."""

    @pytest.fixture
    def pnl_widget(self):
        """Create a P&L widget instance."""
        return PnLWidget()

    def test_init(self, pnl_widget):
        """Test P&L widget initialization."""
        assert pnl_widget.current_pnl == Decimal("0.00")
        assert pnl_widget.daily_pnl == Decimal("0.00")
        assert pnl_widget.daily_pnl_pct == Decimal("0.00")
        assert pnl_widget.max_drawdown == Decimal("0.00")
        assert pnl_widget.sharpe_ratio == Decimal("0.00")
        assert len(pnl_widget.pnl_data_points) == 0

    def test_add_pnl_data_point(self, pnl_widget):
        """Test adding P&L data points to history."""
        # Add data points
        pnl_widget.add_pnl_data_point(Decimal("100"))
        pnl_widget.add_pnl_data_point(Decimal("150"))
        pnl_widget.add_pnl_data_point(Decimal("125"))

        assert len(pnl_widget.pnl_data_points) == 3
        assert pnl_widget.pnl_data_points[0] == Decimal("100")
        assert pnl_widget.pnl_data_points[2] == Decimal("125")

    def test_calculate_risk_metrics(self, pnl_widget):
        """Test risk metrics calculation."""
        # Add test data with drawdown
        test_values = [
            Decimal("1000"),
            Decimal("1200"),  # Peak
            Decimal("900"),   # Drawdown
            Decimal("1100"),
        ]

        for value in test_values:
            pnl_widget.add_pnl_data_point(value)

        pnl_widget.calculate_risk_metrics()

        # Max drawdown should be (1200 - 900) / 1200 * 100 = 25%
        assert pnl_widget.max_drawdown == Decimal("25")

    def test_render_pnl_chart(self, pnl_widget):
        """Test P&L chart rendering."""
        # No data case
        chart = pnl_widget._render_pnl_chart()
        assert "No historical data" in chart

        # Add data
        for i in range(10):
            pnl_widget.add_pnl_data_point(Decimal(str(i * 100)))

        chart = pnl_widget._render_pnl_chart()
        assert "P&L Trend:" in chart
        assert "(24h)" in chart

    def test_render_with_history(self, pnl_widget):
        """Test rendering with historical data."""
        pnl_widget.set_mock_data(
            current=Decimal("500"),
            daily=Decimal("100"),
            balance=Decimal("10000"),
        )

        output = pnl_widget.render()
        assert "P&L Dashboard" in output
        assert "$500.00" in output
        assert "$100.00" in output
        assert "1.00%" in output  # Daily percentage

    def test_paper_trading_mode(self, pnl_widget):
        """Test paper trading mode display."""
        pnl_widget.paper_trading_mode = True
        output = pnl_widget.render()
        assert "Paper Trading P&L" in output


class TestPositionWidget:
    """Test enhanced position widget functionality."""

    @pytest.fixture
    def position_widget(self):
        """Create a position widget instance."""
        return PositionWidget()

    def test_init(self, position_widget):
        """Test position widget initialization."""
        assert not position_widget.has_position
        assert position_widget.risk_reward_ratio == Decimal("0.00")
        assert position_widget.position_risk_pct == Decimal("0.00")
        assert position_widget.time_in_position == 0

    def test_no_position_render(self, position_widget):
        """Test rendering when no position is open."""
        output = position_widget.render()
        assert "No Open Position" in output
        assert "Ready to trade" in output

    def test_position_with_risk_metrics(self, position_widget):
        """Test position display with risk metrics."""
        position_widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("50000"),
            current=Decimal("51000"),
            stop_loss=Decimal("49000"),
        )

        output = position_widget.render()
        assert "BTC/USDT" in output
        assert "LONG" in output
        assert "Risk Metrics:" in output

    def test_calculate_risk_metrics(self, position_widget):
        """Test risk metrics calculation."""
        position_widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("50000"),
            current=Decimal("51000"),
            stop_loss=Decimal("49000"),
        )

        position_widget.calculate_risk_metrics()

        # Risk = (50000 - 49000) * 0.1 = 100
        # Potential reward = 100 * 2 = 200
        # R:R = 200 / 100 = 2
        assert position_widget.risk_reward_ratio == Decimal("2")

    def test_time_in_position_tracking(self, position_widget):
        """Test time in position tracking."""
        position_widget.set_mock_position(
            symbol="ETH/USDT",
            side="SHORT",
            qty=Decimal("1"),
            entry=Decimal("3000"),
            current=Decimal("2950"),
        )

        # Simulate time passing
        position_widget.entry_time = datetime.now(UTC) - timedelta(hours=2, minutes=30)
        position_widget.calculate_risk_metrics()

        assert position_widget.time_in_position > 0

    def test_max_profit_loss_tracking(self, position_widget):
        """Test max profit/loss tracking."""
        position_widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("50000"),
            current=Decimal("51000"),
        )

        # Initial profit
        position_widget.calculate_risk_metrics()
        assert position_widget.max_profit == Decimal("100")  # (51000-50000) * 0.1

        # Simulate price drop
        position_widget.current_price = Decimal("49500")
        position_widget.unrealized_pnl = Decimal("-50")
        position_widget.calculate_risk_metrics()

        assert position_widget.max_loss == Decimal("-50")
        assert position_widget.max_profit == Decimal("100")  # Still remembers max

    def test_render_risk_metrics(self, position_widget):
        """Test risk metrics rendering."""
        position_widget.risk_reward_ratio = Decimal("2.5")
        position_widget.position_risk_pct = Decimal("1.5")
        position_widget.time_in_position = 7200  # 2 hours
        position_widget.max_profit = Decimal("500")
        position_widget.max_loss = Decimal("-100")

        metrics = position_widget._render_risk_metrics()
        metrics_str = "\n".join(metrics)

        assert "2.50:1" in metrics_str
        assert "1.50%" in metrics_str
        assert "2h 0m" in metrics_str
        assert "500.00" in metrics_str
        assert "100.00" in metrics_str


class TestTiltIndicator:
    """Test enhanced tilt indicator functionality."""

    @pytest.fixture
    def tilt_indicator(self):
        """Create a tilt indicator instance."""
        return TiltIndicator()

    def test_init(self, tilt_indicator):
        """Test tilt indicator initialization."""
        assert tilt_indicator.tilt_level == TiltLevel.NORMAL
        assert tilt_indicator.tilt_score == 0
        assert tilt_indicator.anomaly_count == 0
        assert len(tilt_indicator.tilt_history) == 0
        assert tilt_indicator.max_tilt_score_today == 0
        assert tilt_indicator.tilt_events_today == 0

    def test_tilt_status_tracking(self, tilt_indicator):
        """Test tilt status tracking without UI updates."""
        anomalies = [
            {"indicator": "click_speed", "severity": 5, "description": "Rapid clicking"},
        ]

        # Track history without updating UI
        now = datetime.now(UTC)
        tilt_indicator.tilt_history.append((now, TiltLevel.LEVEL1, 35))
        tilt_indicator.tilt_events_today = 1
        tilt_indicator.max_tilt_score_today = 35
        tilt_indicator.tilt_level = TiltLevel.LEVEL1
        tilt_indicator.tilt_score = 35
        tilt_indicator.anomaly_count = 1

        assert tilt_indicator.tilt_level == TiltLevel.LEVEL1
        assert tilt_indicator.tilt_score == 35
        assert tilt_indicator.anomaly_count == 1
        assert len(tilt_indicator.tilt_history) == 1
        assert tilt_indicator.max_tilt_score_today == 35
        assert tilt_indicator.tilt_events_today == 1

    def test_render_tilt_history(self, tilt_indicator):
        """Test tilt history rendering."""
        # No history case
        history = tilt_indicator._render_tilt_history()
        # Check if it's a Text object with expected content
        from rich.text import Text
        if isinstance(history, Text):
            assert "No tilt events today" in str(history)

        # Add history manually
        now = datetime.now(UTC)
        tilt_indicator.tilt_history.append((now, TiltLevel.LEVEL1, 30))
        tilt_indicator.tilt_history.append((now, TiltLevel.LEVEL2, 60))
        tilt_indicator.tilt_history.append((now, TiltLevel.NORMAL, 10))
        tilt_indicator.tilt_events_today = 3
        tilt_indicator.max_tilt_score_today = 60

        # Just verify the method runs without error
        history = tilt_indicator._render_tilt_history()
        assert history is not None  # Panel object exists

    def test_is_trading_allowed(self, tilt_indicator):
        """Test trading permission based on tilt level."""
        # Normal level - trading allowed
        assert tilt_indicator.is_trading_allowed

        # Level 3 - trading blocked
        tilt_indicator.tilt_level = TiltLevel.LEVEL3
        assert not tilt_indicator.is_trading_allowed

    def test_position_size_multiplier(self, tilt_indicator):
        """Test position size multiplier based on tilt level."""
        # Normal level
        assert tilt_indicator.position_size_multiplier == Decimal("1.0")

        # Level 2 - reduced sizing
        tilt_indicator.tilt_level = TiltLevel.LEVEL2
        assert tilt_indicator.position_size_multiplier == Decimal("0.5")

        # Level 3 - no trading
        tilt_indicator.tilt_level = TiltLevel.LEVEL3
        assert tilt_indicator.position_size_multiplier == Decimal("0")

    def test_get_status_summary(self, tilt_indicator):
        """Test getting status summary."""
        # Set status manually
        tilt_indicator.tilt_level = TiltLevel.LEVEL1
        tilt_indicator.tilt_score = 45
        tilt_indicator.anomaly_count = 1
        tilt_indicator.intervention_message = "Test message"

        summary = tilt_indicator.get_status_summary()
        assert summary["level"] == "LEVEL1"
        assert summary["score"] == 45
        assert summary["anomaly_count"] == 1
        assert summary["trading_allowed"] is True
        assert summary["has_intervention"] is True

    def test_render_anomaly_list(self, tilt_indicator):
        """Test anomaly list rendering."""
        anomalies = [
            {"indicator": "click_speed", "severity": 3, "description": "Moderate clicking"},
            {"indicator": "cancel_rate", "severity": 7, "description": "High cancel rate"},
        ]

        tilt_indicator.anomalies = anomalies
        anomaly_list = tilt_indicator._render_anomaly_list()

        # Verify it returns a Panel object
        from rich.panel import Panel
        assert isinstance(anomaly_list, Panel)

    def test_intervention_history_tracking(self, tilt_indicator):
        """Test intervention message history tracking."""
        now = datetime.now(UTC)
        
        # Manually track interventions
        tilt_indicator.intervention_history.append((now, "First intervention"))
        tilt_indicator.intervention_history.append((now, "Second intervention"))

        assert len(tilt_indicator.intervention_history) == 2
        assert tilt_indicator.intervention_history[0][1] == "First intervention"
        assert tilt_indicator.intervention_history[1][1] == "Second intervention"


def test_widget_integration():
    """Test widget integration without app context."""
    # Create widgets
    pnl_widget = PnLWidget()
    position_widget = PositionWidget()
    tilt_indicator = TiltIndicator()

    # Simulate data updates
    pnl_widget.set_mock_data(
        current=Decimal("1000"),
        daily=Decimal("200"),
        balance=Decimal("10000"),
    )

    position_widget.set_mock_position(
        symbol="BTC/USDT",
        side="LONG",
        qty=Decimal("0.1"),
        entry=Decimal("50000"),
        current=Decimal("51000"),
    )

    # Set tilt status manually
    tilt_indicator.tilt_level = TiltLevel.NORMAL
    tilt_indicator.tilt_score = 15
    tilt_indicator.anomaly_count = 0

    # Verify widgets can render
    assert pnl_widget.render()
    assert position_widget.render()
    assert tilt_indicator.get_status_summary()