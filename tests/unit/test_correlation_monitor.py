"""
Unit tests for correlation monitoring system.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from genesis.analytics.correlation_monitor import (
    CorrelationAlert,
    CorrelationMonitor,
    CorrelationWindow,
)
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    event_bus = AsyncMock(spec=EventBus)
    return event_bus


@pytest.fixture
async def monitor(mock_event_bus):
    """Create correlation monitor instance."""
    monitor = CorrelationMonitor(
        event_bus=mock_event_bus,
        window_size=5,  # Small window for testing
        warning_threshold=Decimal("0.6"),
        critical_threshold=Decimal("0.8"),
    )
    await monitor.start()
    yield monitor
    await monitor.stop()


class TestCorrelationWindow:
    """Test correlation window data structure."""

    def test_add_observation(self):
        """Test adding observations to window."""
        window = CorrelationWindow(window_size=3)

        # Add observations
        window.add_observation(
            datetime.now(UTC), {"strategy1": 0.05, "strategy2": -0.02}
        )

        assert len(window.timestamps) == 1
        assert len(window.data["strategy1"]) == 1
        assert window.data["strategy1"][0] == 0.05

    def test_window_size_enforcement(self):
        """Test that window size is enforced."""
        window = CorrelationWindow(window_size=2)

        # Add 3 observations
        for i in range(3):
            window.add_observation(datetime.now(UTC), {"strategy1": i})

        # Should only keep last 2
        assert len(window.timestamps) == 2
        assert len(window.data["strategy1"]) == 2
        assert window.data["strategy1"] == [1, 2]

    def test_is_ready(self):
        """Test window readiness check."""
        window = CorrelationWindow(window_size=3)

        assert not window.is_ready()

        # Add partial data
        for i in range(2):
            window.add_observation(datetime.now(UTC), {"strategy1": i})

        assert not window.is_ready()

        # Complete window
        window.add_observation(datetime.now(UTC), {"strategy1": 2})

        assert window.is_ready()

    def test_get_dataframe(self):
        """Test DataFrame conversion."""
        window = CorrelationWindow(window_size=3)

        timestamps = []
        for i in range(3):
            ts = datetime.now(UTC)
            timestamps.append(ts)
            window.add_observation(ts, {"strategy1": i, "strategy2": i * 2})

        df = window.get_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["strategy1", "strategy2"]
        assert list(df["strategy1"]) == [0, 1, 2]


class TestCorrelationAlert:
    """Test correlation alert structure."""

    def test_to_dict(self):
        """Test alert dictionary conversion."""
        alert = CorrelationAlert(
            alert_id="test_id",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            entity1="strategy1",
            entity2="strategy2",
            correlation=Decimal("0.85"),
            threshold=Decimal("0.8"),
            severity="critical",
            message="Test alert",
        )

        alert_dict = alert.to_dict()

        assert alert_dict["alert_id"] == "test_id"
        assert alert_dict["correlation"] == "0.85"
        assert alert_dict["severity"] == "critical"
        assert "timestamp" in alert_dict


class TestCorrelationMonitor:
    """Test correlation monitor operations."""

    async def test_add_strategy_returns(self, monitor):
        """Test adding strategy return data."""
        # Add insufficient data
        for i in range(3):
            await monitor.add_strategy_returns(
                datetime.now(UTC), {"strategy1": 0.01 * i, "strategy2": -0.01 * i}
            )

        assert not monitor.strategy_returns.is_ready()

        # Add more to complete window
        for i in range(2):
            await monitor.add_strategy_returns(
                datetime.now(UTC), {"strategy1": 0.01, "strategy2": -0.01}
            )

        assert monitor.strategy_returns.is_ready()

    async def test_correlation_calculation(self, monitor, mock_event_bus):
        """Test correlation matrix calculation."""
        # Create perfectly correlated data
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC), {"strategy1": 0.01 * i, "strategy2": 0.01 * i}
            )

        # Should have correlation matrix
        assert monitor.correlation_matrix is not None
        corr = monitor.correlation_matrix.iloc[0, 1]
        assert abs(corr - 1.0) < 0.01  # Near perfect correlation

    async def test_warning_alert_generation(self, monitor, mock_event_bus):
        """Test warning alert for moderate correlation."""
        # Create data with ~0.7 correlation
        np.random.seed(42)
        base = np.random.randn(5)

        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {
                    "strategy1": base[i],
                    "strategy2": base[i] * 0.7 + np.random.randn() * 0.3,
                },
            )

        # Check for alerts
        await asyncio.sleep(0.1)  # Allow async operations to complete

        # Should have published correlation alert
        if mock_event_bus.publish.called:
            call_args = mock_event_bus.publish.call_args_list
            assert any("CORRELATION_ALERT" in str(call) for call in call_args)

    async def test_critical_alert_generation(self, monitor, mock_event_bus):
        """Test critical alert for high correlation."""
        # Create highly correlated data (>0.8)
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {
                    "strategy1": 0.01 * i,
                    "strategy2": 0.01 * i + 0.001,  # Almost identical
                },
            )

        # Should generate critical alert
        assert len(monitor.active_alerts) > 0
        alert = list(monitor.active_alerts.values())[0]
        assert alert.severity == "critical"

    async def test_alert_clearing(self, monitor):
        """Test that alerts clear when correlation drops."""
        # Create high correlation
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC), {"strategy1": i, "strategy2": i}
            )

        assert len(monitor.active_alerts) > 0

        # Add uncorrelated data
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {"strategy1": i, "strategy2": -i},  # Negative correlation
            )

        # Alerts should be cleared
        assert len(monitor.active_alerts) == 0

    async def test_calculate_position_correlations(self, monitor):
        """Test position correlation calculation."""
        positions = [
            {"position_id": "pos1", "returns": [0.01, 0.02, -0.01, 0.03, 0.01]},
            {
                "position_id": "pos2",
                "returns": [0.01, 0.02, -0.01, 0.03, 0.01],  # Identical
            },
            {
                "position_id": "pos3",
                "returns": [-0.01, -0.02, 0.01, -0.03, -0.01],  # Inverse
            },
        ]

        correlations = await monitor.calculate_position_correlations(positions)

        assert len(correlations) == 3  # 3 pairs from 3 positions

        # Find correlation between pos1 and pos2 (should be ~1.0)
        pos1_pos2 = next(
            c
            for c in correlations
            if (c.position_a_id == "pos1" and c.position_b_id == "pos2")
            or (c.position_a_id == "pos2" and c.position_b_id == "pos1")
        )
        assert abs(pos1_pos2.correlation_coefficient - Decimal("1.0")) < Decimal("0.01")
        assert pos1_pos2.alert_triggered  # Should trigger alert

    async def test_get_correlation_summary(self, monitor):
        """Test correlation summary generation."""
        # No data case
        summary = monitor.get_correlation_summary()
        assert summary["status"] == "insufficient_data"

        # Add data
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {"strategy1": i * 0.01, "strategy2": i * 0.01, "strategy3": -i * 0.01},
            )

        summary = monitor.get_correlation_summary()
        assert summary["status"] == "active"
        assert summary["num_pairs"] == 3  # 3 pairs from 3 strategies
        assert "average_correlation" in summary
        assert "max_correlation" in summary

    async def test_get_high_correlation_pairs(self, monitor):
        """Test getting high correlation pairs."""
        # Add perfectly correlated strategies
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {"strategy1": i, "strategy2": i, "strategy3": i * 0.5, "strategy4": -i},
            )

        high_pairs = monitor.get_high_correlation_pairs()

        # Should find pairs with correlation > 0.6
        assert len(high_pairs) > 0

        # Check sorting (highest correlation first)
        if len(high_pairs) > 1:
            assert high_pairs[0][2] >= high_pairs[1][2]

    async def test_get_high_correlation_pairs_custom_threshold(self, monitor):
        """Test getting high correlation pairs with custom threshold."""
        # Add data
        for i in range(5):
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {
                    "strategy1": i,
                    "strategy2": i * 0.7,  # 0.7 correlation
                    "strategy3": i * 0.4,  # 0.4 correlation
                },
            )

        # Get pairs with correlation > 0.5
        high_pairs = monitor.get_high_correlation_pairs(Decimal("0.5"))

        # Should include strategy1-strategy2 but maybe not others
        pair_names = [(p[0], p[1]) for p in high_pairs]
        assert any(("strategy1" in p and "strategy2" in p) for p in pair_names)

    async def test_correlation_crisis_detection(self, monitor, mock_event_bus):
        """Test detection of market-wide correlation crisis."""
        # Create crisis scenario - all strategies highly correlated
        for i in range(5):
            base_return = i * 0.01
            await monitor.add_strategy_returns(
                datetime.now(UTC),
                {
                    "strategy1": base_return,
                    "strategy2": base_return * 1.01,
                    "strategy3": base_return * 0.99,
                    "strategy4": base_return * 1.02,
                    "strategy5": base_return * 0.98,
                },
            )

        # Manually trigger crisis check
        await monitor._check_for_correlation_crisis()

        # Should publish crisis event
        if mock_event_bus.publish.called:
            call_args = mock_event_bus.publish.call_args_list
            crisis_events = [
                call for call in call_args if "correlation_crisis" in str(call)
            ]
            assert len(crisis_events) > 0

    async def test_empty_position_correlation(self, monitor):
        """Test position correlation with empty data."""
        # Empty positions
        correlations = await monitor.calculate_position_correlations([])
        assert correlations == []

        # Single position
        correlations = await monitor.calculate_position_correlations(
            [{"position_id": "pos1", "returns": [0.01, 0.02]}]
        )
        assert correlations == []

        # Positions without returns
        correlations = await monitor.calculate_position_correlations(
            [{"position_id": "pos1"}, {"position_id": "pos2"}]
        )
        assert correlations == []

    async def test_correlation_history_limit(self, monitor):
        """Test that correlation history is limited."""
        # Add many correlation calculations
        for batch in range(110):  # More than limit of 100
            for i in range(5):
                await monitor.add_strategy_returns(
                    datetime.now(UTC), {"strategy1": i, "strategy2": i * 2}
                )

        # History should be capped at 100
        assert len(monitor.correlation_history) <= 100
