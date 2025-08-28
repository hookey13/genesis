"""
Unit tests for behavioral metrics tracking.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from genesis.analytics.behavioral_metrics import (
    ClickLatencyTracker,
    InactivityTracker,
    OrderModificationTracker,
    SessionAnalyzer,
)
from genesis.core.exceptions import ValidationError


class TestClickLatencyTracker:
    """Tests for click-to-trade latency tracking."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ClickLatencyTracker(window_size=50, baseline_window=100)
        assert tracker.window_size == 50
        assert tracker.baseline_window == 100
        assert len(tracker.recent_latencies) == 0
        assert tracker.baseline_mean is None

    def test_track_click_latency_valid(self):
        """Test tracking valid latency values."""
        tracker = ClickLatencyTracker()

        tracker.track_click_latency("market_buy", 150)
        tracker.track_click_latency("limit_sell", 200)

        assert len(tracker.recent_latencies) == 2
        assert tracker.recent_latencies[0] == 150
        assert tracker.recent_latencies[1] == 200
        assert "market_buy" in tracker.action_latencies
        assert "limit_sell" in tracker.action_latencies

    def test_track_click_latency_negative_raises(self):
        """Test that negative latency raises ValidationError."""
        tracker = ClickLatencyTracker()

        with pytest.raises(ValidationError, match="Latency cannot be negative"):
            tracker.track_click_latency("market_buy", -100)

    def test_rolling_window_limit(self):
        """Test that rolling window respects max size."""
        tracker = ClickLatencyTracker(window_size=5)

        for i in range(10):
            tracker.track_click_latency("test", i * 100)

        # Should only keep last 5
        assert len(tracker.recent_latencies) == 5
        assert list(tracker.recent_latencies) == [500, 600, 700, 800, 900]

    def test_baseline_calculation(self):
        """Test baseline mean and standard deviation calculation."""
        tracker = ClickLatencyTracker(baseline_window=10)

        # Add enough samples to trigger baseline calculation
        for i in range(6):
            tracker.track_click_latency("test", 100 + i * 10)

        assert tracker.baseline_mean is not None
        assert tracker.baseline_std is not None
        assert tracker.baseline_mean == pytest.approx(125.0, rel=0.01)

    def test_get_metrics_empty(self):
        """Test metrics when no data available."""
        tracker = ClickLatencyTracker()
        metrics = tracker.get_metrics()

        assert metrics.current == 0
        assert metrics.moving_average == 0.0
        assert metrics.std_deviation == 0.0
        assert metrics.percentile_95 == 0.0
        assert metrics.samples == 0

    def test_get_metrics_with_data(self):
        """Test metrics calculation with data."""
        tracker = ClickLatencyTracker()

        latencies = [100, 150, 200, 250, 300]
        for latency in latencies:
            tracker.track_click_latency("test", latency)

        metrics = tracker.get_metrics()

        assert metrics.current == 300
        assert metrics.moving_average == 200.0
        assert metrics.std_deviation > 0
        assert metrics.percentile_95 == 300  # With 5 samples, 95th percentile is max
        assert metrics.samples == 5

    def test_get_metrics_by_action(self):
        """Test action-specific metrics."""
        tracker = ClickLatencyTracker()

        tracker.track_click_latency("buy", 100)
        tracker.track_click_latency("buy", 150)
        tracker.track_click_latency("sell", 300)

        buy_metrics = tracker.get_metrics("buy")
        sell_metrics = tracker.get_metrics("sell")

        assert buy_metrics.moving_average == 125.0
        assert sell_metrics.moving_average == 300.0

    def test_is_latency_elevated(self):
        """Test elevated latency detection."""
        tracker = ClickLatencyTracker(baseline_window=10)

        # Establish baseline with some variance
        for i in range(10):
            tracker.track_click_latency(
                "test", 100 + i % 3 * 10
            )  # 100, 110, 120 pattern

        # Normal latencies
        tracker.track_click_latency("test", 110)
        assert not tracker.is_latency_elevated(threshold_std=2.0)

        # Add elevated latencies (don't clear recent, let track_click_latency update it)
        for _ in range(10):  # Add enough to dominate the moving average
            tracker.track_click_latency("test", 300)

        # Should detect elevation (300 is much higher than 100-120 baseline)
        assert tracker.is_latency_elevated(threshold_std=2.0)


class TestOrderModificationTracker:
    """Tests for order modification tracking."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = OrderModificationTracker()
        assert len(tracker.modifications) == 0
        assert "5min" in tracker.time_windows
        assert "30min" in tracker.time_windows
        assert "1hr" in tracker.time_windows

    def test_track_order_modification(self):
        """Test tracking order modifications."""
        tracker = OrderModificationTracker()

        tracker.track_order_modification("order1", "price")
        tracker.track_order_modification("order2", "quantity")

        assert len(tracker.modifications) == 2
        assert tracker.modifications[0]["order_id"] == "order1"
        assert tracker.modifications[0]["modification_type"] == "price"

    def test_modification_cleanup(self):
        """Test old modification cleanup."""
        tracker = OrderModificationTracker()

        # Add old modification
        old_mod = {
            "order_id": "old",
            "modification_type": "cancel",
            "timestamp": datetime.utcnow() - timedelta(hours=3),
        }
        tracker.modifications.append(old_mod)

        # Add new modification (triggers cleanup)
        tracker.track_order_modification("new", "price")

        # Old modification should be removed
        assert len(tracker.modifications) == 1
        assert tracker.modifications[0]["order_id"] == "new"

    def test_calculate_modification_frequency(self):
        """Test frequency calculation."""
        tracker = OrderModificationTracker()

        # Add modifications at different times
        now = datetime.utcnow()
        for i in range(5):
            mod = {
                "order_id": f"order{i}",
                "modification_type": "price",
                "timestamp": now - timedelta(minutes=i),
            }
            tracker.modifications.append(mod)

        frequencies = tracker.calculate_modification_frequency()

        assert "5min" in frequencies
        assert frequencies["5min"] == 1.0  # 5 mods in 5 min = 1 per min

    def test_update_baseline(self):
        """Test baseline rate update."""
        tracker = OrderModificationTracker()

        tracker.update_baseline("5min", 0.5)
        assert tracker.baseline_rates["5min"] == 0.5

    @patch("genesis.analytics.behavioral_metrics.logger")
    def test_excessive_modification_warning(self, mock_logger):
        """Test warning for excessive modifications."""
        tracker = OrderModificationTracker()
        tracker.update_baseline("5min", 0.5)

        # Add many modifications quickly
        for i in range(10):
            tracker.track_order_modification(f"order{i}", "price")

        # Should have logged warning
        mock_logger.warning.assert_called()


class TestInactivityTracker:
    """Tests for inactivity tracking."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = InactivityTracker(inactivity_threshold_seconds=60)
        assert tracker.inactivity_threshold == timedelta(seconds=60)
        assert len(tracker.inactivity_periods) == 0
        assert tracker.current_inactive_start is None

    def test_track_activity(self):
        """Test activity tracking."""
        tracker = InactivityTracker()

        # Set up inactive period
        tracker.current_inactive_start = datetime.utcnow() - timedelta(minutes=2)

        # Track activity
        tracker.track_activity()

        # Should have recorded inactivity period
        assert len(tracker.inactivity_periods) == 1
        assert tracker.current_inactive_start is None

    def test_track_inactivity_period_too_short(self):
        """Test that short periods are not tracked."""
        tracker = InactivityTracker(inactivity_threshold_seconds=30)

        start = datetime.utcnow()
        end = start + timedelta(seconds=10)

        tracker.track_inactivity_period(start, end)

        # Too short, should not be recorded
        assert len(tracker.inactivity_periods) == 0

    def test_track_inactivity_period_valid(self):
        """Test tracking valid inactivity period."""
        tracker = InactivityTracker(inactivity_threshold_seconds=30)

        start = datetime.utcnow()
        end = start + timedelta(seconds=60)

        tracker.track_inactivity_period(start, end)

        assert len(tracker.inactivity_periods) == 1
        assert tracker.inactivity_periods[0]["duration_seconds"] == 60

    def test_check_inactivity_active(self):
        """Test checking inactivity when user is active."""
        tracker = InactivityTracker(inactivity_threshold_seconds=30)
        tracker.last_activity = datetime.utcnow()

        assert tracker.check_inactivity() is None

    def test_check_inactivity_inactive(self):
        """Test checking inactivity when user is inactive."""
        tracker = InactivityTracker(inactivity_threshold_seconds=30)
        tracker.last_activity = datetime.utcnow() - timedelta(seconds=45)

        inactivity = tracker.check_inactivity()
        assert inactivity is not None
        assert inactivity >= 45

    def test_get_inactivity_stats_empty(self):
        """Test statistics when no inactivity periods."""
        tracker = InactivityTracker()
        stats = tracker.get_inactivity_stats(hours=1)

        assert stats["total_periods"] == 0
        assert stats["total_duration_seconds"] == 0
        assert stats["average_duration_seconds"] == 0

    def test_get_inactivity_stats_with_data(self):
        """Test statistics with inactivity data."""
        tracker = InactivityTracker()

        # Add inactivity periods
        now = datetime.utcnow()
        periods = [
            {
                "start": now - timedelta(minutes=30),
                "end": now - timedelta(minutes=25),
                "duration_seconds": 300,
            },
            {
                "start": now - timedelta(minutes=20),
                "end": now - timedelta(minutes=15),
                "duration_seconds": 300,
            },
        ]
        tracker.inactivity_periods = periods

        stats = tracker.get_inactivity_stats(hours=1)

        assert stats["total_periods"] == 2
        assert stats["total_duration_seconds"] == 600
        assert stats["average_duration_seconds"] == 300
        assert stats["longest_duration_seconds"] == 300


class TestSessionAnalyzer:
    """Tests for session analysis."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = SessionAnalyzer()
        assert len(analyzer.sessions) == 0
        assert analyzer.current_session_start is None

    def test_start_session(self):
        """Test starting a new session."""
        analyzer = SessionAnalyzer()

        analyzer.start_session("session1")

        assert analyzer.current_session_start is not None

    def test_end_session_without_start(self):
        """Test ending session without starting."""
        analyzer = SessionAnalyzer()

        result = analyzer.end_session("session1")

        assert result is None

    def test_end_session_valid(self):
        """Test ending an active session."""
        analyzer = SessionAnalyzer()

        analyzer.start_session("session1")
        # Simulate some time passing
        analyzer.current_session_start = datetime.utcnow() - timedelta(hours=2)

        metrics = analyzer.end_session("session1")

        assert metrics is not None
        assert metrics["session_id"] == "session1"
        assert metrics["duration_hours"] >= 2.0
        assert len(analyzer.sessions) == 1
        assert analyzer.current_session_start is None

    @patch("genesis.analytics.behavioral_metrics.logger")
    def test_long_session_warning(self, mock_logger):
        """Test warning for long sessions."""
        analyzer = SessionAnalyzer()

        analyzer.start_session("session1")
        # Simulate 5 hour session
        analyzer.current_session_start = datetime.utcnow() - timedelta(hours=5)

        analyzer.end_session("session1")

        # Should have logged warning
        mock_logger.warning.assert_called_with(
            "long_session_detected",
            session_id="session1",
            duration_hours=pytest.approx(5.0, rel=0.01),
        )

    def test_analyze_session_duration(self):
        """Test analyzing specific session."""
        analyzer = SessionAnalyzer()

        # Add sessions
        session1 = {
            "session_id": "session1",
            "start": datetime.utcnow() - timedelta(hours=2),
            "end": datetime.utcnow(),
            "duration_seconds": 7200,
            "duration_hours": 2.0,
        }
        session2 = {
            "session_id": "session2",
            "start": datetime.utcnow() - timedelta(hours=4),
            "end": datetime.utcnow() - timedelta(hours=2),
            "duration_seconds": 7200,
            "duration_hours": 2.0,
        }
        analyzer.sessions = [session1, session2]

        metrics = analyzer.analyze_session_duration("session1")

        assert metrics is not None
        assert metrics["session_id"] == "session1"
        assert metrics["average_duration_hours"] == 2.0
        assert metrics["relative_duration"] == 1.0  # Same as average

    def test_get_session_statistics_empty(self):
        """Test statistics with no sessions."""
        analyzer = SessionAnalyzer()
        stats = analyzer.get_session_statistics()

        assert stats["total_sessions"] == 0
        assert stats["average_duration_hours"] == 0

    def test_get_session_statistics_with_data(self):
        """Test statistics with session data."""
        analyzer = SessionAnalyzer()

        # Add sessions with different durations
        sessions = [
            {"session_id": "s1", "duration_seconds": 3600},  # 1 hour
            {"session_id": "s2", "duration_seconds": 7200},  # 2 hours
            {"session_id": "s3", "duration_seconds": 10800},  # 3 hours
        ]
        analyzer.sessions = sessions

        stats = analyzer.get_session_statistics()

        assert stats["total_sessions"] == 3
        assert stats["average_duration_hours"] == 2.0
        assert stats["longest_session_hours"] == 3.0
        assert stats["shortest_session_hours"] == 1.0
