"""
Unit tests for configuration change tracking.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from genesis.analytics.config_tracker import ConfigurationChangeTracker


class TestConfigurationChangeTracker:
    """Tests for configuration change tracking."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ConfigurationChangeTracker(
            window_size=50, frequent_change_threshold=5
        )

        assert tracker.window_size == 50
        assert tracker.frequent_change_threshold == 5
        assert len(tracker.changes) == 0
        assert len(tracker.reverts) == 0

    def test_track_config_change(self):
        """Test tracking configuration changes."""
        tracker = ConfigurationChangeTracker()

        tracker.track_config_change("position_size", 1000, 2000)
        tracker.track_config_change("stop_loss", 0.02, 0.03)

        assert len(tracker.changes) == 2
        assert tracker.changes[0]["setting"] == "position_size"
        assert tracker.changes[0]["old_value"] == 1000
        assert tracker.changes[0]["new_value"] == 2000

        assert "position_size" in tracker.setting_history
        assert len(tracker.setting_history["position_size"]) == 1

    def test_revert_detection(self):
        """Test detection of configuration reverts."""
        tracker = ConfigurationChangeTracker()

        # Initial value
        tracker.track_config_change("leverage", 1, 2)
        # Change to new value
        tracker.track_config_change("leverage", 2, 3)
        # Revert to original
        tracker.track_config_change("leverage", 3, 1)

        # Should detect the revert
        assert len(tracker.reverts) == 1
        assert tracker.reverts[0]["setting"] == "leverage"
        assert tracker.reverts[0]["new_value"] == 1

    @patch("genesis.analytics.config_tracker.logger")
    def test_frequent_change_warning(self, mock_logger):
        """Test warning for frequent configuration changes."""
        tracker = ConfigurationChangeTracker(frequent_change_threshold=3)

        # Make multiple changes to same setting quickly
        for i in range(4):
            tracker.track_config_change("position_size", i * 1000, (i + 1) * 1000)

        # Should have logged warning
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "frequent_configuration_changes"

    def test_count_recent_changes(self):
        """Test counting recent changes to a setting."""
        tracker = ConfigurationChangeTracker()

        # Add changes at different times
        now = datetime.utcnow()
        tracker.setting_history["test_setting"] = [
            {"value": 1, "timestamp": now - timedelta(hours=2)},
            {"value": 2, "timestamp": now - timedelta(minutes=30)},
            {"value": 3, "timestamp": now - timedelta(minutes=10)},
        ]

        # Count changes in last hour
        count = tracker._count_recent_changes("test_setting", hours=1)
        assert count == 2  # Only the last 2 are within 1 hour

    def test_cleanup_old_data(self):
        """Test cleanup of old data."""
        tracker = ConfigurationChangeTracker()

        # Add old data
        old_time = datetime.utcnow() - timedelta(hours=25)
        new_time = datetime.utcnow()

        tracker.reverts = [
            {"timestamp": old_time, "setting": "old"},
            {"timestamp": new_time, "setting": "new"},
        ]

        tracker.setting_history["test"] = [
            {"value": 1, "timestamp": old_time},
            {"value": 2, "timestamp": new_time},
        ]

        # Cleanup
        tracker._cleanup_old_data()

        # Old data should be removed
        assert len(tracker.reverts) == 1
        assert tracker.reverts[0]["setting"] == "new"
        assert len(tracker.setting_history["test"]) == 1
        assert tracker.setting_history["test"][0]["value"] == 2

    def test_get_change_metrics_empty(self):
        """Test metrics when no changes recorded."""
        tracker = ConfigurationChangeTracker()

        metrics = tracker.get_change_metrics(hours=1)

        assert metrics.total_changes == 0
        assert metrics.change_frequency == 0.0
        assert len(metrics.frequent_settings) == 0
        assert metrics.revert_count == 0
        assert metrics.stability_score == 100.0

    def test_get_change_metrics_with_data(self):
        """Test metrics calculation with changes."""
        tracker = ConfigurationChangeTracker()

        # Add recent changes
        now = datetime.utcnow()
        tracker.changes = [
            {
                "setting": "position_size",
                "old_value": 1000,
                "new_value": 2000,
                "timestamp": now,
            },
            {
                "setting": "position_size",
                "old_value": 2000,
                "new_value": 3000,
                "timestamp": now,
            },
            {"setting": "leverage", "old_value": 1, "new_value": 2, "timestamp": now},
        ]

        tracker.reverts = [{"setting": "stop_loss", "timestamp": now}]

        metrics = tracker.get_change_metrics(hours=1)

        assert metrics.total_changes == 3
        assert metrics.change_frequency == 3.0  # 3 changes per hour
        assert "position_size" in metrics.frequent_settings
        assert metrics.revert_count == 1
        assert metrics.stability_score < 100.0

    def test_stability_score_calculation(self):
        """Test stability score calculation."""
        tracker = ConfigurationChangeTracker()

        # Test stable configuration
        score = tracker._calculate_stability_score(
            change_frequency=0.5, revert_count=0, total_changes=1
        )
        assert score > 80

        # Test unstable configuration
        score = tracker._calculate_stability_score(
            change_frequency=5.0, revert_count=3, total_changes=10
        )
        assert score < 50

        # Test highly unstable
        score = tracker._calculate_stability_score(
            change_frequency=10.0, revert_count=8, total_changes=20
        )
        assert score < 20

    def test_is_configuration_unstable(self):
        """Test configuration stability check."""
        tracker = ConfigurationChangeTracker()

        # Add many changes to make it unstable
        for i in range(10):
            tracker.track_config_change(f"setting_{i}", i, i + 1)

        # Add some reverts
        tracker.reverts = [{"timestamp": datetime.utcnow()} for _ in range(3)]

        assert tracker.is_configuration_unstable(threshold=70) is True

    def test_get_risky_changes(self):
        """Test identification of risky changes."""
        tracker = ConfigurationChangeTracker()

        # Add some changes
        tracker.changes = [
            {
                "setting": "position_size",
                "old_value": 1000,
                "new_value": 5000,
                "timestamp": datetime.utcnow(),
            },
            {
                "setting": "max_leverage",
                "old_value": 2,
                "new_value": 10,
                "timestamp": datetime.utcnow(),
            },
            {
                "setting": "stop_loss",
                "old_value": 0.05,
                "new_value": 0.02,
                "timestamp": datetime.utcnow(),
            },
            {
                "setting": "theme",
                "old_value": "dark",
                "new_value": "light",
                "timestamp": datetime.utcnow(),
            },
        ]

        risky = tracker.get_risky_changes()

        # Should identify the risk-increasing changes
        assert len(risky) == 3  # position_size, leverage, stop_loss

        settings = [r["setting"] for r in risky]
        assert "position_size" in settings
        assert "max_leverage" in settings
        assert "stop_loss" in settings
        assert "theme" not in settings

    def test_is_risk_increase(self):
        """Test risk increase detection."""
        tracker = ConfigurationChangeTracker()

        # Test position size increase (risky)
        assert tracker._is_risk_increase("position_size", 1000, 2000) is True
        assert tracker._is_risk_increase("position_size", 2000, 1000) is False

        # Test leverage increase (risky)
        assert tracker._is_risk_increase("max_leverage", 2, 5) is True
        assert tracker._is_risk_increase("max_leverage", 5, 2) is False

        # Test stop loss decrease (risky)
        assert tracker._is_risk_increase("stop_loss", 0.05, 0.02) is True
        assert tracker._is_risk_increase("stop_loss", 0.02, 0.05) is False

        # Test non-numeric values
        assert tracker._is_risk_increase("some_setting", "low", "high") is True

    def test_get_analysis_summary(self):
        """Test comprehensive analysis summary."""
        tracker = ConfigurationChangeTracker()

        # Add various changes
        now = datetime.utcnow()
        tracker.changes = [
            {
                "setting": "position_size",
                "old_value": 1000,
                "new_value": 2000,
                "timestamp": now,
            },
            {"setting": "leverage", "old_value": 1, "new_value": 2, "timestamp": now},
        ]

        summary = tracker.get_analysis_summary()

        assert "state" in summary
        assert "stability_score" in summary
        assert "changes_last_hour" in summary
        assert "changes_last_24h" in summary
        assert "recommendation" in summary
        assert summary["state"] in [
            "stable",
            "somewhat_unstable",
            "unstable",
            "highly_unstable",
        ]

    def test_recommendation_generation(self):
        """Test recommendation based on configuration patterns."""
        tracker = ConfigurationChangeTracker()

        # Import ConfigChangeMetrics for testing
        from genesis.analytics.config_tracker import ConfigChangeMetrics

        # Test high instability recommendation
        metrics = ConfigChangeMetrics(
            total_changes=10,
            change_frequency=10.0,
            frequent_settings=["position_size"],
            revert_count=1,
            stability_score=25.0,
        )
        recommendation = tracker._get_recommendation(metrics, [])
        assert "Stop changing settings" in recommendation

        # Test risky changes recommendation
        metrics = ConfigChangeMetrics(
            total_changes=3,
            change_frequency=3.0,
            frequent_settings=[],
            revert_count=0,
            stability_score=60.0,
        )
        risky = [{"setting": "leverage"}]
        recommendation = tracker._get_recommendation(metrics, risky)
        assert "Review risk settings" in recommendation

        # Test revert recommendation
        metrics = ConfigChangeMetrics(
            total_changes=5,
            change_frequency=2.0,
            frequent_settings=[],
            revert_count=3,
            stability_score=50.0,
        )
        recommendation = tracker._get_recommendation(metrics, [])
        assert "Commit to settings" in recommendation

        # Test stable configuration
        metrics = ConfigChangeMetrics(
            total_changes=1,
            change_frequency=0.5,
            frequent_settings=[],
            revert_count=0,
            stability_score=90.0,
        )
        recommendation = tracker._get_recommendation(metrics, [])
        assert "stable" in recommendation
