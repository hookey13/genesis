"""
Integration tests for behavioral reporting workflow.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from genesis.analytics.behavior_correlation import BehaviorPnLCorrelator
from genesis.analytics.behavioral_metrics import (
    ClickLatencyTracker,
    InactivityTracker,
    OrderModificationTracker,
    SessionAnalyzer,
)
from genesis.analytics.behavioral_reports import WeeklyBehavioralReportGenerator
from genesis.analytics.config_tracker import ConfigurationChangeTracker
from genesis.tilt.indicators.focus_patterns import FocusPatternDetector


class TestBehavioralReportingWorkflow:
    """Integration tests for complete behavioral reporting workflow."""

    @pytest.fixture
    def setup_components(self):
        """Set up all behavioral tracking components."""
        components = {
            "click_tracker": ClickLatencyTracker(),
            "modification_tracker": OrderModificationTracker(),
            "focus_detector": FocusPatternDetector(),
            "inactivity_tracker": InactivityTracker(),
            "session_analyzer": SessionAnalyzer(),
            "config_tracker": ConfigurationChangeTracker(),
            "correlator": BehaviorPnLCorrelator(),
        }
        return components

    @pytest.fixture
    def report_generator(self, setup_components):
        """Create report generator with all components."""
        return WeeklyBehavioralReportGenerator(**setup_components)

    def test_full_report_generation(self, report_generator):
        """Test generating a complete weekly report."""
        # Simulate a week of data
        base_time = datetime.utcnow()
        week_start = base_time - timedelta(days=7)

        # Add click latency data
        for i in range(100):
            report_generator.click_tracker.track_click_latency(
                "market_buy", 150 + i % 50  # Varying latency
            )

        # Add order modifications
        for i in range(20):
            report_generator.modification_tracker.track_order_modification(
                f"order_{i}", "price" if i % 2 == 0 else "quantity"
            )

        # Add focus events
        for i in range(30):
            report_generator.focus_detector.track_window_focus(
                window_active=i % 2 == 0, duration_ms=5000 + i * 100
            )

        # Add inactivity periods
        for i in range(5):
            start = base_time - timedelta(hours=i * 12)
            end = start + timedelta(minutes=45)
            report_generator.inactivity_tracker.track_inactivity_period(start, end)

        # Add sessions
        for i in range(3):
            session_id = f"session_{i}"
            report_generator.session_analyzer.start_session(session_id)
            report_generator.session_analyzer.current_session_start = (
                base_time - timedelta(hours=i * 24 + 2)
            )
            report_generator.session_analyzer.end_session(session_id)

        # Add config changes
        for i in range(10):
            report_generator.config_tracker.track_config_change(
                "position_size", 1000 + i * 100, 1000 + (i + 1) * 100
            )

        # Add behavior and P&L data for correlation
        for i in range(50):
            time = base_time - timedelta(hours=i)
            report_generator.correlator.add_behavior_data(
                "click_latency", time, 150 + i * 5
            )
            report_generator.correlator.add_pnl_data(time, Decimal(str(100 - i * 2)))

        # Generate report
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Verify report structure
        assert report.profile_id == "test_trader"
        assert report.week_start == week_start
        assert report.week_end == week_start + timedelta(days=7)
        assert report.generation_time is not None

        # Verify metrics collected
        assert report.metrics is not None
        assert report.metrics.average_click_latency_ms > 0
        assert report.metrics.modification_frequency >= 0
        assert report.metrics.average_distraction_score >= 0
        assert report.metrics.total_sessions == 3
        assert report.metrics.config_changes == 10

        # Verify insights generated
        assert len(report.insights) > 0
        assert all(hasattr(i, "category") for i in report.insights)
        assert all(hasattr(i, "recommendation") for i in report.insights)

        # Verify recommendations
        assert len(report.top_recommendations) > 0
        assert all(isinstance(r, str) for r in report.top_recommendations)

        # Verify risk assessment
        assert report.overall_risk_level in ["low", "medium", "high"]
        assert 0 <= report.tilt_risk_score <= 100

    def test_report_export_json(self, report_generator):
        """Test exporting report as JSON."""
        # Generate a simple report
        week_start = datetime.utcnow() - timedelta(days=7)
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Export as JSON
        json_str = report_generator.export_report_json(report)

        # Verify JSON structure
        import json

        data = json.loads(json_str)

        assert data["profile_id"] == "test_trader"
        assert "metrics" in data
        assert "insights" in data
        assert "top_recommendations" in data
        assert "overall_risk_level" in data

    def test_report_export_text(self, report_generator):
        """Test exporting report as text."""
        # Generate a simple report
        week_start = datetime.utcnow() - timedelta(days=7)
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Export as text
        text = report_generator.export_report_text(report)

        # Verify text contains key sections
        assert "WEEKLY BEHAVIORAL REPORT" in text
        assert "RISK ASSESSMENT" in text
        assert "KEY METRICS" in text
        assert "TOP RECOMMENDATIONS" in text
        assert report.profile_id in text

    def test_week_over_week_comparison(self, report_generator):
        """Test comparison between weeks."""
        base_time = datetime.utcnow()

        # Generate first week report
        week1_start = base_time - timedelta(days=14)

        # Add some data for week 1
        for i in range(50):
            report_generator.click_tracker.track_click_latency("test", 100)

        report1 = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week1_start
        )

        # Clear and add different data for week 2
        report_generator.click_tracker.recent_latencies.clear()
        for i in range(50):
            report_generator.click_tracker.track_click_latency("test", 150)

        week2_start = base_time - timedelta(days=7)
        report2 = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week2_start
        )

        # Verify comparison exists
        assert report2.comparison_to_previous is not None
        assert "latency_change" in report2.comparison_to_previous

    def test_baseline_comparison(self, report_generator):
        """Test comparison against baseline."""
        # Set baseline metrics
        from genesis.analytics.behavioral_reports import WeeklyMetrics

        baseline = WeeklyMetrics(
            total_trades=100,
            total_sessions=5,
            average_session_hours=2.0,
            average_click_latency_ms=100.0,
            latency_trend="stable",
            modification_frequency=1.0,
            cancel_rate=0.1,
            average_distraction_score=20.0,
            tab_switches_per_hour=5.0,
            total_stare_time_minutes=30.0,
            longest_inactivity_minutes=5.0,
            config_changes=2,
            config_stability_score=80.0,
            strongest_loss_behavior=None,
            loss_behavior_correlation=None,
        )

        report_generator.set_baseline(baseline)

        # Generate report
        week_start = datetime.utcnow() - timedelta(days=7)

        # Add some different data
        for i in range(50):
            report_generator.click_tracker.track_click_latency("test", 150)

        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Verify baseline comparison
        assert report.comparison_to_baseline is not None
        assert "latency_deviation" in report.comparison_to_baseline

    def test_high_risk_detection(self, report_generator):
        """Test detection of high-risk behavioral patterns."""
        week_start = datetime.utcnow() - timedelta(days=7)

        # Simulate high-risk behaviors

        # High distraction
        for i in range(50):
            report_generator.focus_detector.track_window_focus(
                window_active=i % 2 == 0, duration_ms=1000  # Rapid switching
            )

        # Excessive modifications
        for i in range(100):
            report_generator.modification_tracker.track_order_modification(
                f"order_{i}", "cancel"
            )

        # Configuration instability
        for i in range(30):
            report_generator.config_tracker.track_config_change(
                f"setting_{i % 5}", i, i + 1
            )

        # Long sessions
        for i in range(3):
            report_generator.session_analyzer.sessions.append(
                {"session_id": f"session_{i}", "duration_seconds": 6 * 3600}  # 6 hours
            )

        # Generate report
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Should detect high risk
        assert report.overall_risk_level in ["medium", "high"]
        assert report.tilt_risk_score > 40

        # Should have multiple high-severity insights
        high_severity = [i for i in report.insights if i.severity == "high"]
        assert len(high_severity) > 0

    def test_positive_patterns_detection(self, report_generator):
        """Test detection of positive behavioral patterns."""
        week_start = datetime.utcnow() - timedelta(days=7)

        # Simulate good behaviors

        # Consistent latency
        for i in range(50):
            report_generator.click_tracker.track_click_latency("test", 100)

        # Low modification rate
        for i in range(5):
            report_generator.modification_tracker.track_order_modification(
                f"order_{i}", "price"
            )

        # Good focus
        report_generator.focus_detector.focus_durations.extend([120.0] * 10)

        # Stable configuration
        report_generator.config_tracker.track_config_change("theme", "dark", "light")

        # Generate report
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Should detect low risk
        assert report.overall_risk_level == "low"
        assert report.tilt_risk_score < 40

        # Should have positive or neutral insights
        high_risk = [i for i in report.insights if i.severity == "high"]
        assert len(high_risk) <= 1

    def test_correlation_insights(self, report_generator):
        """Test generation of correlation-based insights."""
        week_start = datetime.utcnow() - timedelta(days=7)
        base_time = datetime.utcnow()

        # Add strongly correlated behavior and P&L data
        for i in range(50):
            time = base_time - timedelta(hours=i)

            # High latency correlates with losses
            latency = 100 if i < 25 else 300
            pnl = 100 if i < 25 else -100

            report_generator.correlator.add_behavior_data("high_latency", time, latency)
            report_generator.correlator.add_pnl_data(time, Decimal(str(pnl)))

        # Generate report
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Should identify loss-correlated behavior
        assert report.metrics.strongest_loss_behavior is not None

        # Should have insight about correlation
        correlation_insights = [
            i for i in report.insights if "correlated" in i.description.lower()
        ]
        assert len(correlation_insights) > 0

    @pytest.mark.asyncio
    async def test_concurrent_data_collection(self, report_generator):
        """Test report generation with concurrent data updates."""
        week_start = datetime.utcnow() - timedelta(days=7)

        async def add_click_data():
            for i in range(100):
                report_generator.click_tracker.track_click_latency("test", 100 + i)
                await asyncio.sleep(0.001)

        async def add_focus_data():
            for i in range(100):
                report_generator.focus_detector.track_window_focus(i % 2 == 0, 5000)
                await asyncio.sleep(0.001)

        async def add_config_data():
            for i in range(50):
                report_generator.config_tracker.track_config_change("setting", i, i + 1)
                await asyncio.sleep(0.002)

        # Run concurrent updates
        await asyncio.gather(add_click_data(), add_focus_data(), add_config_data())

        # Generate report
        report = report_generator.generate_weekly_report(
            profile_id="test_trader", week_start=week_start
        )

        # Verify data was collected
        assert report.metrics.average_click_latency_ms > 0
        assert report.metrics.average_distraction_score >= 0
        assert report.metrics.config_changes > 0
