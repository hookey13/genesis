"""
Weekly behavioral report generation for trader self-awareness.

Generates comprehensive reports analyzing behavioral patterns
and their impact on trading performance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from genesis.analytics.behavior_correlation import BehaviorPnLCorrelator
from genesis.analytics.behavioral_metrics import (
    ClickLatencyTracker,
    InactivityTracker,
    OrderModificationTracker,
    SessionAnalyzer,
)
from genesis.analytics.config_tracker import ConfigurationChangeTracker
from genesis.tilt.indicators.focus_patterns import FocusPatternDetector

logger = structlog.get_logger(__name__)


@dataclass
class WeeklyMetrics:
    """Aggregated weekly behavioral metrics."""

    # Trading activity
    total_trades: int
    total_sessions: int
    average_session_hours: float

    # Click patterns
    average_click_latency_ms: float
    latency_trend: str  # 'improving', 'stable', 'worsening'

    # Order modifications
    modification_frequency: float
    cancel_rate: float

    # Focus patterns
    average_distraction_score: float
    tab_switches_per_hour: float

    # Inactivity
    total_stare_time_minutes: float
    longest_inactivity_minutes: float

    # Configuration
    config_changes: int
    config_stability_score: float

    # P&L correlation
    strongest_loss_behavior: Optional[str]
    loss_behavior_correlation: Optional[float]


@dataclass
class BehavioralInsight:
    """Actionable behavioral insight."""

    category: str  # 'risk', 'opportunity', 'trend'
    severity: str  # 'high', 'medium', 'low'
    description: str
    recommendation: str
    supporting_data: dict[str, Any]


@dataclass
class WeeklyBehavioralReport:
    """Comprehensive weekly behavioral analysis report."""

    profile_id: str
    week_start: datetime
    week_end: datetime
    generation_time: datetime

    # Metrics
    metrics: WeeklyMetrics

    # Week-over-week comparison
    comparison_to_previous: Optional[dict[str, float]]
    comparison_to_baseline: Optional[dict[str, float]]

    # Insights
    insights: list[BehavioralInsight]

    # Recommendations
    top_recommendations: list[str]

    # Risk assessment
    overall_risk_level: str  # 'low', 'medium', 'high'
    tilt_risk_score: float  # 0-100

    # Raw data for export
    raw_data: Optional[dict[str, Any]]


class WeeklyBehavioralReportGenerator:
    """
    Generates weekly behavioral reports with insights.

    Analyzes a week of behavioral data and produces
    actionable insights for traders.
    """

    def __init__(
        self,
        click_tracker: Optional[ClickLatencyTracker] = None,
        modification_tracker: Optional[OrderModificationTracker] = None,
        focus_detector: Optional[FocusPatternDetector] = None,
        inactivity_tracker: Optional[InactivityTracker] = None,
        session_analyzer: Optional[SessionAnalyzer] = None,
        config_tracker: Optional[ConfigurationChangeTracker] = None,
        correlator: Optional[BehaviorPnLCorrelator] = None,
    ) -> None:
        """
        Initialize report generator.

        Args:
            Various behavioral tracking components (optional)
        """
        self.click_tracker = click_tracker
        self.modification_tracker = modification_tracker
        self.focus_detector = focus_detector
        self.inactivity_tracker = inactivity_tracker
        self.session_analyzer = session_analyzer
        self.config_tracker = config_tracker
        self.correlator = correlator

        # Historical data for comparisons
        self.previous_reports: list[WeeklyBehavioralReport] = []
        self.baseline_metrics: Optional[WeeklyMetrics] = None

        logger.info("weekly_report_generator_initialized")

    def generate_weekly_report(
        self, profile_id: str, week_start: datetime
    ) -> WeeklyBehavioralReport:
        """
        Generate weekly behavioral report.

        Args:
            profile_id: Trader profile identifier
            week_start: Start of the week to analyze

        Returns:
            Complete weekly report
        """
        week_end = week_start + timedelta(days=7)

        # Collect metrics
        metrics = self._collect_weekly_metrics(week_start, week_end)

        # Compare to previous week
        comparison_previous = self._compare_to_previous_week(metrics)

        # Compare to baseline
        comparison_baseline = self._compare_to_baseline(metrics)

        # Generate insights
        insights = self._generate_insights(
            metrics, comparison_previous, comparison_baseline
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(insights)

        # Calculate risk assessment
        risk_level, tilt_score = self._assess_risk(metrics, insights)

        # Compile raw data
        raw_data = self._compile_raw_data(week_start, week_end)

        report = WeeklyBehavioralReport(
            profile_id=profile_id,
            week_start=week_start,
            week_end=week_end,
            generation_time=datetime.utcnow(),
            metrics=metrics,
            comparison_to_previous=comparison_previous,
            comparison_to_baseline=comparison_baseline,
            insights=insights,
            top_recommendations=recommendations[:5],  # Top 5
            overall_risk_level=risk_level,
            tilt_risk_score=tilt_score,
            raw_data=raw_data,
        )

        # Store for future comparisons
        self.previous_reports.append(report)
        if len(self.previous_reports) > 12:  # Keep 12 weeks
            self.previous_reports.pop(0)

        logger.info(
            "weekly_report_generated",
            profile_id=profile_id,
            week_start=week_start.isoformat(),
            risk_level=risk_level,
            insights_count=len(insights),
        )

        return report

    def _collect_weekly_metrics(
        self, week_start: datetime, week_end: datetime
    ) -> WeeklyMetrics:
        """
        Collect metrics for the week.

        Args:
            week_start: Start of week
            week_end: End of week

        Returns:
            Aggregated weekly metrics
        """
        # Default values
        metrics = WeeklyMetrics(
            total_trades=0,
            total_sessions=0,
            average_session_hours=0.0,
            average_click_latency_ms=0.0,
            latency_trend="stable",
            modification_frequency=0.0,
            cancel_rate=0.0,
            average_distraction_score=0.0,
            tab_switches_per_hour=0.0,
            total_stare_time_minutes=0.0,
            longest_inactivity_minutes=0.0,
            config_changes=0,
            config_stability_score=100.0,
            strongest_loss_behavior=None,
            loss_behavior_correlation=None,
        )

        # Collect from click tracker
        if self.click_tracker:
            click_metrics = self.click_tracker.get_metrics()
            metrics.average_click_latency_ms = click_metrics.moving_average

            # Determine trend
            if click_metrics.baseline_deviation > 1.5:
                metrics.latency_trend = "worsening"
            elif click_metrics.baseline_deviation < -0.5:
                metrics.latency_trend = "improving"

        # Collect from modification tracker
        if self.modification_tracker:
            mod_frequencies = (
                self.modification_tracker.calculate_modification_frequency()
            )
            metrics.modification_frequency = mod_frequencies.get("1hr", 0.0)

            # Estimate cancel rate (simplified)
            cancels = sum(
                1
                for m in self.modification_tracker.modifications
                if m.get("modification_type") == "cancel"
            )
            total_mods = len(self.modification_tracker.modifications)
            metrics.cancel_rate = cancels / total_mods if total_mods > 0 else 0.0

        # Collect from focus detector
        if self.focus_detector:
            focus_metrics = self.focus_detector.get_focus_metrics(
                window_minutes=60 * 24 * 7
            )
            metrics.average_distraction_score = focus_metrics.distraction_score
            metrics.tab_switches_per_hour = focus_metrics.switch_frequency * 60

        # Collect from inactivity tracker
        if self.inactivity_tracker:
            inactivity_stats = self.inactivity_tracker.get_inactivity_stats(
                hours=24 * 7
            )
            metrics.total_stare_time_minutes = (
                inactivity_stats["total_duration_seconds"] / 60
            )
            metrics.longest_inactivity_minutes = (
                inactivity_stats["longest_duration_seconds"] / 60
            )

        # Collect from session analyzer
        if self.session_analyzer:
            session_stats = self.session_analyzer.get_session_statistics()
            metrics.total_sessions = session_stats["total_sessions"]
            metrics.average_session_hours = session_stats["average_duration_hours"]

        # Collect from config tracker
        if self.config_tracker:
            config_metrics = self.config_tracker.get_change_metrics(hours=24 * 7)
            metrics.config_changes = config_metrics.total_changes
            metrics.config_stability_score = config_metrics.stability_score

        # Collect from correlator
        if self.correlator:
            loss_behaviors = self.correlator.identify_loss_behaviors()
            if loss_behaviors:
                strongest = loss_behaviors[0]
                metrics.strongest_loss_behavior = strongest.behavior_type
                metrics.loss_behavior_correlation = strongest.correlation_coefficient

        return metrics

    def _compare_to_previous_week(
        self, current: WeeklyMetrics
    ) -> Optional[dict[str, float]]:
        """
        Compare current week to previous week.

        Args:
            current: Current week metrics

        Returns:
            Percentage changes or None
        """
        if not self.previous_reports:
            return None

        previous = self.previous_reports[-1].metrics

        comparison = {}

        # Calculate percentage changes
        if previous.average_click_latency_ms > 0:
            comparison["latency_change"] = (
                (current.average_click_latency_ms - previous.average_click_latency_ms)
                / previous.average_click_latency_ms
                * 100
            )

        if previous.modification_frequency > 0:
            comparison["modification_change"] = (
                (current.modification_frequency - previous.modification_frequency)
                / previous.modification_frequency
                * 100
            )

        comparison["config_change_diff"] = (
            current.config_changes - previous.config_changes
        )
        comparison["distraction_change"] = (
            current.average_distraction_score - previous.average_distraction_score
        )

        return comparison

    def _compare_to_baseline(
        self, current: WeeklyMetrics
    ) -> Optional[dict[str, float]]:
        """
        Compare current week to baseline.

        Args:
            current: Current week metrics

        Returns:
            Deviations from baseline or None
        """
        if not self.baseline_metrics:
            return None

        baseline = self.baseline_metrics

        comparison = {}

        # Calculate deviations
        if baseline.average_click_latency_ms > 0:
            comparison["latency_deviation"] = (
                (current.average_click_latency_ms - baseline.average_click_latency_ms)
                / baseline.average_click_latency_ms
                * 100
            )

        comparison["session_length_deviation"] = (
            current.average_session_hours - baseline.average_session_hours
        )

        comparison["cancel_rate_deviation"] = current.cancel_rate - baseline.cancel_rate

        return comparison

    def _generate_insights(
        self,
        metrics: WeeklyMetrics,
        comparison_previous: Optional[dict],
        comparison_baseline: Optional[dict],
    ) -> list[BehavioralInsight]:
        """
        Generate behavioral insights from metrics.

        Args:
            metrics: Current metrics
            comparison_previous: Comparison to previous week
            comparison_baseline: Comparison to baseline

        Returns:
            List of insights
        """
        insights = []

        # Check for high distraction
        if metrics.average_distraction_score > 60:
            insights.append(
                BehavioralInsight(
                    category="risk",
                    severity="high",
                    description="High distraction levels detected",
                    recommendation="Implement focus techniques or take breaks",
                    supporting_data={
                        "distraction_score": metrics.average_distraction_score,
                        "tab_switches_per_hour": metrics.tab_switches_per_hour,
                    },
                )
            )

        # Check for excessive modifications
        if metrics.modification_frequency > 5:
            insights.append(
                BehavioralInsight(
                    category="risk",
                    severity="medium",
                    description="Excessive order modifications indicate indecision",
                    recommendation="Stick to your trading plan",
                    supporting_data={
                        "modification_frequency": metrics.modification_frequency,
                        "cancel_rate": metrics.cancel_rate,
                    },
                )
            )

        # Check for long sessions
        if metrics.average_session_hours > 4:
            insights.append(
                BehavioralInsight(
                    category="risk",
                    severity="medium",
                    description="Sessions are too long, fatigue risk",
                    recommendation="Limit sessions to 2-3 hours",
                    supporting_data={
                        "average_session_hours": metrics.average_session_hours,
                        "total_sessions": metrics.total_sessions,
                    },
                )
            )

        # Check for configuration instability
        if metrics.config_stability_score < 50:
            insights.append(
                BehavioralInsight(
                    category="risk",
                    severity="high",
                    description="Frequent configuration changes detected",
                    recommendation="Commit to settings for at least a week",
                    supporting_data={
                        "config_changes": metrics.config_changes,
                        "stability_score": metrics.config_stability_score,
                    },
                )
            )

        # Check for loss-correlated behaviors
        if metrics.strongest_loss_behavior:
            insights.append(
                BehavioralInsight(
                    category="risk",
                    severity="high",
                    description=f"{metrics.strongest_loss_behavior} strongly correlated with losses",
                    recommendation=f"Monitor and control {metrics.strongest_loss_behavior}",
                    supporting_data={
                        "behavior": metrics.strongest_loss_behavior,
                        "correlation": metrics.loss_behavior_correlation,
                    },
                )
            )

        # Check improvements
        if comparison_previous and comparison_previous.get("latency_change", 0) < -20:
            insights.append(
                BehavioralInsight(
                    category="opportunity",
                    severity="low",
                    description="Decision speed improving",
                    recommendation="Maintain current practices",
                    supporting_data={
                        "latency_improvement": comparison_previous["latency_change"]
                    },
                )
            )

        return insights

    def _generate_recommendations(self, insights: list[BehavioralInsight]) -> list[str]:
        """
        Generate prioritized recommendations.

        Args:
            insights: List of insights

        Returns:
            Prioritized recommendations
        """
        recommendations = []

        # Sort insights by severity
        high_severity = [i for i in insights if i.severity == "high"]
        medium_severity = [i for i in insights if i.severity == "medium"]
        low_severity = [i for i in insights if i.severity == "low"]

        # Add high severity recommendations first
        for insight in high_severity:
            recommendations.append(insight.recommendation)

        # Add medium severity
        for insight in medium_severity:
            recommendations.append(insight.recommendation)

        # Add low severity
        for insight in low_severity:
            recommendations.append(insight.recommendation)

        # Add general recommendations if space
        if len(recommendations) < 5:
            recommendations.append("Review weekly report with mentor or coach")

        if len(recommendations) < 5:
            recommendations.append("Set specific behavioral goals for next week")

        return recommendations

    def _assess_risk(
        self, metrics: WeeklyMetrics, insights: list[BehavioralInsight]
    ) -> Tuple[str, float]:
        """
        Assess overall risk level.

        Args:
            metrics: Weekly metrics
            insights: Generated insights

        Returns:
            Risk level and tilt score
        """
        tilt_score = 0.0

        # Score based on metrics
        tilt_score += min(20, metrics.average_distraction_score / 5)
        tilt_score += min(15, metrics.modification_frequency * 3)
        tilt_score += min(10, max(0, metrics.average_session_hours - 3) * 5)
        tilt_score += min(20, max(0, 100 - metrics.config_stability_score) / 5)

        # Score based on insights
        high_risk_count = sum(1 for i in insights if i.severity == "high")
        medium_risk_count = sum(1 for i in insights if i.severity == "medium")

        tilt_score += high_risk_count * 15
        tilt_score += medium_risk_count * 5

        # Cap at 100
        tilt_score = min(100, tilt_score)

        # Determine risk level
        if tilt_score >= 70:
            risk_level = "high"
        elif tilt_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        return risk_level, tilt_score

    def _compile_raw_data(
        self, week_start: datetime, week_end: datetime
    ) -> dict[str, Any]:
        """
        Compile raw data for export.

        Args:
            week_start: Start of week
            week_end: End of week

        Returns:
            Raw data dictionary
        """
        raw_data = {
            "period": {"start": week_start.isoformat(), "end": week_end.isoformat()},
            "components": {},
        }

        # Add raw component data if available
        if self.click_tracker:
            raw_data["components"]["click_latency"] = {
                "samples": len(self.click_tracker.recent_latencies),
                "values": list(self.click_tracker.recent_latencies),
            }

        if self.modification_tracker:
            raw_data["components"]["modifications"] = {
                "count": len(self.modification_tracker.modifications),
                "types": [
                    m.get("modification_type")
                    for m in self.modification_tracker.modifications
                ],
            }

        return raw_data

    def set_baseline(self, metrics: WeeklyMetrics) -> None:
        """
        Set baseline metrics for comparison.

        Args:
            metrics: Baseline metrics to use
        """
        self.baseline_metrics = metrics
        logger.info("baseline_metrics_set")

    def export_report_json(self, report: WeeklyBehavioralReport) -> str:
        """
        Export report as JSON.

        Args:
            report: Report to export

        Returns:
            JSON string
        """
        # Convert to dictionary
        report_dict = {
            "profile_id": report.profile_id,
            "week_start": report.week_start.isoformat(),
            "week_end": report.week_end.isoformat(),
            "generation_time": report.generation_time.isoformat(),
            "metrics": asdict(report.metrics),
            "comparison_to_previous": report.comparison_to_previous,
            "comparison_to_baseline": report.comparison_to_baseline,
            "insights": [asdict(i) for i in report.insights],
            "top_recommendations": report.top_recommendations,
            "overall_risk_level": report.overall_risk_level,
            "tilt_risk_score": report.tilt_risk_score,
            "raw_data": report.raw_data,
        }

        return json.dumps(report_dict, indent=2)

    def export_report_text(self, report: WeeklyBehavioralReport) -> str:
        """
        Export report as human-readable text.

        Args:
            report: Report to export

        Returns:
            Text report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("WEEKLY BEHAVIORAL REPORT")
        lines.append("=" * 60)
        lines.append(f"Profile: {report.profile_id}")
        lines.append(f"Week: {report.week_start.date()} to {report.week_end.date()}")
        lines.append(f"Generated: {report.generation_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

        lines.append("RISK ASSESSMENT")
        lines.append("-" * 40)
        lines.append(f"Overall Risk Level: {report.overall_risk_level.upper()}")
        lines.append(f"Tilt Risk Score: {report.tilt_risk_score:.1f}/100")
        lines.append("")

        lines.append("KEY METRICS")
        lines.append("-" * 40)
        lines.append(
            f"Sessions: {report.metrics.total_sessions} (avg {report.metrics.average_session_hours:.1f}h)"
        )
        lines.append(
            f"Click Latency: {report.metrics.average_click_latency_ms:.0f}ms ({report.metrics.latency_trend})"
        )
        lines.append(
            f"Distraction Score: {report.metrics.average_distraction_score:.1f}/100"
        )
        lines.append(
            f"Config Stability: {report.metrics.config_stability_score:.1f}/100"
        )

        if report.metrics.strongest_loss_behavior:
            lines.append(
                f"Loss Behavior: {report.metrics.strongest_loss_behavior} ({report.metrics.loss_behavior_correlation:.2f})"
            )
        lines.append("")

        lines.append("KEY INSIGHTS")
        lines.append("-" * 40)
        for i, insight in enumerate(report.insights[:5], 1):
            lines.append(f"{i}. [{insight.severity.upper()}] {insight.description}")
            lines.append(f"   → {insight.recommendation}")
        lines.append("")

        lines.append("TOP RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(report.top_recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

        if report.comparison_to_previous:
            lines.append("WEEK-OVER-WEEK CHANGES")
            lines.append("-" * 40)
            for key, value in report.comparison_to_previous.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:+.1f}%")
                else:
                    lines.append(f"{key}: {value:+d}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
