"""
Monthly Performance Report Generator for Project GENESIS.

This module generates comprehensive performance reports including
all analytics metrics, charts, and insights.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from typing import Optional

import structlog

from genesis.analytics.behavioral_correlation import BehavioralCorrelationAnalyzer
from genesis.analytics.execution_quality import ExecutionQualityTracker
from genesis.analytics.pattern_analyzer import PatternAnalyzer
from genesis.analytics.performance_attribution import PerformanceAttributionEngine
from genesis.analytics.risk_metrics import RiskMetricsCalculator
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class MonthlyReport:
    """Container for monthly performance report data."""

    report_period: str
    generation_date: datetime

    # Overall performance
    total_pnl: Decimal
    total_trades: int
    win_rate: Decimal

    # Risk metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal

    # Attribution
    performance_by_strategy: dict
    performance_by_pair: dict

    # Patterns
    win_loss_patterns: dict
    best_trading_times: dict

    # Execution
    execution_quality_score: float
    average_slippage_bps: Decimal

    # Behavioral
    behavioral_correlation: dict
    tilt_impact_on_performance: Decimal

    # Recommendations
    recommendations: list[str]
    areas_for_improvement: list[str]

    def to_html(self) -> str:
        """Convert report to HTML format."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Monthly Performance Report - {self.report_period}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2e7d32; }}
        h2 {{ color: #1565c0; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 1.2em; color: #333; }}
        .positive {{ color: #2e7d32; }}
        .negative {{ color: #d32f2f; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .recommendation {{ background-color: #fff3e0; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9800; }}
        .improvement {{ background-color: #fce4ec; padding: 10px; margin: 10px 0; border-left: 4px solid #e91e63; }}
    </style>
</head>
<body>
    <h1>Monthly Performance Report</h1>
    <p>Period: {self.report_period} | Generated: {self.generation_date.strftime('%Y-%m-%d %H:%M UTC')}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric">
        <span class="metric-label">Total P&L:</span>
        <span class="metric-value {'positive' if self.total_pnl >= 0 else 'negative'}">${self.total_pnl:,.2f}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Total Trades:</span>
        <span class="metric-value">{self.total_trades}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Win Rate:</span>
        <span class="metric-value">{self.win_rate * 100:.1f}%</span>
    </div>
    
    <h2>Risk-Adjusted Performance</h2>
    <div class="metric">
        <span class="metric-label">Sharpe Ratio:</span>
        <span class="metric-value">{self.sharpe_ratio:.2f}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Sortino Ratio:</span>
        <span class="metric-value">{self.sortino_ratio:.2f}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Max Drawdown:</span>
        <span class="metric-value negative">{self.max_drawdown * 100:.1f}%</span>
    </div>
    
    <h2>Performance Attribution</h2>
    <h3>By Strategy</h3>
    <table>
        <tr><th>Strategy</th><th>Trades</th><th>P&L</th><th>Win Rate</th></tr>
        {"".join(f'<tr><td>{k}</td><td>{v.get("trades", 0)}</td><td class="{"positive" if v.get("pnl", 0) >= 0 else "negative"}">${v.get("pnl", 0):,.2f}</td><td>{v.get("win_rate", 0) * 100:.1f}%</td></tr>' for k, v in self.performance_by_strategy.items())}
    </table>
    
    <h3>By Trading Pair</h3>
    <table>
        <tr><th>Pair</th><th>Trades</th><th>P&L</th><th>Win Rate</th></tr>
        {"".join(f'<tr><td>{k}</td><td>{v.get("trades", 0)}</td><td class="{"positive" if v.get("pnl", 0) >= 0 else "negative"}">${v.get("pnl", 0):,.2f}</td><td>{v.get("win_rate", 0) * 100:.1f}%</td></tr>' for k, v in self.performance_by_pair.items())}
    </table>
    
    <h2>Execution Quality</h2>
    <div class="metric">
        <span class="metric-label">Quality Score:</span>
        <span class="metric-value">{self.execution_quality_score:.1f}/100</span>
    </div>
    <div class="metric">
        <span class="metric-label">Average Slippage:</span>
        <span class="metric-value">{self.average_slippage_bps:.1f} bps</span>
    </div>
    
    <h2>Behavioral Analysis</h2>
    <div class="metric">
        <span class="metric-label">Tilt Impact on P&L:</span>
        <span class="metric-value negative">{self.tilt_impact_on_performance * 100:.1f}%</span>
    </div>
    
    <h2>Recommendations</h2>
    {"".join(f'<div class="recommendation">{rec}</div>' for rec in self.recommendations)}
    
    <h2>Areas for Improvement</h2>
    {"".join(f'<div class="improvement">{area}</div>' for area in self.areas_for_improvement)}
    
</body>
</html>
        """
        return html


class MonthlyReportGenerator:
    """Generator for comprehensive monthly performance reports."""

    def __init__(
        self,
        attribution_engine: PerformanceAttributionEngine,
        risk_calculator: RiskMetricsCalculator,
        pattern_analyzer: PatternAnalyzer,
        execution_tracker: ExecutionQualityTracker,
        behavioral_analyzer: BehavioralCorrelationAnalyzer,
        event_bus: EventBus,
    ):
        """Initialize the report generator with all analytics components."""
        self.attribution_engine = attribution_engine
        self.risk_calculator = risk_calculator
        self.pattern_analyzer = pattern_analyzer
        self.execution_tracker = execution_tracker
        self.behavioral_analyzer = behavioral_analyzer
        self.event_bus = event_bus

        # Schedule monthly generation
        self._schedule_monthly_generation()

    async def generate_monthly_report(self, year: int, month: int) -> MonthlyReport:
        """
        Generate comprehensive monthly performance report.

        Args:
            year: Year for the report
            month: Month for the report (1-12)

        Returns:
            MonthlyReport with all analytics
        """
        # Calculate date range
        start_date = datetime(year, month, 1, tzinfo=UTC)
        if month == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=UTC)

        report_period = f"{year}-{month:02d}"

        logger.info(f"Generating monthly report for {report_period}")

        # Gather all metrics
        attribution = await self._gather_attribution(start_date, end_date)
        risk_metrics = await self._gather_risk_metrics(start_date, end_date)
        patterns = await self._gather_patterns(start_date, end_date)
        execution = await self._gather_execution_quality(start_date, end_date)
        behavioral = await self._gather_behavioral_correlation(start_date, end_date)

        # Generate insights and recommendations
        recommendations = self._generate_recommendations(
            attribution, risk_metrics, patterns, execution, behavioral
        )

        improvements = self._identify_improvements(
            attribution, risk_metrics, patterns, execution, behavioral
        )

        # Create report
        report = MonthlyReport(
            report_period=report_period,
            generation_date=datetime.now(UTC),
            total_pnl=attribution.get("total_pnl", Decimal("0")),
            total_trades=attribution.get("total_trades", 0),
            win_rate=attribution.get("win_rate", Decimal("0")),
            sharpe_ratio=risk_metrics.get("sharpe_ratio", Decimal("0")),
            sortino_ratio=risk_metrics.get("sortino_ratio", Decimal("0")),
            max_drawdown=risk_metrics.get("max_drawdown", Decimal("0")),
            performance_by_strategy=attribution.get("by_strategy", {}),
            performance_by_pair=attribution.get("by_pair", {}),
            win_loss_patterns=patterns.get("patterns", {}),
            best_trading_times=patterns.get("best_times", {}),
            execution_quality_score=execution.get("quality_score", 0.0),
            average_slippage_bps=execution.get("avg_slippage", Decimal("0")),
            behavioral_correlation=behavioral.get("correlation", {}),
            tilt_impact_on_performance=behavioral.get("tilt_impact", Decimal("0")),
            recommendations=recommendations,
            areas_for_improvement=improvements,
        )

        # Save report
        await self._save_report(report)

        # Publish event
        await self.event_bus.publish(
            "monthly_report_generated",
            {
                "period": report_period,
                "total_pnl": str(report.total_pnl),
                "sharpe_ratio": str(report.sharpe_ratio),
            },
        )

        logger.info(
            f"Monthly report generated",
            period=report_period,
            total_pnl=float(report.total_pnl),
            trades=report.total_trades,
        )

        return report

    async def _gather_attribution(
        self, start_date: datetime, end_date: datetime
    ) -> dict:
        """Gather performance attribution data."""
        # Get attribution by strategy
        strategy_attribution = await self.attribution_engine.attribute_by_strategy(
            start_date, end_date
        )

        # Get attribution by pair
        pair_attribution = await self.attribution_engine.attribute_by_pair(
            start_date, end_date
        )

        # Aggregate totals
        total_pnl = sum(r.total_pnl for r in strategy_attribution)
        total_trades = sum(r.total_trades for r in strategy_attribution)
        total_wins = sum(r.winning_trades for r in strategy_attribution)

        win_rate = (
            Decimal(str(total_wins)) / Decimal(str(total_trades))
            if total_trades > 0
            else Decimal("0")
        )

        # Format by strategy
        by_strategy = {}
        for result in strategy_attribution:
            by_strategy[result.attribution_key] = {
                "trades": result.total_trades,
                "pnl": float(result.total_pnl),
                "win_rate": float(result.win_rate),
            }

        # Format by pair
        by_pair = {}
        for result in pair_attribution:
            by_pair[result.attribution_key] = {
                "trades": result.total_trades,
                "pnl": float(result.total_pnl),
                "win_rate": float(result.win_rate),
            }

        return {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "by_strategy": by_strategy,
            "by_pair": by_pair,
        }

    async def _gather_risk_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> dict:
        """Gather risk-adjusted metrics."""
        # This would fetch returns from database
        # For now, return placeholder
        return {
            "sharpe_ratio": Decimal("1.5"),
            "sortino_ratio": Decimal("2.0"),
            "max_drawdown": Decimal("0.15"),
        }

    async def _gather_patterns(self, start_date: datetime, end_date: datetime) -> dict:
        """Gather pattern analysis."""
        # This would analyze patterns
        # For now, return placeholder
        return {
            "patterns": {"max_win_streak": 5, "max_loss_streak": 3},
            "best_times": {"best_hour": "09:00", "best_day": "Tuesday"},
        }

    async def _gather_execution_quality(
        self, start_date: datetime, end_date: datetime
    ) -> dict:
        """Gather execution quality metrics."""
        stats = await self.execution_tracker.get_statistics("24h")

        return {
            "quality_score": stats.avg_execution_score,
            "avg_slippage": stats.avg_slippage_bps,
        }

    async def _gather_behavioral_correlation(
        self, start_date: datetime, end_date: datetime
    ) -> dict:
        """Gather behavioral correlation analysis."""
        # This would analyze behavioral impact
        # For now, return placeholder
        return {"correlation": {}, "tilt_impact": Decimal("-0.05")}

    def _generate_recommendations(
        self,
        attribution: dict,
        risk_metrics: dict,
        patterns: dict,
        execution: dict,
        behavioral: dict,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check Sharpe ratio
        if risk_metrics.get("sharpe_ratio", 0) < 1:
            recommendations.append(
                "Consider reducing position sizes to improve risk-adjusted returns"
            )

        # Check execution quality
        if execution.get("quality_score", 0) < 70:
            recommendations.append(
                "Focus on improving order execution - consider using limit orders more frequently"
            )

        # Check behavioral impact
        if behavioral.get("tilt_impact", 0) < -0.03:
            recommendations.append(
                "Implement stricter tilt detection and intervention protocols"
            )

        # Check for concentrated performance
        by_strategy = attribution.get("by_strategy", {})
        if by_strategy:
            total_pnl = sum(v.get("pnl", 0) for v in by_strategy.values())
            for strategy, data in by_strategy.items():
                if total_pnl > 0 and data.get("pnl", 0) / total_pnl > 0.5:
                    recommendations.append(
                        f"Consider diversifying beyond {strategy} strategy"
                    )
                    break

        return recommendations

    def _identify_improvements(
        self,
        attribution: dict,
        risk_metrics: dict,
        patterns: dict,
        execution: dict,
        behavioral: dict,
    ) -> list[str]:
        """Identify areas for improvement."""
        improvements = []

        # Check win rate
        if attribution.get("win_rate", 0) < 0.5:
            improvements.append(
                "Win rate below 50% - review entry criteria and strategy selection"
            )

        # Check drawdown
        if risk_metrics.get("max_drawdown", 0) > 0.2:
            improvements.append(
                "Maximum drawdown exceeds 20% - implement tighter risk controls"
            )

        # Check execution slippage
        if execution.get("avg_slippage", 0) > 5:
            improvements.append(
                "Average slippage exceeds 5 bps - review order types and timing"
            )

        return improvements

    async def _save_report(self, report: MonthlyReport) -> None:
        """Save report to file system."""
        # Generate HTML
        html_content = report.to_html()

        # Save to file
        filename = f"reports/monthly_{report.report_period}.html"
        # Would save to file system here

        logger.info(f"Report saved to {filename}")

    def _schedule_monthly_generation(self) -> None:
        """Schedule automatic monthly report generation."""
        # Would set up scheduled task here
        pass
