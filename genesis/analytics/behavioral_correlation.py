"""
Behavioral Correlation Analysis for Project GENESIS.

This module correlates trading performance with behavioral patterns,
analyzing how tilt states and interventions affect trading outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import structlog

from genesis.core.models import Trade

logger = structlog.get_logger(__name__)


@dataclass
class BehavioralCorrelation:
    """Container for behavioral correlation analysis results."""

    # Performance by tilt level
    performance_by_tilt: dict[str, dict] = field(default_factory=dict)

    # Trading patterns
    trades_during_tilt: int = 0
    trades_normal_state: int = 0

    # Win rates
    win_rate_during_tilt: Decimal = Decimal("0")
    win_rate_normal_state: Decimal = Decimal("0")

    # P&L metrics
    avg_pnl_during_tilt: Decimal = Decimal("0")
    avg_pnl_normal_state: Decimal = Decimal("0")
    total_pnl_during_tilt: Decimal = Decimal("0")
    total_pnl_normal_state: Decimal = Decimal("0")

    # Intervention effectiveness
    trades_after_intervention: int = 0
    win_rate_after_intervention: Decimal = Decimal("0")
    avg_pnl_after_intervention: Decimal = Decimal("0")
    intervention_recovery_rate: Decimal = Decimal("0")

    # Journal impact
    trades_after_journal: int = 0
    performance_improvement_after_journal: Decimal = Decimal("0")

    # Behavioral patterns
    most_profitable_mental_state: str = ""
    most_dangerous_behavioral_pattern: str = ""
    optimal_break_duration_minutes: int = 0

    # Risk metrics during tilt
    avg_position_size_during_tilt: Decimal = Decimal("0")
    avg_position_size_normal: Decimal = Decimal("0")
    overtrading_frequency_during_tilt: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "performance_by_tilt": self.performance_by_tilt,
            "trades_during_tilt": self.trades_during_tilt,
            "trades_normal_state": self.trades_normal_state,
            "win_rate_during_tilt": str(self.win_rate_during_tilt),
            "win_rate_normal_state": str(self.win_rate_normal_state),
            "avg_pnl_during_tilt": str(self.avg_pnl_during_tilt),
            "avg_pnl_normal_state": str(self.avg_pnl_normal_state),
            "total_pnl_during_tilt": str(self.total_pnl_during_tilt),
            "total_pnl_normal_state": str(self.total_pnl_normal_state),
            "trades_after_intervention": self.trades_after_intervention,
            "win_rate_after_intervention": str(self.win_rate_after_intervention),
            "avg_pnl_after_intervention": str(self.avg_pnl_after_intervention),
            "intervention_recovery_rate": str(self.intervention_recovery_rate),
            "trades_after_journal": self.trades_after_journal,
            "performance_improvement_after_journal": str(self.performance_improvement_after_journal),
            "most_profitable_mental_state": self.most_profitable_mental_state,
            "most_dangerous_behavioral_pattern": self.most_dangerous_behavioral_pattern,
            "optimal_break_duration_minutes": self.optimal_break_duration_minutes,
            "avg_position_size_during_tilt": str(self.avg_position_size_during_tilt),
            "avg_position_size_normal": str(self.avg_position_size_normal),
            "overtrading_frequency_during_tilt": str(self.overtrading_frequency_during_tilt)
        }


class BehavioralCorrelationAnalyzer:
    """Analyzer for correlating behavioral patterns with trading performance."""

    def __init__(self):
        """Initialize the behavioral correlation analyzer."""
        self._correlation_cache: dict[str, BehavioralCorrelation] = {}

    async def analyze_behavioral_correlation(
        self,
        trades: list[Trade],
        tilt_events: list[dict],
        interventions: list[dict],
        journal_entries: list[dict]
    ) -> BehavioralCorrelation:
        """
        Analyze correlation between behavior and trading performance.
        
        Args:
            trades: List of completed trades
            tilt_events: List of tilt detection events
            interventions: List of intervention events
            journal_entries: List of journal entries
            
        Returns:
            BehavioralCorrelation with analysis results
        """
        if not trades:
            return BehavioralCorrelation()

        logger.info(
            f"Analyzing behavioral correlation for {len(trades)} trades, "
            f"{len(tilt_events)} tilt events, {len(interventions)} interventions"
        )

        # Categorize trades by behavioral state
        trades_by_state = self._categorize_trades_by_state(trades, tilt_events)

        # Calculate performance metrics by state
        performance_metrics = self._calculate_performance_by_state(trades_by_state)

        # Analyze intervention effectiveness
        intervention_metrics = self._analyze_intervention_effectiveness(
            trades, interventions
        )

        # Analyze journal impact
        journal_metrics = self._analyze_journal_impact(trades, journal_entries)

        # Identify optimal patterns
        patterns = self._identify_optimal_patterns(
            trades_by_state, tilt_events, interventions
        )

        # Create correlation result
        correlation = BehavioralCorrelation(
            **performance_metrics,
            **intervention_metrics,
            **journal_metrics,
            **patterns
        )

        return correlation

    def _categorize_trades_by_state(
        self,
        trades: list[Trade],
        tilt_events: list[dict]
    ) -> dict[str, list[Trade]]:
        """Categorize trades by behavioral state."""
        trades_by_state = {
            "normal": [],
            "mild_tilt": [],
            "moderate_tilt": [],
            "severe_tilt": []
        }

        for trade in trades:
            # Find tilt state at time of trade
            tilt_state = self._get_tilt_state_at_time(
                trade.timestamp, tilt_events
            )
            trades_by_state[tilt_state].append(trade)

        return trades_by_state

    def _get_tilt_state_at_time(
        self,
        timestamp: datetime,
        tilt_events: list[dict]
    ) -> str:
        """Get tilt state at a specific time."""
        # Find most recent tilt event before timestamp
        relevant_events = [
            e for e in tilt_events
            if e.get("timestamp") and e["timestamp"] <= timestamp
        ]

        if not relevant_events:
            return "normal"

        # Sort by timestamp and get most recent
        latest_event = sorted(
            relevant_events,
            key=lambda e: e["timestamp"],
            reverse=True
        )[0]

        tilt_score = latest_event.get("tilt_score", 0)

        if tilt_score < 30:
            return "normal"
        elif tilt_score < 60:
            return "mild_tilt"
        elif tilt_score < 80:
            return "moderate_tilt"
        else:
            return "severe_tilt"

    def _calculate_performance_by_state(
        self,
        trades_by_state: dict[str, list[Trade]]
    ) -> dict:
        """Calculate performance metrics for each behavioral state."""
        performance_by_tilt = {}
        total_tilt_trades = 0
        total_normal_trades = 0
        tilt_wins = 0
        normal_wins = 0
        tilt_pnl = Decimal("0")
        normal_pnl = Decimal("0")
        tilt_position_sizes = []
        normal_position_sizes = []

        for state, trades in trades_by_state.items():
            if not trades:
                performance_by_tilt[state] = {
                    "trade_count": 0,
                    "win_rate": "0",
                    "avg_pnl": "0",
                    "total_pnl": "0"
                }
                continue

            wins = sum(1 for t in trades if t.pnl_dollars > 0)
            total_pnl = sum(t.pnl_dollars for t in trades)
            avg_pnl = total_pnl / Decimal(str(len(trades)))
            win_rate = Decimal(str(wins)) / Decimal(str(len(trades)))

            performance_by_tilt[state] = {
                "trade_count": len(trades),
                "win_rate": str(win_rate),
                "avg_pnl": str(avg_pnl),
                "total_pnl": str(total_pnl)
            }

            # Aggregate for summary metrics
            if state == "normal":
                total_normal_trades += len(trades)
                normal_wins += wins
                normal_pnl += total_pnl
                normal_position_sizes.extend([
                    t.quantity * t.exit_price for t in trades
                ])
            else:
                total_tilt_trades += len(trades)
                tilt_wins += wins
                tilt_pnl += total_pnl
                tilt_position_sizes.extend([
                    t.quantity * t.exit_price for t in trades
                ])

        # Calculate aggregate metrics
        win_rate_tilt = (
            Decimal(str(tilt_wins)) / Decimal(str(total_tilt_trades))
            if total_tilt_trades > 0 else Decimal("0")
        )
        win_rate_normal = (
            Decimal(str(normal_wins)) / Decimal(str(total_normal_trades))
            if total_normal_trades > 0 else Decimal("0")
        )

        avg_pnl_tilt = (
            tilt_pnl / Decimal(str(total_tilt_trades))
            if total_tilt_trades > 0 else Decimal("0")
        )
        avg_pnl_normal = (
            normal_pnl / Decimal(str(total_normal_trades))
            if total_normal_trades > 0 else Decimal("0")
        )

        avg_position_tilt = (
            sum(tilt_position_sizes) / Decimal(str(len(tilt_position_sizes)))
            if tilt_position_sizes else Decimal("0")
        )
        avg_position_normal = (
            sum(normal_position_sizes) / Decimal(str(len(normal_position_sizes)))
            if normal_position_sizes else Decimal("0")
        )

        # Calculate overtrading frequency (trades per hour during tilt vs normal)
        overtrading_freq = Decimal("0")
        if total_tilt_trades > 0 and total_normal_trades > 0:
            # Simplified: assume tilt periods are 20% of time
            tilt_trade_rate = total_tilt_trades / Decimal("0.2")
            normal_trade_rate = total_normal_trades / Decimal("0.8")
            if normal_trade_rate > 0:
                overtrading_freq = (
                    (tilt_trade_rate - normal_trade_rate) / normal_trade_rate
                ) * Decimal("100")

        return {
            "performance_by_tilt": performance_by_tilt,
            "trades_during_tilt": total_tilt_trades,
            "trades_normal_state": total_normal_trades,
            "win_rate_during_tilt": win_rate_tilt.quantize(Decimal("0.01")),
            "win_rate_normal_state": win_rate_normal.quantize(Decimal("0.01")),
            "avg_pnl_during_tilt": avg_pnl_tilt.quantize(Decimal("0.01")),
            "avg_pnl_normal_state": avg_pnl_normal.quantize(Decimal("0.01")),
            "total_pnl_during_tilt": tilt_pnl.quantize(Decimal("0.01")),
            "total_pnl_normal_state": normal_pnl.quantize(Decimal("0.01")),
            "avg_position_size_during_tilt": avg_position_tilt.quantize(Decimal("0.01")),
            "avg_position_size_normal": avg_position_normal.quantize(Decimal("0.01")),
            "overtrading_frequency_during_tilt": overtrading_freq.quantize(Decimal("0.01"))
        }

    def _analyze_intervention_effectiveness(
        self,
        trades: list[Trade],
        interventions: list[dict]
    ) -> dict:
        """Analyze effectiveness of tilt interventions."""
        if not interventions:
            return {
                "trades_after_intervention": 0,
                "win_rate_after_intervention": Decimal("0"),
                "avg_pnl_after_intervention": Decimal("0"),
                "intervention_recovery_rate": Decimal("0")
            }

        trades_after = []
        recoveries = 0

        for intervention in interventions:
            intervention_time = intervention.get("timestamp")
            if not intervention_time:
                continue

            # Find trades within 2 hours after intervention
            window_end = intervention_time + timedelta(hours=2)
            relevant_trades = [
                t for t in trades
                if intervention_time < t.timestamp <= window_end
            ]

            trades_after.extend(relevant_trades)

            # Check if performance improved
            if relevant_trades:
                post_pnl = sum(t.pnl_dollars for t in relevant_trades)
                if post_pnl > 0:
                    recoveries += 1

        # Calculate metrics
        total_trades = len(trades_after)
        wins = sum(1 for t in trades_after if t.pnl_dollars > 0)

        win_rate = (
            Decimal(str(wins)) / Decimal(str(total_trades))
            if total_trades > 0 else Decimal("0")
        )

        avg_pnl = (
            sum(t.pnl_dollars for t in trades_after) / Decimal(str(total_trades))
            if total_trades > 0 else Decimal("0")
        )

        recovery_rate = (
            Decimal(str(recoveries)) / Decimal(str(len(interventions)))
            if interventions else Decimal("0")
        )

        return {
            "trades_after_intervention": total_trades,
            "win_rate_after_intervention": win_rate.quantize(Decimal("0.01")),
            "avg_pnl_after_intervention": avg_pnl.quantize(Decimal("0.01")),
            "intervention_recovery_rate": recovery_rate.quantize(Decimal("0.01"))
        }

    def _analyze_journal_impact(
        self,
        trades: list[Trade],
        journal_entries: list[dict]
    ) -> dict:
        """Analyze impact of journal entries on performance."""
        if not journal_entries:
            return {
                "trades_after_journal": 0,
                "performance_improvement_after_journal": Decimal("0")
            }

        trades_before = []
        trades_after = []

        for entry in journal_entries:
            entry_time = entry.get("timestamp")
            if not entry_time:
                continue

            # Compare 24h before vs 24h after journal entry
            window_start = entry_time - timedelta(hours=24)
            window_end = entry_time + timedelta(hours=24)

            before = [
                t for t in trades
                if window_start <= t.timestamp < entry_time
            ]
            after = [
                t for t in trades
                if entry_time < t.timestamp <= window_end
            ]

            trades_before.extend(before)
            trades_after.extend(after)

        # Calculate performance improvement
        improvement = Decimal("0")

        if trades_before and trades_after:
            win_rate_before = sum(1 for t in trades_before if t.pnl_dollars > 0) / len(trades_before)
            win_rate_after = sum(1 for t in trades_after if t.pnl_dollars > 0) / len(trades_after)
            improvement = (
                (Decimal(str(win_rate_after)) - Decimal(str(win_rate_before))) /
                Decimal(str(win_rate_before)) * Decimal("100")
                if win_rate_before > 0 else Decimal("0")
            )

        return {
            "trades_after_journal": len(trades_after),
            "performance_improvement_after_journal": improvement.quantize(Decimal("0.01"))
        }

    def _identify_optimal_patterns(
        self,
        trades_by_state: dict[str, list[Trade]],
        tilt_events: list[dict],
        interventions: list[dict]
    ) -> dict:
        """Identify optimal behavioral patterns."""
        # Find most profitable mental state
        best_state = ""
        best_pnl = Decimal("-999999")

        for state, trades in trades_by_state.items():
            if trades:
                avg_pnl = sum(t.pnl_dollars for t in trades) / len(trades)
                if avg_pnl > best_pnl:
                    best_pnl = avg_pnl
                    best_state = state

        # Find most dangerous pattern (worst performance state)
        worst_state = ""
        worst_pnl = Decimal("999999")

        for state, trades in trades_by_state.items():
            if trades and state != "normal":
                avg_pnl = sum(t.pnl_dollars for t in trades) / len(trades)
                if avg_pnl < worst_pnl:
                    worst_pnl = avg_pnl
                    worst_state = state

        # Estimate optimal break duration
        optimal_break = 30  # Default 30 minutes
        if interventions:
            # Find average time between intervention and recovery
            recovery_times = []
            for intervention in interventions:
                if intervention.get("recovery_time_minutes"):
                    recovery_times.append(intervention["recovery_time_minutes"])

            if recovery_times:
                optimal_break = int(sum(recovery_times) / len(recovery_times))

        return {
            "most_profitable_mental_state": best_state,
            "most_dangerous_behavioral_pattern": worst_state,
            "optimal_break_duration_minutes": optimal_break
        }
