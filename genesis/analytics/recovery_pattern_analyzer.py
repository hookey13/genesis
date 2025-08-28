from typing import Optional

"""Recovery pattern analysis for learning from recovery attempts."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from genesis.data.sqlite_repo import SQLiteRepository
from genesis.tilt.recovery_protocols import RecoveryProtocol

logger = structlog.get_logger(__name__)


@dataclass
class RecoveryAnalysis:
    """Recovery pattern analysis results."""

    account_id: str
    total_attempts: int
    successful_recoveries: int
    failed_recoveries: int
    average_recovery_days: float
    best_recovery_days: int
    worst_recovery_days: int
    most_effective_strategies: list[str]
    common_failure_patterns: list[str]
    current_recovery_progress: Optional[float]
    recommendations: list[str]


class RecoveryPatternAnalyzer:
    """Analyzes historical recovery patterns for insights."""

    def __init__(
        self,
        repository: SQLiteRepository,
    ):
        """Initialize recovery pattern analyzer.

        Args:
            repository: Database repository for historical data
        """
        self.repository = repository

    def analyze_recovery_patterns(self, account_id: str) -> RecoveryAnalysis:
        """Analyze recovery patterns for an account.

        Args:
            account_id: Account identifier

        Returns:
            Comprehensive recovery analysis
        """
        try:
            # Get historical recovery protocols
            protocols = self.repository.get_recovery_history(account_id)

            if not protocols:
                return self._create_empty_analysis(account_id)

            # Separate successful and failed recoveries
            successful = []
            failed = []

            for protocol in protocols:
                if protocol.get("is_active"):
                    continue  # Skip currently active

                if protocol.get("recovery_completed_at"):
                    successful.append(protocol)
                else:
                    failed.append(protocol)

            # Calculate metrics
            total_attempts = len(successful) + len(failed)
            successful_recoveries = len(successful)
            failed_recoveries = len(failed)

            # Recovery duration analysis
            recovery_durations = []
            for protocol in successful:
                started = datetime.fromisoformat(protocol["initiated_at"])
                completed = datetime.fromisoformat(protocol["recovery_completed_at"])
                duration_days = (completed - started).days
                recovery_durations.append(duration_days)

            if recovery_durations:
                average_recovery_days = sum(recovery_durations) / len(
                    recovery_durations
                )
                best_recovery_days = min(recovery_durations)
                worst_recovery_days = max(recovery_durations)
            else:
                average_recovery_days = 0
                best_recovery_days = 0
                worst_recovery_days = 0

            # Analyze effective strategies
            most_effective_strategies = self._analyze_effective_strategies(protocols)

            # Identify failure patterns
            common_failure_patterns = self._identify_failure_patterns(failed)

            # Get current recovery progress if active
            current_progress = self._get_current_recovery_progress(account_id)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                successful_recoveries,
                failed_recoveries,
                average_recovery_days,
                common_failure_patterns,
            )

            return RecoveryAnalysis(
                account_id=account_id,
                total_attempts=total_attempts,
                successful_recoveries=successful_recoveries,
                failed_recoveries=failed_recoveries,
                average_recovery_days=average_recovery_days,
                best_recovery_days=best_recovery_days,
                worst_recovery_days=worst_recovery_days,
                most_effective_strategies=most_effective_strategies,
                common_failure_patterns=common_failure_patterns,
                current_recovery_progress=current_progress,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(
                "Failed to analyze recovery patterns",
                account_id=account_id,
                error=str(e),
            )
            return self._create_empty_analysis(account_id)

    def _analyze_effective_strategies(self, protocols: list[dict]) -> list[str]:
        """Analyze which strategies were most effective during recovery.

        Args:
            protocols: List of recovery protocol data

        Returns:
            List of most effective strategy names
        """
        strategy_performance = {}

        for protocol in protocols:
            # Get trades during this recovery period
            started = datetime.fromisoformat(protocol["initiated_at"])
            ended = protocol.get("recovery_completed_at")
            if ended:
                ended = datetime.fromisoformat(ended)
            else:
                ended = datetime.now(UTC)

            trades = self.repository.get_trades_between(started, ended)

            for trade in trades:
                strategy = trade.get("strategy_name", "unknown")
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "total_pnl": Decimal("0"),
                        "trade_count": 0,
                        "win_count": 0,
                    }

                pnl = Decimal(trade.get("profit_loss", "0"))
                strategy_performance[strategy]["total_pnl"] += pnl
                strategy_performance[strategy]["trade_count"] += 1
                if pnl > 0:
                    strategy_performance[strategy]["win_count"] += 1

        # Sort by effectiveness (combination of PnL and win rate)
        ranked_strategies = []
        for strategy, stats in strategy_performance.items():
            if stats["trade_count"] > 0:
                win_rate = stats["win_count"] / stats["trade_count"]
                effectiveness = float(stats["total_pnl"]) * win_rate
                ranked_strategies.append((strategy, effectiveness))

        ranked_strategies.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in ranked_strategies[:3]]  # Top 3 strategies

    def _identify_failure_patterns(self, failed_protocols: list[dict]) -> list[str]:
        """Identify common patterns in failed recovery attempts.

        Args:
            failed_protocols: List of failed recovery protocol data

        Returns:
            List of common failure pattern descriptions
        """
        patterns = []

        if not failed_protocols:
            return patterns

        # Check for quick abandonment (< 3 days)
        quick_abandons = 0
        for protocol in failed_protocols:
            started = datetime.fromisoformat(protocol["initiated_at"])
            last_update = datetime.fromisoformat(
                protocol.get("updated_at", protocol["initiated_at"])
            )
            if (last_update - started).days < 3:
                quick_abandons += 1

        if quick_abandons > len(failed_protocols) * 0.5:
            patterns.append("Tendency to abandon recovery too quickly")

        # Check for over-trading during recovery
        for protocol in failed_protocols:
            if (
                protocol.get("loss_trades_count", 0)
                > protocol.get("profitable_trades_count", 0) * 2
            ):
                patterns.append("Over-trading during recovery periods")
                break

        # Check for insufficient profit ratio
        for protocol in failed_protocols:
            total_profit = Decimal(protocol.get("total_profit", "0"))
            total_loss = Decimal(protocol.get("total_loss", "0"))
            if total_loss > 0 and total_profit / total_loss < Decimal("1.5"):
                patterns.append("Poor risk/reward ratio during recovery")
                break

        return patterns

    def _get_current_recovery_progress(self, account_id: str) -> Optional[float]:
        """Get current recovery progress if actively recovering.

        Args:
            account_id: Account identifier

        Returns:
            Recovery progress percentage or None
        """
        active_protocol = self.repository.get_active_recovery_protocol(account_id)

        if not active_protocol:
            return None

        initial_debt = Decimal(active_protocol.get("initial_debt_amount", "0"))
        current_debt = Decimal(active_protocol.get("current_debt_amount", "0"))

        if initial_debt > 0:
            recovered = initial_debt - current_debt
            progress = (recovered / initial_debt) * 100
            return float(progress)

        return 0.0

    def _generate_recommendations(
        self, successful: int, failed: int, avg_days: float, failure_patterns: list[str]
    ) -> list[str]:
        """Generate personalized recovery recommendations.

        Args:
            successful: Number of successful recoveries
            failed: Number of failed recoveries
            avg_days: Average recovery duration
            failure_patterns: Identified failure patterns

        Returns:
            List of recommendations
        """
        recommendations = []

        # Success rate based recommendations
        if failed > successful:
            recommendations.append(
                "Consider more conservative position sizing during recovery"
            )
            recommendations.append("Focus on fewer, higher-quality trades")

        # Duration based recommendations
        if avg_days > 14:
            recommendations.append("Recovery takes time - maintain patience")
        elif avg_days < 7 and avg_days > 0:
            recommendations.append("Your quick recoveries work - stick to your process")

        # Pattern based recommendations
        if "abandon recovery too quickly" in str(failure_patterns):
            recommendations.append(
                "Commit to at least 7 days before evaluating recovery"
            )

        if "Over-trading" in str(failure_patterns):
            recommendations.append("Limit daily trade count during recovery")

        if "Poor risk/reward" in str(failure_patterns):
            recommendations.append("Focus on trades with 2:1 or better risk/reward")

        # General recommendations
        if not recommendations:
            recommendations.append("Maintain disciplined approach to recovery")
            recommendations.append("Document successful recovery strategies")

        return recommendations

    def _create_empty_analysis(self, account_id: str) -> RecoveryAnalysis:
        """Create empty analysis for accounts with no recovery history.

        Args:
            account_id: Account identifier

        Returns:
            Empty recovery analysis
        """
        return RecoveryAnalysis(
            account_id=account_id,
            total_attempts=0,
            successful_recoveries=0,
            failed_recoveries=0,
            average_recovery_days=0,
            best_recovery_days=0,
            worst_recovery_days=0,
            most_effective_strategies=[],
            common_failure_patterns=[],
            current_recovery_progress=None,
            recommendations=["No recovery history available"],
        )

    def track_recovery_attempt(self, protocol: RecoveryProtocol, outcome: str) -> None:
        """Track a recovery attempt for future analysis.

        Args:
            protocol: Recovery protocol to track
            outcome: Outcome of the attempt (success/failure/abandoned)
        """
        try:
            self.repository.save_recovery_outcome(
                protocol.protocol_id, outcome, datetime.now(UTC)
            )

            logger.info(
                "Recovery attempt tracked",
                protocol_id=protocol.protocol_id,
                outcome=outcome,
            )

        except Exception as e:
            logger.error(
                "Failed to track recovery attempt",
                protocol_id=protocol.protocol_id,
                error=str(e),
            )
