"""
Behavior-P&L correlation analysis for identifying problematic patterns.

Analyzes correlations between behavioral metrics and trading performance
to identify which behaviors lead to losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import structlog

from genesis.core.exceptions import ValidationError

if TYPE_CHECKING:
    from genesis.data.repository import Repository


logger = structlog.get_logger(__name__)


@dataclass
class CorrelationResult:
    """Result of behavior-P&L correlation analysis."""

    behavior_type: str
    correlation_coefficient: float
    p_value: float
    sample_size: int
    significance_level: str  # 'high', 'medium', 'low', 'none'
    impact_direction: str  # 'positive', 'negative', 'neutral'


@dataclass
class BehaviorImpact:
    """Impact assessment of a behavioral pattern."""

    behavior: str
    average_pnl_with: Decimal
    average_pnl_without: Decimal
    pnl_difference: Decimal
    occurrences: int
    recommendation: str


class BehaviorPnLCorrelator:
    """
    Correlates behavioral patterns with P&L outcomes.

    Identifies which behaviors are most associated with losses
    and provides actionable insights for improvement.
    """

    def __init__(
        self,
        min_sample_size: int = 30,
        significance_threshold: float = 0.05,
        repository: Repository | None = None,
        profile_id: str | None = None,
    ) -> None:
        """
        Initialize correlator.

        Args:
            min_sample_size: Minimum samples for correlation
            significance_threshold: P-value threshold for significance
            repository: Repository for persistence
            profile_id: Profile ID for tracking
        """
        self.min_sample_size = min_sample_size
        self.significance_threshold = significance_threshold
        self.repository = repository
        self.profile_id = profile_id

        # Behavior data storage
        self.behavior_data: dict[str, list[dict]] = {}

        # P&L data storage
        self.pnl_data: list[dict] = []

        # Cached correlations
        self.correlation_cache: dict[str, CorrelationResult] = {}

        logger.info(
            "behavior_pnl_correlator_initialized",
            min_sample_size=min_sample_size,
            significance_threshold=significance_threshold,
            has_repository=repository is not None,
        )

    def add_behavior_data(
        self,
        behavior_type: str,
        timestamp: datetime,
        value: float,
        metadata: dict | None = None,
    ) -> None:
        """
        Add behavioral data point.

        Args:
            behavior_type: Type of behavior (e.g., 'click_latency', 'cancel_rate')
            timestamp: When the behavior occurred
            value: Behavioral metric value
            metadata: Additional context
        """
        if behavior_type not in self.behavior_data:
            self.behavior_data[behavior_type] = []

        data_point = {
            "timestamp": timestamp,
            "value": value,
            "metadata": metadata or {},
        }

        self.behavior_data[behavior_type].append(data_point)

        # Invalidate cache for this behavior
        if behavior_type in self.correlation_cache:
            del self.correlation_cache[behavior_type]

    def add_pnl_data(
        self, timestamp: datetime, pnl: Decimal, position_id: str | None = None
    ) -> None:
        """
        Add P&L data point.

        Args:
            timestamp: When P&L was realized
            pnl: Profit/loss amount
            position_id: Optional position identifier
        """
        data_point = {
            "timestamp": timestamp,
            "pnl": float(pnl),  # Convert for correlation calculation
            "position_id": position_id,
        }

        self.pnl_data.append(data_point)

        # Invalidate all cached correlations
        self.correlation_cache.clear()

    async def correlate_behavior_with_pnl(
        self, behavior_type: str, time_window: int
    ) -> float:
        """
        Calculate correlation between behavior and P&L.

        Args:
            behavior_type: Behavior to correlate
            time_window: Window in minutes to match behavior with P&L

        Returns:
            Correlation coefficient (-1 to 1)

        Raises:
            ValidationError: If insufficient data
        """
        if behavior_type not in self.behavior_data:
            raise ValidationError(f"No data for behavior: {behavior_type}")

        if not self.pnl_data:
            raise ValidationError("No P&L data available")

        # Check cache
        cache_key = f"{behavior_type}_{time_window}"
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key].correlation_coefficient

        # Align behavior and P&L data
        aligned_data = self._align_time_series(behavior_type, time_window)

        if len(aligned_data) < self.min_sample_size:
            logger.warning(
                "insufficient_data_for_correlation",
                behavior_type=behavior_type,
                sample_size=len(aligned_data),
                required=self.min_sample_size,
            )
            return 0.0

        # Calculate correlation
        behavior_values = [d["behavior_value"] for d in aligned_data]
        pnl_values = [d["pnl"] for d in aligned_data]

        correlation = np.corrcoef(behavior_values, pnl_values)[0, 1]

        # Calculate p-value using permutation test
        p_value = self._calculate_p_value(behavior_values, pnl_values)

        # Determine significance
        significance = self._determine_significance(p_value)

        # Determine impact direction
        impact = "neutral"
        if abs(correlation) > 0.1:
            impact = "positive" if correlation > 0 else "negative"

        # Cache result
        result = CorrelationResult(
            behavior_type=behavior_type,
            correlation_coefficient=correlation,
            p_value=p_value,
            sample_size=len(aligned_data),
            significance_level=significance,
            impact_direction=impact,
        )

        self.correlation_cache[cache_key] = result

        # Persist to database if repository available
        if self.repository and self.profile_id:
            try:
                await self.repository.save_behavior_correlation(
                    {
                        "profile_id": self.profile_id,
                        "behavior_type": behavior_type,
                        "correlation_coefficient": correlation,
                        "p_value": p_value,
                        "sample_size": len(aligned_data),
                        "time_window_minutes": time_window,
                        "calculated_at": datetime.now(UTC),
                    }
                )
            except Exception as e:
                logger.error("Failed to save behavior correlation", error=str(e))

        return correlation

    def _align_time_series(self, behavior_type: str, time_window: int) -> list[dict]:
        """
        Align behavior and P&L data within time windows.

        Args:
            behavior_type: Behavior to align
            time_window: Window size in minutes

        Returns:
            List of aligned data points
        """
        aligned = []
        window_delta = timedelta(minutes=time_window)

        behavior_data = self.behavior_data[behavior_type]

        for pnl_point in self.pnl_data:
            pnl_time = pnl_point["timestamp"]

            # Find behavior data within window before P&L
            window_start = pnl_time - window_delta

            relevant_behaviors = [
                b for b in behavior_data if window_start <= b["timestamp"] <= pnl_time
            ]

            if relevant_behaviors:
                # Use average behavior value in window
                avg_behavior = np.mean([b["value"] for b in relevant_behaviors])

                aligned.append(
                    {
                        "timestamp": pnl_time,
                        "behavior_value": avg_behavior,
                        "pnl": pnl_point["pnl"],
                    }
                )

        return aligned

    def _calculate_p_value(
        self, x: list[float], y: list[float], n_permutations: int = 1000
    ) -> float:
        """
        Calculate p-value using permutation test.

        Args:
            x: First variable values
            y: Second variable values
            n_permutations: Number of permutations

        Returns:
            P-value
        """
        observed_correlation = np.corrcoef(x, y)[0, 1]

        # Permutation test
        higher_correlations = 0

        for _ in range(n_permutations):
            # Shuffle one variable
            shuffled_y = np.random.permutation(y)
            perm_correlation = np.corrcoef(x, shuffled_y)[0, 1]

            if abs(perm_correlation) >= abs(observed_correlation):
                higher_correlations += 1

        return higher_correlations / n_permutations

    def _determine_significance(self, p_value: float) -> str:
        """
        Determine statistical significance level.

        Args:
            p_value: P-value from correlation test

        Returns:
            Significance level string
        """
        if p_value < 0.01:
            return "high"
        elif p_value < 0.05:
            return "medium"
        elif p_value < 0.1:
            return "low"
        else:
            return "none"

    def get_all_correlations(self, time_window: int = 30) -> list[CorrelationResult]:
        """
        Calculate correlations for all behaviors.

        Args:
            time_window: Window in minutes

        Returns:
            List of correlation results
        """
        results = []

        for behavior_type in self.behavior_data.keys():
            try:
                correlation = self.correlate_behavior_with_pnl(
                    behavior_type, time_window
                )

                cache_key = f"{behavior_type}_{time_window}"
                if cache_key in self.correlation_cache:
                    results.append(self.correlation_cache[cache_key])

            except ValidationError as e:
                logger.warning(
                    "correlation_calculation_failed",
                    behavior_type=behavior_type,
                    error=str(e),
                )

        # Sort by absolute correlation strength
        results.sort(key=lambda r: abs(r.correlation_coefficient), reverse=True)

        return results

    def identify_loss_behaviors(
        self, threshold: float = -0.3
    ) -> list[CorrelationResult]:
        """
        Identify behaviors most correlated with losses.

        Args:
            threshold: Correlation threshold for loss behaviors

        Returns:
            List of behaviors correlated with losses
        """
        all_correlations = self.get_all_correlations()

        loss_behaviors = [
            c
            for c in all_correlations
            if c.correlation_coefficient < threshold
            and c.significance_level in ["high", "medium"]
        ]

        return loss_behaviors

    def calculate_behavior_impact(
        self, behavior_type: str, threshold_percentile: float = 75
    ) -> BehaviorImpact:
        """
        Calculate the P&L impact of a specific behavior.

        Args:
            behavior_type: Behavior to analyze
            threshold_percentile: Percentile to define high behavior

        Returns:
            BehaviorImpact analysis
        """
        if behavior_type not in self.behavior_data:
            raise ValidationError(f"No data for behavior: {behavior_type}")

        # Get aligned data
        aligned = self._align_time_series(behavior_type, 30)

        if not aligned:
            raise ValidationError("No aligned data available")

        # Calculate threshold for "high" behavior
        behavior_values = [d["behavior_value"] for d in aligned]
        threshold = np.percentile(behavior_values, threshold_percentile)

        # Split P&L by behavior level
        pnl_with_behavior = [
            d["pnl"] for d in aligned if d["behavior_value"] >= threshold
        ]

        pnl_without_behavior = [
            d["pnl"] for d in aligned if d["behavior_value"] < threshold
        ]

        # Calculate averages
        avg_with = (
            Decimal(str(np.mean(pnl_with_behavior)))
            if pnl_with_behavior
            else Decimal("0")
        )
        avg_without = (
            Decimal(str(np.mean(pnl_without_behavior)))
            if pnl_without_behavior
            else Decimal("0")
        )
        difference = avg_with - avg_without

        # Generate recommendation
        recommendation = self._generate_impact_recommendation(
            behavior_type, difference, len(pnl_with_behavior)
        )

        return BehaviorImpact(
            behavior=behavior_type,
            average_pnl_with=avg_with,
            average_pnl_without=avg_without,
            pnl_difference=difference,
            occurrences=len(pnl_with_behavior),
            recommendation=recommendation,
        )

    def _generate_impact_recommendation(
        self, behavior_type: str, pnl_difference: Decimal, occurrences: int
    ) -> str:
        """
        Generate recommendation based on behavior impact.

        Args:
            behavior_type: Type of behavior
            pnl_difference: P&L difference with/without behavior
            occurrences: Number of occurrences

        Returns:
            Recommendation string
        """
        if pnl_difference < -100 and occurrences > 10:
            return f"Critical: {behavior_type} strongly associated with losses. Implement controls."
        elif pnl_difference < -50:
            return f"Warning: {behavior_type} negatively impacts performance. Monitor closely."
        elif pnl_difference > 50 and occurrences > 10:
            return f"Positive: {behavior_type} associated with gains. Maintain this pattern."
        else:
            return f"Neutral: {behavior_type} shows no significant P&L impact."

    def get_correlation_summary(self) -> dict[str, any]:
        """
        Get comprehensive correlation analysis summary.

        Returns:
            Dictionary with analysis results
        """
        all_correlations = self.get_all_correlations()
        loss_behaviors = self.identify_loss_behaviors()

        # Find strongest correlations
        strongest_positive = None
        strongest_negative = None

        for corr in all_correlations:
            if corr.significance_level in ["high", "medium"]:
                if (
                    strongest_positive is None
                    or corr.correlation_coefficient
                    > strongest_positive.correlation_coefficient
                ):
                    if corr.correlation_coefficient > 0:
                        strongest_positive = corr

                if (
                    strongest_negative is None
                    or corr.correlation_coefficient
                    < strongest_negative.correlation_coefficient
                ):
                    if corr.correlation_coefficient < 0:
                        strongest_negative = corr

        return {
            "total_behaviors_analyzed": len(all_correlations),
            "significant_correlations": sum(
                1
                for c in all_correlations
                if c.significance_level in ["high", "medium"]
            ),
            "loss_associated_behaviors": len(loss_behaviors),
            "strongest_positive": (
                {
                    "behavior": strongest_positive.behavior_type,
                    "correlation": strongest_positive.correlation_coefficient,
                }
                if strongest_positive
                else None
            ),
            "strongest_negative": (
                {
                    "behavior": strongest_negative.behavior_type,
                    "correlation": strongest_negative.correlation_coefficient,
                }
                if strongest_negative
                else None
            ),
            "top_risk_behaviors": [b.behavior_type for b in loss_behaviors[:3]],
            "recommendation": self._generate_summary_recommendation(loss_behaviors),
        }

    def _generate_summary_recommendation(
        self, loss_behaviors: list[CorrelationResult]
    ) -> str:
        """
        Generate overall recommendation from correlation analysis.

        Args:
            loss_behaviors: Behaviors correlated with losses

        Returns:
            Recommendation string
        """
        if not loss_behaviors:
            return "No concerning behavioral patterns detected"
        elif len(loss_behaviors) >= 3:
            return "Multiple risk behaviors detected - comprehensive intervention recommended"
        elif loss_behaviors[0].correlation_coefficient < -0.5:
            return f"Strong loss correlation with {loss_behaviors[0].behavior_type} - immediate attention required"
        else:
            return f"Monitor {loss_behaviors[0].behavior_type} - moderate correlation with losses"
