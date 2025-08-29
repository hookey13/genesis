"""
Position sizing variance behavioral indicator.

Tracks position size patterns to detect emotional trading behavior.
"""

from collections import deque
from datetime import datetime
from decimal import Decimal

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PositionSizingIndicator:
    """Monitors position sizing patterns for behavioral analysis."""

    def __init__(self, window_size: int = 50):
        """
        Initialize the position sizing indicator.

        Args:
            window_size: Number of recent positions to track
        """
        self.window_size = window_size
        self.position_sizes = deque(maxlen=window_size)
        self.position_timestamps = deque(maxlen=window_size)
        self.position_outcomes = deque(maxlen=window_size)  # win/loss/open

    def record_position(
        self, size: Decimal, timestamp: datetime, outcome: str | None = None
    ) -> dict:
        """
        Record a position and calculate variance metrics.

        Args:
            size: Position size (in base units or USD value)
            timestamp: When position was opened
            outcome: Position outcome (win/loss/open)

        Returns:
            Current variance metrics
        """
        self.position_sizes.append(size)
        self.position_timestamps.append(timestamp)
        self.position_outcomes.append(outcome)

        metrics = self.calculate_variance()

        logger.debug(
            "Position recorded",
            size=float(size),
            outcome=outcome,
            variance=metrics.get("coefficient_of_variation"),
        )

        return metrics

    def calculate_variance(self) -> dict:
        """
        Calculate position size variance metrics.

        Returns:
            Dictionary with variance analysis
        """
        if not self.position_sizes:
            return {"has_data": False, "sample_count": 0}

        if len(self.position_sizes) < 2:
            return {
                "has_data": True,
                "sample_count": len(self.position_sizes),
                "insufficient_data": True,
            }

        # Convert to numpy for calculations
        sizes = np.array([float(s) for s in self.position_sizes])

        # Basic statistics
        mean = Decimal(str(np.mean(sizes)))
        std_dev = Decimal(str(np.std(sizes)))
        min_size = Decimal(str(np.min(sizes)))
        max_size = Decimal(str(np.max(sizes)))

        # Coefficient of variation (normalized measure of dispersion)
        cv = std_dev / mean if mean > 0 else Decimal("0")

        # Check for martingale pattern (doubling after losses)
        martingale_detected = self._detect_martingale()

        # Check for size drift (gradual increase/decrease)
        drift_direction = self._detect_drift()

        # Calculate volatility metrics
        volatility_score = self._calculate_volatility()

        # Calculate recent vs historical comparison
        if len(self.position_sizes) >= 10:
            recent_5 = list(self.position_sizes)[-5:]
            older = list(self.position_sizes)[:-5]

            recent_mean = sum(recent_5) / len(recent_5)
            older_mean = sum(older) / len(older)

            size_trend = (
                "increasing"
                if recent_mean > older_mean * Decimal("1.2")
                else (
                    "decreasing"
                    if recent_mean < older_mean * Decimal("0.8")
                    else "stable"
                )
            )
        else:
            size_trend = "insufficient_data"

        return {
            "has_data": True,
            "sample_count": len(self.position_sizes),
            "mean_size": float(mean),
            "std_dev": float(std_dev),
            "min_size": float(min_size),
            "max_size": float(max_size),
            "coefficient_of_variation": float(cv),
            "high_variance": cv > Decimal("0.3"),
            "volatility_score": volatility_score,
            "high_volatility": volatility_score > 70,
            "martingale_detected": martingale_detected,
            "drift_direction": drift_direction,
            "size_trend": size_trend,
        }

    def _detect_martingale(self) -> bool:
        """
        Detect martingale betting pattern (doubling after losses).

        Returns:
            True if martingale pattern detected
        """
        if len(self.position_sizes) < 3:
            return False

        # Look for pattern of increasing size after losses
        martingale_count = 0

        for i in range(1, len(self.position_sizes)):
            if i < len(self.position_outcomes):
                prev_outcome = self.position_outcomes[i - 1]
                curr_size = self.position_sizes[i]
                prev_size = self.position_sizes[i - 1]

                # Check if size increased significantly after a loss
                if prev_outcome == "loss" and curr_size > prev_size * Decimal("1.5"):
                    martingale_count += 1

        # Detect if pattern occurs frequently
        if martingale_count >= 2:
            logger.warning("Martingale pattern detected", occurrences=martingale_count)
            return True

        return False

    def _detect_drift(self) -> str:
        """
        Detect gradual drift in position sizing.

        Returns:
            Direction of drift: "increasing", "decreasing", or "stable"
        """
        if len(self.position_sizes) < 5:
            return "insufficient_data"

        # Use linear regression to detect trend
        x = np.arange(len(self.position_sizes))
        y = np.array([float(s) for s in self.position_sizes])

        # Calculate slope
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]

        # Normalize slope by mean
        mean = np.mean(y)
        if mean > 0:
            normalized_slope = slope / mean

            if normalized_slope > 0.01:
                return "increasing"
            elif normalized_slope < -0.01:
                return "decreasing"

        return "stable"

    def _calculate_volatility(self) -> int:
        """
        Calculate position size volatility score.

        Returns:
            Volatility score from 0 (stable) to 100 (extremely volatile)
        """
        if len(self.position_sizes) < 3:
            return 0

        # Calculate rolling standard deviation
        sizes = [float(s) for s in self.position_sizes]

        # Calculate differences between consecutive positions
        differences = []
        for i in range(1, len(sizes)):
            diff_pct = (
                abs(sizes[i] - sizes[i - 1]) / sizes[i - 1] if sizes[i - 1] > 0 else 0
            )
            differences.append(diff_pct * 100)

        if not differences:
            return 0

        # Calculate volatility metrics
        avg_change = np.mean(differences)
        max_change = np.max(differences)

        # Score based on average and max changes
        avg_score = min(avg_change * 2, 50)  # Up to 50 points for average
        max_score = min(max_change, 50)  # Up to 50 points for max

        volatility_score = int(avg_score + max_score)

        # Log if high volatility
        if volatility_score > 70:
            logger.warning(
                "High position size volatility detected",
                score=volatility_score,
                avg_change_pct=avg_change,
                max_change_pct=max_change,
            )

        return min(volatility_score, 100)

    def get_risk_score(self) -> Decimal:
        """
        Calculate a risk score based on position sizing behavior.

        Returns:
            Risk score from 0 (low) to 100 (high)
        """
        metrics = self.calculate_variance()

        if not metrics.get("has_data") or metrics.get("insufficient_data"):
            return Decimal("0")

        score = Decimal("0")

        # High variance contribution (0-30 points)
        cv = Decimal(str(metrics["coefficient_of_variation"]))
        variance_score = min(cv * 100, Decimal("30"))
        score += variance_score

        # Martingale pattern contribution (0-40 points)
        if metrics["martingale_detected"]:
            score += Decimal("40")

        # Drift contribution (0-30 points)
        if metrics["drift_direction"] == "increasing":
            # Calculate how much it's increasing
            recent_mean = Decimal(str(metrics["mean_size"]))
            if len(self.position_sizes) >= 10:
                early_sizes = list(self.position_sizes)[:5]
                early_mean = sum(early_sizes) / len(early_sizes)

                if early_mean > 0:
                    increase_ratio = recent_mean / early_mean
                    drift_score = min((increase_ratio - 1) * 50, Decimal("30"))
                    score += drift_score

        return min(score, Decimal("100"))

    def get_last_n_analysis(self, n: int = 10) -> dict:
        """
        Analyze the last N positions.

        Args:
            n: Number of recent positions to analyze

        Returns:
            Analysis of recent positions
        """
        if not self.position_sizes:
            return {"has_data": False}

        recent_sizes = list(self.position_sizes)[-n:]
        recent_outcomes = list(self.position_outcomes)[-n:]

        # Calculate win/loss stats
        wins = recent_outcomes.count("win")
        losses = recent_outcomes.count("loss")

        # Size stats
        avg_size = sum(recent_sizes) / len(recent_sizes)
        max_size = max(recent_sizes)
        min_size = min(recent_sizes)

        return {
            "has_data": True,
            "count": len(recent_sizes),
            "avg_size": float(avg_size),
            "max_size": float(max_size),
            "min_size": float(min_size),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        }

    def reset(self):
        """Reset the indicator state."""
        self.position_sizes.clear()
        self.position_timestamps.clear()
        self.position_outcomes.clear()
        logger.info("Position sizing indicator reset")
