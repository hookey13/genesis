"""
Click speed (latency) behavioral indicator.

Tracks the speed of user actions to detect abnormal decision-making patterns.
"""

from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class ClickSpeedIndicator:
    """Monitors click latency patterns for tilt detection."""

    def __init__(self, window_size: int = 100):
        """
        Initialize the click speed indicator.

        Args:
            window_size: Number of recent actions to track
        """
        self.window_size = window_size
        self.action_timestamps = deque(maxlen=window_size)
        self.latencies_ms = deque(maxlen=window_size)
        self.last_market_update: Optional[datetime] = None

    def record_market_update(self, timestamp: datetime):
        """
        Record when market data was last updated.

        Args:
            timestamp: Market update timestamp
        """
        self.last_market_update = timestamp

    def record_action(self, action_timestamp: datetime) -> Optional[Decimal]:
        """
        Record a user action and calculate latency.

        Args:
            action_timestamp: When the action occurred

        Returns:
            Latency in milliseconds if calculable
        """
        if not self.last_market_update:
            logger.debug("No market update recorded, cannot calculate latency")
            return None

        # Calculate latency from last market update
        latency_seconds = (action_timestamp - self.last_market_update).total_seconds()

        # Ignore negative latencies (clock sync issues)
        if latency_seconds < 0:
            logger.warning(
                "Negative latency detected",
                action_time=action_timestamp,
                market_time=self.last_market_update,
            )
            return None

        latency_ms = Decimal(str(latency_seconds * 1000))

        # Store the measurement
        self.action_timestamps.append(action_timestamp)
        self.latencies_ms.append(latency_ms)

        logger.debug(
            "Click latency recorded",
            latency_ms=float(latency_ms),
            window_count=len(self.latencies_ms),
        )

        return latency_ms

    def get_average_latency(self) -> Optional[Decimal]:
        """
        Calculate average latency over the window.

        Returns:
            Average latency in milliseconds
        """
        if not self.latencies_ms:
            return None

        total = sum(self.latencies_ms)
        avg = total / len(self.latencies_ms)

        return avg

    def get_recent_pattern(self, last_n: int = 10) -> dict:
        """
        Get pattern analysis of recent click speeds.

        Args:
            last_n: Number of recent actions to analyze

        Returns:
            Dictionary with pattern metrics
        """
        if not self.latencies_ms:
            return {"has_data": False, "sample_count": 0}

        recent = list(self.latencies_ms)[-last_n:]

        if len(recent) < 2:
            return {
                "has_data": True,
                "sample_count": len(recent),
                "insufficient_data": True,
            }

        avg = sum(recent) / len(recent)
        min_latency = min(recent)
        max_latency = max(recent)

        # Calculate trend (increasing/decreasing/stable)
        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2 :]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_second > avg_first * Decimal("1.2"):
            trend = "increasing"  # Getting slower
        elif avg_second < avg_first * Decimal("0.8"):
            trend = "decreasing"  # Getting faster
        else:
            trend = "stable"

        # Check for erratic behavior (high variance)
        variance = sum((x - avg) ** 2 for x in recent) / len(recent)
        std_dev = variance ** Decimal("0.5")
        coefficient_of_variation = std_dev / avg if avg > 0 else Decimal("0")

        is_erratic = coefficient_of_variation > Decimal("0.5")

        return {
            "has_data": True,
            "sample_count": len(recent),
            "average_ms": float(avg),
            "min_ms": float(min_latency),
            "max_ms": float(max_latency),
            "trend": trend,
            "is_erratic": is_erratic,
            "std_dev": float(std_dev),
            "cv": float(coefficient_of_variation),
        }

    def detect_panic_clicking(self, threshold_ms: Decimal = Decimal("100")) -> bool:
        """
        Detect panic clicking pattern (very fast consecutive actions).

        Args:
            threshold_ms: Threshold for panic clicking

        Returns:
            True if panic pattern detected
        """
        if len(self.latencies_ms) < 3:
            return False

        # Check last 3 clicks
        recent = list(self.latencies_ms)[-3:]

        # All clicks faster than threshold indicates panic
        panic_detected = all(latency < threshold_ms for latency in recent)

        if panic_detected:
            logger.warning(
                "Panic clicking pattern detected",
                recent_latencies=[float(l) for l in recent],
                threshold_ms=float(threshold_ms),
            )

        return panic_detected

    def reset(self):
        """Reset the indicator state."""
        self.action_timestamps.clear()
        self.latencies_ms.clear()
        self.last_market_update = None
        logger.info("Click speed indicator reset")
