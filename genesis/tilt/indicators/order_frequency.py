"""
Order frequency behavioral indicator.

Monitors the rate of order placement to detect abnormal trading patterns.
"""

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class OrderFrequencyIndicator:
    """Tracks order placement frequency for behavioral analysis."""

    def __init__(self, window_minutes: int = 60):
        """
        Initialize the order frequency indicator.
        
        Args:
            window_minutes: Time window for frequency calculation
        """
        self.window_minutes = window_minutes
        self.order_timestamps = deque()
        self.max_orders = 1000  # Prevent memory issues

    def record_order(self, timestamp: datetime) -> Decimal:
        """
        Record an order placement.
        
        Args:
            timestamp: When the order was placed
            
        Returns:
            Current orders per hour rate
        """
        # Add new timestamp
        self.order_timestamps.append(timestamp)

        # Remove old timestamps outside window
        cutoff = timestamp - timedelta(minutes=self.window_minutes)
        while self.order_timestamps and self.order_timestamps[0] < cutoff:
            self.order_timestamps.popleft()

        # Limit total stored orders
        while len(self.order_timestamps) > self.max_orders:
            self.order_timestamps.popleft()

        # Calculate rate
        orders_per_hour = self._calculate_rate()

        logger.debug(
            "Order recorded",
            timestamp=timestamp,
            window_count=len(self.order_timestamps),
            orders_per_hour=float(orders_per_hour)
        )

        return orders_per_hour

    def _calculate_rate(self) -> Decimal:
        """
        Calculate current orders per hour rate.
        
        Returns:
            Orders per hour
        """
        if not self.order_timestamps:
            return Decimal("0")

        # Get actual time span of orders
        if len(self.order_timestamps) == 1:
            # Single order, can't calculate rate yet
            return Decimal("0")

        time_span = self.order_timestamps[-1] - self.order_timestamps[0]
        hours = Decimal(str(time_span.total_seconds() / 3600))

        if hours == 0:
            # All orders at same timestamp
            return Decimal(str(len(self.order_timestamps) * 60))  # Assume 1 minute window

        rate = Decimal(str(len(self.order_timestamps))) / hours

        return rate

    def get_current_rate(self) -> Decimal:
        """
        Get current order frequency.
        
        Returns:
            Orders per hour
        """
        # Clean old timestamps first
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=self.window_minutes)

        while self.order_timestamps and self.order_timestamps[0] < cutoff:
            self.order_timestamps.popleft()

        return self._calculate_rate()

    def get_pattern_analysis(self) -> dict:
        """
        Analyze order frequency patterns.
        
        Returns:
            Dictionary with pattern analysis
        """
        if not self.order_timestamps:
            return {
                "has_data": False,
                "order_count": 0
            }

        now = datetime.now(UTC)

        # Calculate rates for different time windows
        rates = {}
        windows = [5, 15, 30, 60]  # Minutes

        for window in windows:
            cutoff = now - timedelta(minutes=window)
            recent_orders = [ts for ts in self.order_timestamps if ts >= cutoff]

            if recent_orders:
                if len(recent_orders) > 1:
                    time_span = recent_orders[-1] - recent_orders[0]
                    hours = time_span.total_seconds() / 3600
                    rate = len(recent_orders) / hours if hours > 0 else 0
                else:
                    rate = 0
            else:
                rate = 0

            rates[f"rate_{window}min"] = rate

        # Check for acceleration (increasing frequency)
        is_accelerating = False
        if rates["rate_5min"] > rates["rate_15min"] * 1.5:
            is_accelerating = True

        # Check for burst pattern
        burst_detected = False
        if len(self.order_timestamps) >= 5:
            # Check if 5+ orders in last 2 minutes
            two_min_ago = now - timedelta(minutes=2)
            recent_burst = [ts for ts in self.order_timestamps if ts >= two_min_ago]
            if len(recent_burst) >= 5:
                burst_detected = True

        return {
            "has_data": True,
            "order_count": len(self.order_timestamps),
            "current_rate_per_hour": float(self.get_current_rate()),
            "rates": {k: float(v) for k, v in rates.items()},
            "is_accelerating": is_accelerating,
            "burst_detected": burst_detected
        }

    def detect_overtrading(self, threshold: Decimal) -> bool:
        """
        Check if current rate exceeds overtrading threshold.
        
        Args:
            threshold: Orders per hour threshold
            
        Returns:
            True if overtrading detected
        """
        current_rate = self.get_current_rate()
        overtrading = current_rate > threshold

        if overtrading:
            logger.warning(
                "Overtrading detected",
                current_rate=float(current_rate),
                threshold=float(threshold)
            )

        return overtrading

    def get_inter_order_times(self) -> list[Decimal]:
        """
        Calculate time between consecutive orders.
        
        Returns:
            List of inter-order times in seconds
        """
        if len(self.order_timestamps) < 2:
            return []

        times = []
        timestamps = list(self.order_timestamps)

        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            times.append(Decimal(str(delta)))

        return times

    def reset(self):
        """Reset the indicator state."""
        self.order_timestamps.clear()
        logger.info("Order frequency indicator reset")
