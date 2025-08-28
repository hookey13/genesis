"""
Order cancellation rate behavioral indicator.

Monitors order cancellation patterns to detect indecisive or emotional trading.
"""

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class CancelRateIndicator:
    """Tracks order cancellation rates for behavioral analysis."""

    def __init__(self, window_minutes: int = 60):
        """
        Initialize the cancel rate indicator.

        Args:
            window_minutes: Time window for rate calculation
        """
        self.window_minutes = window_minutes
        self.order_events = deque()  # (timestamp, event_type, order_id)
        self.max_events = 1000  # Prevent memory issues

    def record_order_placed(self, order_id: str, timestamp: datetime):
        """
        Record an order placement.

        Args:
            order_id: Unique order identifier
            timestamp: When the order was placed
        """
        self._add_event(timestamp, "placed", order_id)

    def record_order_cancelled(self, order_id: str, timestamp: datetime):
        """
        Record an order cancellation.

        Args:
            order_id: Unique order identifier
            timestamp: When the order was cancelled
        """
        self._add_event(timestamp, "cancelled", order_id)

    def record_order_filled(self, order_id: str, timestamp: datetime):
        """
        Record an order fill.

        Args:
            order_id: Unique order identifier
            timestamp: When the order was filled
        """
        self._add_event(timestamp, "filled", order_id)

    def _add_event(self, timestamp: datetime, event_type: str, order_id: str):
        """
        Add an order event to the history.

        Args:
            timestamp: Event timestamp
            event_type: Type of event (placed/cancelled/filled)
            order_id: Order identifier
        """
        self.order_events.append((timestamp, event_type, order_id))

        # Clean old events
        cutoff = timestamp - timedelta(minutes=self.window_minutes * 2)
        while self.order_events and self.order_events[0][0] < cutoff:
            self.order_events.popleft()

        # Limit total events
        while len(self.order_events) > self.max_events:
            self.order_events.popleft()

        logger.debug(
            "Order event recorded",
            event_type=event_type,
            order_id=order_id,
            total_events=len(self.order_events),
        )

    def calculate_cancel_rate(self) -> Decimal:
        """
        Calculate current cancellation rate.

        Returns:
            Cancellation rate (0.0 to 1.0)
        """
        if not self.order_events:
            return Decimal("0")

        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=self.window_minutes)

        # Filter recent events
        recent_events = [
            (ts, et, oid) for ts, et, oid in self.order_events if ts >= cutoff
        ]

        if not recent_events:
            return Decimal("0")

        # Count order outcomes
        order_outcomes = {}

        for timestamp, event_type, order_id in recent_events:
            if order_id not in order_outcomes:
                order_outcomes[order_id] = []
            order_outcomes[order_id].append(event_type)

        # Calculate cancellation rate
        total_orders = 0
        cancelled_orders = 0

        for order_id, events in order_outcomes.items():
            if "placed" in events:
                total_orders += 1
                if "cancelled" in events:
                    cancelled_orders += 1

        if total_orders == 0:
            return Decimal("0")

        rate = Decimal(str(cancelled_orders)) / Decimal(str(total_orders))

        return rate

    def get_pattern_analysis(self) -> dict:
        """
        Analyze cancellation patterns.

        Returns:
            Dictionary with pattern analysis
        """
        if not self.order_events:
            return {"has_data": False, "event_count": 0}

        now = datetime.now(UTC)

        # Calculate rates for different time windows
        windows = {"5min": 5, "15min": 15, "30min": 30, "60min": 60}

        rates = {}
        event_counts = {}

        for window_name, minutes in windows.items():
            cutoff = now - timedelta(minutes=minutes)
            window_events = [
                (ts, et, oid) for ts, et, oid in self.order_events if ts >= cutoff
            ]

            # Track outcomes per order
            order_outcomes = {}
            for ts, event_type, order_id in window_events:
                if order_id not in order_outcomes:
                    order_outcomes[order_id] = []
                order_outcomes[order_id].append(event_type)

            # Count cancellations
            total = 0
            cancelled = 0

            for order_id, events in order_outcomes.items():
                if "placed" in events:
                    total += 1
                    if "cancelled" in events:
                        cancelled += 1

            rate = cancelled / total if total > 0 else 0
            rates[window_name] = rate
            event_counts[window_name] = len(window_events)

        # Check for increasing cancellation (panic/indecision)
        is_increasing = False
        if rates.get("5min", 0) > rates.get("30min", 0) * 1.5:
            is_increasing = True

        # Check for rapid cancel pattern (cancel within seconds)
        rapid_cancels = self._detect_rapid_cancels()

        # Calculate time to cancel statistics
        cancel_times = self._get_cancel_times()
        avg_cancel_time = sum(cancel_times) / len(cancel_times) if cancel_times else 0

        return {
            "has_data": True,
            "current_rate": float(self.calculate_cancel_rate()),
            "rates_by_window": {k: float(v) for k, v in rates.items()},
            "event_counts": event_counts,
            "is_increasing": is_increasing,
            "rapid_cancels_detected": rapid_cancels > 2,
            "rapid_cancel_count": rapid_cancels,
            "avg_cancel_time_seconds": float(avg_cancel_time),
        }

    def _detect_rapid_cancels(self, threshold_seconds: int = 5) -> int:
        """
        Detect orders cancelled within seconds of placement.

        Args:
            threshold_seconds: Time threshold for rapid cancellation

        Returns:
            Number of rapid cancellations detected
        """
        # Group events by order ID
        order_events = {}
        for timestamp, event_type, order_id in self.order_events:
            if order_id not in order_events:
                order_events[order_id] = []
            order_events[order_id].append((timestamp, event_type))

        rapid_count = 0

        for order_id, events in order_events.items():
            placed_time = None
            cancelled_time = None

            for timestamp, event_type in events:
                if event_type == "placed":
                    placed_time = timestamp
                elif event_type == "cancelled":
                    cancelled_time = timestamp

            if placed_time and cancelled_time:
                time_to_cancel = (cancelled_time - placed_time).total_seconds()
                if time_to_cancel <= threshold_seconds:
                    rapid_count += 1
                    logger.debug(
                        "Rapid cancel detected",
                        order_id=order_id,
                        time_to_cancel=time_to_cancel,
                    )

        return rapid_count

    def _get_cancel_times(self) -> list[Decimal]:
        """
        Calculate time between order placement and cancellation.

        Returns:
            List of cancellation times in seconds
        """
        # Group events by order ID
        order_events = {}
        for timestamp, event_type, order_id in self.order_events:
            if order_id not in order_events:
                order_events[order_id] = []
            order_events[order_id].append((timestamp, event_type))

        cancel_times = []

        for order_id, events in order_events.items():
            placed_time = None
            cancelled_time = None

            for timestamp, event_type in events:
                if event_type == "placed":
                    placed_time = timestamp
                elif event_type == "cancelled":
                    cancelled_time = timestamp

            if placed_time and cancelled_time:
                time_to_cancel = (cancelled_time - placed_time).total_seconds()
                cancel_times.append(Decimal(str(time_to_cancel)))

        return cancel_times

    def detect_indecision_pattern(self, threshold: Decimal = Decimal("0.5")) -> bool:
        """
        Detect indecision pattern (high cancellation rate).

        Args:
            threshold: Cancellation rate threshold

        Returns:
            True if indecision pattern detected
        """
        current_rate = self.calculate_cancel_rate()
        indecision = current_rate > threshold

        if indecision:
            logger.warning(
                "Indecision pattern detected",
                cancel_rate=float(current_rate),
                threshold=float(threshold),
            )

        return indecision

    def get_streak_analysis(self) -> dict:
        """
        Analyze consecutive cancellation streaks.

        Returns:
            Streak analysis data
        """
        if not self.order_events:
            return {"has_data": False}

        # Sort events by timestamp and order
        sorted_events = sorted(self.order_events, key=lambda x: x[0])

        # Track order outcomes in sequence
        outcomes = []
        processed_orders = set()

        for timestamp, event_type, order_id in sorted_events:
            if order_id not in processed_orders and event_type in [
                "cancelled",
                "filled",
            ]:
                outcomes.append(event_type)
                processed_orders.add(order_id)

        if not outcomes:
            return {"has_data": True, "insufficient_data": True}

        # Find longest cancel streak
        max_streak = 0
        current_streak = 0

        for outcome in outcomes:
            if outcome == "cancelled":
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Current streak at end
        current_cancel_streak = 0
        for outcome in reversed(outcomes):
            if outcome == "cancelled":
                current_cancel_streak += 1
            else:
                break

        return {
            "has_data": True,
            "total_outcomes": len(outcomes),
            "max_cancel_streak": max_streak,
            "current_cancel_streak": current_cancel_streak,
            "streak_warning": max_streak >= 3 or current_cancel_streak >= 3,
        }

    def reset(self):
        """Reset the indicator state."""
        self.order_events.clear()
        logger.info("Cancel rate indicator reset")
