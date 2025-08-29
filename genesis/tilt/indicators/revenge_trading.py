from __future__ import annotations

"""Revenge trading pattern detection indicator."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from genesis.tilt.baseline import BehavioralMetric

logger = structlog.get_logger(__name__)


@dataclass
class TradingLoss:
    """Represents a trading loss event."""

    timestamp: datetime
    amount: Decimal
    symbol: str
    position_size: Decimal


class RevengeTradingDetector:
    """Detects revenge trading patterns after losses."""

    def __init__(
        self,
        loss_streak_threshold: int = 3,
        time_window_minutes: int = 30,
        size_multiplier_threshold: Decimal = Decimal("1.5"),
    ):
        """Initialize revenge trading detector.

        Args:
            loss_streak_threshold: Number of consecutive losses to trigger
            time_window_minutes: Time window to analyze patterns
            size_multiplier_threshold: Position size increase threshold
        """
        self.loss_streak_threshold = loss_streak_threshold
        self.time_window_minutes = time_window_minutes
        self.size_multiplier_threshold = size_multiplier_threshold

        # Track losses per profile
        self.loss_history: dict[str, list[TradingLoss]] = {}
        self.consecutive_losses: dict[str, int] = {}
        self.last_position_sizes: dict[str, Decimal] = {}

    def record_trade_result(
        self,
        profile_id: str,
        pnl: Decimal,
        symbol: str,
        position_size: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a trade result for pattern detection.

        Args:
            profile_id: Profile identifier
            pnl: Profit/loss from trade
            symbol: Trading symbol
            position_size: Size of position
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Initialize if needed
        if profile_id not in self.loss_history:
            self.loss_history[profile_id] = []
            self.consecutive_losses[profile_id] = 0

        # Update consecutive loss counter
        if pnl < 0:
            self.consecutive_losses[profile_id] += 1

            # Record loss
            loss = TradingLoss(
                timestamp=timestamp,
                amount=abs(pnl),
                symbol=symbol,
                position_size=position_size,
            )
            self.loss_history[profile_id].append(loss)

            # Clean old losses outside time window
            self._clean_old_losses(profile_id)
        else:
            # Reset streak on profit
            self.consecutive_losses[profile_id] = 0

        # Track position size
        self.last_position_sizes[profile_id] = position_size

    def detect_revenge_pattern(
        self, profile_id: str, current_metric: BehavioralMetric
    ) -> dict | None:
        """Detect revenge trading pattern.

        Args:
            profile_id: Profile identifier
            current_metric: Current behavioral metric

        Returns:
            Detection result if pattern found, None otherwise
        """
        # Check consecutive losses
        losses = self.consecutive_losses.get(profile_id, 0)
        if losses < self.loss_streak_threshold:
            return None

        # Check position size increase after losses
        if profile_id in self.last_position_sizes:
            last_size = self.last_position_sizes[profile_id]

            # Look for size increase in current metric
            if current_metric.metric_name == "position_size":
                current_size = Decimal(str(current_metric.value))

                if current_size > last_size * self.size_multiplier_threshold:
                    return {
                        "pattern": "revenge_trading",
                        "consecutive_losses": losses,
                        "position_size_increase": float(current_size / last_size),
                        "severity": min(losses * 2, 10),  # Cap at 10
                        "description": f"Position size increased {current_size/last_size:.1f}x after {losses} losses",
                    }

        # Check rapid trading after losses
        if current_metric.metric_name == "order_frequency":
            recent_losses = self._get_recent_losses(profile_id)
            if len(recent_losses) >= self.loss_streak_threshold:
                # Calculate time since last loss
                time_since_loss = (
                    datetime.now(UTC) - recent_losses[-1].timestamp
                ).total_seconds()

                if time_since_loss < 300:  # Within 5 minutes
                    return {
                        "pattern": "revenge_trading_speed",
                        "consecutive_losses": losses,
                        "time_since_loss_seconds": time_since_loss,
                        "severity": 7,
                        "description": f"Rapid trading {time_since_loss:.0f}s after {losses} losses",
                    }

        return None

    def _clean_old_losses(self, profile_id: str) -> None:
        """Remove losses outside the time window.

        Args:
            profile_id: Profile identifier
        """
        if profile_id not in self.loss_history:
            return

        cutoff = datetime.now(UTC) - timedelta(minutes=self.time_window_minutes)
        self.loss_history[profile_id] = [
            loss for loss in self.loss_history[profile_id] if loss.timestamp > cutoff
        ]

    def _get_recent_losses(self, profile_id: str) -> list[TradingLoss]:
        """Get recent losses within time window.

        Args:
            profile_id: Profile identifier

        Returns:
            List of recent losses
        """
        if profile_id not in self.loss_history:
            return []

        cutoff = datetime.now(UTC) - timedelta(minutes=self.time_window_minutes)
        return [
            loss for loss in self.loss_history[profile_id] if loss.timestamp > cutoff
        ]

    def reset_profile(self, profile_id: str) -> None:
        """Reset tracking for a profile.

        Args:
            profile_id: Profile identifier
        """
        if profile_id in self.loss_history:
            self.loss_history[profile_id].clear()
        if profile_id in self.consecutive_losses:
            self.consecutive_losses[profile_id] = 0
        if profile_id in self.last_position_sizes:
            del self.last_position_sizes[profile_id]

        logger.debug("Revenge trading tracking reset", profile_id=profile_id)
