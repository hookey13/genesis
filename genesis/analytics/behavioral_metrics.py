"""
Behavioral metrics tracking and analysis for trader psychology monitoring.

This module provides real-time tracking of micro-behaviors including
click latency, order modifications, and inactivity periods.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Optional, TYPE_CHECKING, Any

import structlog

from genesis.core.exceptions import ValidationError

if TYPE_CHECKING:
    from genesis.data.repository import Repository

logger = structlog.get_logger(__name__)


@dataclass
class LatencyMetrics:
    """Statistical metrics for latency analysis."""

    current: int
    moving_average: float
    std_deviation: float
    percentile_95: float
    baseline_deviation: float
    samples: int


class ClickLatencyTracker:
    """
    Tracks click-to-trade latency with millisecond precision.

    Maintains a rolling window of latency measurements and calculates
    statistical metrics for baseline comparison and anomaly detection.
    """

    def __init__(
        self,
        window_size: int = 100,
        baseline_window: int = 500,
        repository: Optional[Repository] = None,
        profile_id: Optional[str] = None,
    ) -> None:
        """
        Initialize latency tracker.

        Args:
            window_size: Number of samples for moving average
            baseline_window: Number of samples for baseline calculation
            repository: Repository for persistence
            profile_id: Profile ID for tracking
        """
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.repository = repository
        self.profile_id = profile_id

        # Rolling windows for different timeframes
        self.recent_latencies: deque[int] = deque(maxlen=window_size)
        self.baseline_latencies: deque[int] = deque(maxlen=baseline_window)

        # Statistical baselines
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None

        # Action-specific tracking
        self.action_latencies: dict[str, deque[int]] = {}

        logger.info(
            "click_latency_tracker_initialized",
            window_size=window_size,
            baseline_window=baseline_window,
            has_repository=repository is not None,
        )

    async def track_click_latency(
        self, action_type: str, latency_ms: int, session_id: Optional[str] = None
    ) -> None:
        """
        Track click-to-action latency.

        Args:
            action_type: Type of action (e.g., 'market_buy', 'limit_sell')
            latency_ms: Latency in milliseconds
            session_id: Optional session ID for context

        Raises:
            ValidationError: If latency_ms is negative
        """
        if latency_ms < 0:
            raise ValidationError(f"Latency cannot be negative: {latency_ms}")

        # Add to rolling windows
        self.recent_latencies.append(latency_ms)
        self.baseline_latencies.append(latency_ms)

        # Track action-specific latency
        if action_type not in self.action_latencies:
            self.action_latencies[action_type] = deque(maxlen=self.window_size)
        self.action_latencies[action_type].append(latency_ms)

        # Update baseline if enough samples
        if len(self.baseline_latencies) >= self.baseline_window // 2:
            self._update_baseline()

        # Check for anomalies
        if self.baseline_mean is not None:
            deviation = abs(latency_ms - self.baseline_mean)
            if self.baseline_std and deviation > 3 * self.baseline_std:
                logger.warning(
                    "latency_anomaly_detected",
                    action_type=action_type,
                    latency_ms=latency_ms,
                    baseline_mean=self.baseline_mean,
                    std_deviations=deviation / self.baseline_std,
                )

        # Persist to database if repository available
        if self.repository and self.profile_id:
            try:
                await self.repository.save_behavioral_metric(
                    {
                        "profile_id": self.profile_id,
                        "session_id": session_id,
                        "metric_type": "click_latency",
                        "metric_value": Decimal(str(latency_ms)),
                        "metadata": {"action_type": action_type},
                        "timestamp": datetime.now(UTC),
                        "time_of_day_bucket": datetime.now(UTC).hour,
                    }
                )
            except Exception as e:
                logger.error("Failed to save click latency metric", error=str(e))

    def _update_baseline(self) -> None:
        """Update statistical baseline from baseline window."""
        if not self.baseline_latencies:
            return

        latencies = list(self.baseline_latencies)
        self.baseline_mean = sum(latencies) / len(latencies)

        if len(latencies) > 1:
            variance = sum((x - self.baseline_mean) ** 2 for x in latencies) / (
                len(latencies) - 1
            )
            self.baseline_std = variance**0.5
            # If std is 0 (all values same), use a small default based on mean
            if self.baseline_std == 0:
                self.baseline_std = max(
                    self.baseline_mean * 0.1, 10
                )  # 10% of mean or min 10ms
        else:
            self.baseline_std = (
                max(self.baseline_mean * 0.1, 10) if self.baseline_mean else 10
            )

    def get_metrics(self, action_type: Optional[str] = None) -> LatencyMetrics:
        """
        Get statistical metrics for latency.

        Args:
            action_type: Specific action type, or None for overall metrics

        Returns:
            LatencyMetrics object with current statistics
        """
        # Select appropriate data
        if action_type and action_type in self.action_latencies:
            latencies = list(self.action_latencies[action_type])
        else:
            latencies = list(self.recent_latencies)

        if not latencies:
            return LatencyMetrics(
                current=0,
                moving_average=0.0,
                std_deviation=0.0,
                percentile_95=0.0,
                baseline_deviation=0.0,
                samples=0,
            )

        # Calculate statistics
        current = latencies[-1] if latencies else 0
        moving_average = sum(latencies) / len(latencies)

        # Standard deviation
        if len(latencies) > 1:
            variance = sum((x - moving_average) ** 2 for x in latencies) / (
                len(latencies) - 1
            )
            std_deviation = variance**0.5
        else:
            std_deviation = 0.0

        # 95th percentile
        sorted_latencies = sorted(latencies)
        percentile_idx = int(len(sorted_latencies) * 0.95)
        percentile_95 = sorted_latencies[min(percentile_idx, len(sorted_latencies) - 1)]

        # Baseline deviation
        baseline_deviation = 0.0
        if self.baseline_mean is not None:
            baseline_deviation = abs(moving_average - self.baseline_mean)
            if self.baseline_std and self.baseline_std > 0:
                baseline_deviation /= self.baseline_std

        return LatencyMetrics(
            current=current,
            moving_average=moving_average,
            std_deviation=std_deviation,
            percentile_95=percentile_95,
            baseline_deviation=baseline_deviation,
            samples=len(latencies),
        )

    def is_latency_elevated(self, threshold_std: float = 2.0) -> bool:
        """
        Check if current latency is elevated above baseline.

        Args:
            threshold_std: Number of standard deviations for threshold

        Returns:
            True if latency is elevated
        """
        if (
            not self.recent_latencies
            or self.baseline_mean is None
            or self.baseline_std is None
        ):
            return False

        # Check if current average is elevated compared to baseline
        current_avg = sum(self.recent_latencies) / len(self.recent_latencies)
        deviation = abs(current_avg - self.baseline_mean)

        # Use standard deviation if available, otherwise use percentage
        if self.baseline_std > 0:
            normalized_deviation = deviation / self.baseline_std
        else:
            # Use percentage deviation if no std available
            normalized_deviation = deviation / max(self.baseline_mean, 1) * 10

        return normalized_deviation > threshold_std


class OrderModificationTracker:
    """
    Tracks order modification frequency and patterns.

    Monitors how often orders are modified after placement,
    which can indicate indecision or emotional trading.
    """

    def __init__(
        self, repository: Optional[Repository] = None, profile_id: Optional[str] = None
    ) -> None:
        """Initialize modification tracker.

        Args:
            repository: Repository for persistence
            profile_id: Profile ID for tracking
        """
        # Time windows for frequency calculation
        self.time_windows = {
            "5min": timedelta(minutes=5),
            "30min": timedelta(minutes=30),
            "1hr": timedelta(hours=1),
        }

        # Modification history
        self.modifications: list[dict[str, Any]] = []

        # Baseline rates (modifications per minute)
        self.baseline_rates: dict[str, float] = {}

        self.repository = repository
        self.profile_id = profile_id

        logger.info(
            "order_modification_tracker_initialized",
            has_repository=repository is not None,
        )

    async def track_order_modification(
        self, order_id: str, modification_type: str, session_id: Optional[str] = None
    ) -> None:
        """
        Track an order modification event.

        Args:
            order_id: Unique order identifier
            modification_type: Type of modification (e.g., 'price', 'quantity', 'cancel')
            session_id: Optional session ID for context
        """
        modification = {
            "order_id": order_id,
            "modification_type": modification_type,
            "timestamp": datetime.now(UTC),
        }

        self.modifications.append(modification)

        # Persist to database if repository available
        if self.repository and self.profile_id:
            try:
                await self.repository.save_behavioral_metric(
                    {
                        "profile_id": self.profile_id,
                        "session_id": session_id,
                        "metric_type": "order_modification",
                        "metric_value": Decimal("1"),  # Count of modification
                        "metadata": {
                            "order_id": order_id,
                            "modification_type": modification_type,
                        },
                        "timestamp": datetime.now(UTC),
                        "time_of_day_bucket": datetime.now(UTC).hour,
                    }
                )
            except Exception as e:
                logger.error("Failed to save order modification metric", error=str(e))

        # Clean old modifications (keep 2 hours)
        cutoff = datetime.now(UTC) - timedelta(hours=2)
        self.modifications = [m for m in self.modifications if m["timestamp"] > cutoff]

        # Check for excessive modifications
        frequencies = self.calculate_modification_frequency()
        for window, freq in frequencies.items():
            baseline = self.baseline_rates.get(window, 0.5)  # Default baseline
            if freq > baseline * 3:  # 3x baseline indicates potential issue
                logger.warning(
                    "excessive_order_modifications",
                    window=window,
                    frequency=freq,
                    baseline=baseline,
                    order_id=order_id,
                )

    def calculate_modification_frequency(self) -> dict[str, float]:
        """
        Calculate modification frequency for each time window.

        Returns:
            Dictionary of window -> modifications per minute
        """
        frequencies = {}
        now = datetime.now(UTC)

        for window_name, window_delta in self.time_windows.items():
            cutoff = now - window_delta
            window_mods = [m for m in self.modifications if m["timestamp"] > cutoff]

            # Calculate rate per minute
            window_minutes = window_delta.total_seconds() / 60
            frequencies[window_name] = (
                len(window_mods) / window_minutes if window_minutes > 0 else 0
            )

        return frequencies

    def update_baseline(self, window: str, rate: float) -> None:
        """
        Update baseline modification rate for a time window.

        Args:
            window: Time window name (e.g., '5min')
            rate: Baseline modifications per minute
        """
        self.baseline_rates[window] = rate
        logger.info("baseline_modification_rate_updated", window=window, rate=rate)


class InactivityTracker:
    """
    Tracks periods of inactivity ("stare time").

    Monitors when the trader is inactive, which can indicate
    analysis paralysis or emotional freeze.
    """

    def __init__(
        self,
        inactivity_threshold_seconds: int = 30,
        repository: Optional[Repository] = None,
        profile_id: Optional[str] = None,
    ) -> None:
        """
        Initialize inactivity tracker.

        Args:
            inactivity_threshold_seconds: Seconds before considering inactive
            repository: Repository for persistence
            profile_id: Profile ID for tracking
        """
        self.inactivity_threshold = timedelta(seconds=inactivity_threshold_seconds)
        self.inactivity_periods: list[dict[str, Any]] = []
        self.current_inactive_start: Optional[datetime] = None
        self.last_activity: datetime = datetime.now(UTC)
        self.repository = repository
        self.profile_id = profile_id

        logger.info(
            "inactivity_tracker_initialized",
            threshold_seconds=inactivity_threshold_seconds,
            has_repository=repository is not None,
        )

    def track_activity(self) -> None:
        """Record user activity."""
        now = datetime.now(UTC)

        # If was inactive, record the period
        if self.current_inactive_start:
            self.track_inactivity_period(self.current_inactive_start, now)
            self.current_inactive_start = None

        self.last_activity = now

    async def track_inactivity_period(
        self,
        start: datetime,
        end: datetime,
        session_id: Optional[str] = None,
        market_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Track a period of inactivity.

        Args:
            start: Start of inactivity
            end: End of inactivity
            session_id: Optional session ID for context
            market_context: Optional market conditions during inactivity
        """
        duration = (end - start).total_seconds()

        if duration < self.inactivity_threshold.total_seconds():
            return  # Too short to count as inactivity

        period = {"start": start, "end": end, "duration_seconds": duration}

        self.inactivity_periods.append(period)

        # Persist to database if repository available
        if self.repository and self.profile_id:
            try:
                await self.repository.save_behavioral_metric(
                    {
                        "profile_id": self.profile_id,
                        "session_id": session_id,
                        "metric_type": "inactivity_period",
                        "metric_value": Decimal(str(duration)),
                        "metadata": {
                            "start": start.isoformat(),
                            "end": end.isoformat(),
                            "market_context": market_context,
                        },
                        "timestamp": end,
                        "time_of_day_bucket": end.hour,
                    }
                )
            except Exception as e:
                logger.error("Failed to save inactivity period metric", error=str(e))

        # Clean old periods (keep 24 hours)
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        self.inactivity_periods = [
            p for p in self.inactivity_periods if p["end"] > cutoff
        ]

        logger.info("inactivity_period_tracked", duration_seconds=duration)

    def check_inactivity(self) -> Optional[float]:
        """
        Check current inactivity duration.

        Returns:
            Seconds of inactivity, or None if active
        """
        now = datetime.now(UTC)
        time_since_activity = (now - self.last_activity).total_seconds()

        if time_since_activity >= self.inactivity_threshold.total_seconds():
            if not self.current_inactive_start:
                self.current_inactive_start = self.last_activity
            return time_since_activity

        return None

    def get_inactivity_stats(self, hours: int = 1) -> dict[str, Any]:
        """
        Get inactivity statistics for recent period.

        Args:
            hours: Number of hours to analyze

        Returns:
            Dictionary with inactivity statistics
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        recent_periods = [p for p in self.inactivity_periods if p["end"] > cutoff]

        if not recent_periods:
            return {
                "total_periods": 0,
                "total_duration_seconds": 0,
                "average_duration_seconds": 0,
                "longest_duration_seconds": 0,
            }

        total_duration = sum(p["duration_seconds"] for p in recent_periods)
        durations = [p["duration_seconds"] for p in recent_periods]

        return {
            "total_periods": len(recent_periods),
            "total_duration_seconds": total_duration,
            "average_duration_seconds": total_duration / len(recent_periods),
            "longest_duration_seconds": max(durations),
        }


class SessionAnalyzer:
    """
    Analyzes trading session duration and patterns.

    Tracks session length and correlates with performance
    to detect fatigue and optimal trading windows.
    """

    def __init__(self) -> None:
        """Initialize session analyzer."""
        self.sessions: list[dict[str, Any]] = []
        self.current_session_start: Optional[datetime] = None

        logger.info("session_analyzer_initialized")

    def start_session(self, session_id: str) -> None:
        """
        Start tracking a new session.

        Args:
            session_id: Unique session identifier
        """
        self.current_session_start = datetime.now(UTC)
        logger.info(
            "session_started",
            session_id=session_id,
            start_time=self.current_session_start.isoformat(),
        )

    def end_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """
        End current session and calculate metrics.

        Args:
            session_id: Session identifier

        Returns:
            Session metrics or None if no session active
        """
        if not self.current_session_start:
            logger.warning("end_session_called_without_active_session")
            return None

        end_time = datetime.now(UTC)
        duration = (end_time - self.current_session_start).total_seconds()

        session_metrics = {
            "session_id": session_id,
            "start": self.current_session_start,
            "end": end_time,
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
        }

        self.sessions.append(session_metrics)
        self.current_session_start = None

        # Check for unusually long session (fatigue indicator)
        if duration > 14400:  # 4 hours
            logger.warning(
                "long_session_detected",
                session_id=session_id,
                duration_hours=duration / 3600,
            )

        return session_metrics

    def analyze_session_duration(self, session_id: str) -> Optional[dict[str, Any]]:
        """
        Analyze a specific session's duration metrics.

        Args:
            session_id: Session to analyze

        Returns:
            Session metrics or None if not found
        """
        for session in self.sessions:
            if session["session_id"] == session_id:
                # Calculate relative to average
                avg_duration = self._calculate_average_duration()

                metrics = session.copy()
                metrics["average_duration_hours"] = avg_duration / 3600
                metrics["relative_duration"] = (
                    session["duration_seconds"] / avg_duration
                    if avg_duration > 0
                    else 0
                )

                return metrics

        return None

    def _calculate_average_duration(self) -> float:
        """Calculate average session duration in seconds."""
        if not self.sessions:
            return 0.0

        total_duration = sum(s["duration_seconds"] for s in self.sessions)
        return total_duration / len(self.sessions)

    def get_session_statistics(self) -> dict[str, Any]:
        """
        Get overall session statistics.

        Returns:
            Dictionary with session statistics
        """
        if not self.sessions:
            return {
                "total_sessions": 0,
                "average_duration_hours": 0,
                "longest_session_hours": 0,
                "shortest_session_hours": 0,
            }

        durations = [s["duration_seconds"] for s in self.sessions]
        avg_duration = sum(durations) / len(durations)

        return {
            "total_sessions": len(self.sessions),
            "average_duration_hours": avg_duration / 3600,
            "longest_session_hours": max(durations) / 3600,
            "shortest_session_hours": min(durations) / 3600,
        }
