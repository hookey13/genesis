"""
Behavioral baseline calculation and management.

This module provides functionality for establishing and maintaining
behavioral baselines for tilt detection.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Constants
BASELINE_LEARNING_DAYS = 30
ROLLING_WINDOW_DAYS = 7
MIN_SAMPLES_FOR_BASELINE = 100
PERCENTILE_LOWER = 25
PERCENTILE_UPPER = 75
IQR_MULTIPLIER = Decimal("1.5")


@dataclass
class BehavioralMetric:
    """Represents a single behavioral metric measurement."""
    metric_type: str  # click_speed|order_frequency|position_size_variance|cancel_rate
    value: Decimal
    timestamp: datetime
    session_context: Optional[str] = None  # tired|alert|stressed
    time_of_day_bucket: Optional[int] = None  # Hour (0-23)
    profile_id: Optional[str] = None


@dataclass
class MetricRange:
    """Defines the normal range for a metric."""
    lower_bound: Decimal
    upper_bound: Decimal
    mean: Decimal
    std_dev: Decimal
    percentile_25: Decimal
    percentile_75: Decimal
    sample_count: int


@dataclass
class BaselineProfile:
    """Represents a complete behavioral baseline profile."""
    profile_id: str
    learning_start_date: datetime
    learning_end_date: Optional[datetime] = None
    is_mature: bool = False
    metric_ranges: dict[str, MetricRange] = field(default_factory=dict)
    time_of_day_patterns: dict[int, dict[str, MetricRange]] = field(default_factory=dict)
    context: str = "normal"  # normal|tired|alert|stressed
    total_samples: int = 0
    last_updated: Optional[datetime] = None


class BehavioralBaseline:
    """Manages behavioral baseline calculation and updates."""

    def __init__(self, learning_days: int = BASELINE_LEARNING_DAYS):
        """
        Initialize the behavioral baseline calculator.

        Args:
            learning_days: Number of days for initial baseline learning
        """
        self.learning_days = learning_days
        self.rolling_window_days = ROLLING_WINDOW_DAYS
        self.metric_buffers: dict[str, deque] = {}
        self.max_buffer_size = 10000  # Limit memory usage

    def calculate_baseline(self, metrics: list[BehavioralMetric]) -> BaselineProfile:
        """
        Calculate baseline profile from a list of behavioral metrics.

        Args:
            metrics: List of behavioral metrics

        Returns:
            BaselineProfile with calculated normal ranges
        """
        if not metrics:
            raise ValueError("Cannot calculate baseline from empty metrics list")

        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Get profile ID and date range
        profile_id = sorted_metrics[0].profile_id or "default"
        start_date = sorted_metrics[0].timestamp
        end_date = sorted_metrics[-1].timestamp

        # Check if we have enough data
        date_range = (end_date - start_date).days
        is_mature = date_range >= self.learning_days and len(metrics) >= MIN_SAMPLES_FOR_BASELINE

        # Group metrics by type
        metrics_by_type: dict[str, list[Decimal]] = {}
        time_patterns: dict[int, dict[str, list[Decimal]]] = {}

        for metric in sorted_metrics:
            # Overall metrics
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric.value)

            # Time-of-day patterns
            if metric.time_of_day_bucket is not None:
                hour = metric.time_of_day_bucket
                if hour not in time_patterns:
                    time_patterns[hour] = {}
                if metric.metric_type not in time_patterns[hour]:
                    time_patterns[hour][metric.metric_type] = []
                time_patterns[hour][metric.metric_type].append(metric.value)

        # Calculate ranges for each metric type
        metric_ranges = {}
        for metric_type, values in metrics_by_type.items():
            metric_ranges[metric_type] = self._calculate_metric_range(values)

        # Calculate time-of-day patterns
        time_of_day_patterns = {}
        for hour, hour_metrics in time_patterns.items():
            time_of_day_patterns[hour] = {}
            for metric_type, values in hour_metrics.items():
                if len(values) >= 10:  # Need minimum samples per hour
                    time_of_day_patterns[hour][metric_type] = self._calculate_metric_range(values)

        profile = BaselineProfile(
            profile_id=profile_id,
            learning_start_date=start_date,
            learning_end_date=end_date,
            is_mature=is_mature,
            metric_ranges=metric_ranges,
            time_of_day_patterns=time_of_day_patterns,
            total_samples=len(metrics),
            last_updated=datetime.now(UTC)
        )

        logger.info(
            "Baseline calculated",
            profile_id=profile_id,
            is_mature=is_mature,
            total_samples=len(metrics),
            metric_types=list(metric_ranges.keys())
        )

        return profile

    def _calculate_metric_range(self, values: list[Decimal]) -> MetricRange:
        """
        Calculate the normal range for a metric using IQR method.

        Args:
            values: List of metric values

        Returns:
            MetricRange with statistical bounds
        """
        # Convert to numpy array for calculations
        np_values = np.array([float(v) for v in values])

        # Calculate statistics
        mean = Decimal(str(np.mean(np_values)))
        std_dev = Decimal(str(np.std(np_values)))
        percentile_25 = Decimal(str(np.percentile(np_values, PERCENTILE_LOWER)))
        percentile_75 = Decimal(str(np.percentile(np_values, PERCENTILE_UPPER)))

        # Calculate IQR and bounds
        iqr = percentile_75 - percentile_25
        lower_bound = percentile_25 - (IQR_MULTIPLIER * iqr)
        upper_bound = percentile_75 + (IQR_MULTIPLIER * iqr)

        # Ensure bounds are not negative for metrics that can't be negative
        lower_bound = max(lower_bound, Decimal("0"))

        return MetricRange(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mean=mean,
            std_dev=std_dev,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            sample_count=len(values)
        )

    def update_rolling_baseline(
        self,
        current_baseline: BaselineProfile,
        new_metrics: list[BehavioralMetric]
    ) -> BaselineProfile:
        """
        Update baseline using a rolling window approach.

        Args:
            current_baseline: Existing baseline profile
            new_metrics: New metrics to incorporate

        Returns:
            Updated baseline profile
        """
        if not new_metrics:
            return current_baseline

        # Get the cutoff date for rolling window
        cutoff_date = datetime.now(UTC) - timedelta(days=self.rolling_window_days)

        # Filter new metrics to only include recent ones
        recent_metrics = [m for m in new_metrics if m.timestamp >= cutoff_date]

        if not recent_metrics:
            logger.debug("No recent metrics for rolling baseline update")
            return current_baseline

        # Update metric buffers
        for metric in recent_metrics:
            if metric.metric_type not in self.metric_buffers:
                self.metric_buffers[metric.metric_type] = deque(maxlen=self.max_buffer_size)
            self.metric_buffers[metric.metric_type].append(metric)

        # Recalculate baseline with buffered metrics
        all_buffered_metrics = []
        for buffer in self.metric_buffers.values():
            all_buffered_metrics.extend(buffer)

        if len(all_buffered_metrics) >= MIN_SAMPLES_FOR_BASELINE:
            updated_baseline = self.calculate_baseline(all_buffered_metrics)
            # Preserve some fields from original baseline
            updated_baseline.profile_id = current_baseline.profile_id
            updated_baseline.learning_start_date = current_baseline.learning_start_date
            updated_baseline.context = current_baseline.context

            logger.info(
                "Rolling baseline updated",
                profile_id=updated_baseline.profile_id,
                new_samples=len(recent_metrics),
                total_samples=updated_baseline.total_samples
            )

            return updated_baseline
        else:
            # Not enough samples yet, just update the timestamp
            current_baseline.last_updated = datetime.now(UTC)
            return current_baseline

    def is_within_baseline(
        self,
        metric: BehavioralMetric,
        baseline: BaselineProfile,
        use_time_pattern: bool = True
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if a metric is within normal baseline range.

        Args:
            metric: The metric to check
            baseline: The baseline profile to compare against
            use_time_pattern: Whether to use time-of-day specific patterns

        Returns:
            Tuple of (is_normal, deviation_score)
        """
        # Get the appropriate range
        metric_range = None

        if use_time_pattern and metric.time_of_day_bucket is not None:
            # Try to use time-specific pattern
            hour_patterns = baseline.time_of_day_patterns.get(metric.time_of_day_bucket, {})
            metric_range = hour_patterns.get(metric.metric_type)

        # Fall back to overall range if no time pattern
        if metric_range is None:
            metric_range = baseline.metric_ranges.get(metric.metric_type)

        if metric_range is None:
            logger.warning(
                "No baseline range found for metric",
                metric_type=metric.metric_type,
                profile_id=baseline.profile_id
            )
            return True, None  # Assume normal if no baseline

        # Check if within bounds
        is_normal = metric_range.lower_bound <= metric.value <= metric_range.upper_bound

        # Calculate deviation score (z-score)
        if metric_range.std_dev > 0:
            deviation = abs((metric.value - metric_range.mean) / metric_range.std_dev)
        else:
            deviation = Decimal("0")

        return is_normal, deviation

    def reset_baseline(self, profile_id: str) -> BaselineProfile:
        """
        Reset baseline for a profile, clearing all learned patterns.

        Args:
            profile_id: Profile ID to reset

        Returns:
            Empty baseline profile
        """
        # Clear buffers for this profile
        self.metric_buffers.clear()

        # Create new empty baseline
        baseline = BaselineProfile(
            profile_id=profile_id,
            learning_start_date=datetime.now(UTC),
            is_mature=False,
            metric_ranges={},
            time_of_day_patterns={},
            total_samples=0
        )

        logger.info("Baseline reset", profile_id=profile_id)

        return baseline


class BehavioralMetricCollector:
    """Aggregates behavioral indicators and collects metrics."""

    def __init__(self):
        """Initialize the metric collector."""
        self.indicators = {}
        self.last_action_time = None

    def register_indicator(self, name: str, indicator):
        """
        Register a behavioral indicator.

        Args:
            name: Indicator name
            indicator: Indicator instance
        """
        self.indicators[name] = indicator

    def collect_metric(self, action: dict) -> Optional[BehavioralMetric]:
        """
        Collect metric from a user action.

        Args:
            action: User action dictionary

        Returns:
            BehavioralMetric if collectible, None otherwise
        """
        action_type = action.get("type")
        timestamp = action.get("timestamp", datetime.now(UTC))

        # Determine metric type based on action
        metric_type = None
        value = None

        if action_type == "order_placed":
            # Calculate click speed if we have previous action
            if self.last_action_time:
                latency_ms = (timestamp - self.last_action_time).total_seconds() * 1000
                metric_type = "click_speed"
                value = Decimal(str(latency_ms))

        elif action_type == "order_cancelled":
            metric_type = "cancel_rate"
            value = Decimal("1")  # Will be aggregated

        elif action_type == "position_opened":
            size = action.get("size")
            if size:
                metric_type = "position_size_variance"
                value = Decimal(str(size))

        # Update last action time
        self.last_action_time = timestamp

        if metric_type and value is not None:
            # Get time of day bucket
            hour_bucket = timestamp.hour

            return BehavioralMetric(
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                time_of_day_bucket=hour_bucket,
                profile_id=action.get("profile_id")
            )

        return None
