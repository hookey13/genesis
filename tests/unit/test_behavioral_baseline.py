"""
Unit tests for behavioral baseline calculation.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from genesis.tilt.baseline import (
    BaselineProfile,
    BehavioralBaseline,
    BehavioralMetric,
    BehavioralMetricCollector,
    MetricRange,
)


class TestBehavioralBaseline:
    """Test behavioral baseline calculation."""

    @pytest.fixture
    def baseline_calculator(self):
        """Create baseline calculator instance."""
        return BehavioralBaseline(learning_days=30)

    @pytest.fixture
    def sample_metrics(self):
        """Create sample behavioral metrics."""
        metrics = []
        base_time = datetime.now(UTC) - timedelta(days=35)

        # Generate 35 days of metrics
        for day in range(35):
            timestamp = base_time + timedelta(days=day)

            # Click speed metrics (varying throughout the day)
            for hour in [9, 12, 15, 18]:
                metrics.append(
                    BehavioralMetric(
                        metric_type="click_speed",
                        value=Decimal(
                            str(100 + (hour * 5) + (day % 7) * 10)
                        ),  # 100-200ms range
                        timestamp=timestamp.replace(hour=hour),
                        time_of_day_bucket=hour,
                        profile_id="test_profile",
                    )
                )

            # Order frequency metrics
            metrics.append(
                BehavioralMetric(
                    metric_type="order_frequency",
                    value=Decimal(str(10 + (day % 10))),  # 10-20 orders/hour
                    timestamp=timestamp.replace(hour=14),
                    time_of_day_bucket=14,
                    profile_id="test_profile",
                )
            )

            # Cancel rate metrics
            metrics.append(
                BehavioralMetric(
                    metric_type="cancel_rate",
                    value=Decimal(str(0.1 + (day % 5) * 0.05)),  # 0.1-0.35 rate
                    timestamp=timestamp.replace(hour=14),
                    time_of_day_bucket=14,
                    profile_id="test_profile",
                )
            )

            # Position size variance
            metrics.append(
                BehavioralMetric(
                    metric_type="position_size_variance",
                    value=Decimal(str(0.2 + (day % 3) * 0.1)),  # 0.2-0.4 variance
                    timestamp=timestamp.replace(hour=14),
                    time_of_day_bucket=14,
                    profile_id="test_profile",
                )
            )

        return metrics

    def test_calculate_baseline_with_empty_metrics(self, baseline_calculator):
        """Test baseline calculation with empty metrics raises error."""
        with pytest.raises(
            ValueError, match="Cannot calculate baseline from empty metrics"
        ):
            baseline_calculator.calculate_baseline([])

    def test_calculate_baseline_with_sufficient_data(
        self, baseline_calculator, sample_metrics
    ):
        """Test baseline calculation with sufficient data."""
        profile = baseline_calculator.calculate_baseline(sample_metrics)

        assert profile.profile_id == "test_profile"
        assert profile.is_mature is True  # 35 days > 30 day requirement
        assert profile.total_samples == len(sample_metrics)
        assert len(profile.metric_ranges) == 4  # All 4 metric types

        # Check metric ranges exist
        assert "click_speed" in profile.metric_ranges
        assert "order_frequency" in profile.metric_ranges
        assert "cancel_rate" in profile.metric_ranges
        assert "position_size_variance" in profile.metric_ranges

        # Check time patterns
        assert len(profile.time_of_day_patterns) > 0
        assert 14 in profile.time_of_day_patterns  # Most metrics at hour 14

    def test_calculate_baseline_insufficient_samples(self, baseline_calculator):
        """Test baseline with insufficient samples marks as immature."""
        metrics = []
        base_time = datetime.now(UTC)

        # Only 50 samples (less than MIN_SAMPLES_FOR_BASELINE=100)
        for i in range(50):
            metrics.append(
                BehavioralMetric(
                    metric_type="click_speed",
                    value=Decimal("150"),
                    timestamp=base_time - timedelta(hours=i),
                    profile_id="test_profile",
                )
            )

        profile = baseline_calculator.calculate_baseline(metrics)

        assert profile.is_mature is False
        assert profile.total_samples == 50

    def test_metric_range_calculation(self, baseline_calculator):
        """Test metric range calculation using IQR method."""
        values = [
            Decimal(str(x)) for x in [10, 15, 20, 25, 30, 35, 40, 100]
        ]  # 100 is outlier

        metric_range = baseline_calculator._calculate_metric_range(values)

        assert metric_range.sample_count == 8
        assert metric_range.lower_bound >= Decimal("0")  # Never negative
        assert metric_range.upper_bound > metric_range.lower_bound
        assert metric_range.percentile_25 < metric_range.percentile_75

        # Outlier (100) should be outside normal range
        assert Decimal("100") > metric_range.upper_bound

    def test_rolling_baseline_update(self, baseline_calculator, sample_metrics):
        """Test rolling window baseline update."""
        # Initial baseline
        initial_profile = baseline_calculator.calculate_baseline(sample_metrics[:200])

        # New metrics for rolling update
        new_metrics = sample_metrics[200:250]

        updated_profile = baseline_calculator.update_rolling_baseline(
            initial_profile, new_metrics
        )

        assert updated_profile.profile_id == initial_profile.profile_id
        assert updated_profile.last_updated is not None
        assert updated_profile.total_samples > 0

    def test_is_within_baseline(self, baseline_calculator, sample_metrics):
        """Test checking if metric is within baseline range."""
        profile = baseline_calculator.calculate_baseline(sample_metrics)

        # Test normal metric
        normal_metric = BehavioralMetric(
            metric_type="click_speed",
            value=Decimal("150"),  # Should be within normal range
            timestamp=datetime.now(UTC),
            time_of_day_bucket=14,
        )

        is_normal, deviation = baseline_calculator.is_within_baseline(
            normal_metric, profile, use_time_pattern=True
        )

        assert isinstance(is_normal, bool)
        assert deviation is not None
        assert deviation >= Decimal("0")

        # Test extreme metric
        extreme_metric = BehavioralMetric(
            metric_type="click_speed",
            value=Decimal("5000"),  # Way outside normal range
            timestamp=datetime.now(UTC),
            time_of_day_bucket=14,
        )

        is_normal, deviation = baseline_calculator.is_within_baseline(
            extreme_metric, profile, use_time_pattern=True
        )

        assert is_normal is False
        assert deviation > Decimal("3")  # High z-score

    def test_reset_baseline(self, baseline_calculator):
        """Test baseline reset functionality."""
        profile = baseline_calculator.reset_baseline("test_profile")

        assert profile.profile_id == "test_profile"
        assert profile.is_mature is False
        assert len(profile.metric_ranges) == 0
        assert profile.total_samples == 0
        assert len(baseline_calculator.metric_buffers) == 0


class TestBehavioralMetricCollector:
    """Test behavioral metric collection."""

    @pytest.fixture
    def collector(self):
        """Create metric collector instance."""
        return BehavioralMetricCollector()

    def test_collect_order_placed_metric(self, collector):
        """Test collecting metric from order placed action."""
        # Set previous action time
        collector.last_action_time = datetime.now(UTC) - timedelta(milliseconds=250)

        action = {
            "type": "order_placed",
            "timestamp": datetime.now(UTC),
            "profile_id": "test_profile",
        }

        metric = collector.collect_metric(action)

        assert metric is not None
        assert metric.metric_type == "click_speed"
        assert metric.value > Decimal("0")
        assert metric.value < Decimal("1000")  # Reasonable latency
        assert metric.profile_id == "test_profile"

    def test_collect_order_cancelled_metric(self, collector):
        """Test collecting metric from order cancelled action."""
        action = {
            "type": "order_cancelled",
            "timestamp": datetime.now(UTC),
            "profile_id": "test_profile",
        }

        metric = collector.collect_metric(action)

        assert metric is not None
        assert metric.metric_type == "cancel_rate"
        assert metric.value == Decimal("1")

    def test_collect_position_opened_metric(self, collector):
        """Test collecting metric from position opened action."""
        action = {
            "type": "position_opened",
            "size": 1000,
            "timestamp": datetime.now(UTC),
            "profile_id": "test_profile",
        }

        metric = collector.collect_metric(action)

        assert metric is not None
        assert metric.metric_type == "position_size_variance"
        assert metric.value == Decimal("1000")

    def test_time_of_day_bucket(self, collector):
        """Test time of day bucket assignment."""
        timestamp = datetime(2024, 1, 1, 15, 30, 0, tzinfo=UTC)

        action = {"type": "order_cancelled", "timestamp": timestamp}

        metric = collector.collect_metric(action)

        assert metric is not None
        assert metric.time_of_day_bucket == 15

    def test_unknown_action_type(self, collector):
        """Test handling unknown action type."""
        action = {"type": "unknown_action", "timestamp": datetime.now(UTC)}

        metric = collector.collect_metric(action)

        assert metric is None


class TestMetricRange:
    """Test MetricRange dataclass."""

    def test_metric_range_creation(self):
        """Test creating metric range."""
        metric_range = MetricRange(
            lower_bound=Decimal("10"),
            upper_bound=Decimal("100"),
            mean=Decimal("55"),
            std_dev=Decimal("15"),
            percentile_25=Decimal("40"),
            percentile_75=Decimal("70"),
            sample_count=1000,
        )

        assert metric_range.lower_bound < metric_range.upper_bound
        assert metric_range.percentile_25 < metric_range.percentile_75
        assert metric_range.sample_count > 0


class TestBaselineProfile:
    """Test BaselineProfile dataclass."""

    def test_baseline_profile_creation(self):
        """Test creating baseline profile."""
        profile = BaselineProfile(
            profile_id="test_profile",
            learning_start_date=datetime.now(UTC),
            is_mature=False,
            context="normal",
        )

        assert profile.profile_id == "test_profile"
        assert profile.is_mature is False
        assert profile.context == "normal"
        assert profile.total_samples == 0
        assert len(profile.metric_ranges) == 0
        assert len(profile.time_of_day_patterns) == 0
