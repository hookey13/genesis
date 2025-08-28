"""Unit tests for multi-level tilt detection system."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.events import EventType
from genesis.engine.event_bus import EventBus
from genesis.tilt.baseline import BehavioralBaseline, BehavioralMetric
from genesis.tilt.detector import (
    Anomaly,
    AnomalyDetector,
    TiltDetector,
    TiltLevel,
)
from genesis.tilt.profile_manager import ProfileManager


class TestTiltLevel:
    """Test TiltLevel enum."""

    def test_tilt_levels(self):
        """Test tilt level values."""
        assert TiltLevel.NORMAL.value == "NORMAL"
        assert TiltLevel.LEVEL1.value == "LEVEL1"
        assert TiltLevel.LEVEL2.value == "LEVEL2"
        assert TiltLevel.LEVEL3.value == "LEVEL3"


class TestTiltDetector:
    """Test TiltDetector class."""

    @pytest.fixture
    def profile_manager(self):
        """Create mock profile manager."""
        manager = AsyncMock(spec=ProfileManager)
        return manager

    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock(spec=EventBus)
        return bus

    @pytest.fixture
    def detector(self, profile_manager, event_bus):
        """Create tilt detector instance."""
        return TiltDetector(
            profile_manager=profile_manager, event_bus=event_bus, anomaly_buffer_size=50
        )

    @pytest.fixture
    def mock_baseline(self):
        """Create mock baseline."""
        baseline = MagicMock(spec=BehavioralBaseline)
        baseline.get_metric_stats.return_value = {
            "median": 50.0,
            "iqr": 10.0,
            "min": 30.0,
            "max": 70.0,
        }
        return baseline

    @pytest.mark.asyncio
    async def test_detect_tilt_level_normal(
        self, detector, profile_manager, mock_baseline
    ):
        """Test detection of normal tilt level."""
        # Setup
        profile_id = "test_profile"
        profile_manager.get_profile.return_value = MagicMock(baseline=mock_baseline)

        # Create metrics within normal range
        metrics = [
            BehavioralMetric(
                metric_name="click_speed",
                value=48.0,
                timestamp=datetime.now(UTC),
                context={},
            ),
            BehavioralMetric(
                metric_name="order_frequency",
                value=52.0,
                timestamp=datetime.now(UTC),
                context={},
            ),
        ]

        # Test
        result = await detector.detect_tilt_level(profile_id, metrics)

        # Assert
        assert result.tilt_level == TiltLevel.NORMAL
        assert result.tilt_score == 0
        assert len(result.anomalies) == 0
        assert result.detection_time_ms < 50  # Performance requirement

    @pytest.mark.asyncio
    async def test_detect_tilt_level1(self, detector, profile_manager, mock_baseline):
        """Test detection of Level 1 tilt (2-3 anomalies)."""
        # Setup
        profile_id = "test_profile"
        profile_manager.get_profile.return_value = MagicMock(baseline=mock_baseline)

        # Create metrics with 2 anomalies
        metrics = [
            BehavioralMetric(
                metric_name="click_speed",
                value=100.0,  # Anomaly: far from median 50
                timestamp=datetime.now(UTC),
                context={},
            ),
            BehavioralMetric(
                metric_name="order_frequency",
                value=5.0,  # Anomaly: far from median 50
                timestamp=datetime.now(UTC),
                context={},
            ),
            BehavioralMetric(
                metric_name="position_size",
                value=52.0,  # Normal
                timestamp=datetime.now(UTC),
                context={},
            ),
        ]

        # Test
        result = await detector.detect_tilt_level(profile_id, metrics)

        # Assert
        assert result.tilt_level == TiltLevel.LEVEL1
        assert len(result.anomalies) == 2
        assert result.tilt_score > 0

    @pytest.mark.asyncio
    async def test_detect_tilt_level2(self, detector, profile_manager, mock_baseline):
        """Test detection of Level 2 tilt (4-5 anomalies)."""
        # Setup
        profile_id = "test_profile"
        profile_manager.get_profile.return_value = MagicMock(baseline=mock_baseline)

        # Create metrics with 4 anomalies
        metrics = [
            BehavioralMetric(f"metric_{i}", 100.0 + i * 10, datetime.now(UTC), {})
            for i in range(4)  # All anomalies
        ]

        # Test
        result = await detector.detect_tilt_level(profile_id, metrics)

        # Assert
        assert result.tilt_level == TiltLevel.LEVEL2
        assert len(result.anomalies) == 4
        assert result.tilt_score > 20

    @pytest.mark.asyncio
    async def test_detect_tilt_level3(self, detector, profile_manager, mock_baseline):
        """Test detection of Level 3 tilt (6+ anomalies)."""
        # Setup
        profile_id = "test_profile"
        profile_manager.get_profile.return_value = MagicMock(baseline=mock_baseline)

        # Create metrics with 6 anomalies
        metrics = [
            BehavioralMetric(f"metric_{i}", 100.0 + i * 10, datetime.now(UTC), {})
            for i in range(6)  # All anomalies
        ]

        # Test
        result = await detector.detect_tilt_level(profile_id, metrics)

        # Assert
        assert result.tilt_level == TiltLevel.LEVEL3
        assert len(result.anomalies) == 6
        assert result.tilt_score > 30

    def test_calculate_tilt_score(self, detector):
        """Test tilt score calculation."""
        # Test with no anomalies
        assert detector.calculate_tilt_score([]) == 0

        # Test with single anomaly
        anomaly1 = Anomaly(
            indicator_name="test",
            current_value=Decimal("100"),
            baseline_value=Decimal("50"),
            deviation=Decimal("50"),
            severity=5,
            timestamp=datetime.now(UTC),
            description="Test anomaly",
        )
        score = detector.calculate_tilt_score([anomaly1])
        assert score == 35  # 1*10 + 5*5 = 35

        # Test with multiple anomalies
        anomaly2 = Anomaly(
            indicator_name="test2",
            current_value=Decimal("100"),
            baseline_value=Decimal("50"),
            deviation=Decimal("50"),
            severity=8,
            timestamp=datetime.now(UTC),
            description="Test anomaly 2",
        )
        score = detector.calculate_tilt_score([anomaly1, anomaly2])
        assert score == 85  # 2*10 + (5+8)*5 = 85

        # Test score cap at 100
        anomalies = [anomaly1] * 20
        score = detector.calculate_tilt_score(anomalies)
        assert score == 100

    @pytest.mark.asyncio
    async def test_event_publishing(
        self, detector, profile_manager, event_bus, mock_baseline
    ):
        """Test that tilt events are published correctly."""
        # Setup
        profile_id = "test_profile"
        profile_manager.get_profile.return_value = MagicMock(baseline=mock_baseline)

        # Create metrics that trigger Level 1
        metrics = [
            BehavioralMetric(f"metric_{i}", 100.0, datetime.now(UTC), {})
            for i in range(2)
        ]

        # Test
        result = await detector.detect_tilt_level(profile_id, metrics)

        # Assert event was published
        event_bus.publish.assert_called_once()
        call_args = event_bus.publish.call_args
        assert call_args[0][0] == EventType.TILT_LEVEL1_DETECTED
        assert call_args[0][1]["profile_id"] == profile_id
        assert call_args[0][1]["tilt_level"] == "LEVEL1"

    @pytest.mark.asyncio
    async def test_performance_requirement(
        self, detector, profile_manager, mock_baseline
    ):
        """Test that detection completes within 50ms."""
        # Setup
        profile_id = "test_profile"
        profile_manager.get_profile.return_value = MagicMock(baseline=mock_baseline)

        # Create many metrics to stress test
        metrics = [
            BehavioralMetric(f"metric_{i}", 50.0, datetime.now(UTC), {})
            for i in range(100)
        ]

        # Test
        start = asyncio.get_event_loop().time()
        result = await detector.detect_tilt_level(profile_id, metrics)
        duration_ms = (asyncio.get_event_loop().time() - start) * 1000

        # Assert
        assert duration_ms < 50
        assert result.detection_time_ms < 50

    def test_anomaly_buffer_management(self, detector):
        """Test anomaly buffer updates and limits."""
        profile_id = "test_profile"

        # Add anomalies
        anomalies = [
            Anomaly(
                indicator_name=f"test_{i}",
                current_value=Decimal("100"),
                baseline_value=Decimal("50"),
                deviation=Decimal("50"),
                severity=5,
                timestamp=datetime.now(UTC),
                description=f"Test anomaly {i}",
            )
            for i in range(60)  # More than buffer size
        ]

        detector._update_anomaly_buffer(profile_id, anomalies)

        # Check buffer size is limited
        assert len(detector.anomaly_buffer[profile_id]) == 50  # Buffer size limit

        # Check get history
        history = detector.get_anomaly_history(profile_id)
        assert len(history) == 50

        # Check limited history
        limited = detector.get_anomaly_history(profile_id, limit=10)
        assert len(limited) == 10

        # Clear buffer
        detector.clear_anomaly_buffer(profile_id)
        assert len(detector.anomaly_buffer[profile_id]) == 0


class TestAnomalyDetector:
    """Test AnomalyDetector class."""

    @pytest.fixture
    def baseline(self):
        """Create mock baseline."""
        return MagicMock(spec=BehavioralBaseline)

    @pytest.fixture
    def anomaly_detector(self, baseline):
        """Create anomaly detector instance."""
        return AnomalyDetector(baseline)

    def test_register_indicator(self, anomaly_detector):
        """Test registering anomaly indicators."""

        def test_detector(baseline, metric):
            return None

        anomaly_detector.register_indicator("test", test_detector)
        assert "test" in anomaly_detector.indicators
        assert anomaly_detector.indicators["test"] == test_detector

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, anomaly_detector, baseline):
        """Test anomaly detection with multiple indicators."""

        # Register test indicators
        async def detector1(baseline, metric):
            return Anomaly(
                indicator_name="test1",
                current_value=Decimal("100"),
                baseline_value=Decimal("50"),
                deviation=Decimal("50"),
                severity=5,
                timestamp=datetime.now(UTC),
                description="Test 1",
            )

        async def detector2(baseline, metric):
            return None  # No anomaly

        anomaly_detector.register_indicator("test1", detector1)
        anomaly_detector.register_indicator("test2", detector2)

        # Test detection
        metric = BehavioralMetric(
            metric_name="test", value=100.0, timestamp=datetime.now(UTC), context={}
        )

        anomalies = await anomaly_detector.detect_anomalies(baseline, metric)

        # Assert
        assert len(anomalies) == 1
        assert anomalies[0].indicator_name == "test1"
