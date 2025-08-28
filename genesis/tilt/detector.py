from __future__ import annotations

from typing import Optional

"""Multi-level tilt detection system."""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import structlog

from genesis.analytics.behavioral_metrics import (
    ClickLatencyTracker,
    InactivityTracker,
    OrderModificationTracker,
    SessionAnalyzer,
)
from genesis.analytics.config_tracker import ConfigurationChangeTracker
from genesis.core.events import EventType
from genesis.engine.event_bus import EventBus
from genesis.tilt.baseline import BehavioralBaseline, BehavioralMetric
from genesis.tilt.indicators.focus_patterns import FocusPatternDetector
from genesis.tilt.profile_manager import ProfileManager

logger = structlog.get_logger(__name__)


class TiltLevel(Enum):
    """Tilt severity levels."""

    NORMAL = "NORMAL"
    LEVEL1 = "LEVEL1"  # 2-3 anomalies - Yellow warning
    LEVEL2 = "LEVEL2"  # 4-5 anomalies - Orange warning, reduced sizing
    LEVEL3 = "LEVEL3"  # 6+ anomalies - Red lockout


@dataclass
class Anomaly:
    """Represents a detected behavioral anomaly."""

    indicator_name: str
    current_value: Decimal
    baseline_value: Decimal
    deviation: Decimal
    severity: int  # 1-10 scale
    timestamp: datetime
    description: str


@dataclass
class TiltDetectionResult:
    """Result of tilt detection analysis."""

    profile_id: str
    tilt_level: TiltLevel
    tilt_score: int  # 0-100 scale
    anomalies: list[Anomaly]
    detection_time_ms: float
    timestamp: datetime


class TiltDetector:
    """Multi-level tilt detection system."""

    # Thresholds for tilt levels
    LEVEL1_THRESHOLD = 2  # 2-3 anomalies
    LEVEL2_THRESHOLD = 4  # 4-5 anomalies
    LEVEL3_THRESHOLD = 6  # 6+ anomalies

    # Performance target
    MAX_DETECTION_TIME_MS = 50

    def __init__(
        self,
        profile_manager: ProfileManager,
        event_bus: Optional[EventBus] = None,
        anomaly_buffer_size: int = 100,
        click_tracker: Optional[ClickLatencyTracker] = None,
        modification_tracker: Optional[OrderModificationTracker] = None,
        focus_detector: Optional[FocusPatternDetector] = None,
        inactivity_tracker: Optional[InactivityTracker] = None,
        session_analyzer: Optional[SessionAnalyzer] = None,
        config_tracker: Optional[ConfigurationChangeTracker] = None,
    ):
        """Initialize tilt detector.

        Args:
            profile_manager: Manager for behavioral profiles
            event_bus: Event bus for publishing tilt events
            anomaly_buffer_size: Size of anomaly history buffer
            click_tracker: Click latency tracker
            modification_tracker: Order modification tracker
            focus_detector: Focus pattern detector
            inactivity_tracker: Inactivity tracker
            session_analyzer: Session analyzer
            config_tracker: Configuration change tracker
        """
        self.profile_manager = profile_manager
        self.event_bus = event_bus
        self.anomaly_buffer: dict[str, deque] = {}  # profile_id -> deque of anomalies
        self.anomaly_buffer_size = anomaly_buffer_size
        self.anomaly_detectors: dict[str, callable] = {}

        # New behavioral trackers from Story 3.4
        self.click_tracker = click_tracker or ClickLatencyTracker()
        self.modification_tracker = modification_tracker or OrderModificationTracker()
        self.focus_detector = focus_detector or FocusPatternDetector()
        self.inactivity_tracker = inactivity_tracker or InactivityTracker()
        self.session_analyzer = session_analyzer or SessionAnalyzer()
        self.config_tracker = config_tracker or ConfigurationChangeTracker()

        # Cache for baseline data (for performance)
        self.baseline_cache: dict[str, BehavioralBaseline] = {}
        self.cache_ttl_seconds = 60  # Refresh cache every minute
        self.cache_timestamps: dict[str, datetime] = {}

    async def detect_tilt_level(
        self, profile_id: str, metrics: list[BehavioralMetric]
    ) -> TiltDetectionResult:
        """Detect tilt level from behavioral metrics.

        Args:
            profile_id: Profile identifier
            metrics: Current behavioral metrics

        Returns:
            TiltDetectionResult with level, score, and anomalies
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Get or refresh baseline
            baseline = await self._get_cached_baseline(profile_id)
            if not baseline:
                logger.warning("No baseline found for profile", profile_id=profile_id)
                return TiltDetectionResult(
                    profile_id=profile_id,
                    tilt_level=TiltLevel.NORMAL,
                    tilt_score=0,
                    anomalies=[],
                    detection_time_ms=0,
                    timestamp=datetime.now(UTC),
                )

            # Detect anomalies
            anomalies = await self._detect_anomalies(baseline, metrics)

            # Calculate tilt score
            tilt_score = self.calculate_tilt_score(anomalies)

            # Determine tilt level
            tilt_level = self._determine_tilt_level(len(anomalies))

            # Store anomalies in buffer
            self._update_anomaly_buffer(profile_id, anomalies)

            # Calculate detection time
            detection_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Log performance warning if too slow
            if detection_time_ms > self.MAX_DETECTION_TIME_MS:
                logger.warning(
                    "Tilt detection exceeded time budget",
                    detection_time_ms=detection_time_ms,
                    max_ms=self.MAX_DETECTION_TIME_MS,
                )

            result = TiltDetectionResult(
                profile_id=profile_id,
                tilt_level=tilt_level,
                tilt_score=tilt_score,
                anomalies=anomalies,
                detection_time_ms=detection_time_ms,
                timestamp=datetime.now(UTC),
            )

            # Publish event if tilt detected
            if tilt_level != TiltLevel.NORMAL and self.event_bus:
                await self._publish_tilt_event(result)

            return result

        except Exception as e:
            logger.error("Error in tilt detection", error=str(e), profile_id=profile_id)
            raise

    def calculate_tilt_score(self, anomalies: list[Anomaly]) -> int:
        """Calculate tilt score from anomalies (0-100 scale).

        Args:
            anomalies: List of detected anomalies

        Returns:
            Tilt score between 0 and 100
        """
        if not anomalies:
            return 0

        # Base score from anomaly count
        base_score = len(anomalies) * 10

        # Add severity bonus
        severity_bonus = sum(a.severity for a in anomalies) * 5

        # Add bonus for behavioral anomalies from Story 3.4
        behavioral_bonus = 0

        # Check click latency
        if self.click_tracker.is_latency_elevated():
            behavioral_bonus += 10

        # Check distraction
        if self.focus_detector.is_distracted():
            behavioral_bonus += 15

        # Check configuration instability
        if self.config_tracker.is_configuration_unstable():
            behavioral_bonus += 10

        # Check for excessive modifications
        mod_frequencies = self.modification_tracker.calculate_modification_frequency()
        if mod_frequencies.get("5min", 0) > 3:
            behavioral_bonus += 5

        # Cap at 100
        tilt_score = min(base_score + severity_bonus + behavioral_bonus, 100)

        return int(tilt_score)

    async def _get_cached_baseline(
        self, profile_id: str
    ) -> Optional[BehavioralBaseline]:
        """Get baseline from cache or refresh if needed.

        Args:
            profile_id: Profile identifier

        Returns:
            Cached or refreshed baseline
        """
        now = datetime.now(UTC)

        # Check if cache is valid
        if profile_id in self.baseline_cache:
            cache_time = self.cache_timestamps.get(profile_id)
            if (
                cache_time
                and (now - cache_time).total_seconds() < self.cache_ttl_seconds
            ):
                return self.baseline_cache[profile_id]

        # Refresh cache
        profile = await self.profile_manager.get_profile(profile_id)
        if profile and profile.baseline:
            self.baseline_cache[profile_id] = profile.baseline
            self.cache_timestamps[profile_id] = now
            return profile.baseline

        return None

    async def _detect_anomalies(
        self, baseline: BehavioralBaseline, metrics: list[BehavioralMetric]
    ) -> list[Anomaly]:
        """Detect anomalies by comparing metrics to baseline.

        Args:
            baseline: Behavioral baseline
            metrics: Current metrics

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for metric in metrics:
            # Get baseline stats for this metric
            baseline_stats = baseline.get_metric_stats(metric.metric_name)
            if not baseline_stats:
                continue

            # Check if value is anomalous using IQR method
            current_value = Decimal(str(metric.value))
            median = Decimal(str(baseline_stats["median"]))
            iqr = Decimal(str(baseline_stats["iqr"]))

            # Calculate deviation
            deviation = abs(current_value - median)
            threshold = Decimal("1.5") * iqr if iqr > 0 else Decimal("0.1")

            if deviation > threshold:
                # Calculate severity (1-10 scale)
                if iqr > 0:
                    severity = min(int((deviation / iqr).quantize(Decimal("1"))), 10)
                else:
                    severity = 5  # Default severity if no IQR

                anomaly = Anomaly(
                    indicator_name=metric.metric_name,
                    current_value=current_value,
                    baseline_value=median,
                    deviation=deviation,
                    severity=severity,
                    timestamp=metric.timestamp,
                    description=f"{metric.metric_name} deviates {deviation:.2f} from baseline {median:.2f}",
                )
                anomalies.append(anomaly)

                logger.debug(
                    "Anomaly detected",
                    indicator=metric.metric_name,
                    current=float(current_value),
                    baseline=float(median),
                    deviation=float(deviation),
                )

        # Add behavioral anomalies from Story 3.4 trackers
        behavioral_anomalies = self._detect_behavioral_anomalies()
        anomalies.extend(behavioral_anomalies)

        return anomalies

    def _detect_behavioral_anomalies(self) -> list[Anomaly]:
        """Detect anomalies from new behavioral trackers.

        Returns:
            List of behavioral anomalies
        """
        anomalies = []
        now = datetime.now(UTC)

        # Check click latency elevation
        if self.click_tracker.is_latency_elevated(threshold_std=2.0):
            metrics = self.click_tracker.get_metrics()
            anomalies.append(
                Anomaly(
                    indicator_name="click_latency",
                    current_value=Decimal(str(metrics.current)),
                    baseline_value=Decimal(str(metrics.moving_average)),
                    deviation=Decimal(str(metrics.baseline_deviation)),
                    severity=min(int(metrics.baseline_deviation * 3), 10),
                    timestamp=now,
                    description=f"Click latency elevated: {metrics.current}ms (baseline: {metrics.moving_average:.0f}ms)",
                )
            )

        # Check distraction level
        if self.focus_detector.is_distracted(threshold=50.0):
            focus_metrics = self.focus_detector.get_focus_metrics()
            anomalies.append(
                Anomaly(
                    indicator_name="distraction_level",
                    current_value=Decimal(str(focus_metrics.distraction_score)),
                    baseline_value=Decimal("20"),  # Normal distraction baseline
                    deviation=Decimal(str(focus_metrics.distraction_score - 20)),
                    severity=min(int(focus_metrics.distraction_score / 10), 10),
                    timestamp=now,
                    description=f"High distraction: {focus_metrics.distraction_score:.0f}/100",
                )
            )

        # Check configuration instability
        if self.config_tracker.is_configuration_unstable(threshold=50.0):
            config_metrics = self.config_tracker.get_change_metrics()
            anomalies.append(
                Anomaly(
                    indicator_name="config_instability",
                    current_value=Decimal(str(100 - config_metrics.stability_score)),
                    baseline_value=Decimal("20"),  # Normal instability
                    deviation=Decimal(str(80 - config_metrics.stability_score)),
                    severity=min(int((100 - config_metrics.stability_score) / 10), 10),
                    timestamp=now,
                    description=f"Configuration unstable: {config_metrics.total_changes} changes",
                )
            )

        # Check excessive order modifications
        mod_frequencies = self.modification_tracker.calculate_modification_frequency()
        mod_rate = mod_frequencies.get("5min", 0)
        if mod_rate > 3:
            anomalies.append(
                Anomaly(
                    indicator_name="order_modifications",
                    current_value=Decimal(str(mod_rate)),
                    baseline_value=Decimal("1"),  # Normal modification rate
                    deviation=Decimal(str(mod_rate - 1)),
                    severity=min(int(mod_rate * 2), 10),
                    timestamp=now,
                    description=f"Excessive modifications: {mod_rate:.1f} per minute",
                )
            )

        # Check inactivity periods
        inactivity = self.inactivity_tracker.check_inactivity()
        if inactivity and inactivity > 120:  # More than 2 minutes inactive
            anomalies.append(
                Anomaly(
                    indicator_name="inactivity",
                    current_value=Decimal(str(inactivity)),
                    baseline_value=Decimal("30"),  # Normal brief pauses
                    deviation=Decimal(str(inactivity - 30)),
                    severity=min(int(inactivity / 60), 10),
                    timestamp=now,
                    description=f"Extended inactivity: {inactivity:.0f} seconds",
                )
            )

        return anomalies

    def _determine_tilt_level(self, anomaly_count: int) -> TiltLevel:
        """Determine tilt level based on anomaly count.

        Args:
            anomaly_count: Number of detected anomalies

        Returns:
            Appropriate tilt level
        """
        if anomaly_count >= self.LEVEL3_THRESHOLD:
            return TiltLevel.LEVEL3
        elif anomaly_count >= self.LEVEL2_THRESHOLD:
            return TiltLevel.LEVEL2
        elif anomaly_count >= self.LEVEL1_THRESHOLD:
            return TiltLevel.LEVEL1
        else:
            return TiltLevel.NORMAL

    def _update_anomaly_buffer(self, profile_id: str, anomalies: list[Anomaly]) -> None:
        """Update anomaly buffer for profile.

        Args:
            profile_id: Profile identifier
            anomalies: New anomalies to add
        """
        if profile_id not in self.anomaly_buffer:
            self.anomaly_buffer[profile_id] = deque(maxlen=self.anomaly_buffer_size)

        buffer = self.anomaly_buffer[profile_id]
        for anomaly in anomalies:
            buffer.append(anomaly)

    async def _publish_tilt_event(self, result: TiltDetectionResult) -> None:
        """Publish tilt detection event.

        Args:
            result: Tilt detection result
        """
        if not self.event_bus:
            return

        # Map tilt level to event type
        event_type_map = {
            TiltLevel.LEVEL1: EventType.TILT_LEVEL1_DETECTED,
            TiltLevel.LEVEL2: EventType.TILT_LEVEL2_DETECTED,
            TiltLevel.LEVEL3: EventType.TILT_LEVEL3_DETECTED,
        }

        event_type = event_type_map.get(result.tilt_level)
        if event_type:
            await self.event_bus.publish(
                event_type,
                {
                    "profile_id": result.profile_id,
                    "tilt_level": result.tilt_level.value,
                    "tilt_score": result.tilt_score,
                    "anomaly_count": len(result.anomalies),
                    "anomalies": [
                        {
                            "indicator": a.indicator_name,
                            "severity": a.severity,
                            "deviation": float(a.deviation),
                        }
                        for a in result.anomalies
                    ],
                    "timestamp": result.timestamp.isoformat(),
                },
            )

            logger.info(
                "Tilt event published",
                profile_id=result.profile_id,
                level=result.tilt_level.value,
                score=result.tilt_score,
                anomaly_count=len(result.anomalies),
            )

    def get_anomaly_history(
        self, profile_id: str, limit: Optional[int] = None
    ) -> list[Anomaly]:
        """Get anomaly history for profile.

        Args:
            profile_id: Profile identifier
            limit: Maximum number of anomalies to return

        Returns:
            List of historical anomalies
        """
        if profile_id not in self.anomaly_buffer:
            return []

        buffer = self.anomaly_buffer[profile_id]
        if limit:
            return list(buffer)[-limit:]
        return list(buffer)

    def clear_anomaly_buffer(self, profile_id: str) -> None:
        """Clear anomaly buffer for profile.

        Args:
            profile_id: Profile identifier
        """
        if profile_id in self.anomaly_buffer:
            self.anomaly_buffer[profile_id].clear()
            logger.debug("Anomaly buffer cleared", profile_id=profile_id)


class AnomalyDetector:
    """Aggregates anomaly detection from multiple indicators."""

    def __init__(self, baseline: BehavioralBaseline):
        """Initialize anomaly detector.

        Args:
            baseline: Behavioral baseline for comparison
        """
        self.baseline = baseline
        self.indicators: dict[str, callable] = {}

    def register_indicator(self, name: str, detector: callable) -> None:
        """Register an anomaly detector for an indicator.

        Args:
            name: Indicator name
            detector: Detection function
        """
        self.indicators[name] = detector

    async def detect_anomalies(
        self, baseline: BehavioralBaseline, current: BehavioralMetric
    ) -> list[Anomaly]:
        """Detect all anomalies from current metrics.

        Args:
            baseline: Behavioral baseline
            current: Current behavioral metric

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for name, detector in self.indicators.items():
            try:
                if anomaly := await detector(baseline, current):
                    anomalies.append(anomaly)
            except Exception as e:
                logger.error(f"Error in {name} detector", error=str(e))

        return anomalies
