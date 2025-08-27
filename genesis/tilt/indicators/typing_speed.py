from typing import Optional
"""Typing speed behavioral indicator for tilt detection."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class KeystrokeEvent:
    """Represents a keystroke event."""

    timestamp: datetime
    key_count: int
    time_between_keys_ms: float


class TypingSpeedIndicator:
    """Monitors typing speed patterns for behavioral analysis."""

    def __init__(
        self,
        window_size: int = 100,
        burst_threshold_wpm: int = 120,
        slow_threshold_wpm: int = 20
    ):
        """Initialize typing speed indicator.
        
        Args:
            window_size: Number of keystroke events to track
            burst_threshold_wpm: Words per minute threshold for burst detection
            slow_threshold_wpm: Words per minute threshold for slow typing
        """
        self.window_size = window_size
        self.burst_threshold_wpm = burst_threshold_wpm
        self.slow_threshold_wpm = slow_threshold_wpm

        # Track keystroke events
        self.keystroke_events = deque(maxlen=window_size)

        # Track typing bursts
        self.burst_count = 0
        self.last_burst_time: Optional[datetime] = None

        # Average characters per word for WPM calculation
        self.chars_per_word = 5

    def record_keystroke_event(
        self,
        key_count: int,
        duration_ms: float,
        timestamp: Optional[datetime] = None
    ) -> dict:
        """Record a keystroke event and analyze patterns.
        
        Args:
            key_count: Number of keys pressed
            duration_ms: Time taken for the keystrokes
            timestamp: Event timestamp
            
        Returns:
            Analysis of typing patterns
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Calculate time between keys
        time_between_keys = duration_ms / key_count if key_count > 0 else 0

        event = KeystrokeEvent(
            timestamp=timestamp,
            key_count=key_count,
            time_between_keys_ms=time_between_keys
        )

        self.keystroke_events.append(event)

        # Analyze patterns
        analysis = self.analyze_typing_patterns()

        # Check for bursts
        wpm = self._calculate_wpm(key_count, duration_ms)
        if wpm > self.burst_threshold_wpm:
            self.burst_count += 1
            self.last_burst_time = timestamp
            logger.debug(
                "Typing burst detected",
                wpm=wpm,
                burst_count=self.burst_count
            )

        return analysis

    def analyze_typing_patterns(self) -> dict:
        """Analyze typing patterns for anomalies.
        
        Returns:
            Dictionary with typing pattern analysis
        """
        if not self.keystroke_events:
            return {
                "has_data": False,
                "sample_count": 0
            }

        if len(self.keystroke_events) < 3:
            return {
                "has_data": True,
                "sample_count": len(self.keystroke_events),
                "insufficient_data": True
            }

        # Calculate typing speeds
        speeds_wpm = []
        for event in self.keystroke_events:
            if event.key_count > 0 and event.time_between_keys_ms > 0:
                duration_ms = event.key_count * event.time_between_keys_ms
                wpm = self._calculate_wpm(event.key_count, duration_ms)
                speeds_wpm.append(wpm)

        if not speeds_wpm:
            return {
                "has_data": True,
                "sample_count": len(self.keystroke_events),
                "no_valid_speeds": True
            }

        # Calculate statistics
        avg_wpm = sum(speeds_wpm) / len(speeds_wpm)
        max_wpm = max(speeds_wpm)
        min_wpm = min(speeds_wpm)

        # Calculate variance
        variance = sum((s - avg_wpm) ** 2 for s in speeds_wpm) / len(speeds_wpm)
        std_dev = variance ** 0.5

        # Detect patterns
        burst_detected = max_wpm > self.burst_threshold_wpm
        slow_detected = min_wpm < self.slow_threshold_wpm
        erratic = std_dev > 30  # High variance indicates erratic typing

        # Check for recent acceleration
        if len(speeds_wpm) >= 10:
            recent_5 = speeds_wpm[-5:]
            older = speeds_wpm[:-5]
            recent_avg = sum(recent_5) / len(recent_5)
            older_avg = sum(older) / len(older)

            acceleration = "increasing" if recent_avg > older_avg * 1.3 else \
                         "decreasing" if recent_avg < older_avg * 0.7 else \
                         "stable"
        else:
            acceleration = "insufficient_data"

        return {
            "has_data": True,
            "sample_count": len(self.keystroke_events),
            "avg_wpm": avg_wpm,
            "max_wpm": max_wpm,
            "min_wpm": min_wpm,
            "std_dev": std_dev,
            "burst_detected": burst_detected,
            "slow_detected": slow_detected,
            "erratic_typing": erratic,
            "acceleration": acceleration,
            "burst_count": self.burst_count,
            "anomaly_score": self._calculate_anomaly_score(
                avg_wpm, std_dev, burst_detected, erratic
            )
        }

    def _calculate_wpm(self, key_count: int, duration_ms: float) -> float:
        """Calculate words per minute from keystroke data.
        
        Args:
            key_count: Number of keys pressed
            duration_ms: Duration in milliseconds
            
        Returns:
            Words per minute
        """
        if duration_ms <= 0:
            return 0

        # Convert to minutes
        duration_minutes = duration_ms / (1000 * 60)
        if duration_minutes <= 0:
            return 0

        # Calculate words (assuming average word length)
        words = key_count / self.chars_per_word

        # Calculate WPM
        wpm = words / duration_minutes

        return wpm

    def _calculate_anomaly_score(
        self,
        avg_wpm: float,
        std_dev: float,
        burst_detected: bool,
        erratic: bool
    ) -> int:
        """Calculate anomaly score for typing patterns.
        
        Args:
            avg_wpm: Average words per minute
            std_dev: Standard deviation of speeds
            burst_detected: Whether burst typing was detected
            erratic: Whether typing is erratic
            
        Returns:
            Anomaly score from 0 to 100
        """
        score = 0

        # Extreme speeds (too fast or too slow)
        if avg_wpm > 100:
            score += 20
        elif avg_wpm < 30:
            score += 15

        # High variance
        if std_dev > 30:
            score += min(int(std_dev), 30)

        # Burst typing
        if burst_detected:
            score += 25

        # Erratic patterns
        if erratic:
            score += 25

        # Multiple bursts in short time
        if self.burst_count > 3:
            score += min(self.burst_count * 5, 20)

        return min(score, 100)

    def detect_stress_typing(self) -> Optional[dict]:
        """Detect stress-induced typing patterns.
        
        Returns:
            Stress detection result if found, None otherwise
        """
        analysis = self.analyze_typing_patterns()

        if not analysis.get("has_data") or analysis.get("insufficient_data"):
            return None

        # Check for stress indicators
        stress_indicators = []

        if analysis.get("burst_detected"):
            stress_indicators.append("rapid_bursts")

        if analysis.get("erratic_typing"):
            stress_indicators.append("erratic_pattern")

        if analysis.get("acceleration") == "increasing":
            stress_indicators.append("accelerating_speed")

        if analysis.get("anomaly_score", 0) > 60:
            stress_indicators.append("high_anomaly_score")

        if stress_indicators:
            return {
                "stress_detected": True,
                "indicators": stress_indicators,
                "anomaly_score": analysis.get("anomaly_score", 0),
                "avg_wpm": analysis.get("avg_wpm", 0),
                "severity": min(len(stress_indicators) * 3, 10)
            }

        return None

    def reset(self) -> None:
        """Reset the indicator state."""
        self.keystroke_events.clear()
        self.burst_count = 0
        self.last_burst_time = None
        logger.info("Typing speed indicator reset")
