"""
Focus and attention pattern detection for tilt monitoring.

Tracks window focus changes and tab-switching patterns that may
indicate distraction, anxiety, or loss of concentration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FocusMetrics:
    """Metrics for focus and attention patterns."""

    total_switches: int
    switch_frequency: float  # Switches per minute
    average_focus_duration: float  # Seconds
    longest_focus: float  # Seconds
    shortest_focus: float  # Seconds
    rapid_switch_count: int  # Switches within threshold
    distraction_score: float  # 0-100 scale


class FocusPatternDetector:
    """
    Detects patterns in window focus and tab switching.
    
    Monitors when the trading application loses/gains focus
    and identifies patterns that may indicate psychological states.
    """

    def __init__(
        self,
        rapid_switch_threshold_ms: int = 3000,
        window_size: int = 100
    ) -> None:
        """
        Initialize focus pattern detector.
        
        Args:
            rapid_switch_threshold_ms: Time threshold for rapid switching
            window_size: Size of rolling window for pattern analysis
        """
        self.rapid_switch_threshold = timedelta(milliseconds=rapid_switch_threshold_ms)
        self.window_size = window_size

        # Focus event history
        self.focus_events: deque[dict] = deque(maxlen=window_size * 2)

        # Current focus state
        self.window_active = True
        self.last_focus_change = datetime.utcnow()
        self.current_focus_start = datetime.utcnow()

        # Focus duration tracking
        self.focus_durations: deque[float] = deque(maxlen=window_size)
        self.unfocus_durations: deque[float] = deque(maxlen=window_size)

        # Rapid switching detection
        self.rapid_switches: list[datetime] = []

        logger.info(
            "focus_pattern_detector_initialized",
            rapid_switch_threshold_ms=rapid_switch_threshold_ms,
            window_size=window_size
        )

    def track_window_focus(
        self,
        window_active: bool,
        duration_ms: int
    ) -> None:
        """
        Track window focus change event.
        
        Args:
            window_active: True if window gained focus, False if lost
            duration_ms: Duration since last focus change in milliseconds
        """
        now = datetime.utcnow()
        duration_seconds = duration_ms / 1000.0

        # Record focus event
        event = {
            "timestamp": now,
            "window_active": window_active,
            "duration_ms": duration_ms,
            "duration_seconds": duration_seconds
        }
        self.focus_events.append(event)

        # Update duration tracking
        if window_active:
            # Was unfocused, now focused
            if self.unfocus_durations.maxlen and duration_seconds > 0:
                self.unfocus_durations.append(duration_seconds)
        else:
            # Was focused, now unfocused
            if self.focus_durations.maxlen and duration_seconds > 0:
                self.focus_durations.append(duration_seconds)

        # Check for rapid switching
        time_since_last = now - self.last_focus_change
        if time_since_last < self.rapid_switch_threshold:
            self.rapid_switches.append(now)
            logger.warning(
                "rapid_focus_switching_detected",
                time_since_last_ms=time_since_last.total_seconds() * 1000,
                threshold_ms=self.rapid_switch_threshold.total_seconds() * 1000
            )

        # Update state
        self.window_active = window_active
        self.last_focus_change = now
        if window_active:
            self.current_focus_start = now

        # Clean old rapid switches (keep last hour)
        cutoff = now - timedelta(hours=1)
        self.rapid_switches = [
            ts for ts in self.rapid_switches
            if ts > cutoff
        ]

    def get_focus_metrics(self, window_minutes: int = 5) -> FocusMetrics:
        """
        Calculate focus pattern metrics for recent window.
        
        Args:
            window_minutes: Time window to analyze in minutes
            
        Returns:
            FocusMetrics with calculated statistics
        """
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

        # Filter recent events
        recent_events = [
            e for e in self.focus_events
            if e["timestamp"] > cutoff
        ]

        if not recent_events:
            return FocusMetrics(
                total_switches=0,
                switch_frequency=0.0,
                average_focus_duration=0.0,
                longest_focus=0.0,
                shortest_focus=0.0,
                rapid_switch_count=0,
                distraction_score=0.0
            )

        # Calculate switch frequency
        total_switches = len(recent_events)
        switch_frequency = total_switches / window_minutes

        # Calculate focus duration statistics
        avg_focus = 0.0
        longest_focus = 0.0
        shortest_focus = float('inf')

        if self.focus_durations:
            avg_focus = sum(self.focus_durations) / len(self.focus_durations)
            longest_focus = max(self.focus_durations)
            shortest_focus = min(self.focus_durations)

        # Count rapid switches in window
        rapid_switch_count = sum(
            1 for ts in self.rapid_switches
            if ts > cutoff
        )

        # Calculate distraction score (0-100)
        distraction_score = self._calculate_distraction_score(
            switch_frequency,
            rapid_switch_count,
            avg_focus
        )

        return FocusMetrics(
            total_switches=total_switches,
            switch_frequency=switch_frequency,
            average_focus_duration=avg_focus,
            longest_focus=longest_focus,
            shortest_focus=shortest_focus if shortest_focus != float('inf') else 0.0,
            rapid_switch_count=rapid_switch_count,
            distraction_score=distraction_score
        )

    def _calculate_distraction_score(
        self,
        switch_frequency: float,
        rapid_switches: int,
        avg_focus_duration: float
    ) -> float:
        """
        Calculate distraction score from focus metrics.
        
        Args:
            switch_frequency: Switches per minute
            rapid_switches: Number of rapid switches
            avg_focus_duration: Average focus duration in seconds
            
        Returns:
            Distraction score from 0 (focused) to 100 (highly distracted)
        """
        score = 0.0

        # Penalize high switch frequency (>2 per minute is concerning)
        if switch_frequency > 2:
            score += min(30, (switch_frequency - 2) * 10)

        # Penalize rapid switches heavily
        score += min(40, rapid_switches * 10)

        # Penalize short average focus (<30 seconds is problematic)
        if avg_focus_duration > 0 and avg_focus_duration < 30:
            score += min(30, (30 - avg_focus_duration))

        return min(100.0, score)

    def is_distracted(self, threshold: float = 50.0) -> bool:
        """
        Check if user appears distracted based on focus patterns.
        
        Args:
            threshold: Distraction score threshold
            
        Returns:
            True if distraction score exceeds threshold
        """
        metrics = self.get_focus_metrics()
        return metrics.distraction_score > threshold

    def get_pattern_analysis(self) -> dict[str, any]:
        """
        Analyze focus patterns for behavioral insights.
        
        Returns:
            Dictionary with pattern analysis results
        """
        metrics_5min = self.get_focus_metrics(5)
        metrics_15min = self.get_focus_metrics(15)

        # Detect pattern trends
        increasing_distraction = (
            metrics_5min.distraction_score > metrics_15min.distraction_score * 1.2
        )

        # Classify attention state
        attention_state = "focused"
        if metrics_5min.distraction_score > 70:
            attention_state = "highly_distracted"
        elif metrics_5min.distraction_score > 40:
            attention_state = "moderately_distracted"
        elif metrics_5min.distraction_score > 20:
            attention_state = "slightly_distracted"

        return {
            "attention_state": attention_state,
            "distraction_score": metrics_5min.distraction_score,
            "switch_frequency": metrics_5min.switch_frequency,
            "rapid_switches": metrics_5min.rapid_switch_count,
            "average_focus_seconds": metrics_5min.average_focus_duration,
            "trend": "worsening" if increasing_distraction else "stable",
            "recommendation": self._get_recommendation(metrics_5min)
        }

    def _get_recommendation(self, metrics: FocusMetrics) -> str:
        """
        Generate recommendation based on focus metrics.
        
        Args:
            metrics: Current focus metrics
            
        Returns:
            Recommendation string
        """
        if metrics.distraction_score > 70:
            return "Take a break - high distraction detected"
        elif metrics.rapid_switch_count > 3:
            return "Slow down - rapid switching indicates anxiety"
        elif metrics.average_focus_duration < 20:
            return "Practice focus - short attention spans detected"
        else:
            return "Focus patterns normal"
