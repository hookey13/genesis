from __future__ import annotations

"""Mouse pattern behavioral indicator for tilt detection."""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MouseEvent:
    """Represents a mouse event."""

    timestamp: datetime
    event_type: str  # click, move, scroll
    position: tuple[int, int] | None  # (x, y) coordinates
    velocity: float | None  # pixels per second for moves
    click_duration_ms: float | None  # For click events


class MousePatternsIndicator:
    """Monitors mouse movement and click patterns for behavioral analysis."""

    def __init__(
        self,
        window_size: int = 100,
        rapid_click_threshold_ms: int = 200,
        jitter_threshold_pixels: int = 5,
    ):
        """Initialize mouse patterns indicator.

        Args:
            window_size: Number of mouse events to track
            rapid_click_threshold_ms: Time threshold for rapid clicking
            jitter_threshold_pixels: Pixel threshold for jitter detection
        """
        self.window_size = window_size
        self.rapid_click_threshold_ms = rapid_click_threshold_ms
        self.jitter_threshold_pixels = jitter_threshold_pixels

        # Track mouse events
        self.mouse_events = deque(maxlen=window_size)

        # Track click patterns
        self.click_times = deque(maxlen=50)
        self.rapid_click_count = 0
        self.double_click_count = 0

        # Track movement patterns
        self.movement_velocities = deque(maxlen=50)
        self.jitter_count = 0

    def record_click(
        self,
        position: tuple[int, int],
        duration_ms: float,
        timestamp: datetime | None = None,
    ) -> dict:
        """Record a mouse click event.

        Args:
            position: Click position (x, y)
            duration_ms: Click duration in milliseconds
            timestamp: Event timestamp

        Returns:
            Analysis of click patterns
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        event = MouseEvent(
            timestamp=timestamp,
            event_type="click",
            position=position,
            velocity=None,
            click_duration_ms=duration_ms,
        )

        self.mouse_events.append(event)
        self.click_times.append(timestamp)

        # Check for rapid clicking
        if self._detect_rapid_clicking():
            self.rapid_click_count += 1

        # Check for double clicks
        if self._detect_double_click():
            self.double_click_count += 1

        return self.analyze_patterns()

    def record_movement(
        self,
        start_pos: tuple[int, int],
        end_pos: tuple[int, int],
        duration_ms: float,
        timestamp: datetime | None = None,
    ) -> dict:
        """Record a mouse movement event.

        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            duration_ms: Movement duration in milliseconds
            timestamp: Event timestamp

        Returns:
            Analysis of movement patterns
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Calculate velocity
        distance = (
            (end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2
        ) ** 0.5
        velocity = (distance / duration_ms) * 1000 if duration_ms > 0 else 0

        event = MouseEvent(
            timestamp=timestamp,
            event_type="move",
            position=end_pos,
            velocity=velocity,
            click_duration_ms=None,
        )

        self.mouse_events.append(event)
        self.movement_velocities.append(velocity)

        # Check for jitter
        if self._detect_jitter(start_pos, end_pos, distance):
            self.jitter_count += 1

        return self.analyze_patterns()

    def analyze_patterns(self) -> dict:
        """Analyze mouse patterns for anomalies.

        Returns:
            Dictionary with mouse pattern analysis
        """
        if not self.mouse_events:
            return {"has_data": False, "sample_count": 0}

        # Separate events by type
        clicks = [e for e in self.mouse_events if e.event_type == "click"]
        moves = [e for e in self.mouse_events if e.event_type == "move"]

        # Click analysis
        click_analysis = self._analyze_clicks(clicks)

        # Movement analysis
        movement_analysis = self._analyze_movements(moves)

        # Calculate overall anomaly score
        anomaly_score = self._calculate_anomaly_score(click_analysis, movement_analysis)

        return {
            "has_data": True,
            "sample_count": len(self.mouse_events),
            "click_count": len(clicks),
            "movement_count": len(moves),
            **click_analysis,
            **movement_analysis,
            "rapid_click_count": self.rapid_click_count,
            "double_click_count": self.double_click_count,
            "jitter_count": self.jitter_count,
            "anomaly_score": anomaly_score,
        }

    def _analyze_clicks(self, clicks: list[MouseEvent]) -> dict:
        """Analyze click patterns.

        Args:
            clicks: List of click events

        Returns:
            Click pattern analysis
        """
        if not clicks:
            return {"no_clicks": True}

        # Calculate click rate
        if len(clicks) >= 2:
            time_span = (clicks[-1].timestamp - clicks[0].timestamp).total_seconds()
            click_rate = len(clicks) / time_span if time_span > 0 else 0
        else:
            click_rate = 0

        # Calculate average click duration
        durations = [c.click_duration_ms for c in clicks if c.click_duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Detect patterns
        rapid_clicking = click_rate > 5  # More than 5 clicks per second
        long_clicks = avg_duration > 500  # Clicks longer than 500ms

        return {
            "click_rate_per_second": click_rate,
            "avg_click_duration_ms": avg_duration,
            "rapid_clicking": rapid_clicking,
            "long_clicks": long_clicks,
        }

    def _analyze_movements(self, moves: list[MouseEvent]) -> dict:
        """Analyze movement patterns.

        Args:
            moves: List of movement events

        Returns:
            Movement pattern analysis
        """
        if not moves:
            return {"no_movements": True}

        velocities = [m.velocity for m in moves if m.velocity is not None]
        if not velocities:
            return {"no_valid_velocities": True}

        # Calculate statistics
        avg_velocity = sum(velocities) / len(velocities)
        max_velocity = max(velocities)
        min_velocity = min(velocities)

        # Calculate variance
        if len(velocities) > 1:
            variance = sum((v - avg_velocity) ** 2 for v in velocities) / len(
                velocities
            )
            std_dev = variance**0.5
        else:
            std_dev = 0

        # Detect patterns
        erratic_movement = std_dev > avg_velocity * 0.5 if avg_velocity > 0 else False
        very_fast = max_velocity > 2000  # pixels per second
        very_slow = min_velocity < 50 and min_velocity > 0

        return {
            "avg_velocity_pps": avg_velocity,
            "max_velocity_pps": max_velocity,
            "min_velocity_pps": min_velocity,
            "velocity_std_dev": std_dev,
            "erratic_movement": erratic_movement,
            "very_fast_movement": very_fast,
            "very_slow_movement": very_slow,
        }

    def _detect_rapid_clicking(self) -> bool:
        """Detect rapid clicking pattern.

        Returns:
            True if rapid clicking detected
        """
        if len(self.click_times) < 2:
            return False

        # Check time between last two clicks
        time_diff = (self.click_times[-1] - self.click_times[-2]).total_seconds() * 1000

        return time_diff < self.rapid_click_threshold_ms

    def _detect_double_click(self) -> bool:
        """Detect double click pattern.

        Returns:
            True if double click detected
        """
        if len(self.click_times) < 2:
            return False

        # Check if last two clicks were within double-click time
        time_diff = (self.click_times[-1] - self.click_times[-2]).total_seconds() * 1000

        return time_diff < 500  # Standard double-click time

    def _detect_jitter(
        self, start_pos: tuple[int, int], end_pos: tuple[int, int], distance: float
    ) -> bool:
        """Detect jittery movement.

        Args:
            start_pos: Starting position
            end_pos: Ending position
            distance: Distance moved

        Returns:
            True if jitter detected
        """
        # Small, rapid movements indicate jitter
        return distance < self.jitter_threshold_pixels and distance > 0

    def _calculate_anomaly_score(
        self, click_analysis: dict, movement_analysis: dict
    ) -> int:
        """Calculate overall anomaly score.

        Args:
            click_analysis: Click pattern analysis
            movement_analysis: Movement pattern analysis

        Returns:
            Anomaly score from 0 to 100
        """
        score = 0

        # Click anomalies
        if click_analysis.get("rapid_clicking"):
            score += 30
        if click_analysis.get("long_clicks"):
            score += 15
        if self.rapid_click_count > 5:
            score += min(self.rapid_click_count * 2, 20)

        # Movement anomalies
        if movement_analysis.get("erratic_movement"):
            score += 25
        if movement_analysis.get("very_fast_movement"):
            score += 20
        if self.jitter_count > 10:
            score += min(self.jitter_count, 15)

        return min(score, 100)

    def detect_stress_patterns(self) -> dict | None:
        """Detect stress-induced mouse patterns.

        Returns:
            Stress detection result if found, None otherwise
        """
        analysis = self.analyze_patterns()

        if not analysis.get("has_data"):
            return None

        stress_indicators = []

        # Check for stress indicators
        if analysis.get("rapid_clicking"):
            stress_indicators.append("rapid_clicking")

        if analysis.get("erratic_movement"):
            stress_indicators.append("erratic_movement")

        if self.jitter_count > 5:
            stress_indicators.append("mouse_jitter")

        if analysis.get("anomaly_score", 0) > 60:
            stress_indicators.append("high_anomaly_score")

        if stress_indicators:
            return {
                "stress_detected": True,
                "indicators": stress_indicators,
                "anomaly_score": analysis.get("anomaly_score", 0),
                "severity": min(len(stress_indicators) * 3, 10),
            }

        return None

    def reset(self) -> None:
        """Reset the indicator state."""
        self.mouse_events.clear()
        self.click_times.clear()
        self.movement_velocities.clear()
        self.rapid_click_count = 0
        self.double_click_count = 0
        self.jitter_count = 0
        logger.info("Mouse patterns indicator reset")
