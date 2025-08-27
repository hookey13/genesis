from typing import Optional
"""Optional meditation timer for tilt recovery."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class MeditationStatus(Enum):
    """Status of a meditation session."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class MeditationSession:
    """Represents a meditation session."""

    profile_id: str
    duration_minutes: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: MeditationStatus = MeditationStatus.NOT_STARTED
    remaining_seconds: int = 0


class MeditationTimer:
    """Optional meditation timer for recovery support."""

    # Default meditation durations
    DEFAULT_DURATIONS = [5, 10, 15, 20]  # minutes

    def __init__(self, on_tick: Optional[Callable[[int], None]] = None):
        """Initialize meditation timer.

        Args:
            on_tick: Optional callback for timer ticks (receives remaining seconds)
        """
        self.on_tick = on_tick
        self.active_sessions: dict[str, MeditationSession] = {}
        self.timer_tasks: dict[str, asyncio.Task] = {}

    async def start_meditation_session(
        self,
        profile_id: str,
        duration_minutes: int,
    ) -> MeditationSession:
        """Start a meditation session.

        Args:
            profile_id: Profile identifier
            duration_minutes: Duration in minutes

        Returns:
            Started meditation session
        """
        # Cancel any existing session
        if profile_id in self.active_sessions:
            await self.cancel_session(profile_id)

        # Create new session
        session = MeditationSession(
            profile_id=profile_id,
            duration_minutes=duration_minutes,
            started_at=datetime.now(UTC),
            status=MeditationStatus.IN_PROGRESS,
            remaining_seconds=duration_minutes * 60,
        )

        self.active_sessions[profile_id] = session

        # Start timer task
        task = asyncio.create_task(self._run_timer(session))
        self.timer_tasks[profile_id] = task

        logger.info(
            "Meditation session started",
            profile_id=profile_id,
            duration_minutes=duration_minutes,
        )

        return session

    async def _run_timer(self, session: MeditationSession) -> None:
        """Run the timer for a meditation session.

        Args:
            session: Meditation session to time
        """
        try:
            total_seconds = session.duration_minutes * 60

            for elapsed in range(total_seconds):
                if session.status != MeditationStatus.IN_PROGRESS:
                    break

                session.remaining_seconds = total_seconds - elapsed

                # Call tick callback if provided
                if self.on_tick:
                    self.on_tick(session.remaining_seconds)

                # Wait 1 second
                await asyncio.sleep(1)

            # Mark as completed if still in progress
            if session.status == MeditationStatus.IN_PROGRESS:
                session.status = MeditationStatus.COMPLETED
                session.completed_at = datetime.now(UTC)

                logger.info(
                    "Meditation session completed",
                    profile_id=session.profile_id,
                    duration_minutes=session.duration_minutes,
                )

        except asyncio.CancelledError:
            logger.debug("Meditation timer cancelled", profile_id=session.profile_id)
            raise
        except Exception as e:
            logger.error(
                "Meditation timer error",
                profile_id=session.profile_id,
                error=str(e),
            )

    async def cancel_session(self, profile_id: str) -> bool:
        """Cancel an active meditation session.

        Args:
            profile_id: Profile identifier

        Returns:
            True if session was cancelled
        """
        if profile_id not in self.active_sessions:
            return False

        session = self.active_sessions[profile_id]
        session.status = MeditationStatus.CANCELLED

        # Cancel timer task
        if profile_id in self.timer_tasks:
            task = self.timer_tasks[profile_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.timer_tasks[profile_id]

        logger.info("Meditation session cancelled", profile_id=profile_id)
        return True

    def skip_meditation(self, profile_id: str) -> MeditationSession:
        """Mark meditation as skipped (optional component).

        Args:
            profile_id: Profile identifier

        Returns:
            Skipped session record
        """
        # Cancel any active session
        if profile_id in self.active_sessions:
            session = self.active_sessions[profile_id]
            session.status = MeditationStatus.SKIPPED
        else:
            session = MeditationSession(
                profile_id=profile_id,
                duration_minutes=0,
                status=MeditationStatus.SKIPPED,
            )
            self.active_sessions[profile_id] = session

        logger.info("Meditation skipped", profile_id=profile_id)
        return session

    def get_session(self, profile_id: str) -> Optional[MeditationSession]:
        """Get current meditation session for profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Current session or None
        """
        return self.active_sessions.get(profile_id)

    def is_meditation_complete(self, profile_id: str) -> bool:
        """Check if meditation is complete or skipped.

        Args:
            profile_id: Profile identifier

        Returns:
            True if complete or skipped (optional requirement met)
        """
        session = self.get_session(profile_id)

        if not session:
            return False

        return session.status in [
            MeditationStatus.COMPLETED,
            MeditationStatus.SKIPPED,
        ]

    def get_remaining_time(self, profile_id: str) -> int:
        """Get remaining meditation time in seconds.

        Args:
            profile_id: Profile identifier

        Returns:
            Remaining seconds or 0
        """
        session = self.get_session(profile_id)

        if not session or session.status != MeditationStatus.IN_PROGRESS:
            return 0

        return session.remaining_seconds

    def format_time(self, seconds: int) -> str:
        """Format seconds into MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds <= 0:
            return "00:00"

        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    def clear_session(self, profile_id: str) -> None:
        """Clear meditation session for profile.

        Args:
            profile_id: Profile identifier
        """
        if profile_id in self.active_sessions:
            del self.active_sessions[profile_id]

        if profile_id in self.timer_tasks:
            task = self.timer_tasks[profile_id]
            task.cancel()
            del self.timer_tasks[profile_id]

    async def cleanup(self) -> None:
        """Clean up all active timers."""
        # Cancel all timer tasks
        for task in self.timer_tasks.values():
            task.cancel()

        # Wait for all tasks to complete
        if self.timer_tasks:
            await asyncio.gather(
                *self.timer_tasks.values(),
                return_exceptions=True,
            )

        self.timer_tasks.clear()
        self.active_sessions.clear()
