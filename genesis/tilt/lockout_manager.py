"""Graduated lockout management system for tilt recovery."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Optional, Any

import structlog

from genesis.core.events import EventType
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus
from genesis.tilt.detector import TiltLevel

logger = structlog.get_logger(__name__)


class LockoutStatus(Enum):
    """Status of a lockout period."""

    NO_LOCKOUT = "no_lockout"
    ACTIVE = "active"
    EXPIRED = "expired"
    OVERRIDDEN = "overridden"


@dataclass
class Lockout:
    """Represents a trading lockout period."""

    lockout_id: str
    profile_id: str
    tilt_level: TiltLevel
    duration_minutes: int
    started_at: datetime
    expires_at: datetime
    status: LockoutStatus
    occurrence_count: int  # Number of occurrences at this level
    reason: Optional[str] = None


class LockoutManager:
    """Manages graduated lockout periods for tilt recovery."""

    # Base lockout durations in minutes
    BASE_DURATIONS = {
        TiltLevel.LEVEL1: 5,  # 5 minutes base
        TiltLevel.LEVEL2: 30,  # 30 minutes base
        TiltLevel.LEVEL3: 1440,  # 24 hours base
    }

    # Maximum lockout duration (1 week in minutes)
    MAX_LOCKOUT_MINUTES = 10080

    def __init__(
        self,
        repository: Optional[SQLiteRepository] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize lockout manager.

        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing lockout events
        """
        self.repository = repository
        self.event_bus = event_bus

        # In-memory cache of active lockouts
        self.active_lockouts: dict[str, Lockout] = {}

        # Track occurrence counts per profile and level
        self.occurrence_counts: dict[str, dict[TiltLevel, int]] = {}

    def calculate_lockout_duration(
        self, tilt_level: TiltLevel, occurrence: int
    ) -> int:
        """Calculate lockout duration based on level and occurrence count.

        Each repeat occurrence increases duration by 50%.

        Args:
            tilt_level: Severity level of tilt
            occurrence: Number of times this level has been triggered

        Returns:
            Lockout duration in minutes
        """
        if tilt_level not in self.BASE_DURATIONS:
            logger.warning(
                "Unknown tilt level, using default duration",
                tilt_level=tilt_level.value,
            )
            return 5  # Default 5 minutes

        base_duration = self.BASE_DURATIONS[tilt_level]

        # Increase by 50% for each repeat occurrence
        duration = base_duration * (1.5 ** (occurrence - 1))

        # Cap at maximum duration
        return min(int(duration), self.MAX_LOCKOUT_MINUTES)

    async def enforce_lockout(
        self,
        profile_id: str,
        tilt_level: TiltLevel,
        reason: Optional[str] = None,
    ) -> Lockout:
        """Enforce a lockout period for a profile.

        Args:
            profile_id: Profile identifier
            tilt_level: Severity level triggering lockout
            reason: Optional reason for lockout

        Returns:
            Created lockout object
        """
        # Update occurrence count
        if profile_id not in self.occurrence_counts:
            self.occurrence_counts[profile_id] = {}

        if tilt_level not in self.occurrence_counts[profile_id]:
            self.occurrence_counts[profile_id][tilt_level] = 0

        self.occurrence_counts[profile_id][tilt_level] += 1
        occurrence = self.occurrence_counts[profile_id][tilt_level]

        # Calculate duration
        duration_minutes = self.calculate_lockout_duration(tilt_level, occurrence)

        # Create lockout
        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=duration_minutes)

        lockout = Lockout(
            lockout_id=f"{profile_id}_{now.timestamp()}",
            profile_id=profile_id,
            tilt_level=tilt_level,
            duration_minutes=duration_minutes,
            started_at=now,
            expires_at=expires_at,
            status=LockoutStatus.ACTIVE,
            occurrence_count=occurrence,
            reason=reason,
        )

        # Store in cache
        self.active_lockouts[profile_id] = lockout

        # Persist to database
        if self.repository:
            await self._persist_lockout(lockout)

        # Publish event
        if self.event_bus:
            await self._publish_lockout_event(lockout)

        logger.info(
            "Trading lockout enforced",
            profile_id=profile_id,
            level=tilt_level.value,
            duration_minutes=duration_minutes,
            occurrence=occurrence,
            expires_at=expires_at.isoformat(),
        )

        return lockout

    def check_lockout_status(self, profile_id: str) -> LockoutStatus:
        """Check the current lockout status for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Current lockout status
        """
        if profile_id not in self.active_lockouts:
            return LockoutStatus.NO_LOCKOUT

        lockout = self.active_lockouts[profile_id]
        now = datetime.now(UTC)

        # Check if expired
        if lockout.expires_at <= now:
            lockout.status = LockoutStatus.EXPIRED
            # Remove from active cache
            del self.active_lockouts[profile_id]
            return LockoutStatus.EXPIRED

        return lockout.status

    def get_active_lockout(self, profile_id: str) -> Optional[Lockout]:
        """Get active lockout for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Active lockout or None
        """
        status = self.check_lockout_status(profile_id)

        if status == LockoutStatus.ACTIVE:
            return self.active_lockouts.get(profile_id)

        return None

    def get_remaining_minutes(self, profile_id: str) -> int:
        """Get remaining lockout minutes for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Remaining minutes or 0 if no active lockout
        """
        lockout = self.get_active_lockout(profile_id)

        if not lockout:
            return 0

        now = datetime.now(UTC)
        remaining = lockout.expires_at - now
        return max(0, int(remaining.total_seconds() / 60))

    async def release_lockout(self, profile_id: str) -> bool:
        """Manually release a lockout (for emergency override).

        Args:
            profile_id: Profile identifier

        Returns:
            True if lockout was released
        """
        if profile_id not in self.active_lockouts:
            return False

        lockout = self.active_lockouts[profile_id]
        lockout.status = LockoutStatus.OVERRIDDEN

        # Remove from active cache
        del self.active_lockouts[profile_id]

        # Update database
        if self.repository:
            await self._update_lockout_status(lockout)

        # Publish event
        if self.event_bus:
            await self._publish_lockout_release_event(lockout)

        logger.info(
            "Trading lockout manually released",
            profile_id=profile_id,
            lockout_id=lockout.lockout_id,
        )

        return True

    def reset_occurrence_count(
        self, profile_id: str, tilt_level: Optional[TiltLevel] = None
    ) -> None:
        """Reset occurrence count for a profile.

        Args:
            profile_id: Profile identifier
            tilt_level: Specific level to reset, or None for all levels
        """
        if profile_id not in self.occurrence_counts:
            return

        if tilt_level:
            if tilt_level in self.occurrence_counts[profile_id]:
                self.occurrence_counts[profile_id][tilt_level] = 0
                logger.debug(
                    "Occurrence count reset",
                    profile_id=profile_id,
                    level=tilt_level.value,
                )
        else:
            self.occurrence_counts[profile_id] = {}
            logger.debug(
                "All occurrence counts reset",
                profile_id=profile_id,
            )

    async def load_active_lockouts(self) -> None:
        """Load active lockouts from database on startup."""
        if not self.repository:
            return

        try:
            # Query active lockouts from database
            active_lockouts = await self.repository.get_active_lockouts()

            for lockout_data in active_lockouts:
                lockout = self._lockout_from_dict(lockout_data)
                if lockout.status == LockoutStatus.ACTIVE:
                    self.active_lockouts[lockout.profile_id] = lockout

            logger.info(
                "Active lockouts loaded",
                count=len(self.active_lockouts),
            )
        except Exception as e:
            logger.error(
                "Failed to load active lockouts",
                error=str(e),
            )

    async def _persist_lockout(self, lockout: Lockout) -> None:
        """Persist lockout to database.

        Args:
            lockout: Lockout to persist
        """
        if not self.repository:
            return

        try:
            await self.repository.save_lockout(
                {
                    "lockout_id": lockout.lockout_id,
                    "profile_id": lockout.profile_id,
                    "tilt_level": lockout.tilt_level.value,
                    "duration_minutes": lockout.duration_minutes,
                    "started_at": lockout.started_at.isoformat(),
                    "expires_at": lockout.expires_at.isoformat(),
                    "status": lockout.status.value,
                    "occurrence_count": lockout.occurrence_count,
                    "reason": lockout.reason,
                }
            )
        except Exception as e:
            logger.error(
                "Failed to persist lockout",
                lockout_id=lockout.lockout_id,
                error=str(e),
            )

    async def _update_lockout_status(self, lockout: Lockout) -> None:
        """Update lockout status in database.

        Args:
            lockout: Lockout to update
        """
        if not self.repository:
            return

        try:
            await self.repository.update_lockout_status(
                lockout.lockout_id,
                lockout.status.value,
            )
        except Exception as e:
            logger.error(
                "Failed to update lockout status",
                lockout_id=lockout.lockout_id,
                error=str(e),
            )

    async def _publish_lockout_event(self, lockout: Lockout) -> None:
        """Publish lockout enforcement event.

        Args:
            lockout: Enforced lockout
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.TRADING_LOCKOUT if EventType.TRADING_LOCKOUT else "TRADING_LOCKOUT",
            {
                "profile_id": lockout.profile_id,
                "lockout_id": lockout.lockout_id,
                "tilt_level": lockout.tilt_level.value,
                "duration_minutes": lockout.duration_minutes,
                "expires_at": lockout.expires_at.isoformat(),
                "occurrence_count": lockout.occurrence_count,
                "reason": lockout.reason,
                "timestamp": lockout.started_at.isoformat(),
            },
        )

    async def _publish_lockout_release_event(self, lockout: Lockout) -> None:
        """Publish lockout release event.

        Args:
            lockout: Released lockout
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.LOCKOUT_EXPIRED if EventType.LOCKOUT_EXPIRED else "LOCKOUT_EXPIRED",
            {
                "profile_id": lockout.profile_id,
                "lockout_id": lockout.lockout_id,
                "released_at": datetime.now(UTC).isoformat(),
            },
        )

    def _lockout_from_dict(self, data: dict[str, Any]) -> Lockout:
        """Create lockout from dictionary.

        Args:
            data: Lockout data

        Returns:
            Lockout object
        """
        return Lockout(
            lockout_id=data["lockout_id"],
            profile_id=data["profile_id"],
            tilt_level=TiltLevel(data["tilt_level"]),
            duration_minutes=data["duration_minutes"],
            started_at=datetime.fromisoformat(data["started_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            status=LockoutStatus(data["status"]),
            occurrence_count=data.get("occurrence_count", 1),
            reason=data.get("reason"),
        )
