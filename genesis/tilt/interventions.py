from __future__ import annotations

"""Tilt intervention strategies and management."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.events import EventType
from genesis.engine.event_bus import EventBus
from genesis.tilt.detector import TiltLevel

logger = structlog.get_logger(__name__)


class InterventionType(Enum):
    """Types of interventions."""

    MESSAGE = "message"  # Supportive message
    POSITION_REDUCTION = "position_reduction"  # Reduce position sizes
    TRADING_LOCKOUT = "trading_lockout"  # Temporary trading ban
    COOLDOWN = "cooldown"  # Forced break period


@dataclass
class Intervention:
    """Represents an intervention action."""

    intervention_id: str
    profile_id: str
    tilt_level: TiltLevel
    intervention_type: InterventionType
    message: str
    applied_at: datetime
    expires_at: datetime | None
    position_size_multiplier: Decimal | None
    is_active: bool


class InterventionManager:
    """Manages tilt interventions with progressive messaging."""

    # Intervention messages by level
    LEVEL1_MESSAGES = [
        "Taking a moment to breathe can improve your trading decisions.",
        "Your focus is your edge. A brief pause helps maintain clarity.",
        "Market opportunities are constant. Patience creates better entries.",
        "Consider reviewing your trading plan before the next position.",
    ]

    LEVEL2_MESSAGES = [
        "Your trading patterns suggest heightened stress. Position sizes reduced for safety.",
        "Risk management protocol activated. Position sizing adjusted to protect capital.",
        "Emotional state detected. Trading with reduced exposure for better control.",
        "Safety measures engaged. Smaller positions help maintain discipline.",
    ]

    LEVEL3_MESSAGES = [
        "Let's take a break. Trading paused to protect your capital.",
        "Trading lockout activated. This pause is protecting your account.",
        "Time for a reset. Your future self will thank you for this break.",
        "Account protection mode. Trading will resume after cooldown period.",
    ]

    def __init__(
        self,
        event_bus: EventBus | None = None,
        cooldown_minutes: dict[TiltLevel, int] = None,
        emergency_contact_enabled: bool = False,
        emergency_contact_info: dict[str, str] | None = None,
    ):
        """Initialize intervention manager.

        Args:
            event_bus: Event bus for publishing intervention events
            cooldown_minutes: Cooldown periods by tilt level
            emergency_contact_enabled: Whether emergency contact is enabled (default: False)
            emergency_contact_info: Emergency contact information (for future implementation)
        """
        self.event_bus = event_bus
        self.cooldown_minutes = cooldown_minutes or {
            TiltLevel.LEVEL1: 5,
            TiltLevel.LEVEL2: 15,
            TiltLevel.LEVEL3: 30,
        }

        # Track active interventions per profile
        self.active_interventions: dict[str, list[Intervention]] = {}

        # Message rotation tracking
        self.message_indices: dict[str, dict[TiltLevel, int]] = {}

        # Emergency contact configuration (placeholder for future implementation)
        self.emergency_contact_enabled = emergency_contact_enabled
        self.emergency_contact_info = emergency_contact_info or {}

    def get_intervention_message(self, level: TiltLevel) -> str:
        """Get appropriate intervention message for tilt level.

        Args:
            level: Tilt severity level

        Returns:
            Supportive intervention message
        """
        if level == TiltLevel.NORMAL:
            return ""
        elif level == TiltLevel.LEVEL1:
            messages = self.LEVEL1_MESSAGES
        elif level == TiltLevel.LEVEL2:
            messages = self.LEVEL2_MESSAGES
        elif level == TiltLevel.LEVEL3:
            messages = self.LEVEL3_MESSAGES
        else:
            return "Please review your trading state."

        # Rotate through messages to avoid repetition
        return messages[0]  # Simple rotation for now

    async def apply_intervention(
        self, profile_id: str, level: TiltLevel, tilt_score: int
    ) -> Intervention:
        """Apply intervention based on tilt level.

        Args:
            profile_id: Profile identifier
            level: Tilt severity level
            tilt_score: Current tilt score

        Returns:
            Applied intervention
        """
        # Get appropriate message
        message = self.get_intervention_message(level)

        # Determine intervention type and parameters
        intervention_type, params = self._determine_intervention_params(level)

        # Create intervention
        now = datetime.now(UTC)
        cooldown = self.cooldown_minutes.get(level, 5)

        intervention = Intervention(
            intervention_id=f"{profile_id}_{now.timestamp()}",
            profile_id=profile_id,
            tilt_level=level,
            intervention_type=intervention_type,
            message=message,
            applied_at=now,
            expires_at=now + timedelta(minutes=cooldown) if cooldown > 0 else None,
            position_size_multiplier=params.get("position_size_multiplier"),
            is_active=True,
        )

        # Store intervention
        if profile_id not in self.active_interventions:
            self.active_interventions[profile_id] = []
        self.active_interventions[profile_id].append(intervention)

        # Apply intervention effects
        await self._apply_intervention_effects(intervention)

        # Publish event
        if self.event_bus:
            await self._publish_intervention_event(intervention, tilt_score)

        logger.info(
            "Intervention applied",
            profile_id=profile_id,
            level=level.value,
            type=intervention_type.value,
            message=message[:50],
        )

        return intervention

    def _determine_intervention_params(
        self, level: TiltLevel
    ) -> tuple[InterventionType, dict]:
        """Determine intervention type and parameters based on level.

        Args:
            level: Tilt severity level

        Returns:
            Intervention type and parameters
        """
        if level == TiltLevel.LEVEL1:
            # Level 1: Just supportive messaging
            return InterventionType.MESSAGE, {}

        elif level == TiltLevel.LEVEL2:
            # Level 2: Message + position size reduction
            return InterventionType.POSITION_REDUCTION, {
                "position_size_multiplier": Decimal("0.5")  # 50% reduction
            }

        elif level == TiltLevel.LEVEL3:
            # Level 3: Full trading lockout
            return InterventionType.TRADING_LOCKOUT, {
                "position_size_multiplier": Decimal("0")  # No trading
            }

        else:
            return InterventionType.MESSAGE, {}

    async def _apply_intervention_effects(self, intervention: Intervention) -> None:
        """Apply the effects of an intervention.

        Args:
            intervention: Intervention to apply
        """
        if intervention.intervention_type == InterventionType.POSITION_REDUCTION:
            # Position reduction logic would be implemented here
            # This would interact with the risk engine
            logger.debug(
                "Position size reduction applied",
                profile_id=intervention.profile_id,
                multiplier=float(intervention.position_size_multiplier),
            )

        elif intervention.intervention_type == InterventionType.TRADING_LOCKOUT:
            # Trading lockout logic would be implemented here
            # This would interact with the order executor
            logger.debug(
                "Trading lockout applied",
                profile_id=intervention.profile_id,
                expires_at=(
                    intervention.expires_at.isoformat()
                    if intervention.expires_at
                    else None
                ),
            )

    async def _publish_intervention_event(
        self, intervention: Intervention, tilt_score: int
    ) -> None:
        """Publish intervention event.

        Args:
            intervention: Applied intervention
            tilt_score: Current tilt score
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.INTERVENTION_APPLIED,
            {
                "profile_id": intervention.profile_id,
                "intervention_id": intervention.intervention_id,
                "tilt_level": intervention.tilt_level.value,
                "intervention_type": intervention.intervention_type.value,
                "message": intervention.message,
                "tilt_score": tilt_score,
                "position_size_multiplier": (
                    float(intervention.position_size_multiplier)
                    if intervention.position_size_multiplier
                    else None
                ),
                "expires_at": (
                    intervention.expires_at.isoformat()
                    if intervention.expires_at
                    else None
                ),
                "timestamp": intervention.applied_at.isoformat(),
            },
        )

    def get_active_interventions(self, profile_id: str) -> list[Intervention]:
        """Get active interventions for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            List of active interventions
        """
        if profile_id not in self.active_interventions:
            return []

        now = datetime.now(UTC)
        active = []

        for intervention in self.active_interventions[profile_id]:
            # Check if intervention has expired
            if intervention.expires_at and intervention.expires_at <= now:
                intervention.is_active = False

            if intervention.is_active:
                active.append(intervention)

        return active

    def get_position_size_multiplier(self, profile_id: str) -> Decimal:
        """Get current position size multiplier for profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Position size multiplier (1.0 if no reduction)
        """
        active = self.get_active_interventions(profile_id)

        if not active:
            return Decimal("1.0")

        # Return the most restrictive multiplier
        multipliers = [
            i.position_size_multiplier
            for i in active
            if i.position_size_multiplier is not None
        ]

        if multipliers:
            return min(multipliers)

        return Decimal("1.0")

    def is_trading_locked(self, profile_id: str) -> bool:
        """Check if trading is locked for profile.

        Args:
            profile_id: Profile identifier

        Returns:
            True if trading is locked
        """
        active = self.get_active_interventions(profile_id)

        for intervention in active:
            if intervention.intervention_type == InterventionType.TRADING_LOCKOUT:
                return True

        return False

    def clear_interventions(self, profile_id: str) -> None:
        """Clear all interventions for a profile.

        Args:
            profile_id: Profile identifier
        """
        if profile_id in self.active_interventions:
            for intervention in self.active_interventions[profile_id]:
                intervention.is_active = False

            logger.info("Interventions cleared", profile_id=profile_id)

    async def check_recovery(self, profile_id: str) -> bool:
        """Check if profile has recovered from tilt.

        Args:
            profile_id: Profile identifier

        Returns:
            True if recovered
        """
        active = self.get_active_interventions(profile_id)

        # No active interventions means recovered
        if not active:
            return True

        # Check if all interventions have expired
        now = datetime.now(UTC)
        all_expired = all(i.expires_at and i.expires_at <= now for i in active)

        if all_expired:
            # Clear expired interventions
            self.clear_interventions(profile_id)

            # Publish recovery event
            if self.event_bus:
                await self.event_bus.publish(
                    EventType.TILT_RECOVERED,
                    {"profile_id": profile_id, "timestamp": now.isoformat()},
                )

            logger.info("Tilt recovery detected", profile_id=profile_id)
            return True

        return False

    async def notify_emergency_contact(
        self, profile_id: str, tilt_level: TiltLevel, message: str | None = None
    ) -> bool:
        """Notify emergency contact about severe tilt episode (placeholder).

        This is a placeholder for future implementation of emergency contact
        notification. When implemented, this could:
        - Send SMS/email to designated contact
        - Include account status and tilt severity
        - Provide recovery instructions
        - Log notification for audit trail

        Args:
            profile_id: Profile identifier
            tilt_level: Severity level of tilt
            message: Optional custom message

        Returns:
            True if notification would be sent (always False in placeholder)
        """
        if not self.emergency_contact_enabled:
            logger.debug("Emergency contact not enabled", profile_id=profile_id)
            return False

        if not self.emergency_contact_info:
            logger.warning(
                "Emergency contact enabled but no contact info configured",
                profile_id=profile_id,
            )
            return False

        # FUTURE IMPLEMENTATION:
        # - Validate contact information
        # - Check notification rate limits
        # - Format notification message
        # - Send via configured channel (SMS/email/webhook)
        # - Store notification record in database
        # - Handle delivery confirmation

        logger.info(
            "Emergency contact notification (placeholder)",
            profile_id=profile_id,
            tilt_level=tilt_level.value,
            enabled=self.emergency_contact_enabled,
            has_contact_info=bool(self.emergency_contact_info),
            message="Feature not yet implemented",
        )

        # Placeholder always returns False (not actually sent)
        return False
