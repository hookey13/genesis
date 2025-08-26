"""48-hour adjustment period manager for tier transitions.

Provides a protective adjustment period after tier transitions with
reduced position limits and heightened monitoring to prevent
catastrophic losses during the adaptation phase.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from genesis.core.exceptions import ValidationError
from genesis.data.models_db import (
    AccountDB,
    AdjustmentPeriod,
    Session,
    TierTransition,
    TiltProfile,
    get_session,
)

logger = structlog.get_logger(__name__)


class AdjustmentPhase(Enum):
    """Phases of the adjustment period."""
    INITIAL = "INITIAL"  # First 12 hours - most restrictive
    EARLY = "EARLY"      # 12-24 hours - slightly relaxed
    MID = "MID"          # 24-36 hours - moderate restrictions
    LATE = "LATE"        # 36-48 hours - approaching normal
    COMPLETE = "COMPLETE"  # Period complete


# Phase timing configuration (in hours)
PHASE_DURATIONS = {
    AdjustmentPhase.INITIAL: 12,
    AdjustmentPhase.EARLY: 12,
    AdjustmentPhase.MID: 12,
    AdjustmentPhase.LATE: 12
}

# Position limit multipliers by phase
PHASE_POSITION_MULTIPLIERS = {
    AdjustmentPhase.INITIAL: Decimal('0.25'),  # 25% of normal
    AdjustmentPhase.EARLY: Decimal('0.40'),    # 40% of normal
    AdjustmentPhase.MID: Decimal('0.60'),      # 60% of normal
    AdjustmentPhase.LATE: Decimal('0.80'),     # 80% of normal
    AdjustmentPhase.COMPLETE: Decimal('1.00')  # 100% normal
}

# Monitoring sensitivity multipliers by phase
PHASE_MONITORING_MULTIPLIERS = {
    AdjustmentPhase.INITIAL: 3.0,  # 3x sensitivity
    AdjustmentPhase.EARLY: 2.5,    # 2.5x sensitivity
    AdjustmentPhase.MID: 2.0,      # 2x sensitivity
    AdjustmentPhase.LATE: 1.5,     # 1.5x sensitivity
    AdjustmentPhase.COMPLETE: 1.0  # Normal sensitivity
}


@dataclass
class AdjustmentStatus:
    """Current status of adjustment period."""
    period_id: str
    account_id: str
    transition_id: str
    current_phase: AdjustmentPhase
    hours_elapsed: float
    hours_remaining: float
    current_position_limit: Decimal
    normal_position_limit: Decimal
    monitoring_multiplier: float
    interventions_count: int
    phase_progress_percentage: float

    @property
    def is_active(self) -> bool:
        """Check if adjustment period is active."""
        return self.current_phase != AdjustmentPhase.COMPLETE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_id': self.period_id,
            'account_id': self.account_id,
            'transition_id': self.transition_id,
            'current_phase': self.current_phase.value,
            'hours_elapsed': self.hours_elapsed,
            'hours_remaining': self.hours_remaining,
            'current_position_limit': float(self.current_position_limit),
            'normal_position_limit': float(self.normal_position_limit),
            'monitoring_multiplier': self.monitoring_multiplier,
            'interventions_count': self.interventions_count,
            'phase_progress_percentage': self.phase_progress_percentage,
            'is_active': self.is_active
        }


class AdjustmentPeriodManager:
    """Manages 48-hour adjustment periods after tier transitions."""

    def __init__(self, session: Session | None = None):
        """Initialize adjustment period manager.
        
        Args:
            session: Optional database session
        """
        self.session = session or get_session()
        self._active_periods: dict[str, asyncio.Task] = {}

    async def start_adjustment_period(
        self,
        account_id: str,
        tier: str,
        duration_hours: int = 48,
        transition_id: str | None = None
    ) -> str:
        """Start an adjustment period for tier transition.
        
        Args:
            account_id: Account entering adjustment
            tier: New tier level
            duration_hours: Duration of adjustment period
            transition_id: Optional tier transition ID
            
        Returns:
            Period ID for tracking
            
        Raises:
            ValidationError: If period already active
        """
        # Check for existing active period
        existing = self._get_active_period(account_id)
        if existing:
            raise ValidationError(
                f"Adjustment period already active for account: {account_id}"
            )

        # Get account and determine limits
        account = self.session.query(AccountDB).filter_by(
            account_id=account_id
        ).first()

        if not account:
            raise ValidationError(f"Account not found: {account_id}")

        # Determine normal position limit based on tier
        normal_limits = {
            'SNIPER': Decimal('500'),
            'HUNTER': Decimal('2000'),
            'STRATEGIST': Decimal('10000'),
            'ARCHITECT': Decimal('50000'),
            'EMPEROR': Decimal('250000')
        }

        normal_limit = normal_limits.get(tier, Decimal('500'))
        initial_limit = normal_limit * PHASE_POSITION_MULTIPLIERS[AdjustmentPhase.INITIAL]

        # Create adjustment period record
        period_id = str(uuid.uuid4())
        period = AdjustmentPeriod(
            period_id=period_id,
            transition_id=transition_id or str(uuid.uuid4()),
            account_id=account_id,
            original_position_limit=normal_limit,
            reduced_position_limit=initial_limit,
            monitoring_sensitivity_multiplier=PHASE_MONITORING_MULTIPLIERS[AdjustmentPhase.INITIAL],
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=duration_hours),
            current_phase=AdjustmentPhase.INITIAL.value,
            interventions_triggered=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        try:
            self.session.add(period)

            # Update tilt profile monitoring
            await self._update_monitoring_sensitivity(account_id, PHASE_MONITORING_MULTIPLIERS[AdjustmentPhase.INITIAL])

            self.session.commit()

            # Start monitoring task
            task = asyncio.create_task(
                self._monitor_adjustment_period(period_id, duration_hours)
            )
            self._active_periods[period_id] = task

            logger.info(
                "Adjustment period started",
                period_id=period_id,
                account_id=account_id,
                tier=tier,
                duration_hours=duration_hours,
                initial_limit=float(initial_limit)
            )

            return period_id

        except Exception as e:
            logger.error(
                "Failed to start adjustment period",
                account_id=account_id,
                error=str(e)
            )
            self.session.rollback()
            raise

    async def get_adjustment_status(
        self,
        account_id: str
    ) -> AdjustmentStatus | None:
        """Get current adjustment period status.
        
        Args:
            account_id: Account to check
            
        Returns:
            AdjustmentStatus or None if no active period
        """
        period = self._get_active_period(account_id)

        if not period:
            return None

        # Calculate elapsed time
        elapsed = datetime.utcnow() - period.start_time
        hours_elapsed = elapsed.total_seconds() / 3600
        hours_remaining = max(0, (period.end_time - datetime.utcnow()).total_seconds() / 3600)

        # Determine current phase
        current_phase = self._calculate_phase(hours_elapsed)

        # Calculate phase progress
        phase_duration = PHASE_DURATIONS.get(current_phase, 12)
        phase_start_hour = sum(
            PHASE_DURATIONS[p] for p in AdjustmentPhase
            if list(AdjustmentPhase).index(p) < list(AdjustmentPhase).index(current_phase)
        )
        phase_progress = min(100, ((hours_elapsed - phase_start_hour) / phase_duration) * 100)

        # Get current limits
        current_position_limit = (
            period.original_position_limit *
            PHASE_POSITION_MULTIPLIERS[current_phase]
        )

        return AdjustmentStatus(
            period_id=period.period_id,
            account_id=period.account_id,
            transition_id=period.transition_id,
            current_phase=current_phase,
            hours_elapsed=hours_elapsed,
            hours_remaining=hours_remaining,
            current_position_limit=current_position_limit,
            normal_position_limit=period.original_position_limit,
            monitoring_multiplier=PHASE_MONITORING_MULTIPLIERS[current_phase],
            interventions_count=period.interventions_triggered,
            phase_progress_percentage=phase_progress
        )

    async def record_intervention(
        self,
        account_id: str,
        intervention_type: str,
        details: str
    ) -> None:
        """Record an intervention during adjustment period.
        
        Args:
            account_id: Account that triggered intervention
            intervention_type: Type of intervention
            details: Intervention details
        """
        period = self._get_active_period(account_id)

        if not period:
            return  # No active period

        period.interventions_triggered += 1
        period.updated_at = datetime.utcnow()

        self.session.commit()

        logger.info(
            "Adjustment period intervention recorded",
            account_id=account_id,
            period_id=period.period_id,
            intervention_type=intervention_type,
            total_interventions=period.interventions_triggered
        )

        # Check if too many interventions
        if period.interventions_triggered > 5:
            await self._extend_adjustment_period(period.period_id, 12)

    async def force_complete(
        self,
        account_id: str,
        reason: str
    ) -> bool:
        """Force completion of adjustment period.
        
        Args:
            account_id: Account to complete
            reason: Reason for early completion
            
        Returns:
            True if completed successfully
        """
        period = self._get_active_period(account_id)

        if not period:
            return False

        # Update period
        period.end_time = datetime.utcnow()
        period.current_phase = AdjustmentPhase.COMPLETE.value
        period.updated_at = datetime.utcnow()

        # Restore normal monitoring
        await self._update_monitoring_sensitivity(account_id, 1.0)

        self.session.commit()

        # Cancel monitoring task
        if period.period_id in self._active_periods:
            self._active_periods[period.period_id].cancel()
            del self._active_periods[period.period_id]

        logger.info(
            "Adjustment period force completed",
            account_id=account_id,
            period_id=period.period_id,
            reason=reason
        )

        return True

    def _get_active_period(self, account_id: str) -> AdjustmentPeriod | None:
        """Get active adjustment period for account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Active AdjustmentPeriod or None
        """
        now = datetime.utcnow()
        return self.session.query(AdjustmentPeriod).filter(
            AdjustmentPeriod.account_id == account_id,
            AdjustmentPeriod.start_time <= now,
            AdjustmentPeriod.end_time > now
        ).first()

    def _calculate_phase(self, hours_elapsed: float) -> AdjustmentPhase:
        """Calculate current phase based on elapsed time.
        
        Args:
            hours_elapsed: Hours since period start
            
        Returns:
            Current adjustment phase
        """
        cumulative_hours = 0

        for phase in [AdjustmentPhase.INITIAL, AdjustmentPhase.EARLY,
                      AdjustmentPhase.MID, AdjustmentPhase.LATE]:
            cumulative_hours += PHASE_DURATIONS[phase]
            if hours_elapsed < cumulative_hours:
                return phase

        return AdjustmentPhase.COMPLETE

    async def _monitor_adjustment_period(
        self,
        period_id: str,
        duration_hours: int
    ) -> None:
        """Monitor and update adjustment period phases.
        
        Args:
            period_id: Period to monitor
            duration_hours: Total duration
        """
        try:
            phase_check_interval = 3600  # Check every hour
            elapsed_hours = 0

            while elapsed_hours < duration_hours:
                await asyncio.sleep(phase_check_interval)
                elapsed_hours += 1

                # Update period phase
                period = self.session.query(AdjustmentPeriod).filter_by(
                    period_id=period_id
                ).first()

                if not period:
                    break

                # Calculate new phase
                new_phase = self._calculate_phase(elapsed_hours)

                if new_phase.value != period.current_phase:
                    # Phase transition
                    await self._transition_phase(period, new_phase)

            # Complete the period
            await self._complete_adjustment_period(period_id)

        except asyncio.CancelledError:
            # Period was force completed
            pass
        except Exception as e:
            logger.error(
                "Error monitoring adjustment period",
                period_id=period_id,
                error=str(e)
            )

    async def _transition_phase(
        self,
        period: AdjustmentPeriod,
        new_phase: AdjustmentPhase
    ) -> None:
        """Transition to a new adjustment phase.
        
        Args:
            period: Adjustment period
            new_phase: New phase to transition to
        """
        old_phase = period.current_phase

        # Update period
        period.current_phase = new_phase.value
        period.reduced_position_limit = (
            period.original_position_limit *
            PHASE_POSITION_MULTIPLIERS[new_phase]
        )
        period.monitoring_sensitivity_multiplier = PHASE_MONITORING_MULTIPLIERS[new_phase]
        period.updated_at = datetime.utcnow()

        # Update monitoring sensitivity
        await self._update_monitoring_sensitivity(
            period.account_id,
            PHASE_MONITORING_MULTIPLIERS[new_phase]
        )

        self.session.commit()

        logger.info(
            "Adjustment period phase transition",
            period_id=period.period_id,
            account_id=period.account_id,
            old_phase=old_phase,
            new_phase=new_phase.value,
            new_limit=float(period.reduced_position_limit)
        )

    async def _complete_adjustment_period(self, period_id: str) -> None:
        """Complete an adjustment period.
        
        Args:
            period_id: Period to complete
        """
        period = self.session.query(AdjustmentPeriod).filter_by(
            period_id=period_id
        ).first()

        if not period:
            return

        # Mark as complete
        period.current_phase = AdjustmentPhase.COMPLETE.value
        period.end_time = datetime.utcnow()
        period.updated_at = datetime.utcnow()

        # Restore normal monitoring
        await self._update_monitoring_sensitivity(period.account_id, 1.0)

        # Update transition if linked
        if period.transition_id:
            transition = self.session.query(TierTransition).filter_by(
                transition_id=period.transition_id
            ).first()

            if transition:
                transition.adjustment_period_end = datetime.utcnow()
                transition.updated_at = datetime.utcnow()

        self.session.commit()

        # Remove from active periods
        if period_id in self._active_periods:
            del self._active_periods[period_id]

        logger.info(
            "Adjustment period completed",
            period_id=period_id,
            account_id=period.account_id,
            total_interventions=period.interventions_triggered
        )

    async def _extend_adjustment_period(
        self,
        period_id: str,
        extension_hours: int
    ) -> None:
        """Extend adjustment period due to interventions.
        
        Args:
            period_id: Period to extend
            extension_hours: Hours to extend by
        """
        period = self.session.query(AdjustmentPeriod).filter_by(
            period_id=period_id
        ).first()

        if period:
            period.end_time += timedelta(hours=extension_hours)
            period.updated_at = datetime.utcnow()
            self.session.commit()

            logger.warning(
                "Adjustment period extended due to interventions",
                period_id=period_id,
                extension_hours=extension_hours,
                new_end_time=period.end_time.isoformat()
            )

    async def _update_monitoring_sensitivity(
        self,
        account_id: str,
        multiplier: float
    ) -> None:
        """Update tilt monitoring sensitivity.
        
        Args:
            account_id: Account ID
            multiplier: Sensitivity multiplier
        """
        profile = self.session.query(TiltProfile).filter_by(
            account_id=account_id
        ).first()

        if profile:
            # Store original if not already stored
            if not hasattr(profile, '_original_sensitivity'):
                profile._original_sensitivity = profile.monitoring_sensitivity or 1.0

            if multiplier == 1.0:
                # Restore original
                profile.monitoring_sensitivity = profile._original_sensitivity
            else:
                # Apply multiplier
                profile.monitoring_sensitivity = profile._original_sensitivity * multiplier

            self.session.commit()

    async def cleanup(self) -> None:
        """Clean up all monitoring tasks."""
        for task in self._active_periods.values():
            task.cancel()

        self._active_periods.clear()
