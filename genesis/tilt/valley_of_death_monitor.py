"""Valley of Death transition monitoring for tier boundaries.

Provides heightened monitoring and protection during critical tier transitions,
especially around the $2k threshold where many traders fail.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import structlog

from genesis.data.models_db import (
    AccountDB,
    Session,
    TierTransition,
    TiltProfile,
    get_session,
)

logger = structlog.get_logger(__name__)


# Tier thresholds in USDT
TIER_THRESHOLDS = {
    'SNIPER': Decimal('500'),
    'HUNTER': Decimal('2000'),
    'STRATEGIST': Decimal('10000'),
    'ARCHITECT': Decimal('50000'),
    'EMPEROR': Decimal('250000')
}

# Monitoring sensitivity multipliers by proximity to threshold
PROXIMITY_MULTIPLIERS = {
    Decimal('0.90'): 1.5,  # 90% of threshold - start watching
    Decimal('0.95'): 2.0,  # 95% of threshold - heightened monitoring
    Decimal('0.98'): 3.0,  # 98% of threshold - critical monitoring
}


@dataclass
class TransitionProximity:
    """Tracks proximity to tier transition threshold."""
    current_balance: Decimal
    current_tier: str
    next_tier: str
    threshold: Decimal
    distance_dollars: Decimal
    distance_percentage: Decimal
    monitoring_multiplier: float
    is_approaching: bool

    @property
    def is_critical(self) -> bool:
        """Check if in critical proximity (>95% of threshold)."""
        return self.distance_percentage >= Decimal('95')

    @property
    def days_at_current_rate(self) -> Optional[int]:
        """Estimate days to threshold at current growth rate."""
        # This would need historical data to calculate
        # Placeholder for now
        return None


class TransitionMonitor:
    """Monitors account proximity to tier transitions."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize transition monitor.

        Args:
            session: Optional database session
        """
        self.session = session or get_session()
        self._monitoring_tasks: dict[str, asyncio.Task] = {}
        self._approach_events: list[dict[str, Any]] = []

    def check_approaching_transition(
        self,
        balance: Decimal,
        current_tier: str
    ) -> TransitionProximity:
        """Check if balance is approaching next tier threshold.

        Args:
            balance: Current account balance in USDT
            current_tier: Current tier name

        Returns:
            TransitionProximity with details about distance to threshold
        """
        # Get next tier and threshold
        next_tier, threshold = self._get_next_tier_info(current_tier)

        if not next_tier:
            # At highest tier
            return TransitionProximity(
                current_balance=balance,
                current_tier=current_tier,
                next_tier='NONE',
                threshold=Decimal('0'),
                distance_dollars=Decimal('0'),
                distance_percentage=Decimal('0'),
                monitoring_multiplier=1.0,
                is_approaching=False
            )

        # Calculate distances
        distance_dollars = threshold - balance
        # Calculate percentage as whole number (e.g., 95 for 95%)
        distance_percentage = (balance / threshold * 100) if threshold > 0 else Decimal('0')

        # Determine monitoring multiplier
        monitoring_multiplier = 1.0
        is_approaching = False

        # Convert percentage to decimal for comparison (e.g., 95 -> 0.95)
        percentage_decimal = distance_percentage / 100

        for proximity_threshold, multiplier in sorted(
            PROXIMITY_MULTIPLIERS.items(),
            reverse=True
        ):
            if percentage_decimal >= proximity_threshold:
                monitoring_multiplier = multiplier
                is_approaching = True
                break

        return TransitionProximity(
            current_balance=balance,
            current_tier=current_tier,
            next_tier=next_tier,
            threshold=threshold,
            distance_dollars=distance_dollars,
            distance_percentage=distance_percentage,
            monitoring_multiplier=monitoring_multiplier,
            is_approaching=is_approaching
        )

    async def start_monitoring(
        self,
        account_id: str,
        check_interval_seconds: int = 60
    ) -> None:
        """Start monitoring an account for tier transitions.

        Args:
            account_id: Account to monitor
            check_interval_seconds: How often to check proximity
        """
        if account_id in self._monitoring_tasks:
            logger.warning(
                "Monitoring already active",
                account_id=account_id
            )
            return

        task = asyncio.create_task(
            self._monitor_account(account_id, check_interval_seconds)
        )
        self._monitoring_tasks[account_id] = task

        logger.info(
            "Started transition monitoring",
            account_id=account_id,
            interval_seconds=check_interval_seconds
        )

    async def stop_monitoring(self, account_id: str) -> None:
        """Stop monitoring an account.

        Args:
            account_id: Account to stop monitoring
        """
        if account_id not in self._monitoring_tasks:
            return

        task = self._monitoring_tasks[account_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        del self._monitoring_tasks[account_id]

        logger.info(
            "Stopped transition monitoring",
            account_id=account_id
        )

    async def _monitor_account(
        self,
        account_id: str,
        check_interval_seconds: int
    ) -> None:
        """Monitor account for tier transition proximity.

        Args:
            account_id: Account to monitor
            check_interval_seconds: Check interval
        """
        last_proximity: Optional[TransitionProximity] = None

        while True:
            try:
                # Get account and check proximity
                account = self.session.query(AccountDB).filter_by(
                    account_id=account_id
                ).first()

                if not account:
                    logger.error(
                        "Account not found for monitoring",
                        account_id=account_id
                    )
                    break

                proximity = self.check_approaching_transition(
                    balance=account.balance_usdt,
                    current_tier=account.current_tier
                )

                # Check if we've crossed a monitoring threshold
                if proximity.is_approaching:
                    await self._handle_approach_detected(
                        account_id=account_id,
                        proximity=proximity,
                        last_proximity=last_proximity
                    )

                # Store for next iteration
                last_proximity = proximity

                # Adjust check interval based on proximity
                adjusted_interval = check_interval_seconds
                if proximity.monitoring_multiplier > 1:
                    # Check more frequently when approaching
                    adjusted_interval = int(
                        check_interval_seconds / proximity.monitoring_multiplier
                    )

                await asyncio.sleep(adjusted_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(
                    "Error in transition monitoring",
                    account_id=account_id,
                    error=str(e)
                )
                await asyncio.sleep(check_interval_seconds)

    async def _handle_approach_detected(
        self,
        account_id: str,
        proximity: TransitionProximity,
        last_proximity: Optional[TransitionProximity]
    ) -> None:
        """Handle detection of approaching tier transition.

        Args:
            account_id: Account approaching transition
            proximity: Current proximity details
            last_proximity: Previous proximity check
        """
        # Check if this is a new approach event
        is_new_approach = (
            last_proximity is None or
            not last_proximity.is_approaching
        )

        if is_new_approach:
            logger.warning(
                "Tier transition approach detected",
                account_id=account_id,
                current_tier=proximity.current_tier,
                next_tier=proximity.next_tier,
                distance_percentage=float(proximity.distance_percentage),
                distance_dollars=float(proximity.distance_dollars)
            )

            # Create or update tier transition record
            await self._create_transition_record(account_id, proximity)

            # Enhance behavioral monitoring
            await self._enhance_behavioral_monitoring(account_id, proximity)

            # Store approach event
            self._approach_events.append({
                'account_id': account_id,
                'timestamp': datetime.utcnow(),
                'proximity': proximity,
                'event_type': 'APPROACH_DETECTED'
            })

        # Check for critical proximity
        if proximity.is_critical:
            await self._handle_critical_proximity(account_id, proximity)

    async def _create_transition_record(
        self,
        account_id: str,
        proximity: TransitionProximity
    ) -> None:
        """Create or update transition record in database.

        Args:
            account_id: Account ID
            proximity: Proximity details
        """
        try:
            # Check for existing transition
            existing = self.session.query(TierTransition).filter_by(
                account_id=account_id,
                from_tier=proximity.current_tier,
                to_tier=proximity.next_tier,
                transition_status='APPROACHING'
            ).first()

            if not existing:
                # Create new transition record
                transition = TierTransition(
                    transition_id=str(uuid.uuid4()),
                    account_id=account_id,
                    from_tier=proximity.current_tier,
                    to_tier=proximity.next_tier,
                    transition_status='APPROACHING',
                    created_at=datetime.utcnow()
                )
                self.session.add(transition)
                self.session.commit()

                logger.info(
                    "Created transition record",
                    transition_id=transition.transition_id,
                    account_id=account_id
                )

        except Exception as e:
            logger.error(
                "Failed to create transition record",
                account_id=account_id,
                error=str(e)
            )
            self.session.rollback()

    async def _enhance_behavioral_monitoring(
        self,
        account_id: str,
        proximity: TransitionProximity
    ) -> None:
        """Enhance behavioral monitoring for approaching transition.

        Args:
            account_id: Account ID
            proximity: Proximity details
        """
        try:
            # Update tilt profile with enhanced monitoring
            profile = self.session.query(TiltProfile).filter_by(
                account_id=account_id
            ).first()

            if profile:
                # Store original sensitivity for later restoration
                if not hasattr(profile, '_original_sensitivity'):
                    current_sensitivity = getattr(profile, 'monitoring_sensitivity', 1.0)
                    profile._original_sensitivity = current_sensitivity

                # Apply monitoring multiplier
                profile.monitoring_sensitivity = (
                    profile._original_sensitivity * proximity.monitoring_multiplier
                )

                self.session.commit()

                logger.info(
                    "Enhanced behavioral monitoring",
                    account_id=account_id,
                    multiplier=proximity.monitoring_multiplier
                )

        except Exception as e:
            logger.error(
                "Failed to enhance monitoring",
                account_id=account_id,
                error=str(e)
            )
            self.session.rollback()

    async def _handle_critical_proximity(
        self,
        account_id: str,
        proximity: TransitionProximity
    ) -> None:
        """Handle critical proximity to tier threshold.

        Args:
            account_id: Account ID
            proximity: Proximity details
        """
        logger.critical(
            "CRITICAL: Tier transition imminent",
            account_id=account_id,
            current_balance=float(proximity.current_balance),
            threshold=float(proximity.threshold),
            distance_percentage=float(proximity.distance_percentage)
        )

        # Store critical event
        self._approach_events.append({
            'account_id': account_id,
            'timestamp': datetime.utcnow(),
            'proximity': proximity,
            'event_type': 'CRITICAL_PROXIMITY'
        })

        # Additional actions would be triggered here:
        # - Force readiness assessment
        # - Require paper trading
        # - Show warnings in UI
        # These will be implemented in subsequent components

    def _get_next_tier_info(self, current_tier: str) -> tuple[Optional[str], Optional[Decimal]]:
        """Get next tier name and threshold.

        Args:
            current_tier: Current tier name

        Returns:
            Tuple of (next_tier_name, threshold) or (None, None) if at highest
        """
        tier_order = ['SNIPER', 'HUNTER', 'STRATEGIST', 'ARCHITECT', 'EMPEROR']

        try:
            current_index = tier_order.index(current_tier)
            if current_index < len(tier_order) - 1:
                next_tier = tier_order[current_index + 1]
                return next_tier, TIER_THRESHOLDS[next_tier]
        except (ValueError, KeyError):
            logger.error(
                "Invalid tier name",
                current_tier=current_tier
            )

        return None, None

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get monitoring statistics.

        Returns:
            Dictionary of monitoring stats
        """
        return {
            'active_monitors': len(self._monitoring_tasks),
            'monitored_accounts': list(self._monitoring_tasks.keys()),
            'approach_events': len(self._approach_events),
            'recent_events': self._approach_events[-10:]  # Last 10 events
        }

    async def cleanup(self) -> None:
        """Clean up monitoring tasks."""
        tasks = list(self._monitoring_tasks.keys())
        for account_id in tasks:
            await self.stop_monitoring(account_id)
