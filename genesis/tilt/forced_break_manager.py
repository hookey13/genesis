from typing import Optional

"""Forced break management for consecutive losses."""

from datetime import UTC, datetime, timedelta

import structlog

from genesis.core.events import EventType
from genesis.core.models import TradingTier
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class ForcedBreakManager:
    """Manages forced trading breaks after consecutive losses."""

    def __init__(
        self,
        repository: SQLiteRepository,
        event_bus: EventBus,
    ):
        """Initialize forced break manager.

        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing break events
        """
        self.repository = repository
        self.event_bus = event_bus
        self._active_breaks: dict[str, datetime] = {}
        self._loss_counters: dict[str, int] = {}

    @requires_tier(TradingTier.HUNTER)
    def check_consecutive_losses(
        self,
        account_id: str
    ) -> int:
        """Check consecutive loss count for account.

        Args:
            account_id: Account identifier

        Returns:
            Number of consecutive losses
        """
        try:
            # Get tilt profile
            profile = self.repository.get_tilt_profile(account_id)
            if not profile:
                return 0

            consecutive_losses = profile.get("consecutive_losses", 0)
            self._loss_counters[account_id] = consecutive_losses

            logger.debug(
                "Consecutive losses checked",
                account_id=account_id,
                count=consecutive_losses
            )

            return consecutive_losses

        except Exception as e:
            logger.error(
                "Error checking consecutive losses",
                account_id=account_id,
                error=str(e)
            )
            return 0

    def record_trade_result(
        self,
        account_id: str,
        is_profitable: bool
    ) -> Optional[datetime]:
        """Record trade result and check for forced break trigger.

        Args:
            account_id: Account identifier
            is_profitable: Whether trade was profitable

        Returns:
            Break expiration time if break enforced, None otherwise
        """
        try:
            # Get current consecutive losses
            current_losses = self._loss_counters.get(account_id, 0)

            if is_profitable:
                # Reset consecutive loss counter
                current_losses = 0
                self._loss_counters[account_id] = 0

                # Update database
                self.repository.update_tilt_profile(
                    account_id,
                    {"consecutive_losses": 0}
                )

                logger.info(
                    "Consecutive loss streak broken",
                    account_id=account_id
                )
            else:
                # Increment consecutive loss counter
                current_losses += 1
                self._loss_counters[account_id] = current_losses

                # Update database
                self.repository.update_tilt_profile(
                    account_id,
                    {"consecutive_losses": current_losses}
                )

                logger.warning(
                    "Consecutive loss recorded",
                    account_id=account_id,
                    streak=current_losses
                )

                # Check if break threshold reached (default: 3 losses)
                if current_losses >= 3:
                    return self.enforce_trading_break(account_id)

            return None

        except Exception as e:
            logger.error(
                "Error recording trade result",
                account_id=account_id,
                error=str(e)
            )
            return None

    def enforce_trading_break(
        self,
        account_id: str,
        duration_minutes: int = 30
    ) -> datetime:
        """Enforce a trading break for the account.

        Args:
            account_id: Account identifier
            duration_minutes: Break duration in minutes

        Returns:
            Break expiration time
        """
        expiration = datetime.now(UTC) + timedelta(minutes=duration_minutes)
        self._active_breaks[account_id] = expiration

        # Update tilt profile with lockout
        self.repository.update_tilt_profile(
            account_id,
            {
                "lockout_expiration": expiration,
                "journal_entries_required": 1,  # Require journal entry
                "recovery_required": True
            }
        )

        # Publish break event
        self.event_bus.publish({
            "type": EventType.FORCED_BREAK_INITIATED,
            "account_id": account_id,
            "duration_minutes": duration_minutes,
            "expiration": expiration.isoformat(),
            "reason": "consecutive_losses",
            "consecutive_losses": self._loss_counters.get(account_id, 3),
            "timestamp": datetime.now(UTC)
        })

        logger.warning(
            "Trading break enforced",
            account_id=account_id,
            duration_minutes=duration_minutes,
            expiration=expiration.isoformat()
        )

        return expiration

    def is_on_break(
        self,
        account_id: str
    ) -> bool:
        """Check if account is currently on forced break.

        Args:
            account_id: Account identifier

        Returns:
            True if on break
        """
        # Check memory cache first
        if account_id in self._active_breaks:
            expiration = self._active_breaks[account_id]
            if datetime.now(UTC) < expiration:
                return True
            else:
                # Break expired, remove from cache
                del self._active_breaks[account_id]

        # Check database
        profile = self.repository.get_tilt_profile(account_id)
        if profile and profile.get("lockout_expiration"):
            expiration = profile["lockout_expiration"]
            if isinstance(expiration, str):
                expiration = datetime.fromisoformat(expiration)

            if datetime.now(UTC) < expiration:
                self._active_breaks[account_id] = expiration
                return True

        return False

    def get_break_status(
        self,
        account_id: str
    ) -> dict:
        """Get detailed break status for account.

        Args:
            account_id: Account identifier

        Returns:
            Dictionary with break status details
        """
        is_on_break = self.is_on_break(account_id)
        expiration = self._active_breaks.get(account_id)

        if is_on_break and expiration:
            remaining = expiration - datetime.now(UTC)
            remaining_minutes = max(0, int(remaining.total_seconds() / 60))
        else:
            remaining_minutes = 0

        profile = self.repository.get_tilt_profile(account_id)

        return {
            "account_id": account_id,
            "is_on_break": is_on_break,
            "expiration": expiration.isoformat() if expiration else None,
            "remaining_minutes": remaining_minutes,
            "consecutive_losses": self._loss_counters.get(account_id, 0),
            "journal_required": profile.get("journal_entries_required", 0) > 0 if profile else False,
            "recovery_required": profile.get("recovery_required", False) if profile else False
        }

    def clear_break(
        self,
        account_id: str,
        journal_completed: bool = False
    ) -> bool:
        """Clear forced break for account.

        Args:
            account_id: Account identifier
            journal_completed: Whether journal entry was completed

        Returns:
            True if break cleared successfully
        """
        # Check if journal requirement met
        profile = self.repository.get_tilt_profile(account_id)
        if profile and profile.get("journal_entries_required", 0) > 0:
            if not journal_completed:
                logger.warning(
                    "Cannot clear break without journal entry",
                    account_id=account_id
                )
                return False

        # Clear break
        if account_id in self._active_breaks:
            del self._active_breaks[account_id]

        # Reset loss counter
        self._loss_counters[account_id] = 0

        # Update database
        self.repository.update_tilt_profile(
            account_id,
            {
                "lockout_expiration": None,
                "consecutive_losses": 0,
                "journal_entries_required": 0,
                "recovery_required": False
            }
        )

        # Publish break cleared event
        self.event_bus.publish({
            "type": EventType.FORCED_BREAK_CLEARED,
            "account_id": account_id,
            "journal_completed": journal_completed,
            "timestamp": datetime.now(UTC)
        })

        logger.info(
            "Trading break cleared",
            account_id=account_id,
            journal_completed=journal_completed
        )

        return True

    def load_active_breaks(self) -> None:
        """Load active breaks from database on startup."""
        try:
            # Get all profiles with active lockouts
            profiles = self.repository.get_profiles_with_lockouts()

            for profile in profiles:
                account_id = profile["account_id"]
                expiration = profile.get("lockout_expiration")

                if expiration:
                    if isinstance(expiration, str):
                        expiration = datetime.fromisoformat(expiration)

                    if datetime.now(UTC) < expiration:
                        self._active_breaks[account_id] = expiration
                        self._loss_counters[account_id] = profile.get("consecutive_losses", 0)

            logger.info(
                "Active breaks loaded",
                count=len(self._active_breaks)
            )

        except Exception as e:
            logger.error(
                "Failed to load active breaks",
                error=str(e)
            )
