from typing import Optional
"""Drawdown detection and monitoring for recovery protocol."""

from datetime import UTC, datetime
from decimal import Decimal

import structlog

from genesis.core.events import EventType
from genesis.core.models import TradingTier
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class DrawdownDetector:
    """Monitors account balances and detects significant drawdowns."""

    def __init__(
        self,
        repository: SQLiteRepository,
        event_bus: EventBus,
    ):
        """Initialize drawdown detector.
        
        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing drawdown events
        """
        self.repository = repository
        self.event_bus = event_bus
        self._peak_balances: dict[str, Decimal] = {}

    def calculate_drawdown(
        self,
        balance: Decimal,
        peak_balance: Decimal
    ) -> Decimal:
        """Calculate drawdown percentage from peak.
        
        Args:
            balance: Current balance
            peak_balance: Peak balance to compare against
            
        Returns:
            Drawdown percentage as decimal (0.10 = 10%)
        """
        if peak_balance <= 0:
            return Decimal("0")

        drawdown = (peak_balance - balance) / peak_balance
        return max(Decimal("0"), drawdown)

    @requires_tier(TradingTier.HUNTER)
    def detect_drawdown_breach(
        self,
        account_id: str,
        threshold: Optional[Decimal] = None
    ) -> bool:
        """Check if account has breached drawdown threshold.
        
        Args:
            account_id: Account identifier
            threshold: Optional custom threshold (default 0.10 for 10%)
            
        Returns:
            True if drawdown threshold breached
        """
        if threshold is None:
            threshold = self._get_tier_threshold(account_id)

        try:
            account = self.repository.get_account(account_id)
            if not account:
                logger.warning("Account not found", account_id=account_id)
                return False

            peak_balance = self._get_peak_balance(account_id, account.balance)
            current_drawdown = self.calculate_drawdown(
                account.balance,
                peak_balance
            )

            if current_drawdown >= threshold:
                logger.warning(
                    "Drawdown threshold breached",
                    account_id=account_id,
                    current_balance=float(account.balance),
                    peak_balance=float(peak_balance),
                    drawdown_pct=float(current_drawdown * 100),
                    threshold_pct=float(threshold * 100)
                )

                self.event_bus.publish({
                    "type": EventType.DRAWDOWN_DETECTED,
                    "account_id": account_id,
                    "current_balance": account.balance,
                    "peak_balance": peak_balance,
                    "drawdown_pct": current_drawdown,
                    "threshold": threshold,
                    "timestamp": datetime.now(UTC)
                })

                return True

            return False

        except Exception as e:
            logger.error(
                "Error detecting drawdown breach",
                account_id=account_id,
                error=str(e)
            )
            return False

    def _get_peak_balance(
        self,
        account_id: str,
        current_balance: Decimal
    ) -> Decimal:
        """Get or update peak balance for account.
        
        Args:
            account_id: Account identifier
            current_balance: Current account balance
            
        Returns:
            Peak balance for the account
        """
        stored_peak = self.repository.get_peak_balance(account_id)

        if stored_peak is None:
            self.repository.update_peak_balance(account_id, current_balance)
            self._peak_balances[account_id] = current_balance
            return current_balance

        if current_balance > stored_peak:
            self.repository.update_peak_balance(account_id, current_balance)
            self._peak_balances[account_id] = current_balance
            return current_balance

        self._peak_balances[account_id] = stored_peak
        return stored_peak

    def _get_tier_threshold(self, account_id: str) -> Decimal:
        """Get drawdown threshold based on account tier.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Drawdown threshold for the tier
        """
        account = self.repository.get_account(account_id)
        if not account:
            return Decimal("0.10")

        tier_thresholds = {
            TradingTier.SNIPER: Decimal("0.05"),
            TradingTier.HUNTER: Decimal("0.10"),
            TradingTier.STRATEGIST: Decimal("0.15"),
            TradingTier.ARCHITECT: Decimal("0.20")
        }

        return tier_thresholds.get(account.tier, Decimal("0.10"))

    def update_balance_tracking(
        self,
        account_id: str,
        new_balance: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Update balance tracking and return drawdown info.
        
        Args:
            account_id: Account identifier
            new_balance: New balance to track
            
        Returns:
            Tuple of (peak_balance, current_drawdown_pct)
        """
        peak_balance = self._get_peak_balance(account_id, new_balance)
        current_drawdown = self.calculate_drawdown(new_balance, peak_balance)

        logger.debug(
            "Balance tracking updated",
            account_id=account_id,
            balance=float(new_balance),
            peak=float(peak_balance),
            drawdown_pct=float(current_drawdown * 100)
        )

        return peak_balance, current_drawdown

    def get_drawdown_stats(self, account_id: str) -> dict:
        """Get comprehensive drawdown statistics for account.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Dictionary with drawdown statistics
        """
        account = self.repository.get_account(account_id)
        if not account:
            return {}

        peak_balance = self._get_peak_balance(account_id, account.balance)
        current_drawdown = self.calculate_drawdown(account.balance, peak_balance)
        threshold = self._get_tier_threshold(account_id)

        return {
            "account_id": account_id,
            "current_balance": account.balance,
            "peak_balance": peak_balance,
            "drawdown_pct": current_drawdown,
            "drawdown_amount": peak_balance - account.balance,
            "threshold": threshold,
            "threshold_breached": current_drawdown >= threshold,
            "recovery_needed": peak_balance - account.balance if current_drawdown > 0 else Decimal("0"),
            "timestamp": datetime.now(UTC)
        }

    def reset_peak_balance(self, account_id: str) -> None:
        """Reset peak balance tracking for account.
        
        Args:
            account_id: Account identifier
        """
        account = self.repository.get_account(account_id)
        if account:
            self.repository.update_peak_balance(account_id, account.balance)
            self._peak_balances[account_id] = account.balance
            logger.info(
                "Peak balance reset",
                account_id=account_id,
                new_peak=float(account.balance)
            )
