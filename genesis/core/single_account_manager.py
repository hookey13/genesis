"""
Account management for Project GENESIS.

This module handles account balance synchronization, updates,
and persistence with the exchange.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import structlog

from genesis.core.exceptions import ExchangeError
from genesis.core.models import Account, TradingTier, Tier
from genesis.exchange.gateway import BinanceGateway
from genesis.utils.decorators import retry, with_timeout, requires_tier

logger = structlog.get_logger(__name__)


class AccountManager:
    """
    Manages account balance and synchronization with exchange.

    Handles periodic balance updates, persistence, and account state.
    """

    SYNC_INTERVAL = 60  # seconds
    SYNC_TIMEOUT = 10  # seconds

    def __init__(
        self,
        gateway: BinanceGateway,
        account: Optional[Account] = None,
        auto_sync: bool = True,
    ):
        """
        Initialize account manager.

        Args:
            gateway: Exchange gateway for balance fetching
            account: Existing account (optional, will create if None)
            auto_sync: Enable automatic periodic synchronization
        """
        self.gateway = gateway
        self.account = account or Account(
            balance_usdt=Decimal("0"), tier=TradingTier.SNIPER
        )
        self.auto_sync = auto_sync
        self._sync_task: Optional[asyncio.Task] = None
        self._last_sync_error: Optional[str] = None
        self._sync_count = 0

        logger.info(
            "Account manager initialized",
            account_id=self.account.account_id,
            balance=str(self.account.balance_usdt),
            tier=self.account.tier.value,
            auto_sync=auto_sync,
        )

    async def initialize(self) -> None:
        """Initialize account manager and perform first sync."""
        try:
            await self.sync_balance()

            if self.auto_sync:
                self._sync_task = asyncio.create_task(self._periodic_sync())
                logger.info("Automatic balance synchronization started")
        except Exception as e:
            logger.error("Failed to initialize account manager", error=str(e))
            raise

    async def shutdown(self) -> None:
        """Shutdown account manager and stop sync task."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            logger.info("Account synchronization stopped")

    @with_timeout(SYNC_TIMEOUT)
    @retry(max_attempts=3, delay=1.0)
    async def sync_balance(self) -> Decimal:
        """
        Synchronize account balance with exchange.

        Returns:
            Updated balance in USDT

        Raises:
            ExchangeError: If unable to fetch balance
        """
        try:
            # Fetch account info from exchange
            account_info = await self.gateway.get_account_info()

            # Find USDT balance
            usdt_balance = Decimal("0")
            for balance in account_info.get("balances", []):
                if balance["asset"] == "USDT":
                    # Use free balance (available for trading)
                    usdt_balance = Decimal(str(balance["free"]))
                    break

            # Update account
            old_balance = self.account.balance_usdt
            self.account.balance_usdt = usdt_balance
            self.account.last_sync = datetime.now()
            self._sync_count += 1
            self._last_sync_error = None

            # Log balance change
            if old_balance != usdt_balance:
                change = usdt_balance - old_balance
                logger.info(
                    "Account balance updated",
                    old_balance=str(old_balance),
                    new_balance=str(usdt_balance),
                    change=str(change),
                    sync_count=self._sync_count,
                )
            else:
                logger.debug(
                    "Balance unchanged",
                    balance=str(usdt_balance),
                    sync_count=self._sync_count,
                )

            return usdt_balance

        except Exception as e:
            self._last_sync_error = str(e)
            logger.error(
                "Failed to sync account balance",
                error=str(e),
                sync_count=self._sync_count,
            )
            raise ExchangeError(f"Balance sync failed: {e}")

    async def _periodic_sync(self) -> None:
        """Background task for periodic balance synchronization."""
        logger.info(
            "Starting periodic balance sync", interval_seconds=self.SYNC_INTERVAL
        )

        while True:
            try:
                await asyncio.sleep(self.SYNC_INTERVAL)
                await self.sync_balance()
            except asyncio.CancelledError:
                logger.info("Periodic sync cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error in periodic sync",
                    error=str(e),
                    will_retry_in=self.SYNC_INTERVAL,
                )

    def get_balance(self) -> Decimal:
        """
        Get current account balance.

        Returns:
            Current balance in USDT
        """
        return self.account.balance_usdt

    def get_tier(self) -> TradingTier:
        """
        Get current trading tier.

        Returns:
            Current trading tier
        """
        return self.account.tier

    @requires_tier(Tier.SNIPER)
    def update_balance(self, new_balance: Decimal) -> None:
        """
        Manually update account balance.

        Args:
            new_balance: New balance in USDT
        """
        if new_balance < 0:
            raise ValueError("Balance cannot be negative")

        old_balance = self.account.balance_usdt
        self.account.balance_usdt = new_balance

        logger.info(
            "Balance manually updated",
            old_balance=str(old_balance),
            new_balance=str(new_balance),
        )

    @requires_tier(Tier.SNIPER)
    def deduct_balance(self, amount: Decimal) -> None:
        """
        Deduct amount from balance (for position opening).

        Args:
            amount: Amount to deduct

        Raises:
            ValueError: If deduction would make balance negative
        """
        if amount > self.account.balance_usdt:
            raise ValueError(
                f"Cannot deduct ${amount} from balance ${self.account.balance_usdt}"
            )

        self.account.balance_usdt -= amount
        logger.debug(
            "Balance deducted",
            amount=str(amount),
            new_balance=str(self.account.balance_usdt),
        )

    @requires_tier(Tier.SNIPER)
    def add_balance(self, amount: Decimal) -> None:
        """
        Add amount to balance (for position closing).

        Args:
            amount: Amount to add
        """
        self.account.balance_usdt += amount
        logger.debug(
            "Balance added",
            amount=str(amount),
            new_balance=str(self.account.balance_usdt),
        )

    def is_sync_healthy(self) -> bool:
        """
        Check if balance synchronization is healthy.

        Returns:
            True if sync is working, False if errors or stale
        """
        if self._last_sync_error:
            return False

        # Check if last sync is too old
        if self.account.last_sync:
            time_since_sync = datetime.now() - self.account.last_sync
            if time_since_sync > timedelta(seconds=self.SYNC_INTERVAL * 3):
                return False

        return True

    def get_sync_status(self) -> dict[str, Any]:
        """
        Get detailed sync status.

        Returns:
            Dictionary with sync status information
        """
        return {
            "last_sync": (
                self.account.last_sync.isoformat() if self.account.last_sync else None
            ),
            "sync_count": self._sync_count,
            "last_error": self._last_sync_error,
            "is_healthy": self.is_sync_healthy(),
            "auto_sync_enabled": self.auto_sync,
            "sync_interval": self.SYNC_INTERVAL,
        }

    async def validate_balance_constraint(self) -> None:
        """
        Validate that balance meets database CHECK constraint.

        Raises:
            ValueError: If balance is negative
        """
        if self.account.balance_usdt < 0:
            raise ValueError(
                f"Balance constraint violation: ${self.account.balance_usdt} < 0"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert account to dictionary for persistence.

        Returns:
            Dictionary representation of account
        """
        return {
            "account_id": self.account.account_id,
            "balance_usdt": str(self.account.balance_usdt),
            "tier": self.account.tier.value,
            "locked_features": self.account.locked_features,
            "last_sync": (
                self.account.last_sync.isoformat() if self.account.last_sync else None
            ),
            "created_at": self.account.created_at.isoformat(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], gateway: BinanceGateway
    ) -> "AccountManager":
        """
        Create AccountManager from dictionary.

        Args:
            data: Dictionary with account data
            gateway: Exchange gateway

        Returns:
            AccountManager instance
        """
        account = Account(
            account_id=data["account_id"],
            balance_usdt=Decimal(data["balance_usdt"]),
            tier=TradingTier[data["tier"]],
            locked_features=data.get("locked_features", []),
            last_sync=(
                datetime.fromisoformat(data["last_sync"])
                if data.get("last_sync")
                else None
            ),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        return cls(gateway=gateway, account=account)
