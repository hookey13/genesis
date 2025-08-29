"""
Multi-account management infrastructure for Project GENESIS.

Provides account isolation, hierarchy management, and permission controls
for institutional-grade multi-account trading operations.
"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import structlog
from pydantic import ValidationError

from genesis.core.constants import TradingTier
from genesis.core.models import Account, AccountType, Position
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class AccountManager:
    """Manages multiple trading accounts with isolation and permissions."""

    def __init__(self, repository: Repository, event_bus: EventBus):
        """Initialize AccountManager with repository and event bus."""
        self.repository = repository
        self.event_bus = event_bus
        self._accounts_cache: dict[str, Account] = {}
        self._active_account_id: str | None = None
        logger.info("account_manager_initialized")

    @requires_tier(TradingTier.STRATEGIST)
    async def create_account(
        self,
        balance_usdt: Decimal,
        account_type: AccountType = AccountType.MASTER,
        parent_account_id: str | None = None,
        tier: TradingTier = TradingTier.SNIPER,
        permissions: dict | None = None,
        compliance_settings: dict | None = None,
    ) -> Account:
        """
        Create a new trading account.

        Args:
            balance_usdt: Initial account balance in USDT
            account_type: Type of account (MASTER, SUB, PAPER)
            parent_account_id: Parent account ID for sub-accounts
            tier: Trading tier for the account
            permissions: Feature access control settings
            compliance_settings: Regulatory compliance settings

        Returns:
            Created Account instance

        Raises:
            ValidationError: If account validation fails
            ValueError: If parent account doesn't exist for SUB account
        """
        # Validate sub-account parent
        if account_type == AccountType.SUB:
            if not parent_account_id:
                raise ValueError("Sub-account requires parent_account_id")

            parent = await self.repository.get_account(parent_account_id)
            if not parent:
                raise ValueError(f"Parent account {parent_account_id} not found")

            if parent.account_type != AccountType.MASTER:
                raise ValueError(
                    "Sub-accounts can only be created under MASTER accounts"
                )

        try:
            account = Account(
                account_id=str(uuid4()),
                parent_account_id=parent_account_id,
                account_type=account_type,
                balance_usdt=balance_usdt,
                tier=tier,
                permissions=permissions or {},
                compliance_settings=compliance_settings or {},
                last_sync=datetime.now(UTC),
                created_at=datetime.now(UTC),
            )

            # Store in repository
            await self.repository.save_account(account)

            # Cache the account
            self._accounts_cache[account.account_id] = account

            # Publish event
            await self.event_bus.publish(
                "account.created",
                {
                    "account_id": account.account_id,
                    "account_type": account_type.value,
                    "parent_account_id": parent_account_id,
                    "balance_usdt": str(balance_usdt),
                    "tier": tier.value,
                },
            )

            logger.info(
                "account_created",
                account_id=account.account_id,
                account_type=account_type.value,
                parent_id=parent_account_id,
                balance=str(balance_usdt),
            )

            return account

        except ValidationError as e:
            logger.error("account_creation_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def get_account(self, account_id: str) -> Account | None:
        """
        Retrieve an account by ID.

        Args:
            account_id: Account identifier

        Returns:
            Account instance if found, None otherwise
        """
        # Check cache first
        if account_id in self._accounts_cache:
            return self._accounts_cache[account_id]

        # Fetch from repository
        account = await self.repository.get_account(account_id)
        if account:
            self._accounts_cache[account_id] = account

        return account

    @requires_tier(TradingTier.STRATEGIST)
    async def list_accounts(
        self,
        parent_account_id: str | None = None,
        account_type: AccountType | None = None,
    ) -> list[Account]:
        """
        List accounts with optional filtering.

        Args:
            parent_account_id: Filter by parent account
            account_type: Filter by account type

        Returns:
            List of matching accounts
        """
        accounts = await self.repository.list_accounts()

        # Apply filters
        if parent_account_id:
            accounts = [a for a in accounts if a.parent_account_id == parent_account_id]

        if account_type:
            accounts = [a for a in accounts if a.account_type == account_type]

        # Update cache
        for account in accounts:
            self._accounts_cache[account.account_id] = account

        logger.info(
            "accounts_listed",
            count=len(accounts),
            parent_filter=parent_account_id,
            type_filter=account_type.value if account_type else None,
        )

        return accounts

    @requires_tier(TradingTier.STRATEGIST)
    async def switch_account(self, account_id: str) -> bool:
        """
        Switch the active trading account.

        Args:
            account_id: Account to switch to

        Returns:
            True if switch successful, False otherwise
        """
        account = await self.get_account(account_id)
        if not account:
            logger.error(
                "account_switch_failed", account_id=account_id, reason="not_found"
            )
            return False

        old_account_id = self._active_account_id
        self._active_account_id = account_id

        # Publish account switch event
        await self.event_bus.publish(
            "account.switched",
            {
                "old_account_id": old_account_id,
                "new_account_id": account_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        logger.info(
            "account_switched",
            old_account_id=old_account_id,
            new_account_id=account_id,
        )

        return True

    @requires_tier(TradingTier.STRATEGIST)
    async def get_active_account(self) -> Account | None:
        """
        Get the currently active account.

        Returns:
            Active Account instance if set, None otherwise
        """
        if self._active_account_id:
            return await self.get_account(self._active_account_id)
        return None

    @requires_tier(TradingTier.STRATEGIST)
    async def update_account_balance(
        self, account_id: str, new_balance: Decimal
    ) -> bool:
        """
        Update an account's balance.

        Args:
            account_id: Account to update
            new_balance: New balance value

        Returns:
            True if update successful, False otherwise
        """
        account = await self.get_account(account_id)
        if not account:
            logger.error(
                "balance_update_failed", account_id=account_id, reason="not_found"
            )
            return False

        old_balance = account.balance_usdt
        account.balance_usdt = new_balance
        account.last_sync = datetime.now(UTC)

        # Save to repository
        await self.repository.update_account(account)

        # Update cache
        self._accounts_cache[account_id] = account

        # Publish balance update event
        await self.event_bus.publish(
            "account.balance_updated",
            {
                "account_id": account_id,
                "old_balance": str(old_balance),
                "new_balance": str(new_balance),
                "timestamp": account.last_sync.isoformat(),
            },
        )

        logger.info(
            "account_balance_updated",
            account_id=account_id,
            old_balance=str(old_balance),
            new_balance=str(new_balance),
        )

        return True

    @requires_tier(TradingTier.STRATEGIST)
    async def get_account_positions(self, account_id: str) -> list[Position]:
        """
        Get all positions for an account.

        Args:
            account_id: Account identifier

        Returns:
            List of positions for the account
        """
        positions = await self.repository.list_positions(account_id=account_id)

        logger.info(
            "account_positions_retrieved",
            account_id=account_id,
            position_count=len(positions),
        )

        return positions

    @requires_tier(TradingTier.STRATEGIST)
    async def check_account_permissions(self, account_id: str, permission: str) -> bool:
        """
        Check if an account has a specific permission.

        Args:
            account_id: Account to check
            permission: Permission to verify

        Returns:
            True if permission granted, False otherwise
        """
        account = await self.get_account(account_id)
        if not account:
            return False

        # Check if permission exists and is True
        return account.permissions.get(permission, False)

    @requires_tier(TradingTier.STRATEGIST)
    async def update_account_permissions(
        self, account_id: str, permissions: dict
    ) -> bool:
        """
        Update an account's permissions.

        Args:
            account_id: Account to update
            permissions: New permissions dictionary

        Returns:
            True if update successful, False otherwise
        """
        account = await self.get_account(account_id)
        if not account:
            logger.error(
                "permission_update_failed", account_id=account_id, reason="not_found"
            )
            return False

        old_permissions = account.permissions.copy()
        account.permissions = permissions

        # Save to repository
        await self.repository.update_account(account)

        # Update cache
        self._accounts_cache[account_id] = account

        # Publish permission update event
        await self.event_bus.publish(
            "account.permissions_updated",
            {
                "account_id": account_id,
                "old_permissions": old_permissions,
                "new_permissions": permissions,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        logger.info(
            "account_permissions_updated",
            account_id=account_id,
            permission_count=len(permissions),
        )

        return True

    @requires_tier(TradingTier.STRATEGIST)
    async def aggregate_sub_account_balances(self, master_account_id: str) -> Decimal:
        """
        Calculate total balance across master and all sub-accounts.

        Args:
            master_account_id: Master account identifier

        Returns:
            Total aggregated balance
        """
        master = await self.get_account(master_account_id)
        if not master or master.account_type != AccountType.MASTER:
            raise ValueError(f"Invalid master account: {master_account_id}")

        total_balance = master.balance_usdt

        # Get all sub-accounts
        sub_accounts = await self.list_accounts(
            parent_account_id=master_account_id,
            account_type=AccountType.SUB,
        )

        # Sum sub-account balances
        for sub_account in sub_accounts:
            total_balance += sub_account.balance_usdt

        logger.info(
            "aggregated_balance_calculated",
            master_account_id=master_account_id,
            sub_account_count=len(sub_accounts),
            total_balance=str(total_balance),
        )

        return total_balance

    @requires_tier(TradingTier.STRATEGIST)
    async def validate_account_hierarchy(self, account_id: str) -> bool:
        """
        Validate account hierarchy integrity.

        Args:
            account_id: Account to validate

        Returns:
            True if hierarchy is valid, False otherwise
        """
        account = await self.get_account(account_id)
        if not account:
            return False

        # MASTER accounts should not have parents
        if account.account_type == AccountType.MASTER:
            return account.parent_account_id is None

        # SUB and PAPER accounts should have valid parent
        if account.account_type in [AccountType.SUB, AccountType.PAPER]:
            if not account.parent_account_id:
                return False

            parent = await self.get_account(account.parent_account_id)
            return parent is not None and parent.account_type == AccountType.MASTER

        return True

    async def close(self):
        """Clean up resources."""
        self._accounts_cache.clear()
        self._active_account_id = None
        logger.info("account_manager_closed")
