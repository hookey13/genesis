"""Tilt debt tracking and management system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import structlog

from genesis.core.events import EventType
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus
from genesis.tilt.detector import TiltLevel

logger = structlog.get_logger(__name__)


class TransactionType(Enum):
    """Type of debt transaction."""

    DEBT_ADDED = "DEBT_ADDED"
    DEBT_REDUCED = "DEBT_REDUCED"


@dataclass
class DebtTransaction:
    """Represents a tilt debt transaction."""

    ledger_id: str = field(default_factory=lambda: str(uuid4()))
    profile_id: str = ""
    transaction_type: TransactionType = TransactionType.DEBT_ADDED
    amount: Decimal = Decimal("0")
    balance_after: Decimal = Decimal("0")
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class TiltDebtCalculator:
    """Manages tilt debt calculation and tracking."""

    # Debt multipliers by tilt level
    DEBT_MULTIPLIERS = {
        TiltLevel.LEVEL1: Decimal("0.1"),  # 10% of losses become debt
        TiltLevel.LEVEL2: Decimal("0.25"),  # 25% of losses become debt
        TiltLevel.LEVEL3: Decimal("0.50"),  # 50% of losses become debt
    }

    # Minimum debt amount to track
    MIN_DEBT_AMOUNT = Decimal("1.0")

    def __init__(
        self,
        repository: Optional[SQLiteRepository] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """Initialize tilt debt calculator.

        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing debt events
        """
        self.repository = repository
        self.event_bus = event_bus

        # Current debt balances cache
        self.debt_balances: dict[str, Decimal] = {}

        # Transaction history cache (last 100 per profile)
        self.transaction_history: dict[str, list[DebtTransaction]] = {}

    def calculate_tilt_debt(
        self,
        tilt_level: TiltLevel,
        losses: Decimal,
    ) -> Decimal:
        """Calculate tilt debt based on level and losses.

        Args:
            tilt_level: Severity level of tilt
            losses: Amount of losses during tilt

        Returns:
            Calculated debt amount
        """
        if tilt_level not in self.DEBT_MULTIPLIERS:
            logger.warning(
                "No debt multiplier for tilt level",
                tilt_level=tilt_level.value,
            )
            return Decimal("0")

        if losses <= Decimal("0"):
            return Decimal("0")

        multiplier = self.DEBT_MULTIPLIERS[tilt_level]
        debt = losses * multiplier

        # Apply minimum threshold
        if debt < self.MIN_DEBT_AMOUNT:
            return Decimal("0")

        logger.debug(
            "Tilt debt calculated",
            tilt_level=tilt_level.value,
            losses=str(losses),
            multiplier=str(multiplier),
            debt=str(debt),
        )

        return debt

    async def add_to_debt_ledger(
        self,
        profile_id: str,
        debt_amount: Decimal,
        tilt_level: Optional[TiltLevel] = None,
        reason: Optional[str] = None,
    ) -> Optional[DebtTransaction]:
        """Add debt to a profile's ledger.

        Args:
            profile_id: Profile identifier
            debt_amount: Amount of debt to add
            tilt_level: Tilt level that caused debt
            reason: Reason for debt addition

        Returns:
            Created debt transaction or None if amount too small
        """
        if debt_amount < self.MIN_DEBT_AMOUNT:
            logger.debug(
                "Debt amount below minimum threshold",
                profile_id=profile_id,
                amount=str(debt_amount),
            )
            return None

        # Get current balance
        current_balance = self.get_current_debt(profile_id)
        new_balance = current_balance + debt_amount

        # Create transaction
        if not reason and tilt_level:
            reason = f"Tilt {tilt_level.value} episode"

        transaction = DebtTransaction(
            ledger_id=str(uuid4()),
            profile_id=profile_id,
            transaction_type=TransactionType.DEBT_ADDED,
            amount=debt_amount,
            balance_after=new_balance,
            reason=reason,
            timestamp=datetime.now(UTC),
        )

        # Update balance cache
        self.debt_balances[profile_id] = new_balance

        # Add to history cache
        if profile_id not in self.transaction_history:
            self.transaction_history[profile_id] = []
        self.transaction_history[profile_id].append(transaction)

        # Keep only last 100 transactions in cache
        if len(self.transaction_history[profile_id]) > 100:
            self.transaction_history[profile_id] = self.transaction_history[profile_id][
                -100:
            ]

        # Persist to database
        if self.repository:
            await self._persist_transaction(transaction)

        # Publish event
        if self.event_bus:
            await self._publish_debt_added_event(transaction)

        logger.info(
            "Debt added to ledger",
            profile_id=profile_id,
            amount=str(debt_amount),
            new_balance=str(new_balance),
            reason=reason,
        )

        return transaction

    async def reduce_debt(
        self,
        profile_id: str,
        profit_amount: Decimal,
        reason: Optional[str] = None,
    ) -> Decimal:
        """Reduce debt using trading profits.

        Args:
            profile_id: Profile identifier
            profit_amount: Profit amount to apply
            reason: Reason for reduction

        Returns:
            Amount of debt actually reduced
        """
        if profit_amount <= Decimal("0"):
            return Decimal("0")

        current_debt = self.get_current_debt(profile_id)

        if current_debt <= Decimal("0"):
            logger.debug("No debt to reduce", profile_id=profile_id)
            return Decimal("0")

        # Calculate reduction amount (capped at current debt)
        reduction = min(profit_amount, current_debt)
        new_balance = current_debt - reduction

        # Create transaction
        if not reason:
            reason = "Profitable trade debt paydown"

        transaction = DebtTransaction(
            ledger_id=str(uuid4()),
            profile_id=profile_id,
            transaction_type=TransactionType.DEBT_REDUCED,
            amount=reduction,
            balance_after=new_balance,
            reason=reason,
            timestamp=datetime.now(UTC),
        )

        # Update balance cache
        self.debt_balances[profile_id] = new_balance

        # Add to history cache
        if profile_id not in self.transaction_history:
            self.transaction_history[profile_id] = []
        self.transaction_history[profile_id].append(transaction)

        # Persist to database
        if self.repository:
            await self._persist_transaction(transaction)

        # Publish event
        if self.event_bus:
            await self._publish_debt_reduced_event(transaction)

        logger.info(
            "Debt reduced",
            profile_id=profile_id,
            reduction=str(reduction),
            new_balance=str(new_balance),
            reason=reason,
        )

        return reduction

    def get_current_debt(self, profile_id: str) -> Decimal:
        """Get current debt balance for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Current debt amount
        """
        return self.debt_balances.get(profile_id, Decimal("0"))

    def has_outstanding_debt(self, profile_id: str) -> bool:
        """Check if profile has outstanding debt.

        Args:
            profile_id: Profile identifier

        Returns:
            True if debt exists
        """
        return self.get_current_debt(profile_id) > Decimal("0")

    def get_debt_payoff_ratio(self, profile_id: str) -> Decimal:
        """Calculate debt payoff ratio for recovery progress.

        Args:
            profile_id: Profile identifier

        Returns:
            Ratio of debt paid (0 to 1, or 1 if no debt)
        """
        # Get initial debt from first transaction
        if profile_id not in self.transaction_history:
            return Decimal("1")  # No debt history

        history = self.transaction_history[profile_id]
        if not history:
            return Decimal("1")

        # Find first debt addition
        initial_debt = Decimal("0")
        for transaction in history:
            if transaction.transaction_type == TransactionType.DEBT_ADDED:
                initial_debt = max(initial_debt, transaction.balance_after)

        if initial_debt == Decimal("0"):
            return Decimal("1")  # No debt was ever added

        current_debt = self.get_current_debt(profile_id)
        paid_amount = initial_debt - current_debt

        return paid_amount / initial_debt

    async def get_transaction_history(
        self,
        profile_id: str,
        limit: int = 50,
    ) -> list[DebtTransaction]:
        """Get debt transaction history for a profile.

        Args:
            profile_id: Profile identifier
            limit: Maximum number of transactions to return

        Returns:
            List of debt transactions
        """
        # Check cache first
        if profile_id in self.transaction_history:
            cached = self.transaction_history[profile_id]
            if cached:
                return cached[-limit:] if limit else cached

        # Load from database if not in cache
        if self.repository:
            try:
                transactions_data = await self.repository.get_debt_transactions(
                    profile_id, limit
                )
                transactions = [
                    self._transaction_from_dict(data) for data in transactions_data
                ]

                # Update cache
                self.transaction_history[profile_id] = transactions

                return transactions
            except Exception as e:
                logger.error(
                    "Failed to load debt history",
                    profile_id=profile_id,
                    error=str(e),
                )

        return []

    def get_debt_statistics(self, profile_id: str) -> dict[str, Any]:
        """Get debt statistics for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Dictionary of debt statistics
        """
        current_debt = self.get_current_debt(profile_id)
        history = self.transaction_history.get(profile_id, [])

        total_added = Decimal("0")
        total_reduced = Decimal("0")
        peak_debt = Decimal("0")

        for transaction in history:
            if transaction.transaction_type == TransactionType.DEBT_ADDED:
                total_added += transaction.amount
            else:
                total_reduced += transaction.amount
            peak_debt = max(peak_debt, transaction.balance_after)

        return {
            "current_debt": str(current_debt),
            "has_debt": current_debt > Decimal("0"),
            "total_debt_added": str(total_added),
            "total_debt_reduced": str(total_reduced),
            "peak_debt": str(peak_debt),
            "payoff_ratio": str(self.get_debt_payoff_ratio(profile_id)),
            "transaction_count": len(history),
        }

    async def clear_debt(
        self, profile_id: str, reason: str = "Manual debt clearance"
    ) -> bool:
        """Clear all debt for a profile (emergency override).

        Args:
            profile_id: Profile identifier
            reason: Reason for clearance

        Returns:
            True if debt was cleared
        """
        current_debt = self.get_current_debt(profile_id)

        if current_debt <= Decimal("0"):
            return False

        # Create clearance transaction
        transaction = DebtTransaction(
            ledger_id=str(uuid4()),
            profile_id=profile_id,
            transaction_type=TransactionType.DEBT_REDUCED,
            amount=current_debt,
            balance_after=Decimal("0"),
            reason=reason,
            timestamp=datetime.now(UTC),
        )

        # Update balance
        self.debt_balances[profile_id] = Decimal("0")

        # Add to history
        if profile_id not in self.transaction_history:
            self.transaction_history[profile_id] = []
        self.transaction_history[profile_id].append(transaction)

        # Persist
        if self.repository:
            await self._persist_transaction(transaction)

        # Publish event
        if self.event_bus:
            await self._publish_debt_reduced_event(transaction)

        logger.warning(
            "Debt manually cleared",
            profile_id=profile_id,
            amount_cleared=str(current_debt),
            reason=reason,
        )

        return True

    async def load_debt_balances(self) -> None:
        """Load debt balances from database on startup."""
        if not self.repository:
            return

        try:
            balances = await self.repository.get_all_debt_balances()
            self.debt_balances = {
                profile_id: Decimal(balance) for profile_id, balance in balances.items()
            }

            logger.info(
                "Debt balances loaded",
                profiles_with_debt=len(
                    [b for b in self.debt_balances.values() if b > 0]
                ),
            )
        except Exception as e:
            logger.error(
                "Failed to load debt balances",
                error=str(e),
            )

    async def _persist_transaction(self, transaction: DebtTransaction) -> None:
        """Persist debt transaction to database.

        Args:
            transaction: Transaction to persist
        """
        if not self.repository:
            return

        try:
            await self.repository.save_debt_transaction(
                {
                    "ledger_id": transaction.ledger_id,
                    "profile_id": transaction.profile_id,
                    "transaction_type": transaction.transaction_type.value,
                    "amount": str(transaction.amount),
                    "balance_after": str(transaction.balance_after),
                    "reason": transaction.reason,
                    "timestamp": transaction.timestamp.isoformat(),
                    "created_at": transaction.created_at.isoformat(),
                }
            )
        except Exception as e:
            logger.error(
                "Failed to persist debt transaction",
                ledger_id=transaction.ledger_id,
                error=str(e),
            )

    async def _publish_debt_added_event(self, transaction: DebtTransaction) -> None:
        """Publish debt added event.

        Args:
            transaction: Debt addition transaction
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.TILT_DEBT_ADDED,
            {
                "profile_id": transaction.profile_id,
                "ledger_id": transaction.ledger_id,
                "amount_added": str(transaction.amount),
                "new_balance": str(transaction.balance_after),
                "reason": transaction.reason,
                "timestamp": transaction.timestamp.isoformat(),
            },
        )

    async def _publish_debt_reduced_event(self, transaction: DebtTransaction) -> None:
        """Publish debt reduced event.

        Args:
            transaction: Debt reduction transaction
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.TILT_DEBT_REDUCED,
            {
                "profile_id": transaction.profile_id,
                "ledger_id": transaction.ledger_id,
                "amount_reduced": str(transaction.amount),
                "new_balance": str(transaction.balance_after),
                "reason": transaction.reason,
                "timestamp": transaction.timestamp.isoformat(),
            },
        )

    def _transaction_from_dict(self, data: dict[str, Any]) -> DebtTransaction:
        """Create transaction from dictionary.

        Args:
            data: Transaction data

        Returns:
            Debt transaction object
        """
        return DebtTransaction(
            ledger_id=data["ledger_id"],
            profile_id=data["profile_id"],
            transaction_type=TransactionType(data["transaction_type"]),
            amount=Decimal(data["amount"]),
            balance_after=Decimal(data["balance_after"]),
            reason=data.get("reason"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            created_at=datetime.fromisoformat(
                data.get("created_at", data["timestamp"])
            ),
        )
