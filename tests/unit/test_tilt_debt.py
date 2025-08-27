"""Unit tests for tilt debt tracking system."""
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.tilt.detector import TiltLevel
from genesis.tilt.tilt_debt import (
    TiltDebtCalculator,
    TransactionType,
)


@pytest.fixture
def debt_calculator():
    """Create a debt calculator for testing."""
    return TiltDebtCalculator()


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = AsyncMock()
    repo.save_debt_transaction = AsyncMock()
    repo.get_debt_transactions = AsyncMock(return_value=[])
    repo.get_all_debt_balances = AsyncMock(return_value={})
    return repo


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


class TestDebtCalculation:
    """Test debt calculation logic."""

    def test_calculate_debt_level1(self, debt_calculator):
        """Test debt calculation for Level 1 tilt."""
        losses = Decimal("100")
        debt = debt_calculator.calculate_tilt_debt(TiltLevel.LEVEL1, losses)
        assert debt == Decimal("10")  # 10% of 100

    def test_calculate_debt_level2(self, debt_calculator):
        """Test debt calculation for Level 2 tilt."""
        losses = Decimal("100")
        debt = debt_calculator.calculate_tilt_debt(TiltLevel.LEVEL2, losses)
        assert debt == Decimal("25")  # 25% of 100

    def test_calculate_debt_level3(self, debt_calculator):
        """Test debt calculation for Level 3 tilt."""
        losses = Decimal("100")
        debt = debt_calculator.calculate_tilt_debt(TiltLevel.LEVEL3, losses)
        assert debt == Decimal("50")  # 50% of 100

    def test_calculate_debt_below_minimum(self, debt_calculator):
        """Test debt calculation below minimum threshold."""
        losses = Decimal("5")  # Small loss
        debt = debt_calculator.calculate_tilt_debt(TiltLevel.LEVEL1, losses)
        assert debt == Decimal("0")  # Below minimum threshold

    def test_calculate_debt_zero_losses(self, debt_calculator):
        """Test debt calculation with zero losses."""
        debt = debt_calculator.calculate_tilt_debt(TiltLevel.LEVEL3, Decimal("0"))
        assert debt == Decimal("0")

    def test_calculate_debt_negative_losses(self, debt_calculator):
        """Test debt calculation with negative losses (profit)."""
        debt = debt_calculator.calculate_tilt_debt(TiltLevel.LEVEL2, Decimal("-50"))
        assert debt == Decimal("0")


class TestDebtLedger:
    """Test debt ledger operations."""

    @pytest.mark.asyncio
    async def test_add_to_debt_ledger(self, mock_repository, mock_event_bus):
        """Test adding debt to ledger."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"
        debt_amount = Decimal("100")

        transaction = await calculator.add_to_debt_ledger(
            profile_id,
            debt_amount,
            tilt_level=TiltLevel.LEVEL2,
            reason="Test debt"
        )

        assert transaction is not None
        assert transaction.profile_id == profile_id
        assert transaction.amount == debt_amount
        assert transaction.balance_after == debt_amount
        assert transaction.transaction_type == TransactionType.DEBT_ADDED

        # Check persistence and event
        mock_repository.save_debt_transaction.assert_called_once()
        mock_event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_below_minimum_debt(self, debt_calculator):
        """Test adding debt below minimum threshold."""
        profile_id = "test_profile"

        transaction = await debt_calculator.add_to_debt_ledger(
            profile_id,
            Decimal("0.5")  # Below minimum
        )

        assert transaction is None

    @pytest.mark.asyncio
    async def test_cumulative_debt_addition(self, mock_repository, mock_event_bus):
        """Test that debt accumulates correctly."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add first debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))
        assert calculator.get_current_debt(profile_id) == Decimal("100")

        # Add more debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("50"))
        assert calculator.get_current_debt(profile_id) == Decimal("150")


class TestDebtReduction:
    """Test debt reduction operations."""

    @pytest.mark.asyncio
    async def test_reduce_debt(self, mock_repository, mock_event_bus):
        """Test reducing debt with profits."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add initial debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))

        # Reduce debt
        reduction = await calculator.reduce_debt(profile_id, Decimal("30"))

        assert reduction == Decimal("30")
        assert calculator.get_current_debt(profile_id) == Decimal("70")

    @pytest.mark.asyncio
    async def test_reduce_debt_capped_at_balance(self, mock_repository, mock_event_bus):
        """Test that debt reduction is capped at current balance."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("50"))

        # Try to reduce more than current debt
        reduction = await calculator.reduce_debt(profile_id, Decimal("100"))

        assert reduction == Decimal("50")  # Capped at debt amount
        assert calculator.get_current_debt(profile_id) == Decimal("0")

    @pytest.mark.asyncio
    async def test_reduce_debt_with_zero_balance(self, debt_calculator):
        """Test reducing debt when no debt exists."""
        reduction = await debt_calculator.reduce_debt("test_profile", Decimal("50"))
        assert reduction == Decimal("0")

    @pytest.mark.asyncio
    async def test_reduce_debt_with_negative_amount(self, debt_calculator):
        """Test reducing debt with negative amount."""
        reduction = await debt_calculator.reduce_debt("test_profile", Decimal("-50"))
        assert reduction == Decimal("0")


class TestDebtQueries:
    """Test debt query operations."""

    @pytest.mark.asyncio
    async def test_get_current_debt(self, mock_repository, mock_event_bus):
        """Test getting current debt balance."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # No debt initially
        assert calculator.get_current_debt(profile_id) == Decimal("0")

        # Add debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))
        assert calculator.get_current_debt(profile_id) == Decimal("100")

    @pytest.mark.asyncio
    async def test_has_outstanding_debt(self, mock_repository, mock_event_bus):
        """Test checking for outstanding debt."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # No debt
        assert calculator.has_outstanding_debt(profile_id) is False

        # Add debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))
        assert calculator.has_outstanding_debt(profile_id) is True

        # Pay off debt
        await calculator.reduce_debt(profile_id, Decimal("100"))
        assert calculator.has_outstanding_debt(profile_id) is False

    def test_get_debt_payoff_ratio_no_history(self, debt_calculator):
        """Test payoff ratio with no debt history."""
        ratio = debt_calculator.get_debt_payoff_ratio("test_profile")
        assert ratio == Decimal("1")  # No debt = 100% paid

    @pytest.mark.asyncio
    async def test_get_debt_payoff_ratio(self, mock_repository, mock_event_bus):
        """Test calculating debt payoff ratio."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))

        # Pay off 30%
        await calculator.reduce_debt(profile_id, Decimal("30"))

        ratio = calculator.get_debt_payoff_ratio(profile_id)
        assert ratio == Decimal("0.3")  # 30% paid


class TestDebtStatistics:
    """Test debt statistics calculation."""

    @pytest.mark.asyncio
    async def test_get_debt_statistics(self, mock_repository, mock_event_bus):
        """Test getting comprehensive debt statistics."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add and reduce debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))
        await calculator.add_to_debt_ledger(profile_id, Decimal("50"))
        await calculator.reduce_debt(profile_id, Decimal("30"))

        stats = calculator.get_debt_statistics(profile_id)

        assert stats["current_debt"] == "120"  # 100 + 50 - 30
        assert stats["has_debt"] is True
        assert stats["total_debt_added"] == "150"
        assert stats["total_debt_reduced"] == "30"
        assert stats["peak_debt"] == "150"
        assert stats["transaction_count"] == 3

    def test_get_debt_statistics_no_history(self, debt_calculator):
        """Test getting statistics with no debt history."""
        stats = debt_calculator.get_debt_statistics("test_profile")

        assert stats["current_debt"] == "0"
        assert stats["has_debt"] is False
        assert stats["total_debt_added"] == "0"
        assert stats["total_debt_reduced"] == "0"
        assert stats["peak_debt"] == "0"
        assert stats["transaction_count"] == 0


class TestDebtClearance:
    """Test debt clearance operations."""

    @pytest.mark.asyncio
    async def test_clear_debt(self, mock_repository, mock_event_bus):
        """Test clearing all debt."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add debt
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))

        # Clear debt
        cleared = await calculator.clear_debt(profile_id, "Emergency clearance")

        assert cleared is True
        assert calculator.get_current_debt(profile_id) == Decimal("0")

        # Check that clearance transaction was created
        assert mock_event_bus.publish.call_count >= 2  # Add + clear

    @pytest.mark.asyncio
    async def test_clear_debt_no_balance(self, debt_calculator):
        """Test clearing debt when no debt exists."""
        cleared = await debt_calculator.clear_debt("test_profile")
        assert cleared is False


class TestTransactionHistory:
    """Test transaction history management."""

    @pytest.mark.asyncio
    async def test_transaction_history_cache(self, mock_repository, mock_event_bus):
        """Test that transactions are cached."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add transactions
        await calculator.add_to_debt_ledger(profile_id, Decimal("100"))
        await calculator.reduce_debt(profile_id, Decimal("30"))

        # Check cache
        assert profile_id in calculator.transaction_history
        assert len(calculator.transaction_history[profile_id]) == 2

    @pytest.mark.asyncio
    async def test_transaction_history_limit(self, mock_repository, mock_event_bus):
        """Test that transaction history is limited to 100 entries."""
        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )
        profile_id = "test_profile"

        # Add more than 100 transactions
        for i in range(105):
            await calculator.add_to_debt_ledger(profile_id, Decimal("10"))

        # Should only keep last 100
        assert len(calculator.transaction_history[profile_id]) == 100

    @pytest.mark.asyncio
    async def test_get_transaction_history(self, mock_repository):
        """Test retrieving transaction history."""
        calculator = TiltDebtCalculator(repository=mock_repository)

        # Mock some transactions
        mock_repository.get_debt_transactions.return_value = [
            {
                "ledger_id": "txn_1",
                "profile_id": "test_profile",
                "transaction_type": "DEBT_ADDED",
                "amount": "100",
                "balance_after": "100",
                "reason": "Test",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]

        history = await calculator.get_transaction_history("test_profile")
        assert len(history) == 1
        assert history[0].amount == Decimal("100")


class TestDebtPersistence:
    """Test debt persistence operations."""

    @pytest.mark.asyncio
    async def test_load_debt_balances(self, mock_repository):
        """Test loading debt balances from database."""
        mock_repository.get_all_debt_balances.return_value = {
            "profile_1": "100",
            "profile_2": "250.50",
        }

        calculator = TiltDebtCalculator(repository=mock_repository)
        await calculator.load_debt_balances()

        assert calculator.get_current_debt("profile_1") == Decimal("100")
        assert calculator.get_current_debt("profile_2") == Decimal("250.50")

    @pytest.mark.asyncio
    async def test_persist_transaction_error_handling(self, mock_repository, mock_event_bus):
        """Test error handling when persisting transaction."""
        mock_repository.save_debt_transaction.side_effect = Exception("Database error")

        calculator = TiltDebtCalculator(
            repository=mock_repository,
            event_bus=mock_event_bus
        )

        # Should not raise exception
        transaction = await calculator.add_to_debt_ledger(
            "test_profile",
            Decimal("100")
        )

        assert transaction is not None
        mock_repository.save_debt_transaction.assert_called_once()
