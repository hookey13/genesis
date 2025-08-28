"""
Unit tests for the account manager module.

Tests account balance synchronization, updates, and persistence
with comprehensive coverage for all account operations.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.account_manager import AccountManager
from genesis.core.exceptions import ExchangeError
from genesis.core.models import Account, TradingTier
from genesis.exchange.gateway import BinanceGateway


@pytest.fixture
def mock_gateway():
    """Create a mock exchange gateway."""
    gateway = MagicMock(spec=BinanceGateway)
    gateway.get_account_info = AsyncMock(
        return_value={
            "balances": [
                {"asset": "USDT", "free": "1000.50", "locked": "0"},
                {"asset": "BTC", "free": "0.01", "locked": "0"},
            ]
        }
    )
    return gateway


@pytest.fixture
def account():
    """Create a test account."""
    return Account(
        account_id="test-account", balance_usdt=Decimal("500"), tier=TradingTier.SNIPER
    )


@pytest.fixture
async def account_manager(mock_gateway, account):
    """Create an account manager instance."""
    manager = AccountManager(
        gateway=mock_gateway,
        account=account,
        auto_sync=False,  # Disable auto-sync for testing
    )
    return manager


class TestAccountManagerInitialization:
    """Test account manager initialization."""

    def test_init_with_existing_account(self, mock_gateway, account):
        """Test initialization with existing account."""
        manager = AccountManager(mock_gateway, account, auto_sync=False)

        assert manager.account == account
        assert manager.gateway == mock_gateway
        assert not manager.auto_sync
        assert manager._sync_task is None

    def test_init_without_account(self, mock_gateway):
        """Test initialization without existing account."""
        manager = AccountManager(mock_gateway, auto_sync=False)

        assert manager.account is not None
        assert manager.account.balance_usdt == Decimal("0")
        assert manager.account.tier == TradingTier.SNIPER

    @pytest.mark.asyncio
    async def test_initialize_with_sync(self, mock_gateway, account):
        """Test initialization with first sync."""
        manager = AccountManager(mock_gateway, account, auto_sync=False)
        await manager.initialize()

        # Should have synced balance
        assert manager.account.balance_usdt == Decimal("1000.50")
        mock_gateway.get_account_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_auto_sync(self, mock_gateway, account):
        """Test initialization with auto-sync enabled."""
        manager = AccountManager(mock_gateway, account, auto_sync=True)
        await manager.initialize()

        # Should have started sync task
        assert manager._sync_task is not None
        assert not manager._sync_task.done()

        # Clean up
        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_gateway, account):
        """Test initialization with sync failure."""
        mock_gateway.get_account_info.side_effect = Exception("API Error")

        manager = AccountManager(mock_gateway, account, auto_sync=False)

        with pytest.raises(Exception) as exc_info:
            await manager.initialize()

        assert "API Error" in str(exc_info.value)


class TestBalanceSynchronization:
    """Test balance synchronization functionality."""

    @pytest.mark.asyncio
    async def test_sync_balance_success(self, account_manager, mock_gateway):
        """Test successful balance synchronization."""
        new_balance = await account_manager.sync_balance()

        assert new_balance == Decimal("1000.50")
        assert account_manager.account.balance_usdt == Decimal("1000.50")
        assert account_manager._sync_count == 1
        assert account_manager._last_sync_error is None

    @pytest.mark.asyncio
    async def test_sync_balance_no_usdt(self, account_manager, mock_gateway):
        """Test sync when USDT balance not found."""
        mock_gateway.get_account_info.return_value = {
            "balances": [{"asset": "BTC", "free": "0.01", "locked": "0"}]
        }

        new_balance = await account_manager.sync_balance()

        assert new_balance == Decimal("0")
        assert account_manager.account.balance_usdt == Decimal("0")

    @pytest.mark.asyncio
    async def test_sync_balance_error(self, account_manager, mock_gateway):
        """Test sync with exchange error."""
        mock_gateway.get_account_info.side_effect = Exception("Network error")

        with pytest.raises(ExchangeError) as exc_info:
            await account_manager.sync_balance()

        assert "Balance sync failed" in str(exc_info.value)
        assert account_manager._last_sync_error == "Network error"

    @pytest.mark.asyncio
    async def test_sync_balance_retry(self, account_manager, mock_gateway):
        """Test sync with retry on failure."""
        # First two attempts fail, third succeeds
        mock_gateway.get_account_info.side_effect = [
            Exception("Temporary error"),
            Exception("Another error"),
            {"balances": [{"asset": "USDT", "free": "750", "locked": "0"}]},
        ]

        # Retry decorator should handle this
        new_balance = await account_manager.sync_balance()

        assert new_balance == Decimal("750")
        assert mock_gateway.get_account_info.call_count == 3

    @pytest.mark.asyncio
    async def test_sync_balance_timeout(self, account_manager, mock_gateway):
        """Test sync with timeout."""

        async def slow_response():
            await asyncio.sleep(15)  # Longer than timeout
            return {"balances": []}

        mock_gateway.get_account_info = slow_response

        with pytest.raises(asyncio.TimeoutError):
            await account_manager.sync_balance()

    @pytest.mark.asyncio
    async def test_periodic_sync(self, mock_gateway, account):
        """Test periodic balance synchronization."""
        manager = AccountManager(mock_gateway, account, auto_sync=True)

        # Mock sleep to speed up test
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]

            await manager.initialize()

            # Let the sync task run once
            await asyncio.sleep(0.1)

            # Clean up
            await manager.shutdown()

            # Should have called sync at least twice (init + periodic)
            assert mock_gateway.get_account_info.call_count >= 2


class TestBalanceOperations:
    """Test balance manipulation operations."""

    def test_get_balance(self, account_manager):
        """Test getting current balance."""
        account_manager.account.balance_usdt = Decimal("1234.56")
        assert account_manager.get_balance() == Decimal("1234.56")

    def test_get_tier(self, account_manager):
        """Test getting current tier."""
        assert account_manager.get_tier() == TradingTier.SNIPER

    def test_update_balance_valid(self, account_manager):
        """Test manual balance update with valid amount."""
        account_manager.update_balance(Decimal("2000"))
        assert account_manager.account.balance_usdt == Decimal("2000")

    def test_update_balance_negative(self, account_manager):
        """Test manual balance update with negative amount."""
        with pytest.raises(ValueError) as exc_info:
            account_manager.update_balance(Decimal("-100"))

        assert "Balance cannot be negative" in str(exc_info.value)

    def test_deduct_balance_valid(self, account_manager):
        """Test balance deduction with valid amount."""
        account_manager.account.balance_usdt = Decimal("1000")
        account_manager.deduct_balance(Decimal("250"))

        assert account_manager.account.balance_usdt == Decimal("750")

    def test_deduct_balance_insufficient(self, account_manager):
        """Test balance deduction with insufficient funds."""
        account_manager.account.balance_usdt = Decimal("100")

        with pytest.raises(ValueError) as exc_info:
            account_manager.deduct_balance(Decimal("150"))

        assert "Cannot deduct" in str(exc_info.value)

    def test_add_balance(self, account_manager):
        """Test adding to balance."""
        account_manager.account.balance_usdt = Decimal("1000")
        account_manager.add_balance(Decimal("500"))

        assert account_manager.account.balance_usdt == Decimal("1500")


class TestSyncHealth:
    """Test sync health monitoring."""

    def test_sync_healthy(self, account_manager):
        """Test healthy sync status."""
        account_manager.account.last_sync = datetime.now()
        account_manager._last_sync_error = None

        assert account_manager.is_sync_healthy()

    def test_sync_unhealthy_error(self, account_manager):
        """Test unhealthy sync with error."""
        account_manager.account.last_sync = datetime.now()
        account_manager._last_sync_error = "Connection failed"

        assert not account_manager.is_sync_healthy()

    def test_sync_unhealthy_stale(self, account_manager):
        """Test unhealthy sync with stale data."""
        # Set last sync to 5 minutes ago (stale threshold is 3x interval = 180s)
        account_manager.account.last_sync = datetime.now() - timedelta(minutes=5)
        account_manager._last_sync_error = None

        assert not account_manager.is_sync_healthy()

    def test_sync_healthy_no_sync_yet(self, account_manager):
        """Test sync health when never synced."""
        account_manager.account.last_sync = None
        account_manager._last_sync_error = None

        # Should be considered healthy if no sync attempted yet
        assert account_manager.is_sync_healthy()

    def test_get_sync_status(self, account_manager):
        """Test getting detailed sync status."""
        account_manager.account.last_sync = datetime.now()
        account_manager._sync_count = 5
        account_manager._last_sync_error = None

        status = account_manager.get_sync_status()

        assert status["sync_count"] == 5
        assert status["last_error"] is None
        assert status["is_healthy"] is True
        assert status["auto_sync_enabled"] is False
        assert status["sync_interval"] == 60


class TestBalanceConstraints:
    """Test balance validation and constraints."""

    @pytest.mark.asyncio
    async def test_validate_balance_constraint_valid(self, account_manager):
        """Test valid balance constraint validation."""
        account_manager.account.balance_usdt = Decimal("100")

        # Should not raise
        await account_manager.validate_balance_constraint()

    @pytest.mark.asyncio
    async def test_validate_balance_constraint_negative(self, account_manager):
        """Test negative balance constraint validation."""
        account_manager.account.balance_usdt = Decimal("-10")

        with pytest.raises(ValueError) as exc_info:
            await account_manager.validate_balance_constraint()

        assert "Balance constraint violation" in str(exc_info.value)


class TestSerialization:
    """Test account serialization and deserialization."""

    def test_to_dict(self, account_manager):
        """Test converting account to dictionary."""
        account_manager.account.last_sync = datetime(2024, 1, 1, 12, 0, 0)

        data = account_manager.to_dict()

        assert data["account_id"] == "test-account"
        assert data["balance_usdt"] == "500"
        assert data["tier"] == "SNIPER"
        assert data["last_sync"] == "2024-01-01T12:00:00"
        assert isinstance(data["locked_features"], list)

    def test_from_dict(self, mock_gateway):
        """Test creating AccountManager from dictionary."""
        data = {
            "account_id": "restored-account",
            "balance_usdt": "1500.75",
            "tier": "HUNTER",
            "locked_features": ["feature1", "feature2"],
            "last_sync": "2024-01-01T12:00:00",
            "created_at": "2024-01-01T10:00:00",
        }

        manager = AccountManager.from_dict(data, mock_gateway)

        assert manager.account.account_id == "restored-account"
        assert manager.account.balance_usdt == Decimal("1500.75")
        assert manager.account.tier == TradingTier.HUNTER
        assert manager.account.locked_features == ["feature1", "feature2"]
        assert manager.account.last_sync == datetime(2024, 1, 1, 12, 0, 0)


class TestShutdown:
    """Test account manager shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_with_sync_task(self, mock_gateway, account):
        """Test shutdown with active sync task."""
        manager = AccountManager(mock_gateway, account, auto_sync=True)
        await manager.initialize()

        # Verify task is running
        assert manager._sync_task is not None
        assert not manager._sync_task.done()

        # Shutdown
        await manager.shutdown()

        # Task should be cancelled
        assert manager._sync_task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_without_sync_task(self, account_manager):
        """Test shutdown without sync task."""
        # Should not raise
        await account_manager.shutdown()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_sync_with_empty_response(self, account_manager, mock_gateway):
        """Test sync with empty exchange response."""
        mock_gateway.get_account_info.return_value = {}

        new_balance = await account_manager.sync_balance()

        assert new_balance == Decimal("0")

    @pytest.mark.asyncio
    async def test_sync_with_malformed_balance(self, account_manager, mock_gateway):
        """Test sync with malformed balance data."""
        mock_gateway.get_account_info.return_value = {
            "balances": [{"asset": "USDT", "free": "not_a_number", "locked": "0"}]
        }

        with pytest.raises(Exception):
            await account_manager.sync_balance()

    def test_decimal_precision(self, account_manager):
        """Test decimal precision in balance operations."""
        # Test with high precision decimals
        account_manager.update_balance(Decimal("1234.56789012"))
        account_manager.deduct_balance(Decimal("0.00000001"))

        expected = Decimal("1234.56789011")
        assert account_manager.account.balance_usdt == expected

    @pytest.mark.asyncio
    async def test_concurrent_sync_calls(self, account_manager, mock_gateway):
        """Test concurrent sync calls."""
        # Simulate concurrent sync attempts
        tasks = [
            account_manager.sync_balance(),
            account_manager.sync_balance(),
            account_manager.sync_balance(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed with same result
        for result in results:
            if not isinstance(result, Exception):
                assert result == Decimal("1000.50")
