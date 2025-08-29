"""Unit tests for multi-account management infrastructure."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.core.account_manager import AccountManager
from genesis.core.constants import TradingTier
from genesis.core.models import Account, AccountType, Position, PositionSide


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    repo = AsyncMock()
    repo.save_account = AsyncMock()
    repo.update_account = AsyncMock()
    repo.get_account = AsyncMock(return_value=None)
    repo.list_accounts = AsyncMock(return_value=[])
    repo.list_positions = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def account_manager(mock_repository, mock_event_bus):
    """Create AccountManager instance."""
    return AccountManager(mock_repository, mock_event_bus)


@pytest.fixture
def sample_master_account():
    """Create sample master account."""
    return Account(
        account_id=str(uuid4()),
        account_type=AccountType.MASTER,
        balance_usdt=Decimal("10000"),
        tier=TradingTier.STRATEGIST,
        permissions={"trading": True, "withdrawals": True},
        compliance_settings={"reporting": "monthly"},
        last_sync=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_sub_account():
    """Create sample sub-account."""
    parent_id = str(uuid4())
    return Account(
        account_id=str(uuid4()),
        parent_account_id=parent_id,
        account_type=AccountType.SUB,
        balance_usdt=Decimal("5000"),
        tier=TradingTier.STRATEGIST,
        permissions={"trading": True, "withdrawals": False},
        compliance_settings={"reporting": "weekly"},
        last_sync=datetime.now(UTC),
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_create_master_account(account_manager, mock_repository, mock_event_bus):
    """Test creating a master account."""
    # Create master account
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        account = await account_manager.create_account(
            balance_usdt=Decimal("10000"),
            account_type=AccountType.MASTER,
            tier=TradingTier.STRATEGIST,
            permissions={"trading": True},
            compliance_settings={"reporting": "monthly"},
        )

    # Verify account created
    assert account.account_type == AccountType.MASTER
    assert account.balance_usdt == Decimal("10000")
    assert account.tier == TradingTier.STRATEGIST
    assert account.parent_account_id is None
    assert account.permissions == {"trading": True}
    assert account.compliance_settings == {"reporting": "monthly"}

    # Verify repository called
    mock_repository.save_account.assert_called_once()

    # Verify event published
    mock_event_bus.publish.assert_called_once_with(
        "account.created",
        {
            "account_id": account.account_id,
            "account_type": "MASTER",
            "parent_account_id": None,
            "balance_usdt": "10000",
            "tier": "STRATEGIST",
        },
    )


@pytest.mark.asyncio
async def test_create_sub_account_with_valid_parent(
    account_manager, mock_repository, mock_event_bus, sample_master_account
):
    """Test creating a sub-account with valid parent."""
    # Setup parent account
    mock_repository.get_account.return_value = sample_master_account

    # Create sub-account
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        account = await account_manager.create_account(
            balance_usdt=Decimal("5000"),
            account_type=AccountType.SUB,
            parent_account_id=sample_master_account.account_id,
            tier=TradingTier.STRATEGIST,
        )

    # Verify sub-account created
    assert account.account_type == AccountType.SUB
    assert account.parent_account_id == sample_master_account.account_id
    assert account.balance_usdt == Decimal("5000")

    # Verify parent lookup
    mock_repository.get_account.assert_called_once_with(
        sample_master_account.account_id
    )

    # Verify repository save
    mock_repository.save_account.assert_called_once()


@pytest.mark.asyncio
async def test_create_sub_account_without_parent_fails(account_manager):
    """Test creating sub-account without parent fails."""
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        with pytest.raises(ValueError, match="Sub-account requires parent_account_id"):
            await account_manager.create_account(
                balance_usdt=Decimal("5000"),
                account_type=AccountType.SUB,
            )


@pytest.mark.asyncio
async def test_create_sub_account_with_invalid_parent_fails(
    account_manager, mock_repository
):
    """Test creating sub-account with non-existent parent fails."""
    # Parent doesn't exist
    mock_repository.get_account.return_value = None

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        with pytest.raises(ValueError, match="Parent account .* not found"):
            await account_manager.create_account(
                balance_usdt=Decimal("5000"),
                account_type=AccountType.SUB,
                parent_account_id=str(uuid4()),
            )


@pytest.mark.asyncio
async def test_get_account_from_cache(account_manager, sample_master_account):
    """Test retrieving account from cache."""
    # Put account in cache
    account_manager._accounts_cache[sample_master_account.account_id] = (
        sample_master_account
    )

    # Get account
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        account = await account_manager.get_account(sample_master_account.account_id)

    # Verify retrieved from cache
    assert account == sample_master_account
    account_manager.repository.get_account.assert_not_called()


@pytest.mark.asyncio
async def test_get_account_from_repository(
    account_manager, mock_repository, sample_master_account
):
    """Test retrieving account from repository."""
    # Setup repository return
    mock_repository.get_account.return_value = sample_master_account

    # Get account
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        account = await account_manager.get_account(sample_master_account.account_id)

    # Verify retrieved from repository
    assert account == sample_master_account
    mock_repository.get_account.assert_called_once_with(
        sample_master_account.account_id
    )

    # Verify cached
    assert (
        account_manager._accounts_cache[sample_master_account.account_id]
        == sample_master_account
    )


@pytest.mark.asyncio
async def test_list_accounts_with_filters(
    account_manager, mock_repository, sample_master_account, sample_sub_account
):
    """Test listing accounts with filters."""
    # Setup repository return
    mock_repository.list_accounts.return_value = [
        sample_master_account,
        sample_sub_account,
    ]

    # List only sub-accounts
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        accounts = await account_manager.list_accounts(account_type=AccountType.SUB)

    # Verify filter applied
    assert len(accounts) == 1
    assert accounts[0] == sample_sub_account

    # Verify accounts cached (only the filtered ones are in the returned list but both are cached)
    assert sample_sub_account.account_id in account_manager._accounts_cache


@pytest.mark.asyncio
async def test_switch_account(
    account_manager, mock_repository, mock_event_bus, sample_master_account
):
    """Test switching active account."""
    # Setup account
    mock_repository.get_account.return_value = sample_master_account

    # Switch account
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        success = await account_manager.switch_account(sample_master_account.account_id)

    assert success
    assert account_manager._active_account_id == sample_master_account.account_id

    # Verify event published
    mock_event_bus.publish.assert_called_once()
    event_name, event_data = mock_event_bus.publish.call_args[0]
    assert event_name == "account.switched"
    assert event_data["new_account_id"] == sample_master_account.account_id


@pytest.mark.asyncio
async def test_switch_to_nonexistent_account_fails(account_manager, mock_repository):
    """Test switching to non-existent account fails."""
    # Account doesn't exist
    mock_repository.get_account.return_value = None

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        success = await account_manager.switch_account(str(uuid4()))

    assert not success
    assert account_manager._active_account_id is None


@pytest.mark.asyncio
async def test_update_account_balance(
    account_manager, mock_repository, mock_event_bus, sample_master_account
):
    """Test updating account balance."""
    # Setup account
    old_balance = sample_master_account.balance_usdt
    mock_repository.get_account.return_value = sample_master_account

    # Update balance
    new_balance = Decimal("15000")
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        success = await account_manager.update_account_balance(
            sample_master_account.account_id, new_balance
        )

    assert success
    assert sample_master_account.balance_usdt == new_balance

    # Verify repository update
    mock_repository.update_account.assert_called_once_with(sample_master_account)

    # Verify event published
    mock_event_bus.publish.assert_called_once_with(
        "account.balance_updated",
        {
            "account_id": sample_master_account.account_id,
            "old_balance": str(old_balance),
            "new_balance": str(new_balance),
            "timestamp": sample_master_account.last_sync.isoformat(),
        },
    )


@pytest.mark.asyncio
async def test_get_account_positions(account_manager, mock_repository):
    """Test getting account positions."""
    account_id = str(uuid4())

    # Create sample positions
    positions = [
        Position(
            position_id=str(uuid4()),
            account_id=account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            dollar_value=Decimal("5000"),
        ),
        Position(
            position_id=str(uuid4()),
            account_id=account_id,
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("3000"),
            quantity=Decimal("1"),
            dollar_value=Decimal("3000"),
        ),
    ]

    mock_repository.list_positions.return_value = positions

    # Get positions
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        result = await account_manager.get_account_positions(account_id)

    assert len(result) == 2
    assert result == positions
    mock_repository.list_positions.assert_called_once_with(account_id=account_id)


@pytest.mark.asyncio
async def test_check_account_permissions(
    account_manager, mock_repository, sample_master_account
):
    """Test checking account permissions."""
    # Setup account
    mock_repository.get_account.return_value = sample_master_account

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        # Check existing permission
        has_trading = await account_manager.check_account_permissions(
            sample_master_account.account_id, "trading"
        )
        assert has_trading

        # Check non-existent permission
        has_admin = await account_manager.check_account_permissions(
            sample_master_account.account_id, "admin"
        )
        assert not has_admin


@pytest.mark.asyncio
async def test_update_account_permissions(
    account_manager, mock_repository, mock_event_bus, sample_master_account
):
    """Test updating account permissions."""
    # Setup account
    mock_repository.get_account.return_value = sample_master_account
    old_permissions = sample_master_account.permissions.copy()

    # Update permissions
    new_permissions = {"trading": True, "withdrawals": False, "admin": True}
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        success = await account_manager.update_account_permissions(
            sample_master_account.account_id, new_permissions
        )

    assert success
    assert sample_master_account.permissions == new_permissions

    # Verify repository update
    mock_repository.update_account.assert_called_once_with(sample_master_account)

    # Verify event published
    mock_event_bus.publish.assert_called_once_with(
        "account.permissions_updated",
        {
            "account_id": sample_master_account.account_id,
            "old_permissions": old_permissions,
            "new_permissions": new_permissions,
            "timestamp": mock_event_bus.publish.call_args[0][1]["timestamp"],
        },
    )


@pytest.mark.asyncio
async def test_aggregate_sub_account_balances(account_manager, mock_repository):
    """Test aggregating balances across master and sub-accounts."""
    master_id = str(uuid4())

    # Create accounts
    master = Account(
        account_id=master_id,
        account_type=AccountType.MASTER,
        balance_usdt=Decimal("10000"),
        tier=TradingTier.STRATEGIST,
    )

    sub1 = Account(
        account_id=str(uuid4()),
        parent_account_id=master_id,
        account_type=AccountType.SUB,
        balance_usdt=Decimal("5000"),
        tier=TradingTier.STRATEGIST,
    )

    sub2 = Account(
        account_id=str(uuid4()),
        parent_account_id=master_id,
        account_type=AccountType.SUB,
        balance_usdt=Decimal("3000"),
        tier=TradingTier.STRATEGIST,
    )

    # Setup repository
    mock_repository.get_account.return_value = master
    mock_repository.list_accounts.return_value = [master, sub1, sub2]

    # Aggregate balances
    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        total = await account_manager.aggregate_sub_account_balances(master_id)

    assert total == Decimal("18000")  # 10000 + 5000 + 3000


@pytest.mark.asyncio
async def test_aggregate_balances_invalid_master_fails(
    account_manager, mock_repository
):
    """Test aggregating balances with invalid master account fails."""
    # Account doesn't exist
    mock_repository.get_account.return_value = None

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        with pytest.raises(ValueError, match="Invalid master account"):
            await account_manager.aggregate_sub_account_balances(str(uuid4()))


@pytest.mark.asyncio
async def test_validate_account_hierarchy_master(
    account_manager, mock_repository, sample_master_account
):
    """Test validating master account hierarchy."""
    # Setup account
    mock_repository.get_account.return_value = sample_master_account

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        valid = await account_manager.validate_account_hierarchy(
            sample_master_account.account_id
        )

    assert valid  # Master account with no parent is valid


@pytest.mark.asyncio
async def test_validate_account_hierarchy_sub_with_valid_parent(
    account_manager, mock_repository
):
    """Test validating sub-account with valid parent."""
    master_id = str(uuid4())
    sub_id = str(uuid4())

    # Create accounts
    master = Account(
        account_id=master_id,
        account_type=AccountType.MASTER,
        balance_usdt=Decimal("10000"),
        tier=TradingTier.STRATEGIST,
    )

    sub = Account(
        account_id=sub_id,
        parent_account_id=master_id,
        account_type=AccountType.SUB,
        balance_usdt=Decimal("5000"),
        tier=TradingTier.STRATEGIST,
    )

    # Setup repository
    def get_account_side_effect(account_id):
        if account_id == sub_id:
            return sub
        elif account_id == master_id:
            return master
        return None

    mock_repository.get_account.side_effect = get_account_side_effect

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        valid = await account_manager.validate_account_hierarchy(sub_id)

    assert valid


@pytest.mark.asyncio
async def test_validate_account_hierarchy_sub_without_parent_invalid(
    account_manager, mock_repository
):
    """Test sub-account without parent is invalid."""
    sub = Account(
        account_id=str(uuid4()),
        parent_account_id=None,  # Invalid for sub-account
        account_type=AccountType.SUB,
        balance_usdt=Decimal("5000"),
        tier=TradingTier.STRATEGIST,
    )

    mock_repository.get_account.return_value = sub

    with patch("genesis.core.account_manager.requires_tier", lambda x: lambda f: f):
        valid = await account_manager.validate_account_hierarchy(sub.account_id)

    assert not valid


@pytest.mark.asyncio
async def test_close_account_manager(account_manager):
    """Test closing account manager cleans up resources."""
    # Add some data
    account_manager._accounts_cache["test"] = MagicMock()
    account_manager._active_account_id = "test"

    # Close
    await account_manager.close()

    # Verify cleanup
    assert len(account_manager._accounts_cache) == 0
    assert account_manager._active_account_id is None
