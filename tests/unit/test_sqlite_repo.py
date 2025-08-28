"""
Unit tests for SQLite repository implementation.

This module tests the SQLite-specific implementation
of the repository pattern.
"""

import asyncio
import shutil
import tempfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.data.sqlite_repo import SQLiteRepository


@pytest_asyncio.fixture
async def temp_db_path():
    """Create a temporary database path."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield str(db_path)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def sqlite_repo(temp_db_path):
    """Create a SQLite repository for testing."""
    repo = SQLiteRepository(temp_db_path)
    await repo.initialize()
    yield repo
    await repo.shutdown()


@pytest.fixture
def sample_account():
    """Create a sample account."""
    return Account(
        account_id="acc_test_123",
        balance_usdt=Decimal("5000.00"),
        tier=TradingTier.SNIPER,
        locked_features=[],
        last_sync=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_position():
    """Create a sample position."""
    return Position(
        position_id="pos_test_123",
        account_id="acc_test_123",
        symbol="ETH/USDT",
        side=PositionSide.LONG,
        entry_price=Decimal("3000.00"),
        current_price=Decimal("3100.00"),
        quantity=Decimal("0.5"),
        dollar_value=Decimal("1500.00"),
        stop_loss=Decimal("2900.00"),
        pnl_dollars=Decimal("50.00"),
        pnl_percent=Decimal("3.33"),
        priority_score=5,
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_session():
    """Create a sample trading session."""
    return TradingSession(
        session_id="sess_test_123",
        account_id="acc_test_123",
        session_date=datetime.utcnow(),
        starting_balance=Decimal("5000.00"),
        current_balance=Decimal("5100.00"),
        realized_pnl=Decimal("100.00"),
        total_trades=20,
        winning_trades=12,
        losing_trades=8,
        max_drawdown=Decimal("50.00"),
        daily_loss_limit=Decimal("250.00"),
        is_active=True,
        created_at=datetime.utcnow(),
    )


class TestSQLiteRepository:
    """Test SQLite repository implementation."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db_path):
        """Test repository initialization."""
        repo = SQLiteRepository(temp_db_path)
        await repo.initialize()

        # Check that database file exists
        assert Path(temp_db_path).exists()

        # Check that tables were created
        async with aiosqlite.connect(temp_db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]

            assert "accounts" in table_names
            assert "positions" in table_names
            assert "trading_sessions" in table_names
            assert "orders" in table_names
            assert "events" in table_names
            assert "tilt_events" in table_names
            assert "database_info" in table_names

        await repo.shutdown()

    @pytest.mark.asyncio
    async def test_account_operations(self, sqlite_repo, sample_account):
        """Test account CRUD operations."""
        # Create account
        account_id = await sqlite_repo.create_account(sample_account)
        assert account_id == sample_account.account_id

        # Get account
        retrieved = await sqlite_repo.get_account(account_id)
        assert retrieved is not None
        assert retrieved.account_id == sample_account.account_id
        assert retrieved.balance_usdt == sample_account.balance_usdt
        assert retrieved.tier == sample_account.tier

        # Update account
        sample_account.balance_usdt = Decimal("6000.00")
        sample_account.tier = TradingTier.HUNTER
        await sqlite_repo.update_account(sample_account)

        updated = await sqlite_repo.get_account(account_id)
        assert updated.balance_usdt == Decimal("6000.00")
        assert updated.tier == TradingTier.HUNTER

        # Delete account
        await sqlite_repo.delete_account(account_id)
        deleted = await sqlite_repo.get_account(account_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_position_operations(
        self, sqlite_repo, sample_account, sample_position
    ):
        """Test position CRUD operations."""
        # Create account first
        await sqlite_repo.create_account(sample_account)

        # Create position
        position_id = await sqlite_repo.create_position(sample_position)
        assert position_id == sample_position.position_id

        # Get position
        retrieved = await sqlite_repo.get_position(position_id)
        assert retrieved is not None
        assert retrieved.symbol == sample_position.symbol
        assert retrieved.entry_price == sample_position.entry_price

        # Get positions by account
        positions = await sqlite_repo.get_positions_by_account(
            sample_account.account_id
        )
        assert len(positions) == 1
        assert positions[0].position_id == position_id

        # Update position
        sample_position.current_price = Decimal("3200.00")
        sample_position.pnl_dollars = Decimal("100.00")
        await sqlite_repo.update_position(sample_position)

        updated = await sqlite_repo.get_position(position_id)
        assert updated.current_price == Decimal("3200.00")
        assert updated.pnl_dollars == Decimal("100.00")

        # Close position
        await sqlite_repo.close_position(position_id, Decimal("150.00"))
        closed = await sqlite_repo.get_position(position_id)
        assert closed.pnl_dollars == Decimal("150.00")

    @pytest.mark.asyncio
    async def test_session_operations(
        self, sqlite_repo, sample_account, sample_session
    ):
        """Test trading session operations."""
        # Create account first
        await sqlite_repo.create_account(sample_account)

        # Create session
        session_id = await sqlite_repo.create_session(sample_session)
        assert session_id == sample_session.session_id

        # Get session
        retrieved = await sqlite_repo.get_session(session_id)
        assert retrieved is not None
        assert retrieved.starting_balance == sample_session.starting_balance

        # Get active session
        active = await sqlite_repo.get_active_session(sample_account.account_id)
        assert active is not None
        assert active.session_id == session_id
        assert active.is_active is True

        # Update session
        sample_session.current_balance = Decimal("5200.00")
        sample_session.total_trades = 25
        await sqlite_repo.update_session(sample_session)

        updated = await sqlite_repo.get_session(session_id)
        assert updated.current_balance == Decimal("5200.00")
        assert updated.total_trades == 25

        # End session
        await sqlite_repo.end_session(session_id)
        ended = await sqlite_repo.get_session(session_id)
        assert ended.is_active is False

    @pytest.mark.asyncio
    async def test_event_store(self, sqlite_repo):
        """Test event store functionality."""
        # Save events
        event1_id = await sqlite_repo.save_event(
            "OrderExecuted",
            "order_abc",
            {"symbol": "BTC/USDT", "quantity": "0.1", "price": "50000"},
        )

        event2_id = await sqlite_repo.save_event(
            "PositionOpened", "order_abc", {"position_id": "pos_xyz", "side": "LONG"}
        )

        event3_id = await sqlite_repo.save_event(
            "PositionClosed", "pos_xyz", {"pnl": "500.00", "reason": "take_profit"}
        )

        # Get events by aggregate
        order_events = await sqlite_repo.get_events("order_abc")
        assert len(order_events) == 2
        assert order_events[0]["sequence_number"] == 1
        assert order_events[1]["sequence_number"] == 2

        # Get events by type
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow() + timedelta(hours=1)

        position_events = await sqlite_repo.get_events_by_type(
            "PositionOpened", start_time, end_time
        )
        assert len(position_events) == 1
        assert position_events[0]["event_type"] == "PositionOpened"

    @pytest.mark.asyncio
    async def test_order_operations(self, sqlite_repo, sample_account):
        """Test order operations."""
        # Create account first
        await sqlite_repo.create_account(sample_account)

        # Create position first (orders have FK to positions)
        position = Position(
            position_id="pos_test_123",
            account_id=sample_account.account_id,
            symbol="BTC/USDT",
            side="LONG",  # Use uppercase for enum
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            dollar_value=Decimal("500.00"),  # Add required field
        )
        await sqlite_repo.create_position(position)

        order = {
            "order_id": "ord_test_123",
            "position_id": "pos_test_123",
            "account_id": sample_account.account_id,
            "client_order_id": "client_ord_123",
            "exchange_order_id": "exch_ord_123",
            "symbol": "BTC/USDT",
            "type": "MARKET",
            "side": "BUY",
            "quantity": Decimal("0.01"),
            "price": Decimal("50000"),
            "status": "PENDING",
        }

        # Save order
        order_id = await sqlite_repo.save_order(order)
        assert order_id == "ord_test_123"

        # Get order
        retrieved = await sqlite_repo.get_order(order_id)
        assert retrieved is not None
        assert retrieved["symbol"] == "BTC/USDT"
        assert retrieved["quantity"] == Decimal("0.01")

        # Update order status
        exec_time = datetime.utcnow()
        await sqlite_repo.update_order_status(order_id, "FILLED", exec_time)

        updated = await sqlite_repo.get_order(order_id)
        assert updated["status"] == "FILLED"
        assert updated["executed_at"] == exec_time

        # Get orders by position
        position_orders = await sqlite_repo.get_orders_by_position("pos_test_123")
        assert len(position_orders) == 1
        assert position_orders[0]["order_id"] == order_id

    @pytest.mark.asyncio
    async def test_position_recovery(self, sqlite_repo, sample_account):
        """Test position recovery functionality."""
        # Create account
        await sqlite_repo.create_account(sample_account)

        # Create multiple positions
        for i in range(3):
            pos = Position(
                position_id=f"pos_recovery_{i}",
                account_id=sample_account.account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("1"),
                dollar_value=Decimal("100"),
                pnl_dollars=Decimal("0"),
                pnl_percent=Decimal("0"),
                priority_score=i,
                created_at=datetime.utcnow(),
            )
            await sqlite_repo.create_position(pos)

        # Load open positions
        open_positions = await sqlite_repo.load_open_positions(
            sample_account.account_id
        )
        assert len(open_positions) == 3

        # Test reconciliation
        exchange_positions = [
            {"symbol": "TEST0/USDT", "quantity": "1"},
            {"symbol": "TEST1/USDT", "quantity": "1"},
            # TEST2/USDT is missing - should be marked as orphaned
        ]

        orphaned = await sqlite_repo.reconcile_positions(exchange_positions)
        assert len(orphaned) > 0

    @pytest.mark.asyncio
    async def test_backup_and_restore(self, sqlite_repo, sample_account, temp_db_path):
        """Test backup and restore functionality."""
        # Add some data
        await sqlite_repo.create_account(sample_account)

        # Create backup
        backup_dir = Path(temp_db_path).parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = await sqlite_repo.backup()
        assert backup_path.exists()

        # Verify backup contains data
        async with aiosqlite.connect(str(backup_path)) as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM accounts")
            count = await cursor.fetchone()
            assert count[0] == 1

    @pytest.mark.asyncio
    async def test_database_rotation(self, sqlite_repo):
        """Test database rotation functionality."""
        # Get initial size
        initial_size = await sqlite_repo.get_database_size()

        # Add data to increase size
        for i in range(10):
            event_data = {"data": "x" * 1000}  # Large event data
            await sqlite_repo.save_event("TestEvent", f"aggregate_{i}", event_data)

        # Rotate database
        await sqlite_repo.rotate_database()

        # Size should remain manageable after rotation
        final_size = await sqlite_repo.get_database_size()
        assert final_size > 0

    @pytest.mark.asyncio
    async def test_csv_export(self, sqlite_repo, sample_account, temp_db_path):
        """Test CSV export functionality."""
        # Create account
        await sqlite_repo.create_account(sample_account)

        # Create closed positions
        for i in range(3):
            pos = Position(
                position_id=f"pos_export_{i}",
                account_id=sample_account.account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                current_price=Decimal("110"),
                quantity=Decimal("1"),
                dollar_value=Decimal("100"),
                pnl_dollars=Decimal("10"),
                pnl_percent=Decimal("10"),
                priority_score=i,
                created_at=datetime.utcnow(),
            )
            await sqlite_repo.create_position(pos)
            await sqlite_repo.close_position(pos.position_id, Decimal("10"))

        # Export to CSV
        export_path = Path(temp_db_path).parent / "trades.csv"
        start_date = date.today() - timedelta(days=30)
        end_date = date.today()

        result = await sqlite_repo.export_trades_to_csv(
            sample_account.account_id, start_date, end_date, export_path
        )

        assert result == export_path
        assert export_path.exists()

        # Verify CSV content
        with open(export_path) as f:
            content = f.read()
            assert "symbol" in content
            assert "pnl_usd" in content
            assert "TOTAL" in content

    @pytest.mark.asyncio
    async def test_performance_metrics(self, sqlite_repo, sample_account):
        """Test performance metrics calculation."""
        # Create account
        await sqlite_repo.create_account(sample_account)

        # Create closed positions with varying P&L
        pnl_values = [100, -50, 200, -30, 150, -20, 80, -40, 120, 60]

        for i, pnl in enumerate(pnl_values):
            pos = Position(
                position_id=f"pos_perf_{i}",
                account_id=sample_account.account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                current_price=Decimal("100") + Decimal(str(pnl / 10)),
                quantity=Decimal("1"),
                dollar_value=Decimal("100"),
                pnl_dollars=Decimal(str(pnl)),
                pnl_percent=Decimal(str(pnl)),
                priority_score=i,
                created_at=datetime.utcnow() - timedelta(days=10 - i),
            )
            await sqlite_repo.create_position(pos)
            await sqlite_repo.close_position(pos.position_id, Decimal(str(pnl)))

        # Calculate metrics
        metrics = await sqlite_repo.calculate_performance_metrics(
            sample_account.account_id
        )

        assert metrics["total_trades"] == 10
        assert metrics["winning_trades"] == 6
        assert metrics["losing_trades"] == 4
        assert metrics["win_rate"] == 60.0
        assert Decimal(metrics["average_win"]) > 0
        assert Decimal(metrics["average_loss"]) > 0
        assert Decimal(metrics["average_r"]) > 0
        assert Decimal(metrics["profit_factor"]) > 0

    @pytest.mark.asyncio
    async def test_tilt_events(self, sqlite_repo, sample_account, sample_session):
        """Test tilt event operations."""
        # Create account and session
        await sqlite_repo.create_account(sample_account)
        await sqlite_repo.create_session(sample_session)

        # Save tilt event
        event_id = await sqlite_repo.save_tilt_event(
            sample_session.session_id,
            "high_cancel_rate",
            "high",
            {"cancel_rate": 0.85, "threshold": 0.5, "duration_seconds": 300},
            "position_size_reduced",
        )

        assert event_id is not None

        # Get tilt events
        events = await sqlite_repo.get_tilt_events(sample_session.session_id)
        assert len(events) == 1
        assert events[0]["event_type"] == "high_cancel_rate"
        assert events[0]["severity"] == "high"
        assert events[0]["indicator_values"]["cancel_rate"] == 0.85

    @pytest.mark.asyncio
    async def test_transaction_handling(self, sqlite_repo, sample_account):
        """Test transaction commit and rollback."""
        # Create account first (not in transaction)
        await sqlite_repo.create_account(sample_account)

        # Verify account was created with correct balance
        account = await sqlite_repo.get_account(sample_account.account_id)
        assert account is not None
        assert account.balance_usdt == Decimal("5000.00")

        # Test explicit transaction with rollback
        try:
            await sqlite_repo.begin_transaction()
            # Create a new account object with updated balance for the update
            updated_account = Account(
                account_id=sample_account.account_id,
                balance_usdt=Decimal("10000"),
                tier=sample_account.tier,
                locked_features=sample_account.locked_features,
                last_sync=sample_account.last_sync,
                created_at=sample_account.created_at,
            )
            await sqlite_repo.update_account(updated_account)
            # Explicitly rollback
            await sqlite_repo.rollback_transaction()
        except Exception:
            # Ensure rollback happens even on error
            await sqlite_repo.rollback_transaction()
            raise

        # Verify rollback worked (balance should still be original)
        account = await sqlite_repo.get_account(sample_account.account_id)
        assert account.balance_usdt == Decimal("5000.00")

    @pytest.mark.asyncio
    async def test_database_info(self, sqlite_repo):
        """Test database info operations."""
        # Set various info
        await sqlite_repo.set_database_info("version", "1.6.0")
        await sqlite_repo.set_database_info("last_backup", "2024-01-01T00:00:00")
        await sqlite_repo.set_database_info("schema_version", "3")

        # Get info
        version = await sqlite_repo.get_database_info("version")
        assert version == "1.6.0"

        last_backup = await sqlite_repo.get_database_info("last_backup")
        assert last_backup == "2024-01-01T00:00:00"

        schema = await sqlite_repo.get_database_info("schema_version")
        assert schema == "3"

        # Get non-existent info
        missing = await sqlite_repo.get_database_info("non_existent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sqlite_repo, sample_account):
        """Test concurrent database operations."""
        # Create account
        await sqlite_repo.create_account(sample_account)

        # Perform concurrent position creates
        tasks = []
        for i in range(10):
            pos = Position(
                position_id=f"pos_concurrent_{i}",
                account_id=sample_account.account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("1"),
                dollar_value=Decimal("100"),
                pnl_dollars=Decimal("0"),
                pnl_percent=Decimal("0"),
                priority_score=i,
                created_at=datetime.utcnow(),
            )
            tasks.append(sqlite_repo.create_position(pos))

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        assert len(results) == 10

        # Verify all were created
        positions = await sqlite_repo.get_positions_by_account(
            sample_account.account_id
        )
        assert len(positions) == 10

    @pytest.mark.asyncio
    async def test_decimal_precision(self, sqlite_repo, sample_account):
        """Test that decimal values maintain precision."""
        # Create account with precise decimal
        sample_account.balance_usdt = Decimal("12345.67890123")
        await sqlite_repo.create_account(sample_account)

        # Retrieve and verify precision
        account = await sqlite_repo.get_account(sample_account.account_id)
        assert account.balance_usdt == Decimal("12345.67890123")

        # Create position with precise decimals
        pos = Position(
            position_id="pos_precision",
            account_id=sample_account.account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50123.456789"),
            current_price=Decimal("50234.567890"),
            quantity=Decimal("0.123456789"),
            dollar_value=Decimal("6190.44713997"),
            pnl_dollars=Decimal("13.71826858"),
            pnl_percent=Decimal("0.22153846"),
            priority_score=1,
            created_at=datetime.utcnow(),
        )
        await sqlite_repo.create_position(pos)

        # Retrieve and verify precision
        retrieved = await sqlite_repo.get_position(pos.position_id)
        assert retrieved.entry_price == Decimal("50123.456789")
        assert retrieved.quantity == Decimal("0.123456789")
        assert retrieved.pnl_dollars == Decimal("13.71826858")
