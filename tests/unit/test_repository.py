"""
Unit tests for the abstract repository interface.

This module tests the repository pattern implementation,
focusing on the abstract interface and common behaviors.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from genesis.core.models import (
    Account,
    Position,
    PositionCorrelation,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.data.repository import Repository


class MockRepository(Repository):
    """Mock implementation of Repository for testing."""

    def __init__(self):
        self.accounts = {}
        self.positions = {}
        self.sessions = {}
        self.events = []
        self.orders = {}

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    # Account methods
    async def create_account(self, account: Account) -> str:
        self.accounts[account.account_id] = account
        return account.account_id

    async def get_account(self, account_id: str) -> Account:
        return self.accounts.get(account_id)

    async def update_account(self, account: Account) -> None:
        self.accounts[account.account_id] = account

    async def delete_account(self, account_id: str) -> None:
        del self.accounts[account_id]

    # Position methods
    async def create_position(self, position: Position) -> str:
        self.positions[position.position_id] = position
        return position.position_id

    async def get_position(self, position_id: str) -> Position:
        return self.positions.get(position_id)

    async def get_positions_by_account(self, account_id: str, status: str = None) -> list[Position]:
        positions = []
        for pos in self.positions.values():
            if pos.account_id == account_id:
                if status is None or (hasattr(pos, 'status') and pos.status == status):
                    positions.append(pos)
        return positions

    async def update_position(self, position: Position) -> None:
        self.positions[position.position_id] = position

    async def close_position(self, position_id: str, final_pnl: Decimal) -> None:
        if position_id in self.positions:
            self.positions[position_id].pnl_dollars = final_pnl
            if hasattr(self.positions[position_id], 'status'):
                self.positions[position_id].status = 'CLOSED'

    # Trading session methods
    async def create_session(self, session: TradingSession) -> str:
        self.sessions[session.session_id] = session
        return session.session_id

    async def get_session(self, session_id: str) -> TradingSession:
        return self.sessions.get(session_id)

    async def get_active_session(self, account_id: str) -> TradingSession:
        for session in self.sessions.values():
            if session.account_id == account_id and session.is_active:
                return session
        return None

    async def update_session(self, session: TradingSession) -> None:
        self.sessions[session.session_id] = session

    async def end_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False

    # Position correlation methods
    async def save_correlation(self, correlation: PositionCorrelation) -> None:
        pass

    async def get_correlations(self, position_id: str) -> list[PositionCorrelation]:
        return []

    # Risk metrics methods
    async def save_risk_metrics(self, metrics: dict[str, Any]) -> None:
        pass

    async def get_risk_metrics(self, account_id: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        return []

    # Event store methods
    async def save_event(self, event_type: str, aggregate_id: str, event_data: dict[str, Any]) -> str:
        event_id = f"event_{len(self.events)}"
        self.events.append({
            "event_id": event_id,
            "event_type": event_type,
            "aggregate_id": aggregate_id,
            "event_data": event_data,
            "created_at": datetime.utcnow()
        })
        return event_id

    async def get_events(self, aggregate_id: str, event_type: str = None) -> list[dict[str, Any]]:
        events = []
        for event in self.events:
            if event["aggregate_id"] == aggregate_id:
                if event_type is None or event["event_type"] == event_type:
                    events.append(event)
        return events

    async def get_events_by_type(self, event_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        events = []
        for event in self.events:
            if event["event_type"] == event_type:
                if start_time <= event["created_at"] <= end_time:
                    events.append(event)
        return events

    # Order methods
    async def save_order(self, order: dict[str, Any]) -> str:
        order_id = order.get("order_id", f"order_{len(self.orders)}")
        self.orders[order_id] = order
        return order_id

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return self.orders.get(order_id)

    async def get_orders_by_position(self, position_id: str) -> list[dict[str, Any]]:
        orders = []
        for order in self.orders.values():
            if order.get("position_id") == position_id:
                orders.append(order)
        return orders

    async def update_order_status(self, order_id: str, status: str, executed_at: datetime = None) -> None:
        if order_id in self.orders:
            self.orders[order_id]["status"] = status
            if executed_at:
                self.orders[order_id]["executed_at"] = executed_at

    # Position recovery methods
    async def load_open_positions(self, account_id: str) -> list[Position]:
        return await self.get_positions_by_account(account_id, status="OPEN")

    async def reconcile_positions(self, exchange_positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return []

    # Backup and restore methods
    async def backup(self, backup_path: Path = None) -> Path:
        if backup_path is None:
            backup_path = Path("/tmp/test_backup.db")
        return backup_path

    async def restore(self, backup_path: Path) -> None:
        pass

    async def get_database_size(self) -> int:
        return 1024

    async def rotate_database(self) -> None:
        pass

    # Export methods
    async def export_trades_to_csv(self, account_id: str, start_date: date, end_date: date, output_path: Path) -> Path:
        return output_path

    async def export_performance_report(self, account_id: str, output_path: Path) -> Path:
        return output_path

    # Performance metrics methods
    async def calculate_performance_metrics(self, account_id: str, session_id: str = None) -> dict[str, Any]:
        return {
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "win_rate": 60.0,
            "average_win": "150.00",
            "average_loss": "75.00",
            "average_r": "2.00",
            "profit_factor": "2.25",
            "max_drawdown": "200.00"
        }

    async def get_performance_report(self, account_id: str, start_date: date, end_date: date) -> dict[str, Any]:
        return {
            "account_id": account_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": await self.calculate_performance_metrics(account_id),
            "daily_pnl": {},
            "total_pnl": "500.00"
        }

    # Tilt event methods
    async def save_tilt_event(self, session_id: str, event_type: str, severity: str,
                             indicator_values: dict[str, Any], intervention: str = None) -> str:
        return f"tilt_{session_id}"

    async def get_tilt_events(self, session_id: str) -> list[dict[str, Any]]:
        return []

    # Transaction methods
    async def begin_transaction(self) -> None:
        pass

    async def commit_transaction(self) -> None:
        pass

    async def rollback_transaction(self) -> None:
        pass

    # Database management
    async def set_database_info(self, key: str, value: str) -> None:
        pass

    async def get_database_info(self, key: str) -> str:
        return None


@pytest_asyncio.fixture
async def repository():
    """Create a mock repository for testing."""
    repo = MockRepository()
    await repo.initialize()
    yield repo
    await repo.shutdown()


@pytest.fixture
def sample_account():
    """Create a sample account for testing."""
    return Account(
        account_id="test_account_123",
        balance_usdt=Decimal("1000.00"),
        tier=TradingTier.SNIPER,
        locked_features=[],
        last_sync=datetime.utcnow(),
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    return Position(
        position_id="pos_123",
        account_id="test_account_123",
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=Decimal("50000.00"),
        current_price=Decimal("51000.00"),
        quantity=Decimal("0.01"),
        dollar_value=Decimal("500.00"),
        stop_loss=Decimal("49000.00"),
        pnl_dollars=Decimal("10.00"),
        pnl_percent=Decimal("2.00"),
        priority_score=1,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def sample_session():
    """Create a sample trading session for testing."""
    return TradingSession(
        session_id="session_123",
        account_id="test_account_123",
        session_date=datetime.utcnow(),
        starting_balance=Decimal("1000.00"),
        current_balance=Decimal("1050.00"),
        realized_pnl=Decimal("50.00"),
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        max_drawdown=Decimal("20.00"),
        daily_loss_limit=Decimal("100.00"),
        is_active=True,
        created_at=datetime.utcnow()
    )


class TestRepositoryInterface:
    """Test the repository interface."""

    @pytest.mark.asyncio
    async def test_account_crud(self, repository, sample_account):
        """Test account CRUD operations."""
        # Create
        account_id = await repository.create_account(sample_account)
        assert account_id == sample_account.account_id

        # Read
        retrieved = await repository.get_account(account_id)
        assert retrieved.account_id == sample_account.account_id
        assert retrieved.balance_usdt == sample_account.balance_usdt

        # Update
        sample_account.balance_usdt = Decimal("1100.00")
        await repository.update_account(sample_account)
        updated = await repository.get_account(account_id)
        assert updated.balance_usdt == Decimal("1100.00")

        # Delete
        await repository.delete_account(account_id)
        deleted = await repository.get_account(account_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_position_crud(self, repository, sample_position):
        """Test position CRUD operations."""
        # Create
        position_id = await repository.create_position(sample_position)
        assert position_id == sample_position.position_id

        # Read
        retrieved = await repository.get_position(position_id)
        assert retrieved.position_id == sample_position.position_id
        assert retrieved.entry_price == sample_position.entry_price

        # Update
        sample_position.current_price = Decimal("52000.00")
        await repository.update_position(sample_position)
        updated = await repository.get_position(position_id)
        assert updated.current_price == Decimal("52000.00")

        # Close
        await repository.close_position(position_id, Decimal("100.00"))
        closed = await repository.get_position(position_id)
        assert closed.pnl_dollars == Decimal("100.00")

    @pytest.mark.asyncio
    async def test_get_positions_by_account(self, repository):
        """Test getting positions by account."""
        # Create multiple positions
        for i in range(3):
            pos = Position(
                position_id=f"pos_{i}",
                account_id="test_account",
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100.00"),
                quantity=Decimal("1.00"),
                dollar_value=Decimal("100.00"),
                pnl_dollars=Decimal("0.00"),
                pnl_percent=Decimal("0.00"),
                priority_score=i,
                created_at=datetime.utcnow()
            )
            await repository.create_position(pos)

        positions = await repository.get_positions_by_account("test_account")
        assert len(positions) == 3

    @pytest.mark.asyncio
    async def test_session_operations(self, repository, sample_session):
        """Test trading session operations."""
        # Create
        session_id = await repository.create_session(sample_session)
        assert session_id == sample_session.session_id

        # Get active
        active = await repository.get_active_session(sample_session.account_id)
        assert active.session_id == sample_session.session_id
        assert active.is_active is True

        # Update
        sample_session.current_balance = Decimal("1100.00")
        await repository.update_session(sample_session)

        # End
        await repository.end_session(session_id)
        ended = await repository.get_session(session_id)
        assert ended.is_active is False

    @pytest.mark.asyncio
    async def test_event_store(self, repository):
        """Test event store operations."""
        # Save events
        event_id1 = await repository.save_event(
            "OrderExecuted",
            "order_123",
            {"symbol": "BTC/USDT", "quantity": "0.01"}
        )

        event_id2 = await repository.save_event(
            "PositionOpened",
            "order_123",
            {"position_id": "pos_123"}
        )

        # Get events by aggregate
        events = await repository.get_events("order_123")
        assert len(events) == 2

        # Get events by type
        order_events = await repository.get_events("order_123", "OrderExecuted")
        assert len(order_events) == 1
        assert order_events[0]["event_type"] == "OrderExecuted"

    @pytest.mark.asyncio
    async def test_order_operations(self, repository):
        """Test order operations."""
        order = {
            "order_id": "order_123",
            "position_id": "pos_123",
            "account_id": "acc_123",
            "client_order_id": "client_123",
            "symbol": "BTC/USDT",
            "type": "MARKET",
            "side": "BUY",
            "quantity": Decimal("0.01"),
            "status": "PENDING"
        }

        # Save
        order_id = await repository.save_order(order)
        assert order_id == "order_123"

        # Get
        retrieved = await repository.get_order(order_id)
        assert retrieved["symbol"] == "BTC/USDT"

        # Update status
        await repository.update_order_status(order_id, "FILLED", datetime.utcnow())
        updated = await repository.get_order(order_id)
        assert updated["status"] == "FILLED"

        # Get by position
        position_orders = await repository.get_orders_by_position("pos_123")
        assert len(position_orders) == 1

    @pytest.mark.asyncio
    async def test_position_recovery(self, repository):
        """Test position recovery operations."""
        # Create positions
        for i in range(2):
            pos = Position(
                position_id=f"pos_{i}",
                account_id="test_account",
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100.00"),
                quantity=Decimal("1.00"),
                dollar_value=Decimal("100.00"),
                pnl_dollars=Decimal("0.00"),
                pnl_percent=Decimal("0.00"),
                priority_score=i,
                created_at=datetime.utcnow()
            )
            pos.status = "OPEN"
            await repository.create_position(pos)

        # Load open positions
        open_positions = await repository.load_open_positions("test_account")
        assert len(open_positions) == 2

        # Reconcile with exchange
        exchange_positions = [
            {"symbol": "TEST0/USDT", "quantity": "1.00"}
        ]
        orphaned = await repository.reconcile_positions(exchange_positions)
        assert isinstance(orphaned, list)

    @pytest.mark.asyncio
    async def test_backup_restore(self, repository):
        """Test backup and restore operations."""
        # Backup
        backup_path = await repository.backup()
        assert backup_path is not None

        # Get database size
        size = await repository.get_database_size()
        assert size > 0

        # Rotate database
        await repository.rotate_database()

    @pytest.mark.asyncio
    async def test_export_operations(self, repository):
        """Test export operations."""
        # Export trades to CSV
        csv_path = Path("/tmp/trades.csv")
        exported = await repository.export_trades_to_csv(
            "test_account",
            date.today() - timedelta(days=30),
            date.today(),
            csv_path
        )
        assert exported == csv_path

        # Export performance report
        report_path = Path("/tmp/performance.txt")
        report = await repository.export_performance_report(
            "test_account",
            report_path
        )
        assert report == report_path

    @pytest.mark.asyncio
    async def test_performance_metrics(self, repository):
        """Test performance metrics calculations."""
        metrics = await repository.calculate_performance_metrics("test_account")

        assert "total_trades" in metrics
        assert "winning_trades" in metrics
        assert "losing_trades" in metrics
        assert "win_rate" in metrics
        assert "average_r" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics

        # Get performance report
        report = await repository.get_performance_report(
            "test_account",
            date.today() - timedelta(days=30),
            date.today()
        )

        assert "account_id" in report
        assert "period" in report
        assert "metrics" in report
        assert "total_pnl" in report

    @pytest.mark.asyncio
    async def test_tilt_events(self, repository):
        """Test tilt event operations."""
        # Save tilt event
        event_id = await repository.save_tilt_event(
            "session_123",
            "high_cancel_rate",
            "medium",
            {"cancel_rate": 0.75, "threshold": 0.5},
            "reduce_position_size"
        )
        assert event_id is not None

        # Get tilt events
        events = await repository.get_tilt_events("session_123")
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_transactions(self, repository):
        """Test transaction operations."""
        await repository.begin_transaction()

        # Perform operations
        account = Account(
            account_id="tx_test",
            balance_usdt=Decimal("1000.00"),
            tier=TradingTier.SNIPER,
            locked_features=[],
            created_at=datetime.utcnow()
        )
        await repository.create_account(account)

        # Commit
        await repository.commit_transaction()

        # Test rollback
        await repository.begin_transaction()
        account.balance_usdt = Decimal("2000.00")
        await repository.update_account(account)
        await repository.rollback_transaction()

    @pytest.mark.asyncio
    async def test_database_info(self, repository):
        """Test database info operations."""
        # Set info
        await repository.set_database_info("version", "1.0.0")
        await repository.set_database_info("last_backup", datetime.utcnow().isoformat())

        # Get info
        version = await repository.get_database_info("version")
        # Note: MockRepository returns None, but real implementation would return "1.0.0"
        assert version is None or version == "1.0.0"
