"""
SQLite repository implementation for Project GENESIS.

This module implements the repository pattern using SQLite
for the MVP phase of the project.
"""

import csv
import json
import shutil
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
import structlog

from genesis.core.models import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionCorrelation,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.data.repository import Repository

logger = structlog.get_logger(__name__)


def decimal_default(obj):
    """JSON serializer for Decimal objects."""
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError


class SQLiteRepository(Repository):
    """SQLite implementation of the repository pattern."""

    def __init__(self, db_path: str = ".genesis/data/genesis.db"):
        """
        Initialize SQLite repository.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection: aiosqlite.Connection | None = None

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("SQLite repository initialized", db_path=db_path)

    async def initialize(self) -> None:
        """Initialize the repository and create tables."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            self.connection.row_factory = aiosqlite.Row

            # Enable foreign key constraints
            await self.connection.execute("PRAGMA foreign_keys = ON")

            # Create tables
            await self._create_tables()
            await self.connection.commit()

            logger.info("SQLite database initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize SQLite database", error=str(e))
            raise

    async def shutdown(self) -> None:
        """Close database connection."""
        if self.connection:
            await self.connection.close()
            logger.info("SQLite connection closed")

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""

        # Enable WAL mode for better concurrency
        await self.connection.execute("PRAGMA journal_mode=WAL")
        await self.connection.execute("PRAGMA synchronous=NORMAL")

        # Events table (immutable audit trail)
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                event_data TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(aggregate_id, sequence_number)
            )
        """)

        # Accounts table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                balance_usdt TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'SNIPER',
                locked_features TEXT DEFAULT '[]',
                last_sync TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """)

        # Positions table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
                entry_price TEXT NOT NULL,
                current_price TEXT,
                quantity TEXT NOT NULL,
                dollar_value TEXT NOT NULL,
                stop_loss TEXT,
                take_profit TEXT,
                pnl_dollars TEXT NOT NULL DEFAULT '0',
                pnl_percent TEXT NOT NULL DEFAULT '0',
                priority_score INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'OPEN',
                close_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                closed_at TEXT,
                CHECK (close_reason IN ('stop_loss', 'take_profit', 'manual', 'tilt_intervention', 'emergency') OR close_reason IS NULL)
            )
        """)

        # Trading sessions table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS trading_sessions (
                session_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                session_date TEXT NOT NULL,
                starting_balance TEXT NOT NULL,
                current_balance TEXT NOT NULL,
                ending_balance TEXT,
                realized_pnl TEXT NOT NULL DEFAULT '0',
                total_trades INTEGER NOT NULL DEFAULT 0 CHECK (total_trades >= 0),
                winning_trades INTEGER NOT NULL DEFAULT 0 CHECK (winning_trades >= 0),
                losing_trades INTEGER NOT NULL DEFAULT 0 CHECK (losing_trades >= 0),
                win_rate REAL,
                average_r TEXT,
                max_drawdown TEXT NOT NULL DEFAULT '0',
                daily_loss_limit TEXT NOT NULL,
                tilt_events INTEGER DEFAULT 0,
                notes TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                ended_at TEXT
            )
        """)

        # Position correlations table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS position_correlations (
                position_a_id TEXT NOT NULL REFERENCES positions(position_id),
                position_b_id TEXT NOT NULL REFERENCES positions(position_id),
                correlation_coefficient TEXT NOT NULL,
                alert_triggered INTEGER NOT NULL DEFAULT 0,
                calculated_at TEXT NOT NULL,
                PRIMARY KEY (position_a_id, position_b_id),
                CHECK (position_a_id < position_b_id)
            )
        """)

        # Orders table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                position_id TEXT REFERENCES positions(position_id),
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                client_order_id TEXT NOT NULL UNIQUE,
                exchange_order_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('MARKET', 'LIMIT', 'STOP_LIMIT')),
                side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
                price TEXT,
                quantity TEXT NOT NULL,
                filled_quantity TEXT NOT NULL DEFAULT '0',
                executed_price TEXT,
                status TEXT NOT NULL CHECK (status IN ('PENDING', 'PARTIAL', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED', 'FAILED')),
                slice_number INTEGER,
                total_slices INTEGER,
                latency_ms INTEGER,
                slippage_percent TEXT,
                created_at TEXT NOT NULL,
                executed_at TEXT
            )
        """)

        # Risk metrics table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                metric_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                timestamp TEXT NOT NULL,
                total_exposure TEXT NOT NULL,
                position_count INTEGER NOT NULL CHECK (position_count >= 0),
                total_pnl_dollars TEXT NOT NULL,
                total_pnl_percent TEXT NOT NULL,
                max_position_size TEXT NOT NULL,
                daily_pnl TEXT NOT NULL,
                risk_score TEXT
            )
        """)

        # Tilt events table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS tilt_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES trading_sessions(session_id),
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                indicator_values TEXT NOT NULL,
                intervention_taken TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Database info table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS database_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Create indexes
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id, sequence_number)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, created_at)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_position_account ON positions(account_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_position_status ON positions(status)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_account ON trading_sessions(account_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_active ON trading_sessions(is_active)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_position ON orders(position_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_symbol ON orders(symbol)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_status ON orders(status)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_created ON orders(created_at)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_account ON risk_metrics(account_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_timestamp ON risk_metrics(timestamp)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tilt_session ON tilt_events(session_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tilt_created ON tilt_events(created_at)"
        )

    # Event store methods
    async def save_event(self, event_type: str, aggregate_id: str, event_data: dict[str, Any]) -> str:
        """Save an event to the event store."""
        event_id = str(uuid4())

        # Get next sequence number for this aggregate
        cursor = await self.connection.execute(
            "SELECT MAX(sequence_number) as max_seq FROM events WHERE aggregate_id = ?",
            (aggregate_id,)
        )
        row = await cursor.fetchone()
        sequence_number = (row["max_seq"] or 0) + 1 if row else 1

        await self.connection.execute(
            """
            INSERT INTO events (event_id, event_type, aggregate_id, event_data, 
                              sequence_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                aggregate_id,
                json.dumps(event_data, default=decimal_default),
                sequence_number,
                datetime.utcnow().isoformat()
            )
        )
        await self.connection.commit()

        logger.debug("Event saved", event_id=event_id, event_type=event_type)
        return event_id

    async def get_events(self, aggregate_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get events for an aggregate."""
        if event_type:
            cursor = await self.connection.execute(
                """
                SELECT * FROM events 
                WHERE aggregate_id = ? AND event_type = ?
                ORDER BY sequence_number
                """,
                (aggregate_id, event_type)
            )
        else:
            cursor = await self.connection.execute(
                """
                SELECT * FROM events 
                WHERE aggregate_id = ?
                ORDER BY sequence_number
                """,
                (aggregate_id,)
            )

        rows = await cursor.fetchall()
        events = []

        for row in rows:
            events.append({
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "aggregate_id": row["aggregate_id"],
                "event_data": json.loads(row["event_data"]),
                "sequence_number": row["sequence_number"],
                "created_at": datetime.fromisoformat(row["created_at"])
            })

        return events

    async def get_events_by_type(self, event_type: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get events by type and time range."""
        cursor = await self.connection.execute(
            """
            SELECT * FROM events 
            WHERE event_type = ? AND created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
            """,
            (event_type, start_time.isoformat(), end_time.isoformat())
        )

        rows = await cursor.fetchall()
        events = []

        for row in rows:
            events.append({
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "aggregate_id": row["aggregate_id"],
                "event_data": json.loads(row["event_data"]),
                "sequence_number": row["sequence_number"],
                "created_at": datetime.fromisoformat(row["created_at"])
            })

        return events

    # Account methods
    async def create_account(self, account: Account) -> str:
        """Create a new account."""

        await self.connection.execute(
            """
            INSERT INTO accounts (account_id, balance_usdt, tier, locked_features, 
                                 last_sync, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account.account_id,
                str(account.balance_usdt),
                account.tier.value,
                json.dumps(account.locked_features, default=decimal_default),
                account.last_sync.isoformat() if account.last_sync else None,
                account.created_at.isoformat(),
                datetime.utcnow().isoformat()
            )
        )
        await self.connection.commit()

        logger.info("Account created", account_id=account.account_id)
        return account.account_id

    async def get_account(self, account_id: str) -> Account | None:
        """Get account by ID."""
        import json

        cursor = await self.connection.execute(
            "SELECT * FROM accounts WHERE account_id = ?",
            (account_id,)
        )
        row = await cursor.fetchone()

        if row:
            return Account(
                account_id=row["account_id"],
                balance_usdt=Decimal(row["balance_usdt"]),
                tier=TradingTier[row["tier"]],
                locked_features=json.loads(row["locked_features"]),
                last_sync=datetime.fromisoformat(row["last_sync"]) if row["last_sync"] else None,
                created_at=datetime.fromisoformat(row["created_at"])
            )
        return None

    async def update_account(self, account: Account) -> None:
        """Update existing account."""
        import json

        await self.connection.execute(
            """
            UPDATE accounts 
            SET balance_usdt = ?, tier = ?, locked_features = ?, 
                last_sync = ?, updated_at = ?
            WHERE account_id = ?
            """,
            (
                str(account.balance_usdt),
                account.tier.value,
                json.dumps(account.locked_features, default=decimal_default),
                account.last_sync.isoformat() if account.last_sync else None,
                datetime.utcnow().isoformat(),
                account.account_id
            )
        )
        await self.connection.commit()

        logger.debug("Account updated", account_id=account.account_id)

    async def delete_account(self, account_id: str) -> None:
        """Delete account."""
        await self.connection.execute(
            "DELETE FROM accounts WHERE account_id = ?",
            (account_id,)
        )
        await self.connection.commit()

        logger.info("Account deleted", account_id=account_id)

    # Order methods (moved here for better organization)
    async def save_order(self, order: dict[str, Any]) -> str:
        """Save an order."""
        order_id = order.get("order_id", str(uuid4()))

        await self.connection.execute(
            """
            INSERT INTO orders (order_id, position_id, account_id, client_order_id, exchange_order_id,
                              symbol, type, side, price, quantity, filled_quantity, executed_price,
                              status, slice_number, total_slices, latency_ms,
                              slippage_percent, created_at, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                order.get("position_id"),
                order["account_id"],
                order["client_order_id"],
                order.get("exchange_order_id"),
                order["symbol"],
                order["type"],
                order["side"],
                str(order["price"]) if order.get("price") else None,
                str(order["quantity"]),
                str(order.get("filled_quantity", 0)),
                str(order["executed_price"]) if order.get("executed_price") else None,
                order.get("status", "PENDING"),
                order.get("slice_number"),
                order.get("total_slices"),
                order.get("latency_ms"),
                str(order["slippage_percent"]) if order.get("slippage_percent") else None,
                order.get("created_at", datetime.utcnow()).isoformat(),
                order["executed_at"].isoformat() if order.get("executed_at") else None
            )
        )
        await self.connection.commit()

        # Log event
        await self.save_event(
            "OrderCreated",
            order_id,
            order
        )

        logger.info("Order saved", order_id=order_id)
        return order_id

    async def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Get order by ID."""
        cursor = await self.connection.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "order_id": row["order_id"],
            "position_id": row["position_id"],
            "account_id": row["account_id"],
            "client_order_id": row["client_order_id"],
            "exchange_order_id": row["exchange_order_id"],
            "symbol": row["symbol"],
            "type": row["type"],
            "side": row["side"],
            "price": Decimal(row["price"]) if row["price"] else None,
            "quantity": Decimal(row["quantity"]),
            "filled_quantity": Decimal(row["filled_quantity"]),
            "executed_price": Decimal(row["executed_price"]) if row["executed_price"] else None,
            "status": row["status"],
            "slice_number": row["slice_number"],
            "total_slices": row["total_slices"],
            "latency_ms": row["latency_ms"],
            "slippage_percent": Decimal(row["slippage_percent"]) if row["slippage_percent"] else None,
            "created_at": datetime.fromisoformat(row["created_at"]),
            "executed_at": datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None
        }

    async def get_orders_by_position(self, position_id: str) -> list[dict[str, Any]]:
        """Get all orders for a position."""
        cursor = await self.connection.execute(
            "SELECT * FROM orders WHERE position_id = ? ORDER BY created_at DESC",
            (position_id,)
        )
        rows = await cursor.fetchall()

        orders = []
        for row in rows:
            orders.append({
                "order_id": row["order_id"],
                "position_id": row["position_id"],
                "account_id": row["account_id"],
                "client_order_id": row["client_order_id"],
                "exchange_order_id": row["exchange_order_id"],
                "symbol": row["symbol"],
                "type": row["type"],
                "side": row["side"],
                "price": Decimal(row["price"]) if row["price"] else None,
                "quantity": Decimal(row["quantity"]),
                "filled_quantity": Decimal(row["filled_quantity"]),
                "executed_price": Decimal(row["executed_price"]) if row["executed_price"] else None,
                "status": row["status"],
                "slice_number": row["slice_number"],
                "total_slices": row["total_slices"],
                "latency_ms": row["latency_ms"],
                "slippage_percent": Decimal(row["slippage_percent"]) if row["slippage_percent"] else None,
                "created_at": datetime.fromisoformat(row["created_at"]),
                "executed_at": datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None
            })

        return orders

    async def update_order_status(self, order_id: str, status: str, executed_at: datetime | None = None) -> None:
        """Update order status."""
        await self.connection.execute(
            """
            UPDATE orders 
            SET status = ?, executed_at = ?
            WHERE order_id = ?
            """,
            (
                status,
                executed_at.isoformat() if executed_at else None,
                order_id
            )
        )
        await self.connection.commit()

        # Log event
        await self.save_event(
            "OrderStatusUpdated",
            order_id,
            {"status": status, "executed_at": executed_at.isoformat() if executed_at else None}
        )

        logger.debug("Order status updated", order_id=order_id, status=status)

    # Position methods
    async def create_position(self, position: Position) -> str:
        """Create a new position."""
        await self.connection.execute(
            """
            INSERT INTO positions (position_id, account_id, symbol, side, entry_price,
                                  current_price, quantity, dollar_value, stop_loss,
                                  pnl_dollars, pnl_percent, priority_score, status,
                                  created_at, updated_at, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.position_id,
                position.account_id,
                position.symbol,
                position.side.value,
                str(position.entry_price),
                str(position.current_price) if position.current_price else None,
                str(position.quantity),
                str(position.dollar_value),
                str(position.stop_loss) if position.stop_loss else None,
                str(position.pnl_dollars),
                str(position.pnl_percent),
                position.priority_score,
                "OPEN",
                position.created_at.isoformat(),
                position.updated_at.isoformat() if position.updated_at else None,
                None
            )
        )
        await self.connection.commit()

        logger.info("Position created", position_id=position.position_id)
        return position.position_id

    async def get_position(self, position_id: str) -> Position | None:
        """Get position by ID."""
        cursor = await self.connection.execute(
            "SELECT * FROM positions WHERE position_id = ?",
            (position_id,)
        )
        row = await cursor.fetchone()

        if row:
            return Position(
                position_id=row["position_id"],
                account_id=row["account_id"],
                symbol=row["symbol"],
                side=PositionSide[row["side"]],
                entry_price=Decimal(row["entry_price"]),
                current_price=Decimal(row["current_price"]) if row["current_price"] else None,
                quantity=Decimal(row["quantity"]),
                dollar_value=Decimal(row["dollar_value"]),
                stop_loss=Decimal(row["stop_loss"]) if row["stop_loss"] else None,
                pnl_dollars=Decimal(row["pnl_dollars"]),
                pnl_percent=Decimal(row["pnl_percent"]),
                priority_score=row["priority_score"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            )
        return None

    async def get_positions_by_account(self, account_id: str, status: str | None = None) -> list[Position]:
        """Get all positions for an account."""
        if status:
            cursor = await self.connection.execute(
                "SELECT * FROM positions WHERE account_id = ? AND status = ?",
                (account_id, status)
            )
        else:
            cursor = await self.connection.execute(
                "SELECT * FROM positions WHERE account_id = ?",
                (account_id,)
            )

        rows = await cursor.fetchall()
        positions = []

        for row in rows:
            positions.append(Position(
                position_id=row["position_id"],
                account_id=row["account_id"],
                symbol=row["symbol"],
                side=PositionSide[row["side"]],
                entry_price=Decimal(row["entry_price"]),
                current_price=Decimal(row["current_price"]) if row["current_price"] else None,
                quantity=Decimal(row["quantity"]),
                dollar_value=Decimal(row["dollar_value"]),
                stop_loss=Decimal(row["stop_loss"]) if row["stop_loss"] else None,
                pnl_dollars=Decimal(row["pnl_dollars"]),
                pnl_percent=Decimal(row["pnl_percent"]),
                priority_score=row["priority_score"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            ))

        return positions

    async def update_position(self, position: Position) -> None:
        """Update existing position."""
        await self.connection.execute(
            """
            UPDATE positions 
            SET current_price = ?, pnl_dollars = ?, pnl_percent = ?, 
                priority_score = ?, updated_at = ?
            WHERE position_id = ?
            """,
            (
                str(position.current_price) if position.current_price else None,
                str(position.pnl_dollars),
                str(position.pnl_percent),
                position.priority_score,
                datetime.utcnow().isoformat(),
                position.position_id
            )
        )
        await self.connection.commit()

        logger.debug("Position updated", position_id=position.position_id)

    async def close_position(self, position_id: str, final_pnl: Decimal) -> None:
        """Close a position."""
        await self.connection.execute(
            """
            UPDATE positions 
            SET status = 'CLOSED', pnl_dollars = ?, closed_at = ?, updated_at = ?
            WHERE position_id = ?
            """,
            (
                str(final_pnl),
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                position_id
            )
        )
        await self.connection.commit()

        logger.info("Position closed", position_id=position_id, final_pnl=str(final_pnl))

    # Trading session methods
    async def create_session(self, session: TradingSession) -> str:
        """Create a new trading session."""
        await self.connection.execute(
            """
            INSERT INTO trading_sessions (session_id, account_id, session_date, 
                                         starting_balance, current_balance, realized_pnl,
                                         total_trades, winning_trades, losing_trades,
                                         max_drawdown, daily_loss_limit, is_active,
                                         created_at, updated_at, ended_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.account_id,
                session.session_date.isoformat(),
                str(session.starting_balance),
                str(session.current_balance),
                str(session.realized_pnl),
                session.total_trades,
                session.winning_trades,
                session.losing_trades,
                str(session.max_drawdown),
                str(session.daily_loss_limit),
                1 if session.is_active else 0,
                session.created_at.isoformat(),
                session.updated_at.isoformat() if session.updated_at else None,
                None
            )
        )
        await self.connection.commit()

        logger.info("Trading session created", session_id=session.session_id)
        return session.session_id

    async def get_session(self, session_id: str) -> TradingSession | None:
        """Get session by ID."""
        cursor = await self.connection.execute(
            "SELECT * FROM trading_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()

        if row:
            return TradingSession(
                session_id=row["session_id"],
                account_id=row["account_id"],
                session_date=datetime.fromisoformat(row["session_date"]),
                starting_balance=Decimal(row["starting_balance"]),
                current_balance=Decimal(row["current_balance"]),
                realized_pnl=Decimal(row["realized_pnl"]),
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                losing_trades=row["losing_trades"],
                max_drawdown=Decimal(row["max_drawdown"]),
                daily_loss_limit=Decimal(row["daily_loss_limit"]),
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            )
        return None

    async def get_active_session(self, account_id: str) -> TradingSession | None:
        """Get active session for an account."""
        cursor = await self.connection.execute(
            "SELECT * FROM trading_sessions WHERE account_id = ? AND is_active = 1",
            (account_id,)
        )
        row = await cursor.fetchone()

        if row:
            return TradingSession(
                session_id=row["session_id"],
                account_id=row["account_id"],
                session_date=datetime.fromisoformat(row["session_date"]),
                starting_balance=Decimal(row["starting_balance"]),
                current_balance=Decimal(row["current_balance"]),
                realized_pnl=Decimal(row["realized_pnl"]),
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                losing_trades=row["losing_trades"],
                max_drawdown=Decimal(row["max_drawdown"]),
                daily_loss_limit=Decimal(row["daily_loss_limit"]),
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            )
        return None

    async def update_session(self, session: TradingSession) -> None:
        """Update existing session."""
        await self.connection.execute(
            """
            UPDATE trading_sessions 
            SET current_balance = ?, realized_pnl = ?, total_trades = ?,
                winning_trades = ?, losing_trades = ?, max_drawdown = ?,
                updated_at = ?
            WHERE session_id = ?
            """,
            (
                str(session.current_balance),
                str(session.realized_pnl),
                session.total_trades,
                session.winning_trades,
                session.losing_trades,
                str(session.max_drawdown),
                datetime.utcnow().isoformat(),
                session.session_id
            )
        )
        await self.connection.commit()

        logger.debug("Session updated", session_id=session.session_id)

    async def end_session(self, session_id: str) -> None:
        """End a trading session."""
        await self.connection.execute(
            """
            UPDATE trading_sessions 
            SET is_active = 0, ended_at = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                session_id
            )
        )
        await self.connection.commit()

        logger.info("Session ended", session_id=session_id)

    # Position correlation methods
    async def save_correlation(self, correlation: PositionCorrelation) -> None:
        """Save position correlation."""
        # Ensure position_a_id < position_b_id for constraint
        pos_a = min(correlation.position_a_id, correlation.position_b_id)
        pos_b = max(correlation.position_a_id, correlation.position_b_id)

        await self.connection.execute(
            """
            INSERT OR REPLACE INTO position_correlations 
            (position_a_id, position_b_id, correlation_coefficient, alert_triggered, calculated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                pos_a,
                pos_b,
                str(correlation.correlation_coefficient),
                1 if correlation.alert_triggered else 0,
                correlation.calculated_at.isoformat()
            )
        )
        await self.connection.commit()

    async def get_correlations(self, position_id: str) -> list[PositionCorrelation]:
        """Get correlations for a position."""
        cursor = await self.connection.execute(
            """
            SELECT * FROM position_correlations 
            WHERE position_a_id = ? OR position_b_id = ?
            """,
            (position_id, position_id)
        )

        rows = await cursor.fetchall()
        correlations = []

        for row in rows:
            correlations.append(PositionCorrelation(
                position_a_id=row["position_a_id"],
                position_b_id=row["position_b_id"],
                correlation_coefficient=Decimal(row["correlation_coefficient"]),
                alert_triggered=bool(row["alert_triggered"]),
                calculated_at=datetime.fromisoformat(row["calculated_at"])
            ))

        return correlations

    # Risk metrics methods
    async def save_risk_metrics(self, metrics: dict[str, Any]) -> None:
        """Save risk metrics snapshot."""
        from uuid import uuid4

        await self.connection.execute(
            """
            INSERT INTO risk_metrics (metric_id, account_id, timestamp, total_exposure,
                                     position_count, total_pnl_dollars, total_pnl_percent,
                                     max_position_size, daily_pnl, risk_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                metrics["account_id"],
                metrics.get("timestamp", datetime.utcnow()).isoformat(),
                str(metrics["total_exposure"]),
                metrics["position_count"],
                str(metrics["total_pnl_dollars"]),
                str(metrics["total_pnl_percent"]),
                str(metrics["max_position_size"]),
                str(metrics["daily_pnl"]),
                str(metrics.get("risk_score")) if metrics.get("risk_score") else None
            )
        )
        await self.connection.commit()

    async def get_risk_metrics(self, account_id: str, start_time: datetime, end_time: datetime) -> list[dict[str, Any]]:
        """Get risk metrics for time range."""
        cursor = await self.connection.execute(
            """
            SELECT * FROM risk_metrics 
            WHERE account_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            """,
            (account_id, start_time.isoformat(), end_time.isoformat())
        )

        rows = await cursor.fetchall()
        metrics = []

        for row in rows:
            metrics.append({
                "metric_id": row["metric_id"],
                "account_id": row["account_id"],
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "total_exposure": Decimal(row["total_exposure"]),
                "position_count": row["position_count"],
                "total_pnl_dollars": Decimal(row["total_pnl_dollars"]),
                "total_pnl_percent": Decimal(row["total_pnl_percent"]),
                "max_position_size": Decimal(row["max_position_size"]),
                "daily_pnl": Decimal(row["daily_pnl"]),
                "risk_score": Decimal(row["risk_score"]) if row["risk_score"] else None
            })

        return metrics

    # Order methods
    async def create_order(self, order: Order) -> str:
        """Create a new order."""
        await self.connection.execute(
            """
            INSERT INTO orders (order_id, position_id, client_order_id, exchange_order_id,
                              symbol, type, side, price, quantity, filled_quantity,
                              status, slice_number, total_slices, latency_ms,
                              slippage_percent, created_at, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.position_id,
                order.client_order_id,
                order.exchange_order_id,
                order.symbol,
                order.type.value,
                order.side.value,
                str(order.price) if order.price else None,
                str(order.quantity),
                str(order.filled_quantity),
                order.status.value,
                order.slice_number,
                order.total_slices,
                order.latency_ms,
                str(order.slippage_percent) if order.slippage_percent else None,
                order.created_at.isoformat(),
                order.executed_at.isoformat() if order.executed_at else None
            )
        )
        await self.connection.commit()

        logger.info("Order created", order_id=order.order_id)
        return order.order_id

    async def update_order(self, order: Order) -> None:
        """Update an existing order."""
        await self.connection.execute(
            """
            UPDATE orders 
            SET exchange_order_id = ?, filled_quantity = ?, status = ?, 
                latency_ms = ?, slippage_percent = ?, executed_at = ?
            WHERE order_id = ?
            """,
            (
                order.exchange_order_id,
                str(order.filled_quantity),
                order.status.value,
                order.latency_ms,
                str(order.slippage_percent) if order.slippage_percent else None,
                order.executed_at.isoformat() if order.executed_at else None,
                order.order_id
            )
        )
        await self.connection.commit()

        logger.info("Order updated", order_id=order.order_id, status=order.status.value)


    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by symbol."""
        if symbol:
            cursor = await self.connection.execute(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'PARTIAL') AND symbol = ? ORDER BY created_at DESC",
                (symbol,)
            )
        else:
            cursor = await self.connection.execute(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'PARTIAL') ORDER BY created_at DESC"
            )

        rows = await cursor.fetchall()

        orders = []
        for row in rows:
            orders.append(Order(
                order_id=row["order_id"],
                position_id=row["position_id"],
                client_order_id=row["client_order_id"],
                exchange_order_id=row["exchange_order_id"],
                symbol=row["symbol"],
                type=OrderType(row["type"]),
                side=OrderSide(row["side"]),
                price=Decimal(row["price"]) if row["price"] else None,
                quantity=Decimal(row["quantity"]),
                filled_quantity=Decimal(row["filled_quantity"]),
                status=OrderStatus(row["status"]),
                slice_number=row["slice_number"],
                total_slices=row["total_slices"],
                latency_ms=row["latency_ms"],
                slippage_percent=Decimal(row["slippage_percent"]) if row["slippage_percent"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
                executed_at=datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None
            ))

        return orders

    # Position recovery methods
    async def load_open_positions(self, account_id: str) -> list[Position]:
        """Load all open positions for recovery on startup."""
        positions = await self.get_positions_by_account(account_id, status="OPEN")

        # Log recovery event
        await self.save_event(
            "PositionsRecovered",
            account_id,
            {"position_count": len(positions), "account_id": account_id}
        )

        logger.info("Positions recovered", account_id=account_id, count=len(positions))
        return positions

    async def reconcile_positions(self, exchange_positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reconcile database positions with exchange state."""
        orphaned_positions = []

        # Get all open positions from database
        cursor = await self.connection.execute(
            "SELECT * FROM positions WHERE status = 'OPEN'"
        )
        db_positions = await cursor.fetchall()

        # Create lookup for exchange positions
        exchange_lookup = {pos["symbol"]: pos for pos in exchange_positions}

        for db_pos in db_positions:
            symbol = db_pos["symbol"]
            if symbol not in exchange_lookup:
                # Position exists in DB but not on exchange
                orphaned_positions.append({
                    "position_id": db_pos["position_id"],
                    "symbol": symbol,
                    "status": "orphaned",
                    "action": "close_in_db"
                })

                # Close the orphaned position
                await self.close_position(db_pos["position_id"], Decimal("0"))

                # Log reconciliation event
                await self.save_event(
                    "PositionOrphaned",
                    db_pos["position_id"],
                    {"symbol": symbol, "reason": "not_found_on_exchange"}
                )

        logger.info("Position reconciliation complete", orphaned_count=len(orphaned_positions))
        return orphaned_positions

    # Backup and restore methods
    async def backup(self, backup_path: Path | None = None) -> Path:
        """Create a backup of the database."""
        if backup_path is None:
            # Default backup path with timestamp
            backup_dir = Path(".genesis/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"genesis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db"

        # Use SQLite backup API via VACUUM INTO
        await self.connection.execute(f"VACUUM INTO '{backup_path!s}'")
        await self.connection.commit()

        # Log backup event
        await self.save_event(
            "DatabaseBackup",
            "system",
            {"backup_path": str(backup_path), "size_bytes": backup_path.stat().st_size}
        )

        logger.info("Database backed up", path=str(backup_path))
        return backup_path

    async def restore(self, backup_path: Path) -> None:
        """Restore database from backup."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Close current connection
        await self.shutdown()

        # Replace current database with backup
        shutil.copy2(backup_path, self.db_path)

        # Reinitialize connection
        await self.initialize()

        logger.info("Database restored from backup", path=str(backup_path))

    async def get_database_size(self) -> int:
        """Get current database size in bytes."""
        db_file = Path(self.db_path)
        if db_file.exists():
            return db_file.stat().st_size
        return 0

    async def rotate_database(self) -> None:
        """Rotate database when size limit reached (10MB)."""
        MAX_SIZE = 10 * 1024 * 1024  # 10MB

        current_size = await self.get_database_size()
        if current_size > MAX_SIZE:
            # Create backup before rotation
            backup_path = await self.backup()

            # Archive old data (keep last 30 days)
            cutoff_date = (datetime.utcnow() - timedelta(days=30)).isoformat()

            # Delete old events
            await self.connection.execute(
                "DELETE FROM events WHERE created_at < ?",
                (cutoff_date,)
            )

            # Delete old closed positions
            await self.connection.execute(
                "DELETE FROM positions WHERE status = 'CLOSED' AND closed_at < ?",
                (cutoff_date,)
            )

            # Delete old orders
            await self.connection.execute(
                "DELETE FROM orders WHERE created_at < ? AND status IN ('FILLED', 'CANCELLED', 'FAILED')",
                (cutoff_date,)
            )

            # Delete old risk metrics
            await self.connection.execute(
                "DELETE FROM risk_metrics WHERE timestamp < ?",
                (cutoff_date,)
            )

            # Vacuum to reclaim space
            await self.connection.execute("VACUUM")
            await self.connection.commit()

            new_size = await self.get_database_size()
            logger.info("Database rotated", old_size=current_size, new_size=new_size, backup=str(backup_path))

    # Export methods
    async def export_trades_to_csv(self, account_id: str, start_date: date, end_date: date, output_path: Path) -> Path:
        """Export trades to CSV for tax reporting."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Query closed positions in date range
        cursor = await self.connection.execute(
            """
            SELECT p.*, 
                   (SELECT SUM(CAST(o.filled_quantity AS REAL) * CAST(o.executed_price AS REAL))
                    FROM orders o 
                    WHERE o.position_id = p.position_id AND o.side = 'BUY') as total_buy_value,
                   (SELECT SUM(CAST(o.filled_quantity AS REAL) * CAST(o.executed_price AS REAL))
                    FROM orders o 
                    WHERE o.position_id = p.position_id AND o.side = 'SELL') as total_sell_value
            FROM positions p
            WHERE p.account_id = ? 
            AND p.status = 'CLOSED'
            AND DATE(p.closed_at) >= ?
            AND DATE(p.closed_at) <= ?
            ORDER BY p.closed_at
            """,
            (account_id, start_date.isoformat(), end_date.isoformat())
        )

        rows = await cursor.fetchall()

        # Write to CSV
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['date', 'symbol', 'side', 'quantity', 'entry_price',
                         'exit_price', 'pnl_usd', 'fees', 'close_reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            total_pnl = Decimal("0")
            for row in rows:
                # Calculate fees (simplified - would need real fee data)
                fees = Decimal("0")  # Placeholder

                writer.writerow({
                    'date': row['closed_at'],
                    'symbol': row['symbol'],
                    'side': row['side'],
                    'quantity': row['quantity'],
                    'entry_price': row['entry_price'],
                    'exit_price': row['current_price'],
                    'pnl_usd': row['pnl_dollars'],
                    'fees': str(fees),
                    'close_reason': row['close_reason'] or 'manual'
                })

                total_pnl += Decimal(row['pnl_dollars'])

            # Add totals row
            writer.writerow({
                'date': 'TOTAL',
                'symbol': '',
                'side': '',
                'quantity': '',
                'entry_price': '',
                'exit_price': '',
                'pnl_usd': str(total_pnl),
                'fees': '',
                'close_reason': ''
            })

        logger.info("Trades exported to CSV", path=str(output_path), count=len(rows))
        return output_path

    async def export_performance_report(self, account_id: str, output_path: Path) -> Path:
        """Export performance metrics report."""
        metrics = await self.calculate_performance_metrics(account_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("GENESIS Trading Performance Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Account ID: {account_id}\n")
            f.write(f"Report Date: {datetime.utcnow().isoformat()}\n\n")

            f.write("Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        logger.info("Performance report exported", path=str(output_path))
        return output_path

    # Performance metrics methods
    async def calculate_performance_metrics(self, account_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Calculate performance metrics (win rate, average R, etc.)."""
        if session_id:
            # Calculate for specific session
            cursor = await self.connection.execute(
                """
                SELECT * FROM positions 
                WHERE account_id = ? AND status = 'CLOSED'
                AND created_at >= (SELECT created_at FROM trading_sessions WHERE session_id = ?)
                AND closed_at <= (SELECT COALESCE(ended_at, datetime('now')) FROM trading_sessions WHERE session_id = ?)
                """,
                (account_id, session_id, session_id)
            )
        else:
            # Calculate for all time
            cursor = await self.connection.execute(
                "SELECT * FROM positions WHERE account_id = ? AND status = 'CLOSED'",
                (account_id,)
            )

        positions = await cursor.fetchall()

        if not positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "average_win": "0",
                "average_loss": "0",
                "average_r": "0",
                "profit_factor": "0",
                "max_drawdown": "0"
            }

        winning_trades = []
        losing_trades = []

        for pos in positions:
            pnl = Decimal(pos["pnl_dollars"])
            if pnl > 0:
                winning_trades.append(pnl)
            elif pnl < 0:
                losing_trades.append(abs(pnl))

        total_trades = len(positions)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else Decimal("0")
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else Decimal("0")

        # Calculate average R (risk/reward ratio)
        average_r = (avg_win / avg_loss) if avg_loss > 0 else Decimal("0")

        # Calculate profit factor
        total_wins = sum(winning_trades)
        total_losses = sum(losing_trades)
        profit_factor = (total_wins / total_losses) if total_losses > 0 else Decimal("0")

        # Calculate max drawdown (simplified)
        running_pnl = Decimal("0")
        peak = Decimal("0")
        max_drawdown = Decimal("0")

        for pos in sorted(positions, key=lambda x: x["closed_at"]):
            running_pnl += Decimal(pos["pnl_dollars"])
            if running_pnl > peak:
                peak = running_pnl
            drawdown = peak - running_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate": round(win_rate, 2),
            "average_win": str(avg_win.quantize(Decimal("0.01"))),
            "average_loss": str(avg_loss.quantize(Decimal("0.01"))),
            "average_r": str(average_r.quantize(Decimal("0.01"))),
            "profit_factor": str(profit_factor.quantize(Decimal("0.01"))),
            "max_drawdown": str(max_drawdown.quantize(Decimal("0.01")))
        }

    async def get_performance_report(self, account_id: str, start_date: date, end_date: date) -> dict[str, Any]:
        """Get comprehensive performance report."""
        # Get metrics for date range
        cursor = await self.connection.execute(
            """
            SELECT * FROM positions 
            WHERE account_id = ? AND status = 'CLOSED'
            AND DATE(closed_at) >= ? AND DATE(closed_at) <= ?
            """,
            (account_id, start_date.isoformat(), end_date.isoformat())
        )
        positions = await cursor.fetchall()

        # Calculate daily P&L
        daily_pnl = {}
        for pos in positions:
            close_date = pos["closed_at"][:10]  # Extract date part
            if close_date not in daily_pnl:
                daily_pnl[close_date] = Decimal("0")
            daily_pnl[close_date] += Decimal(pos["pnl_dollars"])

        # Get account info
        account = await self.get_account(account_id)

        # Calculate metrics
        metrics = await self.calculate_performance_metrics(account_id)

        return {
            "account_id": account_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "account_balance": str(account.balance_usdt) if account else "0",
            "metrics": metrics,
            "daily_pnl": {date: str(pnl) for date, pnl in daily_pnl.items()},
            "total_pnl": str(sum(daily_pnl.values()))
        }

    # Tilt event methods
    async def save_tilt_event(self, session_id: str, event_type: str, severity: str,
                             indicator_values: dict[str, Any], intervention: str | None = None) -> str:
        """Save a tilt event."""
        event_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO tilt_events (event_id, session_id, event_type, severity,
                                   indicator_values, intervention_taken, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                event_type,
                severity,
                json.dumps(indicator_values, default=decimal_default),
                intervention,
                datetime.utcnow().isoformat()
            )
        )
        await self.connection.commit()

        # Also save to event store
        await self.save_event(
            "TiltDetected",
            session_id,
            {
                "event_id": event_id,
                "event_type": event_type,
                "severity": severity,
                "indicator_values": indicator_values,
                "intervention": intervention
            }
        )

        logger.warning("Tilt event saved", event_id=event_id, severity=severity)
        return event_id

    async def get_tilt_events(self, session_id: str) -> list[dict[str, Any]]:
        """Get all tilt events for a session."""
        cursor = await self.connection.execute(
            "SELECT * FROM tilt_events WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        )
        rows = await cursor.fetchall()

        events = []
        for row in rows:
            events.append({
                "event_id": row["event_id"],
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "severity": row["severity"],
                "indicator_values": json.loads(row["indicator_values"]),
                "intervention_taken": row["intervention_taken"],
                "created_at": datetime.fromisoformat(row["created_at"])
            })

        return events

    # Database management
    async def set_database_info(self, key: str, value: str) -> None:
        """Set database metadata."""
        await self.connection.execute(
            """
            INSERT OR REPLACE INTO database_info (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, datetime.utcnow().isoformat())
        )
        await self.connection.commit()

    async def get_database_info(self, key: str) -> str | None:
        """Get database metadata."""
        cursor = await self.connection.execute(
            "SELECT value FROM database_info WHERE key = ?",
            (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    # Transaction methods
    async def begin_transaction(self) -> None:
        """Begin a database transaction."""
        await self.connection.execute("BEGIN")

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        await self.connection.commit()

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        await self.connection.rollback()
