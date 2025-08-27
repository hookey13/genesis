"""
SQLite repository implementation for Project GENESIS.

This module implements the repository pattern using SQLite
for the MVP phase of the project.
"""

import csv
import json
import shutil
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional, Any
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
        self.connection: Optional[aiosqlite.Connection] = None

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
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                event_data TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(aggregate_id, sequence_number)
            )
        """
        )

        # Accounts table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                balance_usdt TEXT NOT NULL,
                tier TEXT NOT NULL DEFAULT 'SNIPER',
                locked_features TEXT DEFAULT '[]',
                last_sync TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """
        )

        # Positions table
        await self.connection.execute(
            """
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
        """
        )

        # Trading sessions table
        await self.connection.execute(
            """
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
        """
        )

        # Position correlations table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS position_correlations (
                position_a_id TEXT NOT NULL REFERENCES positions(position_id),
                position_b_id TEXT NOT NULL REFERENCES positions(position_id),
                correlation_coefficient TEXT NOT NULL,
                alert_triggered INTEGER NOT NULL DEFAULT 0,
                calculated_at TEXT NOT NULL,
                PRIMARY KEY (position_a_id, position_b_id),
                CHECK (position_a_id < position_b_id)
            )
        """
        )

        # Orders table
        await self.connection.execute(
            """
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
        """
        )

        # Risk metrics table
        await self.connection.execute(
            """
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
        """
        )

        # Tilt profiles table for behavioral baseline tracking
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tilt_profiles (
                profile_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL UNIQUE REFERENCES accounts(account_id),
                baseline_trades_per_hour TEXT,
                baseline_click_latency_ms INTEGER,
                baseline_cancel_rate TEXT,
                current_tilt_score INTEGER NOT NULL DEFAULT 0 CHECK (current_tilt_score >= 0 AND current_tilt_score <= 100),
                tilt_level TEXT NOT NULL DEFAULT 'NORMAL' CHECK (tilt_level IN ('NORMAL', 'CAUTION', 'WARNING', 'LOCKED')),
                consecutive_losses INTEGER NOT NULL DEFAULT 0,
                last_intervention_at TEXT,
                recovery_required INTEGER NOT NULL DEFAULT 0,
                journal_entries_required INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """
        )

        # Behavioral metrics table for tracking trading behavior patterns
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS behavioral_metrics (
                metric_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES tilt_profiles(profile_id),
                metric_type TEXT NOT NULL CHECK (metric_type IN ('click_speed', 'order_frequency', 'position_size_variance', 'cancel_rate')),
                value TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_context TEXT CHECK (session_context IN ('tired', 'alert', 'stressed') OR session_context IS NULL),
                time_of_day_bucket INTEGER CHECK ((time_of_day_bucket >= 0 AND time_of_day_bucket <= 23) OR time_of_day_bucket IS NULL),
                created_at TEXT NOT NULL
            )
        """
        )

        # Tilt events table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tilt_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES trading_sessions(session_id),
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                indicator_values TEXT NOT NULL,
                intervention_taken TEXT,
                created_at TEXT NOT NULL
            )
        """
        )

        # Database info table
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS database_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Market states table for tracking market conditions
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_states (
                state_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('DEAD', 'NORMAL', 'VOLATILE', 'PANIC', 'MAINTENANCE')),
                volatility_atr TEXT,
                spread_basis_points INTEGER,
                volume_24h TEXT,
                liquidity_score TEXT,
                detected_at TEXT NOT NULL,
                state_duration_seconds INTEGER
            )
        """
        )

        # Volume profiles table for tracking volume patterns
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS volume_profiles (
                profile_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                hour INTEGER NOT NULL CHECK (hour >= 0 AND hour <= 23),
                volume TEXT NOT NULL,
                trade_count INTEGER NOT NULL DEFAULT 0,
                average_trade_size TEXT,
                date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                UNIQUE(symbol, hour, date)
            )
        """
        )

        # Global market states table for overall market conditions
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS global_market_states (
                state_id TEXT PRIMARY KEY,
                btc_price TEXT NOT NULL,
                total_market_cap TEXT,
                fear_greed_index INTEGER CHECK (fear_greed_index BETWEEN 0 AND 100),
                correlation_spike BOOLEAN NOT NULL DEFAULT FALSE,
                state TEXT NOT NULL CHECK (state IN ('BULL', 'BEAR', 'CRAB', 'CRASH', 'RECOVERY')),
                vix_crypto TEXT,
                detected_at TEXT NOT NULL
            )
        """
        )

        # Create indexes
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id, sequence_number)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_states_symbol ON market_states(symbol, detected_at)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_states_current ON market_states(symbol, state_id)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_global_market_states_time ON global_market_states(detected_at)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_volume_profile_symbol ON volume_profiles(symbol)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_volume_profile_date ON volume_profiles(date)"
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

        # Arbitrage tables
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS arbitrage_signals (
                signal_id TEXT PRIMARY KEY,
                pair1_symbol TEXT NOT NULL,
                pair2_symbol TEXT NOT NULL,
                zscore TEXT NOT NULL,
                threshold_sigma REAL NOT NULL,
                signal_type TEXT NOT NULL CHECK(signal_type IN ('ENTRY', 'EXIT')),
                confidence_score TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_arbitrage_signals_pairs ON arbitrage_signals(pair1_symbol, pair2_symbol)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_arbitrage_signals_created ON arbitrage_signals(created_at)"
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS spread_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair1_symbol TEXT NOT NULL,
                pair2_symbol TEXT NOT NULL,
                spread_value TEXT NOT NULL,
                recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_spread_history_pairs ON spread_history(pair1_symbol, pair2_symbol)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_spread_history_time ON spread_history(recorded_at)"
        )

        # Liquidity scanner tables
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS liquidity_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                volume_24h TEXT NOT NULL,
                liquidity_tier TEXT NOT NULL CHECK (liquidity_tier IN ('LOW', 'MEDIUM', 'HIGH')),
                spread_basis_points INTEGER NOT NULL CHECK (spread_basis_points >= 0),
                bid_depth_10 TEXT NOT NULL,
                ask_depth_10 TEXT NOT NULL,
                spread_persistence_score TEXT NOT NULL,
                scanned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_liquidity_symbol ON liquidity_snapshots(symbol, scanned_at)"
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_liquidity_volume ON liquidity_snapshots(volume_24h)"
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pair_blacklist (
                blacklist_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL UNIQUE,
                blacklist_reason TEXT NOT NULL,
                consecutive_losses INTEGER NOT NULL CHECK (consecutive_losses >= 0),
                blacklisted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_blacklist_symbol ON pair_blacklist(symbol)"
        )

        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tier_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                tier TEXT NOT NULL CHECK (tier IN ('SNIPER', 'HUNTER', 'STRATEGIST')),
                symbol TEXT NOT NULL,
                volume_24h TEXT NOT NULL,
                liquidity_score TEXT NOT NULL,
                recommended_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        await self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_recommendations_tier ON tier_recommendations(tier, recommended_at)"
        )

    # Event store methods
    async def save_event(
        self, event_type: str, aggregate_id: str, event_data: dict[str, Any]
    ) -> str:
        """Save an event to the event store."""
        event_id = str(uuid4())

        # Get next sequence number for this aggregate
        cursor = await self.connection.execute(
            "SELECT MAX(sequence_number) as max_seq FROM events WHERE aggregate_id = ?",
            (aggregate_id,),
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
                datetime.now(UTC).isoformat(),
            ),
        )
        await self.connection.commit()

        logger.debug("Event saved", event_id=event_id, event_type=event_type)
        return event_id

    async def get_events(
        self, aggregate_id: str, event_type: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get events for an aggregate."""
        if event_type:
            cursor = await self.connection.execute(
                """
                SELECT * FROM events 
                WHERE aggregate_id = ? AND event_type = ?
                ORDER BY sequence_number
                """,
                (aggregate_id, event_type),
            )
        else:
            cursor = await self.connection.execute(
                """
                SELECT * FROM events 
                WHERE aggregate_id = ?
                ORDER BY sequence_number
                """,
                (aggregate_id,),
            )

        rows = await cursor.fetchall()
        events = []

        for row in rows:
            events.append(
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "aggregate_id": row["aggregate_id"],
                    "event_data": json.loads(row["event_data"]),
                    "sequence_number": row["sequence_number"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )

        return events

    async def get_events_by_type(
        self, event_type: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Get events by type and time range."""
        cursor = await self.connection.execute(
            """
            SELECT * FROM events 
            WHERE event_type = ? AND created_at >= ? AND created_at <= ?
            ORDER BY created_at DESC
            """,
            (event_type, start_time.isoformat(), end_time.isoformat()),
        )

        rows = await cursor.fetchall()
        events = []

        for row in rows:
            events.append(
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "aggregate_id": row["aggregate_id"],
                    "event_data": json.loads(row["event_data"]),
                    "sequence_number": row["sequence_number"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )

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
                datetime.now(UTC).isoformat(),
            ),
        )
        await self.connection.commit()

        logger.info("Account created", account_id=account.account_id)
        return account.account_id

    async def get_account(self, account_id: str) -> Optional[Account]:
        """Get account by ID."""
        import json

        cursor = await self.connection.execute(
            "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
        )
        row = await cursor.fetchone()

        if row:
            return Account(
                account_id=row["account_id"],
                balance_usdt=Decimal(row["balance_usdt"]),
                tier=TradingTier[row["tier"]],
                locked_features=json.loads(row["locked_features"]),
                last_sync=(
                    datetime.fromisoformat(row["last_sync"])
                    if row["last_sync"]
                    else None
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
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
                datetime.now(UTC).isoformat(),
                account.account_id,
            ),
        )
        await self.connection.commit()

        logger.debug("Account updated", account_id=account.account_id)

    async def delete_account(self, account_id: str) -> None:
        """Delete account."""
        await self.connection.execute(
            "DELETE FROM accounts WHERE account_id = ?", (account_id,)
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
                (
                    str(order["slippage_percent"])
                    if order.get("slippage_percent")
                    else None
                ),
                order.get("created_at", datetime.now(UTC)).isoformat(),
                order["executed_at"].isoformat() if order.get("executed_at") else None,
            ),
        )
        await self.connection.commit()

        # Log event
        await self.save_event("OrderCreated", order_id, order)

        logger.info("Order saved", order_id=order_id)
        return order_id

    async def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        """Get order by ID."""
        cursor = await self.connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
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
            "executed_price": (
                Decimal(row["executed_price"]) if row["executed_price"] else None
            ),
            "status": row["status"],
            "slice_number": row["slice_number"],
            "total_slices": row["total_slices"],
            "latency_ms": row["latency_ms"],
            "slippage_percent": (
                Decimal(row["slippage_percent"]) if row["slippage_percent"] else None
            ),
            "created_at": datetime.fromisoformat(row["created_at"]),
            "executed_at": (
                datetime.fromisoformat(row["executed_at"])
                if row["executed_at"]
                else None
            ),
        }

    async def get_orders_by_position(self, position_id: str) -> list[dict[str, Any]]:
        """Get all orders for a position."""
        cursor = await self.connection.execute(
            "SELECT * FROM orders WHERE position_id = ? ORDER BY created_at DESC",
            (position_id,),
        )
        rows = await cursor.fetchall()

        orders = []
        for row in rows:
            orders.append(
                {
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
                    "executed_price": (
                        Decimal(row["executed_price"])
                        if row["executed_price"]
                        else None
                    ),
                    "status": row["status"],
                    "slice_number": row["slice_number"],
                    "total_slices": row["total_slices"],
                    "latency_ms": row["latency_ms"],
                    "slippage_percent": (
                        Decimal(row["slippage_percent"])
                        if row["slippage_percent"]
                        else None
                    ),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                    "executed_at": (
                        datetime.fromisoformat(row["executed_at"])
                        if row["executed_at"]
                        else None
                    ),
                }
            )

        return orders

    async def update_order_status(
        self, order_id: str, status: str, executed_at: Optional[datetime] = None
    ) -> None:
        """Update order status."""
        await self.connection.execute(
            """
            UPDATE orders 
            SET status = ?, executed_at = ?
            WHERE order_id = ?
            """,
            (status, executed_at.isoformat() if executed_at else None, order_id),
        )
        await self.connection.commit()

        # Log event
        await self.save_event(
            "OrderStatusUpdated",
            order_id,
            {
                "status": status,
                "executed_at": executed_at.isoformat() if executed_at else None,
            },
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
                None,
            ),
        )
        await self.connection.commit()

        logger.info("Position created", position_id=position.position_id)
        return position.position_id

    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID."""
        cursor = await self.connection.execute(
            "SELECT * FROM positions WHERE position_id = ?", (position_id,)
        )
        row = await cursor.fetchone()

        if row:
            return Position(
                position_id=row["position_id"],
                account_id=row["account_id"],
                symbol=row["symbol"],
                side=PositionSide[row["side"]],
                entry_price=Decimal(row["entry_price"]),
                current_price=(
                    Decimal(row["current_price"]) if row["current_price"] else None
                ),
                quantity=Decimal(row["quantity"]),
                dollar_value=Decimal(row["dollar_value"]),
                stop_loss=Decimal(row["stop_loss"]) if row["stop_loss"] else None,
                pnl_dollars=Decimal(row["pnl_dollars"]),
                pnl_percent=Decimal(row["pnl_percent"]),
                priority_score=row["priority_score"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=(
                    datetime.fromisoformat(row["updated_at"])
                    if row["updated_at"]
                    else None
                ),
            )
        return None

    async def get_positions_by_account(
        self, account_id: str, status: Optional[str] = None
    ) -> list[Position]:
        """Get all positions for an account."""
        if status:
            cursor = await self.connection.execute(
                "SELECT * FROM positions WHERE account_id = ? AND status = ?",
                (account_id, status),
            )
        else:
            cursor = await self.connection.execute(
                "SELECT * FROM positions WHERE account_id = ?", (account_id,)
            )

        rows = await cursor.fetchall()
        positions = []

        for row in rows:
            positions.append(
                Position(
                    position_id=row["position_id"],
                    account_id=row["account_id"],
                    symbol=row["symbol"],
                    side=PositionSide[row["side"]],
                    entry_price=Decimal(row["entry_price"]),
                    current_price=(
                        Decimal(row["current_price"]) if row["current_price"] else None
                    ),
                    quantity=Decimal(row["quantity"]),
                    dollar_value=Decimal(row["dollar_value"]),
                    stop_loss=Decimal(row["stop_loss"]) if row["stop_loss"] else None,
                    pnl_dollars=Decimal(row["pnl_dollars"]),
                    pnl_percent=Decimal(row["pnl_percent"]),
                    priority_score=row["priority_score"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=(
                        datetime.fromisoformat(row["updated_at"])
                        if row["updated_at"]
                        else None
                    ),
                )
            )

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
                datetime.now(UTC).isoformat(),
                position.position_id,
            ),
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
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
                position_id,
            ),
        )
        await self.connection.commit()

        logger.info(
            "Position closed", position_id=position_id, final_pnl=str(final_pnl)
        )

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
                None,
            ),
        )
        await self.connection.commit()

        logger.info("Trading session created", session_id=session.session_id)
        return session.session_id

    async def get_session(self, session_id: str) -> Optional[TradingSession]:
        """Get session by ID."""
        cursor = await self.connection.execute(
            "SELECT * FROM trading_sessions WHERE session_id = ?", (session_id,)
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
                updated_at=(
                    datetime.fromisoformat(row["updated_at"])
                    if row["updated_at"]
                    else None
                ),
            )
        return None

    async def get_active_session(self, account_id: str) -> Optional[TradingSession]:
        """Get active session for an account."""
        cursor = await self.connection.execute(
            "SELECT * FROM trading_sessions WHERE account_id = ? AND is_active = 1",
            (account_id,),
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
                updated_at=(
                    datetime.fromisoformat(row["updated_at"])
                    if row["updated_at"]
                    else None
                ),
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
                datetime.now(UTC).isoformat(),
                session.session_id,
            ),
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
            (datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat(), session_id),
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
                correlation.calculated_at.isoformat(),
            ),
        )
        await self.connection.commit()

    async def get_correlations(self, position_id: str) -> list[PositionCorrelation]:
        """Get correlations for a position."""
        cursor = await self.connection.execute(
            """
            SELECT * FROM position_correlations 
            WHERE position_a_id = ? OR position_b_id = ?
            """,
            (position_id, position_id),
        )

        rows = await cursor.fetchall()
        correlations = []

        for row in rows:
            correlations.append(
                PositionCorrelation(
                    position_a_id=row["position_a_id"],
                    position_b_id=row["position_b_id"],
                    correlation_coefficient=Decimal(row["correlation_coefficient"]),
                    alert_triggered=bool(row["alert_triggered"]),
                    calculated_at=datetime.fromisoformat(row["calculated_at"]),
                )
            )

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
                metrics.get("timestamp", datetime.now(UTC)).isoformat(),
                str(metrics["total_exposure"]),
                metrics["position_count"],
                str(metrics["total_pnl_dollars"]),
                str(metrics["total_pnl_percent"]),
                str(metrics["max_position_size"]),
                str(metrics["daily_pnl"]),
                str(metrics.get("risk_score")) if metrics.get("risk_score") else None,
            ),
        )
        await self.connection.commit()

    async def get_risk_metrics(
        self, account_id: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Get risk metrics for time range."""
        cursor = await self.connection.execute(
            """
            SELECT * FROM risk_metrics 
            WHERE account_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            """,
            (account_id, start_time.isoformat(), end_time.isoformat()),
        )

        rows = await cursor.fetchall()
        metrics = []

        for row in rows:
            metrics.append(
                {
                    "metric_id": row["metric_id"],
                    "account_id": row["account_id"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "total_exposure": Decimal(row["total_exposure"]),
                    "position_count": row["position_count"],
                    "total_pnl_dollars": Decimal(row["total_pnl_dollars"]),
                    "total_pnl_percent": Decimal(row["total_pnl_percent"]),
                    "max_position_size": Decimal(row["max_position_size"]),
                    "daily_pnl": Decimal(row["daily_pnl"]),
                    "risk_score": (
                        Decimal(row["risk_score"]) if row["risk_score"] else None
                    ),
                }
            )

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
                order.executed_at.isoformat() if order.executed_at else None,
            ),
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
                order.order_id,
            ),
        )
        await self.connection.commit()

        logger.info("Order updated", order_id=order.order_id, status=order.status.value)

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open orders, optionally filtered by symbol."""
        if symbol:
            cursor = await self.connection.execute(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'PARTIAL') AND symbol = ? ORDER BY created_at DESC",
                (symbol,),
            )
        else:
            cursor = await self.connection.execute(
                "SELECT * FROM orders WHERE status IN ('PENDING', 'PARTIAL') ORDER BY created_at DESC"
            )

        rows = await cursor.fetchall()

        orders = []
        for row in rows:
            orders.append(
                Order(
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
                    slippage_percent=(
                        Decimal(row["slippage_percent"])
                        if row["slippage_percent"]
                        else None
                    ),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    executed_at=(
                        datetime.fromisoformat(row["executed_at"])
                        if row["executed_at"]
                        else None
                    ),
                )
            )

        return orders

    # Position recovery methods
    async def load_open_positions(self, account_id: str) -> list[Position]:
        """Load all open positions for recovery on startup."""
        positions = await self.get_positions_by_account(account_id, status="OPEN")

        # Log recovery event
        await self.save_event(
            "PositionsRecovered",
            account_id,
            {"position_count": len(positions), "account_id": account_id},
        )

        logger.info("Positions recovered", account_id=account_id, count=len(positions))
        return positions

    async def reconcile_positions(
        self, exchange_positions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                orphaned_positions.append(
                    {
                        "position_id": db_pos["position_id"],
                        "symbol": symbol,
                        "status": "orphaned",
                        "action": "close_in_db",
                    }
                )

                # Close the orphaned position
                await self.close_position(db_pos["position_id"], Decimal("0"))

                # Log reconciliation event
                await self.save_event(
                    "PositionOrphaned",
                    db_pos["position_id"],
                    {"symbol": symbol, "reason": "not_found_on_exchange"},
                )

        logger.info(
            "Position reconciliation complete", orphaned_count=len(orphaned_positions)
        )
        return orphaned_positions

    # Backup and restore methods
    async def backup(self, backup_path: Optional[Path] = None) -> Path:
        """Create a backup of the database."""
        if backup_path is None:
            # Default backup path with timestamp
            backup_dir = Path(".genesis/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = (
                backup_dir / f"genesis_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.db"
            )

        # Use SQLite backup API via VACUUM INTO
        await self.connection.execute(f"VACUUM INTO '{backup_path!s}'")
        await self.connection.commit()

        # Log backup event
        await self.save_event(
            "DatabaseBackup",
            "system",
            {"backup_path": str(backup_path), "size_bytes": backup_path.stat().st_size},
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
            cutoff_date = (datetime.now(UTC) - timedelta(days=30)).isoformat()

            # Delete old events
            await self.connection.execute(
                "DELETE FROM events WHERE created_at < ?", (cutoff_date,)
            )

            # Delete old closed positions
            await self.connection.execute(
                "DELETE FROM positions WHERE status = 'CLOSED' AND closed_at < ?",
                (cutoff_date,),
            )

            # Delete old orders
            await self.connection.execute(
                "DELETE FROM orders WHERE created_at < ? AND status IN ('FILLED', 'CANCELLED', 'FAILED')",
                (cutoff_date,),
            )

            # Delete old risk metrics
            await self.connection.execute(
                "DELETE FROM risk_metrics WHERE timestamp < ?", (cutoff_date,)
            )

            # Vacuum to reclaim space
            await self.connection.execute("VACUUM")
            await self.connection.commit()

            new_size = await self.get_database_size()
            logger.info(
                "Database rotated",
                old_size=current_size,
                new_size=new_size,
                backup=str(backup_path),
            )

    # Export methods
    async def export_trades_to_csv(
        self, account_id: str, start_date: date, end_date: date, output_path: Path
    ) -> Path:
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
            (account_id, start_date.isoformat(), end_date.isoformat()),
        )

        rows = await cursor.fetchall()

        # Write to CSV
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "date",
                "symbol",
                "side",
                "quantity",
                "entry_price",
                "exit_price",
                "pnl_usd",
                "fees",
                "close_reason",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            total_pnl = Decimal("0")
            for row in rows:
                # Calculate fees (simplified - would need real fee data)
                fees = Decimal("0")  # Placeholder

                writer.writerow(
                    {
                        "date": row["closed_at"],
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "quantity": row["quantity"],
                        "entry_price": row["entry_price"],
                        "exit_price": row["current_price"],
                        "pnl_usd": row["pnl_dollars"],
                        "fees": str(fees),
                        "close_reason": row["close_reason"] or "manual",
                    }
                )

                total_pnl += Decimal(row["pnl_dollars"])

            # Add totals row
            writer.writerow(
                {
                    "date": "TOTAL",
                    "symbol": "",
                    "side": "",
                    "quantity": "",
                    "entry_price": "",
                    "exit_price": "",
                    "pnl_usd": str(total_pnl),
                    "fees": "",
                    "close_reason": "",
                }
            )

        logger.info("Trades exported to CSV", path=str(output_path), count=len(rows))
        return output_path

    async def export_performance_report(
        self, account_id: str, output_path: Path
    ) -> Path:
        """Export performance metrics report."""
        metrics = await self.calculate_performance_metrics(account_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("GENESIS Trading Performance Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Account ID: {account_id}\n")
            f.write(f"Report Date: {datetime.now(UTC).isoformat()}\n\n")

            f.write("Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        logger.info("Performance report exported", path=str(output_path))
        return output_path

    # Performance metrics methods
    async def calculate_performance_metrics(
        self, account_id: str, session_id: Optional[str] = None
    ) -> dict[str, Any]:
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
                (account_id, session_id, session_id),
            )
        else:
            # Calculate for all time
            cursor = await self.connection.execute(
                "SELECT * FROM positions WHERE account_id = ? AND status = 'CLOSED'",
                (account_id,),
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
                "max_drawdown": "0",
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

        avg_win = (
            sum(winning_trades) / len(winning_trades)
            if winning_trades
            else Decimal("0")
        )
        avg_loss = (
            sum(losing_trades) / len(losing_trades) if losing_trades else Decimal("0")
        )

        # Calculate average R (risk/reward ratio)
        average_r = (avg_win / avg_loss) if avg_loss > 0 else Decimal("0")

        # Calculate profit factor
        total_wins = sum(winning_trades)
        total_losses = sum(losing_trades)
        profit_factor = (
            (total_wins / total_losses) if total_losses > 0 else Decimal("0")
        )

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
            "max_drawdown": str(max_drawdown.quantize(Decimal("0.01"))),
        }

    async def get_performance_report(
        self, account_id: str, start_date: date, end_date: date
    ) -> dict[str, Any]:
        """Get comprehensive performance report."""
        # Get metrics for date range
        cursor = await self.connection.execute(
            """
            SELECT * FROM positions 
            WHERE account_id = ? AND status = 'CLOSED'
            AND DATE(closed_at) >= ? AND DATE(closed_at) <= ?
            """,
            (account_id, start_date.isoformat(), end_date.isoformat()),
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
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "account_balance": str(account.balance_usdt) if account else "0",
            "metrics": metrics,
            "daily_pnl": {date: str(pnl) for date, pnl in daily_pnl.items()},
            "total_pnl": str(sum(daily_pnl.values())),
        }

    # Tilt event methods
    async def save_tilt_event(
        self,
        session_id: str,
        event_type: str,
        severity: str,
        indicator_values: dict[str, Any],
        intervention: Optional[str] = None,
    ) -> str:
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
                datetime.now(UTC).isoformat(),
            ),
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
                "intervention": intervention,
            },
        )

        logger.warning("Tilt event saved", event_id=event_id, severity=severity)
        return event_id

    async def get_tilt_events(self, session_id: str) -> list[dict[str, Any]]:
        """Get all tilt events for a session."""
        cursor = await self.connection.execute(
            "SELECT * FROM tilt_events WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )
        rows = await cursor.fetchall()

        events = []
        for row in rows:
            events.append(
                {
                    "event_id": row["event_id"],
                    "session_id": row["session_id"],
                    "event_type": row["event_type"],
                    "severity": row["severity"],
                    "indicator_values": json.loads(row["indicator_values"]),
                    "intervention_taken": row["intervention_taken"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )

        return events

    # Database management
    async def set_database_info(self, key: str, value: str) -> None:
        """Set database metadata."""
        await self.connection.execute(
            """
            INSERT OR REPLACE INTO database_info (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, datetime.now(UTC).isoformat()),
        )
        await self.connection.commit()

    async def get_database_info(self, key: str) -> Optional[str]:
        """Get database metadata."""
        cursor = await self.connection.execute(
            "SELECT value FROM database_info WHERE key = ?", (key,)
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

    # Arbitrage signal methods
    async def save_arbitrage_signal(self, signal: dict[str, Any]) -> str:
        """
        Save an arbitrage signal to the database.

        Args:
            signal: Dictionary containing signal data

        Returns:
            Signal ID
        """
        signal_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO arbitrage_signals (
                signal_id, pair1_symbol, pair2_symbol, zscore,
                threshold_sigma, signal_type, confidence_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                signal_id,
                signal["pair1_symbol"],
                signal["pair2_symbol"],
                str(signal["zscore"]),
                float(signal["threshold_sigma"]),
                signal["signal_type"],
                str(signal["confidence_score"]),
                datetime.now(UTC),
            ),
        )

        await self.connection.commit()

        logger.info(
            "Arbitrage signal saved",
            signal_id=signal_id,
            pairs=f"{signal['pair1_symbol']}:{signal['pair2_symbol']}",
        )

        return signal_id

    async def get_arbitrage_signals(
        self,
        pair1: str = None,
        pair2: str = None,
        signal_type: str = None,
        hours_back: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Get arbitrage signals from the database.

        Args:
            pair1: Filter by first pair symbol
            pair2: Filter by second pair symbol
            signal_type: Filter by signal type (ENTRY/EXIT)
            hours_back: Number of hours to look back

        Returns:
            List of signal dictionaries
        """
        query = f"""
            SELECT * FROM arbitrage_signals 
            WHERE created_at > datetime('now', '-{hours_back} hours')
        """

        params = []

        if pair1 and pair2:
            query += " AND pair1_symbol = ? AND pair2_symbol = ?"
            params.extend([pair1, pair2])

        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)

        query += " ORDER BY created_at DESC"

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        signals = []
        for row in rows:
            signals.append(
                {
                    "signal_id": row["signal_id"],
                    "pair1_symbol": row["pair1_symbol"],
                    "pair2_symbol": row["pair2_symbol"],
                    "zscore": Decimal(row["zscore"]),
                    "threshold_sigma": Decimal(str(row["threshold_sigma"])),
                    "signal_type": row["signal_type"],
                    "confidence_score": Decimal(row["confidence_score"]),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )

        return signals

    async def save_spread_history(
        self, pair1: str, pair2: str, spread: Decimal
    ) -> None:
        """
        Save spread history data point.

        Args:
            pair1: First pair symbol
            pair2: Second pair symbol
            spread: Spread value
        """
        await self.connection.execute(
            """
            INSERT INTO spread_history (
                pair1_symbol, pair2_symbol, spread_value, recorded_at
            ) VALUES (?, ?, ?, ?)
        """,
            (pair1, pair2, str(spread), datetime.now(UTC)),
        )

        await self.connection.commit()

        logger.debug("Spread history saved", pairs=f"{pair1}:{pair2}", spread=spread)

    async def get_spread_history(
        self, pair1: str, pair2: str, days_back: int = 20
    ) -> list[dict[str, Any]]:
        """
        Get spread history for a pair.

        Args:
            pair1: First pair symbol
            pair2: Second pair symbol
            days_back: Number of days to look back

        Returns:
            List of spread history points
        """
        cursor = await self.connection.execute(
            f"""
            SELECT * FROM spread_history
            WHERE pair1_symbol = ? AND pair2_symbol = ?
            AND recorded_at > datetime('now', '-{days_back} days')
            ORDER BY recorded_at ASC
        """,
            (pair1, pair2),
        )

        rows = await cursor.fetchall()

        history = []
        for row in rows:
            history.append(
                {
                    "spread_value": Decimal(row["spread_value"]),
                    "recorded_at": datetime.fromisoformat(row["recorded_at"]),
                }
            )

        return history

    # Liquidity scanner methods
    async def save_liquidity_snapshot(self, snapshot: dict[str, Any]) -> str:
        """
        Save a liquidity snapshot to the database.

        Args:
            snapshot: Dictionary containing liquidity metrics

        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO liquidity_snapshots (
                snapshot_id, symbol, volume_24h, liquidity_tier,
                spread_basis_points, bid_depth_10, ask_depth_10,
                spread_persistence_score, scanned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                snapshot_id,
                snapshot["symbol"],
                str(snapshot["volume_24h"]),
                snapshot["liquidity_tier"],
                snapshot["spread_basis_points"],
                str(snapshot["bid_depth_10"]),
                str(snapshot["ask_depth_10"]),
                str(snapshot["spread_persistence_score"]),
                snapshot.get("scanned_at", datetime.now(UTC)),
            ),
        )

        await self.connection.commit()

        logger.debug(
            "Liquidity snapshot saved",
            snapshot_id=snapshot_id,
            symbol=snapshot["symbol"],
        )

        return snapshot_id

    async def get_liquidity_snapshots(
        self, symbol: str = None, tier: str = None, hours_back: int = 24
    ) -> list[dict[str, Any]]:
        """
        Get liquidity snapshots from the database.

        Args:
            symbol: Filter by symbol
            tier: Filter by liquidity tier
            hours_back: Number of hours to look back

        Returns:
            List of snapshot dictionaries
        """
        query = f"""
            SELECT * FROM liquidity_snapshots 
            WHERE scanned_at > datetime('now', '-{hours_back} hours')
        """

        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if tier:
            query += " AND liquidity_tier = ?"
            params.append(tier)

        query += " ORDER BY scanned_at DESC"

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        snapshots = []
        for row in rows:
            snapshots.append(
                {
                    "snapshot_id": row["snapshot_id"],
                    "symbol": row["symbol"],
                    "volume_24h": Decimal(row["volume_24h"]),
                    "liquidity_tier": row["liquidity_tier"],
                    "spread_basis_points": row["spread_basis_points"],
                    "bid_depth_10": Decimal(row["bid_depth_10"]),
                    "ask_depth_10": Decimal(row["ask_depth_10"]),
                    "spread_persistence_score": Decimal(
                        row["spread_persistence_score"]
                    ),
                    "scanned_at": datetime.fromisoformat(row["scanned_at"]),
                }
            )

        return snapshots

    async def save_pair_blacklist(self, blacklist_entry: dict[str, Any]) -> str:
        """
        Add a pair to the blacklist.

        Args:
            blacklist_entry: Dictionary containing blacklist data

        Returns:
            Blacklist ID
        """
        blacklist_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT OR REPLACE INTO pair_blacklist (
                blacklist_id, symbol, blacklist_reason,
                consecutive_losses, blacklisted_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                blacklist_id,
                blacklist_entry["symbol"],
                blacklist_entry["blacklist_reason"],
                blacklist_entry["consecutive_losses"],
                blacklist_entry.get("blacklisted_at", datetime.now(UTC)),
                blacklist_entry.get("expires_at"),
            ),
        )

        await self.connection.commit()

        logger.warning(
            "Pair blacklisted",
            blacklist_id=blacklist_id,
            symbol=blacklist_entry["symbol"],
        )

        return blacklist_id

    async def get_blacklisted_pairs(self) -> list[dict[str, Any]]:
        """
        Get all blacklisted pairs.

        Returns:
            List of blacklisted pair dictionaries
        """
        cursor = await self.connection.execute(
            """
            SELECT * FROM pair_blacklist
            WHERE expires_at IS NULL OR expires_at > datetime('now')
            ORDER BY blacklisted_at DESC
        """
        )

        rows = await cursor.fetchall()

        blacklist = []
        for row in rows:
            blacklist.append(
                {
                    "blacklist_id": row["blacklist_id"],
                    "symbol": row["symbol"],
                    "blacklist_reason": row["blacklist_reason"],
                    "consecutive_losses": row["consecutive_losses"],
                    "blacklisted_at": datetime.fromisoformat(row["blacklisted_at"]),
                    "expires_at": (
                        datetime.fromisoformat(row["expires_at"])
                        if row["expires_at"]
                        else None
                    ),
                }
            )

        return blacklist

    async def is_pair_blacklisted(self, symbol: str) -> bool:
        """
        Check if a pair is blacklisted.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if blacklisted
        """
        cursor = await self.connection.execute(
            """
            SELECT 1 FROM pair_blacklist
            WHERE symbol = ?
            AND (expires_at IS NULL OR expires_at > datetime('now'))
            LIMIT 1
        """,
            (symbol,),
        )

        row = await cursor.fetchone()
        return row is not None

    async def save_tier_recommendation(self, recommendation: dict[str, Any]) -> str:
        """
        Save a tier recommendation.

        Args:
            recommendation: Dictionary containing recommendation data

        Returns:
            Recommendation ID
        """
        recommendation_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO tier_recommendations (
                recommendation_id, tier, symbol,
                volume_24h, liquidity_score, recommended_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                recommendation_id,
                recommendation["tier"],
                recommendation["symbol"],
                str(recommendation["volume_24h"]),
                str(recommendation["liquidity_score"]),
                recommendation.get("recommended_at", datetime.now(UTC)),
            ),
        )

        await self.connection.commit()

        logger.info(
            "Tier recommendation saved",
            recommendation_id=recommendation_id,
            tier=recommendation["tier"],
            symbol=recommendation["symbol"],
        )

        return recommendation_id

    async def get_tier_recommendations(
        self, tier: str, days_back: int = 1
    ) -> list[dict[str, Any]]:
        """
        Get tier recommendations.

        Args:
            tier: Trading tier
            days_back: Number of days to look back

        Returns:
            List of recommendation dictionaries
        """
        cursor = await self.connection.execute(
            f"""
            SELECT * FROM tier_recommendations
            WHERE tier = ?
            AND recommended_at > datetime('now', '-{days_back} days')
            ORDER BY liquidity_score DESC
        """,
            (tier,),
        )

        rows = await cursor.fetchall()

        recommendations = []
        for row in rows:
            recommendations.append(
                {
                    "recommendation_id": row["recommendation_id"],
                    "tier": row["tier"],
                    "symbol": row["symbol"],
                    "volume_24h": Decimal(row["volume_24h"]),
                    "liquidity_score": Decimal(row["liquidity_score"]),
                    "recommended_at": datetime.fromisoformat(row["recommended_at"]),
                }
            )

        return recommendations

    async def save_market_state(self, market_state: dict) -> None:
        """
        Save market state to database.

        Args:
            market_state: Market state data dictionary
        """
        state_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO market_states (
                state_id, symbol, state, volatility_atr, spread_basis_points,
                volume_24h, liquidity_score, detected_at, state_duration_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                state_id,
                market_state["symbol"],
                market_state["state"],
                str(market_state.get("volatility_atr", "0")),
                market_state.get("spread_basis_points", 0),
                str(market_state.get("volume_24h", "0")),
                str(market_state.get("liquidity_score", "0")),
                market_state.get("detected_at", datetime.now(UTC)).isoformat(),
                market_state.get("state_duration_seconds", 0),
            ),
        )

        await self.connection.commit()
        logger.info(
            f"Saved market state for {market_state['symbol']}: {market_state['state']}"
        )

    async def get_latest_market_state(self, symbol: str) -> Optional[dict]:
        """
        Get the latest market state for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Latest market state dictionary or None
        """
        cursor = await self.connection.execute(
            """
            SELECT * FROM market_states
            WHERE symbol = ?
            ORDER BY detected_at DESC
            LIMIT 1
        """,
            (symbol,),
        )

        row = await cursor.fetchone()

        if row:
            return {
                "state_id": row["state_id"],
                "symbol": row["symbol"],
                "state": row["state"],
                "volatility_atr": (
                    Decimal(row["volatility_atr"]) if row["volatility_atr"] else None
                ),
                "spread_basis_points": row["spread_basis_points"],
                "volume_24h": Decimal(row["volume_24h"]) if row["volume_24h"] else None,
                "liquidity_score": (
                    Decimal(row["liquidity_score"]) if row["liquidity_score"] else None
                ),
                "detected_at": datetime.fromisoformat(row["detected_at"]),
                "state_duration_seconds": row["state_duration_seconds"],
            }

        return None

    async def get_market_state_history(
        self, symbol: str, hours_back: int = 24
    ) -> list[dict]:
        """
        Get market state history for a symbol.

        Args:
            symbol: Trading pair symbol
            hours_back: Hours to look back

        Returns:
            List of market state dictionaries
        """
        cursor = await self.connection.execute(
            f"""
            SELECT * FROM market_states
            WHERE symbol = ?
            AND detected_at > datetime('now', '-{hours_back} hours')
            ORDER BY detected_at DESC
        """,
            (symbol,),
        )

        rows = await cursor.fetchall()

        states = []
        for row in rows:
            states.append(
                {
                    "state_id": row["state_id"],
                    "symbol": row["symbol"],
                    "state": row["state"],
                    "volatility_atr": (
                        Decimal(row["volatility_atr"])
                        if row["volatility_atr"]
                        else None
                    ),
                    "spread_basis_points": row["spread_basis_points"],
                    "volume_24h": (
                        Decimal(row["volume_24h"]) if row["volume_24h"] else None
                    ),
                    "liquidity_score": (
                        Decimal(row["liquidity_score"])
                        if row["liquidity_score"]
                        else None
                    ),
                    "detected_at": datetime.fromisoformat(row["detected_at"]),
                    "state_duration_seconds": row["state_duration_seconds"],
                }
            )

        return states

    async def save_global_market_state(self, global_state: dict) -> None:
        """
        Save global market state to database.

        Args:
            global_state: Global market state data dictionary
        """
        state_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO global_market_states (
                state_id, btc_price, total_market_cap, fear_greed_index,
                correlation_spike, state, vix_crypto, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                state_id,
                str(global_state["btc_price"]),
                (
                    str(global_state.get("total_market_cap", "0"))
                    if global_state.get("total_market_cap")
                    else None
                ),
                global_state.get("fear_greed_index"),
                global_state.get("correlation_spike", False),
                global_state["state"],
                (
                    str(global_state.get("vix_crypto", "0"))
                    if global_state.get("vix_crypto")
                    else None
                ),
                global_state.get("detected_at", datetime.now(UTC)).isoformat(),
            ),
        )

        await self.connection.commit()
        logger.info(f"Saved global market state: {global_state['state']}")

    async def get_latest_global_market_state(self) -> Optional[dict]:
        """
        Get the latest global market state.

        Returns:
            Latest global market state dictionary or None
        """
        cursor = await self.connection.execute(
            """
            SELECT * FROM global_market_states
            ORDER BY detected_at DESC
            LIMIT 1
        """
        )

        row = await cursor.fetchone()

        if row:
            return {
                "state_id": row["state_id"],
                "btc_price": Decimal(row["btc_price"]),
                "total_market_cap": (
                    Decimal(row["total_market_cap"])
                    if row["total_market_cap"]
                    else None
                ),
                "fear_greed_index": row["fear_greed_index"],
                "correlation_spike": bool(row["correlation_spike"]),
                "state": row["state"],
                "vix_crypto": Decimal(row["vix_crypto"]) if row["vix_crypto"] else None,
                "detected_at": datetime.fromisoformat(row["detected_at"]),
            }

        return None

    async def save_spread_history(self, spread_data: dict) -> None:
        """
        Save spread history record.

        Args:
            spread_data: Dictionary containing spread metrics
        """
        history_id = str(uuid4())
        timestamp = spread_data.get("timestamp", datetime.now(UTC))

        await self.connection.execute(
            """
            INSERT INTO symbol_spread_history (
                history_id, symbol, spread_bps, bid_price, ask_price,
                bid_volume, ask_volume, order_imbalance, timestamp,
                hour_of_day, day_of_week
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                history_id,
                spread_data["symbol"],
                str(spread_data["spread_bps"]),
                str(spread_data["bid_price"]),
                str(spread_data["ask_price"]),
                (
                    str(spread_data.get("bid_volume", "0"))
                    if spread_data.get("bid_volume")
                    else None
                ),
                (
                    str(spread_data.get("ask_volume", "0"))
                    if spread_data.get("ask_volume")
                    else None
                ),
                (
                    str(spread_data.get("order_imbalance", "0"))
                    if spread_data.get("order_imbalance")
                    else None
                ),
                timestamp.isoformat(),
                timestamp.hour,
                timestamp.weekday(),
            ),
        )

        await self.connection.commit()
        logger.debug(f"Saved spread history for {spread_data['symbol']}")

    async def get_spread_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Get spread history for a symbol.

        Args:
            symbol: Trading pair symbol
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum records to return

        Returns:
            List of spread history dictionaries
        """
        query = """
            SELECT * FROM symbol_spread_history
            WHERE symbol = ?
        """
        params = [symbol]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        history = []
        for row in rows:
            history.append(
                {
                    "history_id": row["history_id"],
                    "symbol": row["symbol"],
                    "spread_bps": Decimal(row["spread_bps"]),
                    "bid_price": Decimal(row["bid_price"]),
                    "ask_price": Decimal(row["ask_price"]),
                    "bid_volume": (
                        Decimal(row["bid_volume"]) if row["bid_volume"] else None
                    ),
                    "ask_volume": (
                        Decimal(row["ask_volume"]) if row["ask_volume"] else None
                    ),
                    "order_imbalance": (
                        Decimal(row["order_imbalance"])
                        if row["order_imbalance"]
                        else None
                    ),
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "hour_of_day": row["hour_of_day"],
                    "day_of_week": row["day_of_week"],
                }
            )

        return history

    async def get_spread_patterns(self, symbol: str, days_back: int = 7) -> dict:
        """
        Get spread patterns by hour and day of week.

        Args:
            symbol: Trading pair symbol
            days_back: Number of days to analyze

        Returns:
            Dictionary with hourly and daily patterns
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)

        # Get hourly patterns
        hourly_cursor = await self.connection.execute(
            """
            SELECT 
                hour_of_day,
                AVG(CAST(spread_bps AS REAL)) as avg_spread,
                MIN(CAST(spread_bps AS REAL)) as min_spread,
                MAX(CAST(spread_bps AS REAL)) as max_spread,
                COUNT(*) as sample_count
            FROM symbol_spread_history
            WHERE symbol = ? AND timestamp >= ?
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """,
            (symbol, cutoff_date.isoformat()),
        )

        hourly_rows = await hourly_cursor.fetchall()
        hourly_patterns = {}
        for row in hourly_rows:
            hourly_patterns[row["hour_of_day"]] = {
                "avg_spread": Decimal(str(row["avg_spread"])),
                "min_spread": Decimal(str(row["min_spread"])),
                "max_spread": Decimal(str(row["max_spread"])),
                "sample_count": row["sample_count"],
            }

        # Get daily patterns
        daily_cursor = await self.connection.execute(
            """
            SELECT 
                day_of_week,
                AVG(CAST(spread_bps AS REAL)) as avg_spread,
                MIN(CAST(spread_bps AS REAL)) as min_spread,
                MAX(CAST(spread_bps AS REAL)) as max_spread,
                COUNT(*) as sample_count
            FROM symbol_spread_history
            WHERE symbol = ? AND timestamp >= ?
            GROUP BY day_of_week
            ORDER BY day_of_week
        """,
            (symbol, cutoff_date.isoformat()),
        )

        daily_rows = await daily_cursor.fetchall()
        daily_patterns = {}
        for row in daily_rows:
            daily_patterns[row["day_of_week"]] = {
                "avg_spread": Decimal(str(row["avg_spread"])),
                "min_spread": Decimal(str(row["min_spread"])),
                "max_spread": Decimal(str(row["max_spread"])),
                "sample_count": row["sample_count"],
            }

        return {"hourly": hourly_patterns, "daily": daily_patterns}

    async def cleanup_old_spread_history(self, retention_days: int = 30) -> int:
        """
        Clean up old spread history records.

        Args:
            retention_days: Number of days to retain

        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)

        cursor = await self.connection.execute(
            """
            DELETE FROM symbol_spread_history
            WHERE timestamp < ?
        """,
            (cutoff_date.isoformat(),),
        )

        await self.connection.commit()
        deleted_count = cursor.rowcount

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old spread history records")

        return deleted_count

    # Behavioral metrics methods for tilt detection
    async def save_behavioral_metric(self, metric: dict) -> str:
        """
        Save a behavioral metric to the database.

        Args:
            metric: Dictionary containing metric data

        Returns:
            Metric ID
        """
        metric_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO behavioral_metrics (
                metric_id, profile_id, metric_type, value, timestamp,
                session_context, time_of_day_bucket, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metric_id,
                metric.get("profile_id"),
                metric.get("metric_type"),
                str(metric.get("value")),
                (
                    metric.get("timestamp").isoformat()
                    if isinstance(metric.get("timestamp"), datetime)
                    else metric.get("timestamp")
                ),
                metric.get("session_context"),
                metric.get("time_of_day_bucket"),
                datetime.now(UTC).isoformat(),
            ),
        )

        await self.connection.commit()

        logger.debug(
            "Behavioral metric saved",
            metric_id=metric_id,
            metric_type=metric.get("metric_type"),
        )

        return metric_id

    async def get_metrics_for_baseline(
        self, profile_id: str, days: int = 30
    ) -> list[dict]:
        """
        Get behavioral metrics for baseline calculation.

        Args:
            profile_id: Profile ID
            days: Number of days to retrieve

        Returns:
            List of metric dictionaries
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        cursor = await self.connection.execute(
            """
            SELECT * FROM behavioral_metrics
            WHERE profile_id = ? AND timestamp >= ?
            ORDER BY timestamp
        """,
            (profile_id, cutoff_date.isoformat()),
        )

        rows = await cursor.fetchall()

        metrics = []
        for row in rows:
            metrics.append(
                {
                    "metric_id": row["metric_id"],
                    "profile_id": row["profile_id"],
                    "metric_type": row["metric_type"],
                    "value": Decimal(row["value"]),
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "session_context": row["session_context"],
                    "time_of_day_bucket": row["time_of_day_bucket"],
                }
            )

        logger.debug(
            "Retrieved metrics for baseline",
            profile_id=profile_id,
            count=len(metrics),
            days=days,
        )

        return metrics

    async def update_tilt_profile_baseline(
        self, profile_id: str, baseline: dict
    ) -> None:
        """
        Update tilt profile with new baseline values.

        Args:
            profile_id: Profile ID
            baseline: Baseline data dictionary
        """
        # Extract baseline metrics
        metric_ranges = baseline.get("metric_ranges", {})

        trades_per_hour = None
        click_latency = None
        cancel_rate = None

        if "order_frequency" in metric_ranges:
            trades_per_hour = str(metric_ranges["order_frequency"].get("mean", 0))

        if "click_speed" in metric_ranges:
            click_latency = int(metric_ranges["click_speed"].get("mean", 0))

        if "cancel_rate" in metric_ranges:
            cancel_rate = str(metric_ranges["cancel_rate"].get("mean", 0))

        await self.connection.execute(
            """
            UPDATE tilt_profiles
            SET baseline_trades_per_hour = ?,
                baseline_click_latency_ms = ?,
                baseline_cancel_rate = ?,
                updated_at = ?
            WHERE profile_id = ?
        """,
            (
                trades_per_hour,
                click_latency,
                cancel_rate,
                datetime.now(UTC).isoformat(),
                profile_id,
            ),
        )

        await self.connection.commit()

        logger.info(
            "Tilt profile baseline updated",
            profile_id=profile_id,
            trades_per_hour=trades_per_hour,
            click_latency=click_latency,
            cancel_rate=cancel_rate,
        )

    async def save_config_change(self, change: dict) -> str:
        """
        Save a configuration change to the database.

        Args:
            change: Dictionary containing config change data

        Returns:
            Change ID
        """
        change_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO config_changes (
                change_id, profile_id, setting_name, old_value, new_value,
                changed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                change_id,
                change.get("profile_id"),
                change.get("setting_name"),
                change.get("old_value"),
                change.get("new_value"),
                (
                    change.get("changed_at").isoformat()
                    if isinstance(change.get("changed_at"), datetime)
                    else change.get("changed_at")
                ),
                datetime.now(UTC).isoformat(),
            ),
        )

        await self.connection.commit()

        logger.debug(
            "Config change saved",
            change_id=change_id,
            setting_name=change.get("setting_name"),
        )

        return change_id

    async def save_behavior_correlation(self, correlation: dict) -> str:
        """
        Save a behavior-PnL correlation result to the database.

        Args:
            correlation: Dictionary containing correlation data

        Returns:
            Correlation ID
        """
        correlation_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO behavior_correlations (
                correlation_id, profile_id, behavior_type, correlation_coefficient,
                p_value, sample_size, time_window_minutes, calculated_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                correlation_id,
                correlation.get("profile_id"),
                correlation.get("behavior_type"),
                str(correlation.get("correlation_coefficient")),
                str(correlation.get("p_value")),
                correlation.get("sample_size"),
                correlation.get("time_window_minutes"),
                (
                    correlation.get("calculated_at").isoformat()
                    if isinstance(correlation.get("calculated_at"), datetime)
                    else correlation.get("calculated_at")
                ),
                datetime.now(UTC).isoformat(),
            ),
        )

        await self.connection.commit()

        logger.debug(
            "Behavior correlation saved",
            correlation_id=correlation_id,
            behavior_type=correlation.get("behavior_type"),
            correlation=correlation.get("correlation_coefficient"),
        )

        return correlation_id

    async def export_baseline_data(self, profile_id: str) -> dict:
        """
        Export baseline data for analysis.

        Args:
            profile_id: Profile ID

        Returns:
            Dictionary with baseline data
        """
        # Get profile data
        profile_cursor = await self.connection.execute(
            """
            SELECT * FROM tilt_profiles
            WHERE profile_id = ?
        """,
            (profile_id,),
        )

        profile_row = await profile_cursor.fetchone()

        if not profile_row:
            raise ValueError(f"Profile {profile_id} not found")

        # Get recent metrics
        metrics = await self.get_metrics_for_baseline(profile_id, days=30)

        # Group metrics by type
        metrics_by_type = {}
        for metric in metrics:
            metric_type = metric["metric_type"]
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(
                {
                    "value": float(metric["value"]),
                    "timestamp": metric["timestamp"].isoformat(),
                    "context": metric.get("session_context"),
                    "hour": metric.get("time_of_day_bucket"),
                }
            )

        export_data = {
            "profile_id": profile_id,
            "account_id": profile_row["account_id"],
            "baseline": {
                "trades_per_hour": (
                    float(profile_row["baseline_trades_per_hour"])
                    if profile_row["baseline_trades_per_hour"]
                    else None
                ),
                "click_latency_ms": profile_row["baseline_click_latency_ms"],
                "cancel_rate": (
                    float(profile_row["baseline_cancel_rate"])
                    if profile_row["baseline_cancel_rate"]
                    else None
                ),
            },
            "current_state": {
                "tilt_score": profile_row["current_tilt_score"],
                "tilt_level": profile_row["tilt_level"],
                "consecutive_losses": profile_row["consecutive_losses"],
                "recovery_required": profile_row["recovery_required"],
            },
            "metrics": metrics_by_type,
            "export_timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Baseline data exported", profile_id=profile_id, metric_count=len(metrics)
        )

        return export_data

    async def create_tilt_profile(self, account_id: str) -> str:
        """
        Create a new tilt profile for an account.

        Args:
            account_id: Account ID

        Returns:
            Profile ID
        """
        profile_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO tilt_profiles (
                profile_id, account_id, current_tilt_score, tilt_level,
                consecutive_losses, recovery_required, journal_entries_required,
                created_at
            ) VALUES (?, ?, 0, 'NORMAL', 0, 0, 0, ?)
        """,
            (profile_id, account_id, datetime.now(UTC).isoformat()),
        )

        await self.connection.commit()

        logger.info(
            "Tilt profile created", profile_id=profile_id, account_id=account_id
        )

        return profile_id

    async def get_tilt_profile(self, account_id: str) -> Optional[dict]:
        """
        Get tilt profile for an account.

        Args:
            account_id: Account ID

        Returns:
            Profile dictionary or None
        """
        cursor = await self.connection.execute(
            """
            SELECT * FROM tilt_profiles
            WHERE account_id = ?
        """,
            (account_id,),
        )

        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "profile_id": row["profile_id"],
            "account_id": row["account_id"],
            "baseline_trades_per_hour": (
                Decimal(row["baseline_trades_per_hour"])
                if row["baseline_trades_per_hour"]
                else None
            ),
            "baseline_click_latency_ms": row["baseline_click_latency_ms"],
            "baseline_cancel_rate": (
                Decimal(row["baseline_cancel_rate"])
                if row["baseline_cancel_rate"]
                else None
            ),
            "current_tilt_score": row["current_tilt_score"],
            "tilt_level": row["tilt_level"],
            "consecutive_losses": row["consecutive_losses"],
            "last_intervention_at": (
                datetime.fromisoformat(row["last_intervention_at"])
                if row["last_intervention_at"]
                else None
            ),
            "recovery_required": bool(row["recovery_required"]),
            "journal_entries_required": row["journal_entries_required"],
        }

    async def save_tilt_event(self, event: dict) -> str:
        """
        Save a tilt event to the database.

        Args:
            event: Tilt event dictionary with keys:
                - profile_id: Profile ID
                - event_type: Type of tilt event
                - tilt_indicators: List of triggered indicators (JSON)
                - tilt_score_before: Score before event
                - tilt_score_after: Score after event
                - intervention_message: Message shown to user
                - timestamp: Event timestamp

        Returns:
            Event ID
        """
        event_id = str(uuid4())

        await self.connection.execute(
            """
            INSERT INTO tilt_events (
                event_id, profile_id, event_type, tilt_indicators,
                tilt_score_before, tilt_score_after, intervention_message,
                timestamp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event_id,
                event["profile_id"],
                event["event_type"],
                json.dumps(event.get("tilt_indicators", [])),
                event["tilt_score_before"],
                event["tilt_score_after"],
                event.get("intervention_message"),
                (
                    event["timestamp"].isoformat()
                    if isinstance(event["timestamp"], datetime)
                    else event["timestamp"]
                ),
                datetime.now(UTC).isoformat(),
            ),
        )

        await self.connection.commit()

        logger.info(
            "Tilt event saved",
            event_id=event_id,
            profile_id=event["profile_id"],
            event_type=event["event_type"],
        )

        return event_id

    async def save_execution_quality(self, quality: Any) -> None:
        """
        Save execution quality metrics to the database.
        
        Args:
            quality: ExecutionQuality object with order execution metrics
        """
        from uuid import uuid4

        quality_id = str(uuid4())
        timestamp = quality.timestamp if hasattr(quality, 'timestamp') else datetime.now(UTC)

        await self.connection.execute(
            """
            INSERT OR REPLACE INTO execution_quality (
                quality_id, order_id, symbol, order_type, routing_method,
                slippage_bps, total_fees, maker_fees, taker_fees,
                time_to_fill_ms, fill_rate, price_improvement_bps,
                execution_score, market_conditions, timestamp, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                quality_id,
                quality.order_id,
                quality.symbol,
                quality.order_type,
                quality.routing_method,
                float(quality.slippage_bps),
                float(quality.total_fees),
                float(quality.maker_fees),
                float(quality.taker_fees),
                quality.time_to_fill_ms,
                float(quality.fill_rate),
                float(quality.price_improvement_bps),
                quality.execution_score,
                quality.market_conditions,  # JSON string
                timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                datetime.now(UTC).isoformat(),
            ),
        )

        await self.connection.commit()
        logger.info("Execution quality saved", order_id=quality.order_id, score=quality.execution_score)

    async def get_execution_quality_records(
        self, start_time: datetime, symbol: Optional[str] = None
    ) -> list[Any]:
        """
        Retrieve execution quality records from the database.
        
        Args:
            start_time: Start time for filtering records
            symbol: Optional symbol filter
            
        Returns:
            List of ExecutionQuality objects
        """
        from genesis.analytics.execution_quality import ExecutionQuality

        query = """
            SELECT * FROM execution_quality 
            WHERE timestamp >= ?
        """
        params = [start_time.isoformat()]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY timestamp DESC"

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        records = []
        for row in rows:
            records.append(ExecutionQuality(
                order_id=row["order_id"],
                symbol=row["symbol"],
                order_type=row["order_type"],
                routing_method=row["routing_method"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                slippage_bps=Decimal(str(row["slippage_bps"])),
                total_fees=Decimal(str(row["total_fees"])),
                maker_fees=Decimal(str(row["maker_fees"])),
                taker_fees=Decimal(str(row["taker_fees"])),
                time_to_fill_ms=row["time_to_fill_ms"],
                fill_rate=Decimal(str(row["fill_rate"])),
                price_improvement_bps=Decimal(str(row["price_improvement_bps"])),
                execution_score=row["execution_score"],
                market_conditions=row["market_conditions"],
            ))

        logger.info("Retrieved execution quality records", count=len(records))
        return records

    async def get_tilt_history(self, profile_id: str, days: int = 7) -> list:
        """
        Get tilt event history for a profile.

        Args:
            profile_id: Profile ID
            days: Number of days of history to retrieve

        Returns:
            List of tilt events
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)

        cursor = await self.connection.execute(
            """
            SELECT * FROM tilt_events
            WHERE profile_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """,
            (profile_id, cutoff.isoformat()),
        )

        rows = await cursor.fetchall()

        events = []
        for row in rows:
            events.append(
                {
                    "event_id": row["event_id"],
                    "profile_id": row["profile_id"],
                    "event_type": row["event_type"],
                    "tilt_indicators": json.loads(row["tilt_indicators"]),
                    "tilt_score_before": row["tilt_score_before"],
                    "tilt_score_after": row["tilt_score_after"],
                    "intervention_message": row["intervention_message"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "created_at": datetime.fromisoformat(row["created_at"]),
                }
            )

        return events

    async def update_tilt_profile_level(
        self, profile_id: str, level: str, score: int
    ) -> None:
        """
        Update tilt profile level and score.

        Args:
            profile_id: Profile ID
            level: New tilt level (NORMAL, LEVEL1, LEVEL2, LEVEL3)
            score: Current tilt score
        """
        await self.connection.execute(
            """
            UPDATE tilt_profiles
            SET tilt_level = ?, current_tilt_score = ?, updated_at = ?
            WHERE profile_id = ?
        """,
            (level, score, datetime.now(UTC).isoformat(), profile_id),
        )

        await self.connection.commit()

        logger.info(
            "Tilt profile updated", profile_id=profile_id, level=level, score=score
        )

    async def get_active_interventions(self, profile_id: str) -> list:
        """
        Get active interventions for a profile.

        Args:
            profile_id: Profile ID

        Returns:
            List of active interventions
        """
        # Get recent tilt events that may have interventions
        cutoff = datetime.now(UTC) - timedelta(hours=1)

        cursor = await self.connection.execute(
            """
            SELECT * FROM tilt_events
            WHERE profile_id = ? 
                AND timestamp > ?
                AND intervention_message IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        """,
            (profile_id, cutoff.isoformat()),
        )

        rows = await cursor.fetchall()

        interventions = []
        for row in rows:
            interventions.append(
                {
                    "event_id": row["event_id"],
                    "profile_id": row["profile_id"],
                    "event_type": row["event_type"],
                    "intervention_message": row["intervention_message"],
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                }
            )

        return interventions
