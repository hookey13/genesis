"""State persistence for paper trading simulation."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PersistenceConfig:
    """Configuration for state persistence."""

    db_path: str = ".genesis/paper_trading.db"
    auto_save_interval_seconds: int = 300
    backup_enabled: bool = True
    max_backups: int = 5


class StatePersistence:
    """Manages state persistence for paper trading."""

    def __init__(self, config: PersistenceConfig = None):
        """Initialize state persistence.

        Args:
            config: Persistence configuration
        """
        self.config = config or PersistenceConfig()
        self._connection = None
        self._init_database()

    def __del__(self):
        """Cleanup when object is destroyed."""
        # Ensure all connections are closed
        self.close()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        db_path = Path(self.config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()

            # Portfolio state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    strategy_id TEXT PRIMARY KEY,
                    current_balance TEXT,
                    positions TEXT,
                    trades TEXT,
                    metrics TEXT,
                    last_updated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Promotion state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS promotion_state (
                    strategy_id TEXT PRIMARY KEY,
                    status TEXT,
                    current_allocation TEXT,
                    target_allocation TEXT,
                    baseline_metrics TEXT,
                    promotion_history TEXT,
                    audit_trail TEXT,
                    last_updated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Order history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_history (
                    order_id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    order_type TEXT,
                    quantity TEXT,
                    price TEXT,
                    filled_quantity TEXT,
                    average_fill_price TEXT,
                    status TEXT,
                    latency_ms REAL,
                    slippage_bps TEXT,
                    timestamp TIMESTAMP,
                    fill_timestamp TIMESTAMP,
                    FOREIGN KEY (strategy_id) REFERENCES portfolio_state(strategy_id)
                )
            """)

            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT,
                    action TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (strategy_id) REFERENCES portfolio_state(strategy_id)
                )
            """)

            conn.commit()
            logger.info("Database initialized", db_path=str(db_path))

    def save_portfolio_state(
        self,
        strategy_id: str,
        current_balance: Decimal,
        positions: dict[str, Any],
        trades: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        """Save portfolio state to database.

        Args:
            strategy_id: Strategy identifier
            current_balance: Current balance
            positions: Current positions
            trades: Trade history
            metrics: Performance metrics
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO portfolio_state
                (strategy_id, current_balance, positions, trades, metrics, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    str(current_balance),
                    json.dumps(positions, default=str),
                    json.dumps(trades, default=str),
                    json.dumps(metrics, default=str),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

        logger.info("Portfolio state saved", strategy_id=strategy_id)

    def load_portfolio_state(
        self, strategy_id: str
    ) -> tuple[Decimal, dict, list, dict] | None:
        """Load portfolio state from database.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Tuple of (balance, positions, trades, metrics) or None if not found
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT current_balance, positions, trades, metrics
                FROM portfolio_state
                WHERE strategy_id = ?
                """,
                (strategy_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return (
            Decimal(row[0]),
            json.loads(row[1]),
            json.loads(row[2]),
            json.loads(row[3]),
        )

    def save_promotion_state(
        self,
        strategy_id: str,
        status: str,
        current_allocation: Decimal,
        target_allocation: Decimal,
        baseline_metrics: dict[str, Any],
        promotion_history: list[dict[str, Any]],
        audit_trail: list[dict[str, Any]],
    ) -> None:
        """Save promotion state to database.

        Args:
            strategy_id: Strategy identifier
            status: Current promotion status
            current_allocation: Current allocation
            target_allocation: Target allocation
            baseline_metrics: Baseline performance metrics
            promotion_history: Promotion history
            audit_trail: Audit trail
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO promotion_state
                (strategy_id, status, current_allocation, target_allocation,
                 baseline_metrics, promotion_history, audit_trail, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_id,
                    status,
                    str(current_allocation),
                    str(target_allocation),
                    json.dumps(baseline_metrics, default=str),
                    json.dumps(promotion_history, default=str),
                    json.dumps(audit_trail, default=str),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

        logger.info("Promotion state saved", strategy_id=strategy_id, status=status)

    def load_promotion_state(self, strategy_id: str) -> dict[str, Any] | None:
        """Load promotion state from database.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Promotion state dictionary or None if not found
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, current_allocation, target_allocation,
                       baseline_metrics, promotion_history, audit_trail
                FROM promotion_state
                WHERE strategy_id = ?
                """,
                (strategy_id,),
            )
            row = cursor.fetchone()

        if not row:
            return None

        return {
            "status": row[0],
            "current_allocation": Decimal(row[1]),
            "target_allocation": Decimal(row[2]),
            "baseline_metrics": json.loads(row[3]),
            "promotion_history": json.loads(row[4]),
            "audit_trail": json.loads(row[5]),
        }

    def save_order(self, order: Any, strategy_id: str | None = None) -> None:
        """Save order to history.

        Args:
            order: Order object to save
            strategy_id: Strategy ID (optional)
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO order_history
                (order_id, strategy_id, symbol, side, order_type, quantity, price,
                 filled_quantity, average_fill_price, status, latency_ms, slippage_bps,
                 timestamp, fill_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    strategy_id or getattr(order, "strategy_id", None),
                    order.symbol,
                    order.side,
                    order.order_type,
                    str(order.quantity),
                    str(order.price) if order.price else None,
                    str(order.filled_quantity) if order.filled_quantity else None,
                    (
                        str(order.average_fill_price)
                        if order.average_fill_price
                        else None
                    ),
                    order.status,
                    order.latency_ms,
                    str(order.slippage) if order.slippage else None,
                    order.timestamp.isoformat() if order.timestamp else None,
                    order.fill_timestamp.isoformat() if order.fill_timestamp else None,
                ),
            )
            conn.commit()

    def add_audit_entry(
        self, strategy_id: str, action: str, details: dict[str, Any]
    ) -> None:
        """Add entry to audit log.

        Args:
            strategy_id: Strategy identifier
            action: Action performed
            details: Action details
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_log (strategy_id, action, details)
                VALUES (?, ?, ?)
                """,
                (strategy_id, action, json.dumps(details, default=str)),
            )
            conn.commit()

    def get_order_history(
        self, strategy_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get order history for a strategy.

        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT order_id, symbol, side, order_type, quantity, price,
                       filled_quantity, average_fill_price, status, latency_ms,
                       slippage_bps, timestamp, fill_timestamp
                FROM order_history
                WHERE strategy_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (strategy_id, limit),
            )
            rows = cursor.fetchall()

        orders = []
        for row in rows:
            orders.append(
                {
                    "order_id": row[0],
                    "symbol": row[1],
                    "side": row[2],
                    "order_type": row[3],
                    "quantity": row[4],
                    "price": row[5],
                    "filled_quantity": row[6],
                    "average_fill_price": row[7],
                    "status": row[8],
                    "latency_ms": row[9],
                    "slippage_bps": row[10],
                    "timestamp": row[11],
                    "fill_timestamp": row[12],
                }
            )

        return orders

    def get_audit_log(
        self, strategy_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get audit log for a strategy.

        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT action, details, timestamp
                FROM audit_log
                WHERE strategy_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (strategy_id, limit),
            )
            rows = cursor.fetchall()

        entries = []
        for row in rows:
            entries.append(
                {
                    "action": row[0],
                    "details": json.loads(row[1]),
                    "timestamp": row[2],
                }
            )

        return entries

    def backup_database(self) -> str | None:
        """Create a backup of the database.

        Returns:
            Path to backup file or None if backup disabled
        """
        if not self.config.backup_enabled:
            return None

        db_path = Path(self.config.db_path)
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"paper_trading_{timestamp}.db"

        with sqlite3.connect(str(db_path)) as source, sqlite3.connect(str(backup_path)) as backup:
            source.backup(backup)

        # Clean up old backups
        backups = sorted(backup_dir.glob("paper_trading_*.db"))
        if len(backups) > self.config.max_backups:
            for old_backup in backups[: -self.config.max_backups]:
                old_backup.unlink()

        logger.info("Database backed up", backup_path=str(backup_path))
        return str(backup_path)

    def clear_old_data(self, days_to_keep: int = 30) -> None:
        """Clear old data from database.

        Args:
            days_to_keep: Number of days of data to keep
        """
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()

            # Clear old orders
            cursor.execute(
                """
                DELETE FROM order_history
                WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """,
                (days_to_keep,),
            )

            # Clear old audit entries
            cursor.execute(
                """
                DELETE FROM audit_log
                WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """,
                (days_to_keep,),
            )

            conn.commit()

        logger.info("Old data cleared", days_to_keep=days_to_keep)

    def close(self) -> None:
        """Close any open database connections."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            finally:
                self._connection = None

