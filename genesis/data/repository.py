"""
Abstract repository pattern for Project GENESIS.

This module defines the repository interface for data persistence,
allowing different implementations (SQLite, PostgreSQL, etc.).
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from genesis.core.models import (
    Account,
    Position,
    PositionCorrelation,
    TradingSession,
)


class Repository(ABC):
    """Abstract base class for data repository."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the repository (create tables, etc.)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the repository (close connections, etc.)."""
        pass

    # Account methods
    @abstractmethod
    async def create_account(self, account: Account) -> str:
        """Create a new account."""
        pass

    @abstractmethod
    async def get_account(self, account_id: str) -> Account | None:
        """Get account by ID."""
        pass

    @abstractmethod
    async def update_account(self, account: Account) -> None:
        """Update existing account."""
        pass

    @abstractmethod
    async def delete_account(self, account_id: str) -> None:
        """Delete account."""
        pass

    @abstractmethod
    async def save_account(self, account: Account) -> None:
        """Save account (create or update)."""
        pass

    @abstractmethod
    async def list_accounts(self) -> list[Account]:
        """List all accounts."""
        pass

    @abstractmethod
    async def list_positions(self, account_id: str | None = None) -> list[Position]:
        """List positions with optional account filter."""
        pass

    # Position methods
    @abstractmethod
    async def create_position(self, position: Position) -> str:
        """Create a new position."""
        pass

    @abstractmethod
    async def get_position(self, position_id: str) -> Position | None:
        """Get position by ID."""
        pass

    @abstractmethod
    async def get_positions_by_account(
        self, account_id: str, status: str | None = None
    ) -> list[Position]:
        """Get all positions for an account."""
        pass

    @abstractmethod
    async def update_position(self, position: Position) -> None:
        """Update existing position."""
        pass

    @abstractmethod
    async def close_position(self, position_id: str, final_pnl: Decimal) -> None:
        """Close a position."""
        pass

    # Trading session methods
    @abstractmethod
    async def create_session(self, session: TradingSession) -> str:
        """Create a new trading session."""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> TradingSession | None:
        """Get session by ID."""
        pass

    @abstractmethod
    async def get_active_session(self, account_id: str) -> TradingSession | None:
        """Get active session for an account."""
        pass

    @abstractmethod
    async def update_session(self, session: TradingSession) -> None:
        """Update existing session."""
        pass

    @abstractmethod
    async def end_session(self, session_id: str) -> None:
        """End a trading session."""
        pass

    # Position correlation methods
    @abstractmethod
    async def save_correlation(self, correlation: PositionCorrelation) -> None:
        """Save position correlation."""
        pass

    @abstractmethod
    async def get_correlations(self, position_id: str) -> list[PositionCorrelation]:
        """Get correlations for a position."""
        pass

    # Risk metrics methods
    @abstractmethod
    async def save_risk_metrics(self, metrics: dict[str, Any]) -> None:
        """Save risk metrics snapshot."""
        pass

    @abstractmethod
    async def get_risk_metrics(
        self, account_id: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Get risk metrics for time range."""
        pass

    # Event store methods
    @abstractmethod
    async def save_event(
        self, event_type: str, aggregate_id: str, event_data: dict[str, Any]
    ) -> str:
        """Save an event to the event store."""
        pass

    @abstractmethod
    async def get_events(
        self, aggregate_id: str, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Get events for an aggregate."""
        pass

    @abstractmethod
    async def get_events_by_type(
        self, event_type: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Get events by type and time range."""
        pass

    @abstractmethod
    async def get_events_by_aggregate(
        self, aggregate_id: str, start_time: datetime, end_time: datetime
    ) -> list[Any]:
        """Get events for an aggregate within time range."""
        pass

    # Compliance and Trade methods
    @abstractmethod
    async def get_trades_by_account(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> list[Any]:
        """Get trades for an account within date range."""
        pass

    @abstractmethod
    async def get_orders_by_account(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> list[Any]:
        """Get orders for an account within date range."""
        pass

    @abstractmethod
    async def get_price_history(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[Any]:
        """Get historical price data for a symbol."""
        pass

    # Reconciliation methods
    @abstractmethod
    async def save_reconciliation_result(self, result: dict[str, Any]) -> None:
        """Save reconciliation result."""
        pass

    @abstractmethod
    async def save_reconciliation_report(self, report: dict[str, Any]) -> None:
        """Save reconciliation report."""
        pass

    # Order methods
    @abstractmethod
    async def save_order(self, order: dict[str, Any]) -> str:
        """Save an order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> dict[str, Any] | None:
        """Get order by ID."""
        pass

    @abstractmethod
    async def get_orders_by_position(self, position_id: str) -> list[dict[str, Any]]:
        """Get all orders for a position."""
        pass

    @abstractmethod
    async def update_order_status(
        self, order_id: str, status: str, executed_at: datetime | None = None
    ) -> None:
        """Update order status."""
        pass

    # Position recovery methods
    @abstractmethod
    async def load_open_positions(self, account_id: str) -> list[Position]:
        """Load all open positions for recovery on startup."""
        pass

    @abstractmethod
    async def reconcile_positions(
        self, exchange_positions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Reconcile database positions with exchange state."""
        pass

    # Backup and restore methods
    @abstractmethod
    async def backup(self, backup_path: Path | None = None) -> Path:
        """Create a backup of the database."""
        pass

    @abstractmethod
    async def restore(self, backup_path: Path) -> None:
        """Restore database from backup."""
        pass

    @abstractmethod
    async def get_database_size(self) -> int:
        """Get current database size in bytes."""
        pass

    @abstractmethod
    async def rotate_database(self) -> None:
        """Rotate database when size limit reached."""
        pass

    # Export methods
    @abstractmethod
    async def export_trades_to_csv(
        self, account_id: str, start_date: date, end_date: date, output_path: Path
    ) -> Path:
        """Export trades to CSV for tax reporting."""
        pass

    @abstractmethod
    async def export_performance_report(
        self, account_id: str, output_path: Path
    ) -> Path:
        """Export performance metrics report."""
        pass

    # Performance metrics methods
    @abstractmethod
    async def calculate_performance_metrics(
        self, account_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        """Calculate performance metrics (win rate, average R, etc.)."""
        pass

    @abstractmethod
    async def get_performance_report(
        self, account_id: str, start_date: date, end_date: date
    ) -> dict[str, Any]:
        """Get comprehensive performance report."""
        pass

    # Tilt event methods
    @abstractmethod
    async def save_tilt_event(
        self,
        session_id: str,
        event_type: str,
        severity: str,
        indicator_values: dict[str, Any],
        intervention: str | None = None,
    ) -> str:
        """Save a tilt event."""
        pass

    @abstractmethod
    async def get_tilt_events(self, session_id: str) -> list[dict[str, Any]]:
        """Get all tilt events for a session."""
        pass

    # Transaction methods
    @abstractmethod
    async def begin_transaction(self) -> None:
        """Begin a database transaction."""
        pass

    @abstractmethod
    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        pass

    # Database management
    @abstractmethod
    async def set_database_info(self, key: str, value: str) -> None:
        """Set database metadata."""
        pass

    @abstractmethod
    async def get_database_info(self, key: str) -> str | None:
        """Get database metadata."""
        pass
