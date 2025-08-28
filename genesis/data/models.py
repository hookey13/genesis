"""
SQLAlchemy database models for Project GENESIS.

All financial values use NUMERIC for exact decimal precision.
All timestamps are UTC. Enums use CHECK constraints for portability.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    NUMERIC,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects import postgresql, sqlite

Base = declarative_base()


# Custom types for cross-database compatibility
def get_uuid_column():
    """Get UUID column that works for both SQLite and PostgreSQL."""
    return Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))


def get_decimal_column(precision=38, scale=18, nullable=True):
    """Get decimal column for exact monetary values."""
    return Column(NUMERIC(precision, scale), nullable=nullable)


def get_timestamp_column(nullable=False, index=False):
    """Get timestamp column with UTC timezone awareness."""
    col = Column(
        DateTime(timezone=True),
        nullable=nullable,
        default=lambda: datetime.now(timezone.utc),
    )
    if index:
        col.index = True
    return col


# === Core Trading Models ===


class Session(Base):
    """Trading session for audit and lifecycle tracking."""

    __tablename__ = "sessions"

    id = get_uuid_column()
    started_at = get_timestamp_column(index=True)
    ended_at = get_timestamp_column(nullable=True)
    state = Column(String(20), nullable=False, default="running")
    reason = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    orders = relationship("Order", back_populates="session")
    pnl_entries = relationship("PnLLedger", back_populates="session")

    __table_args__ = (
        CheckConstraint(
            "state IN ('running', 'paused', 'shutdown')", name="check_session_state"
        ),
        Index("idx_sessions_state", "state"),
    )


class Instrument(Base):
    """Exchange instrument/symbol configuration."""

    __tablename__ = "instruments"

    symbol = Column(String(20), primary_key=True)
    base = Column(String(10), nullable=False)
    quote = Column(String(10), nullable=False)
    tick_size = get_decimal_column(nullable=False)
    lot_step = get_decimal_column(nullable=False)
    min_notional = get_decimal_column(nullable=False)
    updated_at = get_timestamp_column()

    # Relationships
    orders = relationship("Order", back_populates="instrument")
    trades = relationship("Trade", back_populates="instrument")


class Order(Base):
    """Order record with complete lifecycle tracking."""

    __tablename__ = "orders"

    id = get_uuid_column()
    client_order_id = Column(String(64), unique=True, nullable=False)
    exchange_order_id = Column(String(64), unique=True, nullable=True)

    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    symbol = Column(String(20), ForeignKey("instruments.symbol"), nullable=False)

    side = Column(String(10), nullable=False)
    type = Column(String(10), nullable=False)
    status = Column(String(20), nullable=False, default="new")

    qty = get_decimal_column(nullable=False)
    filled_qty = get_decimal_column(nullable=False)
    price = get_decimal_column(nullable=True)  # Null for market orders
    avg_price = get_decimal_column(nullable=True)

    time_in_force = Column(String(10), nullable=True)
    created_at = get_timestamp_column()
    updated_at = get_timestamp_column()
    last_error = Column(Text, nullable=True)

    # Relationships
    session = relationship("Session", back_populates="orders")
    instrument = relationship("Instrument", back_populates="orders")
    trades = relationship("Trade", back_populates="order")

    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="check_order_side"),
        CheckConstraint("type IN ('limit', 'market')", name="check_order_type"),
        CheckConstraint(
            "status IN ('new', 'partially_filled', 'filled', 'cancelled', 'rejected', 'expired')",
            name="check_order_status",
        ),
        CheckConstraint("qty > 0", name="check_order_qty_positive"),
        CheckConstraint("filled_qty >= 0", name="check_filled_qty_non_negative"),
        CheckConstraint("filled_qty <= qty", name="check_filled_qty_not_exceeding"),
        Index("idx_orders_symbol_status", "symbol", "status"),
        Index("idx_orders_session", "session_id"),
    )


class Trade(Base):
    """Individual trade/fill record."""

    __tablename__ = "trades"

    id = get_uuid_column()
    exchange_trade_id = Column(String(64), unique=True, nullable=False)

    order_id = Column(String(36), ForeignKey("orders.id"), nullable=False)
    symbol = Column(String(20), ForeignKey("instruments.symbol"), nullable=False)

    side = Column(String(10), nullable=False)
    qty = get_decimal_column(nullable=False)
    price = get_decimal_column(nullable=False)

    fee_ccy = Column(String(10), nullable=False, default="USDT")
    fee_amount = get_decimal_column(nullable=False)

    trade_time = get_timestamp_column(index=True)

    # Relationships
    order = relationship("Order", back_populates="trades")
    instrument = relationship("Instrument", back_populates="trades")

    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="check_trade_side"),
        CheckConstraint("qty > 0", name="check_trade_qty_positive"),
        CheckConstraint("price > 0", name="check_trade_price_positive"),
        CheckConstraint("fee_amount >= 0", name="check_fee_non_negative"),
        Index("idx_trades_order", "order_id"),
        Index("idx_trades_symbol_time", "symbol", "trade_time"),
    )


class Position(Base):
    """Current position state by symbol."""

    __tablename__ = "positions"

    symbol = Column(String(20), primary_key=True)
    qty = get_decimal_column(nullable=False)
    avg_entry_price = get_decimal_column(nullable=False)
    realised_pnl = get_decimal_column(nullable=False)
    updated_at = get_timestamp_column()

    __table_args__ = (
        CheckConstraint("avg_entry_price >= 0", name="check_avg_price_non_negative"),
    )


class PnLLedger(Base):
    """PnL event ledger for audit trail."""

    __tablename__ = "pnl_ledger"

    id = get_uuid_column()
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)

    event_type = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=True)
    amount_quote = get_decimal_column(nullable=False)

    at = get_timestamp_column(index=True)
    ref_type = Column(String(20), nullable=True)
    ref_id = Column(String(36), nullable=True)

    # Relationships
    session = relationship("Session", back_populates="pnl_entries")

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('trade', 'fee', 'adjustment', 'funding')",
            name="check_pnl_event_type",
        ),
        CheckConstraint(
            "ref_type IN ('order', 'trade') OR ref_type IS NULL",
            name="check_pnl_ref_type",
        ),
        Index("idx_pnl_session_time", "session_id", "at"),
    )


# === Market Data Models ===


class Candle(Base):
    """OHLCV candle data for charting and analytics."""

    __tablename__ = "candles"

    symbol = Column(String(20), nullable=False, primary_key=True)
    timeframe = Column(String(5), nullable=False, primary_key=True)
    open_time = Column(DateTime(timezone=True), nullable=False, primary_key=True)

    open = get_decimal_column(nullable=False)
    high = get_decimal_column(nullable=False)
    low = get_decimal_column(nullable=False)
    close = get_decimal_column(nullable=False)
    volume = get_decimal_column(nullable=False)

    __table_args__ = (
        CheckConstraint(
            "timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d')",
            name="check_candle_timeframe",
        ),
        CheckConstraint("high >= low", name="check_high_gte_low"),
        CheckConstraint("high >= open", name="check_high_gte_open"),
        CheckConstraint("high >= close", name="check_high_gte_close"),
        CheckConstraint("low <= open", name="check_low_lte_open"),
        CheckConstraint("low <= close", name="check_low_lte_close"),
        CheckConstraint("volume >= 0", name="check_volume_non_negative"),
        Index("idx_candles_lookup", "symbol", "timeframe", "open_time"),
    )


# === Operational Models ===


class Journal(Base):
    """System journal for audit and debugging."""

    __tablename__ = "journal"

    id = get_uuid_column()
    at = get_timestamp_column(index=True)
    level = Column(String(10), nullable=False)
    category = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    ctx = Column(JSON, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "level IN ('debug', 'info', 'warn', 'error', 'critical')",
            name="check_journal_level",
        ),
    )


class OutboxEvent(Base):
    """Outbox for reliable event publishing."""

    __tablename__ = "outbox_events"

    id = get_uuid_column()
    created_at = get_timestamp_column()
    event_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=False)
    processed_at = get_timestamp_column(nullable=True)
    retries = Column(Integer, nullable=False, default=0)

    __table_args__ = (Index("idx_outbox_unprocessed", "processed_at"),)


class InboxEvent(Base):
    """Inbox for idempotent event consumption."""

    __tablename__ = "inbox_events"

    id = get_uuid_column()
    dedupe_key = Column(String(128), unique=True, nullable=False)
    created_at = get_timestamp_column()
    event_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=False)
    consumed_at = get_timestamp_column(nullable=True)

    __table_args__ = (Index("idx_inbox_unconsumed", "consumed_at"),)


# SQLite-specific configuration
def configure_sqlite_pragmas(connection):
    """Configure SQLite for production use."""
    if "sqlite" in str(connection.engine.url):
        connection.execute(text("PRAGMA journal_mode = WAL"))
        connection.execute(text("PRAGMA foreign_keys = ON"))
        connection.execute(text("PRAGMA synchronous = NORMAL"))
        connection.execute(text("PRAGMA cache_size = -64000"))  # 64MB cache
        connection.execute(text("PRAGMA temp_store = MEMORY"))
