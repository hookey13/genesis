"""
SQLAlchemy database models for Project GENESIS.

This module defines the database schema using SQLAlchemy ORM
for persistence of positions, accounts, and trading sessions.
All financial values are stored as strings for Decimal precision.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class EventDB(Base):
    """Event store for immutable audit trail."""

    __tablename__ = "events"

    event_id = Column(String(36), primary_key=True)
    event_type = Column(String(50), nullable=False)
    aggregate_id = Column(String(36), nullable=False)
    event_data = Column(Text, nullable=False)  # JSON string
    sequence_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("aggregate_id", "sequence_number"),
        Index("idx_events_aggregate", "aggregate_id", "sequence_number"),
        Index("idx_events_type_time", "event_type", "created_at"),
    )


class AccountDB(Base):
    """Account database model."""

    __tablename__ = "accounts"

    account_id = Column(String(36), primary_key=True)
    balance_usdt = Column(String, nullable=False)  # Decimal as string
    tier = Column(SQLEnum("SNIPER", "HUNTER", "STRATEGIST", "ARCHITECT", name="trading_tier"),
                  nullable=False, default="SNIPER")
    locked_features = Column(JSON, default=list)
    last_sync = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    # Relationships
    positions = relationship("PositionDB", back_populates="account", cascade="all, delete-orphan")
    sessions = relationship("TradingSessionDB", back_populates="account", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("balance_usdt >= 0", name="check_balance_non_negative"),
        Index("idx_account_tier", "tier"),
    )


class PositionDB(Base):
    """Position database model."""

    __tablename__ = "positions"

    position_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum("LONG", "SHORT", name="position_side"), nullable=False)
    entry_price = Column(String, nullable=False)  # Decimal as string
    current_price = Column(String, nullable=True)  # Decimal as string
    quantity = Column(String, nullable=False)  # Decimal as string
    dollar_value = Column(String, nullable=False)  # Decimal as string
    stop_loss = Column(String, nullable=True)  # Decimal as string
    take_profit = Column(String, nullable=True)  # Decimal as string
    pnl_dollars = Column(String, nullable=False, default="0")  # Decimal as string
    pnl_percent = Column(String, nullable=False, default="0")  # Decimal as string
    priority_score = Column(Integer, default=0)
    status = Column(SQLEnum("OPEN", "CLOSED", "PENDING", name="position_status"),
                   nullable=False, default="OPEN")
    close_reason = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)

    # Relationships
    account = relationship("AccountDB", back_populates="positions")

    # Constraints
    __table_args__ = (
        CheckConstraint("side IN ('LONG', 'SHORT')", name="check_position_side"),
        CheckConstraint(
            "close_reason IN ('stop_loss', 'take_profit', 'manual', "
            "'tilt_intervention', 'emergency') OR close_reason IS NULL",
            name="check_close_reason"
        ),
        Index("idx_position_account", "account_id"),
        Index("idx_position_symbol", "symbol"),
        Index("idx_position_status", "status"),
        Index("idx_position_created", "created_at"),
    )


class TradingSessionDB(Base):
    """Trading session database model."""

    __tablename__ = "trading_sessions"

    session_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    session_date = Column(DateTime, nullable=False)
    starting_balance = Column(String, nullable=False)  # Decimal as string
    current_balance = Column(String, nullable=False)  # Decimal as string
    ending_balance = Column(String, nullable=True)  # Decimal as string
    realized_pnl = Column(String, nullable=False, default="0")  # Decimal as string
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(String, nullable=True)  # Store as string for precision  # Stored as float for easy queries
    average_r = Column(String, nullable=True)  # Decimal as string
    max_drawdown = Column(String, nullable=False, default="0")  # Decimal as string
    daily_loss_limit = Column(String, nullable=False)  # Decimal as string
    tilt_events = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)

    # Relationships
    account = relationship("AccountDB", back_populates="sessions")

    # Constraints
    __table_args__ = (
        CheckConstraint("total_trades >= 0", name="check_total_trades_non_negative"),
        CheckConstraint("winning_trades >= 0", name="check_winning_trades_non_negative"),
        CheckConstraint("losing_trades >= 0", name="check_losing_trades_non_negative"),
        Index("idx_session_account", "account_id"),
        Index("idx_session_date", "session_date"),
        Index("idx_session_active", "is_active"),
    )


class PositionCorrelationDB(Base):
    """Position correlation database model."""

    __tablename__ = "position_correlations"

    position_a_id = Column(String(36), ForeignKey("positions.position_id"), primary_key=True)
    position_b_id = Column(String(36), ForeignKey("positions.position_id"), primary_key=True)
    correlation_coefficient = Column(String, nullable=False)  # Decimal as string
    alert_triggered = Column(Boolean, nullable=False, default=False)
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "position_a_id < position_b_id",
            name="check_position_order"
        ),
        Index("idx_correlation_alert", "alert_triggered"),
    )


class OrderDB(Base):
    """Order database model for tracking executed orders."""

    __tablename__ = "orders"

    order_id = Column(String(36), primary_key=True)
    position_id = Column(String(36), ForeignKey("positions.position_id"), nullable=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    client_order_id = Column(String(36), nullable=False, unique=True)  # For idempotency
    exchange_order_id = Column(String(64), nullable=True, unique=True)
    symbol = Column(String(20), nullable=False)
    side = Column(SQLEnum("BUY", "SELL", name="order_side"), nullable=False)
    type = Column(SQLEnum("MARKET", "LIMIT", "STOP_LIMIT", name="order_type"), nullable=False)
    quantity = Column(String, nullable=False)  # Decimal as string
    price = Column(String, nullable=True)  # Decimal as string
    executed_price = Column(String, nullable=True)  # Decimal as string
    executed_quantity = Column(String, nullable=False, default="0")  # Decimal as string
    status = Column(SQLEnum("PENDING", "FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED",
                           name="order_status"), nullable=False, default="PENDING")
    slice_number = Column(Integer, nullable=True)
    total_slices = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    slippage_percent = Column(String, nullable=True)  # Decimal as string
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

    # Constraints
    __table_args__ = (
        Index("idx_order_account", "account_id"),
        Index("idx_order_position", "position_id"),
        Index("idx_order_symbol", "symbol"),
        Index("idx_order_status", "status"),
        Index("idx_order_created", "created_at"),
    )


class RiskMetricsDB(Base):
    """Risk metrics snapshot database model."""

    __tablename__ = "risk_metrics"

    metric_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_exposure = Column(String, nullable=False)  # Decimal as string
    position_count = Column(Integer, nullable=False)
    total_pnl_dollars = Column(String, nullable=False)  # Decimal as string
    total_pnl_percent = Column(String, nullable=False)  # Decimal as string
    max_position_size = Column(String, nullable=False)  # Decimal as string
    daily_pnl = Column(String, nullable=False)  # Decimal as string
    risk_score = Column(String, nullable=True)  # Decimal as string

    # Constraints
    __table_args__ = (
        CheckConstraint("position_count >= 0", name="check_position_count_non_negative"),
        Index("idx_risk_account", "account_id"),
        Index("idx_risk_timestamp", "timestamp"),
    )


class TiltEventDB(Base):
    """Tilt event model for behavioral monitoring."""

    __tablename__ = "tilt_events"

    event_id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("trading_sessions.session_id"), nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    indicator_values = Column(Text, nullable=False)  # JSON string
    intervention_taken = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')"),
        Index("idx_tilt_session", "session_id"),
        Index("idx_tilt_created", "created_at"),
    )


class DatabaseInfo(Base):
    """Database metadata for tracking versions and migrations."""

    __tablename__ = "database_info"

    key = Column(String(50), primary_key=True)
    value = Column(String(255), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
