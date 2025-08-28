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
    tier = Column(
        SQLEnum("SNIPER", "HUNTER", "STRATEGIST", "ARCHITECT", name="trading_tier"),
        nullable=False,
        default="SNIPER",
    )
    locked_features = Column(JSON, default=list)
    last_sync = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    # Relationships
    positions = relationship(
        "PositionDB", back_populates="account", cascade="all, delete-orphan"
    )
    sessions = relationship(
        "TradingSessionDB", back_populates="account", cascade="all, delete-orphan"
    )

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
    status = Column(
        SQLEnum("OPEN", "CLOSED", "PENDING", name="position_status"),
        nullable=False,
        default="OPEN",
    )
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
            name="check_close_reason",
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
    win_rate = Column(
        String, nullable=True
    )  # Store as string for precision  # Stored as float for easy queries
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
        CheckConstraint(
            "winning_trades >= 0", name="check_winning_trades_non_negative"
        ),
        CheckConstraint("losing_trades >= 0", name="check_losing_trades_non_negative"),
        Index("idx_session_account", "account_id"),
        Index("idx_session_date", "session_date"),
        Index("idx_session_active", "is_active"),
    )


class PositionCorrelationDB(Base):
    """Position correlation database model."""

    __tablename__ = "position_correlations"

    position_a_id = Column(
        String(36), ForeignKey("positions.position_id"), primary_key=True
    )
    position_b_id = Column(
        String(36), ForeignKey("positions.position_id"), primary_key=True
    )
    correlation_coefficient = Column(String, nullable=False)  # Decimal as string
    alert_triggered = Column(Boolean, nullable=False, default=False)
    calculated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Constraints
    __table_args__ = (
        CheckConstraint("position_a_id < position_b_id", name="check_position_order"),
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
    type = Column(
        SQLEnum("MARKET", "LIMIT", "STOP_LIMIT", name="order_type"), nullable=False
    )
    quantity = Column(String, nullable=False)  # Decimal as string
    price = Column(String, nullable=True)  # Decimal as string
    executed_price = Column(String, nullable=True)  # Decimal as string
    executed_quantity = Column(String, nullable=False, default="0")  # Decimal as string
    status = Column(
        SQLEnum(
            "PENDING",
            "FILLED",
            "PARTIALLY_FILLED",
            "CANCELLED",
            "REJECTED",
            name="order_status",
        ),
        nullable=False,
        default="PENDING",
    )
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
        CheckConstraint(
            "position_count >= 0", name="check_position_count_non_negative"
        ),
        Index("idx_risk_account", "account_id"),
        Index("idx_risk_timestamp", "timestamp"),
    )


class TiltProfileDB(Base):
    """Tilt profile database model for behavioral baseline tracking."""

    __tablename__ = "tilt_profiles"

    profile_id = Column(String(36), primary_key=True)
    account_id = Column(
        String(36), ForeignKey("accounts.account_id"), nullable=False, unique=True
    )
    baseline_trades_per_hour = Column(String, nullable=True)  # Decimal as string
    baseline_click_latency_ms = Column(Integer, nullable=True)
    baseline_cancel_rate = Column(String, nullable=True)  # Decimal as string
    current_tilt_score = Column(Integer, nullable=False, default=0)
    tilt_level = Column(
        SQLEnum("NORMAL", "LEVEL1", "LEVEL2", "LEVEL3", name="tilt_level"),
        nullable=False,
        default="NORMAL",
    )
    consecutive_losses = Column(Integer, nullable=False, default=0)
    last_intervention_at = Column(DateTime, nullable=True)
    recovery_required = Column(Boolean, nullable=False, default=False)
    journal_entries_required = Column(Integer, nullable=False, default=0)
    # Recovery protocol fields
    lockout_expiration = Column(DateTime, nullable=True)
    recovery_stage = Column(Integer, nullable=False, default=0)
    tilt_debt_amount = Column(String, nullable=False, default="0")  # Decimal as string
    meditation_completed = Column(Boolean, nullable=False, default=False)
    checklist_items_completed = Column(Text, nullable=True)  # JSON array
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    # Relationships
    behavioral_metrics = relationship(
        "BehavioralMetricsDB", back_populates="profile", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("current_tilt_score >= 0 AND current_tilt_score <= 100"),
        CheckConstraint("tilt_level IN ('NORMAL', 'LEVEL1', 'LEVEL2', 'LEVEL3')"),
        Index("idx_tilt_profile_account", "account_id"),
    )


class BehavioralMetricsDB(Base):
    """Behavioral metrics database model for tracking trading behavior patterns."""

    __tablename__ = "behavioral_metrics"

    metric_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    session_id = Column(
        String(36), ForeignKey("trading_sessions.session_id"), nullable=True
    )
    metric_type = Column(String(50), nullable=False)  # Extended types for Story 3.4
    metric_value = Column(String, nullable=False)  # Decimal as string
    metrics_metadata = Column(Text, nullable=True)  # JSON for additional context
    timestamp = Column(DateTime, nullable=False)
    session_context = Column(String(20), nullable=True)  # tired|alert|stressed
    time_of_day_bucket = Column(Integer, nullable=True)  # Hour (0-23)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    profile = relationship("TiltProfileDB", back_populates="behavioral_metrics")

    __table_args__ = (
        CheckConstraint(
            "metric_type IN ('click_speed', 'click_latency', 'order_frequency', 'order_modification', "
            "'position_size_variance', 'cancel_rate', 'tab_switches', 'inactivity_period', "
            "'session_duration', 'config_change', 'focus_duration', 'distraction_score')",
            name="check_metric_type_extended",
        ),
        CheckConstraint(
            "session_context IN ('tired', 'alert', 'stressed') OR session_context IS NULL"
        ),
        CheckConstraint(
            "time_of_day_bucket >= 0 AND time_of_day_bucket <= 23 OR time_of_day_bucket IS NULL"
        ),
        Index("idx_behavioral_metrics_profile", "profile_id", "timestamp"),
        Index(
            "idx_behavioral_metrics_analysis",
            "profile_id",
            "metric_type",
            "time_of_day_bucket",
        ),
        Index("idx_behavioral_metrics_session", "session_id"),
    )


class TiltEventDB(Base):
    """Tilt event model for behavioral monitoring."""

    __tablename__ = "tilt_events"

    event_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    event_type = Column(String(50), nullable=False)
    tilt_indicators = Column(Text, nullable=False)  # JSON array
    tilt_score_before = Column(Integer, nullable=False)
    tilt_score_after = Column(Integer, nullable=False)
    intervention_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("tilt_score_before >= 0 AND tilt_score_before <= 100"),
        CheckConstraint("tilt_score_after >= 0 AND tilt_score_after <= 100"),
        Index("idx_tilt_events_profile", "profile_id", "timestamp"),
        Index("idx_tilt_events_type", "event_type", "timestamp"),
    )


class MarketStateDB(Base):
    """Market state database model for tracking market conditions."""

    __tablename__ = "market_states"

    state_id = Column(String(36), primary_key=True)
    symbol = Column(String(20), nullable=False)
    state = Column(
        SQLEnum(
            "DEAD", "NORMAL", "VOLATILE", "PANIC", "MAINTENANCE", name="market_state"
        ),
        nullable=False,
    )
    volatility_atr = Column(String, nullable=True)  # Decimal as string
    spread_basis_points = Column(Integer, nullable=True)
    volume_24h = Column(String, nullable=True)  # Decimal as string
    liquidity_score = Column(String, nullable=True)  # Decimal as string
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    state_duration_seconds = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_market_states_symbol", "symbol", "detected_at"),
        Index("idx_market_states_current", "symbol", "state_id"),
        CheckConstraint(
            "state IN ('DEAD', 'NORMAL', 'VOLATILE', 'PANIC', 'MAINTENANCE')"
        ),
    )


class GlobalMarketStateDB(Base):
    """Global market state database model for overall market conditions."""

    __tablename__ = "global_market_states"

    state_id = Column(String(36), primary_key=True)
    btc_price = Column(String, nullable=False)  # Decimal as string
    total_market_cap = Column(String, nullable=True)  # Decimal as string
    fear_greed_index = Column(Integer, nullable=True)
    correlation_spike = Column(Boolean, nullable=False, default=False)
    state = Column(
        SQLEnum(
            "BULL", "BEAR", "CRAB", "CRASH", "RECOVERY", name="global_market_state"
        ),
        nullable=False,
    )
    vix_crypto = Column(String, nullable=True)  # Decimal as string
    detected_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("state IN ('BULL', 'BEAR', 'CRAB', 'CRASH', 'RECOVERY')"),
        CheckConstraint(
            "fear_greed_index >= 0 AND fear_greed_index <= 100",
            name="check_fear_greed_range",
        ),
        Index("idx_global_market_states_time", "detected_at"),
    )


class VolumeProfileDB(Base):
    """Volume profile database model for tracking volume patterns."""

    __tablename__ = "volume_profiles"

    profile_id = Column(String(36), primary_key=True)
    symbol = Column(String(20), nullable=False)
    hour = Column(Integer, nullable=False)  # 0-23
    volume = Column(String, nullable=False)  # Decimal as string
    trade_count = Column(Integer, nullable=False, default=0)
    average_trade_size = Column(String, nullable=True)  # Decimal as string
    date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("symbol", "hour", "date", name="unique_volume_profile"),
        Index("idx_volume_profile_symbol", "symbol"),
        Index("idx_volume_profile_date", "date"),
        CheckConstraint("hour >= 0 AND hour <= 23", name="check_hour_range"),
    )


class ArbitrageSignalDB(Base):
    """Arbitrage signal database model for statistical arbitrage."""

    __tablename__ = "arbitrage_signals"

    signal_id = Column(String(36), primary_key=True)
    pair1_symbol = Column(String(20), nullable=False)
    pair2_symbol = Column(String(20), nullable=False)
    zscore = Column(String, nullable=False)  # Decimal as string
    threshold_sigma = Column(Float, nullable=False)
    signal_type = Column(SQLEnum("ENTRY", "EXIT", name="signal_type"), nullable=False)
    confidence_score = Column(String, nullable=False)  # Decimal as string (0-1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("signal_type IN ('ENTRY', 'EXIT')", name="check_signal_type"),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="check_confidence_range",
        ),
        Index("idx_arbitrage_signals_pairs", "pair1_symbol", "pair2_symbol"),
        Index("idx_arbitrage_signals_created", "created_at"),
    )


class SpreadHistoryDB(Base):
    """Spread history database model for tracking spread values over time."""

    __tablename__ = "spread_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair1_symbol = Column(String(20), nullable=False)
    pair2_symbol = Column(String(20), nullable=False)
    spread_value = Column(String, nullable=False)  # Decimal as string
    recorded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_spread_history_pairs", "pair1_symbol", "pair2_symbol"),
        Index("idx_spread_history_time", "recorded_at"),
    )


class SymbolSpreadHistoryDB(Base):
    """Symbol spread history database model for tracking bid-ask spreads."""

    __tablename__ = "symbol_spread_history"

    history_id = Column(String(36), primary_key=True)
    symbol = Column(String(20), nullable=False)
    spread_bps = Column(String, nullable=False)  # Decimal as string
    bid_price = Column(String, nullable=False)  # Decimal as string
    ask_price = Column(String, nullable=False)  # Decimal as string
    bid_volume = Column(String, nullable=True)  # Decimal as string
    ask_volume = Column(String, nullable=True)  # Decimal as string
    order_imbalance = Column(String, nullable=True)  # Decimal as string
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    hour_of_day = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)

    __table_args__ = (
        Index("idx_symbol_spread_history_symbol", "symbol", "timestamp"),
        Index("idx_symbol_spread_patterns", "symbol", "hour_of_day", "day_of_week"),
        CheckConstraint(
            "hour_of_day >= 0 AND hour_of_day <= 23", name="check_spread_hour_range"
        ),
        CheckConstraint(
            "day_of_week >= 0 AND day_of_week <= 6", name="check_spread_day_range"
        ),
    )


class LiquiditySnapshotDB(Base):
    """Liquidity snapshot database model for pair liquidity metrics."""

    __tablename__ = "liquidity_snapshots"

    snapshot_id = Column(String(36), primary_key=True)
    symbol = Column(String(20), nullable=False)
    volume_24h = Column(String, nullable=False)  # Decimal as string
    liquidity_tier = Column(
        SQLEnum("LOW", "MEDIUM", "HIGH", name="liquidity_tier"), nullable=False
    )
    spread_basis_points = Column(Integer, nullable=False)
    bid_depth_10 = Column(String, nullable=False)  # Decimal as string
    ask_depth_10 = Column(String, nullable=False)  # Decimal as string
    spread_persistence_score = Column(
        String, nullable=False
    )  # Decimal as string (0-100)
    scanned_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "liquidity_tier IN ('LOW', 'MEDIUM', 'HIGH')", name="check_liquidity_tier"
        ),
        CheckConstraint("spread_basis_points >= 0", name="check_spread_non_negative"),
        Index("idx_liquidity_symbol", "symbol", "scanned_at"),
        Index("idx_liquidity_volume", "volume_24h"),
    )


class PairBlacklistDB(Base):
    """Pair blacklist database model for unprofitable pairs."""

    __tablename__ = "pair_blacklist"

    blacklist_id = Column(String(36), primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    blacklist_reason = Column(String(255), nullable=False)
    consecutive_losses = Column(Integer, nullable=False)
    blacklisted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint("consecutive_losses >= 0", name="check_losses_non_negative"),
        Index("idx_blacklist_symbol", "symbol"),
    )


class TierRecommendationDB(Base):
    """Tier recommendation database model for pair recommendations."""

    __tablename__ = "tier_recommendations"

    recommendation_id = Column(String(36), primary_key=True)
    tier = Column(
        SQLEnum("SNIPER", "HUNTER", "STRATEGIST", name="trading_tier_rec"),
        nullable=False,
    )
    symbol = Column(String(20), nullable=False)
    volume_24h = Column(String, nullable=False)  # Decimal as string
    liquidity_score = Column(String, nullable=False)  # Decimal as string
    recommended_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "tier IN ('SNIPER', 'HUNTER', 'STRATEGIST')",
            name="check_tier_recommendation",
        ),
        Index("idx_recommendations_tier", "tier", "recommended_at"),
    )


class DatabaseInfo(Base):
    """Database metadata for tracking versions and migrations."""

    __tablename__ = "database_info"

    key = Column(String(50), primary_key=True)
    value = Column(String(255), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RecoveryProtocolDB(Base):
    """Recovery protocol database model for tilt recovery management."""

    __tablename__ = "recovery_protocols"

    protocol_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    initiated_at = Column(DateTime, nullable=False)
    lockout_duration_minutes = Column(Integer, nullable=False)
    initial_debt_amount = Column(String, nullable=False)  # Decimal as string
    current_debt_amount = Column(String, nullable=False)  # Decimal as string
    recovery_stage = Column(Integer, nullable=False, default=0)
    profitable_trades_count = Column(Integer, nullable=False, default=0)
    loss_trades_count = Column(Integer, nullable=False, default=0)
    total_profit = Column(String, nullable=False, default="0")  # Decimal as string
    total_loss = Column(String, nullable=False, default="0")  # Decimal as string
    recovery_completed_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "recovery_stage >= 0 AND recovery_stage <= 3", name="check_recovery_stage"
        ),
        CheckConstraint("lockout_duration_minutes > 0", name="check_lockout_positive"),
        Index("idx_recovery_protocols_profile", "profile_id", "initiated_at"),
        Index("idx_recovery_protocols_active", "is_active", "profile_id"),
    )


class JournalEntryDB(Base):
    """Journal entry database model for tilt recovery reflection."""

    __tablename__ = "journal_entries"

    entry_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    content = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    trigger_analysis = Column(Text, nullable=True)
    prevention_plan = Column(Text, nullable=True)
    submitted_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint("word_count >= 100", name="check_word_count_minimum"),
        Index("idx_journal_entries_profile", "profile_id", "submitted_at"),
    )


class TiltDebtLedgerDB(Base):
    """Tilt debt ledger database model for tracking debt transactions."""

    __tablename__ = "tilt_debt_ledger"

    ledger_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    transaction_type = Column(
        SQLEnum("DEBT_ADDED", "DEBT_REDUCED", name="debt_transaction_type"),
        nullable=False,
    )
    amount = Column(String, nullable=False)  # Decimal as string
    balance_after = Column(String, nullable=False)  # Decimal as string
    reason = Column(String(255), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "transaction_type IN ('DEBT_ADDED', 'DEBT_REDUCED')",
            name="check_debt_transaction_type",
        ),
        Index("idx_tilt_debt_profile", "profile_id", "timestamp"),
    )


class RecoveryChecklistDB(Base):
    """Recovery checklist database model for tracking recovery tasks."""

    __tablename__ = "recovery_checklists"

    checklist_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    checklist_items = Column(Text, nullable=False)  # JSON array of items
    is_complete = Column(Boolean, nullable=False, default=False)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_recovery_checklists_profile", "profile_id", "created_at"),
        Index("idx_recovery_checklists_active", "is_complete", "profile_id"),
    )


class ConfigChangeDB(Base):
    """Configuration change database model for tracking settings changes."""

    __tablename__ = "config_changes"

    change_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    setting_name = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    changed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_config_changes_profile", "profile_id", "changed_at"),
        Index("idx_config_changes_setting", "setting_name", "changed_at"),
    )


class BehaviorCorrelationDB(Base):
    """Behavior correlation database model for P&L correlation analysis."""

    __tablename__ = "behavior_correlations"

    correlation_id = Column(String(36), primary_key=True)
    profile_id = Column(
        String(36), ForeignKey("tilt_profiles.profile_id"), nullable=False
    )
    behavior_type = Column(String(50), nullable=False)
    correlation_coefficient = Column(String, nullable=False)  # Decimal as string
    p_value = Column(String, nullable=False)  # Decimal as string
    sample_size = Column(Integer, nullable=False)
    time_window_minutes = Column(Integer, nullable=False)
    calculated_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_behavior_correlations_profile", "profile_id", "calculated_at"),
        Index(
            "idx_behavior_correlations_type", "behavior_type", "correlation_coefficient"
        ),
        CheckConstraint("p_value >= 0 AND p_value <= 1", name="check_p_value_range"),
        CheckConstraint(
            "correlation_coefficient >= -1 AND correlation_coefficient <= 1",
            name="check_correlation_range",
        ),
    )


class TierTransition(Base):
    """Tier transition tracking for valley of death protection."""

    __tablename__ = "tier_transitions"

    transition_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    from_tier = Column(String(20), nullable=False)
    to_tier = Column(String(20), nullable=False)
    readiness_score = Column(Integer, nullable=True)
    checklist_completed = Column(Boolean, default=False)
    funeral_completed = Column(Boolean, default=False)
    paper_trading_completed = Column(Boolean, default=False)
    adjustment_period_start = Column(DateTime, nullable=True)
    adjustment_period_end = Column(DateTime, nullable=True)
    transition_status = Column(String(20), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "transition_status IN ('APPROACHING', 'READY', 'IN_PROGRESS', 'COMPLETED')",
            name="check_transition_status",
        ),
        Index("idx_tier_transitions_account", "account_id", "created_at"),
    )


class PaperTradingSession(Base):
    """Paper trading session for practicing new execution methods."""

    __tablename__ = "paper_trading_sessions"

    session_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    strategy_name = Column(String(50), nullable=False)
    required_duration_hours = Column(Integer, nullable=False)
    actual_duration_hours = Column(Integer, nullable=True)
    success_rate = Column(String, nullable=True)  # Decimal as string
    total_trades = Column(Integer, default=0)
    profitable_trades = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_paper_trading_account", "account_id", "strategy_name"),
    )


class TransitionChecklist(Base):
    """Transition checklist items for psychological preparation."""

    __tablename__ = "transition_checklists"

    checklist_id = Column(String(36), primary_key=True)
    transition_id = Column(
        String(36), ForeignKey("tier_transitions.transition_id"), nullable=False
    )
    item_name = Column(String(200), nullable=False)
    item_response = Column(Text, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_transition_checklists", "transition_id", "created_at"),
    )


class HabitFuneralRecord(Base):
    """Habit funeral ceremony records."""

    __tablename__ = "habit_funeral_records"

    ceremony_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    transition_id = Column(
        String(36), ForeignKey("tier_transitions.transition_id"), nullable=False
    )
    buried_habits = Column(Text, nullable=False)  # JSON array
    commitments = Column(Text, nullable=True)  # JSON array
    certificate_hash = Column(String(64), nullable=False)
    ceremony_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_habit_funeral_account", "account_id", "ceremony_timestamp"),
    )


class AdjustmentPeriod(Base):
    """48-hour adjustment period tracking."""

    __tablename__ = "adjustment_periods"

    period_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    tier = Column(String(20), nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=False)
    current_phase = Column(String(20), nullable=False)
    position_limit_multiplier = Column(String, nullable=False)  # Decimal as string
    monitoring_sensitivity_multiplier = Column(Float, nullable=False)
    interventions_triggered = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "current_phase IN ('INITIAL', 'EARLY', 'MID', 'FINAL')",
            name="check_adjustment_phase",
        ),
        Index("idx_adjustment_period_account", "account_id", "is_active"),
    )


class Trade(Base):
    """Trade execution records."""

    __tablename__ = "trades"

    trade_id = Column(String(36), primary_key=True)
    account_id = Column(String(36), ForeignKey("accounts.account_id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    price = Column(String, nullable=False)  # Decimal as string
    quantity = Column(String, nullable=False)  # Decimal as string
    pnl = Column(String, nullable=True)  # Decimal as string
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (Index("idx_trades_account", "account_id", "executed_at"),)


class TiltProfile(Base):
    """Alias for TiltProfileDB for backward compatibility."""

    __table__ = TiltProfileDB.__table__


# Helper function placeholder
Session = None  # Will be created by get_session()


def get_session():
    """Get database session."""
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    database_url = os.getenv("DATABASE_URL", "sqlite:///genesis.db")
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
