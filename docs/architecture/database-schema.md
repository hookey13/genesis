# Database Schema

## Phase 1: SQLite Schema (MVP - $500-$2k)

```sql
-- Enable foreign keys and WAL mode for SQLite
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version_id INTEGER PRIMARY KEY,
    migration_name TEXT NOT NULL,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    checksum TEXT NOT NULL,
    rollback_sql TEXT
);

-- Immutable event store for event sourcing and forensics
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,  -- UUID
    event_type TEXT NOT NULL,
    aggregate_id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    event_data TEXT NOT NULL,  -- JSON
    event_metadata TEXT,  -- JSON context
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sequence_number INTEGER NOT NULL UNIQUE
);

-- Indexes for event replay and forensics
CREATE INDEX idx_events_aggregate ON events(aggregate_id, sequence_number);
CREATE INDEX idx_events_type ON events(event_type, created_at);
CREATE INDEX idx_events_created ON events(created_at);

-- Account and tier management
CREATE TABLE IF NOT EXISTS accounts (
    account_id TEXT PRIMARY KEY,
    balance DECIMAL(20,8) NOT NULL CHECK (balance >= 0),
    tier TEXT NOT NULL CHECK (tier IN ('SNIPER', 'HUNTER', 'STRATEGIST', 'ARCHITECT')),
    tier_started_at TIMESTAMP NOT NULL,
    gates_passed TEXT NOT NULL DEFAULT '[]',  -- JSON array
    locked_features TEXT NOT NULL DEFAULT '[]',  -- JSON array
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Positions with stop loss and close tracking
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    quantity DECIMAL(20,8) NOT NULL,
    dollar_value DECIMAL(20,8) NOT NULL,
    stop_loss DECIMAL(20,8),
    pnl_dollars DECIMAL(20,8) NOT NULL DEFAULT 0,
    pnl_percent DECIMAL(10,4) NOT NULL DEFAULT 0,
    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    close_reason TEXT,  -- stop_loss|take_profit|manual|tilt_intervention|emergency
    priority_score INTEGER DEFAULT 0,  -- For emergency close prioritization
    CONSTRAINT valid_dates CHECK (closed_at IS NULL OR closed_at >= opened_at)
);

CREATE INDEX idx_positions_account ON positions(account_id, opened_at);
CREATE INDEX idx_positions_open ON positions(account_id) WHERE closed_at IS NULL;
CREATE INDEX idx_positions_symbol ON positions(symbol, opened_at);
CREATE INDEX idx_positions_priority ON positions(priority_score DESC) WHERE closed_at IS NULL;

-- Orders with idempotency and slicing support
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    position_id TEXT REFERENCES positions(position_id),
    client_order_id TEXT NOT NULL UNIQUE,  -- Idempotency key
    exchange_order_id TEXT UNIQUE,
    type TEXT NOT NULL CHECK (type IN ('MARKET', 'LIMIT', 'STOP_LOSS')),
    side TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    price DECIMAL(20,8),
    quantity DECIMAL(20,8) NOT NULL,
    filled_quantity DECIMAL(20,8) NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'PARTIAL', 'FILLED', 'CANCELLED', 'FAILED')),
    slice_number INTEGER,
    total_slices INTEGER,
    latency_ms INTEGER,
    slippage_percent DECIMAL(10,4),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP
);

CREATE INDEX idx_orders_position ON orders(position_id);
CREATE INDEX idx_orders_status ON orders(status, created_at);
CREATE INDEX idx_orders_client ON orders(client_order_id);

-- Tilt detection and behavioral tracking
CREATE TABLE IF NOT EXISTS tilt_profiles (
    profile_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL UNIQUE REFERENCES accounts(account_id),
    baseline_trades_per_hour DECIMAL(10,2),
    baseline_click_latency_ms INTEGER,
    baseline_cancel_rate DECIMAL(10,4),
    current_tilt_score INTEGER NOT NULL DEFAULT 0 CHECK (current_tilt_score BETWEEN 0 AND 100),
    tilt_level TEXT NOT NULL DEFAULT 'NORMAL' CHECK (tilt_level IN ('NORMAL', 'CAUTION', 'WARNING', 'LOCKED')),
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    last_intervention_at TIMESTAMP,
    recovery_required BOOLEAN NOT NULL DEFAULT FALSE,
    journal_entries_required INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Tilt intervention events
CREATE TABLE IF NOT EXISTS tilt_events (
    event_id TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL REFERENCES tilt_profiles(profile_id),
    position_id TEXT REFERENCES positions(position_id),
    event_type TEXT NOT NULL,
    tilt_indicators TEXT NOT NULL,  -- JSON array
    tilt_score_before INTEGER NOT NULL,
    tilt_score_after INTEGER NOT NULL,
    intervention_message TEXT,
    trader_response TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_tilt_events_profile ON tilt_events(profile_id, created_at);

-- Market state tracking
CREATE TABLE IF NOT EXISTS market_states (
    state_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('DEAD', 'NORMAL', 'VOLATILE', 'PANIC', 'MAINTENANCE')),
    volatility_atr DECIMAL(20,8),
    spread_basis_points INTEGER,
    volume_24h DECIMAL(20,8),
    liquidity_score DECIMAL(10,4),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state_duration_seconds INTEGER
);

CREATE INDEX idx_market_states_symbol ON market_states(symbol, detected_at);
CREATE INDEX idx_market_states_current ON market_states(symbol, state_id DESC);

-- Global market conditions
CREATE TABLE IF NOT EXISTS global_market_states (
    state_id TEXT PRIMARY KEY,
    btc_price DECIMAL(20,8) NOT NULL,
    total_market_cap DECIMAL(20,2),
    fear_greed_index INTEGER CHECK (fear_greed_index BETWEEN 0 AND 100),
    correlation_spike BOOLEAN NOT NULL DEFAULT FALSE,
    state TEXT NOT NULL CHECK (state IN ('BULL', 'BEAR', 'CRAB', 'CRASH', 'RECOVERY')),
    vix_crypto DECIMAL(10,4),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_global_states_time ON global_market_states(detected_at);

-- Position correlation tracking
CREATE TABLE IF NOT EXISTS position_correlations (
    correlation_id TEXT PRIMARY KEY,
    position_1_id TEXT NOT NULL REFERENCES positions(position_id),
    position_2_id TEXT NOT NULL REFERENCES positions(position_id),
    correlation_coefficient DECIMAL(5,4) CHECK (correlation_coefficient BETWEEN -1 AND 1),
    calculation_window INTEGER NOT NULL,  -- minutes
    last_calculated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    alert_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT different_positions CHECK (position_1_id != position_2_id),
    CONSTRAINT ordered_positions CHECK (position_1_id < position_2_id)  -- Prevent duplicates
);

CREATE INDEX idx_correlations_positions ON position_correlations(position_1_id, position_2_id);
CREATE INDEX idx_correlations_alert ON position_correlations(alert_triggered) WHERE alert_triggered = TRUE;

-- Trading sessions for grouping and analysis
CREATE TABLE IF NOT EXISTS trading_sessions (
    session_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts(account_id),
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    starting_balance DECIMAL(20,8) NOT NULL,
    ending_balance DECIMAL(20,8),
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    max_drawdown DECIMAL(20,8),
    tilt_events_count INTEGER NOT NULL DEFAULT 0,
    session_type TEXT NOT NULL CHECK (session_type IN ('NORMAL', 'RECOVERY', 'PAPER'))
);

CREATE INDEX idx_sessions_account ON trading_sessions(account_id, started_at);

-- Bidirectional correlation view
CREATE VIEW position_correlations_bidirectional AS
SELECT 
    correlation_id,
    position_1_id as position_a,
    position_2_id as position_b,
    correlation_coefficient,
    last_calculated
FROM position_correlations
UNION ALL
SELECT 
    correlation_id,
    position_2_id as position_a,
    position_1_id as position_b,
    correlation_coefficient,
    last_calculated
FROM position_correlations;
```

## Phase 2: PostgreSQL Schema ($2k+)

The PostgreSQL schema adds performance optimizations and advanced features. See architecture document for migration strategy and advanced indexes.
