# Data Models

The data models represent the heart of GENESIS's domain logic. Each model enforces business rules that prevent emotional trading and ensure tier-appropriate behavior.

## Account
**Purpose:** Tracks current capital, tier status, and progression gates

**Key Attributes:**
- account_id: UUID - Unique identifier
- balance: Decimal - Current total capital (USDT)
- tier: Enum[SNIPER|HUNTER|STRATEGIST|ARCHITECT] - Current tier (locked by state machine)
- tier_started_at: DateTime - When entered current tier
- gates_passed: JSON - Array of passed gate requirements
- locked_features: JSON - Features disabled at current tier
- created_at: DateTime - Account creation timestamp

**Relationships:**
- Has many Positions
- Has many TradingSession
- Has one TiltProfile

## Position
**Purpose:** Represents an active or historical trading position

**Key Attributes:**
- position_id: UUID - Unique identifier
- account_id: UUID - Link to Account
- symbol: String - Trading pair (e.g., "BTC/USDT")
- side: Enum[LONG|SHORT] - Position direction
- entry_price: Decimal - Average entry price
- current_price: Decimal - Latest market price
- quantity: Decimal - Position size in base currency
- dollar_value: Decimal - Position value in USDT
- stop_loss: Decimal - Stop loss price
- pnl_dollars: Decimal - Unrealized P&L in dollars
- pnl_percent: Decimal - Unrealized P&L percentage
- opened_at: DateTime - Position open time
- closed_at: DateTime - Position close time (null if open)
- close_reason: String - Why position closed (stop_loss|take_profit|manual|tilt_intervention)

**Relationships:**
- Belongs to Account
- Has many Orders
- Has many TiltEvents (interventions on this position)

## Order
**Purpose:** Individual order execution record with slicing support

**Key Attributes:**
- order_id: UUID - Unique identifier  
- position_id: UUID - Parent position
- exchange_order_id: String - Binance order ID
- type: Enum[MARKET|LIMIT|STOP_LOSS] - Order type
- side: Enum[BUY|SELL] - Order side
- price: Decimal - Execution price (null for market)
- quantity: Decimal - Order quantity
- filled_quantity: Decimal - Actually filled amount
- status: Enum[PENDING|PARTIAL|FILLED|CANCELLED] - Current status
- slice_number: Integer - Which slice (1 of N) for iceberg orders
- total_slices: Integer - Total slices for this execution
- latency_ms: Integer - Execution latency
- slippage_percent: Decimal - Difference from expected price
- created_at: DateTime - Order creation
- executed_at: DateTime - Actual execution time

**Relationships:**
- Belongs to Position
- Has many ExecutionEvents

## TiltProfile  
**Purpose:** Tracks behavioral baseline and current psychological state

**Key Attributes:**
- profile_id: UUID - Unique identifier
- account_id: UUID - Link to Account
- baseline_trades_per_hour: Decimal - Normal trading frequency
- baseline_click_latency_ms: Integer - Normal decision speed
- baseline_cancel_rate: Decimal - Normal order cancellation rate
- current_tilt_score: Integer - Current tilt level (0-100)
- tilt_level: Enum[NORMAL|CAUTION|WARNING|LOCKED] - Intervention level
- consecutive_losses: Integer - Loss streak counter
- last_intervention_at: DateTime - Last tilt intervention
- recovery_required: Boolean - Whether recovery protocol active
- journal_entries_required: Integer - Outstanding journal entries

**Relationships:**
- Belongs to Account
- Has many TiltEvents
- Has many BehavioralMetrics

## TiltEvent
**Purpose:** Records psychological interventions and their triggers

**Key Attributes:**
- event_id: UUID - Unique identifier
- profile_id: UUID - Link to TiltProfile
- position_id: UUID - Related position (if applicable)
- event_type: String - Type of intervention
- tilt_indicators: JSON - Array of triggered indicators
- tilt_score_before: Integer - Score before intervention
- tilt_score_after: Integer - Score after intervention
- intervention_message: Text - Message shown to trader
- trader_response: Text - How trader responded
- created_at: DateTime - Event timestamp

**Relationships:**
- Belongs to TiltProfile
- May belong to Position

## MarketState
**Purpose:** Tracks market regime and conditions

**Key Attributes:**
- state_id: UUID - Unique identifier
- symbol: String - Trading pair
- state: Enum[DEAD|NORMAL|VOLATILE|PANIC|MAINTENANCE] - Current classification
- volatility_atr: Decimal - ATR-based volatility
- spread_basis_points: Integer - Current spread in bps
- volume_24h: Decimal - 24-hour volume
- liquidity_score: Decimal - Depth-based liquidity metric
- detected_at: DateTime - When state detected
- state_duration_seconds: Integer - How long in this state

**Relationships:**
- Has many Positions (opened during this state)
- Has many MarketEvents

## TradingSession
**Purpose:** Groups trades within a session for analysis

**Key Attributes:**
- session_id: UUID - Unique identifier
- account_id: UUID - Link to Account
- started_at: DateTime - Session start
- ended_at: DateTime - Session end
- starting_balance: Decimal - Balance at start
- ending_balance: Decimal - Balance at end
- total_trades: Integer - Number of trades
- winning_trades: Integer - Profitable trades
- losing_trades: Integer - Loss-making trades
- max_drawdown: Decimal - Largest drawdown in session
- tilt_events_count: Integer - Number of interventions
- session_type: Enum[NORMAL|RECOVERY|PAPER] - Session context

**Relationships:**
- Belongs to Account
- Has many Positions
- Has many TiltEvents

## Event (Audit Trail)
**Purpose:** Immutable event log for event sourcing and forensic analysis

**Key Attributes:**
- event_id: UUID - Unique identifier
- event_type: String - Event name (e.g., "OrderPlaced", "TiltDetected", "TierTransition")
- aggregate_id: UUID - ID of affected entity
- aggregate_type: String - Type of entity (Account, Position, etc.)
- event_data: JSON - Full event payload
- event_metadata: JSON - Context (user action, system trigger, etc.)
- created_at: DateTime - Event timestamp (immutable)
- sequence_number: BigInt - Global sequence for ordering

**Relationships:**
- Polymorphic - can relate to any entity
- Never updated, only inserted

## PositionCorrelation
**Purpose:** Tracks correlation between active positions for risk management

**Key Attributes:**
- correlation_id: UUID - Unique identifier
- position_1_id: UUID - First position
- position_2_id: UUID - Second position
- correlation_coefficient: Decimal - Current correlation (-1 to 1)
- calculation_window: Integer - Minutes used for calculation
- last_calculated: DateTime - When last updated
- alert_triggered: Boolean - Whether >60% threshold hit

**Relationships:**
- References two Positions
- Belongs to TradingSession

## GlobalMarketState
**Purpose:** Overall market regime affecting all trading

**Key Attributes:**
- state_id: UUID - Unique identifier
- btc_price: Decimal - Bitcoin price (market leader)
- total_market_cap: Decimal - Overall crypto market cap
- fear_greed_index: Integer - Market sentiment (0-100)
- correlation_spike: Boolean - Whether correlations >80%
- state: Enum[BULL|BEAR|CRAB|CRASH|RECOVERY] - Market regime
- vix_crypto: Decimal - Crypto volatility index
- detected_at: DateTime - When state detected

**Relationships:**
- Has many MarketStates (per-symbol states during this regime)
- Has many Positions (opened during this regime)

## SchemaVersion
**Purpose:** Track database migrations for SQLite→PostgreSQL transition

**Key Attributes:**
- version_id: Integer - Sequential version number
- migration_name: String - Descriptive name
- applied_at: DateTime - When migration ran
- checksum: String - Migration file hash
- rollback_sql: Text - SQL to reverse this migration

**Relationships:**
- Standalone migration tracking
