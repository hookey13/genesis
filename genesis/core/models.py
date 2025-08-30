"""
Core domain models for Project GENESIS.

This module contains all domain models used throughout the application,
focusing on trading positions, accounts, and sessions.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# Import TradingTier from constants to avoid duplication
from genesis.core.constants import ConvictionLevel, TradingTier


class Symbol(str):
    """Trading symbol representation."""

    def __new__(cls, value):
        """Create Symbol instance ensuring proper format."""
        return str.__new__(cls, value.upper())

    @property
    def value(self):
        """Get string value of symbol."""
        return str(self)


class Side(str, Enum):
    """Trade/Order side."""

    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class TaxMethod(str, Enum):
    """Tax lot accounting methods."""

    FIFO = "FIFO"  # First In First Out
    LIFO = "LIFO"  # Last In First Out
    HIFO = "HIFO"  # Highest In First Out
    SPECIFIC_LOT = "SPECIFIC_LOT"  # Specific lot selection


class Position(BaseModel):
    """Trading position domain model."""

    position_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    symbol: str
    side: PositionSide
    entry_price: Decimal
    current_price: Decimal | None = None
    quantity: Decimal
    dollar_value: Decimal
    stop_loss: Decimal | None = None
    pnl_dollars: Decimal = Decimal("0")
    pnl_percent: Decimal = Decimal("0")
    priority_score: int = 0
    close_reason: str | None = (
        None  # e.g., "stop_loss", "take_profit", "manual", "tilt_intervention"
    )
    # Tax lot tracking fields
    tax_lot_id: str | None = Field(default=None)
    acquisition_date: datetime | None = None  # For holding period calculations
    cost_basis: Decimal | None = None  # Separate from entry_price for tax purposes
    tax_method: TaxMethod = TaxMethod.FIFO
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None

    @field_validator(
        "entry_price",
        "current_price",
        "quantity",
        "dollar_value",
        "stop_loss",
        "pnl_dollars",
        "pnl_percent",
        "cost_basis",
        mode="before",
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v

    def update_pnl(self, current_price: Decimal) -> None:
        """Update P&L based on current price."""
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            price_change = current_price - self.entry_price
        else:  # SHORT
            price_change = self.entry_price - current_price

        self.pnl_dollars = price_change * self.quantity
        self.pnl_percent = (price_change / self.entry_price) * Decimal("100")
        self.updated_at = datetime.now()


class AccountType(str, Enum):
    """Account type classification."""

    MASTER = "MASTER"
    SUB = "SUB"
    PAPER = "PAPER"


class Account(BaseModel):
    """Trading account domain model."""

    account_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_account_id: str | None = None  # For sub-account hierarchy
    account_type: AccountType = AccountType.MASTER
    balance_usdt: Decimal
    tier: TradingTier = TradingTier.SNIPER
    locked_features: list[str] = Field(default_factory=list)
    permissions: dict = Field(default_factory=dict)  # Feature access control
    compliance_settings: dict = Field(default_factory=dict)  # Regulatory requirements
    last_sync: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("balance_usdt", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))

    @field_validator("balance_usdt")
    @classmethod
    def validate_positive_balance(cls, v):
        """Ensure balance is non-negative."""
        if v < 0:
            raise ValueError("Account balance cannot be negative")
        return v


class TradingSession(BaseModel):
    """Trading session for daily tracking."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    account_id: str
    session_date: datetime = Field(default_factory=datetime.now)
    starting_balance: Decimal
    current_balance: Decimal
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: Decimal = Decimal("0")
    daily_loss_limit: Decimal
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime | None = None

    @field_validator(
        "starting_balance",
        "current_balance",
        "realized_pnl",
        "unrealized_pnl",
        "max_drawdown",
        "daily_loss_limit",
        mode="before",
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))

    def update_trade_result(self, pnl: Decimal) -> None:
        """Update session with trade result."""
        self.total_trades += 1
        self.realized_pnl += pnl
        self.current_balance += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Update max drawdown
        drawdown = self.starting_balance - self.current_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        self.updated_at = datetime.now()

    def is_daily_limit_reached(self) -> bool:
        """Check if daily loss limit has been reached."""
        return self.realized_pnl <= -self.daily_loss_limit


class PositionCorrelation(BaseModel):
    """Position correlation tracking."""

    position_a_id: str
    position_b_id: str
    correlation_coefficient: Decimal
    alert_triggered: bool = False
    calculated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("correlation_coefficient", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))

    @field_validator("correlation_coefficient")
    @classmethod
    def validate_correlation_range(cls, v):
        """Ensure correlation is between -1 and 1."""
        if not -1 <= v <= 1:
            raise ValueError("Correlation coefficient must be between -1 and 1")
        return v


class OrderType(str, Enum):
    """Order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    POST_ONLY = "POST_ONLY"  # Maker-only order
    LIMIT_MAKER = "LIMIT_MAKER"  # Binance-specific post-only


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status states."""

    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class Order(BaseModel):
    """Order domain model."""

    order_id: str = Field(default_factory=lambda: str(uuid4()))
    position_id: str | None = None
    client_order_id: str = Field(default_factory=lambda: str(uuid4()))
    exchange_order_id: str | None = None
    symbol: str
    type: OrderType
    side: OrderSide
    price: Decimal | None = None
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    status: OrderStatus = OrderStatus.PENDING
    conviction_level: ConvictionLevel = ConvictionLevel.MEDIUM
    slice_number: int | None = None
    total_slices: int | None = None
    latency_ms: int | None = None
    slippage_percent: Decimal | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    executed_at: datetime | None = None
    routing_method: str | None = None
    maker_fee_paid: Decimal | None = None
    taker_fee_paid: Decimal | None = None
    execution_score: float | None = None
    # Tax lot tracking
    lot_assignments: list[dict] | None = None  # List of {lot_id, quantity, cost_basis}
    tax_lot_id: str | None = None  # For buy orders that create new lots

    @field_validator(
        "price",
        "quantity",
        "filled_quantity",
        "slippage_percent",
        "maker_fee_paid",
        "taker_fee_paid",
        mode="before",
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v


class SignalType(str, Enum):
    """Signal types for trading decisions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"


class Signal(BaseModel):
    """Trading signal model."""

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    strategy_id: str
    symbol: str
    signal_type: SignalType
    confidence: Decimal = Decimal("0.5")
    price_target: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    quantity: Decimal | None = None
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "confidence",
        "price_target",
        "stop_loss",
        "take_profit",
        "quantity",
        mode="before",
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v


class PriceData(BaseModel):
    """Market price data point."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        return Decimal(str(v))


class Trade(BaseModel):
    """Completed trade model for analytics."""

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    position_id: str | None = None
    strategy_id: str
    symbol: str
    side: OrderSide
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl_dollars: Decimal
    pnl_percent: Decimal
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator(
        "entry_price",
        "exit_price",
        "quantity",
        "pnl_dollars",
        "pnl_percent",
        mode="before",
    )
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal for precision."""
        if v is not None:
            return Decimal(str(v))
        return v
