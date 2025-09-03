"""Core constants and enums for Project GENESIS."""

from decimal import Decimal
from enum import Enum

# Decimal precision for financial calculations
DECIMAL_PRECISION = Decimal('0.00000001')  # 8 decimal places for crypto


class TradingTier(Enum):
    """Trading tier levels with capital ranges."""

    SNIPER = "SNIPER"  # $500 - $2,000
    HUNTER = "HUNTER"  # $2,000 - $10,000
    STRATEGIST = "STRATEGIST"  # $10,000 - $50,000
    ARCHITECT = "ARCHITECT"  # $50,000+


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(Enum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"


class TimeInForce(Enum):
    """Time in force for orders."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    POST_ONLY = "POST_ONLY"  # Maker only


class ConvictionLevel(Enum):
    """Trade conviction level for position sizing overrides."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
