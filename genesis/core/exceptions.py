"""
Custom exceptions for Project GENESIS.

This module contains all custom exceptions used throughout the application
for error handling and control flow.
"""

from decimal import Decimal


class GenesisException(Exception):
    """Base exception for all GENESIS exceptions."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


class RiskLimitExceeded(GenesisException):
    """Raised when a risk limit would be exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        current_value: Decimal,
        limit_value: Decimal,
    ):
        super().__init__(message, code=f"RISK_{limit_type.upper()}_EXCEEDED")
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class TierViolation(GenesisException):
    """Raised when attempting to access tier-locked features."""

    def __init__(self, message: str, required_tier: str, current_tier: str):
        super().__init__(message, code="TIER_VIOLATION")
        self.required_tier = required_tier
        self.current_tier = current_tier


class InsufficientBalance(GenesisException):
    """Raised when account balance is insufficient for operation."""

    def __init__(
        self, message: str, required_amount: Decimal, available_amount: Decimal
    ):
        super().__init__(message, code="INSUFFICIENT_BALANCE")
        self.required_amount = required_amount
        self.available_amount = available_amount


class MinimumPositionSize(GenesisException):
    """Raised when position size is below minimum."""

    def __init__(self, message: str, position_size: Decimal, minimum_size: Decimal):
        super().__init__(message, code="MINIMUM_POSITION_SIZE")
        self.position_size = position_size
        self.minimum_size = minimum_size


class DailyLossLimitReached(GenesisException):
    """Raised when daily loss limit has been reached."""

    def __init__(self, message: str, current_loss: Decimal, daily_limit: Decimal):
        super().__init__(message, code="DAILY_LOSS_LIMIT")
        self.current_loss = current_loss
        self.daily_limit = daily_limit


class ConfigurationError(GenesisException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message, code="CONFIGURATION_ERROR")
        self.config_key = config_key


class ExchangeError(GenesisException):
    """Base class for exchange-related errors."""

    def __init__(self, message: str, exchange: str = "binance"):
        super().__init__(message, code="EXCHANGE_ERROR")
        self.exchange = exchange


class OrderExecutionError(ExchangeError):
    """Raised when order execution fails."""

    def __init__(self, message: str, order_id: str | None = None):
        super().__init__(message)
        self.code = "ORDER_EXECUTION_ERROR"
        self.order_id = order_id


class SlippageAlert(ExchangeError):
    """Raised when slippage exceeds threshold."""

    def __init__(
        self, message: str, slippage: Decimal, threshold: Decimal = Decimal("0.5")
    ):
        super().__init__(message)
        self.code = "SLIPPAGE_ALERT"
        self.slippage = slippage
        self.threshold = threshold


class InsufficientLiquidity(ExchangeError):
    """Raised when market liquidity is insufficient for order execution."""

    def __init__(
        self, message: str, required_liquidity: Decimal, available_liquidity: Decimal
    ):
        super().__init__(message)
        self.code = "INSUFFICIENT_LIQUIDITY"
        self.required_liquidity = required_liquidity
        self.available_liquidity = available_liquidity


class MarketDataError(ExchangeError):
    """Raised when market data operations fail."""

    def __init__(self, message: str):
        super().__init__(message)
        self.code = "MARKET_DATA_ERROR"


class DataError(GenesisException):
    """Raised when data validation or processing fails."""

    def __init__(self, message: str):
        super().__init__(message, code="DATA_ERROR")


class ValidationError(GenesisException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        self.field = field
        if field:
            message = f"Validation error for {field}: {message}"
        super().__init__(message, code="VALIDATION_ERROR")


class StateError(GenesisException):
    """Raised when an operation is attempted in an invalid state."""

    def __init__(self, message: str):
        super().__init__(message, code="STATE_ERROR")


class BackupError(GenesisException):
    """Raised when backup or recovery operations fail."""

    def __init__(self, message: str):
        super().__init__(message, code="BACKUP_ERROR")
