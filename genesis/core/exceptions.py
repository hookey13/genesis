"""
Custom exceptions for Project GENESIS.

This module contains all custom exceptions used throughout the application
for error handling and control flow. Implements a comprehensive exception
hierarchy with GENESIS-XXXX error codes for tracking.
"""

from decimal import Decimal
from typing import Any, Dict, Optional


class BaseError(Exception):
    """
    Base exception for all GENESIS errors with error code support.
    
    Uses GENESIS-XXXX error code format for tracking and categorization.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code or self._generate_code()
        self.details = details or {}
        
    def _generate_code(self) -> str:
        """Generate GENESIS-XXXX error code based on exception type."""
        # Default implementation - can be overridden by subclasses
        class_name = self.__class__.__name__
        hash_val = str(abs(hash(class_name)))[-4:]
        return f"GENESIS-{hash_val}"


class DomainError(BaseError):
    """Base class for domain-specific errors."""
    
    pass


class TradingError(DomainError):
    """Base class for trading-related errors."""
    
    pass


class OrderError(TradingError):
    """Base class for order-specific errors."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)
        self.order_id = order_id
        if order_id:
            self.details["order_id"] = order_id


# Legacy compatibility alias
GenesisException = BaseError


class RiskLimitExceeded(TradingError):
    """Raised when a risk limit would be exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str,
        current_value: Decimal,
        limit_value: Decimal,
    ):
        super().__init__(
            message,
            code=f"GENESIS-5701",  # 5=BUSINESS, 7=HIGH, 01=RiskLimit
            details={
                "limit_type": limit_type,
                "current_value": str(current_value),
                "limit_value": str(limit_value),
            },
        )
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class TierViolation(DomainError):
    """Raised when attempting to access tier-locked features."""

    def __init__(self, message: str, required_tier: str, current_tier: str):
        super().__init__(
            message,
            code="GENESIS-5702",  # 5=BUSINESS, 7=HIGH, 02=TierViolation
            details={
                "required_tier": required_tier,
                "current_tier": current_tier,
            },
        )
        self.required_tier = required_tier
        self.current_tier = current_tier


class TiltInterventionRequired(DomainError):
    """Raised when tilt detection requires intervention."""
    
    def __init__(
        self,
        message: str,
        tilt_score: float,
        threshold: float,
        indicators: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="GENESIS-5903",  # 5=BUSINESS, 9=CRITICAL, 03=TiltIntervention
            details={
                "tilt_score": tilt_score,
                "threshold": threshold,
                "indicators": indicators or {},
            },
        )
        self.tilt_score = tilt_score
        self.threshold = threshold
        self.indicators = indicators or {}


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


class InsufficientBalance(TradingError):
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


# Network and Exchange Specific Errors
class NetworkError(BaseError):
    """Base class for network-related errors."""
    
    pass


class ConnectionTimeout(NetworkError):
    """Raised when connection times out."""
    
    def __init__(self, message: str, timeout_seconds: float):
        super().__init__(
            message,
            code="GENESIS-2501",  # 2=NETWORK, 5=MEDIUM, 01=Timeout
            details={"timeout_seconds": timeout_seconds},
        )
        self.timeout_seconds = timeout_seconds


class RateLimitError(NetworkError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(
            message,
            code="GENESIS-2502",  # 2=NETWORK, 5=MEDIUM, 02=RateLimit
            details={
                "retry_after_seconds": retry_after_seconds,
                "endpoint": endpoint,
            },
        )
        self.retry_after_seconds = retry_after_seconds
        self.endpoint = endpoint


# Database Errors
class DatabaseError(BaseError):
    """Base class for database-related errors."""
    
    pass


class DatabaseLocked(DatabaseError):
    """Raised when database is locked."""
    
    def __init__(self, message: str, table: Optional[str] = None):
        super().__init__(
            message,
            code="GENESIS-3501",  # 3=DATABASE, 5=MEDIUM, 01=Locked
            details={"table": table},
        )
        self.table = table


class TransactionRollback(DatabaseError):
    """Raised when a database transaction needs to be rolled back."""
    
    def __init__(self, message: str, reason: str):
        super().__init__(
            message,
            code="GENESIS-3701",  # 3=DATABASE, 7=HIGH, 01=Rollback
            details={"reason": reason},
        )
        self.reason = reason


# System Errors
class SystemError(BaseError):
    """Base class for system-level errors."""
    
    pass


class MemoryError(SystemError):
    """Raised when memory limits are exceeded."""
    
    def __init__(self, message: str, memory_usage_mb: float, limit_mb: float):
        super().__init__(
            message,
            code="GENESIS-6901",  # 6=SYSTEM, 9=CRITICAL, 01=Memory
            details={
                "memory_usage_mb": memory_usage_mb,
                "limit_mb": limit_mb,
            },
        )
        self.memory_usage_mb = memory_usage_mb
        self.limit_mb = limit_mb


class ConfigurationMissing(SystemError):
    """Raised when required configuration is missing."""
    
    def __init__(self, message: str, config_key: str):
        super().__init__(
            message,
            code="GENESIS-6701",  # 6=SYSTEM, 7=HIGH, 01=ConfigMissing
            details={"config_key": config_key},
        )
        self.config_key = config_key


# Order-specific Errors
class OrderRejected(OrderError):
    """Raised when an order is rejected by the exchange."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(
            message,
            order_id=order_id,
            code="GENESIS-1701",  # 1=EXCHANGE, 7=HIGH, 01=OrderRejected
        )
        self.reason = reason
        if reason:
            self.details["reason"] = reason


class OrderNotFound(OrderError):
    """Raised when an order cannot be found."""
    
    def __init__(self, message: str, order_id: str):
        super().__init__(
            message,
            order_id=order_id,
            code="GENESIS-1502",  # 1=EXCHANGE, 5=MEDIUM, 02=OrderNotFound
        )


class InvalidOrderParameters(OrderError):
    """Raised when order parameters are invalid."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        invalid_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            order_id=order_id,
            code="GENESIS-4501",  # 4=VALIDATION, 5=MEDIUM, 01=InvalidParams
        )
        self.invalid_params = invalid_params
        if invalid_params:
            self.details["invalid_params"] = invalid_params


# Binance Error Mapping
def map_binance_error_to_domain(binance_error_code: int, message: str) -> BaseError:
    """
    Map Binance API error codes to domain-specific exceptions.
    
    Binance error code reference:
    https://binance-docs.github.io/apidocs/spot/en/#error-codes
    """
    error_mapping = {
        # General errors
        -1000: lambda: NetworkError(f"Unknown error: {message}", code="GENESIS-1901"),
        -1001: lambda: NetworkError(f"Internal error: {message}", code="GENESIS-1902"),
        -1002: lambda: ValidationError(f"Unauthorized: {message}", field="api_key"),
        -1003: lambda: RateLimitError(f"Too many requests: {message}", retry_after_seconds=60),
        -1006: lambda: NetworkError(f"Unexpected response: {message}", code="GENESIS-1903"),
        -1007: lambda: ConnectionTimeout(f"Timeout: {message}", timeout_seconds=30),
        
        # Request errors
        -1100: lambda: ValidationError(f"Illegal characters: {message}", field="request"),
        -1101: lambda: ValidationError(f"Too many parameters: {message}", field="request"),
        -1102: lambda: ValidationError(f"Mandatory parameter missing: {message}", field="request"),
        -1103: lambda: ValidationError(f"Unknown parameter: {message}", field="request"),
        
        # Market errors
        -1121: lambda: InvalidOrderParameters(f"Invalid symbol: {message}", invalid_params={"symbol": "invalid"}),
        
        # Order errors
        -2010: lambda: OrderRejected(f"New order rejected: {message}", reason="INSUFFICIENT_BALANCE"),
        -2011: lambda: OrderRejected(f"Cancel rejected: {message}", reason="UNKNOWN_ORDER"),
        -2013: lambda: OrderNotFound(f"Order does not exist: {message}", order_id="unknown"),
        -2014: lambda: ValidationError(f"Invalid API key format: {message}", field="api_key"),
        -2015: lambda: ValidationError(f"Invalid API key or permissions: {message}", field="api_key"),
        
        # Filter failures
        -1013: lambda: InvalidOrderParameters(f"Invalid quantity: {message}", invalid_params={"quantity": "invalid"}),
        -1111: lambda: InvalidOrderParameters(f"Precision over maximum: {message}", invalid_params={"precision": "exceeded"}),
    }
    
    # Get the error factory or return a generic exchange error
    error_factory = error_mapping.get(
        binance_error_code,
        lambda: ExchangeError(f"Binance error {binance_error_code}: {message}"),
    )
    
    return error_factory()
