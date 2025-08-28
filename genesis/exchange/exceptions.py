"""
Exchange-specific exceptions and error handling.

Maps Binance error codes to domain exceptions and provides
retry logic for transient errors.
"""

import asyncio
import functools
from collections.abc import Callable
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class ExchangeError(Exception):
    """Base exception for exchange-related errors."""

    def __init__(
        self,
        message: str,
        code: Optional[int] = None,
        retry_after: Optional[int] = None,
    ):
        """
        Initialize exchange error.

        Args:
            message: Error message
            code: Error code from exchange
            retry_after: Seconds to wait before retry
        """
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class RateLimitError(ExchangeError):
    """Rate limit exceeded error."""

    pass


class AuthenticationError(ExchangeError):
    """Authentication failed error."""

    pass


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance for operation."""

    pass


class OrderNotFoundError(ExchangeError):
    """Order not found error."""

    pass


class InvalidOrderError(ExchangeError):
    """Invalid order parameters error."""

    pass


class NetworkError(ExchangeError):
    """Network-related error."""

    pass


class MaintenanceError(ExchangeError):
    """Exchange under maintenance error."""

    pass


class TimeoutError(ExchangeError):
    """Request timeout error."""

    pass


class UnknownSymbolError(ExchangeError):
    """Unknown trading symbol error."""

    pass


class MinOrderSizeError(ExchangeError):
    """Order size below minimum error."""

    pass


class MaxOrderSizeError(ExchangeError):
    """Order size above maximum error."""

    pass


class ErrorTranslator:
    """
    Translates Binance error codes to domain exceptions.

    Based on Binance API documentation:
    https://binance-docs.github.io/apidocs/spot/en/#error-codes
    """

    # Binance error code mappings
    ERROR_CODES = {
        # 10xx - General Server or Network issues
        -1000: (NetworkError, "Unknown error occurred"),
        -1001: (NetworkError, "Internal server error"),
        -1002: (AuthenticationError, "API key required"),
        -1003: (RateLimitError, "Too many requests"),
        -1004: (MaintenanceError, "Server busy, please retry later"),
        -1005: (NetworkError, "No such IP has been white listed"),
        -1006: (NetworkError, "Unexpected response"),
        -1007: (TimeoutError, "Request timeout"),
        -1008: (RateLimitError, "Server is currently overloaded"),
        -1009: (NetworkError, "Too many orders"),
        -1010: (NetworkError, "Server is currently busy"),
        -1011: (NetworkError, "This IP cannot access this route"),
        -1012: (MaintenanceError, "System maintenance"),
        # 11xx - Request issues
        -1100: (InvalidOrderError, "Illegal characters found in parameter"),
        -1101: (InvalidOrderError, "Too many parameters"),
        -1102: (InvalidOrderError, "Mandatory parameter missing"),
        -1103: (InvalidOrderError, "Unknown parameter"),
        -1104: (InvalidOrderError, "Unread parameters"),
        -1105: (InvalidOrderError, "Parameter empty"),
        -1106: (InvalidOrderError, "Parameter not required"),
        -1111: (InvalidOrderError, "Precision is over the maximum"),
        -1112: (InvalidOrderError, "No depth"),
        -1114: (InvalidOrderError, "TimeInForce not required"),
        -1115: (InvalidOrderError, "Invalid timeInForce"),
        -1116: (InvalidOrderError, "Invalid orderType"),
        -1117: (InvalidOrderError, "Invalid side"),
        -1118: (InvalidOrderError, "New client order ID was empty"),
        -1119: (InvalidOrderError, "Original client order ID was empty"),
        -1120: (InvalidOrderError, "Invalid interval"),
        -1121: (UnknownSymbolError, "Invalid symbol"),
        -1125: (InvalidOrderError, "Invalid listenKey"),
        -1127: (InvalidOrderError, "Invalid quantity"),
        -1128: (InvalidOrderError, "Invalid price"),
        # 20xx - Processing issues
        -2008: (InvalidOrderError, "Invalid API-key format"),
        -2009: (
            InvalidOrderError,
            "Margin account are not allowed to trade this trading pair",
        ),
        -2010: (InsufficientBalanceError, "Account has insufficient balance"),
        -2011: (OrderNotFoundError, "Order does not exist"),
        -2012: (InvalidOrderError, "API-key not found"),
        -2013: (OrderNotFoundError, "Order does not exist"),
        -2014: (AuthenticationError, "Invalid API-key"),
        -2015: (AuthenticationError, "Invalid API-key or permissions"),
        -2016: (InvalidOrderError, "No trading window could be found"),
        -2018: (InsufficientBalanceError, "Balance is insufficient"),
        -2019: (InsufficientBalanceError, "Margin is insufficient"),
        -2020: (InvalidOrderError, "Unable to fill"),
        -2021: (InvalidOrderError, "Order would immediately trigger"),
        -2022: (InvalidOrderError, "Order would immediately match and take"),
        -2023: (InvalidOrderError, "Cannot place order"),
        -2024: (InsufficientBalanceError, "Balance is insufficient"),
        -2025: (InvalidOrderError, "Reach max open order limit"),
        -2026: (InvalidOrderError, "This order type is not supported"),
        -2027: (InvalidOrderError, "Exceeded the maximum allowable position"),
        -2028: (InvalidOrderError, "Position is not sufficient"),
        # Filters
        -9000: (InvalidOrderError, "Filter failure"),
        # LOT_SIZE
        -1013: (MinOrderSizeError, "Order quantity below minimum"),
        # MIN_NOTIONAL
        -1016: (MinOrderSizeError, "Order value below minimum"),
        # MARKET_LOT_SIZE
        -1020: (InvalidOrderError, "Market order quantity precision invalid"),
        # MAX_NUM_ORDERS
        -1021: (InvalidOrderError, "Maximum number of orders exceeded"),
    }

    @classmethod
    def translate(cls, error: Exception) -> ExchangeError:
        """
        Translate an exchange error to a domain exception.

        Args:
            error: Original exception

        Returns:
            Translated domain exception
        """
        # Extract error code from the exception
        error_str = str(error)
        error_code = None

        # Try to extract Binance error code
        if "code=" in error_str:
            try:
                code_part = error_str.split("code=")[1].split()[0]
                error_code = int(code_part.rstrip(",)"))
            except (IndexError, ValueError):
                pass

        # Look up error code in mapping
        if error_code and error_code in cls.ERROR_CODES:
            exception_class, message = cls.ERROR_CODES[error_code]
            return exception_class(message, code=error_code)

        # Check for common error patterns in message
        error_lower = error_str.lower()

        if "rate limit" in error_lower or "too many requests" in error_lower:
            return RateLimitError(error_str)
        elif "insufficient balance" in error_lower:
            return InsufficientBalanceError(error_str)
        elif "authentication" in error_lower or "api key" in error_lower:
            return AuthenticationError(error_str)
        elif "not found" in error_lower:
            return OrderNotFoundError(error_str)
        elif "timeout" in error_lower:
            return TimeoutError(error_str)
        elif "maintenance" in error_lower:
            return MaintenanceError(error_str)
        elif "network" in error_lower or "connection" in error_lower:
            return NetworkError(error_str)

        # Default to generic exchange error
        return ExchangeError(error_str)


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.

    Args:
        error: Exception to check

    Returns:
        True if error is transient
    """
    transient_types = (NetworkError, TimeoutError, RateLimitError, MaintenanceError)

    return isinstance(error, transient_types)


def retry_on_transient_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
):
    """
    Decorator to retry function on transient errors.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for exponential backoff
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    # Translate the error
                    translated_error = ErrorTranslator.translate(e)
                    last_error = translated_error

                    # Check if we should retry
                    if attempt < max_retries and is_transient_error(translated_error):
                        # Use retry_after if provided
                        if (
                            hasattr(translated_error, "retry_after")
                            and translated_error.retry_after
                        ):
                            wait_time = translated_error.retry_after
                        else:
                            wait_time = min(delay, max_delay)

                        logger.warning(
                            f"Transient error on attempt {attempt + 1}/{max_retries + 1}, "
                            f"retrying in {wait_time}s",
                            error=str(translated_error),
                            function=func.__name__,
                        )

                        await asyncio.sleep(wait_time)

                        # Exponential backoff
                        delay *= backoff_factor
                    else:
                        # Non-transient error or max retries reached
                        raise translated_error

            # Should not reach here, but just in case
            if last_error:
                raise last_error
            else:
                raise ExchangeError("Unknown error after retries")

        return wrapper

    return decorator
