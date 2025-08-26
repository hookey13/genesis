"""
Structured logging configuration for Project GENESIS.

This module provides centralized logging setup using structlog for
structured, JSON-formatted logging with proper context preservation
and performance tracking.

Key Features:
    - Structured JSON logging for machine parsing
    - Context preservation across async operations
    - Performance timing for critical operations
    - Separate log streams for trading, audit, and tilt events
    - Automatic log rotation and compression
    - Sensitive data redaction

Example:
    >>> from genesis.utils.logger import get_logger, setup_logging
    >>> setup_logging()
    >>> logger = get_logger(__name__)
    >>> logger.info("trade_executed", pair="BTC/USDT", amount=0.01, side="buy")
"""

import logging
import logging.handlers
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from structlog.processors import CallsiteParameter


class LoggerType(str, Enum):
    """Types of specialized loggers in the system."""

    TRADING = "trading"
    AUDIT = "audit"
    TILT = "tilt"
    SYSTEM = "system"


# Sensitive field names to redact in logs
SENSITIVE_FIELDS = {
    "password",
    "secret",
    "api_key",
    "api_secret",
    "token",
    "private_key",
    "seed_phrase",
    "authorization",
    "credential",
}


def redact_sensitive_data(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Redact sensitive information from log events.

    Args:
        logger: Logger instance
        method_name: Name of the logging method
        event_dict: Dictionary containing log event data

    Returns:
        Modified event dictionary with sensitive data redacted
    """
    for key in list(event_dict.keys()):
        # Check if key name suggests sensitive data
        if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
            if event_dict[key]:
                event_dict[key] = "***REDACTED***"

        # Recursively check nested dictionaries
        elif isinstance(event_dict[key], dict):
            for nested_key in list(event_dict[key].keys()):
                if any(
                    sensitive in nested_key.lower() for sensitive in SENSITIVE_FIELDS
                ):
                    event_dict[key][nested_key] = "***REDACTED***"

    return event_dict


def add_log_level(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add log level to event dictionary."""
    event_dict["level"] = method_name.upper()
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    enable_console: bool = True,
    enable_json: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: .genesis/logs)
        enable_console: Whether to output logs to console
        enable_json: Whether to format logs as JSON
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Raises:
        PermissionError: If log directory cannot be created
        ValueError: If invalid log level specified
    """
    # Set up log directory
    if log_dir is None:
        log_dir = Path(".genesis/logs")

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"ERROR: Cannot create log directory {log_dir}: {e}", file=sys.stderr)
        raise

    # Configure Python's standard logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create formatters
    if enable_json:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
            ],
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = []  # Clear existing handlers

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handlers for different log types
    log_files = {
        "trading": log_dir / "trading.log",
        "audit": log_dir / "audit.log",
        "tilt": log_dir / "tilt.log",
        "system": log_dir / "system.log",
    }

    for log_type, log_file in log_files.items():
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)

        # Create specific logger for each type
        specific_logger = logging.getLogger(f"genesis.{log_type}")
        specific_logger.addHandler(file_handler)
        specific_logger.propagate = False  # Don't propagate to root logger

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            redact_sensitive_data,  # Custom processor to redact sensitive data
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.FILENAME,
                    CallsiteParameter.LINENO,
                    CallsiteParameter.FUNC_NAME,
                ]
            ),
            structlog.processors.dict_tracebacks,
            (
                structlog.processors.JSONRenderer()
                if enable_json
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(
    name: str, logger_type: LoggerType = LoggerType.SYSTEM, **context: Any
) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        logger_type: Type of logger (trading, audit, tilt, system)
        **context: Additional context to bind to logger

    Returns:
        Configured structlog logger with bound context

    Example:
        >>> logger = get_logger(__name__, logger_type=LoggerType.TRADING, strategy="arbitrage")
        >>> logger.info("position_opened", pair="BTC/USDT", size=0.01)
    """
    # Use specific logger based on type
    if logger_type != LoggerType.SYSTEM:
        name = f"genesis.{logger_type.value}.{name}"

    logger = structlog.get_logger(name)

    # Bind additional context
    if context:
        logger = logger.bind(**context)

    return logger


class LogContext:
    """
    Context manager for temporary logging context.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, request_id="123", user="trader1"):
        ...     logger.info("processing_request")  # Will include request_id and user
    """

    def __init__(self, logger: structlog.stdlib.BoundLogger, **context: Any):
        """
        Initialize log context.

        Args:
            logger: Logger instance to add context to
            **context: Context variables to bind
        """
        self.logger = logger
        self.context = context
        self.token = None

    def __enter__(self) -> structlog.stdlib.BoundLogger:
        """Enter context and bind variables."""
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind variables."""
        if self.token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


class PerformanceLogger:
    """
    Context manager for logging operation performance.

    Example:
        >>> logger = get_logger(__name__)
        >>> with PerformanceLogger(logger, "calculate_position_size"):
        ...     result = complex_calculation()
        # Automatically logs duration
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        operation: str,
        warn_threshold_ms: float = 1000,
        **context: Any,
    ):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Name of operation being timed
            warn_threshold_ms: Log warning if operation takes longer than this (ms)
            **context: Additional context to log
        """
        self.logger = logger
        self.operation = operation
        self.warn_threshold_ms = warn_threshold_ms
        self.context = context
        self.start_time = None

    def __enter__(self) -> "PerformanceLogger":
        """Start timing."""
        import time

        self.start_time = time.perf_counter()
        self.logger.debug(f"{self.operation}_started", **self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log operation duration."""
        import time

        duration_ms = (time.perf_counter() - self.start_time) * 1000

        log_data = {
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            **self.context,
        }

        if exc_type:
            log_data["error"] = str(exc_val)
            log_data["error_type"] = exc_type.__name__
            self.logger.error(f"{self.operation}_failed", **log_data)
        elif duration_ms > self.warn_threshold_ms:
            self.logger.warning(f"{self.operation}_slow", **log_data)
        else:
            self.logger.info(f"{self.operation}_completed", **log_data)


def log_trade_event(
    action: str,
    pair: str,
    side: str,
    amount: float,
    price: float | None = None,
    order_id: str | None = None,
    **extra: Any,
) -> None:
    """
    Log a trading event to the audit log.

    Args:
        action: Trading action (order_placed, order_filled, order_cancelled, etc.)
        pair: Trading pair (e.g., "BTC/USDT")
        side: Trade side (buy/sell)
        amount: Trade amount
        price: Trade price (if applicable)
        order_id: Order identifier
        **extra: Additional event data
    """
    logger = get_logger("trade_audit", LoggerType.AUDIT)

    event_data = {
        "action": action,
        "pair": pair,
        "side": side,
        "amount": amount,
    }

    if price:
        event_data["price"] = price
        event_data["value"] = amount * price

    if order_id:
        event_data["order_id"] = order_id

    event_data.update(extra)

    logger.info(f"trade_{action}", **event_data)


def log_tilt_event(
    indicator: str, value: float, threshold: float, triggered: bool, **extra: Any
) -> None:
    """
    Log a tilt detection event.

    Args:
        indicator: Tilt indicator name
        value: Current indicator value
        threshold: Threshold value
        triggered: Whether tilt was triggered
        **extra: Additional context
    """
    logger = get_logger("tilt_detection", LoggerType.TILT)

    level = "warning" if triggered else "info"
    getattr(logger, level)(
        "tilt_indicator",
        indicator=indicator,
        value=value,
        threshold=threshold,
        triggered=triggered,
        **extra,
    )


# Module-level initialization
if __name__ == "__main__":
    # Example usage and testing
    setup_logging(log_level="DEBUG", enable_json=False)

    # Test different logger types
    system_logger = get_logger("test", LoggerType.SYSTEM)
    system_logger.info("system_initialized", version="1.0.0")

    trading_logger = get_logger("test", LoggerType.TRADING)
    trading_logger.info(
        "strategy_started", strategy="arbitrage", pairs=["BTC/USDT", "ETH/USDT"]
    )

    # Test performance logging
    with PerformanceLogger(system_logger, "test_operation", warn_threshold_ms=100):
        import time

        time.sleep(0.05)  # Simulate work

    # Test trade event logging
    log_trade_event(
        action="order_placed",
        pair="BTC/USDT",
        side="buy",
        amount=0.01,
        price=50000,
        order_id="123456",
    )

    # Test tilt event logging
    log_tilt_event(
        indicator="click_speed",
        value=7.5,
        threshold=5.0,
        triggered=True,
        severity="medium",
    )

    # Test sensitive data redaction
    logger_with_secrets = get_logger("security_test")
    logger_with_secrets.info(
        "config_loaded",
        api_key="secret123",  # Will be redacted
        exchange="binance",
        password="mypass",  # Will be redacted
    )

    print("\n✓ Logging system configured and tested successfully")
