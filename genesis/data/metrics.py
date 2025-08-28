"""
Database metrics collection for monitoring.

Tracks repository operations, transaction times, and data flow.
"""

import time
import functools
import logging
from typing import Any, Callable
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# === Transaction Metrics ===

db_transaction_duration = Histogram(
    "genesis_db_transaction_duration_seconds",
    "Database transaction duration",
    ["operation"],
)

db_transaction_failures = Counter(
    "genesis_db_transaction_failures_total",
    "Total database transaction failures",
    ["operation", "error_type"],
)

# === Order Metrics ===

orders_created = Counter(
    "genesis_orders_created_total", "Total orders created", ["symbol", "side", "type"]
)

orders_filled = Counter(
    "genesis_orders_filled_total", "Total orders filled", ["symbol", "side"]
)

orders_cancelled = Counter(
    "genesis_orders_cancelled_total", "Total orders cancelled", ["symbol", "reason"]
)

orders_open = Gauge("genesis_orders_open", "Current number of open orders", ["symbol"])

# === Trade Metrics ===

trades_ingested = Counter(
    "genesis_trades_ingested_total", "Total trades ingested", ["symbol", "side"]
)

trades_duplicate = Counter(
    "genesis_trades_duplicate_total", "Total duplicate trades skipped", ["symbol"]
)

trade_volume = Counter(
    "genesis_trade_volume_quote_total",
    "Total trade volume in quote currency",
    ["symbol", "side"],
)

# === Position Metrics ===

position_updates = Counter(
    "genesis_position_updates_total",
    "Total position updates",
    ["symbol", "action"],  # action: open, increase, reduce, close, reverse
)

position_pnl_realized = Counter(
    "genesis_position_pnl_realized_total",
    "Total realized PnL",
    ["symbol", "direction"],  # direction: profit, loss
)

positions_open = Gauge("genesis_positions_open", "Current number of open positions", [])

# === Candle Metrics ===

candles_upserted = Counter(
    "genesis_candles_upserted_total", "Total candles upserted", ["symbol", "timeframe"]
)

candle_gaps = Counter(
    "genesis_candle_gaps_total", "Total candle gaps detected", ["symbol", "timeframe"]
)

# === Event Metrics ===

events_published = Counter(
    "genesis_events_published_total", "Total events published to outbox", ["event_type"]
)

events_consumed = Counter(
    "genesis_events_consumed_total",
    "Total events consumed from inbox",
    ["event_type", "status"],  # status: new, duplicate
)

events_pending = Gauge("genesis_events_pending", "Number of pending outbox events", [])

# === Decorators ===


def track_db_operation(operation: str):
    """
    Decorator to track database operation metrics.

    Args:
        operation: Name of the operation being tracked
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                db_transaction_duration.labels(operation=operation).observe(
                    time.time() - start_time
                )
                return result
            except Exception as e:
                db_transaction_failures.labels(
                    operation=operation, error_type=type(e).__name__
                ).inc()
                raise

        return wrapper

    return decorator


@contextmanager
def track_transaction(operation: str):
    """
    Context manager to track database transaction timing.

    Args:
        operation: Name of the transaction operation
    """
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        db_transaction_duration.labels(operation=operation).observe(duration)
    except Exception as e:
        db_transaction_failures.labels(
            operation=operation, error_type=type(e).__name__
        ).inc()
        raise


def track_order_created(symbol: str, side: str, order_type: str):
    """Track order creation."""
    orders_created.labels(symbol=symbol, side=side, type=order_type).inc()


def track_order_filled(symbol: str, side: str):
    """Track order fill."""
    orders_filled.labels(symbol=symbol, side=side).inc()


def track_order_cancelled(symbol: str, reason: str = "user"):
    """Track order cancellation."""
    orders_cancelled.labels(symbol=symbol, reason=reason).inc()


def track_trade_ingested(
    symbol: str, side: str, volume_quote: float, is_duplicate: bool = False
):
    """Track trade ingestion."""
    if is_duplicate:
        trades_duplicate.labels(symbol=symbol).inc()
    else:
        trades_ingested.labels(symbol=symbol, side=side).inc()
        trade_volume.labels(symbol=symbol, side=side).inc(volume_quote)


def track_position_update(symbol: str, action: str):
    """
    Track position update.

    Args:
        symbol: Trading symbol
        action: One of: open, increase, reduce, close, reverse
    """
    position_updates.labels(symbol=symbol, action=action).inc()


def track_realized_pnl(symbol: str, amount: float):
    """Track realized PnL."""
    direction = "profit" if amount > 0 else "loss"
    position_pnl_realized.labels(symbol=symbol, direction=direction).inc(abs(amount))


def track_candles_upserted(symbol: str, timeframe: str, count: int):
    """Track candle upserts."""
    candles_upserted.labels(symbol=symbol, timeframe=timeframe).inc(count)


def track_event_published(event_type: str):
    """Track event publication."""
    events_published.labels(event_type=event_type).inc()


def track_event_consumed(event_type: str, is_duplicate: bool = False):
    """Track event consumption."""
    status = "duplicate" if is_duplicate else "new"
    events_consumed.labels(event_type=event_type, status=status).inc()


def update_open_orders_gauge(counts_by_symbol: dict):
    """Update open orders gauge."""
    for symbol, count in counts_by_symbol.items():
        orders_open.labels(symbol=symbol).set(count)


def update_open_positions_gauge(count: int):
    """Update open positions gauge."""
    positions_open.set(count)


def update_pending_events_gauge(count: int):
    """Update pending events gauge."""
    events_pending.set(count)
