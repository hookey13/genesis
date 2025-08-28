"""
Exchange event contracts for event-driven architecture.

This module uses composition over inheritance to avoid dataclass
issues while maintaining a clean event structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import uuid4


class EventType(str, Enum):
    """Exchange event types."""

    # Market data events
    MARKET_TICK = "market_tick"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_UPDATE = "trade_update"

    # Order lifecycle events
    ORDER_ACK = "order_ack"
    ORDER_FILL = "order_fill"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_CANCEL = "order_cancel"
    ORDER_REJECT = "order_reject"
    ORDER_EXPIRE = "order_expire"

    # System events
    EXCHANGE_HEARTBEAT = "exchange_heartbeat"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    WS_CONNECTED = "ws_connected"
    WS_DISCONNECTED = "ws_disconnected"
    WS_RECONNECTING = "ws_reconnecting"
    RECONCILIATION_START = "reconciliation_start"
    RECONCILIATION_COMPLETE = "reconciliation_complete"
    CLOCK_SKEW_DETECTED = "clock_skew_detected"


def create_event_id() -> str:
    """Generate unique event ID."""
    return str(uuid4())


def current_timestamp() -> datetime:
    """Get current timestamp."""
    return datetime.utcnow()


@dataclass
class MarketTick:
    """Real-time price tick event."""

    symbol: str
    price: Decimal
    ts: datetime

    # Optional fields with defaults
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_qty: Optional[Decimal] = None
    ask_qty: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None

    # Event metadata
    event_type: EventType = field(default=EventType.MARKET_TICK, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None


@dataclass
class OrderAck:
    """Order acknowledged by exchange."""

    client_order_id: str
    exchange_order_id: str
    ts: datetime

    # Optional fields
    symbol: Optional[str] = None
    side: Optional[str] = None
    order_type: Optional[str] = None
    quantity: Optional[Decimal] = None
    price: Optional[Decimal] = None
    time_in_force: Optional[str] = None
    status: str = "NEW"

    # Event metadata
    event_type: EventType = field(default=EventType.ORDER_ACK, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None


@dataclass
class OrderFill:
    """Order fill/execution event."""

    client_order_id: str
    exchange_trade_id: str
    qty: Decimal
    price: Decimal
    fee_ccy: str
    fee_amt: Decimal
    ts: datetime

    # Optional fields
    exchange_order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    cumulative_qty: Optional[Decimal] = None
    remaining_qty: Optional[Decimal] = None
    is_partial: bool = False
    status: str = "FILLED"

    # Event metadata
    event_type: EventType = field(default=EventType.ORDER_FILL, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None

    def __post_init__(self):
        """Set event type based on fill type."""
        if self.is_partial:
            self.event_type = EventType.ORDER_PARTIAL_FILL


@dataclass
class OrderCancel:
    """Order cancellation confirmation."""

    client_order_id: str
    reason: str
    ts: datetime

    # Optional fields
    exchange_order_id: Optional[str] = None
    symbol: Optional[str] = None
    canceled_qty: Optional[Decimal] = None
    executed_qty: Optional[Decimal] = None
    status: str = "CANCELED"

    # Event metadata
    event_type: EventType = field(default=EventType.ORDER_CANCEL, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None


@dataclass
class OrderReject:
    """Order rejection event."""

    client_order_id: str
    reason: str
    ts: datetime

    # Optional fields
    symbol: Optional[str] = None
    side: Optional[str] = None
    order_type: Optional[str] = None
    quantity: Optional[Decimal] = None
    price: Optional[Decimal] = None
    error_code: Optional[int] = None
    status: str = "REJECTED"

    # Event metadata
    event_type: EventType = field(default=EventType.ORDER_REJECT, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None


@dataclass
class ExchangeHeartbeat:
    """Exchange connectivity heartbeat."""

    ok: bool
    detail: str
    ts: datetime

    # Optional health metrics
    exchange: str = "binance"
    ws_connected: Optional[bool] = None
    rest_responsive: Optional[bool] = None
    latency_ms: Optional[int] = None
    open_orders: Optional[int] = None
    rate_limit_remaining: Optional[int] = None
    listen_key_valid: Optional[bool] = None

    # Event metadata
    event_type: EventType = field(default=EventType.EXCHANGE_HEARTBEAT, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker state change."""

    tripped: bool
    reason: str
    ts: datetime

    # Optional fields
    state: str = "CLOSED"  # OPEN, CLOSED, HALF_OPEN
    error_count: Optional[int] = None
    error_rate: Optional[float] = None
    cooldown_seconds: Optional[int] = None

    # Event metadata
    event_type: EventType = field(default=EventType.CIRCUIT_BREAKER_OPEN, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None

    def __post_init__(self):
        """Set event type and state based on tripped status."""
        if self.tripped:
            self.event_type = EventType.CIRCUIT_BREAKER_OPEN
            self.state = "OPEN"
        else:
            self.event_type = EventType.CIRCUIT_BREAKER_CLOSE
            self.state = "CLOSED"


@dataclass
class WebSocketEvent:
    """WebSocket connection event."""

    status: str  # CONNECTED, DISCONNECTED, RECONNECTING
    stream_type: str  # market_data, user_data
    ts: datetime

    # Optional fields
    url: Optional[str] = None
    reconnect_attempt: int = 0
    error: Optional[str] = None

    # Event metadata
    event_type: EventType = field(default=EventType.WS_CONNECTED, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None

    def __post_init__(self):
        """Set event type based on status."""
        if self.status == "CONNECTED":
            self.event_type = EventType.WS_CONNECTED
        elif self.status == "DISCONNECTED":
            self.event_type = EventType.WS_DISCONNECTED
        else:
            self.event_type = EventType.WS_RECONNECTING


@dataclass
class ReconciliationEvent:
    """Position/order reconciliation event."""

    phase: str  # START, COMPLETE
    ts: datetime

    # Reconciliation metrics
    orders_reconciled: int = 0
    positions_reconciled: int = 0
    discrepancies_found: int = 0
    corrections_made: int = 0
    duration_ms: Optional[int] = None

    # Event metadata
    event_type: EventType = field(default=EventType.RECONCILIATION_START, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None

    def __post_init__(self):
        """Set event type based on phase."""
        if self.phase == "START":
            self.event_type = EventType.RECONCILIATION_START
        else:
            self.event_type = EventType.RECONCILIATION_COMPLETE


@dataclass
class ClockSkewEvent:
    """Clock synchronization issue detected."""

    local_time: datetime
    server_time: datetime
    skew_ms: int
    ts: datetime

    # Optional fields
    threshold_ms: int = 5000
    action_taken: str = "WARNING"  # WARNING, HALT_TRADING, SYNC_ATTEMPTED

    # Event metadata
    event_type: EventType = field(default=EventType.CLOCK_SKEW_DETECTED, init=False)
    event_id: str = field(default_factory=create_event_id)
    sequence: Optional[int] = None


class EventBus:
    """
    Simple event bus for publishing exchange events.

    Provides publish/subscribe mechanism for event-driven architecture.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Any]] = {}
        self._all_subscribers: List[Any] = []
        self._sequence = 0
        self._event_history: List[Any] = []
        self._max_history = 1000

    def publish(self, event: Any) -> None:
        """
        Publish event to all subscribers.

        Args:
            event: Event to publish (must have event_type attribute)
        """
        # Assign sequence number
        self._sequence += 1
        if hasattr(event, "sequence"):
            event.sequence = self._sequence

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Notify specific subscribers
        event_type = getattr(event, "event_type", None)
        if event_type and event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    import logging

                    logging.error(f"Event subscriber error for {event_type}: {e}")

        # Notify all-event subscribers
        for callback in self._all_subscribers:
            try:
                callback(event)
            except Exception as e:
                import logging

                logging.error(f"All-event subscriber error: {e}")

    def subscribe(self, event_type: Optional[EventType], callback: Any) -> None:
        """
        Subscribe to events.

        Args:
            event_type: Type of events to subscribe to (None for all events)
            callback: Function to call when event is published
        """
        if event_type is None:
            self._all_subscribers.append(callback)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: Optional[EventType], callback: Any) -> None:
        """
        Unsubscribe from events.

        Args:
            event_type: Type of events to unsubscribe from
            callback: Callback function to remove
        """
        if event_type is None:
            if callback in self._all_subscribers:
                self._all_subscribers.remove(callback)
        elif event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def get_history(
        self, event_type: Optional[EventType] = None, limit: int = 100
    ) -> List[Any]:
        """
        Get event history.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of historical events
        """
        if event_type is None:
            return self._event_history[-limit:]
        else:
            filtered = [
                e
                for e in self._event_history
                if getattr(e, "event_type", None) == event_type
            ]
            return filtered[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def reset(self) -> None:
        """Reset the event bus."""
        self._subscribers.clear()
        self._all_subscribers.clear()
        self._event_history.clear()
        self._sequence = 0


# Global event bus instance
event_bus = EventBus()
