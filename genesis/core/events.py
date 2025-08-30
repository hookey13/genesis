"""
Event system for Project GENESIS.

Defines event types, priorities, and base event model for the event-driven architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    """Event types in the system."""

    # Market Data Events
    MARKET_DATA_UPDATED = "market_data_updated"
    SPREAD_ALERT = "spread_alert"
    SPREAD_COMPRESSION = "spread_compression"
    ORDER_IMBALANCE = "order_imbalance"
    VOLUME_ANOMALY = "volume_anomaly"
    ORDER_BOOK_SNAPSHOT = "order_book_snapshot"

    # Trading Events
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FAILED = "order_failed"

    # Position Events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"

    # Risk Events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    DAILY_LOSS_LIMIT_REACHED = "daily_loss_limit_reached"
    CORRELATION_ALERT = "correlation_alert"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"

    # Arbitrage Events
    ARBITRAGE_SIGNAL = "arbitrage_signal"
    ARBITRAGE_THRESHOLD_BREACH = "arbitrage_threshold_breach"

    # Tilt Events
    TILT_WARNING = "tilt_warning"
    
    # Audit Events
    AUDIT_LOG_CREATED = "audit_log_created"
    AUDIT_VERIFICATION_PASSED = "audit_verification_passed"
    AUDIT_VERIFICATION_FAILED = "audit_verification_failed"
    COMPLIANCE_EVENT = "compliance_event"
    DATA_RETENTION_EXECUTED = "data_retention_executed"
    DATA_EXPORT_REQUESTED = "data_export_requested"
    DATA_DELETION_REQUESTED = "data_deletion_requested"
    TILT_DETECTED = "tilt_detected"
    TILT_RECOVERY = "tilt_recovery"
    TILT_LEVEL1_DETECTED = "tilt_level1_detected"  # Yellow border warning
    TILT_LEVEL2_DETECTED = "tilt_level2_detected"  # Orange border, reduced sizing
    TILT_LEVEL3_DETECTED = "tilt_level3_detected"  # Red border, trading lockout
    TILT_ANOMALY_DETECTED = "tilt_anomaly_detected"  # Individual anomaly
    TILT_RECOVERED = "tilt_recovered"  # Recovery from tilt
    INTERVENTION_APPLIED = "intervention_applied"  # Intervention action taken

    # Recovery Protocol Events
    RECOVERY_PROTOCOL_INITIATED = (
        "recovery_protocol_initiated"  # Start recovery process
    )
    TRADING_LOCKOUT = "trading_lockout"  # Trading lockout enforced
    LOCKOUT_EXPIRED = "lockout_expired"  # Trading lockout period ended
    JOURNAL_ENTRY_SUBMITTED = "journal_entry_submitted"  # Journal requirement completed
    RECOVERY_CHECKLIST_UPDATED = (
        "recovery_checklist_updated"  # Checklist progress changed
    )
    RECOVERY_STAGE_ADVANCED = "recovery_stage_advanced"  # Position size increased
    TILT_DEBT_ADDED = "tilt_debt_added"  # Debt added to ledger
    TILT_DEBT_REDUCED = "tilt_debt_reduced"  # Debt paid down
    RECOVERY_COMPLETED = "recovery_completed"  # Full recovery achieved

    # Drawdown Recovery Events
    DRAWDOWN_DETECTED = "drawdown_detected"  # Significant drawdown detected
    DRAWDOWN_RECOVERY_INITIATED = (
        "drawdown_recovery_initiated"  # Drawdown recovery started
    )
    FORCED_BREAK_INITIATED = "forced_break_initiated"  # Forced trading break started
    FORCED_BREAK_CLEARED = "forced_break_cleared"  # Forced break cleared

    # Behavioral Baseline Events
    BASELINE_CALCULATION_STARTED = "baseline_calculation_started"
    BASELINE_CALCULATION_COMPLETE = "baseline_calculation_complete"
    BASELINE_RESET = "baseline_reset"
    BEHAVIORAL_METRIC_RECORDED = "behavioral_metric_recorded"

    # Market State Events
    MARKET_STATE_CHANGE = "market_state_change"
    GLOBAL_MARKET_STATE_CHANGE = "global_market_state_change"
    POSITION_SIZE_ADJUSTMENT = "position_size_adjustment"

    # Strategy Events
    STRATEGY_REGISTERED = "strategy_registered"
    STRATEGY_UNREGISTERED = "strategy_unregistered"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_PAUSED = "strategy_paused"
    STRATEGY_RESUMED = "strategy_resumed"
    STRATEGY_RECOVERED = "strategy_recovered"
    STRATEGY_CONFLICT = "strategy_conflict"
    STRATEGY_CAPITAL_ADJUSTED = "strategy_capital_adjusted"

    # A/B Testing Events
    AB_TEST_CREATED = "ab_test_created"
    AB_TEST_STARTED = "ab_test_started"
    AB_TEST_COMPLETED = "ab_test_completed"
    AB_TEST_ABORTED = "ab_test_aborted"

    # System Events
    TIER_PROGRESSION = "tier_progression"
    TIER_DEMOTION = "tier_demotion"
    TIER_GRADUATION = "tier_graduation"  # Alert for tier graduation eligibility
    GATE_COMPLETED = "gate_completed"  # Tier gate requirement fulfilled
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_HEARTBEAT = "system_heartbeat"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"


class EventPriority(str, Enum):
    """Event priority levels."""

    CRITICAL = "critical"  # System-critical events
    HIGH = "high"  # Execution and risk events
    NORMAL = "normal"  # Market data and regular updates
    LOW = "low"  # Informational events


@dataclass
class Event:
    """Base event model."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = EventType.SYSTEM_STARTUP
    aggregate_id: str = ""  # ID of the entity this event relates to
    event_data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sequence_number: int | None = None
    correlation_id: str | None = None  # For tracking related events
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> EventType:
        """Alias for event_type for backward compatibility."""
        return self.event_type

    @property
    def data(self) -> dict[str, Any]:
        """Alias for event_data for backward compatibility."""
        return self.event_data

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "event_data": self.event_data,
            "created_at": self.created_at.isoformat(),
            "sequence_number": self.sequence_number,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid4())),
            event_type=EventType(data.get("event_type", "system_startup")),
            aggregate_id=data.get("aggregate_id", ""),
            event_data=data.get("event_data", {}),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat())
            ),
            sequence_number=data.get("sequence_number"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ArbitrageSignalEvent(Event):
    """Event for arbitrage signal detection."""

    def __init__(
        self,
        pair1_symbol: str,
        pair2_symbol: str,
        zscore: Decimal,
        threshold_sigma: Decimal,
        signal_type: str,
        confidence_score: Decimal,
        **kwargs,
    ):
        """
        Initialize arbitrage signal event.

        Args:
            pair1_symbol: First trading pair
            pair2_symbol: Second trading pair
            zscore: Z-score deviation
            threshold_sigma: Sigma threshold level
            signal_type: 'ENTRY' or 'EXIT'
            confidence_score: Signal confidence (0-1)
        """
        super().__init__(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "pair1_symbol": pair1_symbol,
                "pair2_symbol": pair2_symbol,
                "zscore": str(zscore),
                "threshold_sigma": str(threshold_sigma),
                "signal_type": signal_type,
                "confidence_score": str(confidence_score),
            },
            **kwargs,
        )


@dataclass
class TierGraduationEvent(Event):
    """Event for tier graduation eligibility."""

    def __init__(
        self,
        current_tier: str,
        recommended_tier: str,
        current_capital: Decimal,
        message: str,
        **kwargs,
    ):
        """
        Initialize tier graduation event.

        Args:
            current_tier: Current trading tier
            recommended_tier: Recommended tier based on capital
            current_capital: Current capital amount
            message: Graduation message
        """
        super().__init__(
            event_type=EventType.TIER_GRADUATION,
            event_data={
                "current_tier": current_tier,
                "recommended_tier": recommended_tier,
                "current_capital": str(current_capital),
                "message": message,
            },
            **kwargs,
        )


@dataclass
class TierProgressionEvent(Event):
    """Event for tier progression."""

    def __init__(
        self,
        account_id: str,
        from_tier: str,
        to_tier: str,
        reason: str,
        gates_passed: list[str],
        **kwargs,
    ):
        """
        Initialize tier progression event.

        Args:
            account_id: Account ID
            from_tier: Previous tier
            to_tier: New tier
            reason: Reason for progression
            gates_passed: List of gates that were passed
        """
        super().__init__(
            event_type=EventType.TIER_PROGRESSION,
            aggregate_id=account_id,
            event_data={
                "from_tier": from_tier,
                "to_tier": to_tier,
                "reason": reason,
                "gates_passed": gates_passed,
            },
            **kwargs,
        )


@dataclass
class TierDemotionEvent(Event):
    """Event for tier demotion."""

    def __init__(
        self,
        account_id: str,
        from_tier: str,
        to_tier: str,
        reason: str,
        triggers: list[str],
        **kwargs,
    ):
        """
        Initialize tier demotion event.

        Args:
            account_id: Account ID
            from_tier: Previous tier
            to_tier: New (lower) tier
            reason: Reason for demotion
            triggers: List of triggers that caused demotion
        """
        super().__init__(
            event_type=EventType.TIER_DEMOTION,
            aggregate_id=account_id,
            event_data={
                "from_tier": from_tier,
                "to_tier": to_tier,
                "reason": reason,
                "triggers": triggers,
            },
            **kwargs,
        )


@dataclass
class GateCompletedEvent(Event):
    """Event for gate completion."""

    def __init__(
        self,
        account_id: str,
        gate_name: str,
        current_tier: str,
        target_tier: str,
        completion_value: Any,
        **kwargs,
    ):
        """
        Initialize gate completed event.

        Args:
            account_id: Account ID
            gate_name: Name of the completed gate
            current_tier: Current tier
            target_tier: Target tier for this gate
            completion_value: Value that satisfied the gate
        """
        super().__init__(
            event_type=EventType.GATE_COMPLETED,
            aggregate_id=account_id,
            event_data={
                "gate_name": gate_name,
                "current_tier": current_tier,
                "target_tier": target_tier,
                "completion_value": str(completion_value),
            },
            **kwargs,
        )
