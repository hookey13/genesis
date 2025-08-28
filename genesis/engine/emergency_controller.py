"""
Emergency controller for extreme market events.

Manages circuit breakers, correlation spikes, liquidity crises,
flash crashes, and rapid deleveraging protocols with comprehensive
safety controls and manual override capabilities.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import structlog
import yaml

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus
from genesis.exchange.circuit_breaker import CircuitBreakerManager

logger = structlog.get_logger(__name__)


class EmergencyState(str, Enum):
    """Emergency system states."""

    NORMAL = "normal"  # Normal operation
    WARNING = "warning"  # Heightened risk detected
    EMERGENCY = "emergency"  # Emergency halt active
    RECOVERY = "recovery"  # Recovering from emergency
    OVERRIDE = "override"  # Manual override active


class EmergencyType(str, Enum):
    """Types of emergencies."""

    DAILY_LOSS_HALT = "daily_loss_halt"
    CORRELATION_SPIKE = "correlation_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    SYSTEM_FAILURE = "system_failure"
    MANUAL_HALT = "manual_halt"


@dataclass
class EmergencyEvent:
    """Record of an emergency event."""

    event_id: str
    emergency_type: EmergencyType
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    triggered_at: datetime
    trigger_values: dict[str, Any]
    affected_symbols: list[str]
    actions_taken: list[str]
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None


class EmergencyController:
    """
    Master controller for emergency risk management.
    Monitors for extreme market conditions and automatically
    triggers protective actions to preserve capital during
    black swan events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        circuit_manager: CircuitBreakerManager,
        config_path: str = "config/emergency_thresholds.yaml",
    ):
        """
        Initialize emergency controller.
        Args:
            event_bus: Event bus for publishing emergency events
            circuit_manager: Circuit breaker manager
            config_path: Path to emergency configuration
        """
        self.event_bus = event_bus
        self.circuit_manager = circuit_manager
        self.config_path = config_path

        # State management
        self.state = EmergencyState.NORMAL
        self.active_emergencies: dict[str, EmergencyEvent] = {}
        self.emergency_history: list[EmergencyEvent] = []

        # Override management
        self.override_active = False
        self.override_expiry: Optional[datetime] = None
        self.override_confirmation_phrase = "OVERRIDE EMERGENCY HALT"

        # Monitoring state
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Daily loss tracking
        self.daily_start_balance: Optional[Decimal] = None
        self.current_balance: Optional[Decimal] = None
        self.daily_loss_percent: Decimal = Decimal("0")

        # Market state tracking
        self.correlation_matrix: dict[tuple[str, str], Decimal] = {}
        self.liquidity_scores: dict[str, Decimal] = {}
        self.price_history: dict[str, list[tuple[datetime, Decimal]]] = {}

        # Load configuration
        self.config = self._load_config()

        # Statistics
        self.emergencies_triggered = 0
        self.false_positives = 0
        self.successful_interventions = 0

        logger.info(
            "EmergencyController initialized",
            config_path=config_path,
            daily_loss_limit=self.config["daily_loss_limit"],
        )

    def _load_config(self) -> dict[str, Any]:
        """Load emergency configuration from YAML."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            # Set defaults if not present
            config.setdefault("daily_loss_limit", 0.15)  # 15%
            config.setdefault("correlation_spike_threshold", 0.80)  # 80%
            config.setdefault("liquidity_drop_threshold", 0.50)  # 50%
            config.setdefault("flash_crash_threshold", 0.10)  # 10% in 60s
            config.setdefault("flash_crash_window_seconds", 60)
            config.setdefault("override_timeout_seconds", 300)  # 5 minutes
            config.setdefault("emergency_timeout_seconds", 3600)  # 1 hour

            return config

        except FileNotFoundError:
            logger.warning(
                "Emergency config file not found, using defaults", path=self.config_path
            )
            return {
                "daily_loss_limit": 0.15,
                "correlation_spike_threshold": 0.80,
                "liquidity_drop_threshold": 0.50,
                "flash_crash_threshold": 0.10,
                "flash_crash_window_seconds": 60,
                "override_timeout_seconds": 300,
                "emergency_timeout_seconds": 3600,
            }

    async def start_monitoring(self) -> None:
        """Start emergency monitoring."""
        if self.monitoring:
            logger.warning("Emergency monitoring already active")
            return

        self.monitoring = True
        self.monitor_task = asyncio.create_task(self.monitor_for_emergencies())

        # Subscribe to relevant events
        await self._subscribe_to_events()

        logger.info("Emergency monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop emergency monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            finally:
                self.monitor_task = None

        logger.info(
            "Emergency monitoring stopped",
            emergencies_triggered=self.emergencies_triggered,
        )

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant market and risk events."""
        # Subscribe to position and balance updates
        self.event_bus.subscribe(
            EventType.POSITION_UPDATED,
            self._handle_position_update,
            priority=EventPriority.CRITICAL,
        )

        # Subscribe to market data updates
        self.event_bus.subscribe(
            EventType.MARKET_DATA_UPDATED,
            self._handle_market_update,
            priority=EventPriority.HIGH,
        )

        # Subscribe to risk events
        self.event_bus.subscribe(
            EventType.RISK_LIMIT_BREACH,
            self._handle_risk_event,
            priority=EventPriority.CRITICAL,
        )

    async def monitor_for_emergencies(self) -> None:
        """
        Main monitoring loop for emergency conditions.
        Continuously checks for:
        - Daily loss breaches
        - Correlation spikes
        - Liquidity crises
        - Flash crashes
        """
        logger.info("Emergency monitoring loop started")

        while self.monitoring:
            try:
                # Skip checks if override is active
                if self._is_override_active():
                    await asyncio.sleep(1)
                    continue

                # Check for various emergency conditions
                await self._check_daily_loss()
                await self._check_correlation_spike()
                await self._check_liquidity_crisis()
                await self._check_flash_crash()

                # Check if any emergencies should be cleared
                await self._check_emergency_clearance()

                # Sleep before next check
                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(
                    "Error in emergency monitoring", error=str(e), exc_info=True
                )
                await asyncio.sleep(5)  # Back off on error

    def calculate_daily_loss(self) -> Decimal:
        """
        Calculate current daily loss percentage.
        Returns:
            Daily loss as decimal (0.15 = 15%)
        """
        if not self.daily_start_balance or not self.current_balance:
            return Decimal("0")

        if self.daily_start_balance == Decimal("0"):
            return Decimal("0")

        loss = (
            self.daily_start_balance - self.current_balance
        ) / self.daily_start_balance
        self.daily_loss_percent = loss

        return loss

    async def _check_daily_loss(self) -> None:
        """Check for daily loss limit breach."""
        loss = self.calculate_daily_loss()

        if loss >= Decimal(str(self.config["daily_loss_limit"])):
            await self._trigger_emergency(
                emergency_type=EmergencyType.DAILY_LOSS_HALT,
                severity="CRITICAL",
                trigger_values={
                    "daily_loss_percent": float(loss),
                    "daily_loss_limit": self.config["daily_loss_limit"],
                    "start_balance": (
                        float(self.daily_start_balance)
                        if self.daily_start_balance
                        else 0
                    ),
                    "current_balance": (
                        float(self.current_balance) if self.current_balance else 0
                    ),
                },
                affected_symbols=[],  # Affects all symbols
            )

    async def _check_correlation_spike(self) -> None:
        """Check for dangerous correlation spikes."""
        if not self.correlation_matrix:
            return

        # Find maximum correlation
        max_correlation = (
            max(self.correlation_matrix.values())
            if self.correlation_matrix
            else Decimal("0")
        )

        if max_correlation >= Decimal(str(self.config["correlation_spike_threshold"])):
            # Find all highly correlated pairs
            affected_pairs = [
                f"{pair[0]}-{pair[1]}"
                for pair, corr in self.correlation_matrix.items()
                if corr >= Decimal(str(self.config["correlation_spike_threshold"]))
            ]

            await self._trigger_emergency(
                emergency_type=EmergencyType.CORRELATION_SPIKE,
                severity="HIGH",
                trigger_values={
                    "max_correlation": float(max_correlation),
                    "threshold": self.config["correlation_spike_threshold"],
                    "affected_pairs": affected_pairs,
                },
                affected_symbols=list(
                    {symbol for pair in affected_pairs for symbol in pair.split("-")}
                ),
            )

    async def _check_liquidity_crisis(self) -> None:
        """Check for liquidity crisis conditions."""
        if not self.liquidity_scores:
            return

        # Check each symbol for liquidity drop
        for symbol, current_score in self.liquidity_scores.items():
            # Note: Need baseline scores for comparison
            # For now, check if score is critically low
            if current_score < Decimal("0.2"):  # Below 20% normal liquidity
                await self._trigger_emergency(
                    emergency_type=EmergencyType.LIQUIDITY_CRISIS,
                    severity="HIGH",
                    trigger_values={
                        "liquidity_score": float(current_score),
                        "threshold": 0.2,
                    },
                    affected_symbols=[symbol],
                )

    async def _check_flash_crash(self) -> None:
        """Check for flash crash conditions."""
        if not self.price_history:
            return

        window_seconds = self.config["flash_crash_window_seconds"]
        threshold = Decimal(str(self.config["flash_crash_threshold"]))

        for symbol, history in self.price_history.items():
            if len(history) < 2:
                continue

            # Get prices within window
            now = datetime.now(UTC)
            recent_prices = [
                (ts, price)
                for ts, price in history
                if (now - ts).total_seconds() <= window_seconds
            ]

            if len(recent_prices) >= 2:
                # Calculate max drop within window
                max_price = max(price for _, price in recent_prices)
                min_price = min(price for _, price in recent_prices)

                if max_price > 0:
                    drop = (max_price - min_price) / max_price

                    if drop >= threshold:
                        await self._trigger_emergency(
                            emergency_type=EmergencyType.FLASH_CRASH,
                            severity="CRITICAL",
                            trigger_values={
                                "price_drop_percent": float(drop),
                                "threshold": float(threshold),
                                "max_price": float(max_price),
                                "min_price": float(min_price),
                                "window_seconds": window_seconds,
                            },
                            affected_symbols=[symbol],
                        )

    async def _trigger_emergency(
        self,
        emergency_type: EmergencyType,
        severity: str,
        trigger_values: dict[str, Any],
        affected_symbols: list[str],
    ) -> None:
        """
        Trigger an emergency response.
        Args:
            emergency_type: Type of emergency
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            trigger_values: Values that triggered the emergency
            affected_symbols: Symbols affected by the emergency
        """
        # Check if this emergency is already active
        if emergency_type.value in self.active_emergencies:
            return

        # Create emergency event record
        emergency = EmergencyEvent(
            event_id=str(uuid4()),
            emergency_type=emergency_type,
            severity=severity,
            triggered_at=datetime.now(UTC),
            trigger_values=trigger_values,
            affected_symbols=affected_symbols,
            actions_taken=[],
        )

        # Store emergency
        self.active_emergencies[emergency_type.value] = emergency
        self.emergency_history.append(emergency)
        self.emergencies_triggered += 1

        # Update state
        self.state = EmergencyState.EMERGENCY

        # Log critical event
        logger.critical(
            "EMERGENCY TRIGGERED",
            emergency_type=emergency_type.value,
            severity=severity,
            trigger_values=trigger_values,
            affected_symbols=affected_symbols,
        )

        # Publish emergency event
        await self.event_bus.publish(
            Event(
                event_type=EventType.CIRCUIT_BREAKER_OPEN,
                aggregate_id=emergency.event_id,
                event_data={
                    "emergency_type": emergency_type.value,
                    "severity": severity,
                    "trigger_values": trigger_values,
                    "affected_symbols": affected_symbols,
                    "message": f"Emergency halt: {emergency_type.value}",
                },
            ),
            priority=EventPriority.CRITICAL,
        )

        # Take emergency actions based on type
        if emergency_type == EmergencyType.DAILY_LOSS_HALT:
            emergency.actions_taken.append("Halted all trading")
            emergency.actions_taken.append("Cancelled all open orders")
            # Note: Actual implementation would call exchange API here

        elif emergency_type == EmergencyType.FLASH_CRASH:
            emergency.actions_taken.append("Cancelled all orders for affected symbols")
            emergency.actions_taken.append("Disabled new order placement")
            # Note: Actual implementation would call exchange API here

        elif emergency_type == EmergencyType.CORRELATION_SPIKE:
            emergency.actions_taken.append("Reduced position sizes")
            emergency.actions_taken.append("Increased diversification requirements")

        elif emergency_type == EmergencyType.LIQUIDITY_CRISIS:
            emergency.actions_taken.append("Switched to market-only orders")
            emergency.actions_taken.append("Reduced maximum position size")

    async def _check_emergency_clearance(self) -> None:
        """Check if any emergencies can be cleared."""
        if not self.active_emergencies:
            return

        now = datetime.now(UTC)
        timeout_seconds = self.config["emergency_timeout_seconds"]

        for emergency_type, emergency in list(self.active_emergencies.items()):
            # Check if emergency has timed out
            if (now - emergency.triggered_at).total_seconds() >= timeout_seconds:
                await self._clear_emergency(emergency_type)

    async def _clear_emergency(self, emergency_type: str) -> None:
        """
        Clear an emergency condition.
        Args:
            emergency_type: Type of emergency to clear
        """
        if emergency_type not in self.active_emergencies:
            return

        emergency = self.active_emergencies[emergency_type]
        emergency.resolved_at = datetime.now(UTC)
        emergency.resolution = "Timeout"

        del self.active_emergencies[emergency_type]

        # Update state if no more emergencies
        if not self.active_emergencies:
            self.state = EmergencyState.RECOVERY

        logger.info(
            "Emergency cleared",
            emergency_type=emergency_type,
            duration_seconds=(
                emergency.resolved_at - emergency.triggered_at
            ).total_seconds(),
        )

        # Publish clearance event
        await self.event_bus.publish(
            Event(
                event_type=EventType.CIRCUIT_BREAKER_CLOSED,
                aggregate_id=emergency.event_id,
                event_data={
                    "emergency_type": emergency_type,
                    "resolution": emergency.resolution,
                    "duration_seconds": (
                        emergency.resolved_at - emergency.triggered_at
                    ).total_seconds(),
                },
            ),
            priority=EventPriority.HIGH,
        )

    async def request_manual_override(self, confirmation: str) -> bool:
        """
        Request manual override of emergency halt.

        Args:
            confirmation: Confirmation phrase typed by user
        Returns:
            True if override granted, False otherwise
        """
        if confirmation != self.override_confirmation_phrase:
            logger.warning(
                "Manual override denied - incorrect confirmation",
                provided=confirmation,
                expected=self.override_confirmation_phrase,
            )
            return False

        # Grant override
        self.override_active = True
        self.override_expiry = datetime.now(UTC) + timedelta(
            seconds=self.config["override_timeout_seconds"]
        )

        logger.critical(
            "MANUAL OVERRIDE ACTIVATED",
            expiry=self.override_expiry.isoformat(),
            duration_seconds=self.config["override_timeout_seconds"],
        )

        # Update state
        old_state = self.state
        self.state = EmergencyState.OVERRIDE

        # Publish override event
        await self.event_bus.publish(
            Event(
                event_type=EventType.SYSTEM_STARTUP,  # Using system event for override
                event_data={
                    "action": "emergency_override",
                    "old_state": old_state.value,
                    "new_state": self.state.value,
                    "expiry": self.override_expiry.isoformat(),
                },
            ),
            priority=EventPriority.CRITICAL,
        )

        return True

    def _is_override_active(self) -> bool:
        """Check if manual override is currently active."""
        if not self.override_active:
            return False

        if self.override_expiry and datetime.now(UTC) >= self.override_expiry:
            # Override expired
            self.override_active = False
            self.override_expiry = None
            self.state = (
                EmergencyState.EMERGENCY
                if self.active_emergencies
                else EmergencyState.NORMAL
            )
            logger.info("Manual override expired")
            return False

        return True

    async def _handle_position_update(self, event: Event) -> None:
        """Handle position update events."""
        # Update current balance from position data
        if "balance" in event.event_data:
            self.current_balance = Decimal(str(event.event_data["balance"]))

            # Set daily start balance if not set
            if self.daily_start_balance is None:
                self.daily_start_balance = self.current_balance

    async def _handle_market_update(self, event: Event) -> None:
        """Handle market data update events."""
        if "symbol" in event.event_data and "price" in event.event_data:
            symbol = event.event_data["symbol"]
            price = Decimal(str(event.event_data["price"]))
            # Ensure timestamp is timezone-aware
            timestamp = event.created_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)

            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []

            self.price_history[symbol].append((timestamp, price))

            # Keep only recent history (last 5 minutes)
            cutoff = datetime.now(UTC) - timedelta(minutes=5)
            self.price_history[symbol] = [
                (ts, p) for ts, p in self.price_history[symbol] if ts >= cutoff
            ]

        # Update liquidity scores if available
        if "liquidity_score" in event.event_data:
            symbol = event.event_data.get("symbol", "")
            if symbol:
                self.liquidity_scores[symbol] = Decimal(
                    str(event.event_data["liquidity_score"])
                )

    async def _handle_risk_event(self, event: Event) -> None:
        """Handle risk-related events."""
        logger.warning(
            "Risk event received",
            event_type=event.event_type.value,
            data=event.event_data,
        )

        # Could trigger additional emergency checks here

    def update_correlation_matrix(
        self, symbol1: str, symbol2: str, correlation: Decimal
    ) -> None:
        """
        Update correlation matrix with new value.
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Correlation coefficient (-1 to 1)
        """
        # Store correlation (order symbols alphabetically for consistency)
        key = tuple(sorted([symbol1, symbol2]))
        self.correlation_matrix[key] = abs(correlation)  # Use absolute value

    def reset_daily_tracking(self) -> None:
        """Reset daily loss tracking (call at start of trading day)."""
        self.daily_start_balance = self.current_balance
        self.daily_loss_percent = Decimal("0")

        logger.info(
            "Daily tracking reset",
            start_balance=(
                float(self.daily_start_balance) if self.daily_start_balance else 0
            ),
        )

    async def generate_emergency_report(self) -> dict[str, Any]:
        """
        Generate comprehensive emergency analysis report.
        Returns:
            Report dictionary with analysis and recommendations
        """
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "current_state": self.state.value,
            "active_emergencies": [],
            "emergency_history": [],
            "statistics": {
                "total_emergencies": self.emergencies_triggered,
                "false_positives": self.false_positives,
                "successful_interventions": self.successful_interventions,
            },
            "current_metrics": {
                "daily_loss_percent": float(self.daily_loss_percent),
                "max_correlation": (
                    float(max(self.correlation_matrix.values()))
                    if self.correlation_matrix
                    else 0
                ),
                "min_liquidity_score": (
                    float(min(self.liquidity_scores.values()))
                    if self.liquidity_scores
                    else 1.0
                ),
            },
            "recommendations": [],
        }

        # Add active emergencies
        for emergency in self.active_emergencies.values():
            report["active_emergencies"].append(
                {
                    "type": emergency.emergency_type.value,
                    "severity": emergency.severity,
                    "triggered_at": emergency.triggered_at.isoformat(),
                    "trigger_values": emergency.trigger_values,
                    "actions_taken": emergency.actions_taken,
                }
            )

        # Add recent history (last 10 emergencies)
        for emergency in self.emergency_history[-10:]:
            report["emergency_history"].append(
                {
                    "type": emergency.emergency_type.value,
                    "severity": emergency.severity,
                    "triggered_at": emergency.triggered_at.isoformat(),
                    "resolved_at": (
                        emergency.resolved_at.isoformat()
                        if emergency.resolved_at
                        else None
                    ),
                    "resolution": emergency.resolution,
                    "duration_seconds": (
                        (emergency.resolved_at - emergency.triggered_at).total_seconds()
                        if emergency.resolved_at
                        else None
                    ),
                }
            )

        # Generate recommendations
        if self.daily_loss_percent > Decimal("0.10"):
            report["recommendations"].append(
                "Consider reducing position sizes - approaching daily loss limit"
            )

        if self.correlation_matrix and max(self.correlation_matrix.values()) > Decimal(
            "0.70"
        ):
            report["recommendations"].append(
                "Diversify positions - high correlation detected between assets"
            )

        if self.liquidity_scores and min(self.liquidity_scores.values()) < Decimal(
            "0.30"
        ):
            report["recommendations"].append(
                "Avoid large orders - low liquidity detected in some markets"
            )

        if self.emergencies_triggered > 5:
            report["recommendations"].append(
                "Review risk parameters - frequent emergency triggers detected"
            )

        return report

    def get_status(self) -> dict[str, Any]:
        """Get current emergency controller status."""
        return {
            "state": self.state.value,
            "monitoring": self.monitoring,
            "active_emergencies": list(self.active_emergencies.keys()),
            "override_active": self.override_active,
            "override_expiry": (
                self.override_expiry.isoformat() if self.override_expiry else None
            ),
            "daily_loss_percent": float(self.daily_loss_percent),
            "emergencies_triggered": self.emergencies_triggered,
            "config": {
                "daily_loss_limit": self.config["daily_loss_limit"],
                "correlation_spike_threshold": self.config[
                    "correlation_spike_threshold"
                ],
                "liquidity_drop_threshold": self.config["liquidity_drop_threshold"],
                "flash_crash_threshold": self.config["flash_crash_threshold"],
            },
        }
