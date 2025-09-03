"""VWAP Execution Strategy for large order execution with minimal market impact."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import structlog

from genesis.core.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
)
from genesis.strategies.base import BaseStrategy, StrategyConfig, StrategyState

logger = structlog.get_logger(__name__)


class UrgencyLevel(str, Enum):
    """Execution urgency levels for VWAP orders."""

    LOW = "LOW"  # 0-25%: Follow volume curve strictly
    MEDIUM = "MEDIUM"  # 26-50%: Moderate acceleration, up to 1.5x normal rate
    HIGH = "HIGH"  # 51-75%: Aggressive execution, up to 2x normal rate
    CRITICAL = "CRITICAL"  # 76-100%: Maximum speed within participation limits
    EMERGENCY = "EMERGENCY"  # Override all constraints, immediate execution


@dataclass
class VWAPOrderConfig:
    """Configuration for a VWAP parent order."""

    parent_order_id: UUID = field(default_factory=uuid4)
    symbol: str = "BTCUSDT"
    side: OrderSide = OrderSide.BUY
    total_quantity: Decimal = Decimal("0")
    target_participation_rate: Decimal = Decimal("0.10")  # Max 10% of volume
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime = field(
        default_factory=lambda: datetime.now(UTC) + timedelta(hours=1)
    )
    min_slice_size: Decimal = Decimal("0.001")  # Exchange minimum
    max_slice_size: Decimal = Decimal("1.0")
    allow_partial: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VWAPExecutionState:
    """Runtime state for VWAP execution."""

    executed_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    vwap_benchmark: Decimal = Decimal("0")
    implementation_shortfall: Decimal = Decimal("0")
    child_orders: List[Order] = field(default_factory=list)
    completed_orders: List[Order] = field(default_factory=list)
    schedule: List[Dict[str, Any]] = field(default_factory=list)
    current_slice_index: int = 0
    total_slices: int = 0
    market_impact_bps: Decimal = Decimal("0")
    execution_score: Decimal = Decimal("0")
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class VWAPExecutionConfig(StrategyConfig):
    """Configuration for VWAP execution strategy."""

    max_participation_rate: Decimal = Decimal("0.10")  # 10% max
    default_urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    volume_lookback_days: int = 20
    volume_curve_intervals: int = 48  # 30-minute intervals for 24 hours
    adaptive_scheduling: bool = True
    anti_gaming_enabled: bool = True
    shortfall_threshold_bps: Decimal = Decimal("50")  # 50 basis points
    emergency_liquidation_threshold: Decimal = Decimal(
        "0.90"
    )  # 90% urgency triggers emergency


class VWAPExecutionStrategy(BaseStrategy):
    """VWAP execution algorithm with intraday volume curves."""

    def __init__(self, config: VWAPExecutionConfig | None = None):
        """Initialize VWAP execution strategy."""
        super().__init__(
            config
            or VWAPExecutionConfig(name="VWAPExecution", tier_required="STRATEGIST")
        )
        self.config: VWAPExecutionConfig = self.config
        self.parent_orders: Dict[UUID, VWAPOrderConfig] = {}
        self.order_states: Dict[UUID, VWAPExecutionState] = {}
        self.volume_curves: Dict[str, List[Decimal]] = {}
        self.historical_volumes: Dict[str, List[Dict[str, Any]]] = {}

    async def generate_signals(self) -> list[Signal]:
        """Generate trading signals based on VWAP execution schedule.

        Returns:
            List of signals for child orders based on schedule.
        """
        signals = []

        for parent_id, parent_config in self.parent_orders.items():
            state = self.order_states.get(parent_id)
            if not state or state.remaining_quantity <= 0:
                continue

            # Check if we need to execute next slice
            if await self._should_execute_slice(parent_id):
                signal = await self._generate_slice_signal(parent_id)
                if signal:
                    signals.append(signal)

        return signals

    async def analyze(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze market data and generate trading signal.

        Args:
            market_data: Market data from analyzer.

        Returns:
            Trading signal or None if no opportunity.
        """
        # VWAP strategy generates signals based on schedule, not market analysis
        # This method is here for base class compatibility
        return None

    async def manage_positions(self) -> list[Signal]:
        """Manage existing positions and generate exit signals.

        Returns:
            List of exit signals for position management.
        """
        # VWAP manages orders, not positions directly
        return []

    async def create_parent_order(self, config: VWAPOrderConfig) -> UUID:
        """Create a new parent order for VWAP execution.

        Args:
            config: Configuration for the parent order.

        Returns:
            Parent order ID.
        """
        parent_id = config.parent_order_id
        self.parent_orders[parent_id] = config

        # Initialize execution state
        state = VWAPExecutionState(
            remaining_quantity=config.total_quantity, executed_quantity=Decimal("0")
        )
        self.order_states[parent_id] = state

        # Generate execution schedule
        await self._generate_execution_schedule(parent_id)

        logger.info(
            "Created VWAP parent order",
            parent_id=str(parent_id),
            symbol=config.symbol,
            quantity=str(config.total_quantity),
            urgency=config.urgency.value,
        )

        return parent_id

    async def _should_execute_slice(self, parent_id: UUID) -> bool:
        """Check if next slice should be executed.

        Args:
            parent_id: Parent order ID.

        Returns:
            True if slice should be executed.
        """
        config = self.parent_orders.get(parent_id)
        state = self.order_states.get(parent_id)

        if not config or not state:
            return False

        # Check if we have schedule
        if not state.schedule or state.current_slice_index >= len(state.schedule):
            return False

        # Get current slice from schedule
        current_slice = state.schedule[state.current_slice_index]

        # Check if it's time to execute
        now = datetime.now(UTC)
        slice_time = current_slice.get("execution_time")

        if isinstance(slice_time, str):
            slice_time = datetime.fromisoformat(slice_time)

        return now >= slice_time

    async def _generate_slice_signal(self, parent_id: UUID) -> Signal | None:
        """Generate signal for next slice execution.

        Args:
            parent_id: Parent order ID.

        Returns:
            Signal for child order or None.
        """
        config = self.parent_orders.get(parent_id)
        state = self.order_states.get(parent_id)

        if not config or not state:
            return None

        # Get current slice
        if state.current_slice_index >= len(state.schedule):
            return None

        slice_info = state.schedule[state.current_slice_index]

        # Create signal for child order
        signal = Signal(
            strategy_id=str(self.strategy_id),
            symbol=config.symbol,
            signal_type=SignalType.BUY
            if config.side == OrderSide.BUY
            else SignalType.SELL,
            quantity=Decimal(str(slice_info["quantity"])),
            confidence=Decimal("0.95"),  # High confidence for scheduled execution
            metadata={
                "parent_order_id": str(parent_id),
                "slice_number": state.current_slice_index + 1,
                "total_slices": state.total_slices,
                "urgency": config.urgency.value,
                "execution_type": "VWAP",
            },
        )

        # Update state
        state.current_slice_index += 1

        logger.info(
            "Generated slice signal",
            parent_id=str(parent_id),
            slice=state.current_slice_index,
            quantity=str(slice_info["quantity"]),
        )

        return signal

    async def _generate_execution_schedule(self, parent_id: UUID) -> None:
        """Generate execution schedule based on volume curve.

        Args:
            parent_id: Parent order ID.
        """
        config = self.parent_orders.get(parent_id)
        state = self.order_states.get(parent_id)

        if not config or not state:
            return

        # Get or estimate volume curve
        volume_curve = await self._get_volume_curve(config.symbol)

        # Calculate time intervals
        duration = config.end_time - config.start_time
        num_slices = self._calculate_slice_count(config, duration)

        # Generate schedule based on urgency
        schedule = await self._create_schedule(
            config=config, volume_curve=volume_curve, num_slices=num_slices
        )

        state.schedule = schedule
        state.total_slices = num_slices

        logger.info(
            "Generated execution schedule",
            parent_id=str(parent_id),
            slices=num_slices,
            duration_hours=duration.total_seconds() / 3600,
        )

    async def _get_volume_curve(self, symbol: str) -> List[Decimal]:
        """Get or estimate volume curve for symbol.

        Args:
            symbol: Trading symbol.

        Returns:
            List of volume percentages by time interval.
        """
        # Check cache
        if symbol in self.volume_curves:
            return self.volume_curves[symbol]

        # Generate default intraday volume curve (U-shaped)
        intervals = self.config.volume_curve_intervals
        curve = []

        for i in range(intervals):
            # U-shaped curve: higher at open/close, lower mid-day
            hour = (i * 24) / intervals
            if hour < 2:  # Opening hours
                volume_pct = Decimal("0.08")
            elif hour < 4:
                volume_pct = Decimal("0.05")
            elif hour < 10:
                volume_pct = Decimal("0.03")
            elif hour < 14:
                volume_pct = Decimal("0.02")
            elif hour < 20:
                volume_pct = Decimal("0.03")
            elif hour < 22:
                volume_pct = Decimal("0.05")
            else:  # Closing hours
                volume_pct = Decimal("0.08")

            curve.append(volume_pct)

        # Normalize to sum to 1
        total = sum(curve)
        curve = [v / total for v in curve]

        self.volume_curves[symbol] = curve
        return curve

    def _calculate_slice_count(
        self, config: VWAPOrderConfig, duration: timedelta
    ) -> int:
        """Calculate optimal number of slices.

        Args:
            config: Parent order configuration.
            duration: Execution duration.

        Returns:
            Number of slices.
        """
        # Base calculation on duration and urgency
        hours = duration.total_seconds() / 3600

        if config.urgency == UrgencyLevel.EMERGENCY:
            return 1  # Single immediate execution
        elif config.urgency == UrgencyLevel.CRITICAL:
            return max(1, int(hours * 2))  # 2 slices per hour
        elif config.urgency == UrgencyLevel.HIGH:
            return max(1, int(hours * 4))  # 4 slices per hour
        elif config.urgency == UrgencyLevel.MEDIUM:
            return max(1, int(hours * 6))  # 6 slices per hour
        else:  # LOW
            return max(1, int(hours * 12))  # 12 slices per hour

    async def _create_schedule(
        self, config: VWAPOrderConfig, volume_curve: List[Decimal], num_slices: int
    ) -> List[Dict[str, Any]]:
        """Create execution schedule.

        Args:
            config: Parent order configuration.
            volume_curve: Volume distribution curve.
            num_slices: Number of slices.

        Returns:
            List of scheduled executions.
        """
        schedule = []
        remaining = config.total_quantity

        # Calculate slice times
        duration = config.end_time - config.start_time
        slice_duration = duration / num_slices

        for i in range(num_slices):
            # Calculate execution time
            exec_time = config.start_time + (slice_duration * i)

            # Calculate quantity based on volume curve and urgency
            base_quantity = config.total_quantity / num_slices

            # Adjust for urgency
            urgency_multiplier = self._get_urgency_multiplier(config.urgency)
            quantity = min(
                base_quantity * urgency_multiplier, remaining, config.max_slice_size
            )

            # Ensure minimum size
            quantity = max(quantity, config.min_slice_size)

            schedule.append(
                {
                    "slice_number": i + 1,
                    "execution_time": exec_time.isoformat(),
                    "quantity": float(quantity),
                    "urgency": config.urgency.value,
                }
            )

            remaining -= quantity
            if remaining <= 0:
                break

        return schedule

    def _get_urgency_multiplier(self, urgency: UrgencyLevel) -> Decimal:
        """Get execution rate multiplier based on urgency.

        Args:
            urgency: Urgency level.

        Returns:
            Multiplier for execution rate.
        """
        multipliers = {
            UrgencyLevel.LOW: Decimal("1.0"),
            UrgencyLevel.MEDIUM: Decimal("1.5"),
            UrgencyLevel.HIGH: Decimal("2.0"),
            UrgencyLevel.CRITICAL: Decimal("3.0"),
            UrgencyLevel.EMERGENCY: Decimal("10.0"),  # Execute immediately
        }
        return multipliers.get(urgency, Decimal("1.0"))

    async def update_child_order(self, parent_id: UUID, child_order: Order) -> None:
        """Update child order status.

        Args:
            parent_id: Parent order ID.
            child_order: Child order with updated status.
        """
        state = self.order_states.get(parent_id)
        if not state:
            return

        # Update state based on order status
        if child_order.status == OrderStatus.FILLED:
            state.executed_quantity += child_order.filled_quantity
            state.remaining_quantity -= child_order.filled_quantity
            state.completed_orders.append(child_order)

            # Update average price
            if state.executed_quantity > 0:
                total_value = sum(
                    o.filled_quantity * (o.price or Decimal("0"))
                    for o in state.completed_orders
                )
                state.average_price = total_value / state.executed_quantity

        state.child_orders.append(child_order)
        state.last_update = datetime.now(UTC)

    async def calculate_vwap_benchmark(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> Decimal:
        """Calculate VWAP benchmark from market data.

        Args:
            symbol: Trading symbol.
            start_time: Start of calculation period.
            end_time: End of calculation period.

        Returns:
            VWAP benchmark price.
        """
        # This would fetch actual market data in production
        # For now, return a placeholder
        return Decimal("50000")  # Placeholder for BTC price

    async def calculate_implementation_shortfall(self, parent_id: UUID) -> Decimal:
        """Calculate implementation shortfall for parent order.

        Args:
            parent_id: Parent order ID.

        Returns:
            Implementation shortfall in basis points.
        """
        config = self.parent_orders.get(parent_id)
        state = self.order_states.get(parent_id)

        if not config or not state or state.executed_quantity == 0:
            return Decimal("0")

        # Calculate benchmark VWAP
        benchmark = await self.calculate_vwap_benchmark(
            config.symbol, config.start_time, datetime.now(UTC)
        )

        # Calculate shortfall
        if benchmark > 0:
            shortfall_pct = ((state.average_price - benchmark) / benchmark) * Decimal(
                "10000"
            )  # In bps
            state.implementation_shortfall = shortfall_pct
            return shortfall_pct

        return Decimal("0")

    def enforce_participation_limits(
        self, quantity: Decimal, market_volume: Decimal
    ) -> Decimal:
        """Enforce participation rate limits.

        Args:
            quantity: Desired order quantity.
            market_volume: Current market volume.

        Returns:
            Adjusted quantity within participation limits.
        """
        max_quantity = market_volume * self.config.max_participation_rate
        return min(quantity, max_quantity)

    async def trigger_emergency_liquidation(self, parent_id: UUID) -> None:
        """Trigger emergency liquidation mode for parent order.

        Args:
            parent_id: Parent order ID.
        """
        config = self.parent_orders.get(parent_id)
        state = self.order_states.get(parent_id)

        if not config or not state:
            return

        logger.warning(
            "Emergency liquidation triggered",
            parent_id=str(parent_id),
            remaining=str(state.remaining_quantity),
        )

        # Update urgency to emergency
        config.urgency = UrgencyLevel.EMERGENCY

        # Clear existing schedule
        state.schedule = []
        state.current_slice_index = 0

        # Create single immediate execution
        state.schedule = [
            {
                "slice_number": 1,
                "execution_time": datetime.now(UTC).isoformat(),
                "quantity": float(state.remaining_quantity),
                "urgency": UrgencyLevel.EMERGENCY.value,
            }
        ]
        state.total_slices = 1
    
    async def on_order_filled(self, order: Order) -> None:
        """Handle order fill event.
        
        Args:
            order: The filled order.
        """
        # Find parent order from metadata
        metadata = order.metadata or {}
        parent_id_str = metadata.get("parent_order_id")
        
        if parent_id_str:
            try:
                parent_id = UUID(parent_id_str)
                await self.update_child_order(parent_id, order)
            except ValueError:
                logger.error("Invalid parent order ID", parent_id=parent_id_str)
        
        logger.info(
            "Order filled",
            order_id=str(order.order_id),
            symbol=order.symbol,
            quantity=str(order.filled_quantity),
            price=str(order.price)
        )
    
    async def on_position_closed(self, position: Position) -> None:
        """Handle position close event.
        
        Args:
            position: The closed position.
        """
        # VWAP strategy doesn't directly manage positions
        # Log for monitoring purposes
        logger.info(
            "Position closed (VWAP monitoring)",
            position_id=position.position_id,
            symbol=position.symbol,
            pnl=str(position.pnl_dollars)
        )
