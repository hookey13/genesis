"""
Rapid deleveraging protocol for emergency position reduction.

Manages progressive position closure during extreme risk events,
prioritizing positions based on risk metrics and ensuring orderly
unwinding to minimize losses.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class DeleveragingStage(str, Enum):
    """Stages of deleveraging process."""

    IDLE = "idle"  # No deleveraging active
    STAGE_1 = "stage_1"  # 25% reduction
    STAGE_2 = "stage_2"  # 50% reduction
    STAGE_3 = "stage_3"  # 75% reduction
    STAGE_4 = "stage_4"  # 100% reduction (full closure)


class ClosurePriority(str, Enum):
    """Position closure priority levels."""

    CRITICAL = "critical"  # Close immediately
    HIGH = "high"  # Close urgently
    MEDIUM = "medium"  # Close soon
    LOW = "low"  # Close when convenient


@dataclass
class Position:
    """Trading position details."""

    symbol: str
    side: str  # "long" or "short"
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    margin_used: Decimal
    opened_at: datetime

    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return Decimal("0")

        if self.side == "long":
            return ((self.current_price - self.entry_price) / self.entry_price) * Decimal("100")
        else:  # short
            return ((self.entry_price - self.current_price) / self.entry_price) * Decimal("100")

    @property
    def position_value(self) -> Decimal:
        """Calculate current position value."""
        return self.size * self.current_price


@dataclass
class DeleveragingPlan:
    """Plan for position reduction."""

    position: Position
    priority: ClosurePriority
    reduction_percentage: Decimal
    reduction_size: Decimal
    expected_slippage: Decimal
    execution_order: int  # Order in which to execute
    reason: str


@dataclass
class DeleveragingResult:
    """Result of deleveraging execution."""

    symbol: str
    original_size: Decimal
    closed_size: Decimal
    remaining_size: Decimal
    execution_price: Decimal
    slippage: Decimal
    realized_pnl: Decimal
    success: bool
    error_message: Optional[str] = None


class DeleveragingProtocol:
    """
    Manages emergency position reduction during crisis events.

    Features:
    - Progressive position reduction (25%, 50%, 75%, 100%)
    - Intelligent position prioritization
    - Slippage estimation and management
    - Real-time progress tracking
    """

    def __init__(
        self,
        event_bus: EventBus,
        max_slippage_pct: Decimal = Decimal("0.02"),  # 2% max slippage
        execution_delay_ms: int = 100  # Delay between orders
    ):
        """
        Initialize deleveraging protocol.

        Args:
            event_bus: Event bus for publishing updates
            max_slippage_pct: Maximum acceptable slippage
            execution_delay_ms: Delay between order executions
        """
        self.event_bus = event_bus
        self.max_slippage_pct = max_slippage_pct
        self.execution_delay_ms = execution_delay_ms

        # Current state
        self.current_stage = DeleveragingStage.IDLE
        self.active_deleveraging = False
        self.emergency_halt = False

        # Positions and plans
        self.positions: dict[str, Position] = {}
        self.deleveraging_plans: list[DeleveragingPlan] = []
        self.execution_queue: list[DeleveragingPlan] = []

        # Progress tracking
        self.total_positions = 0
        self.positions_closed = 0
        self.total_value_reduced = Decimal("0")
        self.total_realized_pnl = Decimal("0")

        # History
        self.deleveraging_history: list[DeleveragingResult] = []

        # Statistics
        self.deleveraging_events = 0
        self.positions_force_closed = 0
        self.total_slippage_cost = Decimal("0")

        logger.info(
            "DeleveragingProtocol initialized",
            max_slippage_pct=float(max_slippage_pct),
            execution_delay_ms=execution_delay_ms
        )

    def update_positions(self, positions: list[Position]) -> None:
        """
        Update current positions.

        Args:
            positions: List of current positions
        """
        self.positions = {p.symbol: p for p in positions}
        self.total_positions = len(positions)

        logger.debug(
            "Positions updated",
            count=len(positions),
            total_value=float(sum(p.position_value for p in positions))
        )

    def calculate_position_priorities(self) -> list[tuple[Position, ClosurePriority]]:
        """
        Calculate closure priority for each position.

        Priority factors:
        - Loss percentage (losing positions closed first)
        - Position age (newer positions closed first)
        - Position size (larger positions reduced first)
        - Correlation risk (highly correlated positions prioritized)

        Returns:
            List of positions with priorities
        """
        priorities = []

        for position in self.positions.values():
            # Start with medium priority
            priority = ClosurePriority.MEDIUM

            # Check P&L
            pnl_pct = position.pnl_percentage
            if pnl_pct < Decimal("-10"):
                priority = ClosurePriority.CRITICAL  # Large loss
            elif pnl_pct < Decimal("-5"):
                priority = ClosurePriority.HIGH  # Moderate loss
            elif pnl_pct > Decimal("5"):
                priority = ClosurePriority.LOW  # Profitable position

            # Check position age (newer = higher priority for closure)
            age_hours = (datetime.now(UTC) - position.opened_at).total_seconds() / 3600
            if age_hours < 1:
                # Very new position, might be a panic trade
                if priority == ClosurePriority.MEDIUM:
                    priority = ClosurePriority.HIGH

            # Check position size relative to others
            if self.total_positions > 0:
                avg_size = sum(p.position_value for p in self.positions.values()) / len(self.positions)
                if position.position_value > avg_size * Decimal("2"):
                    # Oversized position
                    if priority in [ClosurePriority.MEDIUM, ClosurePriority.LOW]:
                        priority = ClosurePriority.HIGH

            priorities.append((position, priority))

        # Sort by priority and P&L
        priorities.sort(
            key=lambda x: (
                self._priority_to_int(x[1]),  # Priority level
                x[0].pnl_percentage  # P&L (losses first)
            )
        )

        return priorities

    def _priority_to_int(self, priority: ClosurePriority) -> int:
        """Convert priority to integer for sorting."""
        return {
            ClosurePriority.CRITICAL: 0,
            ClosurePriority.HIGH: 1,
            ClosurePriority.MEDIUM: 2,
            ClosurePriority.LOW: 3
        }[priority]

    async def initiate_deleveraging(
        self,
        stage: DeleveragingStage,
        reason: str = "Emergency deleveraging"
    ) -> list[DeleveragingPlan]:
        """
        Initiate deleveraging at specified stage.

        Args:
            stage: Target deleveraging stage
            reason: Reason for deleveraging

        Returns:
            List of deleveraging plans
        """
        if self.active_deleveraging:
            logger.warning("Deleveraging already active", current_stage=self.current_stage.value)
            return []

        self.active_deleveraging = True
        self.current_stage = stage
        self.deleveraging_events += 1

        logger.critical(
            "DELEVERAGING INITIATED",
            stage=stage.value,
            reason=reason,
            positions_count=len(self.positions)
        )

        # Calculate reduction percentage based on stage
        reduction_pct = {
            DeleveragingStage.STAGE_1: Decimal("0.25"),
            DeleveragingStage.STAGE_2: Decimal("0.50"),
            DeleveragingStage.STAGE_3: Decimal("0.75"),
            DeleveragingStage.STAGE_4: Decimal("1.00")
        }.get(stage, Decimal("0"))

        # Get position priorities
        priorities = self.calculate_position_priorities()

        # Create deleveraging plans
        self.deleveraging_plans = []

        for i, (position, priority) in enumerate(priorities):
            # Calculate reduction size
            reduction_size = position.size * reduction_pct

            # Estimate slippage based on priority and market conditions
            if priority == ClosurePriority.CRITICAL:
                expected_slippage = self.max_slippage_pct
            elif priority == ClosurePriority.HIGH:
                expected_slippage = self.max_slippage_pct * Decimal("0.75")
            elif priority == ClosurePriority.MEDIUM:
                expected_slippage = self.max_slippage_pct * Decimal("0.50")
            else:
                expected_slippage = self.max_slippage_pct * Decimal("0.25")

            plan = DeleveragingPlan(
                position=position,
                priority=priority,
                reduction_percentage=reduction_pct,
                reduction_size=reduction_size,
                expected_slippage=expected_slippage,
                execution_order=i,
                reason=reason
            )

            self.deleveraging_plans.append(plan)

        # Sort plans by execution order
        self.execution_queue = sorted(
            self.deleveraging_plans,
            key=lambda p: p.execution_order
        )

        # Publish deleveraging event
        await self.event_bus.publish(
            Event(
                event_type=EventType.POSITION_SIZE_ADJUSTMENT,
                event_data={
                    "action": "deleveraging_initiated",
                    "stage": stage.value,
                    "reduction_percentage": float(reduction_pct),
                    "positions_affected": len(self.deleveraging_plans),
                    "reason": reason
                }
            ),
            priority=EventPriority.CRITICAL
        )

        return self.deleveraging_plans

    async def execute_deleveraging(self) -> list[DeleveragingResult]:
        """
        Execute the deleveraging plan.

        Returns:
            List of execution results
        """
        if not self.execution_queue:
            logger.warning("No deleveraging plans to execute")
            return []

        results = []

        for plan in self.execution_queue:
            if self.emergency_halt:
                logger.warning("Deleveraging halted", symbol=plan.position.symbol)
                break

            # Execute position reduction
            result = await self._execute_position_reduction(plan)
            results.append(result)

            # Update progress
            if result.success:
                self.positions_closed += 1
                self.total_value_reduced += result.closed_size * plan.position.current_price
                self.total_realized_pnl += result.realized_pnl
                self.total_slippage_cost += result.slippage * result.closed_size

            # Publish progress update
            await self._publish_progress_update(plan, result)

            # Delay between executions to avoid overwhelming the market
            await asyncio.sleep(self.execution_delay_ms / 1000)

        # Mark deleveraging complete
        self.active_deleveraging = False

        # Publish completion event
        await self._publish_completion_event(results)

        return results

    async def _execute_position_reduction(
        self,
        plan: DeleveragingPlan
    ) -> DeleveragingResult:
        """
        Execute a single position reduction.

        Args:
            plan: Deleveraging plan to execute

        Returns:
            Execution result
        """
        position = plan.position

        try:
            # Simulate order execution (would integrate with exchange)
            # In production, this would submit market orders to the exchange

            # Calculate execution price with slippage
            if position.side == "long":
                # Selling long position, price goes down with slippage
                execution_price = position.current_price * (Decimal("1") - plan.expected_slippage)
            else:
                # Buying back short position, price goes up with slippage
                execution_price = position.current_price * (Decimal("1") + plan.expected_slippage)

            # Calculate realized P&L
            if position.side == "long":
                realized_pnl = (execution_price - position.entry_price) * plan.reduction_size
            else:
                realized_pnl = (position.entry_price - execution_price) * plan.reduction_size

            # Create result
            result = DeleveragingResult(
                symbol=position.symbol,
                original_size=position.size,
                closed_size=plan.reduction_size,
                remaining_size=position.size - plan.reduction_size,
                execution_price=execution_price,
                slippage=abs(execution_price - position.current_price) / position.current_price,
                realized_pnl=realized_pnl,
                success=True
            )

            # Update position
            position.size = result.remaining_size
            if position.size == 0:
                del self.positions[position.symbol]

            # Store in history
            self.deleveraging_history.append(result)
            self.positions_force_closed += 1

            logger.info(
                "Position reduced",
                symbol=position.symbol,
                closed_size=float(plan.reduction_size),
                remaining_size=float(result.remaining_size),
                realized_pnl=float(realized_pnl)
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to reduce position",
                symbol=position.symbol,
                error=str(e),
                exc_info=True
            )

            return DeleveragingResult(
                symbol=position.symbol,
                original_size=position.size,
                closed_size=Decimal("0"),
                remaining_size=position.size,
                execution_price=position.current_price,
                slippage=Decimal("0"),
                realized_pnl=Decimal("0"),
                success=False,
                error_message=str(e)
            )

    async def _publish_progress_update(
        self,
        plan: DeleveragingPlan,
        result: DeleveragingResult
    ) -> None:
        """
        Publish deleveraging progress update.

        Args:
            plan: Executed plan
            result: Execution result
        """
        progress_pct = (self.positions_closed / self.total_positions * 100) if self.total_positions > 0 else 0

        await self.event_bus.publish(
            Event(
                event_type=EventType.POSITION_UPDATED,
                aggregate_id=result.symbol,
                event_data={
                    "action": "deleveraging_progress",
                    "symbol": result.symbol,
                    "closed_size": float(result.closed_size),
                    "remaining_size": float(result.remaining_size),
                    "realized_pnl": float(result.realized_pnl),
                    "progress_percentage": progress_pct,
                    "positions_closed": self.positions_closed,
                    "positions_total": self.total_positions
                }
            ),
            priority=EventPriority.HIGH
        )

    async def _publish_completion_event(self, results: list[DeleveragingResult]) -> None:
        """
        Publish deleveraging completion event.

        Args:
            results: All execution results
        """
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        await self.event_bus.publish(
            Event(
                event_type=EventType.POSITION_SIZE_ADJUSTMENT,
                event_data={
                    "action": "deleveraging_completed",
                    "stage": self.current_stage.value,
                    "positions_processed": len(results),
                    "successful": successful,
                    "failed": failed,
                    "total_value_reduced": float(self.total_value_reduced),
                    "total_realized_pnl": float(self.total_realized_pnl),
                    "total_slippage_cost": float(self.total_slippage_cost)
                }
            ),
            priority=EventPriority.HIGH
        )

        logger.info(
            "Deleveraging completed",
            stage=self.current_stage.value,
            successful=successful,
            failed=failed,
            realized_pnl=float(self.total_realized_pnl)
        )

    def halt_deleveraging(self) -> None:
        """Emergency halt of deleveraging process."""
        self.emergency_halt = True
        logger.critical("Deleveraging HALTED by emergency stop")

    def resume_deleveraging(self) -> None:
        """Resume halted deleveraging."""
        self.emergency_halt = False
        logger.info("Deleveraging resumed")

    def get_deleveraging_report(self) -> dict[str, Any]:
        """
        Generate deleveraging status report.

        Returns:
            Status report dictionary
        """
        return {
            "current_stage": self.current_stage.value,
            "active": self.active_deleveraging,
            "halted": self.emergency_halt,
            "positions_remaining": len(self.positions),
            "positions_closed": self.positions_closed,
            "total_value_reduced": float(self.total_value_reduced),
            "total_realized_pnl": float(self.total_realized_pnl),
            "total_slippage_cost": float(self.total_slippage_cost),
            "plans_pending": len(self.execution_queue),
            "statistics": {
                "deleveraging_events": self.deleveraging_events,
                "positions_force_closed": self.positions_force_closed,
                "average_slippage": float(
                    self.total_slippage_cost / self.positions_force_closed
                ) if self.positions_force_closed > 0 else 0
            }
        }

    def reset(self) -> None:
        """Reset protocol state (useful for testing)."""
        self.current_stage = DeleveragingStage.IDLE
        self.active_deleveraging = False
        self.emergency_halt = False
        self.positions.clear()
        self.deleveraging_plans.clear()
        self.execution_queue.clear()
        self.deleveraging_history.clear()
        self.total_positions = 0
        self.positions_closed = 0
        self.total_value_reduced = Decimal("0")
        self.total_realized_pnl = Decimal("0")
        self.deleveraging_events = 0
        self.positions_force_closed = 0
        self.total_slippage_cost = Decimal("0")

        logger.info("Deleveraging protocol reset")
