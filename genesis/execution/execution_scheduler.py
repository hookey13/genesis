"""Execution scheduler for VWAP and other advanced execution algorithms."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from genesis.execution.order_slicer import OrderSlice
from genesis.execution.volume_curve import VolumeProfile
from genesis.strategies.strategist.vwap_execution import UrgencyLevel

logger = structlog.get_logger(__name__)


class ScheduleType(str, Enum):
    """Schedule types for execution."""

    IMMEDIATE = "IMMEDIATE"  # Execute now
    SCHEDULED = "SCHEDULED"  # Execute at specific times
    ADAPTIVE = "ADAPTIVE"  # Adjust based on market conditions
    PASSIVE = "PASSIVE"  # Wait for favorable conditions
    AGGRESSIVE = "AGGRESSIVE"  # Push through regardless


@dataclass
class ExecutionTask:
    """Individual execution task in schedule."""

    task_id: str
    order_slice: OrderSlice
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    status: str = "PENDING"  # PENDING, EXECUTING, COMPLETED, FAILED
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 0
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order."""

    plan_id: str
    parent_order_id: str
    tasks: List[ExecutionTask]
    schedule_type: ScheduleType
    start_time: datetime
    end_time: datetime
    total_quantity: Decimal
    executed_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    """Configuration for execution scheduler."""

    min_interval_seconds: int = 5  # Minimum time between executions
    max_interval_seconds: int = 300  # Maximum time between executions
    adaptive_reschedule: bool = True  # Allow dynamic rescheduling
    passive_spread_threshold: Decimal = Decimal("0.001")  # 0.1% spread for passive
    aggressive_premium: Decimal = Decimal("0.002")  # 0.2% premium for aggressive
    anti_gaming_delay: int = 3  # Random 0-3 second delay
    max_concurrent_tasks: int = 10  # Maximum concurrent executions


class ExecutionScheduler:
    """Schedules and manages order execution."""

    def __init__(self, config: SchedulerConfig | None = None):
        """Initialize execution scheduler.

        Args:
            config: Scheduler configuration.
        """
        self.config = config or SchedulerConfig()
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None

    async def create_execution_plan(
        self,
        parent_order_id: str,
        slices: List[OrderSlice],
        urgency: UrgencyLevel,
        volume_profile: Optional[VolumeProfile] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ExecutionPlan:
        """Create execution plan from order slices.

        Args:
            parent_order_id: Parent order ID.
            slices: Order slices to schedule.
            urgency: Execution urgency level.
            volume_profile: Volume profile for scheduling.
            start_time: Execution start time.
            end_time: Execution end time.

        Returns:
            Execution plan.
        """
        if not start_time:
            start_time = datetime.now(UTC)
        if not end_time:
            end_time = start_time + timedelta(hours=1)

        # Determine schedule type based on urgency
        schedule_type = self._determine_schedule_type(urgency)

        # Create tasks from slices
        tasks = await self._create_tasks(
            slices=slices,
            urgency=urgency,
            volume_profile=volume_profile,
            start_time=start_time,
            end_time=end_time,
            schedule_type=schedule_type,
        )

        # Calculate total quantity
        total_quantity = sum(slice.quantity for slice in slices)

        # Create execution plan
        plan = ExecutionPlan(
            plan_id=f"plan_{parent_order_id}",
            parent_order_id=parent_order_id,
            tasks=tasks,
            schedule_type=schedule_type,
            start_time=start_time,
            end_time=end_time,
            total_quantity=total_quantity,
            metadata={"urgency": urgency.value, "num_slices": len(slices)},
        )

        self.execution_plans[plan.plan_id] = plan

        logger.info(
            "Created execution plan",
            plan_id=plan.plan_id,
            num_tasks=len(tasks),
            schedule_type=schedule_type.value,
            urgency=urgency.value,
        )

        return plan

    def _determine_schedule_type(self, urgency: UrgencyLevel) -> ScheduleType:
        """Determine schedule type based on urgency.

        Args:
            urgency: Urgency level.

        Returns:
            Schedule type.
        """
        if urgency == UrgencyLevel.EMERGENCY:
            return ScheduleType.IMMEDIATE
        elif urgency == UrgencyLevel.CRITICAL:
            return ScheduleType.AGGRESSIVE
        elif urgency == UrgencyLevel.HIGH:
            return ScheduleType.ADAPTIVE
        elif urgency == UrgencyLevel.MEDIUM:
            return ScheduleType.SCHEDULED
        else:  # LOW
            return ScheduleType.PASSIVE

    async def _create_tasks(
        self,
        slices: List[OrderSlice],
        urgency: UrgencyLevel,
        volume_profile: Optional[VolumeProfile],
        start_time: datetime,
        end_time: datetime,
        schedule_type: ScheduleType,
    ) -> List[ExecutionTask]:
        """Create execution tasks from slices.

        Args:
            slices: Order slices.
            urgency: Urgency level.
            volume_profile: Volume profile.
            start_time: Start time.
            end_time: End time.
            schedule_type: Schedule type.

        Returns:
            List of execution tasks.
        """
        tasks = []

        # Calculate time intervals
        duration = end_time - start_time

        if schedule_type == ScheduleType.IMMEDIATE:
            # All tasks execute immediately
            for i, slice in enumerate(slices):
                task = ExecutionTask(
                    task_id=f"task_{i}_{datetime.now(UTC).timestamp()}",
                    order_slice=slice,
                    scheduled_time=start_time,
                    priority=100,  # Highest priority
                    urgency=urgency,
                )
                tasks.append(task)

        elif schedule_type == ScheduleType.AGGRESSIVE:
            # Front-loaded schedule
            interval = duration / (len(slices) * 2)  # Compress to first half
            for i, slice in enumerate(slices):
                task_time = start_time + (interval * i)
                task = ExecutionTask(
                    task_id=f"task_{i}_{datetime.now(UTC).timestamp()}",
                    order_slice=slice,
                    scheduled_time=task_time,
                    priority=90 - i,  # Decreasing priority
                    urgency=urgency,
                )
                tasks.append(task)

        elif schedule_type == ScheduleType.ADAPTIVE:
            # Use volume profile if available
            if volume_profile and volume_profile.intervals:
                tasks = await self._create_volume_based_tasks(
                    slices, urgency, volume_profile, start_time
                )
            else:
                # Fall back to even distribution
                interval = duration / len(slices)
                for i, slice in enumerate(slices):
                    task_time = start_time + (interval * i)
                    task = ExecutionTask(
                        task_id=f"task_{i}_{datetime.now(UTC).timestamp()}",
                        order_slice=slice,
                        scheduled_time=task_time,
                        priority=50,
                        urgency=urgency,
                    )
                    tasks.append(task)

        elif schedule_type == ScheduleType.PASSIVE:
            # Spread out over full duration
            interval = duration / len(slices)
            for i, slice in enumerate(slices):
                task_time = start_time + (interval * i)
                # Add anti-gaming randomness
                import random

                jitter = timedelta(
                    seconds=random.randint(0, self.config.anti_gaming_delay)
                )
                task_time += jitter

                task = ExecutionTask(
                    task_id=f"task_{i}_{datetime.now(UTC).timestamp()}",
                    order_slice=slice,
                    scheduled_time=task_time,
                    priority=10,  # Low priority
                    urgency=urgency,
                    metadata={"passive": True},
                )
                tasks.append(task)

        else:  # SCHEDULED
            # Even distribution
            interval = duration / len(slices)
            for i, slice in enumerate(slices):
                task_time = start_time + (interval * i)
                task = ExecutionTask(
                    task_id=f"task_{i}_{datetime.now(UTC).timestamp()}",
                    order_slice=slice,
                    scheduled_time=task_time,
                    priority=30,
                    urgency=urgency,
                )
                tasks.append(task)

        return tasks

    async def _create_volume_based_tasks(
        self,
        slices: List[OrderSlice],
        urgency: UrgencyLevel,
        volume_profile: VolumeProfile,
        start_time: datetime,
    ) -> List[ExecutionTask]:
        """Create tasks based on volume profile.

        Args:
            slices: Order slices.
            urgency: Urgency level.
            volume_profile: Volume profile.
            start_time: Start time.

        Returns:
            List of volume-based tasks.
        """
        tasks = []

        # Map slices to volume intervals
        num_intervals = len(volume_profile.intervals)
        slices_per_interval = max(1, len(slices) // num_intervals)

        slice_idx = 0
        for interval_idx, interval_time in enumerate(volume_profile.intervals):
            # Get volume weight for this interval
            volume_weight = volume_profile.normalized_volumes[interval_idx]

            # Determine number of slices for this interval
            if volume_weight > Decimal("0.05"):  # High volume interval
                interval_slices = int(slices_per_interval * 1.5)
            elif volume_weight > Decimal("0.02"):  # Medium volume
                interval_slices = slices_per_interval
            else:  # Low volume
                interval_slices = max(1, slices_per_interval // 2)

            # Create tasks for this interval
            for i in range(interval_slices):
                if slice_idx >= len(slices):
                    break

                slice = slices[slice_idx]

                # Calculate task time within interval
                if interval_idx < num_intervals - 1:
                    next_interval = volume_profile.intervals[interval_idx + 1]
                    interval_duration = next_interval - interval_time
                else:
                    interval_duration = timedelta(minutes=30)  # Default

                offset = interval_duration * (i / max(1, interval_slices))
                task_time = interval_time + offset

                # Adjust for start time
                if task_time < start_time:
                    task_time = start_time + timedelta(seconds=slice_idx * 10)

                task = ExecutionTask(
                    task_id=f"task_{slice_idx}_{datetime.now(UTC).timestamp()}",
                    order_slice=slice,
                    scheduled_time=task_time,
                    priority=int(
                        50 + (volume_weight * 100)
                    ),  # Higher priority for high volume
                    urgency=urgency,
                    metadata={
                        "volume_weight": float(volume_weight),
                        "interval_idx": interval_idx,
                    },
                )
                tasks.append(task)
                slice_idx += 1

        # Handle remaining slices
        while slice_idx < len(slices):
            slice = slices[slice_idx]
            task_time = (
                tasks[-1].scheduled_time + timedelta(seconds=30)
                if tasks
                else start_time
            )

            task = ExecutionTask(
                task_id=f"task_{slice_idx}_{datetime.now(UTC).timestamp()}",
                order_slice=slice,
                scheduled_time=task_time,
                priority=30,
                urgency=urgency,
                metadata={"overflow": True},
            )
            tasks.append(task)
            slice_idx += 1

        return tasks

    async def start(self) -> None:
        """Start the execution scheduler."""
        if self.is_running:
            return

        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._run_scheduler())

        logger.info("Execution scheduler started")

    async def stop(self) -> None:
        """Stop the execution scheduler."""
        self.is_running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Execution scheduler stopped")

    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check for tasks ready to execute
                await self._check_scheduled_tasks()

                # Process adaptive rescheduling if enabled
                if self.config.adaptive_reschedule:
                    await self._adaptive_reschedule()

                # Sleep briefly
                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Scheduler error", error=str(e))
                await asyncio.sleep(5)

    async def _check_scheduled_tasks(self) -> None:
        """Check for tasks ready to execute."""
        now = datetime.now(UTC)

        for plan in self.execution_plans.values():
            for task in plan.tasks:
                if (
                    task.status == "PENDING"
                    and task.scheduled_time <= now
                    and task.task_id not in self.active_tasks
                    and len(self.active_tasks) < self.config.max_concurrent_tasks
                ):
                    # Queue task for execution
                    await self.task_queue.put(task)
                    self.active_tasks[task.task_id] = task
                    task.status = "EXECUTING"

                    logger.debug(
                        "Queued task for execution",
                        task_id=task.task_id,
                        slice_number=task.order_slice.slice_number,
                    )

    async def _adaptive_reschedule(self) -> None:
        """Adaptively reschedule tasks based on conditions."""
        now = datetime.now(UTC)

        for plan in self.execution_plans.values():
            if plan.schedule_type != ScheduleType.ADAPTIVE:
                continue

            # Check if we're behind schedule
            expected_progress = self._calculate_expected_progress(plan, now)
            actual_progress = (
                plan.executed_quantity / plan.total_quantity
                if plan.total_quantity > 0
                else Decimal("0")
            )

            if actual_progress < expected_progress - Decimal("0.1"):  # 10% behind
                # Accelerate remaining tasks
                await self._accelerate_schedule(plan, now)
            elif actual_progress > expected_progress + Decimal("0.1"):  # 10% ahead
                # Decelerate remaining tasks
                await self._decelerate_schedule(plan, now)

    def _calculate_expected_progress(
        self, plan: ExecutionPlan, current_time: datetime
    ) -> Decimal:
        """Calculate expected execution progress.

        Args:
            plan: Execution plan.
            current_time: Current time.

        Returns:
            Expected progress percentage.
        """
        if current_time >= plan.end_time:
            return Decimal("1.0")
        elif current_time <= plan.start_time:
            return Decimal("0.0")
        else:
            elapsed = (current_time - plan.start_time).total_seconds()
            total = (plan.end_time - plan.start_time).total_seconds()
            return Decimal(str(elapsed / total))

    async def _accelerate_schedule(
        self, plan: ExecutionPlan, current_time: datetime
    ) -> None:
        """Accelerate execution schedule.

        Args:
            plan: Execution plan.
            current_time: Current time.
        """
        pending_tasks = [t for t in plan.tasks if t.status == "PENDING"]

        if not pending_tasks:
            return

        # Compress schedule by 20%
        compression_factor = Decimal("0.8")

        for task in pending_tasks:
            if task.scheduled_time > current_time:
                time_until = task.scheduled_time - current_time
                compressed_time = time_until.total_seconds() * float(compression_factor)
                task.scheduled_time = current_time + timedelta(seconds=compressed_time)

        plan.updated_at = current_time

        logger.info(
            "Accelerated execution schedule",
            plan_id=plan.plan_id,
            pending_tasks=len(pending_tasks),
        )

    async def _decelerate_schedule(
        self, plan: ExecutionPlan, current_time: datetime
    ) -> None:
        """Decelerate execution schedule.

        Args:
            plan: Execution plan.
            current_time: Current time.
        """
        pending_tasks = [t for t in plan.tasks if t.status == "PENDING"]

        if not pending_tasks:
            return

        # Expand schedule by 20%
        expansion_factor = Decimal("1.2")

        for task in pending_tasks:
            if task.scheduled_time > current_time:
                time_until = task.scheduled_time - current_time
                expanded_time = time_until.total_seconds() * float(expansion_factor)

                # Don't exceed end time
                new_time = current_time + timedelta(seconds=expanded_time)
                if new_time < plan.end_time:
                    task.scheduled_time = new_time

        plan.updated_at = current_time

        logger.info(
            "Decelerated execution schedule",
            plan_id=plan.plan_id,
            pending_tasks=len(pending_tasks),
        )

    async def get_next_task(self) -> Optional[ExecutionTask]:
        """Get next task from queue.

        Returns:
            Next task to execute or None.
        """
        try:
            task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
            return task
        except asyncio.TimeoutError:
            return None

    def complete_task(self, task_id: str, success: bool = True) -> None:
        """Mark task as completed.

        Args:
            task_id: Task ID.
            success: Whether task succeeded.
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = "COMPLETED" if success else "FAILED"
            task.actual_time = datetime.now(UTC)
            del self.active_tasks[task_id]

            # Update plan
            for plan in self.execution_plans.values():
                if task in plan.tasks:
                    if success:
                        plan.executed_quantity += task.order_slice.quantity
                    plan.updated_at = datetime.now(UTC)
                    break

            logger.debug("Task completed", task_id=task_id, success=success)
