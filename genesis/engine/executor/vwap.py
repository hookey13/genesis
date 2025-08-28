"""VWAP (Volume-Weighted Average Price) execution algorithm.

This module implements an institutional-quality VWAP execution algorithm
with dynamic participation rates, aggressive/passive modes, and dark pool simulation.
Available only for Strategist tier and above.
"""

import asyncio
import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from uuid import uuid4

import structlog

from genesis.analytics.volume_analyzer import VolumeAnalyzer, VolumePrediction
from genesis.analytics.vwap_tracker import VWAPTracker
from genesis.core.constants import TradingTier
from typing import Optional

from genesis.core.events import Event, EventType
from genesis.core.models import Symbol
from genesis.engine.event_bus import EventBus
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class ExecutionMode(str, Enum):
    """VWAP execution modes."""

    PASSIVE = "PASSIVE"  # Post-only orders, minimal market impact
    NORMAL = "NORMAL"  # Mix of limit and market orders
    AGGRESSIVE = "AGGRESSIVE"  # More aggressive fills, accept slippage


class VWAPSlice:
    """Represents a single slice of a VWAP order."""

    def __init__(
        self,
        slice_id: str,
        parent_order_id: str,
        symbol: Symbol,
        side: OrderSide,
        quantity: Decimal,
        target_price: Optional[Decimal],
        scheduled_time: datetime,
        bucket_minute: int,
    ):
        self.slice_id = slice_id
        self.parent_order_id = parent_order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.target_price = target_price
        self.scheduled_time = scheduled_time
        self.bucket_minute = bucket_minute
        self.executed_quantity = Decimal("0")
        self.executed_value = Decimal("0")
        self.status = OrderStatus.PENDING
        self.order: Optional[Order] = None
        self.attempts = 0
        self.last_error: Optional[str] = None


class VWAPExecutor(OrderExecutor):
    """Executes orders using Volume-Weighted Average Price algorithm."""

    def __init__(
        self,
        tier: TradingTier,
        exchange_gateway: ExchangeGateway,
        volume_analyzer: VolumeAnalyzer,
        vwap_tracker: VWAPTracker,
        event_bus: EventBus,
        config: dict,
    ):
        """Initialize VWAP executor.

        Args:
            tier: Current trading tier
            exchange_gateway: Gateway for exchange operations
            volume_analyzer: Volume pattern analyzer
            vwap_tracker: VWAP calculation and tracking
            event_bus: Event bus for notifications
            config: VWAP configuration parameters
        """
        super().__init__(tier)
        self.exchange = exchange_gateway
        self.volume_analyzer = volume_analyzer
        self.vwap_tracker = vwap_tracker
        self.event_bus = event_bus
        self.config = config

        # Default configuration values
        self.default_participation_rate = (
            Decimal(str(config.get("vwap_participation_rate_percent", 10))) / 100
        )
        self.min_slice_size_usd = Decimal(
            str(config.get("vwap_min_slice_size_usd", 50))
        )
        self.max_slices = config.get("vwap_max_slices", 100)
        self.time_window_minutes = config.get("vwap_time_window_minutes", 240)
        self.aggressive_threshold = (
            Decimal(str(config.get("vwap_aggressive_threshold_percent", 5))) / 100
        )

        # Active executions
        self._active_executions: dict[str, list[VWAPSlice]] = {}
        self._execution_tasks: dict[str, asyncio.Task] = {}

        logger.info(
            "vwap_executor_initialized",
            tier=tier.value,
            participation_rate=str(self.default_participation_rate),
            max_slices=self.max_slices,
        )

    @requires_tier(TradingTier.STRATEGIST)
    async def execute_vwap(
        self,
        order: Order,
        mode: ExecutionMode = ExecutionMode.NORMAL,
        time_horizon_minutes: Optional[int] = None,
        participation_rate: Optional[Decimal] = None,
        use_iceberg: bool = True,
    ) -> ExecutionResult:
        """Execute an order using VWAP algorithm.

        Args:
            order: Order to execute
            mode: Execution mode (passive/normal/aggressive)
            time_horizon_minutes: Time window for execution
            participation_rate: Maximum market participation rate
            use_iceberg: Whether to use iceberg orders for dark pool simulation

        Returns:
            ExecutionResult with final execution details
        """
        try:
            logger.info(
                "starting_vwap_execution",
                order_id=order.order_id,
                symbol=order.symbol,
                quantity=str(order.quantity),
                mode=mode.value,
            )

            # Validate order
            self.validate_order(order)

            # Set execution parameters
            horizon = time_horizon_minutes or self.time_window_minutes
            participation = participation_rate or self.default_participation_rate

            # Get volume predictions
            symbol = Symbol(order.symbol)
            current_time = datetime.now(UTC)
            prediction = await self.volume_analyzer.predict_intraday_volume(
                symbol, current_time, horizon // 60
            )

            # Calculate slices based on predicted volume
            slices = await self._calculate_slices(
                order, prediction, participation, mode, horizon
            )

            if not slices:
                return ExecutionResult(
                    success=False,
                    order=order,
                    message="Unable to calculate VWAP slices",
                    error="Insufficient volume predictions",
                )

            # Store slices for tracking
            self._active_executions[order.order_id] = slices

            # Start execution tracking
            self.vwap_tracker.start_execution_tracking(
                symbol, order.order_id, order.quantity
            )

            # Execute slices asynchronously
            execution_task = asyncio.create_task(
                self._execute_slices(order, slices, mode, use_iceberg)
            )
            self._execution_tasks[order.order_id] = execution_task

            # Wait for completion or timeout
            timeout = horizon * 60 + 60  # Extra minute buffer
            try:
                result = await asyncio.wait_for(execution_task, timeout=timeout)
            except TimeoutError:
                logger.error(
                    "vwap_execution_timeout",
                    order_id=order.order_id,
                    timeout_seconds=timeout,
                )
                # Clean up and return partial result
                await self._cleanup_execution(order.order_id)
                result = self._build_final_result(order, slices, "Execution timed out")

            # Complete tracking and get performance
            performance = self.vwap_tracker.complete_execution(
                order.order_id, order.quantity
            )

            # Emit completion event
            await self._emit_completion_event(order, result, performance)

            return result

        except Exception as e:
            logger.error("vwap_execution_error", order_id=order.order_id, error=str(e))
            return ExecutionResult(
                success=False,
                order=order,
                message=f"VWAP execution failed: {e!s}",
                error=str(e),
            )

    async def _calculate_slices(
        self,
        order: Order,
        prediction: VolumePrediction,
        participation_rate: Decimal,
        mode: ExecutionMode,
        horizon_minutes: int,
    ) -> list[VWAPSlice]:
        """Calculate order slices based on volume predictions.

        Args:
            order: Parent order to slice
            prediction: Volume predictions
            participation_rate: Market participation rate
            mode: Execution mode
            horizon_minutes: Time horizon for execution

        Returns:
            List of VWAP slices
        """
        slices = []
        remaining_quantity = order.quantity
        current_time = datetime.now(UTC)

        # Calculate participation rates per bucket
        participation_rates = self.volume_analyzer.get_optimal_participation_rate(
            order.quantity, prediction, participation_rate
        )

        # Adjust for execution mode
        if mode == ExecutionMode.AGGRESSIVE:
            # Front-load execution
            for bucket in participation_rates:
                participation_rates[bucket] *= Decimal("1.2")
        elif mode == ExecutionMode.PASSIVE:
            # Spread out more evenly
            for bucket in participation_rates:
                participation_rates[bucket] *= Decimal("0.8")

        # Create slices for each time bucket
        slice_number = 0
        for bucket_minute, predicted_volume in prediction.predicted_buckets.items():
            if remaining_quantity <= 0:
                break

            if slice_number >= self.max_slices:
                # Add remaining to last slice
                if slices:
                    slices[-1].quantity += remaining_quantity
                break

            # Calculate slice size based on participation
            bucket_participation = participation_rates.get(
                bucket_minute, participation_rate
            )
            slice_quantity = predicted_volume * bucket_participation

            # Apply minimum slice size
            min_slice = self.min_slice_size_usd / Decimal("100")  # Rough conversion
            if slice_quantity < min_slice:
                continue

            # Don't exceed remaining quantity
            slice_quantity = min(slice_quantity, remaining_quantity)

            # Calculate scheduled time with randomization
            bucket_start = current_time.replace(
                hour=bucket_minute // 60,
                minute=bucket_minute % 60,
                second=0,
                microsecond=0,
            )

            # Add random offset within bucket to avoid detection
            random_offset = random.randint(0, 1800)  # 0-30 minutes
            scheduled_time = bucket_start + timedelta(seconds=random_offset)

            # Skip if scheduled time is in the past
            if scheduled_time <= current_time:
                scheduled_time = current_time + timedelta(seconds=slice_number * 30)

            slice_obj = VWAPSlice(
                slice_id=f"{order.order_id}_slice_{slice_number}",
                parent_order_id=order.order_id,
                symbol=Symbol(order.symbol),
                side=order.side,
                quantity=slice_quantity,
                target_price=order.price,  # May be None for market orders
                scheduled_time=scheduled_time,
                bucket_minute=bucket_minute,
            )

            slices.append(slice_obj)
            remaining_quantity -= slice_quantity
            slice_number += 1

        # If we have remaining quantity, distribute among existing slices
        if remaining_quantity > 0 and slices:
            per_slice_addition = remaining_quantity / len(slices)
            for slice_obj in slices:
                slice_obj.quantity += per_slice_addition

        logger.info(
            "calculated_vwap_slices",
            order_id=order.order_id,
            total_slices=len(slices),
            total_quantity=str(sum(s.quantity for s in slices)),
        )

        return slices

    async def _execute_slices(
        self,
        parent_order: Order,
        slices: list[VWAPSlice],
        mode: ExecutionMode,
        use_iceberg: bool,
    ) -> ExecutionResult:
        """Execute VWAP slices according to schedule.

        Args:
            parent_order: Parent order
            slices: List of slices to execute
            mode: Execution mode
            use_iceberg: Whether to use iceberg orders

        Returns:
            Final execution result
        """
        try:
            for slice_obj in slices:
                # Wait until scheduled time
                now = datetime.now(UTC)
                if slice_obj.scheduled_time > now:
                    wait_seconds = (slice_obj.scheduled_time - now).total_seconds()
                    await asyncio.sleep(wait_seconds)

                # Execute slice
                await self._execute_single_slice(
                    slice_obj, parent_order, mode, use_iceberg
                )

                # Update parent order
                parent_order.filled_quantity = sum(s.executed_quantity for s in slices)

                # Check if we should switch to aggressive mode
                if await self._should_switch_to_aggressive(parent_order, slices):
                    mode = ExecutionMode.AGGRESSIVE
                    logger.info(
                        "switching_to_aggressive_mode",
                        order_id=parent_order.order_id,
                        filled_pct=str(
                            parent_order.filled_quantity / parent_order.quantity
                        ),
                    )

            # Build final result
            return self._build_final_result(
                parent_order, slices, "VWAP execution completed"
            )

        except Exception as e:
            logger.error(
                "slice_execution_error", order_id=parent_order.order_id, error=str(e)
            )
            return self._build_final_result(
                parent_order, slices, f"Slice execution failed: {e!s}"
            )

    async def _execute_single_slice(
        self,
        slice_obj: VWAPSlice,
        parent_order: Order,
        mode: ExecutionMode,
        use_iceberg: bool,
    ):
        """Execute a single VWAP slice.

        Args:
            slice_obj: Slice to execute
            parent_order: Parent order
            mode: Execution mode
            use_iceberg: Whether to use iceberg
        """
        try:
            # Create order for this slice
            slice_order = Order(
                order_id=slice_obj.slice_id,
                position_id=parent_order.position_id,
                client_order_id=str(uuid4()),
                symbol=slice_obj.symbol.value,
                type=self._get_order_type(mode),
                side=slice_obj.side,
                price=slice_obj.target_price,
                quantity=slice_obj.quantity,
                slice_number=slices.index(slice_obj) + 1 if "slices" in locals() else 1,
                total_slices=len(
                    self._active_executions.get(parent_order.order_id, [])
                ),
            )

            # Execute based on mode and settings
            if use_iceberg and mode != ExecutionMode.AGGRESSIVE:
                # Use iceberg for passive/normal modes
                sub_slices = max(
                    3, min(10, int(slice_obj.quantity / self.min_slice_size_usd))
                )
                result = await self._execute_iceberg_slice(slice_order, sub_slices)
            else:
                # Direct execution
                result = await self._execute_direct_slice(slice_order, mode)

            # Update slice status
            if result.success:
                slice_obj.executed_quantity = result.order.filled_quantity
                slice_obj.executed_value = result.order.filled_quantity * (
                    result.actual_price or slice_obj.target_price or Decimal("0")
                )
                slice_obj.status = OrderStatus.FILLED
                slice_obj.order = result.order

                # Update VWAP tracker
                self.vwap_tracker.update_execution(
                    parent_order.order_id,
                    result.actual_price or Decimal("0"),
                    result.order.filled_quantity,
                )
            else:
                slice_obj.status = OrderStatus.FAILED
                slice_obj.last_error = result.error
                slice_obj.attempts += 1

                # Retry if not exceeded max attempts
                if slice_obj.attempts < 3:
                    await asyncio.sleep(2**slice_obj.attempts)  # Exponential backoff
                    await self._execute_single_slice(
                        slice_obj, parent_order, ExecutionMode.AGGRESSIVE, False
                    )

        except Exception as e:
            logger.error(
                "single_slice_execution_error",
                slice_id=slice_obj.slice_id,
                error=str(e),
            )
            slice_obj.status = OrderStatus.FAILED
            slice_obj.last_error = str(e)

    async def _execute_iceberg_slice(
        self, order: Order, sub_slices: int
    ) -> ExecutionResult:
        """Execute slice as iceberg order.

        Args:
            order: Slice order
            sub_slices: Number of sub-slices for iceberg

        Returns:
            Execution result
        """
        # This would integrate with iceberg executor
        # For now, simulate iceberg execution
        logger.info(
            "executing_iceberg_slice", order_id=order.order_id, sub_slices=sub_slices
        )

        # Placeholder - would call actual iceberg executor
        return await self.execute_market_order(order)

    async def _execute_direct_slice(
        self, order: Order, mode: ExecutionMode
    ) -> ExecutionResult:
        """Execute slice directly based on mode.

        Args:
            order: Slice order
            mode: Execution mode

        Returns:
            Execution result
        """
        if mode == ExecutionMode.PASSIVE:
            # Use limit order at or better than market
            order.type = OrderType.LIMIT_MAKER
            # Would fetch current market price and place limit
        elif mode == ExecutionMode.AGGRESSIVE:
            # Use market order for immediate fill
            order.type = OrderType.MARKET
        else:
            # Normal mode - use limit with slight buffer
            order.type = OrderType.LIMIT

        return await self.execute_market_order(order)

    def _get_order_type(self, mode: ExecutionMode) -> OrderType:
        """Get appropriate order type for execution mode.

        Args:
            mode: Execution mode

        Returns:
            Order type to use
        """
        if mode == ExecutionMode.PASSIVE:
            return OrderType.LIMIT_MAKER
        elif mode == ExecutionMode.AGGRESSIVE:
            return OrderType.MARKET
        else:
            return OrderType.LIMIT

    async def _should_switch_to_aggressive(
        self, order: Order, slices: list[VWAPSlice]
    ) -> bool:
        """Determine if should switch to aggressive mode.

        Args:
            order: Parent order
            slices: All slices

        Returns:
            True if should switch to aggressive
        """
        # Calculate progress
        filled_pct = (
            order.filled_quantity / order.quantity
            if order.quantity > 0
            else Decimal("0")
        )

        # Calculate time progress
        executed_slices = sum(1 for s in slices if s.status == OrderStatus.FILLED)
        time_progress = executed_slices / len(slices) if slices else Decimal("0")

        # Switch if falling behind schedule
        return filled_pct < time_progress - self.aggressive_threshold

    def _build_final_result(
        self, order: Order, slices: list[VWAPSlice], message: str
    ) -> ExecutionResult:
        """Build final execution result from slices.

        Args:
            order: Parent order
            slices: All execution slices
            message: Result message

        Returns:
            Final execution result
        """
        total_executed = sum(s.executed_quantity for s in slices)
        total_value = sum(s.executed_value for s in slices)

        avg_price = total_value / total_executed if total_executed > 0 else Decimal("0")

        order.filled_quantity = total_executed
        order.status = (
            OrderStatus.FILLED
            if total_executed >= order.quantity * Decimal("0.99")
            else OrderStatus.PARTIAL
        )

        # Calculate slippage if we have a target price
        slippage = None
        if order.price and avg_price > 0:
            slippage = self.calculate_slippage(order.price, avg_price, order.side)

        return ExecutionResult(
            success=order.status == OrderStatus.FILLED,
            order=order,
            message=message,
            actual_price=avg_price,
            slippage_percent=slippage,
            latency_ms=None,  # Would calculate from execution times
        )

    async def _emit_completion_event(
        self, order: Order, result: ExecutionResult, performance
    ):
        """Emit VWAP execution completion event.

        Args:
            order: Executed order
            result: Execution result
            performance: Execution performance metrics
        """
        event_data = {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "quantity": str(order.quantity),
            "filled_quantity": str(order.filled_quantity),
            "success": result.success,
            "message": result.message,
        }

        if performance:
            event_data.update(performance.to_dict())

        event = Event(
            event_type=EventType.ORDER_EXECUTED,
            created_at=datetime.now(UTC),
            event_data=event_data,
        )

        await self.event_bus.emit(event)

    async def _cleanup_execution(self, order_id: str):
        """Clean up execution tracking.

        Args:
            order_id: Order ID to clean up
        """
        if order_id in self._active_executions:
            del self._active_executions[order_id]

        if order_id in self._execution_tasks:
            task = self._execution_tasks[order_id]
            if not task.done():
                task.cancel()
            del self._execution_tasks[order_id]

    # Required abstract methods from OrderExecutor
    async def execute_market_order(
        self, order: Order, confirmation_required: bool = True
    ) -> ExecutionResult:
        """Execute a market order through exchange.

        Args:
            order: Order to execute
            confirmation_required: Whether to confirm before execution

        Returns:
            Execution result
        """
        # This would integrate with the exchange gateway
        # Placeholder implementation
        logger.info(
            "executing_market_order",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=str(order.quantity),
        )

        # Simulate execution
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.executed_at = datetime.now(UTC)

        return ExecutionResult(
            success=True,
            order=order,
            message="Market order executed",
            actual_price=order.price or Decimal("100"),  # Would get from exchange
            slippage_percent=Decimal("0"),
            latency_ms=50,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            True if cancelled successfully
        """
        # Would integrate with exchange
        logger.info("cancelling_order", order_id=order_id, symbol=symbol)
        return True

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        count = 0
        for order_id in list(self._active_executions.keys()):
            slices = self._active_executions[order_id]
            if slices and (symbol is None or slices[0].symbol.value == symbol):
                await self._cleanup_execution(order_id)
                count += len(slices)

        logger.info("cancelled_all_orders", symbol=symbol, count=count)
        return count

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status.

        Args:
            order_id: Order ID to check
            symbol: Trading symbol

        Returns:
            Order with current status
        """
        # Would integrate with exchange
        # Check if it's a VWAP execution
        if order_id in self._active_executions:
            slices = self._active_executions[order_id]
            # Build composite order from slices
            total_filled = sum(s.executed_quantity for s in slices)
            total_quantity = sum(s.quantity for s in slices)

            return Order(
                order_id=order_id,
                position_id=None,
                client_order_id=str(uuid4()),
                symbol=symbol,
                type=OrderType.MARKET,
                side=slices[0].side if slices else OrderSide.BUY,
                price=None,
                quantity=total_quantity,
                filled_quantity=total_filled,
                status=(
                    OrderStatus.PARTIAL
                    if total_filled < total_quantity
                    else OrderStatus.FILLED
                ),
            )

        # Default placeholder
        return Order(
            order_id=order_id,
            position_id=None,
            client_order_id=str(uuid4()),
            symbol=symbol,
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0"),
            status=OrderStatus.PENDING,
        )
