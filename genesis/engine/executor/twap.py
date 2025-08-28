"""
TWAP (Time-Weighted Average Price) executor for Strategist tier in Project GENESIS.

This module implements sophisticated time-weighted order execution to achieve
better average entry prices by distributing orders across a time window.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
from uuid import uuid4

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.exceptions import OrderExecutionError, ValidationError
from genesis.core.models import Account, TradingTier
from genesis.data.market_data_service import MarketDataService, VolumeProfile
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.risk_engine import RiskEngine
from genesis.exchange.gateway import BinanceGateway
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


# TWAP configuration constants
MIN_DURATION_MINUTES = 5
MAX_DURATION_MINUTES = 30
DEFAULT_DURATION_MINUTES = 15
MAX_PARTICIPATION_RATE = Decimal("0.10")  # 10% max volume participation
EARLY_COMPLETION_THRESHOLD = Decimal(
    "0.002"
)  # 0.2% better price triggers early completion
MIN_SLICE_INTERVAL_SECONDS = 30
MAX_SLICE_INTERVAL_SECONDS = 300
SLICE_TIME_JITTER = 0.2  # ±20% timing variation


@dataclass
class TimeSlice:
    """Individual time slice for TWAP execution."""

    slice_number: int
    target_time: datetime
    target_quantity: Decimal
    volume_weight: Decimal = Decimal("1.0")
    participation_rate: Decimal = MAX_PARTICIPATION_RATE


@dataclass
class TwapExecution:
    """TWAP execution tracking."""

    execution_id: str
    symbol: str
    side: OrderSide
    total_quantity: Decimal
    duration_minutes: int
    slices: list[TimeSlice]
    arrival_price: Decimal
    benchmark_price: Optional[Decimal] = None
    executed_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    twap_price: Optional[Decimal] = None
    implementation_shortfall: Optional[Decimal] = None
    participation_rate: Decimal = Decimal("0")
    status: str = "ACTIVE"
    early_completion: bool = False
    early_completion_reason: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    executed_slices: list[dict[str, Any]] = field(default_factory=list)
    background_task: Optional[asyncio.Task] = None


class TwapExecutor(OrderExecutor):
    """
    TWAP executor for Strategist tier.

    Distributes order execution across a time window to achieve better
    average prices. Includes adaptive slice timing based on volume patterns,
    participation rate limiting, and early completion on favorable prices.
    """

    def __init__(
        self,
        gateway: BinanceGateway,
        account: Account,
        market_executor: MarketOrderExecutor,
        repository: Repository,
        market_data_service: MarketDataService,
        risk_engine: RiskEngine,
        event_bus: EventBus,
    ):
        """
        Initialize the TWAP executor.

        Args:
            gateway: Binance gateway for exchange interaction
            account: Trading account
            market_executor: Base market order executor for individual slices
            repository: Data repository for persistence
            market_data_service: Market data service for volume profiles
            risk_engine: Risk engine for position validation
            event_bus: Event bus for publishing execution events
        """
        super().__init__(TradingTier.STRATEGIST)

        # Validate tier requirement
        if account.tier.value < TradingTier.STRATEGIST.value:
            raise OrderExecutionError(
                f"TWAP execution requires {TradingTier.STRATEGIST.value} tier or above (current: {account.tier.value})"
            )

        self.gateway = gateway
        self.account = account
        self.market_executor = market_executor
        self.repository = repository
        self.market_data_service = market_data_service
        self.risk_engine = risk_engine
        self.event_bus = event_bus

        # Track active TWAP executions
        self.active_executions: dict[str, TwapExecution] = {}

        logger.info(
            "TWAP executor initialized",
            account_id=account.account_id,
            tier=account.tier.value,
        )

    @requires_tier(TradingTier.STRATEGIST)
    async def execute_twap(
        self, order: Order, duration_minutes: int = DEFAULT_DURATION_MINUTES
    ) -> ExecutionResult:
        """
        Execute an order using TWAP strategy.

        Args:
            order: Order to execute
            duration_minutes: Time window for execution (5-30 minutes)

        Returns:
            ExecutionResult with execution details

        Raises:
            OrderExecutionError: If execution fails
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            self.validate_order(order)
            self._validate_duration(duration_minutes)

            # Track arrival price (benchmark)
            arrival_price = await self.track_arrival_price(order.symbol)

            logger.info(
                "Starting TWAP execution",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=str(order.quantity),
                duration_minutes=duration_minutes,
                arrival_price=str(arrival_price),
            )

            # Get volume profile for adaptive timing
            volume_profile = await self.market_data_service.get_volume_profile(
                order.symbol
            )

            # Calculate time slices with adaptive timing
            time_slices = await self.calculate_time_slices(
                duration_minutes, volume_profile, order.quantity
            )

            # Create TWAP execution tracker
            execution_id = str(uuid4())
            execution = TwapExecution(
                execution_id=execution_id,
                symbol=order.symbol,
                side=order.side,
                total_quantity=order.quantity,
                duration_minutes=duration_minutes,
                slices=time_slices,
                arrival_price=arrival_price,
                remaining_quantity=order.quantity,
                started_at=datetime.now(),
            )

            self.active_executions[execution_id] = execution

            # Save execution to database
            await self._save_execution_to_db(execution)

            # Publish TWAP started event
            await self.event_bus.publish(
                Event(
                    type=EventType.ORDER_PLACED,
                    data={
                        "execution_id": execution_id,
                        "type": "TWAP_STARTED",
                        "symbol": order.symbol,
                        "quantity": str(order.quantity),
                        "duration_minutes": duration_minutes,
                    },
                ),
                priority=EventPriority.HIGH,
            )

            # Start background task for execution
            execution.background_task = asyncio.create_task(
                self._execute_slices(execution, order)
            )

            # Wait for completion or early termination
            await execution.background_task

            # Calculate final metrics
            execution_time = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            implementation_shortfall = (
                self.calculate_implementation_shortfall(
                    execution.arrival_price, execution.twap_price
                )
                if execution.twap_price
                else None
            )

            logger.info(
                "TWAP execution completed",
                execution_id=execution_id,
                status=execution.status,
                executed_quantity=str(execution.executed_quantity),
                average_price=str(execution.average_price),
                twap_price=str(execution.twap_price),
                implementation_shortfall=str(implementation_shortfall),
                early_completion=execution.early_completion,
                execution_time_seconds=execution_time,
            )

            # Update original order
            order.filled_quantity = execution.executed_quantity
            order.status = (
                OrderStatus.FILLED
                if execution.executed_quantity == order.quantity
                else OrderStatus.PARTIAL
            )

            return ExecutionResult(
                success=execution.status == "COMPLETED",
                order=order,
                message=f"TWAP execution {execution.status.lower()}",
                actual_price=execution.average_price,
                slippage_percent=implementation_shortfall,
                latency_ms=int(execution_time * 1000),
                error=(
                    execution.early_completion_reason
                    if not execution.early_completion
                    else None
                ),
            )

        except Exception as e:
            logger.error("TWAP execution failed", order_id=order.order_id, error=str(e))
            raise OrderExecutionError(
                f"Failed to execute TWAP order: {e!s}", order_id=order.order_id
            )

    async def _execute_slices(
        self, execution: TwapExecution, original_order: Order
    ) -> None:
        """
        Execute time slices in background.

        Args:
            execution: TWAP execution tracker
            original_order: Original order being executed
        """
        try:
            cumulative_quantity = Decimal("0")
            cumulative_value = Decimal("0")

            for slice_info in execution.slices:
                # Check if paused
                while execution.status == "PAUSED":
                    await asyncio.sleep(1)
                    continue

                # Check if cancelled or failed
                if execution.status in ["CANCELLED", "FAILED"]:
                    break

                # Wait until target time with jitter
                now = datetime.now()
                wait_time = (slice_info.target_time - now).total_seconds()
                if wait_time > 0:
                    # Add random jitter
                    jitter = wait_time * SLICE_TIME_JITTER * random.uniform(-1, 1)
                    actual_wait = max(0, wait_time + jitter)
                    await asyncio.sleep(actual_wait)

                # Check for early completion opportunity
                if await self.check_early_completion(
                    execution.symbol, execution.side, execution.arrival_price
                ):
                    # Execute remaining quantity
                    slice_info.target_quantity = execution.remaining_quantity
                    execution.early_completion = True
                    execution.early_completion_reason = "Favorable price detected"
                    logger.info(
                        "Early completion triggered",
                        execution_id=execution.execution_id,
                        remaining_quantity=str(execution.remaining_quantity),
                    )

                # Apply participation rate limit
                adjusted_quantity = await self.enforce_participation_limit(
                    slice_info.target_quantity,
                    execution.symbol,
                    slice_info.participation_rate,
                )

                # Validate with risk engine before execution
                risk_decision = await self.risk_engine.check_risk_limits(
                    {
                        "symbol": execution.symbol,
                        "side": execution.side,
                        "quantity": adjusted_quantity,
                    }
                )

                if not risk_decision.approved:
                    logger.warning(
                        "Slice rejected by risk engine",
                        execution_id=execution.execution_id,
                        slice_number=slice_info.slice_number,
                        reason=risk_decision.reason,
                    )
                    continue

                # Create slice order
                slice_order = Order(
                    order_id=str(uuid4()),
                    position_id=original_order.position_id,
                    client_order_id=self.generate_client_order_id(),
                    symbol=execution.symbol,
                    type=OrderType.MARKET,
                    side=execution.side,
                    price=None,
                    quantity=adjusted_quantity,
                    slice_number=slice_info.slice_number,
                    total_slices=len(execution.slices),
                    created_at=datetime.now(),
                )

                # Execute slice
                slice_result = await self.market_executor.execute_market_order(
                    slice_order, confirmation_required=False
                )

                if slice_result.success:
                    # Update execution tracking
                    executed_quantity = slice_order.filled_quantity
                    execution_price = slice_result.actual_price or Decimal("0")

                    cumulative_quantity += executed_quantity
                    cumulative_value += executed_quantity * execution_price

                    execution.executed_quantity = cumulative_quantity
                    execution.remaining_quantity = (
                        execution.total_quantity - cumulative_quantity
                    )
                    execution.average_price = (
                        cumulative_value / cumulative_quantity
                        if cumulative_quantity > 0
                        else Decimal("0")
                    )

                    # Track slice execution
                    slice_data = {
                        "slice_id": str(uuid4()),
                        "execution_id": execution.execution_id,
                        "slice_number": slice_info.slice_number,
                        "target_quantity": str(slice_info.target_quantity),
                        "executed_quantity": str(executed_quantity),
                        "execution_price": str(execution_price),
                        "market_price": str(
                            execution_price
                        ),  # For TWAP, these are the same
                        "slippage_bps": (
                            str(slice_result.slippage_percent * 100)
                            if slice_result.slippage_percent
                            else "0"
                        ),
                        "participation_rate": str(slice_info.participation_rate * 100),
                        "status": "EXECUTED",
                        "executed_at": datetime.now().isoformat(),
                    }
                    execution.executed_slices.append(slice_data)

                    # Save slice to database
                    await self._save_slice_to_db(slice_data)

                    # Publish slice executed event
                    await self.event_bus.publish(
                        Event(
                            type=EventType.ORDER_FILLED,
                            data={
                                "execution_id": execution.execution_id,
                                "type": "TWAP_SLICE_EXECUTED",
                                "slice_number": slice_info.slice_number,
                                "quantity": str(executed_quantity),
                                "price": str(execution_price),
                            },
                        ),
                        priority=EventPriority.NORMAL,
                    )

                    # Check for early completion
                    if execution.early_completion:
                        break

                else:
                    logger.warning(
                        "Slice execution failed",
                        execution_id=execution.execution_id,
                        slice_number=slice_info.slice_number,
                        error=slice_result.error,
                    )

            # Finalize execution
            execution.completed_at = datetime.now()
            execution.status = (
                "COMPLETED" if execution.executed_quantity > 0 else "FAILED"
            )

            # Calculate TWAP price
            execution.twap_price = self.calculate_twap_price(execution.executed_slices)

            # Calculate implementation shortfall
            if execution.twap_price:
                execution.implementation_shortfall = (
                    self.calculate_implementation_shortfall(
                        execution.arrival_price, execution.twap_price
                    )
                )

            # Update database
            await self._update_execution_in_db(execution)

            # Publish completion event
            await self.event_bus.publish(
                Event(
                    type=EventType.ORDER_FILLED,
                    data={
                        "execution_id": execution.execution_id,
                        "type": "TWAP_COMPLETED",
                        "status": execution.status,
                        "executed_quantity": str(execution.executed_quantity),
                        "twap_price": str(execution.twap_price),
                        "implementation_shortfall": str(
                            execution.implementation_shortfall
                        ),
                    },
                ),
                priority=EventPriority.HIGH,
            )

            # Remove from active executions
            del self.active_executions[execution.execution_id]

        except Exception as e:
            logger.error(
                "Error executing TWAP slices",
                execution_id=execution.execution_id,
                error=str(e),
            )
            execution.status = "FAILED"
            execution.completed_at = datetime.now()
            await self._update_execution_in_db(execution)

    async def calculate_time_slices(
        self,
        duration_minutes: int,
        volume_profile: VolumeProfile,
        total_quantity: Decimal,
    ) -> list[TimeSlice]:
        """
        Calculate time slices with adaptive timing based on volume patterns.

        Args:
            duration_minutes: Total execution duration
            volume_profile: Market volume profile
            total_quantity: Total quantity to execute

        Returns:
            List of time slices with target times and quantities
        """
        # Get hourly volumes from profile
        hourly_volumes = volume_profile.get_hourly_volumes()

        # Calculate number of slices based on duration
        interval_seconds = min(
            MAX_SLICE_INTERVAL_SECONDS,
            max(MIN_SLICE_INTERVAL_SECONDS, (duration_minutes * 60) / 10),
        )
        slice_count = int((duration_minutes * 60) / interval_seconds)

        # Calculate base quantity per slice
        base_quantity = total_quantity / Decimal(str(slice_count))

        # Create slices with volume-weighted quantities
        slices = []
        start_time = datetime.now()
        total_volume_weight = sum(hourly_volumes.values())

        for i in range(slice_count):
            # Calculate target time
            target_time = start_time + timedelta(seconds=interval_seconds * (i + 1))

            # Get volume weight for this time
            hour = target_time.hour
            volume_weight = hourly_volumes.get(hour, Decimal("1.0"))
            normalized_weight = (
                volume_weight / total_volume_weight
                if total_volume_weight > 0
                else Decimal("1.0")
            )

            # Adjust quantity based on volume weight
            adjusted_quantity = base_quantity * (
                Decimal("0.8") + normalized_weight * Decimal("0.4")
            )

            # Calculate participation rate (lower during low volume)
            participation_rate = MAX_PARTICIPATION_RATE * normalized_weight

            slices.append(
                TimeSlice(
                    slice_number=i + 1,
                    target_time=target_time,
                    target_quantity=adjusted_quantity,
                    volume_weight=normalized_weight,
                    participation_rate=participation_rate,
                )
            )

        # Adjust last slice to ensure total quantity matches
        total_allocated = sum(s.target_quantity for s in slices[:-1])
        slices[-1].target_quantity = total_quantity - total_allocated

        logger.debug(
            "Calculated TWAP time slices",
            slice_count=len(slices),
            duration_minutes=duration_minutes,
            interval_seconds=interval_seconds,
        )

        return slices

    async def check_early_completion(
        self, symbol: str, side: OrderSide, target_price: Decimal
    ) -> bool:
        """
        Check if current price is favorable for early completion.

        Args:
            symbol: Trading symbol
            side: Order side
            target_price: Target/arrival price

        Returns:
            True if early completion should trigger
        """
        try:
            current_price = await self.market_data_service.get_current_price(symbol)

            if side == OrderSide.BUY:
                # For buys, complete early if price is sufficiently below target
                improvement = (target_price - current_price) / target_price
            else:
                # For sells, complete early if price is sufficiently above target
                improvement = (current_price - target_price) / target_price

            return improvement >= EARLY_COMPLETION_THRESHOLD

        except Exception as e:
            logger.warning(
                "Failed to check early completion", symbol=symbol, error=str(e)
            )
            return False

    async def pause(self, execution_id: str) -> bool:
        """
        Pause a TWAP execution.

        Args:
            execution_id: Execution to pause

        Returns:
            True if successfully paused
        """
        if execution_id not in self.active_executions:
            raise ValidationError(f"Execution {execution_id} not found")

        execution = self.active_executions[execution_id]

        if execution.status != "ACTIVE":
            raise ValidationError(
                f"Cannot pause execution in {execution.status} status"
            )

        execution.status = "PAUSED"
        execution.paused_at = datetime.now()

        await self._update_execution_in_db(execution)

        logger.info("TWAP execution paused", execution_id=execution_id)

        return True

    async def resume(self, execution_id: str) -> bool:
        """
        Resume a paused TWAP execution.

        Args:
            execution_id: Execution to resume

        Returns:
            True if successfully resumed
        """
        if execution_id not in self.active_executions:
            raise ValidationError(f"Execution {execution_id} not found")

        execution = self.active_executions[execution_id]

        if execution.status != "PAUSED":
            raise ValidationError(
                f"Cannot resume execution in {execution.status} status"
            )

        execution.status = "ACTIVE"
        execution.resumed_at = datetime.now()

        await self._update_execution_in_db(execution)

        logger.info("TWAP execution resumed", execution_id=execution_id)

        return True

    async def track_arrival_price(self, symbol: str) -> Decimal:
        """
        Capture arrival price for benchmark tracking.

        Args:
            symbol: Trading symbol

        Returns:
            Current market price at arrival
        """
        return await self.market_data_service.get_current_price(symbol)

    def calculate_twap_price(self, slices: list[dict[str, Any]]) -> Decimal:
        """
        Calculate time-weighted average price from executed slices.

        Args:
            slices: List of executed slice data

        Returns:
            TWAP price
        """
        if not slices:
            return Decimal("0")

        total_value = Decimal("0")
        total_quantity = Decimal("0")

        for slice_data in slices:
            quantity = Decimal(slice_data["executed_quantity"])
            price = Decimal(slice_data["execution_price"])
            total_value += quantity * price
            total_quantity += quantity

        return total_value / total_quantity if total_quantity > 0 else Decimal("0")

    def calculate_implementation_shortfall(
        self, arrival_price: Decimal, execution_price: Decimal
    ) -> Decimal:
        """
        Calculate implementation shortfall (slippage from arrival price).

        Args:
            arrival_price: Price at order arrival
            execution_price: Average execution price

        Returns:
            Implementation shortfall in percent
        """
        if arrival_price == 0:
            return Decimal("0")

        shortfall = ((execution_price - arrival_price) / arrival_price) * Decimal("100")
        return shortfall.quantize(Decimal("0.0001"))

    async def enforce_participation_limit(
        self, slice_size: Decimal, symbol: str, max_participation: Decimal
    ) -> Decimal:
        """
        Enforce participation rate limit based on current volume.

        Args:
            slice_size: Target slice size
            symbol: Trading symbol
            max_participation: Maximum participation rate

        Returns:
            Adjusted slice size respecting participation limit
        """
        try:
            # Get current 24h volume
            ticker = await self.gateway.get_ticker(symbol)
            volume_24h = ticker.volume

            # Calculate volume per minute
            volume_per_minute = volume_24h / (24 * 60)

            # Maximum allowed based on participation rate
            max_allowed = volume_per_minute * max_participation

            # Check for volume anomaly
            if await self.market_data_service.is_volume_anomaly(symbol):
                # Reduce participation during anomalies
                max_allowed *= Decimal("0.5")
                logger.warning(
                    "Volume anomaly detected, reducing participation",
                    symbol=symbol,
                    original_max=str(volume_per_minute * max_participation),
                    reduced_max=str(max_allowed),
                )

            return min(slice_size, max_allowed)

        except Exception as e:
            logger.warning(
                "Failed to enforce participation limit", symbol=symbol, error=str(e)
            )
            # Conservative fallback
            return slice_size * Decimal("0.5")

    def _validate_duration(self, duration_minutes: int) -> None:
        """
        Validate TWAP duration.

        Args:
            duration_minutes: Requested duration

        Raises:
            ValidationError: If duration is invalid
        """
        if duration_minutes < MIN_DURATION_MINUTES:
            raise ValidationError(
                f"Duration must be at least {MIN_DURATION_MINUTES} minutes"
            )

        if duration_minutes > MAX_DURATION_MINUTES:
            raise ValidationError(
                f"Duration cannot exceed {MAX_DURATION_MINUTES} minutes"
            )

    async def _save_execution_to_db(self, execution: TwapExecution) -> None:
        """Save TWAP execution to database."""
        if self.repository:
            await self.repository.save_twap_execution(execution)

    async def _save_slice_to_db(self, slice_data: dict[str, Any]) -> None:
        """Save TWAP slice to database."""
        if self.repository:
            await self.repository.save_twap_slice(slice_data)

    async def _update_execution_in_db(self, execution: TwapExecution) -> None:
        """Update TWAP execution in database."""
        if self.repository:
            await self.repository.update_twap_execution(execution)

    # Implement required abstract methods from OrderExecutor

    async def execute_market_order(
        self, order: Order, confirmation_required: bool = True
    ) -> ExecutionResult:
        """
        Execute a market order using TWAP strategy.

        Routes to TWAP execution with default duration.
        """
        return await self.execute_twap(order, DEFAULT_DURATION_MINUTES)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        # Find if this is part of an active TWAP execution
        for execution in self.active_executions.values():
            if execution.symbol == symbol and execution.status == "ACTIVE":
                execution.status = "CANCELLED"
                if execution.background_task:
                    execution.background_task.cancel()
                await self._update_execution_in_db(execution)
                return True

        return await self.market_executor.cancel_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all orders."""
        cancelled_count = 0

        # Cancel active TWAP executions
        for execution_id, execution in list(self.active_executions.items()):
            if symbol is None or execution.symbol == symbol:
                if execution.status == "ACTIVE":
                    execution.status = "CANCELLED"
                    if execution.background_task:
                        execution.background_task.cancel()
                    await self._update_execution_in_db(execution)
                    cancelled_count += 1

        # Also cancel market orders
        cancelled_count += await self.market_executor.cancel_all_orders(symbol)

        return cancelled_count

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        return await self.market_executor.get_order_status(order_id, symbol)
