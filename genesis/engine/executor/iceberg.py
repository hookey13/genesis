"""
Iceberg order executor for Hunter tier in Project GENESIS.

This module implements sophisticated order slicing to minimize market impact
when executing large orders. Orders are automatically divided into smaller
slices with randomized timing and size variation.
"""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
from uuid import uuid4

import structlog

from genesis.core.exceptions import (
    OrderExecutionError,
    ValidationError,
)
from genesis.core.models import Account, TradingTier
from genesis.data.repository import Repository
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderBook
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


# Iceberg configuration constants
MIN_ORDER_VALUE_USDT = Decimal("200")  # Minimum order value for iceberg execution
MIN_SLICES = 3  # Minimum number of slices
MAX_SLICES = 10  # Maximum number of slices
SLICE_VARIATION_PERCENT = Decimal("20")  # ±20% size variation
MIN_DELAY_SECONDS = 1.0  # Minimum delay between slices
MAX_DELAY_SECONDS = 5.0  # Maximum delay between slices
SLIPPAGE_ABORT_THRESHOLD = Decimal("0.5")  # Abort if slippage exceeds 0.5%


@dataclass
class LiquidityProfile:
    """Market liquidity analysis result."""

    total_bid_volume: Decimal
    total_ask_volume: Decimal
    bid_depth_1pct: Decimal  # Volume to move price 1%
    ask_depth_1pct: Decimal
    bid_depth_2pct: Decimal  # Volume to move price 2%
    ask_depth_2pct: Decimal
    spread_percent: Decimal
    optimal_slice_count: int
    timestamp: datetime


@dataclass
class IcebergExecution:
    """Iceberg execution tracking."""

    execution_id: str
    order: Order
    total_slices: int
    slices: list[Order]
    slice_sizes: list[Decimal]
    slice_delays: list[float]
    completed_slices: int = 0
    failed_slices: int = 0
    cumulative_slippage: Decimal = Decimal("0")
    max_slice_slippage: Decimal = Decimal("0")
    status: str = "PENDING"
    abort_reason: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class IcebergOrderExecutor(OrderExecutor):
    """
    Iceberg order executor for Hunter tier.

    Automatically slices large orders into smaller pieces to minimize
    market impact. Includes smart sizing based on order book depth,
    random delays, and automatic abort on excessive slippage.
    """

    def __init__(
        self,
        gateway: BinanceGateway,
        account: Account,
        market_executor: MarketOrderExecutor,
        repository: Repository,
        risk_engine: Optional["RiskEngine"] = None,
        min_order_value: Decimal = MIN_ORDER_VALUE_USDT,
        slippage_threshold: Decimal = SLIPPAGE_ABORT_THRESHOLD,
    ):
        """
        Initialize the iceberg order executor.

        Args:
            gateway: Binance gateway for exchange interaction
            account: Trading account
            market_executor: Base market order executor for individual slices
            repository: Data repository for persistence
            risk_engine: Risk engine for position validation
            min_order_value: Minimum order value for iceberg execution
            slippage_threshold: Maximum slippage before abort
        """
        super().__init__(TradingTier.HUNTER)

        # Validate tier requirement
        if account.tier.value < TradingTier.HUNTER.value:
            raise OrderExecutionError(
                f"Iceberg execution requires {TradingTier.HUNTER.value} tier or above",
                details={"current_tier": account.tier.value},
            )

        self.gateway = gateway
        self.account = account
        self.market_executor = market_executor
        self.repository = repository
        self.risk_engine = risk_engine
        self.min_order_value = min_order_value
        self.slippage_threshold = slippage_threshold

        # Track active executions
        self.active_executions: dict[str, IcebergExecution] = {}

        # Order book cache (5-second TTL)
        self._order_book_cache: dict[str, tuple[OrderBook, datetime]] = {}
        self._cache_ttl = timedelta(seconds=5)

        logger.info(
            "Iceberg order executor initialized",
            account_id=account.account_id,
            tier=account.tier.value,
            min_order_value=str(min_order_value),
            slippage_threshold=str(slippage_threshold),
        )

    @requires_tier(TradingTier.HUNTER)
    async def execute_iceberg_order(
        self, order: Order, force_iceberg: bool = False
    ) -> ExecutionResult:
        """
        Execute an order using iceberg slicing if it meets criteria.

        Args:
            order: Order to execute
            force_iceberg: Force iceberg execution regardless of order value

        Returns:
            ExecutionResult with execution details

        Raises:
            OrderExecutionError: If execution fails
            SlippageAlert: If cumulative slippage exceeds threshold
        """
        try:
            # Validate order
            self.validate_order(order)

            # Get current ticker for value calculation
            ticker = await self.gateway.get_ticker(order.symbol)
            order_value = order.quantity * ticker.last_price

            # Check if order qualifies for iceberg execution
            if not force_iceberg and order_value < self.min_order_value:
                logger.info(
                    "Order value below iceberg threshold, using standard execution",
                    order_id=order.order_id,
                    order_value=str(order_value),
                    threshold=str(self.min_order_value),
                )
                return await self.market_executor.execute_market_order(
                    order, confirmation_required=False
                )

            logger.info(
                "Starting iceberg order execution",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=str(order.quantity),
                value_usdt=str(order_value),
            )

            # Analyze order book depth
            order_book = await self._get_cached_order_book(order.symbol)
            liquidity_profile = self.analyze_liquidity_depth(
                order_book, order.side, order_value
            )

            # Calculate slice sizes with variation
            slice_sizes = self.calculate_slice_sizes(order_value, liquidity_profile)

            # Generate random delays
            slice_delays = [self.generate_random_delay() for _ in slice_sizes]

            # Create iceberg execution tracker
            execution_id = str(uuid4())
            execution = IcebergExecution(
                execution_id=execution_id,
                order=order,
                total_slices=len(slice_sizes),
                slices=[],
                slice_sizes=slice_sizes,
                slice_delays=slice_delays,
                started_at=datetime.now(),
            )

            self.active_executions[execution_id] = execution

            # Save execution to database
            await self._save_execution_to_db(execution)

            # Execute slices sequentially with delays
            cumulative_quantity = Decimal("0")
            for i, (slice_size, delay) in enumerate(
                zip(slice_sizes, slice_delays, strict=False)
            ):
                slice_number = i + 1

                try:
                    # Calculate slice quantity based on size in USDT
                    slice_quantity = slice_size / ticker.last_price
                    slice_quantity = slice_quantity.quantize(order.quantity)

                    # Ensure we don't exceed total order quantity
                    remaining_quantity = order.quantity - cumulative_quantity
                    slice_quantity = min(slice_quantity, remaining_quantity)

                    # Create slice order
                    slice_order = Order(
                        order_id=str(uuid4()),
                        position_id=order.position_id,
                        client_order_id=self.generate_client_order_id(),
                        symbol=order.symbol,
                        type=OrderType.MARKET,
                        side=order.side,
                        price=None,
                        quantity=slice_quantity,
                        slice_number=slice_number,
                        total_slices=execution.total_slices,
                        created_at=datetime.now(),
                    )

                    execution.slices.append(slice_order)

                    logger.info(
                        f"Executing slice {slice_number}/{execution.total_slices}",
                        execution_id=execution_id,
                        slice_id=slice_order.order_id,
                        quantity=str(slice_quantity),
                        delay_seconds=delay,
                    )

                    # Execute slice
                    slice_result = await self.market_executor.execute_market_order(
                        slice_order, confirmation_required=False
                    )

                    if slice_result.success:
                        execution.completed_slices += 1
                        cumulative_quantity += slice_order.filled_quantity

                        # Update slippage tracking
                        if slice_result.slippage_percent:
                            execution.cumulative_slippage += (
                                slice_result.slippage_percent
                            )
                            execution.max_slice_slippage = max(
                                execution.max_slice_slippage,
                                abs(slice_result.slippage_percent),
                            )

                        # Check slippage threshold
                        avg_slippage = (
                            execution.cumulative_slippage / execution.completed_slices
                        )
                        if abs(avg_slippage) > self.slippage_threshold:
                            logger.warning(
                                "Slippage threshold exceeded, aborting iceberg execution",
                                execution_id=execution_id,
                                avg_slippage=str(avg_slippage),
                                threshold=str(self.slippage_threshold),
                            )
                            execution.status = "ABORTED"
                            execution.abort_reason = (
                                f"Slippage {avg_slippage}% exceeded threshold"
                            )
                            await self._abort_execution(execution)
                            break

                        # Save slice to database
                        await self._save_slice_to_db(
                            execution_id, slice_order, slice_result
                        )

                        # Delay before next slice (except for last slice)
                        if i < len(slice_sizes) - 1:
                            await asyncio.sleep(delay)
                    else:
                        execution.failed_slices += 1
                        logger.warning(
                            f"Slice {slice_number} failed",
                            execution_id=execution_id,
                            error=slice_result.error,
                        )

                        # Abort if too many failures
                        if execution.failed_slices >= 3:
                            execution.status = "FAILED"
                            execution.abort_reason = "Too many slice failures"
                            await self._abort_execution(execution)
                            break

                except Exception as e:
                    logger.error(
                        f"Error executing slice {slice_number}",
                        execution_id=execution_id,
                        error=str(e),
                    )
                    execution.failed_slices += 1

                    if execution.failed_slices >= 3:
                        execution.status = "FAILED"
                        execution.abort_reason = str(e)
                        await self._abort_execution(execution)
                        break

            # Finalize execution
            execution.completed_at = datetime.now()
            execution_time = (
                execution.completed_at - execution.started_at
            ).total_seconds()

            if execution.status == "PENDING":
                execution.status = "COMPLETED"

            # Update original order
            order.filled_quantity = cumulative_quantity
            order.status = (
                OrderStatus.FILLED
                if cumulative_quantity == order.quantity
                else OrderStatus.PARTIAL
            )
            order.total_slices = execution.total_slices

            # Calculate average slippage
            avg_slippage = (
                execution.cumulative_slippage / execution.completed_slices
                if execution.completed_slices > 0
                else Decimal("0")
            )

            # Update database
            await self._update_execution_in_db(execution)

            # Remove from active executions
            del self.active_executions[execution_id]

            logger.info(
                "Iceberg execution completed",
                execution_id=execution_id,
                status=execution.status,
                completed_slices=execution.completed_slices,
                failed_slices=execution.failed_slices,
                avg_slippage=str(avg_slippage),
                max_slippage=str(execution.max_slice_slippage),
                execution_time_seconds=execution_time,
            )

            return ExecutionResult(
                success=execution.status == "COMPLETED",
                order=order,
                message=f"Iceberg execution {execution.status.lower()}: {execution.completed_slices}/{execution.total_slices} slices",
                slippage_percent=avg_slippage,
                latency_ms=int(execution_time * 1000),
                error=execution.abort_reason,
            )

        except Exception as e:
            logger.error(
                "Iceberg execution failed", order_id=order.order_id, error=str(e)
            )
            raise OrderExecutionError(
                f"Failed to execute iceberg order: {e!s}", order_id=order.order_id
            )

    def calculate_slice_sizes(
        self, order_value: Decimal, liquidity_profile: LiquidityProfile
    ) -> list[Decimal]:
        """
        Calculate optimal slice sizes based on liquidity profile.

        Args:
            order_value: Total order value in USDT
            liquidity_profile: Market liquidity analysis

        Returns:
            List of slice sizes in USDT
        """
        # Determine number of slices
        slice_count = max(
            MIN_SLICES, min(liquidity_profile.optimal_slice_count, MAX_SLICES)
        )

        # Base slice size
        base_size = order_value / Decimal(str(slice_count))

        # Generate sizes with variation
        slice_sizes = []
        remaining_value = order_value

        for i in range(slice_count - 1):
            # Add random variation (±20%)
            variation = self.add_slice_variation(base_size, SLICE_VARIATION_PERCENT)

            # Ensure we don't exceed remaining value
            slice_size = min(variation, remaining_value)
            slice_sizes.append(slice_size)
            remaining_value -= slice_size

        # Last slice gets the remainder
        slice_sizes.append(remaining_value)

        # Shuffle to avoid predictable patterns
        random.shuffle(slice_sizes)

        logger.debug(
            "Calculated slice sizes",
            total_value=str(order_value),
            slice_count=slice_count,
            sizes=[str(s) for s in slice_sizes],
        )

        return slice_sizes

    def add_slice_variation(
        self, base_size: Decimal, variation_percent: Decimal
    ) -> Decimal:
        """
        Add random variation to slice size.

        Args:
            base_size: Base slice size
            variation_percent: Maximum variation percentage

        Returns:
            Varied slice size
        """
        # Generate random variation between -variation_percent and +variation_percent
        variation = Decimal(
            str(random.uniform(-float(variation_percent), float(variation_percent)))
        ) / Decimal("100")
        varied_size = base_size * (Decimal("1") + variation)

        # Ensure positive value
        return max(varied_size, Decimal("1"))

    def generate_random_delay(self) -> float:
        """
        Generate random delay between slices.

        Returns:
            Delay in seconds (1-5 seconds)
        """
        return random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)

    def analyze_liquidity_depth(
        self, order_book: OrderBook, side: OrderSide, order_value: Decimal
    ) -> LiquidityProfile:
        """
        Analyze order book liquidity depth.

        Args:
            order_book: Current order book
            side: Order side (buy/sell)
            order_value: Total order value

        Returns:
            Liquidity profile analysis
        """
        # Calculate total volumes
        total_bid_volume = sum(Decimal(str(level[1])) for level in order_book.bids)
        total_ask_volume = sum(Decimal(str(level[1])) for level in order_book.asks)

        # Calculate spread
        best_bid = (
            Decimal(str(order_book.bids[0][0])) if order_book.bids else Decimal("0")
        )
        best_ask = (
            Decimal(str(order_book.asks[0][0])) if order_book.asks else Decimal("0")
        )
        spread_percent = (
            ((best_ask - best_bid) / best_bid * Decimal("100"))
            if best_bid > 0
            else Decimal("0")
        )

        # Calculate depth to move price by 1% and 2%
        if side == OrderSide.BUY:
            # For buys, look at ask side
            depth_1pct = self._calculate_depth_to_price_level(
                order_book.asks, best_ask * Decimal("1.01"), is_ask=True
            )
            depth_2pct = self._calculate_depth_to_price_level(
                order_book.asks, best_ask * Decimal("1.02"), is_ask=True
            )
            available_liquidity = total_ask_volume
        else:
            # For sells, look at bid side
            depth_1pct = self._calculate_depth_to_price_level(
                order_book.bids, best_bid * Decimal("0.99"), is_ask=False
            )
            depth_2pct = self._calculate_depth_to_price_level(
                order_book.bids, best_bid * Decimal("0.98"), is_ask=False
            )
            available_liquidity = total_bid_volume

        # Calculate optimal slice count based on liquidity
        if order_value <= depth_1pct:
            optimal_slices = MIN_SLICES
        elif order_value <= depth_2pct:
            optimal_slices = 5
        else:
            # For large orders relative to liquidity, use more slices
            liquidity_ratio = (
                order_value / depth_2pct if depth_2pct > 0 else Decimal("10")
            )
            optimal_slices = min(int(liquidity_ratio * 3) + MIN_SLICES, MAX_SLICES)

        return LiquidityProfile(
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            bid_depth_1pct=depth_1pct if side == OrderSide.SELL else Decimal("0"),
            ask_depth_1pct=depth_1pct if side == OrderSide.BUY else Decimal("0"),
            bid_depth_2pct=depth_2pct if side == OrderSide.SELL else Decimal("0"),
            ask_depth_2pct=depth_2pct if side == OrderSide.BUY else Decimal("0"),
            spread_percent=spread_percent,
            optimal_slice_count=optimal_slices,
            timestamp=datetime.now(),
        )

    def _calculate_depth_to_price_level(
        self, levels: list[list[float]], target_price: Decimal, is_ask: bool
    ) -> Decimal:
        """
        Calculate total volume to reach a target price level.

        Args:
            levels: Order book levels (price, quantity pairs)
            target_price: Target price to reach
            is_ask: True for ask side, False for bid side

        Returns:
            Total volume in base currency
        """
        total_volume = Decimal("0")

        for price, quantity in levels:
            price_decimal = Decimal(str(price))
            quantity_decimal = Decimal(str(quantity))

            if is_ask:
                # For asks, accumulate until price exceeds target
                if price_decimal > target_price:
                    break
            else:
                # For bids, accumulate until price falls below target
                if price_decimal < target_price:
                    break

            total_volume += quantity_decimal * price_decimal

        return total_volume

    async def _get_cached_order_book(self, symbol: str) -> OrderBook:
        """
        Get order book with caching.

        Args:
            symbol: Trading symbol

        Returns:
            Order book snapshot
        """
        now = datetime.now()

        # Check cache
        if symbol in self._order_book_cache:
            cached_book, cached_time = self._order_book_cache[symbol]
            if now - cached_time < self._cache_ttl:
                return cached_book

        # Fetch fresh order book
        order_book = await self.gateway.get_order_book(symbol, depth=50)
        self._order_book_cache[symbol] = (order_book, now)

        return order_book

    async def _abort_execution(self, execution: IcebergExecution) -> None:
        """
        Abort an iceberg execution and cancel remaining slices.

        Args:
            execution: Execution to abort
        """
        logger.warning(
            "Aborting iceberg execution",
            execution_id=execution.execution_id,
            reason=execution.abort_reason,
            completed_slices=execution.completed_slices,
            total_slices=execution.total_slices,
        )

        # Update execution status in database
        await self._update_execution_in_db(execution)

        # Send high-priority abort event (would integrate with Event Bus)
        # await self.event_bus.publish(AbortEvent(execution_id=execution.execution_id))

    async def _save_execution_to_db(self, execution: IcebergExecution) -> None:
        """Save iceberg execution to database."""
        if self.repository:
            await self.repository.save_iceberg_execution(execution)

    async def _save_slice_to_db(
        self, execution_id: str, slice_order: Order, result: ExecutionResult
    ) -> None:
        """Save slice execution to database."""
        if self.repository:
            await self.repository.save_iceberg_slice(execution_id, slice_order, result)

    async def _update_execution_in_db(self, execution: IcebergExecution) -> None:
        """Update iceberg execution in database."""
        if self.repository:
            await self.repository.update_iceberg_execution(execution)

    # Implement required abstract methods from OrderExecutor

    async def execute_market_order(
        self, order: Order, confirmation_required: bool = True
    ) -> ExecutionResult:
        """
        Execute a market order, using iceberg if it qualifies.

        This method routes to iceberg execution for qualifying orders.
        """
        return await self.execute_iceberg_order(order, force_iceberg=False)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        return await self.market_executor.cancel_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders."""
        return await self.market_executor.cancel_all_orders(symbol)

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Get order status."""
        return await self.market_executor.get_order_status(order_id, symbol)

    async def rollback_partial_execution(
        self, execution_id: str, confirmed_by: str = None
    ) -> dict[str, Any]:
        """
        Rollback a partially filled iceberg execution.

        Args:
            execution_id: Execution to rollback
            confirmed_by: Manual confirmation from authorized user (required)

        Returns:
            Rollback result details
        """
        if not confirmed_by:
            raise ValidationError("Manual confirmation required for rollbacks")

        if execution_id not in self.active_executions:
            raise OrderExecutionError(f"Execution {execution_id} not found")

        execution = self.active_executions[execution_id]

        logger.warning(
            "Starting rollback for partial execution",
            execution_id=execution_id,
            completed_slices=execution.completed_slices,
        )

        # Calculate rollback details
        rollback_orders = []
        total_rollback_quantity = Decimal("0")

        for slice_order in execution.slices:
            if slice_order.status == OrderStatus.FILLED:
                # Create compensating sell/buy order
                rollback_order = Order(
                    order_id=str(uuid4()),
                    position_id=slice_order.position_id,
                    client_order_id=self.generate_client_order_id(),
                    symbol=slice_order.symbol,
                    type=OrderType.MARKET,
                    side=(
                        OrderSide.SELL
                        if slice_order.side == OrderSide.BUY
                        else OrderSide.BUY
                    ),
                    price=None,
                    quantity=slice_order.filled_quantity,
                    created_at=datetime.now(),
                )
                rollback_orders.append(rollback_order)
                total_rollback_quantity += slice_order.filled_quantity

        # Execute rollback orders
        rollback_results = []
        rollback_cost = Decimal("0")

        for rollback_order in rollback_orders:
            try:
                result = await self.market_executor.execute_market_order(
                    rollback_order, confirmation_required=False
                )
                rollback_results.append(result)

                # Calculate cost (fees + slippage)
                if result.slippage_percent:
                    rollback_cost += (
                        abs(result.slippage_percent) * rollback_order.quantity
                    )

            except Exception as e:
                logger.error(
                    "Failed to execute rollback order",
                    rollback_order_id=rollback_order.order_id,
                    error=str(e),
                )

        # Update execution status
        execution.status = "ROLLED_BACK"
        await self._update_execution_in_db(execution)

        # Store rollback history with confirmation details
        await self._store_rollback_history(
            execution_id=execution_id,
            confirmed_by=confirmed_by,
            rollback_orders=rollback_orders,
            rollback_cost=rollback_cost,
        )

        return {
            "execution_id": execution_id,
            "rollback_orders": len(rollback_orders),
            "rollback_quantity": str(total_rollback_quantity),
            "rollback_cost_estimate": str(rollback_cost),
            "confirmed_by": confirmed_by,
            "success": len(rollback_results) == len(rollback_orders),
        }

    async def _update_execution_in_db(self, execution: IcebergExecution):
        """Update execution status in database."""
        # TODO: Implement database update
        pass

    async def _store_rollback_history(
        self,
        execution_id: str,
        confirmed_by: str,
        rollback_orders: list[Order],
        rollback_cost: Decimal,
    ):
        """Store rollback history with confirmation details."""
        # TODO: Implement rollback history storage
        pass
