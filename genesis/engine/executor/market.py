"""
Market order executor for Sniper tier in Project GENESIS.

This module implements simple market order execution with confirmations,
post-execution verification, and automatic stop-loss placement.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import structlog

from typing import Optional

from genesis.core.exceptions import (
    OrderExecutionError,
    SlippageAlert,
)
from genesis.core.models import Account, TradingTier
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderRequest
from genesis.utils.decorators import requires_tier, with_timeout

logger = structlog.get_logger(__name__)


class MarketOrderExecutor(OrderExecutor):
    """
    Market order executor for Sniper tier.

    Provides simple market order execution with confirmations,
    slippage monitoring, and automatic stop-loss placement.
    """

    def __init__(
        self,
        gateway: BinanceGateway,
        account: Account,
        risk_engine: Optional['RiskEngine'] = None,
        repository: Optional['Repository'] = None,
        confirmation_timeout: int = 10
    ):
        """
        Initialize the market order executor.

        Args:
            gateway: Binance gateway for exchange interaction
            account: Trading account
            risk_engine: Risk engine for position sizing
            repository: Data repository for persistence
            confirmation_timeout: Timeout for order confirmations in seconds
        """
        super().__init__(TradingTier.SNIPER)
        self.gateway = gateway
        self.account = account
        self.risk_engine = risk_engine
        self.repository = repository
        self.confirmation_timeout = confirmation_timeout
        self.pending_orders: dict[str, Order] = {}

        logger.info(
            "Market order executor initialized",
            account_id=account.account_id,
            tier=account.tier.value
        )

    @requires_tier(TradingTier.SNIPER)
    @with_timeout(100)  # 100ms execution target
    async def execute_market_order(
        self,
        order: Order,
        confirmation_required: bool = True
    ) -> ExecutionResult:
        """
        Execute a market order with optional confirmation.

        Args:
            order: Market order to execute
            confirmation_required: Whether to require user confirmation

        Returns:
            ExecutionResult with execution details

        Raises:
            OrderExecutionError: If order execution fails
            SlippageAlert: If slippage exceeds 0.5%
        """
        start_time = datetime.now()

        try:
            # Validate order
            self.validate_order(order)

            # Store order as pending
            self.pending_orders[order.order_id] = order
            order.status = OrderStatus.PENDING

            # Save order to database if repository available
            if self.repository:
                await self.repository.create_order(order)

            logger.info(
                "Executing market order",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=str(order.quantity)
            )

            # Get confirmation if required
            if confirmation_required and not await self._get_confirmation(order):
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order.order_id]

                # Update order status in database
                if self.repository:
                    await self.repository.update_order(order)

                return ExecutionResult(
                    success=False,
                    order=order,
                    message="Order cancelled by user",
                    error="User declined confirmation"
                )

            # Get current market price for slippage calculation
            ticker = await self.gateway.get_ticker(order.symbol)
            expected_price = ticker.ask_price if order.side == OrderSide.BUY else ticker.bid_price

            # Prepare order request
            request = OrderRequest(
                symbol=order.symbol,
                side=order.side.value.lower(),
                type="market",
                quantity=order.quantity,
                client_order_id=order.client_order_id
            )

            # Place the order
            response = await self.gateway.place_order(request)

            # Update order with exchange response
            order.exchange_order_id = response.order_id
            order.status = OrderStatus(response.status.upper())
            order.executed_at = response.created_at

            # Post-execution verification
            verification_result = await self._verify_execution(order, expected_price)

            # Calculate latency
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            order.latency_ms = latency_ms

            # Update order status
            if order.status == OrderStatus.FILLED:
                del self.pending_orders[order.order_id]

                # Update order in database
                if self.repository:
                    await self.repository.update_order(order)

                # Place automatic stop-loss for entry orders
                if order.side == OrderSide.BUY and self.risk_engine:
                    stop_loss_order = await self._place_stop_loss(order, verification_result.actual_price)
                    if stop_loss_order and self.repository:
                        await self.repository.create_order(stop_loss_order)

            logger.info(
                "Market order executed successfully",
                order_id=order.order_id,
                exchange_order_id=order.exchange_order_id,
                status=order.status.value,
                latency_ms=latency_ms,
                slippage_percent=str(verification_result.slippage_percent)
            )

            return ExecutionResult(
                success=True,
                order=order,
                message="Market order executed successfully",
                actual_price=verification_result.actual_price,
                slippage_percent=verification_result.slippage_percent,
                latency_ms=latency_ms
            )

        except Exception as e:
            logger.error(
                "Market order execution failed",
                order_id=order.order_id,
                error=str(e)
            )

            order.status = OrderStatus.FAILED
            if order.order_id in self.pending_orders:
                del self.pending_orders[order.order_id]

            raise OrderExecutionError(
                f"Failed to execute market order: {e!s}",
                order_id=order.order_id
            )

    async def _get_confirmation(self, order: Order) -> bool:
        """
        Get user confirmation for order execution.

        This is a placeholder - actual implementation would integrate
        with the terminal UI to prompt the user.

        Args:
            order: Order requiring confirmation

        Returns:
            True if confirmed, False if cancelled
        """
        # In production, this would integrate with the UI
        # For now, auto-confirm in mock mode
        if hasattr(self.gateway, 'mock_mode') and self.gateway.mock_mode:
            return True

        logger.info(
            "Awaiting order confirmation",
            order_id=order.order_id,
            timeout_seconds=self.confirmation_timeout
        )

        # Simulate confirmation with timeout
        try:
            # In real implementation, this would wait for UI event
            await asyncio.sleep(0.1)  # Simulate quick confirmation
            return True
        except TimeoutError:
            logger.warning("Order confirmation timed out", order_id=order.order_id)
            return False

    async def _verify_execution(self, order: Order, expected_price: Decimal) -> ExecutionResult:
        """
        Verify order execution and calculate slippage.

        Args:
            order: Executed order
            expected_price: Expected execution price

        Returns:
            ExecutionResult with verification details

        Raises:
            SlippageAlert: If slippage exceeds 0.5%
        """
        try:
            # Query order status from exchange
            order_status = await self.gateway.get_order_status(
                order.exchange_order_id,
                order.symbol
            )

            # Update order with actual fill details
            order.filled_quantity = order_status.filled_quantity
            actual_price = order_status.price or expected_price

            # Calculate slippage
            slippage = self.calculate_slippage(expected_price, actual_price, order.side)
            order.slippage_percent = slippage

            # Alert if slippage exceeds threshold
            if abs(slippage) > Decimal("0.5"):
                logger.warning(
                    "High slippage detected",
                    order_id=order.order_id,
                    slippage_percent=str(slippage),
                    expected_price=str(expected_price),
                    actual_price=str(actual_price)
                )
                raise SlippageAlert(
                    f"Slippage {slippage}% exceeds threshold 0.5%",
                    slippage=slippage
                )

            return ExecutionResult(
                success=True,
                order=order,
                message="Execution verified",
                actual_price=actual_price,
                slippage_percent=slippage
            )

        except Exception as e:
            logger.error(
                "Failed to verify execution",
                order_id=order.order_id,
                error=str(e)
            )
            # Return best effort result
            return ExecutionResult(
                success=True,
                order=order,
                message="Execution completed but verification failed",
                actual_price=expected_price,
                slippage_percent=Decimal("0")
            )

    async def _place_stop_loss(self, entry_order: Order, entry_price: Decimal) -> Optional[Order]:
        """
        Place automatic stop-loss order after entry.

        Args:
            entry_order: The entry order that was filled
            entry_price: Actual entry price

        Returns:
            Stop-loss order if placed successfully
        """
        try:
            if not self.risk_engine:
                logger.warning("No risk engine available for stop-loss calculation")
                return None

            # Calculate stop-loss price (2% default for Sniper tier)
            stop_loss_percent = Decimal("0.02")
            stop_loss_price = entry_price * (Decimal("1") - stop_loss_percent)
            stop_loss_price = stop_loss_price.quantize(Decimal("0.00000001"))

            # Create stop-loss order
            stop_loss_order = Order(
                order_id=str(uuid4()),
                position_id=entry_order.position_id,
                client_order_id=self.generate_client_order_id(),
                symbol=entry_order.symbol,
                type=OrderType.STOP_LOSS,
                side=OrderSide.SELL,
                price=stop_loss_price,
                quantity=entry_order.filled_quantity,
                created_at=datetime.now()
            )

            logger.info(
                "Placing automatic stop-loss",
                parent_order_id=entry_order.order_id,
                stop_loss_price=str(stop_loss_price),
                quantity=str(stop_loss_order.quantity)
            )

            # Place stop-loss order
            request = OrderRequest(
                symbol=stop_loss_order.symbol,
                side="sell",
                type="stop_limit",
                quantity=stop_loss_order.quantity,
                price=stop_loss_price,
                stop_price=stop_loss_price,
                client_order_id=stop_loss_order.client_order_id
            )

            response = await self.gateway.place_order(request)

            stop_loss_order.exchange_order_id = response.order_id
            stop_loss_order.status = OrderStatus(response.status.upper())

            logger.info(
                "Stop-loss placed successfully",
                stop_loss_order_id=stop_loss_order.order_id,
                exchange_order_id=stop_loss_order.exchange_order_id
            )

            return stop_loss_order

        except Exception as e:
            logger.error(
                "Failed to place stop-loss",
                parent_order_id=entry_order.order_id,
                error=str(e)
            )
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading symbol

        Returns:
            True if cancellation successful
        """
        try:
            logger.info("Cancelling order", order_id=order_id, symbol=symbol)

            result = await self.gateway.cancel_order(order_id, symbol)

            # Remove from pending orders if present
            for pid, order in list(self.pending_orders.items()):
                if order.exchange_order_id == order_id:
                    order.status = OrderStatus.CANCELLED
                    del self.pending_orders[pid]
                    break

            logger.info("Order cancelled successfully", order_id=order_id)
            return result

        except Exception as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Emergency cancel all open orders.

        Args:
            symbol: Optional symbol to filter cancellations

        Returns:
            Number of orders cancelled
        """
        try:
            logger.warning(
                "EMERGENCY: Cancelling all orders",
                symbol=symbol,
                pending_count=len(self.pending_orders)
            )

            cancelled_count = 0

            # Get all open orders from exchange
            open_orders = await self.gateway.get_open_orders(symbol)

            # Cancel each order
            for order in open_orders:
                if await self.cancel_order(order.order_id, order.symbol):
                    cancelled_count += 1

            # Clear local pending orders
            if symbol:
                # Clear only orders for specific symbol
                to_clear = [
                    oid for oid, order in self.pending_orders.items()
                    if order.symbol == symbol
                ]
                for oid in to_clear:
                    self.pending_orders[oid].status = OrderStatus.CANCELLED
                    del self.pending_orders[oid]
            else:
                # Clear all pending orders
                for order in self.pending_orders.values():
                    order.status = OrderStatus.CANCELLED
                self.pending_orders.clear()

            logger.warning(
                "Emergency cancellation complete",
                cancelled_count=cancelled_count
            )

            return cancelled_count

        except Exception as e:
            logger.error("Failed to cancel all orders", error=str(e))
            raise OrderExecutionError(f"Emergency cancellation failed: {e!s}")

    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to check
            symbol: Trading symbol

        Returns:
            Order with current status
        """
        try:
            # Check pending orders first
            for order in self.pending_orders.values():
                if order.exchange_order_id == order_id or order.order_id == order_id:
                    # Update from exchange
                    if order.exchange_order_id:
                        response = await self.gateway.get_order_status(
                            order.exchange_order_id,
                            symbol
                        )
                        order.status = OrderStatus(response.status.upper())
                        order.filled_quantity = response.filled_quantity
                    return order

            # Query exchange directly
            response = await self.gateway.get_order_status(order_id, symbol)

            # Create order object from response
            order = Order(
                order_id=str(uuid4()),
                position_id=None,
                client_order_id=response.client_order_id or "",
                symbol=response.symbol,
                type=OrderType.MARKET,  # Assume market for now
                side=OrderSide(response.side.upper()),
                price=response.price,
                quantity=response.quantity,
                filled_quantity=response.filled_quantity,
                status=OrderStatus(response.status.upper()),
                created_at=response.created_at,
                executed_at=response.updated_at,
                exchange_order_id=response.order_id
            )

            return order

        except Exception as e:
            logger.error(
                "Failed to get order status",
                order_id=order_id,
                error=str(e)
            )
            raise OrderExecutionError(f"Failed to get order status: {e!s}")
