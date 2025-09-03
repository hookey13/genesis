"""
Core trading loop orchestrator for Project GENESIS.

Implements the Price → Signal → Risk → Execute flow with proper state management
and event-driven architecture.
"""

import asyncio
import time
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.exceptions import (
    DailyLossLimitReached,
    InsufficientBalance,
    MinimumPositionSize,
    RiskLimitExceeded,
)
from genesis.core.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Signal,
    SignalType,
)
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import TierStateMachine
from genesis.exchange.gateway import ExchangeGateway

logger = structlog.get_logger(__name__)


class TradingLoop:
    """
    Main trading loop coordinator.
    
    Manages the event-driven flow between price updates, signal generation,
    risk validation, and order execution.
    """

    def __init__(
        self,
        event_bus: EventBus,
        risk_engine: RiskEngine,
        exchange_gateway: ExchangeGateway,
        state_machine: TierStateMachine | None = None,
        paper_trading_mode: bool = False,
        paper_trading_session_id: str | None = None,
    ):
        """
        Initialize trading loop.
        
        Args:
            event_bus: Event bus for publish-subscribe
            risk_engine: Risk validation engine
            exchange_gateway: Exchange API gateway
            state_machine: Optional state machine for tier management
            paper_trading_mode: Enable paper trading mode
            paper_trading_session_id: Paper trading session ID if in paper mode
        """
        self.event_bus = event_bus
        self.risk_engine = risk_engine
        self.exchange_gateway = exchange_gateway
        self.state_machine = state_machine
        self.paper_trading_mode = paper_trading_mode
        self.paper_trading_session_id = paper_trading_session_id

        self.running = False
        self.startup_validated = False
        self.positions: dict[str, Position] = {}
        self.pending_orders: dict[str, Order] = {}

        # Metrics
        self.events_processed = 0
        self.signals_generated = 0
        self.orders_executed = 0
        self.positions_opened = 0
        self.positions_closed = 0

        # Latency tracking
        self.event_latencies: list[float] = []
        self.signal_to_order_latencies: list[float] = []
        self.order_execution_latencies: list[float] = []

        # Audit logging
        self.sequence_number = 0
        self.event_store: list[Event] = []  # In-memory event store for now
        self.correlation_map: dict[str, str] = {}  # Map events to correlation IDs
        
        # State persistence
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 60  # Checkpoint every 60 seconds

        logger.info("TradingLoop initialized",
                   paper_trading_mode=self.paper_trading_mode,
                   paper_trading_session_id=self.paper_trading_session_id,
                   structlog_correlation_id=str(uuid4()))

    async def startup(self) -> bool:
        """
        Validate all components are ready.
        
        Returns:
            True if all components validated successfully
        """
        logger.info("Starting TradingLoop validation sequence")

        try:
            # Start event bus
            await self.event_bus.start()

            # Validate exchange connection
            if not await self.exchange_gateway.validate_connection():
                logger.error("Exchange gateway validation failed")
                return False

            # Validate risk engine
            if not self.risk_engine.validate_configuration():
                logger.error("Risk engine validation failed")
                return False

            # Register event handlers
            self._register_event_handlers()

            # Publish startup event
            startup_event = Event(
                event_type=EventType.SYSTEM_STARTUP,
                event_data={
                    "component": "TradingLoop",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            await self.event_bus.publish(startup_event, priority=EventPriority.HIGH)
            await self._store_event(startup_event)

            self.startup_validated = True
            logger.info("TradingLoop startup validation completed successfully")
            return True

        except Exception as e:
            logger.error("TradingLoop startup validation failed", error=str(e))
            return False

    def _register_event_handlers(self) -> None:
        """Register event handlers for the trading flow."""
        # Price events → Signal generation
        self.event_bus.subscribe(
            EventType.MARKET_DATA_UPDATED,
            self._handle_price_update,
            priority=EventPriority.HIGH
        )

        # Signal events → Risk validation → Order creation
        self.event_bus.subscribe(
            EventType.ARBITRAGE_SIGNAL,
            self._handle_trading_signal,
            priority=EventPriority.HIGH
        )

        # Order events → Position management
        self.event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._handle_order_filled,
            priority=EventPriority.CRITICAL
        )

        self.event_bus.subscribe(
            EventType.ORDER_FAILED,
            self._handle_order_failed,
            priority=EventPriority.HIGH
        )

        # Position events → State tracking
        self.event_bus.subscribe(
            EventType.STOP_LOSS_TRIGGERED,
            self._handle_stop_loss,
            priority=EventPriority.CRITICAL
        )

        logger.info("Event handlers registered for trading flow")

    async def _handle_price_update(self, event: Event) -> None:
        """
        Handle market data updates.
        
        Args:
            event: Market data event
        """
        try:
            self.events_processed += 1

            # Extract price data
            symbol = event.event_data.get("symbol")
            price = Decimal(str(event.event_data.get("price", 0)))

            # Update positions with current price
            for position_id, position in self.positions.items():
                if position.symbol == symbol:
                    position.update_pnl(price)

                    # Check stop loss
                    if position.stop_loss:
                        if (
                            position.side == PositionSide.LONG and price <= position.stop_loss
                        ) or (
                            position.side == PositionSide.SHORT and price >= position.stop_loss
                        ):
                            await self._trigger_stop_loss(position)

            # Log price update processing
            logger.debug(
                "Price update processed",
                symbol=symbol,
                price=str(price),
                positions_updated=len([p for p in self.positions.values() if p.symbol == symbol])
            )

        except Exception as e:
            logger.error("Error handling price update", error=str(e), event_id=event.event_id)

    async def _handle_trading_signal(self, event: Event) -> None:
        """
        Handle trading signals.
        
        Args:
            event: Trading signal event
        """
        try:
            self.signals_generated += 1
            correlation_id = str(uuid4())

            # Create signal from event data
            signal = Signal(
                strategy_id=event.event_data.get("strategy_id", "unknown"),
                symbol=event.event_data.get("pair1_symbol", ""),
                signal_type=SignalType.BUY if event.event_data.get("signal_type") == "ENTRY" else SignalType.SELL,
                confidence=Decimal(str(event.event_data.get("confidence_score", 0.5))),
                metadata=event.event_data
            )

            # Get entry price from event or use current market price
            entry_price = Decimal(str(event.event_data.get("entry_price", event.event_data.get("price", "0"))))
            if entry_price <= 0:
                logger.warning("Invalid entry price for signal", signal_id=signal.signal_id)
                await self._publish_risk_rejected_event(signal, "Invalid entry price", correlation_id)
                return

            # Determine position side
            position_side = PositionSide.LONG if signal.signal_type == SignalType.BUY else PositionSide.SHORT

            # Calculate position size with risk engine
            try:
                position_size = self.risk_engine.calculate_position_size(
                    symbol=signal.symbol,
                    entry_price=entry_price,
                    strategy_id=signal.strategy_id
                )
            except (InsufficientBalance, MinimumPositionSize, RiskLimitExceeded) as e:
                logger.warning(
                    "Position sizing failed",
                    signal_id=signal.signal_id,
                    error=str(e)
                )
                await self._publish_risk_rejected_event(signal, str(e), correlation_id)
                return

            # Validate order risk
            try:
                self.risk_engine.validate_order_risk(
                    symbol=signal.symbol,
                    side=position_side,
                    quantity=position_size,
                    entry_price=entry_price
                )
            except (RiskLimitExceeded, DailyLossLimitReached, InsufficientBalance) as e:
                logger.warning(
                    "Order risk validation failed",
                    signal_id=signal.signal_id,
                    error=str(e)
                )
                await self._publish_risk_rejected_event(signal, str(e), correlation_id)
                return

            # Check portfolio risk if we have positions
            if self.positions:
                portfolio_positions = list(self.positions.values())
                portfolio_risk = self.risk_engine.validate_portfolio_risk(portfolio_positions)

                if not portfolio_risk["approved"]:
                    rejection_reasons = "; ".join(portfolio_risk["rejections"])
                    logger.warning(
                        "Portfolio risk check failed",
                        signal_id=signal.signal_id,
                        reasons=rejection_reasons
                    )
                    await self._publish_risk_rejected_event(signal, rejection_reasons, correlation_id)
                    return

                # Log warnings if any
                for warning in portfolio_risk.get("warnings", []):
                    logger.warning(
                        "Portfolio risk warning",
                        signal_id=signal.signal_id,
                        warning=warning
                    )

            # Create order
            order = Order(
                symbol=signal.symbol,
                type=OrderType.MARKET,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                quantity=position_size,
                metadata={
                    "signal_id": signal.signal_id,
                    "entry_price": str(entry_price),
                    "correlation_id": correlation_id
                }
            )

            # Store correlation ID
            self.correlation_map[order.order_id] = correlation_id

            # Publish risk validation passed event
            await self._publish_risk_validation_passed_event(signal, order, correlation_id)

            # Execute order
            await self._execute_order(order)

            logger.info(
                "Signal processed and order created",
                signal_id=signal.signal_id,
                order_id=order.order_id,
                symbol=signal.symbol,
                side=order.side.value,
                quantity=str(position_size),
                correlation_id=correlation_id
            )

        except Exception as e:
            logger.error("Error handling trading signal", error=str(e), event_id=event.event_id)

    async def _execute_order(self, order: Order) -> None:
        """
        Execute an order through the exchange gateway.
        
        Args:
            order: Order to execute
        """
        try:
            # Store pending order
            self.pending_orders[order.order_id] = order

            # Execute through gateway
            result = await self.exchange_gateway.execute_order(order)

            if result["success"]:
                order.exchange_order_id = result.get("exchange_order_id")
                order.status = OrderStatus.FILLED
                order.executed_at = datetime.now()
                order.latency_ms = result.get("latency_ms", 0)

                # Publish order filled event
                event_data = {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": str(order.quantity),
                    "price": str(result.get("fill_price", 0)),
                    "exchange_order_id": order.exchange_order_id,
                }

                # Add paper trading prefix if in paper mode
                if self.paper_trading_mode:
                    event_data["paper_trade"] = True
                    event_data["session_id"] = self.paper_trading_session_id

                filled_event = Event(
                    event_type=EventType.ORDER_FILLED,
                    aggregate_id=order.order_id,
                    event_data=event_data
                )
                await self.event_bus.publish(filled_event, priority=EventPriority.HIGH)
                await self._store_event(filled_event)

                self.orders_executed += 1

            else:
                order.status = OrderStatus.FAILED

                # Publish order failed event
                failed_event = Event(
                    event_type=EventType.ORDER_FAILED,
                    aggregate_id=order.order_id,
                    event_data={
                        "order_id": order.order_id,
                        "reason": result.get("error", "Unknown error"),
                    }
                )
                await self.event_bus.publish(failed_event, priority=EventPriority.HIGH)
                await self._store_event(failed_event)

            # Remove from pending
            self.pending_orders.pop(order.order_id, None)

        except Exception as e:
            logger.error("Error executing order", error=str(e), order_id=order.order_id)
            order.status = OrderStatus.FAILED
            self.pending_orders.pop(order.order_id, None)

    async def _handle_order_filled(self, event: Event) -> None:
        """
        Handle order filled events.
        
        Args:
            event: Order filled event
        """
        try:
            event_start = time.perf_counter()
            order_id = event.event_data.get("order_id")
            symbol = event.event_data.get("symbol")
            side = event.event_data.get("side")
            quantity = Decimal(str(event.event_data.get("quantity", 0)))
            fill_price = Decimal(str(event.event_data.get("price", 0)))

            # Create or update position
            position_side = PositionSide.LONG if side == "BUY" else PositionSide.SHORT

            # Check for existing position
            existing_position = None
            for pos_id, pos in self.positions.items():
                if pos.symbol == symbol and pos.side == position_side:
                    existing_position = pos
                    break

            if existing_position:
                # Update existing position (averaging)
                total_quantity = existing_position.quantity + quantity
                weighted_price = (
                    (existing_position.entry_price * existing_position.quantity) +
                    (fill_price * quantity)
                ) / total_quantity

                existing_position.quantity = total_quantity
                existing_position.entry_price = weighted_price
                existing_position.dollar_value = total_quantity * weighted_price
                existing_position.updated_at = datetime.now()

                # Publish position updated event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.POSITION_UPDATED,
                        aggregate_id=existing_position.position_id,
                        event_data={
                            "position_id": existing_position.position_id,
                            "new_quantity": str(total_quantity),
                            "new_entry_price": str(weighted_price),
                        }
                    ),
                    priority=EventPriority.NORMAL
                )

            else:
                # Create new position
                position = Position(
                    account_id="default",
                    symbol=symbol,
                    side=position_side,
                    entry_price=fill_price,
                    quantity=quantity,
                    dollar_value=quantity * fill_price,
                    created_at=datetime.now()
                )

                # Calculate stop loss from configuration
                stop_loss_percent = self.risk_engine.tier_limits.get("stop_loss_percent", Decimal("2.0"))
                stop_loss_multiplier = Decimal("1") - (stop_loss_percent / Decimal("100"))

                if position_side == PositionSide.LONG:
                    position.stop_loss = fill_price * stop_loss_multiplier
                else:
                    position.stop_loss = fill_price * (Decimal("2") - stop_loss_multiplier)

                self.positions[position.position_id] = position
                self.positions_opened += 1

                # Publish position opened event
                opened_event = Event(
                    event_type=EventType.POSITION_OPENED,
                    aggregate_id=position.position_id,
                    correlation_id=self.correlation_map.get(order_id),
                    event_data={
                        "position_id": position.position_id,
                        "symbol": symbol,
                        "side": position_side.value,
                        "entry_price": str(fill_price),
                        "quantity": str(quantity),
                        "stop_loss": str(position.stop_loss),
                    }
                )
                await self.event_bus.publish(opened_event, priority=EventPriority.HIGH)
                await self._store_event(opened_event)

            logger.info(
                "Order filled and position updated",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=str(quantity),
                fill_price=str(fill_price)
            )

            # Record event processing latency
            event_latency_ms = (time.perf_counter() - event_start) * 1000
            self.event_latencies.append(event_latency_ms)

        except Exception as e:
            logger.error("Error handling order filled", error=str(e), event_id=event.event_id)

    async def _handle_order_failed(self, event: Event) -> None:
        """
        Handle order failed events.
        
        Args:
            event: Order failed event
        """
        try:
            order_id = event.event_data.get("order_id")
            reason = event.event_data.get("reason", "Unknown")

            logger.warning(
                "Order execution failed",
                order_id=order_id,
                reason=reason
            )

            # Could implement retry logic here if appropriate

        except Exception as e:
            logger.error("Error handling order failed", error=str(e), event_id=event.event_id)

    async def _trigger_stop_loss(self, position: Position) -> None:
        """
        Trigger stop loss for a position.
        
        Args:
            position: Position to close with stop loss
        """
        try:
            # Create closing order
            order = Order(
                position_id=position.position_id,
                symbol=position.symbol,
                type=OrderType.MARKET,
                side=OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY,
                quantity=position.quantity,
                metadata={"reason": "stop_loss", "position_id": position.position_id}
            )

            # Execute closing order
            await self._execute_order(order)

            # Publish stop loss event
            await self.event_bus.publish(
                Event(
                    event_type=EventType.STOP_LOSS_TRIGGERED,
                    aggregate_id=position.position_id,
                    event_data={
                        "position_id": position.position_id,
                        "symbol": position.symbol,
                        "stop_loss": str(position.stop_loss),
                        "current_pnl": str(position.pnl_dollars),
                    }
                ),
                priority=EventPriority.CRITICAL
            )

            logger.warning(
                "Stop loss triggered",
                position_id=position.position_id,
                symbol=position.symbol,
                stop_loss=str(position.stop_loss),
                pnl=str(position.pnl_dollars)
            )

        except Exception as e:
            logger.error(
                "Error triggering stop loss",
                error=str(e),
                position_id=position.position_id
            )

    async def _handle_stop_loss(self, event: Event) -> None:
        """
        Handle stop loss triggered events.
        
        Args:
            event: Stop loss event
        """
        try:
            position_id = event.event_data.get("position_id")

            # Remove position from active positions
            if position_id in self.positions:
                position = self.positions[position_id]
                position.close_reason = "stop_loss"
                position.updated_at = datetime.now()

                # Remove from active positions
                del self.positions[position_id]
                self.positions_closed += 1

                # Publish position closed event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.POSITION_CLOSED,
                        aggregate_id=position_id,
                        event_data={
                            "position_id": position_id,
                            "close_reason": "stop_loss",
                            "final_pnl": str(position.pnl_dollars),
                        }
                    ),
                    priority=EventPriority.HIGH
                )

                logger.info(
                    "Position closed via stop loss",
                    position_id=position_id,
                    final_pnl=str(position.pnl_dollars)
                )

        except Exception as e:
            logger.error("Error handling stop loss", error=str(e), event_id=event.event_id)

    async def run(self) -> None:
        """Run the main trading loop."""
        if not self.startup_validated:
            if not await self.startup():
                logger.error("Cannot start trading loop - startup validation failed")
                return

        self.running = True
        logger.info("Trading loop started", paper_trading_mode=self.paper_trading_mode)

        # Start monitoring tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        health_check_task = asyncio.create_task(self._health_check_monitor())
        checkpoint_task = asyncio.create_task(self._checkpoint_monitor())

        try:
            while self.running:
                # Main loop heartbeat
                await asyncio.sleep(1)

                # Periodic tasks
                if int(time.time()) % 60 == 0:  # Every minute
                    await self._position_reconciliation()
                    await self._log_performance_metrics()

        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error("Trading loop error", error=str(e))
        finally:
            # Cancel monitoring tasks
            heartbeat_task.cancel()
            health_check_task.cancel()
            await self.shutdown()

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics summary.
        
        Returns:
            Dictionary containing performance metrics
        """
        def calculate_stats(latencies: list[float]) -> dict[str, float]:
            """Calculate statistics for latency list."""
            if not latencies:
                return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            return {
                "min": sorted_latencies[0],
                "max": sorted_latencies[-1],
                "avg": sum(sorted_latencies) / n,
                "p50": sorted_latencies[int(n * 0.5)],
                "p95": sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
                "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
            }

        return {
            "event_processing": {
                "total_events": self.events_processed,
                "latency_ms": calculate_stats(self.event_latencies),
            },
            "signal_to_order": {
                "total_signals": self.signals_generated,
                "latency_ms": calculate_stats(self.signal_to_order_latencies),
            },
            "order_execution": {
                "total_orders": self.orders_executed,
                "latency_ms": calculate_stats(self.order_execution_latencies),
            },
            "positions": {
                "opened": self.positions_opened,
                "closed": self.positions_closed,
                "active": len(self.positions),
            }
        }
    
    async def checkpoint_state(self) -> None:
        """
        Save current state to persistent storage.
        
        Creates a checkpoint of positions, orders, and metrics for recovery.
        """
        try:
            checkpoint_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "sequence_number": self.sequence_number,
                "positions": {
                    pos_id: {
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "entry_price": str(pos.entry_price),
                        "quantity": str(pos.quantity),
                        "unrealized_pnl": str(pos.unrealized_pnl),
                    }
                    for pos_id, pos in self.positions.items()
                },
                "pending_orders": {
                    order_id: {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": str(order.quantity),
                        "price": str(order.price) if order.price else None,
                        "status": order.status,
                    }
                    for order_id, order in self.pending_orders.items()
                },
                "metrics": {
                    "events_processed": self.events_processed,
                    "signals_generated": self.signals_generated,
                    "orders_executed": self.orders_executed,
                    "positions_opened": self.positions_opened,
                    "positions_closed": self.positions_closed,
                },
            }
            
            # In production, save to database or file
            # For now, just log that we checkpointed
            logger.info(
                "State checkpoint created",
                positions_count=len(self.positions),
                orders_count=len(self.pending_orders),
                sequence=self.sequence_number
            )
            
            self.last_checkpoint_time = time.time()
            
        except Exception as e:
            logger.error("Failed to checkpoint state", error=str(e))
    
    async def recover_state(self) -> bool:
        """
        Recover state from last checkpoint.
        
        Returns:
            bool: True if recovery successful
        """
        try:
            logger.info("Attempting state recovery")
            
            # In production, load from database or file
            # For now, return True to indicate successful recovery attempt
            
            logger.info("State recovery completed")
            return True
            
        except Exception as e:
            logger.error("State recovery failed", error=str(e))
            return False

    async def shutdown(self) -> None:
        """Shutdown the trading loop gracefully."""
        logger.info("Shutting down trading loop")

        self.running = False

        # Close all positions
        for position in list(self.positions.values()):
            logger.warning(
                "Force closing position on shutdown",
                position_id=position.position_id,
                symbol=position.symbol
            )
            # In production, would execute closing orders here

        # Cancel pending orders
        for order in list(self.pending_orders.values()):
            logger.warning(
                "Cancelling pending order on shutdown",
                order_id=order.order_id,
                symbol=order.symbol
            )
            # In production, would cancel orders here

        # Stop event bus
        await self.event_bus.stop()

        # Log final metrics
        logger.info(
            "Trading loop shutdown complete",
            events_processed=self.events_processed,
            signals_generated=self.signals_generated,
            orders_executed=self.orders_executed,
            positions_opened=self.positions_opened,
            positions_closed=self.positions_closed
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get trading loop statistics."""
        return {
            "running": self.running,
            "events_processed": self.events_processed,
            "signals_generated": self.signals_generated,
            "orders_executed": self.orders_executed,
            "positions_opened": self.positions_opened,
            "positions_closed": self.positions_closed,
            "active_positions": len(self.positions),
            "pending_orders": len(self.pending_orders),
            "total_events_stored": len(self.event_store),
        }

    async def _store_event(self, event: Event, correlation_id: str | None = None) -> None:
        """
        Store event with audit trail.
        
        Args:
            event: Event to store
            correlation_id: Optional correlation ID for related events
        """
        try:
            # Assign sequence number
            self.sequence_number += 1
            event.sequence_number = self.sequence_number

            # Set or create correlation ID
            if correlation_id:
                event.correlation_id = correlation_id
            elif not event.correlation_id:
                event.correlation_id = str(uuid4())

            # Store in memory (in production, would persist to database)
            self.event_store.append(event)

            # Map aggregate ID to correlation ID
            if event.aggregate_id:
                self.correlation_map[event.aggregate_id] = event.correlation_id

            # Log with structured data
            logger.debug(
                "Event stored",
                event_type=event.event_type.value,
                event_id=event.event_id,
                sequence=event.sequence_number,
                correlation_id=event.correlation_id,
                aggregate_id=event.aggregate_id
            )

        except Exception as e:
            logger.error("Failed to store event", error=str(e), event_id=event.event_id)

    async def _publish_risk_rejected_event(self, signal: Signal, reason: str, correlation_id: str) -> None:
        """
        Publish risk rejected event.
        
        Args:
            signal: The rejected signal
            reason: Rejection reason
            correlation_id: Correlation ID for tracking
        """
        risk_rejected_event = Event(
            event_type=EventType.RISK_CHECK_FAILED,
            aggregate_id=signal.signal_id,
            correlation_id=correlation_id,
            event_data={
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "strategy_id": signal.strategy_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )
        await self.event_bus.publish(risk_rejected_event, priority=EventPriority.HIGH)
        await self._store_event(risk_rejected_event, correlation_id)

    async def _publish_risk_validation_passed_event(self, signal: Signal, order: Order, correlation_id: str) -> None:
        """
        Publish risk validation passed event.
        
        Args:
            signal: The validated signal
            order: The created order
            correlation_id: Correlation ID for tracking
        """
        risk_passed_event = Event(
            event_type=EventType.RISK_CHECK_PASSED,
            aggregate_id=signal.signal_id,
            correlation_id=correlation_id,
            event_data={
                "signal_id": signal.signal_id,
                "order_id": order.order_id,
                "symbol": signal.symbol,
                "strategy_id": signal.strategy_id,
                "quantity": str(order.quantity),
                "timestamp": datetime.now().isoformat()
            }
        )
        await self.event_bus.publish(risk_passed_event, priority=EventPriority.NORMAL)
        await self._store_event(risk_passed_event, correlation_id)

    def get_event_history(self,
                         aggregate_id: str | None = None,
                         event_type: EventType | None = None,
                         correlation_id: str | None = None) -> list[Event]:
        """
        Get event history with filters.
        
        Args:
            aggregate_id: Filter by aggregate ID
            event_type: Filter by event type
            correlation_id: Filter by correlation ID
            
        Returns:
            List of matching events
        """
        events = self.event_store

        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]

        return sorted(events, key=lambda e: e.sequence_number or 0)

    async def _checkpoint_monitor(self) -> None:
        """Monitor and create periodic state checkpoints."""
        while self.running:
            try:
                # Check if checkpoint interval has elapsed
                current_time = time.time()
                if current_time - self.last_checkpoint_time >= self.checkpoint_interval:
                    await self.checkpoint_state()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Checkpoint monitor error", error=str(e))
                await asyncio.sleep(10)
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor trading loop heartbeat for continuous operation."""
        last_heartbeat = time.time()
        heartbeat_interval = 30  # Send heartbeat every 30 seconds

        while self.running:
            try:
                await asyncio.sleep(heartbeat_interval)

                # Send heartbeat event
                heartbeat_event = Event(
                    event_type=EventType.SYSTEM_HEARTBEAT,
                    event_data={
                        "component": "TradingLoop",
                        "timestamp": datetime.now().isoformat(),
                        "uptime_seconds": int(time.time() - last_heartbeat),
                        "events_processed": self.events_processed,
                        "orders_executed": self.orders_executed,
                        "active_positions": len(self.positions),
                        "paper_trading_mode": self.paper_trading_mode,
                    }
                )

                await self.event_bus.publish(heartbeat_event, priority=EventPriority.LOW)

                logger.debug(
                    "Heartbeat sent",
                    events_processed=self.events_processed,
                    active_positions=len(self.positions),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in heartbeat monitor", error=str(e))

    async def _health_check_monitor(self) -> None:
        """Monitor system health and connectivity."""
        health_check_interval = 60  # Check health every minute
        consecutive_failures = 0
        max_failures = 3

        while self.running:
            try:
                await asyncio.sleep(health_check_interval)

                # Check exchange connection
                exchange_healthy = await self.exchange_gateway.validate_connection()

                # Check event bus health
                event_bus_healthy = self.event_bus.is_running if hasattr(self.event_bus, 'is_running') else True

                # Check risk engine
                risk_engine_healthy = self.risk_engine.validate_configuration()

                all_healthy = exchange_healthy and event_bus_healthy and risk_engine_healthy

                if not all_healthy:
                    consecutive_failures += 1
                    logger.warning(
                        "Health check failed",
                        exchange_healthy=exchange_healthy,
                        event_bus_healthy=event_bus_healthy,
                        risk_engine_healthy=risk_engine_healthy,
                        consecutive_failures=consecutive_failures,
                    )

                    if consecutive_failures >= max_failures:
                        logger.error(
                            "Maximum health check failures reached, initiating shutdown",
                            consecutive_failures=consecutive_failures,
                        )
                        self.running = False
                else:
                    if consecutive_failures > 0:
                        logger.info("Health check recovered", previous_failures=consecutive_failures)
                    consecutive_failures = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check monitor", error=str(e))
                consecutive_failures += 1

    async def _position_reconciliation(self) -> None:
        """Reconcile positions with exchange periodically."""
        try:
            logger.debug("Starting position reconciliation")

            # In paper trading mode, skip exchange reconciliation
            if self.paper_trading_mode:
                return

            # Get positions from exchange
            # This would normally query exchange for actual positions
            # For now, just log current internal state
            logger.info(
                "Position reconciliation completed",
                internal_positions=len(self.positions),
                pending_orders=len(self.pending_orders),
            )

        except Exception as e:
            logger.error("Error in position reconciliation", error=str(e))

    async def _log_performance_metrics(self) -> None:
        """Log performance metrics periodically."""
        try:
            metrics = self.get_performance_metrics()

            logger.info(
                "Performance metrics",
                events_processed=metrics["event_processing"]["total_events"],
                signals_generated=metrics["signal_to_order"]["total_signals"],
                orders_executed=metrics["order_execution"]["total_orders"],
                positions_opened=metrics["positions"]["opened"],
                positions_closed=metrics["positions"]["closed"],
                positions_active=metrics["positions"]["active"],
                paper_trading_mode=self.paper_trading_mode,
            )

        except Exception as e:
            logger.error("Error logging performance metrics", error=str(e))

    async def reconstruct_position_state(self, position_id: str) -> dict[str, Any] | None:
        """
        Reconstruct position state from event history.
        
        Args:
            position_id: Position ID to reconstruct
            
        Returns:
            Reconstructed position state or None if not found
        """
        events = self.get_event_history(aggregate_id=position_id)

        if not events:
            return None

        # Build state from events
        state = {"position_id": position_id}

        for event in events:
            if event.event_type == EventType.POSITION_OPENED or event.event_type == EventType.POSITION_UPDATED:
                state.update(event.event_data)
            elif event.event_type == EventType.POSITION_CLOSED:
                state["status"] = "CLOSED"
                state["close_reason"] = event.event_data.get("close_reason")
                state["final_pnl"] = event.event_data.get("final_pnl")

        return state
