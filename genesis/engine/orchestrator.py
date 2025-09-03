"""
Trading engine orchestration for coordinating all components.

Manages the flow of data between market feeds, strategies, signals, and execution.
Ensures proper sequencing and coordination of trading operations.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, AsyncIterator
from uuid import uuid4
from collections import defaultdict, deque

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.exceptions import RiskLimitExceeded
from genesis.core.models import Order, Position, Signal
from genesis.data.market_data_service import MarketDataService
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.signal_queue import SignalQueue
from genesis.engine.state_machine import TierStateMachine
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.strategy_registry import StrategyRegistry
from genesis.exchange.gateway import ExchangeGateway

logger = structlog.get_logger(__name__)


@dataclass
class TraceSpan:
    """Represents a trace span for distributed tracing."""
    
    span_id: str
    trace_id: str
    operation: str
    start_time: float
    end_time: float | None = None
    parent_span_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class RateLimiter:
    """Rate limiter for strategy signal generation."""
    
    def __init__(self, max_signals_per_minute: int = 10, burst_allowance: int = 3):
        """
        Initialize rate limiter.
        
        Args:
            max_signals_per_minute: Maximum signals per minute per strategy
            burst_allowance: Allow burst of signals (up to this many in quick succession)
        """
        self.max_signals_per_minute = max_signals_per_minute
        self.burst_allowance = burst_allowance
        self.strategy_signals: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.strategy_burst_count: dict[str, int] = defaultdict(int)
        self.last_reset_time: dict[str, float] = defaultdict(float)
        
    async def check_rate_limit(self, strategy_name: str) -> tuple[bool, str]:
        """
        Check if a strategy is within rate limits.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        current_time = time.time()
        
        # Reset burst count every 10 seconds
        if current_time - self.last_reset_time[strategy_name] > 10:
            self.strategy_burst_count[strategy_name] = 0
            self.last_reset_time[strategy_name] = current_time
        
        # Check burst limit
        if self.strategy_burst_count[strategy_name] >= self.burst_allowance:
            # Check if enough time has passed
            time_since_reset = current_time - self.last_reset_time[strategy_name]
            if time_since_reset < 10:
                return False, f"Burst limit exceeded ({self.burst_allowance} signals in 10s)"
        
        # Get signals in the last minute
        minute_ago = current_time - 60
        recent_signals = [t for t in self.strategy_signals[strategy_name] if t > minute_ago]
        
        # Check rate limit
        if len(recent_signals) >= self.max_signals_per_minute:
            return False, f"Rate limit exceeded ({self.max_signals_per_minute} signals/minute)"
        
        # Record this signal
        self.strategy_signals[strategy_name].append(current_time)
        self.strategy_burst_count[strategy_name] += 1
        
        return True, ""
    
    def get_strategy_rates(self) -> dict[str, float]:
        """
        Get current signal rates for all strategies.
        
        Returns:
            Dictionary of strategy name to signals per minute
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        rates = {}
        for strategy, timestamps in self.strategy_signals.items():
            recent = [t for t in timestamps if t > minute_ago]
            rates[strategy] = len(recent)
        
        return rates


class DistributedTracer:
    """Simple distributed tracing implementation for complex flows."""
    
    def __init__(self):
        self.active_spans: dict[str, TraceSpan] = {}
        self.completed_spans: list[TraceSpan] = []
        self.current_trace_id: str | None = None
    
    @asynccontextmanager
    async def trace(self, operation: str, parent_span_id: str | None = None, **tags) -> AsyncIterator[TraceSpan]:
        """Create a trace span for an operation."""
        span_id = str(uuid4())
        
        # Use existing trace ID or create new one
        if parent_span_id and parent_span_id in self.active_spans:
            trace_id = self.active_spans[parent_span_id].trace_id
        elif self.current_trace_id:
            trace_id = self.current_trace_id
        else:
            trace_id = str(uuid4())
            self.current_trace_id = trace_id
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            operation=operation,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            tags=tags
        )
        
        self.active_spans[span_id] = span
        
        try:
            logger.debug(
                "trace_span_started",
                span_id=span_id,
                trace_id=trace_id,
                operation=operation,
                tags=tags
            )
            yield span
        finally:
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            
            self.active_spans.pop(span_id, None)
            self.completed_spans.append(span)
            
            # Keep only last 1000 spans to prevent memory growth
            if len(self.completed_spans) > 1000:
                self.completed_spans = self.completed_spans[-1000:]
            
            logger.debug(
                "trace_span_completed",
                span_id=span_id,
                trace_id=trace_id,
                operation=operation,
                duration_ms=span.duration_ms
            )
    
    def get_trace_flow(self, trace_id: str) -> list[TraceSpan]:
        """Get all spans for a specific trace ID."""
        return [
            span for span in self.completed_spans
            if span.trace_id == trace_id
        ]
    
    def clear_traces(self) -> None:
        """Clear all completed traces."""
        self.completed_spans.clear()
        self.current_trace_id = None


@dataclass
class OrchestratorMetrics:
    """Performance metrics for orchestrator monitoring."""

    signals_processed: int = 0
    signals_rejected: int = 0
    orders_created: int = 0
    orders_executed: int = 0
    orders_failed: int = 0
    total_pnl: Decimal = Decimal("0")

    # Latency tracking (in milliseconds)
    avg_signal_latency: float = 0.0
    avg_execution_latency: float = 0.0
    max_signal_latency: float = 0.0
    max_execution_latency: float = 0.0

    # Queue metrics
    signal_queue_size: int = 0
    pending_orders: int = 0

    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))


class TradingOrchestrator:
    """
    Central orchestrator for trading operations.

    Coordinates the flow between:
    - Market data ingestion
    - Strategy signal generation
    - Risk validation
    - Order execution
    - Position management
    """

    def __init__(
        self,
        event_bus: EventBus,
        risk_engine: RiskEngine,
        exchange_gateway: ExchangeGateway,
        strategy_registry: StrategyRegistry,
        strategy_orchestrator: StrategyOrchestrator,
        signal_queue: SignalQueue,
        state_machine: TierStateMachine | None = None,
        market_data_service: MarketDataService | None = None,
    ):
        """
        Initialize the trading orchestrator.

        Args:
            event_bus: Central event bus
            risk_engine: Risk validation engine
            exchange_gateway: Exchange API gateway
            strategy_registry: Strategy registry
            strategy_orchestrator: Strategy orchestrator
            signal_queue: Signal priority queue
            state_machine: Optional tier state machine
            market_data_service: Optional market data service
        """
        self.event_bus = event_bus
        self.risk_engine = risk_engine
        self.exchange_gateway = exchange_gateway
        self.strategy_registry = strategy_registry
        self.strategy_orchestrator = strategy_orchestrator
        self.signal_queue = signal_queue
        self.state_machine = state_machine
        self.market_data_service = market_data_service

        self.running = False
        self.positions: dict[str, Position] = {}
        self.pending_orders: dict[str, Order] = {}
        self.metrics = OrchestratorMetrics()
        
        # Distributed tracing
        self.tracer = DistributedTracer()
        
        # Rate limiting for strategy signals
        self.rate_limiter = RateLimiter(max_signals_per_minute=20, burst_allowance=5)

        # Processing tasks
        self.market_data_task: asyncio.Task | None = None
        self.signal_processing_task: asyncio.Task | None = None
        self.order_monitoring_task: asyncio.Task | None = None

        # Event subscriptions
        self._setup_event_subscriptions()

        logger.info("TradingOrchestrator initialized")

    def _setup_event_subscriptions(self) -> None:
        """Set up event bus subscriptions."""
        # Market data events
        self.event_bus.subscribe(
            EventType.MARKET_DATA,
            self._handle_market_data,
            priority=EventPriority.HIGH
        )

        # Signal events
        self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED,
            self._handle_signal,
            priority=EventPriority.CRITICAL
        )

        # Order events
        self.event_bus.subscribe(
            EventType.ORDER_FILLED,
            self._handle_order_filled,
            priority=EventPriority.HIGH
        )
        self.event_bus.subscribe(
            EventType.ORDER_CANCELLED,
            self._handle_order_cancelled,
            priority=EventPriority.NORMAL
        )
        self.event_bus.subscribe(
            EventType.ORDER_REJECTED,
            self._handle_order_rejected,
            priority=EventPriority.NORMAL
        )

        # Position events
        self.event_bus.subscribe(
            EventType.POSITION_OPENED,
            self._handle_position_opened,
            priority=EventPriority.HIGH
        )
        self.event_bus.subscribe(
            EventType.POSITION_CLOSED,
            self._handle_position_closed,
            priority=EventPriority.HIGH
        )

    async def start(self) -> None:
        """Start the orchestrator and all processing tasks."""
        if self.running:
            logger.warning("Orchestrator already running")
            return

        self.running = True
        logger.info("Starting trading orchestrator")

        # Start processing tasks
        self.market_data_task = asyncio.create_task(
            self._market_data_processor()
        )
        self.signal_processing_task = asyncio.create_task(
            self._signal_processor()
        )
        self.order_monitoring_task = asyncio.create_task(
            self._order_monitor()
        )

        # Start strategy orchestrator
        await self.strategy_orchestrator.start()

        logger.info("Trading orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and clean up."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping trading orchestrator")

        # Cancel processing tasks
        tasks = [
            self.market_data_task,
            self.signal_processing_task,
            self.order_monitoring_task
        ]

        for task in tasks:
            if task and not task.done():
                task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)

        # Stop strategy orchestrator
        await self.strategy_orchestrator.stop()

        # Close all pending orders
        await self._cancel_all_pending_orders()

        logger.info("Trading orchestrator stopped")

    async def _market_data_processor(self) -> None:
        """Process incoming market data and distribute to strategies."""
        while self.running:
            try:
                if not self.market_data_service:
                    await asyncio.sleep(1)
                    continue

                # Get latest market data
                market_data = await self.market_data_service.get_latest_data()

                if market_data:
                    # Publish to event bus
                    event = Event(
                        type=EventType.MARKET_DATA,
                        data=market_data,
                        priority=EventPriority.HIGH,
                        correlation_id=str(uuid4())
                    )
                    await self.event_bus.publish(event)

                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Market data processing error", error=str(e))
                await asyncio.sleep(1)

    async def _signal_processor(self) -> None:
        """Process signals from the signal queue."""
        while self.running:
            try:
                # Get next signal from queue
                signal = await self.signal_queue.get_next()

                if not signal:
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.time()

                # Process the signal
                await self._process_signal(signal)

                # Update metrics
                latency = (time.time() - start_time) * 1000  # Convert to ms
                self._update_signal_latency(latency)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Signal processing error", error=str(e))
                await asyncio.sleep(1)

    async def _process_signal(self, signal: Signal) -> None:
        """
        Process a trading signal with distributed tracing.

        Args:
            signal: Trading signal to process
        """
        self.metrics.signals_processed += 1

        async with self.tracer.trace(
            "process_signal",
            signal_id=signal.signal_id,
            strategy=signal.strategy_name,
            symbol=signal.symbol
        ) as trace_span:
            try:
                # Check rate limits first
                async with self.tracer.trace(
                    "rate_limit_check",
                    parent_span_id=trace_span.span_id,
                    strategy=signal.strategy_name
                ):
                    allowed, reason = await self.rate_limiter.check_rate_limit(signal.strategy_name)
                    if not allowed:
                        self.metrics.signals_rejected += 1
                        logger.warning(
                            "Signal rejected - rate limit exceeded",
                            signal_id=signal.signal_id,
                            strategy=signal.strategy_name,
                            reason=reason,
                            trace_id=trace_span.trace_id
                        )
                        return
                
                # Validate signal through risk engine
                async with self.tracer.trace(
                    "risk_validation",
                    parent_span_id=trace_span.span_id,
                    signal_id=signal.signal_id
                ):
                    is_valid = await self.risk_engine.validate_signal(signal)

                if not is_valid:
                    self.metrics.signals_rejected += 1
                    logger.warning(
                        "Signal rejected by risk engine",
                        signal_id=signal.signal_id,
                        reason="risk_validation_failed",
                        trace_id=trace_span.trace_id
                    )
                    return

                # Check tier restrictions
                if self.state_machine:
                    async with self.tracer.trace(
                        "tier_validation",
                        parent_span_id=trace_span.span_id,
                        strategy=signal.strategy_name
                    ):
                        tier_allowed = await self.state_machine.can_execute_strategy(
                            signal.strategy_name
                        )
                        if not tier_allowed:
                            self.metrics.signals_rejected += 1
                            logger.warning(
                                "Signal rejected - tier restriction",
                                signal_id=signal.signal_id,
                                strategy=signal.strategy_name,
                                current_tier=self.state_machine.current_tier,
                                trace_id=trace_span.trace_id
                            )
                            return

                # Create and execute order
                async with self.tracer.trace(
                    "order_creation",
                    parent_span_id=trace_span.span_id,
                    signal_id=signal.signal_id
                ):
                    order = await self._create_order_from_signal(signal)

                if order:
                    self.metrics.orders_created += 1
                    async with self.tracer.trace(
                        "order_execution",
                        parent_span_id=trace_span.span_id,
                        order_id=order.order_id
                    ):
                        await self._execute_order(order)

            except RiskLimitExceeded as e:
                self.metrics.signals_rejected += 1
                logger.warning(
                    "Signal rejected - risk limit exceeded",
                    signal_id=signal.signal_id,
                    error=str(e),
                    trace_id=trace_span.trace_id
                )
            except Exception as e:
                logger.error(
                    "Signal processing failed",
                    signal_id=signal.signal_id,
                    error=str(e),
                    trace_id=trace_span.trace_id
                )

    async def _create_order_from_signal(self, signal: Signal) -> Order | None:
        """
        Create an order from a trading signal.

        Args:
            signal: Trading signal

        Returns:
            Created order or None if creation failed
        """
        try:
            # Calculate position size
            position_size = await self.risk_engine.calculate_position_size(
                signal.symbol,
                signal.confidence,
                signal.risk_reward_ratio
            )

            if position_size <= 0:
                logger.warning(
                    "Invalid position size calculated",
                    signal_id=signal.signal_id,
                    position_size=position_size
                )
                return None

            # Get current market price
            market_price = await self.exchange_gateway.get_market_price(
                signal.symbol
            )

            # Create order
            order = Order(
                order_id=str(uuid4()),
                client_order_id=f"genesis_{signal.signal_id}",
                symbol=signal.symbol,
                side=signal.side,
                order_type=signal.order_type,
                quantity=position_size,
                price=signal.entry_price or market_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=signal.strategy_name,
                signal_id=signal.signal_id
            )

            logger.info(
                "Order created from signal",
                order_id=order.order_id,
                signal_id=signal.signal_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity
            )

            return order

        except Exception as e:
            logger.error(
                "Order creation failed",
                signal_id=signal.signal_id,
                error=str(e)
            )
            return None

    async def _execute_order(self, order: Order, max_retries: int = 3) -> None:
        """
        Execute an order on the exchange with retry logic and exponential backoff.

        Args:
            order: Order to execute
            max_retries: Maximum number of retry attempts
        """
        retry_count = 0
        backoff_time = 0.1  # Initial backoff: 100ms
        max_backoff = 30.0  # Max backoff: 30 seconds
        
        while retry_count <= max_retries:
            try:
                start_time = time.time()

                # Add to pending orders
                if retry_count == 0:
                    self.pending_orders[order.order_id] = order

                # Execute on exchange
                result = await self.exchange_gateway.place_order(order)

                if result:
                    self.metrics.orders_executed += 1

                    # Update execution latency
                    latency = (time.time() - start_time) * 1000
                    self._update_execution_latency(latency)

                    logger.info(
                        "Order executed successfully",
                        order_id=order.order_id,
                        latency_ms=latency,
                        retry_count=retry_count
                    )

                    # Publish order executed event
                    event = Event(
                        type=EventType.ORDER_PLACED,
                        data=order,
                        priority=EventPriority.HIGH
                    )
                    await self.event_bus.publish(event)
                    return  # Success, exit retry loop

                else:
                    # Order failed but gateway returned False (not exception)
                    raise Exception("Order placement returned False")

            except Exception as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    # Max retries exceeded, fail permanently
                    self.metrics.orders_failed += 1
                    self.pending_orders.pop(order.order_id, None)
                    
                    logger.error(
                        "Order execution failed after retries",
                        order_id=order.order_id,
                        error=str(e),
                        retries=retry_count - 1
                    )
                    return
                
                # Log retry attempt
                logger.warning(
                    "Order execution failed, retrying",
                    order_id=order.order_id,
                    error=str(e),
                    retry_count=retry_count,
                    backoff_seconds=backoff_time
                )
                
                # Exponential backoff
                await asyncio.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff)

    async def _order_monitor(self) -> None:
        """Monitor pending orders for status updates."""
        while self.running:
            try:
                if not self.pending_orders:
                    await asyncio.sleep(1)
                    continue

                # Check status of pending orders
                for order_id, order in list(self.pending_orders.items()):
                    try:
                        status = await self.exchange_gateway.get_order_status(
                            order_id
                        )

                        # Update order status
                        order.status = status

                        # Remove from pending if completed
                        if status in ["FILLED", "CANCELLED", "REJECTED"]:
                            self.pending_orders.pop(order_id, None)

                            # Publish appropriate event
                            event_type = {
                                "FILLED": EventType.ORDER_FILLED,
                                "CANCELLED": EventType.ORDER_CANCELLED,
                                "REJECTED": EventType.ORDER_REJECTED
                            }.get(status)

                            if event_type:
                                event = Event(
                                    type=event_type,
                                    data=order,
                                    priority=EventPriority.HIGH
                                )
                                await self.event_bus.publish(event)

                    except Exception as e:
                        logger.error(
                            "Order status check failed",
                            order_id=order_id,
                            error=str(e)
                        )

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Order monitoring error", error=str(e))
                await asyncio.sleep(5)

    async def _cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders."""
        for order_id, order in list(self.pending_orders.items()):
            try:
                await self.exchange_gateway.cancel_order(order_id)
                logger.info("Order cancelled", order_id=order_id)
            except Exception as e:
                logger.error(
                    "Failed to cancel order",
                    order_id=order_id,
                    error=str(e)
                )

        self.pending_orders.clear()

    async def _handle_market_data(self, event: Event) -> None:
        """Handle market data events."""
        # Market data is distributed by the event bus
        # Strategies will receive it through their subscriptions
        pass

    async def _handle_signal(self, event: Event) -> None:
        """Handle signal generated events."""
        signal = event.data
        if isinstance(signal, Signal):
            # Add to signal queue for processing
            await self.signal_queue.add(signal)
            self.metrics.signal_queue_size = self.signal_queue.size()

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        order = event.data
        logger.info(
            "Order filled",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity,
            price=order.price
        )

        # Create or update position
        # This would be handled by a position manager in a complete implementation

    async def _handle_order_cancelled(self, event: Event) -> None:
        """Handle order cancelled events."""
        order = event.data
        logger.info("Order cancelled", order_id=order.order_id)

    async def _handle_order_rejected(self, event: Event) -> None:
        """Handle order rejected events."""
        order = event.data
        logger.warning("Order rejected", order_id=order.order_id)

    async def _handle_position_opened(self, event: Event) -> None:
        """Handle position opened events."""
        position = event.data
        if isinstance(position, Position):
            self.positions[position.position_id] = position
            logger.info(
                "Position opened",
                position_id=position.position_id,
                symbol=position.symbol
            )

    async def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed events."""
        position = event.data
        if isinstance(position, Position):
            self.positions.pop(position.position_id, None)
            self.metrics.total_pnl += position.realized_pnl
            logger.info(
                "Position closed",
                position_id=position.position_id,
                pnl=position.realized_pnl
            )

    def _update_signal_latency(self, latency: float) -> None:
        """Update signal processing latency metrics."""
        if self.metrics.signals_processed == 1:
            self.metrics.avg_signal_latency = latency
        else:
            # Running average
            count = self.metrics.signals_processed
            self.metrics.avg_signal_latency = (
                (self.metrics.avg_signal_latency * (count - 1) + latency) / count
            )

        self.metrics.max_signal_latency = max(
            self.metrics.max_signal_latency,
            latency
        )

    def _update_execution_latency(self, latency: float) -> None:
        """Update order execution latency metrics."""
        if self.metrics.orders_executed == 1:
            self.metrics.avg_execution_latency = latency
        else:
            # Running average
            count = self.metrics.orders_executed
            self.metrics.avg_execution_latency = (
                (self.metrics.avg_execution_latency * (count - 1) + latency) / count
            )

        self.metrics.max_execution_latency = max(
            self.metrics.max_execution_latency,
            latency
        )

    def get_metrics(self) -> OrchestratorMetrics:
        """
        Get current orchestrator metrics.

        Returns:
            Current metrics snapshot
        """
        self.metrics.signal_queue_size = self.signal_queue.size()
        self.metrics.pending_orders = len(self.pending_orders)
        self.metrics.last_update = datetime.now(UTC)
        return self.metrics

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on orchestrator components.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy" if self.running else "stopped",
            "running": self.running,
            "metrics": {
                "signals_processed": self.metrics.signals_processed,
                "orders_executed": self.metrics.orders_executed,
                "pending_orders": len(self.pending_orders),
                "active_positions": len(self.positions),
                "signal_queue_size": self.signal_queue.size(),
                "avg_signal_latency_ms": round(self.metrics.avg_signal_latency, 2),
                "avg_execution_latency_ms": round(self.metrics.avg_execution_latency, 2)
            },
            "tracing": {
                "active_spans": len(self.tracer.active_spans),
                "completed_spans": len(self.tracer.completed_spans)
            },
            "rate_limiting": {
                "strategy_rates": self.rate_limiter.get_strategy_rates(),
                "max_signals_per_minute": self.rate_limiter.max_signals_per_minute,
                "burst_allowance": self.rate_limiter.burst_allowance
            },
            "components": {
                "event_bus": "healthy" if self.event_bus else "unavailable",
                "risk_engine": "healthy" if self.risk_engine else "unavailable",
                "exchange_gateway": "healthy" if self.exchange_gateway else "unavailable",
                "strategy_registry": "healthy" if self.strategy_registry else "unavailable"
            }
        }

        # Check if processing tasks are running
        tasks_healthy = all([
            self.market_data_task and not self.market_data_task.done(),
            self.signal_processing_task and not self.signal_processing_task.done(),
            self.order_monitoring_task and not self.order_monitoring_task.done()
        ])

        if not tasks_healthy and self.running:
            health["status"] = "degraded"
            health["issues"] = ["Some processing tasks are not running"]

        return health
