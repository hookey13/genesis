"""
Main orchestration engine for multi-strategy trading system.

Coordinates strategy lifecycle, capital allocation, correlation monitoring,
conflict resolution, and aggregate risk management.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from genesis.analytics.correlation_monitor import CorrelationMonitor
from genesis.analytics.strategy_performance import StrategyPerformanceTracker
from genesis.core.events import Event, EventType
from genesis.core.models import Order, Position, Trade
from genesis.engine.capital_allocator import CapitalAllocator, StrategyAllocation
from genesis.engine.conflict_resolver import ConflictResolver
from genesis.engine.event_bus import EventBus
from genesis.engine.market_regime_detector import MarketRegime, MarketRegimeDetector
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.strategy_registry import (
    StrategyMetadata,
    StrategyPriority,
    StrategyRegistry,
    StrategyState,
)

logger = structlog.get_logger(__name__)


class OrchestrationMode(str, Enum):
    """Orchestration operation modes."""
    NORMAL = "normal"  # Standard multi-strategy operation
    CONSERVATIVE = "conservative"  # Reduced risk, lower allocations
    AGGRESSIVE = "aggressive"  # Higher risk tolerance
    DEFENSIVE = "defensive"  # Capital preservation mode
    EMERGENCY = "emergency"  # Crisis mode, minimal operations


@dataclass
class OrchestrationConfig:
    """Configuration for strategy orchestration."""
    max_concurrent_strategies: int = 10
    min_strategy_capital: Decimal = Decimal("100")
    correlation_check_interval: int = 300  # seconds
    performance_update_interval: int = 3600  # seconds
    regime_check_interval: int = 900  # seconds
    rebalance_interval: int = 86400  # daily
    conflict_resolution_enabled: bool = True
    auto_regime_adjustment: bool = True
    emergency_stop_loss: Decimal = Decimal("0.15")  # 15% portfolio loss


class StrategySignal(BaseModel):
    """Trading signal from a strategy."""
    strategy_id: str
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    action: str  # "buy", "sell", "close"
    quantity: Decimal
    confidence: Decimal = Decimal("1.0")
    urgency: str = "normal"  # "low", "normal", "high", "critical"
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StrategyOrchestrator:
    """
    Main orchestrator for multi-strategy trading system.

    Coordinates all aspects of running multiple strategies concurrently
    while managing risk, capital allocation, and conflicts.
    """

    def __init__(
        self,
        event_bus: EventBus,
        risk_engine: RiskEngine,
        total_capital: Decimal,
        config: OrchestrationConfig | None = None
    ):
        """
        Initialize strategy orchestrator.

        Args:
            event_bus: Event bus for system communication
            risk_engine: Risk management engine
            total_capital: Total available capital
            config: Orchestration configuration
        """
        self.event_bus = event_bus
        self.risk_engine = risk_engine
        self.total_capital = Decimal(str(total_capital))
        self.config = config or OrchestrationConfig()

        # Initialize components
        self.strategy_registry = StrategyRegistry(event_bus, None)
        self.capital_allocator = CapitalAllocator(event_bus, total_capital)
        self.correlation_monitor = CorrelationMonitor(event_bus)
        self.performance_tracker = StrategyPerformanceTracker(event_bus)
        self.regime_detector = MarketRegimeDetector(event_bus)
        self.conflict_resolver = ConflictResolver()

        # State tracking
        self.mode = OrchestrationMode.NORMAL
        self.active_positions: dict[str, list[Position]] = {}  # strategy_id -> positions
        self.pending_signals: list[StrategySignal] = []
        self.strategy_allocations: dict[str, StrategyAllocation] = {}

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the orchestration engine."""
        logger.info("Starting strategy orchestrator")

        # Start components
        await self.strategy_registry.start()
        await self.correlation_monitor.start()
        await self.performance_tracker.start()
        await self.regime_detector.start()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._correlation_monitor_task()),
            asyncio.create_task(self._performance_update_task()),
            asyncio.create_task(self._regime_monitor_task()),
            asyncio.create_task(self._rebalance_task()),
            asyncio.create_task(self._signal_processor_task())
        ]

        # Subscribe to events
        await self._subscribe_to_events()

        logger.info(
            "Strategy orchestrator started",
            mode=self.mode.value,
            total_capital=float(self.total_capital)
        )

    async def stop(self) -> None:
        """Stop the orchestration engine."""
        logger.info("Stopping strategy orchestrator")

        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop components
        await self.strategy_registry.stop()
        await self.correlation_monitor.stop()
        await self.performance_tracker.stop()
        await self.regime_detector.stop()

        logger.info("Strategy orchestrator stopped")

    async def register_strategy(
        self,
        account_id: str,
        strategy_name: str,
        metadata: StrategyMetadata,
        initial_allocation: Decimal | None = None
    ) -> str:
        """
        Register a new strategy with the orchestrator.

        Args:
            account_id: Account ID
            strategy_name: Strategy name
            metadata: Strategy metadata
            initial_allocation: Initial capital allocation

        Returns:
            Strategy ID

        Raises:
            ValueError: If registration fails
        """
        # Check concurrent strategy limit
        active_strategies = self.strategy_registry.get_active_strategies(account_id)
        if len(active_strategies) >= self.config.max_concurrent_strategies:
            raise ValueError(
                f"Maximum concurrent strategies ({self.config.max_concurrent_strategies}) reached"
            )

        # Register with registry
        strategy_id = await self.strategy_registry.register_strategy(
            account_id, strategy_name, metadata
        )

        # Create allocation
        allocation = StrategyAllocation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            target_allocation=initial_allocation or self.config.min_strategy_capital,
            min_allocation=self.config.min_strategy_capital,
            max_allocation=self.total_capital * metadata.max_capital_percent / Decimal("100")
        )

        self.strategy_allocations[strategy_id] = allocation
        self.active_positions[strategy_id] = []

        # Allocate capital
        await self._allocate_capital_to_strategy(strategy_id)

        logger.info(
            "Strategy registered with orchestrator",
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            initial_allocation=float(allocation.current_allocation)
        )

        return strategy_id

    async def start_strategy(self, strategy_id: str) -> bool:
        """
        Start a registered strategy.

        Args:
            strategy_id: Strategy ID to start

        Returns:
            True if successfully started
        """
        # Check market regime compatibility
        current_regime = await self.regime_detector.detect_regime()
        metadata = self.strategy_registry._metadata.get(strategy_id)

        if metadata and not self._is_regime_compatible(metadata, current_regime):
            logger.warning(
                "Strategy not compatible with current market regime",
                strategy_id=strategy_id,
                regime=current_regime.value
            )
            if not self.config.auto_regime_adjustment:
                return False

        # Start strategy
        result = await self.strategy_registry.start_strategy(strategy_id)

        if result:
            # Initialize performance tracking
            await self.performance_tracker.initialize_strategy(strategy_id)

        return result

    async def stop_strategy(self, strategy_id: str, close_positions: bool = True) -> bool:
        """
        Stop a running strategy.

        Args:
            strategy_id: Strategy ID to stop
            close_positions: Whether to close open positions

        Returns:
            True if successfully stopped
        """
        # Close positions if requested
        if close_positions and strategy_id in self.active_positions:
            positions = self.active_positions[strategy_id]
            for position in positions:
                await self._close_position(position)

        # Pause strategy first
        await self.strategy_registry.pause_strategy(strategy_id)

        # Release capital allocation
        if strategy_id in self.strategy_allocations:
            allocation = self.strategy_allocations[strategy_id]
            self.capital_allocator.unlock_capital(
                strategy_id, allocation.locked_capital
            )

        # Unregister strategy
        result = await self.strategy_registry.unregister_strategy(strategy_id)

        # Clean up tracking
        if strategy_id in self.active_positions:
            del self.active_positions[strategy_id]
        if strategy_id in self.strategy_allocations:
            del self.strategy_allocations[strategy_id]

        return result

    async def submit_signal(self, signal: StrategySignal) -> bool:
        """
        Submit a trading signal from a strategy.

        Args:
            signal: Trading signal to process

        Returns:
            True if signal accepted for processing
        """
        # Validate strategy is active
        state = self.strategy_registry.get_strategy_state(signal.strategy_id)
        if state != StrategyState.RUNNING:
            logger.warning(
                "Signal from inactive strategy rejected",
                strategy_id=signal.strategy_id,
                state=state
            )
            return False

        # Check risk limits
        if not await self._check_signal_risk(signal):
            logger.warning(
                "Signal failed risk check",
                signal_id=signal.signal_id,
                strategy_id=signal.strategy_id
            )
            return False

        # Add to pending signals
        self.pending_signals.append(signal)

        # Process immediately if critical
        if signal.urgency == "critical":
            await self._process_signal(signal)

        return True

    async def get_portfolio_status(self) -> dict:
        """
        Get current portfolio status across all strategies.

        Returns:
            Portfolio status dictionary
        """
        active_strategies = self.strategy_registry.get_active_strategies()

        # Calculate aggregate metrics
        total_positions = sum(len(positions) for positions in self.active_positions.values())
        total_pnl = Decimal("0")
        total_locked = Decimal("0")

        for _strategy_id, positions in self.active_positions.items():
            for position in positions:
                total_pnl += position.pnl_dollars

        for allocation in self.strategy_allocations.values():
            total_locked += allocation.locked_capital

        # Get correlation summary
        correlation_summary = self.correlation_monitor.get_correlation_summary()

        # Get performance summary
        performance_summary = await self.performance_tracker.get_portfolio_summary()

        return {
            "mode": self.mode.value,
            "active_strategies": len(active_strategies),
            "total_positions": total_positions,
            "total_capital": str(self.total_capital),
            "locked_capital": str(total_locked),
            "available_capital": str(self.total_capital - total_locked),
            "unrealized_pnl": str(total_pnl),
            "correlation_status": correlation_summary,
            "performance": performance_summary,
            "current_regime": (await self.regime_detector.detect_regime()).value,
            "pending_signals": len(self.pending_signals)
        }

    async def rebalance_allocations(self) -> None:
        """Rebalance capital allocations based on performance."""
        performances = await self.performance_tracker.get_all_performances()
        await self.capital_allocator.rebalance(performances)

    async def pause_strategy(self, strategy_id: str) -> None:
        """Pause a strategy."""
        await self.strategy_registry.pause_strategy(strategy_id)

    async def resume_strategy(self, strategy_id: str) -> None:
        """Resume a strategy."""
        await self.strategy_registry.resume_strategy(strategy_id)

    async def set_mode(self, mode: OrchestrationMode) -> None:
        """
        Set orchestration mode.

        Args:
            mode: New orchestration mode
        """
        old_mode = self.mode
        self.mode = mode

        logger.info(
            "Orchestration mode changed",
            old_mode=old_mode.value,
            new_mode=mode.value
        )

        # Adjust behavior based on mode
        if mode == OrchestrationMode.EMERGENCY:
            await self._enter_emergency_mode()
        elif mode == OrchestrationMode.DEFENSIVE:
            await self._enter_defensive_mode()
        elif old_mode == OrchestrationMode.EMERGENCY:
            await self._exit_emergency_mode()

        # Publish mode change event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_SHUTDOWN if mode == OrchestrationMode.EMERGENCY
                     else EventType.SYSTEM_STARTUP,
            event_data={
                "mode": mode.value,
                "timestamp": datetime.now(UTC).isoformat()
            }
        ))

    async def _allocate_capital_to_strategy(self, strategy_id: str) -> None:
        """Allocate capital to a strategy."""
        if strategy_id not in self.strategy_allocations:
            return

        allocation = self.strategy_allocations[strategy_id]

        # Use capital allocator
        allocations = await self.capital_allocator.allocate_capital([allocation])

        if strategy_id in allocations:
            allocation.current_allocation = allocations[strategy_id]
            allocation.available_capital = allocations[strategy_id]

    async def _process_signal(self, signal: StrategySignal) -> None:
        """Process a trading signal."""
        try:
            # Check for conflicts
            if self.config.conflict_resolution_enabled:
                conflicts = await self._check_signal_conflicts(signal)
                if conflicts:
                    resolved_signal = await self.conflict_resolver.resolve(
                        signal, conflicts
                    )
                    if not resolved_signal:
                        logger.warning(
                            "Signal rejected due to conflicts",
                            signal_id=signal.signal_id
                        )
                        return
                    signal = resolved_signal

            # Check capital availability
            allocation = self.strategy_allocations.get(signal.strategy_id)
            if not allocation:
                return

            required_capital = await self._calculate_required_capital(signal)
            if required_capital > allocation.available_capital:
                logger.warning(
                    "Insufficient capital for signal",
                    signal_id=signal.signal_id,
                    required=float(required_capital),
                    available=float(allocation.available_capital)
                )
                return

            # Lock capital
            self.capital_allocator.lock_capital(signal.strategy_id, required_capital)

            # Create and submit order
            order = await self._create_order_from_signal(signal)

            # Track position
            if signal.action in ["buy", "sell"]:
                position = await self._create_position_from_order(order)
                self.active_positions[signal.strategy_id].append(position)

            # Update performance
            await self.performance_tracker.record_trade(
                signal.strategy_id,
                Trade(
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    side=order.side,
                    entry_price=order.price or Decimal("0"),
                    exit_price=Decimal("0"),
                    quantity=order.quantity,
                    pnl_dollars=Decimal("0"),
                    pnl_percent=Decimal("0")
                )
            )

        except Exception as e:
            logger.error(
                "Failed to process signal",
                signal_id=signal.signal_id,
                error=str(e)
            )

    async def _check_signal_risk(self, signal: StrategySignal) -> bool:
        """Check if signal passes risk checks."""
        # Check portfolio-level risk
        portfolio_risk = await self.risk_engine.calculate_portfolio_risk()

        if self.mode == OrchestrationMode.DEFENSIVE:
            if portfolio_risk > 0.5:  # 50% of risk limit
                return False

        # Check correlation impact
        if signal.strategy_id in self.active_positions:
            positions = self.active_positions[signal.strategy_id]
            correlations = await self.correlation_monitor.calculate_position_correlations(
                [{"position_id": p.position_id, "symbol": p.symbol} for p in positions]
            )

            high_correlations = [c for c in correlations if c.correlation_coefficient > Decimal("0.8")]
            if high_correlations:
                logger.warning(
                    "High correlation detected for signal",
                    signal_id=signal.signal_id,
                    num_high_correlations=len(high_correlations)
                )
                if self.mode in [OrchestrationMode.CONSERVATIVE, OrchestrationMode.DEFENSIVE]:
                    return False

        return True

    async def _check_signal_conflicts(self, signal: StrategySignal) -> list[StrategySignal]:
        """Check for conflicting signals."""
        conflicts = []

        for pending in self.pending_signals:
            if pending.signal_id == signal.signal_id:
                continue

            # Check for opposite signals on same symbol
            if pending.symbol == signal.symbol:
                if (signal.action == "buy" and pending.action == "sell") or \
                   (signal.action == "sell" and pending.action == "buy"):
                    conflicts.append(pending)

        return conflicts

    async def _calculate_required_capital(self, signal: StrategySignal) -> Decimal:
        """Calculate capital required for a signal."""
        # Simplified calculation - would need market data in production
        return signal.quantity * Decimal("100")  # Placeholder

    async def _create_order_from_signal(self, signal: StrategySignal) -> Order:
        """Create an order from a signal."""
        from genesis.core.models import OrderSide, OrderType

        return Order(
            symbol=signal.symbol,
            type=OrderType.MARKET,
            side=OrderSide.BUY if signal.action == "buy" else OrderSide.SELL,
            quantity=signal.quantity
        )

    async def _create_position_from_order(self, order: Order) -> Position:
        """Create a position from an order."""
        from genesis.core.models import PositionSide

        return Position(
            position_id=str(uuid4()),
            account_id="default",
            symbol=order.symbol,
            side=PositionSide.LONG if order.side.value == "BUY" else PositionSide.SHORT,
            entry_price=order.price or Decimal("0"),
            quantity=order.quantity,
            dollar_value=order.quantity * (order.price or Decimal("0"))
        )

    async def _calculate_portfolio_metrics(self) -> dict:
        """Calculate portfolio metrics."""
        total_value = Decimal("0")
        total_pnl = Decimal("0")

        for positions in self.active_positions.values():
            for position in positions:
                total_value += position.dollar_value
                total_pnl += position.pnl_dollars

        return {
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.total_capital * 100) if self.total_capital else Decimal("0")
        }

    async def _close_all_positions(self) -> None:
        """Close all positions."""
        for positions in self.active_positions.values():
            for position in positions:
                await self._close_position(position)

    async def _close_position(self, position: Position) -> None:
        """Close a position."""
        # Create closing order
        from genesis.core.models import OrderSide, OrderType

        Order(
            symbol=position.symbol,
            type=OrderType.MARKET,
            side=OrderSide.SELL if position.side.value == "LONG" else OrderSide.BUY,
            quantity=position.quantity
        )

        # Process closing order
        # ... implementation ...

    def _is_regime_compatible(self, metadata: StrategyMetadata, regime: MarketRegime) -> bool:
        """Check if strategy is compatible with market regime."""
        # Simplified compatibility check
        if regime == MarketRegime.CRASH:
            return metadata.priority == StrategyPriority.CRITICAL
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return "volatility" in metadata.name.lower()
        return True

    async def emergency_stop(self, reason: str = "Emergency stop triggered") -> None:
        """Emergency stop all trading operations."""
        logger.critical(f"Emergency stop: {reason}")
        self.mode = OrchestrationMode.EMERGENCY

        # Stop all strategies
        for strategy_id in await self.strategy_registry.get_active_strategies():
            await self.strategy_registry.stop_strategy(strategy_id)

        # Close all positions
        await self._close_all_positions()

        # Set shutdown flag
        self._shutdown_event.set()

    async def _enter_emergency_mode(self) -> None:
        """Enter emergency mode."""
        logger.warning("Entering emergency mode")

        # Pause all non-critical strategies
        for strategy_id in list(self.active_positions.keys()):
            metadata = self.strategy_registry._metadata.get(strategy_id)
            if metadata and metadata.priority != StrategyPriority.CRITICAL:
                await self.strategy_registry.pause_strategy(strategy_id)

        # Close risky positions
        for strategy_id, positions in self.active_positions.items():
            for position in positions:
                if position.pnl_percent < -Decimal("5"):  # 5% loss
                    await self._close_position(position)

    async def _enter_defensive_mode(self) -> None:
        """Enter defensive mode."""
        logger.info("Entering defensive mode")

        # Reduce allocations
        for allocation in self.strategy_allocations.values():
            allocation.target_allocation *= Decimal("0.5")  # Halve allocations

        # Trigger rebalance
        await self.capital_allocator.rebalance(force=True)

    async def _exit_emergency_mode(self) -> None:
        """Exit emergency mode."""
        logger.info("Exiting emergency mode")

        # Resume paused strategies
        for strategy_id in self.strategy_allocations:
            state = self.strategy_registry.get_strategy_state(strategy_id)
            if state == StrategyState.PAUSED:
                await self.strategy_registry.resume_strategy(strategy_id)

    async def _correlation_monitor_task(self) -> None:
        """Background task for correlation monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Get all positions
                all_positions = []
                for positions in self.active_positions.values():
                    all_positions.extend(positions)

                if all_positions:
                    # Check correlations
                    alerts = await self.correlation_monitor.check_correlation_thresholds(
                        all_positions
                    )

                    # Handle alerts
                    for alert in alerts:
                        if alert.severity == "critical":
                            # Consider entering defensive mode
                            if self.mode == OrchestrationMode.NORMAL:
                                await self.set_mode(OrchestrationMode.DEFENSIVE)

            except Exception as e:
                logger.error(f"Correlation monitor task error: {e}")

            await asyncio.sleep(self.config.correlation_check_interval)

    async def _performance_update_task(self) -> None:
        """Background task for performance updates."""
        while not self._shutdown_event.is_set():
            try:
                # Update performance scores
                for strategy_id in self.strategy_allocations:
                    performance = await self.performance_tracker.get_strategy_performance(
                        strategy_id
                    )

                    if performance:
                        # Update capital allocator
                        self.capital_allocator.update_strategy_performance(
                            strategy_id,
                            performance.get("sharpe_ratio", Decimal("1.0")),
                            performance.get("volatility", Decimal("1.0"))
                        )

            except Exception as e:
                logger.error(f"Performance update task error: {e}")

            await asyncio.sleep(self.config.performance_update_interval)

    async def _regime_monitor_task(self) -> None:
        """Background task for market regime monitoring."""
        while not self._shutdown_event.is_set():
            try:
                # Detect current regime
                regime = await self.regime_detector.detect_regime()

                # Adjust strategies based on regime
                if self.config.auto_regime_adjustment:
                    await self._adjust_for_regime(regime)

            except Exception as e:
                logger.error(f"Regime monitor task error: {e}")

            await asyncio.sleep(self.config.regime_check_interval)

    async def _rebalance_task(self) -> None:
        """Background task for portfolio rebalancing."""
        while not self._shutdown_event.is_set():
            try:
                # Trigger rebalance
                await self.capital_allocator.rebalance()

                # Update strategy allocations
                for strategy_id, allocation in self.strategy_allocations.items():
                    new_capital = self.capital_allocator.get_available_capital(strategy_id)
                    allocation.available_capital = new_capital

            except Exception as e:
                logger.error(f"Rebalance task error: {e}")

            await asyncio.sleep(self.config.rebalance_interval)

    async def _signal_processor_task(self) -> None:
        """Background task for processing signals."""
        while not self._shutdown_event.is_set():
            try:
                # Process pending signals
                while self.pending_signals:
                    signal = self.pending_signals.pop(0)
                    await self._process_signal(signal)

            except Exception as e:
                logger.error(f"Signal processor task error: {e}")

            await asyncio.sleep(1)  # Process signals every second

    async def _adjust_for_regime(self, regime: MarketRegime) -> None:
        """Adjust strategies for market regime."""
        # Map regimes to modes
        if regime == MarketRegime.CRASH:
            if self.mode != OrchestrationMode.EMERGENCY:
                await self.set_mode(OrchestrationMode.EMERGENCY)
        elif regime == MarketRegime.HIGH_VOLATILITY:
            if self.mode == OrchestrationMode.NORMAL:
                await self.set_mode(OrchestrationMode.CONSERVATIVE)
        elif regime == MarketRegime.NORMAL:
            if self.mode != OrchestrationMode.NORMAL:
                await self.set_mode(OrchestrationMode.NORMAL)

    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Subscribe to risk events (subscribe is synchronous)
        self.event_bus.subscribe(
            EventType.RISK_LIMIT_BREACH,
            self._handle_risk_event
        )

        # Subscribe to market events
        self.event_bus.subscribe(
            EventType.MARKET_STATE_CHANGE,
            self._handle_market_event
        )

    async def _handle_risk_event(self, event: Event) -> None:
        """Handle risk-related events."""
        if event.event_type == EventType.RISK_LIMIT_BREACH:
            # Enter defensive mode on risk breach
            if self.mode == OrchestrationMode.NORMAL:
                await self.set_mode(OrchestrationMode.DEFENSIVE)

    async def _handle_market_event(self, event: Event) -> None:
        """Handle market-related events."""
        if event.event_type == EventType.MARKET_STATE_CHANGE:
            # Regime change handled by background task
            pass

    async def _handle_correlation_event(self, event: Event) -> None:
        """Handle correlation-related events."""
        if event.event_type == EventType.CORRELATION_ALERT:
            correlation = event.data.get("correlation", Decimal("0"))
            if correlation > Decimal("0.8"):
                # High correlation detected
                await self.set_mode(OrchestrationMode.DEFENSIVE)

    async def _handle_strategy_signal(self, event: Event) -> None:
        """Handle strategy signal events."""
        if event.event_type == EventType.STRATEGY_SIGNAL:
            signal = event.data.get("signal")
            if signal:
                await self.submit_signal(signal)
