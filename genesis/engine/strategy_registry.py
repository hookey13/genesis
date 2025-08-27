"""
Strategy registry and lifecycle management for multi-strategy orchestration.

Manages strategy registration, state transitions, health monitoring,
and automatic recovery for concurrent strategy execution.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus
from genesis.strategies.loader import StrategyLoader

logger = structlog.get_logger(__name__)


class StrategyState(str, Enum):
    """Strategy lifecycle states."""
    IDLE = "IDLE"  # Registered but not started
    STARTING = "STARTING"  # Initialization in progress
    RUNNING = "RUNNING"  # Active and processing
    PAUSED = "PAUSED"  # Temporarily suspended
    STOPPING = "STOPPING"  # Shutdown in progress
    STOPPED = "STOPPED"  # Terminated
    ERROR = "ERROR"  # Failed state requiring intervention
    RECOVERING = "RECOVERING"  # Attempting automatic recovery


class StrategyPriority(int, Enum):
    """Strategy execution priority levels."""
    CRITICAL = 1  # Highest priority (e.g., risk management)
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5  # Lowest priority


@dataclass
class StrategyMetadata:
    """Metadata for registered strategy."""
    strategy_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    version: str = "1.0.0"
    tier_required: str = "STRATEGIST"
    priority: StrategyPriority = StrategyPriority.NORMAL
    max_positions: int = 5
    max_capital_percent: Decimal = Decimal("20")  # Max 20% of portfolio
    min_capital_usdt: Decimal = Decimal("100")
    requires_market_data: bool = True
    compatible_symbols: list[str] = field(default_factory=list)
    conflicting_strategies: set[str] = field(default_factory=set)
    dependencies: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.max_capital_percent, Decimal):
            self.max_capital_percent = Decimal(str(self.max_capital_percent))
        if not isinstance(self.min_capital_usdt, Decimal):
            self.min_capital_usdt = Decimal(str(self.min_capital_usdt))


@dataclass
class StrategyHealth:
    """Health metrics for strategy monitoring."""
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_count: int = 0
    restart_count: int = 0
    last_error: str | None = None
    last_restart: datetime | None = None
    performance_score: float = 1.0  # 0-1, affects capital allocation
    is_healthy: bool = True

    def update_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = datetime.now(UTC)

    def record_error(self, error_msg: str) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error = error_msg
        self.is_healthy = self.error_count < 5  # Mark unhealthy after 5 errors

    def record_restart(self) -> None:
        """Record a restart event."""
        self.restart_count += 1
        self.last_restart = datetime.now(UTC)
        self.error_count = 0  # Reset error count on restart


class StrategyInstance(BaseModel):
    """Runtime instance of a strategy."""

    strategy_id: str
    account_id: str
    metadata: dict[str, Any]
    state: StrategyState = StrategyState.IDLE
    health: dict[str, Any] = Field(default_factory=dict)
    allocated_capital: Decimal = Decimal("0")
    active_positions: int = 0
    total_trades: int = 0
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    last_signal_at: datetime | None = None

    class Config:
        arbitrary_types_allowed = True


class StrategyRegistry:
    """
    Central registry for all trading strategies.

    Manages strategy lifecycle, health monitoring, and automatic recovery.
    """

    def __init__(self, event_bus: EventBus, strategy_loader: StrategyLoader):
        """
        Initialize strategy registry.

        Args:
            event_bus: Event bus for strategy events
            strategy_loader: Strategy loader for tier management
        """
        self.event_bus = event_bus
        self.strategy_loader = strategy_loader
        self._strategies: dict[str, StrategyInstance] = {}
        self._metadata: dict[str, StrategyMetadata] = {}
        self._health: dict[str, StrategyHealth] = {}
        self._strategy_tasks: dict[str, asyncio.Task] = {}
        self._monitor_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the strategy registry and health monitor."""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._health_monitor())
            logger.info("Strategy registry started")

    async def stop(self) -> None:
        """Stop all strategies and the registry."""
        self._shutdown_event.set()

        # Stop all running strategies
        for strategy_id in list(self._strategies.keys()):
            await self.unregister_strategy(strategy_id)

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Strategy registry stopped")

    async def register_strategy(
        self,
        account_id: str,
        strategy_name: str,
        metadata: StrategyMetadata,
        strategy_callable: Callable | None = None
    ) -> str:
        """
        Register a new strategy instance.

        Args:
            account_id: Account ID
            strategy_name: Strategy name
            metadata: Strategy metadata
            strategy_callable: Optional strategy execution function

        Returns:
            Strategy ID

        Raises:
            ValueError: If strategy cannot be registered
        """
        # Check if strategy is enabled for account tier
        if not self.strategy_loader.is_strategy_enabled(account_id, strategy_name):
            raise ValueError(f"Strategy {strategy_name} not enabled for account {account_id}")

        # Check for conflicts with existing strategies
        for sid, instance in self._strategies.items():
            existing_meta = self._metadata.get(sid)
            if existing_meta and strategy_name in existing_meta.conflicting_strategies:
                if instance.state in [StrategyState.RUNNING, StrategyState.STARTING]:
                    raise ValueError(
                        f"Strategy {strategy_name} conflicts with running strategy {existing_meta.name}"
                    )

        # Create instance
        strategy_id = metadata.strategy_id
        instance = StrategyInstance(
            strategy_id=strategy_id,
            account_id=account_id,
            metadata=metadata.__dict__,
            health=StrategyHealth().__dict__
        )

        # Store registration
        self._strategies[strategy_id] = instance
        self._metadata[strategy_id] = metadata
        self._health[strategy_id] = StrategyHealth()

        # Publish registration event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_REGISTERED,
            event_data={
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "account_id": account_id,
                "metadata": metadata.__dict__
            }
        ))

        logger.info(
            "Strategy registered",
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            account_id=account_id
        )

        return strategy_id

    async def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister and stop a strategy.

        Args:
            strategy_id: Strategy ID to unregister

        Returns:
            True if successfully unregistered
        """
        if strategy_id not in self._strategies:
            logger.warning("Strategy not found for unregistration", strategy_id=strategy_id)
            return False

        # Stop strategy if running
        instance = self._strategies[strategy_id]
        if instance.state in [StrategyState.RUNNING, StrategyState.STARTING]:
            await self._stop_strategy(strategy_id)

        # Cancel strategy task if exists
        if strategy_id in self._strategy_tasks:
            task = self._strategy_tasks[strategy_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._strategy_tasks[strategy_id]

        # Remove from registries
        del self._strategies[strategy_id]
        del self._metadata[strategy_id]
        del self._health[strategy_id]

        # Publish event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_UNREGISTERED,
            event_data={"strategy_id": strategy_id}
        ))

        logger.info("Strategy unregistered", strategy_id=strategy_id)
        return True

    async def start_strategy(self, strategy_id: str) -> bool:
        """
        Start a registered strategy.

        Args:
            strategy_id: Strategy ID to start

        Returns:
            True if successfully started
        """
        if strategy_id not in self._strategies:
            logger.error("Strategy not found", strategy_id=strategy_id)
            return False

        instance = self._strategies[strategy_id]

        # Check current state
        if instance.state not in [StrategyState.IDLE, StrategyState.STOPPED, StrategyState.ERROR]:
            logger.warning(
                "Cannot start strategy in current state",
                strategy_id=strategy_id,
                current_state=instance.state
            )
            return False

        # Update state
        instance.state = StrategyState.STARTING
        instance.started_at = datetime.now(UTC)

        # Create strategy task (placeholder - actual execution would go here)
        self._strategy_tasks[strategy_id] = asyncio.create_task(
            self._run_strategy(strategy_id)
        )

        # Update state to running
        instance.state = StrategyState.RUNNING

        # Publish event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_STARTED,
            event_data={
                "strategy_id": strategy_id,
                "started_at": instance.started_at.isoformat()
            }
        ))

        logger.info("Strategy started", strategy_id=strategy_id)
        return True

    async def pause_strategy(self, strategy_id: str) -> bool:
        """
        Pause a running strategy.

        Args:
            strategy_id: Strategy ID to pause

        Returns:
            True if successfully paused
        """
        if strategy_id not in self._strategies:
            return False

        instance = self._strategies[strategy_id]

        if instance.state != StrategyState.RUNNING:
            logger.warning(
                "Cannot pause strategy not in RUNNING state",
                strategy_id=strategy_id,
                current_state=instance.state
            )
            return False

        instance.state = StrategyState.PAUSED

        # Publish event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_PAUSED,
            event_data={"strategy_id": strategy_id}
        ))

        logger.info("Strategy paused", strategy_id=strategy_id)
        return True

    async def resume_strategy(self, strategy_id: str) -> bool:
        """
        Resume a paused strategy.

        Args:
            strategy_id: Strategy ID to resume

        Returns:
            True if successfully resumed
        """
        if strategy_id not in self._strategies:
            return False

        instance = self._strategies[strategy_id]

        if instance.state != StrategyState.PAUSED:
            logger.warning(
                "Cannot resume strategy not in PAUSED state",
                strategy_id=strategy_id,
                current_state=instance.state
            )
            return False

        instance.state = StrategyState.RUNNING

        # Publish event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_RESUMED,
            event_data={"strategy_id": strategy_id}
        ))

        logger.info("Strategy resumed", strategy_id=strategy_id)
        return True

    async def _stop_strategy(self, strategy_id: str) -> None:
        """
        Internal method to stop a strategy.

        Args:
            strategy_id: Strategy ID to stop
        """
        instance = self._strategies[strategy_id]
        instance.state = StrategyState.STOPPING

        # Wait for graceful shutdown (max 5 seconds)
        await asyncio.sleep(0.5)

        instance.state = StrategyState.STOPPED
        instance.stopped_at = datetime.now(UTC)

        # Publish event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_STOPPED,
            event_data={
                "strategy_id": strategy_id,
                "stopped_at": instance.stopped_at.isoformat()
            }
        ))

    def get_active_strategies(self, account_id: str | None = None) -> list[StrategyInstance]:
        """
        Get list of active strategies.

        Args:
            account_id: Optional account ID filter

        Returns:
            List of active strategy instances
        """
        strategies = []

        for instance in self._strategies.values():
            if account_id and instance.account_id != account_id:
                continue

            if instance.state in [StrategyState.RUNNING, StrategyState.STARTING]:
                strategies.append(instance)

        return strategies

    def get_strategy_state(self, strategy_id: str) -> StrategyState | None:
        """
        Get current state of a strategy.

        Args:
            strategy_id: Strategy ID

        Returns:
            Current state or None if not found
        """
        instance = self._strategies.get(strategy_id)
        return instance.state if instance else None

    def get_strategy_health(self, strategy_id: str) -> StrategyHealth | None:
        """
        Get health metrics for a strategy.

        Args:
            strategy_id: Strategy ID

        Returns:
            Health metrics or None if not found
        """
        return self._health.get(strategy_id)

    async def _run_strategy(self, strategy_id: str) -> None:
        """
        Strategy execution loop (placeholder).

        Args:
            strategy_id: Strategy ID to run
        """
        instance = self._strategies[strategy_id]
        health = self._health[strategy_id]

        try:
            while instance.state in [StrategyState.RUNNING, StrategyState.PAUSED]:
                if instance.state == StrategyState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Update heartbeat
                health.update_heartbeat()

                # Strategy execution would go here
                # This is a placeholder for the actual strategy logic
                await asyncio.sleep(5)

        except Exception as e:
            logger.error(
                "Strategy execution error",
                strategy_id=strategy_id,
                error=str(e)
            )
            health.record_error(str(e))
            instance.state = StrategyState.ERROR

    async def _health_monitor(self) -> None:
        """Monitor strategy health and perform automatic recovery."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now(UTC)

                for strategy_id, health in self._health.items():
                    instance = self._strategies.get(strategy_id)
                    if not instance:
                        continue

                    # Check heartbeat timeout (30 seconds)
                    if instance.state == StrategyState.RUNNING:
                        time_since_heartbeat = (
                            current_time - health.last_heartbeat
                        ).total_seconds()

                        if time_since_heartbeat > 30:
                            logger.warning(
                                "Strategy heartbeat timeout",
                                strategy_id=strategy_id,
                                timeout_seconds=time_since_heartbeat
                            )

                            # Attempt automatic recovery
                            if health.restart_count < 3:
                                await self._recover_strategy(strategy_id)
                            else:
                                logger.error(
                                    "Strategy recovery limit reached",
                                    strategy_id=strategy_id,
                                    restart_count=health.restart_count
                                )
                                instance.state = StrategyState.ERROR

                    # Check error state recovery (wait 60 seconds before retry)
                    elif instance.state == StrategyState.ERROR:
                        if health.last_restart:
                            time_since_restart = (
                                current_time - health.last_restart
                            ).total_seconds()

                            if time_since_restart > 60 and health.restart_count < 3:
                                await self._recover_strategy(strategy_id)

            except Exception as e:
                logger.error("Health monitor error", error=str(e))

            await asyncio.sleep(10)  # Check every 10 seconds

    async def _recover_strategy(self, strategy_id: str) -> None:
        """
        Attempt to recover a failed strategy.

        Args:
            strategy_id: Strategy ID to recover
        """
        instance = self._strategies.get(strategy_id)
        health = self._health.get(strategy_id)

        if not instance or not health:
            return

        logger.info(
            "Attempting strategy recovery",
            strategy_id=strategy_id,
            restart_count=health.restart_count
        )

        instance.state = StrategyState.RECOVERING
        health.record_restart()

        # Cancel existing task if any
        if strategy_id in self._strategy_tasks:
            task = self._strategy_tasks[strategy_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Restart strategy
        await asyncio.sleep(2)  # Brief pause before restart
        await self.start_strategy(strategy_id)

        # Publish recovery event
        await self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_RECOVERED,
            event_data={
                "strategy_id": strategy_id,
                "restart_count": health.restart_count
            }
        ))
