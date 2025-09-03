"""Base strategy class for all trading strategies."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import structlog

from genesis.core.events import Event
from genesis.core.models import Order, Position, Signal

logger = structlog.get_logger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""

    strategy_id: UUID = field(default_factory=uuid4)
    name: str = ""
    symbol: str = "BTCUSDT"
    max_position_usdt: Decimal = Decimal("1000")
    position_multiplier: Decimal = Decimal("1.0")
    risk_limit: Decimal = Decimal("0.02")  # 2% risk per trade
    enabled: bool = True
    tier_required: str = "SNIPER"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyState:
    """Runtime state for a strategy."""

    status: str = "IDLE"  # IDLE, RUNNING, PAUSED, STOPPED
    positions: list[Position] = field(default_factory=list)
    pending_orders: list[Order] = field(default_factory=list)
    last_signal: Signal | None = None
    pnl_usdt: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    trades_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: StrategyConfig | None = None):
        """Initialize strategy with configuration."""
        self.config = config or StrategyConfig()
        self.state = StrategyState()
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def strategy_id(self) -> UUID:
        """Get strategy ID."""
        return self.config.strategy_id

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.config.name or self.__class__.__name__

    @abstractmethod
    async def generate_signals(self) -> list[Signal]:
        """Generate trading signals based on strategy logic.

        Returns:
            List of signals to be processed by the executor.
        """
        pass

    @abstractmethod
    async def analyze(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze market data and generate trading signal.

        Args:
            market_data: Market data from analyzer.

        Returns:
            Trading signal or None if no opportunity.
        """
        pass

    @abstractmethod
    async def manage_positions(self) -> list[Signal]:
        """Manage existing positions and generate exit signals.

        Returns:
            List of exit signals for position management.
        """
        pass

    @abstractmethod
    async def on_order_filled(self, order: Order) -> None:
        """Handle order fill event.

        Args:
            order: The filled order.
        """
        pass

    @abstractmethod
    async def on_position_closed(self, position: Position) -> None:
        """Handle position close event.

        Args:
            position: The closed position.
        """
        pass

    async def start(self) -> None:
        """Start the strategy."""
        if self._running:
            logger.warning(f"Strategy {self.name} already running")
            return

        self._running = True
        self.state.status = "RUNNING"
        self._task = asyncio.create_task(self._run())
        logger.info(f"Strategy {self.name} started")

    async def stop(self) -> None:
        """Stop the strategy."""
        if not self._running:
            logger.warning(f"Strategy {self.name} not running")
            return

        self._running = False
        self.state.status = "STOPPED"

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(f"Strategy {self.name} stopped")

    async def pause(self) -> None:
        """Pause the strategy."""
        if self.state.status != "RUNNING":
            logger.warning(f"Strategy {self.name} not running, cannot pause")
            return

        self.state.status = "PAUSED"
        logger.info(f"Strategy {self.name} paused")

    async def resume(self) -> None:
        """Resume the strategy."""
        if self.state.status != "PAUSED":
            logger.warning(f"Strategy {self.name} not paused, cannot resume")
            return

        self.state.status = "RUNNING"
        logger.info(f"Strategy {self.name} resumed")

    async def _run(self) -> None:
        """Main strategy loop."""
        try:
            while self._running:
                if self.state.status == "RUNNING":
                    try:
                        # Generate signals
                        signals = await self.generate_signals()

                        # Process signals
                        for signal in signals:
                            if self._should_process_signal(signal):
                                await self._process_signal(signal)

                    except Exception as e:
                        logger.error(f"Error in strategy {self.name}: {e}")

                # Sleep to prevent busy waiting
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Strategy {self.name} task cancelled")
            raise

    def _should_process_signal(self, signal: Signal) -> bool:
        """Check if signal should be processed.

        Args:
            signal: The signal to check.

        Returns:
            True if signal should be processed.
        """
        # Check if strategy is enabled
        if not self.config.enabled:
            return False

        # Check if we're within risk limits
        current_exposure = sum(p.dollar_value for p in self.state.positions)
        if current_exposure >= self.config.max_position_usdt:
            logger.warning(f"Strategy {self.name} at max position size")
            return False

        return True

    async def _process_signal(self, signal: Signal) -> None:
        """Process a trading signal.

        Args:
            signal: The signal to process.
        """
        self.state.last_signal = signal
        self.state.last_update = datetime.now(UTC)

        # Emit signal event
        event = Event(
            event_type="STRATEGY_SIGNAL",
            data={
                "strategy_id": str(self.strategy_id),
                "strategy_name": self.name,
                "signal": (
                    signal.__dict__ if hasattr(signal, "__dict__") else str(signal)
                ),
            },
        )
        await self.event_queue.put(event)

    def update_performance_metrics(self, pnl: Decimal, is_win: bool) -> None:
        """Update strategy performance metrics.

        Args:
            pnl: Profit/loss amount.
            is_win: Whether the trade was profitable.
        """
        self.state.pnl_usdt += pnl
        self.state.trades_count += 1

        if is_win:
            self.state.wins_count += 1
        else:
            self.state.losses_count += 1

        # Update win rate
        if self.state.trades_count > 0:
            self.state.win_rate = Decimal(self.state.wins_count) / Decimal(
                self.state.trades_count
            )

        # Update max drawdown
        if pnl < 0 and abs(pnl) > self.state.max_drawdown:
            self.state.max_drawdown = abs(pnl)

    def get_stats(self) -> dict[str, Any]:
        """Get strategy statistics.

        Returns:
            Dictionary of strategy statistics.
        """
        return {
            "strategy_id": str(self.strategy_id),
            "name": self.name,
            "status": self.state.status,
            "pnl_usdt": float(self.state.pnl_usdt),
            "win_rate": float(self.state.win_rate),
            "trades_count": self.state.trades_count,
            "wins_count": self.state.wins_count,
            "losses_count": self.state.losses_count,
            "max_drawdown": float(self.state.max_drawdown),
            "sharpe_ratio": float(self.state.sharpe_ratio),
            "positions_count": len(self.state.positions),
            "last_update": self.state.last_update.isoformat(),
        }

    async def save_state(self) -> dict[str, Any]:
        """Save strategy state for persistence.

        Returns:
            Dictionary containing the strategy state.
        """
        return {
            "config": {
                "strategy_id": str(self.config.strategy_id),
                "name": self.config.name,
                "symbol": self.config.symbol,
                "max_position_usdt": str(self.config.max_position_usdt),
                "position_multiplier": str(self.config.position_multiplier),
                "risk_limit": str(self.config.risk_limit),
                "enabled": self.config.enabled,
                "tier_required": self.config.tier_required,
                "metadata": self.config.metadata,
            },
            "state": {
                "status": self.state.status,
                "positions": [p.to_dict() if hasattr(p, "to_dict") else str(p) for p in self.state.positions],
                "pending_orders": [o.to_dict() if hasattr(o, "to_dict") else str(o) for o in self.state.pending_orders],
                "last_signal": self.state.last_signal.to_dict() if self.state.last_signal and hasattr(self.state.last_signal, "to_dict") else None,
                "pnl_usdt": str(self.state.pnl_usdt),
                "win_rate": str(self.state.win_rate),
                "trades_count": self.state.trades_count,
                "wins_count": self.state.wins_count,
                "losses_count": self.state.losses_count,
                "max_drawdown": str(self.state.max_drawdown),
                "sharpe_ratio": str(self.state.sharpe_ratio),
                "last_update": self.state.last_update.isoformat(),
            },
        }

    async def load_state(self, state_data: dict[str, Any]) -> None:
        """Load strategy state from persistence.

        Args:
            state_data: Dictionary containing the strategy state.
        """
        if "config" in state_data:
            config = state_data["config"]
            self.config.strategy_id = UUID(config.get("strategy_id", str(self.config.strategy_id)))
            self.config.name = config.get("name", self.config.name)
            self.config.symbol = config.get("symbol", self.config.symbol)
            self.config.max_position_usdt = Decimal(config.get("max_position_usdt", "1000"))
            self.config.position_multiplier = Decimal(config.get("position_multiplier", "1.0"))
            self.config.risk_limit = Decimal(config.get("risk_limit", "0.02"))
            self.config.enabled = config.get("enabled", True)
            self.config.tier_required = config.get("tier_required", "SNIPER")
            self.config.metadata = config.get("metadata", {})

        if "state" in state_data:
            state = state_data["state"]
            self.state.status = state.get("status", "IDLE")
            self.state.pnl_usdt = Decimal(state.get("pnl_usdt", "0"))
            self.state.win_rate = Decimal(state.get("win_rate", "0"))
            self.state.trades_count = state.get("trades_count", 0)
            self.state.wins_count = state.get("wins_count", 0)
            self.state.losses_count = state.get("losses_count", 0)
            self.state.max_drawdown = Decimal(state.get("max_drawdown", "0"))
            self.state.sharpe_ratio = Decimal(state.get("sharpe_ratio", "0"))
