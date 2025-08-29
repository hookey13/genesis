"""Real-time VWAP calculation and tracking for execution benchmarking.

This module calculates Volume-Weighted Average Price (VWAP) from trade streams
and tracks execution performance against market benchmarks.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from genesis.core.events import Event, EventType
from genesis.core.models import Side, Symbol
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Individual trade data."""

    timestamp: datetime
    price: Decimal
    volume: Decimal
    side: Side | None = None

    @property
    def value(self) -> Decimal:
        """Calculate trade value (price * volume)."""
        return self.price * self.volume


@dataclass
class VWAPMetrics:
    """VWAP calculation metrics."""

    symbol: Symbol
    timestamp: datetime
    vwap: Decimal
    total_volume: Decimal
    total_value: Decimal
    trade_count: int
    time_window_minutes: int

    def to_dict(self) -> dict:
        """Convert to dictionary for event emission."""
        return {
            "symbol": self.symbol.value,
            "timestamp": self.timestamp.isoformat(),
            "vwap": str(self.vwap),
            "total_volume": str(self.total_volume),
            "total_value": str(self.total_value),
            "trade_count": self.trade_count,
            "time_window_minutes": self.time_window_minutes,
        }


@dataclass
class ExecutionPerformance:
    """Execution performance metrics against VWAP benchmark."""

    symbol: Symbol
    execution_id: str
    start_time: datetime
    end_time: datetime | None
    executed_volume: Decimal
    executed_value: Decimal
    execution_vwap: Decimal
    market_vwap: Decimal
    slippage_bps: Decimal  # Basis points difference
    fill_rate: Decimal  # Percentage of target filled
    trades_executed: int

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "symbol": self.symbol.value,
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "executed_volume": str(self.executed_volume),
            "executed_value": str(self.executed_value),
            "execution_vwap": str(self.execution_vwap),
            "market_vwap": str(self.market_vwap),
            "slippage_bps": str(self.slippage_bps),
            "fill_rate": str(self.fill_rate),
            "trades_executed": self.trades_executed,
        }


class VWAPTracker:
    """Tracks VWAP from market data and benchmarks execution performance."""

    def __init__(self, event_bus: EventBus, window_minutes: int = 30):
        """Initialize VWAP tracker.

        Args:
            event_bus: Event bus for emitting metrics
            window_minutes: Rolling window for VWAP calculation
        """
        self.event_bus = event_bus
        self.window_minutes = window_minutes

        # Trade history by symbol
        self._trades: dict[str, deque[Trade]] = {}

        # Current VWAP metrics
        self._current_vwap: dict[str, VWAPMetrics] = {}

        # Execution tracking
        self._executions: dict[str, ExecutionPerformance] = {}

        # Performance history
        self._performance_history: list[ExecutionPerformance] = []

        # Update interval
        self._update_interval = 1.0  # seconds
        self._update_task: asyncio.Task | None = None

    async def start(self):
        """Start VWAP tracking with periodic updates."""
        if not self._update_task:
            self._update_task = asyncio.create_task(self._update_loop())
            logger.info("vwap_tracker_started", window_minutes=self.window_minutes)

    async def stop(self):
        """Stop VWAP tracking."""
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
            logger.info("vwap_tracker_stopped")

    async def _update_loop(self):
        """Periodic update loop for VWAP calculations."""
        while True:
            try:
                await asyncio.sleep(self._update_interval)

                # Update VWAP for all tracked symbols
                for symbol in list(self._trades.keys()):
                    self._clean_old_trades(symbol)
                    metrics = self._calculate_vwap(symbol)
                    if metrics:
                        await self._emit_metrics(metrics)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("vwap_update_error", error=str(e))

    def add_trade(self, symbol: Symbol, trade: Trade):
        """Add a new trade to tracking.

        Args:
            symbol: Trading symbol
            trade: Trade data
        """
        symbol_str = symbol.value

        if symbol_str not in self._trades:
            self._trades[symbol_str] = deque(maxlen=10000)  # Limit memory usage

        self._trades[symbol_str].append(trade)

        # Immediate VWAP update for this symbol
        metrics = self._calculate_vwap(symbol_str)
        if metrics:
            self._current_vwap[symbol_str] = metrics

    def _clean_old_trades(self, symbol: str):
        """Remove trades older than the time window.

        Args:
            symbol: Trading symbol string
        """
        if symbol not in self._trades:
            return

        cutoff_time = datetime.now(UTC) - timedelta(minutes=self.window_minutes)
        trades = self._trades[symbol]

        # Remove old trades from the left
        while trades and trades[0].timestamp < cutoff_time:
            trades.popleft()

    def _calculate_vwap(self, symbol: str) -> VWAPMetrics | None:
        """Calculate current VWAP for a symbol.

        Args:
            symbol: Trading symbol string

        Returns:
            VWAP metrics or None if no trades
        """
        if symbol not in self._trades or not self._trades[symbol]:
            return None

        trades = self._trades[symbol]

        # Calculate totals
        total_volume = Decimal("0")
        total_value = Decimal("0")

        for trade in trades:
            total_volume += trade.volume
            total_value += trade.value

        if total_volume == Decimal("0"):
            return None

        vwap = total_value / total_volume

        return VWAPMetrics(
            symbol=Symbol(symbol),
            timestamp=datetime.now(UTC),
            vwap=vwap,
            total_volume=total_volume,
            total_value=total_value,
            trade_count=len(trades),
            time_window_minutes=self.window_minutes,
        )

    async def _emit_metrics(self, metrics: VWAPMetrics):
        """Emit VWAP metrics via event bus.

        Args:
            metrics: VWAP metrics to emit
        """
        event = Event(
            event_type=EventType.METRICS_UPDATE,
            created_at=metrics.timestamp,
            event_data={"metric_type": "vwap", **metrics.to_dict()},
        )

        await self.event_bus.emit(event)

    def get_current_vwap(self, symbol: Symbol) -> Decimal | None:
        """Get current VWAP for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current VWAP or None if not available
        """
        symbol_str = symbol.value

        if symbol_str in self._current_vwap:
            return self._current_vwap[symbol_str].vwap

        # Try to calculate if we have trades
        metrics = self._calculate_vwap(symbol_str)
        if metrics:
            self._current_vwap[symbol_str] = metrics
            return metrics.vwap

        return None

    def start_execution_tracking(
        self, symbol: Symbol, execution_id: str, target_volume: Decimal
    ) -> ExecutionPerformance:
        """Start tracking an execution against VWAP benchmark.

        Args:
            symbol: Trading symbol
            execution_id: Unique execution identifier
            target_volume: Target volume to execute

        Returns:
            New execution performance tracker
        """
        performance = ExecutionPerformance(
            symbol=symbol,
            execution_id=execution_id,
            start_time=datetime.now(UTC),
            end_time=None,
            executed_volume=Decimal("0"),
            executed_value=Decimal("0"),
            execution_vwap=Decimal("0"),
            market_vwap=self.get_current_vwap(symbol) or Decimal("0"),
            slippage_bps=Decimal("0"),
            fill_rate=Decimal("0"),
            trades_executed=0,
        )

        self._executions[execution_id] = performance

        logger.info(
            "execution_tracking_started",
            symbol=symbol.value,
            execution_id=execution_id,
            target_volume=str(target_volume),
        )

        return performance

    def update_execution(self, execution_id: str, price: Decimal, volume: Decimal):
        """Update execution with a new fill.

        Args:
            execution_id: Execution identifier
            price: Fill price
            volume: Fill volume
        """
        if execution_id not in self._executions:
            logger.warning("execution_not_found", execution_id=execution_id)
            return

        performance = self._executions[execution_id]

        # Update execution metrics
        performance.executed_volume += volume
        performance.executed_value += price * volume
        performance.trades_executed += 1

        # Recalculate execution VWAP
        if performance.executed_volume > 0:
            performance.execution_vwap = (
                performance.executed_value / performance.executed_volume
            )

        # Update market VWAP
        current_market_vwap = self.get_current_vwap(performance.symbol)
        if current_market_vwap:
            performance.market_vwap = current_market_vwap

            # Calculate slippage in basis points
            if performance.market_vwap > 0:
                slippage_ratio = (
                    performance.execution_vwap - performance.market_vwap
                ) / performance.market_vwap
                performance.slippage_bps = slippage_ratio * Decimal("10000")

    def complete_execution(
        self, execution_id: str, target_volume: Decimal
    ) -> ExecutionPerformance | None:
        """Complete execution tracking and return final performance.

        Args:
            execution_id: Execution identifier
            target_volume: Original target volume

        Returns:
            Final execution performance or None
        """
        if execution_id not in self._executions:
            logger.warning("execution_not_found", execution_id=execution_id)
            return None

        performance = self._executions[execution_id]
        performance.end_time = datetime.now(UTC)

        # Calculate fill rate
        if target_volume > 0:
            performance.fill_rate = (
                performance.executed_volume / target_volume
            ) * Decimal("100")

        # Move to history
        self._performance_history.append(performance)
        del self._executions[execution_id]

        logger.info(
            "execution_completed",
            execution_id=execution_id,
            symbol=performance.symbol.value,
            executed_volume=str(performance.executed_volume),
            execution_vwap=str(performance.execution_vwap),
            market_vwap=str(performance.market_vwap),
            slippage_bps=str(performance.slippage_bps),
            fill_rate=str(performance.fill_rate),
        )

        return performance

    def get_performance_stats(
        self, symbol: Symbol | None = None, hours: int = 24
    ) -> dict:
        """Get aggregated performance statistics.

        Args:
            symbol: Filter by symbol (optional)
            hours: Hours of history to analyze

        Returns:
            Performance statistics dictionary
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        # Filter relevant executions
        relevant = [
            p
            for p in self._performance_history
            if p.end_time
            and p.end_time > cutoff_time
            and (symbol is None or p.symbol == symbol)
        ]

        if not relevant:
            return {
                "executions": 0,
                "avg_slippage_bps": "0",
                "avg_fill_rate": "0",
                "total_volume": "0",
            }

        # Calculate statistics
        total_slippage = sum(p.slippage_bps for p in relevant)
        total_fill_rate = sum(p.fill_rate for p in relevant)
        total_volume = sum(p.executed_volume for p in relevant)

        return {
            "executions": len(relevant),
            "avg_slippage_bps": str(total_slippage / len(relevant)),
            "avg_fill_rate": str(total_fill_rate / len(relevant)),
            "total_volume": str(total_volume),
            "best_execution": (
                min(relevant, key=lambda x: x.slippage_bps).execution_id
                if relevant
                else None
            ),
            "worst_execution": (
                max(relevant, key=lambda x: x.slippage_bps).execution_id
                if relevant
                else None
            ),
        }

    async def calculate_real_time_vwap(
        self, trades: list[Trade], time_window: timedelta | None = None
    ) -> Decimal:
        """Calculate VWAP from a list of trades.

        Args:
            trades: List of trades
            time_window: Optional time window to filter trades

        Returns:
            Calculated VWAP
        """
        if not trades:
            return Decimal("0")

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now(UTC) - time_window
            trades = [t for t in trades if t.timestamp > cutoff]

        if not trades:
            return Decimal("0")

        total_volume = sum(t.volume for t in trades)
        total_value = sum(t.value for t in trades)

        if total_volume == Decimal("0"):
            return Decimal("0")

        return total_value / total_volume
