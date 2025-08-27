"""
Performance tracking and analysis for individual strategies.
"""

from datetime import UTC, datetime
from decimal import Decimal

import structlog

from genesis.core.models import Trade
from genesis.engine.event_bus import EventBus
from typing import Optional

logger = structlog.get_logger(__name__)


class StrategyPerformanceTracker:
    """Track and analyze performance metrics for strategies."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.performance_data: dict[str, dict] = {}

    async def start(self) -> None:
        """Start performance tracker."""
        logger.info("Performance tracker started")

    async def stop(self) -> None:
        """Stop performance tracker."""
        logger.info("Performance tracker stopped")

    async def initialize_strategy(self, strategy_id: str) -> None:
        """Initialize tracking for a strategy."""
        self.performance_data[strategy_id] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": Decimal("0"),
            "max_drawdown": Decimal("0"),
            "sharpe_ratio": Decimal("1.0"),
            "win_rate": Decimal("0"),
            "average_win": Decimal("0"),
            "average_loss": Decimal("0"),
            "volatility": Decimal("0.1"),
            "started_at": datetime.now(UTC)
        }

    async def record_trade(self, strategy_id: str, trade: Trade) -> None:
        """Record a completed trade."""
        if strategy_id not in self.performance_data:
            await self.initialize_strategy(strategy_id)

        data = self.performance_data[strategy_id]
        data["total_trades"] += 1
        data["total_pnl"] += trade.pnl_dollars

        if trade.pnl_dollars > 0:
            data["winning_trades"] += 1
            data["average_win"] = (
                (data["average_win"] * (data["winning_trades"] - 1) + trade.pnl_dollars) /
                data["winning_trades"]
            )
        else:
            data["losing_trades"] += 1
            data["average_loss"] = (
                (data["average_loss"] * (data["losing_trades"] - 1) + abs(trade.pnl_dollars)) /
                data["losing_trades"]
            )

        # Update win rate
        if data["total_trades"] > 0:
            data["win_rate"] = Decimal(str(data["winning_trades"])) / Decimal(str(data["total_trades"]))

        # Simple Sharpe ratio approximation
        if data["volatility"] > 0:
            data["sharpe_ratio"] = data["total_pnl"] / (data["volatility"] * Decimal("100"))

    async def get_strategy_performance(self, strategy_id: str) -> Optional[dict]:
        """Get performance metrics for a strategy."""
        return self.performance_data.get(strategy_id)

    async def get_portfolio_summary(self) -> dict:
        """Get portfolio-wide performance summary."""
        total_pnl = Decimal("0")
        total_trades = 0

        for data in self.performance_data.values():
            total_pnl += data["total_pnl"]
            total_trades += data["total_trades"]

        return {
            "total_pnl": str(total_pnl),
            "total_trades": total_trades,
            "num_strategies": len(self.performance_data)
        }
