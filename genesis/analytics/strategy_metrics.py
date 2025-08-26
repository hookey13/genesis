"""
Strategy performance metrics tracking.

Tracks per-strategy performance for Kelly Criterion calculations
and optimal position sizing.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID
import logging

from genesis.core.models import Trade, Position

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Performance metrics for a trading strategy."""
    
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_win_amount: Decimal = Decimal("0")
    total_loss_amount: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_from_trade(self, trade: Trade) -> None:
        """Update metrics with a new trade."""
        self.total_trades += 1
        
        if trade.pnl_dollars > 0:
            self.winning_trades += 1
            self.total_win_amount += trade.pnl_dollars
        elif trade.pnl_dollars < 0:
            self.losing_trades += 1
            self.total_loss_amount += abs(trade.pnl_dollars)
        
        self.total_pnl += trade.pnl_dollars
        self._recalculate_derived_metrics()
        self.last_updated = datetime.now(timezone.utc)
    
    def _recalculate_derived_metrics(self) -> None:
        """Recalculate derived metrics."""
        if self.total_trades > 0:
            self.win_rate = Decimal(str(self.winning_trades / self.total_trades))
        
        if self.winning_trades > 0:
            self.average_win = self.total_win_amount / self.winning_trades
        
        if self.losing_trades > 0:
            self.average_loss = self.total_loss_amount / self.losing_trades
        
        if self.total_loss_amount > 0:
            self.profit_factor = self.total_win_amount / self.total_loss_amount


class StrategyPerformanceTracker:
    """
    Tracks performance metrics for multiple trading strategies.
    
    Provides strategy-specific edge calculations for Kelly sizing.
    """
    
    def __init__(self, cache_ttl_minutes: int = 5):
        """
        Initialize performance tracker.
        
        Args:
            cache_ttl_minutes: Cache TTL in minutes
        """
        self._metrics: Dict[str, StrategyMetrics] = {}
        self._trade_history: Dict[str, List[Trade]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._last_calculation: Dict[str, datetime] = {}
    
    def record_trade(self, strategy_id: str, trade: Trade) -> None:
        """
        Record a completed trade for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            trade: Completed trade
        """
        # Initialize if needed
        if strategy_id not in self._metrics:
            self._metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
            self._trade_history[strategy_id] = []
        
        # Update metrics
        self._metrics[strategy_id].update_from_trade(trade)
        
        # Store trade history
        self._trade_history[strategy_id].append(trade)
        
        # Limit history size (keep last 1000 trades)
        if len(self._trade_history[strategy_id]) > 1000:
            self._trade_history[strategy_id] = self._trade_history[strategy_id][-1000:]
        
        logger.debug("Recorded trade for strategy %s: PnL=%s", 
                    strategy_id, trade.pnl_dollars)
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            StrategyMetrics or None if no data
        """
        return self._metrics.get(strategy_id)
    
    def get_recent_trades(
        self,
        strategy_id: str,
        window_days: int = 30
    ) -> List[Trade]:
        """
        Get recent trades for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            window_days: Days to look back
            
        Returns:
            List of recent trades
        """
        if strategy_id not in self._trade_history:
            return []
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        
        return [
            trade for trade in self._trade_history[strategy_id]
            if trade.timestamp >= cutoff
        ]
    
    def calculate_strategy_edge(
        self,
        strategy_id: str,
        window_days: int = 30
    ) -> Dict[str, Decimal]:
        """
        Calculate trading edge for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            window_days: Days to look back
            
        Returns:
            Dict with win_rate, win_loss_ratio, sample_size
        """
        # Check cache
        last_calc = self._last_calculation.get(strategy_id)
        if last_calc:
            age = datetime.now(timezone.utc) - last_calc
            if age < self._cache_ttl:
                metrics = self._metrics.get(strategy_id)
                if metrics:
                    return {
                        "win_rate": metrics.win_rate,
                        "win_loss_ratio": metrics.average_win / metrics.average_loss 
                                        if metrics.average_loss > 0 else Decimal("0"),
                        "sample_size": metrics.total_trades
                    }
        
        # Get recent trades
        recent_trades = self.get_recent_trades(strategy_id, window_days)
        
        if not recent_trades:
            return {
                "win_rate": Decimal("0"),
                "win_loss_ratio": Decimal("0"),
                "sample_size": 0
            }
        
        # Calculate metrics
        wins = [t for t in recent_trades if t.pnl_dollars > 0]
        losses = [t for t in recent_trades if t.pnl_dollars < 0]
        
        win_rate = Decimal(str(len(wins) / len(recent_trades))) if recent_trades else Decimal("0")
        
        avg_win = sum(t.pnl_dollars for t in wins) / len(wins) if wins else Decimal("0")
        avg_loss = sum(abs(t.pnl_dollars) for t in losses) / len(losses) if losses else Decimal("1")
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal("0")
        
        # Update cache
        self._last_calculation[strategy_id] = datetime.now(timezone.utc)
        
        return {
            "win_rate": win_rate.quantize(Decimal("0.0001")),
            "win_loss_ratio": win_loss_ratio.quantize(Decimal("0.01")),
            "sample_size": len(recent_trades)
        }
    
    def calculate_drawdown(
        self,
        strategy_id: str,
        window_days: Optional[int] = None
    ) -> Decimal:
        """
        Calculate maximum drawdown for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            window_days: Days to look back (None for all history)
            
        Returns:
            Maximum drawdown as decimal (0.20 = 20%)
        """
        trades = self._trade_history.get(strategy_id, [])
        
        if not trades:
            return Decimal("0")
        
        if window_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
            trades = [t for t in trades if t.timestamp >= cutoff]
        
        # Calculate cumulative PnL
        cumulative_pnl = []
        running_total = Decimal("0")
        
        for trade in trades:
            running_total += trade.pnl_dollars
            cumulative_pnl.append(running_total)
        
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_drawdown = Decimal("0")
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            
            drawdown = (peak - pnl) / peak if peak > 0 else Decimal("0")
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown.quantize(Decimal("0.0001"))
    
    def get_winning_streak(self, strategy_id: str) -> int:
        """
        Get current winning streak for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Number of consecutive wins
        """
        trades = self._trade_history.get(strategy_id, [])
        
        if not trades:
            return 0
        
        streak = 0
        for trade in reversed(trades):
            if trade.pnl_dollars > 0:
                streak += 1
            else:
                break
        
        return streak
    
    def get_losing_streak(self, strategy_id: str) -> int:
        """
        Get current losing streak for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Number of consecutive losses
        """
        trades = self._trade_history.get(strategy_id, [])
        
        if not trades:
            return 0
        
        streak = 0
        for trade in reversed(trades):
            if trade.pnl_dollars < 0:
                streak += 1
            else:
                break
        
        return streak
    
    def get_all_strategy_ids(self) -> List[str]:
        """Get list of all tracked strategy IDs."""
        return list(self._metrics.keys())
    
    def reset_strategy_metrics(self, strategy_id: str) -> None:
        """
        Reset metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id in self._metrics:
            self._metrics[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
            self._trade_history[strategy_id] = []
            self._last_calculation.pop(strategy_id, None)
            logger.info("Reset metrics for strategy: %s", strategy_id)