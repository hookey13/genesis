"""Real-time strategy performance monitoring system."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import structlog
from genesis.core.models import Order, Position, Trade, PositionSide
from genesis.monitoring.performance_attribution import PerformanceAttributor

logger = structlog.get_logger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""
    
    strategy_id: str
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    peak_pnl: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    drawdown_start: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.utcnow)
    positions: Dict[str, Position] = field(default_factory=dict)
    recent_trades: List[Trade] = field(default_factory=list)
    slippage_total: Decimal = Decimal("0")
    expected_fills: Dict[str, Decimal] = field(default_factory=dict)
    
    @property
    def win_rate(self) -> Decimal:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return Decimal("0")
        return (Decimal(self.winning_trades) / Decimal(self.total_trades)) * Decimal("100")
    
    @property
    def profit_factor(self) -> Decimal:
        """Calculate profit factor."""
        if self.gross_loss == Decimal("0"):
            return Decimal("0") if self.gross_profit == Decimal("0") else Decimal("999")
        return abs(self.gross_profit / self.gross_loss)
    
    @property
    def average_win(self) -> Decimal:
        """Calculate average winning trade."""
        if self.winning_trades == 0:
            return Decimal("0")
        return self.gross_profit / Decimal(self.winning_trades)
    
    @property
    def average_loss(self) -> Decimal:
        """Calculate average losing trade."""
        if self.losing_trades == 0:
            return Decimal("0")
        return abs(self.gross_loss / Decimal(self.losing_trades))
    
    @property
    def expectancy(self) -> Decimal:
        """Calculate expectancy per trade."""
        if self.total_trades == 0:
            return Decimal("0")
        return self.total_pnl / Decimal(self.total_trades)
    
    def update_drawdown(self) -> None:
        """Update drawdown metrics."""
        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl
            self.current_drawdown = Decimal("0")
            self.drawdown_start = None
        else:
            self.current_drawdown = self.peak_pnl - self.total_pnl
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            if self.drawdown_start is None and self.current_drawdown > Decimal("0"):
                self.drawdown_start = datetime.utcnow()


class StrategyPerformanceMonitor:
    """Monitor and track real-time strategy performance."""
    
    def __init__(self, attribution_analyzer: Optional[PerformanceAttributor] = None):
        """Initialize the strategy performance monitor.
        
        Args:
            attribution_analyzer: Performance attribution analyzer instance
        """
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.attribution_analyzer = attribution_analyzer or PerformanceAttributor()
        self.update_interval = 1.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        self.rotation_rules: Dict[str, Any] = {}
        self._price_cache: Dict[str, Decimal] = {}
        self._monitoring_active = False
        
    async def start_monitoring(self) -> None:
        """Start the real-time monitoring loop."""
        if self._monitoring_active:
            logger.warning("monitoring_already_active")
            return
            
        self._monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("monitoring_started")
        
    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("monitoring_stopped")
        
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                await self._update_all_strategies()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(self.update_interval)
                
    async def _update_all_strategies(self) -> None:
        """Update metrics for all active strategies."""
        for strategy_id in list(self.strategies.keys()):
            await self.update_strategy_metrics(strategy_id)
            
    async def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy for monitoring.
        
        Args:
            strategy_id: Unique identifier for the strategy
        """
        if strategy_id not in self.strategies:
            self.strategies[strategy_id] = StrategyMetrics(strategy_id=strategy_id)
            logger.info("strategy_registered", strategy_id=strategy_id)
            
    async def unregister_strategy(self, strategy_id: str) -> None:
        """Unregister a strategy from monitoring.
        
        Args:
            strategy_id: Strategy identifier to remove
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            logger.info("strategy_unregistered", strategy_id=strategy_id)
            
    async def update_position(self, strategy_id: str, position: Position) -> None:
        """Update position for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            position: Updated position object
        """
        if strategy_id not in self.strategies:
            await self.register_strategy(strategy_id)
            
        metrics = self.strategies[strategy_id]
        metrics.positions[position.symbol] = position
        await self._calculate_unrealized_pnl(metrics)
        
    async def record_trade(self, strategy_id: str, trade: Trade, expected_price: Optional[Decimal] = None) -> None:
        """Record a completed trade for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            trade: Completed trade object
            expected_price: Expected execution price for slippage calculation
        """
        if strategy_id not in self.strategies:
            await self.register_strategy(strategy_id)
            
        metrics = self.strategies[strategy_id]
        metrics.recent_trades.append(trade)
        metrics.total_trades += 1
        
        # Calculate P&L
        trade_pnl = trade.pnl_dollars
        metrics.realized_pnl += trade_pnl
        
        if trade_pnl > Decimal("0"):
            metrics.winning_trades += 1
            metrics.gross_profit += trade_pnl
        else:
            metrics.losing_trades += 1
            metrics.gross_loss += trade_pnl
            
        # Calculate slippage if expected price provided
        if expected_price:
            # Use exit_price as the actual executed price
            slippage = abs(trade.exit_price - expected_price) * trade.quantity
            metrics.slippage_total += slippage
            
        # Update total P&L and drawdown
        await self._update_total_pnl(metrics)
        metrics.update_drawdown()
        
        # Keep only recent trades (last 100)
        if len(metrics.recent_trades) > 100:
            metrics.recent_trades = metrics.recent_trades[-100:]
            
        metrics.last_update = datetime.utcnow()
        
    async def _calculate_unrealized_pnl(self, metrics: StrategyMetrics) -> None:
        """Calculate unrealized P&L for open positions.
        
        Args:
            metrics: Strategy metrics to update
        """
        unrealized_total = Decimal("0")
        
        for symbol, position in metrics.positions.items():
            if position.quantity != Decimal("0"):
                current_price = await self._get_current_price(symbol)
                if current_price:
                    if position.side == PositionSide.LONG:  # Long position
                        unrealized = (current_price - position.entry_price) * position.quantity
                    else:  # Short position (SHORT)
                        unrealized = (position.entry_price - current_price) * position.quantity
                    unrealized_total += unrealized
                    
        metrics.unrealized_pnl = unrealized_total
        await self._update_total_pnl(metrics)
        
    async def _update_total_pnl(self, metrics: StrategyMetrics) -> None:
        """Update total P&L for a strategy.
        
        Args:
            metrics: Strategy metrics to update
        """
        metrics.total_pnl = metrics.realized_pnl + metrics.unrealized_pnl
        
    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        # This would connect to the exchange gateway in production
        # For now, return cached price or None
        return self._price_cache.get(symbol)
        
    async def update_price(self, symbol: str, price: Decimal) -> None:
        """Update cached price for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current market price
        """
        self._price_cache[symbol] = price
        
    async def update_strategy_metrics(self, strategy_id: str) -> None:
        """Force update of strategy metrics.
        
        Args:
            strategy_id: Strategy to update
        """
        if strategy_id in self.strategies:
            metrics = self.strategies[strategy_id]
            await self._calculate_unrealized_pnl(metrics)
            metrics.update_drawdown()
            metrics.last_update = datetime.utcnow()
            
    async def get_strategy_metrics(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """Get current metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Strategy metrics or None if not found
        """
        return self.strategies.get(strategy_id)
        
    async def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies.
        
        Returns:
            Dictionary of all strategy metrics
        """
        return self.strategies.copy()
        
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all strategies.
        
        Returns:
            Summary statistics dictionary
        """
        total_pnl = Decimal("0")
        total_trades = 0
        total_winning = 0
        active_strategies = len(self.strategies)
        
        for metrics in self.strategies.values():
            total_pnl += metrics.total_pnl
            total_trades += metrics.total_trades
            total_winning += metrics.winning_trades
            
        overall_win_rate = Decimal("0")
        if total_trades > 0:
            overall_win_rate = (Decimal(total_winning) / Decimal(total_trades)) * Decimal("100")
            
        return {
            "total_pnl": total_pnl,
            "active_strategies": active_strategies,
            "total_trades": total_trades,
            "overall_win_rate": overall_win_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def estimate_strategy_capacity(self, strategy_id: str, market_depth: Dict[str, Decimal]) -> Decimal:
        """Estimate the capacity of a strategy based on market depth.
        
        Args:
            strategy_id: Strategy identifier
            market_depth: Market depth data by symbol
            
        Returns:
            Estimated capacity in base currency
        """
        if strategy_id not in self.strategies:
            return Decimal("0")
            
        metrics = self.strategies[strategy_id]
        capacity = Decimal("0")
        
        for symbol, position in metrics.positions.items():
            if symbol in market_depth:
                # Estimate based on not exceeding 10% of market depth
                symbol_capacity = market_depth[symbol] * Decimal("0.10")
                capacity += symbol_capacity
                
        return capacity
        
    async def check_rotation_criteria(self) -> List[Tuple[str, str]]:
        """Check if any strategies should be rotated based on performance.
        
        Returns:
            List of (strategy_id, reason) tuples for strategies to rotate
        """
        rotation_candidates = []
        
        for strategy_id, metrics in self.strategies.items():
            # Check various rotation criteria
            if metrics.max_drawdown > Decimal("0.20"):  # 20% drawdown
                rotation_candidates.append((strategy_id, "max_drawdown_exceeded"))
                
            if metrics.total_trades >= 20 and metrics.win_rate < Decimal("30"):
                rotation_candidates.append((strategy_id, "low_win_rate"))
                
            if metrics.profit_factor < Decimal("0.8") and metrics.total_trades >= 10:
                rotation_candidates.append((strategy_id, "poor_profit_factor"))
                
            # Check if in drawdown for too long
            if metrics.drawdown_start:
                drawdown_duration = datetime.utcnow() - metrics.drawdown_start
                if drawdown_duration > timedelta(days=7):
                    rotation_candidates.append((strategy_id, "extended_drawdown"))
                    
        return rotation_candidates
        
    async def get_attribution_analysis(self, strategy_id: str, period_hours: int = 24) -> Dict[str, Any]:
        """Get performance attribution analysis for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            period_hours: Analysis period in hours
            
        Returns:
            Attribution analysis results
        """
        if strategy_id not in self.strategies:
            return {}
            
        metrics = self.strategies[strategy_id]
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        recent_trades = [t for t in metrics.recent_trades if t.timestamp >= cutoff_time]
        
        # Convert trades to returns format for PerformanceAttributor
        if not recent_trades:
            return {}
            
        portfolio_returns = []
        timestamps = []
        for trade in recent_trades:
            if trade.pnl_dollars:
                # Calculate return as percentage using entry price as cost basis
                cost_basis = trade.entry_price * trade.quantity
                if cost_basis > 0:
                    return_pct = float(trade.pnl_dollars / cost_basis)
                    portfolio_returns.append(return_pct)
                    timestamps.append(trade.timestamp)
        
        if not portfolio_returns:
            return {}
            
        # Simple attribution without full portfolio weights
        result = {
            "strategy_id": strategy_id,
            "period_hours": period_hours,
            "trade_count": len(recent_trades),
            "total_return": float(sum(portfolio_returns) * 100),
            "average_return": float((sum(portfolio_returns) / len(portfolio_returns)) * 100) if portfolio_returns else 0,
            "timestamps": [t.isoformat() for t in timestamps]
        }
        
        return result
        
    def set_rotation_rule(self, rule_name: str, criteria: Dict[str, Any]) -> None:
        """Set a custom rotation rule for strategies.
        
        Args:
            rule_name: Name of the rotation rule
            criteria: Dictionary of criteria for rotation
        """
        self.rotation_rules[rule_name] = criteria
        logger.info("rotation_rule_set", rule_name=rule_name, criteria=criteria)