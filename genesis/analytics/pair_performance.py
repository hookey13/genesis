from typing import Optional
"""Performance attribution system for multi-pair trading."""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

import numpy as np
import structlog

from genesis.core.models import Position
from genesis.data.repository import Repository

logger = structlog.get_logger(__name__)


class PeriodType(Enum):
    """Performance period types."""
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    CUSTOM = "CUSTOM"


@dataclass
class PairMetrics:
    """Metrics for a single trading pair."""
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl_dollars: Decimal
    average_win_dollars: Decimal
    average_loss_dollars: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    max_drawdown_dollars: Decimal
    volume_traded_base: Decimal
    volume_traded_quote: Decimal
    fees_paid_dollars: Decimal
    best_trade_pnl: Decimal
    worst_trade_pnl: Decimal
    average_hold_time_minutes: float

    @property
    def expectancy(self) -> Decimal:
        """Calculate trade expectancy."""
        if self.total_trades == 0:
            return Decimal("0")
        return self.total_pnl_dollars / Decimal(self.total_trades)

    @property
    def risk_reward_ratio(self) -> Decimal:
        """Calculate risk/reward ratio."""
        if self.average_loss_dollars == Decimal("0"):
            return Decimal("0")
        return abs(self.average_win_dollars / self.average_loss_dollars)


@dataclass
class AttributionReport:
    """Performance attribution report across pairs."""
    period_start: datetime
    period_end: datetime
    total_pnl_dollars: Decimal
    pair_contributions: dict[str, Decimal]  # symbol -> P&L contribution
    pair_weights: dict[str, Decimal]  # symbol -> % of total trading
    best_performer: Optional[str]
    worst_performer: Optional[str]
    correlation_impact: Decimal  # P&L impact from correlations
    diversification_benefit: Decimal  # Benefit from diversification
    recommendations: list[str] = field(default_factory=list)


class PairPerformanceTracker:
    """Tracks and analyzes performance attribution across trading pairs."""

    def __init__(self, repository: Repository, account_id: str):
        """Initialize performance tracker.
        
        Args:
            repository: Data repository for persistence
            account_id: Account identifier
        """
        self.repository = repository
        self.account_id = account_id
        self._metrics_cache: dict[str, PairMetrics] = {}
        self._cache_expiry = timedelta(minutes=5)
        self._cache_timestamps: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def track_trade(self, position: Position) -> None:
        """Track a completed trade for performance analysis.
        
        Args:
            position: Closed position to track
        """
        if not position.closed_at:
            logger.warning(
                "attempted_to_track_open_position",
                position_id=position.position_id,
                symbol=position.symbol
            )
            return

        async with self._lock:
            # Calculate trade metrics
            hold_time_minutes = (position.closed_at - position.opened_at).total_seconds() / 60
            is_winner = position.pnl_dollars > Decimal("0")

            # Get or create current period metrics
            period_start, period_end = self._get_current_period(PeriodType.DAILY)

            try:
                # Store trade in database
                await self.repository.save_trade_performance({
                    "trade_id": str(uuid.uuid4()),
                    "account_id": self.account_id,
                    "position_id": position.position_id,
                    "symbol": position.symbol,
                    "pnl_dollars": position.pnl_dollars,
                    "is_winner": is_winner,
                    "hold_time_minutes": hold_time_minutes,
                    "volume_base": position.quantity,
                    "volume_quote": position.dollar_value,
                    "fees_paid": position.fees_paid if hasattr(position, 'fees_paid') else Decimal("0"),
                    "closed_at": position.closed_at,
                    "period_start": period_start,
                    "period_end": period_end
                })

                # Invalidate cache for this symbol
                if position.symbol in self._metrics_cache:
                    del self._metrics_cache[position.symbol]
                    del self._cache_timestamps[position.symbol]

                logger.info(
                    "trade_tracked",
                    position_id=position.position_id,
                    symbol=position.symbol,
                    pnl=position.pnl_dollars,
                    is_winner=is_winner,
                    hold_time_minutes=hold_time_minutes
                )

            except Exception as e:
                logger.error(
                    "failed_to_track_trade",
                    position_id=position.position_id,
                    error=str(e)
                )

    async def get_pair_metrics(
        self,
        symbol: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> PairMetrics:
        """Get performance metrics for a specific pair.
        
        Args:
            symbol: Trading pair symbol
            period_start: Start of period (defaults to 30 days ago)
            period_end: End of period (defaults to now)
            
        Returns:
            PairMetrics with performance data
        """
        # Use defaults if not specified
        if not period_end:
            period_end = datetime.utcnow()
        if not period_start:
            period_start = period_end - timedelta(days=30)

        # Check cache
        cache_key = f"{symbol}_{period_start}_{period_end}"
        if cache_key in self._metrics_cache:
            if datetime.utcnow() - self._cache_timestamps[cache_key] < self._cache_expiry:
                return self._metrics_cache[cache_key]

        async with self._lock:
            # Load trades from database
            trades = await self.repository.get_trades_by_symbol(
                account_id=self.account_id,
                symbol=symbol,
                start_time=period_start,
                end_time=period_end
            )

            if not trades:
                # Return empty metrics
                return PairMetrics(
                    symbol=symbol,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    total_pnl_dollars=Decimal("0"),
                    average_win_dollars=Decimal("0"),
                    average_loss_dollars=Decimal("0"),
                    win_rate=Decimal("0"),
                    profit_factor=Decimal("0"),
                    sharpe_ratio=Decimal("0"),
                    max_drawdown_dollars=Decimal("0"),
                    volume_traded_base=Decimal("0"),
                    volume_traded_quote=Decimal("0"),
                    fees_paid_dollars=Decimal("0"),
                    best_trade_pnl=Decimal("0"),
                    worst_trade_pnl=Decimal("0"),
                    average_hold_time_minutes=0
                )

            # Calculate metrics
            metrics = self._calculate_pair_metrics(symbol, trades)

            # Cache results
            self._metrics_cache[cache_key] = metrics
            self._cache_timestamps[cache_key] = datetime.utcnow()

            return metrics

    async def generate_attribution_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> AttributionReport:
        """Generate performance attribution report across all pairs.
        
        Args:
            period_start: Start of period
            period_end: End of period
            
        Returns:
            AttributionReport with attribution analysis
        """
        if not period_end:
            period_end = datetime.utcnow()
        if not period_start:
            period_start = period_end - timedelta(days=30)

        async with self._lock:
            # Get all traded symbols in period
            symbols = await self.repository.get_traded_symbols(
                account_id=self.account_id,
                start_time=period_start,
                end_time=period_end
            )

            if not symbols:
                return AttributionReport(
                    period_start=period_start,
                    period_end=period_end,
                    total_pnl_dollars=Decimal("0"),
                    pair_contributions={},
                    pair_weights={},
                    best_performer=None,
                    worst_performer=None,
                    correlation_impact=Decimal("0"),
                    diversification_benefit=Decimal("0")
                )

            # Get metrics for each pair
            pair_metrics = {}
            for symbol in symbols:
                metrics = await self.get_pair_metrics(symbol, period_start, period_end)
                pair_metrics[symbol] = metrics

            # Calculate total P&L
            total_pnl = sum(m.total_pnl_dollars for m in pair_metrics.values())

            # Calculate contributions and weights
            pair_contributions = {
                symbol: metrics.total_pnl_dollars
                for symbol, metrics in pair_metrics.items()
            }

            total_volume = sum(m.volume_traded_quote for m in pair_metrics.values())
            pair_weights = {}
            if total_volume > Decimal("0"):
                pair_weights = {
                    symbol: metrics.volume_traded_quote / total_volume
                    for symbol, metrics in pair_metrics.items()
                }

            # Find best and worst performers
            if pair_contributions:
                best_performer = max(pair_contributions, key=pair_contributions.get)
                worst_performer = min(pair_contributions, key=pair_contributions.get)
            else:
                best_performer = worst_performer = None

            # Calculate correlation impact
            correlation_impact = await self._calculate_correlation_impact(
                pair_metrics, period_start, period_end
            )

            # Calculate diversification benefit
            diversification_benefit = self._calculate_diversification_benefit(
                pair_metrics, pair_weights
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                pair_metrics, pair_weights, correlation_impact
            )

            return AttributionReport(
                period_start=period_start,
                period_end=period_end,
                total_pnl_dollars=total_pnl,
                pair_contributions=pair_contributions,
                pair_weights=pair_weights,
                best_performer=best_performer,
                worst_performer=worst_performer,
                correlation_impact=correlation_impact,
                diversification_benefit=diversification_benefit,
                recommendations=recommendations
            )

    async def get_historical_performance(
        self,
        symbol: str,
        periods: int = 30,
        period_type: PeriodType = PeriodType.DAILY
    ) -> list[PairMetrics]:
        """Get historical performance for a pair.
        
        Args:
            symbol: Trading pair symbol
            periods: Number of periods to retrieve
            period_type: Type of period
            
        Returns:
            List of PairMetrics for each period
        """
        historical_metrics = []

        # Calculate period duration
        if period_type == PeriodType.HOURLY:
            delta = timedelta(hours=1)
        elif period_type == PeriodType.DAILY:
            delta = timedelta(days=1)
        elif period_type == PeriodType.WEEKLY:
            delta = timedelta(weeks=1)
        elif period_type == PeriodType.MONTHLY:
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)

        end_time = datetime.utcnow()

        for i in range(periods):
            period_end = end_time - (delta * i)
            period_start = period_end - delta

            metrics = await self.get_pair_metrics(symbol, period_start, period_end)
            historical_metrics.append(metrics)

        # Reverse to chronological order
        historical_metrics.reverse()

        return historical_metrics

    async def compare_pairs(
        self,
        symbols: list[str],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> dict[str, dict[str, any]]:
        """Compare performance across multiple pairs.
        
        Args:
            symbols: List of symbols to compare
            period_start: Start of comparison period
            period_end: End of comparison period
            
        Returns:
            Dictionary with comparison data
        """
        if not period_end:
            period_end = datetime.utcnow()
        if not period_start:
            period_start = period_end - timedelta(days=30)

        comparison = {}

        for symbol in symbols:
            metrics = await self.get_pair_metrics(symbol, period_start, period_end)

            comparison[symbol] = {
                "pnl": metrics.total_pnl_dollars,
                "trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown_dollars,
                "expectancy": metrics.expectancy,
                "risk_reward": metrics.risk_reward_ratio,
                "profit_factor": metrics.profit_factor
            }

        # Add rankings
        metrics_to_rank = ["pnl", "win_rate", "sharpe_ratio", "expectancy", "profit_factor"]

        for metric in metrics_to_rank:
            values = [(s, comparison[s][metric]) for s in symbols]
            values.sort(key=lambda x: x[1], reverse=True)

            for rank, (symbol, _) in enumerate(values, 1):
                comparison[symbol][f"{metric}_rank"] = rank

        return comparison

    # Private methods

    def _calculate_pair_metrics(self, symbol: str, trades: list[dict]) -> PairMetrics:
        """Calculate metrics from trade data.
        
        Args:
            symbol: Trading pair symbol
            trades: List of trade records
            
        Returns:
            Calculated PairMetrics
        """
        if not trades:
            return self._empty_metrics(symbol)

        # Separate winners and losers
        winners = [t for t in trades if t["pnl_dollars"] > 0]
        losers = [t for t in trades if t["pnl_dollars"] < 0]

        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)

        total_pnl = sum(Decimal(str(t["pnl_dollars"])) for t in trades)

        # Calculate averages
        average_win = (
            sum(Decimal(str(t["pnl_dollars"])) for t in winners) / Decimal(winning_trades)
            if winning_trades > 0 else Decimal("0")
        )

        average_loss = (
            sum(Decimal(str(t["pnl_dollars"])) for t in losers) / Decimal(losing_trades)
            if losing_trades > 0 else Decimal("0")
        )

        # Calculate win rate
        win_rate = (
            Decimal(winning_trades) / Decimal(total_trades)
            if total_trades > 0 else Decimal("0")
        )

        # Calculate profit factor
        gross_profit = sum(Decimal(str(t["pnl_dollars"])) for t in winners)
        gross_loss = abs(sum(Decimal(str(t["pnl_dollars"])) for t in losers))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > Decimal("0") else Decimal("0")
        )

        # Calculate Sharpe ratio
        returns = [float(t["pnl_dollars"]) for t in trades]
        if len(returns) > 1:
            sharpe_ratio = Decimal(str(self._calculate_sharpe_ratio(returns)))
        else:
            sharpe_ratio = Decimal("0")

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(trades)

        # Calculate volumes and fees
        volume_base = sum(Decimal(str(t.get("volume_base", 0))) for t in trades)
        volume_quote = sum(Decimal(str(t.get("volume_quote", 0))) for t in trades)
        fees_paid = sum(Decimal(str(t.get("fees_paid", 0))) for t in trades)

        # Find best and worst trades
        pnls = [Decimal(str(t["pnl_dollars"])) for t in trades]
        best_trade = max(pnls) if pnls else Decimal("0")
        worst_trade = min(pnls) if pnls else Decimal("0")

        # Calculate average hold time
        hold_times = [t.get("hold_time_minutes", 0) for t in trades]
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

        return PairMetrics(
            symbol=symbol,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl_dollars=total_pnl,
            average_win_dollars=average_win,
            average_loss_dollars=average_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_dollars=max_drawdown,
            volume_traded_base=volume_base,
            volume_traded_quote=volume_quote,
            fees_paid_dollars=fees_paid,
            best_trade_pnl=best_trade,
            worst_trade_pnl=worst_trade,
            average_hold_time_minutes=avg_hold_time
        )

    def _calculate_sharpe_ratio(self, returns: list[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0:
            return 0.0

        # Annualized Sharpe (assuming daily returns)
        return (mean_excess / std_excess) * np.sqrt(252)

    def _calculate_max_drawdown(self, trades: list[dict]) -> Decimal:
        """Calculate maximum drawdown.
        
        Args:
            trades: List of trade records
            
        Returns:
            Maximum drawdown amount
        """
        if not trades:
            return Decimal("0")

        # Calculate cumulative P&L
        cumulative_pnl = []
        running_total = Decimal("0")

        for trade in sorted(trades, key=lambda t: t.get("closed_at", datetime.min)):
            running_total += Decimal(str(trade["pnl_dollars"]))
            cumulative_pnl.append(running_total)

        # Find maximum drawdown
        peak = Decimal("0")
        max_dd = Decimal("0")

        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    async def _calculate_correlation_impact(
        self,
        pair_metrics: dict[str, PairMetrics],
        period_start: datetime,
        period_end: datetime
    ) -> Decimal:
        """Calculate P&L impact from correlations.
        
        Args:
            pair_metrics: Metrics for each pair
            period_start: Period start
            period_end: Period end
            
        Returns:
            Estimated correlation impact on P&L
        """
        # This would integrate with the correlation monitor
        # For now, return a placeholder
        return Decimal("0")

    def _calculate_diversification_benefit(
        self,
        pair_metrics: dict[str, PairMetrics],
        pair_weights: dict[str, Decimal]
    ) -> Decimal:
        """Calculate diversification benefit.
        
        Args:
            pair_metrics: Metrics for each pair
            pair_weights: Weight of each pair in portfolio
            
        Returns:
            Diversification benefit amount
        """
        if len(pair_metrics) < 2:
            return Decimal("0")

        # Calculate portfolio Sharpe vs average individual Sharpe
        individual_sharpes = [m.sharpe_ratio for m in pair_metrics.values()]
        avg_sharpe = sum(individual_sharpes) / len(individual_sharpes)

        # Portfolio Sharpe (simplified)
        portfolio_return = sum(
            m.total_pnl_dollars * pair_weights.get(s, Decimal("0"))
            for s, m in pair_metrics.items()
        )

        # Simplified benefit calculation
        if avg_sharpe > Decimal("0"):
            benefit = portfolio_return * Decimal("0.1")  # 10% benefit estimate
        else:
            benefit = Decimal("0")

        return benefit

    def _generate_recommendations(
        self,
        pair_metrics: dict[str, PairMetrics],
        pair_weights: dict[str, Decimal],
        correlation_impact: Decimal
    ) -> list[str]:
        """Generate performance recommendations.
        
        Args:
            pair_metrics: Metrics for each pair
            pair_weights: Weight of each pair
            correlation_impact: Correlation impact
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for underperforming pairs
        for symbol, metrics in pair_metrics.items():
            if metrics.total_trades > 10:
                if metrics.win_rate < Decimal("0.4"):
                    recommendations.append(
                        f"Review strategy for {symbol} - win rate below 40%"
                    )
                if metrics.sharpe_ratio < Decimal("0"):
                    recommendations.append(
                        f"Consider reducing exposure to {symbol} - negative Sharpe ratio"
                    )

        # Check concentration
        max_weight = max(pair_weights.values()) if pair_weights else Decimal("0")
        if max_weight > Decimal("0.5"):
            recommendations.append(
                "Portfolio highly concentrated - consider diversifying"
            )

        # Check correlation impact
        if abs(correlation_impact) > Decimal("100"):
            recommendations.append(
                "High correlation impact detected - review pair selection"
            )

        return recommendations[:5]  # Limit to top 5 recommendations

    def _get_current_period(self, period_type: PeriodType) -> tuple[datetime, datetime]:
        """Get current period boundaries.
        
        Args:
            period_type: Type of period
            
        Returns:
            Tuple of (start, end) datetime
        """
        now = datetime.utcnow()

        if period_type == PeriodType.HOURLY:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period_type == PeriodType.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period_type == PeriodType.WEEKLY:
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(weeks=1)
        elif period_type == PeriodType.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Get first day of next month
            if now.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        else:
            start = now
            end = now

        return start, end

    def _empty_metrics(self, symbol: str) -> PairMetrics:
        """Create empty metrics for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Empty PairMetrics
        """
        return PairMetrics(
            symbol=symbol,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl_dollars=Decimal("0"),
            average_win_dollars=Decimal("0"),
            average_loss_dollars=Decimal("0"),
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            max_drawdown_dollars=Decimal("0"),
            volume_traded_base=Decimal("0"),
            volume_traded_quote=Decimal("0"),
            fees_paid_dollars=Decimal("0"),
            best_trade_pnl=Decimal("0"),
            worst_trade_pnl=Decimal("0"),
            average_hold_time_minutes=0
        )
