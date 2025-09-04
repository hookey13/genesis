"""Performance metrics calculation for backtesting and live trading.

This module provides comprehensive performance metrics calculation including
Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, win rate,
profit factor, and other industry-standard risk-adjusted return metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DrawdownInfo:
    """Information about drawdown periods."""
    
    max_drawdown: Decimal
    max_drawdown_duration: timedelta
    current_drawdown: Decimal
    peak_value: Decimal
    trough_value: Decimal
    recovery_time: Optional[timedelta]
    drawdown_start: Optional[datetime]
    drawdown_end: Optional[datetime]


@dataclass
class TradeStatistics:
    """Detailed trade statistics."""
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float
    profit_factor: float
    avg_win: Decimal
    avg_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    avg_trade: Decimal
    avg_trade_duration: timedelta
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    expectancy: Decimal
    payoff_ratio: float


@dataclass
class RollingMetrics:
    """Rolling window performance metrics."""
    
    window_size: int
    returns: List[float]
    sharpe_ratios: List[float]
    sortino_ratios: List[float]
    win_rates: List[float]
    drawdowns: List[float]
    volatilities: List[float]


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    
    # Returns
    total_return: Decimal
    annualized_return: float
    compound_annual_growth_rate: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: Decimal
    max_drawdown_duration: timedelta
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: Optional[float] = None
    
    # Trade statistics
    trade_stats: Optional[TradeStatistics] = None
    
    # Period metrics
    best_day: Decimal = Decimal('0')
    worst_day: Decimal = Decimal('0')
    best_month: Decimal = Decimal('0')
    worst_month: Decimal = Decimal('0')
    positive_days: int = 0
    negative_days: int = 0
    
    # Benchmark comparison
    beta: Optional[float] = None
    alpha: Optional[float] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Rolling metrics
    rolling_metrics: Optional[RollingMetrics] = None
    
    # Additional metrics
    recovery_factor: Optional[float] = None
    ulcer_index: Optional[float] = None
    kelly_criterion: Optional[float] = None


class PerformanceCalculator:
    """Calculate comprehensive performance metrics for trading strategies."""
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        mar: float = 0.0,  # Minimum Acceptable Return for Sortino
        periods_per_year: int = 252,
        cache_size: int = 128
    ):
        """Initialize performance calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            mar: Minimum Acceptable Return for Sortino calculation
            periods_per_year: Trading periods per year (252 for daily)
            cache_size: Size of LRU cache for expensive calculations
        """
        self.risk_free_rate = risk_free_rate
        self.mar = mar
        self.periods_per_year = periods_per_year
        self._cache: Dict[str, Any] = {}
        
        # Configure cache size
        self._configure_cache(cache_size)
        
        logger.info(
            "Performance calculator initialized",
            risk_free_rate=risk_free_rate,
            mar=mar,
            periods_per_year=periods_per_year
        )
    
    def _configure_cache(self, cache_size: int) -> None:
        """Configure LRU cache for expensive calculations."""
        # Recreate cached methods with new cache size
        self._sharpe_ratio = lru_cache(maxsize=cache_size)(self._sharpe_ratio_impl)
        self._sortino_ratio = lru_cache(maxsize=cache_size)(self._sortino_ratio_impl)
        self._information_ratio = lru_cache(maxsize=cache_size)(self._information_ratio_impl)
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics for edge cases."""
        return PerformanceMetrics(
            total_return=Decimal('0'),
            annualized_return=0.0,
            compound_annual_growth_rate=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            max_drawdown=Decimal('0'),
            max_drawdown_duration=timedelta(0),
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0
        )
    
    def calculate_metrics(
        self,
        equity_curve: List[Decimal],
        trades: List[Dict],
        timestamps: List[datetime],
        benchmark: Optional[List[float]] = None,
        initial_capital: Decimal = Decimal('10000')
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.
        
        Args:
            equity_curve: List of portfolio equity values
            trades: List of trade dictionaries with pnl, duration, etc.
            timestamps: List of timestamps for equity curve
            benchmark: Optional benchmark returns for comparison
            initial_capital: Starting capital for return calculations
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        # Input validation
        if not equity_curve:
            logger.warning("Empty equity curve provided")
            return self._create_empty_metrics()
        
        if len(equity_curve) != len(timestamps):
            logger.error(
                "Equity curve and timestamps length mismatch",
                equity_len=len(equity_curve),
                timestamps_len=len(timestamps)
            )
            raise ValueError("Equity curve and timestamps must have same length")
        
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        # Convert to numpy arrays for calculations
        equity_array = np.array([float(e) for e in equity_curve])
        returns = self._calculate_returns(equity_array)
        
        # Calculate base metrics
        total_return = self._total_return(equity_curve, initial_capital)
        annualized_return = self._annualized_return(returns)
        cagr = self._calculate_cagr(equity_curve, timestamps)
        
        # Risk metrics
        volatility = self._calculate_volatility(returns)
        downside_vol = self._calculate_downside_volatility(returns)
        drawdown_info = self._calculate_drawdown(equity_curve, timestamps)
        var_95, cvar_95 = self._calculate_var_cvar(returns)
        
        # Risk-adjusted returns
        sharpe = self._sharpe_ratio(tuple(returns))
        sortino = self._sortino_ratio(tuple(returns))
        calmar = self._calmar_ratio(annualized_return, drawdown_info.max_drawdown)
        
        # Trade statistics
        trade_stats = None
        if trades:
            trade_stats = self._calculate_trade_statistics(trades)
        
        # Period metrics
        daily_returns = self._calculate_daily_returns(equity_curve, timestamps)
        best_day, worst_day = self._calculate_best_worst_day(daily_returns)
        monthly_returns = self._calculate_monthly_returns(equity_curve, timestamps)
        best_month, worst_month = self._calculate_best_worst_month(monthly_returns)
        positive_days, negative_days = self._count_positive_negative_days(daily_returns)
        
        # Create base metrics
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            compound_annual_growth_rate=cagr,
            volatility=volatility,
            downside_volatility=downside_vol,
            max_drawdown=drawdown_info.max_drawdown,
            max_drawdown_duration=drawdown_info.max_drawdown_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=0.0,  # Will be calculated if benchmark provided
            trade_stats=trade_stats,
            best_day=best_day,
            worst_day=worst_day,
            best_month=best_month,
            worst_month=worst_month,
            positive_days=positive_days,
            negative_days=negative_days
        )
        
        # Benchmark comparison if provided
        if benchmark is not None:
            self._add_benchmark_metrics(metrics, returns, benchmark)
        
        # Additional metrics
        metrics.recovery_factor = self._calculate_recovery_factor(
            total_return, drawdown_info.max_drawdown
        )
        metrics.ulcer_index = self._calculate_ulcer_index(equity_curve)
        
        if trade_stats:
            metrics.kelly_criterion = self._calculate_kelly_criterion(trade_stats)
        
        logger.info(
            "Performance metrics calculated",
            total_return=float(total_return),
            sharpe_ratio=sharpe,
            max_drawdown=float(drawdown_info.max_drawdown)
        )
        
        return metrics
    
    def _calculate_returns(self, equity_array: np.ndarray) -> np.ndarray:
        """Calculate returns from equity curve."""
        if len(equity_array) < 2:
            return np.array([])
        
        # Calculate percentage returns
        returns = np.diff(equity_array) / equity_array[:-1]
        return returns[~np.isnan(returns)]  # Remove NaN values
    
    def _total_return(
        self, 
        equity_curve: List[Decimal], 
        initial_capital: Decimal
    ) -> Decimal:
        """Calculate total return."""
        if not equity_curve:
            return Decimal('0')
        
        final_equity = equity_curve[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        return total_return.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate compound return
        cumulative_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0.0
        
        # Annualize
        years = n_periods / self.periods_per_year
        if years <= 0:
            return 0.0
        
        annualized = (1 + cumulative_return) ** (1 / years) - 1
        return float(annualized)
    
    def _calculate_cagr(
        self,
        equity_curve: List[Decimal],
        timestamps: List[datetime]
    ) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(equity_curve) < 2 or len(timestamps) < 2:
            return 0.0
        
        initial_value = float(equity_curve[0])
        final_value = float(equity_curve[-1])
        
        if initial_value <= 0:
            return 0.0
        
        # Calculate time period in years
        time_delta = timestamps[-1] - timestamps[0]
        years = time_delta.days / 365.25
        
        if years <= 0:
            return 0.0
        
        # CAGR = (Final/Initial)^(1/years) - 1
        cagr = (final_value / initial_value) ** (1 / years) - 1
        return float(cagr)
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0
        
        # Calculate standard deviation of returns
        std_return = np.std(returns, ddof=1)
        
        # Annualize
        annual_volatility = std_return * np.sqrt(self.periods_per_year)
        return float(annual_volatility)
    
    def _calculate_downside_volatility(self, returns: np.ndarray) -> float:
        """Calculate downside deviation for Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        # Filter returns below MAR
        downside_returns = returns[returns < self.mar]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_dev = np.sqrt(np.mean(downside_returns ** 2))
        
        # Annualize
        annual_downside = downside_dev * np.sqrt(self.periods_per_year)
        return float(annual_downside)
    
    def _calculate_drawdown(
        self,
        equity_curve: List[Decimal],
        timestamps: List[datetime]
    ) -> DrawdownInfo:
        """Calculate maximum drawdown and related metrics."""
        if not equity_curve:
            return DrawdownInfo(
                max_drawdown=Decimal('0'),
                max_drawdown_duration=timedelta(0),
                current_drawdown=Decimal('0'),
                peak_value=Decimal('0'),
                trough_value=Decimal('0'),
                recovery_time=None,
                drawdown_start=None,
                drawdown_end=None
            )
        
        # Convert to numpy for calculations
        equity_array = np.array([float(e) for e in equity_curve])
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdown series
        drawdown = (equity_array - running_max) / running_max * 100
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_drawdown = abs(drawdown[max_dd_idx])
        
        # Find peak before max drawdown
        peak_idx = np.where(equity_array[:max_dd_idx + 1] == running_max[max_dd_idx])[0]
        if len(peak_idx) > 0:
            peak_idx = peak_idx[-1]
        else:
            peak_idx = 0
        
        # Calculate drawdown duration
        drawdown_start = timestamps[peak_idx] if peak_idx < len(timestamps) else None
        drawdown_end = timestamps[max_dd_idx] if max_dd_idx < len(timestamps) else None
        
        duration = timedelta(0)
        if drawdown_start and drawdown_end:
            duration = drawdown_end - drawdown_start
        
        # Check for recovery
        recovery_time = None
        if max_dd_idx < len(equity_array) - 1:
            recovery_idx = np.where(
                equity_array[max_dd_idx + 1:] >= equity_array[peak_idx]
            )[0]
            if len(recovery_idx) > 0:
                recovery_idx = recovery_idx[0] + max_dd_idx + 1
                if recovery_idx < len(timestamps):
                    recovery_time = timestamps[recovery_idx] - drawdown_end
        
        return DrawdownInfo(
            max_drawdown=Decimal(str(max_drawdown)),
            max_drawdown_duration=duration,
            current_drawdown=Decimal(str(abs(drawdown[-1]))),
            peak_value=equity_curve[peak_idx] if peak_idx < len(equity_curve) else Decimal('0'),
            trough_value=equity_curve[max_dd_idx] if max_dd_idx < len(equity_curve) else Decimal('0'),
            recovery_time=recovery_time,
            drawdown_start=drawdown_start,
            drawdown_end=drawdown_end
        )
    
    def _calculate_var_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Calculate percentile for VaR
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)
        
        # Calculate CVaR (expected loss beyond VaR)
        cvar_returns = returns[returns <= var]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var
        
        return float(var), float(cvar)
    
    def _sharpe_ratio_impl(self, returns: Tuple[float, ...]) -> float:
        """Calculate Sharpe ratio (implementation)."""
        returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return 0.0
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        annual_return = mean_return * self.periods_per_year
        annual_std = std_return * np.sqrt(self.periods_per_year)
        
        sharpe = (annual_return - self.risk_free_rate) / annual_std
        return float(sharpe)
    
    def _sharpe_ratio(self, returns: Tuple[float, ...]) -> float:
        """Calculate Sharpe ratio (cached wrapper)."""
        return self._sharpe_ratio_impl(returns)
    
    def _sortino_ratio_impl(self, returns: Tuple[float, ...]) -> float:
        """Calculate Sortino ratio (implementation)."""
        returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return 0.0
        
        mean_return = np.mean(returns_array)
        
        # Calculate downside deviation
        downside_returns = returns_array[returns_array < self.mar]
        
        if len(downside_returns) == 0:
            return 0.0  # No downside risk
        
        downside_dev = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_dev == 0:
            return 0.0
        
        # Annualize
        annual_return = mean_return * self.periods_per_year
        annual_downside = downside_dev * np.sqrt(self.periods_per_year)
        
        sortino = (annual_return - self.mar) / annual_downside
        return float(sortino)
    
    def _sortino_ratio(self, returns: Tuple[float, ...]) -> float:
        """Calculate Sortino ratio (cached wrapper)."""
        return self._sortino_ratio_impl(returns)
    
    def _calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: Decimal
    ) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        
        calmar = annualized_return / abs(float(max_drawdown) / 100)
        return float(calmar)
    
    def _calculate_trade_statistics(self, trades: List[Dict]) -> TradeStatistics:
        """Calculate comprehensive trade statistics."""
        if not trades:
            return TradeStatistics(
                total_trades=0, winning_trades=0, losing_trades=0,
                breakeven_trades=0, win_rate=0.0, loss_rate=0.0,
                profit_factor=0.0, avg_win=Decimal('0'), avg_loss=Decimal('0'),
                largest_win=Decimal('0'), largest_loss=Decimal('0'),
                avg_trade=Decimal('0'), avg_trade_duration=timedelta(0),
                max_consecutive_wins=0, max_consecutive_losses=0,
                current_streak=0, expectancy=Decimal('0'), payoff_ratio=0.0
            )
        
        # Classify trades
        winning_trades = [t for t in trades if Decimal(str(t.get('pnl', 0))) > 0]
        losing_trades = [t for t in trades if Decimal(str(t.get('pnl', 0))) < 0]
        breakeven_trades = [t for t in trades if Decimal(str(t.get('pnl', 0))) == 0]
        
        # Basic counts
        total_trades = len(trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        num_breakeven = len(breakeven_trades)
        
        # Win/loss rates
        win_rate = num_wins / total_trades if total_trades > 0 else 0.0
        loss_rate = num_losses / total_trades if total_trades > 0 else 0.0
        
        # Calculate averages
        avg_win = Decimal('0')
        avg_loss = Decimal('0')
        largest_win = Decimal('0')
        largest_loss = Decimal('0')
        
        if winning_trades:
            wins = [Decimal(str(t['pnl'])) for t in winning_trades]
            avg_win = sum(wins) / len(wins)
            largest_win = max(wins)
        
        if losing_trades:
            losses = [abs(Decimal(str(t['pnl']))) for t in losing_trades]
            avg_loss = sum(losses) / len(losses)
            largest_loss = max(losses)
        
        # Profit factor with better edge case handling
        gross_profit = sum(Decimal(str(t['pnl'])) for t in winning_trades) if winning_trades else Decimal('0')
        gross_loss = abs(sum(Decimal(str(t['pnl'])) for t in losing_trades)) if losing_trades else Decimal('0')
        
        # Handle edge cases for profit factor
        if gross_loss == 0 and gross_profit > 0:
            # All trades profitable - use a large but finite value
            profit_factor = 999.99
            logger.info("All trades profitable - profit factor set to maximum")
        elif gross_loss > 0:
            profit_factor = float(gross_profit / gross_loss)
        else:
            profit_factor = 0.0
        
        # Average trade
        all_pnl = [Decimal(str(t['pnl'])) for t in trades]
        avg_trade = sum(all_pnl) / len(all_pnl) if all_pnl else Decimal('0')
        
        # Trade duration
        durations = []
        for t in trades:
            if 'duration' in t:
                if isinstance(t['duration'], timedelta):
                    durations.append(t['duration'])
                elif isinstance(t['duration'], (int, float)):
                    durations.append(timedelta(seconds=t['duration']))
        
        avg_trade_duration = (
            sum(durations, timedelta(0)) / len(durations)
            if durations else timedelta(0)
        )
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses, current_streak = self._calculate_streaks(trades)
        
        # Expectancy
        expectancy = (Decimal(str(win_rate)) * avg_win) - (Decimal(str(loss_rate)) * avg_loss)
        
        # Payoff ratio
        payoff_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
        
        return TradeStatistics(
            total_trades=total_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            breakeven_trades=num_breakeven,
            win_rate=win_rate,
            loss_rate=loss_rate,
            profit_factor=profit_factor,
            avg_win=avg_win.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            avg_loss=avg_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            largest_win=largest_win.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            largest_loss=largest_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            avg_trade=avg_trade.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            avg_trade_duration=avg_trade_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            current_streak=current_streak,
            expectancy=expectancy.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            payoff_ratio=payoff_ratio
        )
    
    def _calculate_streaks(self, trades: List[Dict]) -> Tuple[int, int, int]:
        """Calculate consecutive win/loss streaks."""
        if not trades:
            return 0, 0, 0
        
        max_wins = 0
        max_losses = 0
        current_streak = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            pnl = Decimal(str(trade.get('pnl', 0)))
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                current_streak = current_wins
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                current_streak = -current_losses
            else:
                # Breakeven trade resets streak
                current_wins = 0
                current_losses = 0
                current_streak = 0
            
            max_wins = max(max_wins, current_wins)
            max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses, current_streak
    
    def _calculate_daily_returns(
        self,
        equity_curve: List[Decimal],
        timestamps: List[datetime]
    ) -> List[Tuple[datetime, Decimal]]:
        """Calculate daily returns from equity curve."""
        if len(equity_curve) < 2 or len(timestamps) < 2:
            return []
        
        daily_returns = []
        current_date = timestamps[0].date()
        day_start_equity = equity_curve[0]
        
        for i in range(1, len(timestamps)):
            if timestamps[i].date() != current_date:
                # New day - calculate return for previous day
                day_end_equity = equity_curve[i - 1]
                if day_start_equity > 0:
                    day_return = ((day_end_equity - day_start_equity) / day_start_equity) * 100
                    daily_returns.append((timestamps[i - 1], day_return))
                
                # Reset for new day
                current_date = timestamps[i].date()
                day_start_equity = equity_curve[i]
        
        # Handle last day
        if len(equity_curve) > 1 and day_start_equity > 0:
            day_return = ((equity_curve[-1] - day_start_equity) / day_start_equity) * 100
            daily_returns.append((timestamps[-1], day_return))
        
        return daily_returns
    
    def _calculate_best_worst_day(
        self,
        daily_returns: List[Tuple[datetime, Decimal]]
    ) -> Tuple[Decimal, Decimal]:
        """Find best and worst daily returns."""
        if not daily_returns:
            return Decimal('0'), Decimal('0')
        
        returns = [r[1] for r in daily_returns]
        return max(returns), min(returns)
    
    def _calculate_monthly_returns(
        self,
        equity_curve: List[Decimal],
        timestamps: List[datetime]
    ) -> List[Tuple[datetime, Decimal]]:
        """Calculate monthly returns from equity curve."""
        if len(equity_curve) < 2 or len(timestamps) < 2:
            return []
        
        monthly_returns = []
        current_month = (timestamps[0].year, timestamps[0].month)
        month_start_equity = equity_curve[0]
        
        for i in range(1, len(timestamps)):
            month = (timestamps[i].year, timestamps[i].month)
            
            if month != current_month:
                # New month - calculate return for previous month
                month_end_equity = equity_curve[i - 1]
                if month_start_equity > 0:
                    month_return = ((month_end_equity - month_start_equity) / month_start_equity) * 100
                    monthly_returns.append((timestamps[i - 1], month_return))
                
                # Reset for new month
                current_month = month
                month_start_equity = equity_curve[i]
        
        # Handle last month
        if len(equity_curve) > 1 and month_start_equity > 0:
            month_return = ((equity_curve[-1] - month_start_equity) / month_start_equity) * 100
            monthly_returns.append((timestamps[-1], month_return))
        
        return monthly_returns
    
    def _calculate_best_worst_month(
        self,
        monthly_returns: List[Tuple[datetime, Decimal]]
    ) -> Tuple[Decimal, Decimal]:
        """Find best and worst monthly returns."""
        if not monthly_returns:
            return Decimal('0'), Decimal('0')
        
        returns = [r[1] for r in monthly_returns]
        return max(returns), min(returns)
    
    def _count_positive_negative_days(
        self,
        daily_returns: List[Tuple[datetime, Decimal]]
    ) -> Tuple[int, int]:
        """Count positive and negative return days."""
        if not daily_returns:
            return 0, 0
        
        positive = sum(1 for _, r in daily_returns if r > 0)
        negative = sum(1 for _, r in daily_returns if r < 0)
        
        return positive, negative
    
    def _add_benchmark_metrics(
        self,
        metrics: PerformanceMetrics,
        returns: np.ndarray,
        benchmark: List[float]
    ) -> None:
        """Add benchmark comparison metrics."""
        if len(returns) != len(benchmark):
            logger.warning(
                "Benchmark length mismatch",
                returns_len=len(returns),
                benchmark_len=len(benchmark)
            )
            return
        
        benchmark_array = np.array(benchmark)
        
        # Calculate beta
        if np.var(benchmark_array) > 0:
            metrics.beta = float(np.cov(returns, benchmark_array)[0, 1] / np.var(benchmark_array))
        
        # Calculate alpha (Jensen's alpha)
        if metrics.beta is not None:
            benchmark_return = np.mean(benchmark_array) * self.periods_per_year
            expected_return = self.risk_free_rate + metrics.beta * (benchmark_return - self.risk_free_rate)
            actual_return = np.mean(returns) * self.periods_per_year
            metrics.alpha = float(actual_return - expected_return)
        
        # Calculate correlation
        if len(returns) > 1 and len(benchmark_array) > 1:
            correlation_matrix = np.corrcoef(returns, benchmark_array)
            metrics.correlation = float(correlation_matrix[0, 1])
        
        # Calculate tracking error
        tracking_diff = returns - benchmark_array
        metrics.tracking_error = float(np.std(tracking_diff) * np.sqrt(self.periods_per_year))
        
        # Calculate information ratio
        if metrics.tracking_error > 0:
            excess_return = np.mean(tracking_diff) * self.periods_per_year
            metrics.information_ratio = float(excess_return / metrics.tracking_error)
        
        # Calculate Treynor ratio
        if metrics.beta and metrics.beta != 0:
            portfolio_return = np.mean(returns) * self.periods_per_year
            metrics.treynor_ratio = float((portfolio_return - self.risk_free_rate) / metrics.beta)
    
    def _information_ratio_impl(
        self,
        returns: Tuple[float, ...],
        benchmark: Tuple[float, ...]
    ) -> float:
        """Calculate information ratio (implementation)."""
        if len(returns) != len(benchmark):
            return 0.0
        
        returns_array = np.array(returns)
        benchmark_array = np.array(benchmark)
        
        # Calculate excess returns
        excess_returns = returns_array - benchmark_array
        
        if len(excess_returns) == 0:
            return 0.0
        
        # Calculate tracking error
        tracking_error = np.std(excess_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
        
        # Annualize
        annual_excess = np.mean(excess_returns) * self.periods_per_year
        annual_tracking = tracking_error * np.sqrt(self.periods_per_year)
        
        return float(annual_excess / annual_tracking)
    
    def _information_ratio(
        self,
        returns: Tuple[float, ...],
        benchmark: Tuple[float, ...]
    ) -> float:
        """Calculate information ratio (cached wrapper)."""
        return self._information_ratio_impl(returns, benchmark)
    
    def _calculate_recovery_factor(
        self,
        total_return: Decimal,
        max_drawdown: Decimal
    ) -> float:
        """Calculate recovery factor (return / max drawdown)."""
        if max_drawdown == 0:
            return 0.0
        
        return float(abs(total_return) / abs(max_drawdown))
    
    def _calculate_ulcer_index(self, equity_curve: List[Decimal]) -> float:
        """Calculate Ulcer Index (measure of downside volatility)."""
        if len(equity_curve) < 2:
            return 0.0
        
        # Convert to numpy array
        equity_array = np.array([float(e) for e in equity_curve])
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate percentage drawdown from peak
        drawdown_pct = ((equity_array - running_max) / running_max) * 100
        
        # Calculate Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
        
        return float(ulcer_index)
    
    def _calculate_kelly_criterion(self, trade_stats: TradeStatistics) -> float:
        """Calculate Kelly Criterion for position sizing."""
        if trade_stats.win_rate == 0 or trade_stats.payoff_ratio == 0:
            return 0.0
        
        # Kelly % = (p * b - q) / b
        # p = win rate, q = loss rate, b = payoff ratio
        p = trade_stats.win_rate
        q = trade_stats.loss_rate
        b = trade_stats.payoff_ratio
        
        if b == 0:
            return 0.0
        
        kelly_pct = (p * b - q) / b
        
        # Cap at reasonable levels (typically use 25% of Kelly)
        return float(min(max(kelly_pct, 0), 0.25))
    
    def calculate_rolling_metrics(
        self,
        equity_curve: List[Decimal],
        timestamps: List[datetime],
        window_size: int = 30,
        step_size: int = 1
    ) -> RollingMetrics:
        """Calculate rolling window performance metrics.
        
        Args:
            equity_curve: List of portfolio equity values
            timestamps: List of timestamps
            window_size: Size of rolling window (in periods)
            step_size: Step size for rolling window
            
        Returns:
            RollingMetrics object with time series of metrics
        """
        if len(equity_curve) < window_size:
            logger.warning(
                "Insufficient data for rolling metrics",
                data_len=len(equity_curve),
                window_size=window_size
            )
            return RollingMetrics(
                window_size=window_size,
                returns=[],
                sharpe_ratios=[],
                sortino_ratios=[],
                win_rates=[],
                drawdowns=[],
                volatilities=[]
            )
        
        rolling_returns = []
        rolling_sharpe = []
        rolling_sortino = []
        rolling_volatility = []
        rolling_drawdowns = []
        
        for i in range(window_size, len(equity_curve), step_size):
            # Get window data
            window_equity = equity_curve[i - window_size:i]
            window_timestamps = timestamps[i - window_size:i] if timestamps else []
            
            # Convert to array and calculate returns
            equity_array = np.array([float(e) for e in window_equity])
            returns = self._calculate_returns(equity_array)
            
            if len(returns) > 0:
                # Calculate metrics for this window
                rolling_returns.append(float(np.mean(returns) * self.periods_per_year))
                rolling_sharpe.append(self._sharpe_ratio(tuple(returns)))
                rolling_sortino.append(self._sortino_ratio(tuple(returns)))
                rolling_volatility.append(self._calculate_volatility(returns))
                
                # Calculate drawdown
                drawdown_info = self._calculate_drawdown(window_equity, window_timestamps)
                rolling_drawdowns.append(float(drawdown_info.max_drawdown))
        
        return RollingMetrics(
            window_size=window_size,
            returns=rolling_returns,
            sharpe_ratios=rolling_sharpe,
            sortino_ratios=rolling_sortino,
            win_rates=[],  # Would need trade data for this
            drawdowns=rolling_drawdowns,
            volatilities=rolling_volatility
        )
    
    def generate_performance_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate a summary dictionary of key performance metrics.
        
        Args:
            metrics: PerformanceMetrics object
            
        Returns:
            Dictionary with formatted metric values
        """
        summary = {
            'returns': {
                'total_return': f"{metrics.total_return:.2f}%",
                'annualized_return': f"{metrics.annualized_return:.2%}",
                'cagr': f"{metrics.compound_annual_growth_rate:.2%}"
            },
            'risk': {
                'volatility': f"{metrics.volatility:.2%}",
                'max_drawdown': f"{metrics.max_drawdown:.2f}%",
                'max_dd_duration': str(metrics.max_drawdown_duration),
                'var_95': f"{metrics.var_95:.2%}",
                'cvar_95': f"{metrics.cvar_95:.2%}"
            },
            'risk_adjusted': {
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'information_ratio': f"{metrics.information_ratio:.2f}"
            }
        }
        
        if metrics.trade_stats:
            summary['trades'] = {
                'total_trades': metrics.trade_stats.total_trades,
                'win_rate': f"{metrics.trade_stats.win_rate:.1%}",
                'profit_factor': f"{metrics.trade_stats.profit_factor:.2f}",
                'avg_win': f"{metrics.trade_stats.avg_win:.2f}",
                'avg_loss': f"{metrics.trade_stats.avg_loss:.2f}",
                'expectancy': f"{metrics.trade_stats.expectancy:.2f}",
                'max_consecutive_wins': metrics.trade_stats.max_consecutive_wins,
                'max_consecutive_losses': metrics.trade_stats.max_consecutive_losses
            }
        
        if metrics.beta is not None:
            summary['benchmark'] = {
                'beta': f"{metrics.beta:.2f}",
                'alpha': f"{metrics.alpha:.2%}" if metrics.alpha else "N/A",
                'correlation': f"{metrics.correlation:.2f}" if metrics.correlation else "N/A",
                'tracking_error': f"{metrics.tracking_error:.2%}" if metrics.tracking_error else "N/A"
            }
        
        return summary