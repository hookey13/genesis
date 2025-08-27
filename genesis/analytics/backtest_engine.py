"""Backtesting framework for statistical arbitrage strategies."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Optional, Any

import numpy as np
import pandas as pd
import structlog

# Set precision for financial calculations
getcontext().prec = 10

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Represents a single trade in backtesting."""

    entry_time: datetime
    exit_time: Optional[datetime]
    pair1_symbol: str
    pair2_symbol: str
    entry_zscore: Decimal
    exit_zscore: Optional[Decimal]
    entry_price1: Decimal
    entry_price2: Decimal
    exit_price1: Optional[Decimal]
    exit_price2: Optional[Decimal]
    position_size: Decimal
    pnl: Optional[Decimal] = None
    pnl_percent: Optional[Decimal] = None
    is_open: bool = True

    def calculate_pnl(self) -> Decimal:
        """Calculate P&L for the trade."""
        if not self.exit_price1 or not self.exit_price2:
            return Decimal("0")

        # Calculate spread change
        entry_spread = self.entry_price1 / self.entry_price2
        exit_spread = self.exit_price1 / self.exit_price2

        # Calculate P&L based on spread change
        spread_change = (exit_spread - entry_spread) / entry_spread

        # Adjust for direction (positive zscore means spread is too high, should decrease)
        if self.entry_zscore > 0:
            self.pnl = -spread_change * self.position_size
        else:
            self.pnl = spread_change * self.position_size

        self.pnl_percent = (self.pnl / self.position_size) * Decimal("100")

        return self.pnl


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_percent: Decimal
    total_pnl: Decimal
    total_return_percent: Decimal
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)
    performance_by_pair: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": str(self.initial_capital),
            "final_capital": str(self.final_capital),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": str(self.win_rate),
            "average_win": str(self.average_win),
            "average_loss": str(self.average_loss),
            "profit_factor": str(self.profit_factor),
            "sharpe_ratio": str(self.sharpe_ratio),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_percent": str(self.max_drawdown_percent),
            "total_pnl": str(self.total_pnl),
            "total_return_percent": str(self.total_return_percent),
            "trade_count": len(self.trades),
            "performance_by_pair": self.performance_by_pair,
        }


class BacktestEngine:
    """Engine for backtesting statistical arbitrage strategies."""

    def __init__(
        self,
        initial_capital: Decimal = Decimal("10000"),
        position_size_percent: Decimal = Decimal("0.1"),
        transaction_cost_percent: Decimal = Decimal("0.001"),
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            position_size_percent: Percentage of capital per trade
            transaction_cost_percent: Transaction cost as percentage
        """
        self.initial_capital = initial_capital
        self.position_size_percent = position_size_percent
        self.transaction_cost_percent = transaction_cost_percent
        self.current_capital = initial_capital
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, Decimal]] = []

    async def run_backtest(
        self,
        strategy,
        historical_data: dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        entry_threshold: Decimal = Decimal("2"),
        exit_threshold: Decimal = Decimal("0.5"),
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            strategy: Statistical arbitrage strategy instance
            historical_data: Dictionary mapping symbol pairs to price DataFrames
            start_date: Backtest start date
            end_date: Backtest end date
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(
            "Starting backtest",
            start_date=start_date,
            end_date=end_date,
            pairs=len(historical_data),
        )

        # Reset state
        self.current_capital = self.initial_capital
        self.open_trades = []
        self.closed_trades = []
        self.equity_curve = [(start_date, self.initial_capital)]

        # Process data chronologically
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index)

        timestamps = sorted(all_timestamps)
        timestamps = [t for t in timestamps if start_date <= t <= end_date]

        for timestamp in timestamps:
            # Check for signals on each pair
            for pair_key, df in historical_data.items():
                if timestamp not in df.index:
                    continue

                pair1, pair2 = pair_key.split(":")

                # Get current prices
                if "price1" not in df.columns or "price2" not in df.columns:
                    continue

                price1 = Decimal(str(df.loc[timestamp, "price1"]))
                price2 = Decimal(str(df.loc[timestamp, "price2"]))

                # Calculate z-score
                window = min(20, len(df[:timestamp]))
                if window < 5:  # Need minimum data
                    continue

                prices1 = [Decimal(str(p)) for p in df["price1"][:timestamp][-window:]]
                prices2 = [Decimal(str(p)) for p in df["price2"][:timestamp][-window:]]

                zscore = strategy.calculate_zscore(
                    price1, price2, window, prices1, prices2
                )

                # Check for entry signal
                if abs(zscore) >= entry_threshold:
                    await self._enter_trade(
                        timestamp, pair1, pair2, zscore, price1, price2
                    )

                # Check for exit signals on open trades
                await self._check_exits(
                    timestamp, pair1, pair2, zscore, price1, price2, exit_threshold
                )

            # Update equity curve
            equity = await self._calculate_equity(timestamp)
            self.equity_curve.append((timestamp, equity))

        # Close any remaining open trades
        for trade in self.open_trades:
            trade.is_open = False
            self.closed_trades.append(trade)

        # Calculate performance metrics
        result = await self._calculate_metrics(start_date, end_date)

        logger.info(
            "Backtest complete",
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            total_return=result.total_return_percent,
        )

        return result

    async def _enter_trade(
        self,
        timestamp: datetime,
        pair1: str,
        pair2: str,
        zscore: Decimal,
        price1: Decimal,
        price2: Decimal,
    ) -> None:
        """Enter a new trade."""
        # Check if already in position for this pair
        for trade in self.open_trades:
            if trade.pair1_symbol == pair1 and trade.pair2_symbol == pair2:
                return  # Already in position

        # Calculate position size
        position_size = self.current_capital * self.position_size_percent

        # Apply transaction costs
        cost = position_size * self.transaction_cost_percent
        self.current_capital -= cost

        # Create trade
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            pair1_symbol=pair1,
            pair2_symbol=pair2,
            entry_zscore=zscore,
            exit_zscore=None,
            entry_price1=price1,
            entry_price2=price2,
            exit_price1=None,
            exit_price2=None,
            position_size=position_size,
            is_open=True,
        )

        self.open_trades.append(trade)

        logger.debug(
            "Trade entered",
            timestamp=timestamp,
            pairs=f"{pair1}:{pair2}",
            zscore=zscore,
            position_size=position_size,
        )

    async def _check_exits(
        self,
        timestamp: datetime,
        pair1: str,
        pair2: str,
        zscore: Decimal,
        price1: Decimal,
        price2: Decimal,
        exit_threshold: Decimal,
    ) -> None:
        """Check if any open trades should be exited."""
        trades_to_close = []

        for trade in self.open_trades:
            if trade.pair1_symbol != pair1 or trade.pair2_symbol != pair2:
                continue

            # Exit if z-score has returned to mean
            if abs(zscore) <= exit_threshold:
                trade.exit_time = timestamp
                trade.exit_zscore = zscore
                trade.exit_price1 = price1
                trade.exit_price2 = price2
                trade.is_open = False

                # Calculate P&L
                pnl = trade.calculate_pnl()

                # Apply transaction costs
                cost = trade.position_size * self.transaction_cost_percent
                pnl -= cost
                trade.pnl = pnl

                # Update capital
                self.current_capital += trade.position_size + pnl

                trades_to_close.append(trade)

                logger.debug(
                    "Trade exited",
                    timestamp=timestamp,
                    pairs=f"{pair1}:{pair2}",
                    pnl=pnl,
                    pnl_percent=trade.pnl_percent,
                )

        # Move closed trades
        for trade in trades_to_close:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)

    async def _calculate_equity(self, _timestamp: datetime) -> Decimal:
        """Calculate current equity including open positions."""
        equity = self.current_capital

        # Add unrealized P&L from open trades
        for trade in self.open_trades:
            # Simplified: assume no change in open trade value
            # In production, would mark-to-market
            equity += trade.position_size

        return equity

    async def _calculate_metrics(
        self, start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """Calculate performance metrics from trades."""
        # Basic metrics
        total_trades = len(self.closed_trades)

        if total_trades == 0:
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=Decimal("0"),
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                profit_factor=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                max_drawdown=Decimal("0"),
                max_drawdown_percent=Decimal("0"),
                total_pnl=Decimal("0"),
                total_return_percent=Decimal("0"),
            )

        # Win/loss statistics
        winning_trades = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl and t.pnl <= 0]

        win_rate = Decimal(str(len(winning_trades))) / Decimal(str(total_trades))

        # Average win/loss
        average_win = (
            sum(t.pnl for t in winning_trades) / len(winning_trades)
            if winning_trades
            else Decimal("0")
        )
        average_loss = (
            sum(abs(t.pnl) for t in losing_trades) / len(losing_trades)
            if losing_trades
            else Decimal("0")
        )

        # Profit factor
        gross_profit = (
            sum(t.pnl for t in winning_trades) if winning_trades else Decimal("0")
        )
        gross_loss = (
            sum(abs(t.pnl) for t in losing_trades) if losing_trades else Decimal("0")
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("999")

        # Total P&L and return
        total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl)
        final_capital = (
            self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        )
        total_return_percent = (
            (final_capital - self.initial_capital)
            / self.initial_capital
            * Decimal("100")
        )

        # Sharpe ratio (simplified daily)
        if len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i - 1][1]
                curr_equity = self.equity_curve[i][1]
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(float(daily_return))

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = Decimal(
                    str(avg_return / std_return * np.sqrt(252) if std_return > 0 else 0)
                )
            else:
                sharpe_ratio = Decimal("0")
        else:
            sharpe_ratio = Decimal("0")

        # Maximum drawdown
        max_drawdown, max_drawdown_percent = await self._calculate_drawdown()

        # Performance by pair
        performance_by_pair = await self._calculate_pair_performance()

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            total_pnl=total_pnl,
            total_return_percent=total_return_percent,
            trades=self.closed_trades,
            equity_curve=self.equity_curve,
            performance_by_pair=performance_by_pair,
        )

    async def _calculate_drawdown(self) -> tuple[Decimal, Decimal]:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return Decimal("0"), Decimal("0")

        peak = self.equity_curve[0][1]
        max_drawdown = Decimal("0")
        max_drawdown_percent = Decimal("0")

        for _timestamp, equity in self.equity_curve:
            if equity > peak:
                peak = equity

            drawdown = peak - equity
            drawdown_percent = (
                drawdown / peak * Decimal("100") if peak > 0 else Decimal("0")
            )

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = drawdown_percent

        return max_drawdown, max_drawdown_percent

    async def _calculate_pair_performance(self) -> dict[str, dict[str, Any]]:
        """Calculate performance metrics by pair."""
        pair_performance = {}

        for trade in self.closed_trades:
            pair_key = f"{trade.pair1_symbol}:{trade.pair2_symbol}"

            if pair_key not in pair_performance:
                pair_performance[pair_key] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": Decimal("0"),
                    "win_rate": Decimal("0"),
                }

            stats = pair_performance[pair_key]
            stats["trades"] += 1

            if trade.pnl and trade.pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

            if trade.pnl:
                stats["total_pnl"] += trade.pnl

            stats["win_rate"] = Decimal(str(stats["wins"])) / Decimal(
                str(stats["trades"])
            )

        # Convert Decimals to strings for serialization
        for _pair_key, stats in pair_performance.items():
            stats["total_pnl"] = str(stats["total_pnl"])
            stats["win_rate"] = str(stats["win_rate"])

        return pair_performance

    def plot_equity_curve(self) -> None:
        """Plot equity curve (placeholder for visualization)."""
        # In production, would use matplotlib or plotly
        logger.info(
            "Equity curve data available for plotting",
            data_points=len(self.equity_curve),
        )
