"""
Backtest Engine Core Implementation

Provides the main backtesting engine for historical strategy validation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Any, AsyncIterator, Tuple
from enum import Enum

import structlog

logger = structlog.get_logger()


class BacktestStatus(Enum):
    """Status of a backtest run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("1000")
    symbols: List[str] = field(default_factory=list)
    resolution: str = "1m"
    slippage_model: str = "linear"
    fee_model: str = "binance"
    max_drawdown: Decimal = Decimal("0.20")
    enable_shorting: bool = False
    use_leverage: bool = False
    max_leverage: Decimal = Decimal("1.0")
    

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    avg_win: Decimal
    avg_loss: Decimal
    portfolio_history: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    """Point-in-time market data."""
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_volume: Optional[Decimal] = None
    ask_volume: Optional[Decimal] = None


class BacktestEngine:
    """
    Core backtesting engine for strategy validation.
    
    Handles historical data replay, strategy execution simulation,
    and performance metric calculation.
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize the backtest engine."""
        self.config = config
        self.status = BacktestStatus.PENDING
        self.data_provider = None
        self.execution_simulator = None
        self.portfolio = None
        self.event_log = []
        self.tick_count = 0
        self._stop_requested = False
        
    async def initialize(self) -> None:
        """
        Initialize engine components.
        
        Sets up data provider, execution simulator, and portfolio.
        """
        from genesis.backtesting.data_provider import HistoricalDataProvider
        from genesis.backtesting.execution_simulator import ExecutionSimulator
        from genesis.backtesting.portfolio import Portfolio
        
        self.data_provider = HistoricalDataProvider()
        self.execution_simulator = ExecutionSimulator(
            slippage_model=self.config.slippage_model,
            fee_model=self.config.fee_model
        )
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            enable_shorting=self.config.enable_shorting,
            use_leverage=self.config.use_leverage,
            max_leverage=self.config.max_leverage
        )
        
        logger.info(
            "backtest_engine_initialized",
            config=self.config,
            components={
                "data_provider": self.data_provider is not None,
                "execution_simulator": self.execution_simulator is not None,
                "portfolio": self.portfolio is not None
            }
        )
        
    async def run_backtest(
        self,
        strategy: Any,  # Will be BaseStrategy once available
        progress_callback: Optional[callable] = None
    ) -> BacktestResult:
        """
        Run a backtest for the given strategy.
        
        Args:
            strategy: The strategy to backtest
            progress_callback: Optional callback for progress updates
            
        Returns:
            BacktestResult containing performance metrics
        """
        try:
            self.status = BacktestStatus.RUNNING
            await self.initialize()
            
            logger.info(
                "backtest_started",
                strategy=strategy.__class__.__name__ if hasattr(strategy, '__class__') else str(strategy),
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                symbols=self.config.symbols
            )
            
            # Load historical data - this returns an async generator, not a coroutine
            data_stream = self.data_provider.load_data(
                symbols=self.config.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                resolution=self.config.resolution
            )
            
            # Main backtest loop - replay historical data
            async for timestamp, market_snapshot in self._replay_data(data_stream):
                if self._stop_requested:
                    logger.warning("backtest_cancelled")
                    self.status = BacktestStatus.CANCELLED
                    break
                    
                self.tick_count += 1
                
                # Update portfolio with current market prices
                await self.portfolio.mark_to_market(market_snapshot)
                
                # Check drawdown limit
                if self.portfolio.current_drawdown > self.config.max_drawdown:
                    logger.warning(
                        "max_drawdown_exceeded",
                        drawdown=float(self.portfolio.current_drawdown),
                        limit=float(self.config.max_drawdown)
                    )
                    break
                
                # Generate trading signals
                signal = await strategy.analyze(market_snapshot)
                
                if signal:
                    # Simulate order execution
                    fill = await self.execution_simulator.simulate_fill(
                        signal=signal,
                        market_data=market_snapshot,
                        portfolio=self.portfolio
                    )
                    
                    if fill:
                        # Update portfolio with the fill
                        await self.portfolio.process_fill(fill)
                        self.event_log.append({
                            "type": "fill",
                            "timestamp": timestamp,
                            "data": fill
                        })
                
                # Progress callback
                if progress_callback and self.tick_count % 100 == 0:
                    progress = self._calculate_progress(timestamp)
                    await progress_callback(progress)
            
            # Generate final results
            result = await self._generate_results(strategy)
            self.status = BacktestStatus.COMPLETED
            
            logger.info(
                "backtest_completed",
                strategy=strategy.__class__.__name__ if hasattr(strategy, '__class__') else str(strategy),
                total_trades=result.total_trades,
                final_pnl=float(result.final_capital - result.initial_capital),
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=float(result.max_drawdown)
            )
            
            return result
            
        except Exception as e:
            self.status = BacktestStatus.FAILED
            logger.error(
                "backtest_failed",
                error=str(e),
                strategy=strategy.__class__.__name__ if hasattr(strategy, '__class__') else str(strategy)
            )
            raise
    
    async def _replay_data(
        self,
        data_stream: AsyncIterator
    ) -> AsyncIterator[Tuple[datetime, MarketSnapshot]]:
        """
        Replay historical data chronologically.
        
        Args:
            data_stream: Stream of historical market data
            
        Yields:
            Tuples of (timestamp, market_snapshot)
        """
        async for data_point in data_stream:
            if isinstance(data_point, dict):
                # Convert dict to MarketSnapshot
                snapshot = MarketSnapshot(
                    timestamp=data_point.get('timestamp'),
                    symbol=data_point.get('symbol'),
                    open=Decimal(str(data_point.get('open', 0))),
                    high=Decimal(str(data_point.get('high', 0))),
                    low=Decimal(str(data_point.get('low', 0))),
                    close=Decimal(str(data_point.get('close', 0))),
                    volume=Decimal(str(data_point.get('volume', 0))),
                    bid_price=Decimal(str(data_point.get('bid_price'))) if data_point.get('bid_price') else None,
                    ask_price=Decimal(str(data_point.get('ask_price'))) if data_point.get('ask_price') else None,
                )
                yield snapshot.timestamp, snapshot
            else:
                yield data_point.timestamp, data_point
    
    def _calculate_progress(self, current_time: datetime) -> float:
        """
        Calculate backtest progress as percentage.
        
        Args:
            current_time: Current timestamp in replay
            
        Returns:
            Progress percentage (0-100)
        """
        total_duration = (self.config.end_date - self.config.start_date).total_seconds()
        elapsed = (current_time - self.config.start_date).total_seconds()
        return min(100.0, (elapsed / total_duration) * 100) if total_duration > 0 else 0.0
    
    async def _generate_results(self, strategy: Any) -> BacktestResult:
        """
        Generate comprehensive backtest results.
        
        Args:
            strategy: The strategy that was tested
            
        Returns:
            BacktestResult with all metrics calculated
        """
        # Get portfolio metrics
        portfolio_stats = await self.portfolio.get_statistics()
        
        # Calculate performance metrics
        trades = self.portfolio.closed_trades
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        
        total_wins = sum(Decimal(str(t.get('pnl', 0))) for t in winning_trades)
        total_losses = abs(sum(Decimal(str(t.get('pnl', 0))) for t in losing_trades))
        
        return BacktestResult(
            strategy_name=strategy.__class__.__name__ if hasattr(strategy, '__class__') else str(strategy),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=self.portfolio.total_equity,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            max_drawdown=portfolio_stats.get('max_drawdown', Decimal('0')),
            sharpe_ratio=portfolio_stats.get('sharpe_ratio', 0.0),
            sortino_ratio=portfolio_stats.get('sortino_ratio', 0.0),
            calmar_ratio=portfolio_stats.get('calmar_ratio', 0.0),
            profit_factor=float(total_wins / total_losses) if total_losses > 0 else 0.0,
            win_rate=len(winning_trades) / len(trades) if trades else 0.0,
            avg_win=total_wins / len(winning_trades) if winning_trades else Decimal('0'),
            avg_loss=total_losses / len(losing_trades) if losing_trades else Decimal('0'),
            portfolio_history=self.portfolio.history,
            trades=trades,
            metadata={
                'tick_count': self.tick_count,
                'config': self.config.__dict__,
                'events': len(self.event_log)
            }
        )
    
    async def stop(self) -> None:
        """Request graceful stop of the backtest."""
        self._stop_requested = True
        logger.info("backtest_stop_requested")
    
    async def validate_data_availability(self) -> bool:
        """
        Check if required historical data is available.
        
        Returns:
            True if data is available, False otherwise
        """
        if not self.data_provider:
            await self.initialize()
            
        return await self.data_provider.check_data_availability(
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            resolution=self.config.resolution
        )