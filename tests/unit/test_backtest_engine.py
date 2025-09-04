"""
Unit tests for the BacktestEngine.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from genesis.backtesting.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestStatus,
    MarketSnapshot
)
from genesis.backtesting.portfolio import Portfolio, Position
from genesis.backtesting.execution_simulator import (
    ExecutionSimulator,
    Signal,
    Fill,
    OrderSide,
    OrderType,
    FillStatus
)


class TestBacktestEngine:
    """Test suite for BacktestEngine."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            initial_capital=Decimal("10000"),
            symbols=["BTC/USDT", "ETH/USDT"],
            resolution="1m",
            slippage_model="linear",
            fee_model="binance",
            max_drawdown=Decimal("0.20")
        )
    
    @pytest.fixture
    def engine(self, config):
        """Create test engine."""
        return BacktestEngine(config)
    
    @pytest.fixture
    def mock_strategy(self):
        """Create mock strategy."""
        strategy = AsyncMock()
        strategy.__class__.__name__ = "TestStrategy"
        strategy.analyze = AsyncMock(return_value=None)
        return strategy
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine, config):
        """Test engine initializes correctly."""
        assert engine.config == config
        assert engine.status == BacktestStatus.PENDING
        assert engine.data_provider is None
        assert engine.execution_simulator is None
        assert engine.portfolio is None
        assert engine.event_log == []
        assert engine.tick_count == 0
    
    @pytest.mark.asyncio
    async def test_initialize_components(self, engine):
        """Test component initialization."""
        await engine.initialize()
        
        assert engine.data_provider is not None
        assert engine.execution_simulator is not None
        assert engine.portfolio is not None
        assert engine.portfolio.initial_capital == engine.config.initial_capital
    
    @pytest.mark.asyncio
    async def test_run_backtest_basic(self, engine, mock_strategy):
        """Test basic backtest execution."""
        # Mock data provider
        mock_data = [
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, 0),
                symbol="BTC/USDT",
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41900"),
                close=Decimal("42050"),
                volume=Decimal("100")
            ),
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, 1),
                symbol="BTC/USDT",
                open=Decimal("42050"),
                high=Decimal("42150"),
                low=Decimal("42000"),
                close=Decimal("42100"),
                volume=Decimal("110")
            )
        ]
        
        with patch.object(engine, 'initialize', new=AsyncMock()):
            with patch.object(engine, '_replay_data', new=AsyncMock(return_value=[(d.timestamp, d) for d in mock_data])):
                # Mock portfolio and execution simulator
                engine.portfolio = Mock(spec=Portfolio)
                engine.portfolio.total_equity = engine.config.initial_capital
                engine.portfolio.current_drawdown = Decimal("0")
                engine.portfolio.closed_trades = []
                engine.portfolio.history = []
                engine.portfolio.mark_to_market = AsyncMock()
                engine.portfolio.process_fill = AsyncMock(return_value=True)
                engine.portfolio.get_statistics = AsyncMock(return_value={
                    'max_drawdown': Decimal('0.05'),
                    'sharpe_ratio': 1.2,
                    'sortino_ratio': 1.5,
                    'calmar_ratio': 2.0
                })
                
                engine.execution_simulator = Mock(spec=ExecutionSimulator)
                engine.execution_simulator.simulate_fill = AsyncMock(return_value=None)
                
                engine.data_provider = Mock()
                engine.data_provider.load_data = AsyncMock(
                    return_value=self._async_generator(mock_data)
                )
                
                # Run backtest
                result = await engine.run_backtest(mock_strategy)
                
                assert isinstance(result, BacktestResult)
                assert engine.status == BacktestStatus.COMPLETED
                assert engine.tick_count == 2
                assert mock_strategy.analyze.call_count == 2
    
    @pytest.mark.asyncio
    async def test_drawdown_limit(self, engine, mock_strategy):
        """Test backtest stops when drawdown limit exceeded."""
        await engine.initialize()
        
        # Set high drawdown
        engine.portfolio.current_drawdown = Decimal("0.25")
        engine.portfolio.max_drawdown = Decimal("0.25")
        
        mock_data = [
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, 0),
                symbol="BTC/USDT",
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41900"),
                close=Decimal("42050"),
                volume=Decimal("100")
            )
        ]
        
        with patch.object(engine.data_provider, 'load_data', new=AsyncMock(
            return_value=self._async_generator(mock_data)
        )):
            with patch.object(engine.portfolio, 'get_statistics', new=AsyncMock(return_value={})):
                result = await engine.run_backtest(mock_strategy)
                
                # Should stop early due to drawdown
                assert engine.tick_count == 1
    
    @pytest.mark.asyncio
    async def test_signal_execution(self, engine, mock_strategy):
        """Test signal generation and execution."""
        await engine.initialize()
        
        # Create a buy signal
        signal = Signal(
            timestamp=datetime(2024, 1, 1, 9, 0),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            order_type=OrderType.MARKET
        )
        
        # Create a fill
        fill = Fill(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            price=Decimal("42000"),
            fee=Decimal("4.2"),
            slippage=Decimal("10"),
            status=FillStatus.FILLED,
            order_id="TEST-001",
            value=Decimal("4200")
        )
        
        mock_strategy.analyze = AsyncMock(return_value=signal)
        engine.execution_simulator.simulate_fill = AsyncMock(return_value=fill)
        engine.portfolio.process_fill = AsyncMock(return_value=True)
        
        mock_data = [
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, 0),
                symbol="BTC/USDT",
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41900"),
                close=Decimal("42050"),
                volume=Decimal("100")
            )
        ]
        
        with patch.object(engine.data_provider, 'load_data', new=AsyncMock(
            return_value=self._async_generator(mock_data)
        )):
            with patch.object(engine.portfolio, 'get_statistics', new=AsyncMock(return_value={})):
                result = await engine.run_backtest(mock_strategy)
                
                # Verify signal was processed
                engine.execution_simulator.simulate_fill.assert_called_once()
                engine.portfolio.process_fill.assert_called_once_with(fill)
                assert len(engine.event_log) == 1
                assert engine.event_log[0]['type'] == 'fill'
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, engine, mock_strategy):
        """Test progress callback is called."""
        progress_values = []
        
        async def progress_callback(progress):
            progress_values.append(progress)
        
        # Create 200 data points to trigger progress callback
        mock_data = [
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, i),
                symbol="BTC/USDT",
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41900"),
                close=Decimal("42050"),
                volume=Decimal("100")
            )
            for i in range(200)
        ]
        
        await engine.initialize()
        
        with patch.object(engine.data_provider, 'load_data', new=AsyncMock(
            return_value=self._async_generator(mock_data)
        )):
            with patch.object(engine.portfolio, 'get_statistics', new=AsyncMock(return_value={})):
                result = await engine.run_backtest(mock_strategy, progress_callback)
                
                # Progress should have been reported
                assert len(progress_values) > 0
                assert all(0 <= p <= 100 for p in progress_values)
    
    @pytest.mark.asyncio
    async def test_stop_backtest(self, engine, mock_strategy):
        """Test stopping backtest mid-execution."""
        await engine.initialize()
        
        # Create many data points
        mock_data = [
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, i),
                symbol="BTC/USDT",
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41900"),
                close=Decimal("42050"),
                volume=Decimal("100")
            )
            for i in range(100)
        ]
        
        # Stop after 10 ticks
        async def stop_after_10():
            while engine.tick_count < 10:
                await asyncio.sleep(0.001)
            await engine.stop()
        
        with patch.object(engine.data_provider, 'load_data', new=AsyncMock(
            return_value=self._async_generator(mock_data)
        )):
            with patch.object(engine.portfolio, 'get_statistics', new=AsyncMock(return_value={})):
                # Run stop task concurrently
                stop_task = asyncio.create_task(stop_after_10())
                result = await engine.run_backtest(mock_strategy)
                await stop_task
                
                assert engine.status == BacktestStatus.CANCELLED
                assert engine.tick_count < 100
    
    @pytest.mark.asyncio
    async def test_validate_data_availability(self, engine):
        """Test data availability validation."""
        await engine.initialize()
        
        with patch.object(engine.data_provider, 'check_data_availability', new=AsyncMock(return_value=True)):
            available = await engine.validate_data_availability()
            assert available is True
        
        with patch.object(engine.data_provider, 'check_data_availability', new=AsyncMock(return_value=False)):
            available = await engine.validate_data_availability()
            assert available is False
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, engine, mock_strategy):
        """Test exception handling during backtest."""
        await engine.initialize()
        
        # Make strategy throw an exception
        mock_strategy.analyze = AsyncMock(side_effect=Exception("Strategy error"))
        
        mock_data = [
            MarketSnapshot(
                timestamp=datetime(2024, 1, 1, 9, 0),
                symbol="BTC/USDT",
                open=Decimal("42000"),
                high=Decimal("42100"),
                low=Decimal("41900"),
                close=Decimal("42050"),
                volume=Decimal("100")
            )
        ]
        
        with patch.object(engine.data_provider, 'load_data', new=AsyncMock(
            return_value=self._async_generator(mock_data)
        )):
            with pytest.raises(Exception) as exc_info:
                await engine.run_backtest(mock_strategy)
            
            assert "Strategy error" in str(exc_info.value)
            assert engine.status == BacktestStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_result_generation(self, engine, mock_strategy):
        """Test backtest result generation."""
        await engine.initialize()
        
        # Setup portfolio with trades
        engine.portfolio.total_equity = Decimal("11000")
        engine.portfolio.closed_trades = [
            {
                'symbol': 'BTC/USDT',
                'pnl': 100,
                'entry_time': datetime(2024, 1, 1, 9, 0),
                'exit_time': datetime(2024, 1, 1, 10, 0)
            },
            {
                'symbol': 'ETH/USDT',
                'pnl': -50,
                'entry_time': datetime(2024, 1, 1, 11, 0),
                'exit_time': datetime(2024, 1, 1, 12, 0)
            },
            {
                'symbol': 'BTC/USDT',
                'pnl': 200,
                'entry_time': datetime(2024, 1, 1, 13, 0),
                'exit_time': datetime(2024, 1, 1, 14, 0)
            }
        ]
        
        engine.portfolio.history = [
            {'timestamp': datetime(2024, 1, 1, 9, 0), 'total_equity': 10000},
            {'timestamp': datetime(2024, 1, 1, 10, 0), 'total_equity': 10100},
            {'timestamp': datetime(2024, 1, 1, 11, 0), 'total_equity': 10050},
            {'timestamp': datetime(2024, 1, 1, 12, 0), 'total_equity': 11000}
        ]
        
        with patch.object(engine.portfolio, 'get_statistics', new=AsyncMock(return_value={
            'max_drawdown': Decimal('0.05'),
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8,
            'calmar_ratio': 2.5
        })):
            result = await engine._generate_results(mock_strategy)
            
            assert result.strategy_name == "TestStrategy"
            assert result.initial_capital == engine.config.initial_capital
            assert result.final_capital == Decimal("11000")
            assert result.total_trades == 3
            assert result.winning_trades == 2
            assert result.losing_trades == 1
            assert result.win_rate == pytest.approx(0.667, rel=0.01)
            assert result.max_drawdown == Decimal('0.05')
            assert result.sharpe_ratio == 1.5
    
    async def _async_generator(self, items):
        """Helper to create async generator from list."""
        for item in items:
            yield item


class TestPortfolio:
    """Test suite for Portfolio management."""
    
    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return Portfolio(
            initial_capital=Decimal("10000"),
            enable_shorting=False,
            use_leverage=False
        )
    
    def test_portfolio_initialization(self, portfolio):
        """Test portfolio initializes correctly."""
        assert portfolio.initial_capital == Decimal("10000")
        assert portfolio.cash == Decimal("10000")
        assert portfolio.positions == {}
        assert not portfolio.enable_shorting
        assert not portfolio.use_leverage
        assert portfolio.max_leverage == Decimal("1.0")
    
    def test_total_equity_calculation(self, portfolio):
        """Test total equity calculation."""
        # Add a position
        portfolio.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            current_price=Decimal("42000"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("200")
        )
        
        portfolio.cash = Decimal("5800")  # 10000 - 4200 (cost)
        
        assert portfolio.total_equity == Decimal("10000")  # 5800 + 4200
    
    @pytest.mark.asyncio
    async def test_process_fill_buy(self, portfolio):
        """Test processing buy fill."""
        fill = Fill(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("40000"),
            fee=Decimal("4"),
            slippage=Decimal("10"),
            status=FillStatus.FILLED,
            order_id="TEST-001",
            value=Decimal("4000")
        )
        
        success = await portfolio.process_fill(fill)
        
        assert success is True
        assert portfolio.cash == Decimal("5996")  # 10000 - 4000 - 4
        assert "BTC/USDT" in portfolio.positions
        assert portfolio.positions["BTC/USDT"].quantity == Decimal("0.1")
        assert portfolio.trade_count == 1
        assert portfolio.total_fees == Decimal("4")
    
    @pytest.mark.asyncio
    async def test_process_fill_sell(self, portfolio):
        """Test processing sell fill."""
        # First buy
        portfolio.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            current_price=Decimal("42000"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("200"),
            entry_time=datetime.now()
        )
        portfolio.cash = Decimal("6000")
        
        # Sell fill
        fill = Fill(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            price=Decimal("42000"),
            fee=Decimal("4.2"),
            slippage=Decimal("-10"),
            status=FillStatus.FILLED,
            order_id="TEST-002",
            value=Decimal("4200")
        )
        
        success = await portfolio.process_fill(fill)
        
        assert success is True
        assert portfolio.cash == Decimal("10195.8")  # 6000 + 4200 - 4.2
        assert "BTC/USDT" not in portfolio.positions  # Position closed
        assert len(portfolio.closed_trades) == 1
        assert portfolio.closed_trades[0]['pnl'] == 200  # Profit from trade
    
    @pytest.mark.asyncio
    async def test_mark_to_market(self, portfolio):
        """Test mark-to-market updates."""
        # Add position
        portfolio.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            current_price=Decimal("40000"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0")
        )
        
        # Market snapshot with new price
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            open=Decimal("40000"),
            high=Decimal("42100"),
            low=Decimal("39900"),
            close=Decimal("42000"),
            volume=Decimal("1000")
        )
        
        await portfolio.mark_to_market(snapshot)
        
        position = portfolio.positions["BTC/USDT"]
        assert position.current_price == Decimal("42000")
        assert position.unrealized_pnl == Decimal("200")  # (42000 - 40000) * 0.1
    
    @pytest.mark.asyncio
    async def test_drawdown_tracking(self, portfolio):
        """Test drawdown tracking."""
        # Initial state
        assert portfolio.current_drawdown == Decimal("0")
        assert portfolio.max_drawdown == Decimal("0")
        assert portfolio.peak_equity == Decimal("10000")
        
        # Simulate loss
        portfolio.cash = Decimal("9000")
        portfolio.positions = {}
        
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            symbol="DUMMY",
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=Decimal("100")
        )
        
        await portfolio.mark_to_market(snapshot)
        
        # Drawdown should be 10%
        assert portfolio.current_drawdown == Decimal("0.1")
        assert portfolio.max_drawdown == Decimal("0.1")
    
    @pytest.mark.asyncio
    async def test_statistics_calculation(self, portfolio):
        """Test portfolio statistics calculation."""
        # Add some history
        portfolio.history = [
            {'total_equity': Decimal('10000'), 'timestamp': datetime(2024, 1, 1)},
            {'total_equity': Decimal('10100'), 'timestamp': datetime(2024, 1, 2)},
            {'total_equity': Decimal('10050'), 'timestamp': datetime(2024, 1, 3)},
            {'total_equity': Decimal('10200'), 'timestamp': datetime(2024, 1, 4)}
        ]
        
        portfolio.closed_trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 150}
        ]
        
        portfolio.max_drawdown = Decimal("0.05")
        portfolio.trade_count = 3
        
        stats = await portfolio.get_statistics()
        
        assert stats['initial_capital'] == 10000
        assert stats['trade_count'] == 3
        assert stats['closed_trades'] == 3
        assert stats['max_drawdown'] == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])