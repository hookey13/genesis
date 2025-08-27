"""Unit tests for Strategy Performance Tracker"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from genesis.analytics.strategy_performance import StrategyPerformanceTracker
from genesis.core.models import Trade
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_event_bus():
    """Create mock event bus"""
    event_bus = Mock(spec=EventBus)
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def tracker(mock_event_bus):
    """Create performance tracker instance"""
    return StrategyPerformanceTracker(mock_event_bus)


@pytest.fixture
def sample_trade():
    """Create a sample trade"""
    trade = Mock(spec=Trade)
    trade.pnl_dollars = Decimal("100")
    trade.symbol = "BTC/USDT"
    trade.quantity = Decimal("1")
    return trade


class TestStrategyPerformanceTracker:
    """Test strategy performance tracking functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, tracker):
        """Test tracker initialization"""
        assert tracker.performance_data == {}
        assert tracker.event_bus is not None
    
    @pytest.mark.asyncio
    async def test_start_stop(self, tracker):
        """Test starting and stopping the tracker"""
        await tracker.start()
        # Should log start message
        
        await tracker.stop()
        # Should log stop message
    
    @pytest.mark.asyncio
    async def test_initialize_strategy(self, tracker):
        """Test initializing a new strategy"""
        await tracker.initialize_strategy("momentum")
        
        assert "momentum" in tracker.performance_data
        data = tracker.performance_data["momentum"]
        assert data["total_trades"] == 0
        assert data["winning_trades"] == 0
        assert data["losing_trades"] == 0
        assert data["total_pnl"] == Decimal("0")
        assert data["max_drawdown"] == Decimal("0")
        assert data["sharpe_ratio"] == Decimal("1.0")
        assert data["win_rate"] == Decimal("0")
        assert data["average_win"] == Decimal("0")
        assert data["average_loss"] == Decimal("0")
        assert data["volatility"] == Decimal("0.1")
        assert "started_at" in data
    
    @pytest.mark.asyncio
    async def test_record_winning_trade(self, tracker, sample_trade):
        """Test recording a winning trade"""
        sample_trade.pnl_dollars = Decimal("500")
        
        await tracker.record_trade("momentum", sample_trade)
        
        data = tracker.performance_data["momentum"]
        assert data["total_trades"] == 1
        assert data["winning_trades"] == 1
        assert data["losing_trades"] == 0
        assert data["total_pnl"] == Decimal("500")
        assert data["average_win"] == Decimal("500")
        assert data["win_rate"] == Decimal("1")
    
    @pytest.mark.asyncio
    async def test_record_losing_trade(self, tracker):
        """Test recording a losing trade"""
        trade = Mock(spec=Trade)
        trade.pnl_dollars = Decimal("-200")
        
        await tracker.record_trade("momentum", trade)
        
        data = tracker.performance_data["momentum"]
        assert data["total_trades"] == 1
        assert data["winning_trades"] == 0
        assert data["losing_trades"] == 1
        assert data["total_pnl"] == Decimal("-200")
        assert data["average_loss"] == Decimal("200")
        assert data["win_rate"] == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_record_multiple_trades(self, tracker):
        """Test recording multiple trades"""
        trades = [
            Mock(spec=Trade, pnl_dollars=Decimal("100")),
            Mock(spec=Trade, pnl_dollars=Decimal("200")),
            Mock(spec=Trade, pnl_dollars=Decimal("-50")),
            Mock(spec=Trade, pnl_dollars=Decimal("150")),
            Mock(spec=Trade, pnl_dollars=Decimal("-100"))
        ]
        
        for trade in trades:
            await tracker.record_trade("momentum", trade)
        
        data = tracker.performance_data["momentum"]
        assert data["total_trades"] == 5
        assert data["winning_trades"] == 3
        assert data["losing_trades"] == 2
        assert data["total_pnl"] == Decimal("300")
        assert data["win_rate"] == Decimal("0.6")
    
    @pytest.mark.asyncio
    async def test_average_calculations(self, tracker):
        """Test average win/loss calculations"""
        # Add winning trades
        for pnl in [100, 200, 300]:
            trade = Mock(spec=Trade, pnl_dollars=Decimal(str(pnl)))
            await tracker.record_trade("momentum", trade)
        
        # Add losing trades
        for pnl in [-50, -150]:
            trade = Mock(spec=Trade, pnl_dollars=Decimal(str(pnl)))
            await tracker.record_trade("momentum", trade)
        
        data = tracker.performance_data["momentum"]
        assert data["average_win"] == Decimal("200")  # (100+200+300)/3
        assert data["average_loss"] == Decimal("100")  # (50+150)/2
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio_calculation(self, tracker):
        """Test Sharpe ratio calculation"""
        await tracker.initialize_strategy("momentum")
        tracker.performance_data["momentum"]["volatility"] = Decimal("0.2")
        
        trade = Mock(spec=Trade, pnl_dollars=Decimal("1000"))
        await tracker.record_trade("momentum", trade)
        
        data = tracker.performance_data["momentum"]
        # Sharpe ratio = total_pnl / (volatility * 100)
        expected_sharpe = Decimal("1000") / (Decimal("0.2") * Decimal("100"))
        assert data["sharpe_ratio"] == expected_sharpe
    
    @pytest.mark.asyncio
    async def test_get_strategy_performance(self, tracker):
        """Test getting strategy performance"""
        await tracker.initialize_strategy("momentum")
        
        performance = await tracker.get_strategy_performance("momentum")
        assert performance is not None
        assert performance["total_trades"] == 0
        
        # Non-existent strategy
        performance = await tracker.get_strategy_performance("nonexistent")
        assert performance is None
    
    @pytest.mark.asyncio
    async def test_get_portfolio_summary(self, tracker):
        """Test getting portfolio-wide summary"""
        # Initialize multiple strategies
        await tracker.initialize_strategy("momentum")
        await tracker.initialize_strategy("mean_reversion")
        
        # Add trades
        trade1 = Mock(spec=Trade, pnl_dollars=Decimal("500"))
        trade2 = Mock(spec=Trade, pnl_dollars=Decimal("300"))
        trade3 = Mock(spec=Trade, pnl_dollars=Decimal("-200"))
        
        await tracker.record_trade("momentum", trade1)
        await tracker.record_trade("momentum", trade2)
        await tracker.record_trade("mean_reversion", trade3)
        
        summary = await tracker.get_portfolio_summary()
        
        assert summary["total_pnl"] == "600"  # 500 + 300 - 200
        assert summary["total_trades"] == 3
        assert summary["num_strategies"] == 2
    
    @pytest.mark.asyncio
    async def test_auto_initialize_on_record(self, tracker):
        """Test automatic strategy initialization on first trade"""
        trade = Mock(spec=Trade, pnl_dollars=Decimal("100"))
        
        # Record trade for non-initialized strategy
        await tracker.record_trade("new_strategy", trade)
        
        # Should auto-initialize
        assert "new_strategy" in tracker.performance_data
        assert tracker.performance_data["new_strategy"]["total_trades"] == 1
    
    @pytest.mark.asyncio
    async def test_win_rate_edge_cases(self, tracker):
        """Test win rate calculation edge cases"""
        await tracker.initialize_strategy("momentum")
        
        # Zero trades
        data = tracker.performance_data["momentum"]
        assert data["win_rate"] == Decimal("0")
        
        # All winning trades
        for _ in range(3):
            trade = Mock(spec=Trade, pnl_dollars=Decimal("100"))
            await tracker.record_trade("momentum", trade)
        assert data["win_rate"] == Decimal("1")
        
        # Mix of trades
        trade = Mock(spec=Trade, pnl_dollars=Decimal("-50"))
        await tracker.record_trade("momentum", trade)
        assert data["win_rate"] == Decimal("0.75")  # 3/4
    
    @pytest.mark.asyncio
    async def test_zero_volatility_sharpe(self, tracker):
        """Test Sharpe ratio with zero volatility"""
        await tracker.initialize_strategy("momentum")
        tracker.performance_data["momentum"]["volatility"] = Decimal("0")
        
        trade = Mock(spec=Trade, pnl_dollars=Decimal("100"))
        await tracker.record_trade("momentum", trade)
        
        # Should not crash, Sharpe remains at initial value
        data = tracker.performance_data["momentum"]
        assert data["sharpe_ratio"] == Decimal("1.0")  # Initial value maintained