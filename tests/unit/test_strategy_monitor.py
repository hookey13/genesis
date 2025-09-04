"""Unit tests for strategy performance monitoring."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from genesis.core.models import Order, Position, Trade, OrderSide, OrderStatus, PositionSide
from genesis.monitoring.strategy_monitor import (
    StrategyMetrics,
    StrategyPerformanceMonitor
)
from genesis.monitoring.performance_attribution import PerformanceAttributor


@pytest.fixture
def mock_trade():
    """Create a mock trade."""
    return Trade(
        trade_id="trade1",
        order_id="order1",
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        entry_price=Decimal("49900"),
        exit_price=Decimal("50000"),
        quantity=Decimal("0.1"),
        pnl_dollars=Decimal("100"),
        pnl_percent=Decimal("0.2"),
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def mock_position():
    """Create a mock position."""
    return Position(
        position_id="pos1",
        account_id="account1",
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        entry_price=Decimal("49000"),
        current_price=Decimal("50000"),
        quantity=Decimal("0.5"),
        dollar_value=Decimal("25000"),
        pnl_dollars=Decimal("500"),
        pnl_percent=Decimal("2.04")
    )


@pytest.fixture
def strategy_monitor():
    """Create a strategy performance monitor."""
    return StrategyPerformanceMonitor()


class TestStrategyMetrics:
    """Test StrategyMetrics class."""
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        metrics = StrategyMetrics(strategy_id="test")
        metrics.total_trades = 10
        metrics.winning_trades = 7
        
        assert metrics.win_rate == Decimal("70")
        
    def test_win_rate_no_trades(self):
        """Test win rate with no trades."""
        metrics = StrategyMetrics(strategy_id="test")
        
        assert metrics.win_rate == Decimal("0")
        
    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        metrics = StrategyMetrics(strategy_id="test")
        metrics.gross_profit = Decimal("1000")
        metrics.gross_loss = Decimal("-500")
        
        assert metrics.profit_factor == Decimal("2")
        
    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        metrics = StrategyMetrics(strategy_id="test")
        metrics.gross_profit = Decimal("1000")
        metrics.gross_loss = Decimal("0")
        
        assert metrics.profit_factor == Decimal("999")
        
    def test_average_win_calculation(self):
        """Test average win calculation."""
        metrics = StrategyMetrics(strategy_id="test")
        metrics.gross_profit = Decimal("1000")
        metrics.winning_trades = 5
        
        assert metrics.average_win == Decimal("200")
        
    def test_average_loss_calculation(self):
        """Test average loss calculation."""
        metrics = StrategyMetrics(strategy_id="test")
        metrics.gross_loss = Decimal("-600")
        metrics.losing_trades = 3
        
        assert metrics.average_loss == Decimal("200")
        
    def test_expectancy_calculation(self):
        """Test expectancy calculation."""
        metrics = StrategyMetrics(strategy_id="test")
        metrics.total_pnl = Decimal("500")
        metrics.total_trades = 10
        
        assert metrics.expectancy == Decimal("50")
        
    def test_update_drawdown(self):
        """Test drawdown update logic."""
        metrics = StrategyMetrics(strategy_id="test")
        
        # Initial state
        metrics.total_pnl = Decimal("100")
        metrics.update_drawdown()
        assert metrics.peak_pnl == Decimal("100")
        assert metrics.current_drawdown == Decimal("0")
        
        # Drawdown occurs
        metrics.total_pnl = Decimal("70")
        metrics.update_drawdown()
        assert metrics.peak_pnl == Decimal("100")
        assert metrics.current_drawdown == Decimal("30")
        assert metrics.max_drawdown == Decimal("30")
        
        # Recovery
        metrics.total_pnl = Decimal("120")
        metrics.update_drawdown()
        assert metrics.peak_pnl == Decimal("120")
        assert metrics.current_drawdown == Decimal("0")
        assert metrics.max_drawdown == Decimal("30")  # Max drawdown persists


class TestStrategyPerformanceMonitor:
    """Test StrategyPerformanceMonitor class."""
    
    @pytest.mark.asyncio
    async def test_register_strategy(self, strategy_monitor):
        """Test strategy registration."""
        await strategy_monitor.register_strategy("test_strategy")
        
        assert "test_strategy" in strategy_monitor.strategies
        assert isinstance(strategy_monitor.strategies["test_strategy"], StrategyMetrics)
        
    @pytest.mark.asyncio
    async def test_unregister_strategy(self, strategy_monitor):
        """Test strategy unregistration."""
        await strategy_monitor.register_strategy("test_strategy")
        await strategy_monitor.unregister_strategy("test_strategy")
        
        assert "test_strategy" not in strategy_monitor.strategies
        
    @pytest.mark.asyncio
    async def test_update_position(self, strategy_monitor, mock_position):
        """Test position update."""
        await strategy_monitor.update_position("test_strategy", mock_position)
        
        metrics = strategy_monitor.strategies["test_strategy"]
        assert mock_position.symbol in metrics.positions
        assert metrics.positions[mock_position.symbol] == mock_position
        
    @pytest.mark.asyncio
    async def test_record_trade_winning(self, strategy_monitor, mock_trade):
        """Test recording a winning trade."""
        await strategy_monitor.record_trade("test_strategy", mock_trade)
        
        metrics = strategy_monitor.strategies["test_strategy"]
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.realized_pnl == Decimal("100")
        assert metrics.gross_profit == Decimal("100")
        assert mock_trade in metrics.recent_trades
        
    @pytest.mark.asyncio
    async def test_record_trade_losing(self, strategy_monitor):
        """Test recording a losing trade."""
        losing_trade = Trade(
            trade_id="trade2",
            order_id="order2",
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            entry_price=Decimal("50000"),
            exit_price=Decimal("49500"),
            quantity=Decimal("0.1"),
            pnl_dollars=Decimal("-50"),
            pnl_percent=Decimal("-1"),
            timestamp=datetime.utcnow()
        )
        
        await strategy_monitor.record_trade("test_strategy", losing_trade)
        
        metrics = strategy_monitor.strategies["test_strategy"]
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 1
        assert metrics.realized_pnl == Decimal("-50")
        assert metrics.gross_loss == Decimal("-50")
        
    @pytest.mark.asyncio
    async def test_record_trade_with_slippage(self, strategy_monitor, mock_trade):
        """Test slippage calculation."""
        expected_price = Decimal("49900")
        await strategy_monitor.record_trade("test_strategy", mock_trade, expected_price)
        
        metrics = strategy_monitor.strategies["test_strategy"]
        # Use exit_price for slippage calculation since that's the actual executed price
        slippage = abs(mock_trade.exit_price - expected_price) * mock_trade.quantity
        assert metrics.slippage_total == slippage
        
    @pytest.mark.asyncio
    async def test_recent_trades_limit(self, strategy_monitor):
        """Test that recent trades list is limited to 100."""
        # Create 150 trades
        for i in range(150):
            trade = Trade(
                trade_id=f"trade{i}",
                order_id=f"order{i}",
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("49900"),
                exit_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                pnl_dollars=Decimal("10"),
                pnl_percent=Decimal("0.02"),
                timestamp=datetime.utcnow()
            )
            await strategy_monitor.record_trade("test_strategy", trade)
            
        metrics = strategy_monitor.strategies["test_strategy"]
        assert len(metrics.recent_trades) == 100
        
    @pytest.mark.asyncio
    async def test_update_price(self, strategy_monitor):
        """Test price update."""
        await strategy_monitor.update_price("BTC/USDT", Decimal("51000"))
        
        assert strategy_monitor._price_cache["BTC/USDT"] == Decimal("51000")
        
    @pytest.mark.asyncio
    async def test_get_strategy_metrics(self, strategy_monitor):
        """Test getting strategy metrics."""
        await strategy_monitor.register_strategy("test_strategy")
        
        metrics = await strategy_monitor.get_strategy_metrics("test_strategy")
        assert metrics is not None
        assert metrics.strategy_id == "test_strategy"
        
        # Non-existent strategy
        metrics = await strategy_monitor.get_strategy_metrics("non_existent")
        assert metrics is None
        
    @pytest.mark.asyncio
    async def test_get_all_metrics(self, strategy_monitor):
        """Test getting all strategy metrics."""
        await strategy_monitor.register_strategy("strategy1")
        await strategy_monitor.register_strategy("strategy2")
        
        all_metrics = await strategy_monitor.get_all_metrics()
        assert len(all_metrics) == 2
        assert "strategy1" in all_metrics
        assert "strategy2" in all_metrics
        
    @pytest.mark.asyncio
    async def test_get_performance_summary(self, strategy_monitor):
        """Test performance summary generation."""
        # Register strategies and add trades
        await strategy_monitor.register_strategy("strategy1")
        await strategy_monitor.register_strategy("strategy2")
        
        trade1 = Trade(
            trade_id="trade1",
            order_id="order1",
            strategy_id="strategy1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("49900"),
            exit_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            pnl_dollars=Decimal("100"),
            pnl_percent=Decimal("0.2"),
            timestamp=datetime.utcnow()
        )
        
        trade2 = Trade(
            trade_id="trade2",
            order_id="order2",
            strategy_id="strategy2",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            entry_price=Decimal("3000"),
            exit_price=Decimal("2950"),
            quantity=Decimal("1"),
            pnl_dollars=Decimal("-50"),
            pnl_percent=Decimal("-1.67"),
            timestamp=datetime.utcnow()
        )
        
        await strategy_monitor.record_trade("strategy1", trade1)
        await strategy_monitor.record_trade("strategy2", trade2)
        
        summary = await strategy_monitor.get_performance_summary()
        
        assert summary["total_pnl"] == Decimal("50")
        assert summary["active_strategies"] == 2
        assert summary["total_trades"] == 2
        assert summary["overall_win_rate"] == Decimal("50")
        
    @pytest.mark.asyncio
    async def test_estimate_strategy_capacity(self, strategy_monitor, mock_position):
        """Test strategy capacity estimation."""
        await strategy_monitor.register_strategy("test_strategy")
        await strategy_monitor.update_position("test_strategy", mock_position)
        
        market_depth = {"BTC/USDT": Decimal("1000000")}
        capacity = await strategy_monitor.estimate_strategy_capacity("test_strategy", market_depth)
        
        # Capacity should be 10% of market depth
        assert capacity == Decimal("100000")
        
    @pytest.mark.asyncio
    async def test_check_rotation_criteria_drawdown(self, strategy_monitor):
        """Test rotation criteria for max drawdown."""
        await strategy_monitor.register_strategy("test_strategy")
        
        metrics = strategy_monitor.strategies["test_strategy"]
        metrics.max_drawdown = Decimal("25")  # Exceeds 20% threshold
        
        rotation_candidates = await strategy_monitor.check_rotation_criteria()
        
        assert len(rotation_candidates) == 1
        assert rotation_candidates[0] == ("test_strategy", "max_drawdown_exceeded")
        
    @pytest.mark.asyncio
    async def test_check_rotation_criteria_win_rate(self, strategy_monitor):
        """Test rotation criteria for low win rate."""
        await strategy_monitor.register_strategy("test_strategy")
        
        metrics = strategy_monitor.strategies["test_strategy"]
        metrics.total_trades = 25
        metrics.winning_trades = 5  # 20% win rate
        # Set profit factor to avoid triggering that rule
        metrics.gross_profit = Decimal("100")
        metrics.gross_loss = Decimal("-100")  # Profit factor = 1.0
        
        rotation_candidates = await strategy_monitor.check_rotation_criteria()
        
        assert len(rotation_candidates) == 1
        assert rotation_candidates[0] == ("test_strategy", "low_win_rate")
        
    @pytest.mark.asyncio
    async def test_check_rotation_criteria_profit_factor(self, strategy_monitor):
        """Test rotation criteria for poor profit factor."""
        await strategy_monitor.register_strategy("test_strategy")
        
        metrics = strategy_monitor.strategies["test_strategy"]
        metrics.total_trades = 15
        metrics.gross_profit = Decimal("100")
        metrics.gross_loss = Decimal("-200")  # Profit factor = 0.5
        
        rotation_candidates = await strategy_monitor.check_rotation_criteria()
        
        assert len(rotation_candidates) == 1
        assert rotation_candidates[0] == ("test_strategy", "poor_profit_factor")
        
    @pytest.mark.asyncio
    async def test_check_rotation_criteria_extended_drawdown(self, strategy_monitor):
        """Test rotation criteria for extended drawdown."""
        await strategy_monitor.register_strategy("test_strategy")
        
        metrics = strategy_monitor.strategies["test_strategy"]
        metrics.drawdown_start = datetime.utcnow() - timedelta(days=8)
        
        rotation_candidates = await strategy_monitor.check_rotation_criteria()
        
        assert len(rotation_candidates) == 1
        assert rotation_candidates[0] == ("test_strategy", "extended_drawdown")
        
    @pytest.mark.asyncio
    async def test_set_rotation_rule(self, strategy_monitor):
        """Test setting custom rotation rules."""
        criteria = {"max_consecutive_losses": 5, "min_sharpe_ratio": 1.0}
        strategy_monitor.set_rotation_rule("custom_rule", criteria)
        
        assert "custom_rule" in strategy_monitor.rotation_rules
        assert strategy_monitor.rotation_rules["custom_rule"] == criteria
        
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, strategy_monitor):
        """Test monitoring start and stop."""
        await strategy_monitor.start_monitoring()
        assert strategy_monitor._monitoring_active
        assert strategy_monitor.monitoring_task is not None
        
        await strategy_monitor.stop_monitoring()
        assert not strategy_monitor._monitoring_active
        
    @pytest.mark.asyncio
    async def test_get_attribution_analysis(self, strategy_monitor):
        """Test attribution analysis retrieval."""
        await strategy_monitor.register_strategy("test_strategy")
        
        # Add some trades
        for i in range(5):
            trade = Trade(
                trade_id=f"trade{i}",
                order_id=f"order{i}",
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("49900"),
                exit_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                pnl_dollars=Decimal("100"),
                pnl_percent=Decimal("0.2"),
                timestamp=datetime.utcnow()
            )
            await strategy_monitor.record_trade("test_strategy", trade)
            
        analysis = await strategy_monitor.get_attribution_analysis("test_strategy", period_hours=24)
        
        assert analysis["strategy_id"] == "test_strategy"
        assert analysis["trade_count"] == 5
        assert "total_return" in analysis
        assert "average_return" in analysis
        
    @pytest.mark.asyncio
    async def test_calculate_unrealized_pnl_long(self, strategy_monitor):
        """Test unrealized P&L calculation for long positions."""
        await strategy_monitor.register_strategy("test_strategy")
        
        position = Position(
            position_id="pos1",
            account_id="account1",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            current_price=Decimal("50000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("25000"),
            pnl_dollars=Decimal("0"),
            pnl_percent=Decimal("0")
        )
        
        await strategy_monitor.update_position("test_strategy", position)
        await strategy_monitor.update_price("BTC/USDT", Decimal("51000"))
        
        metrics = strategy_monitor.strategies["test_strategy"]
        await strategy_monitor._calculate_unrealized_pnl(metrics)
        
        # (51000 - 49000) * 0.5 = 1000
        assert metrics.unrealized_pnl == Decimal("1000")
        
    @pytest.mark.asyncio
    async def test_calculate_unrealized_pnl_short(self, strategy_monitor):
        """Test unrealized P&L calculation for short positions."""
        await strategy_monitor.register_strategy("test_strategy")
        
        position = Position(
            position_id="pos1",
            account_id="account1",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("51000"),
            current_price=Decimal("50000"),
            quantity=Decimal("0.5"),  # Short position uses positive quantity
            dollar_value=Decimal("25000"),
            pnl_dollars=Decimal("0"),
            pnl_percent=Decimal("0")
        )
        
        await strategy_monitor.update_position("test_strategy", position)
        await strategy_monitor.update_price("BTC/USDT", Decimal("50000"))
        
        metrics = strategy_monitor.strategies["test_strategy"]
        await strategy_monitor._calculate_unrealized_pnl(metrics)
        
        # (51000 - 50000) * 0.5 = 500
        assert metrics.unrealized_pnl == Decimal("500")