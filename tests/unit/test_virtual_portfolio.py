"""Unit tests for virtual portfolio module."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from genesis.paper_trading.virtual_portfolio import (
    Position,
    Trade,
    VirtualPortfolio,
)


class TestPosition:
    """Test Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            side="long",
            opened_at=datetime.now(),
        )
        
        assert position.symbol == "BTC/USDT"
        assert position.quantity == Decimal("0.5")
        assert position.entry_price == Decimal("50000")
        assert position.current_price == Decimal("51000")
        assert position.side == "long"

    def test_position_pnl_calculation(self):
        """Test P&L calculation for positions."""
        # Long position with profit
        long_pos = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            side="long",
            opened_at=datetime.now(),
        )
        assert long_pos.unrealized_pnl == Decimal("1000")
        assert long_pos.unrealized_pnl_percent == Decimal("2.0")
        
        # Long position with loss
        long_loss = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
            side="long",
            opened_at=datetime.now(),
        )
        assert long_loss.unrealized_pnl == Decimal("-1000")
        assert long_loss.unrealized_pnl_percent == Decimal("-2.0")
        
        # Short position with profit
        short_pos = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("49000"),
            side="short",
            opened_at=datetime.now(),
        )
        assert short_pos.unrealized_pnl == Decimal("1000")
        assert short_pos.unrealized_pnl_percent == Decimal("2.0")

    def test_position_value_calculation(self):
        """Test position value calculation."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            side="long",
            opened_at=datetime.now(),
        )
        
        assert position.current_value == Decimal("102000")
        assert position.entry_value == Decimal("100000")


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            trade_id="trade123",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
            commission=Decimal("50"),
            pnl=Decimal("1000"),
        )
        
        assert trade.trade_id == "trade123"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.quantity == Decimal("1.0")
        assert trade.price == Decimal("50000")
        assert trade.commission == Decimal("50")
        assert trade.pnl == Decimal("1000")

    def test_trade_total_value(self):
        """Test trade total value calculation."""
        trade = Trade(
            trade_id="trade123",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
            commission=Decimal("100"),
        )
        
        assert trade.total_value == Decimal("100000")
        assert trade.net_value == Decimal("100100")  # Including commission


class TestVirtualPortfolio:
    """Test VirtualPortfolio class."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return VirtualPortfolio(
            portfolio_id="test_portfolio",
            initial_balance=Decimal("10000"),
        )

    def test_initialization(self, portfolio):
        """Test portfolio initialization."""
        assert portfolio.portfolio_id == "test_portfolio"
        assert portfolio.initial_balance == Decimal("10000")
        assert portfolio.balance == Decimal("10000")
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert portfolio.total_trades == 0

    def test_open_position(self, portfolio):
        """Test opening a position."""
        position = portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        assert position.symbol == "BTC/USDT"
        assert position.quantity == Decimal("0.1")
        assert position.entry_price == Decimal("50000")
        assert "BTC/USDT" in portfolio.positions
        assert portfolio.positions["BTC/USDT"] == position
        
        # Balance should be reduced by position value
        expected_balance = Decimal("10000") - (Decimal("0.1") * Decimal("50000"))
        assert portfolio.balance == expected_balance

    def test_open_position_insufficient_balance(self, portfolio):
        """Test opening position with insufficient balance."""
        with pytest.raises(ValueError, match="Insufficient balance"):
            portfolio.open_position(
                symbol="BTC/USDT",
                side="long",
                quantity=Decimal("1.0"),
                price=Decimal("50000"),  # 50k > 10k balance
            )

    def test_close_position(self, portfolio):
        """Test closing a position."""
        # Open a position first
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        # Close with profit
        trade = portfolio.close_position(
            symbol="BTC/USDT",
            price=Decimal("51000"),
        )
        
        assert trade.symbol == "BTC/USDT"
        assert trade.quantity == Decimal("0.1")
        assert trade.price == Decimal("51000")
        assert trade.pnl == Decimal("100")  # 0.1 * (51000 - 50000)
        assert "BTC/USDT" not in portfolio.positions
        assert len(portfolio.trades) == 1
        
        # Balance should reflect profit
        expected_balance = Decimal("10000") + Decimal("100")
        assert portfolio.balance == expected_balance

    def test_close_nonexistent_position(self, portfolio):
        """Test closing a position that doesn't exist."""
        with pytest.raises(ValueError, match="No position found"):
            portfolio.close_position("BTC/USDT", Decimal("50000"))

    def test_update_position_price(self, portfolio):
        """Test updating position price."""
        # Open position
        position = portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        # Update price
        portfolio.update_position_price("BTC/USDT", Decimal("51000"))
        
        assert position.current_price == Decimal("51000")
        assert position.unrealized_pnl == Decimal("100")

    def test_add_to_position(self, portfolio):
        """Test adding to an existing position."""
        # Open initial position
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        # Add to position
        portfolio.add_to_position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
        )
        
        position = portfolio.positions["BTC/USDT"]
        assert position.quantity == Decimal("0.2")
        # Average entry price should be (0.1*50000 + 0.1*51000) / 0.2 = 50500
        assert position.entry_price == Decimal("50500")

    def test_reduce_position(self, portfolio):
        """Test reducing a position."""
        # Open position
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.2"),
            price=Decimal("50000"),
        )
        
        # Reduce position
        trade = portfolio.reduce_position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
        )
        
        position = portfolio.positions["BTC/USDT"]
        assert position.quantity == Decimal("0.1")
        assert trade.pnl == Decimal("100")  # 0.1 * (51000 - 50000)

    def test_record_trade(self, portfolio):
        """Test recording a trade."""
        trade = Trade(
            trade_id="test123",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
            pnl=Decimal("100"),
        )
        
        portfolio.record_trade(trade)
        
        assert len(portfolio.trades) == 1
        assert portfolio.trades[0] == trade
        assert portfolio.total_trades == 1

    def test_get_metrics(self, portfolio):
        """Test getting portfolio metrics."""
        # Make some trades
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        portfolio.close_position("BTC/USDT", Decimal("51000"))  # Profit
        
        portfolio.open_position(
            symbol="ETH/USDT",
            side="long",
            quantity=Decimal("1.0"),
            price=Decimal("3000"),
        )
        portfolio.close_position("ETH/USDT", Decimal("2900"))  # Loss
        
        metrics = portfolio.get_metrics()
        
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_pnl" in metrics
        assert "avg_win" in metrics
        assert "avg_loss" in metrics
        
        assert metrics["total_trades"] == 2
        assert metrics["win_rate"] == Decimal("50")  # 1 win, 1 loss

    def test_calculate_sharpe_ratio(self, portfolio):
        """Test Sharpe ratio calculation."""
        # Add daily returns
        portfolio.daily_returns.extend([
            Decimal("0.01"),
            Decimal("-0.005"),
            Decimal("0.02"),
            Decimal("0.015"),
            Decimal("-0.01"),
        ])
        
        sharpe = portfolio.calculate_sharpe_ratio()
        assert sharpe is not None
        assert isinstance(sharpe, Decimal)

    def test_calculate_max_drawdown(self, portfolio):
        """Test maximum drawdown calculation."""
        # Simulate equity curve
        portfolio.equity_curve = [
            Decimal("10000"),
            Decimal("11000"),
            Decimal("10500"),
            Decimal("9500"),
            Decimal("10000"),
        ]
        
        drawdown = portfolio.calculate_max_drawdown()
        # Max drawdown from 11000 to 9500 = 1500/11000 = 0.1364
        assert drawdown == pytest.approx(Decimal("0.1364"), rel=0.01)

    def test_calculate_profit_factor(self, portfolio):
        """Test profit factor calculation."""
        # Add winning and losing trades
        portfolio.trades = [
            Trade("1", "BTC/USDT", "sell", Decimal("1"), Decimal("50000"),
                  datetime.now(), Decimal("0"), Decimal("1000")),  # Win
            Trade("2", "ETH/USDT", "sell", Decimal("1"), Decimal("3000"),
                  datetime.now(), Decimal("0"), Decimal("500")),   # Win
            Trade("3", "BNB/USDT", "sell", Decimal("1"), Decimal("300"),
                  datetime.now(), Decimal("0"), Decimal("-300")),  # Loss
            Trade("4", "SOL/USDT", "sell", Decimal("1"), Decimal("100"),
                  datetime.now(), Decimal("0"), Decimal("-200")),  # Loss
        ]
        
        profit_factor = portfolio.calculate_profit_factor()
        # Total wins: 1500, Total losses: 500
        # Profit factor = 1500 / 500 = 3.0
        assert profit_factor == Decimal("3.0")

    def test_reset(self, portfolio):
        """Test resetting portfolio."""
        # Make some trades and positions
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        portfolio.close_position("BTC/USDT", Decimal("51000"))
        
        # Reset
        portfolio.reset()
        
        assert portfolio.balance == portfolio.initial_balance
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert portfolio.total_trades == 0
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0] == portfolio.initial_balance

    def test_get_total_value(self, portfolio):
        """Test getting total portfolio value."""
        # Open some positions
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        portfolio.update_position_price("BTC/USDT", Decimal("51000"))
        
        portfolio.open_position(
            symbol="ETH/USDT",
            side="long",
            quantity=Decimal("1.0"),
            price=Decimal("3000"),
        )
        portfolio.update_position_price("ETH/USDT", Decimal("3100"))
        
        total_value = portfolio.get_total_value()
        # Balance: 10000 - 5000 - 3000 = 2000
        # BTC position: 0.1 * 51000 = 5100
        # ETH position: 1.0 * 3100 = 3100
        # Total: 2000 + 5100 + 3100 = 10200
        assert total_value == Decimal("10200")

    def test_get_position_summary(self, portfolio):
        """Test getting position summary."""
        # Open positions
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        portfolio.update_position_price("BTC/USDT", Decimal("51000"))
        
        summary = portfolio.get_position_summary()
        
        assert len(summary) == 1
        assert summary[0]["symbol"] == "BTC/USDT"
        assert summary[0]["quantity"] == Decimal("0.1")
        assert summary[0]["entry_price"] == Decimal("50000")
        assert summary[0]["current_price"] == Decimal("51000")
        assert summary[0]["unrealized_pnl"] == Decimal("100")

    def test_equity_curve_tracking(self, portfolio):
        """Test equity curve tracking."""
        initial_equity = portfolio.equity_curve[-1]
        
        # Make profitable trade
        portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        portfolio.close_position("BTC/USDT", Decimal("51000"))
        
        portfolio.update_equity_curve()
        
        assert len(portfolio.equity_curve) > 1
        assert portfolio.equity_curve[-1] > initial_equity

    def test_commission_handling(self, portfolio):
        """Test commission handling in trades."""
        portfolio.commission_rate = Decimal("0.001")  # 0.1%
        
        # Open position with commission
        position = portfolio.open_position(
            symbol="BTC/USDT",
            side="long",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        # Commission should be deducted from balance
        position_value = Decimal("0.1") * Decimal("50000")
        commission = position_value * Decimal("0.001")
        expected_balance = Decimal("10000") - position_value - commission
        assert portfolio.balance == expected_balance

    def test_multiple_positions(self, portfolio):
        """Test managing multiple positions."""
        # Open multiple positions
        portfolio.open_position("BTC/USDT", "long", Decimal("0.1"), Decimal("50000"))
        portfolio.open_position("ETH/USDT", "long", Decimal("1.0"), Decimal("3000"))
        portfolio.open_position("BNB/USDT", "short", Decimal("10"), Decimal("300"))
        
        assert len(portfolio.positions) == 3
        assert "BTC/USDT" in portfolio.positions
        assert "ETH/USDT" in portfolio.positions
        assert "BNB/USDT" in portfolio.positions
        
        # Update prices
        portfolio.update_position_price("BTC/USDT", Decimal("51000"))
        portfolio.update_position_price("ETH/USDT", Decimal("3100"))
        portfolio.update_position_price("BNB/USDT", Decimal("290"))
        
        # Check P&L
        btc_pnl = portfolio.positions["BTC/USDT"].unrealized_pnl
        eth_pnl = portfolio.positions["ETH/USDT"].unrealized_pnl
        bnb_pnl = portfolio.positions["BNB/USDT"].unrealized_pnl
        
        assert btc_pnl == Decimal("100")  # Long profit
        assert eth_pnl == Decimal("100")  # Long profit
        assert bnb_pnl == Decimal("100")  # Short profit

    def test_win_rate_calculation(self, portfolio):
        """Test win rate calculation."""
        # Add trades
        portfolio.trades = [
            Trade("1", "BTC/USDT", "sell", Decimal("1"), Decimal("50000"),
                  datetime.now(), Decimal("0"), Decimal("1000")),  # Win
            Trade("2", "ETH/USDT", "sell", Decimal("1"), Decimal("3000"),
                  datetime.now(), Decimal("0"), Decimal("-500")),  # Loss
            Trade("3", "BNB/USDT", "sell", Decimal("1"), Decimal("300"),
                  datetime.now(), Decimal("0"), Decimal("100")),   # Win
        ]
        
        metrics = portfolio.get_metrics()
        # 2 wins out of 3 trades = 66.67%
        assert metrics["win_rate"] == pytest.approx(Decimal("66.67"), rel=0.01)

    def test_daily_returns_tracking(self, portfolio):
        """Test daily returns tracking."""
        # Simulate daily P&L
        starting_balance = portfolio.balance
        
        # Day 1: Profit
        portfolio.balance = starting_balance * Decimal("1.02")
        portfolio.record_daily_return()
        
        # Day 2: Loss
        portfolio.balance = starting_balance * Decimal("1.01")
        portfolio.record_daily_return()
        
        assert len(portfolio.daily_returns) >= 2
        assert portfolio.daily_returns[-2] == Decimal("0.02")
        assert portfolio.daily_returns[-1] == pytest.approx(Decimal("-0.0098"), rel=0.01)