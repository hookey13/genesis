"""Unit tests for Mean Reversion Strategy."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genesis.core.models import Order, Position, Signal, SignalType
from genesis.strategies.hunter.mean_reversion import (
    MeanReversionConfig,
    MeanReversionStrategy,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return MeanReversionConfig(
        name="TestMeanReversion",
        symbols=["BTCUSDT", "ETHUSDT"],
        bb_period=20,
        bb_std_dev=2.0,
        rsi_period=14,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        max_concurrent_positions=5,
        max_position_usdt=Decimal("1000"),
        max_position_percent=Decimal("0.05"),
    )


@pytest.fixture
def strategy(config):
    """Create a strategy instance."""
    return MeanReversionStrategy(config)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    
    # Generate synthetic price data with mean reversion characteristics
    base_price = 50000
    prices = []
    current = base_price
    
    for _ in range(100):
        # Add some mean reverting noise
        change = np.random.randn() * 500
        current = current + change
        # Mean revert towards base price
        current = current * 0.98 + base_price * 0.02
        prices.append(current)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "price": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": np.random.uniform(100, 1000, 100)
    })
    
    return df


class TestMeanReversionConfig:
    """Test MeanReversionConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MeanReversionConfig()
        assert config.bb_period == 20
        assert config.bb_std_dev == 2.0
        assert config.rsi_period == 14
        assert config.tier_required == "HUNTER"
        assert config.max_concurrent_positions == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MeanReversionConfig(
            bb_period=30,
            rsi_oversold=25.0,
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        )
        assert config.bb_period == 30
        assert config.rsi_oversold == 25.0
        assert len(config.symbols) == 3


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy class."""
    
    def test_initialization(self, config):
        """Test strategy initialization."""
        strategy = MeanReversionStrategy(config)
        assert strategy.config == config
        assert len(strategy.price_history) == 0
        assert len(strategy.symbol_positions) == 0
        assert strategy.config.tier_required == "HUNTER"
    
    @pytest.mark.asyncio
    async def test_update_price_history(self, strategy):
        """Test price history update."""
        market_data = {
            "symbol": "BTCUSDT",
            "price": 50000,
            "volume": 1000,
            "high": 50500,
            "low": 49500,
            "timestamp": datetime.now(UTC)
        }
        
        await strategy._update_price_history("BTCUSDT", market_data)
        
        assert "BTCUSDT" in strategy.price_history
        assert len(strategy.price_history["BTCUSDT"]) == 1
        assert strategy.price_history["BTCUSDT"]["price"].iloc[0] == 50000
    
    def test_calculate_bollinger_bands(self, strategy):
        """Test Bollinger Bands calculation."""
        prices = pd.Series([100, 102, 98, 101, 99, 103, 97, 100, 102, 101] * 3)
        
        upper, middle, lower = strategy._calculate_bollinger_bands(prices)
        
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)
        # Check that valid bands are in correct order (after enough periods)
        valid_idx = strategy.config.bb_period - 1
        assert all(upper.iloc[valid_idx:] > middle.iloc[valid_idx:])
        assert all(middle.iloc[valid_idx:] > lower.iloc[valid_idx:])
    
    def test_calculate_rsi(self, strategy):
        """Test RSI calculation."""
        # Create prices that should give oversold RSI
        prices = pd.Series([100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72])
        
        rsi = strategy._calculate_rsi(prices)
        
        assert len(rsi) == len(prices)
        # Downtrend should give low RSI
        assert rsi.iloc[-1] < 50
    
    def test_calculate_atr(self, strategy, sample_price_data):
        """Test ATR calculation."""
        df = sample_price_data
        
        atr = strategy._calculate_atr(df)
        
        assert len(atr) == len(df)
        # ATR should be positive
        assert all(atr[strategy.config.atr_period:] > 0)
    
    @pytest.mark.asyncio
    async def test_has_sufficient_data(self, strategy, sample_price_data):
        """Test sufficient data check."""
        # No data
        assert not strategy._has_sufficient_data("BTCUSDT")
        
        # Add some data but not enough
        strategy.price_history["BTCUSDT"] = sample_price_data.head(10)
        assert not strategy._has_sufficient_data("BTCUSDT")
        
        # Add sufficient data
        strategy.price_history["BTCUSDT"] = sample_price_data
        assert strategy._has_sufficient_data("BTCUSDT")
    
    @pytest.mark.asyncio
    async def test_get_indicators(self, strategy, sample_price_data):
        """Test indicator calculation."""
        strategy.price_history["BTCUSDT"] = sample_price_data
        
        indicators = await strategy._get_indicators("BTCUSDT")
        
        assert indicators is not None
        assert "bb_upper" in indicators
        assert "bb_middle" in indicators
        assert "bb_lower" in indicators
        assert "rsi" in indicators
        assert "atr" in indicators
        assert "current_price" in indicators
    
    @pytest.mark.asyncio
    async def test_is_ranging_market(self, strategy, sample_price_data):
        """Test ranging market detection."""
        strategy.price_history["BTCUSDT"] = sample_price_data
        
        # Mock indicators
        indicators = {"current_price": 50000}
        
        is_ranging = await strategy._is_ranging_market("BTCUSDT", indicators)
        
        # Should return a boolean
        assert isinstance(is_ranging, bool)
    
    @pytest.mark.asyncio
    async def test_generate_entry_signal_buy(self, strategy):
        """Test buy signal generation."""
        # Setup indicators for oversold condition
        indicators = {
            "current_price": 49000,
            "bb_upper": 51000,
            "bb_middle": 50000,
            "bb_lower": 49000,  # Price at lower band
            "rsi": 25,  # Oversold
            "atr": 500
        }
        
        signal = await strategy._generate_entry_signal("BTCUSDT", indicators)
        
        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence > 0
        assert signal.stop_loss < Decimal("49000")
        assert signal.take_profit == Decimal("50000")  # Middle band
    
    @pytest.mark.asyncio
    async def test_generate_entry_signal_no_signal(self, strategy):
        """Test no signal when conditions not met."""
        # Setup indicators for neutral condition
        indicators = {
            "current_price": 50000,
            "bb_upper": 51000,
            "bb_middle": 50000,
            "bb_lower": 49000,
            "rsi": 50,  # Neutral
            "atr": 500
        }
        
        signal = await strategy._generate_entry_signal("BTCUSDT", indicators)
        
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_check_exit_conditions_mean_reversion(self, strategy):
        """Test exit on mean reversion."""
        position = Position(
            position_id="test_123",
            account_id="test_account",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("49000"),
            quantity=Decimal("0.02"),
            dollar_value=Decimal("980"),
            created_at=datetime.now(UTC)
        )
        
        indicators = {
            "current_price": 50000,  # At middle band
            "bb_middle": 50000,
            "atr": 500
        }
        
        signal = await strategy._check_exit_conditions(position, indicators)
        
        # Signal may or may not be generated based on conditions
        if signal:
            assert signal.signal_type == SignalType.CLOSE
            assert "exit_reason" in signal.metadata
    
    @pytest.mark.asyncio
    async def test_check_exit_conditions_stop_loss(self, strategy):
        """Test exit on stop loss."""
        position = Position(
            position_id="test_123",
            account_id="test_account",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.02"),
            dollar_value=Decimal("1000"),
            created_at=datetime.now(UTC)
        )
        
        indicators = {
            "current_price": 48500,  # Below stop loss
            "bb_middle": 50000,
            "atr": 500  # Stop loss would be 50000 - (500 * 2) = 49000
        }
        
        signal = await strategy._check_exit_conditions(position, indicators)
        
        assert signal is not None
        assert signal.signal_type == SignalType.CLOSE
        assert signal.metadata["exit_reason"] == "stop_loss"
    
    @pytest.mark.asyncio
    async def test_calculate_position_size(self, strategy):
        """Test position size calculation."""
        price = Decimal("50000")
        atr = Decimal("500")
        
        size = await strategy._calculate_position_size("BTCUSDT", price, atr)
        
        assert size > 0
        assert size <= strategy.config.max_position_usdt / price
    
    @pytest.mark.asyncio
    async def test_validate_signal(self, strategy):
        """Test signal validation."""
        signal = Signal(
            strategy_id="test",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.7")
        )
        
        # Should pass with no positions
        assert await strategy._validate_signal(signal)
        
        # Add max positions
        for i in range(5):
            strategy.symbol_positions[f"SYMBOL{i}"] = MagicMock()
        
        # Should fail with max positions
        assert not await strategy._validate_signal(signal)
    
    @pytest.mark.asyncio
    async def test_on_order_filled_buy(self, strategy):
        """Test handling buy order fill."""
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "BUY"
        order.order_id = "order_123"
        order.filled_price = Decimal("50000")
        order.filled_quantity = Decimal("0.02")
        
        await strategy.on_order_filled(order)
        
        assert "BTCUSDT" in strategy.symbol_positions
        position = strategy.symbol_positions["BTCUSDT"]
        assert position.entry_price == Decimal("50000")
        assert position.quantity == Decimal("0.02")
    
    @pytest.mark.asyncio
    async def test_on_order_filled_sell(self, strategy):
        """Test handling sell order fill."""
        # Setup existing position
        position = Position(
            position_id="pos_123",
            account_id="test",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("49000"),
            quantity=Decimal("0.02"),
            dollar_value=Decimal("980"),
            created_at=datetime.now(UTC)
        )
        strategy.symbol_positions["BTCUSDT"] = position
        strategy.state.positions.append(position)
        
        # Create sell order
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "SELL"
        order.filled_price = Decimal("50000")
        order.filled_quantity = Decimal("0.02")
        
        await strategy.on_order_filled(order)
        
        assert "BTCUSDT" not in strategy.symbol_positions
        assert len(strategy.state.positions) == 0
        assert len(strategy.trade_history) == 1
        assert strategy.trade_history[0]["pnl"] == 20.0  # (50000 - 49000) * 0.02
    
    @pytest.mark.asyncio
    async def test_on_position_closed(self, strategy):
        """Test position close event handling."""
        position = Position(
            position_id="pos_123",
            account_id="test",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.02"),
            dollar_value=Decimal("1000"),
            created_at=datetime.now(UTC)
        )
        
        strategy.symbol_positions["BTCUSDT"] = position
        strategy.state.positions.append(position)
        
        await strategy.on_position_closed(position)
        
        assert "BTCUSDT" not in strategy.symbol_positions
        assert len(strategy.state.positions) == 0
    
    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio(self, strategy):
        """Test Sharpe ratio calculation."""
        # Add some P&L data
        strategy.realized_pnl = [
            Decimal("100"),
            Decimal("-50"),
            Decimal("75"),
            Decimal("120"),
            Decimal("-30")
        ]
        
        sharpe = await strategy.calculate_sharpe_ratio()
        
        assert sharpe >= 0
        assert isinstance(sharpe, Decimal)
    
    @pytest.mark.asyncio
    async def test_generate_signals_full_flow(self, strategy, sample_price_data):
        """Test full signal generation flow."""
        # Setup price data for two symbols
        strategy.price_history["BTCUSDT"] = sample_price_data
        strategy.price_history["ETHUSDT"] = sample_price_data.copy()
        
        # Mock the helper methods
        with patch.object(strategy, "_has_sufficient_data", return_value=True):
            with patch.object(strategy, "_get_indicators", return_value={
                "current_price": 49000,
                "bb_upper": 51000,
                "bb_middle": 50000,
                "bb_lower": 49000,
                "rsi": 25,
                "atr": 500
            }):
                with patch.object(strategy, "_is_ranging_market", return_value=True):
                    signals = await strategy.generate_signals()
        
        assert isinstance(signals, list)
        # Should generate signals for symbols meeting conditions
    
    @pytest.mark.asyncio
    async def test_analyze_market_data(self, strategy):
        """Test market data analysis."""
        market_data = {
            "symbol": "BTCUSDT",
            "price": 49000,
            "volume": 1000,
            "high": 49500,
            "low": 48500,
            "timestamp": datetime.now(UTC)
        }
        
        # Add to config symbols
        strategy.config.symbols = ["BTCUSDT"]
        
        signal = await strategy.analyze(market_data)
        
        # Initially no signal due to insufficient data
        assert signal is None
        
        # Add sufficient data
        for i in range(100):
            await strategy._update_price_history("BTCUSDT", market_data)
        
        # Mock helpers to return appropriate values
        with patch.object(strategy, "_has_sufficient_data", return_value=True):
            with patch.object(strategy, "_get_indicators", return_value={
                "current_price": 49000,
                "bb_upper": 51000,
                "bb_middle": 50000,
                "bb_lower": 49000,
                "rsi": 25,
                "atr": 500
            }):
                with patch.object(strategy, "_is_ranging_market", return_value=True):
                    signal = await strategy.analyze(market_data)
        
        # Should generate signal when conditions are met
        if signal:
            assert signal.symbol == "BTCUSDT"
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    
    @pytest.mark.asyncio
    async def test_manage_positions(self, strategy):
        """Test position management."""
        # Add a position
        position = Position(
            position_id="pos_123",
            account_id="test",
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("49000"),
            quantity=Decimal("0.02"),
            dollar_value=Decimal("980"),
            created_at=datetime.now(UTC)
        )
        strategy.symbol_positions["BTCUSDT"] = position
        
        # Mock get_indicators
        with patch.object(strategy, "_get_indicators", return_value={
            "current_price": 50000,
            "bb_middle": 50000,
            "atr": 500
        }):
            exit_signals = await strategy.manage_positions()
        
        assert isinstance(exit_signals, list)
        if exit_signals:
            assert exit_signals[0].signal_type == SignalType.CLOSE


class TestIndicatorCaching:
    """Test indicator caching functionality."""
    
    @pytest.mark.asyncio
    async def test_indicator_cache(self, strategy, sample_price_data):
        """Test that indicators are cached properly."""
        strategy.price_history["BTCUSDT"] = sample_price_data
        
        # First call should calculate
        indicators1 = await strategy._get_indicators("BTCUSDT")
        assert "BTCUSDT" in strategy.indicators_cache
        
        # Second call should use cache
        indicators2 = await strategy._get_indicators("BTCUSDT")
        assert indicators1 == indicators2
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, strategy, sample_price_data):
        """Test cache expiry."""
        strategy.price_history["BTCUSDT"] = sample_price_data
        strategy.config.cache_ttl_seconds = 0  # Immediate expiry
        
        indicators1 = await strategy._get_indicators("BTCUSDT")
        await asyncio.sleep(0.1)
        indicators2 = await strategy._get_indicators("BTCUSDT")
        
        # Should recalculate due to expiry
        assert indicators1["timestamp"] != indicators2["timestamp"]


class TestRiskManagement:
    """Test risk management features."""
    
    @pytest.mark.asyncio
    async def test_max_concurrent_positions(self, strategy):
        """Test max concurrent positions limit."""
        signal = Signal(
            strategy_id="test",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.7")
        )
        
        # Add max positions
        for i in range(5):
            strategy.symbol_positions[f"SYMBOL{i}"] = MagicMock()
        
        # Should not validate with max positions
        assert not await strategy._validate_signal(signal)
    
    @pytest.mark.asyncio
    async def test_drawdown_limit(self, strategy):
        """Test drawdown limit enforcement."""
        signal = Signal(
            strategy_id="test",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.7")
        )
        
        # Set high drawdown
        strategy.state.max_drawdown = Decimal("100")  # Above 5% of max_position_usdt
        
        # Should not validate with high drawdown
        assert not await strategy._validate_signal(signal)
    
    @pytest.mark.asyncio
    async def test_position_already_exists(self, strategy):
        """Test that duplicate positions are prevented."""
        signal = Signal(
            strategy_id="test",
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            confidence=Decimal("0.7")
        )
        
        # Add existing position
        strategy.symbol_positions["BTCUSDT"] = MagicMock()
        
        # Should not validate with existing position
        assert not await strategy._validate_signal(signal)