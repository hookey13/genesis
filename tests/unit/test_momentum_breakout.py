"""
Unit tests for MomentumBreakoutStrategy.

Tests breakout detection, volume confirmation, trend validation,
risk management, and signal generation.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from genesis.core.models import Order, OrderSide, Position, PositionSide, Signal, SignalType
from genesis.strategies.sniper.momentum_breakout import (
    MarketData,
    MomentumBreakoutConfig,
    MomentumBreakoutStrategy,
)


@pytest.fixture
def strategy_config():
    """Create test strategy configuration."""
    return MomentumBreakoutConfig(
        symbol="BTCUSDT",
        lookback_period=20,
        breakout_threshold=Decimal("0.02"),
        volume_multiplier=Decimal("1.5"),
        adx_threshold=Decimal("25"),
        trail_distance=Decimal("0.005"),
        false_breakout_candles=3,
        max_position_percent=Decimal("0.02"),
    )


@pytest.fixture
def strategy(strategy_config):
    """Create test strategy instance."""
    return MomentumBreakoutStrategy(strategy_config)


@pytest.fixture
def trending_market_data():
    """Generate trending market data with breakout."""
    base_price = Decimal("100")
    data = []
    
    # Generate 20 periods of ranging data
    for i in range(20):
        price = base_price + Decimal(str(i % 3))  # Range between 100-102
        data.append({
            "timestamp": datetime.now(UTC),
            "open": price,
            "high": price + Decimal("1"),
            "low": price - Decimal("0.5"),
            "close": price + Decimal("0.5"),
            "volume": Decimal("1000"),
        })
    
    # Add breakout candle
    # Max high in range is 103, so breakout needs to be > 103 * 1.02 = 105.06
    breakout_price = base_price + Decimal("5.1")  # 105.1 to ensure clear breakout above threshold
    data.append({
        "timestamp": datetime.now(UTC),
        "open": base_price + Decimal("2"),
        "high": breakout_price + Decimal("1"),
        "low": base_price + Decimal("2"),
        "close": breakout_price,
        "volume": Decimal("2000"),  # High volume
    })
    
    return data


@pytest.fixture
def ranging_market_data():
    """Generate sideways/ranging market data."""
    base_price = Decimal("100")
    data = []
    
    for i in range(25):
        # Oscillate between 99 and 101
        price = base_price + Decimal(str((i % 3) - 1))
        data.append({
            "timestamp": datetime.now(UTC),
            "open": price,
            "high": price + Decimal("0.5"),
            "low": price - Decimal("0.5"),
            "close": price,
            "volume": Decimal("1000"),
        })
    
    return data


@pytest.fixture
def false_breakout_data():
    """Generate data with false breakout."""
    base_price = Decimal("100")
    data = []
    
    # Generate initial ranging data
    for i in range(20):
        price = base_price + Decimal(str(i % 2))
        data.append({
            "timestamp": datetime.now(UTC),
            "open": price,
            "high": price + Decimal("0.5"),
            "low": price - Decimal("0.5"),
            "close": price,
            "volume": Decimal("1000"),
        })
    
    # Add false breakout (breaks above but falls back)
    # Max high in range is 101.5, so breakout needs > 101.5 * 1.02 = 103.53
    data.append({
        "timestamp": datetime.now(UTC),
        "open": base_price,
        "high": base_price + Decimal("4"),  # 104 > 103.53
        "low": base_price,
        "close": base_price + Decimal("3.6"),  # 103.6 > 103.53 - Initial breakout
        "volume": Decimal("1600"),  # High volume
    })
    
    # Price falls back below breakout level (false breakout)
    data.append({
        "timestamp": datetime.now(UTC),
        "open": base_price + Decimal("3.6"),
        "high": base_price + Decimal("3.6"),
        "low": base_price - Decimal("1"),
        "close": base_price,  # Falls back below resistance
        "volume": Decimal("1200"),
    })
    
    return data


class TestMomentumBreakoutStrategy:
    """Test suite for MomentumBreakoutStrategy."""
    
    def test_initialization(self, strategy, strategy_config):
        """Test strategy initialization."""
        assert strategy.config.symbol == "BTCUSDT"
        assert strategy.config.lookback_period == 20
        assert strategy.config.breakout_threshold == Decimal("0.02")
        assert strategy.config.volume_multiplier == Decimal("1.5")
        assert strategy.config.adx_threshold == Decimal("25")
        assert len(strategy.price_history) == 0
        assert len(strategy.volume_history) == 0
    
    @pytest.mark.asyncio
    async def test_breakout_detection_upward(self, strategy, trending_market_data):
        """Test upward breakout detection."""
        # Feed historical data
        for data in trending_market_data[:-1]:
            await strategy.analyze(data)
        
        # Feed breakout candle
        signal = await strategy.analyze(trending_market_data[-1])
        
        # Should detect breakout (though may not generate signal without full confirmation)
        assert strategy.breakout_direction == "UP"
        assert strategy.breakout_level is not None
    
    @pytest.mark.asyncio
    async def test_breakout_detection_downward(self, strategy):
        """Test downward breakout detection."""
        # Create downward breakout data
        base_price = Decimal("100")
        
        # Feed ranging data
        for i in range(20):
            data = {
                "timestamp": datetime.now(UTC),
                "open": base_price,
                "high": base_price + Decimal("1"),
                "low": base_price - Decimal("0.5"),
                "close": base_price,
                "volume": Decimal("1000"),
            }
            await strategy.analyze(data)
        
        # Feed downward breakout
        breakout_data = {
            "timestamp": datetime.now(UTC),
            "open": base_price,
            "high": base_price,
            "low": base_price - Decimal("5"),
            "close": base_price - Decimal("4"),  # Clear breakout below support
            "volume": Decimal("2000"),
        }
        
        await strategy.analyze(breakout_data)
        
        assert strategy.breakout_direction == "DOWN"
        assert strategy.breakout_level is not None
    
    @pytest.mark.asyncio
    async def test_no_breakout_in_range(self, strategy, ranging_market_data):
        """Test no breakout detected in ranging market."""
        for data in ranging_market_data:
            signal = await strategy.analyze(data)
            assert signal is None  # No signal in ranging market
        
        assert strategy.breakout_direction is None
    
    @pytest.mark.asyncio
    async def test_volume_confirmation_success(self, strategy):
        """Test volume confirmation with high volume."""
        # Setup price history
        for i in range(20):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
            strategy.volume_history.append(Decimal("1000"))
        
        # Update indicators
        data_point = MarketData(
            timestamp=datetime.now(UTC),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("100"),
            close=Decimal("104"),
            volume=Decimal("2000"),  # 2x average volume
        )
        
        strategy._update_indicators(data_point)
        
        # Test volume confirmation
        confirmed = strategy._confirm_volume(data_point)
        assert confirmed is True
    
    @pytest.mark.asyncio
    async def test_volume_confirmation_failure(self, strategy):
        """Test volume confirmation with low volume."""
        # Setup price history
        for i in range(20):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
            strategy.volume_history.append(Decimal("1000"))
        
        # Update indicators
        data_point = MarketData(
            timestamp=datetime.now(UTC),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("100"),
            close=Decimal("104"),
            volume=Decimal("1200"),  # Only 1.2x average (need 1.5x)
        )
        
        strategy._update_indicators(data_point)
        
        # Test volume confirmation
        confirmed = strategy._confirm_volume(data_point)
        assert confirmed is False
    
    def test_trend_validation_strong_trend(self, strategy):
        """Test trend validation with strong ADX."""
        strategy.indicators.adx = Decimal("30")  # Strong trend
        strategy.indicators.plus_di = Decimal("35")
        strategy.indicators.minus_di = Decimal("15")
        strategy.breakout_direction = "UP"
        
        valid = strategy._validate_trend()
        assert valid is True
    
    def test_trend_validation_weak_trend(self, strategy):
        """Test trend validation with weak ADX."""
        strategy.indicators.adx = Decimal("20")  # Weak trend
        strategy.indicators.plus_di = Decimal("25")
        strategy.indicators.minus_di = Decimal("20")
        strategy.breakout_direction = "UP"
        
        valid = strategy._validate_trend()
        assert valid is False
    
    def test_trend_validation_wrong_direction(self, strategy):
        """Test trend validation with wrong direction."""
        strategy.indicators.adx = Decimal("30")  # Strong trend
        strategy.indicators.plus_di = Decimal("15")  # But minus > plus
        strategy.indicators.minus_di = Decimal("35")
        strategy.breakout_direction = "UP"  # Trying to go up in downtrend
        
        valid = strategy._validate_trend()
        assert valid is False
    
    @pytest.mark.asyncio
    async def test_false_breakout_filter(self, strategy, false_breakout_data):
        """Test false breakout detection and filtering."""
        # Feed data including false breakout
        for data in false_breakout_data:
            signal = await strategy.analyze(data)
        
        # Should have detected and rejected false breakout
        assert strategy.false_breakouts > 0
        assert strategy.breakout_direction is None  # Reset after false breakout
    
    def test_stop_loss_calculation_long(self, strategy):
        """Test stop loss calculation for long position."""
        # Setup price history with clear swing low
        for i in range(5):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("102"),
                    low=Decimal("98") if i == 2 else Decimal("99"),  # Swing low at index 2
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
        
        strategy.indicators.atr = Decimal("2")
        strategy.breakout_direction = "UP"
        
        data_point = strategy.price_history[-1]
        stop_loss = strategy._calculate_stop_loss(data_point)
        
        # Stop should be below swing low minus ATR buffer
        expected = Decimal("98") - (Decimal("2") * Decimal("0.5"))
        assert stop_loss == expected
    
    def test_stop_loss_calculation_short(self, strategy):
        """Test stop loss calculation for short position."""
        # Setup price history with clear swing high
        for i in range(5):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("103") if i == 2 else Decimal("101"),  # Swing high at index 2
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
        
        strategy.indicators.atr = Decimal("2")
        strategy.breakout_direction = "DOWN"
        
        data_point = strategy.price_history[-1]
        stop_loss = strategy._calculate_stop_loss(data_point)
        
        # Stop should be above swing high plus ATR buffer
        expected = Decimal("103") + (Decimal("2") * Decimal("0.5"))
        assert stop_loss == expected
    
    def test_position_sizing(self, strategy):
        """Test position size calculation."""
        account_balance = Decimal("10000")
        position_size = strategy._calculate_position_size(account_balance)
        
        # Should be 2% of account
        expected = Decimal("200")
        assert position_size == expected
    
    def test_signal_generation_buy(self, strategy):
        """Test buy signal generation."""
        # Setup for buy signal
        strategy.breakout_direction = "UP"
        strategy.breakout_level = Decimal("100")
        strategy.breakout_confirmation_count = 3
        
        # Setup indicators
        strategy.indicators.adx = Decimal("30")
        strategy.indicators.atr = Decimal("2")
        strategy.indicators.volume_average = Decimal("1000")
        
        # Setup price history for stop loss calculation
        for i in range(5):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
            strategy.volume_history.append(Decimal("1000"))
        
        data_point = MarketData(
            timestamp=datetime.now(UTC),
            open=Decimal("102"),
            high=Decimal("105"),
            low=Decimal("102"),
            close=Decimal("104"),
            volume=Decimal("2000"),
        )
        
        signal = strategy._generate_signal(data_point)
        
        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
        assert signal.confidence > Decimal("0.5")
    
    def test_signal_generation_sell(self, strategy):
        """Test sell signal generation."""
        # Setup for sell signal
        strategy.breakout_direction = "DOWN"
        strategy.breakout_level = Decimal("100")
        strategy.breakout_confirmation_count = 3
        
        # Setup indicators
        strategy.indicators.adx = Decimal("30")
        strategy.indicators.atr = Decimal("2")
        strategy.indicators.volume_average = Decimal("1000")
        
        # Setup price history
        for i in range(5):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
            strategy.volume_history.append(Decimal("1000"))
        
        data_point = MarketData(
            timestamp=datetime.now(UTC),
            open=Decimal("98"),
            high=Decimal("98"),
            low=Decimal("95"),
            close=Decimal("96"),
            volume=Decimal("2000"),
        )
        
        signal = strategy._generate_signal(data_point)
        
        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
    
    @pytest.mark.asyncio
    async def test_trailing_stop_activation(self, strategy):
        """Test trailing stop activation on profit."""
        # Create a profitable position
        position = Position(
            position_id="test-001",
            account_id="test-account",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            current_price=Decimal("101"),  # 1% profit
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
            pnl_percent=Decimal("1"),  # 1% profit
        )
        
        strategy.state.positions = [position]
        
        # Generate exit signals
        signals = await strategy.manage_positions()
        
        # Trailing stop should be activated
        assert strategy.trailing_stop_active is True
        assert strategy.high_water_mark == Decimal("101")
    
    @pytest.mark.asyncio
    async def test_trailing_stop_trigger(self, strategy):
        """Test trailing stop trigger on price drop."""
        # Setup trailing stop active
        strategy.trailing_stop_active = True
        strategy.high_water_mark = Decimal("105")
        
        # Create position that hits trailing stop
        position = Position(
            position_id="test-001",
            account_id="test-account",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            current_price=Decimal("104.4"),  # Below trailing stop of 105 * 0.995
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )
        
        strategy.state.positions = [position]
        
        # Generate exit signals
        signals = await strategy.manage_positions()
        
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.CLOSE
        assert signals[0].metadata["reason"] == "trailing_stop_hit"
    
    @pytest.mark.asyncio
    async def test_on_order_filled(self, strategy):
        """Test order fill handling."""
        from genesis.core.models import OrderType
        
        order = Order(
            order_id="test-order",
            symbol="BTCUSDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            price=Decimal("100"),
        )
        
        await strategy.on_order_filled(order)
        # Should log the fill (no assertions needed for logging)
    
    @pytest.mark.asyncio
    async def test_on_position_closed_win(self, strategy):
        """Test position close handling for winning trade."""
        position = Position(
            position_id="test-001",
            account_id="test-account",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
            pnl_dollars=Decimal("10"),  # Profit
        )
        
        await strategy.on_position_closed(position)
        
        assert strategy.state.wins_count == 1
        assert strategy.state.losses_count == 0
        assert strategy.profit_factor_sum == Decimal("10")
        assert strategy.loss_factor_sum == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_on_position_closed_loss(self, strategy):
        """Test position close handling for losing trade."""
        position = Position(
            position_id="test-001",
            account_id="test-account",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
            pnl_dollars=Decimal("-5"),  # Loss
        )
        
        await strategy.on_position_closed(position)
        
        assert strategy.state.wins_count == 0
        assert strategy.state.losses_count == 1
        assert strategy.profit_factor_sum == Decimal("0")
        assert strategy.loss_factor_sum == Decimal("5")
    
    def test_performance_metrics(self, strategy):
        """Test performance metrics calculation."""
        # Simulate some trades
        strategy.total_signals = 10
        strategy.successful_breakouts = 7
        strategy.false_breakouts = 3
        strategy.profit_factor_sum = Decimal("100")
        strategy.loss_factor_sum = Decimal("50")
        strategy.win_rate_history.extend([True, True, False, True, False])
        
        metrics = strategy.get_performance_metrics()
        
        assert metrics["total_signals"] == 10
        assert metrics["successful_breakouts"] == 7
        assert metrics["false_breakouts"] == 3
        assert metrics["false_breakout_rate"] == 0.3  # 3/10
        assert metrics["profit_factor"] == 2.0  # 100/50
        assert metrics["win_rate"] == 0.6  # 3/5
    
    def test_confidence_calculation(self, strategy):
        """Test confidence calculation based on indicators."""
        # Test with strong indicators
        strategy.indicators.adx = Decimal("45")  # Very strong trend
        strategy.indicators.volume_average = Decimal("1000")
        strategy.volume_history.append(Decimal("3500"))  # 3.5x volume
        
        confidence = strategy._calculate_confidence()
        
        # Should have high confidence
        assert confidence >= Decimal("0.8")
        assert confidence <= Decimal("0.9")  # Capped at 0.9
    
    @pytest.mark.asyncio
    async def test_invalid_market_data(self, strategy):
        """Test handling of invalid market data."""
        # Test with missing fields
        invalid_data = {
            "timestamp": datetime.now(UTC),
            "open": "100",
            # Missing other required fields
        }
        
        signal = await strategy.analyze(invalid_data)
        assert signal is None  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_insufficient_history(self, strategy):
        """Test behavior with insufficient price history."""
        # Feed less than lookback period
        for i in range(10):  # Only 10 periods (need 20)
            data = {
                "timestamp": datetime.now(UTC),
                "open": Decimal("100"),
                "high": Decimal("101"),
                "low": Decimal("99"),
                "close": Decimal("100"),
                "volume": Decimal("1000"),
            }
            signal = await strategy.analyze(data)
            assert signal is None  # Should not generate signals
    
    def test_edge_case_exact_threshold(self, strategy):
        """Test breakout detection at exact threshold."""
        # Setup price history
        for i in range(20):
            strategy.price_history.append(
                MarketData(
                    timestamp=datetime.now(UTC),
                    open=Decimal("100"),
                    high=Decimal("100"),  # Max high is 100
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=Decimal("1000"),
                )
            )
        
        # Test with price exactly at threshold
        data_point = MarketData(
            timestamp=datetime.now(UTC),
            open=Decimal("100"),
            high=Decimal("102.1"),
            low=Decimal("100"),
            close=Decimal("102"),  # Exactly 2% above resistance
            volume=Decimal("1500"),
        )
        
        breakout = strategy._detect_price_breakout(data_point)
        assert breakout is True  # Should detect breakout