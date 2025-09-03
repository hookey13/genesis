"""Unit tests for pairs trading strategy."""

import asyncio
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from genesis.strategies.hunter.pairs_trading import (
    PairsTradingConfig,
    PairsTradingState,
    PairsTradingStrategy,
    TradingPair,
)
from genesis.core.models import Order, OrderType, OrderSide, Position, PositionSide, Signal


class TestTradingPair:
    """Test TradingPair dataclass."""
    
    def test_trading_pair_creation(self):
        """Test creating a trading pair."""
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.85"),
            cointegration_pvalue=Decimal("0.03"),
            hedge_ratio=Decimal("1.5")
        )
        
        assert pair.symbol1 == "BTCUSDT"
        assert pair.symbol2 == "ETHUSDT"
        assert pair.correlation == Decimal("0.85")
        assert pair.cointegration_pvalue == Decimal("0.03")
        assert pair.hedge_ratio == Decimal("1.5")
        assert not pair.is_active
    
    def test_is_cointegrated(self):
        """Test cointegration check."""
        # Cointegrated pair
        pair1 = TradingPair(cointegration_pvalue=Decimal("0.03"))
        assert pair1.is_cointegrated()
        
        # Not cointegrated
        pair2 = TradingPair(cointegration_pvalue=Decimal("0.10"))
        assert not pair2.is_cointegrated()
    
    def test_needs_recalibration(self):
        """Test recalibration check."""
        from datetime import UTC
        # Recent calibration
        pair1 = TradingPair(last_calibration=datetime.now(UTC))
        assert not pair1.needs_recalibration(recalibration_hours=168)
        
        # Old calibration
        pair2 = TradingPair(last_calibration=datetime.now(UTC) - timedelta(days=8))
        assert pair2.needs_recalibration(recalibration_hours=168)


class TestPairsTradingConfig:
    """Test PairsTradingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PairsTradingConfig()
        
        assert config.max_pairs == 5
        assert config.correlation_threshold == Decimal("0.8")
        assert config.cointegration_pvalue_threshold == Decimal("0.05")
        assert config.entry_zscore == Decimal("2.0")
        assert config.exit_zscore == Decimal("0.5")
        assert config.stop_loss_zscore == Decimal("3.0")
        assert config.lookback_window == 100
        assert config.recalibration_frequency_hours == 168
        assert config.max_holding_period_days == 30
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PairsTradingConfig(
            max_pairs=3,
            entry_zscore=Decimal("2.5"),
            lookback_window=150
        )
        
        assert config.max_pairs == 3
        assert config.entry_zscore == Decimal("2.5")
        assert config.lookback_window == 150


class TestPairsTradingStrategy:
    """Test PairsTradingStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        config = PairsTradingConfig(
            name="TestPairsStrategy",
            max_pairs=3,
            lookback_window=50
        )
        return PairsTradingStrategy(config)
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        
        # Create correlated price series
        np.random.seed(42)
        btc_returns = np.random.normal(0.001, 0.02, 100)
        btc_prices = 40000 * np.exp(np.cumsum(btc_returns))
        
        # ETH correlated with BTC
        eth_returns = 0.8 * btc_returns + np.random.normal(0, 0.01, 100)
        eth_prices = 2500 * np.exp(np.cumsum(eth_returns))
        
        btc_df = pd.DataFrame({
            'timestamp': dates,
            'close': btc_prices,
            'volume': np.random.uniform(100, 200, 100)
        })
        
        eth_df = pd.DataFrame({
            'timestamp': dates,
            'close': eth_prices,
            'volume': np.random.uniform(50, 100, 100)
        })
        
        return {'BTCUSDT': btc_df, 'ETHUSDT': eth_df}
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "TestPairsStrategy"
        assert strategy.config.max_pairs == 3
        assert strategy.config.lookback_window == 50
        assert isinstance(strategy.state, PairsTradingState)
        assert strategy.cointegration_tester is not None
        assert strategy.spread_calculator is not None
    
    @pytest.mark.asyncio
    async def test_generate_signals_no_pairs(self, strategy):
        """Test signal generation with no active pairs."""
        signals = await strategy.generate_signals()
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_generate_signals_with_pairs(self, strategy):
        """Test signal generation with active pairs."""
        # Add an active pair
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.85"),
            cointegration_pvalue=Decimal("0.03"),
            hedge_ratio=Decimal("1.5"),
            current_zscore=Decimal("2.5")
        )
        strategy.state.active_pairs.append(pair)
        
        # Mock the helper methods
        strategy._should_scan_for_pairs = AsyncMock(return_value=False)
        strategy._generate_pair_signal = AsyncMock(return_value=None)
        strategy.manage_positions = AsyncMock(return_value=[])
        
        signals = await strategy.generate_signals()
        
        strategy._generate_pair_signal.assert_called_once_with(pair)
        strategy.manage_positions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_market_data(self, strategy):
        """Test analyzing market data."""
        market_data = {
            "symbol": "BTCUSDT",
            "price": 40000,
            "volume": 150
        }
        
        # Mock update method
        strategy._update_market_data = AsyncMock()
        
        signal = await strategy.analyze(market_data)
        
        strategy._update_market_data.assert_called_once_with("BTCUSDT", market_data)
        assert signal is None  # No pairs yet
    
    @pytest.mark.asyncio
    async def test_manage_positions_no_active(self, strategy):
        """Test position management with no active positions."""
        signals = await strategy.manage_positions()
        assert signals == []
    
    @pytest.mark.asyncio
    async def test_manage_positions_with_exit(self, strategy):
        """Test position management with exit conditions."""
        # Add active pair with position
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            is_active=True,
            current_zscore=Decimal("0.3"),  # Below exit threshold
            entry_time=datetime.now(UTC) - timedelta(days=5)
        )
        strategy.state.active_pairs.append(pair)
        
        # Mock helper methods
        strategy._check_pair_exit = AsyncMock(
            return_value=Signal(
                signal_id=str(uuid4()),
                strategy_id=str(strategy.config.strategy_id),
                symbol="BTCUSDT/ETHUSDT",
                signal_type="CLOSE",
                confidence=Decimal("1")
            )
        )
        strategy._check_stop_loss = AsyncMock(return_value=None)
        strategy._create_timeout_exit = AsyncMock(return_value=None)
        
        signals = await strategy.manage_positions()
        
        assert len(signals) == 1
        assert signals[0].signal_type == "CLOSE"
    
    @pytest.mark.asyncio
    async def test_on_order_filled(self, strategy):
        """Test order fill handling."""
        pair = TradingPair(
            pair_id=uuid4(),
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            is_active=False
        )
        strategy.state.active_pairs.append(pair)
        
        order = Order(
            order_id=str(uuid4()),
            symbol="BTCUSDT/ETHUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("1"),
            metadata={"pair_id": str(pair.pair_id)}
        )
        
        await strategy.on_order_filled(order)
        
        assert pair.is_active
        assert pair.entry_time is not None
    
    @pytest.mark.asyncio
    async def test_on_position_closed(self, strategy):
        """Test position close handling."""
        pair = TradingPair(
            pair_id=uuid4(),
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            is_active=True
        )
        strategy.state.active_pairs.append(pair)
        
        position = Position(
            position_id=str(uuid4()),
            account_id="test_account",
            symbol="BTCUSDT/ETHUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1"),
            entry_price=Decimal("40000"),
            dollar_value=Decimal("40000")
        )
        
        await strategy.on_position_closed(position)
        
        assert not pair.is_active
        assert pair.entry_time is None
        assert strategy.state.trades_count == 1
        assert strategy.state.wins_count == 1
    
    @pytest.mark.asyncio
    async def test_should_scan_for_pairs(self, strategy):
        """Test pair scanning conditions."""
        # Should scan when no pairs
        assert await strategy._should_scan_for_pairs()
        
        # Add max pairs
        for i in range(3):
            strategy.state.active_pairs.append(TradingPair())
        
        # Should not scan when at max
        assert not await strategy._should_scan_for_pairs()
        
        # Should scan after 24 hours
        strategy.state.last_scan_time = datetime.now(UTC) - timedelta(hours=25)
        assert await strategy._should_scan_for_pairs()
    
    @pytest.mark.asyncio
    async def test_scan_for_pairs(self, strategy, mock_market_data):
        """Test scanning for cointegrated pairs."""
        strategy.market_data_cache = mock_market_data
        
        # Mock the cointegration and spread calculator methods
        with patch.object(strategy.spread_calculator, 'calculate_correlation') as mock_corr:
            with patch.object(strategy.cointegration_tester, 'test_engle_granger') as mock_coint:
                with patch.object(strategy.spread_calculator, 'calculate_spread') as mock_spread:
                    with patch.object(strategy.spread_calculator, 'analyze_spread_quality') as mock_quality:
                        
                        # Setup mocks
                        mock_corr.return_value = MagicMock(
                            pearson_correlation=Decimal("0.85"),
                            is_stable=True
                        )
                        mock_coint.return_value = MagicMock(
                            is_cointegrated=True,
                            p_value=Decimal("0.03"),
                            hedge_ratio=Decimal("1.5")
                        )
                        mock_spread.return_value = MagicMock(
                            mean=Decimal("0"),
                            std_dev=Decimal("1"),
                            current_zscore=Decimal("0.5")
                        )
                        mock_quality.return_value = {
                            'is_tradeable': True,
                            'quality_score': 0.75
                        }
                        
                        await strategy._scan_for_pairs()
                        
                        assert len(strategy.state.active_pairs) == 1
                        pair = strategy.state.active_pairs[0]
                        assert pair.symbol1 in ["BTCUSDT", "ETHUSDT"]
                        assert pair.symbol2 in ["BTCUSDT", "ETHUSDT"]
                        assert pair.correlation == Decimal("0.85")
    
    @pytest.mark.asyncio
    async def test_recalibrate_pair(self, strategy, mock_market_data):
        """Test pair recalibration."""
        strategy.market_data_cache = mock_market_data
        
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.80"),
            cointegration_pvalue=Decimal("0.04"),
            hedge_ratio=Decimal("1.4")
        )
        strategy.state.active_pairs.append(pair)
        
        # Mock the calculator methods
        with patch.object(strategy.spread_calculator, 'calculate_correlation') as mock_corr:
            with patch.object(strategy.cointegration_tester, 'test_engle_granger') as mock_coint:
                with patch.object(strategy.spread_calculator, 'calculate_spread') as mock_spread:
                    
                    mock_corr.return_value = MagicMock(
                        pearson_correlation=Decimal("0.82")
                    )
                    mock_coint.return_value = MagicMock(
                        is_cointegrated=True,
                        p_value=Decimal("0.035"),
                        hedge_ratio=Decimal("1.45")
                    )
                    mock_spread.return_value = MagicMock(
                        mean=Decimal("0.1"),
                        std_dev=Decimal("0.9"),
                        current_zscore=Decimal("0.2")
                    )
                    
                    await strategy._recalibrate_pair(pair)
                    
                    assert pair.correlation == Decimal("0.82")
                    assert pair.cointegration_pvalue == Decimal("0.035")
                    assert pair.hedge_ratio == Decimal("1.45")
    
    @pytest.mark.asyncio
    async def test_create_entry_signal(self, strategy):
        """Test entry signal creation."""
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            current_zscore=Decimal("2.5"),
            hedge_ratio=Decimal("1.5"),
            correlation=Decimal("0.85")
        )
        
        signal = await strategy._create_entry_signal(pair)
        
        assert signal.symbol == "BTCUSDT/ETHUSDT"
        assert signal.metadata["direction"] == "SHORT_SPREAD"  # z-score > 2.0
        assert signal.metadata["pair_id"] == str(pair.pair_id)
        assert signal.metadata["hedge_ratio"] == "1.5"
    
    @pytest.mark.asyncio
    async def test_create_exit_signal(self, strategy):
        """Test exit signal creation."""
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            current_zscore=Decimal("0.3")
        )
        
        signal = await strategy._create_exit_signal(pair, "MEAN_REVERSION")
        
        assert signal.symbol == "BTCUSDT/ETHUSDT"
        assert signal.signal_type == "CLOSE"
        assert signal.metadata["exit_reason"] == "MEAN_REVERSION"
    
    def test_check_pair_independence(self, strategy):
        """Test pair independence checking."""
        # Add existing pair
        existing_pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT"
        )
        strategy.state.active_pairs.append(existing_pair)
        
        # Test overlapping pair (should fail)
        new_pair1 = TradingPair(
            symbol1="BTCUSDT",
            symbol2="BNBUSDT"
        )
        assert not strategy._check_pair_independence(new_pair1)
        
        # Test independent pair (should pass)
        new_pair2 = TradingPair(
            symbol1="ADAUSDT",
            symbol2="SOLUSDT"
        )
        assert strategy._check_pair_independence(new_pair2)
    
    def test_get_pair_statistics(self, strategy):
        """Test getting pair statistics."""
        # Add some performance data
        strategy.state.total_pairs_traded = 10
        strategy.state.successful_pairs = 7
        strategy.state.pair_performance["BTC/ETH"] = {
            "trades": 5,
            "wins": 3,
            "total_pnl": Decimal("1500"),
            "best_trade": Decimal("600"),
            "worst_trade": Decimal("-200")
        }
        
        stats = strategy.get_pair_statistics()
        
        assert stats["total_pairs_traded"] == 10
        assert stats["successful_pairs"] == 7
        assert "BTC/ETH" in stats["pair_performance"]
        assert stats["pair_performance"]["BTC/ETH"]["win_rate"] == 0.6