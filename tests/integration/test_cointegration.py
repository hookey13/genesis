"""Integration tests for cointegration and pairs trading flow."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from genesis.analytics.cointegration import CointegrationTester
from genesis.analytics.spread_calculator import SpreadCalculator
from genesis.core.models import Order, Position, Signal
from genesis.strategies.hunter.pairs_trading import (
    PairsTradingConfig,
    PairsTradingStrategy,
    TradingPair,
)


class TestPairsTradingIntegration:
    """Integration tests for pairs trading workflow."""
    
    @pytest.fixture
    def strategy(self):
        """Create configured strategy."""
        config = PairsTradingConfig(
            name="IntegrationTest",
            max_pairs=3,
            lookback_window=50,
            entry_zscore=Decimal("2.0"),
            exit_zscore=Decimal("0.5"),
            correlation_threshold=Decimal("0.75")
        )
        return PairsTradingStrategy(config)
    
    @pytest.fixture
    def market_data(self):
        """Generate realistic market data."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range(start='2024-01-01', periods=n, freq='h')
        
        # Create base trend
        trend = np.cumsum(np.random.normal(0.0005, 0.01, n))
        
        # BTC prices
        btc_returns = trend + np.random.normal(0, 0.015, n)
        btc_prices = 40000 * np.exp(btc_returns)
        
        # ETH highly correlated with BTC
        eth_returns = 0.85 * trend + np.random.normal(0, 0.01, n)
        eth_prices = 2500 * np.exp(eth_returns)
        
        # BNB moderately correlated
        bnb_returns = 0.6 * trend + np.random.normal(0, 0.02, n)
        bnb_prices = 300 * np.exp(bnb_returns)
        
        # ADA weakly correlated
        ada_returns = 0.3 * trend + np.random.normal(0, 0.025, n)
        ada_prices = 0.5 * np.exp(ada_returns)
        
        # SOL uncorrelated
        sol_returns = np.cumsum(np.random.normal(0.0003, 0.02, n))
        sol_prices = 100 * np.exp(sol_returns)
        
        return {
            'BTCUSDT': pd.DataFrame({'timestamp': dates, 'close': btc_prices, 'volume': 1000000}),
            'ETHUSDT': pd.DataFrame({'timestamp': dates, 'close': eth_prices, 'volume': 800000}),
            'BNBUSDT': pd.DataFrame({'timestamp': dates, 'close': bnb_prices, 'volume': 500000}),
            'ADAUSDT': pd.DataFrame({'timestamp': dates, 'close': ada_prices, 'volume': 300000}),
            'SOLUSDT': pd.DataFrame({'timestamp': dates, 'close': sol_prices, 'volume': 400000})
        }
    
    @pytest.mark.asyncio
    async def test_full_pair_identification_flow(self, strategy, market_data):
        """Test complete flow from data to pair identification."""
        # Load market data
        strategy.market_data_cache = market_data
        
        # Scan for pairs
        await strategy._scan_for_pairs()
        
        # Should find BTC/ETH as highly correlated and cointegrated
        assert len(strategy.state.active_pairs) > 0
        
        # Check that identified pairs are valid
        for pair in strategy.state.active_pairs:
            assert pair.correlation > strategy.config.correlation_threshold
            assert pair.is_cointegrated()
            assert pair.hedge_ratio > Decimal("0")
    
    @pytest.mark.asyncio
    async def test_signal_generation_flow(self, strategy, market_data):
        """Test signal generation for identified pairs."""
        # Setup
        strategy.market_data_cache = market_data
        
        # Create a pair with entry condition
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.85"),
            cointegration_pvalue=Decimal("0.03"),
            hedge_ratio=Decimal("15.5"),  # Approximate BTC/ETH ratio
            current_zscore=Decimal("2.5"),  # Above entry threshold
            spread_mean=Decimal("0"),
            spread_std=Decimal("1")
        )
        strategy.state.active_pairs.append(pair)
        
        # Generate signals
        signals = await strategy.generate_signals()
        
        # Should generate entry signal
        assert len(signals) > 0
        signal = signals[0]
        assert signal.symbol == "BTCUSDT/ETHUSDT"
        assert signal.direction in ["LONG_SPREAD", "SHORT_SPREAD"]
    
    @pytest.mark.asyncio
    async def test_position_lifecycle(self, strategy, market_data):
        """Test complete position lifecycle from entry to exit."""
        strategy.market_data_cache = market_data
        
        # Setup pair with position
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.85"),
            cointegration_pvalue=Decimal("0.03"),
            hedge_ratio=Decimal("15.5"),
            is_active=True,
            entry_time=datetime.now() - timedelta(days=3),
            entry_zscore=Decimal("2.2")
        )
        strategy.state.active_pairs.append(pair)
        
        # Simulate order fill
        order = Order(
            order_id="test_order_1",
            strategy_id=strategy.config.strategy_id,
            symbol="BTCUSDT/ETHUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("1"),
            metadata={"pair_id": str(pair.pair_id)}
        )
        await strategy.on_order_filled(order)
        
        assert pair.is_active
        assert pair.entry_time is not None
        
        # Simulate mean reversion (exit condition)
        pair.current_zscore = Decimal("0.3")
        
        # Generate exit signals
        exit_signals = await strategy.manage_positions()
        
        assert len(exit_signals) > 0
        exit_signal = exit_signals[0]
        assert exit_signal.direction == "CLOSE"
        assert exit_signal.metadata["exit_reason"] in ["MEAN_REVERSION", "STOP_LOSS", "MAX_HOLDING_PERIOD"]
        
        # Simulate position close
        position = Position(
            position_id="test_pos_1",
            strategy_id=strategy.config.strategy_id,
            symbol="BTCUSDT/ETHUSDT",
            side="LONG",
            quantity=Decimal("1"),
            entry_price=Decimal("40000"),
            realized_pnl=Decimal("500"),
            metadata={"pair_id": str(pair.pair_id)}
        )
        await strategy.on_position_closed(position)
        
        assert not pair.is_active
        assert strategy.state.trades_count == 1
        assert strategy.state.wins_count == 1
    
    @pytest.mark.asyncio
    async def test_recalibration_flow(self, strategy, market_data):
        """Test pair recalibration process."""
        strategy.market_data_cache = market_data
        
        # Create pair that needs recalibration
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.80"),
            cointegration_pvalue=Decimal("0.04"),
            hedge_ratio=Decimal("15.0"),
            last_calibration=datetime.now() - timedelta(days=8)  # Old calibration
        )
        strategy.state.active_pairs.append(pair)
        
        # Trigger recalibration
        await strategy._recalibrate_pair(pair)
        
        # Check pair was updated
        assert pair.last_calibration > datetime.now() - timedelta(minutes=1)
        # Values should be updated (exact values depend on mock data)
        assert pair.correlation > Decimal("0")
        assert pair.hedge_ratio > Decimal("0")
    
    @pytest.mark.asyncio
    async def test_concurrent_pairs_management(self, strategy, market_data):
        """Test managing multiple pairs concurrently."""
        strategy.market_data_cache = market_data
        
        # Add multiple pairs
        pairs = [
            TradingPair(
                symbol1="BTCUSDT",
                symbol2="ETHUSDT",
                correlation=Decimal("0.85"),
                cointegration_pvalue=Decimal("0.03"),
                current_zscore=Decimal("2.1")
            ),
            TradingPair(
                symbol1="BNBUSDT",
                symbol2="ADAUSDT",
                correlation=Decimal("0.78"),
                cointegration_pvalue=Decimal("0.04"),
                current_zscore=Decimal("-2.2")
            ),
            TradingPair(
                symbol1="SOLUSDT",
                symbol2="ADAUSDT",
                correlation=Decimal("0.76"),
                cointegration_pvalue=Decimal("0.045"),
                current_zscore=Decimal("0.5")
            )
        ]
        
        strategy.state.active_pairs = pairs
        
        # Generate signals for all pairs
        signals = await strategy.generate_signals()
        
        # Should generate signals for pairs meeting entry criteria
        entry_pairs = [p for p in pairs if abs(p.current_zscore) >= strategy.config.entry_zscore]
        assert len([s for s in signals if s.direction != "CLOSE"]) >= len(entry_pairs)
    
    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, strategy):
        """Test stop loss trigger for diverging spread."""
        # Create pair with extreme z-score
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            is_active=True,
            current_zscore=Decimal("3.5"),  # Beyond stop loss
            entry_zscore=Decimal("2.0"),
            entry_time=datetime.now() - timedelta(days=2)
        )
        strategy.state.active_pairs.append(pair)
        
        # Check stop loss
        signal = await strategy._check_stop_loss(pair)
        
        assert signal is not None
        assert signal.direction == "CLOSE"
        assert signal.metadata["exit_reason"] == "STOP_LOSS"
    
    @pytest.mark.asyncio
    async def test_max_holding_period(self, strategy):
        """Test exit on maximum holding period."""
        # Create pair exceeding max holding period
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            is_active=True,
            current_zscore=Decimal("1.5"),
            entry_time=datetime.now() - timedelta(days=35)  # Beyond 30 days
        )
        strategy.state.active_pairs.append(pair)
        
        # Check for timeout exit
        signals = await strategy.manage_positions()
        
        assert len(signals) > 0
        timeout_signal = [s for s in signals if s.metadata.get("exit_reason") == "MAX_HOLDING_PERIOD"]
        assert len(timeout_signal) > 0
    
    @pytest.mark.asyncio
    async def test_pair_independence_check(self, strategy):
        """Test pair independence validation."""
        # Add existing pair
        existing = TradingPair(symbol1="BTCUSDT", symbol2="ETHUSDT")
        strategy.state.active_pairs.append(existing)
        
        # Test overlapping pairs
        overlap1 = TradingPair(symbol1="BTCUSDT", symbol2="BNBUSDT")
        overlap2 = TradingPair(symbol1="ADAUSDT", symbol2="ETHUSDT")
        
        assert not strategy._check_pair_independence(overlap1)
        assert not strategy._check_pair_independence(overlap2)
        
        # Test independent pair
        independent = TradingPair(symbol1="SOLUSDT", symbol2="ADAUSDT")
        assert strategy._check_pair_independence(independent)
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, strategy):
        """Test performance metrics tracking."""
        # Simulate multiple trades
        trades = [
            (Decimal("500"), True),   # Win
            (Decimal("-200"), False),  # Loss
            (Decimal("300"), True),    # Win
            (Decimal("-100"), False),  # Loss
            (Decimal("400"), True),    # Win
        ]
        
        for pnl, is_win in trades:
            strategy.update_performance_metrics(pnl, is_win)
        
        assert strategy.state.trades_count == 5
        assert strategy.state.wins_count == 3
        assert strategy.state.losses_count == 2
        assert strategy.state.win_rate == Decimal("0.6")
        assert strategy.state.pnl_usdt == Decimal("900")
        assert strategy.state.max_drawdown == Decimal("200")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, strategy):
        """Test error handling and recovery."""
        # Test with missing market data
        pair = TradingPair(
            symbol1="MISSING1",
            symbol2="MISSING2"
        )
        
        # Should handle gracefully
        await strategy._recalibrate_pair(pair)
        await strategy._update_pair_zscore(pair)
        
        # Pair should remain unchanged or be handled appropriately
        assert pair.current_zscore == Decimal("0")  # Default value
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, strategy):
        """Test saving and loading strategy state."""
        # Setup some state
        pair = TradingPair(
            symbol1="BTCUSDT",
            symbol2="ETHUSDT",
            correlation=Decimal("0.85")
        )
        strategy.state.active_pairs.append(pair)
        strategy.state.trades_count = 10
        strategy.state.wins_count = 7
        
        # Save state
        state_data = await strategy.save_state()
        
        assert "config" in state_data
        assert "state" in state_data
        assert state_data["state"]["trades_count"] == 10
        
        # Create new strategy and load state
        new_strategy = PairsTradingStrategy()
        await new_strategy.load_state(state_data)
        
        assert new_strategy.state.trades_count == 10
        assert new_strategy.state.wins_count == 7