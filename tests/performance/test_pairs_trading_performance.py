"""Performance tests for pairs trading strategy."""

import asyncio
import time
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pytest

from genesis.analytics.cointegration import CointegrationTester
from genesis.analytics.spread_calculator import SpreadCalculator
from genesis.strategies.hunter.pairs_trading import (
    PairsTradingConfig,
    PairsTradingStrategy,
    TradingPair,
)


class TestPairsTradingPerformance:
    """Performance tests for pairs trading components."""
    
    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        n = 10000  # 10k data points
        
        # Generate multiple correlated series
        base = np.cumsum(np.random.normal(0, 0.01, n))
        
        series = {}
        for i in range(20):  # 20 symbols
            correlation = 0.5 + 0.4 * np.random.random()
            noise = np.random.normal(0, 0.005, n)
            prices = 100 * np.exp(correlation * base + noise)
            series[f"SYMBOL{i:02d}"] = pd.DataFrame({
                'close': prices,
                'volume': np.random.uniform(10000, 100000, n)
            })
        
        return series
    
    @pytest.fixture
    def calculator(self):
        """Create spread calculator."""
        return SpreadCalculator(lookback_window=100)
    
    @pytest.fixture
    def tester(self):
        """Create cointegration tester."""
        return CointegrationTester(confidence_level=0.95)
    
    def test_cointegration_speed(self, tester, large_dataset):
        """Test speed of cointegration testing."""
        # Get two series
        series1 = large_dataset["SYMBOL00"]["close"]
        series2 = large_dataset["SYMBOL01"]["close"]
        
        # Time Engle-Granger test
        start = time.time()
        result = tester.test_engle_granger(series1, series2)
        eg_time = time.time() - start
        
        assert eg_time < 0.5  # Should complete in less than 500ms
        assert result.p_value is not None
        
        # Time Johansen test
        start = time.time()
        result = tester.test_johansen([series1, series2])
        joh_time = time.time() - start
        
        assert joh_time < 1.0  # Should complete in less than 1 second
    
    def test_correlation_calculation_speed(self, calculator, large_dataset):
        """Test speed of correlation calculation."""
        series1 = large_dataset["SYMBOL00"]["close"]
        series2 = large_dataset["SYMBOL01"]["close"]
        
        start = time.time()
        metrics = calculator.calculate_correlation(series1, series2)
        calc_time = time.time() - start
        
        assert calc_time < 0.2  # Should complete in less than 200ms
        assert metrics.pearson_correlation is not None
    
    def test_spread_calculation_speed(self, calculator, large_dataset):
        """Test speed of spread calculation."""
        series1 = large_dataset["SYMBOL00"]["close"]
        series2 = large_dataset["SYMBOL01"]["close"]
        
        start = time.time()
        spread = calculator.calculate_spread(series1, series2, hedge_ratio=1.5)
        calc_time = time.time() - start
        
        assert calc_time < 0.3  # Should complete in less than 300ms
        assert len(spread.spread_values) == len(series1)
    
    def test_hedge_ratio_calculation_speed(self, calculator, large_dataset):
        """Test speed of hedge ratio calculation."""
        series1 = large_dataset["SYMBOL00"]["close"]
        series2 = large_dataset["SYMBOL01"]["close"]
        
        # Test OLS method
        start = time.time()
        hedge_ols = calculator.calculate_hedge_ratio(series1, series2, method="ols")
        ols_time = time.time() - start
        
        assert ols_time < 0.1  # Should complete in less than 100ms
        
        # Test rolling method (slower)
        start = time.time()
        hedge_rolling = calculator.calculate_hedge_ratio(series1, series2, method="rolling")
        rolling_time = time.time() - start
        
        assert rolling_time < 2.0  # Should complete in less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_pair_scanning_speed(self, large_dataset):
        """Test speed of scanning multiple pairs."""
        strategy = PairsTradingStrategy(PairsTradingConfig(max_pairs=5))
        strategy.market_data_cache = large_dataset
        
        start = time.time()
        await strategy._scan_for_pairs()
        scan_time = time.time() - start
        
        # With 20 symbols, we have C(20,2) = 190 pairs to test
        assert scan_time < 30.0  # Should complete in less than 30 seconds
        print(f"Scanned 190 pairs in {scan_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, large_dataset):
        """Test concurrent signal generation for multiple pairs."""
        strategy = PairsTradingStrategy(PairsTradingConfig(max_pairs=5))
        
        # Add multiple active pairs
        for i in range(5):
            pair = TradingPair(
                symbol1=f"SYMBOL{i:02d}",
                symbol2=f"SYMBOL{i+1:02d}",
                correlation=Decimal("0.8"),
                current_zscore=Decimal(str(2.5 - i * 0.5))
            )
            strategy.state.active_pairs.append(pair)
        
        strategy.market_data_cache = large_dataset
        
        start = time.time()
        signals = await strategy.generate_signals()
        gen_time = time.time() - start
        
        assert gen_time < 1.0  # Should complete in less than 1 second
        assert isinstance(signals, list)
    
    def test_zscore_calculation_speed(self, calculator):
        """Test z-score calculation speed on large arrays."""
        # Create large spread array
        spread = np.random.normal(0, 1, 10000)
        
        start = time.time()
        z_scores = calculator.calculate_zscore(spread, window=100)
        calc_time = time.time() - start
        
        assert calc_time < 0.1  # Should complete in less than 100ms
        assert len(z_scores) == len(spread)
    
    def test_batch_cointegration_testing(self, tester):
        """Test batch processing of multiple pairs."""
        np.random.seed(42)
        n = 1000
        n_pairs = 50
        
        # Generate pairs
        pairs = []
        for _ in range(n_pairs):
            base = np.cumsum(np.random.normal(0, 0.01, n))
            series1 = base + np.random.normal(0, 0.5, n)
            series2 = base * 1.5 + np.random.normal(0, 0.5, n)
            pairs.append((series1, series2))
        
        # Time batch testing
        start = time.time()
        results = []
        for s1, s2 in pairs:
            result = tester.test_engle_granger(s1, s2)
            results.append(result)
        batch_time = time.time() - start
        
        assert batch_time < 10.0  # Should complete 50 pairs in less than 10 seconds
        assert len(results) == n_pairs
        
        # Check some results are cointegrated
        cointegrated = [r for r in results if r.is_cointegrated]
        assert len(cointegrated) > 0
    
    def test_memory_usage_large_cache(self, large_dataset):
        """Test memory usage with large market data cache."""
        strategy = PairsTradingStrategy(PairsTradingConfig())
        
        # Load all data into cache
        strategy.market_data_cache = large_dataset
        
        # Add many pairs
        for i in range(10):
            pair = TradingPair(
                symbol1=f"SYMBOL{i:02d}",
                symbol2=f"SYMBOL{i+1:02d}"
            )
            strategy.state.active_pairs.append(pair)
        
        # Check memory is managed (cache should have max size)
        for symbol, df in strategy.market_data_cache.items():
            assert len(df) == 10000  # Original size
        
        # Update with new data (should trim old data)
        for symbol in list(strategy.market_data_cache.keys())[:5]:
            new_data = {
                'symbol': symbol,
                'close': 100.0,
                'volume': 50000
            }
            asyncio.run(strategy._update_market_data(symbol, new_data))
        
        # Check data was added and old data trimmed if needed
        for symbol in list(strategy.market_data_cache.keys())[:5]:
            assert len(strategy.market_data_cache[symbol]) <= 10001
    
    def test_spread_quality_analysis_speed(self, calculator):
        """Test speed of spread quality analysis."""
        # Create spread metrics
        spread_values = np.random.normal(0, 1, 5000)
        
        from genesis.analytics.spread_calculator import SpreadMetrics
        
        metrics = SpreadMetrics(
            spread_values=spread_values,
            mean=Decimal("0"),
            std_dev=Decimal("1"),
            current_value=Decimal("0.5"),
            current_zscore=Decimal("0.5"),
            half_life=20,
            hurst_exponent=0.35,
            adf_pvalue=Decimal("0.02"),
            max_zscore=Decimal("3"),
            min_zscore=Decimal("-3"),
            spread_type="log"
        )
        
        start = time.time()
        quality = calculator.analyze_spread_quality(metrics)
        analysis_time = time.time() - start
        
        assert analysis_time < 0.01  # Should complete in less than 10ms
        assert "quality_score" in quality
    
    @pytest.mark.asyncio
    async def test_recalibration_performance(self, large_dataset):
        """Test performance of recalibration process."""
        strategy = PairsTradingStrategy(PairsTradingConfig())
        strategy.market_data_cache = {
            k: v for k, v in list(large_dataset.items())[:5]  # Use subset
        }
        
        # Create pairs needing recalibration
        pairs = []
        for i in range(3):
            pair = TradingPair(
                symbol1=f"SYMBOL{i:02d}",
                symbol2=f"SYMBOL{i+1:02d}",
                last_calibration=pd.Timestamp.now() - pd.Timedelta(days=8)
            )
            pairs.append(pair)
            strategy.state.active_pairs.append(pair)
        
        # Time recalibration
        start = time.time()
        for pair in pairs:
            await strategy._recalibrate_pair(pair)
        recal_time = time.time() - start
        
        assert recal_time < 3.0  # Should complete 3 pairs in less than 3 seconds
    
    def test_parallel_calculations(self, calculator):
        """Test parallel calculation capabilities."""
        # Create multiple series
        n_series = 10
        series_list = []
        for _ in range(n_series):
            series = pd.Series(np.random.randn(1000))
            series_list.append(series)
        
        # Time sequential processing
        start = time.time()
        results_seq = []
        for i in range(0, n_series, 2):
            corr = calculator.calculate_correlation(series_list[i], series_list[i+1])
            results_seq.append(corr)
        seq_time = time.time() - start
        
        assert seq_time < 1.0  # Should complete in reasonable time
        assert len(results_seq) == n_series // 2