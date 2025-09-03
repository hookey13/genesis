"""Unit tests for spread calculator module."""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from genesis.analytics.spread_calculator import (
    CorrelationMetrics,
    HedgeRatio,
    SpreadCalculator,
    SpreadMetrics,
)


class TestCorrelationMetrics:
    """Test CorrelationMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating correlation metrics."""
        metrics = CorrelationMetrics(
            pearson_correlation=Decimal("0.85"),
            spearman_correlation=Decimal("0.83"),
            kendall_correlation=Decimal("0.78"),
            rolling_correlation=np.array([0.8, 0.85, 0.82]),
            correlation_stability=Decimal("0.05"),
            correlation_trend=Decimal("0.001"),
            is_stable=True
        )
        
        assert metrics.pearson_correlation == Decimal("0.85")
        assert metrics.is_stable
        assert len(metrics.rolling_correlation) == 3


class TestHedgeRatio:
    """Test HedgeRatio dataclass."""
    
    def test_hedge_ratio_creation(self):
        """Test creating hedge ratio."""
        ratio = HedgeRatio(
            ratio=Decimal("1.5"),
            method="ols",
            r_squared=Decimal("0.85"),
            confidence_interval=(Decimal("1.4"), Decimal("1.6")),
            stability_score=Decimal("0.8"),
            last_updated=datetime.now()
        )
        
        assert ratio.ratio == Decimal("1.5")
        assert ratio.method == "ols"
        assert ratio.r_squared == Decimal("0.85")


class TestSpreadMetrics:
    """Test SpreadMetrics dataclass."""
    
    def test_spread_metrics_creation(self):
        """Test creating spread metrics."""
        spread_values = np.array([0.1, 0.2, -0.1, 0.0])
        
        metrics = SpreadMetrics(
            spread_values=spread_values,
            mean=Decimal("0.05"),
            std_dev=Decimal("0.12"),
            current_value=Decimal("0.0"),
            current_zscore=Decimal("-0.42"),
            half_life=15,
            hurst_exponent=0.35,
            adf_pvalue=Decimal("0.02"),
            max_zscore=Decimal("2.5"),
            min_zscore=Decimal("-2.1"),
            spread_type="log"
        )
        
        assert len(metrics.spread_values) == 4
        assert metrics.half_life == 15
        assert metrics.hurst_exponent < 0.5  # Mean reverting


class TestSpreadCalculator:
    """Test SpreadCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SpreadCalculator(lookback_window=50)
    
    @pytest.fixture
    def correlated_series(self):
        """Create correlated price series."""
        np.random.seed(42)
        n = 100
        
        # Base series
        base = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
        
        # Correlated series
        noise = np.random.normal(0, 5, n)
        series1 = pd.Series(base + noise)
        series2 = pd.Series(base * 1.5 + noise * 2)
        
        return series1, series2
    
    @pytest.fixture
    def uncorrelated_series(self):
        """Create uncorrelated price series."""
        np.random.seed(42)
        n = 100
        
        series1 = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n))))
        series2 = pd.Series(50 * np.exp(np.cumsum(np.random.normal(0, 0.03, n))))
        
        return series1, series2
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.lookback_window == 50
        assert calculator.min_periods == 10
    
    def test_calculate_correlation_high(self, calculator, correlated_series):
        """Test correlation calculation for highly correlated series."""
        series1, series2 = correlated_series
        
        metrics = calculator.calculate_correlation(series1, series2)
        
        assert metrics.pearson_correlation > Decimal("0.8")
        assert metrics.spearman_correlation > Decimal("0.7")
        assert len(metrics.rolling_correlation) > 0
        assert metrics.is_stable
        assert metrics.metadata["n_observations"] == 100
    
    def test_calculate_correlation_low(self, calculator, uncorrelated_series):
        """Test correlation calculation for uncorrelated series."""
        series1, series2 = uncorrelated_series
        
        metrics = calculator.calculate_correlation(series1, series2)
        
        assert metrics.pearson_correlation < Decimal("0.5")
        assert not metrics.is_stable
    
    def test_calculate_correlation_with_window(self, calculator, correlated_series):
        """Test correlation with custom window."""
        series1, series2 = correlated_series
        
        metrics = calculator.calculate_correlation(series1, series2, window=30)
        
        assert metrics.metadata["window_size"] == 30
    
    def test_calculate_correlation_numpy_input(self, calculator):
        """Test correlation with numpy array input."""
        series1 = np.random.randn(100)
        series2 = series1 + np.random.randn(100) * 0.5
        
        metrics = calculator.calculate_correlation(series1, series2)
        
        assert metrics.pearson_correlation > Decimal("0.5")
    
    def test_calculate_hedge_ratio_ols(self, calculator, correlated_series):
        """Test OLS hedge ratio calculation."""
        series1, series2 = correlated_series
        
        hedge = calculator.calculate_hedge_ratio(series1, series2, method="ols")
        
        assert hedge.method == "ols"
        assert hedge.ratio > Decimal("0")
        assert hedge.r_squared > Decimal("0.5")
        assert hedge.confidence_interval[0] < hedge.ratio < hedge.confidence_interval[1]
        assert hedge.stability_score > Decimal("0")
    
    def test_calculate_hedge_ratio_tls(self, calculator, correlated_series):
        """Test TLS hedge ratio calculation."""
        series1, series2 = correlated_series
        
        hedge = calculator.calculate_hedge_ratio(series1, series2, method="tls")
        
        assert hedge.method == "tls"
        assert hedge.ratio > Decimal("0")
        assert hedge.r_squared > Decimal("0")
    
    def test_calculate_hedge_ratio_rolling(self, calculator, correlated_series):
        """Test rolling hedge ratio calculation."""
        series1, series2 = correlated_series
        
        hedge = calculator.calculate_hedge_ratio(series1, series2, method="rolling")
        
        assert hedge.method == "rolling"
        assert hedge.ratio > Decimal("0")
        assert hedge.stability_score > Decimal("0")
    
    def test_calculate_hedge_ratio_numpy(self, calculator):
        """Test hedge ratio with numpy input."""
        series1 = np.random.randn(100)
        series2 = series1 * 1.5 + np.random.randn(100) * 0.5
        
        hedge = calculator.calculate_hedge_ratio(series1, series2)
        
        # Should be close to 1.5
        assert Decimal("1.2") < hedge.ratio < Decimal("1.8")
    
    def test_calculate_spread_log(self, calculator, correlated_series):
        """Test log spread calculation."""
        series1, series2 = correlated_series
        
        metrics = calculator.calculate_spread(series1, series2, hedge_ratio=1.5, spread_type="log")
        
        assert metrics.spread_type == "log"
        assert len(metrics.spread_values) == 100
        assert metrics.mean is not None
        assert metrics.std_dev > Decimal("0")
        assert metrics.current_zscore is not None
        assert metrics.metadata["is_stationary"] is not None
    
    def test_calculate_spread_ratio(self, calculator, correlated_series):
        """Test ratio spread calculation."""
        series1, series2 = correlated_series
        
        metrics = calculator.calculate_spread(series1, series2, hedge_ratio=1.5, spread_type="ratio")
        
        assert metrics.spread_type == "ratio"
        assert len(metrics.spread_values) == 100
    
    def test_calculate_spread_dollar(self, calculator, correlated_series):
        """Test dollar spread calculation."""
        series1, series2 = correlated_series
        
        metrics = calculator.calculate_spread(series1, series2, hedge_ratio=1.5, spread_type="dollar")
        
        assert metrics.spread_type == "dollar"
        assert len(metrics.spread_values) == 100
    
    def test_calculate_zscore(self, calculator):
        """Test z-score calculation."""
        # Create spread with known properties
        spread = pd.Series([0, 1, -1, 2, -2, 0, 1, -1] * 10)  # 80 values
        
        z_scores = calculator.calculate_zscore(spread, window=20)
        
        assert len(z_scores) == 80
        # Z-scores should be bounded in reasonable range
        assert np.all(np.abs(z_scores[20:]) < 5)  # After initial window
    
    def test_calculate_zscore_numpy(self, calculator):
        """Test z-score with numpy input."""
        spread = np.random.randn(100)
        
        z_scores = calculator.calculate_zscore(spread)
        
        assert len(z_scores) == 100
    
    def test_analyze_spread_quality_good(self, calculator):
        """Test spread quality analysis for good spread."""
        # Create good spread metrics
        spread_values = np.random.normal(0, 1, 100)
        
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
        
        quality = calculator.analyze_spread_quality(metrics)
        
        assert quality["is_tradeable"]
        assert quality["quality_score"] >= 0.5
        assert len(quality["strengths"]) > 0
        assert "Spread is stationary" in quality["strengths"]
        assert "Mean reverting" in quality["strengths"][1]
    
    def test_analyze_spread_quality_poor(self, calculator):
        """Test spread quality analysis for poor spread."""
        # Create poor spread metrics
        metrics = SpreadMetrics(
            spread_values=np.array([]),
            mean=Decimal("0"),
            std_dev=Decimal("1"),
            current_value=Decimal("0"),
            current_zscore=Decimal("0"),
            half_life=None,
            hurst_exponent=0.65,  # Trending
            adf_pvalue=Decimal("0.15"),  # Not stationary
            max_zscore=Decimal("1"),
            min_zscore=Decimal("-1"),
            spread_type="log"
        )
        
        quality = calculator.analyze_spread_quality(metrics)
        
        assert not quality["is_tradeable"]
        assert quality["quality_score"] < 0.5
        assert len(quality["issues"]) > 0
        assert "Spread is not stationary" in quality["issues"]
    
    def test_calculate_half_life_mean_reverting(self, calculator):
        """Test half-life calculation for mean-reverting series."""
        # Create mean-reverting spread
        n = 100
        spread = np.zeros(n)
        theta = 0.1
        for i in range(1, n):
            spread[i] = (1 - theta) * spread[i-1] + np.random.normal(0, 0.5)
        
        half_life = calculator._calculate_half_life(spread)
        
        assert half_life is not None
        assert 5 < half_life < 50
    
    def test_calculate_half_life_random_walk(self, calculator):
        """Test half-life for random walk."""
        spread = np.cumsum(np.random.normal(0, 1, 100))
        
        half_life = calculator._calculate_half_life(spread)
        
        # Should be None or very large
        assert half_life is None or half_life > 100
    
    def test_calculate_hurst_exponent(self, calculator):
        """Test Hurst exponent calculation."""
        # Mean-reverting series
        n = 200
        mr_series = np.zeros(n)
        for i in range(1, n):
            mr_series[i] = 0.5 * mr_series[i-1] + np.random.normal(0, 1)
        
        hurst = calculator._calculate_hurst_exponent(mr_series)
        
        assert 0 <= hurst <= 1
        assert hurst < 0.5  # Should be mean-reverting
    
    def test_error_handling_correlation(self, calculator):
        """Test error handling in correlation calculation."""
        # Empty series
        result = calculator.calculate_correlation(pd.Series([]), pd.Series([]))
        
        assert result.pearson_correlation == Decimal("0")
        assert not result.is_stable
    
    def test_error_handling_hedge_ratio(self, calculator):
        """Test error handling in hedge ratio calculation."""
        # Invalid method
        result = calculator.calculate_hedge_ratio(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3]),
            method="invalid"
        )
        
        assert result.ratio == Decimal("1")
        assert result.r_squared == Decimal("0")
    
    def test_error_handling_spread(self, calculator):
        """Test error handling in spread calculation."""
        # Invalid spread type
        result = calculator.calculate_spread(
            pd.Series([1, 2, 3]),
            pd.Series([1, 2, 3]),
            spread_type="invalid"
        )
        
        assert len(result.spread_values) == 0
        assert result.mean == Decimal("0")