"""Unit tests for cointegration testing module."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from genesis.analytics.cointegration import CointegrationResult, CointegrationTester


class TestCointegrationResult:
    """Test CointegrationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating cointegration result."""
        result = CointegrationResult(
            test_type="ADF",
            is_cointegrated=True,
            p_value=Decimal("0.03"),
            test_statistic=Decimal("-3.5"),
            critical_values={"5%": Decimal("-2.9")},
            hedge_ratio=Decimal("1.5"),
            half_life=15
        )
        
        assert result.test_type == "ADF"
        assert result.is_cointegrated
        assert result.p_value == Decimal("0.03")
        assert result.test_statistic == Decimal("-3.5")
        assert result.hedge_ratio == Decimal("1.5")
        assert result.half_life == 15


class TestCointegrationTester:
    """Test CointegrationTester class."""
    
    @pytest.fixture
    def tester(self):
        """Create tester instance."""
        return CointegrationTester(confidence_level=0.95)
    
    @pytest.fixture
    def stationary_series(self):
        """Create a stationary series."""
        np.random.seed(42)
        # Mean-reverting series
        n = 100
        series = np.zeros(n)
        for i in range(1, n):
            series[i] = 0.5 * series[i-1] + np.random.normal(0, 1)
        return pd.Series(series)
    
    @pytest.fixture
    def random_walk_series(self):
        """Create a random walk (non-stationary) series."""
        np.random.seed(42)
        returns = np.random.normal(0, 1, 100)
        return pd.Series(np.cumsum(returns))
    
    @pytest.fixture
    def cointegrated_pair(self):
        """Create a cointegrated pair of series."""
        np.random.seed(42)
        n = 100
        
        # Generate first series (random walk)
        series1 = np.cumsum(np.random.normal(0, 1, n))
        
        # Generate second series cointegrated with first
        hedge_ratio = 1.5
        noise = np.random.normal(0, 0.5, n)
        series2 = series1 / hedge_ratio + noise
        
        return pd.Series(series1), pd.Series(series2)
    
    @pytest.fixture
    def non_cointegrated_pair(self):
        """Create a non-cointegrated pair."""
        np.random.seed(42)
        n = 100
        
        # Two independent random walks
        series1 = np.cumsum(np.random.normal(0, 1, n))
        series2 = np.cumsum(np.random.normal(0, 1, n))
        
        return pd.Series(series1), pd.Series(series2)
    
    def test_tester_initialization(self, tester):
        """Test tester initialization."""
        assert tester.confidence_level == 0.95
        assert tester.p_value_threshold == Decimal("0.05")
    
    def test_adf_stationary(self, tester, stationary_series):
        """Test ADF test on stationary series."""
        result = tester.test_adf(stationary_series)
        
        assert result.test_type == "ADF"
        assert result.is_cointegrated  # Stationary = cointegrated for single series
        assert result.p_value < Decimal("0.05")
        assert "1%" in result.critical_values
        assert "5%" in result.critical_values
        assert "10%" in result.critical_values
    
    def test_adf_non_stationary(self, tester, random_walk_series):
        """Test ADF test on non-stationary series."""
        result = tester.test_adf(random_walk_series)
        
        assert result.test_type == "ADF"
        assert not result.is_cointegrated
        assert result.p_value > Decimal("0.05")
    
    def test_adf_with_numpy_array(self, tester, stationary_series):
        """Test ADF test with numpy array input."""
        result = tester.test_adf(stationary_series.values)
        
        assert result.test_type == "ADF"
        assert result.is_cointegrated
    
    def test_engle_granger_cointegrated(self, tester, cointegrated_pair):
        """Test Engle-Granger test on cointegrated pair."""
        series1, series2 = cointegrated_pair
        result = tester.test_engle_granger(series1, series2)
        
        assert result.test_type == "Engle-Granger"
        assert result.is_cointegrated
        assert result.p_value < Decimal("0.05")
        assert result.hedge_ratio is not None
        assert result.half_life is not None
        assert result.metadata is not None
        assert "r_squared" in result.metadata
    
    def test_engle_granger_non_cointegrated(self, tester, non_cointegrated_pair):
        """Test Engle-Granger test on non-cointegrated pair."""
        series1, series2 = non_cointegrated_pair
        result = tester.test_engle_granger(series1, series2)
        
        assert result.test_type == "Engle-Granger"
        assert not result.is_cointegrated
        assert result.p_value > Decimal("0.05")
    
    def test_engle_granger_with_numpy(self, tester, cointegrated_pair):
        """Test Engle-Granger with numpy arrays."""
        series1, series2 = cointegrated_pair
        result = tester.test_engle_granger(series1.values, series2.values)
        
        assert result.test_type == "Engle-Granger"
        assert result.hedge_ratio is not None
    
    def test_johansen_cointegrated(self, tester, cointegrated_pair):
        """Test Johansen test on cointegrated series."""
        series1, series2 = cointegrated_pair
        result = tester.test_johansen([series1, series2])
        
        assert result.test_type == "Johansen"
        # Johansen test may have different results due to its methodology
        assert result.p_value is not None
        assert "trace_95%" in result.critical_values
        assert "max_eigen_95%" in result.critical_values
        assert result.metadata is not None
        assert "n_cointegrating_relations" in result.metadata
    
    def test_johansen_multiple_series(self, tester):
        """Test Johansen with multiple series."""
        np.random.seed(42)
        n = 100
        
        # Create three related series
        base = np.cumsum(np.random.normal(0, 1, n))
        series1 = pd.Series(base + np.random.normal(0, 0.5, n))
        series2 = pd.Series(base * 1.5 + np.random.normal(0, 0.5, n))
        series3 = pd.Series(base * 0.8 + np.random.normal(0, 0.5, n))
        
        result = tester.test_johansen([series1, series2, series3])
        
        assert result.test_type == "Johansen"
        assert result.metadata["eigenvalues"] is not None
    
    def test_phillips_ouliaris(self, tester, cointegrated_pair):
        """Test Phillips-Ouliaris test."""
        series1, series2 = cointegrated_pair
        result = tester.test_phillips_ouliaris(series1, series2)
        
        assert result.test_type == "Phillips-Ouliaris"
        assert result.hedge_ratio is not None
        assert result.metadata is not None
    
    def test_calculate_half_life(self, tester):
        """Test half-life calculation."""
        np.random.seed(42)
        n = 100
        
        # Create mean-reverting spread
        spread = np.zeros(n)
        theta = 0.1  # Mean reversion speed
        for i in range(1, n):
            spread[i] = (1 - theta) * spread[i-1] + np.random.normal(0, 0.5)
        
        half_life = tester._calculate_half_life(spread)
        
        assert half_life is not None
        assert half_life > 0
        # Expected half-life approximately -log(2)/log(1-theta)
        expected = int(-np.log(2) / np.log(1 - theta))
        assert abs(half_life - expected) < 10  # Allow some variance
    
    def test_calculate_half_life_no_mean_reversion(self, tester):
        """Test half-life with no mean reversion."""
        # Random walk has no mean reversion
        spread = np.cumsum(np.random.normal(0, 1, 100))
        half_life = tester._calculate_half_life(spread)
        
        # Should be None or very large
        assert half_life is None or half_life > 100
    
    def test_calculate_hurst_exponent(self, tester):
        """Test Hurst exponent calculation."""
        np.random.seed(42)
        
        # Mean-reverting series (H < 0.5)
        n = 200
        mr_series = np.zeros(n)
        for i in range(1, n):
            mr_series[i] = 0.5 * mr_series[i-1] + np.random.normal(0, 1)
        
        hurst_mr = tester.calculate_hurst_exponent(mr_series)
        assert hurst_mr < Decimal("0.5")
        
        # Random walk (H ≈ 0.5)
        rw_series = np.cumsum(np.random.normal(0, 1, n))
        hurst_rw = tester.calculate_hurst_exponent(rw_series)
        assert Decimal("0.4") < hurst_rw < Decimal("0.6")
        
        # Trending series (H > 0.5)
        trend = np.linspace(0, 10, n) + np.random.normal(0, 0.1, n)
        hurst_trend = tester.calculate_hurst_exponent(trend)
        assert hurst_trend > Decimal("0.5")
    
    def test_validate_cointegration(self, tester, cointegrated_pair):
        """Test validation with multiple methods."""
        series1, series2 = cointegrated_pair
        
        results = tester.validate_cointegration(
            series1, series2,
            methods=["engle_granger", "phillips_ouliaris"]
        )
        
        assert "engle_granger" in results
        assert "phillips_ouliaris" in results
        assert results["engle_granger"].test_type == "Engle-Granger"
        assert results["phillips_ouliaris"].test_type == "Phillips-Ouliaris"
        
        # Check Hurst exponent was added
        for result in results.values():
            assert "hurst_exponent" in result.metadata
    
    def test_validate_cointegration_all_methods(self, tester, cointegrated_pair):
        """Test validation with all methods."""
        series1, series2 = cointegrated_pair
        
        results = tester.validate_cointegration(series1, series2)
        
        assert "engle_granger" in results
        assert "phillips_ouliaris" in results
        assert "johansen" in results
    
    def test_error_handling_adf(self, tester):
        """Test error handling in ADF test."""
        # Test with invalid input
        result = tester.test_adf([])  # Empty array
        
        assert result.test_type == "ADF"
        assert not result.is_cointegrated
        assert result.p_value == Decimal("1")
    
    def test_error_handling_engle_granger(self, tester):
        """Test error handling in Engle-Granger test."""
        # Test with mismatched lengths
        series1 = pd.Series([1, 2, 3])
        series2 = pd.Series([1, 2])
        
        result = tester.test_engle_granger(series1, series2)
        
        # Should handle gracefully
        assert result.test_type == "Engle-Granger"
    
    def test_error_handling_johansen(self, tester):
        """Test error handling in Johansen test."""
        # Test with too few observations
        series1 = pd.Series([1, 2])
        series2 = pd.Series([1, 2])
        
        result = tester.test_johansen([series1, series2])
        
        assert result.test_type == "Johansen"
        assert not result.is_cointegrated