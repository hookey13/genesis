"""
Unit tests for Sharpe Ratio Calculator

Tests calculation accuracy, edge cases, and performance requirements.
"""

from decimal import Decimal

import numpy as np
import pytest

from genesis.analytics.sharpe_ratio import (
    SharpeRatioCalculator,
    SharpeRatioResult,
    TimePeriod,
)
from genesis.core.exceptions import DataError as InvalidDataError


class TestSharpeRatioCalculator:
    """Test suite for Sharpe ratio calculations"""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance"""
        return SharpeRatioCalculator()

    @pytest.fixture
    def sample_returns(self):
        """Generate sample return data"""
        # Simulate 100 daily returns with positive expected value
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% mean, 2% std dev
        return [Decimal(str(r)) for r in returns]

    @pytest.fixture
    def negative_returns(self):
        """Generate negative return data"""
        np.random.seed(43)
        returns = np.random.normal(-0.002, 0.03, 50)  # -0.2% mean, 3% std dev
        return [Decimal(str(r)) for r in returns]

    @pytest.mark.asyncio
    async def test_basic_sharpe_calculation(self, calculator, sample_returns):
        """Test basic Sharpe ratio calculation"""
        result = await calculator.calculate_sharpe_ratio(
            sample_returns,
            risk_free_rate=Decimal("0.02"),
            period=TimePeriod.DAILY
        )

        assert isinstance(result, SharpeRatioResult)
        assert result.sharpe_ratio is not None
        assert result.mean_return is not None
        assert result.std_deviation is not None
        assert result.num_periods == 100
        assert result.period == TimePeriod.DAILY

    @pytest.mark.asyncio
    async def test_negative_sharpe_ratio(self, calculator, negative_returns):
        """Test calculation with negative returns"""
        result = await calculator.calculate_sharpe_ratio(
            negative_returns,
            risk_free_rate=Decimal("0.02"),
            period=TimePeriod.DAILY
        )

        # Should have negative Sharpe ratio
        assert result.sharpe_ratio < Decimal("0")
        assert result.mean_return < Decimal("0")

    @pytest.mark.asyncio
    async def test_zero_volatility(self, calculator):
        """Test handling of zero volatility"""
        # All returns are the same
        constant_returns = [Decimal("0.01")] * 30

        result = await calculator.calculate_sharpe_ratio(
            constant_returns,
            risk_free_rate=Decimal("0.00"),
            period=TimePeriod.DAILY
        )

        # Should handle zero volatility gracefully
        assert result.std_deviation == Decimal("0")
        assert result.sharpe_ratio == Decimal("999")  # Max value for positive excess return

    @pytest.mark.asyncio
    async def test_different_time_periods(self, calculator, sample_returns):
        """Test calculation for different time periods"""
        periods = [
            TimePeriod.DAILY,
            TimePeriod.WEEKLY,
            TimePeriod.MONTHLY,
            TimePeriod.YEARLY
        ]

        results = {}
        for period in periods:
            result = await calculator.calculate_sharpe_ratio(
                sample_returns,
                risk_free_rate=Decimal("0.02"),
                period=period
            )
            results[period] = result

        # Verify all calculations completed
        assert len(results) == 4

        # Daily should have different annualization than yearly
        assert results[TimePeriod.DAILY].sharpe_ratio != results[TimePeriod.YEARLY].sharpe_ratio

    @pytest.mark.asyncio
    async def test_confidence_intervals(self, calculator, sample_returns):
        """Test confidence interval calculation"""
        result = await calculator.calculate_sharpe_ratio(
            sample_returns,
            confidence_level=0.95
        )

        # Should have confidence intervals
        assert result.confidence_interval_lower is not None
        assert result.confidence_interval_upper is not None

        # CI should contain the point estimate
        assert result.confidence_interval_lower <= result.sharpe_ratio
        assert result.confidence_interval_upper >= result.sharpe_ratio

    @pytest.mark.asyncio
    async def test_insufficient_data_confidence_intervals(self, calculator):
        """Test that CI is not calculated with insufficient data"""
        small_returns = [Decimal("0.01"), Decimal("-0.02")] * 10  # Only 20 periods

        result = await calculator.calculate_sharpe_ratio(
            small_returns,
            confidence_level=0.95
        )

        # Should not calculate CI with < 30 periods
        assert result.confidence_interval_lower is None
        assert result.confidence_interval_upper is None

    @pytest.mark.asyncio
    async def test_rolling_sharpe(self, calculator, sample_returns):
        """Test rolling Sharpe ratio calculation"""
        window_size = 20

        results = await calculator.calculate_rolling_sharpe(
            sample_returns,
            window_size=window_size
        )

        # Should have correct number of windows
        expected_windows = len(sample_returns) - window_size + 1
        assert len(results) == expected_windows

        # Each result should be valid
        for result in results:
            assert isinstance(result, SharpeRatioResult)
            assert result.num_periods == window_size

    @pytest.mark.asyncio
    async def test_caching(self, calculator, sample_returns):
        """Test that results are cached"""
        # First calculation
        result1 = await calculator.calculate_sharpe_ratio(sample_returns)

        # Second calculation with same data
        result2 = await calculator.calculate_sharpe_ratio(sample_returns)

        # Should return same result (from cache)
        assert result1.sharpe_ratio == result2.sharpe_ratio
        assert result1.calculated_at == result2.calculated_at  # Same timestamp

    @pytest.mark.asyncio
    async def test_cache_clear(self, calculator, sample_returns):
        """Test cache clearing"""
        # Calculate and cache
        await calculator.calculate_sharpe_ratio(sample_returns)

        # Clear cache
        await calculator.clear_cache()

        # Recalculate - should not use cache
        result = await calculator.calculate_sharpe_ratio(sample_returns)
        assert result is not None

    @pytest.mark.asyncio
    async def test_invalid_returns_data(self, calculator):
        """Test validation of invalid return data"""
        # Empty returns
        with pytest.raises(InvalidDataError):
            await calculator.calculate_sharpe_ratio([])

        # Single return
        with pytest.raises(InvalidDataError):
            await calculator.calculate_sharpe_ratio([Decimal("0.01")])

    @pytest.mark.asyncio
    async def test_extreme_values(self, calculator):
        """Test handling of extreme values"""
        # Very large returns
        large_returns = [Decimal("10"), Decimal("-10")] * 20

        result = await calculator.calculate_sharpe_ratio(large_returns)
        assert result is not None

        # Very small returns
        small_returns = [Decimal("0.000001"), Decimal("-0.000001")] * 30

        result = await calculator.calculate_sharpe_ratio(small_returns)
        assert result is not None

    @pytest.mark.asyncio
    async def test_annualization_factors(self, calculator):
        """Test correct annualization factors"""
        # Create returns with known statistics
        returns = [Decimal("0.01")] * 50 + [Decimal("-0.01")] * 50

        daily_result = await calculator.calculate_sharpe_ratio(
            returns,
            period=TimePeriod.DAILY,
            risk_free_rate=Decimal("0")
        )

        weekly_result = await calculator.calculate_sharpe_ratio(
            returns,
            period=TimePeriod.WEEKLY,
            risk_free_rate=Decimal("0")
        )

        # Weekly Sharpe should be different due to annualization
        assert daily_result.sharpe_ratio != weekly_result.sharpe_ratio

    @pytest.mark.asyncio
    async def test_rolling_window_too_large(self, calculator):
        """Test error handling for rolling window larger than data"""
        returns = [Decimal("0.01")] * 10

        with pytest.raises(InvalidDataError):
            await calculator.calculate_rolling_sharpe(
                returns,
                window_size=20  # Larger than data
            )

    @pytest.mark.asyncio
    async def test_performance_requirement(self, calculator):
        """Test calculation completes within performance requirements"""
        # Generate large dataset
        large_returns = [Decimal(str(np.random.normal(0.001, 0.02)))
                        for _ in range(1000)]

        import time
        start = time.time()

        result = await calculator.calculate_sharpe_ratio(
            large_returns,
            confidence_level=0.95  # Include bootstrap CI
        )

        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 5.0  # Should complete within 5 seconds timeout
