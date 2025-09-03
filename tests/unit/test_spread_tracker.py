"""
Unit tests for Enhanced Spread Tracker
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from genesis.analytics.spread_tracker_enhanced import (
    EnhancedSpreadTracker,
    SpreadTrackerConfig,
)
from genesis.core.exceptions import ValidationError


@pytest.fixture
def config():
    """Create test configuration"""
    return SpreadTrackerConfig(
        spread_window=100,
        baseline_period=300,
        volatility_halflife=60,
        anomaly_threshold=Decimal("3.0"),
        cache_ttl=30,
        max_memory_mb=10,
        ema_alpha=Decimal("0.2")
    )


@pytest.fixture
def tracker(config):
    """Create spread tracker instance"""
    return EnhancedSpreadTracker(config)


@pytest.fixture
def sample_market_data():
    """Generate sample market data"""
    return {
        "symbol": "BTC/USDT",
        "bid_price": Decimal("50000"),
        "ask_price": Decimal("50010"),
        "bid_volume": Decimal("10"),
        "ask_volume": Decimal("12")
    }


class TestSpreadTrackerInitialization:
    """Test spread tracker initialization"""

    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        tracker = EnhancedSpreadTracker()
        assert tracker.config.spread_window == 1000
        assert tracker.config.baseline_period == 3600
        assert tracker.config.max_memory_mb == 50

    def test_init_with_custom_config(self, config):
        """Test initialization with custom configuration"""
        tracker = EnhancedSpreadTracker(config)
        assert tracker.config.spread_window == 100
        assert tracker.config.baseline_period == 300
        assert tracker.config.max_memory_mb == 10

    def test_data_structures_initialized(self, tracker):
        """Test that all data structures are properly initialized"""
        assert hasattr(tracker, "_spread_history")
        assert hasattr(tracker, "_timestamp_history")
        assert hasattr(tracker, "_volume_history")
        assert hasattr(tracker, "_ema_spreads")
        assert hasattr(tracker, "_metrics_cache")


class TestSpreadCalculation:
    """Test spread calculation accuracy"""

    @pytest.mark.asyncio
    async def test_basic_spread_calculation(self, tracker, sample_market_data):
        """Test basic spread calculation in basis points"""
        metrics = await tracker.update_spread(**sample_market_data)

        # Expected spread: (50010 - 50000) / 50005 * 10000 = 1.9998 bps
        expected_spread = Decimal("1.9998000599880024")
        assert abs(metrics.current_spread_bps - expected_spread) < Decimal("0.001")

    @pytest.mark.asyncio
    async def test_spread_with_different_prices(self, tracker):
        """Test spread calculation with various price levels"""
        test_cases = [
            (Decimal("100"), Decimal("101"), Decimal("99.502")),  # ~100 bps
            (Decimal("1000"), Decimal("1001"), Decimal("9.995")),  # ~10 bps
            (Decimal("10000"), Decimal("10002"), Decimal("1.9998")),  # ~2 bps
        ]

        for bid, ask, expected_bps in test_cases:
            metrics = await tracker.update_spread(
                symbol="TEST",
                bid_price=bid,
                ask_price=ask,
                bid_volume=Decimal("1"),
                ask_volume=Decimal("1")
            )
            assert abs(metrics.current_spread_bps - expected_bps) < Decimal("1")

    @pytest.mark.asyncio
    async def test_invalid_prices_rejected(self, tracker):
        """Test that invalid prices are rejected"""
        # Negative price
        with pytest.raises(ValidationError):
            await tracker.update_spread(
                symbol="TEST",
                bid_price=Decimal("-100"),
                ask_price=Decimal("101"),
                bid_volume=Decimal("1"),
                ask_volume=Decimal("1")
            )

        # Ask <= Bid
        with pytest.raises(ValidationError):
            await tracker.update_spread(
                symbol="TEST",
                bid_price=Decimal("100"),
                ask_price=Decimal("99"),
                bid_volume=Decimal("1"),
                ask_volume=Decimal("1")
            )


class TestBaselineCalculation:
    """Test historical baseline calculation"""

    @pytest.mark.asyncio
    async def test_ema_spread_calculation(self, tracker):
        """Test exponential moving average spread calculation"""
        symbol = "BTC/USDT"

        # Add multiple spread updates
        for i in range(10):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000") + Decimal(str(i)),
                ask_price=Decimal("50010") + Decimal(str(i)),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        # Check EMA is calculated
        assert symbol in tracker._ema_spreads
        ema = tracker._ema_spreads[symbol]
        assert ema > Decimal("0")

    @pytest.mark.asyncio
    async def test_percentile_calculation(self, tracker):
        """Test percentile-based baseline calculation"""
        symbol = "BTC/USDT"

        # Add spread data with known distribution
        spreads = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]

        for _i, spread_offset in enumerate(spreads):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50000") + spread_offset,
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        metrics = await tracker.get_cached_metrics(symbol)

        # Check percentiles
        assert metrics.percentile_25 > Decimal("0")
        assert metrics.percentile_50 > metrics.percentile_25
        assert metrics.percentile_75 > metrics.percentile_50

    @pytest.mark.asyncio
    async def test_baseline_with_time_period(self, tracker):
        """Test baseline calculation with specific time period"""
        symbol = "BTC/USDT"

        # Add historical data
        for i in range(20):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010") + Decimal(str(i)),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        # Calculate baseline for last 60 seconds
        baseline = await tracker.calculate_baseline(symbol, period_seconds=60)

        assert "mean" in baseline
        assert "median" in baseline
        assert "std_dev" in baseline
        assert baseline["mean"] > Decimal("0")


class TestVolatilityMeasurement:
    """Test spread volatility measurement"""

    @pytest.mark.asyncio
    async def test_standard_deviation_calculation(self, tracker):
        """Test standard deviation calculation"""
        symbol = "BTC/USDT"

        # Add spreads with known variance
        prices = [Decimal("50000"), Decimal("50005"), Decimal("50010"), Decimal("50005")]

        for price in prices:
            await tracker.update_spread(
                symbol=symbol,
                bid_price=price,
                ask_price=price + Decimal("10"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        metrics = await tracker.get_cached_metrics(symbol)
        assert metrics.std_deviation > Decimal("0")

    @pytest.mark.asyncio
    async def test_ewma_volatility(self, tracker):
        """Test EWMA volatility calculation"""
        symbol = "BTC/USDT"

        # Add volatile spread data
        for i in range(20):
            offset = Decimal("10") if i % 2 == 0 else Decimal("20")
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50000") + offset,
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        metrics = await tracker.get_cached_metrics(symbol)
        assert metrics.ewma_volatility > Decimal("0")

    @pytest.mark.asyncio
    async def test_volatility_regime_detection(self, tracker):
        """Test volatility regime classification"""
        symbol = "BTC/USDT"

        # Create stable market conditions
        for _ in range(30):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        metrics = await tracker.get_cached_metrics(symbol)
        assert metrics.volatility_regime in ["low", "normal", "high", "extreme"]

        # Create volatile conditions
        for i in range(10):
            offset = Decimal(str(i * 10))
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010") + offset,
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        metrics = await tracker.get_cached_metrics(symbol)
        assert metrics.volatility_regime in ["normal", "high", "extreme"]


class TestWeightedSpreadCalculations:
    """Test VWAP and TWAP spread calculations"""

    @pytest.mark.asyncio
    async def test_vwap_spread_calculation(self, tracker):
        """Test volume-weighted average spread"""
        symbol = "BTC/USDT"

        # Add spreads with different volumes
        test_data = [
            (Decimal("50000"), Decimal("50010"), Decimal("10"), Decimal("10")),  # Small volume
            (Decimal("50000"), Decimal("50020"), Decimal("100"), Decimal("100")),  # Large volume
            (Decimal("50000"), Decimal("50015"), Decimal("50"), Decimal("50")),  # Medium volume
        ]

        for bid, ask, bid_vol, ask_vol in test_data:
            await tracker.update_spread(
                symbol=symbol,
                bid_price=bid,
                ask_price=ask,
                bid_volume=bid_vol,
                ask_volume=ask_vol
            )

        metrics = await tracker.get_cached_metrics(symbol)

        # VWAP should be weighted towards high-volume spread
        assert metrics.vwap_spread > Decimal("0")
        # Should be closer to the high-volume spread (20 bps) than simple average
        assert abs(metrics.vwap_spread - Decimal("3")) < Decimal("1")

    @pytest.mark.asyncio
    async def test_twap_spread_calculation(self, tracker):
        """Test time-weighted average spread"""
        symbol = "BTC/USDT"

        # Add spreads at different time intervals
        for i in range(5):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010") + Decimal(str(i)),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )
            await asyncio.sleep(0.01)  # Small delay for time weighting

        metrics = await tracker.get_cached_metrics(symbol)
        assert metrics.twap_spread > Decimal("0")

    @pytest.mark.asyncio
    async def test_effective_spread_calculation(self, tracker):
        """Test effective spread with market impact"""
        symbol = "BTC/USDT"

        await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50020"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        metrics = await tracker.get_cached_metrics(symbol)

        # Effective spread should account for typical slippage
        assert metrics.effective_spread > Decimal("0")
        assert metrics.effective_spread != metrics.current_spread_bps


class TestAnomalyDetection:
    """Test spread anomaly detection"""

    @pytest.mark.asyncio
    async def test_anomaly_detection_with_outlier(self, tracker):
        """Test detection of anomalous spreads"""
        symbol = "BTC/USDT"

        # Create normal spread pattern with low variance for consistent baseline
        for _ in range(20):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        # Add anomalous spread - needs to be extreme enough to exceed z-score threshold of 3
        # Normal spread is ~2 bps, so 100x that should definitely trigger anomaly
        metrics = await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("51000"),  # 100x normal spread (2000 bps vs 2 bps)
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        assert metrics.is_anomaly is True
        assert metrics.anomaly_score > tracker.config.anomaly_threshold

    @pytest.mark.asyncio
    async def test_no_anomaly_in_normal_conditions(self, tracker):
        """Test that normal spreads are not flagged as anomalies"""
        symbol = "BTC/USDT"

        # Create consistent spread pattern
        for i in range(20):
            metrics = await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010") + Decimal(str(i % 2)),  # Small variation
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

            if i > 10:  # After enough history
                assert metrics.is_anomaly is False


class TestDataPersistence:
    """Test data persistence and caching"""

    @pytest.mark.asyncio
    async def test_metrics_caching(self, tracker):
        """Test that metrics are properly cached"""
        symbol = "BTC/USDT"

        # Update spread
        original_metrics = await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        # Get cached metrics
        cached_metrics = await tracker.get_cached_metrics(symbol)

        assert cached_metrics is not None
        assert cached_metrics.current_spread_bps == original_metrics.current_spread_bps

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(self, tracker):
        """Test that cache expires after TTL"""
        symbol = "BTC/USDT"
        tracker.config.cache_ttl = 0  # Immediate expiry

        await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        # Manually set cache timestamp to past
        tracker._cache_timestamps[symbol] = datetime.now(UTC) - timedelta(seconds=10)

        cached_metrics = await tracker.get_cached_metrics(symbol)
        assert cached_metrics is None

    @pytest.mark.asyncio
    async def test_memory_cleanup(self, tracker):
        """Test memory cleanup of old data"""
        symbol = "BTC/USDT"

        # Add lots of data
        for i in range(100):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

            # Add hourly baseline data
            tracker._hourly_baselines[symbol][i % 24].append(Decimal(str(i)))

        # Trigger cleanup
        await tracker._cleanup_old_data()

        # Check that data is trimmed
        for hour in tracker._hourly_baselines[symbol].values():
            assert len(hour) <= 100


class TestSpreadChangeDetection:
    """Test spread change rate detection"""

    @pytest.mark.asyncio
    async def test_spread_change_rate_calculation(self, tracker):
        """Test calculation of spread change rate"""
        symbol = "BTC/USDT"

        # First update
        await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        # Second update with wider spread
        metrics = await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50020"),  # Doubled spread
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        # Should show ~100% increase
        assert abs(metrics.spread_change_rate - Decimal("100")) < Decimal("1")


class TestConcurrency:
    """Test thread-safety with concurrent updates"""

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, tracker):
        """Test that concurrent updates are handled safely"""
        async def update_spread(symbol, index):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000") + Decimal(str(index)),
                ask_price=Decimal("50010") + Decimal(str(index)),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        # Create concurrent tasks
        tasks = [
            update_spread("BTC/USDT", i)
            for i in range(10)
        ]

        # Execute concurrently
        await asyncio.gather(*tasks)

        # Check that all updates were processed
        assert len(tracker._spread_history["BTC/USDT"]) == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_single_data_point(self, tracker):
        """Test behavior with only one data point"""
        symbol = "BTC/USDT"

        metrics = await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        # Should handle gracefully
        assert metrics.current_spread_bps > Decimal("0")
        assert metrics.std_deviation == Decimal("0")  # No variance with one point
        assert metrics.spread_change_rate == Decimal("0")  # No previous to compare

    @pytest.mark.asyncio
    async def test_very_tight_spread(self, tracker):
        """Test handling of very tight spreads"""
        symbol = "BTC/USDT"

        metrics = await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("50000.01"),  # 0.01 spread
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        assert metrics.current_spread_bps > Decimal("0")
        assert metrics.current_spread_bps < Decimal("1")  # Very small

    @pytest.mark.asyncio
    async def test_very_wide_spread(self, tracker):
        """Test handling of very wide spreads"""
        symbol = "BTC/USDT"

        # First establish normal spreads with low variance
        for _ in range(20):
            await tracker.update_spread(
                symbol=symbol,
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10")
            )

        # Add very wide spread - must be extreme to exceed z-score threshold of 3
        # With low variance baseline, 100x normal spread should trigger anomaly
        metrics = await tracker.update_spread(
            symbol=symbol,
            bid_price=Decimal("50000"),
            ask_price=Decimal("51000"),  # 1000 spread (100x normal of 10)
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10")
        )

        assert metrics.current_spread_bps > Decimal("190")  # ~200 bps (1000 spread on 50500 midpoint)
        assert metrics.is_anomaly is True  # Should be detected as anomaly

