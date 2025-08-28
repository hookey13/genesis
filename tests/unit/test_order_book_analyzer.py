"""Unit tests for the Order Book Analyzer."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from genesis.analytics.order_book_analyzer import (
    LiquidityLevel,
    LiquidityProfile,
    OrderBookAnalyzer,
)
from genesis.engine.executor.base import OrderSide
from genesis.exchange.models import OrderBook


@pytest.fixture
def analyzer():
    """Create an order book analyzer instance."""
    return OrderBookAnalyzer()


@pytest.fixture
def deep_order_book():
    """Create a deep liquidity order book."""
    return OrderBook(
        symbol="BTCUSDT",
        bids=[
            [40000.0, 2.0],  # $80,000
            [39999.0, 3.0],  # $119,997
            [39998.0, 2.5],  # $99,995
            [39997.0, 2.0],  # $79,994
            [39996.0, 1.5],  # $59,994
            [39995.0, 2.0],  # $79,990
            [39994.0, 1.8],  # $71,989
            [39993.0, 2.2],  # $87,985
            [39992.0, 1.5],  # $59,988
            [39991.0, 2.0],  # $79,982
        ],
        asks=[
            [40001.0, 2.0],  # $80,002
            [40002.0, 3.0],  # $120,006
            [40003.0, 2.5],  # $100,008
            [40004.0, 2.0],  # $80,008
            [40005.0, 1.5],  # $60,008
            [40006.0, 2.0],  # $80,012
            [40007.0, 1.8],  # $72,013
            [40008.0, 2.2],  # $88,018
            [40009.0, 1.5],  # $60,014
            [40010.0, 2.0],  # $80,020
        ],
        timestamp=datetime.now(),
    )


@pytest.fixture
def shallow_order_book():
    """Create a shallow liquidity order book."""
    return OrderBook(
        symbol="ETHUSDT",
        bids=[
            [3000.0, 0.1],  # $300
            [2999.0, 0.2],  # $599.8
            [2998.0, 0.15],  # $449.7
            [2997.0, 0.1],  # $299.7
            [2996.0, 0.05],  # $149.8
        ],
        asks=[
            [3001.0, 0.1],  # $300.1
            [3002.0, 0.2],  # $600.4
            [3003.0, 0.15],  # $450.45
            [3004.0, 0.1],  # $300.4
            [3005.0, 0.05],  # $150.25
        ],
        timestamp=datetime.now(),
    )


@pytest.fixture
def imbalanced_order_book():
    """Create an imbalanced order book (more asks than bids)."""
    return OrderBook(
        symbol="SOLUSDT",
        bids=[[100.0, 10.0], [99.9, 5.0], [99.8, 3.0]],  # $1,000  # $499.5  # $299.4
        asks=[
            [100.1, 50.0],  # $5,005
            [100.2, 40.0],  # $4,008
            [100.3, 30.0],  # $3,009
            [100.4, 20.0],  # $2,008
            [100.5, 10.0],  # $1,005
        ],
        timestamp=datetime.now(),
    )


class TestOrderBookAnalyzer:
    """Test suite for OrderBookAnalyzer."""

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer._cache == {}
        assert analyzer.MIN_SLICES == 3
        assert analyzer.MAX_SLICES == 10

    def test_analyze_liquidity_depth_deep_market(self, analyzer, deep_order_book):
        """Test liquidity analysis for deep market."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        assert profile.symbol == "BTCUSDT"
        assert profile.total_bid_volume > 0
        assert profile.total_ask_volume > 0
        assert profile.best_bid == Decimal("40000")
        assert profile.best_ask == Decimal("40001")
        assert profile.spread_percent < Decimal("0.01")  # Very tight spread
        assert profile.liquidity_level in [LiquidityLevel.DEEP, LiquidityLevel.MODERATE]
        assert profile.optimal_slice_count >= analyzer.MIN_SLICES
        assert profile.max_safe_order_size > 0

    def test_analyze_liquidity_depth_shallow_market(self, analyzer, shallow_order_book):
        """Test liquidity analysis for shallow market."""
        profile = analyzer.analyze_liquidity_depth(shallow_order_book)

        assert profile.symbol == "ETHUSDT"
        assert profile.liquidity_level in [
            LiquidityLevel.SHALLOW,
            LiquidityLevel.CRITICAL,
        ]
        assert profile.max_safe_order_size < Decimal("1000")  # Limited safe size
        assert profile.concentration_risk > Decimal("0.5")  # High concentration

    def test_analyze_liquidity_depth_imbalanced_market(
        self, analyzer, imbalanced_order_book
    ):
        """Test liquidity analysis for imbalanced market."""
        profile = analyzer.analyze_liquidity_depth(imbalanced_order_book)

        assert profile.symbol == "SOLUSDT"
        assert profile.imbalance_ratio != 0  # Should show imbalance
        assert profile.total_ask_volume > profile.total_bid_volume

    def test_calculate_optimal_slice_count_small_order(self, analyzer, deep_order_book):
        """Test slice calculation for small orders."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        # Small order (below safe size)
        slices = analyzer.calculate_optimal_slice_count(
            profile.max_safe_order_size * Decimal("0.5"), profile
        )
        assert slices == analyzer.MIN_SLICES

    def test_calculate_optimal_slice_count_large_order(self, analyzer, deep_order_book):
        """Test slice calculation for large orders."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        # Large order (3x safe size)
        slices = analyzer.calculate_optimal_slice_count(
            profile.max_safe_order_size * Decimal("3"), profile
        )
        assert slices > analyzer.MIN_SLICES
        assert slices <= analyzer.MAX_SLICES

    def test_calculate_optimal_slice_count_critical_liquidity(self, analyzer):
        """Test slice calculation with critical liquidity."""
        # Create a critical liquidity profile
        profile = LiquidityProfile(
            symbol="TEST",
            timestamp=datetime.now(),
            total_bid_volume=Decimal("100"),
            total_ask_volume=Decimal("100"),
            imbalance_ratio=Decimal("0"),
            bid_depth_0_5pct=Decimal("50"),
            bid_depth_1pct=Decimal("75"),
            bid_depth_2pct=Decimal("100"),
            ask_depth_0_5pct=Decimal("50"),
            ask_depth_1pct=Decimal("75"),
            ask_depth_2pct=Decimal("100"),
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            spread_absolute=Decimal("1"),
            spread_percent=Decimal("1"),
            liquidity_level=LiquidityLevel.CRITICAL,
            optimal_slice_count=10,
            max_safe_order_size=Decimal("10"),
            expected_slippage_1x=Decimal("0.5"),
            expected_slippage_2x=Decimal("1.5"),
            concentration_risk=Decimal("0.8"),
            depth_consistency=Decimal("0.2"),
        )

        # Large order in critical liquidity
        slices = analyzer.calculate_optimal_slice_count(Decimal("100"), profile)
        assert slices == analyzer.MAX_SLICES  # Should use maximum slicing

    def test_depth_calculation(self, analyzer, deep_order_book):
        """Test depth level calculations."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        # Verify depth calculations
        assert profile.bid_depth_0_5pct > 0
        assert profile.bid_depth_1pct > profile.bid_depth_0_5pct
        assert profile.bid_depth_2pct > profile.bid_depth_1pct

        assert profile.ask_depth_0_5pct > 0
        assert profile.ask_depth_1pct > profile.ask_depth_0_5pct
        assert profile.ask_depth_2pct > profile.ask_depth_1pct

    def test_spread_calculation(self, analyzer, deep_order_book):
        """Test spread calculations."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        expected_spread = Decimal("40001") - Decimal("40000")
        assert profile.spread_absolute == expected_spread

        expected_spread_pct = (expected_spread / Decimal("40000")) * Decimal("100")
        assert abs(profile.spread_percent - expected_spread_pct) < Decimal("0.001")

    def test_concentration_risk_calculation(self, analyzer, shallow_order_book):
        """Test concentration risk calculation."""
        profile = analyzer.analyze_liquidity_depth(shallow_order_book)

        # Shallow book should have higher concentration risk
        assert profile.concentration_risk > Decimal("0.3")
        assert profile.concentration_risk <= Decimal("1")

    def test_depth_consistency_calculation(self, analyzer, deep_order_book):
        """Test depth consistency calculation."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        # Well-distributed book should have decent consistency
        assert profile.depth_consistency >= Decimal("0")
        assert profile.depth_consistency <= Decimal("1")

    def test_slippage_estimation(self, analyzer, deep_order_book):
        """Test slippage estimation."""
        profile = analyzer.analyze_liquidity_depth(deep_order_book)

        # Slippage should increase with order size
        assert profile.expected_slippage_1x >= Decimal("0")
        assert profile.expected_slippage_2x > profile.expected_slippage_1x

    def test_cache_functionality(self, analyzer, deep_order_book):
        """Test caching of liquidity profiles."""
        # First call - should analyze and cache
        profile1 = analyzer.analyze_liquidity_depth(deep_order_book)
        assert "BTCUSDT" in analyzer._cache

        # Second call - should return cached result
        profile2 = analyzer.analyze_liquidity_depth(deep_order_book)
        assert profile1.timestamp == profile2.timestamp  # Same cached result

        # Clear cache
        analyzer.clear_cache("BTCUSDT")
        assert "BTCUSDT" not in analyzer._cache

        # Third call - should re-analyze
        profile3 = analyzer.analyze_liquidity_depth(deep_order_book)
        assert profile3.timestamp > profile1.timestamp

    def test_cache_expiry(self, analyzer, deep_order_book):
        """Test cache TTL expiry."""
        # Analyze and cache
        profile1 = analyzer.analyze_liquidity_depth(deep_order_book)

        # Manually expire the cache
        cached_profile, _ = analyzer._cache["BTCUSDT"]
        expired_time = datetime.now() - timedelta(seconds=10)
        analyzer._cache["BTCUSDT"] = (cached_profile, expired_time)

        # Should re-analyze due to expiry
        profile2 = analyzer.analyze_liquidity_depth(deep_order_book)
        assert profile2.timestamp > profile1.timestamp

    def test_clear_all_cache(self, analyzer, deep_order_book, shallow_order_book):
        """Test clearing all cached profiles."""
        # Cache multiple profiles
        analyzer.analyze_liquidity_depth(deep_order_book)
        analyzer.analyze_liquidity_depth(shallow_order_book)

        assert len(analyzer._cache) == 2

        # Clear all
        analyzer.clear_cache()
        assert len(analyzer._cache) == 0

    def test_empty_order_book(self, analyzer):
        """Test handling of empty order book."""
        empty_book = OrderBook(
            symbol="EMPTY", bids=[], asks=[], timestamp=datetime.now()
        )

        profile = analyzer.analyze_liquidity_depth(empty_book)

        assert profile.liquidity_level == LiquidityLevel.CRITICAL
        assert profile.total_bid_volume == Decimal("0")
        assert profile.total_ask_volume == Decimal("0")
        assert profile.optimal_slice_count == analyzer.MIN_SLICES

    def test_one_sided_order_book(self, analyzer):
        """Test handling of one-sided order book."""
        one_sided_book = OrderBook(
            symbol="ONESIDED",
            bids=[[100.0, 10.0]],
            asks=[],  # No asks
            timestamp=datetime.now(),
        )

        profile = analyzer.analyze_liquidity_depth(one_sided_book)

        assert profile.liquidity_level == LiquidityLevel.CRITICAL
        assert profile.total_ask_volume == Decimal("0")

    def test_side_specific_analysis(self, analyzer, deep_order_book):
        """Test analysis focused on specific side."""
        # Analyze for buy side
        buy_profile = analyzer.analyze_liquidity_depth(deep_order_book, OrderSide.BUY)
        assert buy_profile.ask_depth_1pct > 0  # Should analyze ask side for buys

        # Analyze for sell side
        sell_profile = analyzer.analyze_liquidity_depth(deep_order_book, OrderSide.SELL)
        assert sell_profile.bid_depth_1pct > 0  # Should analyze bid side for sells

    def test_minimum_slices_enforcement(self, analyzer):
        """Test that minimum 3 slices are always recommended."""
        profile = LiquidityProfile(
            symbol="TEST",
            timestamp=datetime.now(),
            total_bid_volume=Decimal("10000"),
            total_ask_volume=Decimal("10000"),
            imbalance_ratio=Decimal("0"),
            bid_depth_0_5pct=Decimal("1000"),
            bid_depth_1pct=Decimal("2000"),
            bid_depth_2pct=Decimal("4000"),
            ask_depth_0_5pct=Decimal("1000"),
            ask_depth_1pct=Decimal("2000"),
            ask_depth_2pct=Decimal("4000"),
            best_bid=Decimal("100"),
            best_ask=Decimal("100.1"),
            spread_absolute=Decimal("0.1"),
            spread_percent=Decimal("0.1"),
            liquidity_level=LiquidityLevel.DEEP,
            optimal_slice_count=5,
            max_safe_order_size=Decimal("1000"),
            expected_slippage_1x=Decimal("0.1"),
            expected_slippage_2x=Decimal("0.3"),
            concentration_risk=Decimal("0.2"),
            depth_consistency=Decimal("0.8"),
        )

        # Very small order
        slices = analyzer.calculate_optimal_slice_count(Decimal("50"), profile)
        assert slices >= 3  # AC: Ensure minimum 3 slices

    def test_calculate_depth_to_price_levels(self, analyzer):
        """Test internal depth calculation method."""
        bids = [[100.0, 10.0], [99.5, 15.0], [99.0, 20.0], [98.5, 25.0]]

        # Test bid side depth calculation
        depths = analyzer._calculate_depth_levels(bids, Decimal("100"), is_bid=True)

        assert len(depths) == 3  # 0.5%, 1%, 2%
        assert all(d >= 0 for d in depths)
        assert depths[2] >= depths[1] >= depths[0]  # Should be increasing
