"""
Unit tests for the SmartRouter class.

Tests intelligent order routing decisions, market condition analysis,
and execution quality scoring.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.models import TradingTier
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.engine.executor.smart_router import (
    ExtendedOrderType,
    LiquidityLevel,
    MarketConditions,
    OrderBook,
    SmartRouter,
    TimeFactor,
    UrgencyLevel,
)


@pytest.fixture
def mock_exchange_gateway():
    """Create a mock exchange gateway."""
    gateway = MagicMock()
    gateway.get_order_book = AsyncMock()
    gateway.get_ticker = AsyncMock()
    return gateway


@pytest.fixture
def smart_router(mock_exchange_gateway):
    """Create a SmartRouter instance with mock gateway."""
    router = SmartRouter(mock_exchange_gateway)
    # Set tier to HUNTER to allow access to tier-restricted methods
    router.tier = TradingTier.HUNTER
    return router


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id="test-order-1",
        position_id="pos-1",
        client_order_id="client-1",
        symbol="BTC/USDT",
        type=OrderType.LIMIT,
        side=OrderSide.BUY,
        price=Decimal("50000"),
        quantity=Decimal("0.01"),
        filled_quantity=Decimal("0"),
        status=OrderStatus.PENDING,
    )


@pytest.fixture
def sample_order_book():
    """Create a sample order book."""
    return OrderBook(
        symbol="BTC/USDT",
        bids=[
            (Decimal("49995"), Decimal("1.5")),
            (Decimal("49990"), Decimal("2.0")),
            (Decimal("49985"), Decimal("3.0")),
        ],
        asks=[
            (Decimal("50005"), Decimal("1.2")),
            (Decimal("50010"), Decimal("1.8")),
            (Decimal("50015"), Decimal("2.5")),
        ],
        timestamp=datetime.now(),
    )


@pytest.fixture
def tight_spread_conditions():
    """Create market conditions with tight spread."""
    return MarketConditions(
        spread_percent=Decimal("0.0003"),  # 0.03%
        bid_liquidity=Decimal("100"),
        ask_liquidity=Decimal("100"),
        liquidity_level=LiquidityLevel.DEEP,
        time_factor=TimeFactor.US_OPEN,
        volatility=Decimal("0.02"),
        order_book_imbalance=Decimal("0"),
        timestamp=datetime.now(),
    )


@pytest.fixture
def wide_spread_conditions():
    """Create market conditions with wide spread."""
    return MarketConditions(
        spread_percent=Decimal("0.003"),  # 0.3%
        bid_liquidity=Decimal("50"),
        ask_liquidity=Decimal("50"),
        liquidity_level=LiquidityLevel.NORMAL,
        time_factor=TimeFactor.ASIA_OPEN,
        volatility=Decimal("0.03"),
        order_book_imbalance=Decimal("0.1"),
        timestamp=datetime.now(),
    )


@pytest.fixture
def volatile_conditions():
    """Create high volatility market conditions."""
    return MarketConditions(
        spread_percent=Decimal("0.001"),
        bid_liquidity=Decimal("20"),
        ask_liquidity=Decimal("20"),
        liquidity_level=LiquidityLevel.SHALLOW,
        time_factor=TimeFactor.EUROPE_OPEN,
        volatility=Decimal("0.08"),  # 8% volatility
        order_book_imbalance=Decimal("-0.2"),
        timestamp=datetime.now(),
    )


@pytest.fixture
def thin_liquidity_conditions():
    """Create thin liquidity market conditions."""
    return MarketConditions(
        spread_percent=Decimal("0.002"),
        bid_liquidity=Decimal("5"),
        ask_liquidity=Decimal("5"),
        liquidity_level=LiquidityLevel.THIN,
        time_factor=TimeFactor.QUIET,
        volatility=Decimal("0.04"),
        order_book_imbalance=Decimal("0.5"),
        timestamp=datetime.now(),
    )


class TestMarketConditionAnalysis:
    """Test market condition analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_market_conditions(self, smart_router, mock_exchange_gateway):
        """Test analyzing market conditions from order book and ticker."""
        # Setup mock data
        mock_exchange_gateway.get_order_book.return_value = {
            "bids": [[49995, 1.5], [49990, 2.0]],
            "asks": [[50005, 1.2], [50010, 1.8]],
        }
        mock_exchange_gateway.get_ticker.return_value = {"priceChangePercent": "3.5"}

        conditions = await smart_router.analyze_market_conditions("BTC/USDT")

        assert conditions.spread_percent > 0
        assert conditions.bid_liquidity == Decimal("3.5")
        assert conditions.ask_liquidity == Decimal("3.0")
        assert conditions.volatility == Decimal("0.035")
        assert conditions.timestamp is not None

    @pytest.mark.asyncio
    async def test_market_conditions_caching(self, smart_router, mock_exchange_gateway):
        """Test that market conditions are cached for TTL period."""
        mock_exchange_gateway.get_order_book.return_value = {
            "bids": [[49995, 1.5]],
            "asks": [[50005, 1.2]],
        }
        mock_exchange_gateway.get_ticker.return_value = {"priceChangePercent": "2.0"}

        # First call should fetch from exchange
        conditions1 = await smart_router.analyze_market_conditions("BTC/USDT")
        assert mock_exchange_gateway.get_order_book.call_count == 1

        # Second immediate call should use cache
        conditions2 = await smart_router.analyze_market_conditions("BTC/USDT")
        assert mock_exchange_gateway.get_order_book.call_count == 1
        assert conditions1.timestamp == conditions2.timestamp

    @pytest.mark.asyncio
    async def test_market_conditions_error_handling(
        self, smart_router, mock_exchange_gateway
    ):
        """Test fallback to conservative defaults on API error."""
        mock_exchange_gateway.get_order_book.side_effect = Exception("API Error")

        conditions = await smart_router.analyze_market_conditions("BTC/USDT")

        # Should return conservative defaults
        assert conditions.liquidity_level == LiquidityLevel.THIN
        assert conditions.volatility == Decimal("0.01")
        assert conditions.spread_percent == Decimal("0.001")

    def test_calculate_spread_percentage(self, smart_router, sample_order_book):
        """Test spread percentage calculation."""
        spread = smart_router.calculate_spread_percentage(sample_order_book)

        # Spread = (50005 - 49995) / 50000 * 100 = 0.02%
        expected = Decimal("0.02")
        assert abs(spread - expected) < Decimal("0.001")

    def test_calculate_spread_percentage_no_market(self, smart_router):
        """Test spread calculation with empty order book."""
        empty_book = OrderBook(
            symbol="BTC/USDT", bids=[], asks=[], timestamp=datetime.now()
        )

        spread = smart_router.calculate_spread_percentage(empty_book)
        assert spread == Decimal("999")  # No market indicator

    def test_assess_liquidity_depth(self, smart_router, sample_order_book):
        """Test liquidity depth assessment."""
        # Small order relative to liquidity
        level = smart_router.assess_liquidity_depth(sample_order_book, Decimal("0.01"))
        assert level == LiquidityLevel.DEEP

        # Large order relative to liquidity
        level = smart_router.assess_liquidity_depth(sample_order_book, Decimal("10"))
        assert level == LiquidityLevel.THIN

    def test_get_time_of_day_factor(self, smart_router):
        """Test time of day factor determination."""
        with patch("genesis.engine.executor.smart_router.datetime") as mock_datetime:
            # Test Asia open
            mock_datetime.utcnow.return_value.hour = 4
            assert smart_router.get_time_of_day_factor() == TimeFactor.ASIA_OPEN

            # Test Europe open
            mock_datetime.utcnow.return_value.hour = 10
            assert smart_router.get_time_of_day_factor() == TimeFactor.EUROPE_OPEN

            # Test US open
            mock_datetime.utcnow.return_value.hour = 16
            assert smart_router.get_time_of_day_factor() == TimeFactor.US_OPEN

            # Test quiet hours
            mock_datetime.utcnow.return_value.hour = 22
            assert smart_router.get_time_of_day_factor() == TimeFactor.QUIET

    def test_estimate_market_impact(self, smart_router):
        """Test market impact estimation."""
        # Small order in deep liquidity
        impact = smart_router.estimate_market_impact(
            Decimal("0.1"), LiquidityLevel.DEEP
        )
        assert impact < Decimal("0.001")

        # Large order in thin liquidity
        impact = smart_router.estimate_market_impact(Decimal("10"), LiquidityLevel.THIN)
        assert impact > Decimal("0.01")

        # Test impact cap at 5%
        impact = smart_router.estimate_market_impact(
            Decimal("10000"), LiquidityLevel.THIN
        )
        assert impact == Decimal("5")


class TestOrderTypeSelection:
    """Test order type selection logic."""

    def test_high_urgency_selection(
        self, smart_router, sample_order, tight_spread_conditions
    ):
        """Test that high urgency always selects market orders."""
        order_type = smart_router.select_order_type(
            sample_order, tight_spread_conditions, UrgencyLevel.HIGH
        )
        assert order_type == ExtendedOrderType.MARKET

    def test_tight_spread_deep_liquidity(
        self, smart_router, sample_order, tight_spread_conditions
    ):
        """Test order selection with tight spread and deep liquidity."""
        order_type = smart_router.select_order_type(
            sample_order, tight_spread_conditions, UrgencyLevel.NORMAL
        )
        assert order_type == ExtendedOrderType.MARKET

    def test_tight_spread_thin_liquidity(self, smart_router, sample_order):
        """Test order selection with tight spread but thin liquidity."""
        conditions = MarketConditions(
            spread_percent=Decimal("0.0003"),
            bid_liquidity=Decimal("5"),
            ask_liquidity=Decimal("5"),
            liquidity_level=LiquidityLevel.THIN,
            time_factor=TimeFactor.US_OPEN,
            volatility=Decimal("0.02"),
            order_book_imbalance=Decimal("0"),
            timestamp=datetime.now(),
        )

        order_type = smart_router.select_order_type(
            sample_order, conditions, UrgencyLevel.NORMAL
        )
        assert order_type == ExtendedOrderType.IOC

    def test_wide_spread_low_urgency(
        self, smart_router, sample_order, wide_spread_conditions
    ):
        """Test order selection with wide spread and low urgency."""
        order_type = smart_router.select_order_type(
            sample_order, wide_spread_conditions, UrgencyLevel.LOW
        )
        assert order_type == ExtendedOrderType.POST_ONLY

    def test_wide_spread_normal_urgency(
        self, smart_router, sample_order, wide_spread_conditions
    ):
        """Test order selection with wide spread and normal urgency."""
        order_type = smart_router.select_order_type(
            sample_order, wide_spread_conditions, UrgencyLevel.NORMAL
        )
        assert order_type == ExtendedOrderType.LIMIT

    def test_high_volatility_thin_liquidity(self, smart_router, sample_order):
        """Test FOK selection in high volatility with thin liquidity."""
        conditions = MarketConditions(
            spread_percent=Decimal("0.001"),
            bid_liquidity=Decimal("5"),
            ask_liquidity=Decimal("5"),
            liquidity_level=LiquidityLevel.THIN,
            time_factor=TimeFactor.US_OPEN,
            volatility=Decimal("0.08"),  # High volatility
            order_book_imbalance=Decimal("0"),
            timestamp=datetime.now(),
        )

        order_type = smart_router.select_order_type(
            sample_order, conditions, UrgencyLevel.NORMAL
        )
        assert order_type == ExtendedOrderType.FOK

    def test_high_volatility_good_liquidity(self, smart_router, sample_order):
        """Test IOC selection in high volatility with good liquidity."""
        conditions = MarketConditions(
            spread_percent=Decimal("0.001"),
            bid_liquidity=Decimal("100"),
            ask_liquidity=Decimal("100"),
            liquidity_level=LiquidityLevel.DEEP,
            time_factor=TimeFactor.US_OPEN,
            volatility=Decimal("0.08"),  # High volatility
            order_book_imbalance=Decimal("0"),
            timestamp=datetime.now(),
        )

        order_type = smart_router.select_order_type(
            sample_order, conditions, UrgencyLevel.NORMAL
        )
        assert order_type == ExtendedOrderType.IOC

    def test_fee_optimization_selection(self, smart_router, sample_order):
        """Test POST_ONLY selection for fee optimization."""
        conditions = MarketConditions(
            spread_percent=Decimal("0.0008"),  # Medium spread
            bid_liquidity=Decimal("50"),
            ask_liquidity=Decimal("50"),
            liquidity_level=LiquidityLevel.NORMAL,
            time_factor=TimeFactor.US_OPEN,
            volatility=Decimal("0.02"),  # Low volatility
            order_book_imbalance=Decimal("0"),
            timestamp=datetime.now(),
        )

        order_type = smart_router.select_order_type(
            sample_order, conditions, UrgencyLevel.LOW
        )
        assert order_type == ExtendedOrderType.POST_ONLY


class TestExecutionScoring:
    """Test execution quality scoring."""

    def test_perfect_execution_score(
        self, smart_router, sample_order, tight_spread_conditions
    ):
        """Test score for perfect execution."""
        result = ExecutionResult(
            success=True,
            order=sample_order,
            message="Order executed",
            actual_price=Decimal("50000"),
            slippage_percent=Decimal("0"),
            latency_ms=50,
        )

        score = smart_router.calculate_execution_score(
            sample_order, result, tight_spread_conditions
        )
        assert score > 95  # Near perfect score

    def test_high_slippage_penalty(
        self, smart_router, sample_order, tight_spread_conditions
    ):
        """Test score penalty for high slippage."""
        result = ExecutionResult(
            success=True,
            order=sample_order,
            message="Order executed",
            actual_price=Decimal("50500"),
            slippage_percent=Decimal("1.0"),  # 1% slippage
            latency_ms=100,
        )

        score = smart_router.calculate_execution_score(
            sample_order, result, tight_spread_conditions
        )
        assert score < 90  # Penalized for slippage

    def test_high_latency_penalty(
        self, smart_router, sample_order, tight_spread_conditions
    ):
        """Test score penalty for high latency."""
        result = ExecutionResult(
            success=True,
            order=sample_order,
            message="Order executed",
            actual_price=Decimal("50000"),
            slippage_percent=Decimal("0"),
            latency_ms=1500,  # Very high latency
        )

        score = smart_router.calculate_execution_score(
            sample_order, result, tight_spread_conditions
        )
        assert score <= 80  # Heavily penalized for latency

    def test_favorable_execution_bonus(
        self, smart_router, sample_order, volatile_conditions
    ):
        """Test bonus for favorable execution in volatile markets."""
        result = ExecutionResult(
            success=True,
            order=sample_order,
            message="Order executed",
            actual_price=Decimal("49900"),
            slippage_percent=Decimal("-0.2"),  # Favorable slippage
            latency_ms=100,
        )

        score = smart_router.calculate_execution_score(
            sample_order, result, volatile_conditions
        )
        assert score > 100  # Bonus applied (capped at 100)

    def test_post_only_success_bonus(
        self, smart_router, sample_order, wide_spread_conditions
    ):
        """Test bonus for successful post-only execution."""
        sample_order.routing_method = "POST_ONLY"

        result = ExecutionResult(
            success=True,
            order=sample_order,
            message="Order executed as maker",
            actual_price=Decimal("50000"),
            slippage_percent=Decimal("0"),
            latency_ms=100,
        )

        score = smart_router.calculate_execution_score(
            sample_order, result, wide_spread_conditions
        )
        assert score > 95  # Includes post-only bonus


class TestOrderRouting:
    """Test complete order routing flow."""

    @pytest.mark.asyncio
    async def test_route_order_high_urgency(
        self, smart_router, sample_order, mock_exchange_gateway
    ):
        """Test routing with high urgency."""
        mock_exchange_gateway.get_order_book.return_value = {
            "bids": [[49995, 1.5]],
            "asks": [[50005, 1.2]],
        }
        mock_exchange_gateway.get_ticker.return_value = {"priceChangePercent": "2.0"}

        routed = await smart_router.route_order(sample_order, UrgencyLevel.HIGH)

        assert routed.selected_type == ExtendedOrderType.MARKET
        assert "high urgency" in routed.routing_reason.lower()
        assert routed.expected_fee_rate == smart_router.TAKER_FEE_RATE

    @pytest.mark.asyncio
    async def test_route_order_fee_optimization(
        self, smart_router, sample_order, mock_exchange_gateway
    ):
        """Test routing for fee optimization."""
        # Setup wide spread conditions
        mock_exchange_gateway.get_order_book.return_value = {
            "bids": [[49990, 1.5]],
            "asks": [[50010, 1.2]],
        }
        mock_exchange_gateway.get_ticker.return_value = {"priceChangePercent": "1.5"}

        routed = await smart_router.route_order(sample_order, UrgencyLevel.LOW)

        assert routed.selected_type == ExtendedOrderType.POST_ONLY
        assert "optimizing maker fees" in routed.routing_reason.lower()
        assert routed.expected_fee_rate == smart_router.MAKER_FEE_RATE

    @pytest.mark.asyncio
    async def test_route_order_with_metadata(
        self, smart_router, sample_order, mock_exchange_gateway
    ):
        """Test that routing adds metadata to order."""
        mock_exchange_gateway.get_order_book.return_value = {
            "bids": [[49995, 1.5]],
            "asks": [[50005, 1.2]],
        }
        mock_exchange_gateway.get_ticker.return_value = {"priceChangePercent": "2.0"}

        routed = await smart_router.route_order(sample_order, UrgencyLevel.NORMAL)

        assert hasattr(routed.order, "routing_method")
        assert routed.order.routing_method == routed.selected_type.value
        assert routed.market_conditions is not None
        assert routed.post_only_retry_count == 0


class TestTierRestriction:
    """Test tier-based access restrictions."""

    def test_requires_hunter_tier(self):
        """Test that SmartRouter requires HUNTER tier."""
        # Check that the router has a required tier
        assert hasattr(SmartRouter, "REQUIRED_TIER")
        assert SmartRouter.REQUIRED_TIER == TradingTier.HUNTER

    def test_smart_router_initialization(self, mock_exchange_gateway):
        """Test SmartRouter can be initialized with proper tier."""
        router = SmartRouter(mock_exchange_gateway)
        assert router.exchange_gateway == mock_exchange_gateway
        assert router._conditions_cache == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
