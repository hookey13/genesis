"""
Unit tests for execution quality tracking and scoring.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.analytics.execution_quality import (
    ExecutionQuality,
    ExecutionQualityTracker,
    ExecutionScorer,
    ExecutionStats,
)
from genesis.core.models import Order, OrderSide, OrderStatus, OrderType


@pytest.fixture
def sample_order():
    """Create a sample completed order."""
    return Order(
        order_id="test-order-1",
        client_order_id="client-1",
        symbol="BTC/USDT",
        type=OrderType.LIMIT,
        side=OrderSide.BUY,
        price=Decimal("50000"),
        quantity=Decimal("0.01"),
        filled_quantity=Decimal("0.01"),
        status=OrderStatus.FILLED,
        routing_method="SMART",
        maker_fee_paid=Decimal("0.00001"),  # 0.1% maker fee
        taker_fee_paid=Decimal("0"),
        created_at=datetime.now() - timedelta(seconds=2),
        executed_at=datetime.now(),
    )


@pytest.fixture
def scorer():
    """Create an ExecutionScorer instance."""
    return ExecutionScorer()


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.save_execution_quality = AsyncMock()
    repo.get_execution_qualities = AsyncMock()
    return repo


@pytest.fixture
def tracker(mock_repository):
    """Create an ExecutionQualityTracker instance."""
    return ExecutionQualityTracker(mock_repository)


class TestExecutionScorer:
    """Test the ExecutionScorer class."""

    def test_calculate_slippage_bps_buy_unfavorable(self, scorer):
        """Test slippage calculation for buy order with unfavorable price."""
        slippage = scorer._calculate_slippage_bps(
            expected_price=Decimal("50000"),
            actual_price=Decimal("50100"),  # Paid more
            side=OrderSide.BUY,
        )
        assert slippage == Decimal("20")  # 20 basis points unfavorable

    def test_calculate_slippage_bps_buy_favorable(self, scorer):
        """Test slippage calculation for buy order with favorable price."""
        slippage = scorer._calculate_slippage_bps(
            expected_price=Decimal("50000"),
            actual_price=Decimal("49900"),  # Paid less
            side=OrderSide.BUY,
        )
        assert slippage == Decimal("-20")  # 20 basis points favorable

    def test_calculate_slippage_bps_sell_unfavorable(self, scorer):
        """Test slippage calculation for sell order with unfavorable price."""
        slippage = scorer._calculate_slippage_bps(
            expected_price=Decimal("50000"),
            actual_price=Decimal("49900"),  # Sold for less
            side=OrderSide.SELL,
        )
        assert slippage == Decimal("20")  # 20 basis points unfavorable

    def test_calculate_slippage_bps_sell_favorable(self, scorer):
        """Test slippage calculation for sell order with favorable price."""
        slippage = scorer._calculate_slippage_bps(
            expected_price=Decimal("50000"),
            actual_price=Decimal("50100"),  # Sold for more
            side=OrderSide.SELL,
        )
        assert slippage == Decimal("-20")  # 20 basis points favorable

    def test_calculate_price_improvement_buy(self, scorer):
        """Test price improvement calculation for buy order."""
        improvement = scorer._calculate_price_improvement_bps(
            expected_price=Decimal("50000"),
            actual_price=Decimal("49950"),  # Bought below mid
            market_mid_price=Decimal("50000"),
            side=OrderSide.BUY,
        )
        assert improvement == Decimal("10")  # 10 bps improvement

    def test_calculate_price_improvement_sell(self, scorer):
        """Test price improvement calculation for sell order."""
        improvement = scorer._calculate_price_improvement_bps(
            expected_price=Decimal("50000"),
            actual_price=Decimal("50050"),  # Sold above mid
            market_mid_price=Decimal("50000"),
            side=OrderSide.SELL,
        )
        assert improvement == Decimal("10")  # 10 bps improvement

    def test_score_slippage_excellent(self, scorer):
        """Test slippage scoring for excellent execution."""
        score = scorer._score_slippage(Decimal("3"))  # 3 bps
        assert score == 100.0

    def test_score_slippage_good(self, scorer):
        """Test slippage scoring for good execution."""
        score = scorer._score_slippage(Decimal("15"))  # 15 bps
        assert 60 < score < 80

    def test_score_slippage_poor(self, scorer):
        """Test slippage scoring for poor execution."""
        score = scorer._score_slippage(Decimal("75"))  # 75 bps
        assert score < 20

    def test_score_fees_maker(self, scorer):
        """Test fee scoring for maker execution."""
        score = scorer._score_fees(Decimal("5"))  # 5 bps (maker rate)
        assert score == 100.0

    def test_score_fees_taker(self, scorer):
        """Test fee scoring for taker execution."""
        score = scorer._score_fees(Decimal("10"))  # 10 bps (taker rate)
        assert score == 80.0

    def test_score_time_excellent(self, scorer):
        """Test time scoring for excellent speed."""
        score = scorer._score_time_to_fill(50)  # 50ms
        assert score == 100.0

    def test_score_time_good(self, scorer):
        """Test time scoring for good speed."""
        score = scorer._score_time_to_fill(300)  # 300ms
        assert 80 < score < 90

    def test_score_time_poor(self, scorer):
        """Test time scoring for poor speed."""
        score = scorer._score_time_to_fill(3000)  # 3 seconds
        assert score < 40

    def test_score_fill_rate_complete(self, scorer):
        """Test fill rate scoring for complete fill."""
        score = scorer._score_fill_rate(Decimal("100"))
        assert score == 100.0

    def test_score_fill_rate_partial(self, scorer):
        """Test fill rate scoring for partial fill."""
        score = scorer._score_fill_rate(Decimal("85"))
        assert score < 80

    def test_calculate_score_perfect_execution(self, scorer, sample_order):
        """Test scoring for perfect execution."""
        score, quality = scorer.calculate_score(
            order=sample_order,
            actual_price=Decimal("50000"),  # No slippage
            time_to_fill_ms=50,  # Very fast
            market_mid_price=Decimal("50000"),
        )

        assert score > 95  # Near perfect score
        assert quality.order_id == sample_order.order_id
        assert quality.slippage_bps == Decimal("0")
        assert quality.time_to_fill_ms == 50

    def test_calculate_score_with_slippage(self, scorer, sample_order):
        """Test scoring with slippage."""
        score, quality = scorer.calculate_score(
            order=sample_order,
            actual_price=Decimal("50200"),  # 40 bps slippage
            time_to_fill_ms=500,
            market_mid_price=Decimal("50100"),
        )

        assert score < 80  # Lower score due to slippage
        assert quality.slippage_bps == Decimal("40")

    def test_calculate_score_with_high_fees(self, scorer, sample_order):
        """Test scoring with high fees."""
        sample_order.taker_fee_paid = Decimal("0.00005")  # 0.5% fee

        score, quality = scorer.calculate_score(
            order=sample_order,
            actual_price=Decimal("50000"),
            time_to_fill_ms=100,
            market_mid_price=Decimal("50000"),
        )

        assert score < 90  # Lower due to high fees
        assert quality.total_fees == Decimal("0.00006")


class TestExecutionQualityTracker:
    """Test the ExecutionQualityTracker class."""

    @pytest.mark.asyncio
    async def test_track_execution(self, tracker, sample_order):
        """Test tracking an order execution."""
        score = await tracker.track_execution(
            order=sample_order,
            actual_price=Decimal("50000"),
            time_to_fill_ms=100,
            market_mid_price=Decimal("50000"),
        )

        assert score > 0
        assert len(tracker._quality_cache) == 1
        assert sample_order.execution_score == score

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, tracker):
        """Test getting statistics with no data."""
        stats = await tracker.get_statistics("1h")

        assert stats.total_orders == 0
        assert stats.avg_slippage_bps == Decimal("0")
        assert stats.avg_execution_score == 0.0

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, tracker, sample_order):
        """Test getting statistics with tracked executions."""
        # Track a few executions
        await tracker.track_execution(
            sample_order, Decimal("50000"), 100, Decimal("50000")
        )

        sample_order.order_id = "test-order-2"
        await tracker.track_execution(
            sample_order, Decimal("50100"), 200, Decimal("50000")  # Some slippage
        )

        stats = await tracker.get_statistics("1h")

        assert stats.total_orders == 2
        assert stats.avg_slippage_bps > 0
        assert stats.avg_time_to_fill_ms == 150
        assert stats.avg_execution_score > 0
        assert stats.best_execution_score > stats.worst_execution_score

    @pytest.mark.asyncio
    async def test_get_statistics_by_symbol(self, tracker, sample_order):
        """Test filtering statistics by symbol."""
        # Track orders for different symbols
        await tracker.track_execution(
            sample_order, Decimal("50000"), 100, Decimal("50000")
        )

        sample_order.order_id = "test-order-2"
        sample_order.symbol = "ETH/USDT"
        await tracker.track_execution(
            sample_order, Decimal("3000"), 150, Decimal("3000")
        )

        # Get stats for BTC only
        stats = await tracker.get_statistics("1h", "BTC/USDT")
        assert stats.total_orders == 1

        # Get stats for ETH only
        stats = await tracker.get_statistics("1h", "ETH/USDT")
        assert stats.total_orders == 1

    @pytest.mark.asyncio
    async def test_get_statistics_time_periods(self, tracker, sample_order):
        """Test different time period calculations."""
        # Track an old execution
        old_quality = ExecutionQuality(
            order_id="old-order",
            symbol="BTC/USDT",
            order_type="LIMIT",
            routing_method="SMART",
            timestamp=datetime.now() - timedelta(days=2),
            slippage_bps=Decimal("10"),
            total_fees=Decimal("0.0001"),
            maker_fees=Decimal("0.0001"),
            taker_fees=Decimal("0"),
            time_to_fill_ms=100,
            fill_rate=Decimal("100"),
            price_improvement_bps=Decimal("5"),
            execution_score=85.0,
            market_conditions=None,
        )
        tracker._quality_cache.append(old_quality)

        # Track a recent execution
        await tracker.track_execution(
            sample_order, Decimal("50000"), 100, Decimal("50000")
        )

        # Get 24h stats - should only include recent
        stats_24h = await tracker.get_statistics("24h")
        assert stats_24h.total_orders == 1

        # Get 7d stats - should include both
        stats_7d = await tracker.get_statistics("7d")
        assert stats_7d.total_orders == 2

    def test_generate_report(self, tracker):
        """Test report generation."""
        stats = ExecutionStats(
            period="24h",
            total_orders=100,
            avg_slippage_bps=Decimal("15"),
            total_fees=Decimal("0.5"),
            avg_maker_fees=Decimal("0.002"),
            avg_taker_fees=Decimal("0.003"),
            avg_time_to_fill_ms=250,
            avg_fill_rate=Decimal("98.5"),
            price_improvement_rate=Decimal("35"),
            avg_execution_score=82.5,
            best_execution_score=98.0,
            worst_execution_score=45.0,
            rejection_rate=Decimal("2"),
            orders_by_type={"LIMIT": 60, "MARKET": 30, "POST_ONLY": 10},
            orders_by_routing={"SMART": 70, "DIRECT": 30},
        )

        report = tracker.generate_report(stats)

        assert "Total Orders: 100" in report
        assert "Average Execution Score: 82.5/100" in report
        assert "Average Slippage: 15.0 bps" in report
        assert "LIMIT: 60 (60.0%)" in report
        assert "SMART: 70 (70.0%)" in report


class TestExecutionQuality:
    """Test ExecutionQuality dataclass."""

    def test_execution_quality_creation(self):
        """Test creating an ExecutionQuality instance."""
        quality = ExecutionQuality(
            order_id="test-1",
            symbol="BTC/USDT",
            order_type="LIMIT",
            routing_method="SMART",
            timestamp=datetime.now(),
            slippage_bps=Decimal("10"),
            total_fees=Decimal("0.0001"),
            maker_fees=Decimal("0.0001"),
            taker_fees=Decimal("0"),
            time_to_fill_ms=100,
            fill_rate=Decimal("100"),
            price_improvement_bps=Decimal("5"),
            execution_score=85.0,
            market_conditions='{"spread": 0.001}',
        )

        assert quality.order_id == "test-1"
        assert quality.slippage_bps == Decimal("10")
        assert quality.execution_score == 85.0
        assert "spread" in quality.market_conditions


class TestIntegrationScenarios:
    """Test complete execution quality scenarios."""

    @pytest.mark.asyncio
    async def test_market_order_execution(self, tracker):
        """Test tracking a market order execution."""
        market_order = Order(
            order_id="market-1",
            client_order_id="client-market-1",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,  # Market orders don't have price
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            routing_method="MARKET",
            taker_fee_paid=Decimal("0.0001"),  # Always taker
            maker_fee_paid=Decimal("0"),
            created_at=datetime.now() - timedelta(milliseconds=150),
            executed_at=datetime.now(),
        )

        score = await tracker.track_execution(
            market_order,
            actual_price=Decimal("50100"),  # Some slippage expected
            time_to_fill_ms=150,
            market_mid_price=Decimal("50000"),
        )

        assert score > 0
        assert score < 100  # Not perfect due to taker fee and slippage

    @pytest.mark.asyncio
    async def test_post_only_order_execution(self, tracker):
        """Test tracking a post-only order execution."""
        post_only_order = Order(
            order_id="post-1",
            client_order_id="client-post-1",
            symbol="BTC/USDT",
            type=OrderType.POST_ONLY,
            side=OrderSide.SELL,
            price=Decimal("50100"),
            quantity=Decimal("0.05"),
            filled_quantity=Decimal("0.05"),
            status=OrderStatus.FILLED,
            routing_method="POST_ONLY",
            maker_fee_paid=Decimal("0.00005"),  # Always maker
            taker_fee_paid=Decimal("0"),
            created_at=datetime.now() - timedelta(seconds=5),
            executed_at=datetime.now(),
        )

        score = await tracker.track_execution(
            post_only_order,
            actual_price=Decimal("50100"),  # Executed at limit price
            time_to_fill_ms=5000,  # Slower but expected for post-only
            market_mid_price=Decimal("50000"),
        )

        assert score > 70  # Good score despite slower execution

        stats = await tracker.get_statistics("1h")
        assert "POST_ONLY" in stats.orders_by_routing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
