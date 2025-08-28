"""Unit tests for Price Impact Model."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.analytics.price_impact_model import PriceImpactModel


class TestPriceImpactModel:
    """Test suite for PriceImpactModel."""

    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def impact_model(self, event_bus):
        """Create PriceImpactModel instance."""
        return PriceImpactModel(event_bus)

    @pytest.mark.asyncio
    async def test_initialization(self, impact_model):
        """Test model initialization."""
        assert impact_model.event_bus is not None
        assert impact_model.kyle_lambda == Decimal("0.1")
        assert impact_model.temp_impact_decay == Decimal("0.5")
        assert len(impact_model.market_depth_cache) == 0

    @pytest.mark.asyncio
    async def test_calculate_impact_small_order(self, impact_model):
        """Test impact calculation for small order."""
        # Setup
        impact_model.market_depth_cache["BTC/USDT"] = {
            "avg_spread": Decimal("10"),
            "avg_depth": Decimal("100000"),
            "volatility": Decimal("0.02")
        }

        # Calculate impact
        impact = await impact_model.calculate_impact(
            symbol="BTC/USDT",
            order_size=Decimal("0.1"),
            side="BUY",
            current_price=Decimal("50000")
        )

        # Assertions
        assert "temporary_impact" in impact
        assert "permanent_impact" in impact
        assert "total_impact" in impact
        assert "expected_price" in impact
        assert "slippage_bps" in impact
        assert "confidence" in impact

        # Small orders should have minimal impact
        assert impact["total_impact"] < Decimal("0.01")
        assert impact["slippage_bps"] < 10

    @pytest.mark.asyncio
    async def test_calculate_impact_large_order(self, impact_model):
        """Test impact calculation for large order."""
        # Setup
        impact_model.market_depth_cache["BTC/USDT"] = {
            "avg_spread": Decimal("10"),
            "avg_depth": Decimal("100000"),
            "volatility": Decimal("0.02")
        }

        # Large order relative to market depth
        impact = await impact_model.calculate_impact(
            symbol="BTC/USDT",
            order_size=Decimal("100"),  # Large order
            side="SELL",
            current_price=Decimal("50000")
        )

        # Large orders should have significant impact
        assert impact["total_impact"] > Decimal("0.01")
        assert impact["slippage_bps"] > 10
        assert impact["confidence"] <= Decimal("1.0")

    @pytest.mark.asyncio
    async def test_estimate_kyle_lambda(self, impact_model):
        """Test Kyle's lambda estimation."""
        # Mock historical executions
        executions = [
            {
                "size": Decimal("1"),
                "price_change": Decimal("5"),
                "timestamp": datetime.now(UTC)
            },
            {
                "size": Decimal("2"),
                "price_change": Decimal("12"),
                "timestamp": datetime.now(UTC)
            },
            {
                "size": Decimal("0.5"),
                "price_change": Decimal("2"),
                "timestamp": datetime.now(UTC)
            }
        ]

        kyle_lambda = await impact_model.estimate_kyle_lambda(
            symbol="BTC/USDT",
            executions=executions
        )

        assert kyle_lambda > 0
        assert kyle_lambda < 10  # Reasonable range

    @pytest.mark.asyncio
    async def test_update_market_depth(self, impact_model):
        """Test market depth update."""
        await impact_model.update_market_depth(
            symbol="ETH/USDT",
            bid_depth=Decimal("50000"),
            ask_depth=Decimal("55000"),
            spread=Decimal("5"),
            volatility=Decimal("0.025")
        )

        assert "ETH/USDT" in impact_model.market_depth_cache
        cache = impact_model.market_depth_cache["ETH/USDT"]
        assert cache["avg_depth"] == Decimal("52500")
        assert cache["avg_spread"] == Decimal("5")
        assert cache["volatility"] == Decimal("0.025")

    @pytest.mark.asyncio
    async def test_almgren_chriss_trajectory(self, impact_model):
        """Test Almgren-Chriss optimal execution trajectory."""
        trajectory = await impact_model.almgren_chriss_trajectory(
            total_size=Decimal("10"),
            time_horizon=3600,  # 1 hour
            risk_aversion=Decimal("0.5"),
            volatility=Decimal("0.02"),
            avg_volume=Decimal("1000")
        )

        assert "schedule" in trajectory
        assert "expected_cost" in trajectory
        assert "risk" in trajectory
        assert len(trajectory["schedule"]) > 0

        # Check schedule sums to total size
        total_scheduled = sum(s["size"] for s in trajectory["schedule"])
        assert abs(total_scheduled - Decimal("10")) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_pre_trade_analysis(self, impact_model):
        """Test pre-trade analysis."""
        # Setup market depth
        impact_model.market_depth_cache["BTC/USDT"] = {
            "avg_spread": Decimal("10"),
            "avg_depth": Decimal("100000"),
            "volatility": Decimal("0.02")
        }

        analysis = await impact_model.pre_trade_analysis(
            symbol="BTC/USDT",
            order_size=Decimal("5"),
            side="BUY",
            current_price=Decimal("50000"),
            urgency="NORMAL"
        )

        assert "impact_estimate" in analysis
        assert "optimal_slices" in analysis
        assert "recommended_limit_price" in analysis
        assert "execution_schedule" in analysis
        assert "risk_metrics" in analysis
        assert analysis["optimal_slices"] >= 1

    @pytest.mark.asyncio
    async def test_post_trade_analysis(self, impact_model, event_bus):
        """Test post-trade analysis."""
        analysis = await impact_model.post_trade_analysis(
            symbol="BTC/USDT",
            expected_impact=Decimal("0.005"),
            realized_impact=Decimal("0.007"),
            order_size=Decimal("5"),
            execution_time=120
        )

        assert "impact_accuracy" in analysis
        assert "excess_impact" in analysis
        assert "execution_quality" in analysis
        assert analysis["excess_impact"] == Decimal("0.002")

        # Check event was published
        event_bus.publish.assert_called_once()
        event_name, event_data = event_bus.publish.call_args[0]
        assert event_name == "price_impact.analyzed"

    @pytest.mark.asyncio
    async def test_calculate_slippage(self, impact_model):
        """Test slippage calculation."""
        slippage = await impact_model.calculate_slippage(
            expected_price=Decimal("50000"),
            executed_price=Decimal("50050"),
            side="BUY"
        )

        assert "absolute" in slippage
        assert "percentage" in slippage
        assert "basis_points" in slippage
        assert slippage["absolute"] == Decimal("50")
        assert slippage["basis_points"] == 10

    @pytest.mark.asyncio
    async def test_estimate_market_depth(self, impact_model):
        """Test market depth estimation from order book."""
        order_book = {
            "bids": [
                {"price": Decimal("49990"), "quantity": Decimal("10")},
                {"price": Decimal("49980"), "quantity": Decimal("15")},
                {"price": Decimal("49970"), "quantity": Decimal("20")}
            ],
            "asks": [
                {"price": Decimal("50010"), "quantity": Decimal("12")},
                {"price": Decimal("50020"), "quantity": Decimal("18")},
                {"price": Decimal("50030"), "quantity": Decimal("25")}
            ]
        }

        depth = await impact_model.estimate_market_depth(
            order_book=order_book,
            price_levels=3
        )

        assert "bid_depth" in depth
        assert "ask_depth" in depth
        assert "total_depth" in depth
        assert "imbalance" in depth
        assert depth["bid_depth"] == Decimal("45")
        assert depth["ask_depth"] == Decimal("55")

    @pytest.mark.asyncio
    async def test_no_market_data_fallback(self, impact_model):
        """Test behavior when no market data is available."""
        # No cached data for this symbol
        impact = await impact_model.calculate_impact(
            symbol="NEW/PAIR",
            order_size=Decimal("1"),
            side="BUY",
            current_price=Decimal("100")
        )

        # Should use default parameters
        assert impact is not None
        assert impact["confidence"] < Decimal("0.5")  # Low confidence without data

    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, impact_model):
        """Test impact calculation in extreme conditions."""
        # Very high volatility
        impact_model.market_depth_cache["VOLATILE/USDT"] = {
            "avg_spread": Decimal("100"),
            "avg_depth": Decimal("1000"),
            "volatility": Decimal("0.5")  # 50% volatility
        }

        impact = await impact_model.calculate_impact(
            symbol="VOLATILE/USDT",
            order_size=Decimal("10"),
            side="SELL",
            current_price=Decimal("1000")
        )

        # High volatility should increase impact
        assert impact["total_impact"] > Decimal("0.05")
        assert impact["confidence"] < Decimal("0.8")  # Lower confidence in extreme conditions
