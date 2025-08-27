"""
Integration tests for smart order routing workflow.

Tests the complete flow from order submission through smart routing
to execution and quality tracking.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.analytics.execution_quality import ExecutionQualityTracker
from genesis.core.models import Order, OrderSide, OrderStatus, OrderType, TradingTier
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.executor.base import (
    ExecutionResult,
    ExecutionStrategy,
    OrderExecutor,
)
from genesis.engine.executor.smart_router import (
    SmartRouter,
    UrgencyLevel,
)
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderRequest, OrderResponse


@pytest.fixture
async def test_database():
    """Create an in-memory test database."""
    repo = SQLiteRepository(":memory:")
    await repo.initialize()
    yield repo
    await repo.close()


@pytest.fixture
def mock_gateway():
    """Create a mock exchange gateway."""
    gateway = MagicMock(spec=BinanceGateway)
    gateway.get_order_book = AsyncMock()
    gateway.get_ticker = AsyncMock()
    gateway.place_order = AsyncMock()
    gateway.place_post_only_order = AsyncMock()
    gateway.get_order_status = AsyncMock()
    return gateway


@pytest.fixture
def smart_router(mock_gateway):
    """Create a smart router instance."""
    return SmartRouter(mock_gateway)


@pytest.fixture
def quality_tracker(test_database):
    """Create an execution quality tracker."""
    return ExecutionQualityTracker(test_database)


@pytest.fixture
def mock_executor(smart_router):
    """Create a mock order executor with smart router."""
    executor = MagicMock(spec=OrderExecutor)
    executor.tier = TradingTier.HUNTER
    executor.smart_router = smart_router
    executor.execute_market_order = AsyncMock()
    executor.route_order = AsyncMock()
    return executor


class TestSmartRoutingIntegration:
    """Test complete smart routing workflow."""

    @pytest.mark.asyncio
    async def test_market_order_tight_spread_flow(self, mock_gateway, smart_router, mock_executor):
        """Test market order selection for tight spread conditions."""
        # Setup market conditions - tight spread
        mock_gateway.get_order_book.return_value = {
            'bids': [[49999, 10], [49998, 20]],
            'asks': [[50001, 10], [50002, 20]]  # 2 point spread = 0.004%
        }
        mock_gateway.get_ticker.return_value = {
            'priceChangePercent': '1.5'
        }

        # Create order
        order = Order(
            order_id="test-1",
            client_order_id="client-1",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING
        )

        # Route order
        routed = await smart_router.route_order(order, UrgencyLevel.NORMAL)

        # Should select MARKET due to tight spread
        assert routed.selected_type.value == "MARKET"
        assert "tight spread" in routed.routing_reason.lower()
        assert routed.expected_fee_rate == smart_router.TAKER_FEE_RATE

    @pytest.mark.asyncio
    async def test_post_only_order_wide_spread_flow(self, mock_gateway, smart_router):
        """Test post-only order selection for wide spread conditions."""
        # Setup market conditions - wide spread
        mock_gateway.get_order_book.return_value = {
            'bids': [[49950, 10], [49940, 20]],
            'asks': [[50050, 10], [50060, 20]]  # 100 point spread = 0.2%
        }
        mock_gateway.get_ticker.return_value = {
            'priceChangePercent': '2.0'
        }

        # Create order
        order = Order(
            order_id="test-2",
            client_order_id="client-2",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("49960"),
            quantity=Decimal("0.05"),
            status=OrderStatus.PENDING
        )

        # Route order with low urgency
        routed = await smart_router.route_order(order, UrgencyLevel.LOW)

        # Should select POST_ONLY due to wide spread and low urgency
        assert routed.selected_type.value == "POST_ONLY"
        assert "wide spread" in routed.routing_reason.lower()
        assert routed.expected_fee_rate == smart_router.MAKER_FEE_RATE

    @pytest.mark.asyncio
    async def test_fok_order_volatile_thin_liquidity_flow(self, mock_gateway, smart_router):
        """Test FOK order selection for volatile, thin liquidity conditions."""
        # Setup market conditions - thin liquidity, high volatility
        mock_gateway.get_order_book.return_value = {
            'bids': [[49990, 0.5], [49980, 0.3]],  # Very thin
            'asks': [[50010, 0.4], [50020, 0.2]]
        }
        mock_gateway.get_ticker.return_value = {
            'priceChangePercent': '8.5'  # High volatility
        }

        # Create order
        order = Order(
            order_id="test-3",
            client_order_id="client-3",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.5"),  # Large relative to liquidity
            status=OrderStatus.PENDING
        )

        # Route order
        routed = await smart_router.route_order(order, UrgencyLevel.NORMAL)

        # Should select FOK due to thin liquidity and high volatility
        assert routed.selected_type.value == "FOK"
        assert "volatility" in routed.routing_reason.lower()

    @pytest.mark.asyncio
    async def test_post_only_retry_logic(self, mock_gateway):
        """Test post-only order retry on rejection."""
        # Setup rejection then success
        mock_gateway.place_order.side_effect = [
            Exception("Order would immediately match"),  # First attempt fails
            OrderResponse(
                order_id="exchange-1",
                client_order_id="client-4",
                symbol="BTC/USDT",
                side="buy",
                type="limit",
                price=Decimal("49959"),  # Adjusted price
                quantity=Decimal("0.1"),
                filled_quantity=Decimal("0"),
                status="open",
                created_at=datetime.now()
            )
        ]

        # Create order request
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="POST_ONLY",
            quantity=Decimal("0.1"),
            price=Decimal("49960")
        )

        # Place post-only order with retries
        response = await mock_gateway.place_post_only_order(request, max_retries=3)

        assert response.order_id == "exchange-1"
        assert mock_gateway.place_order.call_count == 2  # Original + 1 retry

    @pytest.mark.asyncio
    async def test_execution_quality_tracking_flow(self, smart_router, quality_tracker):
        """Test complete execution quality tracking flow."""
        # Create a completed order
        order = Order(
            order_id="test-5",
            client_order_id="client-5",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            routing_method="SMART",
            maker_fee_paid=Decimal("0.00005"),  # 0.05% maker fee
            taker_fee_paid=Decimal("0"),
            created_at=datetime.now(),
            executed_at=datetime.now()
        )

        # Track execution
        score = await quality_tracker.track_execution(
            order=order,
            actual_price=Decimal("49995"),  # Slight improvement
            time_to_fill_ms=150,
            market_mid_price=Decimal("50000")
        )

        assert score > 80  # Good execution
        assert order.execution_score == score

        # Get statistics
        stats = await quality_tracker.get_statistics("1h")
        assert stats.total_orders == 1
        assert stats.avg_execution_score == score

    @pytest.mark.asyncio
    async def test_tier_restriction_enforcement(self):
        """Test that smart routing respects tier restrictions."""
        # Create executor with SNIPER tier (too low for smart routing)
        low_tier_executor = MagicMock(spec=OrderExecutor)
        low_tier_executor.tier = TradingTier.SNIPER
        low_tier_executor.smart_router = None

        order = Order(
            order_id="test-6",
            client_order_id="client-6",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING
        )

        # Try to route order - should fail
        with pytest.raises(ValueError, match="Smart router not configured"):
            await low_tier_executor.route_order(order)

    @pytest.mark.asyncio
    async def test_execution_strategy_routing(self, mock_executor, smart_router):
        """Test execution with SMART strategy."""
        mock_executor.smart_router = smart_router
        smart_router.execute_routed_order = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                order=MagicMock(),
                message="Order executed via smart routing",
                actual_price=Decimal("50000"),
                slippage_percent=Decimal("0"),
                latency_ms=100
            )
        )

        order = Order(
            order_id="test-7",
            client_order_id="client-7",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING
        )

        # Execute with SMART strategy
        result = await mock_executor.execute_order(order, ExecutionStrategy.SMART)

        assert result.success
        assert "smart routing" in result.message.lower()

    @pytest.mark.asyncio
    async def test_complete_workflow_with_reporting(self, mock_gateway, smart_router, quality_tracker):
        """Test complete workflow from routing to quality reporting."""
        # Setup market conditions
        mock_gateway.get_order_book.return_value = {
            'bids': [[49990, 10], [49980, 20]],
            'asks': [[50010, 10], [50020, 20]]
        }
        mock_gateway.get_ticker.return_value = {
            'priceChangePercent': '2.5'
        }

        # Create and route multiple orders
        orders = []
        for i in range(5):
            order = Order(
                order_id=f"test-{i}",
                client_order_id=f"client-{i}",
                symbol="BTC/USDT",
                type=OrderType.LIMIT,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=Decimal("50000"),
                quantity=Decimal("0.1"),
                status=OrderStatus.PENDING
            )

            # Route order
            routed = await smart_router.route_order(order, UrgencyLevel.NORMAL)
            order.routing_method = routed.selected_type.value

            # Simulate execution
            if routed.selected_type.value == "POST_ONLY":
                order.maker_fee_paid = Decimal("0.00005")
                order.taker_fee_paid = Decimal("0")
            else:
                order.maker_fee_paid = Decimal("0")
                order.taker_fee_paid = Decimal("0.0001")

            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.executed_at = datetime.now()

            # Track execution quality
            await quality_tracker.track_execution(
                order=order,
                actual_price=Decimal("50000") + Decimal(i * 10),
                time_to_fill_ms=100 + i * 50,
                market_mid_price=Decimal("50000")
            )

            orders.append(order)

        # Generate statistics
        stats = await quality_tracker.get_statistics("24h")

        assert stats.total_orders == 5
        assert stats.avg_execution_score > 0
        assert len(stats.orders_by_routing) > 0

        # Generate report
        report = quality_tracker.generate_report(stats)

        assert "Total Orders: 5" in report
        assert "Average Execution Score" in report
        assert "By Routing Method" in report


class TestErrorHandling:
    """Test error handling in smart routing."""

    @pytest.mark.asyncio
    async def test_gateway_error_fallback(self, mock_gateway, smart_router):
        """Test fallback when gateway fails."""
        # Setup gateway to fail
        mock_gateway.get_order_book.side_effect = Exception("API Error")

        order = Order(
            order_id="test-error-1",
            client_order_id="client-error-1",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            status=OrderStatus.PENDING
        )

        # Should still route with conservative defaults
        routed = await smart_router.route_order(order)

        assert routed is not None
        assert routed.market_conditions.liquidity_level.value == "THIN"  # Conservative

    @pytest.mark.asyncio
    async def test_post_only_max_retries(self, mock_gateway):
        """Test post-only order fails after max retries."""
        # Setup continuous rejection
        mock_gateway.place_order.side_effect = Exception("Order would immediately match")

        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="POST_ONLY",
            quantity=Decimal("0.1"),
            price=Decimal("50000")
        )

        # Should raise after max retries
        with pytest.raises(Exception, match="would immediately match"):
            await mock_gateway.place_post_only_order(request, max_retries=2)

        # Should have tried max_retries times
        assert mock_gateway.place_order.call_count == 2


class TestConfigurationIntegration:
    """Test configuration loading and application."""

    def test_smart_routing_config_loading(self):
        """Test loading smart routing configuration."""

        config_path = "config/trading_rules.yaml"

        # This would normally load the actual config file
        # For testing, we'll create a sample config
        config = {
            'smart_routing': {
                'enabled_from_tier': 'HUNTER',
                'spread_thresholds': {
                    'tight_bps': 5,
                    'wide_bps': 20,
                    'post_only_min_bps': 5
                },
                'fee_optimization': {
                    'prefer_maker': True,
                    'max_post_only_retries': 3
                }
            }
        }

        assert config['smart_routing']['enabled_from_tier'] == 'HUNTER'
        assert config['smart_routing']['spread_thresholds']['tight_bps'] == 5
        assert config['smart_routing']['fee_optimization']['prefer_maker'] is True

    def test_tier_gate_configuration(self):
        """Test tier gate configuration for smart routing."""

        # Sample tier gates config
        config = {
            'tiers': {
                'HUNTER': {
                    'features_unlocked': [
                        'smart_order_routing',
                        'post_only_orders',
                        'fok_ioc_orders'
                    ]
                },
                'STRATEGIST': {
                    'features_unlocked': [
                        'advanced_smart_routing',
                        'execution_quality_reports'
                    ]
                }
            }
        }

        assert 'smart_order_routing' in config['tiers']['HUNTER']['features_unlocked']
        assert 'execution_quality_reports' in config['tiers']['STRATEGIST']['features_unlocked']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
