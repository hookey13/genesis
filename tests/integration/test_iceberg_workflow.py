"""End-to-end integration tests for iceberg order execution workflow."""

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from genesis.analytics.iceberg_report_generator import IcebergReportGenerator
from genesis.analytics.market_impact_monitor import MarketImpactMonitor
from genesis.analytics.order_book_analyzer import OrderBookAnalyzer
from genesis.core.exceptions import (
    InsufficientLiquidity,
    OrderExecutionError,
    TierGateViolation,
    ValidationError,
)
from genesis.core.models import Order, OrderSide, OrderStatus, OrderType
from genesis.engine.executor.iceberg import IcebergOrderExecutor
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.state_machine import TierStateMachine, TradingTier
from genesis.exchange.gateway import ExchangeGateway
from genesis.exchange.models import OrderBook, OrderBookLevel


class TestIcebergWorkflow:
    """Integration tests for complete iceberg order workflow."""

    @pytest.fixture
    async def setup_infrastructure(self):
        """Setup complete infrastructure for testing."""
        # Mock exchange gateway
        exchange = Mock(spec=ExchangeGateway)
        exchange.place_order = AsyncMock()
        exchange.get_order_status = AsyncMock()
        exchange.get_order_book = AsyncMock()
        exchange.cancel_order = AsyncMock()

        # Mock state machine at Hunter tier
        state_machine = Mock(spec=TierStateMachine)
        state_machine.current_tier = TradingTier.HUNTER
        state_machine.check_tier_requirement = Mock(return_value=True)

        # Mock repository
        repository = Mock()
        repository.save_iceberg_execution = AsyncMock()
        repository.get_iceberg_execution = AsyncMock()
        repository.save_execution_report = AsyncMock()

        # Create components
        market_executor = MarketOrderExecutor(exchange, state_machine)
        order_book_analyzer = OrderBookAnalyzer()
        impact_monitor = MarketImpactMonitor(repository)
        report_generator = IcebergReportGenerator(repository)

        # Create iceberg executor
        iceberg_executor = IcebergOrderExecutor(
            market_executor=market_executor,
            order_book_analyzer=order_book_analyzer,
            impact_monitor=impact_monitor,
            state_machine=state_machine,
            repository=repository
        )

        return {
            "iceberg": iceberg_executor,
            "exchange": exchange,
            "state_machine": state_machine,
            "repository": repository,
            "report_generator": report_generator
        }

    @pytest.fixture
    def sample_order_book(self):
        """Create sample order book with good liquidity."""
        return OrderBook(
            symbol="BTC/USDT",
            bids=[
                OrderBookLevel(price=Decimal("49950"), quantity=Decimal("0.5")),
                OrderBookLevel(price=Decimal("49940"), quantity=Decimal("0.8")),
                OrderBookLevel(price=Decimal("49930"), quantity=Decimal("1.2")),
                OrderBookLevel(price=Decimal("49920"), quantity=Decimal("1.5")),
                OrderBookLevel(price=Decimal("49910"), quantity=Decimal("2.0"))
            ],
            asks=[
                OrderBookLevel(price=Decimal("50050"), quantity=Decimal("0.5")),
                OrderBookLevel(price=Decimal("50060"), quantity=Decimal("0.8")),
                OrderBookLevel(price=Decimal("50070"), quantity=Decimal("1.2")),
                OrderBookLevel(price=Decimal("50080"), quantity=Decimal("1.5")),
                OrderBookLevel(price=Decimal("50090"), quantity=Decimal("2.0"))
            ],
            timestamp=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_successful_iceberg_execution(self, setup_infrastructure, sample_order_book):
        """Test successful end-to-end iceberg order execution."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        # Setup exchange mock responses
        exchange.get_order_book.return_value = sample_order_book

        # Mock successful order placements
        async def mock_place_order(order):
            return Order(
                order_id=str(uuid.uuid4()),
                position_id=order.position_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                filled_quantity=order.quantity,
                filled_price=Decimal("50000"),
                status=OrderStatus.FILLED,
                created_at=order.created_at,
                updated_at=datetime.now()
            )

        exchange.place_order.side_effect = mock_place_order

        # Create buy order for $500 (should trigger iceberg)
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),  # $500 worth at $50k
            price=None,
            created_at=datetime.now()
        )

        # Execute iceberg order
        result = await iceberg.execute_iceberg_order(order)

        # Verify execution
        assert result is not None
        assert result.execution_id is not None
        assert result.completed_slices >= 3  # Minimum slices
        assert result.status == "COMPLETED"
        assert result.total_filled == order.quantity

        # Verify exchange was called multiple times (once per slice)
        assert exchange.place_order.call_count >= 3

        # Verify repository was updated
        infra["repository"].save_iceberg_execution.assert_called()

    @pytest.mark.asyncio
    async def test_iceberg_abort_on_high_slippage(self, setup_infrastructure, sample_order_book):
        """Test iceberg execution aborts when slippage exceeds threshold."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        exchange.get_order_book.return_value = sample_order_book

        # Mock orders with increasing slippage
        slice_count = 0

        async def mock_place_order_with_slippage(order):
            nonlocal slice_count
            slice_count += 1

            # Third slice has high slippage
            if slice_count >= 3:
                filled_price = Decimal("50300")  # 0.6% slippage
            else:
                filled_price = Decimal("50100")  # 0.2% slippage

            return Order(
                order_id=str(uuid.uuid4()),
                position_id=order.position_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                filled_quantity=order.quantity,
                filled_price=filled_price,
                status=OrderStatus.FILLED,
                created_at=order.created_at,
                updated_at=datetime.now()
            )

        exchange.place_order.side_effect = mock_place_order_with_slippage

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=None,
            created_at=datetime.now()
        )

        # Execute should abort due to high slippage
        result = await iceberg.execute_iceberg_order(order)

        assert result.status == "ABORTED"
        assert result.abort_reason is not None
        assert "slippage" in result.abort_reason.lower()
        assert result.completed_slices < result.total_slices

    @pytest.mark.asyncio
    async def test_tier_gate_enforcement(self, setup_infrastructure):
        """Test that Sniper tier cannot use iceberg execution."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        state_machine = infra["state_machine"]

        # Set tier to Sniper
        state_machine.current_tier = TradingTier.SNIPER
        state_machine.check_tier_requirement.return_value = False

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=None,
            created_at=datetime.now()
        )

        # Should raise tier gate violation
        with pytest.raises(TierGateViolation):
            await iceberg.execute_iceberg_order(order)

    @pytest.mark.asyncio
    async def test_insufficient_liquidity_handling(self, setup_infrastructure):
        """Test handling of insufficient order book liquidity."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        # Create thin order book
        thin_order_book = OrderBook(
            symbol="BTC/USDT",
            bids=[
                OrderBookLevel(price=Decimal("49950"), quantity=Decimal("0.001"))
            ],
            asks=[
                OrderBookLevel(price=Decimal("50050"), quantity=Decimal("0.001"))
            ],
            timestamp=datetime.now()
        )

        exchange.get_order_book.return_value = thin_order_book

        # Create large order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),  # Much larger than available liquidity
            price=None,
            created_at=datetime.now()
        )

        # Should raise insufficient liquidity error
        with pytest.raises(InsufficientLiquidity):
            await iceberg.execute_iceberg_order(order)

    @pytest.mark.asyncio
    async def test_partial_fill_rollback(self, setup_infrastructure, sample_order_book):
        """Test rollback of partially filled iceberg order."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        exchange.get_order_book.return_value = sample_order_book

        # Mock partial execution (2 of 5 slices)
        slice_count = 0

        async def mock_partial_execution(order):
            nonlocal slice_count
            slice_count += 1

            if slice_count <= 2:
                return Order(
                    order_id=str(uuid.uuid4()),
                    position_id=order.position_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    type=order.type,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=order.quantity,
                    filled_price=Decimal("50000"),
                    status=OrderStatus.FILLED,
                    created_at=order.created_at
                )
            else:
                # Simulate failure on third slice
                raise OrderExecutionError("Network error")

        exchange.place_order.side_effect = mock_partial_execution

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=None,
            created_at=datetime.now()
        )

        # Execute with expected failure
        try:
            result = await iceberg.execute_iceberg_order(order)
        except OrderExecutionError:
            # Order failed, now test rollback
            pass

        # Get execution ID from active executions
        execution_id = list(iceberg.active_executions.keys())[0]

        # Test rollback without confirmation (should fail)
        with pytest.raises(ValidationError, match="Manual confirmation required"):
            await iceberg.rollback_partial_execution(execution_id)

        # Test rollback with confirmation
        exchange.place_order.side_effect = mock_partial_execution  # Reset mock

        rollback_result = await iceberg.rollback_partial_execution(
            execution_id=execution_id,
            confirmed_by="test_user"
        )

        assert rollback_result["confirmed_by"] == "test_user"
        assert rollback_result["rollback_orders"] > 0
        assert rollback_result["execution_id"] == execution_id

    @pytest.mark.asyncio
    async def test_concurrent_iceberg_executions(self, setup_infrastructure, sample_order_book):
        """Test handling multiple concurrent iceberg executions."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        exchange.get_order_book.return_value = sample_order_book

        # Mock successful order placement
        async def mock_place_order(order):
            await asyncio.sleep(0.01)  # Simulate network delay
            return Order(
                order_id=str(uuid.uuid4()),
                position_id=order.position_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                filled_price=Decimal("50000"),
                status=OrderStatus.FILLED,
                created_at=order.created_at
            )

        exchange.place_order.side_effect = mock_place_order

        # Create multiple orders
        orders = [
            Order(
                order_id=str(uuid.uuid4()),
                position_id=str(uuid.uuid4()),
                client_order_id=f"test-{i}-{uuid.uuid4()}",
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=Decimal("0.01"),
                price=None,
                created_at=datetime.now()
            )
            for i in range(3)
        ]

        # Execute concurrently
        tasks = [iceberg.execute_iceberg_order(order) for order in orders]
        results = await asyncio.gather(*tasks)

        # Verify all executed successfully
        assert len(results) == 3
        assert all(r.status == "COMPLETED" for r in results)
        assert len(iceberg.active_executions) == 3

    @pytest.mark.asyncio
    async def test_execution_tracking_and_reporting(self, setup_infrastructure, sample_order_book):
        """Test real-time execution tracking and report generation."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]
        report_generator = infra["report_generator"]

        exchange.get_order_book.return_value = sample_order_book

        # Track execution progress
        execution_updates = []

        async def mock_place_order_with_tracking(order):
            # Simulate progress tracking
            execution_updates.append({
                "timestamp": datetime.now(),
                "slice": len(execution_updates) + 1
            })

            return Order(
                order_id=str(uuid.uuid4()),
                position_id=order.position_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                filled_price=Decimal("50000") + Decimal(len(execution_updates) * 10),
                status=OrderStatus.FILLED,
                created_at=order.created_at
            )

        exchange.place_order.side_effect = mock_place_order_with_tracking

        # Create and execute order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=None,
            created_at=datetime.now()
        )

        result = await iceberg.execute_iceberg_order(order)

        # Generate report
        execution = iceberg.active_executions[result.execution_id]
        report = await report_generator.generate_report(execution)

        assert report is not None
        assert report.execution_id == result.execution_id
        assert report.total_slices >= 3
        assert report.metrics is not None
        assert report.metrics.total_slippage is not None
        assert report.quality_score is not None

    @pytest.mark.asyncio
    async def test_order_below_threshold(self, setup_infrastructure):
        """Test that orders below $200 don't trigger iceberg execution."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]

        # Create small order ($150)
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.003"),  # $150 at $50k
            price=None,
            created_at=datetime.now()
        )

        # Should not trigger iceberg execution
        with pytest.raises(ValidationError, match="below threshold"):
            await iceberg.execute_iceberg_order(order)

    @pytest.mark.asyncio
    async def test_recovery_from_network_failure(self, setup_infrastructure, sample_order_book):
        """Test recovery mechanism when network fails during execution."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        exchange.get_order_book.return_value = sample_order_book

        # Simulate network failure after 2 slices
        slice_count = 0

        async def mock_network_failure(order):
            nonlocal slice_count
            slice_count += 1

            if slice_count <= 2:
                return Order(
                    order_id=str(uuid.uuid4()),
                    position_id=order.position_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    type=order.type,
                    side=order.side,
                    quantity=order.quantity,
                    filled_quantity=order.quantity,
                    filled_price=Decimal("50000"),
                    status=OrderStatus.FILLED,
                    created_at=order.created_at
                )
            else:
                # Simulate network timeout
                raise asyncio.TimeoutError("Network timeout")

        exchange.place_order.side_effect = mock_network_failure

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=None,
            created_at=datetime.now()
        )

        # Execute with expected failure
        with pytest.raises(asyncio.TimeoutError):
            await iceberg.execute_iceberg_order(order)

        # Verify partial execution was tracked
        assert len(iceberg.active_executions) == 1
        execution = list(iceberg.active_executions.values())[0]
        assert execution.completed_slices == 2
        assert execution.status != "COMPLETED"

    @pytest.mark.asyncio
    async def test_market_impact_monitoring(self, setup_infrastructure, sample_order_book):
        """Test real-time market impact monitoring during execution."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        exchange.get_order_book.return_value = sample_order_book

        # Track impact for each slice
        impacts = []

        async def mock_place_order_with_impact(order):
            # Simulate increasing market impact
            slice_num = len(impacts) + 1
            impact = Decimal("0.05") * slice_num  # 0.05%, 0.10%, 0.15%, etc.
            impacts.append(impact)

            filled_price = Decimal("50000") * (Decimal("1") + impact / Decimal("100"))

            return Order(
                order_id=str(uuid.uuid4()),
                position_id=order.position_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                filled_price=filled_price,
                status=OrderStatus.FILLED,
                created_at=order.created_at
            )

        exchange.place_order.side_effect = mock_place_order_with_impact

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
            price=None,
            created_at=datetime.now()
        )

        result = await iceberg.execute_iceberg_order(order)

        # Verify impact was monitored
        assert result.status == "COMPLETED"
        assert len(impacts) >= 3  # At least 3 slices

        # Check cumulative impact
        execution = iceberg.active_executions[result.execution_id]
        assert execution.cumulative_impact > Decimal("0")

    @pytest.mark.asyncio
    async def test_order_exactly_at_threshold(self, setup_infrastructure, sample_order_book):
        """Test order exactly at $200 threshold triggers iceberg."""
        infra = await setup_infrastructure
        iceberg = infra["iceberg"]
        exchange = infra["exchange"]

        exchange.get_order_book.return_value = sample_order_book

        async def mock_place_order(order):
            return Order(
                order_id=str(uuid.uuid4()),
                position_id=order.position_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                type=order.type,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity,
                filled_price=Decimal("50000"),
                status=OrderStatus.FILLED,
                created_at=order.created_at
            )

        exchange.place_order.side_effect = mock_place_order

        # Create order for exactly $200
        order = Order(
            order_id=str(uuid.uuid4()),
            position_id=str(uuid.uuid4()),
            client_order_id=f"test-{uuid.uuid4()}",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.004"),  # Exactly $200 at $50k
            price=None,
            created_at=datetime.now()
        )

        result = await iceberg.execute_iceberg_order(order)

        assert result.status == "COMPLETED"
        assert result.completed_slices >= 3  # Should trigger slicing
