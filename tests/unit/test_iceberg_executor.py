"""Unit tests for the Iceberg Order Executor."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from genesis.core.exceptions import OrderExecutionError
from genesis.core.models import Account, TradingTier
from genesis.data.repository import Repository
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.engine.executor.iceberg import (
    MIN_ORDER_VALUE_USDT,
    MIN_SLICES,
    SLICE_VARIATION_PERCENT,
    SLIPPAGE_ABORT_THRESHOLD,
    IcebergExecution,
    IcebergOrderExecutor,
    LiquidityProfile,
)
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.models import OrderBook, Ticker


@pytest.fixture
def mock_gateway():
    """Create a mock gateway."""
    gateway = Mock(spec=BinanceGateway)
    gateway.get_ticker = AsyncMock()
    gateway.get_order_book = AsyncMock()
    gateway.place_order = AsyncMock()
    gateway.get_order_status = AsyncMock()
    gateway.cancel_order = AsyncMock()
    gateway.get_open_orders = AsyncMock()
    return gateway


@pytest.fixture
def mock_account():
    """Create a mock Hunter tier account."""
    account = Mock(spec=Account)
    account.account_id = "test_account_123"
    account.tier = TradingTier.HUNTER
    account.balance_usdt = Decimal("5000")
    return account


@pytest.fixture
def mock_market_executor():
    """Create a mock market executor."""
    executor = Mock(spec=MarketOrderExecutor)
    executor.execute_market_order = AsyncMock()
    executor.cancel_order = AsyncMock()
    executor.cancel_all_orders = AsyncMock()
    executor.get_order_status = AsyncMock()
    executor.generate_client_order_id = Mock(return_value=str(uuid4()))
    return executor


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = Mock(spec=Repository)
    repo.save_iceberg_execution = AsyncMock()
    repo.save_iceberg_slice = AsyncMock()
    repo.update_iceberg_execution = AsyncMock()
    repo.create_order = AsyncMock()
    repo.update_order = AsyncMock()
    return repo


@pytest.fixture
def iceberg_executor(mock_gateway, mock_account, mock_market_executor, mock_repository):
    """Create an iceberg executor instance."""
    return IcebergOrderExecutor(
        gateway=mock_gateway,
        account=mock_account,
        market_executor=mock_market_executor,
        repository=mock_repository,
    )


@pytest.fixture
def sample_order():
    """Create a sample order."""
    return Order(
        order_id=str(uuid4()),
        position_id=str(uuid4()),
        client_order_id=str(uuid4()),
        symbol="BTCUSDT",
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        price=None,
        quantity=Decimal("0.01"),
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_order_book():
    """Create a sample order book."""
    return OrderBook(
        symbol="BTCUSDT",
        bids=[
            [40000.0, 0.5],
            [39999.0, 1.0],
            [39998.0, 0.8],
            [39997.0, 1.2],
            [39996.0, 0.6],
        ],
        asks=[
            [40001.0, 0.5],
            [40002.0, 1.0],
            [40003.0, 0.8],
            [40004.0, 1.2],
            [40005.0, 0.6],
        ],
        timestamp=datetime.now(),
    )


@pytest.fixture
def sample_ticker():
    """Create a sample ticker."""
    ticker = Mock(spec=Ticker)
    ticker.symbol = "BTCUSDT"
    ticker.last_price = Decimal("40000")
    ticker.bid_price = Decimal("39999")
    ticker.ask_price = Decimal("40001")
    ticker.volume = Decimal("1000")
    return ticker


class TestIcebergOrderExecutor:
    """Test suite for IcebergOrderExecutor."""

    def test_initialization_with_valid_tier(
        self, mock_gateway, mock_account, mock_market_executor, mock_repository
    ):
        """Test successful initialization with Hunter tier."""
        executor = IcebergOrderExecutor(
            gateway=mock_gateway,
            account=mock_account,
            market_executor=mock_market_executor,
            repository=mock_repository,
        )

        assert executor.tier == TradingTier.HUNTER
        assert executor.min_order_value == MIN_ORDER_VALUE_USDT
        assert executor.slippage_threshold == SLIPPAGE_ABORT_THRESHOLD
        assert len(executor.active_executions) == 0

    def test_initialization_with_invalid_tier(
        self, mock_gateway, mock_market_executor, mock_repository
    ):
        """Test initialization fails with Sniper tier."""
        sniper_account = Mock(spec=Account)
        sniper_account.account_id = "sniper_123"
        sniper_account.tier = TradingTier.SNIPER

        with pytest.raises(OrderExecutionError) as exc_info:
            IcebergOrderExecutor(
                gateway=mock_gateway,
                account=sniper_account,
                market_executor=mock_market_executor,
                repository=mock_repository,
            )

        assert "Hunter tier or above" in str(exc_info.value)

    def test_calculate_slice_sizes_minimum_slices(self, iceberg_executor):
        """Test slice size calculation with minimum slices."""
        order_value = Decimal("300")  # Just above minimum
        liquidity_profile = LiquidityProfile(
            total_bid_volume=Decimal("10000"),
            total_ask_volume=Decimal("10000"),
            bid_depth_1pct=Decimal("500"),
            ask_depth_1pct=Decimal("500"),
            bid_depth_2pct=Decimal("1000"),
            ask_depth_2pct=Decimal("1000"),
            spread_percent=Decimal("0.1"),
            optimal_slice_count=MIN_SLICES,
            timestamp=datetime.now(),
        )

        slices = iceberg_executor.calculate_slice_sizes(order_value, liquidity_profile)

        assert len(slices) == MIN_SLICES
        assert sum(slices) == order_value
        assert all(s > 0 for s in slices)

    def test_calculate_slice_sizes_with_variation(self, iceberg_executor):
        """Test that slice sizes have variation."""
        order_value = Decimal("1000")
        liquidity_profile = LiquidityProfile(
            total_bid_volume=Decimal("10000"),
            total_ask_volume=Decimal("10000"),
            bid_depth_1pct=Decimal("2000"),
            ask_depth_1pct=Decimal("2000"),
            bid_depth_2pct=Decimal("4000"),
            ask_depth_2pct=Decimal("4000"),
            spread_percent=Decimal("0.1"),
            optimal_slice_count=5,
            timestamp=datetime.now(),
        )

        slices = iceberg_executor.calculate_slice_sizes(order_value, liquidity_profile)

        assert len(slices) == 5
        assert sum(slices) == order_value
        # Check that not all slices are the same (variation applied)
        assert len(set(slices)) > 1

    def test_add_slice_variation(self, iceberg_executor):
        """Test slice variation calculation."""
        base_size = Decimal("100")

        # Test multiple variations
        variations = []
        for _ in range(10):
            varied = iceberg_executor.add_slice_variation(
                base_size, SLICE_VARIATION_PERCENT
            )
            variations.append(varied)

            # Check within expected range
            min_expected = base_size * Decimal("0.8")  # -20%
            max_expected = base_size * Decimal("1.2")  # +20%
            assert min_expected <= varied <= max_expected

        # Check that we get different values (randomness)
        assert len(set(variations)) > 1

    def test_generate_random_delay(self, iceberg_executor):
        """Test random delay generation."""
        delays = []
        for _ in range(10):
            delay = iceberg_executor.generate_random_delay()
            delays.append(delay)
            assert 1.0 <= delay <= 5.0

        # Check randomness
        assert len(set(delays)) > 1

    def test_analyze_liquidity_depth_buy_side(
        self, iceberg_executor, sample_order_book
    ):
        """Test liquidity analysis for buy orders."""
        order_value = Decimal("20000")  # 0.5 BTC at $40000

        profile = iceberg_executor.analyze_liquidity_depth(
            sample_order_book, OrderSide.BUY, order_value
        )

        assert profile.total_ask_volume > 0
        assert profile.ask_depth_1pct > 0
        assert profile.spread_percent > 0
        assert profile.optimal_slice_count >= MIN_SLICES

    def test_analyze_liquidity_depth_sell_side(
        self, iceberg_executor, sample_order_book
    ):
        """Test liquidity analysis for sell orders."""
        order_value = Decimal("20000")

        profile = iceberg_executor.analyze_liquidity_depth(
            sample_order_book, OrderSide.SELL, order_value
        )

        assert profile.total_bid_volume > 0
        assert profile.bid_depth_1pct > 0
        assert profile.optimal_slice_count >= MIN_SLICES

    @pytest.mark.asyncio
    async def test_execute_iceberg_order_below_threshold(
        self,
        iceberg_executor,
        sample_order,
        mock_gateway,
        mock_market_executor,
        sample_ticker,
    ):
        """Test that small orders bypass iceberg execution."""
        # Set up order value below threshold
        sample_order.quantity = Decimal("0.004")  # $160 at $40000
        mock_gateway.get_ticker.return_value = sample_ticker

        # Mock standard execution result
        standard_result = ExecutionResult(
            success=True,
            order=sample_order,
            message="Standard execution",
            actual_price=Decimal("40000"),
            slippage_percent=Decimal("0.1"),
        )
        mock_market_executor.execute_market_order.return_value = standard_result

        result = await iceberg_executor.execute_iceberg_order(sample_order)

        # Should use standard execution
        assert result == standard_result
        mock_market_executor.execute_market_order.assert_called_once()
        assert len(iceberg_executor.active_executions) == 0

    @pytest.mark.asyncio
    async def test_execute_iceberg_order_above_threshold(
        self,
        iceberg_executor,
        sample_order,
        mock_gateway,
        mock_market_executor,
        mock_repository,
        sample_ticker,
        sample_order_book,
    ):
        """Test iceberg execution for large orders."""
        # Set up large order
        sample_order.quantity = Decimal("0.01")  # $400 at $40000
        mock_gateway.get_ticker.return_value = sample_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        # Mock slice executions
        slice_results = []
        for i in range(MIN_SLICES):
            slice_result = ExecutionResult(
                success=True,
                order=Mock(filled_quantity=Decimal("0.003")),
                message=f"Slice {i+1} executed",
                actual_price=Decimal("40000"),
                slippage_percent=Decimal("0.05"),
            )
            slice_results.append(slice_result)

        mock_market_executor.execute_market_order.side_effect = slice_results

        result = await iceberg_executor.execute_iceberg_order(
            sample_order, force_iceberg=True
        )

        assert result.success
        assert "slices" in result.message.lower()
        assert mock_market_executor.execute_market_order.call_count == MIN_SLICES
        mock_repository.save_iceberg_execution.assert_called()

    @pytest.mark.asyncio
    async def test_execute_iceberg_order_with_slippage_abort(
        self,
        iceberg_executor,
        sample_order,
        mock_gateway,
        mock_market_executor,
        mock_repository,
        sample_ticker,
        sample_order_book,
    ):
        """Test that high slippage triggers abort."""
        sample_order.quantity = Decimal("0.01")
        mock_gateway.get_ticker.return_value = sample_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        # Mock high slippage on first slice
        high_slippage_result = ExecutionResult(
            success=True,
            order=Mock(filled_quantity=Decimal("0.003")),
            message="High slippage",
            actual_price=Decimal("40300"),  # 0.75% slippage
            slippage_percent=Decimal("0.75"),
        )

        mock_market_executor.execute_market_order.return_value = high_slippage_result

        result = await iceberg_executor.execute_iceberg_order(
            sample_order, force_iceberg=True
        )

        assert not result.success
        assert "aborted" in result.message.lower()
        assert result.error is not None
        mock_repository.update_iceberg_execution.assert_called()

    @pytest.mark.asyncio
    async def test_execute_iceberg_order_with_failures(
        self,
        iceberg_executor,
        sample_order,
        mock_gateway,
        mock_market_executor,
        mock_repository,
        sample_ticker,
        sample_order_book,
    ):
        """Test handling of slice execution failures."""
        sample_order.quantity = Decimal("0.01")
        mock_gateway.get_ticker.return_value = sample_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        # Mock failures
        failed_result = ExecutionResult(
            success=False,
            order=Mock(),
            message="Execution failed",
            error="Network error",
        )

        mock_market_executor.execute_market_order.return_value = failed_result

        result = await iceberg_executor.execute_iceberg_order(
            sample_order, force_iceberg=True
        )

        assert not result.success
        assert "failed" in result.message.lower()
        # Should abort after 3 failures
        assert mock_market_executor.execute_market_order.call_count <= 3

    @pytest.mark.asyncio
    async def test_order_book_caching(
        self, iceberg_executor, mock_gateway, sample_order_book
    ):
        """Test that order book is cached properly."""
        mock_gateway.get_order_book.return_value = sample_order_book

        # First call - should fetch from gateway
        book1 = await iceberg_executor._get_cached_order_book("BTCUSDT")
        assert mock_gateway.get_order_book.call_count == 1

        # Second call within TTL - should use cache
        book2 = await iceberg_executor._get_cached_order_book("BTCUSDT")
        assert mock_gateway.get_order_book.call_count == 1
        assert book1 == book2

        # Simulate cache expiry
        iceberg_executor._order_book_cache["BTCUSDT"] = (
            sample_order_book,
            datetime.now() - timedelta(seconds=10),
        )

        # Third call after TTL - should fetch again
        book3 = await iceberg_executor._get_cached_order_book("BTCUSDT")
        assert mock_gateway.get_order_book.call_count == 2

    @pytest.mark.asyncio
    async def test_rollback_partial_execution(
        self, iceberg_executor, mock_market_executor, sample_order
    ):
        """Test rollback of partially filled iceberg execution."""
        # Create an active execution
        execution = IcebergExecution(
            execution_id="test_exec_123",
            order=sample_order,
            total_slices=3,
            slices=[],
            slice_sizes=[Decimal("100"), Decimal("100"), Decimal("100")],
            slice_delays=[2.0, 3.0, 4.0],
            completed_slices=2,
            started_at=datetime.now(),
        )

        # Add filled slices
        for i in range(2):
            slice_order = Order(
                order_id=str(uuid4()),
                position_id=sample_order.position_id,
                client_order_id=str(uuid4()),
                symbol="BTCUSDT",
                type=OrderType.MARKET,
                side=OrderSide.BUY,
                price=None,
                quantity=Decimal("0.0025"),
                filled_quantity=Decimal("0.0025"),
                status=OrderStatus.FILLED,
                slice_number=i + 1,
                total_slices=3,
            )
            execution.slices.append(slice_order)

        iceberg_executor.active_executions["test_exec_123"] = execution

        # Mock rollback execution
        rollback_result = ExecutionResult(
            success=True,
            order=Mock(),
            message="Rollback executed",
            slippage_percent=Decimal("0.1"),
        )
        mock_market_executor.execute_market_order.return_value = rollback_result

        result = await iceberg_executor.rollback_partial_execution("test_exec_123")

        assert result["success"]
        assert result["rollback_orders"] == 2
        assert mock_market_executor.execute_market_order.call_count == 2

    def test_order_at_200_boundary(self, iceberg_executor):
        """Test order exactly at $200 triggers iceberg slicing."""
        order_value = Decimal("200.00")
        liquidity_profile = LiquidityProfile(
            total_bid_volume=Decimal("10000"),
            total_ask_volume=Decimal("10000"),
            bid_depth_1pct=Decimal("500"),
            ask_depth_1pct=Decimal("500"),
            bid_depth_2pct=Decimal("1000"),
            ask_depth_2pct=Decimal("1000"),
            spread_percent=Decimal("0.1"),
            optimal_slice_count=MIN_SLICES,
            timestamp=datetime.now(),
        )

        slices = iceberg_executor.calculate_slice_sizes(order_value, liquidity_profile)
        assert len(slices) >= MIN_SLICES

    def test_order_at_199_99(self, iceberg_executor):
        """Test order at $199.99 should not trigger iceberg by default."""
        order_value = Decimal("199.99")
        # This would normally bypass iceberg execution in the main method
        assert order_value < iceberg_executor.min_order_value

    @pytest.mark.asyncio
    async def test_concurrent_iceberg_orders(
        self,
        iceberg_executor,
        mock_gateway,
        mock_market_executor,
        mock_repository,
        sample_ticker,
        sample_order_book,
    ):
        """Test handling of concurrent iceberg executions."""
        mock_gateway.get_ticker.return_value = sample_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        # Create two orders
        order1 = Order(
            order_id="order1",
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTCUSDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.01"),
        )

        order2 = Order(
            order_id="order2",
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="ETHUSDT",
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            price=None,
            quantity=Decimal("0.5"),
        )

        # Mock successful slice executions
        success_result = ExecutionResult(
            success=True,
            order=Mock(filled_quantity=Decimal("0.003")),
            message="Executed",
            slippage_percent=Decimal("0.1"),
        )
        mock_market_executor.execute_market_order.return_value = success_result

        # Execute concurrently
        results = await asyncio.gather(
            iceberg_executor.execute_iceberg_order(order1, force_iceberg=True),
            iceberg_executor.execute_iceberg_order(order2, force_iceberg=True),
            return_exceptions=True,
        )

        # Both should complete
        assert all(isinstance(r, ExecutionResult) for r in results)
        # Multiple executions should have been tracked
        assert mock_repository.save_iceberg_execution.call_count >= 2

    def test_calculate_depth_to_price_level_asks(self, iceberg_executor):
        """Test depth calculation for ask side."""
        asks = [[40000.0, 0.5], [40001.0, 1.0], [40002.0, 0.8]]

        # Calculate depth to reach 40001.5
        depth = iceberg_executor._calculate_depth_to_price_level(
            asks, Decimal("40001.5"), is_ask=True
        )

        # Should include first two levels
        expected = Decimal("0.5") * Decimal("40000") + Decimal("1.0") * Decimal("40001")
        assert depth == expected

    def test_calculate_depth_to_price_level_bids(self, iceberg_executor):
        """Test depth calculation for bid side."""
        bids = [[40000.0, 0.5], [39999.0, 1.0], [39998.0, 0.8]]

        # Calculate depth to reach 39998.5
        depth = iceberg_executor._calculate_depth_to_price_level(
            bids, Decimal("39998.5"), is_ask=False
        )

        # Should include first two levels
        expected = Decimal("0.5") * Decimal("40000") + Decimal("1.0") * Decimal("39999")
        assert depth == expected

    @pytest.mark.asyncio
    async def test_network_failure_during_slice(
        self,
        iceberg_executor,
        sample_order,
        mock_gateway,
        mock_market_executor,
        mock_repository,
        sample_ticker,
        sample_order_book,
    ):
        """Test handling of network failure during slice execution."""
        sample_order.quantity = Decimal("0.01")
        mock_gateway.get_ticker.return_value = sample_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        # Mock network failure
        mock_market_executor.execute_market_order.side_effect = Exception(
            "Network timeout"
        )

        with pytest.raises(OrderExecutionError) as exc_info:
            await iceberg_executor.execute_iceberg_order(
                sample_order, force_iceberg=True
            )

        assert "Network timeout" in str(exc_info.value)
