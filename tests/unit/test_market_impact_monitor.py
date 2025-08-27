"""Unit tests for the Market Impact Monitor."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from genesis.analytics.market_impact_monitor import (
    ImpactSeverity,
    MarketImpactMetric,
    MarketImpactMonitor,
)
from genesis.core.models import TradingTier
from genesis.data.repository import Repository
from genesis.engine.executor.base import Order, OrderSide, OrderStatus, OrderType
from genesis.exchange.models import OrderBook, Ticker


@pytest.fixture
def mock_gateway():
    """Create a mock gateway."""
    gateway = Mock()
    gateway.get_ticker = AsyncMock()
    gateway.get_order_book = AsyncMock()
    return gateway


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = Mock(spec=Repository)
    repo.save_market_impact_metric = AsyncMock()
    repo.save_impact_analysis = AsyncMock()
    return repo


@pytest.fixture
def impact_monitor(mock_gateway, mock_repository):
    """Create a market impact monitor instance."""
    return MarketImpactMonitor(
        gateway=mock_gateway,
        repository=mock_repository,
        tier=TradingTier.HUNTER
    )


@pytest.fixture
def sample_ticker():
    """Create a sample ticker."""
    ticker = Mock(spec=Ticker)
    ticker.symbol = "BTCUSDT"
    ticker.bid_price = Decimal("40000")
    ticker.ask_price = Decimal("40001")
    ticker.last_price = Decimal("40000.5")
    return ticker


@pytest.fixture
def sample_order_book():
    """Create a sample order book."""
    return OrderBook(
        symbol="BTCUSDT",
        bids=[
            [40000.0, 1.0],
            [39999.0, 1.5],
            [39998.0, 2.0],
            [39997.0, 1.2],
            [39996.0, 0.8]
        ],
        asks=[
            [40001.0, 1.0],
            [40002.0, 1.5],
            [40003.0, 2.0],
            [40004.0, 1.2],
            [40005.0, 0.8]
        ],
        timestamp=datetime.now()
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
        quantity=Decimal("0.1"),
        filled_quantity=Decimal("0.1"),
        status=OrderStatus.FILLED,
        slice_number=1,
        total_slices=3,
        created_at=datetime.now()
    )


class TestMarketImpactMonitor:
    """Test suite for MarketImpactMonitor."""

    def test_initialization(self, mock_gateway, mock_repository):
        """Test monitor initialization."""
        monitor = MarketImpactMonitor(
            gateway=mock_gateway,
            repository=mock_repository,
            tier=TradingTier.HUNTER
        )

        assert monitor.gateway == mock_gateway
        assert monitor.repository == mock_repository
        assert monitor.tier == TradingTier.HUNTER
        assert len(monitor.active_monitors) == 0
        assert len(monitor.pre_execution_prices) == 0

    @pytest.mark.asyncio
    async def test_measure_pre_execution_state(
        self, impact_monitor, mock_gateway, sample_ticker, sample_order_book
    ):
        """Test capturing pre-execution market state."""
        execution_id = "test_exec_123"
        symbol = "BTCUSDT"

        mock_gateway.get_ticker.return_value = sample_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        bid, ask, order_book = await impact_monitor.measure_pre_execution_state(
            symbol, execution_id
        )

        assert bid == sample_ticker.bid_price
        assert ask == sample_ticker.ask_price
        assert order_book == sample_order_book
        assert execution_id in impact_monitor.pre_execution_prices
        assert execution_id in impact_monitor.execution_start_times
        assert execution_id in impact_monitor.active_monitors

    @pytest.mark.asyncio
    async def test_measure_slice_impact(
        self, impact_monitor, mock_gateway, sample_order, sample_ticker, sample_order_book
    ):
        """Test measuring impact of a single slice."""
        execution_id = "test_exec_123"
        pre_price = Decimal("40000")
        volume = Decimal("4000")  # 0.1 BTC * $40000

        # Set up post-execution state (slight price increase)
        post_ticker = Mock(spec=Ticker)
        post_ticker.bid_price = Decimal("40005")
        post_ticker.ask_price = Decimal("40010")  # Price moved up

        mock_gateway.get_ticker.return_value = post_ticker
        mock_gateway.get_order_book.return_value = sample_order_book

        metric = await impact_monitor.measure_slice_impact(
            execution_id, sample_order, pre_price, volume
        )

        assert metric.execution_id == execution_id
        assert metric.slice_id == sample_order.order_id
        assert metric.pre_price == pre_price
        assert metric.post_price == post_ticker.ask_price  # Buy order looks at ask
        assert metric.price_impact_percent > 0  # Price moved unfavorably
        assert metric.volume_executed == volume
        assert metric.severity in ImpactSeverity
        assert execution_id in impact_monitor.active_monitors
        assert metric in impact_monitor.active_monitors[execution_id]

    def test_calculate_impact(self, impact_monitor):
        """Test impact percentage calculation."""
        # Test price increase (unfavorable for buy)
        impact = impact_monitor.calculate_impact(
            Decimal("100"),
            Decimal("101"),
            Decimal("1000")
        )
        assert impact == Decimal("1.0000")  # 1% increase

        # Test price decrease (favorable for buy)
        impact = impact_monitor.calculate_impact(
            Decimal("100"),
            Decimal("99"),
            Decimal("1000")
        )
        assert impact == Decimal("-1.0000")  # 1% decrease

        # Test no change
        impact = impact_monitor.calculate_impact(
            Decimal("100"),
            Decimal("100"),
            Decimal("1000")
        )
        assert impact == Decimal("0")

        # Test zero pre-price
        impact = impact_monitor.calculate_impact(
            Decimal("0"),
            Decimal("100"),
            Decimal("1000")
        )
        assert impact == Decimal("0")

    def test_classify_impact_severity(self, impact_monitor):
        """Test impact severity classification."""
        assert impact_monitor._classify_impact_severity(Decimal("0.05")) == ImpactSeverity.NEGLIGIBLE
        assert impact_monitor._classify_impact_severity(Decimal("0.2")) == ImpactSeverity.LOW
        assert impact_monitor._classify_impact_severity(Decimal("0.4")) == ImpactSeverity.MODERATE
        assert impact_monitor._classify_impact_severity(Decimal("0.7")) == ImpactSeverity.HIGH
        assert impact_monitor._classify_impact_severity(Decimal("1.5")) == ImpactSeverity.SEVERE
        assert impact_monitor._classify_impact_severity(Decimal("-0.4")) == ImpactSeverity.MODERATE  # Absolute value

    @pytest.mark.asyncio
    async def test_analyze_execution_impact(self, impact_monitor):
        """Test comprehensive execution impact analysis."""
        execution_id = "test_exec_123"
        symbol = "BTCUSDT"
        total_volume = Decimal("12000")  # 0.3 BTC * $40000
        slice_count = 3

        # Create mock metrics
        metrics = [
            MarketImpactMetric(
                impact_id=str(uuid4()),
                execution_id=execution_id,
                slice_id=f"slice_{i}",
                symbol=symbol,
                side=OrderSide.BUY,
                pre_price=Decimal("40000"),
                post_price=Decimal(f"4000{i}"),
                price_impact_percent=Decimal(f"0.{i}"),
                volume_executed=Decimal("4000"),
                order_book_depth_usdt=Decimal("50000"),
                bid_ask_spread=Decimal("1"),
                liquidity_consumed_percent=None,
                market_depth_1pct=Decimal("50000"),
                market_depth_2pct=Decimal("100000"),
                cumulative_impact=None,
                severity=ImpactSeverity.LOW,
                measured_at=datetime.now() + timedelta(seconds=i*3),
                notes=f"Slice {i+1}/3"
            )
            for i in range(3)
        ]

        impact_monitor.active_monitors[execution_id] = metrics
        impact_monitor.execution_start_times[execution_id] = datetime.now() - timedelta(seconds=10)

        analysis = await impact_monitor.analyze_execution_impact(
            execution_id, symbol, total_volume, slice_count
        )

        assert analysis.execution_id == execution_id
        assert analysis.symbol == symbol
        assert analysis.total_volume == total_volume
        assert analysis.slice_count == slice_count
        assert analysis.average_impact_per_slice == sum(m.price_impact_percent for m in metrics) / 3
        assert analysis.max_slice_impact == Decimal("0.2")
        assert analysis.min_slice_impact == Decimal("0.0")
        assert analysis.severity_distribution[ImpactSeverity.LOW] == 3
        assert analysis.optimal_slice_size > 0
        assert analysis.recommended_delay_seconds > 0
        assert execution_id not in impact_monitor.active_monitors  # Cleaned up

    @pytest.mark.asyncio
    async def test_analyze_execution_impact_no_metrics(self, impact_monitor):
        """Test analysis when no metrics are available."""
        execution_id = "empty_exec"
        symbol = "BTCUSDT"
        total_volume = Decimal("10000")
        slice_count = 5

        analysis = await impact_monitor.analyze_execution_impact(
            execution_id, symbol, total_volume, slice_count
        )

        assert analysis.execution_id == execution_id
        assert analysis.total_impact_percent == Decimal("0")
        assert analysis.average_impact_per_slice == Decimal("0")
        assert all(count == 0 for count in analysis.severity_distribution.values())

    def test_calculate_depth_at_level(self, impact_monitor, sample_order_book):
        """Test order book depth calculation at price levels."""
        # Test buy side (looking at asks)
        depth = impact_monitor._calculate_depth_at_level(
            sample_order_book,
            Decimal("1"),  # 1% level
            OrderSide.BUY
        )
        assert depth > 0

        # Test sell side (looking at bids)
        depth = impact_monitor._calculate_depth_at_level(
            sample_order_book,
            Decimal("1"),
            OrderSide.SELL
        )
        assert depth > 0

        # Test empty order book
        empty_book = OrderBook(
            symbol="EMPTY",
            bids=[],
            asks=[],
            timestamp=datetime.now()
        )
        depth = impact_monitor._calculate_depth_at_level(
            empty_book,
            Decimal("1"),
            OrderSide.BUY
        )
        assert depth == Decimal("0")

    def test_recommend_optimal_slice_size(self, impact_monitor):
        """Test optimal slice size recommendations."""
        total_volume = Decimal("10000")
        slice_count = 5
        current_size = total_volume / slice_count

        # Very low impact - should increase size
        size = impact_monitor._recommend_optimal_slice_size(
            total_volume, slice_count, Decimal("0.05")
        )
        assert size > current_size

        # Moderate impact - should maintain or slightly reduce
        size = impact_monitor._recommend_optimal_slice_size(
            total_volume, slice_count, Decimal("0.25")
        )
        assert size <= current_size

        # High impact - should reduce significantly
        size = impact_monitor._recommend_optimal_slice_size(
            total_volume, slice_count, Decimal("0.8")
        )
        assert size < current_size * Decimal("0.8")

    def test_recommend_delay(self, impact_monitor):
        """Test delay recommendations."""
        # Low impact, low volatility - short delay
        delay = impact_monitor._recommend_delay(Decimal("0.1"), Decimal("0.05"))
        assert delay == 2.0

        # Moderate impact - medium delay
        delay = impact_monitor._recommend_delay(Decimal("0.3"), Decimal("0.1"))
        assert delay == 3.5

        # High impact - maximum delay
        delay = impact_monitor._recommend_delay(Decimal("0.7"), Decimal("0.2"))
        assert delay == 5.0

    def test_calculate_max_safe_volume(self, impact_monitor):
        """Test maximum safe volume calculation."""
        # Create metrics with varying impact
        metrics = [
            MarketImpactMetric(
                impact_id=str(uuid4()),
                execution_id="test",
                slice_id=f"slice_{i}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                pre_price=Decimal("40000"),
                post_price=Decimal("40000"),
                price_impact_percent=Decimal(f"0.{i}"),
                volume_executed=Decimal(f"{1000 * (i+1)}"),
                order_book_depth_usdt=None,
                bid_ask_spread=None,
                liquidity_consumed_percent=None,
                market_depth_1pct=None,
                market_depth_2pct=None,
                cumulative_impact=None,
                severity=ImpactSeverity.LOW,
                measured_at=datetime.now()
            )
            for i in range(5)
        ]

        # Metrics 0, 1, 2 have impact < 0.3%
        safe_volume = impact_monitor._calculate_max_safe_volume(metrics)
        assert safe_volume == Decimal("3000")  # Max of safe volumes

        # No safe volumes
        for metric in metrics:
            metric.price_impact_percent = Decimal("0.5")

        safe_volume = impact_monitor._calculate_max_safe_volume(metrics)
        assert safe_volume == Decimal("500")  # Half of minimum volume

    @pytest.mark.asyncio
    async def test_create_impact_dashboard_widget(self, impact_monitor):
        """Test dashboard widget creation."""
        # Add some active monitors
        execution_id = "test_exec"
        metric = MarketImpactMetric(
            impact_id=str(uuid4()),
            execution_id=execution_id,
            slice_id="slice_1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            pre_price=Decimal("40000"),
            post_price=Decimal("40010"),
            price_impact_percent=Decimal("0.025"),
            volume_executed=Decimal("4000"),
            order_book_depth_usdt=None,
            bid_ask_spread=None,
            liquidity_consumed_percent=None,
            market_depth_1pct=None,
            market_depth_2pct=None,
            cumulative_impact=None,
            severity=ImpactSeverity.NEGLIGIBLE,
            measured_at=datetime.now()
        )

        impact_monitor.active_monitors[execution_id] = [metric]

        widget_data = await impact_monitor.create_impact_dashboard_widget()

        assert widget_data["type"] == "market_impact"
        assert len(widget_data["active_executions"]) == 1
        assert widget_data["active_executions"][0]["execution_id"] == execution_id
        assert widget_data["active_executions"][0]["symbol"] == "BTCUSDT"
        assert widget_data["active_executions"][0]["severity"] == "NEGLIGIBLE"

    @pytest.mark.asyncio
    async def test_error_handling_in_measure_slice_impact(
        self, impact_monitor, mock_gateway, sample_order
    ):
        """Test error handling when measuring slice impact fails."""
        execution_id = "test_exec"
        pre_price = Decimal("40000")
        volume = Decimal("4000")

        # Make gateway raise an error
        mock_gateway.get_ticker.side_effect = Exception("Network error")

        metric = await impact_monitor.measure_slice_impact(
            execution_id, sample_order, pre_price, volume
        )

        # Should return default metric with zero impact
        assert metric.price_impact_percent == Decimal("0")
        assert metric.severity == ImpactSeverity.NEGLIGIBLE
        assert metric.notes == "Measurement failed"

    @pytest.mark.asyncio
    async def test_monitor_price_recovery(self, impact_monitor, mock_gateway):
        """Test price recovery monitoring."""
        execution_id = "test_exec"
        symbol = "BTCUSDT"

        # Set pre-execution prices
        impact_monitor.pre_execution_prices[execution_id] = (
            Decimal("40000"),  # bid
            Decimal("40001")   # ask
        )

        # Mock ticker showing recovery
        recovered_ticker = Mock(spec=Ticker)
        recovered_ticker.bid_price = Decimal("39990")  # Within 0.99 * pre_bid
        recovered_ticker.ask_price = Decimal("40010")  # Within 1.01 * pre_ask

        mock_gateway.get_ticker.return_value = recovered_ticker

        # Use a short monitoring period for testing
        impact_monitor.RECOVERY_MONITORING_PERIOD = timedelta(seconds=1)

        recovery_time = await impact_monitor._monitor_price_recovery(
            execution_id, symbol
        )

        assert recovery_time is not None
        assert recovery_time >= 0
