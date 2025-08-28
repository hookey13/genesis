"""Unit tests for VWAPTracker."""

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from genesis.analytics.vwap_tracker import (
    ExecutionPerformance,
    Trade,
    VWAPMetrics,
    VWAPTracker,
)
from genesis.core.events import Event, EventType
from genesis.core.models import Side, Symbol
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    event_bus = Mock(spec=EventBus)
    event_bus.emit = AsyncMock()
    return event_bus


@pytest.fixture
def vwap_tracker(mock_event_bus):
    """Create a VWAPTracker instance with mock dependencies."""
    return VWAPTracker(mock_event_bus, window_minutes=30)


@pytest.fixture
def sample_trades():
    """Generate sample trade data."""
    base_time = datetime.now(UTC)
    trades = [
        Trade(
            timestamp=base_time - timedelta(minutes=20),
            price=Decimal("100.00"),
            volume=Decimal("10"),
            side=Side.BUY,
        ),
        Trade(
            timestamp=base_time - timedelta(minutes=15),
            price=Decimal("101.00"),
            volume=Decimal("15"),
            side=Side.SELL,
        ),
        Trade(
            timestamp=base_time - timedelta(minutes=10),
            price=Decimal("100.50"),
            volume=Decimal("20"),
            side=Side.BUY,
        ),
        Trade(
            timestamp=base_time - timedelta(minutes=5),
            price=Decimal("99.50"),
            volume=Decimal("25"),
            side=Side.SELL,
        ),
        Trade(
            timestamp=base_time,
            price=Decimal("100.00"),
            volume=Decimal("30"),
            side=Side.BUY,
        ),
    ]
    return trades


class TestTrade:
    """Test Trade dataclass."""

    def test_trade_value_calculation(self):
        """Test that trade value is calculated correctly."""
        trade = Trade(
            timestamp=datetime.now(UTC), price=Decimal("100.50"), volume=Decimal("10.5")
        )

        expected_value = Decimal("100.50") * Decimal("10.5")
        assert trade.value == expected_value

    def test_trade_with_side(self):
        """Test trade with side information."""
        trade = Trade(
            timestamp=datetime.now(UTC),
            price=Decimal("100"),
            volume=Decimal("10"),
            side=Side.BUY,
        )

        assert trade.side == Side.BUY
        assert trade.value == Decimal("1000")


class TestVWAPMetrics:
    """Test VWAPMetrics dataclass."""

    def test_metrics_to_dict(self):
        """Test conversion of metrics to dictionary."""
        metrics = VWAPMetrics(
            symbol=Symbol("BTC/USDT"),
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            vwap=Decimal("100.50"),
            total_volume=Decimal("1000"),
            total_value=Decimal("100500"),
            trade_count=100,
            time_window_minutes=30,
        )

        result = metrics.to_dict()

        assert result["symbol"] == "BTC/USDT"
        assert "2024-01-01" in result["timestamp"]
        assert result["vwap"] == "100.50"
        assert result["total_volume"] == "1000"
        assert result["total_value"] == "100500"
        assert result["trade_count"] == 100
        assert result["time_window_minutes"] == 30


class TestExecutionPerformance:
    """Test ExecutionPerformance dataclass."""

    def test_performance_to_dict(self):
        """Test conversion of performance to dictionary."""
        performance = ExecutionPerformance(
            symbol=Symbol("BTC/USDT"),
            execution_id="exec_123",
            start_time=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            end_time=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
            executed_volume=Decimal("100"),
            executed_value=Decimal("10050"),
            execution_vwap=Decimal("100.50"),
            market_vwap=Decimal("100.00"),
            slippage_bps=Decimal("50"),
            fill_rate=Decimal("95"),
            trades_executed=10,
        )

        result = performance.to_dict()

        assert result["symbol"] == "BTC/USDT"
        assert result["execution_id"] == "exec_123"
        assert "2024-01-01" in result["start_time"]
        assert "2024-01-01" in result["end_time"]
        assert result["executed_volume"] == "100"
        assert result["executed_value"] == "10050"
        assert result["execution_vwap"] == "100.50"
        assert result["market_vwap"] == "100.00"
        assert result["slippage_bps"] == "50"
        assert result["fill_rate"] == "95"
        assert result["trades_executed"] == 10

    def test_performance_to_dict_no_end_time(self):
        """Test conversion with no end time."""
        performance = ExecutionPerformance(
            symbol=Symbol("BTC/USDT"),
            execution_id="exec_123",
            start_time=datetime.now(UTC),
            end_time=None,
            executed_volume=Decimal("0"),
            executed_value=Decimal("0"),
            execution_vwap=Decimal("0"),
            market_vwap=Decimal("100"),
            slippage_bps=Decimal("0"),
            fill_rate=Decimal("0"),
            trades_executed=0,
        )

        result = performance.to_dict()
        assert result["end_time"] is None


class TestVWAPTracker:
    """Test VWAPTracker class."""

    @pytest.mark.asyncio
    async def test_start_stop(self, vwap_tracker):
        """Test starting and stopping the tracker."""
        await vwap_tracker.start()
        assert vwap_tracker._update_task is not None
        assert not vwap_tracker._update_task.done()

        await vwap_tracker.stop()
        assert vwap_tracker._update_task is None

    def test_add_trade(self, vwap_tracker):
        """Test adding trades to the tracker."""
        symbol = Symbol("BTC/USDT")
        trade = Trade(
            timestamp=datetime.now(UTC), price=Decimal("100"), volume=Decimal("10")
        )

        vwap_tracker.add_trade(symbol, trade)

        assert "BTC/USDT" in vwap_tracker._trades
        assert len(vwap_tracker._trades["BTC/USDT"]) == 1
        assert vwap_tracker._trades["BTC/USDT"][0] == trade

    def test_clean_old_trades(self, vwap_tracker):
        """Test removal of old trades outside window."""
        symbol_str = "BTC/USDT"
        now = datetime.now(UTC)

        # Add trades with different timestamps
        old_trade = Trade(
            timestamp=now - timedelta(minutes=40),  # Outside 30-min window
            price=Decimal("100"),
            volume=Decimal("10"),
        )
        recent_trade = Trade(
            timestamp=now - timedelta(minutes=10),  # Within window
            price=Decimal("101"),
            volume=Decimal("15"),
        )

        vwap_tracker._trades[symbol_str] = deque([old_trade, recent_trade])
        vwap_tracker._clean_old_trades(symbol_str)

        # Old trade should be removed
        assert len(vwap_tracker._trades[symbol_str]) == 1
        assert vwap_tracker._trades[symbol_str][0] == recent_trade

    def test_calculate_vwap_single_trade(self, vwap_tracker):
        """Test VWAP calculation with single trade."""
        symbol_str = "BTC/USDT"
        trade = Trade(
            timestamp=datetime.now(UTC), price=Decimal("100"), volume=Decimal("10")
        )

        vwap_tracker._trades[symbol_str] = deque([trade])
        metrics = vwap_tracker._calculate_vwap(symbol_str)

        assert metrics is not None
        assert metrics.vwap == Decimal("100")
        assert metrics.total_volume == Decimal("10")
        assert metrics.total_value == Decimal("1000")
        assert metrics.trade_count == 1

    def test_calculate_vwap_multiple_trades(self, vwap_tracker, sample_trades):
        """Test VWAP calculation with multiple trades."""
        symbol_str = "BTC/USDT"
        vwap_tracker._trades[symbol_str] = deque(sample_trades)

        metrics = vwap_tracker._calculate_vwap(symbol_str)

        # Calculate expected VWAP
        total_value = sum(t.value for t in sample_trades)
        total_volume = sum(t.volume for t in sample_trades)
        expected_vwap = total_value / total_volume

        assert metrics is not None
        assert metrics.vwap == expected_vwap
        assert metrics.total_volume == total_volume
        assert metrics.total_value == total_value
        assert metrics.trade_count == len(sample_trades)

    def test_calculate_vwap_no_trades(self, vwap_tracker):
        """Test VWAP calculation with no trades."""
        metrics = vwap_tracker._calculate_vwap("BTC/USDT")
        assert metrics is None

    def test_calculate_vwap_zero_volume(self, vwap_tracker):
        """Test VWAP calculation with zero volume trades."""
        symbol_str = "BTC/USDT"
        trade = Trade(
            timestamp=datetime.now(UTC), price=Decimal("100"), volume=Decimal("0")
        )

        vwap_tracker._trades[symbol_str] = deque([trade])
        metrics = vwap_tracker._calculate_vwap(symbol_str)

        assert metrics is None  # Should return None for zero volume

    @pytest.mark.asyncio
    async def test_emit_metrics(self, vwap_tracker, mock_event_bus):
        """Test emission of VWAP metrics via event bus."""
        metrics = VWAPMetrics(
            symbol=Symbol("BTC/USDT"),
            timestamp=datetime.now(UTC),
            vwap=Decimal("100"),
            total_volume=Decimal("1000"),
            total_value=Decimal("100000"),
            trade_count=50,
            time_window_minutes=30,
        )

        await vwap_tracker._emit_metrics(metrics)

        mock_event_bus.emit.assert_called_once()
        emitted_event = mock_event_bus.emit.call_args[0][0]

        assert isinstance(emitted_event, Event)
        assert emitted_event.event_type == EventType.METRICS_UPDATE
        assert emitted_event.event_data["metric_type"] == "vwap"
        assert emitted_event.event_data["symbol"] == "BTC/USDT"

    def test_get_current_vwap_cached(self, vwap_tracker):
        """Test getting current VWAP from cache."""
        symbol = Symbol("BTC/USDT")
        cached_metrics = VWAPMetrics(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            vwap=Decimal("100.50"),
            total_volume=Decimal("1000"),
            total_value=Decimal("100500"),
            trade_count=50,
            time_window_minutes=30,
        )

        vwap_tracker._current_vwap["BTC/USDT"] = cached_metrics

        vwap = vwap_tracker.get_current_vwap(symbol)
        assert vwap == Decimal("100.50")

    def test_get_current_vwap_calculate(self, vwap_tracker):
        """Test getting current VWAP with calculation."""
        symbol = Symbol("BTC/USDT")
        trade = Trade(
            timestamp=datetime.now(UTC), price=Decimal("100"), volume=Decimal("10")
        )

        vwap_tracker._trades["BTC/USDT"] = deque([trade])

        vwap = vwap_tracker.get_current_vwap(symbol)
        assert vwap == Decimal("100")

    def test_get_current_vwap_none(self, vwap_tracker):
        """Test getting current VWAP when not available."""
        symbol = Symbol("BTC/USDT")
        vwap = vwap_tracker.get_current_vwap(symbol)
        assert vwap is None

    def test_start_execution_tracking(self, vwap_tracker):
        """Test starting execution tracking."""
        symbol = Symbol("BTC/USDT")
        execution_id = "exec_123"
        target_volume = Decimal("100")

        # Set current market VWAP
        vwap_tracker._current_vwap["BTC/USDT"] = VWAPMetrics(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            vwap=Decimal("100"),
            total_volume=Decimal("1000"),
            total_value=Decimal("100000"),
            trade_count=50,
            time_window_minutes=30,
        )

        performance = vwap_tracker.start_execution_tracking(
            symbol, execution_id, target_volume
        )

        assert performance.symbol == symbol
        assert performance.execution_id == execution_id
        assert performance.market_vwap == Decimal("100")
        assert performance.executed_volume == Decimal("0")
        assert performance.trades_executed == 0
        assert execution_id in vwap_tracker._executions

    def test_update_execution(self, vwap_tracker):
        """Test updating execution with fills."""
        execution_id = "exec_123"
        symbol = Symbol("BTC/USDT")

        # Start tracking
        performance = vwap_tracker.start_execution_tracking(
            symbol, execution_id, Decimal("100")
        )

        # Update with fills
        vwap_tracker.update_execution(execution_id, Decimal("100"), Decimal("10"))
        vwap_tracker.update_execution(execution_id, Decimal("101"), Decimal("15"))

        updated_performance = vwap_tracker._executions[execution_id]

        assert updated_performance.executed_volume == Decimal("25")
        assert updated_performance.executed_value == Decimal("2515")  # 100*10 + 101*15
        assert updated_performance.execution_vwap == Decimal("2515") / Decimal("25")
        assert updated_performance.trades_executed == 2

    def test_update_execution_not_found(self, vwap_tracker):
        """Test updating non-existent execution."""
        # Should not raise, just log warning
        vwap_tracker.update_execution("nonexistent", Decimal("100"), Decimal("10"))
        # No assertion needed - just verify no exception

    def test_complete_execution(self, vwap_tracker):
        """Test completing execution tracking."""
        execution_id = "exec_123"
        symbol = Symbol("BTC/USDT")
        target_volume = Decimal("100")

        # Start and update execution
        vwap_tracker.start_execution_tracking(symbol, execution_id, target_volume)
        vwap_tracker.update_execution(execution_id, Decimal("100"), Decimal("50"))
        vwap_tracker.update_execution(execution_id, Decimal("101"), Decimal("50"))

        # Complete execution
        final_performance = vwap_tracker.complete_execution(execution_id, target_volume)

        assert final_performance is not None
        assert final_performance.end_time is not None
        assert final_performance.fill_rate == Decimal("100")  # 100% filled
        assert execution_id not in vwap_tracker._executions
        assert final_performance in vwap_tracker._performance_history

    def test_complete_execution_partial_fill(self, vwap_tracker):
        """Test completing execution with partial fill."""
        execution_id = "exec_123"
        symbol = Symbol("BTC/USDT")
        target_volume = Decimal("100")

        vwap_tracker.start_execution_tracking(symbol, execution_id, target_volume)
        vwap_tracker.update_execution(execution_id, Decimal("100"), Decimal("30"))

        final_performance = vwap_tracker.complete_execution(execution_id, target_volume)

        assert final_performance.fill_rate == Decimal("30")  # 30% filled

    def test_complete_execution_not_found(self, vwap_tracker):
        """Test completing non-existent execution."""
        result = vwap_tracker.complete_execution("nonexistent", Decimal("100"))
        assert result is None

    def test_get_performance_stats_no_history(self, vwap_tracker):
        """Test getting performance stats with no history."""
        stats = vwap_tracker.get_performance_stats()

        assert stats["executions"] == 0
        assert stats["avg_slippage_bps"] == "0"
        assert stats["avg_fill_rate"] == "0"
        assert stats["total_volume"] == "0"

    def test_get_performance_stats_with_history(self, vwap_tracker):
        """Test getting performance stats with execution history."""
        # Add completed executions to history
        perf1 = ExecutionPerformance(
            symbol=Symbol("BTC/USDT"),
            execution_id="exec_1",
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC),
            executed_volume=Decimal("100"),
            executed_value=Decimal("10050"),
            execution_vwap=Decimal("100.50"),
            market_vwap=Decimal("100"),
            slippage_bps=Decimal("50"),
            fill_rate=Decimal("100"),
            trades_executed=10,
        )

        perf2 = ExecutionPerformance(
            symbol=Symbol("BTC/USDT"),
            execution_id="exec_2",
            start_time=datetime.now(UTC) - timedelta(hours=2),
            end_time=datetime.now(UTC) - timedelta(hours=1),
            executed_volume=Decimal("200"),
            executed_value=Decimal("19900"),
            execution_vwap=Decimal("99.50"),
            market_vwap=Decimal("100"),
            slippage_bps=Decimal("-50"),  # Negative slippage (good)
            fill_rate=Decimal("95"),
            trades_executed=20,
        )

        vwap_tracker._performance_history = [perf1, perf2]

        stats = vwap_tracker.get_performance_stats(hours=24)

        assert stats["executions"] == 2
        assert stats["avg_slippage_bps"] == "0"  # (50 + -50) / 2
        assert stats["avg_fill_rate"] == "97.5"  # (100 + 95) / 2
        assert stats["total_volume"] == "300"
        assert stats["best_execution"] == "exec_2"  # Lower slippage
        assert stats["worst_execution"] == "exec_1"  # Higher slippage

    def test_get_performance_stats_filtered_by_symbol(self, vwap_tracker):
        """Test getting performance stats filtered by symbol."""
        perf1 = ExecutionPerformance(
            symbol=Symbol("BTC/USDT"),
            execution_id="exec_1",
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC),
            executed_volume=Decimal("100"),
            executed_value=Decimal("10000"),
            execution_vwap=Decimal("100"),
            market_vwap=Decimal("100"),
            slippage_bps=Decimal("0"),
            fill_rate=Decimal("100"),
            trades_executed=10,
        )

        perf2 = ExecutionPerformance(
            symbol=Symbol("ETH/USDT"),
            execution_id="exec_2",
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC),
            executed_volume=Decimal("200"),
            executed_value=Decimal("20000"),
            execution_vwap=Decimal("100"),
            market_vwap=Decimal("100"),
            slippage_bps=Decimal("0"),
            fill_rate=Decimal("100"),
            trades_executed=20,
        )

        vwap_tracker._performance_history = [perf1, perf2]

        # Get stats for BTC only
        stats = vwap_tracker.get_performance_stats(symbol=Symbol("BTC/USDT"))
        assert stats["executions"] == 1
        assert stats["total_volume"] == "100"

    @pytest.mark.asyncio
    async def test_calculate_real_time_vwap(self, vwap_tracker, sample_trades):
        """Test real-time VWAP calculation."""
        vwap = await vwap_tracker.calculate_real_time_vwap(sample_trades)

        # Calculate expected VWAP
        total_value = sum(t.value for t in sample_trades)
        total_volume = sum(t.volume for t in sample_trades)
        expected_vwap = total_value / total_volume

        assert vwap == expected_vwap

    @pytest.mark.asyncio
    async def test_calculate_real_time_vwap_with_window(
        self, vwap_tracker, sample_trades
    ):
        """Test real-time VWAP calculation with time window."""
        # Filter trades within 15-minute window
        window = timedelta(minutes=15)
        vwap = await vwap_tracker.calculate_real_time_vwap(sample_trades, window)

        # Calculate expected VWAP for recent trades only
        cutoff = datetime.now(UTC) - window
        recent_trades = [t for t in sample_trades if t.timestamp > cutoff]

        if recent_trades:
            total_value = sum(t.value for t in recent_trades)
            total_volume = sum(t.volume for t in recent_trades)
            expected_vwap = total_value / total_volume
            assert vwap == expected_vwap
        else:
            assert vwap == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_real_time_vwap_empty(self, vwap_tracker):
        """Test real-time VWAP with empty trades."""
        vwap = await vwap_tracker.calculate_real_time_vwap([])
        assert vwap == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_real_time_vwap_zero_volume(self, vwap_tracker):
        """Test real-time VWAP with zero volume trades."""
        trades = [
            Trade(
                timestamp=datetime.now(UTC), price=Decimal("100"), volume=Decimal("0")
            )
        ]

        vwap = await vwap_tracker.calculate_real_time_vwap(trades)
        assert vwap == Decimal("0")
