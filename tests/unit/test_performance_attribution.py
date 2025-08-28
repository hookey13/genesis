"""
Unit tests for Performance Attribution Engine.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.analytics.performance_attribution import (
    AttributionPeriod,
    AttributionResult,
    PerformanceAttributionEngine,
)
from genesis.core.models import OrderSide, Trade


class TestAttributionResult:
    """Test AttributionResult dataclass."""

    def test_attribution_result_creation(self):
        """Test creating an AttributionResult."""
        result = AttributionResult(
            period_start=datetime(2024, 1, 1, tzinfo=UTC),
            period_end=datetime(2024, 1, 31, tzinfo=UTC),
            attribution_type="strategy",
            attribution_key="strategy_1",
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            total_pnl=Decimal("5000"),
            win_rate=Decimal("0.6"),
            average_win=Decimal("150"),
            average_loss=Decimal("100"),
            profit_factor=Decimal("2.25"),
            max_consecutive_wins=5,
            max_consecutive_losses=3,
            largest_win=Decimal("500"),
            largest_loss=Decimal("-300"),
            total_volume=Decimal("100000"),
        )

        assert result.total_trades == 100
        assert result.win_rate == Decimal("0.6")
        assert result.profit_factor == Decimal("2.25")

    def test_attribution_result_to_dict(self):
        """Test converting AttributionResult to dictionary."""
        result = AttributionResult(
            period_start=datetime(2024, 1, 1, tzinfo=UTC),
            period_end=datetime(2024, 1, 31, tzinfo=UTC),
            attribution_type="pair",
            attribution_key="BTC/USDT",
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            total_pnl=Decimal("2500"),
            win_rate=Decimal("0.6"),
            average_win=Decimal("100"),
            average_loss=Decimal("50"),
            profit_factor=Decimal("3.0"),
            max_consecutive_wins=4,
            max_consecutive_losses=2,
            largest_win=Decimal("300"),
            largest_loss=Decimal("-150"),
            total_volume=Decimal("50000"),
            metadata={"extra": "data"},
        )

        result_dict = result.to_dict()

        assert result_dict["attribution_type"] == "pair"
        assert result_dict["attribution_key"] == "BTC/USDT"
        assert result_dict["total_trades"] == 50
        assert result_dict["total_pnl"] == "2500"
        assert result_dict["metadata"] == {"extra": "data"}


class TestPerformanceAttributionEngine:
    """Test PerformanceAttributionEngine class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository."""
        repo = AsyncMock()
        repo.query_events = AsyncMock(return_value=[])
        repo.query_positions_with_mae = AsyncMock(return_value=[])
        repo.store_attribution_result = AsyncMock()
        return repo

    @pytest.fixture
    def attribution_engine(self, mock_repository):
        """Create a PerformanceAttributionEngine instance."""
        return PerformanceAttributionEngine(mock_repository)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        trades = []

        # Create winning trades
        for i in range(6):
            trades.append(
                Trade(
                    trade_id=f"trade_{i}",
                    order_id=f"order_{i}",
                    strategy_id="strategy_1",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("51000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal("100"),
                    pnl_percent=Decimal("2"),
                    timestamp=base_time + timedelta(hours=i),
                )
            )

        # Create losing trades
        for i in range(4):
            trades.append(
                Trade(
                    trade_id=f"trade_loss_{i}",
                    order_id=f"order_loss_{i}",
                    strategy_id="strategy_1",
                    symbol="BTC/USDT",
                    side=OrderSide.SELL,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("49500"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal("-50"),
                    pnl_percent=Decimal("-1"),
                    timestamp=base_time + timedelta(hours=6 + i),
                )
            )

        return trades

    @pytest.mark.asyncio
    async def test_attribute_by_strategy(
        self, attribution_engine, mock_repository, sample_trades
    ):
        """Test attribution by strategy."""
        # Setup mock to return trade events
        mock_events = [
            {
                "event_data": {
                    "trade_id": trade.trade_id,
                    "order_id": trade.order_id,
                    "position_id": trade.position_id,
                    "strategy_id": trade.strategy_id,
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "entry_price": str(trade.entry_price),
                    "exit_price": str(trade.exit_price),
                    "quantity": str(trade.quantity),
                    "pnl_dollars": str(trade.pnl_dollars),
                    "pnl_percent": str(trade.pnl_percent),
                },
                "created_at": trade.timestamp.isoformat(),
            }
            for trade in sample_trades
        ]
        mock_repository.query_events.return_value = mock_events

        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        results = await attribution_engine.attribute_by_strategy(start_date, end_date)

        assert len(results) == 1
        result = results[0]
        assert result.attribution_type == "strategy"
        assert result.attribution_key == "strategy_1"
        assert result.total_trades == 10
        assert result.winning_trades == 6
        assert result.losing_trades == 4
        assert result.total_pnl == Decimal("400")  # 6*100 - 4*50
        assert result.win_rate == Decimal("0.6")
        assert result.profit_factor == Decimal("3")  # 600/200

        # Verify repository was called
        mock_repository.query_events.assert_called_once()
        mock_repository.store_attribution_result.assert_called()

    @pytest.mark.asyncio
    async def test_attribute_by_pair(self, attribution_engine, mock_repository):
        """Test attribution by trading pair."""
        # Create trades for different pairs
        trades_btc = [
            {
                "event_data": {
                    "trade_id": "t1",
                    "order_id": "o1",
                    "strategy_id": "s1",
                    "symbol": "BTC/USDT",
                    "side": "BUY",
                    "entry_price": "50000",
                    "exit_price": "51000",
                    "quantity": "0.1",
                    "pnl_dollars": "100",
                    "pnl_percent": "2",
                },
                "created_at": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
            }
        ]

        trades_eth = [
            {
                "event_data": {
                    "trade_id": "t2",
                    "order_id": "o2",
                    "strategy_id": "s1",
                    "symbol": "ETH/USDT",
                    "side": "BUY",
                    "entry_price": "3000",
                    "exit_price": "3100",
                    "quantity": "1",
                    "pnl_dollars": "100",
                    "pnl_percent": "3.33",
                },
                "created_at": datetime(2024, 1, 2, tzinfo=UTC).isoformat(),
            }
        ]

        mock_repository.query_events.return_value = trades_btc + trades_eth

        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        results = await attribution_engine.attribute_by_pair(start_date, end_date)

        assert len(results) == 2

        # Check BTC/USDT results
        btc_result = next(r for r in results if r.attribution_key == "BTC/USDT")
        assert btc_result.attribution_type == "pair"
        assert btc_result.total_trades == 1
        assert btc_result.total_pnl == Decimal("100")

        # Check ETH/USDT results
        eth_result = next(r for r in results if r.attribution_key == "ETH/USDT")
        assert eth_result.attribution_type == "pair"
        assert eth_result.total_trades == 1
        assert eth_result.total_pnl == Decimal("100")

    @pytest.mark.asyncio
    async def test_attribute_by_time_period_daily(
        self, attribution_engine, mock_repository
    ):
        """Test attribution by daily time period."""
        # Create trades across different days
        trades = []
        for day in range(3):
            timestamp = datetime(2024, 1, day + 1, 12, 0, tzinfo=UTC)
            trades.append(
                {
                    "event_data": {
                        "trade_id": f"t{day}",
                        "order_id": f"o{day}",
                        "strategy_id": "s1",
                        "symbol": "BTC/USDT",
                        "side": "BUY",
                        "entry_price": "50000",
                        "exit_price": "51000",
                        "quantity": "0.1",
                        "pnl_dollars": "100",
                        "pnl_percent": "2",
                    },
                    "created_at": timestamp.isoformat(),
                }
            )

        mock_repository.query_events.return_value = trades

        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        results = await attribution_engine.attribute_by_time_period(
            start_date, end_date, AttributionPeriod.DAILY
        )

        assert len(results) == 3

        # Check that each day has one trade
        for result in results:
            assert result.attribution_type == "time"
            assert result.total_trades == 1
            assert result.total_pnl == Decimal("100")

    @pytest.mark.asyncio
    async def test_calculate_attribution_empty_trades(self, attribution_engine):
        """Test attribution calculation with no trades."""
        result = attribution_engine._calculate_attribution(
            trades=[],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
            attribution_type="strategy",
            attribution_key="empty_strategy",
        )

        assert result.total_trades == 0
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.total_pnl == Decimal("0")
        assert result.win_rate == Decimal("0")
        assert result.profit_factor == Decimal("0")

    def test_calculate_attribution_with_trades(self, attribution_engine, sample_trades):
        """Test attribution calculation with trades."""
        result = attribution_engine._calculate_attribution(
            trades=sample_trades,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
            attribution_type="strategy",
            attribution_key="test_strategy",
        )

        assert result.total_trades == 10
        assert result.winning_trades == 6
        assert result.losing_trades == 4
        assert result.total_pnl == Decimal("400")
        assert result.win_rate == Decimal("0.6")
        assert result.average_win == Decimal("100")
        assert result.average_loss == Decimal("50")
        assert result.profit_factor == Decimal("3")
        assert result.max_consecutive_wins == 6  # All wins are consecutive
        assert result.max_consecutive_losses == 4  # All losses are consecutive
        assert result.largest_win == Decimal("100")
        assert result.largest_loss == Decimal("-50")

    @pytest.mark.asyncio
    async def test_get_mae_analysis(self, attribution_engine, mock_repository):
        """Test MAE analysis."""
        # Mock positions with MAE data
        positions = [
            {
                "position_id": "p1",
                "strategy_id": "s1",
                "symbol": "BTC/USDT",
                "max_adverse_excursion": Decimal("100"),
                "pnl_dollars": Decimal("200"),  # Recovered
            },
            {
                "position_id": "p2",
                "strategy_id": "s1",
                "symbol": "ETH/USDT",
                "max_adverse_excursion": Decimal("150"),
                "pnl_dollars": Decimal("-50"),  # Did not recover
            },
            {
                "position_id": "p3",
                "strategy_id": "s2",
                "symbol": "BTC/USDT",
                "max_adverse_excursion": Decimal("75"),
                "pnl_dollars": Decimal("100"),  # Recovered
            },
        ]

        mock_repository.query_positions_with_mae.return_value = positions

        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        mae_stats = await attribution_engine.get_mae_analysis(start_date, end_date)

        assert mae_stats["total_positions"] == 3
        assert mae_stats["recovered_positions"] == 2
        assert mae_stats["average_mae"] == Decimal("325") / Decimal("3")
        assert mae_stats["max_mae"] == Decimal("150")
        assert "s1" in mae_stats["mae_by_strategy"]
        assert "s2" in mae_stats["mae_by_strategy"]
        assert "BTC/USDT" in mae_stats["mae_by_pair"]
        assert "ETH/USDT" in mae_stats["mae_by_pair"]

    def test_get_period_key_hourly(self, attribution_engine):
        """Test period key generation for hourly periods."""
        timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        key = attribution_engine._get_period_key(timestamp, AttributionPeriod.HOURLY)
        assert key == "2024-01-15 14:00"

    def test_get_period_key_daily(self, attribution_engine):
        """Test period key generation for daily periods."""
        timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        key = attribution_engine._get_period_key(timestamp, AttributionPeriod.DAILY)
        assert key == "2024-01-15"

    def test_get_period_key_weekly(self, attribution_engine):
        """Test period key generation for weekly periods."""
        timestamp = datetime(2024, 1, 15, tzinfo=UTC)  # Monday
        key = attribution_engine._get_period_key(timestamp, AttributionPeriod.WEEKLY)
        assert key == "2024-W03"

    def test_get_period_key_monthly(self, attribution_engine):
        """Test period key generation for monthly periods."""
        timestamp = datetime(2024, 1, 15, tzinfo=UTC)
        key = attribution_engine._get_period_key(timestamp, AttributionPeriod.MONTHLY)
        assert key == "2024-01"

    def test_get_period_key_quarterly(self, attribution_engine):
        """Test period key generation for quarterly periods."""
        timestamp = datetime(2024, 4, 15, tzinfo=UTC)  # Q2
        key = attribution_engine._get_period_key(timestamp, AttributionPeriod.QUARTERLY)
        assert key == "2024-Q2"

    def test_get_period_key_yearly(self, attribution_engine):
        """Test period key generation for yearly periods."""
        timestamp = datetime(2024, 6, 15, tzinfo=UTC)
        key = attribution_engine._get_period_key(timestamp, AttributionPeriod.YEARLY)
        assert key == "2024"

    def test_get_period_bounds_daily(self, attribution_engine):
        """Test period bounds calculation for daily periods."""
        start, end = attribution_engine._get_period_bounds(
            "2024-01-15", AttributionPeriod.DAILY
        )
        assert start == datetime(2024, 1, 15, tzinfo=UTC)
        assert end == datetime(2024, 1, 16, tzinfo=UTC)

    def test_get_period_bounds_monthly(self, attribution_engine):
        """Test period bounds calculation for monthly periods."""
        start, end = attribution_engine._get_period_bounds(
            "2024-01", AttributionPeriod.MONTHLY
        )
        assert start == datetime(2024, 1, 1, tzinfo=UTC)
        assert end == datetime(2024, 2, 1, tzinfo=UTC)

    def test_get_period_bounds_quarterly(self, attribution_engine):
        """Test period bounds calculation for quarterly periods."""
        start, end = attribution_engine._get_period_bounds(
            "2024-Q2", AttributionPeriod.QUARTERLY
        )
        assert start == datetime(2024, 4, 1, tzinfo=UTC)
        assert end == datetime(2024, 7, 1, tzinfo=UTC)

    def test_get_period_bounds_yearly(self, attribution_engine):
        """Test period bounds calculation for yearly periods."""
        start, end = attribution_engine._get_period_bounds(
            "2024", AttributionPeriod.YEARLY
        )
        assert start == datetime(2024, 1, 1, tzinfo=UTC)
        assert end == datetime(2025, 1, 1, tzinfo=UTC)

    async def test_mae_recovery_edge_case_no_recovery(
        self, attribution_engine, mock_repository
    ):
        """Test MAE analysis when no positions recover from drawdown."""
        positions = [
            {
                "position_id": "p1",
                "strategy_id": "s1",
                "symbol": "BTC/USDT",
                "max_adverse_excursion": Decimal("500"),
                "recovered_from_mae": False,
                "pnl_dollars": Decimal("-500"),
            },
            {
                "position_id": "p2",
                "strategy_id": "s1",
                "symbol": "ETH/USDT",
                "max_adverse_excursion": Decimal("300"),
                "recovered_from_mae": False,
                "pnl_dollars": Decimal("-300"),
            },
        ]

        mock_repository.query_positions_with_mae.return_value = positions

        mae_stats = await attribution_engine.get_mae_analysis(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )

        assert mae_stats["total_positions"] == 2
        assert mae_stats["recovered_positions"] == 0
        assert mae_stats["recovery_rate"] == Decimal("0")
        assert mae_stats["average_mae"] == Decimal("400")

    async def test_mae_recovery_edge_case_all_recover(
        self, attribution_engine, mock_repository
    ):
        """Test MAE analysis when all positions recover from drawdown."""
        positions = [
            {
                "position_id": "p1",
                "strategy_id": "s1",
                "symbol": "BTC/USDT",
                "max_adverse_excursion": Decimal("200"),
                "recovered_from_mae": True,
                "pnl_dollars": Decimal("100"),
            },
            {
                "position_id": "p2",
                "strategy_id": "s1",
                "symbol": "ETH/USDT",
                "max_adverse_excursion": Decimal("150"),
                "recovered_from_mae": True,
                "pnl_dollars": Decimal("75"),
            },
        ]

        mock_repository.query_positions_with_mae.return_value = positions

        mae_stats = await attribution_engine.get_mae_analysis(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )

        assert mae_stats["total_positions"] == 2
        assert mae_stats["recovered_positions"] == 2
        assert mae_stats["recovery_rate"] == Decimal("1.0")
        assert mae_stats["average_mae"] == Decimal("175")

    async def test_mae_recovery_edge_case_zero_mae(
        self, attribution_engine, mock_repository
    ):
        """Test MAE analysis with positions having zero MAE."""
        positions = [
            {
                "position_id": "p1",
                "strategy_id": "s1",
                "symbol": "BTC/USDT",
                "max_adverse_excursion": Decimal("0"),
                "recovered_from_mae": True,
                "pnl_dollars": Decimal("500"),
            }
        ]

        mock_repository.query_positions_with_mae.return_value = positions

        mae_stats = await attribution_engine.get_mae_analysis(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )

        assert mae_stats["total_positions"] == 1
        assert mae_stats["average_mae"] == Decimal("0")
        assert mae_stats["min_mae"] == Decimal("0")
        assert mae_stats["max_mae"] == Decimal("0")

    async def test_mae_recovery_edge_case_large_drawdowns(
        self, attribution_engine, mock_repository
    ):
        """Test MAE analysis with extremely large drawdowns."""
        positions = [
            {
                "position_id": "p1",
                "strategy_id": "high_risk",
                "symbol": "BTC/USDT",
                "max_adverse_excursion": Decimal("10000"),
                "recovered_from_mae": False,
                "pnl_dollars": Decimal("-8000"),
            },
            {
                "position_id": "p2",
                "strategy_id": "high_risk",
                "symbol": "ETH/USDT",
                "max_adverse_excursion": Decimal("15000"),
                "recovered_from_mae": True,
                "pnl_dollars": Decimal("500"),
            },
        ]

        mock_repository.query_positions_with_mae.return_value = positions

        mae_stats = await attribution_engine.get_mae_analysis(
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)
        )

        assert mae_stats["total_positions"] == 2
        assert mae_stats["max_mae"] == Decimal("15000")
        assert mae_stats["average_mae"] == Decimal("12500")
        assert mae_stats["recovered_positions"] == 1
        assert mae_stats["recovery_rate"] == Decimal("0.5")
