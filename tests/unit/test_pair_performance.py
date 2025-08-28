"""Unit tests for pair performance tracking system."""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.analytics.pair_performance import (
    PairMetrics,
    PairPerformanceTracker,
    PeriodType,
)
from genesis.core.models import Position, PositionSide


class TestPairMetrics:
    """Test PairMetrics class."""

    def test_expectancy_calculation(self):
        """Test expectancy calculation."""
        metrics = PairMetrics(
            symbol="BTC/USDT",
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl_dollars=Decimal("500"),
            average_win_dollars=Decimal("200"),
            average_loss_dollars=Decimal("-125"),
            win_rate=Decimal("0.6"),
            profit_factor=Decimal("1.5"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_dollars=Decimal("300"),
            volume_traded_base=Decimal("10"),
            volume_traded_quote=Decimal("500000"),
            fees_paid_dollars=Decimal("50"),
            best_trade_pnl=Decimal("400"),
            worst_trade_pnl=Decimal("-200"),
            average_hold_time_minutes=30.5,
        )

        assert metrics.expectancy == Decimal("50")  # 500 / 10

    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        metrics = PairMetrics(
            symbol="BTC/USDT",
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl_dollars=Decimal("500"),
            average_win_dollars=Decimal("200"),
            average_loss_dollars=Decimal("-100"),
            win_rate=Decimal("0.6"),
            profit_factor=Decimal("2.0"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown_dollars=Decimal("200"),
            volume_traded_base=Decimal("10"),
            volume_traded_quote=Decimal("500000"),
            fees_paid_dollars=Decimal("50"),
            best_trade_pnl=Decimal("400"),
            worst_trade_pnl=Decimal("-150"),
            average_hold_time_minutes=25,
        )

        assert metrics.risk_reward_ratio == Decimal("2")  # 200 / 100


class TestPairPerformanceTracker:
    """Test PairPerformanceTracker class."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.save_trade_performance = AsyncMock()
        repo.get_trades_by_symbol = AsyncMock(return_value=[])
        repo.get_traded_symbols = AsyncMock(return_value=[])
        return repo

    @pytest.fixture
    def account_id(self):
        """Generate test account ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def tracker(self, mock_repository, account_id):
        """Create PairPerformanceTracker instance."""
        return PairPerformanceTracker(repository=mock_repository, account_id=account_id)

    @pytest.fixture
    def sample_position_closed(self, account_id):
        """Create sample closed position."""
        position = Position(
            position_id=str(uuid.uuid4()),
            account_id=account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("25500"),
            pnl_dollars=Decimal("500"),
            opened_at=datetime.utcnow() - timedelta(hours=2),
            closed_at=datetime.utcnow(),
        )
        return position

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return [
            {
                "trade_id": str(uuid.uuid4()),
                "pnl_dollars": Decimal("200"),
                "volume_base": Decimal("0.1"),
                "volume_quote": Decimal("5000"),
                "fees_paid": Decimal("5"),
                "hold_time_minutes": 30,
                "closed_at": datetime.utcnow() - timedelta(hours=1),
            },
            {
                "trade_id": str(uuid.uuid4()),
                "pnl_dollars": Decimal("-100"),
                "volume_base": Decimal("0.05"),
                "volume_quote": Decimal("2500"),
                "fees_paid": Decimal("2.5"),
                "hold_time_minutes": 45,
                "closed_at": datetime.utcnow() - timedelta(hours=2),
            },
            {
                "trade_id": str(uuid.uuid4()),
                "pnl_dollars": Decimal("300"),
                "volume_base": Decimal("0.15"),
                "volume_quote": Decimal("7500"),
                "fees_paid": Decimal("7.5"),
                "hold_time_minutes": 20,
                "closed_at": datetime.utcnow() - timedelta(hours=3),
            },
        ]

    @pytest.mark.asyncio
    async def test_track_trade(self, tracker, mock_repository, sample_position_closed):
        """Test tracking a completed trade."""
        await tracker.track_trade(sample_position_closed)

        # Should save to repository
        mock_repository.save_trade_performance.assert_called_once()
        saved_data = mock_repository.save_trade_performance.call_args[0][0]

        assert saved_data["account_id"] == tracker.account_id
        assert saved_data["position_id"] == sample_position_closed.position_id
        assert saved_data["symbol"] == "BTC/USDT"
        assert saved_data["pnl_dollars"] == Decimal("500")
        assert saved_data["is_winner"] is True

    @pytest.mark.asyncio
    async def test_track_trade_open_position(
        self, tracker, mock_repository, account_id
    ):
        """Test that open positions are not tracked."""
        open_position = Position(
            position_id=str(uuid.uuid4()),
            account_id=account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            opened_at=datetime.utcnow(),
            closed_at=None,  # Still open
        )

        await tracker.track_trade(open_position)

        # Should not save
        mock_repository.save_trade_performance.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_pair_metrics_no_trades(self, tracker, mock_repository):
        """Test getting metrics with no trades."""
        mock_repository.get_trades_by_symbol.return_value = []

        metrics = await tracker.get_pair_metrics("BTC/USDT")

        assert metrics.symbol == "BTC/USDT"
        assert metrics.total_trades == 0
        assert metrics.total_pnl_dollars == Decimal("0")
        assert metrics.win_rate == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_pair_metrics_with_trades(
        self, tracker, mock_repository, sample_trades
    ):
        """Test getting metrics with trade data."""
        mock_repository.get_trades_by_symbol.return_value = sample_trades

        metrics = await tracker.get_pair_metrics("BTC/USDT")

        assert metrics.symbol == "BTC/USDT"
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.total_pnl_dollars == Decimal("400")  # 200 - 100 + 300
        assert metrics.win_rate == Decimal("2") / Decimal("3")

    @pytest.mark.asyncio
    async def test_metrics_caching(self, tracker, mock_repository, sample_trades):
        """Test that metrics are cached."""
        mock_repository.get_trades_by_symbol.return_value = sample_trades

        # First call
        metrics1 = await tracker.get_pair_metrics("BTC/USDT")

        # Second call (should use cache)
        metrics2 = await tracker.get_pair_metrics("BTC/USDT")

        # Repository should only be called once
        assert mock_repository.get_trades_by_symbol.call_count == 1
        assert metrics1 == metrics2

    @pytest.mark.asyncio
    async def test_generate_attribution_report_empty(self, tracker, mock_repository):
        """Test attribution report with no data."""
        mock_repository.get_traded_symbols.return_value = []

        report = await tracker.generate_attribution_report()

        assert report.total_pnl_dollars == Decimal("0")
        assert report.pair_contributions == {}
        assert report.best_performer is None
        assert report.worst_performer is None

    @pytest.mark.asyncio
    async def test_generate_attribution_report(self, tracker, mock_repository):
        """Test attribution report generation."""
        # Mock traded symbols
        mock_repository.get_traded_symbols.return_value = ["BTC/USDT", "ETH/USDT"]

        # Mock trades for each symbol
        btc_trades = [
            {
                "pnl_dollars": Decimal("500"),
                "volume_quote": Decimal("25000"),
                "hold_time_minutes": 30,
                "closed_at": datetime.utcnow(),
            }
        ]

        eth_trades = [
            {
                "pnl_dollars": Decimal("-200"),
                "volume_quote": Decimal("10000"),
                "hold_time_minutes": 45,
                "closed_at": datetime.utcnow(),
            }
        ]

        # Setup mock to return different trades for different symbols
        def get_trades_side_effect(account_id, symbol, **kwargs):
            if symbol == "BTC/USDT":
                return btc_trades
            elif symbol == "ETH/USDT":
                return eth_trades
            return []

        mock_repository.get_trades_by_symbol.side_effect = get_trades_side_effect

        report = await tracker.generate_attribution_report()

        assert report.total_pnl_dollars == Decimal("300")  # 500 - 200
        assert report.pair_contributions["BTC/USDT"] == Decimal("500")
        assert report.pair_contributions["ETH/USDT"] == Decimal("-200")
        assert report.best_performer == "BTC/USDT"
        assert report.worst_performer == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_get_historical_performance(self, tracker, mock_repository):
        """Test getting historical performance."""
        # Mock trades for different periods
        mock_repository.get_trades_by_symbol.return_value = [
            {
                "pnl_dollars": Decimal("100"),
                "volume_quote": Decimal("5000"),
                "hold_time_minutes": 30,
                "closed_at": datetime.utcnow(),
            }
        ]

        historical = await tracker.get_historical_performance(
            "BTC/USDT", periods=7, period_type=PeriodType.DAILY
        )

        assert len(historical) == 7
        assert all(m.symbol == "BTC/USDT" for m in historical)

    @pytest.mark.asyncio
    async def test_compare_pairs(self, tracker, mock_repository):
        """Test comparing multiple pairs."""

        # Setup different metrics for each pair
        def get_trades_side_effect(account_id, symbol, **kwargs):
            if symbol == "BTC/USDT":
                return [
                    {"pnl_dollars": Decimal("1000"), "volume_quote": Decimal("50000")}
                ]
            elif symbol == "ETH/USDT":
                return [
                    {"pnl_dollars": Decimal("500"), "volume_quote": Decimal("30000")}
                ]
            elif symbol == "SOL/USDT":
                return [
                    {"pnl_dollars": Decimal("-200"), "volume_quote": Decimal("10000")}
                ]
            return []

        mock_repository.get_trades_by_symbol.side_effect = get_trades_side_effect

        comparison = await tracker.compare_pairs(["BTC/USDT", "ETH/USDT", "SOL/USDT"])

        assert len(comparison) == 3
        assert "BTC/USDT" in comparison
        assert "ETH/USDT" in comparison
        assert "SOL/USDT" in comparison

        # Check rankings
        assert comparison["BTC/USDT"]["pnl_rank"] == 1
        assert comparison["ETH/USDT"]["pnl_rank"] == 2
        assert comparison["SOL/USDT"]["pnl_rank"] == 3

    def test_calculate_sharpe_ratio(self, tracker):
        """Test Sharpe ratio calculation."""
        returns = [100, -50, 200, -25, 150, 75, -100, 300, 50, -75]

        sharpe = tracker._calculate_sharpe_ratio(returns)

        # Should return a reasonable Sharpe ratio
        assert isinstance(sharpe, float)
        assert -10 < sharpe < 10  # Reasonable range

    def test_calculate_max_drawdown(self, tracker):
        """Test maximum drawdown calculation."""
        trades = [
            {"pnl_dollars": 100, "closed_at": datetime.utcnow() - timedelta(days=5)},
            {"pnl_dollars": 200, "closed_at": datetime.utcnow() - timedelta(days=4)},
            {"pnl_dollars": -150, "closed_at": datetime.utcnow() - timedelta(days=3)},
            {"pnl_dollars": -100, "closed_at": datetime.utcnow() - timedelta(days=2)},
            {"pnl_dollars": 50, "closed_at": datetime.utcnow() - timedelta(days=1)},
        ]

        max_dd = tracker._calculate_max_drawdown(trades)

        # Peak was 300 (100+200), dropped to 50, so drawdown is 250
        assert max_dd == Decimal("250")

    def test_generate_recommendations(self, tracker):
        """Test recommendation generation."""
        pair_metrics = {
            "BTC/USDT": PairMetrics(
                symbol="BTC/USDT",
                total_trades=20,
                winning_trades=6,
                losing_trades=14,
                total_pnl_dollars=Decimal("-500"),
                average_win_dollars=Decimal("100"),
                average_loss_dollars=Decimal("-100"),
                win_rate=Decimal("0.3"),  # Low win rate
                profit_factor=Decimal("0.5"),
                sharpe_ratio=Decimal("-1.2"),  # Negative Sharpe
                max_drawdown_dollars=Decimal("1000"),
                volume_traded_base=Decimal("10"),
                volume_traded_quote=Decimal("500000"),
                fees_paid_dollars=Decimal("100"),
                best_trade_pnl=Decimal("200"),
                worst_trade_pnl=Decimal("-300"),
                average_hold_time_minutes=30,
            ),
            "ETH/USDT": PairMetrics(
                symbol="ETH/USDT",
                total_trades=15,
                winning_trades=10,
                losing_trades=5,
                total_pnl_dollars=Decimal("1000"),
                average_win_dollars=Decimal("150"),
                average_loss_dollars=Decimal("-100"),
                win_rate=Decimal("0.67"),
                profit_factor=Decimal("2.0"),
                sharpe_ratio=Decimal("1.5"),
                max_drawdown_dollars=Decimal("200"),
                volume_traded_base=Decimal("50"),
                volume_traded_quote=Decimal("150000"),
                fees_paid_dollars=Decimal("50"),
                best_trade_pnl=Decimal("300"),
                worst_trade_pnl=Decimal("-150"),
                average_hold_time_minutes=25,
            ),
        }

        pair_weights = {
            "BTC/USDT": Decimal("0.7"),  # High concentration
            "ETH/USDT": Decimal("0.3"),
        }

        recommendations = tracker._generate_recommendations(
            pair_metrics, pair_weights, Decimal("0")
        )

        assert len(recommendations) > 0
        # Should recommend reviewing BTC/USDT due to poor performance
        assert any("BTC/USDT" in r for r in recommendations)
        # Should recommend diversifying due to high concentration
        assert any("concentrated" in r.lower() for r in recommendations)
