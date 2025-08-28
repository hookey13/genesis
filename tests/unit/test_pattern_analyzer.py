"""
Unit tests for Win/Loss Pattern Analyzer.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from genesis.analytics.pattern_analyzer import (
    PatternAnalyzer,
    WinLossPattern,
)
from genesis.core.models import OrderSide, Position, PositionSide, Trade


class TestWinLossPattern:
    """Test WinLossPattern dataclass."""

    def test_win_loss_pattern_creation(self):
        """Test creating a WinLossPattern."""
        pattern = WinLossPattern(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=Decimal("0.6"),
            current_streak=3,
            max_win_streak=8,
            max_loss_streak=5,
            average_win_streak=Decimal("2.5"),
            average_loss_streak=Decimal("1.8"),
            average_win_size=Decimal("150"),
            average_loss_size=Decimal("100"),
            win_loss_ratio=Decimal("1.5"),
            expectancy=Decimal("50"),
            average_win_duration_hours=Decimal("24"),
            average_loss_duration_hours=Decimal("12"),
            median_win_duration_hours=Decimal("20"),
            median_loss_duration_hours=Decimal("10"),
        )

        assert pattern.total_trades == 100
        assert pattern.win_rate == Decimal("0.6")
        assert pattern.win_loss_ratio == Decimal("1.5")
        assert pattern.expectancy == Decimal("50")

    def test_win_loss_pattern_to_dict(self):
        """Test converting WinLossPattern to dictionary."""
        pattern = WinLossPattern(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=Decimal("0.6"),
            current_streak=-2,
            max_win_streak=5,
            max_loss_streak=3,
            average_win_streak=Decimal("2.0"),
            average_loss_streak=Decimal("1.5"),
            average_win_size=Decimal("100"),
            average_loss_size=Decimal("50"),
            win_loss_ratio=Decimal("2.0"),
            expectancy=Decimal("40"),
            average_win_duration_hours=Decimal("18"),
            average_loss_duration_hours=Decimal("8"),
            median_win_duration_hours=Decimal("16"),
            median_loss_duration_hours=Decimal("7"),
            win_rate_by_hour={12: Decimal("0.7"), 14: Decimal("0.5")},
            close_reason_distribution={"stop_loss": 10, "take_profit": 20},
        )

        result = pattern.to_dict()

        assert result["total_trades"] == 50
        assert result["win_rate"] == "0.6"
        assert result["current_streak"] == -2
        assert result["win_rate_by_hour"]["12"] == "0.7"
        assert result["close_reason_distribution"]["stop_loss"] == 10


class TestPatternAnalyzer:
    """Test PatternAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a PatternAnalyzer instance."""
        return PatternAnalyzer()

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        trades = []

        # Create a pattern: W, W, W, L, L, W, L, W, W, L
        pnl_values = [100, 150, 120, -80, -60, 200, -100, 180, 90, -70]

        for i, pnl in enumerate(pnl_values):
            trades.append(
                Trade(
                    trade_id=f"trade_{i}",
                    order_id=f"order_{i}",
                    strategy_id="test_strategy",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY if pnl > 0 else OrderSide.SELL,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("51000") if pnl > 0 else Decimal("49000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal(str(pnl)),
                    pnl_percent=Decimal(str(pnl / 100)),
                    timestamp=base_time + timedelta(hours=i),
                )
            )

        return trades

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        positions = []

        for i in range(5):
            position = Position(
                position_id=f"pos_{i}",
                account_id="test_account",
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.1"),
                dollar_value=Decimal("5000"),
                pnl_dollars=Decimal("100") if i % 2 == 0 else Decimal("-50"),
                pnl_percent=Decimal("2") if i % 2 == 0 else Decimal("-1"),
                created_at=base_time,
                updated_at=base_time + timedelta(hours=24 if i % 2 == 0 else 12),
            )
            # Add close_reason attribute
            if i == 0:
                position.close_reason = "take_profit"
            elif i == 1:
                position.close_reason = "stop_loss"
            elif i == 2:
                position.close_reason = "take_profit"
            elif i == 3:
                position.close_reason = "manual"
            else:
                position.close_reason = "tilt_intervention"

            positions.append(position)

        return positions

    def test_analyze_patterns_empty_trades(self, analyzer):
        """Test pattern analysis with no trades."""
        pattern = analyzer.analyze_patterns([])

        assert pattern.total_trades == 0
        assert pattern.winning_trades == 0
        assert pattern.losing_trades == 0
        assert pattern.win_rate == Decimal("0")
        assert pattern.max_win_streak == 0
        assert pattern.max_loss_streak == 0

    def test_analyze_patterns_basic(self, analyzer, sample_trades):
        """Test basic pattern analysis."""
        pattern = analyzer.analyze_patterns(sample_trades)

        assert pattern.total_trades == 10
        assert pattern.winning_trades == 6
        assert pattern.losing_trades == 4
        assert pattern.win_rate == Decimal("0.6")

    def test_analyze_streaks(self, analyzer, sample_trades):
        """Test streak analysis."""
        pattern = analyzer.analyze_patterns(sample_trades)

        # Pattern: W, W, W, L, L, W, L, W, W, L
        assert pattern.max_win_streak == 3  # First 3 wins
        assert pattern.max_loss_streak == 2  # Two consecutive losses
        assert pattern.current_streak == -1  # Ends with a loss

    def test_analyze_sizes(self, analyzer, sample_trades):
        """Test win/loss size analysis."""
        pattern = analyzer.analyze_patterns(sample_trades)

        # Average win: (100 + 150 + 120 + 200 + 180 + 90) / 6 = 140
        assert pattern.average_win_size == Decimal("140")

        # Average loss: (80 + 60 + 100 + 70) / 4 = 77.5
        assert pattern.average_loss_size == Decimal("77.50")

        # Win/loss ratio: 140 / 77.5 ≈ 1.81
        assert abs(pattern.win_loss_ratio - Decimal("1.81")) < Decimal("0.01")

        # Expectancy: (0.6 * 140) - (0.4 * 77.5) = 84 - 31 = 53
        assert abs(pattern.expectancy - Decimal("53")) < Decimal("1")

    def test_analyze_durations_with_positions(
        self, analyzer, sample_trades, sample_positions
    ):
        """Test duration analysis with position data."""
        pattern = analyzer.analyze_patterns(sample_trades, sample_positions)

        # With our sample positions:
        # Winning positions (0, 2, 4): 24 hours each
        # Losing positions (1, 3): 12 hours each
        assert pattern.average_win_duration_hours == Decimal("24.0")
        assert pattern.average_loss_duration_hours == Decimal("12.0")

    def test_analyze_durations_without_positions(self, analyzer, sample_trades):
        """Test duration analysis without position data."""
        pattern = analyzer.analyze_patterns(sample_trades, positions=None)

        # Should use default values
        assert pattern.average_win_duration_hours == Decimal("24.0")
        assert pattern.average_loss_duration_hours == Decimal("12.0")

    def test_analyze_temporal_patterns(self, analyzer):
        """Test temporal pattern analysis."""
        # Create trades at specific times
        trades = []
        base_date = datetime(2024, 1, 1, tzinfo=UTC)

        # Morning trades (9-11 AM): mostly wins
        for i in range(5):
            trades.append(
                Trade(
                    trade_id=f"morning_{i}",
                    order_id=f"order_morning_{i}",
                    strategy_id="test",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("51000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal("100") if i < 4 else Decimal("-50"),
                    pnl_percent=Decimal("2"),
                    timestamp=base_date.replace(hour=9 + i % 3),
                )
            )

        # Afternoon trades (14-16 PM): mostly losses
        for i in range(5):
            trades.append(
                Trade(
                    trade_id=f"afternoon_{i}",
                    order_id=f"order_afternoon_{i}",
                    strategy_id="test",
                    symbol="BTC/USDT",
                    side=OrderSide.SELL,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("49000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal("-50") if i < 4 else Decimal("100"),
                    pnl_percent=Decimal("-1"),
                    timestamp=base_date.replace(hour=14 + i % 3),
                )
            )

        pattern = analyzer.analyze_patterns(trades)

        # Check hour-based win rates
        assert 9 in pattern.win_rate_by_hour
        assert 14 in pattern.win_rate_by_hour

        # Morning hours should have higher win rate
        morning_win_rates = [
            pattern.win_rate_by_hour.get(h, Decimal("0")) for h in [9, 10, 11]
        ]
        afternoon_win_rates = [
            pattern.win_rate_by_hour.get(h, Decimal("0")) for h in [14, 15, 16]
        ]

        assert max(morning_win_rates) > max(afternoon_win_rates)

    def test_analyze_close_reasons(self, analyzer, sample_trades, sample_positions):
        """Test close reason analysis."""
        pattern = analyzer.analyze_patterns(sample_trades, sample_positions)

        # Check close reason distribution
        assert pattern.close_reason_distribution["take_profit"] == 2
        assert pattern.close_reason_distribution["stop_loss"] == 1
        assert pattern.close_reason_distribution["manual"] == 1
        assert pattern.close_reason_distribution["tilt_intervention"] == 1

        # Check win/loss close reasons
        assert pattern.win_close_reasons["take_profit"] == 2
        assert pattern.loss_close_reasons["stop_loss"] == 1

    def test_analyze_recovery_patterns(self, analyzer):
        """Test recovery pattern analysis."""
        trades = []
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)

        # Create pattern: L(-100), W(150), L(-80), W(70), L(-50), W(200)
        pnl_sequence = [-100, 150, -80, 70, -50, 200]

        for i, pnl in enumerate(pnl_sequence):
            trades.append(
                Trade(
                    trade_id=f"trade_{i}",
                    order_id=f"order_{i}",
                    strategy_id="test",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("51000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal(str(pnl)),
                    pnl_percent=Decimal("1"),
                    timestamp=base_time + timedelta(hours=i),
                )
            )

        pattern = analyzer.analyze_patterns(trades)

        # After losses: 3 recovery attempts
        # Successful recoveries: 150 > 100, 200 > 50 (2 successful)
        assert pattern.trades_recovering_from_drawdown == 2
        assert pattern.recovery_success_rate == Decimal("0.67")  # 2/3

    def test_identify_best_trading_times(self, analyzer):
        """Test identification of best trading times."""
        pattern = WinLossPattern(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=Decimal("0.6"),
            current_streak=0,
            max_win_streak=5,
            max_loss_streak=3,
            average_win_streak=Decimal("2"),
            average_loss_streak=Decimal("1.5"),
            average_win_size=Decimal("100"),
            average_loss_size=Decimal("50"),
            win_loss_ratio=Decimal("2"),
            expectancy=Decimal("40"),
            average_win_duration_hours=Decimal("20"),
            average_loss_duration_hours=Decimal("10"),
            median_win_duration_hours=Decimal("18"),
            median_loss_duration_hours=Decimal("9"),
            win_rate_by_hour={
                9: Decimal("0.75"),
                10: Decimal("0.70"),
                11: Decimal("0.65"),
                14: Decimal("0.35"),
                15: Decimal("0.30"),
                16: Decimal("0.40"),
            },
            win_rate_by_day_of_week={
                0: Decimal("0.65"),  # Monday
                1: Decimal("0.70"),  # Tuesday
                2: Decimal("0.55"),  # Wednesday
                3: Decimal("0.35"),  # Thursday
                4: Decimal("0.45"),  # Friday
            },
        )

        best_times = analyzer.identify_best_trading_times(pattern)

        # Check best hours
        assert len(best_times["best_trading_hours"]) <= 3
        assert best_times["best_trading_hours"][0]["hour"] == 9
        assert best_times["best_trading_hours"][0]["win_rate"] == "0.75"

        # Check best days
        assert len(best_times["best_trading_days"]) <= 2
        assert best_times["best_trading_days"][0]["day"] == "Tuesday"

        # Check worst hours
        assert len(best_times["worst_trading_hours"]) <= 3
        assert any(h["hour"] == 15 for h in best_times["worst_trading_hours"])

        # Check worst days
        assert len(best_times["worst_trading_days"]) <= 2
        assert any(d["day"] == "Thursday" for d in best_times["worst_trading_days"])

    def test_empty_pattern_method(self, analyzer):
        """Test _empty_pattern method."""
        pattern = analyzer._empty_pattern()

        assert pattern.total_trades == 0
        assert pattern.win_rate == Decimal("0")
        assert pattern.max_win_streak == 0
        assert pattern.average_win_size == Decimal("0")
        assert pattern.expectancy == Decimal("0")
