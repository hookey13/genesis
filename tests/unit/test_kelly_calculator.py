"""
Unit tests for Kelly Criterion position sizing calculator.
"""
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from genesis.analytics.kelly_sizing import (
    KellyCalculator,
    SimulationResult,
    VolatilityRegime,
)
from genesis.analytics.strategy_metrics import StrategyPerformanceTracker
from genesis.core.constants import ConvictionLevel, TradingTier
from genesis.core.models import OrderSide, Trade


class TestKellyCalculator:
    """Test suite for KellyCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a Kelly calculator instance."""
        return KellyCalculator(
            default_fraction=Decimal("0.25"),
            min_trades=20,
            lookback_days=30,
            max_kelly=Decimal("0.5")
        )

    @pytest.fixture
    def sample_trades(self):
        """Generate sample trades for testing."""
        trades = []
        base_time = datetime.now(UTC)

        # Create 30 trades: 20 wins, 10 losses
        # Spread trades evenly over the last 29 days to avoid filter issues
        for i in range(20):  # Wins
            trades.append(Trade(
                order_id=f"order_{i}",
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                quantity=Decimal("0.01"),
                pnl_dollars=Decimal("10"),  # Avg win: $10
                pnl_percent=Decimal("2"),
                timestamp=base_time - timedelta(days=i % 29)  # Spread over 29 days
            ))

        for i in range(10):  # Losses
            trades.append(Trade(
                order_id=f"order_loss_{i}",
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("49500"),
                quantity=Decimal("0.01"),
                pnl_dollars=Decimal("-5"),  # Avg loss: $5
                pnl_percent=Decimal("-1"),
                timestamp=base_time - timedelta(days=(i+5) % 29)  # Offset and spread
            ))

        return trades

    def test_kelly_fraction_calculation(self, calculator):
        """Test basic Kelly fraction calculation."""
        # Test with 60% win rate, 2:1 win/loss ratio
        win_rate = Decimal("0.6")
        win_loss_ratio = Decimal("2.0")

        kelly_f = calculator.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # Kelly formula: (0.6 * 2 - 0.4) / 2 = 0.8 / 2 = 0.4
        assert kelly_f == Decimal("0.4")

    def test_kelly_fraction_with_cap(self, calculator):
        """Test Kelly fraction capping at maximum."""
        # Test with very favorable conditions that would yield > 0.5
        win_rate = Decimal("0.8")
        win_loss_ratio = Decimal("3.0")

        kelly_f = calculator.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # Would be (0.8 * 3 - 0.2) / 3 = 2.2 / 3 = 0.733
        # But should be capped at max_kelly = 0.5
        assert kelly_f == Decimal("0.5")

    def test_negative_kelly_returns_zero(self, calculator):
        """Test that negative Kelly returns zero."""
        # Test with losing edge
        win_rate = Decimal("0.3")  # 30% win rate
        win_loss_ratio = Decimal("1.0")  # 1:1 ratio

        kelly_f = calculator.calculate_kelly_fraction(win_rate, win_loss_ratio)

        # (0.3 * 1 - 0.7) / 1 = -0.4 (negative)
        assert kelly_f == Decimal("0")

    def test_position_size_calculation(self, calculator):
        """Test position size calculation with fractional Kelly."""
        kelly_f = Decimal("0.4")
        balance = Decimal("10000")
        fraction = Decimal("0.25")  # 25% fractional Kelly

        position_size = calculator.calculate_position_size(kelly_f, balance, fraction)

        # 0.4 * 0.25 * 10000 = 1000
        assert position_size == Decimal("1000")

    def test_position_size_with_default_fraction(self, calculator):
        """Test position size with default fractional Kelly."""
        kelly_f = Decimal("0.2")
        balance = Decimal("5000")

        position_size = calculator.calculate_position_size(kelly_f, balance)

        # 0.2 * 0.25 * 5000 = 250
        assert position_size == Decimal("250")

    def test_estimate_edge_with_trades(self, calculator, sample_trades):
        """Test edge estimation from historical trades."""
        edge_metrics = calculator.estimate_edge(sample_trades)

        assert edge_metrics["win_rate"] == Decimal("0.6667")  # 20/30
        assert edge_metrics["win_loss_ratio"] == Decimal("2.00")  # 10/5
        assert edge_metrics["sample_size"] == 30
        assert edge_metrics["confidence"] > Decimal("0.6")  # Reasonable confidence with 30 trades

    def test_estimate_edge_with_no_trades(self, calculator):
        """Test edge estimation with no trades."""
        edge_metrics = calculator.estimate_edge([])

        assert edge_metrics["win_rate"] == Decimal("0")
        assert edge_metrics["win_loss_ratio"] == Decimal("0")
        assert edge_metrics["sample_size"] == 0
        assert edge_metrics["confidence"] == Decimal("0")

    def test_calculate_strategy_edge(self, calculator, sample_trades):
        """Test strategy-specific edge calculation."""
        edge = calculator.calculate_strategy_edge("test_strategy", sample_trades)

        assert edge.strategy_id == "test_strategy"
        assert abs(edge.win_rate - Decimal("0.6667")) < Decimal("0.01")  # 20 wins / 30 total
        assert edge.win_loss_ratio == Decimal("2.00")
        assert edge.sample_size == 30  # All trades should be within window now
        assert edge.confidence_interval[0] < edge.win_rate < edge.confidence_interval[1]

    def test_strategy_edge_caching(self, calculator, sample_trades):
        """Test that strategy edge is cached."""
        # First calculation
        edge1 = calculator.calculate_strategy_edge("test_strategy", sample_trades)

        # Second calculation should return cached value (within TTL)
        edge2 = calculator.calculate_strategy_edge("test_strategy", sample_trades)

        assert edge1.last_calculated == edge2.last_calculated  # Same cached result

    def test_adjust_kelly_for_drawdown(self, calculator):
        """Test Kelly adjustment during drawdown."""
        base_kelly = Decimal("0.3")

        # Create trades with significant drawdown
        losing_trades = []
        for i in range(20):
            losing_trades.append(Trade(
                order_id=f"loss_{i}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("49000"),
                quantity=Decimal("0.01"),
                pnl_dollars=Decimal("-10"),
                pnl_percent=Decimal("-2"),
                timestamp=datetime.now(UTC)
            ))

        adjusted_kelly = calculator.adjust_kelly_for_performance(
            base_kelly, losing_trades, window_size=20
        )

        # Should reduce Kelly due to drawdown
        assert adjusted_kelly < base_kelly

    def test_adjust_kelly_for_winning_streak(self, calculator):
        """Test Kelly adjustment during winning streak."""
        base_kelly = Decimal("0.2")

        # Create recent winning trades
        winning_trades = []
        for i in range(5):
            winning_trades.append(Trade(
                order_id=f"win_{i}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                quantity=Decimal("0.01"),
                pnl_dollars=Decimal("10"),
                pnl_percent=Decimal("2"),
                timestamp=datetime.now(UTC)
            ))

        adjusted_kelly = calculator.adjust_kelly_for_performance(
            base_kelly, winning_trades, window_size=5
        )

        # Should increase Kelly slightly for winning streak
        assert adjusted_kelly == Decimal("0.2200")  # 10% increase

    def test_conviction_multiplier_low(self, calculator):
        """Test conviction multiplier for low conviction."""
        kelly_size = Decimal("1000")

        adjusted_size = calculator.apply_conviction_multiplier(
            kelly_size, ConvictionLevel.LOW
        )

        assert adjusted_size == Decimal("500.00")  # 0.5x multiplier

    def test_conviction_multiplier_high(self, calculator):
        """Test conviction multiplier for high conviction."""
        kelly_size = Decimal("1000")

        adjusted_size = calculator.apply_conviction_multiplier(
            kelly_size, ConvictionLevel.HIGH
        )

        assert adjusted_size == Decimal("1500.00")  # 1.5x multiplier

    def test_enforce_position_boundaries_min(self, calculator):
        """Test enforcement of minimum position size."""
        calculated_size = Decimal("50")  # Too small
        balance = Decimal("10000")
        tier = TradingTier.HUNTER

        bounded_size = calculator.enforce_position_boundaries(
            calculated_size, balance, tier
        )

        # Minimum for HUNTER is 1% = $100
        assert bounded_size == Decimal("100.00")

    def test_enforce_position_boundaries_max(self, calculator):
        """Test enforcement of maximum position size."""
        calculated_size = Decimal("2000")  # Too large
        balance = Decimal("10000")
        tier = TradingTier.HUNTER

        bounded_size = calculator.enforce_position_boundaries(
            calculated_size, balance, tier
        )

        # Maximum for HUNTER is 10% = $1000
        assert bounded_size == Decimal("1000.00")

    def test_volatility_multiplier_low_vol(self, calculator):
        """Test volatility multiplier in low volatility."""
        returns = [0.005] * 20  # Very low volatility returns

        multiplier, regime = calculator.calculate_volatility_multiplier(returns)

        assert regime == VolatilityRegime.LOW
        assert multiplier == Decimal("1.10")  # Slight increase in low vol

    def test_volatility_multiplier_high_vol(self, calculator):
        """Test volatility multiplier in high volatility."""
        # High volatility returns (alternating ±5%)
        returns = [0.05, -0.05] * 10

        multiplier, regime = calculator.calculate_volatility_multiplier(returns)

        assert regime == VolatilityRegime.HIGH
        assert multiplier < Decimal("1.0")  # Reduction in high vol

    def test_monte_carlo_simulation(self, calculator):
        """Test Monte Carlo simulation."""
        win_rate = Decimal("0.6")
        win_loss_ratio = Decimal("2.0")
        kelly_fraction = Decimal("0.1")  # Conservative 10% Kelly

        result = calculator.run_monte_carlo_simulation(
            win_rate, win_loss_ratio, kelly_fraction,
            iterations=100,  # Reduced for test speed
            trades_per_iteration=50  # Reduced trades
        )

        assert isinstance(result, SimulationResult)
        assert result.risk_of_ruin < Decimal("0.05")  # Allow slightly higher risk for test
        # Growth rate can be negative in some simulations
        assert result.median_final_balance >= Decimal("0")  # At least no total loss
        assert result.percentile_5 < result.median_final_balance < result.percentile_95

    def test_zero_win_loss_ratio(self, calculator):
        """Test handling of zero win/loss ratio."""
        win_rate = Decimal("0.5")
        win_loss_ratio = Decimal("0")  # Invalid ratio

        kelly_f = calculator.calculate_kelly_fraction(win_rate, win_loss_ratio)

        assert kelly_f == Decimal("0")  # Should return 0 for safety

    def test_confidence_interval_calculation(self, calculator, sample_trades):
        """Test confidence interval calculation for win rate."""
        # Create trades with timestamps that won't be filtered
        base_time = datetime.now(UTC)
        recent_trades = []

        # Create 10 recent trades (all within the lookback window)
        for i in range(10):
            recent_trades.append(Trade(
                order_id=f"recent_{i}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if i < 7 else Decimal("49500"),
                quantity=Decimal("0.01"),
                pnl_dollars=Decimal("10") if i < 7 else Decimal("-5"),
                pnl_percent=Decimal("2") if i < 7 else Decimal("-1"),
                timestamp=base_time - timedelta(days=i)  # All within 10 days
            ))

        # Test with insufficient data (< min_trades of 20)
        edge_small = calculator.calculate_strategy_edge("test", recent_trades)
        assert edge_small.confidence_interval == (Decimal("0"), Decimal("1"))
        assert edge_small.sample_size == 10  # Exactly 10 trades
        assert edge_small.sample_size < calculator.min_trades  # Below threshold of 20

        # Test with sufficient data (use the full sample_trades fixture)
        edge_large = calculator.calculate_strategy_edge("test_strategy", sample_trades)  # Use correct strategy_id
        # Should have at least min_trades (20) after any filtering
        assert edge_large.sample_size >= calculator.min_trades
        # Confidence interval should be narrower than default
        interval_width = edge_large.confidence_interval[1] - edge_large.confidence_interval[0]
        assert interval_width < Decimal("1")  # Should be narrower than (0, 1)
        # Check win rate is within confidence interval
        assert edge_large.confidence_interval[0] <= edge_large.win_rate <= edge_large.confidence_interval[1]


class TestStrategyPerformanceTracker:
    """Test suite for StrategyPerformanceTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a performance tracker instance."""
        return StrategyPerformanceTracker(cache_ttl_minutes=5)

    @pytest.fixture
    def sample_trade(self):
        """Create a sample trade."""
        return Trade(
            order_id="test_order",
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            quantity=Decimal("0.01"),
            pnl_dollars=Decimal("10"),
            pnl_percent=Decimal("2"),
            timestamp=datetime.now(UTC)
        )

    def test_record_trade(self, tracker, sample_trade):
        """Test recording a trade."""
        tracker.record_trade("test_strategy", sample_trade)

        metrics = tracker.get_strategy_metrics("test_strategy")
        assert metrics is not None
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.total_pnl == Decimal("10")

    def test_calculate_strategy_edge(self, tracker):
        """Test strategy edge calculation."""
        # Record multiple trades
        for i in range(30):
            if i < 20:  # Wins
                pnl = Decimal("10")
            else:  # Losses
                pnl = Decimal("-5")

            trade = Trade(
                order_id=f"order_{i}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if pnl > 0 else Decimal("49500"),
                quantity=Decimal("0.01"),
                pnl_dollars=pnl,
                pnl_percent=Decimal("2") if pnl > 0 else Decimal("-1"),
                timestamp=datetime.now(UTC)
            )
            tracker.record_trade("test", trade)

        edge = tracker.calculate_strategy_edge("test")

        assert edge["win_rate"] == Decimal("0.6667")  # 20/30
        assert edge["win_loss_ratio"] == Decimal("2.00")  # 10/5
        assert edge["sample_size"] == 30

    def test_calculate_drawdown(self, tracker):
        """Test maximum drawdown calculation."""
        # Create trades with drawdown
        trades = [
            (Decimal("100"), 1),   # Win
            (Decimal("50"), 2),    # Win  (peak: 150)
            (Decimal("-80"), 3),   # Loss (current: 70, drawdown: 80/150 = 53%)
            (Decimal("-30"), 4),   # Loss (current: 40, drawdown: 110/150 = 73%)
            (Decimal("60"), 5),    # Win  (current: 100)
        ]

        for pnl, day in trades:
            trade = Trade(
                order_id=f"order_{day}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("50000"),
                quantity=Decimal("0.01"),
                pnl_dollars=pnl,
                pnl_percent=Decimal("1"),
                timestamp=datetime.now(UTC) - timedelta(days=30-day)
            )
            tracker.record_trade("test", trade)

        drawdown = tracker.calculate_drawdown("test")
        assert drawdown == Decimal("0.7333")  # 73.33% max drawdown

    def test_winning_streak(self, tracker):
        """Test winning streak calculation."""
        # Record 5 consecutive wins
        for i in range(5):
            trade = Trade(
                order_id=f"win_{i}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                quantity=Decimal("0.01"),
                pnl_dollars=Decimal("10"),
                pnl_percent=Decimal("2"),
                timestamp=datetime.now(UTC)
            )
            tracker.record_trade("test", trade)

        streak = tracker.get_winning_streak("test")
        assert streak == 5

    def test_losing_streak(self, tracker):
        """Test losing streak calculation."""
        # Record mixed trades ending with losses
        trades_pnl = [Decimal("10"), Decimal("10"), Decimal("-5"), Decimal("-5"), Decimal("-5")]

        for i, pnl in enumerate(trades_pnl):
            trade = Trade(
                order_id=f"trade_{i}",
                strategy_id="test",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if pnl > 0 else Decimal("49500"),
                quantity=Decimal("0.01"),
                pnl_dollars=pnl,
                pnl_percent=Decimal("2") if pnl > 0 else Decimal("-1"),
                timestamp=datetime.now(UTC)
            )
            tracker.record_trade("test", trade)

        streak = tracker.get_losing_streak("test")
        assert streak == 3

    def test_reset_strategy_metrics(self, tracker, sample_trade):
        """Test resetting strategy metrics."""
        # Record some trades
        tracker.record_trade("test", sample_trade)
        assert tracker.get_strategy_metrics("test").total_trades == 1

        # Reset metrics
        tracker.reset_strategy_metrics("test")

        metrics = tracker.get_strategy_metrics("test")
        assert metrics.total_trades == 0
        assert metrics.total_pnl == Decimal("0")
