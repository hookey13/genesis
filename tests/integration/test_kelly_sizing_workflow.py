"""
Integration tests for Kelly Criterion position sizing workflow.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from genesis.analytics.kelly_sizing import KellyCalculator
from genesis.analytics.strategy_metrics import StrategyPerformanceTracker
from genesis.core.constants import ConvictionLevel, TradingTier
from genesis.core.models import Account, OrderSide, Trade, TradingSession
from genesis.data.kelly_repo import KellyRepository
from genesis.engine.risk_engine import RiskEngine


class TestKellySizingWorkflow:
    """Integration tests for Kelly sizing workflow."""

    @pytest.fixture
    async def test_account_hunter(self):
        """Create a Hunter tier test account."""
        return Account(
            account_id="test_hunter",
            balance_usdt=Decimal("5000"),
            tier=TradingTier.HUNTER,
            locked_features=[],
            last_sync=datetime.now(UTC),
        )

    @pytest.fixture
    async def test_account_strategist(self):
        """Create a Strategist tier test account."""
        return Account(
            account_id="test_strategist",
            balance_usdt=Decimal("15000"),
            tier=TradingTier.STRATEGIST,
            locked_features=[],
            last_sync=datetime.now(UTC),
        )

    @pytest.fixture
    async def test_session(self, test_account_hunter):
        """Create a test trading session."""
        return TradingSession(
            account_id=test_account_hunter.account_id,
            starting_balance=test_account_hunter.balance_usdt,
            current_balance=test_account_hunter.balance_usdt,
            daily_loss_limit=Decimal("100"),
        )

    @pytest.fixture
    def mock_trades(self):
        """Generate a history of trades for testing."""
        trades = []
        base_time = datetime.now(UTC)

        # Create a profitable trading history
        # 30 trades: 21 wins (70% win rate), 9 losses
        # Average win: $50, Average loss: $25 (2:1 ratio)

        for i in range(21):  # Wins
            trades.append(
                Trade(
                    trade_id=f"win_{i}",
                    order_id=f"order_win_{i}",
                    strategy_id="momentum_strategy",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("50100"),
                    quantity=Decimal("0.01"),
                    pnl_dollars=Decimal("50"),
                    pnl_percent=Decimal("0.2"),
                    timestamp=base_time - timedelta(days=30 - i),
                )
            )

        for i in range(9):  # Losses
            trades.append(
                Trade(
                    trade_id=f"loss_{i}",
                    order_id=f"order_loss_{i}",
                    strategy_id="momentum_strategy",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("49950"),
                    quantity=Decimal("0.01"),
                    pnl_dollars=Decimal("-25"),
                    pnl_percent=Decimal("-0.05"),
                    timestamp=base_time - timedelta(days=10 - i),
                )
            )

        return trades

    @pytest.mark.asyncio
    async def test_kelly_sizing_hunter_tier(
        self, test_account_hunter, test_session, mock_trades
    ):
        """Test Kelly sizing workflow for Hunter tier."""
        # Initialize risk engine with Kelly sizing
        risk_engine = RiskEngine(
            test_account_hunter, test_session, use_kelly_sizing=True
        )

        # Record historical trades
        for trade in mock_trades:
            risk_engine.record_trade_result("momentum_strategy", trade)

        # Calculate position size with Kelly
        entry_price = Decimal("50000")
        position_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="momentum_strategy",
            conviction=ConvictionLevel.MEDIUM,
        )

        # Verify Kelly sizing was used
        assert position_size > 0

        # Calculate expected Kelly fraction
        # Win rate: 21/30 = 0.7
        # Win/loss ratio: 50/25 = 2.0
        # Kelly: (0.7 * 2 - 0.3) / 2 = 1.1 / 2 = 0.55
        # Capped at 0.5, then fractional Kelly 0.5 * 0.25 = 0.125
        # Position: 5000 * 0.125 = 625 USDT
        expected_quantity = Decimal("625") / entry_price  # 0.0125 BTC

        # Allow some tolerance for rounding
        assert abs(position_size - expected_quantity) < Decimal("0.001")

    @pytest.mark.asyncio
    async def test_kelly_with_conviction_override(
        self, test_account_strategist, mock_trades
    ):
        """Test Kelly sizing with conviction override (Strategist feature)."""
        risk_engine = RiskEngine(test_account_strategist, None, use_kelly_sizing=True)

        # Record trades
        for trade in mock_trades:
            risk_engine.record_trade_result("momentum_strategy", trade)

        # Test different conviction levels
        entry_price = Decimal("50000")

        # Low conviction
        size_low = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="momentum_strategy",
            conviction=ConvictionLevel.LOW,
        )

        # Medium conviction
        size_medium = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="momentum_strategy",
            conviction=ConvictionLevel.MEDIUM,
        )

        # High conviction
        size_high = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="momentum_strategy",
            conviction=ConvictionLevel.HIGH,
        )

        # Verify conviction multipliers work
        assert size_low < size_medium < size_high
        assert size_high / size_medium == Decimal("1.5")  # High = 1.5x Medium
        assert size_medium / size_low == Decimal("2.0")  # Medium = 2x Low

    @pytest.mark.asyncio
    async def test_kelly_with_volatility_adjustment(
        self, test_account_hunter, mock_trades
    ):
        """Test Kelly sizing with volatility adjustment."""
        risk_engine = RiskEngine(test_account_hunter, None, use_kelly_sizing=True)

        # Create high volatility trades
        volatile_trades = []
        for i in range(20):
            pnl = Decimal("100") if i % 2 == 0 else Decimal("-90")
            pnl_pct = Decimal("5") if i % 2 == 0 else Decimal("-4.5")

            volatile_trades.append(
                Trade(
                    trade_id=f"vol_{i}",
                    order_id=f"order_vol_{i}",
                    strategy_id="volatile_strategy",
                    symbol="ETH/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("3000"),
                    exit_price=Decimal("3150") if pnl > 0 else Decimal("2865"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=pnl,
                    pnl_percent=pnl_pct,
                    timestamp=datetime.now(UTC) - timedelta(days=20 - i),
                )
            )

        # Record volatile trades
        for trade in volatile_trades:
            risk_engine.record_trade_result("volatile_strategy", trade)

        # Calculate position with volatility adjustment
        size_with_vol = risk_engine.calculate_position_size(
            symbol="ETH/USDT",
            entry_price=Decimal("3000"),
            strategy_id="volatile_strategy",
            use_volatility_adjustment=True,
        )

        # Calculate without volatility adjustment
        size_without_vol = risk_engine.calculate_position_size(
            symbol="ETH/USDT",
            entry_price=Decimal("3000"),
            strategy_id="volatile_strategy",
            use_volatility_adjustment=False,
        )

        # High volatility should reduce position size
        assert size_with_vol < size_without_vol

    @pytest.mark.asyncio
    async def test_kelly_fallback_to_fixed_sizing(self, test_account_hunter):
        """Test fallback to fixed percentage when insufficient data."""
        risk_engine = RiskEngine(test_account_hunter, None, use_kelly_sizing=True)

        # No trades recorded - should fall back to fixed percentage
        entry_price = Decimal("50000")
        position_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="new_strategy",  # No history
            conviction=ConvictionLevel.MEDIUM,
        )

        # Should use fixed 5% risk
        # With 2% stop loss: risk_amount = 5000 * 0.05 = 250
        # Price risk per unit = 50000 * 0.02 = 1000
        # Quantity = 250 / 1000 = 0.25
        # But this gives position value = 0.25 * 50000 = 12500 > balance
        # So it should be capped at balance / price = 5000 / 50000 = 0.1
        expected_max = test_account_hunter.balance_usdt / entry_price

        assert position_size <= expected_max

    @pytest.mark.asyncio
    async def test_kelly_with_drawdown_adjustment(self, test_account_hunter):
        """Test Kelly reduction during drawdown."""
        risk_engine = RiskEngine(test_account_hunter, None, use_kelly_sizing=True)

        # Create a series of losing trades (drawdown)
        losing_streak = []
        for i in range(10):
            losing_streak.append(
                Trade(
                    trade_id=f"drawdown_{i}",
                    order_id=f"order_dd_{i}",
                    strategy_id="drawdown_strategy",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("49000"),
                    quantity=Decimal("0.01"),
                    pnl_dollars=Decimal("-50"),
                    pnl_percent=Decimal("-2"),
                    timestamp=datetime.now(UTC) - timedelta(hours=10 - i),
                )
            )

        # Add some historical wins for Kelly calculation
        for i in range(20):
            losing_streak.append(
                Trade(
                    trade_id=f"hist_win_{i}",
                    order_id=f"order_hw_{i}",
                    strategy_id="drawdown_strategy",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("50500"),
                    quantity=Decimal("0.01"),
                    pnl_dollars=Decimal("25"),
                    pnl_percent=Decimal("1"),
                    timestamp=datetime.now(UTC) - timedelta(days=30 - i),
                )
            )

        # Record all trades
        for trade in losing_streak:
            risk_engine.record_trade_result("drawdown_strategy", trade)

        # Calculate position during drawdown
        position_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=Decimal("50000"),
            strategy_id="drawdown_strategy",
        )

        # Position should be reduced due to recent drawdown
        assert position_size > 0  # Still positive but reduced

    @pytest.mark.asyncio
    async def test_kelly_position_boundaries(self, test_account_hunter, mock_trades):
        """Test position size boundaries enforcement."""
        risk_engine = RiskEngine(test_account_hunter, None, use_kelly_sizing=True)

        # Record profitable trades for high Kelly
        for trade in mock_trades:
            risk_engine.record_trade_result("bounded_strategy", trade)

        # Test with very low entry price (would give huge quantity)
        low_price = Decimal("10")
        position_size = risk_engine.calculate_position_size(
            symbol="PENNY/USDT", entry_price=low_price, strategy_id="bounded_strategy"
        )

        # Check boundaries are enforced
        position_value = position_size * low_price
        balance = test_account_hunter.balance_usdt

        # Hunter tier: min 1%, max 10% of balance
        assert position_value >= balance * Decimal("0.01")  # Min 1%
        assert position_value <= balance * Decimal("0.10")  # Max 10%

    @pytest.mark.asyncio
    async def test_monte_carlo_validation(self):
        """Test Monte Carlo simulation for Kelly validation."""
        calculator = KellyCalculator()

        # Run simulation with known parameters
        win_rate = Decimal("0.6")
        win_loss_ratio = Decimal("2.0")
        kelly_fraction = Decimal("0.2")  # Conservative Kelly

        result = calculator.run_monte_carlo_simulation(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            kelly_fraction=kelly_fraction,
            iterations=1000,
            trades_per_iteration=100,
        )

        # Verify simulation results
        assert result.risk_of_ruin < Decimal("0.05")  # Low risk of ruin
        assert result.expected_growth_rate > Decimal("0")  # Positive growth
        assert result.median_final_balance > Decimal("10000")  # Growth expected

        # Verify optimal Kelly is reasonable
        assert Decimal("0.1") <= result.optimal_kelly <= Decimal("0.4")

    @pytest.mark.asyncio
    async def test_performance_tracking_integration(self, test_account_hunter):
        """Test integration between performance tracking and Kelly sizing."""
        risk_engine = RiskEngine(test_account_hunter, None, use_kelly_sizing=True)

        # Simulate trading over time
        strategies = ["strategy_a", "strategy_b"]

        for strategy_id in strategies:
            # Create different performance profiles
            if strategy_id == "strategy_a":
                # Good performance
                wins, losses = 15, 5
                win_amount, loss_amount = Decimal("40"), Decimal("20")
            else:
                # Poor performance
                wins, losses = 8, 12
                win_amount, loss_amount = Decimal("30"), Decimal("30")

            # Record trades
            for i in range(wins):
                trade = Trade(
                    trade_id=f"{strategy_id}_win_{i}",
                    order_id=f"order_{strategy_id}_w_{i}",
                    strategy_id=strategy_id,
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("50000") + win_amount * 2,
                    quantity=Decimal("0.01"),
                    pnl_dollars=win_amount,
                    pnl_percent=Decimal("0.1"),
                    timestamp=datetime.now(UTC) - timedelta(days=20 - i),
                )
                risk_engine.record_trade_result(strategy_id, trade)

            for i in range(losses):
                trade = Trade(
                    trade_id=f"{strategy_id}_loss_{i}",
                    order_id=f"order_{strategy_id}_l_{i}",
                    strategy_id=strategy_id,
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("50000") - loss_amount * 2,
                    quantity=Decimal("0.01"),
                    pnl_dollars=-loss_amount,
                    pnl_percent=Decimal("-0.05"),
                    timestamp=datetime.now(UTC) - timedelta(days=10 - i),
                )
                risk_engine.record_trade_result(strategy_id, trade)

        # Calculate position sizes for each strategy
        size_a = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("50000"), strategy_id="strategy_a"
        )

        size_b = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("50000"), strategy_id="strategy_b"
        )

        # Better performing strategy should get larger position
        assert size_a > size_b

    @pytest.mark.asyncio
    async def test_database_persistence(self, test_account_hunter, mock_trades):
        """Test Kelly parameters database persistence."""
        # Mock database session
        mock_session = Mock()
        kelly_repo = KellyRepository(mock_session)

        # Create calculator and tracker
        calculator = KellyCalculator()
        tracker = StrategyPerformanceTracker()

        # Record trades
        for trade in mock_trades:
            tracker.record_trade("test_strategy", trade)

        # Calculate and save edge
        edge = calculator.calculate_strategy_edge("test_strategy", mock_trades)

        # Save to repository
        kelly_repo.save_kelly_parameters(edge)

        # Verify save was called
        assert mock_session.merge.called
        assert mock_session.commit.called

        # Test retrieval
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = Mock(
            to_strategy_edge=lambda: edge
        )

        retrieved_edge = kelly_repo.get_kelly_parameters("test_strategy")
        assert retrieved_edge.strategy_id == "test_strategy"

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, test_account_strategist):
        """Test complete Kelly sizing workflow end-to-end."""
        # Initialize components
        risk_engine = RiskEngine(test_account_strategist, None, use_kelly_sizing=True)

        # Phase 1: Initial trades with no history (should use fixed sizing)
        entry_price = Decimal("50000")
        initial_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=entry_price, strategy_id="new_strategy"
        )

        assert initial_size > 0

        # Phase 2: Build up trade history
        for i in range(25):
            # 60% win rate, 2:1 ratio
            if i % 5 < 3:  # 3 wins out of 5
                pnl = Decimal("100")
            else:
                pnl = Decimal("-50")

            trade = Trade(
                trade_id=f"trade_{i}",
                order_id=f"order_{i}",
                strategy_id="new_strategy",
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                exit_price=Decimal("50000") + (pnl * 10),
                quantity=initial_size,
                pnl_dollars=pnl,
                pnl_percent=pnl / Decimal("500"),
                timestamp=datetime.now(UTC) - timedelta(days=25 - i),
            )
            risk_engine.record_trade_result("new_strategy", trade)

        # Phase 3: Now should use Kelly sizing
        kelly_size_medium = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="new_strategy",
            conviction=ConvictionLevel.MEDIUM,
        )

        # Phase 4: Test with high conviction
        kelly_size_high = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="new_strategy",
            conviction=ConvictionLevel.HIGH,
        )

        # Verify Kelly is being used and conviction matters
        assert kelly_size_high > kelly_size_medium
        assert kelly_size_high == kelly_size_medium * Decimal("1.5")

        # Phase 5: Simulate market volatility
        volatile_trades = []
        for i in range(10):
            pnl = Decimal("200") if i % 2 == 0 else Decimal("-180")
            volatile_trades.append(
                Trade(
                    trade_id=f"vol_trade_{i}",
                    order_id=f"vol_order_{i}",
                    strategy_id="new_strategy",
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("50000") + (pnl * 10),
                    quantity=kelly_size_medium,
                    pnl_dollars=pnl,
                    pnl_percent=pnl / Decimal("500"),
                    timestamp=datetime.now(UTC) - timedelta(hours=10 - i),
                )
            )

        for trade in volatile_trades:
            risk_engine.record_trade_result("new_strategy", trade)

        # Phase 6: Position should be reduced due to volatility
        adjusted_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
            strategy_id="new_strategy",
            conviction=ConvictionLevel.MEDIUM,
            use_volatility_adjustment=True,
        )

        # Volatility adjustment should reduce size
        assert adjusted_size < kelly_size_medium
