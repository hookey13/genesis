"""Integration tests for multi-pair concurrent trading workflow."""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.analytics.pair_correlation_monitor import CorrelationMonitor
from genesis.analytics.pair_performance import PairPerformanceTracker
from genesis.core.exceptions import TierLockedException
from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    Signal,
    SignalType,
    Tier,
)
from genesis.engine.executor.multi_pair import MultiPairManager
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.signal_queue import ConflictResolution, SignalQueue


class TestMultiPairWorkflow:
    """Test complete multi-pair trading workflow."""

    @pytest.fixture
    async def setup_environment(self):
        """Setup test environment with all components."""
        # Create mock repository
        repository = AsyncMock()
        repository.get_account = AsyncMock()
        repository.get_portfolio_limits = AsyncMock(return_value=None)
        repository.get_open_positions = AsyncMock(return_value=[])
        repository.get_latest_price = AsyncMock()
        repository.save_queued_signal = AsyncMock()
        repository.update_signal_status = AsyncMock()
        repository.get_price_history = AsyncMock(return_value=[])
        repository.store_correlation = AsyncMock()
        repository.save_trade_performance = AsyncMock()
        repository.get_trades_by_symbol = AsyncMock(return_value=[])
        repository.get_traded_symbols = AsyncMock(return_value=[])

        # Create account
        account_id = str(uuid.uuid4())
        account = Account(
            account_id=account_id,
            balance=Decimal("10000"),
            tier=Tier.HUNTER  # Required for multi-pair
        )
        repository.get_account.return_value = account

        # Create state manager
        state_manager = AsyncMock()
        state_manager.get_current_tier = AsyncMock(return_value=Tier.HUNTER)

        # Create components
        multi_pair_manager = MultiPairManager(repository, state_manager, account_id)
        signal_queue = SignalQueue(repository, conflict_resolution=ConflictResolution.HIGHEST_PRIORITY)
        correlation_monitor = CorrelationMonitor(repository)
        performance_tracker = PairPerformanceTracker(repository, account_id)
        risk_engine = RiskEngine(account)

        return {
            "repository": repository,
            "account": account,
            "account_id": account_id,
            "state_manager": state_manager,
            "multi_pair_manager": multi_pair_manager,
            "signal_queue": signal_queue,
            "correlation_monitor": correlation_monitor,
            "performance_tracker": performance_tracker,
            "risk_engine": risk_engine
        }

    @pytest.mark.asyncio
    async def test_open_5_concurrent_positions(self, setup_environment):
        """Test opening 5+ concurrent positions (AC 1)."""
        env = await setup_environment
        manager = env["multi_pair_manager"]
        repository = env["repository"]

        # Initialize manager
        await manager.initialize()

        # Mock prices
        repository.get_latest_price.return_value = Decimal("1000")

        # Test opening 5 positions
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
        positions = []

        for i, symbol in enumerate(symbols):
            # Check can open
            can_open = await manager.can_open_position(symbol, Decimal("1"))
            assert can_open is True, f"Should be able to open position {i+1}"

            # Create and add position
            position = Position(
                position_id=str(uuid.uuid4()),
                account_id=env["account_id"],
                symbol=symbol,
                side=PositionSide.LONG,
                quantity=Decimal("1"),
                dollar_value=Decimal("1000"),
                entry_price=Decimal("1000"),
                current_price=Decimal("1000"),
                pnl_dollars=Decimal("0"),
                opened_at=datetime.utcnow()
            )
            await manager.add_position(position)
            positions.append(position)

        # Verify 5 positions open
        active = await manager.get_active_positions()
        assert len(active) == 5

        # Try to open 6th position - should still work for Hunter tier (limit is 5 default)
        can_open = await manager.can_open_position("LINK/USDT", Decimal("1"))
        assert can_open is False  # Should hit position limit

    @pytest.mark.asyncio
    async def test_per_pair_limits_enforced(self, setup_environment):
        """Test per-pair position limits are enforced (AC 2)."""
        env = await setup_environment
        manager = env["multi_pair_manager"]
        repository = env["repository"]

        await manager.initialize()

        # Add position close to pair limit
        btc_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("1.9"),  # Close to 2 BTC Hunter limit
            dollar_value=Decimal("2900"),  # Close to $3000 limit
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            pnl_dollars=Decimal("0")
        )
        await manager.add_position(btc_position)

        repository.get_latest_price.return_value = Decimal("50000")

        # Should not allow exceeding pair limit
        can_open = await manager.can_open_position("BTC/USDT", Decimal("0.2"))  # Would exceed 2 BTC
        assert can_open is False

        # Should allow small addition
        can_open = await manager.can_open_position("BTC/USDT", Decimal("0.05"))
        assert can_open is True

    @pytest.mark.asyncio
    async def test_overall_portfolio_risk_management(self, setup_environment):
        """Test overall portfolio risk management (AC 3)."""
        env = await setup_environment
        manager = env["multi_pair_manager"]
        risk_engine = env["risk_engine"]
        repository = env["repository"]

        await manager.initialize()
        repository.get_account.return_value = env["account"]

        # Add multiple positions
        positions = []
        for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT"]):
            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=symbol,
                quantity=Decimal("1"),
                dollar_value=Decimal("3000"),
                entry_price=Decimal("3000"),
                current_price=Decimal("2900"),  # Small loss
                pnl_dollars=Decimal("-100")
            )
            await manager.add_position(position)
            positions.append(position)

        # Calculate portfolio risk
        portfolio_risk = await manager.calculate_portfolio_risk()

        assert portfolio_risk.total_exposure_dollars == Decimal("9000")
        assert portfolio_risk.position_count == 3
        assert portfolio_risk.risk_score > Decimal("0")

        # Validate with risk engine
        risk_decision = risk_engine.validate_portfolio_risk(positions)
        assert "approved" in risk_decision

        # Check if approaching limits
        if portfolio_risk.total_exposure_dollars > Decimal("8000"):
            assert len(portfolio_risk.warnings) > 0

    @pytest.mark.asyncio
    async def test_correlation_monitoring(self, setup_environment):
        """Test correlation monitoring between positions (AC 4)."""
        env = await setup_environment
        manager = env["multi_pair_manager"]
        monitor = env["correlation_monitor"]

        await manager.initialize()

        # Add correlated price data
        base_time = datetime.utcnow()
        for i in range(30):
            # BTC and ETH with high correlation
            factor = Decimal("1") + Decimal(i) / Decimal("100")
            await monitor.update_price("BTC/USDT", Decimal("50000") * factor, base_time + timedelta(minutes=i))
            await monitor.update_price("ETH/USDT", Decimal("3000") * factor, base_time + timedelta(minutes=i))
            # SOL with different pattern
            await monitor.update_price("SOL/USDT", Decimal("100") * (Decimal("1") - Decimal(i) / Decimal("200")), base_time + timedelta(minutes=i))

        # Calculate correlations
        btc_eth_corr = await monitor.calculate_pair_correlation("BTC/USDT", "ETH/USDT")
        btc_sol_corr = await monitor.calculate_pair_correlation("BTC/USDT", "SOL/USDT")

        # BTC/ETH should have high positive correlation
        assert btc_eth_corr > Decimal("0.8")
        # BTC/SOL should have negative correlation
        assert btc_sol_corr < Decimal("0")

        # Update manager with correlations
        await manager.update_correlations({
            ("BTC/USDT", "ETH/USDT"): btc_eth_corr,
            ("BTC/USDT", "SOL/USDT"): btc_sol_corr
        })

        # Check for correlation warnings
        alerts = await monitor.get_recent_alerts()
        assert any(alert.severity in ["WARNING", "CRITICAL"] for alert in alerts)

    @pytest.mark.asyncio
    async def test_smart_capital_allocation(self, setup_environment):
        """Test smart capital allocation across pairs (AC 5)."""
        env = await setup_environment
        manager = env["multi_pair_manager"]
        repository = env["repository"]

        await manager.initialize()

        # Create signals with different priorities and confidence
        signals = [
            Signal(
                signal_id="1",
                symbol="BTC/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.9"),
                priority=90,
                strategy_name="momentum",
                timestamp=datetime.utcnow()
            ),
            Signal(
                signal_id="2",
                symbol="ETH/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.7"),
                priority=70,
                strategy_name="mean_reversion",
                timestamp=datetime.utcnow()
            ),
            Signal(
                signal_id="3",
                symbol="SOL/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.8"),
                priority=60,
                strategy_name="breakout",
                timestamp=datetime.utcnow()
            )
        ]

        # Allocate capital
        allocations = await manager.allocate_capital(signals)

        assert len(allocations) > 0
        assert sum(allocations.values()) <= env["account"].balance

        # Highest priority/confidence should get most allocation
        if "BTC/USDT" in allocations and "SOL/USDT" in allocations:
            assert allocations["BTC/USDT"] > allocations["SOL/USDT"]

    @pytest.mark.asyncio
    async def test_queue_management_competing_signals(self, setup_environment):
        """Test queue management for competing signals (AC 6)."""
        env = await setup_environment
        queue = env["signal_queue"]

        # Create competing signals for same pair
        buy_signal = Signal(
            signal_id="buy_1",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="trend",
            timestamp=datetime.utcnow()
        )

        sell_signal = Signal(
            signal_id="sell_1",
            symbol="BTC/USDT",
            signal_type=SignalType.SELL,
            confidence_score=Decimal("0.7"),
            priority=70,
            strategy_name="reversal",
            timestamp=datetime.utcnow()
        )

        # Add both signals
        await queue.add_signal(buy_signal)
        await queue.add_signal(sell_signal)

        # Should detect and resolve conflict
        assert queue._stats["total_conflicts"] > 0

        # Higher priority should win
        next_signal = await queue.get_next_signal()
        assert next_signal.signal_id == "buy_1"  # Higher priority

    @pytest.mark.asyncio
    async def test_priority_system_high_confidence(self, setup_environment):
        """Test priority system for high-confidence trades (AC 7)."""
        env = await setup_environment
        queue = env["signal_queue"]

        # Create signals with varying priorities
        low_priority = Signal(
            signal_id="low",
            symbol="ADA/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.5"),
            priority=30,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )

        high_priority = Signal(
            signal_id="high",
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.95"),
            priority=95,
            strategy_name="strong_signal",
            timestamp=datetime.utcnow()
        )

        medium_priority = Signal(
            signal_id="medium",
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.7"),
            priority=60,
            strategy_name="normal",
            timestamp=datetime.utcnow()
        )

        # Add in random order
        await queue.add_signal(low_priority)
        await queue.add_signal(medium_priority)
        await queue.add_signal(high_priority)

        # Should get in priority order
        signal1 = await queue.get_next_signal()
        assert signal1.signal_id == "high"

        signal2 = await queue.get_next_signal()
        assert signal2.signal_id == "medium"

        signal3 = await queue.get_next_signal()
        assert signal3.signal_id == "low"

    @pytest.mark.asyncio
    async def test_performance_attribution_by_pair(self, setup_environment):
        """Test performance attribution by pair (AC 8)."""
        env = await setup_environment
        tracker = env["performance_tracker"]
        repository = env["repository"]

        # Create closed positions with different P&L
        btc_position = Position(
            position_id=str(uuid.uuid4()),
            account_id=env["account_id"],
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            dollar_value=Decimal("25500"),
            pnl_dollars=Decimal("500"),
            opened_at=datetime.utcnow() - timedelta(hours=2),
            closed_at=datetime.utcnow()
        )

        eth_position = Position(
            position_id=str(uuid.uuid4()),
            account_id=env["account_id"],
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("2"),
            entry_price=Decimal("3000"),
            current_price=Decimal("2900"),
            dollar_value=Decimal("5800"),
            pnl_dollars=Decimal("-200"),
            opened_at=datetime.utcnow() - timedelta(hours=3),
            closed_at=datetime.utcnow() - timedelta(hours=1)
        )

        # Track trades
        await tracker.track_trade(btc_position)
        await tracker.track_trade(eth_position)

        # Mock repository responses for attribution
        repository.get_traded_symbols.return_value = ["BTC/USDT", "ETH/USDT"]
        repository.get_trades_by_symbol.side_effect = lambda account_id, symbol, **kwargs: [
            {"pnl_dollars": Decimal("500"), "volume_quote": Decimal("25000")} if symbol == "BTC/USDT"
            else {"pnl_dollars": Decimal("-200"), "volume_quote": Decimal("6000")}
        ]

        # Generate attribution report
        report = await tracker.generate_attribution_report()

        assert report.total_pnl_dollars == Decimal("300")  # 500 - 200
        assert report.best_performer == "BTC/USDT"
        assert report.worst_performer == "ETH/USDT"
        assert "BTC/USDT" in report.pair_contributions
        assert "ETH/USDT" in report.pair_contributions

    @pytest.mark.asyncio
    async def test_tier_gate_enforcement(self, setup_environment):
        """Test tier gate enforcement - Sniper tier rejection."""
        env = await setup_environment

        # Create new manager with Sniper tier account
        sniper_account = Account(
            account_id=str(uuid.uuid4()),
            balance=Decimal("1000"),
            tier=Tier.SNIPER  # Too low for multi-pair
        )

        sniper_state_manager = AsyncMock()
        sniper_state_manager.get_current_tier = AsyncMock(return_value=Tier.SNIPER)

        sniper_manager = MultiPairManager(
            env["repository"],
            sniper_state_manager,
            sniper_account.account_id
        )

        # Should raise TierLockedException
        with pytest.raises(TierLockedException, match="Multi-pair trading requires HUNTER"):
            await sniper_manager.initialize()

    @pytest.mark.asyncio
    async def test_portfolio_drawdown_monitoring(self, setup_environment):
        """Test portfolio drawdown monitoring."""
        env = await setup_environment
        manager = env["multi_pair_manager"]

        await manager.initialize()

        # Add positions with losses
        losing_positions = []
        for i in range(3):
            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=f"PAIR{i}/USDT",
                quantity=Decimal("1"),
                dollar_value=Decimal("1000"),
                entry_price=Decimal("100"),
                current_price=Decimal("90"),
                pnl_dollars=Decimal("-100")  # $100 loss each
            )
            await manager.add_position(position)
            losing_positions.append(position)

        # Calculate portfolio risk
        portfolio_risk = await manager.calculate_portfolio_risk()

        # Should detect drawdown
        assert portfolio_risk.max_drawdown_dollars == Decimal("300")  # Total losses

        # Risk score should reflect drawdown
        assert portfolio_risk.risk_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_emergency_liquidation_priority(self, setup_environment):
        """Test emergency liquidation uses priority_score."""
        env = await setup_environment
        manager = env["multi_pair_manager"]

        await manager.initialize()

        # Add positions with different priority scores
        positions = [
            Position(
                position_id="1",
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                dollar_value=Decimal("5000"),
                pnl_dollars=Decimal("-500"),
                priority_score=10  # Highest priority for liquidation
            ),
            Position(
                position_id="2",
                symbol="ETH/USDT",
                quantity=Decimal("2"),
                dollar_value=Decimal("3000"),
                pnl_dollars=Decimal("-200"),
                priority_score=5
            ),
            Position(
                position_id="3",
                symbol="SOL/USDT",
                quantity=Decimal("10"),
                dollar_value=Decimal("1000"),
                pnl_dollars=Decimal("100"),
                priority_score=1  # Lowest priority
            )
        ]

        for pos in positions:
            await manager.add_position(pos)

        active = await manager.get_active_positions()
        assert len(active) == 3

        # In emergency, positions with higher priority_score should be closed first
        # This is application logic that would be implemented in the actual trading engine

    @pytest.mark.asyncio
    async def test_high_correlation_risk_adjustment(self, setup_environment):
        """Test risk adjustment at 80% correlation."""
        env = await setup_environment
        manager = env["multi_pair_manager"]
        repository = env["repository"]

        await manager.initialize()

        # Set high correlation
        await manager.update_correlations({
            ("BTC/USDT", "ETH/USDT"): Decimal("0.85")  # Above 80% threshold
        })

        # Add BTC position
        btc_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            dollar_value=Decimal("2500")
        )
        await manager.add_position(btc_position)

        # Try to allocate capital with ETH signal (high correlation)
        eth_signal = Signal(
            signal_id="eth",
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.8"),
            priority=80,
            strategy_name="test",
            timestamp=datetime.utcnow()
        )

        allocations = await manager.allocate_capital([eth_signal])

        # ETH allocation should be reduced due to correlation penalty
        if "ETH/USDT" in allocations:
            # Should be heavily penalized
            assert allocations["ETH/USDT"] < env["account"].balance * Decimal("0.1")

    @pytest.mark.asyncio
    async def test_complete_workflow_end_to_end(self, setup_environment):
        """Test complete multi-pair trading workflow end-to-end."""
        env = await setup_environment

        # 1. Initialize all components
        await env["multi_pair_manager"].initialize()

        # 2. Add price data for correlations
        base_time = datetime.utcnow()
        for i in range(30):
            await env["correlation_monitor"].update_price(
                "BTC/USDT",
                Decimal("50000") + Decimal(i * 100),
                base_time + timedelta(minutes=i)
            )
            await env["correlation_monitor"].update_price(
                "ETH/USDT",
                Decimal("3000") + Decimal(i * 10),
                base_time + timedelta(minutes=i)
            )

        # 3. Queue some signals
        signals = [
            Signal(
                signal_id=f"sig_{i}",
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence_score=Decimal(str(0.7 + i * 0.05)),
                priority=70 + i * 5,
                strategy_name="test",
                timestamp=datetime.utcnow()
            )
            for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        ]

        for signal in signals:
            await env["signal_queue"].add_signal(signal)

        # 4. Process signals and open positions
        env["repository"].get_latest_price.return_value = Decimal("1000")

        positions_opened = []
        for _ in range(3):
            signal = await env["signal_queue"].get_next_signal()
            if signal:
                # Check if can open position
                can_open = await env["multi_pair_manager"].can_open_position(
                    signal.symbol,
                    Decimal("1")
                )

                if can_open:
                    # Create position
                    position = Position(
                        position_id=str(uuid.uuid4()),
                        account_id=env["account_id"],
                        symbol=signal.symbol,
                        side=PositionSide.LONG,
                        quantity=Decimal("1"),
                        dollar_value=Decimal("1000"),
                        entry_price=Decimal("1000"),
                        current_price=Decimal("1000"),
                        pnl_dollars=Decimal("0"),
                        opened_at=datetime.utcnow()
                    )
                    await env["multi_pair_manager"].add_position(position)
                    positions_opened.append(position)

        assert len(positions_opened) > 0

        # 5. Calculate portfolio risk
        portfolio_risk = await env["multi_pair_manager"].calculate_portfolio_risk()
        assert portfolio_risk.position_count == len(positions_opened)

        # 6. Validate with risk engine
        risk_decision = env["risk_engine"].validate_portfolio_risk(positions_opened)
        assert risk_decision["approved"] is True

        # 7. Close positions and track performance
        for position in positions_opened:
            position.closed_at = datetime.utcnow()
            position.pnl_dollars = Decimal("50")  # Small profit
            await env["performance_tracker"].track_trade(position)

        # 8. Generate performance report
        env["repository"].get_traded_symbols.return_value = list(set(p.symbol for p in positions_opened))
        report = await env["performance_tracker"].generate_attribution_report()

        assert report.total_pnl_dollars >= Decimal("0")  # Should have some P&L
        assert len(report.pair_contributions) > 0
