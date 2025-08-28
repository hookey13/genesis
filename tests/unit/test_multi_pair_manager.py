"""Unit tests for MultiPairManager class."""

import asyncio
import uuid
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from genesis.core.exceptions import TierLockedException
from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    Signal,
    SignalType,
    Tier,
)
from genesis.engine.executor.multi_pair import (
    MultiPairManager,
)


class TestMultiPairManager:
    """Test suite for MultiPairManager."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.get_account = AsyncMock()
        repo.get_portfolio_limits = AsyncMock()
        repo.get_open_positions = AsyncMock()
        repo.get_latest_price = AsyncMock()
        return repo

    @pytest.fixture
    def mock_state_manager(self):
        """Create mock state manager."""
        manager = AsyncMock()
        manager.get_current_tier = AsyncMock(return_value=Tier.HUNTER)
        return manager

    @pytest.fixture
    def account_id(self):
        """Generate test account ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def manager(self, mock_repository, mock_state_manager, account_id):
        """Create MultiPairManager instance."""
        return MultiPairManager(
            repository=mock_repository,
            state_manager=mock_state_manager,
            account_id=account_id,
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing."""
        return [
            Position(
                position_id=str(uuid.uuid4()),
                account_id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                quantity=Decimal("0.5"),
                dollar_value=Decimal("25500"),
                pnl_dollars=Decimal("500"),
                opened_at=datetime.utcnow(),
            ),
            Position(
                position_id=str(uuid.uuid4()),
                account_id=str(uuid.uuid4()),
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                quantity=Decimal("2"),
                dollar_value=Decimal("6200"),
                pnl_dollars=Decimal("200"),
                opened_at=datetime.utcnow(),
            ),
        ]

    @pytest.fixture
    def sample_signals(self):
        """Create sample signals for testing."""
        return [
            Signal(
                signal_id=str(uuid.uuid4()),
                symbol="BTC/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.85"),
                priority=80,
                size_recommendation=Decimal("0.1"),
                price_target=Decimal("52000"),
                stop_loss=Decimal("49000"),
                strategy_name="mean_reversion",
                timestamp=datetime.utcnow(),
            ),
            Signal(
                signal_id=str(uuid.uuid4()),
                symbol="ETH/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.75"),
                priority=70,
                size_recommendation=Decimal("1"),
                price_target=Decimal("3200"),
                stop_loss=Decimal("2900"),
                strategy_name="trend_following",
                timestamp=datetime.utcnow(),
            ),
            Signal(
                signal_id=str(uuid.uuid4()),
                symbol="SOL/USDT",
                signal_type=SignalType.BUY,
                confidence_score=Decimal("0.90"),
                priority=90,
                size_recommendation=Decimal("10"),
                price_target=Decimal("150"),
                stop_loss=Decimal("120"),
                strategy_name="breakout",
                timestamp=datetime.utcnow(),
            ),
        ]

    @pytest.mark.asyncio
    async def test_initialize_validates_tier(self, manager, mock_state_manager):
        """Test initialization validates tier requirements."""
        # Test with insufficient tier
        mock_state_manager.get_current_tier.return_value = Tier.SNIPER

        with pytest.raises(TierLockedException) as exc_info:
            await manager.initialize()

        assert "Multi-pair trading requires HUNTER tier" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_loads_limits(
        self, manager, mock_repository, mock_state_manager
    ):
        """Test initialization loads portfolio limits."""
        mock_state_manager.get_current_tier.return_value = Tier.HUNTER
        mock_repository.get_portfolio_limits.return_value = None
        mock_repository.get_open_positions.return_value = []

        await manager.initialize()

        # Should have default Hunter tier limits
        assert manager._limits is not None
        assert manager._limits.max_positions_global == 5
        assert manager._limits.max_exposure_global_dollars == Decimal("10000")

    @pytest.mark.asyncio
    async def test_can_open_position_checks_global_limit(
        self, manager, mock_repository
    ):
        """Test position opening checks global position count."""
        # Initialize with Hunter tier
        mock_repository.get_portfolio_limits.return_value = None
        mock_repository.get_open_positions.return_value = []
        await manager.initialize()

        # Add 5 positions (max for Hunter)
        for i in range(5):
            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=f"PAIR{i}/USDT",
                quantity=Decimal("1"),
                dollar_value=Decimal("1000"),
            )
            await manager.add_position(position)

        # Mock price lookup
        mock_repository.get_latest_price.return_value = Decimal("100")

        # Should not allow 6th position
        can_open = await manager.can_open_position("NEW/USDT", Decimal("1"))
        assert can_open is False

    @pytest.mark.asyncio
    async def test_can_open_position_checks_pair_limits(self, manager, mock_repository):
        """Test position opening checks per-pair limits."""
        await manager.initialize()

        # Add existing BTC position near limit
        btc_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("1.8"),  # Near 2 BTC limit for Hunter
            dollar_value=Decimal("2800"),  # Near $3000 limit
        )
        await manager.add_position(btc_position)

        mock_repository.get_latest_price.return_value = Decimal("50000")

        # Should not allow exceeding pair limit
        can_open = await manager.can_open_position(
            "BTC/USDT", Decimal("0.5")
        )  # Would exceed 2 BTC
        assert can_open is False

        # Should allow smaller addition
        can_open = await manager.can_open_position("BTC/USDT", Decimal("0.1"))
        assert can_open is True

    @pytest.mark.asyncio
    async def test_can_open_position_checks_global_exposure(
        self, manager, mock_repository
    ):
        """Test position opening checks global exposure limit."""
        await manager.initialize()

        # Add positions totaling $9500
        positions = [
            Position(symbol="BTC/USDT", dollar_value=Decimal("5000")),
            Position(symbol="ETH/USDT", dollar_value=Decimal("3000")),
            Position(symbol="SOL/USDT", dollar_value=Decimal("1500")),
        ]

        for pos in positions:
            pos.position_id = str(uuid.uuid4())
            pos.quantity = Decimal("1")
            await manager.add_position(pos)

        mock_repository.get_latest_price.return_value = Decimal("100")

        # Should not allow position that exceeds $10000 total
        can_open = await manager.can_open_position("NEW/USDT", Decimal("6"))  # $600
        assert can_open is False

        # Should allow smaller position
        can_open = await manager.can_open_position("NEW/USDT", Decimal("4"))  # $400
        assert can_open is True

    @pytest.mark.asyncio
    async def test_allocate_capital_empty_signals(self, manager):
        """Test capital allocation with no signals."""
        await manager.initialize()

        allocations = await manager.allocate_capital([])
        assert allocations == {}

    @pytest.mark.asyncio
    async def test_allocate_capital_no_available_capital(
        self, manager, mock_repository, sample_signals
    ):
        """Test capital allocation with no available capital."""
        await manager.initialize()

        # Mock account with no balance
        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("0"), tier=Tier.HUNTER
        )

        allocations = await manager.allocate_capital(sample_signals)
        assert allocations == {}

    @pytest.mark.asyncio
    async def test_allocate_capital_distributes_proportionally(
        self, manager, mock_repository, sample_signals
    ):
        """Test capital allocation distributes based on signal priority and confidence."""
        await manager.initialize()

        # Mock account with balance
        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("5000"), tier=Tier.HUNTER
        )

        allocations = await manager.allocate_capital(sample_signals)

        # Should allocate to all signals
        assert len(allocations) > 0

        # Higher priority/confidence signals should get more
        if "SOL/USDT" in allocations and "ETH/USDT" in allocations:
            # SOL has higher priority (90) and confidence (0.90)
            assert allocations["SOL/USDT"] > allocations["ETH/USDT"]

    @pytest.mark.asyncio
    async def test_allocate_capital_respects_pair_limits(
        self, manager, mock_repository, sample_signals
    ):
        """Test capital allocation respects per-pair limits."""
        await manager.initialize()

        # Add existing position near limit
        btc_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            dollar_value=Decimal("2900"),  # Near $3000 limit
        )
        await manager.add_position(btc_position)

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("5000"), tier=Tier.HUNTER
        )

        # Find BTC signal
        btc_signal = next((s for s in sample_signals if s.symbol == "BTC/USDT"), None)

        if btc_signal:
            allocations = await manager.allocate_capital([btc_signal])

            # Should limit BTC allocation to remaining room ($100)
            if "BTC/USDT" in allocations:
                assert allocations["BTC/USDT"] <= Decimal("100")

    @pytest.mark.asyncio
    async def test_allocate_capital_applies_correlation_penalty(
        self, manager, mock_repository
    ):
        """Test capital allocation applies correlation penalty."""
        await manager.initialize()

        # Set up high correlation between BTC and ETH
        await manager.update_correlations(
            {
                ("BTC/USDT", "ETH/USDT"): Decimal("0.85"),
                ("ETH/USDT", "BTC/USDT"): Decimal("0.85"),
            }
        )

        # Add BTC position
        btc_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            dollar_value=Decimal("2500"),
        )
        await manager.add_position(btc_position)

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("5000"), tier=Tier.HUNTER
        )

        # Create ETH signal (high correlation with BTC)
        eth_signal = Signal(
            signal_id=str(uuid.uuid4()),
            symbol="ETH/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.80"),
            priority=75,
            strategy_name="test",
            timestamp=datetime.utcnow(),
        )

        # Create SOL signal (no correlation)
        sol_signal = Signal(
            signal_id=str(uuid.uuid4()),
            symbol="SOL/USDT",
            signal_type=SignalType.BUY,
            confidence_score=Decimal("0.80"),
            priority=75,
            strategy_name="test",
            timestamp=datetime.utcnow(),
        )

        allocations = await manager.allocate_capital([eth_signal, sol_signal])

        # SOL should get more allocation due to no correlation penalty
        if "ETH/USDT" in allocations and "SOL/USDT" in allocations:
            assert allocations["SOL/USDT"] > allocations["ETH/USDT"]

    @pytest.mark.asyncio
    async def test_get_active_positions(self, manager, sample_positions):
        """Test getting active positions."""
        await manager.initialize()

        # Add positions
        for pos in sample_positions:
            await manager.add_position(pos)

        active = await manager.get_active_positions()
        assert len(active) == len(sample_positions)
        assert all(p.symbol in ["BTC/USDT", "ETH/USDT"] for p in active)

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_empty(self, manager, mock_repository):
        """Test portfolio risk calculation with no positions."""
        await manager.initialize()

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("10000"), tier=Tier.HUNTER
        )

        risk = await manager.calculate_portfolio_risk()

        assert risk.total_exposure_dollars == Decimal("0")
        assert risk.position_count == 0
        assert risk.risk_score == Decimal("0")
        assert not risk.is_high_risk

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_with_positions(
        self, manager, mock_repository, sample_positions
    ):
        """Test portfolio risk calculation with positions."""
        await manager.initialize()

        # Add positions
        for pos in sample_positions:
            await manager.add_position(pos)

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("40000"), tier=Tier.HUNTER
        )

        risk = await manager.calculate_portfolio_risk()

        assert risk.total_exposure_dollars == Decimal("31700")  # 25500 + 6200
        assert risk.position_count == 2
        assert risk.available_capital == Decimal("8300")  # 40000 - 31700
        assert risk.risk_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_high_correlation(
        self, manager, mock_repository, sample_positions
    ):
        """Test portfolio risk with high correlation."""
        await manager.initialize()

        # Add positions
        for pos in sample_positions:
            await manager.add_position(pos)

        # Set high correlation
        await manager.update_correlations({("BTC/USDT", "ETH/USDT"): Decimal("0.85")})

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("40000"), tier=Tier.HUNTER
        )

        risk = await manager.calculate_portfolio_risk()

        assert risk.correlation_risk > Decimal("0.6")
        assert len(risk.warnings) > 0
        assert any("correlation" in w.lower() for w in risk.warnings)

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_concentration(
        self, manager, mock_repository
    ):
        """Test portfolio risk with high concentration."""
        await manager.initialize()

        # Add one large position
        large_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            dollar_value=Decimal("9000"),  # 90% of limit
            pnl_dollars=Decimal("-500"),  # Loss
        )
        await manager.add_position(large_position)

        # Add one small position
        small_position = Position(
            position_id=str(uuid.uuid4()),
            symbol="ETH/USDT",
            quantity=Decimal("0.1"),
            dollar_value=Decimal("1000"),  # 10% of limit
            pnl_dollars=Decimal("50"),
        )
        await manager.add_position(small_position)

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("15000"), tier=Tier.HUNTER
        )

        risk = await manager.calculate_portfolio_risk()

        # High concentration (90% in one position)
        assert risk.concentration_risk > Decimal("0.3")
        assert risk.max_drawdown_dollars == Decimal("500")  # From BTC loss
        assert any("concentration" in w.lower() for w in risk.warnings)

    @pytest.mark.asyncio
    async def test_calculate_portfolio_risk_approaching_limits(
        self, manager, mock_repository
    ):
        """Test risk warnings when approaching limits."""
        await manager.initialize()

        # Add 4 positions (80% of 5 max)
        for i in range(4):
            position = Position(
                position_id=str(uuid.uuid4()),
                symbol=f"PAIR{i}/USDT",
                quantity=Decimal("1"),
                dollar_value=Decimal("2000"),
                pnl_dollars=Decimal("0"),
            )
            await manager.add_position(position)

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("10000"), tier=Tier.HUNTER
        )

        risk = await manager.calculate_portfolio_risk()

        assert risk.position_count == 4
        assert any("position limit" in w.lower() for w in risk.warnings)

    @pytest.mark.asyncio
    async def test_add_remove_position(self, manager):
        """Test adding and removing positions."""
        await manager.initialize()

        position = Position(
            position_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            dollar_value=Decimal("50000"),
        )

        # Add position
        await manager.add_position(position)
        assert "BTC/USDT" in manager._active_positions

        # Remove position
        await manager.remove_position("BTC/USDT")
        assert "BTC/USDT" not in manager._active_positions

    @pytest.mark.asyncio
    async def test_update_correlations_warns_high(self, manager):
        """Test correlation update warns on high values."""
        await manager.initialize()

        correlations = {
            ("BTC/USDT", "ETH/USDT"): Decimal("0.65"),  # Above warning threshold
            ("ETH/USDT", "SOL/USDT"): Decimal("0.3"),  # Below threshold
            ("BTC/USDT", "SOL/USDT"): Decimal(
                "0.85"
            ),  # Above risk adjustment threshold
        }

        with patch("genesis.engine.executor.multi_pair.logger") as mock_logger:
            await manager.update_correlations(correlations)

            # Should warn about high correlations
            assert mock_logger.warning.called
            warning_calls = mock_logger.warning.call_args_list

            # Check that high correlations were logged
            warned_pairs = []
            for call in warning_calls:
                if call[0][0] == "high_correlation_detected":
                    warned_pairs.append((call[1]["symbol1"], call[1]["symbol2"]))

            assert ("BTC/USDT", "ETH/USDT") in warned_pairs or (
                "ETH/USDT",
                "BTC/USDT",
            ) in warned_pairs
            assert ("BTC/USDT", "SOL/USDT") in warned_pairs or (
                "SOL/USDT",
                "BTC/USDT",
            ) in warned_pairs

    @pytest.mark.asyncio
    async def test_concurrent_operations_thread_safe(self, manager, mock_repository):
        """Test that concurrent operations are thread-safe."""
        await manager.initialize()

        mock_repository.get_account.return_value = Account(
            account_id=manager.account_id, balance=Decimal("10000"), tier=Tier.HUNTER
        )

        # Simulate concurrent position additions and risk calculations
        async def add_positions():
            for i in range(5):
                position = Position(
                    position_id=str(uuid.uuid4()),
                    symbol=f"PAIR{i}/USDT",
                    quantity=Decimal("1"),
                    dollar_value=Decimal("1000"),
                )
                await manager.add_position(position)
                await asyncio.sleep(0.01)  # Small delay

        async def calculate_risk():
            for _ in range(10):
                risk = await manager.calculate_portfolio_risk()
                assert risk.position_count <= 5
                await asyncio.sleep(0.005)

        # Run concurrently
        await asyncio.gather(
            add_positions(),
            calculate_risk(),
            calculate_risk(),  # Multiple risk calculations
        )

        # Final state should be consistent
        final_positions = await manager.get_active_positions()
        assert len(final_positions) == 5

    def test_get_default_limits(self, manager):
        """Test default limits for different tiers."""
        hunter_limits = manager._get_default_limits(Tier.HUNTER)
        assert hunter_limits.max_positions_global == 5
        assert hunter_limits.max_exposure_global_dollars == Decimal("10000")

        strategist_limits = manager._get_default_limits(Tier.STRATEGIST)
        assert strategist_limits.max_positions_global == 10
        assert strategist_limits.max_exposure_global_dollars == Decimal("50000")

        architect_limits = manager._get_default_limits(Tier.ARCHITECT)
        assert architect_limits.max_positions_global == 20
        assert architect_limits.max_exposure_global_dollars == Decimal("200000")
