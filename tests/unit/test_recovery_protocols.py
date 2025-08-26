"""Unit tests for recovery protocols with position size management."""
import pytest
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from genesis.tilt.recovery_protocols import (
    RecoveryProtocolManager,
    RecoveryProtocol,
    RecoveryStage,
)


@pytest.fixture
def recovery_manager():
    """Create a recovery protocol manager for testing."""
    manager = RecoveryProtocolManager()
    return manager


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = AsyncMock()
    repo.save_recovery_protocol = AsyncMock()
    repo.update_recovery_protocol = AsyncMock()
    repo.get_active_recovery_protocols = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


class TestPositionSizeCalculation:
    """Test position size calculations during recovery."""

    def test_calculate_recovery_position_size_stage0(self, recovery_manager):
        """Test position size at Stage 0 (25%)."""
        base_size = Decimal("1000")
        adjusted = recovery_manager.calculate_recovery_position_size(base_size, 0)
        assert adjusted == Decimal("250")  # 25% of 1000

    def test_calculate_recovery_position_size_stage1(self, recovery_manager):
        """Test position size at Stage 1 (50%)."""
        base_size = Decimal("1000")
        adjusted = recovery_manager.calculate_recovery_position_size(base_size, 1)
        assert adjusted == Decimal("500")  # 50% of 1000

    def test_calculate_recovery_position_size_stage2(self, recovery_manager):
        """Test position size at Stage 2 (75%)."""
        base_size = Decimal("1000")
        adjusted = recovery_manager.calculate_recovery_position_size(base_size, 2)
        assert adjusted == Decimal("750")  # 75% of 1000

    def test_calculate_recovery_position_size_stage3(self, recovery_manager):
        """Test position size at Stage 3 (100% - fully recovered)."""
        base_size = Decimal("1000")
        adjusted = recovery_manager.calculate_recovery_position_size(base_size, 3)
        assert adjusted == Decimal("1000")  # 100% of 1000

    def test_calculate_recovery_position_size_invalid_stage(self, recovery_manager):
        """Test position size with invalid stage defaults to Stage 0."""
        base_size = Decimal("1000")
        adjusted = recovery_manager.calculate_recovery_position_size(base_size, 99)
        assert adjusted == Decimal("250")  # Defaults to 25%


class TestProtocolInitiation:
    """Test recovery protocol initiation."""

    @pytest.mark.asyncio
    async def test_initiate_recovery_protocol(self, mock_repository, mock_event_bus):
        """Test initiating a new recovery protocol."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"
        debt = Decimal("500")

        protocol = await manager.initiate_recovery_protocol(
            profile_id=profile_id,
            lockout_duration_minutes=30,
            initial_debt_amount=debt,
        )

        assert protocol.profile_id == profile_id
        assert protocol.initial_debt_amount == debt
        assert protocol.current_debt_amount == debt
        assert protocol.recovery_stage == RecoveryStage.STAGE_0
        assert protocol.is_active is True

        # Check persistence and event
        mock_repository.save_recovery_protocol.assert_called_once()
        mock_event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_initiate_protocol_already_active(self, mock_repository, mock_event_bus):
        """Test initiating protocol when one already exists."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        # First protocol
        protocol1 = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("500")
        )

        # Attempt second protocol - should return existing
        protocol2 = await manager.initiate_recovery_protocol(
            profile_id, 60, Decimal("1000")
        )

        assert protocol2 == protocol1
        assert mock_repository.save_recovery_protocol.call_count == 1  # Only once


class TestTradeRecording:
    """Test recording trade results during recovery."""

    @pytest.mark.asyncio
    async def test_record_profitable_trade(self, mock_repository, mock_event_bus):
        """Test recording a profitable trade."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        # Initiate protocol with debt
        protocol = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Record profitable trade
        profit = Decimal("200")
        await manager.record_trade_result(profile_id, profit, is_profitable=True)

        # Check updates
        assert protocol.profitable_trades_count == 1
        assert protocol.total_profit == profit
        assert protocol.current_debt_amount == Decimal("800")  # 1000 - 200

        # Check debt reduction event
        assert mock_event_bus.publish.call_count >= 2  # init + debt reduction

    @pytest.mark.asyncio
    async def test_record_loss_trade(self, mock_repository, mock_event_bus):
        """Test recording a losing trade."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        protocol = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Record loss trade
        loss = Decimal("100")
        await manager.record_trade_result(profile_id, loss, is_profitable=False)

        # Check updates
        assert protocol.loss_trades_count == 1
        assert protocol.total_loss == loss
        assert protocol.current_debt_amount == Decimal("1000")  # No reduction

    @pytest.mark.asyncio
    async def test_debt_reduction_capped_at_debt_amount(self, mock_repository, mock_event_bus):
        """Test that debt reduction is capped at current debt amount."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        protocol = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("100")
        )

        # Record profit larger than debt
        await manager.record_trade_result(profile_id, Decimal("500"), is_profitable=True)

        # Debt should be zero, not negative
        assert protocol.current_debt_amount == Decimal("0")


class TestStageAdvancement:
    """Test recovery stage advancement."""

    @pytest.mark.asyncio
    async def test_advance_stage_requirements_not_met(self, recovery_manager):
        """Test that stage doesn't advance without meeting requirements."""
        profile_id = "test_profile"

        protocol = await recovery_manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Try to advance without meeting requirements
        advanced = await recovery_manager.advance_recovery_stage(profile_id)
        assert advanced is False
        assert protocol.recovery_stage == RecoveryStage.STAGE_0

    @pytest.mark.asyncio
    async def test_advance_stage_requirements_met(self, mock_repository, mock_event_bus):
        """Test stage advancement when requirements are met."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        protocol = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Simulate meeting Stage 0 requirements
        # Need: 3 profitable trades, profit > 1.5x losses, 25% debt paid
        protocol.profitable_trades_count = 3
        protocol.total_profit = Decimal("300")
        protocol.total_loss = Decimal("100")  # Profit ratio = 3.0
        protocol.current_debt_amount = Decimal("700")  # 30% paid

        advanced = await manager.advance_recovery_stage(profile_id)
        assert advanced is True
        assert protocol.recovery_stage == RecoveryStage.STAGE_1

    @pytest.mark.asyncio
    async def test_advance_to_full_recovery(self, mock_repository, mock_event_bus):
        """Test advancing to full recovery (Stage 3)."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        protocol = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Set at Stage 2, ready for Stage 3
        protocol.recovery_stage = RecoveryStage.STAGE_2
        protocol.profitable_trades_count = 10
        protocol.total_profit = Decimal("1000")
        protocol.total_loss = Decimal("200")
        protocol.current_debt_amount = Decimal("100")

        advanced = await manager.advance_recovery_stage(profile_id)
        assert advanced is True
        assert protocol.recovery_stage == RecoveryStage.STAGE_3
        assert protocol.recovery_completed_at is not None
        assert protocol.is_active is False

    def test_check_advancement_requirements_profit_ratio(self, recovery_manager):
        """Test advancement requirements checking for profit ratio."""
        protocol = RecoveryProtocol(
            profile_id="test",
            recovery_stage=RecoveryStage.STAGE_0,
            initial_debt_amount=Decimal("1000"),
            current_debt_amount=Decimal("700"),
            profitable_trades_count=5,
            total_profit=Decimal("100"),
            total_loss=Decimal("200"),  # Profit ratio = 0.5 (needs 1.5)
        )

        # Should fail due to profit ratio
        meets = recovery_manager._check_advancement_requirements(protocol)
        assert meets is False

    def test_check_advancement_requirements_debt_payment(self, recovery_manager):
        """Test advancement requirements checking for debt payment."""
        protocol = RecoveryProtocol(
            profile_id="test",
            recovery_stage=RecoveryStage.STAGE_0,
            initial_debt_amount=Decimal("1000"),
            current_debt_amount=Decimal("900"),  # Only 10% paid (needs 25%)
            profitable_trades_count=5,
            total_profit=Decimal("300"),
            total_loss=Decimal("100"),
        )

        # Should fail due to insufficient debt payment
        meets = recovery_manager._check_advancement_requirements(protocol)
        assert meets is False


class TestPositionSizeMultiplier:
    """Test getting position size multiplier for profiles."""

    @pytest.mark.asyncio
    async def test_get_multiplier_no_protocol(self, recovery_manager):
        """Test multiplier when no recovery protocol exists."""
        multiplier = recovery_manager.get_position_size_multiplier("unknown_profile")
        assert multiplier == Decimal("1.0")  # Full size

    @pytest.mark.asyncio
    async def test_get_multiplier_with_protocol(self, recovery_manager):
        """Test multiplier with active recovery protocol."""
        profile_id = "test_profile"

        await recovery_manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("500")
        )

        # Stage 0 = 25% multiplier
        multiplier = recovery_manager.get_position_size_multiplier(profile_id)
        assert multiplier == Decimal("0.25")


class TestRecoveryStatistics:
    """Test recovery statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_statistics_no_protocol(self, recovery_manager):
        """Test statistics when no protocol exists."""
        stats = recovery_manager.get_recovery_statistics("unknown_profile")

        assert stats["has_active_protocol"] is False
        assert stats["recovery_stage"] is None
        assert stats["position_size_multiplier"] == "1.0"

    @pytest.mark.asyncio
    async def test_get_statistics_with_protocol(self, recovery_manager):
        """Test statistics with active protocol."""
        profile_id = "test_profile"

        protocol = await recovery_manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Add some trades
        protocol.profitable_trades_count = 2
        protocol.loss_trades_count = 1
        protocol.total_profit = Decimal("150")
        protocol.total_loss = Decimal("50")
        protocol.current_debt_amount = Decimal("850")

        stats = recovery_manager.get_recovery_statistics(profile_id)

        assert stats["has_active_protocol"] is True
        assert stats["protocol_id"] == protocol.protocol_id
        assert stats["recovery_stage"] == "STAGE_0"
        assert stats["position_size_multiplier"] == "0.25"
        assert stats["initial_debt"] == "1000"
        assert stats["current_debt"] == "850"
        assert stats["debt_paid"] == "150"
        assert stats["profitable_trades"] == 2
        assert stats["loss_trades"] == 1
        assert stats["total_profit"] == "150"
        assert stats["total_loss"] == "50"


class TestForceCompletion:
    """Test force completing recovery protocols."""

    @pytest.mark.asyncio
    async def test_force_complete_recovery(self, mock_repository, mock_event_bus):
        """Test force completing an active recovery protocol."""
        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        protocol = await manager.initiate_recovery_protocol(
            profile_id, 30, Decimal("1000")
        )

        # Force complete
        completed = await manager.force_complete_recovery(profile_id)
        assert completed is True
        assert protocol.is_active is False
        assert protocol.recovery_completed_at is not None
        assert profile_id not in manager.active_protocols

    @pytest.mark.asyncio
    async def test_force_complete_no_protocol(self, recovery_manager):
        """Test force completing when no protocol exists."""
        completed = await recovery_manager.force_complete_recovery("unknown")
        assert completed is False


class TestProtocolPersistence:
    """Test protocol persistence and loading."""

    @pytest.mark.asyncio
    async def test_load_active_protocols(self, mock_repository):
        """Test loading active protocols from database."""
        # Mock database response
        now = datetime.now(UTC)
        mock_repository.get_active_recovery_protocols.return_value = [
            {
                "protocol_id": "proto_1",
                "profile_id": "profile_1",
                "initiated_at": now.isoformat(),
                "lockout_duration_minutes": 30,
                "initial_debt_amount": "1000",
                "current_debt_amount": "800",
                "recovery_stage": 1,
                "profitable_trades_count": 2,
                "loss_trades_count": 1,
                "total_profit": "300",
                "total_loss": "100",
                "recovery_completed_at": None,
                "is_active": True,
            }
        ]

        manager = RecoveryProtocolManager(repository=mock_repository)
        await manager.load_active_protocols()

        assert "profile_1" in manager.active_protocols
        protocol = manager.active_protocols["profile_1"]
        assert protocol.protocol_id == "proto_1"
        assert protocol.recovery_stage == RecoveryStage.STAGE_1
        assert protocol.current_debt_amount == Decimal("800")

    @pytest.mark.asyncio
    async def test_persist_error_handling(self, mock_repository, mock_event_bus):
        """Test error handling when persisting protocol."""
        mock_repository.save_recovery_protocol.side_effect = Exception("Database error")

        manager = RecoveryProtocolManager(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )

        # Should not raise exception
        protocol = await manager.initiate_recovery_protocol(
            "test_profile", 30, Decimal("500")
        )

        assert protocol is not None
        mock_repository.save_recovery_protocol.assert_called_once()