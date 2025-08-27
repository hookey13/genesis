"""Unit tests for tilt intervention system."""
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.core.events import EventType
from genesis.engine.event_bus import EventBus
from genesis.tilt.detector import TiltLevel
from genesis.tilt.interventions import (
    Intervention,
    InterventionManager,
    InterventionType,
)


class TestInterventionType:
    """Test InterventionType enum."""

    def test_intervention_types(self):
        """Test intervention type values."""
        assert InterventionType.MESSAGE.value == "message"
        assert InterventionType.POSITION_REDUCTION.value == "position_reduction"
        assert InterventionType.TRADING_LOCKOUT.value == "trading_lockout"
        assert InterventionType.COOLDOWN.value == "cooldown"


class TestInterventionManager:
    """Test InterventionManager class."""

    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        return AsyncMock(spec=EventBus)

    @pytest.fixture
    def manager(self, event_bus):
        """Create intervention manager instance."""
        return InterventionManager(
            event_bus=event_bus,
            cooldown_minutes={
                TiltLevel.LEVEL1: 5,
                TiltLevel.LEVEL2: 10,
                TiltLevel.LEVEL3: 30
            }
        )

    def test_get_intervention_message(self, manager):
        """Test getting appropriate intervention messages."""
        # Test Level 1 message
        msg = manager.get_intervention_message(TiltLevel.LEVEL1)
        assert msg in manager.LEVEL1_MESSAGES

        # Test Level 2 message
        msg = manager.get_intervention_message(TiltLevel.LEVEL2)
        assert msg in manager.LEVEL2_MESSAGES

        # Test Level 3 message
        msg = manager.get_intervention_message(TiltLevel.LEVEL3)
        assert msg in manager.LEVEL3_MESSAGES

        # Test normal level (no message)
        msg = manager.get_intervention_message(TiltLevel.NORMAL)
        assert msg == ""

    @pytest.mark.asyncio
    async def test_apply_intervention_level1(self, manager, event_bus):
        """Test applying Level 1 intervention (message only)."""
        profile_id = "test_profile"
        level = TiltLevel.LEVEL1
        tilt_score = 30

        # Apply intervention
        intervention = await manager.apply_intervention(profile_id, level, tilt_score)

        # Assert intervention properties
        assert intervention.profile_id == profile_id
        assert intervention.tilt_level == level
        assert intervention.intervention_type == InterventionType.MESSAGE
        assert intervention.message in manager.LEVEL1_MESSAGES
        assert intervention.position_size_multiplier is None
        assert intervention.is_active

        # Check expiration time (5 minutes for Level 1)
        expected_expiry = intervention.applied_at + timedelta(minutes=5)
        assert abs((intervention.expires_at - expected_expiry).total_seconds()) < 1

        # Check event was published
        event_bus.publish.assert_called_once()
        call_args = event_bus.publish.call_args
        assert call_args[0][0] == EventType.INTERVENTION_APPLIED
        assert call_args[0][1]['profile_id'] == profile_id

    @pytest.mark.asyncio
    async def test_apply_intervention_level2(self, manager, event_bus):
        """Test applying Level 2 intervention (message + position reduction)."""
        profile_id = "test_profile"
        level = TiltLevel.LEVEL2
        tilt_score = 60

        # Apply intervention
        intervention = await manager.apply_intervention(profile_id, level, tilt_score)

        # Assert intervention properties
        assert intervention.intervention_type == InterventionType.POSITION_REDUCTION
        assert intervention.position_size_multiplier == Decimal("0.5")  # 50% reduction
        assert intervention.message in manager.LEVEL2_MESSAGES

        # Check expiration time (10 minutes for Level 2)
        expected_expiry = intervention.applied_at + timedelta(minutes=10)
        assert abs((intervention.expires_at - expected_expiry).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_apply_intervention_level3(self, manager, event_bus):
        """Test applying Level 3 intervention (full lockout)."""
        profile_id = "test_profile"
        level = TiltLevel.LEVEL3
        tilt_score = 90

        # Apply intervention
        intervention = await manager.apply_intervention(profile_id, level, tilt_score)

        # Assert intervention properties
        assert intervention.intervention_type == InterventionType.TRADING_LOCKOUT
        assert intervention.position_size_multiplier == Decimal("0")  # No trading
        assert intervention.message in manager.LEVEL3_MESSAGES

        # Check expiration time (30 minutes for Level 3)
        expected_expiry = intervention.applied_at + timedelta(minutes=30)
        assert abs((intervention.expires_at - expected_expiry).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_get_active_interventions(self, manager):
        """Test getting active interventions."""
        profile_id = "test_profile"

        # Apply multiple interventions
        await manager.apply_intervention(profile_id, TiltLevel.LEVEL1, 30)
        await manager.apply_intervention(profile_id, TiltLevel.LEVEL2, 60)

        # Get active interventions
        active = manager.get_active_interventions(profile_id)
        assert len(active) == 2
        assert all(i.is_active for i in active)

    @pytest.mark.asyncio
    async def test_expired_interventions(self, manager):
        """Test that expired interventions are not returned as active."""
        profile_id = "test_profile"

        # Apply intervention
        intervention = await manager.apply_intervention(profile_id, TiltLevel.LEVEL1, 30)

        # Manually expire it
        intervention.expires_at = datetime.now(UTC) - timedelta(minutes=1)

        # Get active interventions
        active = manager.get_active_interventions(profile_id)
        assert len(active) == 0  # Expired intervention not included

    def test_get_position_size_multiplier(self, manager):
        """Test getting position size multiplier."""
        profile_id = "test_profile"

        # No interventions - normal multiplier
        assert manager.get_position_size_multiplier(profile_id) == Decimal("1.0")

        # Add Level 2 intervention manually
        intervention = Intervention(
            intervention_id="test",
            profile_id=profile_id,
            tilt_level=TiltLevel.LEVEL2,
            intervention_type=InterventionType.POSITION_REDUCTION,
            message="Test",
            applied_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            position_size_multiplier=Decimal("0.5"),
            is_active=True
        )
        manager.active_interventions[profile_id] = [intervention]

        # Check multiplier
        assert manager.get_position_size_multiplier(profile_id) == Decimal("0.5")

    def test_is_trading_locked(self, manager):
        """Test checking if trading is locked."""
        profile_id = "test_profile"

        # No interventions - trading allowed
        assert not manager.is_trading_locked(profile_id)

        # Add Level 3 intervention manually
        intervention = Intervention(
            intervention_id="test",
            profile_id=profile_id,
            tilt_level=TiltLevel.LEVEL3,
            intervention_type=InterventionType.TRADING_LOCKOUT,
            message="Test",
            applied_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
            position_size_multiplier=Decimal("0"),
            is_active=True
        )
        manager.active_interventions[profile_id] = [intervention]

        # Check trading is locked
        assert manager.is_trading_locked(profile_id)

    def test_clear_interventions(self, manager):
        """Test clearing interventions."""
        profile_id = "test_profile"

        # Add interventions
        intervention = Intervention(
            intervention_id="test",
            profile_id=profile_id,
            tilt_level=TiltLevel.LEVEL2,
            intervention_type=InterventionType.POSITION_REDUCTION,
            message="Test",
            applied_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=10),
            position_size_multiplier=Decimal("0.5"),
            is_active=True
        )
        manager.active_interventions[profile_id] = [intervention]

        # Clear interventions
        manager.clear_interventions(profile_id)

        # Check intervention is inactive
        assert not intervention.is_active
        assert len(manager.get_active_interventions(profile_id)) == 0

    @pytest.mark.asyncio
    async def test_check_recovery(self, manager, event_bus):
        """Test checking for recovery from tilt."""
        profile_id = "test_profile"

        # No interventions - already recovered
        assert await manager.check_recovery(profile_id)

        # Add active intervention
        intervention = Intervention(
            intervention_id="test",
            profile_id=profile_id,
            tilt_level=TiltLevel.LEVEL1,
            intervention_type=InterventionType.MESSAGE,
            message="Test",
            applied_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
            position_size_multiplier=None,
            is_active=True
        )
        manager.active_interventions[profile_id] = [intervention]

        # Not recovered yet
        assert not await manager.check_recovery(profile_id)

        # Expire intervention
        intervention.expires_at = datetime.now(UTC) - timedelta(minutes=1)

        # Now recovered
        assert await manager.check_recovery(profile_id)

        # Check recovery event was published
        event_bus.publish.assert_called()
        call_args = event_bus.publish.call_args
        assert call_args[0][0] == EventType.TILT_RECOVERED
        assert call_args[0][1]['profile_id'] == profile_id

    def test_most_restrictive_multiplier(self, manager):
        """Test that most restrictive position multiplier is returned."""
        profile_id = "test_profile"

        # Add multiple interventions with different multipliers
        interventions = [
            Intervention(
                intervention_id="test1",
                profile_id=profile_id,
                tilt_level=TiltLevel.LEVEL1,
                intervention_type=InterventionType.MESSAGE,
                message="Test",
                applied_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
                position_size_multiplier=Decimal("0.8"),
                is_active=True
            ),
            Intervention(
                intervention_id="test2",
                profile_id=profile_id,
                tilt_level=TiltLevel.LEVEL2,
                intervention_type=InterventionType.POSITION_REDUCTION,
                message="Test",
                applied_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(minutes=10),
                position_size_multiplier=Decimal("0.5"),  # Most restrictive
                is_active=True
            )
        ]
        manager.active_interventions[profile_id] = interventions

        # Should return the most restrictive (smallest) multiplier
        assert manager.get_position_size_multiplier(profile_id) == Decimal("0.5")
