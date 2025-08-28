"""Unit tests for the lockout management system."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from genesis.tilt.detector import TiltLevel
from genesis.tilt.lockout_manager import (
    LockoutManager,
    LockoutStatus,
)


@pytest.fixture
def lockout_manager():
    """Create a lockout manager instance for testing."""
    manager = LockoutManager()
    return manager


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = AsyncMock()
    repo.save_lockout = AsyncMock()
    repo.update_lockout_status = AsyncMock()
    repo.get_active_lockouts = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


class TestLockoutDurationCalculation:
    """Test lockout duration calculations."""

    def test_base_durations(self, lockout_manager):
        """Test base duration calculations for first occurrence."""
        # Level 1: 5 minutes
        assert lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL1, 1) == 5

        # Level 2: 30 minutes
        assert lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL2, 1) == 30

        # Level 3: 1440 minutes (24 hours)
        assert lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL3, 1) == 1440

    def test_graduated_durations(self, lockout_manager):
        """Test graduated duration increases with occurrences."""
        # Level 1 with multiple occurrences
        assert lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL1, 1) == 5
        assert (
            lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL1, 2) == 7
        )  # 5 * 1.5
        assert (
            lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL1, 3) == 11
        )  # 5 * 1.5^2

        # Level 2 with multiple occurrences
        assert lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL2, 1) == 30
        assert (
            lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL2, 2) == 45
        )  # 30 * 1.5
        assert (
            lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL2, 3) == 67
        )  # 30 * 1.5^2

    def test_maximum_duration_cap(self, lockout_manager):
        """Test that duration is capped at maximum."""
        # Very high occurrence count should hit the cap
        duration = lockout_manager.calculate_lockout_duration(TiltLevel.LEVEL3, 10)
        assert duration == LockoutManager.MAX_LOCKOUT_MINUTES  # 10080 minutes (1 week)

    def test_unknown_level_default(self, lockout_manager):
        """Test default duration for unknown level."""
        # Mock an unknown level
        duration = lockout_manager.calculate_lockout_duration(TiltLevel.NORMAL, 1)
        assert duration == 5  # Default 5 minutes


class TestLockoutEnforcement:
    """Test lockout enforcement."""

    @pytest.mark.asyncio
    async def test_enforce_lockout_first_time(self, mock_repository, mock_event_bus):
        """Test enforcing lockout for first occurrence."""
        manager = LockoutManager(repository=mock_repository, event_bus=mock_event_bus)
        profile_id = "test_profile"

        lockout = await manager.enforce_lockout(
            profile_id=profile_id,
            tilt_level=TiltLevel.LEVEL1,
            reason="Test lockout",
        )

        assert lockout.profile_id == profile_id
        assert lockout.tilt_level == TiltLevel.LEVEL1
        assert lockout.duration_minutes == 5
        assert lockout.status == LockoutStatus.ACTIVE
        assert lockout.occurrence_count == 1
        assert lockout.reason == "Test lockout"

        # Check that lockout was persisted
        mock_repository.save_lockout.assert_called_once()

        # Check that event was published
        mock_event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_enforce_lockout_multiple_occurrences(
        self, mock_repository, mock_event_bus
    ):
        """Test enforcing lockout with increasing occurrences."""
        manager = LockoutManager(repository=mock_repository, event_bus=mock_event_bus)
        profile_id = "test_profile"

        # First occurrence
        lockout1 = await manager.enforce_lockout(profile_id, TiltLevel.LEVEL2)
        assert lockout1.duration_minutes == 30
        assert lockout1.occurrence_count == 1

        # Second occurrence
        lockout2 = await manager.enforce_lockout(profile_id, TiltLevel.LEVEL2)
        assert lockout2.duration_minutes == 45  # 30 * 1.5
        assert lockout2.occurrence_count == 2

        # Third occurrence
        lockout3 = await manager.enforce_lockout(profile_id, TiltLevel.LEVEL2)
        assert lockout3.duration_minutes == 67  # 30 * 1.5^2
        assert lockout3.occurrence_count == 3

    @pytest.mark.asyncio
    async def test_lockout_expiration_times(self, lockout_manager):
        """Test that expiration times are calculated correctly."""
        profile_id = "test_profile"

        with patch("genesis.tilt.lockout_manager.datetime") as mock_datetime:
            now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )

            lockout = await lockout_manager.enforce_lockout(
                profile_id=profile_id,
                tilt_level=TiltLevel.LEVEL1,
            )

            assert lockout.started_at == now
            assert lockout.expires_at == now + timedelta(minutes=5)


class TestLockoutStatusChecking:
    """Test lockout status checking."""

    def test_no_lockout_status(self, lockout_manager):
        """Test status when no lockout exists."""
        status = lockout_manager.check_lockout_status("unknown_profile")
        assert status == LockoutStatus.NO_LOCKOUT

    @pytest.mark.asyncio
    async def test_active_lockout_status(self, lockout_manager):
        """Test status for active lockout."""
        profile_id = "test_profile"

        await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL1)
        status = lockout_manager.check_lockout_status(profile_id)
        assert status == LockoutStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_expired_lockout_status(self, lockout_manager):
        """Test status for expired lockout."""
        profile_id = "test_profile"

        # Create a lockout that's already expired
        with patch("genesis.tilt.lockout_manager.datetime") as mock_datetime:
            # Set time to past for creation
            past = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = past
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )

            await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL1)

            # Move time forward past expiration
            future = past + timedelta(minutes=10)
            mock_datetime.now.return_value = future

            status = lockout_manager.check_lockout_status(profile_id)
            assert status == LockoutStatus.EXPIRED

            # Lockout should be removed from active cache
            assert profile_id not in lockout_manager.active_lockouts


class TestLockoutRetrieval:
    """Test lockout retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_active_lockout(self, lockout_manager):
        """Test retrieving active lockout."""
        profile_id = "test_profile"

        # No lockout initially
        assert lockout_manager.get_active_lockout(profile_id) is None

        # Enforce lockout
        await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL2)

        # Should now retrieve the lockout
        lockout = lockout_manager.get_active_lockout(profile_id)
        assert lockout is not None
        assert lockout.profile_id == profile_id
        assert lockout.tilt_level == TiltLevel.LEVEL2

    @pytest.mark.asyncio
    async def test_get_remaining_minutes(self, lockout_manager):
        """Test getting remaining lockout minutes."""
        profile_id = "test_profile"

        # No lockout initially
        assert lockout_manager.get_remaining_minutes(profile_id) == 0

        with patch("genesis.tilt.lockout_manager.datetime") as mock_datetime:
            now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )

            # Enforce 30-minute lockout
            await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL2)

            # Check immediately - should be 30 minutes
            assert lockout_manager.get_remaining_minutes(profile_id) == 30

            # Move forward 10 minutes
            mock_datetime.now.return_value = now + timedelta(minutes=10)
            assert lockout_manager.get_remaining_minutes(profile_id) == 20

            # Move forward 30 minutes total (expired)
            mock_datetime.now.return_value = now + timedelta(minutes=30)
            assert lockout_manager.get_remaining_minutes(profile_id) == 0


class TestLockoutRelease:
    """Test manual lockout release."""

    @pytest.mark.asyncio
    async def test_release_active_lockout(self, mock_repository, mock_event_bus):
        """Test releasing an active lockout."""
        manager = LockoutManager(repository=mock_repository, event_bus=mock_event_bus)
        profile_id = "test_profile"

        # Enforce lockout
        await manager.enforce_lockout(profile_id, TiltLevel.LEVEL3)
        assert profile_id in manager.active_lockouts

        # Release lockout
        result = await manager.release_lockout(profile_id)
        assert result is True
        assert profile_id not in manager.active_lockouts

        # Check database update
        mock_repository.update_lockout_status.assert_called_once()

        # Check event published
        assert mock_event_bus.publish.call_count == 2  # enforce + release

    @pytest.mark.asyncio
    async def test_release_nonexistent_lockout(self, lockout_manager):
        """Test releasing a lockout that doesn't exist."""
        result = await lockout_manager.release_lockout("unknown_profile")
        assert result is False


class TestOccurrenceCountManagement:
    """Test occurrence count management."""

    @pytest.mark.asyncio
    async def test_occurrence_count_tracking(self, lockout_manager):
        """Test that occurrence counts are tracked correctly."""
        profile_id = "test_profile"

        # Multiple Level 1 occurrences
        await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL1)
        await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL1)
        await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL1)

        assert lockout_manager.occurrence_counts[profile_id][TiltLevel.LEVEL1] == 3

        # Different level
        await lockout_manager.enforce_lockout(profile_id, TiltLevel.LEVEL2)
        assert lockout_manager.occurrence_counts[profile_id][TiltLevel.LEVEL2] == 1

    def test_reset_specific_level_count(self, lockout_manager):
        """Test resetting occurrence count for specific level."""
        profile_id = "test_profile"

        # Set up counts
        lockout_manager.occurrence_counts[profile_id] = {
            TiltLevel.LEVEL1: 3,
            TiltLevel.LEVEL2: 2,
        }

        # Reset Level 1 only
        lockout_manager.reset_occurrence_count(profile_id, TiltLevel.LEVEL1)

        assert lockout_manager.occurrence_counts[profile_id][TiltLevel.LEVEL1] == 0
        assert lockout_manager.occurrence_counts[profile_id][TiltLevel.LEVEL2] == 2

    def test_reset_all_counts(self, lockout_manager):
        """Test resetting all occurrence counts."""
        profile_id = "test_profile"

        # Set up counts
        lockout_manager.occurrence_counts[profile_id] = {
            TiltLevel.LEVEL1: 3,
            TiltLevel.LEVEL2: 2,
            TiltLevel.LEVEL3: 1,
        }

        # Reset all
        lockout_manager.reset_occurrence_count(profile_id)

        assert lockout_manager.occurrence_counts[profile_id] == {}


class TestLockoutPersistence:
    """Test lockout persistence and loading."""

    @pytest.mark.asyncio
    async def test_load_active_lockouts_on_startup(self, mock_repository):
        """Test loading active lockouts from database."""
        # Mock database response
        now = datetime.now(UTC)
        future = now + timedelta(minutes=30)

        mock_repository.get_active_lockouts.return_value = [
            {
                "lockout_id": "lockout_1",
                "profile_id": "profile_1",
                "tilt_level": "level2",
                "duration_minutes": 30,
                "started_at": now.isoformat(),
                "expires_at": future.isoformat(),
                "status": "active",
                "occurrence_count": 2,
                "reason": "Test lockout",
            }
        ]

        manager = LockoutManager(repository=mock_repository)
        await manager.load_active_lockouts()

        assert "profile_1" in manager.active_lockouts
        lockout = manager.active_lockouts["profile_1"]
        assert lockout.lockout_id == "lockout_1"
        assert lockout.tilt_level == TiltLevel.LEVEL2
        assert lockout.occurrence_count == 2

    @pytest.mark.asyncio
    async def test_persist_lockout_error_handling(
        self, mock_repository, mock_event_bus
    ):
        """Test error handling when persisting lockout."""
        mock_repository.save_lockout.side_effect = Exception("Database error")

        manager = LockoutManager(repository=mock_repository, event_bus=mock_event_bus)

        # Should not raise exception, just log error
        lockout = await manager.enforce_lockout("test_profile", TiltLevel.LEVEL1)

        assert lockout is not None
        assert lockout.status == LockoutStatus.ACTIVE
        mock_repository.save_lockout.assert_called_once()
