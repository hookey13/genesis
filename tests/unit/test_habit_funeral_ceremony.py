"""Unit tests for habit funeral ceremony system."""

import asyncio
import hashlib
from unittest.mock import MagicMock, patch

import pytest

from genesis.core.exceptions import ValidationError
from genesis.tilt.habit_funeral_ceremony import (
    CeremonyRecord,
    HabitCommitment,
    HabitFuneralCeremony,
)


class TestHabitFuneralCeremony:
    """Test suite for HabitFuneralCeremony."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.commit = MagicMock()
        session.rollback = MagicMock()
        return session

    @pytest.fixture
    def ceremony(self, mock_session):
        """Create HabitFuneralCeremony instance with mocked dependencies."""
        with patch(
            "genesis.tilt.habit_funeral_ceremony.get_session", return_value=mock_session
        ):
            return HabitFuneralCeremony(
                account_id="test-account-123",
                transition_id="trans-456",
                from_tier="SNIPER",
                to_tier="HUNTER",
            )

    @pytest.mark.asyncio
    async def test_initialization(self, ceremony):
        """Test ceremony initialization."""
        assert ceremony.account_id == "test-account-123"
        assert ceremony.transition_id == "trans-456"
        assert ceremony.from_tier == "SNIPER"
        assert ceremony.to_tier == "HUNTER"
        assert ceremony.ceremony_record is None

    @pytest.mark.asyncio
    async def test_conduct_funeral_valid(self, ceremony):
        """Test conducting funeral with valid habits."""
        old_habits = [
            "Revenge trading after losses",
            "Oversizing positions when confident",
            "Checking PnL every minute",
        ]

        ceremony_record = await ceremony.conduct_funeral(old_habits)

        assert isinstance(ceremony_record, CeremonyRecord)
        assert ceremony_record.account_id == "test-account-123"
        assert ceremony_record.from_tier == "SNIPER"
        assert ceremony_record.to_tier == "HUNTER"
        assert len(ceremony_record.buried_habits) == 3
        assert ceremony_record.ceremony_timestamp is not None
        assert ceremony_record.certificate_hash is not None

    @pytest.mark.asyncio
    async def test_conduct_funeral_too_few_habits(self, ceremony):
        """Test conducting funeral with too few habits."""
        old_habits = ["Only one habit"]

        with pytest.raises(ValidationError) as exc_info:
            await ceremony.conduct_funeral(old_habits)

        assert "at least 3 habits" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_conduct_funeral_too_many_habits(self, ceremony):
        """Test conducting funeral with too many habits."""
        old_habits = [f"Habit {i}" for i in range(6)]

        with pytest.raises(ValidationError) as exc_info:
            await ceremony.conduct_funeral(old_habits)

        assert "maximum of 5 habits" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_conduct_funeral_empty_habits(self, ceremony):
        """Test conducting funeral with empty habit strings."""
        old_habits = ["Valid habit", "", "Another habit"]

        with pytest.raises(ValidationError) as exc_info:
            await ceremony.conduct_funeral(old_habits)

        assert "Empty habit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_certificate(self, ceremony):
        """Test certificate generation."""
        old_habits = ["Bad habit 1", "Bad habit 2", "Bad habit 3"]

        ceremony_record = await ceremony.conduct_funeral(old_habits)
        certificate = await ceremony.generate_certificate(ceremony_record)

        assert isinstance(certificate, str)
        assert "HABIT FUNERAL CERTIFICATE" in certificate
        assert "test-account-123" in certificate
        assert "SNIPER" in certificate
        assert "HUNTER" in certificate
        assert all(habit in certificate for habit in old_habits)
        assert ceremony_record.certificate_hash in certificate

    @pytest.mark.asyncio
    async def test_certificate_integrity(self, ceremony):
        """Test certificate integrity hash."""
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]

        ceremony_record = await ceremony.conduct_funeral(old_habits)
        certificate = await ceremony.generate_certificate(ceremony_record)

        # Verify hash matches content
        content_to_hash = f"{ceremony_record.ceremony_id}{ceremony_record.account_id}{ceremony_record.ceremony_timestamp.isoformat()}"
        expected_hash = hashlib.sha256(content_to_hash.encode()).hexdigest()

        assert ceremony_record.certificate_hash == expected_hash
        assert expected_hash in certificate

    @pytest.mark.asyncio
    async def test_add_commitment(self, ceremony):
        """Test adding commitment after ceremony."""
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]
        ceremony_record = await ceremony.conduct_funeral(old_habits)

        commitment = await ceremony.add_commitment(
            "I commit to using proper position sizing according to tier limits"
        )

        assert isinstance(commitment, HabitCommitment)
        assert (
            commitment.commitment_text
            == "I commit to using proper position sizing according to tier limits"
        )
        assert commitment.ceremony_id == ceremony_record.ceremony_id
        assert commitment.created_at is not None

    @pytest.mark.asyncio
    async def test_add_commitment_before_ceremony(self, ceremony):
        """Test adding commitment before conducting ceremony."""
        with pytest.raises(ValidationError) as exc_info:
            await ceremony.add_commitment("Some commitment")

        assert "No ceremony conducted" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_commitment_empty(self, ceremony):
        """Test adding empty commitment."""
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]
        await ceremony.conduct_funeral(old_habits)

        with pytest.raises(ValidationError) as exc_info:
            await ceremony.add_commitment("")

        assert "Empty commitment" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_is_complete(self, ceremony):
        """Test checking ceremony completion status."""
        # Initially not complete
        assert not await ceremony.is_complete()

        # Conduct ceremony
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]
        await ceremony.conduct_funeral(old_habits)

        # Now should be complete
        assert await ceremony.is_complete()

    @pytest.mark.asyncio
    async def test_get_ceremony_record(self, ceremony):
        """Test retrieving ceremony record."""
        # Initially None
        assert await ceremony.get_ceremony_record() is None

        # Conduct ceremony
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]
        ceremony_record = await ceremony.conduct_funeral(old_habits)

        # Should return the record
        retrieved = await ceremony.get_ceremony_record()
        assert retrieved == ceremony_record

    @pytest.mark.asyncio
    async def test_database_persistence(self, ceremony, mock_session):
        """Test database persistence of ceremony."""
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]

        await ceremony.conduct_funeral(old_habits)

        # Verify database methods were called
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_ceremony_idempotency(self, ceremony):
        """Test ceremony can only be conducted once."""
        old_habits = ["Habit 1", "Habit 2", "Habit 3"]

        # First ceremony should succeed
        ceremony_record1 = await ceremony.conduct_funeral(old_habits)
        assert ceremony_record1 is not None

        # Second attempt should raise error
        with pytest.raises(ValidationError) as exc_info:
            await ceremony.conduct_funeral(old_habits)

        assert "already conducted" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tier_specific_habits(self):
        """Test different tiers have different suggested bad habits."""
        with patch("genesis.tilt.habit_funeral_ceremony.get_session"):
            sniper_ceremony = HabitFuneralCeremony(
                account_id="test",
                transition_id="trans1",
                from_tier="SNIPER",
                to_tier="HUNTER",
            )

            hunter_ceremony = HabitFuneralCeremony(
                account_id="test",
                transition_id="trans2",
                from_tier="HUNTER",
                to_tier="STRATEGIST",
            )

            sniper_suggestions = await sniper_ceremony.get_suggested_habits()
            hunter_suggestions = await hunter_ceremony.get_suggested_habits()

            # Different tiers should have different suggested habits
            assert sniper_suggestions != hunter_suggestions

    @pytest.mark.asyncio
    async def test_concurrent_ceremonies(self):
        """Test handling multiple concurrent ceremonies."""
        with patch("genesis.tilt.habit_funeral_ceremony.get_session"):
            ceremonies = []
            for i in range(3):
                ceremony = HabitFuneralCeremony(
                    account_id=f"account-{i}",
                    transition_id=f"trans-{i}",
                    from_tier="SNIPER",
                    to_tier="HUNTER",
                )
                ceremonies.append(ceremony)

            # Conduct ceremonies concurrently
            tasks = []
            for ceremony in ceremonies:
                habits = [f"Habit {j}" for j in range(3)]
                tasks.append(ceremony.conduct_funeral(habits))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            assert all(isinstance(r, CeremonyRecord) for r in results)

            # Each should have unique ceremony ID
            ceremony_ids = [r.ceremony_id for r in results]
            assert len(ceremony_ids) == len(set(ceremony_ids))
