"""Unit tests for the journal entry system."""
import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from genesis.tilt.journal_system import (
    JournalSystem,
    JournalEntry,
)


@pytest.fixture
def journal_system():
    """Create a journal system instance for testing."""
    system = JournalSystem(min_word_count=100)
    return system


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = AsyncMock()
    repo.save_journal_entry = AsyncMock()
    repo.get_journal_entries = AsyncMock(return_value=[])
    repo.get_journal_entry = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def sample_content():
    """Create sample journal content that meets requirements."""
    return (
        "Today I experienced a significant tilt episode during trading. "
        "It started when I saw a sudden price movement that went against my position. "
        "Instead of sticking to my plan, I immediately doubled down trying to recover. "
        "This emotional response led to a cascade of poor decisions. "
        "I ignored my stop loss rules and kept adding to losing positions. "
        "The frustration built up and I started revenge trading. "
        "Looking back, I can see the warning signs were there. "
        "My clicking speed increased dramatically and I was canceling orders frequently. "
        "In the future, I need to recognize these behavioral changes earlier. "
        "I commit to taking a break when I notice these patterns emerging. "
        "This reflection helps me understand my emotional triggers better. "
        "I will implement a cooling off period after any significant loss. "
        "Trading discipline is more important than trying to recover losses quickly. "
        "I need to accept that losses are part of trading and not chase them. "
        "Tomorrow I will start fresh with a clear mind and strict risk management."
    )


class TestWordCounting:
    """Test word counting functionality."""

    def test_count_words_empty(self, journal_system):
        """Test counting words in empty text."""
        assert journal_system.count_words("") == 0
        assert journal_system.count_words("   ") == 0
        assert journal_system.count_words("\n\n") == 0

    def test_count_words_simple(self, journal_system):
        """Test counting words in simple text."""
        assert journal_system.count_words("Hello world") == 2
        assert journal_system.count_words("One two three four five") == 5

    def test_count_words_with_punctuation(self, journal_system):
        """Test counting words with punctuation."""
        text = "Hello, world! How are you today?"
        assert journal_system.count_words(text) == 6

    def test_count_words_multiline(self, journal_system):
        """Test counting words in multiline text."""
        text = """
        This is line one.
        This is line two.
        And line three.
        """
        assert journal_system.count_words(text) == 10


class TestContentValidation:
    """Test journal content validation."""

    def test_validate_sufficient_content(self, journal_system, sample_content):
        """Test validation of sufficient content."""
        is_valid, message = journal_system.validate_entry_content(sample_content)
        assert is_valid is True
        assert message == "Entry is valid."

    def test_validate_insufficient_word_count(self, journal_system):
        """Test validation with insufficient word count."""
        content = "This is a very short entry that doesn't meet requirements."
        is_valid, message = journal_system.validate_entry_content(content)
        assert is_valid is False
        assert "must be at least 100 words" in message

    def test_validate_repeated_words(self, journal_system):
        """Test validation of repeated words (not meaningful content)."""
        content = " ".join(["word"] * 150)  # 150 repeated words
        is_valid, message = journal_system.validate_entry_content(content)
        assert is_valid is False
        assert "meaningful reflection" in message

    def test_validate_with_trigger_analysis(self, journal_system, sample_content):
        """Test validation with trigger analysis."""
        trigger = "Market moved against my position and I panicked."
        is_valid, message = journal_system.validate_entry_content(
            sample_content,
            trigger_analysis=trigger,
        )
        assert is_valid is True

    def test_validate_insufficient_trigger_analysis(self, journal_system, sample_content):
        """Test validation with insufficient trigger analysis."""
        trigger = "Bad"  # Too short
        is_valid, message = journal_system.validate_entry_content(
            sample_content,
            trigger_analysis=trigger,
        )
        assert is_valid is False
        assert "Trigger analysis must be more detailed" in message

    def test_validate_with_prevention_plan(self, journal_system, sample_content):
        """Test validation with prevention plan."""
        plan = "I will implement strict stop losses and take breaks when emotional."
        is_valid, message = journal_system.validate_entry_content(
            sample_content,
            prevention_plan=plan,
        )
        assert is_valid is True


class TestJournalSubmission:
    """Test journal entry submission."""

    @pytest.mark.asyncio
    async def test_submit_valid_entry(
        self,
        mock_repository,
        mock_event_bus,
        sample_content,
    ):
        """Test submitting a valid journal entry."""
        system = JournalSystem(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        entry = await system.submit_journal_entry(
            profile_id=profile_id,
            content=sample_content,
            trigger_analysis="Sudden market movement triggered emotional response",
            prevention_plan="Implement cooling off periods and strict stop losses",
        )

        assert entry is not None
        assert entry.profile_id == profile_id
        assert entry.content == sample_content
        assert entry.word_count > 100
        assert entry.is_valid is True

        # Check persistence
        mock_repository.save_journal_entry.assert_called_once()

        # Check event published
        mock_event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_invalid_entry(self, journal_system):
        """Test submitting an invalid journal entry."""
        profile_id = "test_profile"
        content = "Too short"

        entry = await journal_system.submit_journal_entry(
            profile_id=profile_id,
            content=content,
        )

        assert entry is None

    @pytest.mark.asyncio
    async def test_submit_entry_updates_requirements(
        self,
        mock_repository,
        mock_event_bus,
        sample_content,
    ):
        """Test that submitting entry reduces pending requirements."""
        system = JournalSystem(
            repository=mock_repository,
            event_bus=mock_event_bus,
        )
        profile_id = "test_profile"

        # Add requirement
        system.add_journal_requirement(profile_id, 2)
        assert system.get_pending_requirements(profile_id) == 2

        # Submit first entry
        await system.submit_journal_entry(profile_id, sample_content)
        assert system.get_pending_requirements(profile_id) == 1

        # Submit second entry
        await system.submit_journal_entry(profile_id, sample_content)
        assert system.get_pending_requirements(profile_id) == 0

    @pytest.mark.asyncio
    async def test_recent_entries_cache(
        self,
        mock_repository,
        sample_content,
    ):
        """Test that recent entries are cached."""
        system = JournalSystem(repository=mock_repository)
        profile_id = "test_profile"

        # Submit multiple entries
        entry1 = await system.submit_journal_entry(profile_id, sample_content)
        entry2 = await system.submit_journal_entry(profile_id, sample_content)

        # Check cache
        assert profile_id in system.recent_entries
        assert len(system.recent_entries[profile_id]) == 2
        assert system.recent_entries[profile_id][0] == entry1
        assert system.recent_entries[profile_id][1] == entry2


class TestRequirementsManagement:
    """Test journal requirements management."""

    def test_add_journal_requirement(self, journal_system):
        """Test adding journal requirements."""
        profile_id = "test_profile"

        # Initially no requirements
        assert journal_system.get_pending_requirements(profile_id) == 0

        # Add requirements
        journal_system.add_journal_requirement(profile_id, 1)
        assert journal_system.get_pending_requirements(profile_id) == 1

        # Add more
        journal_system.add_journal_requirement(profile_id, 2)
        assert journal_system.get_pending_requirements(profile_id) == 3

    def test_has_completed_requirements(self, journal_system):
        """Test checking if requirements are completed."""
        profile_id = "test_profile"

        # Initially completed (no requirements)
        assert journal_system.has_completed_requirements(profile_id) is True

        # Add requirement
        journal_system.add_journal_requirement(profile_id, 1)
        assert journal_system.has_completed_requirements(profile_id) is False

        # Clear requirements
        journal_system.clear_requirements(profile_id)
        assert journal_system.has_completed_requirements(profile_id) is True

    def test_clear_requirements(self, journal_system):
        """Test clearing requirements."""
        profile_id = "test_profile"

        # Add requirements
        journal_system.add_journal_requirement(profile_id, 5)
        assert journal_system.get_pending_requirements(profile_id) == 5

        # Clear
        journal_system.clear_requirements(profile_id)
        assert journal_system.get_pending_requirements(profile_id) == 0


class TestEntriesRetrieval:
    """Test retrieving journal entries."""

    @pytest.mark.asyncio
    async def test_get_recent_entries_from_cache(
        self,
        mock_repository,
        sample_content,
    ):
        """Test getting recent entries from cache."""
        system = JournalSystem(repository=mock_repository)
        profile_id = "test_profile"

        # Submit entries
        await system.submit_journal_entry(profile_id, sample_content)
        await system.submit_journal_entry(profile_id, sample_content)

        # Get from cache
        entries = await system.get_recent_entries(profile_id)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_get_recent_entries_from_database(self, mock_repository):
        """Test getting recent entries from database."""
        # Mock database response
        mock_entries = [
            {
                "entry_id": str(uuid4()),
                "profile_id": "test_profile",
                "content": "Test entry",
                "word_count": 100,
                "trigger_analysis": None,
                "prevention_plan": None,
                "submitted_at": datetime.now(UTC).isoformat(),
            }
        ]
        mock_repository.get_journal_entries.return_value = mock_entries

        system = JournalSystem(repository=mock_repository)
        entries = await system.get_recent_entries("test_profile")

        assert len(entries) == 1
        assert entries[0].profile_id == "test_profile"
        mock_repository.get_journal_entries.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entry_by_id(self, mock_repository, sample_content):
        """Test getting specific entry by ID."""
        system = JournalSystem(repository=mock_repository)
        profile_id = "test_profile"

        # Submit entry
        entry = await system.submit_journal_entry(profile_id, sample_content)

        # Get from cache
        retrieved = await system.get_entry_by_id(entry.entry_id)
        assert retrieved == entry

    @pytest.mark.asyncio
    async def test_get_entry_by_id_from_database(self, mock_repository):
        """Test getting entry by ID from database."""
        entry_id = str(uuid4())
        mock_entry = {
            "entry_id": entry_id,
            "profile_id": "test_profile",
            "content": "Test entry",
            "word_count": 100,
            "trigger_analysis": None,
            "prevention_plan": None,
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        mock_repository.get_journal_entry.return_value = mock_entry

        system = JournalSystem(repository=mock_repository)
        entry = await system.get_entry_by_id(entry_id)

        assert entry is not None
        assert entry.entry_id == entry_id
        mock_repository.get_journal_entry.assert_called_once_with(entry_id)


class TestIntrospectionPrompts:
    """Test introspection prompts."""

    def test_get_introspection_prompts(self, journal_system):
        """Test getting introspection prompts."""
        prompts = journal_system.get_introspection_prompts()
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) > 10 for p in prompts)

    def test_generate_reflection_template(self, journal_system):
        """Test generating reflection template."""
        template = journal_system.generate_reflection_template()
        assert "TILT RECOVERY JOURNAL" in template
        assert f"Minimum {journal_system.min_word_count} words" in template
        assert "Consider these questions:" in template
        for prompt in journal_system.PROMPTS:
            assert prompt in template


class TestErrorHandling:
    """Test error handling in journal system."""

    @pytest.mark.asyncio
    async def test_persist_error_handling(
        self,
        mock_repository,
        sample_content,
    ):
        """Test error handling when persisting entry."""
        mock_repository.save_journal_entry.side_effect = Exception("Database error")

        system = JournalSystem(repository=mock_repository)

        # Should not raise exception
        entry = await system.submit_journal_entry("test_profile", sample_content)

        assert entry is not None  # Entry still created despite persistence error
        mock_repository.save_journal_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_entries_error_handling(self, mock_repository):
        """Test error handling when loading entries."""
        mock_repository.get_journal_entries.side_effect = Exception("Database error")

        system = JournalSystem(repository=mock_repository)
        entries = await system.get_recent_entries("test_profile")

        assert entries == []  # Returns empty list on error
        mock_repository.get_journal_entries.assert_called_once()