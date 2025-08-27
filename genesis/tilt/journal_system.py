"""Journal entry system for Level 3 tilt recovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import uuid4

import structlog

from genesis.core.events import EventType
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class JournalEntry:
    """Represents a journal entry for tilt recovery."""

    entry_id: str = field(default_factory=lambda: str(uuid4()))
    profile_id: str = ""
    content: str = ""
    word_count: int = 0
    trigger_analysis: Optional[str] = None
    prevention_plan: Optional[str] = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_valid: bool = False


class JournalSystem:
    """Manages journal entries for tilt recovery."""

    # Minimum word count requirement
    MIN_WORD_COUNT = 100

    # Introspection prompts
    PROMPTS = [
        "What emotional state triggered this tilt episode?",
        "What specific market events or losses led to this?",
        "How could you have recognized the warning signs earlier?",
        "What would you do differently next time?",
        "What recovery strategies will you commit to?",
    ]

    def __init__(
        self,
        repository: Optional[SQLiteRepository] = None,
        event_bus: Optional[EventBus] = None,
        min_word_count: int = MIN_WORD_COUNT,
    ):
        """Initialize journal system.

        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing journal events
            min_word_count: Minimum required word count
        """
        self.repository = repository
        self.event_bus = event_bus
        self.min_word_count = min_word_count

        # Cache of pending journal requirements per profile
        self.pending_requirements: dict[str, int] = {}

        # Cache of recent entries for quick access
        self.recent_entries: dict[str, list[JournalEntry]] = {}

    def get_introspection_prompts(self) -> list[str]:
        """Get introspection prompts for journal entry.

        Returns:
            List of prompts to guide reflection
        """
        return self.PROMPTS.copy()

    def count_words(self, text: str) -> int:
        """Count words in text.

        Args:
            text: Text to count words in

        Returns:
            Word count
        """
        if not text:
            return 0

        # Simple word counting - split by whitespace
        words = text.strip().split()
        return len(words)

    def validate_entry_content(
        self,
        content: str,
        trigger_analysis: Optional[str] = None,
        prevention_plan: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Validate journal entry content.

        Args:
            content: Main journal content
            trigger_analysis: Analysis of what triggered tilt
            prevention_plan: Plan to prevent future occurrences

        Returns:
            Tuple of (is_valid, validation_message)
        """
        # Check main content word count
        word_count = self.count_words(content)
        if word_count < self.min_word_count:
            return (
                False,
                f"Entry must be at least {self.min_word_count} words. "
                f"Current: {word_count} words.",
            )

        # Check for meaningful content (not just repeated words)
        unique_words = set(content.lower().split())
        if len(unique_words) < 20:
            return (
                False,
                "Entry must contain meaningful reflection, not repeated text.",
            )

        # Optional: Check trigger analysis if provided
        if trigger_analysis and len(trigger_analysis.strip()) < 10:
            return (
                False,
                "Trigger analysis must be more detailed.",
            )

        # Optional: Check prevention plan if provided
        if prevention_plan and len(prevention_plan.strip()) < 10:
            return (
                False,
                "Prevention plan must be more detailed.",
            )

        return True, "Entry is valid."

    async def submit_journal_entry(
        self,
        profile_id: str,
        content: str,
        trigger_analysis: Optional[str] = None,
        prevention_plan: Optional[str] = None,
    ) -> Optional[JournalEntry]:
        """Submit a journal entry for recovery.

        Args:
            profile_id: Profile identifier
            content: Journal entry content
            trigger_analysis: Analysis of what triggered tilt
            prevention_plan: Plan to prevent future occurrences

        Returns:
            Created journal entry if valid, None otherwise
        """
        # Validate content
        is_valid, message = self.validate_entry_content(
            content, trigger_analysis, prevention_plan
        )

        if not is_valid:
            logger.warning(
                "Invalid journal entry submission",
                profile_id=profile_id,
                reason=message,
            )
            return None

        # Create journal entry
        word_count = self.count_words(content)
        now = datetime.now(UTC)

        entry = JournalEntry(
            entry_id=str(uuid4()),
            profile_id=profile_id,
            content=content,
            word_count=word_count,
            trigger_analysis=trigger_analysis,
            prevention_plan=prevention_plan,
            submitted_at=now,
            created_at=now,
            is_valid=True,
        )

        # Store in cache
        if profile_id not in self.recent_entries:
            self.recent_entries[profile_id] = []
        self.recent_entries[profile_id].append(entry)

        # Keep only last 10 entries in cache
        if len(self.recent_entries[profile_id]) > 10:
            self.recent_entries[profile_id] = self.recent_entries[profile_id][-10:]

        # Persist to database
        if self.repository:
            await self._persist_entry(entry)

        # Update pending requirements
        if profile_id in self.pending_requirements:
            self.pending_requirements[profile_id] = max(
                0, self.pending_requirements[profile_id] - 1
            )

        # Publish event
        if self.event_bus:
            await self._publish_journal_event(entry)

        logger.info(
            "Journal entry submitted",
            profile_id=profile_id,
            entry_id=entry.entry_id,
            word_count=word_count,
            has_trigger_analysis=trigger_analysis is not None,
            has_prevention_plan=prevention_plan is not None,
        )

        return entry

    def add_journal_requirement(self, profile_id: str, count: int = 1) -> None:
        """Add journal entry requirement for a profile.

        Args:
            profile_id: Profile identifier
            count: Number of entries required
        """
        if profile_id not in self.pending_requirements:
            self.pending_requirements[profile_id] = 0

        self.pending_requirements[profile_id] += count

        logger.info(
            "Journal requirement added",
            profile_id=profile_id,
            required_count=self.pending_requirements[profile_id],
        )

    def get_pending_requirements(self, profile_id: str) -> int:
        """Get number of pending journal entries required.

        Args:
            profile_id: Profile identifier

        Returns:
            Number of entries still required
        """
        return self.pending_requirements.get(profile_id, 0)

    def has_completed_requirements(self, profile_id: str) -> bool:
        """Check if journal requirements are completed.

        Args:
            profile_id: Profile identifier

        Returns:
            True if all requirements met
        """
        return self.get_pending_requirements(profile_id) == 0

    async def get_recent_entries(
        self,
        profile_id: str,
        limit: int = 10,
    ) -> list[JournalEntry]:
        """Get recent journal entries for a profile.

        Args:
            profile_id: Profile identifier
            limit: Maximum number of entries to return

        Returns:
            List of recent journal entries
        """
        # Check cache first
        if profile_id in self.recent_entries:
            cached = self.recent_entries[profile_id]
            if cached:
                return cached[-limit:] if limit else cached

        # Load from database if not in cache
        if self.repository:
            try:
                entries_data = await self.repository.get_journal_entries(
                    profile_id, limit
                )
                entries = [
                    self._entry_from_dict(data) for data in entries_data
                ]

                # Update cache
                self.recent_entries[profile_id] = entries

                return entries
            except Exception as e:
                logger.error(
                    "Failed to load journal entries",
                    profile_id=profile_id,
                    error=str(e),
                )

        return []

    async def get_entry_by_id(self, entry_id: str) -> Optional[JournalEntry]:
        """Get a specific journal entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            Journal entry or None if not found
        """
        # Check cache first
        for entries in self.recent_entries.values():
            for entry in entries:
                if entry.entry_id == entry_id:
                    return entry

        # Load from database
        if self.repository:
            try:
                data = await self.repository.get_journal_entry(entry_id)
                if data:
                    return self._entry_from_dict(data)
            except Exception as e:
                logger.error(
                    "Failed to load journal entry",
                    entry_id=entry_id,
                    error=str(e),
                )

        return None

    def clear_requirements(self, profile_id: str) -> None:
        """Clear all journal requirements for a profile.

        Args:
            profile_id: Profile identifier
        """
        if profile_id in self.pending_requirements:
            del self.pending_requirements[profile_id]

        logger.info(
            "Journal requirements cleared",
            profile_id=profile_id,
        )

    async def _persist_entry(self, entry: JournalEntry) -> None:
        """Persist journal entry to database.

        Args:
            entry: Journal entry to persist
        """
        if not self.repository:
            return

        try:
            await self.repository.save_journal_entry(
                {
                    "entry_id": entry.entry_id,
                    "profile_id": entry.profile_id,
                    "content": entry.content,
                    "word_count": entry.word_count,
                    "trigger_analysis": entry.trigger_analysis,
                    "prevention_plan": entry.prevention_plan,
                    "submitted_at": entry.submitted_at.isoformat(),
                    "created_at": entry.created_at.isoformat(),
                }
            )
        except Exception as e:
            logger.error(
                "Failed to persist journal entry",
                entry_id=entry.entry_id,
                error=str(e),
            )

    async def _publish_journal_event(self, entry: JournalEntry) -> None:
        """Publish journal submission event.

        Args:
            entry: Submitted journal entry
        """
        if not self.event_bus:
            return

        remaining = self.get_pending_requirements(entry.profile_id)

        await self.event_bus.publish(
            EventType.JOURNAL_ENTRY_SUBMITTED,
            {
                "profile_id": entry.profile_id,
                "entry_id": entry.entry_id,
                "word_count": entry.word_count,
                "has_trigger_analysis": entry.trigger_analysis is not None,
                "has_prevention_plan": entry.prevention_plan is not None,
                "requirements_remaining": remaining,
                "all_requirements_met": remaining == 0,
                "timestamp": entry.submitted_at.isoformat(),
            },
        )

    def _entry_from_dict(self, data: dict[str, Any]) -> JournalEntry:
        """Create journal entry from dictionary.

        Args:
            data: Entry data

        Returns:
            Journal entry object
        """
        return JournalEntry(
            entry_id=data["entry_id"],
            profile_id=data["profile_id"],
            content=data["content"],
            word_count=data["word_count"],
            trigger_analysis=data.get("trigger_analysis"),
            prevention_plan=data.get("prevention_plan"),
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            created_at=datetime.fromisoformat(data.get("created_at", data["submitted_at"])),
            is_valid=True,
        )

    def generate_reflection_template(self) -> str:
        """Generate a template for journal reflection.

        Returns:
            Template text with prompts
        """
        template = "=== TILT RECOVERY JOURNAL ===\n\n"
        template += "Please reflect on your recent trading experience.\n"
        template += f"Minimum {self.min_word_count} words required.\n\n"

        template += "Consider these questions:\n"
        for i, prompt in enumerate(self.PROMPTS, 1):
            template += f"{i}. {prompt}\n"

        template += "\n--- Begin your reflection below ---\n\n"

        return template
