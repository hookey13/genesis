"""Recovery checklist system for tilt recovery protocols."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from genesis.core.events import EventType
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class ChecklistItemType(Enum):
    """Type of checklist item."""

    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass
class ChecklistItem:
    """Represents a single checklist item."""

    item_id: str
    name: str
    description: str
    item_type: ChecklistItemType
    is_completed: bool = False
    completed_at: datetime | None = None


@dataclass
class RecoveryChecklist:
    """Represents a recovery checklist for a profile."""

    checklist_id: str = field(default_factory=lambda: str(uuid4()))
    profile_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    items: list[ChecklistItem] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_complete: bool = False
    completed_at: datetime | None = None


class RecoveryChecklistManager:
    """Manages recovery checklists for tilt recovery."""

    # Default checklist items
    DEFAULT_ITEMS = [
        {
            "name": "Journal Entry",
            "description": "Complete a reflective journal entry about the tilt episode",
            "item_type": ChecklistItemType.REQUIRED,
        },
        {
            "name": "Performance Review",
            "description": "Review trading performance and identify patterns",
            "item_type": ChecklistItemType.REQUIRED,
        },
        {
            "name": "Meditation Session",
            "description": "Complete a meditation session to reset mentally",
            "item_type": ChecklistItemType.OPTIONAL,
        },
        {
            "name": "Commitment Statement",
            "description": "Write a commitment to follow trading rules",
            "item_type": ChecklistItemType.REQUIRED,
        },
        {
            "name": "Break Time",
            "description": "Take a minimum 15-minute break from screens",
            "item_type": ChecklistItemType.REQUIRED,
        },
        {
            "name": "Risk Review",
            "description": "Review and confirm position sizing rules",
            "item_type": ChecklistItemType.REQUIRED,
        },
    ]

    def __init__(
        self,
        repository: SQLiteRepository | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize recovery checklist manager.

        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing checklist events
        """
        self.repository = repository
        self.event_bus = event_bus

        # Active checklists cache
        self.active_checklists: dict[str, RecoveryChecklist] = {}

    def create_checklist(
        self,
        profile_id: str,
        custom_items: list[dict[str, Any]] | None = None,
    ) -> RecoveryChecklist:
        """Create a new recovery checklist for a profile.

        Args:
            profile_id: Profile identifier
            custom_items: Optional custom checklist items

        Returns:
            Created recovery checklist
        """
        # Use custom items if provided, otherwise use defaults
        items_config = custom_items if custom_items else self.DEFAULT_ITEMS

        # Create checklist items
        items = []
        for item_config in items_config:
            item = ChecklistItem(
                item_id=str(uuid4()),
                name=item_config["name"],
                description=item_config["description"],
                item_type=item_config["item_type"],
                is_completed=False,
            )
            items.append(item)

        # Create checklist
        checklist = RecoveryChecklist(
            checklist_id=str(uuid4()),
            profile_id=profile_id,
            created_at=datetime.now(UTC),
            items=items,
            is_complete=False,
        )

        # Store in cache
        self.active_checklists[profile_id] = checklist

        logger.info(
            "Recovery checklist created",
            profile_id=profile_id,
            checklist_id=checklist.checklist_id,
            item_count=len(items),
            required_count=len(
                [i for i in items if i.item_type == ChecklistItemType.REQUIRED]
            ),
        )

        return checklist

    def get_checklist(self, profile_id: str) -> RecoveryChecklist | None:
        """Get active checklist for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Active checklist or None
        """
        return self.active_checklists.get(profile_id)

    def complete_item(
        self,
        profile_id: str,
        item_name: str,
    ) -> bool:
        """Mark a checklist item as complete.

        Args:
            profile_id: Profile identifier
            item_name: Name of the item to complete

        Returns:
            True if item was marked complete
        """
        checklist = self.get_checklist(profile_id)
        if not checklist:
            logger.warning(
                "No active checklist found",
                profile_id=profile_id,
            )
            return False

        # Find the item
        item_found = False
        for item in checklist.items:
            if item.name == item_name and not item.is_completed:
                item.is_completed = True
                item.completed_at = datetime.now(UTC)
                item_found = True
                break

        if not item_found:
            logger.warning(
                "Checklist item not found or already completed",
                profile_id=profile_id,
                item_name=item_name,
            )
            return False

        # Update last modified
        checklist.last_updated = datetime.now(UTC)

        # Check if all required items are complete
        if self._check_completion(checklist):
            checklist.is_complete = True
            checklist.completed_at = datetime.now(UTC)

        # Persist changes
        if self.repository:
            asyncio.create_task(self._persist_checklist(checklist))

        # Publish event
        if self.event_bus:
            asyncio.create_task(self._publish_checklist_update(checklist, item_name))

        logger.info(
            "Checklist item completed",
            profile_id=profile_id,
            item_name=item_name,
            checklist_complete=checklist.is_complete,
        )

        return True

    def validate_checklist_completion(self, profile_id: str) -> bool:
        """Validate if all required checklist items are complete.

        Args:
            profile_id: Profile identifier

        Returns:
            True if all required items are complete
        """
        checklist = self.get_checklist(profile_id)
        if not checklist:
            return False

        return self._check_completion(checklist)

    def _check_completion(self, checklist: RecoveryChecklist) -> bool:
        """Check if all required items in checklist are complete.

        Args:
            checklist: Checklist to check

        Returns:
            True if all required items are complete
        """
        for item in checklist.items:
            if item.item_type == ChecklistItemType.REQUIRED and not item.is_completed:
                return False
        return True

    def get_progress(self, profile_id: str) -> dict[str, Any]:
        """Get checklist progress for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Progress statistics
        """
        checklist = self.get_checklist(profile_id)
        if not checklist:
            return {
                "has_checklist": False,
                "progress_percentage": 0,
                "required_complete": 0,
                "required_total": 0,
                "optional_complete": 0,
                "optional_total": 0,
            }

        required_items = [
            i for i in checklist.items if i.item_type == ChecklistItemType.REQUIRED
        ]
        optional_items = [
            i for i in checklist.items if i.item_type == ChecklistItemType.OPTIONAL
        ]

        required_complete = len([i for i in required_items if i.is_completed])
        optional_complete = len([i for i in optional_items if i.is_completed])

        total_complete = required_complete + optional_complete
        total_items = len(checklist.items)

        progress_percentage = (
            (total_complete / total_items * 100) if total_items > 0 else 0
        )

        return {
            "has_checklist": True,
            "checklist_id": checklist.checklist_id,
            "is_complete": checklist.is_complete,
            "progress_percentage": round(progress_percentage, 1),
            "required_complete": required_complete,
            "required_total": len(required_items),
            "optional_complete": optional_complete,
            "optional_total": len(optional_items),
            "items": [
                {
                    "name": item.name,
                    "description": item.description,
                    "type": item.item_type.value,
                    "is_completed": item.is_completed,
                    "completed_at": (
                        item.completed_at.isoformat() if item.completed_at else None
                    ),
                }
                for item in checklist.items
            ],
        }

    def reset_checklist(self, profile_id: str) -> bool:
        """Reset a checklist to uncompleted state.

        Args:
            profile_id: Profile identifier

        Returns:
            True if checklist was reset
        """
        checklist = self.get_checklist(profile_id)
        if not checklist:
            return False

        # Reset all items
        for item in checklist.items:
            item.is_completed = False
            item.completed_at = None

        # Reset checklist
        checklist.is_complete = False
        checklist.completed_at = None
        checklist.last_updated = datetime.now(UTC)

        logger.info(
            "Recovery checklist reset",
            profile_id=profile_id,
            checklist_id=checklist.checklist_id,
        )

        return True

    def clear_checklist(self, profile_id: str) -> bool:
        """Clear checklist for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            True if checklist was cleared
        """
        if profile_id in self.active_checklists:
            del self.active_checklists[profile_id]
            logger.info(
                "Recovery checklist cleared",
                profile_id=profile_id,
            )
            return True
        return False

    def can_resume_trading(self, profile_id: str) -> bool:
        """Check if profile can resume trading based on checklist.

        Args:
            profile_id: Profile identifier

        Returns:
            True if all required items are complete
        """
        return self.validate_checklist_completion(profile_id)

    def get_incomplete_required_items(self, profile_id: str) -> list[str]:
        """Get list of incomplete required items.

        Args:
            profile_id: Profile identifier

        Returns:
            List of incomplete required item names
        """
        checklist = self.get_checklist(profile_id)
        if not checklist:
            return []

        incomplete = []
        for item in checklist.items:
            if item.item_type == ChecklistItemType.REQUIRED and not item.is_completed:
                incomplete.append(item.name)

        return incomplete

    async def _persist_checklist(self, checklist: RecoveryChecklist) -> None:
        """Persist checklist to database.

        Args:
            checklist: Checklist to persist
        """
        if not self.repository:
            return

        try:
            checklist_data = {
                "checklist_id": checklist.checklist_id,
                "profile_id": checklist.profile_id,
                "created_at": checklist.created_at.isoformat(),
                "last_updated": checklist.last_updated.isoformat(),
                "is_complete": checklist.is_complete,
                "completed_at": (
                    checklist.completed_at.isoformat()
                    if checklist.completed_at
                    else None
                ),
                "items": [
                    {
                        "item_id": item.item_id,
                        "name": item.name,
                        "description": item.description,
                        "item_type": item.item_type.value,
                        "is_completed": item.is_completed,
                        "completed_at": (
                            item.completed_at.isoformat() if item.completed_at else None
                        ),
                    }
                    for item in checklist.items
                ],
            }

            await self.repository.save_recovery_checklist(checklist_data)
        except Exception as e:
            logger.error(
                "Failed to persist checklist",
                checklist_id=checklist.checklist_id,
                error=str(e),
            )

    async def _publish_checklist_update(
        self,
        checklist: RecoveryChecklist,
        item_name: str,
    ) -> None:
        """Publish checklist update event.

        Args:
            checklist: Updated checklist
            item_name: Name of completed item
        """
        if not self.event_bus:
            return

        progress = self.get_progress(checklist.profile_id)

        await self.event_bus.publish(
            EventType.RECOVERY_CHECKLIST_UPDATED,
            {
                "profile_id": checklist.profile_id,
                "checklist_id": checklist.checklist_id,
                "item_completed": item_name,
                "is_complete": checklist.is_complete,
                "progress_percentage": progress["progress_percentage"],
                "required_remaining": progress["required_total"]
                - progress["required_complete"],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    async def load_active_checklists(self) -> None:
        """Load active checklists from database on startup."""
        if not self.repository:
            return

        try:
            checklists_data = await self.repository.get_active_recovery_checklists()

            for data in checklists_data:
                checklist = self._checklist_from_dict(data)
                if not checklist.is_complete:
                    self.active_checklists[checklist.profile_id] = checklist

            logger.info(
                "Active recovery checklists loaded",
                count=len(self.active_checklists),
            )
        except Exception as e:
            logger.error(
                "Failed to load active checklists",
                error=str(e),
            )

    def _checklist_from_dict(self, data: dict[str, Any]) -> RecoveryChecklist:
        """Create checklist from dictionary.

        Args:
            data: Checklist data

        Returns:
            Recovery checklist object
        """
        items = []
        for item_data in data.get("items", []):
            item = ChecklistItem(
                item_id=item_data["item_id"],
                name=item_data["name"],
                description=item_data["description"],
                item_type=ChecklistItemType(item_data["item_type"]),
                is_completed=item_data.get("is_completed", False),
                completed_at=(
                    datetime.fromisoformat(item_data["completed_at"])
                    if item_data.get("completed_at")
                    else None
                ),
            )
            items.append(item)

        return RecoveryChecklist(
            checklist_id=data["checklist_id"],
            profile_id=data["profile_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            items=items,
            last_updated=datetime.fromisoformat(
                data.get("last_updated", data["created_at"])
            ),
            is_complete=data.get("is_complete", False),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
        )
