"""Unit tests for recovery checklist system."""
import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from genesis.tilt.recovery_checklist import (
    RecoveryChecklistManager,
    RecoveryChecklist,
    ChecklistItem,
    ChecklistItemType,
)


@pytest.fixture
def checklist_manager():
    """Create a checklist manager for testing."""
    return RecoveryChecklistManager()


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = AsyncMock()
    repo.save_recovery_checklist = AsyncMock()
    repo.get_active_recovery_checklists = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


class TestChecklistCreation:
    """Test checklist creation."""

    def test_create_default_checklist(self, checklist_manager):
        """Test creating checklist with default items."""
        profile_id = "test_profile"
        
        checklist = checklist_manager.create_checklist(profile_id)
        
        assert checklist.profile_id == profile_id
        assert len(checklist.items) == len(RecoveryChecklistManager.DEFAULT_ITEMS)
        assert not checklist.is_complete
        
        # Check items
        required_count = sum(
            1 for item in checklist.items 
            if item.item_type == ChecklistItemType.REQUIRED
        )
        assert required_count > 0

    def test_create_custom_checklist(self, checklist_manager):
        """Test creating checklist with custom items."""
        profile_id = "test_profile"
        custom_items = [
            {
                "name": "Custom Task 1",
                "description": "First custom task",
                "item_type": ChecklistItemType.REQUIRED,
            },
            {
                "name": "Custom Task 2",
                "description": "Second custom task",
                "item_type": ChecklistItemType.OPTIONAL,
            },
        ]
        
        checklist = checklist_manager.create_checklist(profile_id, custom_items)
        
        assert len(checklist.items) == 2
        assert checklist.items[0].name == "Custom Task 1"
        assert checklist.items[1].name == "Custom Task 2"

    def test_checklist_stored_in_cache(self, checklist_manager):
        """Test that created checklist is stored in cache."""
        profile_id = "test_profile"
        
        checklist = checklist_manager.create_checklist(profile_id)
        
        assert profile_id in checklist_manager.active_checklists
        assert checklist_manager.active_checklists[profile_id] == checklist


class TestItemCompletion:
    """Test checklist item completion."""

    def test_complete_item(self, checklist_manager):
        """Test completing a checklist item."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Get first item name
        first_item = checklist.items[0]
        
        # Complete the item
        success = checklist_manager.complete_item(profile_id, first_item.name)
        
        assert success is True
        assert first_item.is_completed is True
        assert first_item.completed_at is not None

    def test_complete_nonexistent_item(self, checklist_manager):
        """Test completing an item that doesn't exist."""
        profile_id = "test_profile"
        checklist_manager.create_checklist(profile_id)
        
        success = checklist_manager.complete_item(profile_id, "Nonexistent Item")
        
        assert success is False

    def test_complete_already_completed_item(self, checklist_manager):
        """Test completing an already completed item."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        first_item = checklist.items[0]
        
        # Complete once
        checklist_manager.complete_item(profile_id, first_item.name)
        
        # Try to complete again
        success = checklist_manager.complete_item(profile_id, first_item.name)
        
        assert success is False

    def test_checklist_completion_detection(self, checklist_manager):
        """Test that checklist completion is detected."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete all required items
        for item in checklist.items:
            if item.item_type == ChecklistItemType.REQUIRED:
                checklist_manager.complete_item(profile_id, item.name)
        
        # Checklist should be complete
        assert checklist.is_complete is True
        assert checklist.completed_at is not None


class TestChecklistValidation:
    """Test checklist validation."""

    def test_validate_incomplete_checklist(self, checklist_manager):
        """Test validation when checklist is incomplete."""
        profile_id = "test_profile"
        checklist_manager.create_checklist(profile_id)
        
        is_valid = checklist_manager.validate_checklist_completion(profile_id)
        
        assert is_valid is False

    def test_validate_complete_checklist(self, checklist_manager):
        """Test validation when all required items are complete."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete all required items
        for item in checklist.items:
            if item.item_type == ChecklistItemType.REQUIRED:
                checklist_manager.complete_item(profile_id, item.name)
        
        is_valid = checklist_manager.validate_checklist_completion(profile_id)
        
        assert is_valid is True

    def test_validate_with_incomplete_optional(self, checklist_manager):
        """Test validation with incomplete optional items."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete only required items
        for item in checklist.items:
            if item.item_type == ChecklistItemType.REQUIRED:
                checklist_manager.complete_item(profile_id, item.name)
        
        # Should still be valid (optional items not required)
        is_valid = checklist_manager.validate_checklist_completion(profile_id)
        
        assert is_valid is True

    def test_validate_no_checklist(self, checklist_manager):
        """Test validation when no checklist exists."""
        is_valid = checklist_manager.validate_checklist_completion("unknown_profile")
        
        assert is_valid is False


class TestProgressTracking:
    """Test checklist progress tracking."""

    def test_get_progress_no_checklist(self, checklist_manager):
        """Test getting progress when no checklist exists."""
        progress = checklist_manager.get_progress("unknown_profile")
        
        assert progress["has_checklist"] is False
        assert progress["progress_percentage"] == 0

    def test_get_progress_empty_checklist(self, checklist_manager):
        """Test getting progress for new checklist."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        progress = checklist_manager.get_progress(profile_id)
        
        assert progress["has_checklist"] is True
        assert progress["progress_percentage"] == 0
        assert progress["required_complete"] == 0
        assert progress["optional_complete"] == 0

    def test_get_progress_partial_completion(self, checklist_manager):
        """Test getting progress with partial completion."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete first item
        checklist_manager.complete_item(profile_id, checklist.items[0].name)
        
        progress = checklist_manager.get_progress(profile_id)
        
        assert progress["progress_percentage"] > 0
        assert progress["progress_percentage"] < 100

    def test_get_progress_full_completion(self, checklist_manager):
        """Test getting progress with full completion."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete all items
        for item in checklist.items:
            checklist_manager.complete_item(profile_id, item.name)
        
        progress = checklist_manager.get_progress(profile_id)
        
        assert progress["progress_percentage"] == 100
        assert progress["is_complete"] is True


class TestChecklistReset:
    """Test checklist reset functionality."""

    def test_reset_checklist(self, checklist_manager):
        """Test resetting a checklist."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete some items
        for item in checklist.items[:2]:
            checklist_manager.complete_item(profile_id, item.name)
        
        # Reset
        success = checklist_manager.reset_checklist(profile_id)
        
        assert success is True
        
        # All items should be uncompleted
        for item in checklist.items:
            assert item.is_completed is False
            assert item.completed_at is None
        
        assert checklist.is_complete is False

    def test_reset_nonexistent_checklist(self, checklist_manager):
        """Test resetting when no checklist exists."""
        success = checklist_manager.reset_checklist("unknown_profile")
        
        assert success is False


class TestTradingResumption:
    """Test trading resumption checks."""

    def test_can_resume_trading_incomplete(self, checklist_manager):
        """Test trading resumption with incomplete checklist."""
        profile_id = "test_profile"
        checklist_manager.create_checklist(profile_id)
        
        can_resume = checklist_manager.can_resume_trading(profile_id)
        
        assert can_resume is False

    def test_can_resume_trading_complete(self, checklist_manager):
        """Test trading resumption with complete checklist."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Complete all required items
        for item in checklist.items:
            if item.item_type == ChecklistItemType.REQUIRED:
                checklist_manager.complete_item(profile_id, item.name)
        
        can_resume = checklist_manager.can_resume_trading(profile_id)
        
        assert can_resume is True

    def test_get_incomplete_required_items(self, checklist_manager):
        """Test getting list of incomplete required items."""
        profile_id = "test_profile"
        checklist = checklist_manager.create_checklist(profile_id)
        
        # Get all required item names
        required_names = [
            item.name for item in checklist.items
            if item.item_type == ChecklistItemType.REQUIRED
        ]
        
        # Initially all required items are incomplete
        incomplete = checklist_manager.get_incomplete_required_items(profile_id)
        assert set(incomplete) == set(required_names)
        
        # Complete one required item
        first_required = next(
            item for item in checklist.items
            if item.item_type == ChecklistItemType.REQUIRED
        )
        checklist_manager.complete_item(profile_id, first_required.name)
        
        # Should have one less incomplete item
        incomplete = checklist_manager.get_incomplete_required_items(profile_id)
        assert len(incomplete) == len(required_names) - 1
        assert first_required.name not in incomplete


class TestChecklistManagement:
    """Test checklist management operations."""

    def test_get_checklist(self, checklist_manager):
        """Test getting an active checklist."""
        profile_id = "test_profile"
        created = checklist_manager.create_checklist(profile_id)
        
        retrieved = checklist_manager.get_checklist(profile_id)
        
        assert retrieved == created

    def test_get_nonexistent_checklist(self, checklist_manager):
        """Test getting a checklist that doesn't exist."""
        checklist = checklist_manager.get_checklist("unknown_profile")
        
        assert checklist is None

    def test_clear_checklist(self, checklist_manager):
        """Test clearing a checklist."""
        profile_id = "test_profile"
        checklist_manager.create_checklist(profile_id)
        
        success = checklist_manager.clear_checklist(profile_id)
        
        assert success is True
        assert profile_id not in checklist_manager.active_checklists

    def test_clear_nonexistent_checklist(self, checklist_manager):
        """Test clearing a checklist that doesn't exist."""
        success = checklist_manager.clear_checklist("unknown_profile")
        
        assert success is False


class TestChecklistPersistence:
    """Test checklist persistence."""

    @pytest.mark.asyncio
    async def test_load_active_checklists(self, mock_repository):
        """Test loading active checklists from database."""
        # Mock database response
        mock_repository.get_active_recovery_checklists.return_value = [
            {
                "checklist_id": "checklist_1",
                "profile_id": "profile_1",
                "created_at": datetime.now(UTC).isoformat(),
                "last_updated": datetime.now(UTC).isoformat(),
                "is_complete": False,
                "completed_at": None,
                "items": [
                    {
                        "item_id": "item_1",
                        "name": "Test Item",
                        "description": "Test description",
                        "item_type": "required",
                        "is_completed": False,
                        "completed_at": None,
                    }
                ],
            }
        ]
        
        manager = RecoveryChecklistManager(repository=mock_repository)
        await manager.load_active_checklists()
        
        assert "profile_1" in manager.active_checklists
        checklist = manager.active_checklists["profile_1"]
        assert checklist.checklist_id == "checklist_1"
        assert len(checklist.items) == 1