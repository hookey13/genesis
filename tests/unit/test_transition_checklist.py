"""Unit tests for transition checklist system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

from genesis.tilt.transition_checklist import (
    TransitionChecklist,
    ChecklistItem,
    ChecklistCategory,
    ChecklistStatus
)
from genesis.core.exceptions import ValidationError


class TestTransitionChecklist:
    """Test suite for TransitionChecklist."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.commit = MagicMock()
        session.rollback = MagicMock()
        return session
    
    @pytest.fixture
    def checklist(self, mock_session):
        """Create TransitionChecklist instance with mocked dependencies."""
        with patch('genesis.tilt.transition_checklist.get_session', return_value=mock_session):
            return TransitionChecklist(
                account_id="test-account-123",
                transition_id="trans-456",
                target_tier="HUNTER"
            )
    
    @pytest.mark.asyncio
    async def test_initialization(self, checklist):
        """Test checklist initialization."""
        assert checklist.account_id == "test-account-123"
        assert checklist.transition_id == "trans-456"
        assert checklist.target_tier == "HUNTER"
        assert len(checklist.items) == 7  # HUNTER tier has 7 checklist items
    
    @pytest.mark.asyncio
    async def test_get_checklist_items(self, checklist):
        """Test retrieval of checklist items."""
        items = await checklist.get_checklist_items()
        
        assert len(items) > 0
        assert all(isinstance(item, ChecklistItem) for item in items)
        assert all(item.status == ChecklistStatus.PENDING for item in items)
    
    @pytest.mark.asyncio
    async def test_complete_item_valid(self, checklist):
        """Test completing a checklist item with valid response."""
        items = await checklist.get_checklist_items()
        item_id = items[0].item_id
        
        response = "This is a detailed response about my trading goals and risk management strategy for the new tier."
        
        await checklist.complete_item(item_id, response)
        
        # Check item is marked completed
        updated_items = await checklist.get_checklist_items()
        completed_item = next(i for i in updated_items if i.item_id == item_id)
        assert completed_item.status == ChecklistStatus.COMPLETED
        assert completed_item.response == response
        assert completed_item.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_complete_item_short_response(self, checklist):
        """Test completing item with response too short."""
        items = await checklist.get_checklist_items()
        item_id = items[0].item_id
        
        with pytest.raises(ValidationError) as exc_info:
            await checklist.complete_item(item_id, "Too short")
        
        assert "at least 50 characters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_complete_item_invalid_id(self, checklist):
        """Test completing item with invalid ID."""
        with pytest.raises(ValidationError) as exc_info:
            await checklist.complete_item("invalid-id", "A" * 100)
        
        assert "not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_complete_item_already_completed(self, checklist):
        """Test completing already completed item."""
        items = await checklist.get_checklist_items()
        item_id = items[0].item_id
        response = "A" * 100
        
        # Complete once
        await checklist.complete_item(item_id, response)
        
        # Try to complete again
        with pytest.raises(ValidationError) as exc_info:
            await checklist.complete_item(item_id, response)
        
        assert "already completed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_is_complete_all_items(self, checklist):
        """Test checking if all items are complete."""
        # Initially not complete
        assert not await checklist.is_complete()
        
        # Complete all items
        items = await checklist.get_checklist_items()
        for item in items:
            response = f"Detailed response for {item.name} " + "x" * 50
            await checklist.complete_item(item.item_id, response)
        
        # Now should be complete
        assert await checklist.is_complete()
    
    @pytest.mark.asyncio
    async def test_get_completion_percentage(self, checklist):
        """Test calculating completion percentage."""
        assert await checklist.get_completion_percentage() == 0
        
        items = await checklist.get_checklist_items()
        total = len(items)
        
        # Complete half the items
        for i, item in enumerate(items[:total//2]):
            response = f"Response for {item.name} " + "x" * 50
            await checklist.complete_item(item.item_id, response)
        
        percentage = await checklist.get_completion_percentage()
        assert 40 < percentage < 60  # Around 50%
    
    @pytest.mark.asyncio
    async def test_reset_checklist(self, checklist):
        """Test resetting checklist."""
        # Complete some items
        items = await checklist.get_checklist_items()
        for item in items[:2]:
            await checklist.complete_item(item.item_id, "x" * 100)
        
        # Reset
        await checklist.reset()
        
        # Check all items are pending again
        reset_items = await checklist.get_checklist_items()
        assert all(item.status == ChecklistStatus.PENDING for item in reset_items)
        assert all(item.response is None for item in reset_items)
        assert all(item.completed_at is None for item in reset_items)
    
    @pytest.mark.asyncio
    async def test_tier_specific_checklists(self):
        """Test different tiers have different checklist items."""
        with patch('genesis.tilt.transition_checklist.get_session'):
            hunter_checklist = TransitionChecklist(
                account_id="test",
                transition_id="trans1",
                target_tier="HUNTER"
            )
            
            strategist_checklist = TransitionChecklist(
                account_id="test",
                transition_id="trans2", 
                target_tier="STRATEGIST"
            )
            
            hunter_items = await hunter_checklist.get_checklist_items()
            strategist_items = await strategist_checklist.get_checklist_items()
            
            # Different tiers should have different questions
            assert len(hunter_items) != len(strategist_items)
    
    @pytest.mark.asyncio
    async def test_database_persistence(self, checklist, mock_session):
        """Test database persistence of checklist completion."""
        items = await checklist.get_checklist_items()
        item_id = items[0].item_id
        response = "x" * 100
        
        await checklist.complete_item(item_id, response)
        
        # Verify database methods were called
        assert mock_session.add.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_validation_enforcement(self, checklist):
        """Test validation of responses."""
        items = await checklist.get_checklist_items()
        item_id = items[0].item_id
        
        # Test empty response
        with pytest.raises(ValidationError):
            await checklist.complete_item(item_id, "")
        
        # Test whitespace-only response
        with pytest.raises(ValidationError):
            await checklist.complete_item(item_id, "   \n\t   ")
        
        # Test response with just spaces to meet length
        with pytest.raises(ValidationError):
            await checklist.complete_item(item_id, " " * 100)
    
    @pytest.mark.asyncio
    async def test_concurrent_completion(self, checklist):
        """Test handling concurrent item completions."""
        items = await checklist.get_checklist_items()
        
        # Simulate concurrent completions
        tasks = []
        for item in items[:3]:
            response = f"Response for {item.name} " + "x" * 50
            tasks.append(checklist.complete_item(item.item_id, response))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed without conflicts
        assert all(r is None for r in results)
        
        # Verify items are completed
        updated_items = await checklist.get_checklist_items()
        completed_count = sum(1 for i in updated_items if i.status == ChecklistStatus.COMPLETED)
        assert completed_count == 3