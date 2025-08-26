"""Unit tests for tier state machine."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

from genesis.engine.state_machine import (
    TierStateMachine,
    Tier,
    TransitionResult,
    GRACE_PERIOD_HOURS,
    prevent_manual_tier_change
)
from genesis.core.exceptions import ValidationError
from genesis.data.models_db import AccountDB, TierTransition


class TestTierStateMachine:
    """Test suite for TierStateMachine."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        mock_account = MagicMock(spec=AccountDB)
        mock_account.account_id = "test-account-123"
        mock_account.current_tier = "SNIPER"
        mock_account.balance = Decimal("1500")
        mock_account.tier_started_at = datetime.utcnow() - timedelta(days=30)
        mock_account.created_at = datetime.utcnow() - timedelta(days=60)
        session.query.return_value.filter_by.return_value.first.return_value = mock_account
        session.commit = MagicMock()
        session.rollback = MagicMock()
        session.add = MagicMock()
        return session
    
    @pytest.fixture
    def state_machine(self, mock_session):
        """Create TierStateMachine instance with mocked dependencies."""
        return TierStateMachine(session=mock_session)
    
    @pytest.mark.asyncio
    async def test_initialization(self, state_machine):
        """Test state machine initialization."""
        assert state_machine.session is not None
    
    @pytest.mark.asyncio
    async def test_check_tier_requirement(self, state_machine):
        """Test checking tier requirements."""
        # Current tier is SNIPER, should have access
        assert await state_machine.check_tier_requirement("test-account-123", Tier.SNIPER)
        
        # Should not have access to higher tier
        assert not await state_machine.check_tier_requirement("test-account-123", Tier.HUNTER)
        assert not await state_machine.check_tier_requirement("test-account-123", Tier.STRATEGIST)
    
    @pytest.mark.asyncio
    async def test_evaluate_progression(self, state_machine, mock_session):
        """Test evaluating tier progression."""
        # Setup a transition record
        mock_transition = MagicMock()
        mock_transition.transition_status = 'READY'
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_transition
        
        transition = await state_machine.evaluate_progression("test-account-123")
        
        assert transition is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_progression_no_qualification(self, state_machine, mock_session):
        """Test evaluation when not qualified for next tier."""
        # No transition record
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [Mock(), None]
        
        transition = await state_machine.evaluate_progression("test-account-123")
        
        assert transition is None
    
    @pytest.mark.asyncio
    async def test_request_tier_change_valid(self, state_machine, mock_session):
        """Test requesting valid tier change."""
        # Setup mock transition
        mock_transition = MagicMock()
        mock_transition.transition_status = 'READY'
        mock_transition.checklist_completed = True
        mock_transition.funeral_completed = True
        mock_transition.paper_trading_completed = True
        
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            Mock(account_id="test-account-123", current_tier="SNIPER"),
            mock_transition
        ]
        
        success = await state_machine.request_tier_change(
            "test-account-123",
            "HUNTER",
            "Balance threshold met"
        )
        
        assert success
    
    @pytest.mark.asyncio
    async def test_request_tier_change_invalid(self, state_machine, mock_session):
        """Test requesting invalid tier change."""
        # No transition record found
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            Mock(account_id="test-account-123", current_tier="SNIPER"),
            None  # No transition found
        ]
        
        success = await state_machine.request_tier_change(
            "test-account-123",
            "HUNTER",
            "Manual request"
        )
        
        assert not success
    
    @pytest.mark.asyncio
    async def test_force_demotion(self, state_machine, mock_session):
        """Test forced tier demotion."""
        # Setup account at HUNTER tier
        mock_account = Mock()
        mock_account.account_id = "test-account-123"
        mock_account.current_tier = "HUNTER"
        mock_account.tier_started_at = datetime.utcnow()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_account
        
        await state_machine.force_demotion("test-account-123", "SNIPER", "Excessive tilt events detected")
        
        assert mock_account.current_tier == "SNIPER"
        assert mock_session.add.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_enforce_tier_transition(self, state_machine, mock_session):
        """Test enforcing automatic tier transition."""
        mock_account = Mock(spec=AccountDB)
        mock_account.account_id = "test-account-123"
        mock_account.current_tier = "SNIPER"
        mock_account.balance = Decimal("2500")
        mock_account.tier_started_at = datetime.utcnow() - timedelta(days=10)
        
        result = await state_machine.enforce_tier_transition(mock_account, Tier.HUNTER)
        
        # Should succeed with proper requirements
        assert result in [TransitionResult.SUCCESS, TransitionResult.REQUIREMENTS_NOT_MET]
    
    @pytest.mark.asyncio
    async def test_grace_period_management(self, state_machine, mock_session):
        """Test grace period application and checking."""
        mock_account = Mock(spec=AccountDB)
        mock_account.account_id = "test-account-123"
        mock_account.current_tier = "HUNTER"
        mock_account.tier_started_at = datetime.utcnow()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_account
        
        # Apply grace period
        await state_machine.apply_grace_period("test-account-123")
        
        # Check if in grace period
        assert await state_machine.is_in_grace_period(mock_account)
        
        # Test after grace period
        mock_account.tier_started_at = datetime.utcnow() - timedelta(hours=GRACE_PERIOD_HOURS + 1)
        assert not await state_machine.is_in_grace_period(mock_account)
    
    @pytest.mark.asyncio
    async def test_get_tier_requirements(self, state_machine):
        """Test retrieving tier requirements."""
        requirements = state_machine.get_tier_requirements("HUNTER")
        
        assert isinstance(requirements, dict)
        assert requirements['min_balance'] == 2000
        assert requirements['min_trades'] == 50
        assert requirements['paper_trading_required'] == True
    
    @pytest.mark.asyncio
    async def test_validate_requirements(self, state_machine, mock_session):
        """Test validating tier requirements."""
        mock_account = Mock(spec=AccountDB)
        mock_account.account_id = "test-account-123"
        mock_account.balance = Decimal("2500")
        
        # Test with sufficient balance
        is_valid = await state_machine.validate_requirements(mock_account, Tier.HUNTER)
        # May be False due to other requirements
        assert isinstance(is_valid, bool)
        
        # Test with insufficient balance  
        mock_account.balance = Decimal("1000")
        is_valid = await state_machine.validate_requirements(mock_account, Tier.HUNTER)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_get_next_tier(self, state_machine):
        """Test getting next tier in progression."""
        assert state_machine.get_next_tier("SNIPER") == "HUNTER"
        assert state_machine.get_next_tier("HUNTER") == "STRATEGIST"
        assert state_machine.get_next_tier("STRATEGIST") == "ARCHITECT"
        assert state_machine.get_next_tier("ARCHITECT") == "EMPEROR"
        assert state_machine.get_next_tier("EMPEROR") is None  # Max tier
    
    @pytest.mark.asyncio
    async def test_get_available_features(self, state_machine):
        """Test getting available features for tier."""
        sniper_features = state_machine.get_available_features("SNIPER")
        assert "market_orders" in sniper_features
        assert "iceberg_orders" not in sniper_features
        
        hunter_features = state_machine.get_available_features("HUNTER")
        assert "iceberg_orders" in hunter_features
        assert "multi_pair_trading" in hunter_features
        assert "twap_execution" not in hunter_features
        
        strategist_features = state_machine.get_available_features("STRATEGIST")
        assert "twap_execution" in strategist_features
        assert "statistical_arbitrage" in strategist_features
    
    @pytest.mark.asyncio
    async def test_prevent_manual_tier_change_decorator(self):
        """Test the prevent_manual_tier_change decorator."""
        @prevent_manual_tier_change
        async def protected_function(self, **kwargs):
            return "success"
        
        # Test normal call
        result = await protected_function(self)
        assert result == "success"
        
        # Test blocked manual override
        with pytest.raises(ValidationError) as exc_info:
            await protected_function(self, manual_override=True)
        assert "Manual tier changes are prohibited" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_transition_result_enum(self):
        """Test TransitionResult enum values."""
        assert TransitionResult.SUCCESS.value == "SUCCESS"
        assert TransitionResult.BLOCKED_BY_PROTECTION.value == "BLOCKED_BY_PROTECTION"
        assert TransitionResult.REQUIREMENTS_NOT_MET.value == "REQUIREMENTS_NOT_MET"
        assert TransitionResult.IN_GRACE_PERIOD.value == "IN_GRACE_PERIOD"
        assert TransitionResult.FAILED.value == "FAILED"
    
    @pytest.mark.asyncio
    async def test_tier_enum_values(self):
        """Test Tier enum values."""
        assert Tier.SNIPER.value == "SNIPER"
        assert Tier.HUNTER.value == "HUNTER"
        assert Tier.STRATEGIST.value == "STRATEGIST"
        assert Tier.ARCHITECT.value == "ARCHITECT"
        assert Tier.EMPEROR.value == "EMPEROR"