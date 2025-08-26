"""Unit tests for paper trading enforcer."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import uuid

from genesis.engine.paper_trading_enforcer import (
    PaperTradingEnforcer,
    SessionStatus,
    TradingMethod,
    PaperTrade,
    SessionMetrics
)
from genesis.core.exceptions import ValidationError, StateError


class TestPaperTradingEnforcer:
    """Test suite for PaperTradingEnforcer."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.commit = MagicMock()
        session.rollback = MagicMock()
        session.add = MagicMock()
        return session
    
    @pytest.fixture
    def enforcer(self, mock_session):
        """Create PaperTradingEnforcer instance with mocked dependencies."""
        with patch('genesis.engine.paper_trading_enforcer.get_session', return_value=mock_session):
            return PaperTradingEnforcer(account_id="test-account-123")
    
    @pytest.mark.asyncio
    async def test_initialization(self, enforcer):
        """Test enforcer initialization."""
        assert enforcer.account_id == "test-account-123"
        assert enforcer.session is not None
        assert enforcer.active_sessions == {}
    
    @pytest.mark.asyncio
    async def test_require_paper_trading(self, enforcer):
        """Test requiring paper trading for new strategy."""
        session_id = await enforcer.require_paper_trading(
            strategy="iceberg_orders",
            duration_hours=24
        )
        
        assert session_id is not None
        assert session_id in enforcer.active_sessions
        
        session = enforcer.active_sessions[session_id]
        assert session.strategy_name == "iceberg_orders"
        assert session.required_duration_hours == 24
        assert session.status == SessionStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_require_paper_trading_duplicate(self, enforcer):
        """Test requiring paper trading for already active strategy."""
        await enforcer.require_paper_trading("iceberg_orders", 24)
        
        # Try to require same strategy again
        with pytest.raises(StateError) as exc_info:
            await enforcer.require_paper_trading("iceberg_orders", 24)
        
        assert "already active" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_record_paper_trade(self, enforcer):
        """Test recording paper trades."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 24)
        
        # Record successful trade
        await enforcer.record_paper_trade(
            session_id=session_id,
            order_type="iceberg",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            pnl=Decimal("100"),
            success=True
        )
        
        session = enforcer.active_sessions[session_id]
        assert session.total_trades == 1
        assert session.profitable_trades == 1
        assert session.success_rate == Decimal("100")
    
    @pytest.mark.asyncio
    async def test_record_paper_trade_invalid_session(self, enforcer):
        """Test recording trade for invalid session."""
        with pytest.raises(ValidationError) as exc_info:
            await enforcer.record_paper_trade(
                session_id="invalid-session",
                order_type="market",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                pnl=Decimal("100"),
                success=True
            )
        
        assert "Session not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_check_completion_success(self, enforcer):
        """Test checking completion with successful performance."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 1)  # 1 hour for testing
        
        # Record enough successful trades
        for i in range(10):
            success = i < 8  # 80% success rate
            await enforcer.record_paper_trade(
                session_id=session_id,
                order_type="iceberg",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if success else Decimal("49000"),
                pnl=Decimal("100") if success else Decimal("-100"),
                success=success
            )
        
        # Simulate time passing
        session = enforcer.active_sessions[session_id]
        session.started_at = datetime.utcnow() - timedelta(hours=2)
        
        is_complete = await enforcer.check_completion(session_id)
        assert is_complete
        assert session.status == SessionStatus.COMPLETED
        assert session.success_rate >= Decimal("70")  # Minimum required
    
    @pytest.mark.asyncio
    async def test_check_completion_insufficient_success_rate(self, enforcer):
        """Test checking completion with low success rate."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 1)
        
        # Record mostly failed trades
        for i in range(10):
            success = i < 3  # 30% success rate
            await enforcer.record_paper_trade(
                session_id=session_id,
                order_type="iceberg",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if success else Decimal("49000"),
                pnl=Decimal("100") if success else Decimal("-100"),
                success=success
            )
        
        # Simulate time passing
        session = enforcer.active_sessions[session_id]
        session.started_at = datetime.utcnow() - timedelta(hours=2)
        
        is_complete = await enforcer.check_completion(session_id)
        assert not is_complete  # Failed due to low success rate
        assert session.status == SessionStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_check_completion_insufficient_duration(self, enforcer):
        """Test checking completion before duration met."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 24)
        
        # Record successful trades
        for _ in range(5):
            await enforcer.record_paper_trade(
                session_id=session_id,
                order_type="iceberg",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                pnl=Decimal("100"),
                success=True
            )
        
        # Don't simulate time passing - just started
        is_complete = await enforcer.check_completion(session_id)
        assert not is_complete  # Not enough time passed
    
    @pytest.mark.asyncio
    async def test_is_strategy_approved(self, enforcer):
        """Test checking if strategy is approved."""
        # Initially not approved
        assert not await enforcer.is_strategy_approved("iceberg_orders")
        
        # Start and complete paper trading
        session_id = await enforcer.require_paper_trading("iceberg_orders", 1)
        
        # Complete successfully
        session = enforcer.active_sessions[session_id]
        session.status = SessionStatus.COMPLETED
        session.success_rate = Decimal("75")
        
        # Now should be approved
        assert await enforcer.is_strategy_approved("iceberg_orders")
    
    @pytest.mark.asyncio
    async def test_get_session_metrics(self, enforcer):
        """Test retrieving session metrics."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 24)
        
        # Record some trades
        for i in range(5):
            await enforcer.record_paper_trade(
                session_id=session_id,
                order_type="iceberg",
                size=Decimal("0.1"),
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000") if i < 3 else Decimal("49000"),
                pnl=Decimal("100") if i < 3 else Decimal("-100"),
                success=i < 3
            )
        
        metrics = await enforcer.get_session_metrics(session_id)
        
        assert isinstance(metrics, SessionMetrics)
        assert metrics.total_trades == 5
        assert metrics.profitable_trades == 3
        assert metrics.success_rate == Decimal("60")
        assert metrics.total_pnl == Decimal("100")  # 3*100 - 2*100
    
    @pytest.mark.asyncio
    async def test_cancel_session(self, enforcer):
        """Test cancelling paper trading session."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 24)
        
        await enforcer.cancel_session(session_id)
        
        assert session_id not in enforcer.active_sessions
    
    @pytest.mark.asyncio
    async def test_extend_session(self, enforcer):
        """Test extending paper trading session."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 24)
        
        original_duration = enforcer.active_sessions[session_id].required_duration_hours
        
        await enforcer.extend_session(session_id, additional_hours=12)
        
        new_duration = enforcer.active_sessions[session_id].required_duration_hours
        assert new_duration == original_duration + 12
    
    @pytest.mark.asyncio
    async def test_database_persistence(self, enforcer, mock_session):
        """Test database persistence of sessions."""
        await enforcer.require_paper_trading("iceberg_orders", 24)
        
        # Verify database methods were called
        assert mock_session.add.called
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_minimum_trades_requirement(self, enforcer):
        """Test minimum trades requirement for completion."""
        session_id = await enforcer.require_paper_trading("iceberg_orders", 1)
        
        # Record only 1 successful trade
        await enforcer.record_paper_trade(
            session_id=session_id,
            order_type="iceberg",
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            pnl=Decimal("100"),
            success=True
        )
        
        # Simulate time passing
        session = enforcer.active_sessions[session_id]
        session.started_at = datetime.utcnow() - timedelta(hours=2)
        
        is_complete = await enforcer.check_completion(session_id)
        assert not is_complete  # Not enough trades (minimum 10 required)
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, enforcer):
        """Test managing multiple concurrent sessions."""
        # Start multiple sessions for different strategies
        sessions = []
        strategies = ["iceberg_orders", "vwap_execution", "twap_execution"]
        
        for strategy in strategies:
            session_id = await enforcer.require_paper_trading(strategy, 24)
            sessions.append(session_id)
        
        assert len(enforcer.active_sessions) == 3
        
        # Each should be independent
        for session_id in sessions:
            session = enforcer.active_sessions[session_id]
            assert session.status == SessionStatus.ACTIVE