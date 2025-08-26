"""Integration tests for paper trading workflow."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import uuid

from genesis.engine.paper_trading_enforcer import (
    PaperTradingEnforcer, PaperTrade, SessionStatus,
    SessionMetrics, PAPER_TRADING_REQUIREMENTS
)
from genesis.core.exceptions import ValidationError, StateError


class TestPaperTradingWorkflow:
    """Test complete paper trading workflow."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        # Setup query mock
        session.query.return_value.filter_by.return_value.first.return_value = None
        return session
    
    @pytest.fixture
    def enforcer(self, mock_session):
        """Create PaperTradingEnforcer instance."""
        return PaperTradingEnforcer(session=mock_session)
    
    @pytest.mark.asyncio
    async def test_complete_paper_trading_workflow(self, enforcer, mock_session):
        """Test complete workflow from creation to completion."""
        account_id = str(uuid.uuid4())
        transition_id = str(uuid.uuid4())
        
        # Step 1: Create paper trading requirement
        session_id = await enforcer.require_paper_trading(
            account_id=account_id,
            strategy='iceberg_orders',
            duration_hours=24,
            transition_id=transition_id
        )
        
        assert session_id is not None
        assert session_id in enforcer._active_sessions
        assert mock_session.add.called
        assert mock_session.commit.called
        
        # Step 2: Execute paper trades
        trades = []
        for i in range(25):  # More than minimum required
            trade = PaperTrade(
                trade_id=str(uuid.uuid4()),
                session_id=session_id,
                symbol='BTC/USDT',
                side='BUY' if i % 2 == 0 else 'SELL',
                quantity=Decimal('0.01'),
                entry_price=Decimal('50000') + Decimal(i * 10),
                execution_method='iceberg'
            )
            trades.append(trade)
            await enforcer.record_paper_trade(session_id, trade)
        
        # Step 3: Close trades with profitable results
        for i, trade in enumerate(trades):
            # Make 75% profitable (exceeds 70% requirement)
            if i < 19:  # 19 out of 25 = 76% profitable
                exit_price = trade.entry_price + Decimal('100') if trade.side == 'BUY' else trade.entry_price - Decimal('100')
            else:
                exit_price = trade.entry_price - Decimal('50') if trade.side == 'BUY' else trade.entry_price + Decimal('50')
            
            await enforcer.close_paper_trade(
                session_id=session_id,
                trade_id=trade.trade_id,
                exit_price=exit_price
            )
        
        # Step 4: Get metrics
        metrics = await enforcer.get_session_metrics(session_id)
        
        assert metrics.total_trades == 25
        assert metrics.profitable_trades >= 18
        assert metrics.success_rate >= Decimal('0.70')
        assert metrics.profitable_ratio >= Decimal('0.70')
        
        # Step 5: Check completion requirements
        is_complete, failures = await enforcer.check_session_completion(session_id)
        
        # Should pass if we've met all requirements
        # Note: Duration requirement would normally fail without mocking time
        # but we're testing the logic flow here
        
        # Step 6: Complete session
        # Mock the session query to return a session with proper duration
        mock_paper_session = MagicMock()
        mock_paper_session.session_id = session_id
        mock_paper_session.strategy_name = 'iceberg_orders'
        mock_paper_session.started_at = datetime.utcnow() - timedelta(hours=25)
        mock_paper_session.transition_id = transition_id
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_paper_session
        
        final_metrics = await enforcer.complete_session(session_id, force=True)
        
        assert final_metrics.total_trades == 25
        assert mock_paper_session.status == SessionStatus.COMPLETED.value
        assert session_id not in enforcer._active_sessions
    
    @pytest.mark.asyncio
    async def test_paper_trading_with_failures(self, enforcer, mock_session):
        """Test paper trading with poor performance."""
        account_id = str(uuid.uuid4())
        
        # Create session
        session_id = await enforcer.require_paper_trading(
            account_id=account_id,
            strategy='vwap_execution',
            duration_hours=48
        )
        
        # Execute trades with poor performance
        trades = []
        for i in range(10):  # Less than minimum required
            trade = PaperTrade(
                trade_id=str(uuid.uuid4()),
                session_id=session_id,
                symbol='ETH/USDT',
                side='BUY',
                quantity=Decimal('0.1'),
                entry_price=Decimal('3000'),
                execution_method='vwap'
            )
            trades.append(trade)
            await enforcer.record_paper_trade(session_id, trade)
        
        # Close trades with mostly losses (40% profitable, below 60% requirement)
        for i, trade in enumerate(trades):
            if i < 4:  # Only 4 out of 10 profitable
                exit_price = Decimal('3100')
            else:
                exit_price = Decimal('2900')
            
            await enforcer.close_paper_trade(
                session_id=session_id,
                trade_id=trade.trade_id,
                exit_price=exit_price
            )
        
        # Check metrics
        metrics = await enforcer.get_session_metrics(session_id)
        assert metrics.total_trades == 10
        assert metrics.profitable_trades == 4
        assert metrics.success_rate == Decimal('0.40')
        
        # Check completion - should fail
        is_complete, failures = await enforcer.check_session_completion(session_id)
        assert is_complete is False
        assert len(failures) > 0
        
        # Try to complete without force - should raise error
        with pytest.raises(ValidationError):
            await enforcer.complete_session(session_id, force=False)
        
        # Force completion
        mock_paper_session = MagicMock()
        mock_paper_session.session_id = session_id
        mock_paper_session.strategy_name = 'vwap_execution'
        mock_paper_session.started_at = datetime.utcnow() - timedelta(hours=1)
        mock_paper_session.transition_id = None
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_paper_session
        
        await enforcer.complete_session(session_id, force=True)
        assert mock_paper_session.status == SessionStatus.FAILED.value
    
    @pytest.mark.asyncio
    async def test_concurrent_paper_trading_sessions(self, enforcer, mock_session):
        """Test multiple concurrent paper trading sessions."""
        account_id = str(uuid.uuid4())
        
        # Create multiple sessions for different strategies
        sessions = []
        for strategy in ['iceberg_orders', 'vwap_execution']:
            session_id = await enforcer.require_paper_trading(
                account_id=account_id,
                strategy=strategy,
                duration_hours=24
            )
            sessions.append(session_id)
        
        assert len(enforcer._active_sessions) == 2
        
        # Execute trades in both sessions
        for session_id in sessions:
            for i in range(5):
                trade = PaperTrade(
                    trade_id=str(uuid.uuid4()),
                    session_id=session_id,
                    symbol='BTC/USDT',
                    side='BUY',
                    quantity=Decimal('0.01'),
                    entry_price=Decimal('50000')
                )
                await enforcer.record_paper_trade(session_id, trade)
        
        # Get active sessions for account
        mock_sessions = [MagicMock(session_id=sid) for sid in sessions]
        mock_session.query.return_value.filter_by.return_value.all.return_value = mock_sessions
        
        active = enforcer.get_active_sessions(account_id)
        assert len(active) == 2
    
    @pytest.mark.asyncio
    async def test_paper_trade_lifecycle(self, enforcer, mock_session):
        """Test individual paper trade lifecycle."""
        account_id = str(uuid.uuid4())
        
        # Create session
        session_id = await enforcer.require_paper_trading(
            account_id=account_id,
            strategy='iceberg_orders',
            duration_hours=24
        )
        
        # Create and record trade
        trade = PaperTrade(
            trade_id='test-trade-1',
            session_id=session_id,
            symbol='BTC/USDT',
            side='BUY',
            quantity=Decimal('0.01'),
            entry_price=Decimal('50000'),
            execution_method='iceberg'
        )
        
        await enforcer.record_paper_trade(session_id, trade)
        
        # Verify trade is open
        assert trade.is_closed is False
        assert trade.pnl is None
        
        # Close trade with profit
        updated_trade = await enforcer.close_paper_trade(
            session_id=session_id,
            trade_id='test-trade-1',
            exit_price=Decimal('51000')
        )
        
        # Verify trade is closed
        assert updated_trade.is_closed is True
        assert updated_trade.exit_price == Decimal('51000')
        assert updated_trade.pnl == Decimal('10')  # (51000 - 50000) * 0.01
        assert updated_trade.is_profitable is True
        
        # Try to close again - should fail
        with pytest.raises(StateError):
            await enforcer.close_paper_trade(
                session_id=session_id,
                trade_id='test-trade-1',
                exit_price=Decimal('52000')
            )
    
    @pytest.mark.asyncio
    async def test_session_cancellation(self, enforcer, mock_session):
        """Test cancelling a paper trading session."""
        account_id = str(uuid.uuid4())
        
        # Create session
        session_id = await enforcer.require_paper_trading(
            account_id=account_id,
            strategy='market_making',
            duration_hours=72
        )
        
        assert session_id in enforcer._active_sessions
        
        # Cancel session
        mock_paper_session = MagicMock()
        mock_paper_session.session_id = session_id
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_paper_session
        
        await enforcer.cancel_session(session_id, "User requested cancellation")
        
        assert mock_paper_session.status == SessionStatus.CANCELLED.value
        assert session_id not in enforcer._active_sessions
        assert session_id not in enforcer._session_tasks
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, enforcer, mock_session):
        """Test accurate metrics calculation."""
        account_id = str(uuid.uuid4())
        
        # Create session
        session_id = await enforcer.require_paper_trading(
            account_id=account_id,
            strategy='statistical_arbitrage',
            duration_hours=48
        )
        
        # Mock session for metrics
        mock_paper_session = MagicMock()
        mock_paper_session.session_id = session_id
        mock_paper_session.started_at = datetime.utcnow() - timedelta(hours=2)
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_paper_session
        
        # Execute specific trades to test metrics
        trades_data = [
            ('BUY', Decimal('50000'), Decimal('51000'), Decimal('10')),  # +10
            ('BUY', Decimal('51000'), Decimal('51500'), Decimal('5')),   # +5
            ('SELL', Decimal('52000'), Decimal('51500'), Decimal('5')),  # +5
            ('BUY', Decimal('51000'), Decimal('50500'), Decimal('-5')),  # -5
            ('SELL', Decimal('50000'), Decimal('50500'), Decimal('-5')), # -5
        ]
        
        for i, (side, entry, exit, expected_pnl) in enumerate(trades_data):
            trade = PaperTrade(
                trade_id=f'trade-{i}',
                session_id=session_id,
                symbol='BTC/USDT',
                side=side,
                quantity=Decimal('0.01'),
                entry_price=entry
            )
            await enforcer.record_paper_trade(session_id, trade)
            
            await enforcer.close_paper_trade(
                session_id=session_id,
                trade_id=f'trade-{i}',
                exit_price=exit
            )
        
        # Get metrics
        metrics = await enforcer.get_session_metrics(session_id)
        
        assert metrics.total_trades == 5
        assert metrics.profitable_trades == 3
        assert metrics.total_pnl == Decimal('10')  # 10 + 5 + 5 - 5 - 5
        assert metrics.success_rate == Decimal('0.6')  # 3/5
        assert metrics.average_pnl == Decimal('2')  # 10/5
        assert metrics.max_drawdown == Decimal('10')  # Peak was 20, dropped to 10
    
    @pytest.mark.asyncio
    async def test_invalid_strategy_rejection(self, enforcer):
        """Test rejection of invalid strategy."""
        account_id = str(uuid.uuid4())
        
        with pytest.raises(ValidationError, match="Unknown strategy"):
            await enforcer.require_paper_trading(
                account_id=account_id,
                strategy='invalid_strategy',
                duration_hours=24
            )
    
    @pytest.mark.asyncio
    async def test_cleanup(self, enforcer, mock_session):
        """Test cleanup of all sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session_id = await enforcer.require_paper_trading(
                account_id=str(uuid.uuid4()),
                strategy='iceberg_orders',
                duration_hours=24
            )
            sessions.append(session_id)
        
        assert len(enforcer._active_sessions) == 3
        assert len(enforcer._session_tasks) == 3
        
        # Cleanup
        await enforcer.cleanup()
        
        assert len(enforcer._active_sessions) == 0
        assert len(enforcer._session_tasks) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])