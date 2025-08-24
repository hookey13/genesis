"""
Integration tests for the complete risk management flow.

Tests the full integration of risk engine, account manager,
database persistence, and exchange gateway interactions.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock

from genesis.engine.risk_engine import RiskEngine
from genesis.core.account_manager import AccountManager
from genesis.core.models import (
    Account, Position, TradingSession, TradingTier, PositionSide
)
from genesis.core.exceptions import (
    RiskLimitExceeded, DailyLossLimitReached, MinimumPositionSize
)
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.mock_exchange import MockExchange


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        repo = SQLiteRepository(str(db_path))
        await repo.initialize()
        yield repo
        await repo.shutdown()


@pytest.fixture
def mock_exchange():
    """Create a mock exchange for testing."""
    exchange = MockExchange()
    exchange.set_balance("USDT", Decimal("1000"))
    return exchange


@pytest.fixture
async def gateway(mock_exchange):
    """Create a gateway with mock exchange."""
    gateway = BinanceGateway(
        api_key="test_key",
        api_secret="test_secret",
        testnet=True
    )
    # Replace exchange with mock
    gateway.exchange = mock_exchange
    await gateway.initialize()
    yield gateway
    await gateway.shutdown()


@pytest.fixture
async def integrated_system(temp_db, gateway):
    """Create a fully integrated risk management system."""
    # Create account
    account = Account(
        account_id="integration-test",
        balance_usdt=Decimal("1000"),
        tier=TradingTier.SNIPER
    )
    await temp_db.create_account(account)
    
    # Create account manager
    account_manager = AccountManager(
        gateway=gateway,
        account=account,
        auto_sync=False
    )
    
    # Create trading session
    session = TradingSession(
        session_id="session-test",
        account_id=account.account_id,
        session_date=datetime.now(),
        starting_balance=account.balance_usdt,
        current_balance=account.balance_usdt,
        daily_loss_limit=Decimal("25")
    )
    await temp_db.create_session(session)
    
    # Create risk engine
    risk_engine = RiskEngine(account, session)
    
    return {
        "repo": temp_db,
        "gateway": gateway,
        "account_manager": account_manager,
        "risk_engine": risk_engine,
        "account": account,
        "session": session
    }


class TestFullRiskFlow:
    """Test complete risk management workflow."""
    
    @pytest.mark.asyncio
    async def test_position_creation_flow(self, integrated_system):
        """Test full flow of creating a position with risk checks."""
        risk_engine = integrated_system["risk_engine"]
        repo = integrated_system["repo"]
        
        # Calculate position size
        symbol = "BTC/USDT"
        entry_price = Decimal("50000")
        
        position_size = risk_engine.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price
        )
        
        # Validate order risk
        risk_engine.validate_order_risk(
            symbol=symbol,
            side=PositionSide.LONG,
            quantity=position_size,
            entry_price=entry_price
        )
        
        # Create position
        stop_loss = risk_engine.calculate_stop_loss(entry_price, PositionSide.LONG)
        position = Position(
            account_id=integrated_system["account"].account_id,
            symbol=symbol,
            side=PositionSide.LONG,
            entry_price=entry_price,
            quantity=position_size,
            dollar_value=position_size * entry_price,
            stop_loss=stop_loss
        )
        
        # Save to database
        position_id = await repo.create_position(position)
        
        # Add to risk engine
        risk_engine.add_position(position)
        
        # Verify position was created
        saved_position = await repo.get_position(position_id)
        assert saved_position is not None
        assert saved_position.symbol == symbol
        assert saved_position.quantity == position_size
    
    @pytest.mark.asyncio
    async def test_account_sync_flow(self, integrated_system):
        """Test account balance synchronization flow."""
        account_manager = integrated_system["account_manager"]
        repo = integrated_system["repo"]
        mock_exchange = integrated_system["gateway"].exchange
        
        # Update exchange balance
        mock_exchange.set_balance("USDT", Decimal("1500"))
        
        # Sync balance
        new_balance = await account_manager.sync_balance()
        assert new_balance == Decimal("1500")
        
        # Save to database
        await repo.update_account(account_manager.account)
        
        # Verify persistence
        saved_account = await repo.get_account(account_manager.account.account_id)
        assert saved_account.balance_usdt == Decimal("1500")
    
    @pytest.mark.asyncio
    async def test_pnl_update_flow(self, integrated_system):
        """Test P&L calculation and update flow."""
        risk_engine = integrated_system["risk_engine"]
        repo = integrated_system["repo"]
        
        # Create a position
        position = Position(
            account_id=integrated_system["account"].account_id,
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("3000"),
            quantity=Decimal("0.1"),
            dollar_value=Decimal("300")
        )
        
        position_id = await repo.create_position(position)
        risk_engine.add_position(position)
        
        # Update with new price
        new_price = Decimal("3150")  # 5% gain
        position.update_pnl(new_price)
        
        # Save updated P&L
        await repo.update_position(position)
        
        # Verify P&L calculation
        assert position.pnl_dollars == Decimal("15")  # 0.1 * 150
        assert position.pnl_percent == Decimal("5")
        
        # Verify persistence
        saved_position = await repo.get_position(position_id)
        assert saved_position.pnl_dollars == Decimal("15")
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit_flow(self, integrated_system):
        """Test daily loss limit enforcement flow."""
        risk_engine = integrated_system["risk_engine"]
        session = integrated_system["session"]
        repo = integrated_system["repo"]
        
        # Simulate losses approaching limit
        session.realized_pnl = Decimal("-20")
        await repo.update_session(session)
        
        # Should still allow trades
        risk_engine.validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000")
        )
        
        # Hit the daily limit
        session.realized_pnl = Decimal("-25")
        risk_engine.session.realized_pnl = Decimal("-25")
        await repo.update_session(session)
        
        # Should block new trades
        with pytest.raises(DailyLossLimitReached):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000")
            )
    
    @pytest.mark.asyncio
    async def test_position_close_flow(self, integrated_system):
        """Test position closing flow with P&L update."""
        risk_engine = integrated_system["risk_engine"]
        session = integrated_system["session"]
        repo = integrated_system["repo"]
        account_manager = integrated_system["account_manager"]
        
        # Create and track position
        position = Position(
            account_id=integrated_system["account"].account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            dollar_value=Decimal("500")
        )
        
        position_id = await repo.create_position(position)
        risk_engine.add_position(position)
        
        # Simulate closing with profit
        closing_price = Decimal("51000")  # 2% gain
        position.update_pnl(closing_price)
        final_pnl = position.pnl_dollars
        
        # Update session with trade result
        session.update_trade_result(final_pnl)
        await repo.update_session(session)
        
        # Close position in database
        await repo.close_position(position_id, final_pnl)
        
        # Remove from risk engine
        risk_engine.remove_position(position_id)
        
        # Update account balance
        account_manager.add_balance(final_pnl)
        await repo.update_account(account_manager.account)
        
        # Verify session was updated
        assert session.total_trades == 1
        assert session.winning_trades == 1
        assert session.realized_pnl == final_pnl
        
        # Verify position is closed
        closed_position = await repo.get_position(position_id)
        assert closed_position is not None  # Should still exist in DB
        
        # Verify account balance updated
        assert account_manager.account.balance_usdt == Decimal("1000") + final_pnl


class TestMultiPositionScenarios:
    """Test scenarios with multiple positions."""
    
    @pytest.mark.asyncio
    async def test_max_positions_enforcement(self, integrated_system):
        """Test maximum positions limit for tier."""
        risk_engine = integrated_system["risk_engine"]
        repo = integrated_system["repo"]
        
        # Sniper tier allows only 1 position
        position = Position(
            account_id=integrated_system["account"].account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001"),
            dollar_value=Decimal("50")
        )
        
        await repo.create_position(position)
        risk_engine.add_position(position)
        
        # Try to open second position
        with pytest.raises(RiskLimitExceeded) as exc_info:
            risk_engine.validate_order_risk(
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.01"),
                entry_price=Decimal("3000")
            )
        
        assert exc_info.value.limit_type == "max_positions"
    
    @pytest.mark.asyncio
    async def test_total_exposure_tracking(self, integrated_system):
        """Test total exposure across positions."""
        # Upgrade to Hunter tier for multiple positions
        integrated_system["account"].tier = TradingTier.HUNTER
        integrated_system["risk_engine"].tier_limits = RiskEngine.TIER_LIMITS[TradingTier.HUNTER]
        
        risk_engine = integrated_system["risk_engine"]
        repo = integrated_system["repo"]
        
        # Create multiple positions
        positions = [
            Position(
                account_id=integrated_system["account"].account_id,
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.002"),
                dollar_value=Decimal("100")
            ),
            Position(
                account_id=integrated_system["account"].account_id,
                symbol="ETH/USDT",
                side=PositionSide.SHORT,
                entry_price=Decimal("3000"),
                quantity=Decimal("0.05"),
                dollar_value=Decimal("150")
            )
        ]
        
        for position in positions:
            await repo.create_position(position)
            risk_engine.add_position(position)
        
        # Check total exposure
        total_exposure = risk_engine.get_total_exposure()
        assert total_exposure == Decimal("250.00")
        
        # Update prices and check total P&L
        price_updates = {
            "BTC/USDT": Decimal("51000"),  # 2% gain
            "ETH/USDT": Decimal("2900")     # 3.33% gain (short)
        }
        
        risk_engine.update_all_pnl(price_updates)
        
        total_pnl = risk_engine.get_total_pnl()
        assert total_pnl["total_pnl_dollars"] == Decimal("7.00")  # 2 + 5
    
    @pytest.mark.asyncio
    async def test_position_correlation_hunter_tier(self, integrated_system):
        """Test position correlation calculation for Hunter tier."""
        # Upgrade to Hunter tier
        integrated_system["account"].tier = TradingTier.HUNTER
        risk_engine = RiskEngine(
            integrated_system["account"],
            integrated_system["session"]
        )
        
        # Create correlated positions
        positions = [
            Position(
                position_id="pos-btc",
                account_id=integrated_system["account"].account_id,
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.001"),
                dollar_value=Decimal("50")
            ),
            Position(
                position_id="pos-btc2",
                account_id=integrated_system["account"].account_id,
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50100"),
                quantity=Decimal("0.001"),
                dollar_value=Decimal("50.1")
            )
        ]
        
        for position in positions:
            risk_engine.add_position(position)
        
        # Calculate correlations
        correlations = await risk_engine.calculate_position_correlations()
        
        assert len(correlations) == 1
        # Same symbol should have high correlation
        assert correlations[0][2] == Decimal("0.8")


class TestSessionManagement:
    """Test trading session management."""
    
    @pytest.mark.asyncio
    async def test_session_reset_at_midnight(self, integrated_system):
        """Test session reset at UTC midnight."""
        repo = integrated_system["repo"]
        session = integrated_system["session"]
        
        # Simulate end of day
        session.is_active = False
        await repo.end_session(session.session_id)
        
        # Create new session for next day
        new_session = TradingSession(
            account_id=integrated_system["account"].account_id,
            session_date=datetime.now() + timedelta(days=1),
            starting_balance=integrated_system["account"].balance_usdt,
            current_balance=integrated_system["account"].balance_usdt,
            daily_loss_limit=Decimal("25")
        )
        
        await repo.create_session(new_session)
        
        # Verify old session is ended
        old_session = await repo.get_session(session.session_id)
        assert not old_session.is_active
        
        # Verify new session is active
        active_session = await repo.get_active_session(integrated_system["account"].account_id)
        assert active_session.session_id == new_session.session_id
        assert active_session.is_active
    
    @pytest.mark.asyncio
    async def test_session_statistics_tracking(self, integrated_system):
        """Test tracking of session statistics."""
        session = integrated_system["session"]
        repo = integrated_system["repo"]
        
        # Simulate multiple trades
        trades = [
            Decimal("10"),   # Win
            Decimal("-5"),   # Loss
            Decimal("15"),   # Win
            Decimal("-8"),   # Loss
            Decimal("20"),   # Win
        ]
        
        for pnl in trades:
            session.update_trade_result(pnl)
        
        await repo.update_session(session)
        
        # Verify statistics
        assert session.total_trades == 5
        assert session.winning_trades == 3
        assert session.losing_trades == 2
        assert session.realized_pnl == Decimal("32")  # Sum of all trades
        assert session.max_drawdown == Decimal("0")  # Never went below starting


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, integrated_system):
        """Test database transaction rollback on error."""
        repo = integrated_system["repo"]
        
        await repo.begin_transaction()
        
        try:
            # Create position
            position = Position(
                account_id=integrated_system["account"].account_id,
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.01"),
                dollar_value=Decimal("500")
            )
            await repo.create_position(position)
            
            # Simulate error
            raise Exception("Simulated error")
            
        except Exception:
            await repo.rollback_transaction()
        
        # Position should not exist
        positions = await repo.get_positions_by_account(integrated_system["account"].account_id)
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_exchange_connection_failure_recovery(self, integrated_system):
        """Test recovery from exchange connection failure."""
        account_manager = integrated_system["account_manager"]
        
        # Simulate connection failure
        account_manager.gateway.get_account_info = AsyncMock(
            side_effect=Exception("Connection lost")
        )
        
        # Sync should fail but handle gracefully
        with pytest.raises(Exception):
            await account_manager.sync_balance()
        
        # Verify error was recorded
        assert account_manager._last_sync_error == "Connection lost"
        assert not account_manager.is_sync_healthy()
        
        # Restore connection
        account_manager.gateway.get_account_info = AsyncMock(
            return_value={
                "balances": [
                    {"asset": "USDT", "free": "1000", "locked": "0"}
                ]
            }
        )
        
        # Sync should work again
        balance = await account_manager.sync_balance()
        assert balance == Decimal("1000")
        assert account_manager.is_sync_healthy()
    
    @pytest.mark.asyncio
    async def test_position_recovery_from_database(self, integrated_system):
        """Test position recovery from database after restart."""
        repo = integrated_system["repo"]
        original_risk_engine = integrated_system["risk_engine"]
        
        # Create positions
        positions = [
            Position(
                position_id=f"pos-{i}",
                account_id=integrated_system["account"].account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("1"),
                dollar_value=Decimal("100")
            )
            for i in range(3)
        ]
        
        for position in positions:
            await repo.create_position(position)
            original_risk_engine.add_position(position)
        
        # Simulate restart - create new risk engine
        new_risk_engine = RiskEngine(
            integrated_system["account"],
            integrated_system["session"]
        )
        
        # Recover positions from database
        saved_positions = await repo.get_positions_by_account(
            integrated_system["account"].account_id,
            status="OPEN"
        )
        
        for position in saved_positions:
            new_risk_engine.add_position(position)
        
        # Verify all positions recovered
        assert len(new_risk_engine.positions) == 3
        assert all(f"pos-{i}" in new_risk_engine.positions for i in range(3))


class TestPerformance:
    """Test performance requirements."""
    
    @pytest.mark.asyncio
    async def test_risk_calculation_performance(self, integrated_system):
        """Test that risk calculations complete within 10ms."""
        import time
        risk_engine = integrated_system["risk_engine"]
        
        # Measure position size calculation
        start = time.perf_counter()
        risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=Decimal("50000")
        )
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        
        assert elapsed < 10, f"Position size calculation took {elapsed:.2f}ms"
        
        # Measure risk validation
        start = time.perf_counter()
        risk_engine.validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000")
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 10, f"Risk validation took {elapsed:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self, integrated_system):
        """Test database query performance."""
        import time
        repo = integrated_system["repo"]
        
        # Create multiple positions for testing
        for i in range(100):
            position = Position(
                position_id=f"perf-pos-{i}",
                account_id=integrated_system["account"].account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("1"),
                dollar_value=Decimal("100")
            )
            await repo.create_position(position)
        
        # Measure query performance
        start = time.perf_counter()
        positions = await repo.get_positions_by_account(
            integrated_system["account"].account_id
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        assert len(positions) == 100
        assert elapsed < 100, f"Query took {elapsed:.2f}ms for 100 positions"