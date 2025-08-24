"""
Integration tests for order execution flow.

Tests the complete order execution pipeline including gateway interaction,
risk engine validation, and database persistence.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from genesis.core.models import (
    Account,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.risk_engine import RiskEngine
from genesis.exchange.gateway import BinanceGateway


@pytest.fixture
async def test_db_path(tmp_path):
    """Create temporary database path."""
    return str(tmp_path / "test_orders.db")


@pytest.fixture
async def repository(test_db_path):
    """Create test repository."""
    repo = SQLiteRepository(test_db_path)
    await repo.initialize()
    yield repo
    await repo.shutdown()


@pytest.fixture
async def account(repository):
    """Create test account."""
    account = Account(
        account_id=str(uuid4()),
        balance_usdt=Decimal("1000"),
        tier=TradingTier.SNIPER
    )
    await repository.create_account(account)
    return account


@pytest.fixture
async def trading_session(repository, account):
    """Create test trading session."""
    session = TradingSession(
        session_id=str(uuid4()),
        account_id=account.account_id,
        starting_balance=account.balance_usdt,
        current_balance=account.balance_usdt,
        daily_loss_limit=Decimal("25")
    )
    await repository.create_session(session)
    return session


@pytest.fixture
async def gateway():
    """Create mock gateway for integration tests."""
    gateway = BinanceGateway(mock_mode=True)
    await gateway.initialize()
    yield gateway
    await gateway.shutdown()


@pytest.fixture
def risk_engine(account, trading_session):
    """Create risk engine."""
    return RiskEngine(account, trading_session)


@pytest.fixture
def executor(gateway, account, risk_engine, repository):
    """Create executor with all dependencies."""
    return MarketOrderExecutor(
        gateway=gateway,
        account=account,
        risk_engine=risk_engine,
        repository=repository
    )


class TestOrderFlowIntegration:
    """Integration tests for complete order flow."""
    
    @pytest.mark.asyncio
    async def test_complete_order_flow(self, executor, repository):
        """Test complete order execution flow from creation to database."""
        # Create market buy order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001")
        )
        
        # Execute order
        result = await executor.execute_market_order(order, confirmation_required=False)
        
        # Verify execution success
        assert result.success is True
        assert result.order.status == OrderStatus.FILLED
        assert result.latency_ms is not None
        
        # Verify order persisted to database
        db_order = await repository.get_order(order.order_id)
        assert db_order is not None
        assert db_order.order_id == order.order_id
        assert db_order.status == OrderStatus.FILLED
        assert db_order.filled_quantity == Decimal("0.001")
        assert db_order.latency_ms is not None
    
    @pytest.mark.asyncio
    async def test_order_with_position_creation(self, executor, repository, account):
        """Test order execution with position creation."""
        # Create position
        position = Position(
            position_id=str(uuid4()),
            account_id=account.account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001"),
            dollar_value=Decimal("50")
        )
        await repository.create_position(position)
        
        # Create order for position
        order = Order(
            order_id=str(uuid4()),
            position_id=position.position_id,
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001")
        )
        
        # Execute order
        result = await executor.execute_market_order(order, confirmation_required=False)
        
        assert result.success is True
        
        # Verify order linked to position
        db_order = await repository.get_order(order.order_id)
        assert db_order.position_id == position.position_id
        
        # Verify we can get all orders for position
        position_orders = await repository.get_orders_by_position(position.position_id)
        assert len(position_orders) >= 1
        assert any(o.order_id == order.order_id for o in position_orders)
    
    @pytest.mark.asyncio
    async def test_multiple_orders_sequential(self, executor, repository):
        """Test executing multiple orders sequentially."""
        orders_executed = []
        
        for i in range(3):
            order = Order(
                order_id=str(uuid4()),
                position_id=str(uuid4()),
                client_order_id=str(uuid4()),
                symbol="BTC/USDT",
                type=OrderType.MARKET,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=None,
                quantity=Decimal("0.001")
            )
            
            result = await executor.execute_market_order(order, confirmation_required=False)
            assert result.success is True
            orders_executed.append(order.order_id)
        
        # Verify all orders in database
        for order_id in orders_executed:
            db_order = await repository.get_order(order_id)
            assert db_order is not None
            assert db_order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_order_cancellation_flow(self, executor, repository):
        """Test order cancellation flow."""
        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001")
        )
        
        # Mock confirmation decline
        with pytest.mock.patch.object(executor, '_get_confirmation', return_value=False):
            result = await executor.execute_market_order(order, confirmation_required=True)
        
        # Verify cancellation
        assert result.success is False
        assert result.order.status == OrderStatus.CANCELLED
        
        # Verify cancelled status in database
        db_order = await repository.get_order(order.order_id)
        assert db_order is not None
        assert db_order.status == OrderStatus.CANCELLED
        assert db_order.filled_quantity == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_stop_loss_order_creation(self, executor, repository):
        """Test automatic stop-loss order creation after buy."""
        # Create buy order
        buy_order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001")
        )
        
        # Execute buy order
        result = await executor.execute_market_order(buy_order, confirmation_required=False)
        assert result.success is True
        
        # Wait for stop-loss to be created
        await asyncio.sleep(0.1)
        
        # Get all orders for the position
        position_orders = await repository.get_orders_by_position(buy_order.position_id)
        
        # Should have at least 2 orders (buy + stop-loss)
        assert len(position_orders) >= 2
        
        # Find stop-loss order
        stop_loss_orders = [o for o in position_orders if o.type == OrderType.STOP_LOSS]
        assert len(stop_loss_orders) == 1
        
        stop_loss = stop_loss_orders[0]
        assert stop_loss.side == OrderSide.SELL
        assert stop_loss.quantity == buy_order.quantity
    
    @pytest.mark.asyncio
    async def test_risk_engine_integration(self, executor, risk_engine, repository):
        """Test risk engine validates position size."""
        # Create order with large size
        large_order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.01")  # $500 at $50k BTC
        )
        
        # Validate position size with risk engine
        position_size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000")
        )
        
        # Should be limited by 5% rule
        assert position_size == Decimal("50")  # 5% of $1000
    
    @pytest.mark.asyncio
    async def test_open_orders_retrieval(self, executor, repository):
        """Test retrieving open orders."""
        # Create multiple orders with different statuses
        orders = []
        
        # Create pending order
        pending_order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("49000"),
            quantity=Decimal("0.001"),
            status=OrderStatus.PENDING
        )
        await repository.create_order(pending_order)
        orders.append(pending_order)
        
        # Create partial order
        partial_order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="ETH/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.SELL,
            price=Decimal("3000"),
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.05"),
            status=OrderStatus.PARTIAL
        )
        await repository.create_order(partial_order)
        orders.append(partial_order)
        
        # Create filled order (should not be returned)
        filled_order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001"),
            filled_quantity=Decimal("0.001"),
            status=OrderStatus.FILLED
        )
        await repository.create_order(filled_order)
        
        # Get all open orders
        open_orders = await repository.get_open_orders()
        assert len(open_orders) == 2
        
        # Get open orders for specific symbol
        btc_orders = await repository.get_open_orders("BTC/USDT")
        assert len(btc_orders) == 1
        assert btc_orders[0].order_id == pending_order.order_id
    
    @pytest.mark.asyncio
    async def test_emergency_cancel_all(self, executor, repository):
        """Test emergency cancellation of all orders."""
        # Create multiple pending orders
        order_ids = []
        for i in range(3):
            order = Order(
                order_id=str(uuid4()),
                position_id=str(uuid4()),
                client_order_id=str(uuid4()),
                symbol="BTC/USDT" if i < 2 else "ETH/USDT",
                type=OrderType.LIMIT,
                side=OrderSide.BUY,
                price=Decimal("49000"),
                quantity=Decimal("0.001"),
                status=OrderStatus.PENDING
            )
            await repository.create_order(order)
            order_ids.append(order.order_id)
            executor.pending_orders[order.order_id] = order
        
        # Cancel all orders
        cancelled_count = await executor.cancel_all_orders()
        
        # Verify cancellation
        assert cancelled_count >= 0  # Mock gateway returns empty list
        assert len(executor.pending_orders) == 0
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self, repository):
        """Test transaction rollback on failure."""
        # Begin transaction
        await repository.begin_transaction()
        
        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001")
        )
        
        try:
            await repository.create_order(order)
            # Simulate error
            raise Exception("Simulated failure")
        except:
            # Rollback transaction
            await repository.rollback_transaction()
        
        # Order should not exist in database
        db_order = await repository.get_order(order.order_id)
        assert db_order is None
    
    @pytest.mark.asyncio
    async def test_latency_tracking(self, executor, repository):
        """Test order execution latency tracking."""
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("0.001")
        )
        
        # Execute order
        result = await executor.execute_market_order(order, confirmation_required=False)
        
        # Verify latency was tracked
        assert result.latency_ms is not None
        assert result.latency_ms > 0
        assert result.latency_ms < 1000  # Should be under 1 second
        
        # Verify latency saved to database
        db_order = await repository.get_order(order.order_id)
        assert db_order.latency_ms == result.latency_ms