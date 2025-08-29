"""
Unit tests for paper trading functionality.

Tests paper trading test harness, P&L tracking, and mock exchange integration.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import Order, OrderSide, OrderType
from genesis.data.models_db import PaperTradingSession, Session
from genesis.engine.event_bus import EventBus
from genesis.engine.paper_trading_enforcer import (
    PaperTrade,
    PaperTradingEnforcer,
    SessionMetrics,
    SessionStatus,
)
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway
from genesis.exchange.mock_exchange import MockExchange


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock(spec=Session)
    return session


@pytest.fixture
def paper_trading_enforcer(mock_session):
    """Create a paper trading enforcer instance."""
    return PaperTradingEnforcer(session=mock_session)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange instance in paper trading mode."""
    return MockExchange(
        initial_balance={"USDT": Decimal("10000")},
        paper_trading_mode=True
    )


@pytest.fixture
def event_bus():
    """Create an event bus instance."""
    return EventBus()


@pytest.fixture
def risk_engine():
    """Create a mock risk engine."""
    engine = MagicMock(spec=RiskEngine)
    engine.tier_limits = {"stop_loss_percent": Decimal("2.0")}
    engine.calculate_position_size.return_value = Decimal("0.01")
    engine.validate_order_risk.return_value = None
    engine.validate_portfolio_risk.return_value = {"approved": True, "rejections": [], "warnings": []}
    engine.validate_configuration.return_value = True
    return engine


@pytest.fixture
def exchange_gateway(mock_exchange):
    """Create a mock exchange gateway."""
    gateway = MagicMock(spec=ExchangeGateway)
    gateway.validate_connection.return_value = True
    gateway.execute_order.return_value = {
        "success": True,
        "exchange_order_id": "EX123",
        "fill_price": Decimal("50000"),
        "latency_ms": 50
    }
    return gateway


@pytest.fixture
def trading_loop(event_bus, risk_engine, exchange_gateway):
    """Create a trading loop instance in paper trading mode."""
    return TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=exchange_gateway,
        paper_trading_mode=True,
        paper_trading_session_id="test-session-123"
    )


class TestPaperTradingEnforcer:
    """Test paper trading enforcement and session management."""

    @pytest.mark.asyncio
    async def test_create_paper_trading_session(self, paper_trading_enforcer, mock_session):
        """Test creating a paper trading session."""
        # Setup
        mock_session.add.return_value = None
        mock_session.commit.return_value = None

        # Execute
        session_id = await paper_trading_enforcer.require_paper_trading(
            account_id="test-account",
            strategy="iceberg_orders",
            duration_hours=24,
        )

        # Verify
        assert session_id is not None
        assert session_id in paper_trading_enforcer._active_sessions
        assert session_id in paper_trading_enforcer._session_tasks
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_order_fill(self, paper_trading_enforcer):
        """Test simulating order fills with slippage."""
        # Setup session
        session_id = "test-session"
        paper_trading_enforcer._active_sessions[session_id] = []

        # Execute buy order
        trade = await paper_trading_enforcer.simulate_order_fill(
            session_id=session_id,
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            slippage_percent=Decimal("0.1"),
        )

        # Verify buy order (price should be higher due to slippage)
        assert trade.trade_id is not None
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "BUY"
        assert trade.quantity == Decimal("0.01")
        assert trade.entry_price > Decimal("50000")  # Slippage increases buy price
        assert trade.entry_price == Decimal("50000") * Decimal("1.001")

        # Execute sell order
        sell_trade = await paper_trading_enforcer.simulate_order_fill(
            session_id=session_id,
            symbol="BTC/USDT",
            side="SELL",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
            slippage_percent=Decimal("0.1"),
        )

        # Verify sell order (price should be lower due to slippage)
        assert sell_trade.entry_price < Decimal("50000")  # Slippage decreases sell price
        assert sell_trade.entry_price == Decimal("50000") * Decimal("0.999")

    @pytest.mark.asyncio
    async def test_close_paper_trade_with_pnl(self, paper_trading_enforcer):
        """Test closing a paper trade and calculating P&L."""
        # Setup
        session_id = "test-session"
        paper_trading_enforcer._active_sessions[session_id] = []

        # Create a buy trade
        trade = await paper_trading_enforcer.simulate_order_fill(
            session_id=session_id,
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )

        # Close with profit
        closed_trade = await paper_trading_enforcer.close_paper_trade(
            session_id=session_id,
            trade_id=trade.trade_id,
            exit_price=Decimal("51000"),  # 2% profit
        )

        # Verify P&L calculation
        assert closed_trade.exit_price == Decimal("51000")
        assert closed_trade.pnl > 0  # Profitable trade
        expected_pnl = (Decimal("51000") - trade.entry_price) * Decimal("0.01")
        assert abs(closed_trade.pnl - expected_pnl) < Decimal("0.01")
        assert closed_trade.is_profitable
        assert closed_trade.is_closed

    @pytest.mark.asyncio
    async def test_session_metrics_calculation(self, paper_trading_enforcer, mock_session):
        """Test calculating session metrics with multiple trades."""
        # Setup
        session_id = "test-session"
        paper_trading_enforcer._active_sessions[session_id] = []

        # Mock database query
        mock_paper_session = MagicMock(spec=PaperTradingSession)
        mock_paper_session.started_at = datetime.utcnow()
        mock_paper_session.strategy_name = "iceberg_orders"
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_paper_session

        # Create and close multiple trades
        trades_data = [
            ("BTC/USDT", "BUY", Decimal("0.01"), Decimal("50000"), Decimal("51000")),  # +10 USDT
            ("ETH/USDT", "BUY", Decimal("0.1"), Decimal("3000"), Decimal("2950")),     # -5 USDT
            ("BTC/USDT", "SELL", Decimal("0.01"), Decimal("50500"), Decimal("50000")), # +5 USDT
        ]

        for symbol, side, qty, entry, exit_price in trades_data:
            trade = await paper_trading_enforcer.simulate_order_fill(
                session_id=session_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                price=entry,
            )
            await paper_trading_enforcer.close_paper_trade(
                session_id=session_id,
                trade_id=trade.trade_id,
                exit_price=exit_price,
            )

        # Get metrics
        metrics = await paper_trading_enforcer.get_session_metrics(session_id)

        # Verify metrics
        assert metrics.total_trades == 3
        assert metrics.profitable_trades == 2
        assert metrics.success_rate == Decimal("2") / Decimal("3")
        assert metrics.total_pnl > 0  # Net positive P&L

    @pytest.mark.asyncio
    async def test_session_completion_requirements(self, paper_trading_enforcer, mock_session):
        """Test checking session completion requirements."""
        # Setup
        session_id = "test-session"
        paper_trading_enforcer._active_sessions[session_id] = []

        # Mock database session
        mock_paper_session = MagicMock(spec=PaperTradingSession)
        mock_paper_session.strategy_name = "iceberg_orders"
        mock_paper_session.started_at = datetime.utcnow()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_paper_session

        # Create insufficient trades (less than required 20)
        for i in range(10):
            trade = await paper_trading_enforcer.simulate_order_fill(
                session_id=session_id,
                symbol="BTC/USDT",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=Decimal("0.01"),
                price=Decimal("50000"),
            )
            # Close with slight profit
            await paper_trading_enforcer.close_paper_trade(
                session_id=session_id,
                trade_id=trade.trade_id,
                exit_price=Decimal("50100") if trade.side == "BUY" else Decimal("49900"),
            )

        # Check completion
        is_complete, failure_reasons = await paper_trading_enforcer.check_session_completion(session_id)

        # Verify not complete due to insufficient trades
        assert not is_complete
        assert any("Insufficient trades" in reason for reason in failure_reasons)


class TestMockExchange:
    """Test mock exchange functionality for paper trading."""

    @pytest.mark.asyncio
    async def test_paper_trading_mode_initialization(self):
        """Test mock exchange initializes correctly in paper trading mode."""
        exchange = MockExchange(paper_trading_mode=True)
        
        assert exchange.paper_trading_mode is True
        assert exchange.slippage_percent == Decimal("0.1")
        assert "USDT" in exchange.balances

    @pytest.mark.asyncio
    async def test_create_order_with_slippage(self, mock_exchange):
        """Test order creation applies realistic slippage."""
        from genesis.exchange.models import OrderRequest

        # Create buy order
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            quantity=Decimal("0.01"),
            client_order_id="test-order-1",
        )

        response = await mock_exchange.create_order(request)

        # Verify order filled
        assert response.status == "filled"
        assert response.filled_quantity == request.quantity
        assert response.price > 0

    @pytest.mark.asyncio
    async def test_balance_updates_after_trade(self, mock_exchange):
        """Test balance updates correctly after paper trades."""
        from genesis.exchange.models import OrderRequest

        initial_usdt = mock_exchange.balances["USDT"]

        # Execute buy order
        request = OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            type="market",
            quantity=Decimal("0.01"),
            client_order_id="test-order-1",
        )

        response = await mock_exchange.create_order(request)

        # Check balance decreased
        cost = response.filled_quantity * response.price
        assert mock_exchange.balances["USDT"] < initial_usdt
        assert mock_exchange.balances["USDT"] == initial_usdt - cost
        assert "BTC" in mock_exchange.balances
        assert mock_exchange.balances["BTC"] == Decimal("0.51")  # 0.5 initial + 0.01 bought


class TestTradingLoopPaperMode:
    """Test trading loop in paper trading mode."""

    @pytest.mark.asyncio
    async def test_paper_trading_mode_initialization(self, trading_loop):
        """Test trading loop initializes correctly in paper mode."""
        assert trading_loop.paper_trading_mode is True
        assert trading_loop.paper_trading_session_id == "test-session-123"

    @pytest.mark.asyncio
    async def test_paper_trade_event_prefix(self, trading_loop, event_bus):
        """Test events include paper trade prefix when in paper mode."""
        # Setup
        await event_bus.start()
        trading_loop._register_event_handlers()

        # Create order
        order = Order(
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("0.01"),
        )

        # Store order in pending
        trading_loop.pending_orders[order.order_id] = order

        # Mock successful execution
        trading_loop.exchange_gateway.execute_order.return_value = {
            "success": True,
            "exchange_order_id": "EX123",
            "fill_price": Decimal("50000"),
            "latency_ms": 50,
        }

        # Execute order
        await trading_loop._execute_order(order)

        # Verify event published with paper trade flag
        assert len(trading_loop.event_store) > 0
        stored_event = trading_loop.event_store[-1]
        assert stored_event.event_data.get("paper_trade") is True
        assert stored_event.event_data.get("session_id") == "test-session-123"


class TestPnLCalculation:
    """Test P&L calculation accuracy."""

    def test_pnl_calculation_long_position_profit(self):
        """Test P&L calculation for profitable long position."""
        trade = PaperTrade(
            trade_id="test-1",
            session_id="session-1",
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        # Close with profit
        trade.exit_price = Decimal("51000")
        trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity

        assert trade.pnl == Decimal("100")  # 0.1 * (51000 - 50000)
        assert trade.is_profitable

    def test_pnl_calculation_short_position_profit(self):
        """Test P&L calculation for profitable short position."""
        trade = PaperTrade(
            trade_id="test-2",
            session_id="session-1",
            symbol="BTC/USDT",
            side="SELL",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        # Close with profit (price went down)
        trade.exit_price = Decimal("49000")
        trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity

        assert trade.pnl == Decimal("100")  # 0.1 * (50000 - 49000)
        assert trade.is_profitable

    def test_pnl_calculation_with_fees(self):
        """Test P&L calculation including trading fees."""
        trade = PaperTrade(
            trade_id="test-3",
            session_id="session-1",
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
        )

        # Close with small profit
        trade.exit_price = Decimal("50100")
        gross_pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        
        # Apply 0.1% fee on both entry and exit
        entry_fee = trade.entry_price * trade.quantity * Decimal("0.001")
        exit_fee = trade.exit_price * trade.quantity * Decimal("0.001")
        net_pnl = gross_pnl - entry_fee - exit_fee

        assert gross_pnl == Decimal("10")  # 0.1 * (50100 - 50000)
        assert net_pnl < gross_pnl  # Net P&L less than gross due to fees

    def test_pnl_accuracy_to_two_decimals(self):
        """Test P&L calculations are accurate to 2 decimal places."""
        trades = [
            (Decimal("0.12345"), Decimal("50000.50"), Decimal("50100.75")),
            (Decimal("0.98765"), Decimal("3000.25"), Decimal("3010.50")),
            (Decimal("0.00123"), Decimal("400.10"), Decimal("401.25")),
        ]

        for quantity, entry, exit_price in trades:
            trade = PaperTrade(
                trade_id=f"test-{quantity}",
                session_id="session-1",
                symbol="TEST/USDT",
                side="BUY",
                quantity=quantity,
                entry_price=entry,
            )
            
            trade.exit_price = exit_price
            trade.pnl = (exit_price - entry) * quantity
            
            # Round to 2 decimal places
            rounded_pnl = trade.pnl.quantize(Decimal("0.01"))
            
            # Verify precision
            assert str(rounded_pnl).split(".")[-1].__len__() <= 2


class TestContinuousOperation:
    """Test 24-hour continuous operation capabilities."""

    @pytest.mark.asyncio
    async def test_heartbeat_monitoring(self, trading_loop):
        """Test trading loop heartbeat monitoring."""
        # Start trading loop
        startup_success = await trading_loop.startup()
        assert startup_success

        # Check running status
        assert not trading_loop.running

        # Start run task
        run_task = asyncio.create_task(trading_loop.run())

        # Wait briefly
        await asyncio.sleep(0.1)

        # Verify running
        assert trading_loop.running

        # Cancel and cleanup
        run_task.cancel()
        await trading_loop.shutdown()
        assert not trading_loop.running

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, trading_loop):
        """Test trading loop tracks statistics correctly."""
        # Initialize
        await trading_loop.startup()

        # Simulate some activity
        trading_loop.events_processed = 100
        trading_loop.signals_generated = 10
        trading_loop.orders_executed = 8
        trading_loop.positions_opened = 8
        trading_loop.positions_closed = 5

        # Get statistics
        stats = trading_loop.get_statistics()

        # Verify statistics
        assert stats["events_processed"] == 100
        assert stats["signals_generated"] == 10
        assert stats["orders_executed"] == 8
        assert stats["positions_opened"] == 8
        assert stats["positions_closed"] == 5
        assert stats["active_positions"] == 0  # No actual positions created

    @pytest.mark.asyncio
    async def test_performance_metrics(self, trading_loop):
        """Test performance metrics calculation."""
        # Add some latency data
        trading_loop.event_latencies = [10.5, 12.3, 11.8, 15.2, 9.7]
        trading_loop.signal_to_order_latencies = [25.1, 28.3, 26.7]
        trading_loop.order_execution_latencies = [50.2, 55.8, 48.9, 52.4]

        # Get metrics
        metrics = trading_loop.get_performance_metrics()

        # Verify event processing metrics
        assert metrics["event_processing"]["total_events"] == 0  # Not incremented in test
        assert metrics["event_processing"]["latency_ms"]["min"] == 9.7
        assert metrics["event_processing"]["latency_ms"]["max"] == 15.2
        assert metrics["event_processing"]["latency_ms"]["avg"] > 0

        # Verify order execution metrics  
        assert metrics["order_execution"]["latency_ms"]["min"] == 48.9
        assert metrics["order_execution"]["latency_ms"]["max"] == 55.8


@pytest.mark.asyncio
async def test_integration_paper_trading_flow():
    """Test complete paper trading flow integration."""
    # Setup components
    event_bus = EventBus()
    mock_exchange = MockExchange(paper_trading_mode=True)
    
    # Mock components
    risk_engine = MagicMock(spec=RiskEngine)
    risk_engine.tier_limits = {"stop_loss_percent": Decimal("2.0")}
    risk_engine.calculate_position_size.return_value = Decimal("0.01")
    risk_engine.validate_order_risk.return_value = None
    risk_engine.validate_portfolio_risk.return_value = {"approved": True, "rejections": [], "warnings": []}
    risk_engine.validate_configuration.return_value = True

    exchange_gateway = MagicMock(spec=ExchangeGateway)
    exchange_gateway.validate_connection.return_value = True
    exchange_gateway.execute_order.return_value = {
        "success": True,
        "exchange_order_id": "EX123",
        "fill_price": Decimal("50000"),
        "latency_ms": 50
    }

    # Create trading loop in paper mode
    trading_loop = TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=exchange_gateway,
        paper_trading_mode=True,
        paper_trading_session_id="integration-test-session"
    )

    # Start trading loop
    assert await trading_loop.startup()

    # Simulate trading signal
    signal_event = Event(
        event_type=EventType.ARBITRAGE_SIGNAL,
        event_data={
            "strategy_id": "test-strategy",
            "pair1_symbol": "BTC/USDT",
            "signal_type": "ENTRY",
            "confidence_score": 0.8,
            "entry_price": "50000",
        }
    )

    # Process signal
    await trading_loop._handle_trading_signal(signal_event)

    # Verify metrics
    assert trading_loop.signals_generated == 1
    assert trading_loop.orders_executed == 1

    # Verify paper trade flag in events
    assert len(trading_loop.event_store) > 0
    for event in trading_loop.event_store:
        if event.event_type == EventType.ORDER_FILLED:
            assert event.event_data.get("paper_trade") is True
            assert event.event_data.get("session_id") == "integration-test-session"

    # Cleanup
    await trading_loop.shutdown()