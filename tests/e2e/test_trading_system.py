"""End-to-end tests for the complete trading system."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog

from genesis.core.models import Position, Order, OrderType, OrderSide
from genesis.engine.engine import TradingEngine
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import StateMachine, TierState
from genesis.strategies.base import BaseStrategy
from genesis.exchange.gateway import ExchangeGateway

logger = structlog.get_logger(__name__)


class EndToEndTradingTests:
    """Complete end-to-end testing suite for the trading system."""

    def __init__(self):
        self.engine = None
        self.risk_engine = None
        self.state_machine = None
        self.exchange_gateway = None
        self.strategies = []
        self.test_results = {}

    async def setup_test_environment(self):
        """Initialize the test environment with all components."""
        self.state_machine = StateMachine()
        self.risk_engine = RiskEngine(self.state_machine)
        self.exchange_gateway = AsyncMock(spec=ExchangeGateway)
        
        self.engine = TradingEngine(
            exchange_gateway=self.exchange_gateway,
            risk_engine=self.risk_engine,
            state_machine=self.state_machine
        )
        
        await self._setup_mock_exchange()
        await self._setup_test_strategies()
        
        logger.info("Test environment initialized")

    async def _setup_mock_exchange(self):
        """Configure mock exchange responses."""
        self.exchange_gateway.get_balance.return_value = {
            "USDT": Decimal("10000.00"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("10.0")
        }
        
        self.exchange_gateway.get_ticker.return_value = {
            "bid": Decimal("50000.00"),
            "ask": Decimal("50001.00"),
            "last": Decimal("50000.50")
        }
        
        self.exchange_gateway.place_order = AsyncMock(
            return_value={
                "order_id": "test_order_123",
                "status": "filled",
                "filled_qty": Decimal("0.1"),
                "avg_price": Decimal("50000.00")
            }
        )

    async def _setup_test_strategies(self):
        """Initialize test strategies."""
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        from genesis.strategies.hunter.mean_reversion import MeanReversionStrategy
        
        sniper_strategy = SimpleArbitrageStrategy(
            symbol="BTC/USDT",
            min_spread=Decimal("0.001")
        )
        
        hunter_strategy = MeanReversionStrategy(
            symbol="ETH/USDT",
            lookback_period=20,
            entry_threshold=Decimal("2.0")
        )
        
        self.strategies = [sniper_strategy, hunter_strategy]

    async def teardown(self):
        """Clean up test environment."""
        if self.engine:
            await self.engine.stop()
        
        self.engine = None
        self.risk_engine = None
        self.state_machine = None
        self.exchange_gateway = None
        self.strategies = []
        
        logger.info("Test environment cleaned up")


@pytest.fixture
async def trading_system():
    """Pytest fixture for the trading system tests."""
    test_system = EndToEndTradingTests()
    await test_system.setup_test_environment()
    yield test_system
    await test_system.teardown()


@pytest.mark.asyncio
class TestCompleteTradingFlow:
    """Test the complete trading flow from signal to execution."""

    async def test_signal_generation_to_execution(self, trading_system):
        """Test complete flow: signal → validation → execution → fill."""
        signal = {
            "strategy": "simple_arb",
            "symbol": "BTC/USDT",
            "side": OrderSide.BUY,
            "quantity": Decimal("0.1"),
            "signal_strength": Decimal("0.8")
        }
        
        order = await trading_system.engine.process_signal(signal)
        
        assert order is not None
        assert order["order_id"] == "test_order_123"
        assert order["status"] == "filled"
        assert order["filled_qty"] == Decimal("0.1")

    async def test_position_lifecycle(self, trading_system):
        """Test position opening, monitoring, and closure."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50500.00")
        )
        
        await trading_system.engine.open_position(position)
        
        assert position.is_open
        assert position.unrealized_pnl == Decimal("50.00")
        
        await trading_system.engine.close_position(position)
        
        assert not position.is_open
        assert position.realized_pnl == Decimal("50.00")

    async def test_risk_validation(self, trading_system):
        """Test risk engine validation of orders."""
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),  # Large order
            price=Decimal("50000.00")
        )
        
        is_valid = await trading_system.risk_engine.validate_order(order)
        
        assert not is_valid  # Should reject due to position size limits

    async def test_state_transitions(self, trading_system):
        """Test trading state machine transitions."""
        initial_state = trading_system.state_machine.current_state
        assert initial_state == "IDLE"
        
        await trading_system.engine.start()
        assert trading_system.state_machine.current_state == "RUNNING"
        
        await trading_system.engine.pause()
        assert trading_system.state_machine.current_state == "PAUSED"
        
        await trading_system.engine.stop()
        assert trading_system.state_machine.current_state == "STOPPED"


@pytest.mark.asyncio
class TestSystemIntegration:
    """Test integration between system components."""

    async def test_engine_strategy_integration(self, trading_system):
        """Test trading engine and strategy integration."""
        strategy = trading_system.strategies[0]
        await trading_system.engine.register_strategy(strategy)
        
        registered = trading_system.engine.get_active_strategies()
        assert strategy in registered
        
        await trading_system.engine.unregister_strategy(strategy)
        registered = trading_system.engine.get_active_strategies()
        assert strategy not in registered

    async def test_exchange_gateway_integration(self, trading_system):
        """Test exchange gateway integration with the system."""
        balance = await trading_system.exchange_gateway.get_balance()
        assert "USDT" in balance
        assert balance["USDT"] == Decimal("10000.00")
        
        ticker = await trading_system.exchange_gateway.get_ticker("BTC/USDT")
        assert ticker["last"] == Decimal("50000.50")

    async def test_risk_engine_integration(self, trading_system):
        """Test risk engine integration with trading engine."""
        trading_system.risk_engine.set_max_position_size(Decimal("1.0"))
        
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5")
        )
        
        validated = await trading_system.risk_engine.validate_order(order)
        assert validated
        
        large_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0")
        )
        
        validated = await trading_system.risk_engine.validate_order(large_order)
        assert not validated

    async def test_database_persistence(self, trading_system):
        """Test database persistence of trading data."""
        from genesis.data.repository import TradeRepository
        
        repo = TradeRepository()
        
        trade = {
            "order_id": "test_123",
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": Decimal("0.1"),
            "price": Decimal("50000.00"),
            "timestamp": "2025-01-03T12:00:00Z"
        }
        
        await repo.save_trade(trade)
        
        saved_trade = await repo.get_trade("test_123")
        assert saved_trade is not None
        assert saved_trade["symbol"] == "BTC/USDT"
        assert saved_trade["quantity"] == Decimal("0.1")


@pytest.mark.asyncio
class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    async def test_exchange_disconnection_recovery(self, trading_system):
        """Test recovery from exchange disconnection."""
        trading_system.exchange_gateway.get_ticker.side_effect = ConnectionError("Exchange disconnected")
        
        with pytest.raises(ConnectionError):
            await trading_system.exchange_gateway.get_ticker("BTC/USDT")
        
        trading_system.exchange_gateway.get_ticker.side_effect = None
        trading_system.exchange_gateway.get_ticker.return_value = {
            "last": Decimal("50000.00")
        }
        
        ticker = await trading_system.exchange_gateway.get_ticker("BTC/USDT")
        assert ticker["last"] == Decimal("50000.00")

    async def test_order_rejection_handling(self, trading_system):
        """Test handling of order rejections."""
        trading_system.exchange_gateway.place_order.return_value = {
            "order_id": None,
            "status": "rejected",
            "reason": "Insufficient balance"
        }
        
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.0")
        )
        
        result = await trading_system.engine.execute_order(order)
        assert result["status"] == "rejected"
        assert "Insufficient balance" in result["reason"]

    async def test_network_timeout_handling(self, trading_system):
        """Test handling of network timeouts."""
        trading_system.exchange_gateway.place_order.side_effect = asyncio.TimeoutError("Network timeout")
        
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1")
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await trading_system.engine.execute_order(order)
        
        assert trading_system.engine.get_pending_orders() == []

    async def test_partial_fill_handling(self, trading_system):
        """Test handling of partial order fills."""
        trading_system.exchange_gateway.place_order.return_value = {
            "order_id": "partial_123",
            "status": "partially_filled",
            "filled_qty": Decimal("0.05"),
            "remaining_qty": Decimal("0.05"),
            "avg_price": Decimal("50000.00")
        }
        
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49999.00")
        )
        
        result = await trading_system.engine.execute_order(order)
        assert result["status"] == "partially_filled"
        assert result["filled_qty"] == Decimal("0.05")
        assert result["remaining_qty"] == Decimal("0.05")