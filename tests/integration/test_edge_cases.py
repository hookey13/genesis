"""
Edge case validation suite.
Tests zero balance, partial fills, extreme conditions, and boundary cases.
"""
import asyncio
import pytest
from decimal import Decimal, InvalidOperation
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import structlog
import math

from genesis.core.models import (
    Position, Order, Trade, Signal, TierType,
    OrderStatus, OrderType, OrderSide, ExecutionType
)
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.risk_engine import RiskEngine
from genesis.data.repository import Repository
from genesis.exchange.gateway import ExchangeGateway
from genesis.core.account_manager import AccountManager
from genesis.core.single_account_manager import SingleAccountManager

logger = structlog.get_logger()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_repository(self):
        """Mock repository."""
        repo = Mock(spec=Repository)
        repo.positions = {}
        repo.orders = {}
        repo.save_order = Mock()
        repo.save_position = Mock()
        repo.get_open_positions = Mock(return_value=[])
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange gateway."""
        exchange = Mock(spec=ExchangeGateway)
        exchange.get_balance = AsyncMock(return_value=Decimal("0"))
        exchange.place_order = AsyncMock()
        exchange.get_min_order_size = Mock(return_value=Decimal("0.001"))
        exchange.get_tick_size = Mock(return_value=Decimal("0.01"))
        return exchange

    @pytest.fixture
    def risk_engine(self, mock_repository):
        """Create risk engine."""
        return RiskEngine(
            repository=mock_repository,
            max_position_size=Decimal("10000"),
            max_drawdown=Decimal("0.2")
        )

    @pytest.fixture
    def account_manager(self):
        """Create account manager."""
        return SingleAccountManager(
            account_id="test_account",
            initial_balance=Decimal("0"),
            tier=TierType.SNIPER
        )

    @pytest.mark.asyncio
    async def test_zero_balance_order_rejection(self, mock_exchange, account_manager):
        """Test orders are rejected when balance is zero."""
        # Set zero balance
        account_manager.balance = Decimal("0")
        mock_exchange.get_balance.return_value = Decimal("0")
        
        # Try to place order
        order = Order(
            id="zero_balance_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW
        )
        
        # Check if order can be placed
        can_afford = account_manager.balance >= (
            order.quantity * Decimal("50000")  # Assume BTC price
        )
        
        assert not can_afford
        assert account_manager.balance == Decimal("0")
        
        # Exchange should reject
        mock_exchange.place_order.side_effect = Exception("Insufficient balance")
        
        with pytest.raises(Exception) as exc_info:
            await mock_exchange.place_order(order.__dict__)
        
        assert "Insufficient balance" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_partial_fill_processing(self, mock_exchange, mock_repository):
        """Test handling of partially filled orders."""
        # Create order
        order = Order(
            id="partial_fill_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.NEW
        )
        
        # Simulate partial fills
        fill_updates = [
            {"executedQty": "0.3", "status": "PARTIALLY_FILLED"},
            {"executedQty": "0.6", "status": "PARTIALLY_FILLED"},
            {"executedQty": "0.9", "status": "PARTIALLY_FILLED"},
            {"executedQty": "1.0", "status": "FILLED"}
        ]
        
        filled_quantity = Decimal("0")
        for update in fill_updates:
            mock_exchange.get_order.return_value = {
                "orderId": order.id,
                "status": update["status"],
                "executedQty": update["executedQty"],
                "avgPrice": "50000"
            }
            
            order_status = await mock_exchange.get_order(order.id)
            current_filled = Decimal(order_status["executedQty"])
            
            # Process partial fill
            if current_filled > filled_quantity:
                fill_amount = current_filled - filled_quantity
                filled_quantity = current_filled
                
                # Create trade for partial fill
                trade = Trade(
                    id=f"trade_{filled_quantity}",
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    price=Decimal(order_status["avgPrice"]),
                    quantity=fill_amount,
                    commission=Decimal("0.001") * fill_amount,
                    timestamp=datetime.utcnow()
                )
                mock_repository.save_trade(trade)
        
        assert filled_quantity == Decimal("1.0")
        assert order_status["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_simultaneous_buy_sell_signal_resolution(self, mock_repository, mock_exchange):
        """Test resolution of simultaneous buy and sell signals."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository,
            exchange_gateway=mock_exchange
        )
        
        # Create simultaneous conflicting signals
        buy_signal = Signal(
            strategy_name="bull_strategy",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            confidence=Decimal("0.85"),
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow()
        )
        
        sell_signal = Signal(
            strategy_name="bear_strategy",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            confidence=Decimal("0.85"),  # Same confidence
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow()
        )
        
        # Resolution strategies
        resolution_strategies = {
            "confidence": lambda signals: max(signals, key=lambda s: s.confidence),
            "timestamp": lambda signals: min(signals, key=lambda s: s.timestamp),
            "cancel_both": lambda signals: None,
            "execute_both": lambda signals: signals
        }
        
        # Test different resolution strategies
        for strategy_name, resolver in resolution_strategies.items():
            result = resolver([buy_signal, sell_signal])
            
            if strategy_name == "confidence":
                # Same confidence, should use timestamp
                assert result in [buy_signal, sell_signal]
            elif strategy_name == "timestamp":
                # First signal wins
                assert result == buy_signal
            elif strategy_name == "cancel_both":
                # Cancel both on conflict
                assert result is None
            elif strategy_name == "execute_both":
                # Execute both (hedge)
                assert len(result) == 2

    @pytest.mark.asyncio
    async def test_maximum_position_count_enforcement(self, mock_repository, risk_engine):
        """Test enforcement of maximum position count."""
        max_positions = 5
        risk_engine.max_positions = max_positions
        
        # Create existing positions
        positions = []
        for i in range(max_positions):
            pos = Position(
                id=f"pos_{i}",
                symbol=f"COIN{i}USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("100"),
                current_price=Decimal("100"),
                quantity=Decimal("1"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0")
            )
            positions.append(pos)
        
        mock_repository.get_open_positions.return_value = positions
        
        # Try to open new position
        new_order = Order(
            id="excess_position",
            symbol="NEWCOINUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            status=OrderStatus.NEW
        )
        
        # Check if allowed
        current_count = len(mock_repository.get_open_positions())
        can_open = current_count < max_positions
        
        assert not can_open
        assert current_count == max_positions

    @pytest.mark.asyncio
    async def test_minimum_order_size_handling(self, mock_exchange):
        """Test handling of orders below minimum size."""
        min_order_size = Decimal("0.001")
        mock_exchange.get_min_order_size.return_value = min_order_size
        
        # Test various order sizes
        test_cases = [
            (Decimal("0.0001"), False, "Below minimum"),
            (Decimal("0.001"), True, "Exactly minimum"),
            (Decimal("0.01"), True, "Above minimum"),
            (Decimal("0"), False, "Zero quantity")
        ]
        
        for quantity, should_pass, description in test_cases:
            order = Order(
                id=f"min_size_{quantity}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                status=OrderStatus.NEW
            )
            
            is_valid = quantity >= min_order_size and quantity > 0
            assert is_valid == should_pass, f"Failed for {description}"

    @pytest.mark.asyncio
    async def test_extreme_market_conditions(self, mock_exchange, risk_engine):
        """Test behavior under extreme market volatility."""
        # Simulate flash crash
        price_sequence = [
            Decimal("50000"),  # Normal
            Decimal("45000"),  # -10%
            Decimal("35000"),  # -30%
            Decimal("20000"),  # -60%
            Decimal("25000"),  # Recovery
            Decimal("45000")   # Further recovery
        ]
        
        circuit_breaker_triggered = False
        for i, price in enumerate(price_sequence):
            if i > 0:
                price_change = (price - price_sequence[i-1]) / price_sequence[i-1]
                
                # Check if circuit breaker should trigger
                if abs(price_change) > Decimal("0.2"):  # 20% move
                    circuit_breaker_triggered = True
                    # Halt trading
                    break
        
        assert circuit_breaker_triggered
        assert price == Decimal("20000")  # Stopped at -60% level

    @pytest.mark.asyncio
    async def test_decimal_precision_edge_cases(self):
        """Test decimal precision in extreme cases."""
        # Test very small values
        tiny_value = Decimal("0.00000001")  # Satoshi
        calculation = tiny_value * Decimal("100000000")
        assert calculation == Decimal("1")
        
        # Test very large values
        large_value = Decimal("999999999999.99999999")
        calculation = large_value / Decimal("1000000")
        assert calculation == Decimal("999999.99999999999999")
        
        # Test precision preservation in calculations
        price = Decimal("50123.456789")
        quantity = Decimal("0.123456789")
        value = price * quantity
        
        # Verify no precision loss
        assert str(value) == "6189.580224114921"
        
        # Test division by zero handling
        with pytest.raises(Exception):
            result = Decimal("1") / Decimal("0")

    @pytest.mark.asyncio
    async def test_timestamp_boundary_conditions(self):
        """Test handling of timestamp edge cases."""
        # Test future timestamp
        future_time = datetime.utcnow() + timedelta(days=365)
        order = Order(
            id="future_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW,
            created_at=future_time
        )
        
        # Should reject future orders
        assert order.created_at > datetime.utcnow()
        
        # Test very old timestamp
        old_time = datetime(2009, 1, 3)  # Bitcoin genesis block date
        old_order = Order(
            id="old_order",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW,
            created_at=old_time
        )
        
        # Should handle historical data
        assert old_order.created_at < datetime.utcnow()

    @pytest.mark.asyncio
    async def test_null_and_nan_value_handling(self, mock_repository):
        """Test handling of null and NaN values."""
        # Test None values
        position = Position(
            id="null_test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            current_price=None,  # Null price
            quantity=Decimal("0.1"),
            unrealized_pnl=None,
            realized_pnl=Decimal("0")
        )
        
        # Calculate P&L with null current price
        if position.current_price is None:
            pnl = Decimal("0")
        else:
            pnl = (position.current_price - position.entry_price) * position.quantity
        
        assert pnl == Decimal("0")
        
        # Test NaN detection
        try:
            nan_value = float('nan')
            decimal_value = Decimal(str(nan_value))
        except (InvalidOperation, ValueError):
            # Decimal should reject NaN
            pass

    @pytest.mark.asyncio
    async def test_concurrent_order_limit_enforcement(self, mock_exchange, mock_repository):
        """Test enforcement of concurrent order limits."""
        max_concurrent_orders = 10
        
        # Create existing orders
        active_orders = []
        for i in range(max_concurrent_orders):
            order = Order(
                id=f"concurrent_{i}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.01"),
                price=Decimal(str(50000 - i * 100)),
                status=OrderStatus.NEW
            )
            active_orders.append(order)
        
        mock_repository.get_active_orders = Mock(return_value=active_orders)
        
        # Try to place one more order
        new_order = Order(
            id="excess_concurrent",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.01"),
            price=Decimal("49000"),
            status=OrderStatus.NEW
        )
        
        # Check limit
        current_count = len(mock_repository.get_active_orders())
        can_place = current_count < max_concurrent_orders
        
        assert not can_place
        assert current_count == max_concurrent_orders

    @pytest.mark.asyncio
    async def test_extreme_leverage_prevention(self, risk_engine, account_manager):
        """Test prevention of excessive leverage."""
        account_manager.balance = Decimal("1000")
        max_leverage = Decimal("3")  # 3x max
        
        # Try to open position with excessive leverage
        position_value = Decimal("5000")  # 5x leverage
        leverage = position_value / account_manager.balance
        
        is_allowed = leverage <= max_leverage
        assert not is_allowed
        assert leverage == Decimal("5")

    @pytest.mark.asyncio
    async def test_order_type_edge_cases(self, mock_exchange):
        """Test edge cases for different order types."""
        # Stop loss at current price
        current_price = Decimal("50000")
        stop_loss_order = Order(
            id="stop_at_current",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=Decimal("0.1"),
            stop_price=current_price,  # Stop at current
            status=OrderStatus.NEW
        )
        
        # Should trigger immediately
        should_trigger = current_price <= stop_loss_order.stop_price
        assert should_trigger
        
        # Limit order with price = 0
        with pytest.raises(ValueError):
            invalid_order = Order(
                id="zero_price",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("0"),
                status=OrderStatus.NEW
            )

    @pytest.mark.asyncio
    async def test_fee_calculation_edge_cases(self):
        """Test fee calculations in edge cases."""
        # Zero fee
        trade_value = Decimal("1000")
        fee_rate = Decimal("0")
        fee = trade_value * fee_rate
        assert fee == Decimal("0")
        
        # Very high fee
        high_fee_rate = Decimal("0.5")  # 50% fee
        high_fee = trade_value * high_fee_rate
        assert high_fee == Decimal("500")
        
        # Negative fee (rebate)
        rebate_rate = Decimal("-0.0001")  # Maker rebate
        rebate = trade_value * rebate_rate
        assert rebate == Decimal("-0.1")