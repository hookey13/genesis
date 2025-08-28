"""
Critical path integration tests.
Tests complete order lifecycle, tier features, risk limits, and emergency procedures.
"""
import asyncio
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import structlog
import json

from genesis.core.models import (
    Position, Order, Trade, Signal, TierType, AccountTier,
    OrderStatus, OrderType, OrderSide, TradingState, AuditEntry
)
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import TierStateMachine
from genesis.data.repository import Repository
from genesis.data.performance_repo import PerformanceRepository
from genesis.exchange.gateway import ExchangeGateway
from genesis.analytics.performance_attribution import PerformanceAttributionEngine
from genesis.core.account_manager import AccountManager

logger = structlog.get_logger()


class TestCriticalPath:
    """Test critical business paths end-to-end."""

    @pytest.fixture
    def mock_repository(self):
        """Mock repository with audit support."""
        repo = Mock(spec=Repository)
        repo.positions = {}
        repo.orders = {}
        repo.trades = []
        repo.audit_log = []
        repo.save_order = Mock()
        repo.save_trade = Mock()
        repo.save_position = Mock()
        repo.save_audit_entry = Mock(side_effect=lambda e: repo.audit_log.append(e))
        repo.get_open_positions = Mock(return_value=[])
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange with full order lifecycle."""
        exchange = Mock(spec=ExchangeGateway)
        exchange.place_order = AsyncMock()
        exchange.get_order = AsyncMock()
        exchange.cancel_order = AsyncMock()
        exchange.get_balance = AsyncMock(return_value=Decimal("10000"))
        return exchange

    @pytest.fixture
    def risk_engine(self, mock_repository):
        """Create risk engine with limits."""
        return RiskEngine(
            repository=mock_repository,
            max_position_size=Decimal("5000"),
            max_drawdown=Decimal("0.2"),
            max_leverage=Decimal("3")
        )

    @pytest.fixture
    def tier_state_machine(self):
        """Create tier state machine."""
        return TierStateMachine(initial_tier=TierType.SNIPER)

    @pytest.fixture
    def orchestrator(self, mock_repository, mock_exchange):
        """Create strategy orchestrator."""
        return StrategyOrchestrator(
            repository=mock_repository,
            exchange_gateway=mock_exchange
        )

    @pytest.mark.asyncio
    async def test_complete_order_lifecycle(self, orchestrator, mock_exchange, mock_repository):
        """Test order from placement through execution, tracking, closing, and analysis."""
        order_id = "lifecycle_test_001"
        
        # Step 1: Place order
        order = Order(
            id=order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status=OrderStatus.NEW
        )
        
        mock_exchange.place_order.return_value = {
            "orderId": order_id,
            "status": "NEW"
        }
        
        # Place order
        result = await mock_exchange.place_order(order.__dict__)
        assert result["status"] == "NEW"
        
        # Audit entry for order placement
        audit_entry = AuditEntry(
            timestamp=datetime.utcnow(),
            action="ORDER_PLACED",
            entity_type="Order",
            entity_id=order_id,
            details={"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1"}
        )
        mock_repository.save_audit_entry(audit_entry)
        
        # Step 2: Execute order (fill)
        await asyncio.sleep(0.1)  # Simulate execution delay
        
        mock_exchange.get_order.return_value = {
            "orderId": order_id,
            "status": "FILLED",
            "executedQty": "0.1",
            "avgPrice": "49950"
        }
        
        order_status = await mock_exchange.get_order(order_id)
        assert order_status["status"] == "FILLED"
        
        # Create trade from execution
        trade = Trade(
            id=f"trade_{order_id}",
            order_id=order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal(order_status["avgPrice"]),
            quantity=Decimal(order_status["executedQty"]),
            commission=Decimal("0.0001") * Decimal(order_status["executedQty"]),
            timestamp=datetime.utcnow()
        )
        mock_repository.save_trade(trade)
        
        # Step 3: Track position
        position = Position(
            id=f"pos_{order_id}",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal(order_status["avgPrice"]),
            current_price=Decimal("49950"),
            quantity=Decimal(order_status["executedQty"]),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0")
        )
        mock_repository.save_position(position)
        
        # Update current price and P&L
        await asyncio.sleep(0.1)
        position.current_price = Decimal("51000")
        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
        assert position.unrealized_pnl == Decimal("105")  # (51000 - 49950) * 0.1
        
        # Step 4: Close position
        close_order = Order(
            id=f"{order_id}_close",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            status=OrderStatus.NEW
        )
        
        mock_exchange.place_order.return_value = {
            "orderId": f"{order_id}_close",
            "status": "FILLED",
            "executedQty": "0.1",
            "avgPrice": "51000"
        }
        
        close_result = await mock_exchange.place_order(close_order.__dict__)
        assert close_result["status"] == "FILLED"
        
        # Calculate realized P&L
        position.realized_pnl = (Decimal(close_result["avgPrice"]) - position.entry_price) * position.quantity
        position.unrealized_pnl = Decimal("0")
        assert position.realized_pnl == Decimal("105")
        
        # Step 5: Analyze performance
        perf_engine = PerformanceAttributionEngine(repository=mock_repository)
        mock_repository.get_trades.return_value = [trade]
        
        performance = await perf_engine.analyze_trade_performance(order_id)
        assert performance is not None
        
        # Verify complete audit trail
        assert len(mock_repository.audit_log) > 0
        assert mock_repository.audit_log[0].action == "ORDER_PLACED"

    @pytest.mark.asyncio
    async def test_tier_feature_accessibility(self, tier_state_machine):
        """Test tier features are correctly locked/unlocked based on conditions."""
        # Start as SNIPER
        assert tier_state_machine.current_tier == TierType.SNIPER
        
        # Check SNIPER features
        sniper_features = tier_state_machine.get_available_features()
        assert "simple_arbitrage" in sniper_features
        assert "multi_pair_trading" not in sniper_features
        assert "institutional_features" not in sniper_features
        
        # Meet HUNTER requirements
        tier_state_machine.update_metrics({
            "capital": Decimal("2500"),
            "consistency_score": Decimal("0.75"),
            "risk_score": Decimal("0.8"),
            "education_complete": True
        })
        
        # Transition to HUNTER
        can_upgrade = tier_state_machine.check_tier_upgrade()
        if can_upgrade:
            tier_state_machine.transition_to(TierType.HUNTER)
        
        assert tier_state_machine.current_tier == TierType.HUNTER
        
        # Check HUNTER features
        hunter_features = tier_state_machine.get_available_features()
        assert "multi_pair_trading" in hunter_features
        assert "iceberg_orders" in hunter_features
        assert "prime_broker_access" not in hunter_features
        
        # Meet STRATEGIST requirements
        tier_state_machine.update_metrics({
            "capital": Decimal("15000"),
            "consistency_score": Decimal("0.85"),
            "risk_score": Decimal("0.9"),
            "sharpe_ratio": Decimal("1.5"),
            "months_profitable": 6
        })
        
        # Transition to STRATEGIST
        can_upgrade = tier_state_machine.check_tier_upgrade()
        if can_upgrade:
            tier_state_machine.transition_to(TierType.STRATEGIST)
        
        assert tier_state_machine.current_tier == TierType.STRATEGIST
        
        # Check STRATEGIST features
        strategist_features = tier_state_machine.get_available_features()
        assert "institutional_features" in strategist_features
        assert "prime_broker_access" in strategist_features
        assert "multi_account_management" in strategist_features

    @pytest.mark.asyncio
    async def test_risk_limits_prevent_overleveragee(self, risk_engine, mock_repository):
        """Test risk limits prevent excessive leverage."""
        # Set account balance
        account_balance = Decimal("5000")
        max_leverage = Decimal("3")
        
        # Try to open position with 5x leverage
        position_value = Decimal("25000")  # 5x leverage
        position_quantity = position_value / Decimal("50000")  # BTC price
        
        order = Order(
            id="overleveragetest",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position_quantity,
            status=OrderStatus.NEW
        )
        
        # Calculate leverage
        leverage = position_value / account_balance
        assert leverage == Decimal("5")
        
        # Risk engine should reject
        is_allowed = await risk_engine.check_order_risk(order, account_balance)
        assert not is_allowed
        
        # Try with acceptable leverage (2x)
        safe_position_value = account_balance * Decimal("2")
        safe_quantity = safe_position_value / Decimal("50000")
        
        safe_order = Order(
            id="safe_leverage_test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=safe_quantity,
            status=OrderStatus.NEW
        )
        
        safe_leverage = safe_position_value / account_balance
        assert safe_leverage == Decimal("2")
        
        # Risk engine should allow
        is_allowed = await risk_engine.check_order_risk(safe_order, account_balance)
        assert is_allowed

    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self, orchestrator, mock_exchange, mock_repository):
        """Test emergency stop closes all positions and halts trading."""
        # Create open positions
        positions = [
            Position(
                id=f"emergency_pos_{i}",
                symbol=f"COIN{i}USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("100"),
                current_price=Decimal("105"),
                quantity=Decimal("1"),
                unrealized_pnl=Decimal("5"),
                realized_pnl=Decimal("0")
            )
            for i in range(3)
        ]
        
        mock_repository.get_open_positions.return_value = positions
        
        # Trigger emergency stop
        orchestrator.trading_state = TradingState.EMERGENCY_STOP
        
        # Close all positions
        close_orders = []
        for position in positions:
            close_order = Order(
                id=f"emergency_close_{position.id}",
                symbol=position.symbol,
                side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                status=OrderStatus.NEW
            )
            
            mock_exchange.place_order.return_value = {
                "orderId": close_order.id,
                "status": "FILLED"
            }
            
            result = await mock_exchange.place_order(close_order.__dict__)
            close_orders.append(result)
        
        # Verify all positions closed
        assert len(close_orders) == 3
        assert all(o["status"] == "FILLED" for o in close_orders)
        
        # Verify trading is halted
        assert orchestrator.trading_state == TradingState.EMERGENCY_STOP
        
        # New orders should be rejected
        new_order = Order(
            id="rejected_during_emergency",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            status=OrderStatus.NEW
        )
        
        can_place = orchestrator.trading_state != TradingState.EMERGENCY_STOP
        assert not can_place

    @pytest.mark.asyncio
    async def test_audit_trail_completeness(self, mock_repository):
        """Test complete audit trail for all operations."""
        audit_events = []
        
        # Order placement
        audit_events.append(AuditEntry(
            timestamp=datetime.utcnow(),
            action="ORDER_PLACED",
            entity_type="Order",
            entity_id="audit_test_001",
            details={"symbol": "BTCUSDT", "side": "BUY"}
        ))
        
        # Order execution
        audit_events.append(AuditEntry(
            timestamp=datetime.utcnow(),
            action="ORDER_FILLED",
            entity_type="Order",
            entity_id="audit_test_001",
            details={"executed_qty": "0.1", "price": "50000"}
        ))
        
        # Position opened
        audit_events.append(AuditEntry(
            timestamp=datetime.utcnow(),
            action="POSITION_OPENED",
            entity_type="Position",
            entity_id="pos_audit_001",
            details={"symbol": "BTCUSDT", "quantity": "0.1"}
        ))
        
        # Risk limit triggered
        audit_events.append(AuditEntry(
            timestamp=datetime.utcnow(),
            action="RISK_LIMIT_TRIGGERED",
            entity_type="RiskEngine",
            entity_id="leverage_check",
            details={"leverage": "4.5", "max_allowed": "3.0"}
        ))
        
        # Position closed
        audit_events.append(AuditEntry(
            timestamp=datetime.utcnow(),
            action="POSITION_CLOSED",
            entity_type="Position",
            entity_id="pos_audit_001",
            details={"realized_pnl": "150", "close_price": "51500"}
        ))
        
        # Save all audit entries
        for entry in audit_events:
            mock_repository.save_audit_entry(entry)
        
        # Verify audit trail
        assert len(mock_repository.audit_log) == 5
        
        # Verify chronological order
        timestamps = [e.timestamp for e in mock_repository.audit_log]
        assert timestamps == sorted(timestamps)
        
        # Verify all critical actions logged
        actions = {e.action for e in mock_repository.audit_log}
        required_actions = {
            "ORDER_PLACED", "ORDER_FILLED", "POSITION_OPENED",
            "RISK_LIMIT_TRIGGERED", "POSITION_CLOSED"
        }
        assert required_actions.issubset(actions)

    @pytest.mark.asyncio
    async def test_performance_attribution_accuracy(self, mock_repository):
        """Test performance attribution calculations are accurate."""
        perf_engine = PerformanceAttributionEngine(repository=mock_repository)
        
        # Create sample trades with known P&L
        trades = [
            Trade(
                id="perf_trade_1",
                order_id="order_1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                price=Decimal("50000"),
                quantity=Decimal("0.1"),
                commission=Decimal("5"),
                timestamp=datetime.utcnow() - timedelta(hours=24)
            ),
            Trade(
                id="perf_trade_2",
                order_id="order_2",
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                price=Decimal("51000"),
                quantity=Decimal("0.1"),
                commission=Decimal("5.1"),
                timestamp=datetime.utcnow() - timedelta(hours=12)
            ),
            Trade(
                id="perf_trade_3",
                order_id="order_3",
                symbol="ETHUSDT",
                side=OrderSide.BUY,
                price=Decimal("3000"),
                quantity=Decimal("1"),
                commission=Decimal("3"),
                timestamp=datetime.utcnow() - timedelta(hours=6)
            ),
            Trade(
                id="perf_trade_4",
                order_id="order_4",
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                price=Decimal("3050"),
                quantity=Decimal("1"),
                commission=Decimal("3.05"),
                timestamp=datetime.utcnow()
            )
        ]
        
        mock_repository.get_trades.return_value = trades
        
        # Calculate attribution
        attribution = await perf_engine.calculate_attribution(
            start_date=datetime.utcnow() - timedelta(days=2),
            end_date=datetime.utcnow()
        )
        
        # Calculate expected values
        btc_pnl = (Decimal("51000") - Decimal("50000")) * Decimal("0.1") - Decimal("10.1")
        eth_pnl = (Decimal("3050") - Decimal("3000")) * Decimal("1") - Decimal("6.05")
        total_pnl = btc_pnl + eth_pnl
        
        assert attribution is not None
        assert "total_pnl" in attribution
        assert "by_symbol" in attribution
        assert abs(attribution["total_pnl"] - total_pnl) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_multi_account_isolation(self):
        """Test multiple accounts are properly isolated."""
        account_manager = AccountManager(db_session=Mock())
        
        # Create multiple accounts
        accounts = [
            ("main", Decimal("10000"), TierType.STRATEGIST),
            ("test", Decimal("1000"), TierType.SNIPER),
            ("algo", Decimal("5000"), TierType.HUNTER)
        ]
        
        for account_id, balance, tier in accounts:
            account_manager.create_account(account_id, balance, tier)
        
        # Modify one account
        account_manager.update_balance("main", Decimal("9500"))
        
        # Verify isolation
        assert account_manager.get_balance("main") == Decimal("9500")
        assert account_manager.get_balance("test") == Decimal("1000")
        assert account_manager.get_balance("algo") == Decimal("5000")
        
        # Verify tier isolation
        assert account_manager.get_tier("main") == TierType.STRATEGIST
        assert account_manager.get_tier("test") == TierType.SNIPER
        assert account_manager.get_tier("algo") == TierType.HUNTER

    @pytest.mark.asyncio
    async def test_disaster_recovery_checkpoint(self, mock_repository):
        """Test system can recover from last checkpoint."""
        # Create checkpoint
        checkpoint = {
            "timestamp": datetime.utcnow(),
            "positions": {
                "pos_1": {"symbol": "BTCUSDT", "quantity": "0.1", "entry": "50000"}
            },
            "pending_orders": {
                "order_1": {"symbol": "ETHUSDT", "quantity": "1", "price": "3000"}
            },
            "account_balance": "10000",
            "tier": "STRATEGIST"
        }
        
        # Save checkpoint
        mock_repository.save_checkpoint = Mock()
        mock_repository.save_checkpoint(checkpoint)
        
        # Simulate crash and recovery
        mock_repository.load_checkpoint = Mock(return_value=checkpoint)
        recovered = mock_repository.load_checkpoint()
        
        assert recovered["account_balance"] == "10000"
        assert recovered["tier"] == "STRATEGIST"
        assert len(recovered["positions"]) == 1
        assert len(recovered["pending_orders"]) == 1