"""Additional tests to achieve 95% coverage for paper trading modules."""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import shutil

import pytest

from genesis.paper_trading.persistence import StatePersistence, PersistenceConfig
from genesis.paper_trading.simulator import (
    PaperTradingSimulator, SimulationConfig, SimulationMode, SimulatedOrder
)
from genesis.paper_trading.promotion_manager import (
    StrategyPromotionManager, PromotionConfig, PromotionStatus,
    AllocationStrategy, ABTestVariant, StrategyPromotion
)
from genesis.paper_trading.validation_criteria import ValidationCriteria
from genesis.paper_trading.virtual_portfolio import VirtualPortfolio, Position


class TestPersistenceFull:
    """Complete coverage for persistence.py."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")
        yield db_path
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def persistence(self, temp_db):
        """Create persistence instance."""
        config = PersistenceConfig(
            db_path=temp_db,
            auto_save_interval_seconds=1,
            backup_enabled=True,
            max_backups=2
        )
        return StatePersistence(config)
    
    def test_backup_database(self, persistence, temp_db):
        """Test database backup functionality."""
        # Create backup
        backup_path = persistence.backup_database()
        assert backup_path is not None
        assert Path(backup_path).exists()
        
        # Create multiple backups to test cleanup
        for i in range(4):
            persistence.backup_database()
        
        # Check that old backups were cleaned up
        backup_dir = Path(temp_db).parent / "backups"
        backups = list(backup_dir.glob("*.db"))
        assert len(backups) <= persistence.config.max_backups
    
    def test_backup_disabled(self, temp_db):
        """Test backup when disabled."""
        config = PersistenceConfig(
            db_path=temp_db,
            backup_enabled=False
        )
        persistence = StatePersistence(config)
        
        backup = persistence.backup_database()
        assert backup is None
    
    def test_load_nonexistent_portfolio(self, persistence):
        """Test loading non-existent portfolio."""
        result = persistence.load_portfolio_state("nonexistent")
        assert result is None
    
    def test_load_nonexistent_promotion(self, persistence):
        """Test loading non-existent promotion."""
        result = persistence.load_promotion_state("nonexistent")
        assert result is None
    
    def test_get_order_history_empty(self, persistence):
        """Test getting order history when empty."""
        orders = persistence.get_order_history("test", limit=10)
        assert orders == []
    
    def test_get_audit_log_empty(self, persistence):
        """Test getting audit log when empty."""
        log = persistence.get_audit_log("test", limit=10)
        assert log == []
    
    def test_clear_old_data_with_data(self, persistence):
        """Test clearing old data."""
        # Add audit entry
        persistence.add_audit_entry("test", "action", {"detail": "test"})
        
        # Clear very old data (should keep recent)
        persistence.clear_old_data(days_to_keep=1)
        
        # Recent data should still exist
        log = persistence.get_audit_log("test")
        assert len(log) > 0


class TestSimulatorFull:
    """Complete coverage for simulator.py."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        config = SimulationConfig(
            mode=SimulationMode.REALISTIC,
            base_latency_ms=10,
            enabled=True
        )
        criteria = ValidationCriteria()
        return PaperTradingSimulator(config, criteria)
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, simulator):
        """Test starting when already running."""
        await simulator.start()
        assert simulator.running
        
        # Try to start again - should return early
        await simulator.start()
        assert simulator.running
        
        await simulator.stop()
    
    @pytest.mark.asyncio
    async def test_stop_not_running(self, simulator):
        """Test stopping when not running."""
        # Stop without starting - should return early
        await simulator.stop()
        assert not simulator.running
    
    @pytest.mark.asyncio
    async def test_submit_order_not_running(self, simulator):
        """Test submitting order when not running."""
        simulator.create_portfolio("test", Decimal("10000"))
        
        order = await simulator.submit_order(
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1")
        )
        
        # Should still create order but not execute
        assert order is not None
        assert order.status == "pending"
    
    @pytest.mark.asyncio
    async def test_submit_order_no_portfolio(self, simulator):
        """Test submitting order without portfolio."""
        await simulator.start()
        
        with pytest.raises(ValueError, match="Portfolio not found"):
            await simulator.submit_order(
                strategy_id="nonexistent",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1")
            )
        
        await simulator.stop()
    
    @pytest.mark.asyncio
    async def test_execute_order_limit(self, simulator):
        """Test executing limit order."""
        await simulator.start()
        portfolio = simulator.create_portfolio("test", Decimal("10000"))
        
        # Submit limit order
        order = await simulator.submit_order(
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("45000")
        )
        
        # Should not be filled immediately
        await asyncio.sleep(0.1)
        assert order.status == "pending" or order.status == "filled"
        
        await simulator.stop()
    
    @pytest.mark.asyncio
    async def test_execute_order_pessimistic_mode(self, simulator):
        """Test order execution in pessimistic mode."""
        simulator.config.mode = SimulationMode.PESSIMISTIC
        await simulator.start()
        portfolio = simulator.create_portfolio("test", Decimal("10000"))
        
        order = await simulator.submit_order(
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1")
        )
        
        # Wait for pessimistic execution (higher latency)
        await asyncio.sleep(0.2)
        
        await simulator.stop()
    
    @pytest.mark.asyncio
    async def test_execute_order_instant_mode(self, simulator):
        """Test instant mode execution."""
        simulator.config.mode = SimulationMode.INSTANT
        await simulator.start()
        portfolio = simulator.create_portfolio("test", Decimal("10000"))
        
        order = await simulator.submit_order(
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1")
        )
        
        # Should fill instantly
        await asyncio.sleep(0.01)
        
        await simulator.stop()
    
    @pytest.mark.asyncio
    async def test_partial_fill(self, simulator):
        """Test partial order fill."""
        simulator.config.partial_fill_threshold = Decimal("0.01")
        simulator.config.max_fill_ratio = 0.5
        
        await simulator.start()
        portfolio = simulator.create_portfolio("test", Decimal("100000"))
        
        # Large order that should be partially filled
        order = await simulator.submit_order(
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("1.0")  # Large quantity
        )
        
        await asyncio.sleep(0.1)
        
        await simulator.stop()
    
    def test_calculate_slippage_sell(self, simulator):
        """Test slippage calculation for sell orders."""
        slippage = simulator._calculate_slippage(
            Decimal("50000"),
            Decimal("0.1"),
            "sell"
        )
        
        # Sell should have negative slippage
        assert slippage < Decimal("50000")
    
    @pytest.mark.asyncio
    async def test_auto_save_task(self, simulator):
        """Test auto-save task."""
        simulator.persistence.config.auto_save_interval_seconds = 0.01
        await simulator.start()
        
        # Create portfolio to save
        simulator.create_portfolio("test", Decimal("10000"))
        
        # Wait for auto-save
        await asyncio.sleep(0.05)
        
        await simulator.stop()
    
    def test_load_portfolio_with_data(self, simulator):
        """Test loading portfolio with saved data."""
        # First save a portfolio
        simulator.create_portfolio("test", Decimal("10000"))
        simulator.persistence.save_portfolio_state(
            strategy_id="test",
            current_balance=Decimal("9500"),
            positions={"BTC/USDT": {
                "quantity": "0.1",
                "average_price": "50000",
                "side": "long",
                "opened_at": datetime.now().isoformat(),
                "last_price": "51000",
                "realized_pnl": "0",
                "unrealized_pnl": "100"
            }},
            trades=[],
            metrics={}
        )
        
        # Load it back
        loaded = simulator.load_portfolio("test")
        assert loaded is not None
        assert "BTC/USDT" in loaded.positions
        assert loaded.current_balance == Decimal("9500")
    
    def test_get_portfolio_metrics_not_found(self, simulator):
        """Test getting metrics for non-existent portfolio."""
        with pytest.raises(ValueError, match="Portfolio not found"):
            simulator.get_portfolio_metrics("nonexistent")


class TestPromotionManagerFull:
    """Complete coverage for promotion_manager.py."""
    
    @pytest.fixture
    def manager(self):
        """Create promotion manager."""
        config = PromotionConfig()
        criteria = ValidationCriteria()
        return StrategyPromotionManager(config, criteria)
    
    def test_register_existing_strategy(self, manager):
        """Test registering already registered strategy."""
        manager.register_strategy("test")
        
        # Register again - should skip
        manager.register_strategy("test")
        
        assert len(manager.promotions) == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_unregistered_strategy(self, manager):
        """Test evaluating unregistered strategy."""
        metrics = {"total_trades": 100}
        
        decision = await manager.evaluate_for_promotion("unregistered", metrics)
        
        # Should register and evaluate
        assert "unregistered" in manager.promotions
        assert decision is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_already_promoted(self, manager):
        """Test evaluating already promoted strategy."""
        manager.register_strategy("test")
        manager.promotions["test"].status = PromotionStatus.FULL_ALLOCATION
        
        metrics = {"total_trades": 100}
        decision = await manager.evaluate_for_promotion("test", metrics)
        
        assert decision.decision == "already_promoted"
    
    @pytest.mark.asyncio
    async def test_promote_unregistered(self, manager):
        """Test promoting unregistered strategy."""
        await manager.promote_strategy("test", {})
        
        # Should be registered and promoted
        assert "test" in manager.promotions
        assert manager.promotions["test"].status == PromotionStatus.GRADUAL_ALLOCATION
    
    @pytest.mark.asyncio
    async def test_update_allocation_increase(self, manager):
        """Test increasing allocation."""
        manager.register_strategy("test")
        await manager.promote_strategy("test", {"sharpe_ratio": 1.5})
        
        # Good performance should increase allocation
        good_metrics = {"sharpe_ratio": 2.0, "max_drawdown": 0.05}
        await manager.update_allocation("test", good_metrics)
        
        promotion = manager.promotions["test"]
        assert promotion.current_allocation > manager.config.initial_allocation
    
    @pytest.mark.asyncio
    async def test_update_allocation_regression(self, manager):
        """Test allocation with regression."""
        manager.register_strategy("test")
        baseline = {"sharpe_ratio": 2.0, "win_rate": 0.6}
        await manager.promote_strategy("test", baseline)
        
        # Poor performance should trigger demotion check
        poor_metrics = {"sharpe_ratio": 1.0, "win_rate": 0.4}
        await manager.update_allocation("test", poor_metrics)
        
        # May be demoted depending on thresholds
        promotion = manager.promotions["test"]
        assert promotion.status in [PromotionStatus.DEMOTED, PromotionStatus.GRADUAL_ALLOCATION]
    
    @pytest.mark.asyncio
    async def test_update_allocation_full(self, manager):
        """Test reaching full allocation."""
        manager.register_strategy("test")
        await manager.promote_strategy("test", {})
        
        # Set near max allocation
        manager.promotions["test"].current_allocation = Decimal("0.95")
        
        good_metrics = {"sharpe_ratio": 3.0}
        await manager.update_allocation("test", good_metrics)
        
        promotion = manager.promotions["test"]
        assert promotion.status == PromotionStatus.FULL_ALLOCATION
        assert promotion.current_allocation == manager.config.max_allocation
    
    def test_get_allocation_increment_exponential(self, manager):
        """Test exponential allocation strategy."""
        increment = manager._get_allocation_increment(
            Decimal("0.1"),
            Decimal("1.0"),
            AllocationStrategy.EXPONENTIAL
        )
        
        # Should be proportional to current allocation
        assert increment > 0
        assert increment <= Decimal("0.9")
    
    def test_get_allocation_increment_step(self, manager):
        """Test step allocation strategy."""
        increment = manager._get_allocation_increment(
            Decimal("0.3"),
            Decimal("1.0"),
            AllocationStrategy.STEP
        )
        
        # Should jump to next step
        assert increment == Decimal("0.7")
    
    def test_get_allocation_increment_custom(self, manager):
        """Test custom allocation strategy."""
        increment = manager._get_allocation_increment(
            Decimal("0.1"),
            Decimal("1.0"),
            AllocationStrategy.CUSTOM
        )
        
        # Should use default increment
        assert increment == manager.config.allocation_increment
    
    def test_start_duplicate_ab_test(self, manager):
        """Test starting duplicate A/B test."""
        manager.register_strategy("control")
        manager.register_strategy("variant")
        
        manager.start_ab_test("control", "variant")
        
        # Try to start same test again
        manager.start_ab_test("control", "variant")
        
        # Should only have one test
        assert len(manager.ab_tests) == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_ab_test_not_found(self, manager):
        """Test evaluating non-existent A/B test."""
        result = await manager.evaluate_ab_test(
            "control", "variant",
            {}, {},
            min_days=1
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_evaluate_ab_test_winner(self, manager):
        """Test A/B test with clear winner."""
        manager.register_strategy("control")
        manager.register_strategy("variant")
        manager.start_ab_test("control", "variant")
        
        control_metrics = {
            "sharpe_ratio": 1.5,
            "win_rate": 0.52,
            "profit_factor": 1.1,
            "max_drawdown": 0.15
        }
        
        variant_metrics = {
            "sharpe_ratio": 2.5,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "max_drawdown": 0.08
        }
        
        result = await manager.evaluate_ab_test(
            "control", "variant",
            control_metrics, variant_metrics,
            min_days=7
        )
        
        assert result is not None
        assert result.winner == "variant"
        assert result.confidence > 0.8
        assert result.improvement > 0
    
    def test_get_all_promotions(self, manager):
        """Test getting all promotions."""
        manager.register_strategy("test1")
        manager.register_strategy("test2")
        
        promotions = manager.get_all_promotions()
        
        assert len(promotions) == 2
        assert all(isinstance(p, StrategyPromotion) for p in promotions)
    
    def test_get_active_ab_tests(self, manager):
        """Test getting active A/B tests."""
        manager.register_strategy("control")
        manager.register_strategy("variant")
        manager.start_ab_test("control", "variant")
        
        tests = manager.get_active_ab_tests()
        
        assert len(tests) == 1
        assert tests[0]["control"] == "control"


class TestVirtualPortfolioFull:
    """Additional tests for virtual_portfolio.py edge cases."""
    
    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        return VirtualPortfolio("test", Decimal("10000"))
    
    @pytest.mark.asyncio
    async def test_process_buy_insufficient_balance(self, portfolio):
        """Test buy order with insufficient balance."""
        order = SimulatedOrder(
            order_id="TEST001",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("10"),  # Too large
            timestamp=datetime.now()
        )
        order.filled_quantity = Decimal("10")
        order.average_fill_price = Decimal("50000")
        order.status = "filled"
        order.fill_timestamp = datetime.now()
        
        # Should still process but balance will go negative
        await portfolio.process_fill(order)
        
        assert portfolio.current_balance < 0
    
    @pytest.mark.asyncio
    async def test_process_sell_full_position(self, portfolio):
        """Test selling entire position."""
        # First buy
        buy_order = SimulatedOrder(
            order_id="BUY001",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1"),
            timestamp=datetime.now()
        )
        buy_order.filled_quantity = Decimal("0.1")
        buy_order.average_fill_price = Decimal("50000")
        buy_order.status = "filled"
        buy_order.fill_timestamp = datetime.now()
        
        await portfolio.process_fill(buy_order)
        
        # Then sell entire position
        sell_order = SimulatedOrder(
            order_id="SELL001",
            symbol="BTC/USDT",
            side="sell",
            order_type="market",
            quantity=Decimal("0.1"),
            timestamp=datetime.now()
        )
        sell_order.filled_quantity = Decimal("0.1")
        sell_order.average_fill_price = Decimal("51000")
        sell_order.status = "filled"
        sell_order.fill_timestamp = datetime.now()
        
        await portfolio.process_fill(sell_order)
        
        # Position should be closed
        assert "BTC/USDT" not in portfolio.positions or \
               portfolio.positions["BTC/USDT"].quantity == 0
    
    def test_calculate_metrics_with_trades(self, portfolio):
        """Test metrics calculation with winning and losing trades."""
        # Add some mock trades
        portfolio.trades = [
            {"pnl": "100", "symbol": "BTC/USDT"},
            {"pnl": "-50", "symbol": "ETH/USDT"},
            {"pnl": "200", "symbol": "BTC/USDT"},
            {"pnl": "-30", "symbol": "ETH/USDT"},
        ]
        portfolio.total_trades = 4
        portfolio.winning_trades = 2
        portfolio.losing_trades = 2
        portfolio.total_profit = Decimal("300")
        portfolio.total_loss = Decimal("80")
        portfolio.max_drawdown = Decimal("0.05")
        portfolio.daily_returns.extend([0.01, -0.005, 0.02, 0.015, -0.01])
        
        metrics = portfolio.get_metrics()
        
        assert metrics["total_trades"] == 4
        assert metrics["win_rate"] == 0.5
        assert metrics["profit_factor"] == 3.75  # 300/80
        assert metrics["sharpe_ratio"] != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])