"""Comprehensive tests for paper trading modules with full coverage."""

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.paper_trading import (
    PaperTradingSimulator,
    StatePersistence,
    StrategyPromotionManager,
    ValidationCriteria,
    VirtualPortfolio,
)
from genesis.paper_trading.persistence import PersistenceConfig
from genesis.paper_trading.simulator import SimulationConfig, SimulationMode, SimulatedOrder
from genesis.paper_trading.promotion_manager import (
    ABTestResult,
    ABTestVariant,
    AllocationStrategy,
    PromotionConfig,
    PromotionDecision,
    PromotionStatus,
    StrategyPromotion,
)
from genesis.paper_trading.validation_criteria import (
    CriteriaResult,
    ValidationResult,
)
from genesis.paper_trading.virtual_portfolio import Position, Trade


class TestVirtualPortfolio:
    """Test VirtualPortfolio class comprehensively."""

    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        return VirtualPortfolio("test_strategy", Decimal("10000"))

    def test_initialization(self, portfolio):
        """Test portfolio initialization."""
        assert portfolio.strategy_id == "test_strategy"
        assert portfolio.initial_balance == Decimal("10000")
        assert portfolio.current_balance == Decimal("10000")
        assert portfolio.balance == Decimal("10000")
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        assert portfolio.total_trades == 0
        assert portfolio.winning_trades == 0
        assert portfolio.losing_trades == 0

    @pytest.mark.asyncio
    async def test_process_buy_order(self, portfolio):
        """Test processing a buy order."""
        order = SimulatedOrder(
            order_id="TEST001",
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1"),
            price=None,
            timestamp=datetime.now(),
        )
        order.filled_quantity = Decimal("0.1")
        order.average_fill_price = Decimal("50000")
        order.status = "filled"
        order.fill_timestamp = datetime.now()

        await portfolio.process_fill(order)

        assert portfolio.current_balance < Decimal("10000")
        assert "BTC/USDT" in portfolio.positions
        assert portfolio.positions["BTC/USDT"].quantity == Decimal("0.1")
        assert portfolio.total_trades == 1

    @pytest.mark.asyncio
    async def test_process_sell_order(self, portfolio):
        """Test processing a sell order."""
        # First buy
        buy_order = SimulatedOrder(
            order_id="BUY001",
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1"),
            price=None,
            timestamp=datetime.now(),
        )
        buy_order.filled_quantity = Decimal("0.1")
        buy_order.average_fill_price = Decimal("50000")
        buy_order.status = "filled"
        buy_order.fill_timestamp = datetime.now()
        
        await portfolio.process_fill(buy_order)

        # Then sell
        sell_order = SimulatedOrder(
            order_id="SELL001",
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side="sell",
            order_type="market",
            quantity=Decimal("0.05"),
            price=None,
            timestamp=datetime.now(),
        )
        sell_order.filled_quantity = Decimal("0.05")
        sell_order.average_fill_price = Decimal("51000")
        sell_order.status = "filled"
        sell_order.fill_timestamp = datetime.now()

        await portfolio.process_fill(sell_order)

        assert portfolio.positions["BTC/USDT"].quantity == Decimal("0.05")
        assert portfolio.total_trades == 2
        assert portfolio.current_balance > Decimal("5000")

    def test_get_metrics(self, portfolio):
        """Test getting portfolio metrics."""
        metrics = portfolio.get_metrics()
        
        assert "total_value" in metrics
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_pnl" in metrics
        assert "unrealized_pnl" in metrics
        assert "realized_pnl" in metrics

    def test_calculate_total_value(self, portfolio):
        """Test calculating total portfolio value."""
        total = portfolio.calculate_total_value()
        assert total == Decimal("10000")

        # Add a position
        portfolio.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            average_price=Decimal("50000"),
            side="long",
            opened_at=datetime.now(),
            last_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
        )
        portfolio.current_balance = Decimal("5000")

        total = portfolio.calculate_total_value()
        assert total == Decimal("10100")  # 5000 + 5100

    def test_position_update_pnl(self):
        """Test Position PnL calculation."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            average_price=Decimal("50000"),
            side="long",
            opened_at=datetime.now(),
        )
        
        position.update_pnl(Decimal("51000"))
        assert position.unrealized_pnl == Decimal("1000")
        
        position.side = "short"
        position.update_pnl(Decimal("49000"))
        assert position.unrealized_pnl == Decimal("1000")

    def test_trade_dataclass(self):
        """Test Trade dataclass."""
        trade = Trade(
            order_id="TEST001",
            timestamp=datetime.now(),
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            value=Decimal("5000"),
            pnl=Decimal("0"),
            balance_before=Decimal("10000"),
            balance_after=Decimal("5000"),
        )
        
        assert trade.order_id == "TEST001"
        assert trade.symbol == "BTC/USDT"
        assert trade.quantity == Decimal("0.1")
        assert trade.value == Decimal("5000")


class TestValidationCriteria:
    """Test ValidationCriteria class comprehensively."""

    @pytest.fixture
    def criteria(self):
        """Create test validation criteria."""
        return ValidationCriteria(
            min_trades=100,
            min_days=7,
            min_sharpe=1.5,
            max_drawdown=0.10,
            min_win_rate=0.55,
            min_profit_factor=1.2,
        )

    def test_initialization(self, criteria):
        """Test criteria initialization."""
        assert criteria.min_trades == 100
        assert criteria.min_days == 7
        assert criteria.min_sharpe == 1.5
        assert criteria.max_drawdown == 0.10
        assert criteria.min_win_rate == 0.55
        assert criteria.min_profit_factor == 1.2

    def test_is_eligible_pass(self, criteria):
        """Test eligibility check passing."""
        metrics = {
            "total_trades": 150,
            "days_running": 10,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.08,
            "win_rate": 0.60,
            "profit_factor": 1.5,
            "strategy_id": "test",
        }
        
        assert criteria.is_eligible(metrics) is True

    def test_is_eligible_fail(self, criteria):
        """Test eligibility check failing."""
        metrics = {
            "total_trades": 50,  # Too few trades
            "days_running": 5,    # Too few days
            "sharpe_ratio": 1.0,  # Too low
            "max_drawdown": 0.15, # Too high
            "win_rate": 0.45,     # Too low
            "profit_factor": 0.9,  # Too low
            "strategy_id": "test",
        }
        
        assert criteria.is_eligible(metrics) is False

    def test_get_eligibility_report(self, criteria):
        """Test getting eligibility report."""
        metrics = {
            "total_trades": 150,
            "days_running": 10,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.08,
            "win_rate": 0.60,
            "profit_factor": 1.5,
            "strategy_id": "test",
        }
        
        report = criteria.get_eligibility_report(metrics)
        
        assert report["eligible"] is True
        assert "criteria_status" in report
        assert "confidence_score" in report
        assert report["confidence_score"] > 0

    def test_calculate_confidence_score(self, criteria):
        """Test confidence score calculation."""
        metrics = {
            "total_trades": 200,  # 2x requirement
            "days_running": 14,   # 2x requirement
            "sharpe_ratio": 3.0,  # 2x requirement
            "max_drawdown": 0.05, # Half of max
            "win_rate": 0.70,     # Well above min
            "profit_factor": 2.0, # Well above min
        }
        
        score = criteria._calculate_confidence_score(metrics)
        assert score > 0.8  # High confidence

    def test_update_criteria(self, criteria):
        """Test updating criteria."""
        criteria.update_criteria({
            "min_trades": 200,
            "min_sharpe": 2.0,
            "invalid_key": 100,  # Should be ignored
        })
        
        assert criteria.min_trades == 200
        assert criteria.min_sharpe == 2.0
        assert criteria.min_days == 7  # Unchanged

    def test_to_dict(self, criteria):
        """Test converting to dictionary."""
        data = criteria.to_dict()
        
        assert data["min_trades"] == 100
        assert data["min_days"] == 7
        assert data["min_sharpe"] == 1.5
        assert data["max_drawdown"] == 0.10

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "min_trades": 150,
            "min_days": 10,
            "min_sharpe": 2.0,
            "max_drawdown": 0.05,
            "min_win_rate": 0.60,
            "min_profit_factor": 1.5,
            "extra_field": "ignored",
        }
        
        criteria = ValidationCriteria.from_dict(data)
        assert criteria.min_trades == 150
        assert criteria.min_days == 10
        assert criteria.min_sharpe == 2.0

    def test_check_regression(self, criteria):
        """Test regression checking."""
        current = {
            "sharpe_ratio": 1.2,
            "win_rate": 0.50,
            "profit_factor": 1.0,
            "max_drawdown": 0.12,
            "strategy_id": "test",
        }
        
        baseline = {
            "sharpe_ratio": 2.0,
            "win_rate": 0.65,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
        }
        
        assert criteria.check_regression(current, baseline) is True

    def test_criteria_result(self):
        """Test CriteriaResult dataclass."""
        result = CriteriaResult(
            name="min_trades",
            passed=True,
            value=150,
            required=100,
            message="Sufficient trades",
        )
        
        assert result.name == "min_trades"
        assert result.passed is True
        assert result.value == 150

    def test_validation_result(self):
        """Test ValidationResult dataclass."""
        criteria_results = [
            CriteriaResult("trades", True, 150, 100),
            CriteriaResult("sharpe", False, 1.0, 1.5),
        ]
        
        result = ValidationResult(
            passed=False,
            criteria_results=criteria_results,
            confidence_score=0.65,
            message="Failed sharpe ratio requirement",
        )
        
        assert result.passed is False
        assert len(result.criteria_results) == 2
        assert result.confidence_score == 0.65


class TestSimulator:
    """Test PaperTradingSimulator comprehensively."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulationConfig(
            mode=SimulationMode.REALISTIC,
            base_latency_ms=10.0,
            latency_std_ms=2.0,
            base_slippage_bps=5.0,
            slippage_std_bps=1.0,
            partial_fill_threshold=Decimal("10000"),
            max_fill_ratio=0.5,
            enabled=True,
        )

    @pytest.fixture
    def validation_criteria(self):
        """Create test validation criteria."""
        return ValidationCriteria(
            min_trades=10,
            min_days=1,
            min_sharpe=1.0,
            max_drawdown=0.20,
            min_win_rate=0.50,
        )

    @pytest.fixture
    def simulator(self, config, validation_criteria):
        """Create test simulator."""
        return PaperTradingSimulator(config, validation_criteria)

    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.running is False
        assert len(simulator.portfolios) == 0
        assert len(simulator.orders) == 0
        assert simulator.order_counter == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, simulator):
        """Test starting and stopping simulator."""
        await simulator.start()
        assert simulator.running is True
        
        await simulator.stop()
        assert simulator.running is False

    def test_create_portfolio(self, simulator):
        """Test creating portfolio."""
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        assert portfolio is not None
        assert portfolio.strategy_id == "test_strategy"
        assert "test_strategy" in simulator.portfolios

    @pytest.mark.asyncio
    async def test_submit_order(self, simulator):
        """Test submitting an order."""
        await simulator.start()
        
        simulator.create_portfolio("test", Decimal("10000"))
        
        order = await simulator.submit_order(
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1"),
        )
        
        assert order is not None
        assert order.order_id in simulator.orders
        assert order.status == "pending"
        
        await simulator.stop()

    def test_calculate_slippage(self, simulator):
        """Test slippage calculation."""
        slippage = simulator._calculate_slippage(
            Decimal("50000"),
            Decimal("0.1"),
            "buy",
        )
        
        # Should add slippage for buy
        assert slippage > Decimal("50000")

    def test_calculate_latency(self, simulator):
        """Test latency calculation."""
        latency = simulator._calculate_latency()
        assert latency > 0
        assert latency >= simulator.config.base_latency_ms

    def test_get_portfolio_metrics(self, simulator):
        """Test getting portfolio metrics."""
        portfolio = simulator.create_portfolio("test", Decimal("10000"))
        metrics = simulator.get_portfolio_metrics("test")
        
        assert metrics is not None
        assert "total_value" in metrics
        assert "total_trades" in metrics

    def test_check_promotion_eligibility(self, simulator):
        """Test promotion eligibility check."""
        portfolio = simulator.create_portfolio("test", Decimal("10000"))
        
        eligible = simulator.check_promotion_eligibility("test")
        assert eligible is False  # No trades yet

    def test_load_portfolio(self, simulator):
        """Test loading portfolio from persistence."""
        # First save a portfolio
        portfolio = simulator.create_portfolio("test", Decimal("10000"))
        
        # Try to load it (will return None as not persisted)
        loaded = simulator.load_portfolio("nonexistent")
        assert loaded is None

    def test_simulation_mode_enum(self):
        """Test SimulationMode enum."""
        assert SimulationMode.INSTANT.value == "instant"
        assert SimulationMode.REALISTIC.value == "realistic"
        assert SimulationMode.PESSIMISTIC.value == "pessimistic"

    def test_simulated_order(self):
        """Test SimulatedOrder class."""
        order = SimulatedOrder(
            order_id="TEST001",
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
        )
        
        assert order.order_id == "TEST001"
        assert order.status == "pending"
        assert order.filled_quantity is None
        
        order.update_fill(
            filled_quantity=Decimal("0.1"),
            average_price=Decimal("50000"),
        )
        
        assert order.filled_quantity == Decimal("0.1")
        assert order.average_fill_price == Decimal("50000")


class TestPromotionManager:
    """Test StrategyPromotionManager comprehensively."""

    @pytest.fixture
    def config(self):
        """Create test promotion config."""
        return PromotionConfig(
            auto_promote=True,
            initial_allocation=Decimal("0.1"),
            allocation_increment=Decimal("0.1"),
            max_allocation=Decimal("1.0"),
        )

    @pytest.fixture
    def validation_criteria(self):
        """Create test validation criteria."""
        return ValidationCriteria()

    @pytest.fixture
    def manager(self, config, validation_criteria):
        """Create test promotion manager."""
        return StrategyPromotionManager(config, validation_criteria)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config.auto_promote is True
        assert len(manager.promotions) == 0
        assert len(manager.ab_tests) == 0

    def test_register_strategy(self, manager):
        """Test registering a strategy."""
        manager.register_strategy("test_strategy")
        
        assert "test_strategy" in manager.promotions
        promotion = manager.promotions["test_strategy"]
        assert promotion.status == PromotionStatus.PAPER

    @pytest.mark.asyncio
    async def test_evaluate_for_promotion(self, manager):
        """Test evaluating strategy for promotion."""
        manager.register_strategy("test")
        
        metrics = {
            "total_trades": 150,
            "days_running": 10,
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.08,
            "win_rate": 0.60,
            "profit_factor": 1.5,
        }
        
        decision = await manager.evaluate_for_promotion("test", metrics)
        assert decision is not None
        assert decision.strategy_id == "test"
        assert decision.decision == "promote"

    @pytest.mark.asyncio
    async def test_promote_strategy(self, manager):
        """Test promoting a strategy."""
        manager.register_strategy("test")
        
        baseline_metrics = {
            "sharpe_ratio": 2.0,
            "max_drawdown": 0.08,
        }
        
        await manager.promote_strategy("test", baseline_metrics)
        
        promotion = manager.promotions["test"]
        assert promotion.status == PromotionStatus.GRADUAL_ALLOCATION
        assert promotion.current_allocation > Decimal("0")

    @pytest.mark.asyncio
    async def test_update_allocation(self, manager):
        """Test updating allocation."""
        manager.register_strategy("test")
        await manager.promote_strategy("test", {})
        
        metrics = {"sharpe_ratio": 2.5}
        await manager.update_allocation("test", metrics)
        
        promotion = manager.promotions["test"]
        assert promotion.current_allocation >= manager.config.initial_allocation

    @pytest.mark.asyncio
    async def test_demote_strategy(self, manager):
        """Test demoting a strategy."""
        manager.register_strategy("test")
        await manager.promote_strategy("test", {})
        
        await manager.demote_strategy("test", "Poor performance")
        
        promotion = manager.promotions["test"]
        assert promotion.status == PromotionStatus.DEMOTED
        assert promotion.current_allocation == Decimal("0")

    def test_get_allocation_increment(self, manager):
        """Test allocation increment calculation."""
        # Linear strategy
        increment = manager._get_allocation_increment(
            Decimal("0.1"),
            Decimal("1.0"),
            AllocationStrategy.LINEAR,
        )
        assert increment == manager.config.allocation_increment
        
        # Step strategy
        increment = manager._get_allocation_increment(
            Decimal("0.5"),
            Decimal("1.0"),
            AllocationStrategy.STEP,
        )
        assert increment == Decimal("0.5")

    def test_start_ab_test(self, manager):
        """Test starting A/B test."""
        manager.register_strategy("control")
        manager.register_strategy("variant")
        
        manager.start_ab_test("control", "variant")
        
        assert len(manager.ab_tests) == 1
        test = manager.ab_tests[0]
        assert test["control"] == "control"
        assert test["variant"] == "variant"

    @pytest.mark.asyncio
    async def test_evaluate_ab_test(self, manager):
        """Test evaluating A/B test."""
        manager.register_strategy("control")
        manager.register_strategy("variant")
        manager.start_ab_test("control", "variant")
        
        control_metrics = {"sharpe_ratio": 1.5, "win_rate": 0.55}
        variant_metrics = {"sharpe_ratio": 2.0, "win_rate": 0.60}
        
        result = await manager.evaluate_ab_test(
            "control", "variant",
            control_metrics, variant_metrics,
            min_days=7,
        )
        
        assert result is not None
        assert result.winner == "variant"

    def test_enums(self):
        """Test promotion enums."""
        assert PromotionStatus.PAPER.value == "paper"
        assert AllocationStrategy.LINEAR.value == "linear"
        assert ABTestVariant.CONTROL.value == "control"

    def test_dataclasses(self):
        """Test promotion dataclasses."""
        # PromotionDecision
        decision = PromotionDecision(
            strategy_id="test",
            decision="promote",
            reason="Met all criteria",
            timestamp=datetime.now(),
            metrics={},
        )
        assert decision.strategy_id == "test"
        
        # ABTestResult
        result = ABTestResult(
            control_metrics={},
            variant_metrics={},
            winner="variant",
            confidence=0.95,
            improvement=0.20,
            test_duration_days=14,
        )
        assert result.winner == "variant"
        assert result.confidence == 0.95


class TestPersistence:
    """Test StatePersistence comprehensively."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        yield temp_path
        try:
            os.unlink(temp_path)
        except:
            pass

    @pytest.fixture
    def config(self, temp_db):
        """Create test persistence config."""
        return PersistenceConfig(
            db_path=temp_db,
            auto_save_interval_seconds=1,
            backup_enabled=False,
        )

    @pytest.fixture
    def persistence(self, config):
        """Create test persistence."""
        return StatePersistence(config)

    def test_initialization(self, persistence, temp_db):
        """Test persistence initialization."""
        assert persistence.config.db_path == temp_db
        assert Path(temp_db).exists()
        
        # Check tables were created
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert "portfolio_state" in tables
            assert "promotion_state" in tables
            assert "order_history" in tables
            assert "audit_log" in tables

    def test_save_load_portfolio_state(self, persistence):
        """Test saving and loading portfolio state."""
        persistence.save_portfolio_state(
            strategy_id="test",
            current_balance=Decimal("9500"),
            positions={"BTC/USDT": {"quantity": "0.1"}},
            trades=[{"order_id": "TEST001"}],
            metrics={"total_trades": 1},
        )
        
        state = persistence.load_portfolio_state("test")
        assert state is not None
        
        balance, positions, trades, metrics = state
        assert balance == Decimal("9500")
        assert "BTC/USDT" in positions
        assert len(trades) == 1
        assert metrics["total_trades"] == 1

    def test_save_load_promotion_state(self, persistence):
        """Test saving and loading promotion state."""
        persistence.save_promotion_state(
            strategy_id="test",
            status="gradual_allocation",
            current_allocation=Decimal("0.2"),
            target_allocation=Decimal("1.0"),
            baseline_metrics={"sharpe_ratio": 2.0},
            promotion_history=[{"action": "promoted"}],
            audit_trail=[{"timestamp": "2025-01-01"}],
        )
        
        state = persistence.load_promotion_state("test")
        assert state is not None
        assert state["status"] == "gradual_allocation"
        assert state["current_allocation"] == Decimal("0.2")
        assert state["baseline_metrics"]["sharpe_ratio"] == 2.0

    def test_save_order(self, persistence):
        """Test saving order."""
        order = SimulatedOrder(
            order_id="TEST001",
            strategy_id="test",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("0.1"),
            price=None,
            timestamp=datetime.now(),
        )
        order.filled_quantity = Decimal("0.1")
        order.average_fill_price = Decimal("50000")
        order.status = "filled"
        order.fill_timestamp = datetime.now()
        order.latency_ms = 10.5
        order.slippage = Decimal("5")
        
        persistence.save_order(order, "test")
        
        # Verify it was saved
        orders = persistence.get_order_history("test")
        assert len(orders) > 0
        assert orders[0]["order_id"] == "TEST001"

    def test_add_audit_entry(self, persistence):
        """Test adding audit entry."""
        persistence.add_audit_entry(
            strategy_id="test",
            action="strategy_promoted",
            details={"reason": "Met criteria"},
        )
        
        log = persistence.get_audit_log("test")
        assert len(log) > 0
        assert log[0]["action"] == "strategy_promoted"

    def test_get_order_history(self, persistence):
        """Test getting order history."""
        # Save multiple orders
        for i in range(5):
            order = SimulatedOrder(
                order_id=f"TEST{i:03d}",
                strategy_id="test",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
                price=None,
                timestamp=datetime.now(),
            )
            persistence.save_order(order, "test")
        
        history = persistence.get_order_history("test", limit=3)
        assert len(history) == 3

    def test_clear_old_data(self, persistence):
        """Test clearing old data."""
        # Add some data
        persistence.add_audit_entry("test", "action", {})
        
        # Clear data older than 30 days (nothing should be cleared)
        persistence.clear_old_data(days_to_keep=30)
        
        # Verify data still exists
        log = persistence.get_audit_log("test")
        assert len(log) > 0

    def test_close(self, persistence):
        """Test closing connections."""
        persistence.close()
        # Should not raise any errors

    def test_destructor(self, config):
        """Test destructor calls close."""
        persistence = StatePersistence(config)
        del persistence
        # Should not raise any errors


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])