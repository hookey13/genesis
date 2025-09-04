"""Integration tests for paper trading system."""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from genesis.paper_trading.simulator import (
    PaperTradingSimulator,
    SimulationConfig,
    SimulationMode,
)
from genesis.paper_trading.persistence import PersistenceConfig, StatePersistence
from genesis.paper_trading.virtual_portfolio import VirtualPortfolio
from genesis.paper_trading.validation_criteria import ValidationCriteria
from genesis.paper_trading.promotion_manager import (
    StrategyPromotionManager,
    AllocationConfig,
    ABTestConfig,
    PromotionStatus,
    ABTestVariant,
)


@pytest.fixture
async def simulator():
    """Create a paper trading simulator instance."""
    config = SimulationConfig(
        mode=SimulationMode.REALISTIC, base_latency_ms=10.0, base_slippage_bps=2.0
    )
    criteria = ValidationCriteria(
        min_trades=10, min_days=1, min_sharpe=1.0, max_drawdown=0.20
    )
    sim = PaperTradingSimulator(config, criteria)
    await sim.start()
    yield sim
    await sim.stop()


@pytest.fixture
def promotion_manager():
    """Create a strategy promotion manager instance."""
    criteria = ValidationCriteria(min_trades=10, min_days=1, min_sharpe=1.0)
    allocation_config = AllocationConfig(
        auto_promote=True,
        initial_allocation=Decimal("0.10"),
        allocation_increment=Decimal("0.10"),
    )
    ab_config = ABTestConfig(enabled=True, max_variants=3)
    return StrategyPromotionManager(criteria, allocation_config, ab_config)


@pytest.mark.asyncio
async def test_paper_trading_simulation(simulator):
    """Test basic paper trading simulation flow."""
    strategy_id = "test_strategy_1"
    initial_balance = Decimal("10000")

    portfolio = simulator.create_portfolio(strategy_id, initial_balance)
    assert portfolio.strategy_id == strategy_id
    assert portfolio.current_balance == initial_balance

    order = await simulator.submit_order(
        strategy_id=strategy_id,
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        quantity=Decimal("0.1"),
        price=Decimal("50000"),
    )

    assert order.order_id.startswith("SIM-")
    assert order.status == "pending"

    await asyncio.sleep(0.1)

    assert order.status == "filled"
    assert order.filled_quantity > Decimal("0")
    assert order.latency_ms is not None
    assert order.slippage is not None


@pytest.mark.asyncio
async def test_virtual_portfolio_tracking(simulator):
    """Test virtual portfolio position tracking and P&L calculation."""
    strategy_id = "test_strategy_2"
    initial_balance = Decimal("10000")

    portfolio = simulator.create_portfolio(strategy_id, initial_balance)

    buy_order = await simulator.submit_order(
        strategy_id=strategy_id,
        symbol="ETH/USDT",
        side="buy",
        order_type="market",
        quantity=Decimal("1"),
        price=Decimal("3000"),
    )

    await asyncio.sleep(0.1)

    position = portfolio.get_position("ETH/USDT")
    assert position is not None
    assert position.quantity == buy_order.filled_quantity
    assert position.side == "long"

    await portfolio.update_market_prices({"ETH/USDT": Decimal("3100")})

    metrics = portfolio.get_metrics()
    assert metrics["positions"] == 1
    assert metrics["total_trades"] == 1

    sell_order = await simulator.submit_order(
        strategy_id=strategy_id,
        symbol="ETH/USDT",
        side="sell",
        order_type="market",
        quantity=position.quantity,
        price=Decimal("3100"),
    )

    await asyncio.sleep(0.1)

    position = portfolio.get_position("ETH/USDT")
    assert position is None

    final_metrics = portfolio.get_metrics()
    assert final_metrics["total_trades"] == 2
    assert final_metrics["winning_trades"] == 1


@pytest.mark.asyncio
async def test_validation_criteria():
    """Test validation criteria checking."""
    criteria = ValidationCriteria(
        min_trades=100, min_days=7, min_sharpe=1.5, max_drawdown=0.10, min_win_rate=0.55
    )

    good_metrics = {
        "total_trades": 150,
        "days_running": 10,
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.08,
        "win_rate": 0.60,
        "profit_factor": 1.5,
    }

    assert criteria.is_eligible(good_metrics) is True

    bad_metrics = {
        "total_trades": 50,
        "days_running": 5,
        "sharpe_ratio": 1.0,
        "max_drawdown": 0.15,
        "win_rate": 0.50,
        "profit_factor": 0.9,
    }

    assert criteria.is_eligible(bad_metrics) is False

    report = criteria.get_eligibility_report(bad_metrics)
    assert report["eligible"] is False
    assert report["criteria_status"]["trades"]["passed"] is False
    assert report["criteria_status"]["sharpe_ratio"]["passed"] is False


@pytest.mark.asyncio
async def test_strategy_promotion(promotion_manager):
    """Test strategy promotion workflow."""
    strategy_id = "test_strategy_3"

    promotion = promotion_manager.register_strategy(strategy_id)
    assert promotion.status == PromotionStatus.PAPER

    good_metrics = {
        "strategy_id": strategy_id,
        "total_trades": 150,
        "days_running": 10,
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.08,
        "win_rate": 0.60,
        "profit_factor": 1.5,
    }

    eligible = await promotion_manager.check_promotion_eligibility(
        strategy_id, good_metrics
    )
    assert eligible is True

    promoted = await promotion_manager.promote_strategy(strategy_id, good_metrics)
    assert promoted is True

    promotion = promotion_manager.strategies[strategy_id]
    assert promotion.status == PromotionStatus.GRADUAL_ALLOCATION
    assert promotion.current_allocation == Decimal("0.10")

    adjusted = await promotion_manager.adjust_allocation(strategy_id, Decimal("0.50"))
    assert adjusted is True
    assert promotion.current_allocation == Decimal("0.50")


@pytest.mark.asyncio
async def test_ab_testing_framework(promotion_manager):
    """Test A/B testing functionality."""
    ab_group = "test_ab_group"

    control_strategy = promotion_manager.register_strategy(
        "control_strategy", ABTestVariant.CONTROL, ab_group
    )

    variant_a_strategy = promotion_manager.register_strategy(
        "variant_a_strategy", ABTestVariant.VARIANT_A, ab_group
    )

    variant_b_strategy = promotion_manager.register_strategy(
        "variant_b_strategy", ABTestVariant.VARIANT_B, ab_group
    )

    assert control_strategy.ab_test_variant == ABTestVariant.CONTROL
    assert variant_a_strategy.ab_test_variant == ABTestVariant.VARIANT_A
    assert variant_b_strategy.ab_test_variant == ABTestVariant.VARIANT_B

    results = promotion_manager.get_ab_test_results(ab_group)
    assert "variants" in results
    assert len(results["variants"]) == 3
    assert ABTestVariant.CONTROL.value in results["variants"]


@pytest.mark.asyncio
async def test_regression_detection(promotion_manager):
    """Test performance regression detection."""
    strategy_id = "test_strategy_4"

    promotion = promotion_manager.register_strategy(strategy_id)

    baseline_metrics = {
        "strategy_id": strategy_id,
        "sharpe_ratio": 2.0,
        "win_rate": 0.60,
        "profit_factor": 1.5,
        "max_drawdown": 0.08,
    }

    promoted = await promotion_manager.promote_strategy(
        strategy_id, baseline_metrics, force=True
    )
    assert promoted is True

    current_metrics = {
        "strategy_id": strategy_id,
        "sharpe_ratio": 1.4,
        "win_rate": 0.45,
        "profit_factor": 1.0,
        "max_drawdown": 0.15,
    }

    regression_detected = promotion_manager.check_regression(
        strategy_id, current_metrics
    )
    assert regression_detected is True


@pytest.mark.asyncio
async def test_strategy_demotion(promotion_manager):
    """Test strategy demotion functionality."""
    strategy_id = "test_strategy_5"

    promotion = promotion_manager.register_strategy(strategy_id)

    metrics = {
        "strategy_id": strategy_id,
        "total_trades": 150,
        "days_running": 10,
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.08,
        "win_rate": 0.60,
        "profit_factor": 1.5,
    }

    await promotion_manager.promote_strategy(strategy_id, metrics)
    assert promotion.status == PromotionStatus.GRADUAL_ALLOCATION

    demoted = await promotion_manager.demote_strategy(
        strategy_id, "Performance regression detected"
    )
    assert demoted is True
    assert promotion.status == PromotionStatus.DEMOTED
    assert promotion.current_allocation == Decimal("0")


@pytest.mark.asyncio
async def test_audit_trail(simulator, promotion_manager):
    """Test audit trail functionality."""
    strategy_id = "test_strategy_6"
    initial_balance = Decimal("10000")

    portfolio = simulator.create_portfolio(strategy_id, initial_balance)
    promotion = promotion_manager.register_strategy(strategy_id)

    for i in range(15):
        await simulator.submit_order(
            strategy_id=strategy_id,
            symbol="BTC/USDT",
            side="buy" if i % 2 == 0 else "sell",
            order_type="market",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )

    await asyncio.sleep(0.2)

    audit_data = await simulator.export_audit_trail(strategy_id)
    assert "strategy_id" in audit_data
    assert "portfolio_metrics" in audit_data
    assert "trades" in audit_data
    assert "orders" in audit_data
    assert len(audit_data["orders"]) == 15

    promotion_report = promotion_manager.export_promotion_report(strategy_id)
    assert "audit_trail" in promotion_report
    assert len(promotion_report["audit_trail"]) > 0


@pytest.mark.asyncio
async def test_partial_fills(simulator):
    """Test partial order fills for large orders."""
    config = SimulationConfig(
        mode=SimulationMode.REALISTIC,
        partial_fill_threshold=Decimal("100"),
        max_fill_ratio=0.5,
    )
    criteria = ValidationCriteria()
    sim = PaperTradingSimulator(config, criteria)
    await sim.start()

    strategy_id = "test_strategy_7"
    portfolio = sim.create_portfolio(strategy_id, Decimal("100000"))

    large_order = await sim.submit_order(
        strategy_id=strategy_id,
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        quantity=Decimal("200"),
        price=Decimal("50000"),
    )

    await asyncio.sleep(0.1)

    assert large_order.filled_quantity < large_order.quantity
    assert large_order.filled_quantity <= large_order.quantity * Decimal("0.5")

    await sim.stop()


@pytest.mark.asyncio
async def test_gradual_allocation_increment(promotion_manager):
    """Test gradual allocation increment over time."""
    strategy_id = "test_strategy_8"

    allocation_config = AllocationConfig(
        auto_promote=True,
        initial_allocation=Decimal("0.10"),
        allocation_increment=Decimal("0.20"),
        max_allocation=Decimal("1.00"),
        increment_interval_days=0,
        performance_check_interval_hours=0.01,
    )

    manager = StrategyPromotionManager(
        ValidationCriteria(min_trades=1, min_days=0), allocation_config
    )

    promotion = manager.register_strategy(strategy_id)

    metrics = {
        "strategy_id": strategy_id,
        "total_trades": 10,
        "days_running": 1,
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.08,
        "win_rate": 0.60,
        "profit_factor": 1.5,
    }

    await manager.promote_strategy(strategy_id, metrics)
    assert promotion.current_allocation == Decimal("0.10")

    await manager.adjust_allocation(strategy_id, Decimal("0.30"))
    assert promotion.current_allocation == Decimal("0.30")

    await manager.adjust_allocation(strategy_id, Decimal("1.00"))
    assert promotion.status == PromotionStatus.FULL_ALLOCATION

    await manager.cleanup()


@pytest.mark.asyncio
async def test_portfolio_reset(simulator):
    """Test virtual portfolio reset functionality."""
    strategy_id = "test_strategy_9"
    initial_balance = Decimal("10000")

    portfolio = simulator.create_portfolio(strategy_id, initial_balance)

    await simulator.submit_order(
        strategy_id=strategy_id,
        symbol="ETH/USDT",
        side="buy",
        order_type="market",
        quantity=Decimal("1"),
        price=Decimal("3000"),
    )

    await asyncio.sleep(0.1)

    metrics_before = portfolio.get_metrics()
    assert metrics_before["total_trades"] > 0
    assert metrics_before["positions"] > 0

    await portfolio.reset()

    metrics_after = portfolio.get_metrics()
    assert metrics_after["total_trades"] == 0
    assert metrics_after["positions"] == 0
    assert metrics_after["current_balance"] == str(initial_balance)


@pytest.mark.asyncio
async def test_confidence_score_calculation():
    """Test confidence score calculation in validation criteria."""
    criteria = ValidationCriteria(min_trades=100, min_days=7, min_sharpe=1.5)

    metrics_low = {
        "total_trades": 100,
        "days_running": 7,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.10,
        "win_rate": 0.55,
        "profit_factor": 1.2,
    }

    metrics_high = {
        "total_trades": 200,
        "days_running": 14,
        "sharpe_ratio": 3.0,
        "max_drawdown": 0.05,
        "win_rate": 0.70,
        "profit_factor": 2.0,
    }

    report_low = criteria.get_eligibility_report(metrics_low)
    report_high = criteria.get_eligibility_report(metrics_high)

    assert report_high["confidence_score"] > report_low["confidence_score"]
    assert 0 <= report_low["confidence_score"] <= 1
    assert 0 <= report_high["confidence_score"] <= 1


@pytest.mark.asyncio
async def test_state_persistence():
    """Test state persistence and recovery."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_paper_trading.db")
        
        persistence_config = PersistenceConfig(
            db_path=db_path,
            auto_save_interval_seconds=1,
            backup_enabled=True,
        )
        
        config = SimulationConfig(mode=SimulationMode.REALISTIC)
        criteria = ValidationCriteria(min_trades=5, min_days=1)
        
        # Create simulator with persistence
        sim1 = PaperTradingSimulator(config, criteria, persistence_config)
        await sim1.start()
        
        strategy_id = "test_persistence"
        initial_balance = Decimal("10000")
        
        portfolio1 = sim1.create_portfolio(strategy_id, initial_balance)
        
        # Execute some trades
        for i in range(3):
            await sim1.submit_order(
                strategy_id=strategy_id,
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                order_type="market",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
            )
        
        await asyncio.sleep(0.2)
        
        # Save state
        await sim1._save_all_portfolios()
        await sim1.stop()
        
        # Ensure persistence is closed before creating new simulator
        sim1.persistence.close()
        
        # Create new simulator and load state
        sim2 = PaperTradingSimulator(config, criteria, persistence_config)
        portfolio2 = sim2.load_portfolio(strategy_id)
        
        assert portfolio2 is not None
        assert portfolio2.strategy_id == strategy_id
        assert len(portfolio2.trades) == 3
        assert portfolio2.current_balance != initial_balance  # Changed due to trades
        
        # Verify persistence can load order history
        order_history = sim2.persistence.get_order_history(strategy_id)
        assert len(order_history) >= 3
        
        # Cleanup
        sim2.persistence.close()


@pytest.mark.asyncio
async def test_database_backup():
    """Test database backup functionality."""
    import tempfile
    import os
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_backup.db")
        
        persistence_config = PersistenceConfig(
            db_path=db_path,
            backup_enabled=True,
            max_backups=3,
        )
        
        persistence = StatePersistence(persistence_config)
        
        # Create some backups
        backup_paths = []
        for _ in range(5):
            backup_path = persistence.backup_database()
            if backup_path:
                backup_paths.append(backup_path)
            await asyncio.sleep(0.01)  # Ensure different timestamps
        
        # Check that only max_backups are kept
        backup_dir = Path(db_path).parent / "backups"
        existing_backups = list(backup_dir.glob("paper_trading_*.db")) if backup_dir.exists() else []
        assert len(existing_backups) <= persistence_config.max_backups
        
        # Cleanup
        persistence.close()


@pytest.mark.asyncio
async def test_audit_persistence():
    """Test audit trail persistence."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_audit.db")
        
        persistence_config = PersistenceConfig(db_path=db_path)
        persistence = StatePersistence(persistence_config)
        
        strategy_id = "test_audit"
        
        # Add audit entries
        persistence.add_audit_entry(
            strategy_id,
            "STRATEGY_CREATED",
            {"initial_balance": "10000"}
        )
        
        persistence.add_audit_entry(
            strategy_id,
            "ORDER_SUBMITTED",
            {"symbol": "BTC/USDT", "quantity": "0.1"}
        )
        
        # Retrieve audit log
        audit_log = persistence.get_audit_log(strategy_id)
        
        assert len(audit_log) == 2
        assert audit_log[0]["action"] == "ORDER_SUBMITTED"  # Most recent first
        assert audit_log[1]["action"] == "STRATEGY_CREATED"
        
        # Cleanup
        persistence.close()
