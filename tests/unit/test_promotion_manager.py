"""Unit tests for strategy promotion manager module."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.paper_trading.promotion_manager import (
    ABTestResult,
    AllocationStrategy,
    PromotionConfig,
    PromotionDecision,
    StrategyPromotion,
    StrategyPromotionManager,
)
from genesis.paper_trading.validation_criteria import ValidationCriteria, ValidationResult


class TestPromotionConfig:
    """Test PromotionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PromotionConfig()
        assert config.auto_promote is True
        assert config.initial_allocation == Decimal("0.10")
        assert config.allocation_increment == Decimal("0.10")
        assert config.max_allocation == Decimal("1.00")
        assert config.regression_threshold == Decimal("0.20")
        assert config.ab_testing_enabled is True
        assert config.max_ab_variants == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PromotionConfig(
            auto_promote=False,
            initial_allocation=Decimal("0.05"),
            allocation_increment=Decimal("0.05"),
            max_allocation=Decimal("0.50"),
            regression_threshold=Decimal("0.15"),
        )
        assert config.auto_promote is False
        assert config.initial_allocation == Decimal("0.05")
        assert config.allocation_increment == Decimal("0.05")
        assert config.max_allocation == Decimal("0.50")
        assert config.regression_threshold == Decimal("0.15")


class TestStrategyPromotion:
    """Test StrategyPromotion dataclass."""

    def test_promotion_creation(self):
        """Test creating a strategy promotion."""
        promotion = StrategyPromotion(
            strategy_id="test_strategy",
            promoted_at=datetime.now(),
            initial_allocation=Decimal("0.10"),
            current_allocation=Decimal("0.20"),
            performance_baseline={"sharpe": 1.5, "win_rate": 0.60},
            promotion_reason="Met all validation criteria",
        )
        
        assert promotion.strategy_id == "test_strategy"
        assert promotion.initial_allocation == Decimal("0.10")
        assert promotion.current_allocation == Decimal("0.20")
        assert promotion.performance_baseline["sharpe"] == 1.5
        assert promotion.promotion_reason == "Met all validation criteria"

    def test_allocation_history(self):
        """Test allocation history tracking."""
        promotion = StrategyPromotion(
            strategy_id="test_strategy",
            promoted_at=datetime.now(),
            initial_allocation=Decimal("0.10"),
            current_allocation=Decimal("0.10"),
        )
        
        # Add allocation changes
        promotion.allocation_history.append({
            "timestamp": datetime.now(),
            "allocation": Decimal("0.20"),
            "reason": "Performance improvement",
        })
        
        assert len(promotion.allocation_history) == 1
        assert promotion.allocation_history[0]["allocation"] == Decimal("0.20")


class TestPromotionDecision:
    """Test PromotionDecision dataclass."""

    def test_decision_creation(self):
        """Test creating a promotion decision."""
        decision = PromotionDecision(
            strategy_id="test_strategy",
            should_promote=True,
            allocation=Decimal("0.10"),
            reason="Validation criteria met",
            validation_result=MagicMock(passed=True, overall_score=0.85),
        )
        
        assert decision.strategy_id == "test_strategy"
        assert decision.should_promote is True
        assert decision.allocation == Decimal("0.10")
        assert decision.reason == "Validation criteria met"
        assert decision.validation_result.passed is True


class TestABTestResult:
    """Test ABTestResult dataclass."""

    def test_ab_test_result(self):
        """Test A/B test result creation."""
        result = ABTestResult(
            variant_a="strategy_v1",
            variant_b="strategy_v2",
            winner="strategy_v2",
            confidence=0.95,
            performance_diff=Decimal("0.15"),
            test_duration_days=14,
        )
        
        assert result.variant_a == "strategy_v1"
        assert result.variant_b == "strategy_v2"
        assert result.winner == "strategy_v2"
        assert result.confidence == 0.95
        assert result.performance_diff == Decimal("0.15")
        assert result.test_duration_days == 14


class TestStrategyPromotionManager:
    """Test StrategyPromotionManager class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PromotionConfig(
            auto_promote=True,
            initial_allocation=Decimal("0.10"),
            allocation_increment=Decimal("0.10"),
            max_allocation=Decimal("1.00"),
        )

    @pytest.fixture
    def validation_criteria(self):
        """Create test validation criteria."""
        return ValidationCriteria(
            min_trades=100,
            min_days=7,
            min_sharpe_ratio=1.5,
        )

    @pytest.fixture
    def manager(self, config, validation_criteria):
        """Create test promotion manager."""
        return StrategyPromotionManager(config, validation_criteria)

    def test_initialization(self, manager, config, validation_criteria):
        """Test manager initialization."""
        assert manager.config == config
        assert manager.validation_criteria == validation_criteria
        assert len(manager.promoted_strategies) == 0
        assert len(manager.ab_tests) == 0
        assert len(manager.audit_log) == 0

    @pytest.mark.asyncio
    async def test_evaluate_for_promotion_success(self, manager):
        """Test evaluating strategy for promotion - success case."""
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        
        decision = await manager.evaluate_for_promotion("test_strategy", metrics)
        
        assert isinstance(decision, PromotionDecision)
        assert decision.should_promote is True
        assert decision.allocation == manager.config.initial_allocation
        assert "criteria met" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_evaluate_for_promotion_failure(self, manager):
        """Test evaluating strategy for promotion - failure case."""
        metrics = {
            "total_trades": 50,  # Below minimum
            "start_date": datetime.now() - timedelta(days=3),  # Below minimum
            "sharpe_ratio": 1.0,  # Below minimum
            "max_drawdown": Decimal("0.15"),
            "win_rate": Decimal("0.45"),
        }
        
        decision = await manager.evaluate_for_promotion("test_strategy", metrics)
        
        assert decision.should_promote is False
        assert decision.allocation == Decimal("0")
        assert "validation failed" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_promote_strategy(self, manager):
        """Test promoting a strategy."""
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        
        promotion = await manager.promote_strategy("test_strategy", metrics)
        
        assert isinstance(promotion, StrategyPromotion)
        assert promotion.strategy_id == "test_strategy"
        assert promotion.initial_allocation == manager.config.initial_allocation
        assert promotion.current_allocation == manager.config.initial_allocation
        assert "test_strategy" in manager.promoted_strategies
        assert len(manager.audit_log) > 0

    @pytest.mark.asyncio
    async def test_promote_already_promoted(self, manager):
        """Test promoting an already promoted strategy."""
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        
        # Promote once
        await manager.promote_strategy("test_strategy", metrics)
        
        # Try to promote again
        with pytest.raises(ValueError, match="already promoted"):
            await manager.promote_strategy("test_strategy", metrics)

    @pytest.mark.asyncio
    async def test_increase_allocation(self, manager):
        """Test increasing strategy allocation."""
        # First promote the strategy
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        await manager.promote_strategy("test_strategy", metrics)
        
        # Increase allocation
        new_allocation = await manager.increase_allocation(
            "test_strategy",
            reason="Excellent performance",
        )
        
        expected = manager.config.initial_allocation + manager.config.allocation_increment
        assert new_allocation == expected
        
        promotion = manager.promoted_strategies["test_strategy"]
        assert promotion.current_allocation == expected
        assert len(promotion.allocation_history) > 0

    @pytest.mark.asyncio
    async def test_increase_allocation_max_limit(self, manager):
        """Test allocation increase respects maximum limit."""
        # Promote and set allocation near max
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        promotion = await manager.promote_strategy("test_strategy", metrics)
        promotion.current_allocation = Decimal("0.95")
        
        # Try to increase beyond max
        new_allocation = await manager.increase_allocation("test_strategy")
        
        assert new_allocation == manager.config.max_allocation
        assert new_allocation == Decimal("1.00")

    @pytest.mark.asyncio
    async def test_decrease_allocation(self, manager):
        """Test decreasing strategy allocation."""
        # Promote and increase allocation first
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        await manager.promote_strategy("test_strategy", metrics)
        await manager.increase_allocation("test_strategy")
        
        # Decrease allocation
        new_allocation = await manager.decrease_allocation(
            "test_strategy",
            reason="Performance regression",
        )
        
        assert new_allocation == manager.config.initial_allocation
        
        promotion = manager.promoted_strategies["test_strategy"]
        assert promotion.current_allocation == manager.config.initial_allocation

    @pytest.mark.asyncio
    async def test_demote_strategy(self, manager):
        """Test demoting a strategy."""
        # Promote first
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        await manager.promote_strategy("test_strategy", metrics)
        
        # Demote
        await manager.demote_strategy("test_strategy", reason="Consistent losses")
        
        assert "test_strategy" not in manager.promoted_strategies
        assert len(manager.audit_log) > 1
        assert "demoted" in manager.audit_log[-1]["action"]

    @pytest.mark.asyncio
    async def test_check_regression(self, manager):
        """Test regression detection."""
        # Promote with baseline metrics
        baseline_metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        promotion = await manager.promote_strategy("test_strategy", baseline_metrics)
        promotion.performance_baseline = baseline_metrics
        
        # Check with degraded performance
        current_metrics = {
            "sharpe_ratio": 1.4,  # 30% drop from 2.0
            "win_rate": Decimal("0.40"),  # 33% drop from 0.60
        }
        
        has_regression = await manager.check_regression("test_strategy", current_metrics)
        assert has_regression is True
        
        # Check with maintained performance
        good_metrics = {
            "sharpe_ratio": 1.9,
            "win_rate": Decimal("0.58"),
        }
        
        has_regression = await manager.check_regression("test_strategy", good_metrics)
        assert has_regression is False

    @pytest.mark.asyncio
    async def test_start_ab_test(self, manager):
        """Test starting an A/B test."""
        test_id = await manager.start_ab_test(
            variant_a="strategy_v1",
            variant_b="strategy_v2",
            allocation_split=Decimal("0.5"),
        )
        
        assert test_id in manager.ab_tests
        test = manager.ab_tests[test_id]
        assert test["variant_a"] == "strategy_v1"
        assert test["variant_b"] == "strategy_v2"
        assert test["allocation_split"] == Decimal("0.5")
        assert test["status"] == "running"

    @pytest.mark.asyncio
    async def test_ab_test_max_variants(self, manager):
        """Test A/B test maximum variants limit."""
        # Start maximum number of tests
        for i in range(manager.config.max_ab_variants):
            await manager.start_ab_test(
                variant_a=f"strategy_a{i}",
                variant_b=f"strategy_b{i}",
            )
        
        # Try to start one more
        with pytest.raises(ValueError, match="Maximum.*AB tests"):
            await manager.start_ab_test(
                variant_a="strategy_extra_a",
                variant_b="strategy_extra_b",
            )

    @pytest.mark.asyncio
    async def test_evaluate_ab_test(self, manager):
        """Test evaluating A/B test results."""
        # Start test
        test_id = await manager.start_ab_test(
            variant_a="strategy_v1",
            variant_b="strategy_v2",
        )
        
        # Provide metrics for evaluation
        metrics_a = {
            "sharpe_ratio": 1.5,
            "win_rate": Decimal("0.55"),
            "total_pnl": Decimal("1000"),
        }
        
        metrics_b = {
            "sharpe_ratio": 2.0,
            "win_rate": Decimal("0.65"),
            "total_pnl": Decimal("1500"),
        }
        
        result = await manager.evaluate_ab_test(test_id, metrics_a, metrics_b)
        
        assert isinstance(result, ABTestResult)
        assert result.winner == "strategy_v2"
        assert result.performance_diff > Decimal("0")
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_stop_ab_test(self, manager):
        """Test stopping an A/B test."""
        # Start test
        test_id = await manager.start_ab_test(
            variant_a="strategy_v1",
            variant_b="strategy_v2",
        )
        
        # Stop test
        await manager.stop_ab_test(test_id)
        
        assert manager.ab_tests[test_id]["status"] == "stopped"
        assert manager.ab_tests[test_id]["end_time"] is not None

    @pytest.mark.asyncio
    async def test_gradual_allocation_strategy(self, manager):
        """Test gradual allocation increase strategy."""
        # Promote strategy
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        await manager.promote_strategy("test_strategy", metrics)
        
        # Simulate gradual allocation increases
        allocations = []
        for _ in range(5):
            new_allocation = await manager.increase_allocation("test_strategy")
            allocations.append(new_allocation)
        
        # Check allocations increase gradually
        for i in range(1, len(allocations)):
            assert allocations[i] > allocations[i-1]
        
        # Should not exceed max
        assert allocations[-1] <= manager.config.max_allocation

    @pytest.mark.asyncio
    async def test_get_promoted_strategies(self, manager):
        """Test getting list of promoted strategies."""
        # Promote multiple strategies
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        
        await manager.promote_strategy("strategy1", metrics)
        await manager.promote_strategy("strategy2", metrics)
        
        promoted = manager.get_promoted_strategies()
        
        assert len(promoted) == 2
        assert "strategy1" in promoted
        assert "strategy2" in promoted

    @pytest.mark.asyncio
    async def test_get_allocation(self, manager):
        """Test getting current allocation for a strategy."""
        # Not promoted yet
        allocation = manager.get_allocation("test_strategy")
        assert allocation == Decimal("0")
        
        # Promote
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        await manager.promote_strategy("test_strategy", metrics)
        
        allocation = manager.get_allocation("test_strategy")
        assert allocation == manager.config.initial_allocation

    @pytest.mark.asyncio
    async def test_audit_log(self, manager):
        """Test audit log functionality."""
        # Perform various operations
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        
        await manager.promote_strategy("test_strategy", metrics)
        await manager.increase_allocation("test_strategy")
        await manager.decrease_allocation("test_strategy")
        await manager.demote_strategy("test_strategy")
        
        # Check audit log
        assert len(manager.audit_log) >= 4
        
        actions = [entry["action"] for entry in manager.audit_log]
        assert "promoted" in actions
        assert "allocation_increased" in actions
        assert "allocation_decreased" in actions
        assert "demoted" in actions
        
        # Check log entries have required fields
        for entry in manager.audit_log:
            assert "timestamp" in entry
            assert "action" in entry
            assert "strategy_id" in entry
            assert "details" in entry

    @pytest.mark.asyncio
    async def test_rollback_capability(self, manager):
        """Test ability to rollback promotions."""
        # Promote and modify
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
        }
        
        await manager.promote_strategy("test_strategy", metrics)
        await manager.increase_allocation("test_strategy")
        
        # Simulate performance drop and rollback
        await manager.decrease_allocation("test_strategy", reason="Rollback due to losses")
        await manager.demote_strategy("test_strategy", reason="Complete rollback")
        
        assert "test_strategy" not in manager.promoted_strategies

    def test_allocation_strategy_enum(self):
        """Test AllocationStrategy enum values."""
        assert AllocationStrategy.GRADUAL.value == "gradual"
        assert AllocationStrategy.AGGRESSIVE.value == "aggressive"
        assert AllocationStrategy.CONSERVATIVE.value == "conservative"