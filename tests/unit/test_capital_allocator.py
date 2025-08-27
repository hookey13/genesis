"""
Unit tests for capital allocation system.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from genesis.engine.capital_allocator import (
    AllocationConfig,
    AllocationMethod,
    AllocationRule,
    CapitalAllocator,
    RebalanceFrequency,
    StrategyAllocation,
)
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    event_bus = AsyncMock(spec=EventBus)
    return event_bus


@pytest.fixture
def test_config():
    """Create test allocation config."""
    return AllocationConfig(
        method=AllocationMethod.EQUAL_WEIGHT,
        rebalance_frequency=RebalanceFrequency.DAILY,
        max_strategies=5,
        min_allocation_percent=Decimal("10"),
        max_allocation_percent=Decimal("40"),
        reserve_percent=Decimal("10"),
        rebalance_threshold_percent=Decimal("5")
    )


@pytest.fixture
def strategy_allocations():
    """Create test strategy allocations."""
    return [
        StrategyAllocation(
            strategy_id="strategy_1",
            strategy_name="Test Strategy 1",
            performance_score=Decimal("0.8"),
            risk_score=Decimal("1.2")
        ),
        StrategyAllocation(
            strategy_id="strategy_2",
            strategy_name="Test Strategy 2",
            performance_score=Decimal("0.6"),
            risk_score=Decimal("0.8")
        ),
        StrategyAllocation(
            strategy_id="strategy_3",
            strategy_name="Test Strategy 3",
            performance_score=Decimal("0.4"),
            risk_score=Decimal("1.5")
        )
    ]


@pytest.fixture
async def allocator(mock_event_bus):
    """Create capital allocator instance."""
    allocator = CapitalAllocator(
        event_bus=mock_event_bus,
        total_capital=Decimal("10000")
    )
    return allocator


class TestAllocationConfig:
    """Test allocation configuration."""
    
    def test_decimal_conversion(self):
        """Test automatic Decimal conversion."""
        config = AllocationConfig(
            min_allocation_percent=10,
            max_allocation_percent=40,
            reserve_percent=10,
            rebalance_threshold_percent=5,
            kelly_fraction=0.25
        )
        
        assert isinstance(config.min_allocation_percent, Decimal)
        assert isinstance(config.max_allocation_percent, Decimal)
        assert config.min_allocation_percent == Decimal("10")
        assert config.kelly_fraction == Decimal("0.25")
        
    def test_validation_max_greater_than_min(self):
        """Test that max allocation must be greater than min."""
        with pytest.raises(ValueError, match="Max allocation must be greater"):
            AllocationConfig(
                min_allocation_percent=Decimal("40"),
                max_allocation_percent=Decimal("30")
            )


class TestStrategyAllocation:
    """Test strategy allocation data structure."""
    
    def test_decimal_conversion(self):
        """Test automatic Decimal conversion in post_init."""
        alloc = StrategyAllocation(
            strategy_id="test",
            strategy_name="Test Strategy",
            current_allocation=1000,
            performance_score=0.8,
            risk_score=1.2
        )
        
        assert isinstance(alloc.current_allocation, Decimal)
        assert isinstance(alloc.performance_score, Decimal)
        assert alloc.current_allocation == Decimal("1000")
        assert alloc.performance_score == Decimal("0.8")


class TestCapitalAllocator:
    """Test capital allocator operations."""
    
    async def test_equal_weight_allocation(self, allocator, strategy_allocations, mock_event_bus):
        """Test equal weight allocation method."""
        allocator.config.method = AllocationMethod.EQUAL_WEIGHT
        
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # With 10% reserve, 9000 available, 3000 per strategy
        assert len(allocations) == 3
        for strategy_id, amount in allocations.items():
            assert amount == Decimal("3000")
            
        # Check event published
        mock_event_bus.publish.assert_called_once()
        
    async def test_performance_weighted_allocation(self, allocator, strategy_allocations, mock_event_bus):
        """Test performance-weighted allocation."""
        allocator.config.method = AllocationMethod.PERFORMANCE_WEIGHTED
        
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # Total performance: 0.8 + 0.6 + 0.4 = 1.8
        # Available: 9000
        # Strategy 1: 0.8/1.8 * 9000 = 4000
        # Strategy 2: 0.6/1.8 * 9000 = 3000
        # Strategy 3: 0.4/1.8 * 9000 = 2000
        
        assert allocations["strategy_1"] == Decimal("4000")
        assert allocations["strategy_2"] == Decimal("3000")
        assert allocations["strategy_3"] == Decimal("2000")
        
    async def test_risk_parity_allocation(self, allocator, strategy_allocations, mock_event_bus):
        """Test risk parity allocation."""
        allocator.config.method = AllocationMethod.RISK_PARITY
        
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # Inverse risk weighting
        # Strategy 1: 1/1.2 = 0.833
        # Strategy 2: 1/0.8 = 1.25
        # Strategy 3: 1/1.5 = 0.667
        # Total: 2.75
        
        # Check that lower risk gets higher allocation
        assert allocations["strategy_2"] > allocations["strategy_1"]
        assert allocations["strategy_1"] > allocations["strategy_3"]
        
    async def test_kelly_allocation(self, allocator, strategy_allocations, mock_event_bus):
        """Test Kelly criterion allocation."""
        allocator.config.method = AllocationMethod.KELLY_CRITERION
        allocator.config.use_kelly_sizing = True
        allocator.config.kelly_fraction = Decimal("0.25")
        
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # Kelly allocations based on performance edge
        # Higher performance scores get higher allocations
        assert allocations["strategy_1"] > allocations["strategy_2"]
        assert allocations["strategy_2"] > allocations["strategy_3"]
        
    async def test_constraint_application(self, allocator, strategy_allocations):
        """Test min/max constraint application."""
        allocator.config.min_allocation_percent = Decimal("20")
        allocator.config.max_allocation_percent = Decimal("40")
        
        # Set one strategy to have very high performance
        strategy_allocations[0].performance_score = Decimal("10.0")
        allocator.config.method = AllocationMethod.PERFORMANCE_WEIGHTED
        
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # Check max constraint applied
        max_allowed = Decimal("10000") * Decimal("0.9") * Decimal("0.4")  # 3600
        assert allocations["strategy_1"] <= max_allowed
        
        # Check min constraint applied
        min_allowed = Decimal("10000") * Decimal("0.9") * Decimal("0.2")  # 1800
        for amount in allocations.values():
            assert amount >= min_allowed
            
    async def test_lock_unlock_capital(self, allocator, strategy_allocations):
        """Test capital locking and unlocking."""
        await allocator.allocate_capital(strategy_allocations)
        
        # Lock capital
        result = allocator.lock_capital("strategy_1", Decimal("1000"))
        assert result is True
        assert allocator.allocations["strategy_1"].locked_capital == Decimal("1000")
        assert allocator.allocations["strategy_1"].available_capital == Decimal("2000")
        
        # Try to lock more than available
        result = allocator.lock_capital("strategy_1", Decimal("3000"))
        assert result is False
        
        # Unlock capital
        result = allocator.unlock_capital("strategy_1", Decimal("500"))
        assert result is True
        assert allocator.allocations["strategy_1"].locked_capital == Decimal("500")
        assert allocator.allocations["strategy_1"].available_capital == Decimal("2500")
        
    async def test_update_strategy_performance(self, allocator, strategy_allocations):
        """Test updating strategy performance metrics."""
        await allocator.allocate_capital(strategy_allocations)
        
        allocator.update_strategy_performance(
            "strategy_1",
            performance_score=Decimal("0.9"),
            risk_score=Decimal("1.0")
        )
        
        assert allocator.allocations["strategy_1"].performance_score == Decimal("0.9")
        assert allocator.allocations["strategy_1"].risk_score == Decimal("1.0")
        
    async def test_rebalance_daily(self, allocator, strategy_allocations):
        """Test daily rebalancing schedule."""
        allocator.config.rebalance_frequency = RebalanceFrequency.DAILY
        await allocator.allocate_capital(strategy_allocations)
        
        # Should not rebalance immediately
        result = await allocator.rebalance()
        assert result is False
        
        # Simulate time passing
        allocator.last_rebalance = datetime.now(timezone.utc) - timedelta(days=2)
        
        # Should rebalance now
        result = await allocator.rebalance()
        assert result is True
        
    async def test_rebalance_threshold(self, allocator, strategy_allocations):
        """Test threshold-based rebalancing."""
        allocator.config.rebalance_frequency = RebalanceFrequency.THRESHOLD
        allocator.config.rebalance_threshold_percent = Decimal("5")
        
        await allocator.allocate_capital(strategy_allocations)
        
        # Set target different from current
        allocator.allocations["strategy_1"].target_allocation = Decimal("4000")
        allocator.allocations["strategy_1"].current_allocation = Decimal("3000")
        
        # Drift is 25%, should trigger rebalance
        result = await allocator.rebalance()
        assert result is True
        
    async def test_force_rebalance(self, allocator, strategy_allocations):
        """Test forced rebalancing."""
        allocator.config.rebalance_frequency = RebalanceFrequency.WEEKLY
        await allocator.allocate_capital(strategy_allocations)
        
        # Force rebalance regardless of schedule
        result = await allocator.rebalance(force=True)
        assert result is True
        
    def test_get_available_capital(self, allocator):
        """Test getting available capital for a strategy."""
        # Non-existent strategy
        capital = allocator.get_available_capital("non_existent")
        assert capital == Decimal("0")
        
        # Add allocation
        allocator.allocations["test_strategy"] = StrategyAllocation(
            strategy_id="test_strategy",
            strategy_name="Test",
            available_capital=Decimal("5000")
        )
        
        capital = allocator.get_available_capital("test_strategy")
        assert capital == Decimal("5000")
        
    async def test_get_allocation_summary(self, allocator, strategy_allocations):
        """Test getting allocation summary."""
        await allocator.allocate_capital(strategy_allocations)
        
        summary = allocator.get_allocation_summary()
        
        assert summary["total_capital"] == "10000"
        assert summary["num_strategies"] == 3
        assert summary["allocation_method"] == "equal_weight"
        assert "strategies" in summary
        assert len(summary["strategies"]) == 3
        
    def test_add_custom_rule(self, allocator):
        """Test adding custom allocation rules."""
        rule = AllocationRule(
            rule_id="rule_1",
            name="High Performance Boost",
            condition="performance_score > 0.7",
            action="increase_allocation",
            adjustment_percent=Decimal("10"),
            priority=1
        )
        
        allocator.add_rule(rule)
        
        assert len(allocator.allocation_rules) == 1
        assert allocator.allocation_rules[0].rule_id == "rule_1"
        
    async def test_custom_allocation_with_rules(self, allocator, strategy_allocations):
        """Test custom allocation with rules."""
        allocator.config.method = AllocationMethod.CUSTOM
        
        # Add rule to boost high performers
        rule = AllocationRule(
            rule_id="boost_high",
            name="Boost High Performers",
            condition="performance_score > 0.7",
            action="increase_allocation",
            adjustment_percent=Decimal("20")
        )
        allocator.add_rule(rule)
        
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # Strategy 1 with 0.8 performance should get boosted
        # Base equal weight: 3000
        # With 20% boost: 3600
        assert allocations["strategy_1"] == Decimal("3600")
        
    async def test_config_file_loading(self, tmp_path, mock_event_bus):
        """Test loading configuration from file."""
        # Create temp config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "allocation": {
                "method": "performance_weighted",
                "max_strategies": 8,
                "reserve_percent": 15
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
            
        # Create allocator with config file
        allocator = CapitalAllocator(
            event_bus=mock_event_bus,
            total_capital=Decimal("10000"),
            config_path=str(config_file)
        )
        
        assert allocator.config.method == AllocationMethod.PERFORMANCE_WEIGHTED
        assert allocator.config.max_strategies == 8
        assert allocator.config.reserve_percent == Decimal("15")
        
    async def test_zero_total_capital(self, mock_event_bus):
        """Test handling zero total capital."""
        allocator = CapitalAllocator(
            event_bus=mock_event_bus,
            total_capital=Decimal("0")
        )
        
        allocations = await allocator.allocate_capital([
            StrategyAllocation(
                strategy_id="test",
                strategy_name="Test",
                performance_score=Decimal("1.0")
            )
        ])
        
        assert allocations["test"] == Decimal("0")
        
    async def test_negative_performance_scores(self, allocator, strategy_allocations):
        """Test handling negative performance scores."""
        # Set negative performance score
        strategy_allocations[0].performance_score = Decimal("-0.5")
        
        allocator.config.method = AllocationMethod.PERFORMANCE_WEIGHTED
        
        # Should use minimum score of 0.1 to avoid zero allocation
        allocations = await allocator.allocate_capital(strategy_allocations)
        
        # All strategies should still get some allocation
        for amount in allocations.values():
            assert amount > Decimal("0")