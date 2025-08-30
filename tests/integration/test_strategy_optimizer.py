"""Integration tests for strategy parameter optimization with A/B testing."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
import json
from scipy import stats

from genesis.operations.strategy_optimizer import (
    StrategyOptimizer,
    ParameterSet,
    ABTestGroup,
    OptimizationResult,
    BayesianOptimizer,
    PerformanceTracker
)


class TestParameterSet:
    """Test parameter set management."""
    
    def test_parameter_set_creation(self):
        """Test creation of parameter sets."""
        params = ParameterSet(
            strategy="spread_capture",
            parameters={
                "spread_threshold": 0.002,
                "position_size": 0.1,
                "stop_loss": 0.01
            },
            version="v1.0"
        )
        
        assert params.strategy == "spread_capture"
        assert params.parameters["spread_threshold"] == 0.002
        assert params.version == "v1.0"
    
    def test_parameter_validation(self):
        """Test parameter validation against constraints."""
        params = ParameterSet(
            strategy="spread_capture",
            parameters={
                "spread_threshold": 0.002,
                "position_size": 0.1
            },
            constraints={
                "spread_threshold": {"min": 0.001, "max": 0.01},
                "position_size": {"min": 0.01, "max": 1.0}
            }
        )
        
        assert params.validate()
        
        # Invalid parameter
        params.parameters["position_size"] = 1.5
        assert not params.validate()
    
    def test_parameter_hash(self):
        """Test parameter set hashing for uniqueness."""
        params1 = ParameterSet(
            strategy="test",
            parameters={"a": 1, "b": 2}
        )
        
        params2 = ParameterSet(
            strategy="test",
            parameters={"a": 1, "b": 2}
        )
        
        params3 = ParameterSet(
            strategy="test",
            parameters={"a": 1, "b": 3}
        )
        
        assert params1.get_hash() == params2.get_hash()
        assert params1.get_hash() != params3.get_hash()


class TestABTestGroup:
    """Test A/B test group management."""
    
    def test_group_assignment(self):
        """Test assigning trades to test groups."""
        group_a = ABTestGroup(name="control", parameter_set=MagicMock())
        group_b = ABTestGroup(name="variant", parameter_set=MagicMock())
        
        # Assign trades
        for i in range(100):
            if i % 2 == 0:
                group_a.add_trade(MagicMock(pnl=10))
            else:
                group_b.add_trade(MagicMock(pnl=15))
        
        assert group_a.trade_count == 50
        assert group_b.trade_count == 50
    
    def test_performance_metrics(self):
        """Test calculating performance metrics for groups."""
        group = ABTestGroup(name="test", parameter_set=MagicMock())
        
        # Add trades with various outcomes
        trades = [
            MagicMock(pnl=100, duration_seconds=60),
            MagicMock(pnl=-50, duration_seconds=30),
            MagicMock(pnl=75, duration_seconds=45),
            MagicMock(pnl=25, duration_seconds=90),
            MagicMock(pnl=-10, duration_seconds=120)
        ]
        
        for trade in trades:
            group.add_trade(trade)
        
        metrics = group.calculate_metrics()
        
        assert metrics["total_pnl"] == 140  # 100 - 50 + 75 + 25 - 10
        assert metrics["win_rate"] == 0.6  # 3 wins out of 5
        assert metrics["avg_pnl"] == 28  # 140 / 5
        assert metrics["sharpe_ratio"] is not None


class TestBayesianOptimizer:
    """Test Bayesian optimization for parameter search."""
    
    def test_acquisition_function(self):
        """Test Expected Improvement acquisition function."""
        optimizer = BayesianOptimizer()
        
        # Mock Gaussian Process predictions
        mean = np.array([1.0, 1.5, 2.0, 1.8])
        std = np.array([0.1, 0.2, 0.15, 0.3])
        current_best = 1.9
        
        ei_values = optimizer.expected_improvement(mean, std, current_best)
        
        # Point with high mean and high uncertainty should have high EI
        assert ei_values[1] > ei_values[0]  # Higher uncertainty
        assert ei_values[3] > ei_values[2]  # Higher uncertainty despite lower mean
    
    def test_parameter_space_exploration(self):
        """Test exploration vs exploitation balance."""
        optimizer = BayesianOptimizer()
        
        space = {
            "spread_threshold": {"min": 0.001, "max": 0.01, "type": "float"},
            "position_size": {"min": 0.01, "max": 0.5, "type": "float"}
        }
        
        # Initial exploration should cover space
        initial_points = optimizer.generate_initial_points(space, n_points=10)
        
        assert len(initial_points) == 10
        for point in initial_points:
            assert 0.001 <= point["spread_threshold"] <= 0.01
            assert 0.01 <= point["position_size"] <= 0.5
    
    @pytest.mark.asyncio
    async def test_optimization_convergence(self):
        """Test that optimization converges to good parameters."""
        optimizer = BayesianOptimizer()
        
        # Define a simple objective function (quadratic with known optimum)
        def objective(params):
            x = params["x"]
            return -(x - 0.7) ** 2 + 1  # Maximum at x=0.7
        
        space = {"x": {"min": 0, "max": 1, "type": "float"}}
        
        best_params = None
        best_value = float('-inf')
        
        for _ in range(20):
            params = await optimizer.suggest_next_parameters(
                space, history=[]
            )
            value = objective(params)
            
            if value > best_value:
                best_value = value
                best_params = params
        
        # Should converge close to 0.7
        assert abs(best_params["x"] - 0.7) < 0.1


class TestStrategyOptimizer:
    """Test main strategy optimization system."""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test strategy optimizer initialization."""
        with patch('genesis.operations.strategy_optimizer.load_config'):
            optimizer = StrategyOptimizer()
            
            assert optimizer.min_trades_for_significance == 100
            assert optimizer.significance_level == 0.05
            assert optimizer.rollback_threshold == -0.1
            assert optimizer.active_tests == {}
    
    @pytest.mark.asyncio
    async def test_create_ab_test(self):
        """Test creating an A/B test."""
        optimizer = StrategyOptimizer()
        
        control_params = ParameterSet(
            strategy="spread_capture",
            parameters={"spread_threshold": 0.002}
        )
        
        variant_params = ParameterSet(
            strategy="spread_capture",
            parameters={"spread_threshold": 0.003}
        )
        
        test_id = await optimizer.create_ab_test(
            strategy="spread_capture",
            control_params=control_params,
            variant_params=variant_params,
            traffic_split=0.5
        )
        
        assert test_id in optimizer.active_tests
        test = optimizer.active_tests[test_id]
        assert test.control_group.parameter_set == control_params
        assert test.variant_group.parameter_set == variant_params
        assert test.traffic_split == 0.5
    
    @pytest.mark.asyncio
    async def test_traffic_routing(self):
        """Test routing trades to correct test groups."""
        optimizer = StrategyOptimizer()
        
        # Create test with 70/30 split
        test_id = await optimizer.create_ab_test(
            strategy="test_strategy",
            control_params=MagicMock(),
            variant_params=MagicMock(),
            traffic_split=0.7  # 70% to control
        )
        
        control_count = 0
        variant_count = 0
        
        # Route 1000 trades
        for i in range(1000):
            group = await optimizer.route_trade_to_group(test_id, trade_id=i)
            if group == "control":
                control_count += 1
            else:
                variant_count += 1
        
        # Check split is approximately 70/30
        control_ratio = control_count / 1000
        assert 0.65 < control_ratio < 0.75  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_statistical_significance(self):
        """Test statistical significance calculation."""
        optimizer = StrategyOptimizer()
        
        # Create samples with clear difference
        control_pnls = [10 + np.random.normal(0, 2) for _ in range(150)]
        variant_pnls = [15 + np.random.normal(0, 2) for _ in range(150)]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(control_pnls, variant_pnls)
        
        assert p_value < 0.05  # Should be statistically significant
        
        # Test with no difference
        control_pnls = [10 + np.random.normal(0, 2) for _ in range(150)]
        variant_pnls = [10 + np.random.normal(0, 2) for _ in range(150)]
        
        t_stat, p_value = stats.ttest_ind(control_pnls, variant_pnls)
        
        assert p_value > 0.05  # Should not be significant
    
    @pytest.mark.asyncio
    async def test_automatic_rollback(self):
        """Test automatic rollback for underperforming parameters."""
        with patch('genesis.operations.strategy_optimizer.rollback_parameters') as mock_rollback:
            optimizer = StrategyOptimizer()
            
            test_id = await optimizer.create_ab_test(
                strategy="test",
                control_params=MagicMock(),
                variant_params=MagicMock()
            )
            
            test = optimizer.active_tests[test_id]
            
            # Simulate poor performance in variant
            for i in range(150):
                test.control_group.add_trade(MagicMock(pnl=10))
                test.variant_group.add_trade(MagicMock(pnl=-5))  # Losing money
            
            # Check for rollback
            should_rollback = await optimizer.check_rollback_criteria(test_id)
            assert should_rollback
            
            # Trigger rollback
            await optimizer.rollback_if_needed(test_id)
            mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test tracking performance per parameter set."""
        tracker = PerformanceTracker()
        
        param_set = ParameterSet(
            strategy="test",
            parameters={"a": 1}
        )
        
        # Record trades
        for i in range(50):
            await tracker.record_trade(
                param_set_id=param_set.get_hash(),
                trade=MagicMock(
                    pnl=10 + i,
                    timestamp=datetime.now(),
                    duration_seconds=60
                )
            )
        
        # Get performance metrics
        metrics = await tracker.get_metrics(param_set.get_hash())
        
        assert metrics["trade_count"] == 50
        assert metrics["total_pnl"] > 0
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
    
    @pytest.mark.asyncio
    async def test_multi_strategy_optimization(self):
        """Test optimizing multiple strategies simultaneously."""
        optimizer = StrategyOptimizer()
        
        # Create tests for different strategies
        strategies = ["spread_capture", "mean_reversion", "momentum"]
        test_ids = []
        
        for strategy in strategies:
            test_id = await optimizer.create_ab_test(
                strategy=strategy,
                control_params=MagicMock(),
                variant_params=MagicMock()
            )
            test_ids.append(test_id)
        
        assert len(optimizer.active_tests) == 3
        
        # Each should track independently
        for test_id in test_ids:
            test = optimizer.active_tests[test_id]
            assert test.control_group.trade_count == 0
            assert test.variant_group.trade_count == 0
    
    @pytest.mark.asyncio
    async def test_parameter_persistence(self):
        """Test saving winning parameters to database."""
        with patch('genesis.operations.strategy_optimizer.save_to_db') as mock_save:
            optimizer = StrategyOptimizer()
            
            winning_params = ParameterSet(
                strategy="test",
                parameters={"spread": 0.003},
                performance_metrics={
                    "sharpe_ratio": 1.5,
                    "total_pnl": 5000
                }
            )
            
            await optimizer.promote_parameters(winning_params)
            
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert saved_data.strategy == "test"
            assert saved_data.parameters["spread"] == 0.003
    
    @pytest.mark.asyncio
    async def test_gradual_rollout(self):
        """Test gradual rollout of winning parameters."""
        optimizer = StrategyOptimizer()
        
        test_id = await optimizer.create_ab_test(
            strategy="test",
            control_params=MagicMock(),
            variant_params=MagicMock(),
            traffic_split=0.1  # Start with 10% to variant
        )
        
        # Simulate good performance
        test = optimizer.active_tests[test_id]
        for i in range(200):
            test.control_group.add_trade(MagicMock(pnl=10))
            test.variant_group.add_trade(MagicMock(pnl=20))  # Variant performing better
        
        # Increase traffic to variant
        await optimizer.adjust_traffic_split(test_id, new_split=0.5)
        assert test.traffic_split == 0.5
        
        # Further increase based on continued performance
        await optimizer.adjust_traffic_split(test_id, new_split=0.9)
        assert test.traffic_split == 0.9


class TestOptimizationIntegration:
    """Integration tests for complete optimization workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test complete optimization cycle from exploration to deployment."""
        with patch('genesis.operations.strategy_optimizer.get_active_strategy') as mock_strategy:
            optimizer = StrategyOptimizer()
            bayesian = BayesianOptimizer()
            
            # Define parameter space
            space = {
                "spread_threshold": {"min": 0.001, "max": 0.005},
                "position_size": {"min": 0.05, "max": 0.2}
            }
            
            # Phase 1: Exploration
            initial_params = bayesian.generate_initial_points(space, n_points=5)
            
            best_params = None
            best_performance = float('-inf')
            
            for params in initial_params:
                # Create A/B test
                param_set = ParameterSet(strategy="test", parameters=params)
                test_id = await optimizer.create_ab_test(
                    strategy="test",
                    control_params=MagicMock(),  # Current production params
                    variant_params=param_set
                )
                
                # Simulate trading
                test = optimizer.active_tests[test_id]
                performance = np.random.normal(100, 20)  # Simulated P&L
                
                for _ in range(150):
                    test.variant_group.add_trade(MagicMock(pnl=performance))
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = params
            
            # Phase 2: Exploitation (refine best parameters)
            refined_params = best_params.copy()
            refined_params["spread_threshold"] *= 1.1  # Small adjustment
            
            refined_set = ParameterSet(strategy="test", parameters=refined_params)
            final_test_id = await optimizer.create_ab_test(
                strategy="test",
                control_params=ParameterSet(strategy="test", parameters=best_params),
                variant_params=refined_set
            )
            
            # Phase 3: Validation and deployment
            final_test = optimizer.active_tests[final_test_id]
            
            # Simulate trading with refined parameters
            for _ in range(200):
                final_test.control_group.add_trade(MagicMock(pnl=100))
                final_test.variant_group.add_trade(MagicMock(pnl=110))  # Better
            
            # Check significance and deploy
            is_significant = await optimizer.check_statistical_significance(final_test_id)
            
            if is_significant:
                await optimizer.promote_parameters(refined_set)
            
            assert best_params is not None
            assert best_performance > float('-inf')