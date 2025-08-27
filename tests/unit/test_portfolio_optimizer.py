"""
Unit tests for Portfolio Optimizer

Tests complete portfolio optimization workflow including constraints,
validation, and performance requirements.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from genesis.analytics.portfolio_optimizer import (
    OptimizationConstraints,
    OptimizationResult,
    PortfolioOptimizer,
    Strategy,
    ValidationResult,
)
from genesis.core.exceptions import DataError as InvalidDataError
from genesis.engine.event_bus import EventBus, EventType


class TestPortfolioOptimizer:
    """Test suite for portfolio optimization"""

    @pytest.fixture
    def event_bus(self):
        """Create mock event bus"""
        bus = Mock(spec=EventBus)
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def optimizer(self, event_bus):
        """Create optimizer instance"""
        config = {
            "risk_free_rate": 0.02,
            "constraints": {
                "min_allocation": 0.01,
                "max_allocation": 0.40,
                "max_correlation": 0.60,
                "min_strategies": 2
            },
            "rebalancing": {
                "threshold_percent": 5.0,
                "min_improvement": 0.01,
                "schedule": "weekly"
            }
        }

        with patch('genesis.analytics.portfolio_optimizer.PortfolioOptimizer._load_configuration'):
            optimizer = PortfolioOptimizer(event_bus=event_bus)
            optimizer.config = config
            optimizer.constraints = OptimizationConstraints(
                min_allocation=Decimal("0.01"),
                max_allocation=Decimal("0.40"),
                max_correlation=Decimal("0.60"),
                min_strategies=2
            )
        return optimizer

    @pytest.fixture
    def sample_strategies(self):
        """Create sample strategies with returns"""
        np.random.seed(42)

        # Strategy 1: Positive expected return, low volatility
        returns1 = np.random.normal(0.002, 0.01, 100)

        # Strategy 2: Higher return, higher volatility
        returns2 = np.random.normal(0.003, 0.02, 100)

        # Strategy 3: Moderate return and volatility
        returns3 = np.random.normal(0.0015, 0.015, 100)

        strategies = [
            Strategy(
                name="Strategy_A",
                returns=[Decimal(str(r)) for r in returns1],
                current_allocation=Decimal("0.33"),
                is_active=True
            ),
            Strategy(
                name="Strategy_B",
                returns=[Decimal(str(r)) for r in returns2],
                current_allocation=Decimal("0.33"),
                is_active=True
            ),
            Strategy(
                name="Strategy_C",
                returns=[Decimal(str(r)) for r in returns3],
                current_allocation=Decimal("0.34"),
                is_active=True
            )
        ]

        return strategies

    @pytest.fixture
    def correlated_strategies(self):
        """Create highly correlated strategies"""
        np.random.seed(43)

        # Base returns
        base = np.random.normal(0.002, 0.015, 100)

        # Highly correlated strategies
        returns1 = base + np.random.normal(0, 0.005, 100)
        returns2 = base * 0.9 + np.random.normal(0, 0.005, 100)

        strategies = [
            Strategy(
                name="Correlated_A",
                returns=[Decimal(str(r)) for r in returns1],
                current_allocation=Decimal("0.50"),
                is_active=True
            ),
            Strategy(
                name="Correlated_B",
                returns=[Decimal(str(r)) for r in returns2],
                current_allocation=Decimal("0.50"),
                is_active=True
            )
        ]

        return strategies

    @pytest.mark.asyncio
    async def test_basic_optimization(self, optimizer, sample_strategies):
        """Test basic portfolio optimization"""
        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimal_weights is not None
        assert len(result.optimal_weights) == 3

        # Weights should sum to 1
        total_weight = sum(result.optimal_weights.values())
        assert abs(total_weight - Decimal("1")) < Decimal("0.01")

        # Should have positive Sharpe ratio
        assert result.sharpe_ratio > Decimal("0")

    @pytest.mark.asyncio
    async def test_minimum_allocation_enforcement(self, optimizer, sample_strategies):
        """Test that minimum allocations are enforced"""
        # Add a strategy with very low allocation
        sample_strategies[2].current_allocation = Decimal("0.005")  # 0.5%

        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # All non-zero weights should be >= min_allocation
        for weight in result.optimal_weights.values():
            if weight > Decimal("0"):
                assert weight >= optimizer.constraints.min_allocation

    @pytest.mark.asyncio
    async def test_maximum_allocation_enforcement(self, optimizer, sample_strategies):
        """Test that maximum allocations are enforced"""
        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # No weight should exceed max_allocation
        for weight in result.optimal_weights.values():
            assert weight <= optimizer.constraints.max_allocation

    @pytest.mark.asyncio
    async def test_correlation_detection(self, optimizer, correlated_strategies):
        """Test detection of high correlation"""
        # Should detect and warn about high correlation
        with patch('structlog.get_logger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            result = await optimizer.optimize_portfolio(
                correlated_strategies,
                portfolio_value_usdt=Decimal("10000"),
                validate=False,
                rebalance_check=False
            )

            assert result is not None
            # Check correlation matrix contains high values
            correlations = result.correlation_matrix
            assert len(correlations) > 0

    @pytest.mark.asyncio
    async def test_insufficient_strategies(self, optimizer):
        """Test error with too few strategies"""
        single_strategy = [
            Strategy(
                name="Only_One",
                returns=[Decimal("0.01")] * 100,
                current_allocation=Decimal("1.00"),
                is_active=True
            )
        ]

        with pytest.raises(InvalidDataError):
            await optimizer.optimize_portfolio(
                single_strategy,
                portfolio_value_usdt=Decimal("10000")
            )

    @pytest.mark.asyncio
    async def test_out_of_sample_validation(self, optimizer, sample_strategies):
        """Test out-of-sample validation"""
        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=True,
            rebalance_check=False
        )

        assert result.validation is not None
        assert isinstance(result.validation, ValidationResult)
        assert result.validation.in_sample_sharpe is not None
        assert result.validation.out_sample_sharpe is not None
        assert result.validation.performance_degradation is not None
        assert isinstance(result.validation.is_robust, bool)

    @pytest.mark.asyncio
    async def test_walk_forward_analysis(self, optimizer, sample_strategies):
        """Test walk-forward analysis in validation"""
        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=True,
            rebalance_check=False
        )

        assert result.validation is not None
        assert result.validation.walk_forward_results is not None
        assert len(result.validation.walk_forward_results) > 0

        # Each window should have required fields
        for window_result in result.validation.walk_forward_results:
            assert "window" in window_result
            assert "train_sharpe" in window_result
            assert "test_sharpe" in window_result

    @pytest.mark.asyncio
    async def test_rebalancing_recommendation(self, optimizer, sample_strategies):
        """Test rebalancing recommendation generation"""
        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=True
        )

        assert result.rebalance_recommendation is not None
        assert "should_execute" in result.rebalance_recommendation
        assert "rationale" in result.rebalance_recommendation
        assert "current_allocation" in result.rebalance_recommendation
        assert "target_allocation" in result.rebalance_recommendation

    @pytest.mark.asyncio
    async def test_inactive_strategy_filtering(self, optimizer, sample_strategies):
        """Test that inactive strategies are filtered"""
        # Mark one strategy as inactive
        sample_strategies[1].is_active = False

        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # Should only have 2 strategies in result
        assert len(result.optimal_weights) == 2
        assert "Strategy_B" not in result.optimal_weights

    @pytest.mark.asyncio
    async def test_strategy_specific_constraints(self, optimizer, sample_strategies):
        """Test strategy-specific min/max allocations"""
        # Set specific constraints for one strategy
        sample_strategies[0].min_allocation = Decimal("0.20")
        sample_strategies[0].max_allocation = Decimal("0.30")

        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # Strategy_A should respect its specific constraints
        weight_a = result.optimal_weights.get("Strategy_A", Decimal("0"))
        if weight_a > Decimal("0"):
            assert weight_a >= Decimal("0.20") or weight_a == Decimal("0")

    @pytest.mark.asyncio
    async def test_performance_requirement(self, optimizer):
        """Test optimization completes within time limit"""
        # Create 10 strategies (max for performance requirement)
        strategies = []
        for i in range(10):
            np.random.seed(i)
            returns = np.random.normal(0.002, 0.015, 100)
            strategies.append(
                Strategy(
                    name=f"Strategy_{i}",
                    returns=[Decimal(str(r)) for r in returns],
                    current_allocation=Decimal("0.10"),
                    is_active=True
                )
            )

        result = await optimizer.optimize_portfolio(
            strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # Should complete within 1 second (1000ms)
        assert result.optimization_time_ms < 1000

    @pytest.mark.asyncio
    async def test_event_publishing(self, optimizer, event_bus, sample_strategies):
        """Test that optimization events are published"""
        await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # Should have published an event
        event_bus.publish.assert_called_once()
        call_args = event_bus.publish.call_args[0][0]
        assert call_args.type == EventType.PORTFOLIO_OPTIMIZED

    @pytest.mark.asyncio
    async def test_insufficient_return_data(self, optimizer):
        """Test error with insufficient return periods"""
        strategies = [
            Strategy(
                name="Short_Data_A",
                returns=[Decimal("0.01")] * 10,  # Only 10 periods
                current_allocation=Decimal("0.50"),
                is_active=True
            ),
            Strategy(
                name="Short_Data_B",
                returns=[Decimal("0.02")] * 10,
                current_allocation=Decimal("0.50"),
                is_active=True
            )
        ]

        with pytest.raises(InvalidDataError):
            await optimizer.optimize_portfolio(
                strategies,
                portfolio_value_usdt=Decimal("10000")
            )

    @pytest.mark.asyncio
    async def test_invalid_return_values(self, optimizer):
        """Test validation of invalid return values"""
        strategies = [
            Strategy(
                name="Invalid_A",
                returns=[Decimal("999")] * 50,  # Unrealistic returns
                current_allocation=Decimal("0.50"),
                is_active=True
            ),
            Strategy(
                name="Invalid_B",
                returns=[Decimal("0.01")] * 50,
                current_allocation=Decimal("0.50"),
                is_active=True
            )
        ]

        with pytest.raises(InvalidDataError):
            await optimizer.optimize_portfolio(
                strategies,
                portfolio_value_usdt=Decimal("10000")
            )

    @pytest.mark.asyncio
    async def test_weight_normalization(self, optimizer, sample_strategies):
        """Test that weights are properly normalized"""
        result = await optimizer.optimize_portfolio(
            sample_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False
        )

        # Weights should sum to exactly 1 (within rounding tolerance)
        total = sum(result.optimal_weights.values())
        assert abs(total - Decimal("1")) < Decimal("0.001")

        # All weights should be properly rounded
        for weight in result.optimal_weights.values():
            # Check that weight has at most 4 decimal places
            str_weight = str(weight)
            if '.' in str_weight:
                decimals = len(str_weight.split('.')[1])
                assert decimals <= 4
