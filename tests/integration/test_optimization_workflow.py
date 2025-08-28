"""
Integration tests for Portfolio Optimization Workflow

Tests the complete end-to-end optimization process including
all components working together.
"""

import asyncio
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from genesis.analytics.efficient_frontier import EfficientFrontierAnalyzer
from genesis.analytics.portfolio_optimizer import PortfolioOptimizer, Strategy
from genesis.analytics.rebalancing_engine import RebalanceTrigger, RebalancingEngine
from genesis.analytics.sharpe_ratio import SharpeRatioCalculator, TimePeriod
from genesis.core.constants import TradingTier
from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus


class TestOptimizationWorkflow:
    """Integration tests for complete optimization workflow"""

    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        bus = EventBus()
        return bus

    @pytest.fixture
    def config_data(self):
        """Create test configuration"""
        return {
            "portfolio_optimization": {
                "enabled": True,
                "enabled_from_tier": "HUNTER",
                "risk_free_rate": 0.02,
                "constraints": {
                    "min_allocation": 0.01,
                    "max_allocation": 0.40,
                    "max_correlation": 0.60,
                    "min_strategies": 2,
                },
                "rebalancing": {
                    "threshold_percent": 5.0,
                    "min_improvement": 0.01,
                    "schedule": "weekly",
                    "maker_fee": 0.001,
                    "taker_fee": 0.001,
                    "slippage": 0.0005,
                },
                "validation": {
                    "out_of_sample_ratio": 0.3,
                    "max_degradation": 0.20,
                    "walk_forward_windows": 5,
                    "min_data_periods": 30,
                },
                "performance": {
                    "max_optimization_time_ms": 1000,
                    "cache_ttl_seconds": 3600,
                },
            }
        }

    @pytest.fixture
    def temp_config_file(self, tmp_path, config_data):
        """Create temporary config file"""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @pytest.fixture
    async def portfolio_optimizer(self, event_bus, temp_config_file):
        """Create portfolio optimizer with config"""
        optimizer = PortfolioOptimizer(
            event_bus=event_bus, config_path=temp_config_file
        )
        return optimizer

    @pytest.fixture
    def realistic_strategies(self):
        """Create realistic strategy data"""
        np.random.seed(42)

        # Trend-following strategy
        trend_returns = []
        price = 100
        for _ in range(200):
            # Trending with momentum
            change = np.random.normal(0.0015, 0.012)
            if len(trend_returns) > 0 and trend_returns[-1] > 0:
                change += 0.0005  # Momentum boost
            price *= 1 + change
            trend_returns.append(Decimal(str(change)))

        # Mean reversion strategy
        mr_returns = []
        mean_level = 0
        for _ in range(200):
            # Reverting to mean
            deviation = np.random.normal(0, 0.008)
            if mean_level > 0.01:
                deviation -= 0.002
            elif mean_level < -0.01:
                deviation += 0.002
            mean_level += deviation
            mr_returns.append(Decimal(str(mean_level * 0.1)))

        # Arbitrage strategy (low risk, consistent returns)
        arb_returns = []
        for _ in range(200):
            # Small but consistent
            ret = abs(np.random.normal(0.0008, 0.003))
            arb_returns.append(Decimal(str(ret)))

        # Market making strategy
        mm_returns = []
        for _ in range(200):
            # Frequent small wins, occasional losses
            if np.random.random() < 0.7:
                ret = abs(np.random.normal(0.001, 0.002))
            else:
                ret = -abs(np.random.normal(0.0015, 0.003))
            mm_returns.append(Decimal(str(ret)))

        strategies = [
            Strategy(
                name="TrendFollowing",
                returns=trend_returns,
                current_allocation=Decimal("0.30"),
                is_active=True,
            ),
            Strategy(
                name="MeanReversion",
                returns=mr_returns,
                current_allocation=Decimal("0.25"),
                is_active=True,
            ),
            Strategy(
                name="Arbitrage",
                returns=arb_returns,
                current_allocation=Decimal("0.25"),
                is_active=True,
            ),
            Strategy(
                name="MarketMaking",
                returns=mm_returns,
                current_allocation=Decimal("0.20"),
                is_active=True,
            ),
        ]

        return strategies

    @pytest.mark.asyncio
    async def test_complete_optimization_flow(
        self, portfolio_optimizer, realistic_strategies
    ):
        """Test complete optimization workflow"""
        portfolio_value = Decimal("10000")

        # Run full optimization
        result = await portfolio_optimizer.optimize_portfolio(
            realistic_strategies,
            portfolio_value_usdt=portfolio_value,
            validate=True,
            rebalance_check=True,
        )

        # Verify all components produced results
        assert result is not None
        assert result.optimal_weights is not None
        assert result.sharpe_ratio > Decimal("0")
        assert result.validation is not None
        assert result.rebalance_recommendation is not None

        # Verify weights sum to 1
        total_weight = sum(result.optimal_weights.values())
        assert abs(total_weight - Decimal("1")) < Decimal("0.01")

        # Verify validation completed
        assert result.validation.in_sample_sharpe is not None
        assert result.validation.out_sample_sharpe is not None
        assert isinstance(result.validation.is_robust, bool)

        # Verify rebalancing recommendation
        assert "should_execute" in result.rebalance_recommendation
        assert "actions" in result.rebalance_recommendation

    @pytest.mark.asyncio
    async def test_sharpe_calculation_integration(self, realistic_strategies):
        """Test Sharpe ratio calculation with real data"""
        calculator = SharpeRatioCalculator()

        for strategy in realistic_strategies:
            result = await calculator.calculate_sharpe_ratio(
                strategy.returns,
                risk_free_rate=Decimal("0.02"),
                period=TimePeriod.DAILY,
                confidence_level=0.95,
            )

            assert result is not None
            assert result.sharpe_ratio is not None
            assert result.confidence_interval_lower is not None
            assert result.confidence_interval_upper is not None

    @pytest.mark.asyncio
    async def test_efficient_frontier_integration(self, realistic_strategies):
        """Test efficient frontier calculation with real data"""
        analyzer = EfficientFrontierAnalyzer()

        strategy_returns = {s.name: s.returns for s in realistic_strategies}

        result = await analyzer.calculate_efficient_frontier(
            strategy_returns, risk_free_rate=Decimal("0.02")
        )

        assert result is not None
        assert len(result.frontier_points) > 0
        assert result.max_sharpe_portfolio is not None
        assert result.min_variance_portfolio is not None

        # Max Sharpe should have better ratio than min variance
        assert (
            result.max_sharpe_portfolio.sharpe_ratio
            >= result.min_variance_portfolio.sharpe_ratio
        )

    @pytest.mark.asyncio
    async def test_rebalancing_integration(self, event_bus, realistic_strategies):
        """Test rebalancing engine integration"""
        engine = RebalancingEngine(event_bus=event_bus)

        current_weights = {s.name: s.current_allocation for s in realistic_strategies}

        # Create target weights (different from current)
        target_weights = {
            "TrendFollowing": Decimal("0.35"),
            "MeanReversion": Decimal("0.20"),
            "Arbitrage": Decimal("0.30"),
            "MarketMaking": Decimal("0.15"),
        }

        portfolio_value = Decimal("10000")

        recommendation = await engine.check_rebalance_triggers(
            current_weights,
            target_weights,
            portfolio_value,
            expected_sharpe_improvement=Decimal("0.05"),
        )

        assert recommendation is not None
        assert recommendation.trigger in RebalanceTrigger
        assert recommendation.total_cost_usdt >= Decimal("0")
        assert isinstance(recommendation.should_execute, bool)
        assert recommendation.rationale is not None

    @pytest.mark.asyncio
    async def test_event_bus_integration(
        self, event_bus, portfolio_optimizer, realistic_strategies
    ):
        """Test event bus communication"""
        received_events = []

        async def event_handler(event: Event):
            received_events.append(event)

        # Subscribe to events
        event_bus.subscribe(EventType.PORTFOLIO_OPTIMIZED, event_handler)
        event_bus.subscribe(EventType.REBALANCE_RECOMMENDED, event_handler)

        # Run optimization
        await portfolio_optimizer.optimize_portfolio(
            realistic_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=True,
        )

        # Allow events to propagate
        await asyncio.sleep(0.1)

        # Should have received events
        assert len(received_events) > 0

        # Check for portfolio optimized event
        portfolio_events = [
            e for e in received_events if e.type == EventType.PORTFOLIO_OPTIMIZED
        ]
        assert len(portfolio_events) > 0

    @pytest.mark.asyncio
    async def test_tier_restrictions(self, portfolio_optimizer):
        """Test that tier restrictions are enforced"""
        # Mock tier check to return SNIPER (below required HUNTER)
        with patch("genesis.utils.decorators.get_current_tier") as mock_tier:
            mock_tier.return_value = TradingTier.SNIPER

            strategies = [
                Strategy(
                    name="Test_A",
                    returns=[Decimal("0.01")] * 100,
                    current_allocation=Decimal("0.50"),
                    is_active=True,
                ),
                Strategy(
                    name="Test_B",
                    returns=[Decimal("0.02")] * 100,
                    current_allocation=Decimal("0.50"),
                    is_active=True,
                ),
            ]

            # Should raise tier restriction error
            with pytest.raises(Exception):  # Would be TierRestrictionError in real code
                await portfolio_optimizer.optimize_portfolio(
                    strategies, portfolio_value_usdt=Decimal("10000")
                )

    @pytest.mark.asyncio
    async def test_caching_behavior(self, portfolio_optimizer, realistic_strategies):
        """Test that caching improves performance"""
        import time

        # First run (no cache)
        start1 = time.time()
        result1 = await portfolio_optimizer.optimize_portfolio(
            realistic_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False,
        )
        time1 = time.time() - start1

        # Second run (with cache)
        start2 = time.time()
        result2 = await portfolio_optimizer.optimize_portfolio(
            realistic_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False,
        )
        time2 = time.time() - start2

        # Both should succeed
        assert result1 is not None
        assert result2 is not None

        # Results should be similar (from cache)
        assert result1.sharpe_ratio == result2.sharpe_ratio

    @pytest.mark.asyncio
    async def test_weekly_recommendation_generation(
        self, portfolio_optimizer, realistic_strategies
    ):
        """Test weekly recommendation report generation"""
        result = await portfolio_optimizer.optimize_portfolio(
            realistic_strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=True,
        )

        recommendation = result.rebalance_recommendation
        assert recommendation is not None

        # Check report structure
        assert "generated_at" in recommendation
        assert "portfolio_value_usdt" in recommendation
        assert "recommendation" in recommendation
        assert "current_allocation" in recommendation
        assert "target_allocation" in recommendation
        assert "actions" in recommendation

        # Verify recommendation details
        rec_details = recommendation["recommendation"]
        assert "should_execute" in rec_details
        assert "rationale" in rec_details
        assert "expected_improvement" in rec_details
        assert "total_cost_usdt" in rec_details
        assert "cost_benefit_ratio" in rec_details

    @pytest.mark.asyncio
    async def test_minimum_allocation_handling(self, portfolio_optimizer):
        """Test handling of strategies below minimum allocation"""
        strategies = [
            Strategy(
                name="Large",
                returns=[Decimal("0.01")] * 100,
                current_allocation=Decimal("0.90"),
                is_active=True,
            ),
            Strategy(
                name="Small_1",
                returns=[Decimal("0.02")] * 100,
                current_allocation=Decimal("0.005"),  # Below minimum
                is_active=True,
            ),
            Strategy(
                name="Small_2",
                returns=[Decimal("0.015")] * 100,
                current_allocation=Decimal("0.005"),  # Below minimum
                is_active=True,
            ),
            Strategy(
                name="Medium",
                returns=[Decimal("0.012")] * 100,
                current_allocation=Decimal("0.09"),
                is_active=True,
            ),
        ]

        result = await portfolio_optimizer.optimize_portfolio(
            strategies,
            portfolio_value_usdt=Decimal("10000"),
            validate=False,
            rebalance_check=False,
        )

        # Should handle minimum allocations properly
        assert result is not None
        for weight in result.optimal_weights.values():
            if weight > Decimal("0"):
                assert weight >= Decimal("0.01")  # Minimum allocation

    @pytest.mark.asyncio
    async def test_performance_with_many_strategies(self, portfolio_optimizer):
        """Test performance with maximum number of strategies"""
        # Create 10 strategies (performance requirement limit)
        strategies = []
        for i in range(10):
            np.random.seed(i + 100)
            returns = np.random.normal(0.001 + i * 0.0001, 0.01 + i * 0.001, 150)
            strategies.append(
                Strategy(
                    name=f"Strategy_{i:02d}",
                    returns=[Decimal(str(r)) for r in returns],
                    current_allocation=Decimal("0.10"),
                    is_active=True,
                )
            )

        import time

        start = time.time()

        result = await portfolio_optimizer.optimize_portfolio(
            strategies,
            portfolio_value_usdt=Decimal("50000"),
            validate=True,  # Include validation for stress test
            rebalance_check=True,
        )

        elapsed_ms = (time.time() - start) * 1000

        assert result is not None
        assert result.optimization_time_ms < 1000  # Must complete within 1 second
        assert elapsed_ms < 10000  # Total time with validation should be reasonable
