"""Stress tests for multi-strategy orchestration under extreme conditions."""

from __future__ import annotations

import asyncio
import gc
import time
import tracemalloc
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from genesis.core.models import Signal, SignalType
from genesis.engine.strategy_orchestrator import StrategyOrchestrator


@pytest.fixture
def mock_dependencies():
    """Create all mock dependencies for orchestrator."""
    return {
        "event_bus": AsyncMock(),
        "strategy_registry": AsyncMock(),
        "capital_allocator": AsyncMock(),
        "correlation_monitor": AsyncMock(),
        "conflict_resolver": AsyncMock(),
        "risk_engine": AsyncMock(),
        "regime_detector": AsyncMock(),
        "performance_tracker": AsyncMock(),
        "ab_framework": AsyncMock(),
    }


@pytest.fixture
def stress_orchestrator(mock_dependencies):
    """Create orchestrator for stress testing."""
    orchestrator = StrategyOrchestrator(**mock_dependencies)

    # Setup default return values
    mock_dependencies["strategy_registry"].get_active_strategies.return_value = []
    mock_dependencies["capital_allocator"].get_allocation.return_value = Decimal(
        "10000"
    )
    mock_dependencies["risk_engine"].check_risk_limits.return_value = True
    mock_dependencies["risk_engine"].calculate_portfolio_risk.return_value = Decimal(
        "0.05"
    )
    mock_dependencies["regime_detector"].detect_regime.return_value = "BULL"

    return orchestrator


class TestHighConcurrencyStress:
    """Stress tests for high-concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_extreme_concurrent_strategies(self, stress_orchestrator):
        """Test with 1000+ concurrent strategies."""
        num_strategies = 1000
        strategies = []

        for i in range(num_strategies):
            strategy = MagicMock()
            strategy.name = f"strategy_{i}"
            strategy.generate_signals = AsyncMock(
                return_value=[
                    Signal(
                        signal_type=SignalType.BUY if i % 3 == 0 else SignalType.SELL,
                        symbol=f"COIN{i % 100}/USDT",
                        confidence=Decimal(str(0.5 + (i % 50) / 100)),
                        metadata={"strategy_id": i},
                    )
                ]
            )
            strategies.append(strategy)

        stress_orchestrator.strategy_registry.get_active_strategies.return_value = (
            strategies
        )

        # Measure processing under extreme load
        start_time = time.perf_counter()

        # Process all strategies concurrently
        await stress_orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Should handle 1000 strategies in under 5 seconds
        assert (
            elapsed_time < 5.0
        ), f"Processing {num_strategies} strategies took {elapsed_time:.2f}s"

        # Verify all strategies were called
        assert all(s.generate_signals.called for s in strategies)

    @pytest.mark.asyncio
    async def test_signal_burst_processing(self, stress_orchestrator):
        """Test handling sudden burst of 10,000+ signals."""
        # Create a strategy that generates massive signal burst
        burst_strategy = MagicMock()
        burst_strategy.name = "burst_strategy"

        # Generate 10,000 signals in a burst
        burst_signals = [
            Signal(
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                symbol=f"COIN{i % 500}/USDT",
                confidence=Decimal(str(0.5 + (i % 100) / 200)),
                metadata={"burst_index": i, "timestamp": time.time()},
            )
            for i in range(10000)
        ]

        burst_strategy.generate_signals = AsyncMock(return_value=burst_signals)
        stress_orchestrator.strategy_registry.get_active_strategies.return_value = [
            burst_strategy
        ]

        # Setup conflict resolver to handle the burst
        stress_orchestrator.conflict_resolver.resolve_conflicts.return_value = (
            burst_signals[:100]
        )

        # Measure burst processing
        start_time = time.perf_counter()

        await stress_orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Should process 10,000 signal burst in under 2 seconds
        assert (
            elapsed_time < 2.0
        ), f"Processing 10,000 signal burst took {elapsed_time:.2f}s"

    @pytest.mark.asyncio
    async def test_cascading_conflict_resolution(self, stress_orchestrator):
        """Test conflict resolution with 500+ conflicting strategies."""
        num_strategies = 500
        strategies = []

        # All strategies generate conflicting signals for same symbols
        for i in range(num_strategies):
            strategy = MagicMock()
            strategy.name = f"conflict_strategy_{i}"

            # Create conflicting signals across 10 symbols
            signals = [
                Signal(
                    signal_type=SignalType.BUY if (i + j) % 2 == 0 else SignalType.SELL,
                    symbol=f"CONFLICT{j}/USDT",
                    confidence=Decimal(str(0.6 + (i % 40) / 100)),
                    metadata={"strategy": strategy.name, "priority": i},
                )
                for j in range(10)
            ]

            strategy.generate_signals = AsyncMock(return_value=signals)
            strategies.append(strategy)

        stress_orchestrator.strategy_registry.get_active_strategies.return_value = (
            strategies
        )

        # Setup resolver to consolidate conflicts
        resolved_signals = [
            Signal(
                signal_type=SignalType.BUY,
                symbol=f"CONFLICT{j}/USDT",
                confidence=Decimal("0.85"),
                metadata={"resolved": True, "conflict_count": num_strategies},
            )
            for j in range(10)
        ]
        stress_orchestrator.conflict_resolver.resolve_conflicts.return_value = (
            resolved_signals
        )

        # Measure cascading conflict resolution
        start_time = time.perf_counter()

        await stress_orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Should resolve 5000+ conflicts in under 3 seconds
        assert (
            elapsed_time < 3.0
        ), f"Resolving {num_strategies * 10} conflicts took {elapsed_time:.2f}s"

    @pytest.mark.asyncio
    async def test_continuous_reallocation_stress(self, stress_orchestrator):
        """Test continuous capital reallocation under stress."""
        num_strategies = 100
        num_reallocations = 1000

        # Setup strategies with varying performance
        performance_data = {
            f"strategy_{i}": {
                "sharpe_ratio": Decimal(str(np.random.uniform(0.5, 2.5))),
                "win_rate": Decimal(str(np.random.uniform(0.4, 0.7))),
                "max_drawdown": Decimal(str(np.random.uniform(0.05, 0.25))),
            }
            for i in range(num_strategies)
        }

        stress_orchestrator.performance_tracker.get_performance_metrics.return_value = (
            performance_data
        )

        # Measure continuous reallocation performance
        start_time = time.perf_counter()

        for _ in range(num_reallocations):
            # Simulate performance changes
            for key in performance_data:
                performance_data[key]["sharpe_ratio"] *= Decimal(
                    str(np.random.uniform(0.95, 1.05))
                )

            await stress_orchestrator.capital_allocator.allocate_capital(
                total_capital=Decimal("1000000"),
                strategy_names=list(performance_data.keys()),
                performance_metrics=performance_data,
            )

        elapsed_time = time.perf_counter() - start_time

        # Should handle 1000 reallocations in under 5 seconds
        assert (
            elapsed_time < 5.0
        ), f"Processing {num_reallocations} reallocations took {elapsed_time:.2f}s"

    @pytest.mark.asyncio
    async def test_correlation_matrix_explosion(self, stress_orchestrator):
        """Test correlation calculation with 200+ positions."""
        num_positions = 200

        # Simulate correlation updates for all position pairs
        start_time = time.perf_counter()

        # Calculate correlations for all pairs
        correlation_count = 0
        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                await stress_orchestrator.correlation_monitor.update_correlation(
                    f"POS{i}/USDT",
                    f"POS{j}/USDT",
                    Decimal(str(np.random.uniform(-0.5, 0.95))),
                )
                correlation_count += 1

        elapsed_time = time.perf_counter() - start_time

        # Should handle 19,900 correlations in under 10 seconds
        assert (
            elapsed_time < 10.0
        ), f"Calculating {correlation_count} correlations took {elapsed_time:.2f}s"

    @pytest.mark.asyncio
    async def test_event_storm_resilience(self, stress_orchestrator):
        """Test resilience under event storm conditions."""
        num_events = 100000

        # Generate event storm
        start_time = time.perf_counter()

        tasks = []
        for i in range(num_events):
            event_data = {
                "type": (
                    "SIGNAL" if i % 3 == 0 else "RISK" if i % 3 == 1 else "PERFORMANCE"
                ),
                "timestamp": time.time(),
                "data": {"index": i, "critical": i % 100 == 0},
            }

            task = stress_orchestrator.event_bus.publish(event_data)
            tasks.append(task)

            # Process in batches to avoid overwhelming
            if len(tasks) >= 1000:
                await asyncio.gather(*tasks)
                tasks = []

        # Process remaining
        if tasks:
            await asyncio.gather(*tasks)

        elapsed_time = time.perf_counter() - start_time

        # Should handle 100,000 events in under 20 seconds
        assert (
            elapsed_time < 20.0
        ), f"Processing {num_events} events took {elapsed_time:.2f}s"

        # Calculate throughput
        throughput = num_events / elapsed_time
        assert (
            throughput > 5000
        ), f"Event throughput {throughput:.0f} events/s is below minimum"

    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self, stress_orchestrator):
        """Test memory stability under sustained high load."""
        tracemalloc.start()
        initial_snapshot = tracemalloc.take_snapshot()

        # Run sustained load for multiple iterations
        for iteration in range(50):
            # Generate heavy load
            strategies = []
            for i in range(100):
                strategy = MagicMock()
                strategy.name = f"mem_test_{iteration}_{i}"
                strategy.generate_signals = AsyncMock(
                    return_value=[
                        Signal(
                            signal_type=SignalType.BUY,
                            symbol=f"MEM{i}/USDT",
                            confidence=Decimal("0.75"),
                            metadata={"iteration": iteration, "index": i},
                        )
                        for _ in range(100)
                    ]
                )
                strategies.append(strategy)

            stress_orchestrator.strategy_registry.get_active_strategies.return_value = (
                strategies
            )

            # Process signals
            await stress_orchestrator._process_strategy_signals()

            # Force garbage collection
            gc.collect()

        # Check memory growth
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")

        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        growth_mb = total_growth / (1024 * 1024)

        tracemalloc.stop()

        # Memory growth should be less than 50MB for sustained load
        assert growth_mb < 50, f"Memory grew by {growth_mb:.2f}MB under sustained load"

    @pytest.mark.asyncio
    async def test_parallel_ab_test_stress(self, stress_orchestrator):
        """Test stress with 100+ parallel A/B tests."""
        num_tests = 100
        num_trades_per_test = 1000

        # Create parallel A/B tests
        start_time = time.perf_counter()

        tasks = []
        for test_id in range(num_tests):
            for trade_id in range(num_trades_per_test):
                variant = "control" if trade_id % 2 == 0 else "variant"
                profit = Decimal(str(np.random.uniform(-100, 200)))

                task = stress_orchestrator.ab_framework.record_trade(
                    test_id=f"stress_test_{test_id}",
                    variant=variant,
                    profit=profit,
                    metadata={"test": test_id, "trade": trade_id},
                )
                tasks.append(task)

                # Process in batches
                if len(tasks) >= 5000:
                    await asyncio.gather(*tasks)
                    tasks = []

        # Process remaining
        if tasks:
            await asyncio.gather(*tasks)

        elapsed_time = time.perf_counter() - start_time

        # Should handle 100,000 A/B test trades in under 30 seconds
        assert (
            elapsed_time < 30.0
        ), f"Processing {num_tests * num_trades_per_test} A/B trades took {elapsed_time:.2f}s"

    @pytest.mark.asyncio
    async def test_rapid_regime_changes(self, stress_orchestrator):
        """Test rapid market regime changes impact."""
        regimes = ["BULL", "BEAR", "CRAB", "CRASH", "RECOVERY"]
        num_changes = 1000

        # Setup strategies sensitive to regime
        strategies = []
        for i in range(50):
            strategy = MagicMock()
            strategy.name = f"regime_sensitive_{i}"
            strategy.generate_signals = AsyncMock(return_value=[])
            strategies.append(strategy)

        stress_orchestrator.strategy_registry.get_active_strategies.return_value = (
            strategies
        )

        # Simulate rapid regime changes
        start_time = time.perf_counter()

        for i in range(num_changes):
            new_regime = regimes[i % len(regimes)]
            stress_orchestrator.regime_detector.detect_regime.return_value = new_regime

            # Trigger regime-based adjustments
            await stress_orchestrator._handle_regime_change(new_regime)

            # Some strategies should be paused/resumed based on regime
            if i % 10 == 0:
                # Simulate strategy pause/resume
                for strategy in strategies[:10]:
                    await stress_orchestrator.strategy_registry.update_health(
                        strategy.name, is_healthy=new_regime != "CRASH"
                    )

        elapsed_time = time.perf_counter() - start_time

        # Should handle 1000 regime changes in under 5 seconds
        assert (
            elapsed_time < 5.0
        ), f"Processing {num_changes} regime changes took {elapsed_time:.2f}s"

    @pytest.mark.asyncio
    async def test_cascading_risk_limits(self, stress_orchestrator):
        """Test cascading risk limit breaches under stress."""
        num_strategies = 200
        num_risk_checks = 10000

        # Create strategies with varying risk profiles
        strategies = []
        for i in range(num_strategies):
            strategy = MagicMock()
            strategy.name = f"risk_strategy_{i}"
            strategy.risk_level = Decimal(str(np.random.uniform(0.01, 0.15)))
            strategies.append(strategy)

        # Simulate cascading risk checks
        start_time = time.perf_counter()

        for check in range(num_risk_checks):
            # Simulate portfolio risk calculation
            portfolio_risk = Decimal(str(np.random.uniform(0.02, 0.25)))
            stress_orchestrator.risk_engine.calculate_portfolio_risk.return_value = (
                portfolio_risk
            )

            # Check risk limits
            if portfolio_risk > Decimal("0.20"):
                # Trigger risk reduction
                stress_orchestrator.risk_engine.check_risk_limits.return_value = False

                # Reduce positions for high-risk strategies
                for strategy in strategies[:20]:
                    await stress_orchestrator._reduce_strategy_exposure(strategy.name)
            else:
                stress_orchestrator.risk_engine.check_risk_limits.return_value = True

        elapsed_time = time.perf_counter() - start_time

        # Should handle 10,000 risk checks in under 10 seconds
        assert (
            elapsed_time < 10.0
        ), f"Processing {num_risk_checks} risk checks took {elapsed_time:.2f}s"
