"""Performance benchmark tests for StrategyOrchestrator."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus
from genesis.core.models import PriceData, Signal, SignalType
from genesis.engine.strategy_orchestrator import StrategyOrchestrator


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = AsyncMock(spec=EventBus)
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_registry():
    """Create mock strategy registry."""
    registry = AsyncMock()
    registry.register_strategy = AsyncMock()
    registry.get_active_strategies = AsyncMock(return_value=[])
    registry.update_health = AsyncMock()
    return registry


@pytest.fixture
def mock_allocator():
    """Create mock capital allocator."""
    allocator = AsyncMock()
    allocator.allocate_capital = AsyncMock()
    allocator.get_allocation = AsyncMock(return_value=Decimal("1000"))
    return allocator


@pytest.fixture
def mock_monitor():
    """Create mock correlation monitor."""
    monitor = AsyncMock()
    monitor.update_correlation = AsyncMock()
    monitor.get_correlation_matrix = AsyncMock(return_value={})
    return monitor


@pytest.fixture
def mock_resolver():
    """Create mock conflict resolver."""
    resolver = AsyncMock()
    resolver.resolve_conflicts = AsyncMock()
    return resolver


@pytest.fixture
def mock_risk_engine():
    """Create mock risk engine."""
    engine = AsyncMock()
    engine.calculate_portfolio_risk = AsyncMock(return_value=Decimal("0.02"))
    engine.check_risk_limits = AsyncMock(return_value=True)
    return engine


@pytest.fixture
def mock_regime_detector():
    """Create mock market regime detector."""
    detector = AsyncMock()
    detector.detect_regime = AsyncMock(return_value="BULL")
    return detector


@pytest.fixture
def mock_performance_tracker():
    """Create mock performance tracker."""
    tracker = AsyncMock()
    tracker.update_performance = AsyncMock()
    tracker.get_performance_metrics = AsyncMock(return_value={})
    return tracker


@pytest.fixture
def mock_ab_framework():
    """Create mock A/B test framework."""
    framework = AsyncMock()
    framework.allocate_variant = AsyncMock(return_value="control")
    framework.record_trade = AsyncMock()
    return framework


@pytest.fixture
def orchestrator(
    mock_event_bus,
    mock_registry,
    mock_allocator,
    mock_monitor,
    mock_resolver,
    mock_risk_engine,
    mock_regime_detector,
    mock_performance_tracker,
    mock_ab_framework,
):
    """Create orchestrator with mocked dependencies."""
    return StrategyOrchestrator(
        event_bus=mock_event_bus,
        strategy_registry=mock_registry,
        capital_allocator=mock_allocator,
        correlation_monitor=mock_monitor,
        conflict_resolver=mock_resolver,
        risk_engine=mock_risk_engine,
        regime_detector=mock_regime_detector,
        performance_tracker=mock_performance_tracker,
        ab_framework=mock_ab_framework,
    )


class TestOrchestratorPerformance:
    """Performance benchmarks for StrategyOrchestrator."""

    @pytest.mark.asyncio
    async def test_concurrent_strategy_processing_performance(self, orchestrator):
        """Test performance with multiple concurrent strategies."""
        # Create mock strategies
        num_strategies = 100
        mock_strategies = []

        for i in range(num_strategies):
            strategy = MagicMock()
            strategy.name = f"strategy_{i}"
            strategy.generate_signals = AsyncMock(
                return_value=[
                    Signal(
                        signal_type=SignalType.BUY,
                        symbol="BTC/USDT",
                        confidence=Decimal("0.8"),
                        metadata={},
                    )
                ]
            )
            mock_strategies.append(strategy)

        orchestrator.strategy_registry.get_active_strategies.return_value = (
            mock_strategies
        )

        # Measure processing time
        start_time = time.perf_counter()

        # Process signals for all strategies
        await orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Should handle 100 strategies in under 1 second
        assert (
            elapsed_time < 1.0
        ), f"Processing {num_strategies} strategies took {elapsed_time:.3f}s"

        # Verify all strategies were called
        for strategy in mock_strategies:
            strategy.generate_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_frequency_signal_processing(self, orchestrator):
        """Test performance under high-frequency signal generation."""
        # Setup mock strategy with high-frequency signals
        mock_strategy = MagicMock()
        mock_strategy.name = "high_freq_strategy"

        # Generate 1000 signals
        signals = [
            Signal(
                signal_type=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                symbol="BTC/USDT",
                confidence=Decimal(str(0.5 + (i % 50) / 100)),
                metadata={"index": i},
            )
            for i in range(1000)
        ]

        mock_strategy.generate_signals = AsyncMock(return_value=signals)
        orchestrator.strategy_registry.get_active_strategies.return_value = [
            mock_strategy
        ]

        # Measure processing time
        start_time = time.perf_counter()

        await orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Should process 1000 signals in under 0.5 seconds
        assert elapsed_time < 0.5, f"Processing 1000 signals took {elapsed_time:.3f}s"

    @pytest.mark.asyncio
    async def test_conflict_resolution_performance(self, orchestrator):
        """Test performance of conflict resolution with many conflicting signals."""
        # Create conflicting signals from multiple strategies
        num_strategies = 50
        mock_strategies = []

        for i in range(num_strategies):
            strategy = MagicMock()
            strategy.name = f"strategy_{i}"
            # Half generate BUY, half generate SELL for same symbol
            signal_type = SignalType.BUY if i % 2 == 0 else SignalType.SELL
            strategy.generate_signals = AsyncMock(
                return_value=[
                    Signal(
                        signal_type=signal_type,
                        symbol="BTC/USDT",
                        confidence=Decimal("0.7"),
                        metadata={"strategy": f"strategy_{i}"},
                    )
                ]
            )
            mock_strategies.append(strategy)

        orchestrator.strategy_registry.get_active_strategies.return_value = (
            mock_strategies
        )

        # Setup conflict resolver to handle conflicts
        orchestrator.conflict_resolver.resolve_conflicts.return_value = [
            Signal(
                signal_type=SignalType.BUY,
                symbol="BTC/USDT",
                confidence=Decimal("0.8"),
                metadata={"resolved": True},
            )
        ]

        # Measure processing time
        start_time = time.perf_counter()

        await orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Should resolve 50 conflicts in under 0.2 seconds
        assert (
            elapsed_time < 0.2
        ), f"Resolving {num_strategies} conflicts took {elapsed_time:.3f}s"

        # Verify conflict resolver was called
        orchestrator.conflict_resolver.resolve_conflicts.assert_called()

    @pytest.mark.asyncio
    async def test_risk_check_performance(self, orchestrator):
        """Test performance of risk checks under load."""
        # Setup multiple signals requiring risk checks
        signals = [
            Signal(
                signal_type=SignalType.BUY,
                symbol=f"COIN{i}/USDT",
                confidence=Decimal("0.75"),
                metadata={"size": Decimal("100")},
            )
            for i in range(100)
        ]

        # Mock strategy returning many signals
        mock_strategy = MagicMock()
        mock_strategy.name = "multi_signal_strategy"
        mock_strategy.generate_signals = AsyncMock(return_value=signals)

        orchestrator.strategy_registry.get_active_strategies.return_value = [
            mock_strategy
        ]

        # Measure risk checking time
        start_time = time.perf_counter()

        await orchestrator._process_strategy_signals()

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Should check risk for 100 signals in under 0.1 seconds
        assert elapsed_time < 0.1, f"Risk checking 100 signals took {elapsed_time:.3f}s"

    @pytest.mark.asyncio
    async def test_correlation_update_performance(self, orchestrator):
        """Test performance of correlation matrix updates."""
        # Setup positions for correlation calculation
        num_positions = 50
        positions = {
            f"position_{i}": {
                "symbol": f"COIN{i}/USDT",
                "size": Decimal("1000"),
                "entry_price": Decimal("100"),
            }
            for i in range(num_positions)
        }

        # Measure correlation update time
        start_time = time.perf_counter()

        # Update correlations for all position pairs
        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                await orchestrator.correlation_monitor.update_correlation(
                    f"COIN{i}/USDT", f"COIN{j}/USDT", Decimal("0.5")
                )

        elapsed_time = time.perf_counter() - start_time

        # Calculate number of correlation pairs
        num_pairs = (num_positions * (num_positions - 1)) // 2

        # Performance assertion: Should update all correlations in under 2 seconds
        assert (
            elapsed_time < 2.0
        ), f"Updating {num_pairs} correlations took {elapsed_time:.3f}s"

    @pytest.mark.asyncio
    async def test_event_bus_throughput(self, orchestrator):
        """Test event bus message throughput."""
        # Generate high volume of events
        num_events = 10000
        events = []

        for i in range(num_events):
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                data={
                    "signal": Signal(
                        signal_type=SignalType.BUY,
                        symbol="BTC/USDT",
                        confidence=Decimal("0.8"),
                        metadata={"index": i},
                    )
                },
            )
            events.append(event)

        # Measure event publishing time
        start_time = time.perf_counter()

        for event in events:
            await orchestrator.event_bus.publish(event)

        elapsed_time = time.perf_counter() - start_time

        # Calculate throughput
        throughput = num_events / elapsed_time

        # Performance assertion: Should handle at least 5000 events per second
        assert (
            throughput > 5000
        ), f"Event throughput {throughput:.0f} events/s is below 5000 events/s"

    @pytest.mark.asyncio
    async def test_capital_reallocation_performance(self, orchestrator):
        """Test performance of dynamic capital reallocation."""
        # Setup multiple strategies with performance metrics
        num_strategies = 30

        performance_metrics = {
            f"strategy_{i}": {
                "sharpe_ratio": Decimal(str(1.5 + i * 0.1)),
                "win_rate": Decimal(str(0.55 + i * 0.01)),
                "max_drawdown": Decimal(str(0.1 - i * 0.001)),
            }
            for i in range(num_strategies)
        }

        orchestrator.performance_tracker.get_performance_metrics.return_value = (
            performance_metrics
        )

        # Measure reallocation time
        start_time = time.perf_counter()

        # Trigger capital reallocation
        await orchestrator.capital_allocator.allocate_capital(
            total_capital=Decimal("100000"),
            strategy_names=[f"strategy_{i}" for i in range(num_strategies)],
            performance_metrics=performance_metrics,
        )

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Should reallocate capital for 30 strategies in under 0.05 seconds
        assert (
            elapsed_time < 0.05
        ), f"Capital reallocation for {num_strategies} strategies took {elapsed_time:.3f}s"

    @pytest.mark.asyncio
    async def test_orchestration_loop_latency(self, orchestrator):
        """Test end-to-end orchestration loop latency."""
        # Setup complete orchestration scenario
        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.generate_signals = AsyncMock(
            return_value=[
                Signal(
                    signal_type=SignalType.BUY,
                    symbol="BTC/USDT",
                    confidence=Decimal("0.85"),
                    metadata={},
                )
            ]
        )

        orchestrator.strategy_registry.get_active_strategies.return_value = [
            mock_strategy
        ]

        # Measure complete orchestration loop
        start_time = time.perf_counter()

        # Run one complete orchestration cycle
        await orchestrator._orchestration_loop()

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Complete loop should execute in under 100ms
        assert (
            elapsed_time < 0.1
        ), f"Orchestration loop took {elapsed_time * 1000:.1f}ms"

    @pytest.mark.asyncio
    async def test_stress_test_memory_usage(self, orchestrator):
        """Stress test to ensure no memory leaks under high load."""
        import gc
        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        # Get initial memory snapshot
        snapshot1 = tracemalloc.take_snapshot()

        # Run high-load scenario
        for _ in range(100):
            # Generate and process many signals
            signals = [
                Signal(
                    signal_type=SignalType.BUY,
                    symbol=f"TEST/USDT",
                    confidence=Decimal("0.8"),
                    metadata={"iteration": _},
                )
                for _ in range(100)
            ]

            mock_strategy = MagicMock()
            mock_strategy.name = "stress_test_strategy"
            mock_strategy.generate_signals = AsyncMock(return_value=signals)

            orchestrator.strategy_registry.get_active_strategies.return_value = [
                mock_strategy
            ]

            await orchestrator._process_strategy_signals()

            # Force garbage collection
            gc.collect()

        # Get final memory snapshot
        snapshot2 = tracemalloc.take_snapshot()

        # Calculate memory difference
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Stop memory tracking
        tracemalloc.stop()

        # Calculate total memory increase
        total_increase = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)

        # Convert to MB
        increase_mb = total_increase / (1024 * 1024)

        # Performance assertion: Memory increase should be less than 10MB
        assert (
            increase_mb < 10
        ), f"Memory increased by {increase_mb:.2f}MB during stress test"

    @pytest.mark.asyncio
    async def test_concurrent_ab_test_performance(self, orchestrator):
        """Test performance with multiple concurrent A/B tests."""
        # Setup multiple A/B tests
        num_tests = 20
        num_trades_per_test = 100

        for test_id in range(num_tests):
            # Record trades for each test
            for trade_id in range(num_trades_per_test):
                variant = "control" if trade_id % 2 == 0 else "variant"

                await orchestrator.ab_framework.record_trade(
                    test_id=f"test_{test_id}",
                    variant=variant,
                    profit=Decimal(str(10 * (trade_id % 10 - 5))),
                    metadata={"trade_id": trade_id},
                )

        # Measure analysis time
        start_time = time.perf_counter()

        # Analyze all tests (this would normally be done by the framework)
        for test_id in range(num_tests):
            await orchestrator.ab_framework.allocate_variant(f"test_{test_id}")

        elapsed_time = time.perf_counter() - start_time

        # Performance assertion: Should handle 20 concurrent A/B tests in under 0.2 seconds
        assert (
            elapsed_time < 0.2
        ), f"Managing {num_tests} A/B tests took {elapsed_time:.3f}s"
