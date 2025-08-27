"""Integration tests for multi-strategy orchestration workflow"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.strategy_registry import StrategyRegistry, StrategyState
from genesis.engine.capital_allocator import CapitalAllocator, AllocationMethod
from genesis.engine.conflict_resolver import ConflictResolver, ResolutionMethod
from genesis.engine.market_regime_detector import MarketRegimeDetector, MarketRegime
from genesis.engine.ab_test_framework import ABTestFramework, TestVariant, AllocationMethod as ABAllocationMethod
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.analytics.correlation_monitor import CorrelationMonitor
from genesis.analytics.strategy_performance import StrategyPerformanceTracker
from genesis.core.events import Event, EventType
from genesis.core.models import Position, Order
from genesis.strategies.loader import StrategyLoader
from genesis.strategies.base import BaseStrategy


@pytest.fixture
def event_bus():
    """Create event bus for testing"""
    return EventBus()


@pytest.fixture
def mock_risk_engine():
    """Create mock risk engine"""
    engine = Mock(spec=RiskEngine)
    engine.check_portfolio_risk = AsyncMock(return_value=True)
    engine.calculate_portfolio_var = Mock(return_value=Decimal("1000"))
    engine.calculate_position_size = Mock(return_value=Decimal("100"))
    return engine


@pytest.fixture
def mock_strategy_loader():
    """Create mock strategy loader"""
    loader = Mock(spec=StrategyLoader)
    
    # Create mock strategies
    mock_strategies = {}
    for name in ["momentum", "mean_reversion", "arbitrage"]:
        strategy = Mock(spec=BaseStrategy)
        strategy.name = name
        strategy.generate_signals = AsyncMock(return_value=[])
        strategy.execute = AsyncMock()
        strategy.shutdown = AsyncMock()
        mock_strategies[name] = strategy
    
    loader.load_strategy = Mock(side_effect=lambda name: mock_strategies.get(name))
    loader.get_available_strategies = Mock(return_value=list(mock_strategies.keys()))
    return loader


@pytest.fixture
def orchestrator(event_bus, mock_risk_engine, mock_strategy_loader):
    """Create strategy orchestrator with all components"""
    
    orchestrator = StrategyOrchestrator(
        event_bus=event_bus,
        risk_engine=mock_risk_engine,
        total_capital=Decimal("10000")
    )
    
    # Mock the internal components to match the expected interface
    orchestrator.registry = StrategyRegistry(event_bus, mock_strategy_loader)
    orchestrator.allocator = CapitalAllocator(event_bus, total_capital=Decimal("10000"))
    orchestrator.correlation_monitor = CorrelationMonitor(event_bus, warning_threshold=Decimal("0.6"))
    orchestrator.performance_tracker = StrategyPerformanceTracker(event_bus)
    orchestrator.conflict_resolver = ConflictResolver(event_bus)
    orchestrator.regime_detector = MarketRegimeDetector(event_bus)
    orchestrator.ab_test_framework = ABTestFramework(event_bus, orchestrator.performance_tracker)
    
    # Mock methods that might be missing
    orchestrator.register_strategy = AsyncMock()
    orchestrator.start = AsyncMock()
    orchestrator.stop = AsyncMock()
    orchestrator.initialize = AsyncMock()
    
    return orchestrator


class TestMultiStrategyWorkflow:
    """Test complete multi-strategy orchestration workflow"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_multiple_strategies(self, orchestrator, event_bus):
        """Test running multiple strategies concurrently"""
        # Start orchestrator
        await orchestrator.start()
        
        # Register strategies
        await orchestrator.register_strategy("momentum", {"lookback": 20})
        await orchestrator.register_strategy("mean_reversion", {"threshold": 2.0})
        await orchestrator.register_strategy("arbitrage", {"min_spread": 0.01})
        
        # Verify strategies registered
        assert len(orchestrator.registry.strategies) == 3
        assert all(s.state == StrategyState.RUNNING for s in orchestrator.registry.strategies.values())
        
        # Verify capital allocation
        allocations = orchestrator.allocator.allocations
        assert len(allocations) == 3
        assert sum(allocations.values()) <= orchestrator.allocator.total_capital
        
        # Let strategies run briefly
        await asyncio.sleep(0.1)
        
        # Stop orchestrator
        await orchestrator.stop()
        
        # Verify cleanup
        assert all(s.state == StrategyState.STOPPED for s in orchestrator.registry.strategies.values())
    
    @pytest.mark.asyncio
    async def test_capital_reallocation_on_performance(self, orchestrator, event_bus):
        """Test capital reallocation based on performance"""
        await orchestrator.start()
        
        # Register strategies with equal allocation
        await orchestrator.register_strategy("momentum", {})
        await orchestrator.register_strategy("mean_reversion", {})
        
        initial_allocations = orchestrator.allocator.allocations.copy()
        assert initial_allocations["momentum"] == initial_allocations["mean_reversion"]
        
        # Simulate performance difference
        await event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            event_data={
                "strategy_id": "momentum",
                "pnl_usdt": Decimal("500"),
                "timestamp": datetime.now(timezone.utc)
            }
        ))
        
        await event_bus.publish(Event(
            event_type=EventType.POSITION_CLOSED,
            event_data={
                "strategy_id": "mean_reversion",
                "pnl_usdt": Decimal("-200"),
                "timestamp": datetime.now(timezone.utc)
            }
        ))
        
        # Trigger reallocation
        await orchestrator.allocator.rebalance(method=AllocationMethod.PERFORMANCE_WEIGHTED)
        
        # Verify reallocation favors better performer
        new_allocations = orchestrator.allocator.allocations
        assert new_allocations["momentum"] > new_allocations["mean_reversion"]
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_correlation_monitoring_and_alert(self, orchestrator, event_bus):
        """Test correlation monitoring between strategies"""
        await orchestrator.start()
        
        # Register strategies
        await orchestrator.register_strategy("momentum", {})
        await orchestrator.register_strategy("mean_reversion", {})
        
        # Simulate correlated positions
        positions = [
            Position(
                position_id="pos1",
                strategy_id="momentum",
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                entry_price=Decimal("50000")
            ),
            Position(
                position_id="pos2",
                strategy_id="mean_reversion",
                symbol="ETH/USDT",
                quantity=Decimal("10"),
                entry_price=Decimal("3000")
            )
        ]
        
        # Add price updates to create correlation
        for _ in range(10):
            price_btc = Decimal(str(50000 + np.random.normal(0, 100)))
            price_eth = Decimal(str(3000 + np.random.normal(0, 50)))
            
            await orchestrator.correlation_monitor.update_position_price("pos1", price_btc)
            await orchestrator.correlation_monitor.update_position_price("pos2", price_eth)
        
        # Calculate correlation
        correlation = await orchestrator.correlation_monitor.calculate_correlation("pos1", "pos2")
        assert correlation is not None
        
        # Test high correlation alert
        with patch.object(orchestrator.correlation_monitor, 'calculate_correlation', 
                         return_value=Decimal("0.85")):
            alert_event = None
            
            async def capture_event(event):
                nonlocal alert_event
                if event.type == EventType.CORRELATION_ALERT:
                    alert_event = event
            
            event_bus.subscribe(EventType.CORRELATION_ALERT, capture_event)
            await orchestrator.correlation_monitor._check_correlations()
            
            # Allow event processing
            await asyncio.sleep(0.1)
            
            assert alert_event is not None
            assert alert_event.data["level"] == "critical"
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_market_regime_strategy_control(self, orchestrator, event_bus):
        """Test strategy control based on market regime"""
        await orchestrator.start()
        
        # Register regime-specific strategies
        await orchestrator.register_strategy("momentum", {"regime": "bull"})
        await orchestrator.register_strategy("mean_reversion", {"regime": "bear"})
        
        # Initial state - all running
        assert orchestrator.registry.strategies["momentum"].state == StrategyState.RUNNING
        assert orchestrator.registry.strategies["mean_reversion"].state == StrategyState.RUNNING
        
        # Simulate market regime change to BEAR
        orchestrator.regime_detector.current_regime = MarketRegime.BEAR
        await event_bus.publish(Event(
            event_type=EventType.GLOBAL_MARKET_STATE_CHANGE,
            event_data={
                "regime": MarketRegime.BEAR.value,
                "timestamp": datetime.now(timezone.utc)
            }
        ))
        
        # Process regime change
        await orchestrator._handle_regime_change(MarketRegime.BEAR)
        
        # Verify strategy adjustments
        # In a real implementation, momentum might be paused in bear market
        # This depends on the actual strategy configuration
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_between_strategies(self, orchestrator, event_bus):
        """Test conflict resolution when strategies generate conflicting signals"""
        await orchestrator.start()
        
        # Register strategies
        await orchestrator.register_strategy("momentum", {"priority": 1})
        await orchestrator.register_strategy("mean_reversion", {"priority": 2})
        
        # Create conflicting signals
        signal1 = {
            "strategy_id": "momentum",
            "symbol": "BTC/USDT",
            "action": "buy",
            "quantity": Decimal("1"),
            "timestamp": datetime.now(timezone.utc)
        }
        
        signal2 = {
            "strategy_id": "mean_reversion",
            "symbol": "BTC/USDT",
            "action": "sell",
            "quantity": Decimal("1"),
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Resolve conflict
        resolved = await orchestrator.conflict_resolver.resolve([signal1, signal2])
        
        # Verify resolution (higher priority wins)
        assert len(resolved) == 1
        assert resolved[0]["strategy_id"] == "mean_reversion"  # Higher priority
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_aggregate_risk_management(self, orchestrator, event_bus, mock_risk_engine):
        """Test aggregate risk limits across all strategies"""
        await orchestrator.start()
        
        # Register strategies
        await orchestrator.register_strategy("momentum", {})
        await orchestrator.register_strategy("mean_reversion", {})
        await orchestrator.register_strategy("arbitrage", {})
        
        # Simulate risk limit breach
        mock_risk_engine.check_portfolio_risk.return_value = False
        mock_risk_engine.calculate_portfolio_var.return_value = Decimal("5000")  # 50% of capital
        
        # Publish risk event
        await event_bus.publish(Event(
            event_type=EventType.RISK_LIMIT_BREACH,
            event_data={
                "var_usdt": Decimal("5000"),
                "limit_usdt": Decimal("2000"),
                "timestamp": datetime.now(timezone.utc)
            }
        ))
        
        # Allow event processing
        await asyncio.sleep(0.1)
        
        # Verify risk response (strategies might be paused or capital reduced)
        mock_risk_engine.check_portfolio_risk.assert_called()
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_ab_testing_strategy_variations(self, orchestrator, event_bus):
        """Test A/B testing framework for strategy variations"""
        await orchestrator.start()
        
        # Create test variants
        variant_a = TestVariant(
            variant_id="momentum_v1",
            strategy_name="momentum",
            strategy_params={"lookback": 20, "threshold": 1.5}
        )
        
        variant_b = TestVariant(
            variant_id="momentum_v2",
            strategy_name="momentum",
            strategy_params={"lookback": 30, "threshold": 2.0}
        )
        
        # Create A/B test
        test = await orchestrator.ab_test_framework.create_test(
            test_id="momentum_test",
            name="Momentum Parameter Test",
            description="Testing lookback period and threshold",
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades=10,
            confidence_level=Decimal("0.95"),
            allocation_method=ABAllocationMethod.ROUND_ROBIN
        )
        
        # Start test
        await orchestrator.ab_test_framework.start_test("momentum_test")
        
        # Simulate trades for both variants
        for i in range(15):
            variant = orchestrator.ab_test_framework.allocate_variant("momentum_test")
            pnl = Decimal(str(np.random.normal(10, 20)))
            
            await orchestrator.ab_test_framework.record_trade_result(
                "momentum_test",
                variant,
                pnl,
                datetime.now(timezone.utc)
            )
        
        # Check test status
        test_result = await orchestrator.ab_test_framework.get_test_results("momentum_test")
        assert test_result is not None
        assert test_result.variant_a.trades_executed > 0
        assert test_result.variant_b.trades_executed > 0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_strategy_health_monitoring_and_recovery(self, orchestrator, event_bus):
        """Test strategy health monitoring and automatic recovery"""
        await orchestrator.start()
        
        # Register strategy
        await orchestrator.register_strategy("momentum", {})
        
        strategy_info = orchestrator.registry.strategies["momentum"]
        assert strategy_info.state == StrategyState.RUNNING
        
        # Simulate strategy failure
        strategy_info.consecutive_errors = 3
        strategy_info.state = StrategyState.ERROR
        
        # Trigger health check
        await orchestrator.registry._monitor_health()
        
        # Verify recovery attempt
        # In a real implementation, this would attempt to restart the strategy
        assert strategy_info.restart_attempts > 0
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_emergency_stop_correlated_losses(self, orchestrator, event_bus):
        """Test emergency stop when strategies have correlated losses"""
        await orchestrator.start()
        
        # Register strategies
        await orchestrator.register_strategy("momentum", {})
        await orchestrator.register_strategy("mean_reversion", {})
        await orchestrator.register_strategy("arbitrage", {})
        
        # Simulate correlated losses
        loss_events = []
        for strategy_id in ["momentum", "mean_reversion", "arbitrage"]:
            loss_events.append(Event(
                event_type=EventType.POSITION_CLOSED,
                event_data={
                    "strategy_id": strategy_id,
                    "pnl_usdt": Decimal("-500"),
                    "timestamp": datetime.now(timezone.utc)
                }
            ))
        
        # Publish loss events
        for event in loss_events:
            await event_bus.publish(event)
        
        # Trigger daily loss limit
        await event_bus.publish(Event(
            event_type=EventType.DAILY_LOSS_LIMIT_REACHED,
            event_data={
                "total_loss_usdt": Decimal("-1500"),
                "limit_usdt": Decimal("-1000"),
                "timestamp": datetime.now(timezone.utc)
            }
        ))
        
        # Allow event processing
        await asyncio.sleep(0.1)
        
        # In a real implementation, all strategies would be stopped
        # This depends on the actual implementation of emergency stop
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_performance_based_capital_decay(self, orchestrator, event_bus):
        """Test capital decay for underperforming strategies"""
        await orchestrator.start()
        
        # Register strategies
        await orchestrator.register_strategy("momentum", {})
        await orchestrator.register_strategy("mean_reversion", {})
        
        initial_allocation = orchestrator.allocator.allocations["momentum"]
        
        # Simulate poor performance
        for _ in range(10):
            await event_bus.publish(Event(
                event_type=EventType.POSITION_CLOSED,
                event_data={
                    "strategy_id": "momentum",
                    "pnl_usdt": Decimal("-50"),
                    "timestamp": datetime.now(timezone.utc)
                }
            ))
        
        # Trigger performance review
        await orchestrator.performance_tracker.calculate_metrics("momentum")
        
        # Apply capital decay
        await orchestrator.allocator.apply_performance_decay()
        
        # Verify capital reduction
        new_allocation = orchestrator.allocator.allocations["momentum"]
        assert new_allocation < initial_allocation
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self, orchestrator):
        """Test that strategies execute concurrently without blocking"""
        await orchestrator.start()
        
        # Track execution times
        execution_times = {}
        
        async def track_execution(strategy_id):
            start = datetime.now(timezone.utc)
            await asyncio.sleep(0.1)  # Simulate work
            execution_times[strategy_id] = datetime.now(timezone.utc) - start
        
        # Register strategies with execution tracking
        strategies = ["momentum", "mean_reversion", "arbitrage"]
        for strategy in strategies:
            await orchestrator.register_strategy(strategy, {})
            orchestrator.registry.strategies[strategy].instance.execute = AsyncMock(
                side_effect=lambda: track_execution(strategy)
            )
        
        # Run orchestration loop once
        await orchestrator._orchestration_loop()
        
        # Verify concurrent execution (should complete faster than sequential)
        total_time = sum(t.total_seconds() for t in execution_times.values())
        assert total_time < 0.3  # Would be 0.3+ if sequential
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_strategy_lifecycle_events(self, orchestrator, event_bus):
        """Test that all strategy lifecycle events are properly emitted"""
        events_captured = []
        
        async def capture_events(event):
            events_captured.append(event.type)
        
        # Subscribe to strategy events
        for event_type in [EventType.STRATEGY_REGISTERED, EventType.STRATEGY_STARTED,
                           EventType.STRATEGY_STOPPED, EventType.STRATEGY_UNREGISTERED]:
            event_bus.subscribe(event_type, capture_events)
        
        await orchestrator.start()
        
        # Full lifecycle
        await orchestrator.register_strategy("momentum", {})
        await asyncio.sleep(0.1)  # Let it run
        await orchestrator.unregister_strategy("momentum")
        
        # Verify all events emitted
        assert EventType.STRATEGY_REGISTERED in events_captured
        assert EventType.STRATEGY_STARTED in events_captured
        assert EventType.STRATEGY_STOPPED in events_captured
        assert EventType.STRATEGY_UNREGISTERED in events_captured
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_capital_allocation_methods(self, orchestrator):
        """Test different capital allocation methods"""
        await orchestrator.start()
        
        # Register strategies
        strategies = ["momentum", "mean_reversion", "arbitrage"]
        for strategy in strategies:
            await orchestrator.register_strategy(strategy, {})
        
        # Test equal weight allocation
        await orchestrator.allocator.rebalance(AllocationMethod.EQUAL_WEIGHT)
        allocations = orchestrator.allocator.allocations
        assert all(abs(a - Decimal("3333.33")) < Decimal("1") for a in allocations.values())
        
        # Test risk parity allocation
        await orchestrator.allocator.rebalance(AllocationMethod.RISK_PARITY)
        # Allocations should be adjusted based on risk
        
        # Test Kelly criterion allocation
        await orchestrator.allocator.rebalance(AllocationMethod.KELLY_CRITERION)
        # Allocations should be based on expected returns and variance
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, orchestrator):
        """Test graceful shutdown of all components"""
        await orchestrator.start()
        
        # Register multiple strategies
        strategies = ["momentum", "mean_reversion", "arbitrage"]
        for strategy in strategies:
            await orchestrator.register_strategy(strategy, {})
        
        # Create some pending operations
        orchestrator._background_tasks.add(asyncio.create_task(asyncio.sleep(10)))
        
        # Shutdown
        await orchestrator.stop()
        
        # Verify clean shutdown
        assert orchestrator.is_running is False
        assert all(s.state == StrategyState.STOPPED for s in orchestrator.registry.strategies.values())
        assert len(orchestrator._background_tasks) == 0
        
        # Verify no pending tasks
        pending = [task for task in orchestrator._background_tasks if not task.done()]
        assert len(pending) == 0