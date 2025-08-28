"""Unit tests for StrategyOrchestrator."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import structlog

from genesis.analytics.correlation_monitor import CorrelationMonitor
from genesis.analytics.strategy_performance import StrategyPerformanceTracker
from genesis.core.events import Event, EventType
from genesis.core.models import Order, Position, Trade
from genesis.engine.capital_allocator import CapitalAllocator, StrategyAllocation
from genesis.engine.conflict_resolver import ConflictResolver
from genesis.engine.event_bus import EventBus
from genesis.engine.market_regime_detector import MarketRegime, MarketRegimeDetector
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.strategy_orchestrator import (
    OrchestrationConfig,
    OrchestrationMode,
    StrategyOrchestrator,
    StrategySignal,
)
from genesis.engine.strategy_registry import (
    StrategyMetadata,
    StrategyPriority,
    StrategyRegistry,
    StrategyState,
)

# Configure structured logging for tests
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)


class TestOrchestrationConfig:
    """Tests for OrchestrationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OrchestrationConfig()
        assert config.max_concurrent_strategies == 10
        assert config.min_strategy_capital == Decimal("100")
        assert config.correlation_check_interval == 300
        assert config.performance_update_interval == 3600
        assert config.regime_check_interval == 900
        assert config.rebalance_interval == 86400
        assert config.conflict_resolution_enabled is True
        assert config.auto_regime_adjustment is True
        assert config.emergency_stop_loss == Decimal("0.15")

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OrchestrationConfig(
            max_concurrent_strategies=5,
            min_strategy_capital=Decimal("500"),
            emergency_stop_loss=Decimal("0.10"),
        )
        assert config.max_concurrent_strategies == 5
        assert config.min_strategy_capital == Decimal("500")
        assert config.emergency_stop_loss == Decimal("0.10")


class TestStrategySignal:
    """Tests for StrategySignal."""

    def test_signal_creation(self):
        """Test creating a strategy signal."""
        signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            action="buy",
            quantity=Decimal("0.1"),
        )
        assert signal.strategy_id == "test_strategy"
        assert signal.symbol == "BTC/USDT"
        assert signal.action == "buy"
        assert signal.quantity == Decimal("0.1")
        assert signal.confidence == Decimal("1.0")
        assert signal.urgency == "normal"
        assert len(signal.signal_id) > 0

    def test_signal_with_metadata(self):
        """Test signal with custom metadata."""
        metadata = {"reason": "breakout", "strength": "high"}
        signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            action="buy",
            quantity=Decimal("0.1"),
            confidence=Decimal("0.8"),
            urgency="high",
            metadata=metadata,
        )
        assert signal.confidence == Decimal("0.8")
        assert signal.urgency == "high"
        assert signal.metadata == metadata


class TestStrategyOrchestrator:
    """Tests for StrategyOrchestrator."""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create a test orchestrator instance."""
        event_bus = EventBus()
        risk_engine = Mock(spec=RiskEngine)
        risk_engine.check_portfolio_risk = Mock(return_value=True)
        risk_engine.check_risk_limits = Mock(return_value=True)
        risk_engine.calculate_portfolio_risk = AsyncMock(
            return_value={"var": Decimal("0.05"), "sharpe": Decimal("1.2")}
        )

        orchestrator = StrategyOrchestrator(
            event_bus=event_bus,
            risk_engine=risk_engine,
            total_capital=Decimal("10000"),
            config=OrchestrationConfig(),
        )

        # Mock dependencies
        # These are already created in __init__, just need to mock methods
        orchestrator.strategy_registry = Mock(spec=StrategyRegistry)
        orchestrator.strategy_registry.get_active_strategies = Mock(return_value=[])
        orchestrator.strategy_registry._strategies = {}
        orchestrator.strategy_registry.stop_strategy = AsyncMock()
        orchestrator.strategy_registry.unregister_strategy = AsyncMock()
        orchestrator.strategy_registry.pause_strategy = AsyncMock()
        orchestrator.strategy_registry.resume_strategy = AsyncMock()

        orchestrator.capital_allocator = Mock(spec=CapitalAllocator)
        orchestrator.capital_allocator.get_allocations = Mock(return_value={})
        orchestrator.capital_allocator.allocate_capital = AsyncMock(return_value={})

        orchestrator.correlation_monitor = Mock(spec=CorrelationMonitor)
        orchestrator.correlation_monitor.get_correlation_matrix = AsyncMock(
            return_value={}
        )
        orchestrator.correlation_monitor.get_correlation_summary = Mock(return_value={})

        orchestrator.performance_tracker = Mock(spec=StrategyPerformanceTracker)
        orchestrator.performance_tracker.get_performance = AsyncMock(return_value={})
        orchestrator.performance_tracker.initialize_strategy = AsyncMock()
        orchestrator.performance_tracker.get_portfolio_summary = AsyncMock(
            return_value={}
        )

        orchestrator.regime_detector = Mock(spec=MarketRegimeDetector)
        orchestrator.regime_detector.detect_regime = AsyncMock(
            return_value=Mock(value="NORMAL")
        )
        orchestrator.conflict_resolver = Mock(spec=ConflictResolver)

        yield orchestrator

        # Cleanup
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test orchestrator initialization."""
        event_bus = EventBus()
        risk_engine = Mock(spec=RiskEngine)

        orchestrator = StrategyOrchestrator(
            event_bus=event_bus, risk_engine=risk_engine, total_capital=Decimal("10000")
        )

        assert orchestrator.total_capital == Decimal("10000")
        assert orchestrator.mode == OrchestrationMode.NORMAL
        assert len(orchestrator._tasks) == 0
        assert isinstance(orchestrator.config, OrchestrationConfig)
        assert isinstance(orchestrator.strategy_registry, StrategyRegistry)
        assert isinstance(orchestrator.capital_allocator, CapitalAllocator)

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_start_orchestrator(self, orchestrator):
        """Test starting the orchestrator."""
        orchestrator.strategy_registry.start = AsyncMock()
        orchestrator.performance_tracker.start = AsyncMock()
        orchestrator.correlation_monitor.start = AsyncMock()
        orchestrator.regime_detector.start = AsyncMock()

        await orchestrator.start()

        assert len(orchestrator._tasks) > 0  # Tasks should be running
        orchestrator.strategy_registry.start.assert_called_once()
        orchestrator.performance_tracker.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_orchestrator(self, orchestrator):
        """Test stopping the orchestrator."""
        orchestrator._shutdown_event.clear()  # Simulate running state
        orchestrator._background_tasks = [
            asyncio.create_task(asyncio.sleep(10)),
            asyncio.create_task(asyncio.sleep(10)),
        ]

        orchestrator.strategy_registry.stop = AsyncMock()
        orchestrator.performance_tracker.stop = AsyncMock()
        orchestrator.correlation_monitor.stop = AsyncMock()

        await orchestrator.stop()

        assert orchestrator._shutdown_event.is_set()  # Should be stopped
        orchestrator.strategy_registry.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_strategy(self, orchestrator):
        """Test registering a new strategy."""
        strategy_id = str(uuid4())
        metadata = StrategyMetadata(
            name="test_strategy",
            version="1.0.0",
            tier_required="STRATEGIST",
            priority=StrategyPriority.NORMAL,
        )

        orchestrator.strategy_registry.get_active_strategies = Mock(return_value=[])
        orchestrator.strategy_registry.register_strategy = AsyncMock(
            return_value=strategy_id
        )
        orchestrator.capital_allocator.allocate_capital = AsyncMock(
            return_value={
                strategy_id: StrategyAllocation(
                    strategy_id=strategy_id,
                    strategy_name="test_strategy",
                    current_allocation=Decimal("1000"),
                    target_allocation=Decimal("1000"),
                    available_capital=Decimal("900"),
                )
            }
        )

        result = await orchestrator.register_strategy(
            account_id="test_account", strategy_name="test_strategy", metadata=metadata
        )

        assert result == strategy_id
        orchestrator.strategy_registry.register_strategy.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_strategy(self, orchestrator):
        """Test unregistering a strategy."""
        strategy_id = str(uuid4())

        orchestrator.strategy_registry.pause_strategy = AsyncMock()
        orchestrator.strategy_registry.stop_strategy = AsyncMock(return_value=True)
        orchestrator.capital_allocator.unlock_capital = Mock()
        orchestrator.strategy_allocations[strategy_id] = Mock(
            locked_capital=Decimal("100")
        )

        await orchestrator.unregister_strategy(strategy_id)

        # unregister_strategy calls stop_strategy which calls pause_strategy
        orchestrator.strategy_registry.pause_strategy.assert_called_once_with(
            strategy_id
        )

    @pytest.mark.asyncio
    async def test_process_signal(self, orchestrator):
        """Test processing a strategy signal."""
        signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            action="buy",
            quantity=Decimal("0.1"),
        )

        orchestrator.strategy_registry.get_strategy_state = Mock(
            return_value=StrategyState.RUNNING
        )
        orchestrator.conflict_resolver.check_conflicts = Mock(return_value=[])
        orchestrator.risk_engine.check_risk_limits = Mock(return_value=True)
        orchestrator._check_signal_risk = AsyncMock(return_value=True)
        orchestrator._process_signal = AsyncMock()

        result = await orchestrator.submit_signal(signal)

        assert result == True
        assert signal in orchestrator.pending_signals

    @pytest.mark.asyncio
    async def test_process_signal_with_conflict(self, orchestrator):
        """Test processing a signal with conflicts."""
        signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            action="buy",
            quantity=Decimal("0.1"),
        )

        orchestrator.strategy_registry.get_strategy_state = Mock(
            return_value=StrategyState.RUNNING
        )
        orchestrator._check_signal_risk = AsyncMock(return_value=True)
        orchestrator._check_signal_conflicts = AsyncMock(return_value=[signal])
        orchestrator._process_signal = AsyncMock()

        result = await orchestrator.submit_signal(signal)

        assert result == True
        orchestrator._check_signal_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_signal_risk_rejected(self, orchestrator):
        """Test signal rejected by risk engine."""
        signal = StrategySignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            action="buy",
            quantity=Decimal("0.1"),
        )

        orchestrator.strategy_registry.get_strategy_state = Mock(
            return_value=StrategyState.RUNNING
        )
        orchestrator._check_signal_risk = AsyncMock(return_value=False)
        orchestrator._process_signal = AsyncMock()

        result = await orchestrator.submit_signal(signal)

        assert result == False
        orchestrator._process_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_correlation_alert(self, orchestrator):
        """Test handling correlation alerts."""
        orchestrator.mode = OrchestrationMode.NORMAL
        orchestrator._enter_defensive_mode = AsyncMock()

        # Create correlation alert
        alert = {"correlation": Decimal("0.95"), "symbols": ["BTC/USDT", "ETH/USDT"]}

        await orchestrator.handle_correlation_alert(alert)

        # High correlation should trigger defensive mode
        orchestrator._enter_defensive_mode.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_regime_change(self, orchestrator):
        """Test handling market regime changes."""
        orchestrator.pause_strategy = AsyncMock()
        orchestrator.resume_strategy = AsyncMock()

        await orchestrator.handle_regime_change("BULL")

        # Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_emergency_stop(self, orchestrator):
        """Test emergency stop functionality."""
        orchestrator._shutdown_event.clear()  # Simulate running state
        orchestrator.strategy_registry.get_active_strategies = Mock(
            return_value=["strat1", "strat2"]
        )
        orchestrator.strategy_registry.stop_strategy = AsyncMock()
        orchestrator._close_all_positions = AsyncMock()

        await orchestrator.emergency_stop("Critical loss detected")

        assert orchestrator.mode == OrchestrationMode.EMERGENCY
        assert orchestrator.strategy_registry.stop_strategy.call_count == 2
        orchestrator._close_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_allocations(self, orchestrator):
        """Test capital rebalancing."""
        performances = {
            "strat1": {"sharpe_ratio": 1.5, "total_pnl": Decimal("500")},
            "strat2": {"sharpe_ratio": 0.8, "total_pnl": Decimal("-100")},
        }

        orchestrator.performance_tracker.get_all_performances = AsyncMock(
            return_value=performances
        )
        orchestrator.capital_allocator.rebalance = AsyncMock()

        await orchestrator.rebalance_allocations()

        orchestrator.capital_allocator.rebalance.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_portfolio_status(self, orchestrator):
        """Test getting portfolio status."""
        orchestrator.strategy_registry.get_active_strategies = Mock(
            return_value=["strat1"]
        )
        orchestrator.capital_allocator.get_allocations = Mock(
            return_value={
                "strat1": StrategyAllocation(
                    strategy_id="strat1",
                    strategy_name="strat1",
                    current_allocation=Decimal("1000"),
                    target_allocation=Decimal("1000"),
                    available_capital=Decimal("900"),
                )
            }
        )
        orchestrator.performance_tracker.get_performance = Mock(
            return_value={"total_pnl": Decimal("100"), "win_rate": Decimal("0.6")}
        )
        orchestrator._calculate_portfolio_metrics = Mock(
            return_value={
                "total_value": Decimal("10100"),
                "total_pnl": Decimal("100"),
                "total_pnl_pct": Decimal("0.01"),
            }
        )

        status = await orchestrator.get_portfolio_status()

        assert status["mode"] == OrchestrationMode.NORMAL.value
        assert status["active_strategies"] == 1
        assert "available_capital" in status
        assert "performance" in status
        assert "correlation_status" in status

    @pytest.mark.asyncio
    async def test_monitor_portfolio_risk(self, orchestrator):
        """Test portfolio risk monitoring."""
        orchestrator.risk_engine.calculate_portfolio_risk = AsyncMock(
            return_value={"var": Decimal("0.05"), "max_drawdown": Decimal("0.15")}
        )

        risk_metrics = await orchestrator.monitor_portfolio_risk()

        assert "var" in risk_metrics
        assert risk_metrics["var"] == Decimal("0.05")
        orchestrator.risk_engine.calculate_portfolio_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_performance_metrics(self, orchestrator):
        """Test updating performance metrics."""
        orchestrator.strategy_allocations = {
            "strat1": Mock(performance_score=Decimal("1.0"))
        }
        orchestrator.performance_tracker.get_performance = AsyncMock(
            return_value={"sharpe_ratio": Decimal("1.5")}
        )

        await orchestrator.update_performance_metrics()

        # Should have updated the performance score
        assert orchestrator.strategy_allocations["strat1"].performance_score == Decimal(
            "1.5"
        )

    @pytest.mark.asyncio
    async def test_check_correlations(self, orchestrator):
        """Test correlation checking."""
        orchestrator.correlation_monitor.get_correlation_matrix = AsyncMock(
            return_value={"BTC/ETH": Decimal("0.85")}
        )

        correlations = await orchestrator.check_correlations()

        assert "BTC/ETH" in correlations
        assert correlations["BTC/ETH"] == Decimal("0.85")

    def test_orchestration_mode_values(self):
        """Test orchestration mode enum values."""
        assert OrchestrationMode.NORMAL == "normal"
        assert OrchestrationMode.CONSERVATIVE == "conservative"
        assert OrchestrationMode.AGGRESSIVE == "aggressive"
        assert OrchestrationMode.DEFENSIVE == "defensive"
        assert OrchestrationMode.EMERGENCY == "emergency"

    @pytest.mark.asyncio
    async def test_strategy_lifecycle(self, orchestrator):
        """Test complete strategy lifecycle."""
        strategy_id = str(uuid4())
        metadata = StrategyMetadata(
            name="test_strategy", version="1.0.0", tier_required="STRATEGIST"
        )

        # Mock registry methods
        orchestrator.strategy_registry.get_active_strategies = Mock(return_value=[])
        orchestrator.strategy_registry.register_strategy = AsyncMock(
            return_value=strategy_id
        )
        orchestrator.strategy_registry._metadata = {}
        orchestrator.strategy_registry.start_strategy = AsyncMock(return_value=True)
        orchestrator.strategy_registry.pause_strategy = AsyncMock()
        orchestrator.strategy_registry.resume_strategy = AsyncMock()
        orchestrator.strategy_registry.stop_strategy = AsyncMock()
        orchestrator.strategy_registry.unregister_strategy = AsyncMock()

        # Mock allocator
        orchestrator.capital_allocator.allocate_capital = AsyncMock(
            return_value={
                strategy_id: StrategyAllocation(
                    strategy_id=strategy_id,
                    strategy_name="test_strategy",
                    current_allocation=Decimal("1000"),
                    target_allocation=Decimal("1000"),
                    available_capital=Decimal("900"),
                )
            }
        )
        orchestrator.capital_allocator.release_allocation = Mock()
        orchestrator.capital_allocator.unlock_capital = Mock()
        orchestrator.performance_tracker.initialize_strategy = AsyncMock()
        orchestrator.performance_tracker.get_performance = AsyncMock(return_value={})

        # Register
        result_id = await orchestrator.register_strategy(
            "test_account", "test_strategy", metadata
        )
        assert result_id == strategy_id

        # Start
        await orchestrator.start_strategy(strategy_id)
        orchestrator.strategy_registry.start_strategy.assert_called_once_with(
            strategy_id
        )

        # Pause
        await orchestrator.pause_strategy(strategy_id)
        orchestrator.strategy_registry.pause_strategy.assert_called_once_with(
            strategy_id
        )

        # Resume
        await orchestrator.resume_strategy(strategy_id)
        orchestrator.strategy_registry.resume_strategy.assert_called_once_with(
            strategy_id
        )

        # Stop (orchestrator's stop_strategy calls pause_strategy on registry)
        orchestrator.strategy_allocations[strategy_id] = Mock(
            locked_capital=Decimal("100")
        )
        await orchestrator.stop_strategy(strategy_id)
        orchestrator.strategy_registry.pause_strategy.assert_called_with(strategy_id)

        # Unregister (calls stop_strategy again which calls pause_strategy again)
        orchestrator.performance_tracker.stop_tracking_strategy = Mock()
        await orchestrator.unregister_strategy(strategy_id)

    @pytest.mark.asyncio
    async def test_concurrent_strategy_limit(self, orchestrator):
        """Test max concurrent strategies limit."""
        orchestrator.config.max_concurrent_strategies = 2
        orchestrator.strategy_registry.get_active_strategies = Mock(
            return_value=["strat1", "strat2"]
        )

        metadata = StrategyMetadata(
            name="test_strategy", version="1.0.0", tier_required="STRATEGIST"
        )

        # Should fail due to limit
        with pytest.raises(ValueError, match="Maximum concurrent strategies"):
            await orchestrator.register_strategy(
                "test_account", "test_strategy", metadata
            )

    @pytest.mark.asyncio
    async def test_min_capital_requirement(self, orchestrator):
        """Test minimum capital requirement for strategies."""
        orchestrator.config.min_strategy_capital = Decimal("1000")
        orchestrator.strategy_registry.get_active_strategies = Mock(return_value=[])
        orchestrator.strategy_registry.register_strategy = AsyncMock(
            return_value="strat1"
        )
        orchestrator.capital_allocator.allocate_capital = AsyncMock(return_value={})
        orchestrator.performance_tracker.initialize_strategy = AsyncMock()

        metadata = StrategyMetadata(
            name="test_strategy", version="1.0.0", tier_required="STRATEGIST"
        )

        # Should succeed and create proper allocation
        result = await orchestrator.register_strategy(
            "test_account", "test_strategy", metadata
        )

        assert result == "strat1"
        assert "strat1" in orchestrator.strategy_allocations
        assert orchestrator.strategy_allocations["strat1"].min_allocation == Decimal(
            "1000"
        )
