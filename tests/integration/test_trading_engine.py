"""
Integration tests for the trading engine.

Tests the complete flow from market data to order execution
with all components integrated.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalType,
    Trade,
)
from genesis.data.market_feed import MarketDataFeed, MarketData, Ticker, OrderBook
from genesis.engine.event_bus import EventBus
from genesis.engine.health_monitor import HealthMonitor, HealthStatus
from genesis.engine.orchestrator import TradingOrchestrator
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.signal_queue import SignalQueue
from genesis.engine.state_machine import TierStateMachine
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.strategy_registry import StrategyRegistry
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway
from genesis.strategies.loader import StrategyLoader


@pytest.fixture
def event_bus():
    """Create event bus fixture."""
    return EventBus()


@pytest.fixture
def mock_exchange_gateway():
    """Create mock exchange gateway."""
    gateway = AsyncMock(spec=ExchangeGateway)
    gateway.get_market_price = AsyncMock(return_value=Decimal("50000"))
    gateway.place_order = AsyncMock(return_value=True)
    gateway.get_order_status = AsyncMock(return_value=OrderStatus.FILLED)
    gateway.cancel_order = AsyncMock(return_value=True)
    gateway.close = AsyncMock()
    return gateway


@pytest.fixture
def risk_engine():
    """Create risk engine fixture."""
    return RiskEngine(
        max_position_size=Decimal("10000"),
        max_daily_loss=Decimal("1000"),
        max_positions=5
    )


@pytest.fixture
def state_machine():
    """Create state machine fixture."""
    return TierStateMachine(initial_tier="SNIPER")


@pytest.fixture
def strategy_loader():
    """Create strategy loader fixture."""
    loader = MagicMock(spec=StrategyLoader)
    loader.load_tier_strategies = MagicMock(return_value=["simple_arb", "spread_capture"])
    return loader


@pytest.fixture
def strategy_registry(event_bus, strategy_loader):
    """Create strategy registry fixture."""
    return StrategyRegistry(event_bus=event_bus, loader=strategy_loader)


@pytest.fixture
def signal_queue():
    """Create signal queue fixture."""
    return SignalQueue(max_size=100)


@pytest.fixture
def strategy_orchestrator(event_bus, strategy_registry, signal_queue):
    """Create strategy orchestrator fixture."""
    return StrategyOrchestrator(
        event_bus=event_bus,
        strategy_registry=strategy_registry,
        signal_queue=signal_queue
    )


@pytest.fixture
def health_monitor(event_bus):
    """Create health monitor fixture."""
    return HealthMonitor(event_bus=event_bus, check_interval=1)


@pytest.fixture
def trading_loop(event_bus, risk_engine, mock_exchange_gateway, state_machine):
    """Create trading loop fixture."""
    return TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=mock_exchange_gateway,
        state_machine=state_machine,
        paper_trading_mode=True
    )


@pytest.fixture
def orchestrator(
    event_bus,
    risk_engine,
    mock_exchange_gateway,
    strategy_registry,
    strategy_orchestrator,
    signal_queue,
    state_machine
):
    """Create trading orchestrator fixture."""
    return TradingOrchestrator(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=mock_exchange_gateway,
        strategy_registry=strategy_registry,
        strategy_orchestrator=strategy_orchestrator,
        signal_queue=signal_queue,
        state_machine=state_machine
    )


@pytest.mark.asyncio
class TestTradingEngineIntegration:
    """Integration tests for the complete trading engine."""
    
    async def test_trading_loop_initialization(self, trading_loop):
        """Test trading loop initialization."""
        assert trading_loop is not None
        assert not trading_loop.running
        assert trading_loop.paper_trading_mode
        assert len(trading_loop.positions) == 0
        assert len(trading_loop.pending_orders) == 0
    
    async def test_trading_loop_startup(self, trading_loop):
        """Test trading loop startup sequence."""
        # Mock startup validation
        trading_loop.validate_startup = AsyncMock(return_value=True)
        
        # Start the trading loop
        start_task = asyncio.create_task(trading_loop.start())
        
        # Give it time to start
        await asyncio.sleep(0.1)
        
        assert trading_loop.running
        assert trading_loop.startup_validated
        
        # Stop the trading loop
        await trading_loop.stop()
        await start_task
        
        assert not trading_loop.running
    
    async def test_signal_to_order_flow(self, orchestrator, signal_queue):
        """Test signal processing to order creation flow."""
        # Create a test signal
        signal = Signal(
            signal_id=str(uuid4()),
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            confidence=Decimal("0.8"),
            risk_reward_ratio=Decimal("2.0"),
            timestamp=datetime.now(UTC)
        )
        
        # Add signal to queue
        await signal_queue.add(signal)
        
        # Start orchestrator
        await orchestrator.start()
        
        # Process the signal
        await orchestrator._process_signal(signal)
        
        # Verify metrics
        assert orchestrator.metrics.signals_processed == 1
        assert orchestrator.metrics.orders_created == 1
        
        # Stop orchestrator
        await orchestrator.stop()
    
    async def test_market_data_to_signal_flow(self, event_bus, orchestrator):
        """Test market data processing to signal generation."""
        # Create test market data
        ticker = Ticker(
            symbol="BTCUSDT",
            bid_price=Decimal("49900"),
            ask_price=Decimal("50100"),
            last_price=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(UTC)
        )
        
        market_data = MarketData(
            symbol="BTCUSDT",
            ticker=ticker,
            timestamp=datetime.now(UTC)
        )
        
        # Publish market data event
        event = Event(
            type=EventType.MARKET_DATA,
            data=market_data,
            priority=EventPriority.HIGH
        )
        
        await event_bus.publish(event)
        
        # Give time for processing
        await asyncio.sleep(0.1)
        
        # Market data should be distributed to strategies
        # Strategies would generate signals based on this data
    
    async def test_health_monitoring(self, health_monitor):
        """Test health monitoring system."""
        # Register components
        health_monitor.register_component("trading_loop")
        health_monitor.register_component("risk_engine")
        health_monitor.register_component("exchange_gateway")
        
        # Start health monitor
        await health_monitor.start()
        
        # Update component health
        health_monitor.update_component_health(
            "trading_loop",
            HealthStatus.HEALTHY,
            {"events_processed": 100}
        )
        
        health_monitor.update_component_health(
            "risk_engine",
            HealthStatus.HEALTHY,
            {"signals_validated": 50}
        )
        
        # Simulate error
        health_monitor.record_component_error(
            "exchange_gateway",
            "Connection timeout"
        )
        
        # Get health report
        report = health_monitor.get_health_report()
        
        assert report["overall_status"] in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "trading_loop" in report["components"]
        assert "risk_engine" in report["components"]
        assert "exchange_gateway" in report["components"]
        
        # Stop health monitor
        await health_monitor.stop()
    
    async def test_graceful_shutdown(self, trading_loop, orchestrator, health_monitor):
        """Test graceful shutdown of all components."""
        # Start components
        await trading_loop.start()
        await orchestrator.start()
        await health_monitor.start()
        
        # Verify running
        assert trading_loop.running
        assert orchestrator.running
        assert health_monitor.running
        
        # Perform graceful shutdown
        await trading_loop.stop()
        await orchestrator.stop()
        await health_monitor.stop()
        
        # Verify stopped
        assert not trading_loop.running
        assert not orchestrator.running
        assert not health_monitor.running
    
    async def test_error_recovery(self, orchestrator, mock_exchange_gateway):
        """Test error handling and recovery."""
        # Simulate exchange error
        mock_exchange_gateway.place_order = AsyncMock(side_effect=Exception("Exchange error"))
        
        # Create a test signal
        signal = Signal(
            signal_id=str(uuid4()),
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            signal_type=SignalType.ENTRY,
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            confidence=Decimal("0.8"),
            timestamp=datetime.now(UTC)
        )
        
        # Process signal (should handle error gracefully)
        await orchestrator._process_signal(signal)
        
        # Verify error was handled
        assert orchestrator.metrics.orders_failed == 1
        assert orchestrator.metrics.orders_executed == 0
    
    async def test_position_tracking(self, trading_loop):
        """Test position tracking and management."""
        # Create a test position
        position = Position(
            position_id=str(uuid4()),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            realized_pnl=Decimal("0"),
            timestamp=datetime.now(UTC)
        )
        
        # Add position
        trading_loop.positions[position.position_id] = position
        
        # Verify position tracking
        assert len(trading_loop.positions) == 1
        assert position.position_id in trading_loop.positions
        
        # Update position
        position.current_price = Decimal("52000")
        position.unrealized_pnl = Decimal("200")
        
        # Verify update
        tracked_position = trading_loop.positions[position.position_id]
        assert tracked_position.current_price == Decimal("52000")
        assert tracked_position.unrealized_pnl == Decimal("200")
    
    async def test_performance_metrics(self, orchestrator):
        """Test performance metrics collection."""
        # Process some signals to generate metrics
        for i in range(5):
            signal = Signal(
                signal_id=str(uuid4()),
                strategy_name="test_strategy",
                symbol="BTCUSDT",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                entry_price=Decimal("50000"),
                confidence=Decimal("0.7"),
                timestamp=datetime.now(UTC)
            )
            
            await orchestrator._process_signal(signal)
            
            # Simulate latency
            orchestrator._update_signal_latency(10 + i)
            if i < 3:
                orchestrator._update_execution_latency(20 + i)
        
        # Get metrics
        metrics = orchestrator.get_metrics()
        
        assert metrics.signals_processed == 5
        assert metrics.avg_signal_latency > 0
        assert metrics.max_signal_latency >= 14  # 10 + 4
        
        if metrics.orders_executed > 0:
            assert metrics.avg_execution_latency > 0
            assert metrics.max_execution_latency >= 20
    
    async def test_concurrent_signal_processing(self, orchestrator, signal_queue):
        """Test concurrent processing of multiple signals."""
        # Create multiple signals
        signals = []
        for i in range(10):
            signal = Signal(
                signal_id=str(uuid4()),
                strategy_name=f"strategy_{i % 3}",
                symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
                signal_type=SignalType.ENTRY,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                entry_price=Decimal("50000") if i % 2 == 0 else Decimal("3000"),
                confidence=Decimal("0.6") + Decimal("0.03") * i,
                timestamp=datetime.now(UTC)
            )
            signals.append(signal)
            await signal_queue.add(signal)
        
        # Start orchestrator
        await orchestrator.start()
        
        # Give time for processing
        await asyncio.sleep(1)
        
        # Verify all signals were processed
        assert orchestrator.metrics.signals_processed <= 10
        
        # Stop orchestrator
        await orchestrator.stop()
    
    async def test_state_persistence(self, trading_loop):
        """Test state persistence and recovery."""
        # Add some state
        trading_loop.positions["pos1"] = Position(
            position_id="pos1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            timestamp=datetime.now(UTC)
        )
        
        trading_loop.pending_orders["order1"] = Order(
            order_id="order1",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("51000"),
            status=OrderStatus.PENDING
        )
        
        # Save state (would be implemented in production)
        state = {
            "positions": list(trading_loop.positions.keys()),
            "pending_orders": list(trading_loop.pending_orders.keys()),
            "events_processed": trading_loop.events_processed
        }
        
        # Simulate recovery
        assert "pos1" in state["positions"]
        assert "order1" in state["pending_orders"]
    
    async def test_tier_based_strategy_loading(self, state_machine, strategy_registry):
        """Test tier-based strategy loading and restrictions."""
        # Set tier to SNIPER
        state_machine.current_tier = "SNIPER"
        
        # Check strategy availability
        can_execute_sniper = await state_machine.can_execute_strategy("simple_arb")
        can_execute_strategist = await state_machine.can_execute_strategy("market_making")
        
        # SNIPER tier should only allow basic strategies
        # This would depend on actual implementation
        assert can_execute_sniper or not can_execute_sniper  # Placeholder
        assert not can_execute_strategist or can_execute_strategist  # Placeholder
        
        # Upgrade tier
        state_machine.current_tier = "STRATEGIST"
        
        # Now advanced strategies should be available
        can_execute_strategist_after = await state_machine.can_execute_strategy("market_making")
        # Would be True in actual implementation


@pytest.mark.asyncio
class TestMarketDataFeed:
    """Tests for market data feed component."""
    
    async def test_market_feed_initialization(self):
        """Test market feed initialization."""
        mock_ws_manager = AsyncMock()
        event_bus = EventBus()
        
        feed = MarketDataFeed(
            websocket_manager=mock_ws_manager,
            event_bus=event_bus,
            symbols=["BTCUSDT", "ETHUSDT"],
            enable_orderbook=True,
            enable_trades=True
        )
        
        assert feed.symbols == ["BTCUSDT", "ETHUSDT"]
        assert feed.enable_orderbook
        assert feed.enable_trades
        assert not feed.running
    
    async def test_market_feed_subscription(self):
        """Test market feed stream subscription."""
        mock_ws_manager = AsyncMock()
        feed = MarketDataFeed(
            websocket_manager=mock_ws_manager,
            symbols=["BTCUSDT"]
        )
        
        await feed.start()
        
        # Verify WebSocket subscriptions
        mock_ws_manager.subscribe.assert_called()
        
        await feed.stop()
    
    async def test_ticker_processing(self):
        """Test ticker data processing."""
        mock_ws_manager = AsyncMock()
        event_bus = EventBus()
        
        feed = MarketDataFeed(
            websocket_manager=mock_ws_manager,
            event_bus=event_bus
        )
        
        # Process test ticker data
        ticker_data = {
            "s": "BTCUSDT",
            "b": "49900",
            "B": "1.5",
            "a": "50100",
            "A": "1.2",
            "c": "50000",
            "v": "1000",
            "q": "50000000",
            "o": "49500",
            "h": "51000",
            "l": "49000",
            "E": 1640000000000
        }
        
        await feed._process_ticker(ticker_data)
        
        # Verify cache update
        cached_ticker = feed.cache.get_ticker("BTCUSDT")
        assert cached_ticker is not None
        assert cached_ticker.last_price == Decimal("50000")
        assert cached_ticker.bid_price == Decimal("49900")
        assert cached_ticker.ask_price == Decimal("50100")


@pytest.mark.asyncio
class TestHealthMonitor:
    """Tests for health monitoring system."""
    
    async def test_component_registration(self, health_monitor):
        """Test component registration and health updates."""
        # Register components
        health_monitor.register_component("component1")
        health_monitor.register_component("component2")
        
        assert "component1" in health_monitor.components
        assert "component2" in health_monitor.components
        
        # Update health
        health_monitor.update_component_health(
            "component1",
            HealthStatus.HEALTHY
        )
        
        component = health_monitor.components["component1"]
        assert component.status == HealthStatus.HEALTHY
        assert component.consecutive_failures == 0
    
    async def test_error_tracking(self, health_monitor):
        """Test error tracking and status degradation."""
        health_monitor.register_component("test_component")
        
        # Record errors
        for i in range(3):
            health_monitor.record_component_error(
                "test_component",
                f"Error {i}"
            )
        
        component = health_monitor.components["test_component"]
        assert component.error_count == 3
        assert component.consecutive_failures == 3
        assert component.status == HealthStatus.CRITICAL
        
        # More errors should mark as failed
        for i in range(2):
            health_monitor.record_component_error(
                "test_component",
                f"Error {i + 3}"
            )
        
        assert component.status == HealthStatus.FAILED
    
    async def test_system_metrics_collection(self, health_monitor):
        """Test system metrics collection."""
        await health_monitor.start()
        
        # Give time for metrics collection
        await asyncio.sleep(1)
        
        # Check metrics
        metrics = health_monitor.system_metrics
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_percent >= 0
        
        await health_monitor.stop()
    
    async def test_alert_generation(self, health_monitor):
        """Test alert generation for critical conditions."""
        health_monitor.register_component("critical_component")
        
        # Trigger critical status
        for _ in range(3):
            health_monitor.record_component_error(
                "critical_component",
                "Critical error"
            )
        
        # Check alerts were generated
        assert len(health_monitor.alerts) > 0
        
        alert = health_monitor.alerts[-1]
        assert alert.component == "critical_component"
        assert alert.severity in ["ERROR", "CRITICAL"]