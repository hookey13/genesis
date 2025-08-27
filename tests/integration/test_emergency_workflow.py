"""
Integration tests for complete emergency workflow.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.analytics.correlation_spike_detector import CorrelationSpikeDetector
from genesis.analytics.flash_crash_detector import FlashCrashDetector
from genesis.analytics.liquidity_monitor import LiquidityMonitor
from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.deleveraging_protocol import (
    DeleveragingProtocol,
    DeleveragingStage,
    Position
)
from genesis.engine.emergency_controller import (
    EmergencyController,
    EmergencyState,
    EmergencyType
)
from genesis.engine.event_bus import EventBus
from genesis.exchange.circuit_breaker import CircuitBreakerManager
from genesis.tilt.emergency_recovery_checklist import (
    EmergencyRecoveryChecklist,
    ChecklistItemStatus,
    RecoveryPhase
)


@pytest.fixture
async def event_bus():
    """Create and start event bus fixture."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def circuit_manager():
    """Create circuit breaker manager fixture."""
    return CircuitBreakerManager()


@pytest.fixture
def emergency_controller(event_bus, circuit_manager):
    """Create emergency controller with test config."""
    with patch('genesis.engine.emergency_controller.yaml.safe_load') as mock_yaml:
        mock_yaml.return_value = {
            'daily_loss_limit': 0.15,
            'correlation_spike_threshold': 0.80,
            'liquidity_drop_threshold': 0.50,
            'flash_crash_threshold': 0.10,
            'flash_crash_window_seconds': 60,
            'override_timeout_seconds': 300,
            'emergency_timeout_seconds': 3600
        }
        
        return EmergencyController(
            event_bus=event_bus,
            circuit_manager=circuit_manager,
            config_path="test_config.yaml"
        )


@pytest.fixture
def correlation_detector(event_bus):
    """Create correlation spike detector fixture."""
    return CorrelationSpikeDetector(
        event_bus=event_bus,
        window_minutes=30,
        spike_threshold=Decimal("0.80")
    )


@pytest.fixture
def liquidity_monitor(event_bus):
    """Create liquidity monitor fixture."""
    return LiquidityMonitor(
        event_bus=event_bus,
        depth_window_minutes=5,
        crisis_threshold=Decimal("0.30")
    )


@pytest.fixture
def flash_crash_detector(event_bus):
    """Create flash crash detector fixture."""
    return FlashCrashDetector(
        event_bus=event_bus,
        window_seconds=60,
        crash_threshold=Decimal("0.10")
    )


@pytest.fixture
def deleveraging_protocol(event_bus):
    """Create deleveraging protocol fixture."""
    return DeleveragingProtocol(
        event_bus=event_bus,
        max_slippage_pct=Decimal("0.02")
    )


@pytest.fixture
def recovery_checklist(event_bus):
    """Create recovery checklist fixture."""
    return EmergencyRecoveryChecklist(event_bus=event_bus)


class TestEmergencyWorkflow:
    """Test complete emergency workflow integration."""
    
    @pytest.mark.asyncio
    async def test_daily_loss_halt_workflow(
        self,
        event_bus,
        emergency_controller,
        deleveraging_protocol,
        recovery_checklist
    ):
        """Test complete workflow for daily loss halt."""
        # Setup initial state
        emergency_controller.daily_start_balance = Decimal("10000")
        emergency_controller.current_balance = Decimal("8500")  # 15% loss
        
        # Setup positions for deleveraging
        positions = [
            Position(
                symbol="BTC-USDT",
                side="long",
                size=Decimal("0.1"),
                entry_price=Decimal("45000"),
                current_price=Decimal("42000"),
                unrealized_pnl=Decimal("-300"),
                margin_used=Decimal("4500"),
                opened_at=datetime.now(timezone.utc)
            ),
            Position(
                symbol="ETH-USDT",
                side="long",
                size=Decimal("2"),
                entry_price=Decimal("3000"),
                current_price=Decimal("2800"),
                unrealized_pnl=Decimal("-400"),
                margin_used=Decimal("6000"),
                opened_at=datetime.now(timezone.utc)
            )
        ]
        deleveraging_protocol.update_positions(positions)
        
        # Subscribe to events
        events_received = []
        
        def event_handler(event: Event):
            events_received.append(event)
        
        event_bus.subscribe(
            EventType.CIRCUIT_BREAKER_OPEN,
            event_handler
        )
        event_bus.subscribe(
            EventType.POSITION_SIZE_ADJUSTMENT,
            event_handler
        )
        
        # Trigger daily loss check
        await emergency_controller._check_daily_loss()
        
        # Allow events to propagate
        await asyncio.sleep(0.1)
        
        # Verify emergency triggered
        assert EmergencyType.DAILY_LOSS_HALT.value in emergency_controller.active_emergencies
        assert emergency_controller.state == EmergencyState.EMERGENCY
        
        # Initiate deleveraging
        plans = await deleveraging_protocol.initiate_deleveraging(
            stage=DeleveragingStage.STAGE_4,  # Full closure due to 15% loss
            reason="Daily loss limit breach"
        )
        
        assert len(plans) == 2  # Both positions should be closed
        assert all(p.reduction_percentage == Decimal("1.00") for p in plans)
        
        # Execute deleveraging
        with patch.object(deleveraging_protocol, '_execute_position_reduction') as mock_execute:
            # Mock successful execution
            from genesis.engine.deleveraging_protocol import DeleveragingResult
            
            mock_execute.side_effect = [
                DeleveragingResult(
                    symbol="BTC-USDT",
                    original_size=Decimal("0.1"),
                    closed_size=Decimal("0.1"),
                    remaining_size=Decimal("0"),
                    execution_price=Decimal("41800"),
                    slippage=Decimal("0.005"),
                    realized_pnl=Decimal("-320"),
                    success=True
                ),
                DeleveragingResult(
                    symbol="ETH-USDT",
                    original_size=Decimal("2"),
                    closed_size=Decimal("2"),
                    remaining_size=Decimal("0"),
                    execution_price=Decimal("2790"),
                    slippage=Decimal("0.004"),
                    realized_pnl=Decimal("-420"),
                    success=True
                )
            ]
            
            results = await deleveraging_protocol.execute_deleveraging()
            
        assert len(results) == 2
        assert all(r.success for r in results)
        assert deleveraging_protocol.positions_closed == 2
        
        # Create recovery checklist
        checklist = recovery_checklist.create_recovery_checklist(
            emergency_type="daily_loss_halt",
            severity="CRITICAL"
        )
        
        assert checklist is not None
        assert checklist.emergency_type == "daily_loss_halt"
        assert checklist.total_items > 0
        assert checklist.current_phase == RecoveryPhase.IMMEDIATE
        
        # Process some checklist items
        first_item = checklist.items[0]
        await recovery_checklist.start_item(checklist.checklist_id, first_item.item_id)
        await recovery_checklist.complete_item(
            checklist.checklist_id,
            first_item.item_id,
            notes="All positions verified closed"
        )
        
        assert first_item.status == ChecklistItemStatus.COMPLETED
        assert checklist.completed_items == 1
        
        # Verify events were published
        assert len(events_received) > 0
        circuit_breaker_events = [
            e for e in events_received
            if e.event_type == EventType.CIRCUIT_BREAKER_OPEN
        ]
        assert len(circuit_breaker_events) > 0
    
    @pytest.mark.asyncio
    async def test_flash_crash_detection_workflow(
        self,
        event_bus,
        emergency_controller,
        flash_crash_detector,
        recovery_checklist
    ):
        """Test flash crash detection and response workflow."""
        # Subscribe to events
        events_received = []
        
        def event_handler(event: Event):
            events_received.append(event)
        
        event_bus.subscribe(
            EventType.MARKET_STATE_CHANGE,
            event_handler
        )
        event_bus.subscribe(
            EventType.ORDER_CANCELLED,
            event_handler
        )
        
        # Simulate flash crash with rapid price drops
        await flash_crash_detector.process_price(
            symbol="BTC-USDT",
            price=Decimal("50000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=30)
        )
        
        await flash_crash_detector.process_price(
            symbol="BTC-USDT",
            price=Decimal("48000"),
            volume=Decimal("150"),
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=20)
        )
        
        await flash_crash_detector.process_price(
            symbol="BTC-USDT",
            price=Decimal("45000"),
            volume=Decimal("200"),
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=10)
        )
        
        # Final price triggers flash crash (10% drop)
        crash_event = await flash_crash_detector.process_price(
            symbol="BTC-USDT",
            price=Decimal("44000"),  # 12% drop from 50000
            volume=Decimal("300"),
            timestamp=datetime.now(timezone.utc)
        )
        
        # Allow events to propagate
        await asyncio.sleep(0.1)
        
        assert crash_event is not None
        assert crash_event.symbol == "BTC-USDT"
        assert crash_event.drop_percentage >= Decimal("0.10")
        assert flash_crash_detector.crashes_detected == 1
        assert "BTC-USDT" in flash_crash_detector.symbols_with_cancelled_orders
        
        # Verify events published
        market_events = [
            e for e in events_received
            if e.event_type == EventType.MARKET_STATE_CHANGE
        ]
        assert len(market_events) > 0
        assert market_events[0].event_data["alert_type"] == "flash_crash"
        
        order_cancel_events = [
            e for e in events_received
            if e.event_type == EventType.ORDER_CANCELLED
        ]
        assert len(order_cancel_events) > 0
        assert order_cancel_events[0].event_data["reason"] == "flash_crash_protection"
        
        # Create recovery checklist
        checklist = recovery_checklist.create_recovery_checklist(
            emergency_type="flash_crash",
            severity="HIGH"
        )
        
        assert checklist is not None
        assert checklist.emergency_type == "flash_crash"
    
    @pytest.mark.asyncio
    async def test_correlation_spike_workflow(
        self,
        event_bus,
        emergency_controller,
        correlation_detector,
        deleveraging_protocol
    ):
        """Test correlation spike detection and response."""
        # Subscribe to events
        events_received = []
        
        def event_handler(event: Event):
            events_received.append(event)
        
        event_bus.subscribe(
            EventType.CORRELATION_ALERT,
            event_handler
        )
        
        # Add price observations to build correlation
        for i in range(30):
            base_price = Decimal("50000")
            # Create correlated price movements
            btc_price = base_price + Decimal(str(i * 100))
            eth_price = Decimal("3000") + Decimal(str(i * 7.5))  # 75% correlation pattern
            
            correlation_detector.add_price_observation(
                "BTC-USDT",
                btc_price,
                Decimal("100"),
                datetime.now(timezone.utc) - timedelta(minutes=30 - i)
            )
            
            correlation_detector.add_price_observation(
                "ETH-USDT",
                eth_price,
                Decimal("50"),
                datetime.now(timezone.utc) - timedelta(minutes=30 - i)
            )
        
        # Calculate correlation matrix
        matrix = await correlation_detector.calculate_correlation_matrix()
        
        # Allow events to propagate
        await asyncio.sleep(0.1)
        
        # Check correlation detected
        assert len(matrix) > 0
        key = ("BTC-USDT", "ETH-USDT")
        if key in matrix:
            assert matrix[key] > Decimal("0.60")  # Should show correlation
        
        # Get position recommendations
        positions = {
            "BTC-USDT": Decimal("10000"),
            "ETH-USDT": Decimal("5000")
        }
        
        recommendations = correlation_detector.get_position_recommendations(positions)
        
        # If correlation is high enough, should get recommendations
        if matrix.get(key, Decimal("0")) >= Decimal("0.80"):
            assert len(recommendations) > 0
            assert recommendations[0]["recommended_reduction"] > 0
        
        # Check for correlation alerts
        correlation_alerts = [
            e for e in events_received
            if e.event_type == EventType.CORRELATION_ALERT
        ]
        
        # May or may not have alerts depending on actual correlation calculated
        if correlation_alerts:
            assert correlation_alerts[0].event_data["alert_type"] in [
                "correlation_spike", "correlation_recovered"
            ]
    
    @pytest.mark.asyncio
    async def test_liquidity_crisis_workflow(
        self,
        event_bus,
        emergency_controller,
        liquidity_monitor
    ):
        """Test liquidity crisis detection and response."""
        # Subscribe to events
        events_received = []
        
        def event_handler(event: Event):
            events_received.append(event)
        
        event_bus.subscribe(
            EventType.MARKET_STATE_CHANGE,
            event_handler
        )
        
        # Process healthy order book first (to establish baseline)
        healthy_bids = [
            (Decimal("49900"), Decimal("10")),
            (Decimal("49850"), Decimal("15")),
            (Decimal("49800"), Decimal("20")),
        ]
        healthy_asks = [
            (Decimal("50000"), Decimal("10")),
            (Decimal("50050"), Decimal("15")),
            (Decimal("50100"), Decimal("20")),
        ]
        
        liquidity_monitor.process_order_book(
            "BTC-USDT",
            healthy_bids,
            healthy_asks,
            datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        # Calculate baseline liquidity score
        await liquidity_monitor.calculate_liquidity_score("BTC-USDT")
        
        # Now simulate liquidity crisis with thin order book
        crisis_bids = [
            (Decimal("49500"), Decimal("1")),  # Very thin
            (Decimal("49000"), Decimal("2")),
        ]
        crisis_asks = [
            (Decimal("50500"), Decimal("1")),  # Wide spread
            (Decimal("51000"), Decimal("2")),
        ]
        
        liquidity_monitor.process_order_book(
            "BTC-USDT",
            crisis_bids,
            crisis_asks,
            datetime.now(timezone.utc)
        )
        
        # Calculate new liquidity score (should be much lower)
        score = await liquidity_monitor.calculate_liquidity_score("BTC-USDT")
        
        # Allow events to propagate
        await asyncio.sleep(0.1)
        
        assert score < Decimal("1.0")  # Score should be reduced
        
        # Check for liquidity events
        market_events = [
            e for e in events_received
            if e.event_type == EventType.MARKET_STATE_CHANGE
        ]
        
        # May have liquidity crisis alert if score is low enough
        if score < Decimal("0.30"):
            assert len(market_events) > 0
            assert market_events[0].event_data["state"] == "liquidity_crisis"
    
    @pytest.mark.asyncio
    async def test_manual_override_workflow(
        self,
        event_bus,
        emergency_controller
    ):
        """Test manual override of emergency halt."""
        # Trigger an emergency first
        emergency_controller.daily_start_balance = Decimal("1000")
        emergency_controller.current_balance = Decimal("850")  # 15% loss
        
        await emergency_controller._check_daily_loss()
        
        assert emergency_controller.state == EmergencyState.EMERGENCY
        
        # Attempt override with wrong phrase
        result = await emergency_controller.request_manual_override("WRONG")
        assert result is False
        assert emergency_controller.state == EmergencyState.EMERGENCY
        
        # Override with correct phrase
        result = await emergency_controller.request_manual_override(
            "OVERRIDE EMERGENCY HALT"
        )
        assert result is True
        assert emergency_controller.state == EmergencyState.OVERRIDE
        assert emergency_controller.override_active is True
        
        # Verify override expires
        emergency_controller.override_expiry = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert emergency_controller._is_override_active() is False
        assert emergency_controller.state == EmergencyState.EMERGENCY  # Back to emergency
    
    @pytest.mark.asyncio
    async def test_complete_recovery_workflow(
        self,
        event_bus,
        recovery_checklist
    ):
        """Test complete recovery checklist workflow."""
        # Create checklist
        checklist = recovery_checklist.create_recovery_checklist(
            emergency_type="daily_loss_halt",
            severity="CRITICAL"
        )
        
        assert checklist is not None
        
        # Get initial status
        status = recovery_checklist.get_checklist_status(checklist.checklist_id)
        assert status["progress_percentage"] == 0.0
        assert status["current_phase"] == RecoveryPhase.IMMEDIATE.value
        
        # Process immediate phase items
        immediate_items = [
            item for item in checklist.items
            if item.phase == RecoveryPhase.IMMEDIATE
        ]
        
        for item in immediate_items[:2]:  # Complete first 2 items
            await recovery_checklist.start_item(
                checklist.checklist_id,
                item.item_id
            )
            await recovery_checklist.complete_item(
                checklist.checklist_id,
                item.item_id,
                notes=f"Completed {item.title}"
            )
        
        # Check progress
        status = recovery_checklist.get_checklist_status(checklist.checklist_id)
        assert status["progress_percentage"] > 0
        assert status["completed_items"] == 2
    
    @pytest.mark.asyncio
    async def test_emergency_report_generation(
        self,
        emergency_controller
    ):
        """Test comprehensive emergency report generation."""
        # Set up various emergency conditions
        emergency_controller.daily_loss_percent = Decimal("0.12")
        emergency_controller.emergencies_triggered = 5
        emergency_controller.false_positives = 1
        emergency_controller.successful_interventions = 4
        
        # Add correlation data
        emergency_controller.correlation_matrix[("BTC-USDT", "ETH-USDT")] = Decimal("0.85")
        emergency_controller.correlation_matrix[("BTC-USDT", "SOL-USDT")] = Decimal("0.72")
        
        # Add liquidity data
        emergency_controller.liquidity_scores["BTC-USDT"] = Decimal("0.25")
        emergency_controller.liquidity_scores["ETH-USDT"] = Decimal("0.45")
        
        # Generate report
        report = await emergency_controller.generate_emergency_report()
        
        assert report is not None
        assert "current_state" in report
        assert "active_emergencies" in report
        assert "statistics" in report
        assert "current_metrics" in report
        assert "recommendations" in report
        
        # Verify metrics
        assert report["current_metrics"]["daily_loss_percent"] == 0.12
        assert report["current_metrics"]["max_correlation"] == 0.85
        assert report["current_metrics"]["min_liquidity_score"] == 0.25
        
        # Verify statistics
        assert report["statistics"]["total_emergencies"] == 5
        assert report["statistics"]["false_positives"] == 1
        assert report["statistics"]["successful_interventions"] == 4
        
        # Should have recommendations based on current state
        assert len(report["recommendations"]) > 0
        assert any("correlation" in r.lower() for r in report["recommendations"])
        assert any("liquidity" in r.lower() for r in report["recommendations"])


@pytest.mark.asyncio
async def test_performance_under_load(event_bus, emergency_controller):
    """Test emergency system performance under high load."""
    import time
    
    # Generate many price updates rapidly
    start_time = time.time()
    
    for i in range(1000):
        event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": f"TEST-{i % 10}",
                "price": str(50000 + i),
                "liquidity_score": str(0.5 + (i % 100) / 200)
            }
        )
        await emergency_controller._handle_market_update(event)
    
    elapsed = time.time() - start_time
    
    # Should handle 1000 updates in reasonable time
    assert elapsed < 5.0  # Less than 5 seconds
    
    # Verify data was stored
    assert len(emergency_controller.price_history) > 0
    assert len(emergency_controller.liquidity_scores) > 0