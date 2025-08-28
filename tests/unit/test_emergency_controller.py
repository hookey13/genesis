"""
Unit tests for emergency controller.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.emergency_controller import (
    EmergencyController,
    EmergencyState,
    EmergencyType,
)
from genesis.engine.event_bus import EventBus
from genesis.exchange.circuit_breaker import CircuitBreakerManager


@pytest.fixture
def event_bus():
    """Create event bus fixture."""
    bus = EventBus()
    return bus


@pytest.fixture
def circuit_manager():
    """Create circuit breaker manager fixture."""
    return CircuitBreakerManager()


@pytest.fixture
def emergency_controller(event_bus, circuit_manager):
    """Create emergency controller fixture."""
    with patch("genesis.engine.emergency_controller.yaml.safe_load") as mock_yaml:
        mock_yaml.return_value = {
            "daily_loss_limit": 0.15,
            "correlation_spike_threshold": 0.80,
            "liquidity_drop_threshold": 0.50,
            "flash_crash_threshold": 0.10,
            "flash_crash_window_seconds": 60,
            "override_timeout_seconds": 300,
            "emergency_timeout_seconds": 3600,
        }

        controller = EmergencyController(
            event_bus=event_bus,
            circuit_manager=circuit_manager,
            config_path="test_config.yaml",
        )

        return controller


class TestEmergencyController:
    """Test emergency controller functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, emergency_controller):
        """Test controller initializes correctly."""
        assert emergency_controller.state == EmergencyState.NORMAL
        assert len(emergency_controller.active_emergencies) == 0
        assert emergency_controller.override_active is False
        assert emergency_controller.emergencies_triggered == 0

    @pytest.mark.asyncio
    async def test_daily_loss_calculation(self, emergency_controller):
        """Test daily loss percentage calculation."""
        # Set balances
        emergency_controller.daily_start_balance = Decimal("1000")
        emergency_controller.current_balance = Decimal("850")

        # Calculate loss (should be 15%)
        loss = emergency_controller.calculate_daily_loss()
        assert loss == Decimal("0.15")
        assert emergency_controller.daily_loss_percent == Decimal("0.15")

        # Test with no loss
        emergency_controller.current_balance = Decimal("1000")
        loss = emergency_controller.calculate_daily_loss()
        assert loss == Decimal("0")

        # Test with gain
        emergency_controller.current_balance = Decimal("1100")
        loss = emergency_controller.calculate_daily_loss()
        assert loss == Decimal("-0.1")  # Negative means gain

    @pytest.mark.asyncio
    async def test_daily_loss_circuit_breaker(self, emergency_controller):
        """Test daily loss circuit breaker triggers correctly."""
        # Mock event bus publish
        emergency_controller.event_bus.publish = AsyncMock()

        # Set balances at exactly 15% loss
        emergency_controller.daily_start_balance = Decimal("1000")
        emergency_controller.current_balance = Decimal("850")

        # Check daily loss
        await emergency_controller._check_daily_loss()

        # Verify emergency triggered
        assert (
            EmergencyType.DAILY_LOSS_HALT.value
            in emergency_controller.active_emergencies
        )
        assert emergency_controller.state == EmergencyState.EMERGENCY
        assert emergency_controller.emergencies_triggered == 1

        # Verify event published
        emergency_controller.event_bus.publish.assert_called()
        call_args = emergency_controller.event_bus.publish.call_args
        assert call_args[0][0].event_type == EventType.CIRCUIT_BREAKER_OPEN
        assert call_args[1]["priority"] == EventPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_daily_loss_below_threshold(self, emergency_controller):
        """Test daily loss below threshold doesn't trigger."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Set balances at 14.9% loss (just below threshold)
        emergency_controller.daily_start_balance = Decimal("1000")
        emergency_controller.current_balance = Decimal("851")

        # Check daily loss
        await emergency_controller._check_daily_loss()

        # Verify no emergency triggered
        assert len(emergency_controller.active_emergencies) == 0
        assert emergency_controller.state == EmergencyState.NORMAL
        assert emergency_controller.emergencies_triggered == 0

    @pytest.mark.asyncio
    async def test_correlation_spike_detection(self, emergency_controller):
        """Test correlation spike detection."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Set high correlation
        emergency_controller.update_correlation_matrix(
            "BTC-USDT", "ETH-USDT", Decimal("0.85")
        )

        # Check correlation spike
        await emergency_controller._check_correlation_spike()

        # Verify emergency triggered
        assert (
            EmergencyType.CORRELATION_SPIKE.value
            in emergency_controller.active_emergencies
        )
        assert emergency_controller.state == EmergencyState.EMERGENCY

    @pytest.mark.asyncio
    async def test_liquidity_crisis_detection(self, emergency_controller):
        """Test liquidity crisis detection."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Set low liquidity score
        emergency_controller.liquidity_scores["BTC-USDT"] = Decimal("0.15")

        # Check liquidity crisis
        await emergency_controller._check_liquidity_crisis()

        # Verify emergency triggered
        assert (
            EmergencyType.LIQUIDITY_CRISIS.value
            in emergency_controller.active_emergencies
        )
        assert emergency_controller.state == EmergencyState.EMERGENCY

    @pytest.mark.asyncio
    async def test_flash_crash_detection(self, emergency_controller):
        """Test flash crash detection."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Add price history simulating 15% drop
        now = datetime.now(UTC)
        emergency_controller.price_history["BTC-USDT"] = [
            (now - timedelta(seconds=30), Decimal("50000")),
            (now, Decimal("42500")),  # 15% drop
        ]

        # Check flash crash
        await emergency_controller._check_flash_crash()

        # Verify emergency triggered
        assert (
            EmergencyType.FLASH_CRASH.value in emergency_controller.active_emergencies
        )
        assert emergency_controller.state == EmergencyState.EMERGENCY

    @pytest.mark.asyncio
    async def test_manual_override_correct_phrase(self, emergency_controller):
        """Test manual override with correct confirmation."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Request override with correct phrase
        result = await emergency_controller.request_manual_override(
            "OVERRIDE EMERGENCY HALT"
        )

        assert result is True
        assert emergency_controller.override_active is True
        assert emergency_controller.state == EmergencyState.OVERRIDE
        assert emergency_controller.override_expiry is not None

    @pytest.mark.asyncio
    async def test_manual_override_wrong_phrase(self, emergency_controller):
        """Test manual override with wrong confirmation."""
        # Request override with wrong phrase
        result = await emergency_controller.request_manual_override("WRONG PHRASE")

        assert result is False
        assert emergency_controller.override_active is False
        assert emergency_controller.state == EmergencyState.NORMAL

    @pytest.mark.asyncio
    async def test_override_expiry(self, emergency_controller):
        """Test override expires correctly."""
        emergency_controller.override_active = True
        emergency_controller.override_expiry = datetime.now(UTC) - timedelta(seconds=1)

        # Check if override is active (should be expired)
        is_active = emergency_controller._is_override_active()

        assert is_active is False
        assert emergency_controller.override_active is False
        assert emergency_controller.override_expiry is None

    @pytest.mark.asyncio
    async def test_emergency_clearance(self, emergency_controller):
        """Test emergency auto-clearance after timeout."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Create an expired emergency
        from genesis.engine.emergency_controller import EmergencyEvent

        emergency = EmergencyEvent(
            event_id="test_001",
            emergency_type=EmergencyType.DAILY_LOSS_HALT,
            severity="CRITICAL",
            triggered_at=datetime.now(UTC) - timedelta(hours=2),  # 2 hours ago
            trigger_values={},
            affected_symbols=[],
            actions_taken=[],
        )

        emergency_controller.active_emergencies[EmergencyType.DAILY_LOSS_HALT.value] = (
            emergency
        )
        emergency_controller.state = EmergencyState.EMERGENCY

        # Check for clearance
        await emergency_controller._check_emergency_clearance()

        # Verify emergency cleared
        assert (
            EmergencyType.DAILY_LOSS_HALT.value
            not in emergency_controller.active_emergencies
        )
        assert emergency_controller.state == EmergencyState.RECOVERY

    @pytest.mark.asyncio
    async def test_emergency_report_generation(self, emergency_controller):
        """Test emergency report generation."""
        # Set some test data
        emergency_controller.daily_loss_percent = Decimal("0.12")
        emergency_controller.correlation_matrix[("BTC-USDT", "ETH-USDT")] = Decimal(
            "0.75"
        )
        emergency_controller.liquidity_scores["BTC-USDT"] = Decimal("0.45")
        emergency_controller.emergencies_triggered = 3

        # Generate report
        report = await emergency_controller.generate_emergency_report()

        assert report["current_state"] == EmergencyState.NORMAL.value
        assert report["current_metrics"]["daily_loss_percent"] == 0.12
        assert report["current_metrics"]["max_correlation"] == 0.75
        assert report["current_metrics"]["min_liquidity_score"] == 0.45
        assert len(report["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_position_update_handling(self, emergency_controller):
        """Test handling of position update events."""
        # Create position update event
        event = Event(
            event_type=EventType.POSITION_UPDATED, event_data={"balance": "950.50"}
        )

        # Handle event
        await emergency_controller._handle_position_update(event)

        assert emergency_controller.current_balance == Decimal("950.50")
        # First update should also set daily start balance
        assert emergency_controller.daily_start_balance == Decimal("950.50")

    @pytest.mark.asyncio
    async def test_market_update_handling(self, emergency_controller):
        """Test handling of market update events."""
        # Create market update event
        event = Event(
            event_type=EventType.MARKET_DATA_UPDATED,
            event_data={
                "symbol": "BTC-USDT",
                "price": "45000",
                "liquidity_score": "0.85",
            },
        )

        # Handle event
        await emergency_controller._handle_market_update(event)

        assert emergency_controller.liquidity_scores["BTC-USDT"] == Decimal("0.85")
        assert len(emergency_controller.price_history["BTC-USDT"]) == 1
        assert emergency_controller.price_history["BTC-USDT"][0][1] == Decimal("45000")

    @pytest.mark.asyncio
    async def test_reset_daily_tracking(self, emergency_controller):
        """Test daily tracking reset."""
        # Set current state
        emergency_controller.current_balance = Decimal("1200")
        emergency_controller.daily_loss_percent = Decimal("0.08")

        # Reset daily tracking
        emergency_controller.reset_daily_tracking()

        assert emergency_controller.daily_start_balance == Decimal("1200")
        assert emergency_controller.daily_loss_percent == Decimal("0")

    @pytest.mark.asyncio
    async def test_status_retrieval(self, emergency_controller):
        """Test getting controller status."""
        # Set some state
        emergency_controller.state = EmergencyState.WARNING
        emergency_controller.daily_loss_percent = Decimal("0.10")
        emergency_controller.emergencies_triggered = 5

        status = emergency_controller.get_status()

        assert status["state"] == EmergencyState.WARNING.value
        assert status["monitoring"] is False
        assert status["daily_loss_percent"] == 0.10
        assert status["emergencies_triggered"] == 5
        assert "config" in status

    @pytest.mark.asyncio
    async def test_multiple_emergencies_handling(self, emergency_controller):
        """Test handling multiple simultaneous emergencies."""
        emergency_controller.event_bus.publish = AsyncMock()

        # Trigger multiple emergencies
        emergency_controller.daily_start_balance = Decimal("1000")
        emergency_controller.current_balance = Decimal("850")  # 15% loss
        emergency_controller.correlation_matrix[("BTC-USDT", "ETH-USDT")] = Decimal(
            "0.90"
        )

        # Check both conditions
        await emergency_controller._check_daily_loss()
        await emergency_controller._check_correlation_spike()

        # Verify both emergencies active
        assert len(emergency_controller.active_emergencies) == 2
        assert (
            EmergencyType.DAILY_LOSS_HALT.value
            in emergency_controller.active_emergencies
        )
        assert (
            EmergencyType.CORRELATION_SPIKE.value
            in emergency_controller.active_emergencies
        )
        assert emergency_controller.state == EmergencyState.EMERGENCY


@pytest.mark.asyncio
async def test_monitoring_lifecycle(emergency_controller):
    """Test starting and stopping monitoring."""
    # Start monitoring
    await emergency_controller.start_monitoring()
    assert emergency_controller.monitoring is True
    assert emergency_controller.monitor_task is not None

    # Stop monitoring
    await emergency_controller.stop_monitoring()
    assert emergency_controller.monitoring is False
    assert emergency_controller.monitor_task is None


@pytest.mark.asyncio
async def test_edge_case_zero_balance(emergency_controller):
    """Test edge case with zero balance."""
    emergency_controller.daily_start_balance = Decimal("0")
    emergency_controller.current_balance = Decimal("0")

    loss = emergency_controller.calculate_daily_loss()
    assert loss == Decimal("0")

    # Test with zero start but positive current (shouldn't happen but test anyway)
    emergency_controller.current_balance = Decimal("100")
    loss = emergency_controller.calculate_daily_loss()
    assert loss == Decimal("0")  # Can't calculate loss from zero
