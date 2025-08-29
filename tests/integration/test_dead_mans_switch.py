"""
Integration tests for dead man's switch functionality.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

from genesis.exchange.dead_mans_switch import (
    DeadMansSwitch,
    ConnectivityStatus,
)
from genesis.exchange.websocket_manager import ConnectionState


@pytest.fixture
async def mock_gateway():
    """Create a mock gateway."""
    gateway = MagicMock()
    gateway.get_server_time = AsyncMock(return_value=int(time.time() * 1000))
    return gateway


@pytest.fixture
async def mock_websocket_manager():
    """Create a mock WebSocket manager."""
    manager = MagicMock()
    manager.get_connection_states = MagicMock(
        return_value={
            "execution": ConnectionState.CONNECTED,
            "monitoring": ConnectionState.CONNECTED,
        }
    )
    manager.get_statistics = MagicMock(
        return_value={
            "connections": {
                "execution": {
                    "last_message_time": time.time() - 5,  # 5 seconds ago
                },
                "monitoring": {
                    "last_message_time": time.time() - 10,  # 10 seconds ago
                },
            }
        }
    )
    return manager


@pytest.fixture
async def mock_event_bus():
    """Create a mock event bus."""
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    return event_bus


class TestDeadMansSwitch:
    """Test dead man's switch functionality."""
    
    async def test_initialization(self, mock_gateway, mock_websocket_manager):
        """Test dead man's switch initialization."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            websocket_manager=mock_websocket_manager,
            threshold_seconds=60,
            check_interval=5,
        )
        
        assert dms.threshold_seconds == 60
        assert dms.check_interval == 5
        assert dms.monitoring_active is False
        assert dms.emergency_triggered is False
    
    async def test_start_stop_monitoring(self, mock_gateway):
        """Test starting and stopping monitoring."""
        dms = DeadMansSwitch(gateway=mock_gateway, check_interval=1)
        
        # Start monitoring
        await dms.start_monitoring()
        assert dms.monitoring_active is True
        assert dms.monitor_task is not None
        
        # Stop monitoring
        await dms.stop_monitoring()
        assert dms.monitoring_active is False
    
    async def test_healthy_connectivity(self, mock_gateway, mock_websocket_manager):
        """Test detection of healthy connectivity."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            websocket_manager=mock_websocket_manager,
        )
        
        # Check connectivity
        api_connected = await dms._check_api_connectivity()
        assert api_connected is True
        
        ws_connected = dms._check_websocket_connectivity()
        assert ws_connected is True
        
        # Status should be healthy
        status = dms._determine_status(5)  # 5 seconds elapsed
        assert status == ConnectivityStatus.HEALTHY
    
    async def test_degraded_connectivity(self, mock_gateway):
        """Test detection of degraded connectivity."""
        dms = DeadMansSwitch(gateway=mock_gateway)
        
        # Test degraded status (10-30 seconds)
        status = dms._determine_status(15)
        assert status == ConnectivityStatus.DEGRADED
        
        # Test critical status (30-60 seconds)
        status = dms._determine_status(45)
        assert status == ConnectivityStatus.CRITICAL
        
        # Test lost status (>60 seconds)
        status = dms._determine_status(70)
        assert status == ConnectivityStatus.LOST
    
    async def test_api_connectivity_failure(self, mock_gateway):
        """Test handling of API connectivity failure."""
        mock_gateway.get_server_time = AsyncMock(side_effect=Exception("Connection failed"))
        
        dms = DeadMansSwitch(gateway=mock_gateway)
        
        api_connected = await dms._check_api_connectivity()
        assert api_connected is False
        assert dms.failed_checks == 1
    
    async def test_websocket_connectivity_failure(self):
        """Test handling of WebSocket connectivity failure."""
        mock_ws_manager = MagicMock()
        mock_ws_manager.get_connection_states = MagicMock(
            return_value={
                "execution": ConnectionState.DISCONNECTED,
                "monitoring": ConnectionState.RECONNECTING,
            }
        )
        
        dms = DeadMansSwitch(
            gateway=MagicMock(),
            websocket_manager=mock_ws_manager,
        )
        
        ws_connected = dms._check_websocket_connectivity()
        assert ws_connected is False
        assert dms.failed_checks == 1
    
    async def test_emergency_trigger(self, mock_gateway, mock_event_bus):
        """Test emergency trigger when connectivity lost."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            event_bus=mock_event_bus,
            threshold_seconds=60,
        )
        
        # Mock emergency close execution
        with patch.object(dms, "_execute_emergency_close", new_callable=AsyncMock) as mock_execute:
            await dms._trigger_emergency_closure(65)  # 65 seconds elapsed
            
            assert dms.emergency_triggered is True
            assert dms.emergency_activations == 1
            mock_execute.assert_called_once()
            
            # Verify emergency event was published
            mock_event_bus.publish.assert_called_once()
            event = mock_event_bus.publish.call_args[0][0]
            assert event.event_type.value == "EMERGENCY_SHUTDOWN"
    
    async def test_emergency_trigger_callback(self, mock_gateway):
        """Test custom callback execution on emergency."""
        dms = DeadMansSwitch(gateway=mock_gateway)
        
        # Set custom callback
        callback_executed = False
        async def custom_callback():
            nonlocal callback_executed
            callback_executed = True
        
        dms.on_emergency_trigger = custom_callback
        
        with patch.object(dms, "_execute_emergency_close", new_callable=AsyncMock):
            await dms._trigger_emergency_closure(65)
            
            assert callback_executed is True
    
    async def test_emergency_script_execution(self, mock_gateway):
        """Test emergency close script execution."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            emergency_close_script="scripts/emergency_close.py",
        )
        
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Success"
            
            await dms._execute_emergency_close()
            
            mock_run.assert_called_once_with(
                ["python", "scripts/emergency_close.py"],
                capture_output=True,
                text=True,
                timeout=30,
            )
    
    async def test_market_condition_threshold_adjustment(self, mock_gateway):
        """Test threshold adjustment based on market conditions."""
        dms = DeadMansSwitch(gateway=mock_gateway, threshold_seconds=60)
        
        # Normal condition
        assert dms.current_threshold == 60
        
        # Volatile market - shorter threshold
        dms.set_market_condition("volatile")
        assert dms.current_threshold == 30
        
        # Maintenance mode - longer threshold
        dms.set_market_condition("maintenance")
        assert dms.current_threshold == 120
        
        # Back to normal
        dms.set_market_condition("normal")
        assert dms.current_threshold == 60
    
    async def test_monitoring_loop_with_simulated_outage(
        self, mock_gateway, mock_websocket_manager, mock_event_bus
    ):
        """Test monitoring loop behavior during simulated outage."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            websocket_manager=mock_websocket_manager,
            event_bus=mock_event_bus,
            threshold_seconds=2,  # Short threshold for testing
            check_interval=0.5,  # Fast checks for testing
        )
        
        # Start monitoring
        await dms.start_monitoring()
        
        # Simulate healthy connectivity for 1 second
        await asyncio.sleep(1)
        
        # Simulate connectivity loss
        mock_gateway.get_server_time = AsyncMock(side_effect=Exception("Connection lost"))
        mock_websocket_manager.get_connection_states = MagicMock(
            return_value={"execution": ConnectionState.DISCONNECTED}
        )
        
        # Mock emergency execution to prevent actual script run
        with patch.object(dms, "_execute_emergency_close", new_callable=AsyncMock):
            # Wait for threshold to be exceeded
            await asyncio.sleep(3)
            
            # Emergency should have been triggered
            assert dms.emergency_triggered is True
            assert dms.failed_checks > 0
        
        # Stop monitoring
        await dms.stop_monitoring()
    
    async def test_reset_functionality(self, mock_gateway):
        """Test resetting the dead man's switch."""
        dms = DeadMansSwitch(gateway=mock_gateway)
        
        # Set some state
        dms.emergency_triggered = True
        dms.last_successful_api_time = time.time() - 100
        dms.last_successful_ws_time = time.time() - 100
        
        # Reset
        dms.reset()
        
        assert dms.emergency_triggered is False
        assert time.time() - dms.last_successful_api_time < 1
        assert time.time() - dms.last_successful_ws_time < 1
    
    async def test_status_reporting(self, mock_gateway, mock_websocket_manager):
        """Test status reporting functionality."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            websocket_manager=mock_websocket_manager,
            threshold_seconds=60,
        )
        
        # Set some tracking data
        dms.total_checks = 100
        dms.failed_checks = 5
        dms.degraded_periods = 10
        dms.emergency_activations = 1
        dms.last_successful_api_time = time.time() - 10
        dms.last_successful_ws_time = time.time() - 5
        
        status = dms.get_status()
        
        assert status["monitoring_active"] is False
        assert status["emergency_triggered"] is False
        assert status["current_status"] == ConnectivityStatus.DEGRADED
        assert 4 < status["min_elapsed_seconds"] < 6
        assert status["time_until_trigger"] > 50
        assert status["statistics"]["total_checks"] == 100
        assert status["statistics"]["failed_checks"] == 5
        assert status["statistics"]["failure_rate"] == 5.0
    
    async def test_no_false_triggers(self, mock_gateway, mock_websocket_manager):
        """Test that emergency is not triggered when connectivity is healthy."""
        dms = DeadMansSwitch(
            gateway=mock_gateway,
            websocket_manager=mock_websocket_manager,
            threshold_seconds=2,  # Short threshold for testing
            check_interval=0.5,
        )
        
        # Start monitoring
        await dms.start_monitoring()
        
        # Run for longer than threshold with healthy connectivity
        await asyncio.sleep(3)
        
        # Should not have triggered
        assert dms.emergency_triggered is False
        assert dms.emergency_activations == 0
        
        # Stop monitoring
        await dms.stop_monitoring()