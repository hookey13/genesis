"""
Dead man's switch for emergency position closure on connection loss.

Monitors connectivity to the exchange and triggers emergency position closure
if connection is lost for more than the configured threshold (default 60s).
"""

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from enum import Enum

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.websocket_manager import ConnectionState, WebSocketManager

logger = structlog.get_logger(__name__)


class ConnectivityStatus(str, Enum):
    """Connectivity status states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    LOST = "lost"


class DeadMansSwitch:
    """
    Dead man's switch for emergency position closure.
    
    Monitors both REST API and WebSocket connectivity. If no successful
    communication occurs for the configured threshold, triggers emergency
    position closure to prevent unmanaged risk.
    """

    def __init__(
        self,
        gateway: BinanceGateway,
        websocket_manager: WebSocketManager | None = None,
        event_bus: EventBus | None = None,
        threshold_seconds: int = 60,
        check_interval: int = 5,
        emergency_close_script: str = "scripts/emergency_close.py",
    ):
        """
        Initialize the dead man's switch.
        
        Args:
            gateway: Binance gateway for REST API monitoring
            websocket_manager: WebSocket manager for stream monitoring
            event_bus: Event bus for notifications
            threshold_seconds: Seconds without connectivity before triggering (default 60)
            check_interval: How often to check connectivity in seconds (default 5)
            emergency_close_script: Path to emergency close script
        """
        self.gateway = gateway
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.threshold_seconds = threshold_seconds
        self.check_interval = check_interval
        self.emergency_close_script = emergency_close_script

        # Tracking
        self.last_successful_api_time: float | None = None
        self.last_successful_ws_time: float | None = None
        self.monitoring_active = False
        self.emergency_triggered = False
        self.monitor_task: asyncio.Task | None = None

        # Statistics
        self.total_checks = 0
        self.failed_checks = 0
        self.degraded_periods = 0
        self.emergency_activations = 0

        # Callbacks
        self.on_emergency_trigger: Callable | None = None

        # Configurable thresholds for different conditions
        self.thresholds = {
            "normal": threshold_seconds,
            "volatile": threshold_seconds // 2,  # Shorter threshold in volatile markets
            "maintenance": threshold_seconds * 2,  # Longer during maintenance
        }
        self.current_threshold = threshold_seconds

        logger.info(
            "DeadMansSwitch initialized",
            threshold_seconds=threshold_seconds,
            check_interval=check_interval,
        )

    async def start_monitoring(self) -> None:
        """Start monitoring connectivity."""
        if self.monitoring_active:
            logger.warning("Dead man's switch monitoring already active")
            return

        self.monitoring_active = True
        self.emergency_triggered = False
        self.last_successful_api_time = time.time()
        self.last_successful_ws_time = time.time()

        self.monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Dead man's switch monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring connectivity."""
        self.monitoring_active = False

        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Dead man's switch monitoring stopped",
            total_checks=self.total_checks,
            failed_checks=self.failed_checks,
            emergency_activations=self.emergency_activations,
        )

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_connectivity()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error("Error in dead man's switch monitor", error=str(e))
                await asyncio.sleep(self.check_interval)

    async def _check_connectivity(self) -> None:
        """Check connectivity and trigger emergency if necessary."""
        current_time = time.time()
        self.total_checks += 1

        # Check REST API connectivity
        api_connected = await self._check_api_connectivity()

        # Check WebSocket connectivity
        ws_connected = self._check_websocket_connectivity()

        # Update last successful times
        if api_connected:
            self.last_successful_api_time = current_time
        if ws_connected:
            self.last_successful_ws_time = current_time

        # Calculate time since last successful communication
        api_elapsed = current_time - (self.last_successful_api_time or 0)
        ws_elapsed = current_time - (self.last_successful_ws_time or 0)

        # Use the minimum elapsed time (best connectivity indicator)
        min_elapsed = min(api_elapsed, ws_elapsed)

        # Determine connectivity status
        status = self._determine_status(min_elapsed)

        # Log status
        if status != ConnectivityStatus.HEALTHY:
            logger.warning(
                "Connectivity degraded",
                status=status,
                api_elapsed=f"{api_elapsed:.1f}s",
                ws_elapsed=f"{ws_elapsed:.1f}s",
                threshold=self.current_threshold,
            )
            self.degraded_periods += 1

        # Publish status event
        if self.event_bus:
            await self._publish_status_event(status, api_elapsed, ws_elapsed)

        # Check if we should trigger emergency closure
        if min_elapsed > self.current_threshold and not self.emergency_triggered:
            await self._trigger_emergency_closure(min_elapsed)

    async def _check_api_connectivity(self) -> bool:
        """
        Check REST API connectivity.
        
        Returns:
            True if API is reachable
        """
        try:
            # Use server time endpoint as lightweight connectivity check
            server_time = await self.gateway.get_server_time()

            if server_time and server_time > 0:
                logger.debug("REST API connectivity confirmed", server_time=server_time)
                return True
            else:
                logger.warning("REST API returned invalid server time")
                self.failed_checks += 1
                return False

        except Exception as e:
            logger.warning("REST API connectivity check failed", error=str(e))
            self.failed_checks += 1
            return False

    def _check_websocket_connectivity(self) -> bool:
        """
        Check WebSocket connectivity.
        
        Returns:
            True if WebSocket is connected and receiving data
        """
        if not self.websocket_manager:
            return True  # Assume OK if no WebSocket manager

        try:
            # Check connection states
            states = self.websocket_manager.get_connection_states()

            # Check if any connection is healthy
            healthy_connections = sum(
                1 for state in states.values()
                if state == ConnectionState.CONNECTED
            )

            if healthy_connections == 0:
                logger.warning("No healthy WebSocket connections")
                self.failed_checks += 1
                return False

            # Check if receiving recent messages
            stats = self.websocket_manager.get_statistics()
            for conn_name, conn_stats in stats.get("connections", {}).items():
                last_msg_time = conn_stats.get("last_message_time")
                if last_msg_time:
                    msg_age = time.time() - last_msg_time
                    if msg_age < 30:  # Messages received within 30s
                        logger.debug(
                            "WebSocket connectivity confirmed",
                            connection=conn_name,
                            message_age=f"{msg_age:.1f}s",
                        )
                        return True

            logger.warning("WebSocket connected but no recent messages")
            return False

        except Exception as e:
            logger.warning("WebSocket connectivity check failed", error=str(e))
            self.failed_checks += 1
            return False

    def _determine_status(self, elapsed_seconds: float) -> ConnectivityStatus:
        """
        Determine connectivity status based on elapsed time.
        
        Args:
            elapsed_seconds: Seconds since last successful communication
            
        Returns:
            Current connectivity status
        """
        if elapsed_seconds < 10:
            return ConnectivityStatus.HEALTHY
        elif elapsed_seconds < 30:
            return ConnectivityStatus.DEGRADED
        elif elapsed_seconds < self.current_threshold:
            return ConnectivityStatus.CRITICAL
        else:
            return ConnectivityStatus.LOST

    async def _trigger_emergency_closure(self, elapsed_seconds: float) -> None:
        """
        Trigger emergency position closure.
        
        Args:
            elapsed_seconds: Seconds without connectivity
        """
        self.emergency_triggered = True
        self.emergency_activations += 1

        logger.critical(
            "EMERGENCY: Dead man's switch triggered!",
            elapsed_seconds=elapsed_seconds,
            threshold=self.current_threshold,
            emergency_activations=self.emergency_activations,
        )

        # Publish emergency event
        if self.event_bus:
            event = Event(
                event_type=EventType.EMERGENCY_SHUTDOWN,
                aggregate_id="DEAD_MANS_SWITCH",
                event_data={
                    "reason": "Connection lost",
                    "elapsed_seconds": elapsed_seconds,
                    "threshold": self.current_threshold,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            await self.event_bus.publish(event, priority=EventPriority.CRITICAL)

        # Execute callback if configured
        if self.on_emergency_trigger:
            try:
                await self.on_emergency_trigger()
            except Exception as e:
                logger.error("Error in emergency trigger callback", error=str(e))

        # Execute emergency close script
        await self._execute_emergency_close()

    async def _execute_emergency_close(self) -> None:
        """Execute the emergency position closure script."""
        try:
            import subprocess

            logger.info(
                "Executing emergency close script",
                script=self.emergency_close_script,
            )

            # Run the emergency close script
            result = subprocess.run(
                ["python", self.emergency_close_script],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for emergency close
            )

            if result.returncode == 0:
                logger.info(
                    "Emergency close script executed successfully",
                    stdout=result.stdout,
                )
            else:
                logger.error(
                    "Emergency close script failed",
                    returncode=result.returncode,
                    stderr=result.stderr,
                )

        except subprocess.TimeoutExpired:
            logger.error("Emergency close script timed out")
        except Exception as e:
            logger.error("Failed to execute emergency close script", error=str(e))

    async def _publish_status_event(
        self,
        status: ConnectivityStatus,
        api_elapsed: float,
        ws_elapsed: float,
    ) -> None:
        """
        Publish connectivity status event.
        
        Args:
            status: Current connectivity status
            api_elapsed: Seconds since last API success
            ws_elapsed: Seconds since last WebSocket success
        """
        event = Event(
            event_type=EventType.CONNECTIVITY_STATUS,
            aggregate_id="DEAD_MANS_SWITCH",
            event_data={
                "status": status,
                "api_elapsed_seconds": api_elapsed,
                "ws_elapsed_seconds": ws_elapsed,
                "threshold": self.current_threshold,
                "emergency_triggered": self.emergency_triggered,
                "timestamp": datetime.now().isoformat(),
            },
        )
        await self.event_bus.publish(event, priority=EventPriority.HIGH)

    def set_market_condition(self, condition: str) -> None:
        """
        Adjust threshold based on market conditions.
        
        Args:
            condition: One of "normal", "volatile", "maintenance"
        """
        if condition in self.thresholds:
            self.current_threshold = self.thresholds[condition]
            logger.info(
                "Dead man's switch threshold adjusted",
                condition=condition,
                threshold=self.current_threshold,
            )
        else:
            logger.warning(
                "Unknown market condition",
                condition=condition,
                available=list(self.thresholds.keys()),
            )

    def reset(self) -> None:
        """Reset the dead man's switch state."""
        self.emergency_triggered = False
        self.last_successful_api_time = time.time()
        self.last_successful_ws_time = time.time()

        logger.info("Dead man's switch reset")

    def get_status(self) -> dict:
        """
        Get current dead man's switch status.
        
        Returns:
            Dictionary with current status information
        """
        current_time = time.time()
        api_elapsed = current_time - (self.last_successful_api_time or 0)
        ws_elapsed = current_time - (self.last_successful_ws_time or 0)
        min_elapsed = min(api_elapsed, ws_elapsed)

        return {
            "monitoring_active": self.monitoring_active,
            "emergency_triggered": self.emergency_triggered,
            "current_status": self._determine_status(min_elapsed),
            "api_elapsed_seconds": api_elapsed,
            "ws_elapsed_seconds": ws_elapsed,
            "min_elapsed_seconds": min_elapsed,
            "current_threshold": self.current_threshold,
            "time_until_trigger": max(0, self.current_threshold - min_elapsed),
            "statistics": {
                "total_checks": self.total_checks,
                "failed_checks": self.failed_checks,
                "degraded_periods": self.degraded_periods,
                "emergency_activations": self.emergency_activations,
                "failure_rate": (
                    self.failed_checks / self.total_checks * 100
                    if self.total_checks > 0
                    else 0
                ),
            },
        }
