"""Dead man's switch for automatic emergency closure."""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class DeadMansSwitch:
    """Automatic emergency closure if heartbeat stops."""
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        check_interval_seconds: int = 30,
        emergency_callback: Optional[Callable] = None
    ):
        """Initialize dead man's switch.
        
        Args:
            timeout_seconds: Seconds without heartbeat before triggering
            check_interval_seconds: How often to check heartbeat
            emergency_callback: Function to call on trigger
        """
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self.emergency_callback = emergency_callback
        
        # State tracking
        self.last_heartbeat = datetime.utcnow()
        self.armed = False
        self.triggered = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def arm(self) -> None:
        """Arm the dead man's switch."""
        if self.armed:
            logger.warning("Dead man's switch already armed")
            return
        
        self.armed = True
        self.triggered = False
        self.last_heartbeat = datetime.utcnow()
        
        # Start monitoring
        if not self.monitor_task or self.monitor_task.done():
            self.monitor_task = asyncio.create_task(self._monitor())
        
        logger.info(
            "Dead man's switch ARMED",
            timeout_seconds=self.timeout_seconds
        )
    
    def disarm(self) -> None:
        """Disarm the dead man's switch."""
        if not self.armed:
            return
        
        self.armed = False
        
        # Cancel monitoring
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
        
        logger.info("Dead man's switch disarmed")
    
    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
        
        if self.triggered:
            # Reset if triggered
            self.triggered = False
            logger.info("Dead man's switch reset after trigger")
    
    async def _monitor(self) -> None:
        """Monitor heartbeat and trigger if timeout."""
        logger.info("Dead man's switch monitoring started")
        
        while self.armed:
            try:
                await asyncio.sleep(self.check_interval_seconds)
                
                # Check timeout
                time_since_heartbeat = (
                    datetime.utcnow() - self.last_heartbeat
                ).total_seconds()
                
                if time_since_heartbeat > self.timeout_seconds:
                    await self._trigger()
                    break
                    
            except asyncio.CancelledError:
                logger.info("Dead man's switch monitoring cancelled")
                break
                
            except Exception as e:
                logger.error("Dead man's switch error", error=str(e))
                await asyncio.sleep(1)
    
    async def _trigger(self) -> None:
        """Trigger emergency action."""
        if self.triggered:
            return
        
        self.triggered = True
        
        logger.critical(
            "DEAD MAN'S SWITCH TRIGGERED",
            last_heartbeat=self.last_heartbeat.isoformat(),
            timeout_seconds=self.timeout_seconds
        )
        
        # Execute emergency callback
        if self.emergency_callback:
            try:
                if asyncio.iscoroutinefunction(self.emergency_callback):
                    await self.emergency_callback()
                else:
                    self.emergency_callback()
                    
                logger.info("Emergency callback executed")
                
            except Exception as e:
                logger.error(
                    "Emergency callback failed",
                    error=str(e)
                )
        
        # Disarm after trigger
        self.disarm()
    
    def get_status(self) -> dict:
        """Get current status.
        
        Returns:
            Status dictionary
        """
        time_since_heartbeat = (
            datetime.utcnow() - self.last_heartbeat
        ).total_seconds()
        
        return {
            "armed": self.armed,
            "triggered": self.triggered,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "time_since_heartbeat": time_since_heartbeat,
            "timeout_seconds": self.timeout_seconds,
            "will_trigger_in": max(
                0,
                self.timeout_seconds - time_since_heartbeat
            ) if self.armed else None
        }