"""
Time synchronization module for Binance API.

Ensures accurate timestamps for request signatures by synchronizing
with the exchange server time.
"""

import asyncio
import time

import structlog

from config.settings import get_settings
from typing import Optional

logger = structlog.get_logger(__name__)


class TimeSync:
    """
    Manages time synchronization with Binance servers.
    
    Critical for API authentication which requires timestamps
    to be within a certain window of server time.
    """

    def __init__(self):
        """Initialize the time synchronization manager."""
        self.settings = get_settings()
        self.time_offset: int = 0  # Milliseconds difference between local and server
        self.last_sync_time: Optional[float] = None
        self.sync_interval: int = 300  # 5 minutes default
        self.recv_window: int = 5000  # 5 second tolerance window
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.sync_count = 0
        self.max_offset_seen = 0
        self.min_offset_seen = 0

        logger.info(
            "TimeSync initialized",
            sync_interval=self.sync_interval,
            recv_window=self.recv_window
        )

    async def start(self, gateway=None) -> None:
        """
        Start the time synchronization service.
        
        Args:
            gateway: BinanceGateway instance for fetching server time
        """
        if self._running:
            return

        self._running = True
        self.gateway = gateway

        # Initial sync
        await self.sync_time()

        # Start periodic sync task
        self._sync_task = asyncio.create_task(self._periodic_sync())

        logger.info("TimeSync service started")

    async def stop(self) -> None:
        """Stop the time synchronization service."""
        self._running = False

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("TimeSync service stopped")

    async def _periodic_sync(self) -> None:
        """Periodically synchronize with server time."""
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval)
                await self.sync_time()
            except Exception as e:
                logger.error("Periodic time sync failed", error=str(e))
                await asyncio.sleep(30)  # Retry after 30 seconds on error

    async def sync_time(self) -> None:
        """Synchronize with Binance server time."""
        try:
            if not self.gateway:
                logger.warning("No gateway available for time sync")
                return

            # Measure round-trip time
            local_time_before = self._get_timestamp()

            # Fetch server time
            server_time = await self.gateway.get_server_time()

            local_time_after = self._get_timestamp()

            # Calculate round-trip time
            round_trip_time = local_time_after - local_time_before

            # Estimate server time at the moment of our local_time_after
            # Account for half the round-trip time
            estimated_server_time = server_time + (round_trip_time // 2)

            # Calculate offset
            self.time_offset = estimated_server_time - local_time_after

            # Update statistics
            self.sync_count += 1
            self.max_offset_seen = max(self.max_offset_seen, abs(self.time_offset))
            self.min_offset_seen = min(self.min_offset_seen, abs(self.time_offset))
            self.last_sync_time = time.time()

            logger.info(
                "Time synchronized with server",
                offset_ms=self.time_offset,
                round_trip_ms=round_trip_time,
                sync_count=self.sync_count
            )

            # Warn if offset is large
            if abs(self.time_offset) > 1000:  # More than 1 second
                logger.warning(
                    "Large time offset detected",
                    offset_ms=self.time_offset,
                    recommendation="Consider syncing system time with NTP"
                )

        except Exception as e:
            logger.error("Failed to sync time with server", error=str(e))
            # Don't update offset on failure, keep using the last known good value

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)

    def get_synchronized_timestamp(self) -> int:
        """
        Get timestamp synchronized with server time.
        
        Returns:
            Timestamp in milliseconds adjusted for server time offset
        """
        return self._get_timestamp() + self.time_offset

    def get_recv_window(self) -> int:
        """
        Get the receive window for API requests.
        
        Returns:
            Receive window in milliseconds
        """
        return self.recv_window

    def is_synchronized(self) -> bool:
        """
        Check if time is synchronized.
        
        Returns:
            True if synchronized within the last sync interval
        """
        if self.last_sync_time is None:
            return False

        time_since_sync = time.time() - self.last_sync_time
        return time_since_sync < (self.sync_interval * 2)  # Allow 2x interval as buffer

    def get_offset(self) -> int:
        """
        Get current time offset.
        
        Returns:
            Time offset in milliseconds
        """
        return self.time_offset

    def get_statistics(self) -> dict:
        """
        Get time synchronization statistics.
        
        Returns:
            Dictionary with sync statistics
        """
        return {
            "is_synchronized": self.is_synchronized(),
            "current_offset_ms": self.time_offset,
            "last_sync_time": self.last_sync_time,
            "sync_count": self.sync_count,
            "max_offset_seen_ms": self.max_offset_seen,
            "min_offset_seen_ms": self.min_offset_seen,
            "recv_window_ms": self.recv_window,
            "time_since_last_sync": (
                time.time() - self.last_sync_time if self.last_sync_time else None
            )
        }

    async def ensure_synchronized(self) -> None:
        """
        Ensure time is synchronized, sync if necessary.
        
        Raises:
            RuntimeError: If synchronization fails
        """
        if not self.is_synchronized():
            logger.info("Time not synchronized, syncing now")
            await self.sync_time()

            if not self.is_synchronized():
                raise RuntimeError("Failed to synchronize time with server")
