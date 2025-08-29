"""
Time synchronization utilities for Project GENESIS.

Provides clock drift detection and validation to ensure
trading operations remain synchronized with exchange servers.
"""

import asyncio
import logging
import time

import aiohttp
import ntplib
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ClockDriftResult(BaseModel):
    """Result of clock drift check."""

    local_timestamp_ms: int
    remote_timestamp_ms: int
    drift_ms: int
    is_acceptable: bool
    source: str
    error: str | None = None


async def check_binance_time() -> int | None:
    """
    Get current server time from Binance API.

    Returns:
        Server timestamp in milliseconds or None if failed
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Use testnet URL for now
            url = "https://testnet.binance.vision/api/v3/time"
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("serverTime")
    except Exception as e:
        logger.error(f"Failed to get Binance server time: {e}")
        return None


def check_ntp_time(server: str = "pool.ntp.org") -> int | None:
    """
    Get current time from NTP server.

    Args:
        server: NTP server hostname

    Returns:
        NTP timestamp in milliseconds or None if failed
    """
    try:
        client = ntplib.NTPClient()
        response = client.request(server, version=3, timeout=5)
        # Convert NTP timestamp to milliseconds
        return int(response.tx_time * 1000)
    except Exception as e:
        logger.error(f"Failed to get NTP time from {server}: {e}")
        return None


async def check_clock_drift_ms(max_drift_ms: int = 1000) -> ClockDriftResult:
    """
    Check clock drift against exchange server.

    Args:
        max_drift_ms: Maximum acceptable drift in milliseconds

    Returns:
        ClockDriftResult with drift information
    """
    local_time_ms = int(time.time() * 1000)

    # Try Binance first (most relevant for trading)
    remote_time_ms = await check_binance_time()
    source = "binance"

    # Fallback to NTP if Binance fails
    if remote_time_ms is None:
        remote_time_ms = check_ntp_time()
        source = "ntp"

    if remote_time_ms is None:
        return ClockDriftResult(
            local_timestamp_ms=local_time_ms,
            remote_timestamp_ms=0,
            drift_ms=0,
            is_acceptable=False,
            source="none",
            error="Failed to get remote time from all sources",
        )

    drift_ms = abs(local_time_ms - remote_time_ms)

    return ClockDriftResult(
        local_timestamp_ms=local_time_ms,
        remote_timestamp_ms=remote_time_ms,
        drift_ms=drift_ms,
        is_acceptable=drift_ms <= max_drift_ms,
        source=source,
        error=(
            None
            if drift_ms <= max_drift_ms
            else f"Drift {drift_ms}ms exceeds maximum {max_drift_ms}ms"
        ),
    )


def sync_system_time() -> bool:
    """
    Attempt to sync system time (requires root/admin).

    Returns:
        True if sync successful, False otherwise
    """
    import platform
    import subprocess

    system = platform.system()

    try:
        if system == "Linux":
            # Try timedatectl first (systemd)
            result = subprocess.run(
                ["timedatectl", "set-ntp", "true"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                # Fallback to ntpdate
                result = subprocess.run(
                    ["ntpdate", "-b", "pool.ntp.org"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            return result.returncode == 0

        elif system == "Darwin":  # macOS
            # Use sntp command
            result = subprocess.run(
                ["sntp", "-sS", "time.apple.com"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0

        elif system == "Windows":
            # Use w32tm command
            result = subprocess.run(
                ["w32tm", "/resync", "/force"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True,
            )
            return result.returncode == 0

        else:
            logger.warning(f"Time sync not supported on {system}")
            return False

    except Exception as e:
        logger.error(f"Failed to sync system time: {e}")
        return False


async def continuous_drift_monitor(
    max_drift_ms: int, check_interval_seconds: int, callback=None
) -> None:
    """
    Continuously monitor clock drift.

    Args:
        max_drift_ms: Maximum acceptable drift
        check_interval_seconds: How often to check
        callback: Optional callback function(ClockDriftResult)
    """
    while True:
        try:
            result = await check_clock_drift_ms(max_drift_ms)

            if not result.is_acceptable:
                logger.error(
                    f"Clock drift exceeded: {result.drift_ms}ms "
                    f"(max: {max_drift_ms}ms) from {result.source}"
                )
            else:
                logger.debug(
                    f"Clock drift OK: {result.drift_ms}ms from {result.source}"
                )

            if callback:
                await callback(result)

        except Exception as e:
            logger.error(f"Clock drift monitor error: {e}")

        await asyncio.sleep(check_interval_seconds)
