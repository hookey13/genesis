"""
Unit tests for time synchronization utilities.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.utils.time_sync import (
    check_binance_time,
    check_clock_drift_ms,
    check_ntp_time,
)


@pytest.mark.asyncio
class TestClockDriftCheck:
    """Tests for clock drift checking."""

    async def test_check_clock_drift_acceptable(self):
        """Test clock drift within acceptable range."""
        current_time_ms = int(time.time() * 1000)

        with patch("genesis.utils.time_sync.check_binance_time") as mock_binance:
            mock_binance.return_value = current_time_ms + 100  # 100ms drift

            result = await check_clock_drift_ms(max_drift_ms=1000)

            assert result.is_acceptable is True
            assert result.drift_ms <= 200  # Allow some timing variance
            assert result.source == "binance"
            assert result.error is None

    async def test_check_clock_drift_exceeded(self):
        """Test clock drift exceeds maximum."""
        current_time_ms = int(time.time() * 1000)

        with patch("genesis.utils.time_sync.check_binance_time") as mock_binance:
            mock_binance.return_value = current_time_ms + 2000  # 2000ms drift

            result = await check_clock_drift_ms(max_drift_ms=1000)

            assert result.is_acceptable is False
            assert result.drift_ms >= 1900  # Allow some timing variance
            assert result.source == "binance"
            assert "exceeds maximum" in result.error

    async def test_check_clock_drift_fallback_to_ntp(self):
        """Test fallback to NTP when Binance fails."""
        current_time_ms = int(time.time() * 1000)

        with patch("genesis.utils.time_sync.check_binance_time") as mock_binance:
            with patch("genesis.utils.time_sync.check_ntp_time") as mock_ntp:
                mock_binance.return_value = None  # Binance fails
                mock_ntp.return_value = current_time_ms + 50  # 50ms drift

                result = await check_clock_drift_ms(max_drift_ms=1000)

                assert result.is_acceptable is True
                assert result.drift_ms <= 150  # Allow some timing variance
                assert result.source == "ntp"

    async def test_check_clock_drift_all_sources_fail(self):
        """Test when all time sources fail."""
        with patch("genesis.utils.time_sync.check_binance_time") as mock_binance:
            with patch("genesis.utils.time_sync.check_ntp_time") as mock_ntp:
                mock_binance.return_value = None
                mock_ntp.return_value = None

                result = await check_clock_drift_ms(max_drift_ms=1000)

                assert result.is_acceptable is False
                assert result.source == "none"
                assert "Failed to get remote time" in result.error


@pytest.mark.asyncio
class TestBinanceTime:
    """Tests for Binance time retrieval."""

    async def test_check_binance_time_success(self):
        """Test successful Binance time retrieval."""
        mock_time = 1234567890000

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"serverTime": mock_time})

            mock_session.get = AsyncMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock()

            result = await check_binance_time()
            assert result == mock_time

    async def test_check_binance_time_http_error(self):
        """Test Binance time retrieval with HTTP error."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 500  # Server error

            mock_session.get = AsyncMock(
                return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock()

            result = await check_binance_time()
            assert result is None

    async def test_check_binance_time_timeout(self):
        """Test Binance time retrieval with timeout."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.get = AsyncMock(side_effect=TimeoutError())
            mock_session_class.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_session_class.return_value.__aexit__ = AsyncMock()

            result = await check_binance_time()
            assert result is None


class TestNTPTime:
    """Tests for NTP time retrieval."""

    def test_check_ntp_time_success(self):
        """Test successful NTP time retrieval."""
        mock_time = 1234567890.123

        with patch("ntplib.NTPClient") as mock_ntp_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.tx_time = mock_time
            mock_client.request = MagicMock(return_value=mock_response)
            mock_ntp_client.return_value = mock_client

            result = check_ntp_time("pool.ntp.org")
            assert result == int(mock_time * 1000)

    def test_check_ntp_time_failure(self):
        """Test NTP time retrieval failure."""
        with patch("ntplib.NTPClient") as mock_ntp_client:
            mock_client = MagicMock()
            mock_client.request = MagicMock(side_effect=Exception("NTP error"))
            mock_ntp_client.return_value = mock_client

            result = check_ntp_time("pool.ntp.org")
            assert result is None
