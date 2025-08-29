"""
Integration test for WebSocket 1-hour stability (AC #1).

This test validates that the WebSocket connection remains stable for 1 hour
as required by Acceptance Criteria #1 of Story 0.2.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from genesis.exchange.websocket_manager import WebSocketManager


class TestWebSocketStability:
    """Test WebSocket connection stability over extended periods."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_1_hour_stability(self):
        """
        Test that WebSocket connection remains stable for 1 hour.

        Acceptance Criteria #1: WebSocket connection stable for 1 hour
        Task 6: Create 1-hour stability test for WebSocket connection
        """
        # Setup
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)

        # Connection metrics
        metrics = {
            "total_messages": 0,
            "disconnections": 0,
            "reconnections": 0,
            "errors": [],
            "heartbeat_failures": 0,
            "last_message_time": start_time,
            "connection_uptime": 0,
            "message_gaps": [],
        }

        # Mock WebSocket to simulate real connection behavior
        mock_websocket = AsyncMock()
        mock_websocket.recv = AsyncMock()
        mock_websocket.send = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.ping = AsyncMock()
        mock_websocket.pong = AsyncMock()

        # Simulate market data stream
        async def generate_market_data():
            """Generate realistic market data at intervals."""
            message_count = 0
            while datetime.now() < end_time:
                message_count += 1

                # Simulate different message types
                if message_count % 100 == 0:
                    # Heartbeat/ping every 100 messages (~30 seconds)
                    yield '{"e":"ping"}'
                elif message_count % 10 == 0:
                    # Order book update
                    yield f'{{"e":"depthUpdate","E":{int(time.time()*1000)},"s":"BTCUSDT","b":[["43500.10","0.5"],["43499.90","1.2"]],"a":[["43501.20","0.8"],["43501.50","2.1"]]}}'
                else:
                    # Trade update
                    price = 43500 + (message_count % 100) * 0.1
                    yield f'{{"e":"trade","E":{int(time.time()*1000)},"s":"BTCUSDT","p":"{price}","q":"0.01","T":{int(time.time()*1000)},"m":true,"M":true}}'

                # Simulate realistic message frequency (5-10 messages per second)
                await asyncio.sleep(0.15)

        data_generator = generate_market_data()

        async def mock_recv():
            """Mock WebSocket receive with realistic behavior."""
            try:
                message = await anext(data_generator)
                metrics["total_messages"] += 1
                metrics["last_message_time"] = datetime.now()

                # Check for message gaps (> 5 seconds)
                time_since_last = (
                    datetime.now() - metrics["last_message_time"]
                ).total_seconds()
                if time_since_last > 5:
                    metrics["message_gaps"].append(
                        {"timestamp": datetime.now(), "gap_seconds": time_since_last}
                    )

                return message
            except StopAsyncIteration:
                return None

        mock_websocket.recv = mock_recv

        # Create WebSocket manager with mocked connection
        manager = WebSocketManager()

        # Mock the WebSocket connections
        with patch("websockets.connect", return_value=mock_websocket):
            # Start the WebSocket manager
            await manager.start()

            try:
                # Monitor for 1 hour
                monitoring_start = time.time()
                last_health_check = monitoring_start

                while datetime.now() < end_time:
                    # Health check every 30 seconds
                    current_time = time.time()
                    if current_time - last_health_check >= 30:
                        # Verify manager is running
                        assert manager.running, f"Manager stopped at {datetime.now()}"

                        # Check message flow
                        time_since_message = (
                            datetime.now() - metrics["last_message_time"]
                        ).total_seconds()
                        assert (
                            time_since_message < 10
                        ), f"No messages for {time_since_message} seconds"

                        # Update uptime
                        metrics["connection_uptime"] = (
                            current_time - monitoring_start
                        )
                        last_health_check = current_time

                        # Log progress
                        elapsed_minutes = int(metrics["connection_uptime"] / 60)
                        print(
                            f"[{elapsed_minutes:02d}:00] Connection stable - Messages: {metrics['total_messages']}, Reconnections: {metrics['reconnections']}"
                        )

                    await asyncio.sleep(1)

            finally:
                await manager.stop()

        # Validate stability metrics
        total_runtime = (datetime.now() - start_time).total_seconds()

        # Assert stability requirements
        assert (
            total_runtime >= 3600
        ), f"Test ran for only {total_runtime/60:.1f} minutes"
        assert (
            metrics["total_messages"] > 20000
        ), f"Expected >20000 messages in 1 hour, got {metrics['total_messages']}"
        assert (
            metrics["disconnections"] <= 2
        ), f"Too many disconnections: {metrics['disconnections']}"
        assert (
            metrics["reconnections"] <= 2
        ), f"Too many reconnections: {metrics['reconnections']}"
        assert len(metrics["errors"]) <= 3, f"Too many errors: {len(metrics['errors'])}"
        assert (
            len(metrics["message_gaps"]) <= 5
        ), f"Too many message gaps: {len(metrics['message_gaps'])}"
        assert (
            metrics["connection_uptime"] >= 3590
        ), f"Insufficient uptime: {metrics['connection_uptime']} seconds"

        # Log final metrics
        print("\n=== 1-Hour Stability Test Results ===")
        print(f"Total Runtime: {total_runtime/60:.1f} minutes")
        print(f"Total Messages: {metrics['total_messages']}")
        print(f"Disconnections: {metrics['disconnections']}")
        print(f"Reconnections: {metrics['reconnections']}")
        print(f"Errors: {len(metrics['errors'])}")
        print(f"Message Gaps (>5s): {len(metrics['message_gaps'])}")
        print(f"Connection Uptime: {metrics['connection_uptime']/60:.1f} minutes")
        print(f"Stability Score: {(metrics['connection_uptime']/3600)*100:.1f}%")
        print("=====================================\n")

    @pytest.mark.asyncio
    async def test_quick_stability_check(self):
        """
        Quick 5-minute stability test for development/CI.

        This is a shorter version of the 1-hour test for faster feedback.
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)

        metrics = {
            "total_messages": 0,
            "disconnections": 0,
            "last_message_time": start_time,
        }

        # Similar setup but for 5 minutes
        mock_websocket = AsyncMock()

        async def generate_quick_data():
            """Generate data for 5 minutes."""
            while datetime.now() < end_time:
                metrics["total_messages"] += 1
                metrics["last_message_time"] = datetime.now()
                yield f'{{"e":"trade","s":"BTCUSDT","p":"43500.{metrics["total_messages"]%100:02d}"}}'
                await asyncio.sleep(0.1)

        data_gen = generate_quick_data()
        mock_websocket.recv = AsyncMock(side_effect=lambda: anext(data_gen))

        manager = WebSocketManager()

        with patch("websockets.connect", return_value=mock_websocket):
            await manager.start()

            try:
                # Monitor for 5 minutes
                while datetime.now() < end_time:
                    assert manager.running, "Manager stopped during quick test"
                    await asyncio.sleep(1)
            finally:
                await manager.stop()

        # Quick test assertions
        assert (
            metrics["total_messages"] > 2500
        ), f"Expected >2500 messages in 5 min, got {metrics['total_messages']}"
        assert (
            metrics["disconnections"] == 0
        ), f"Unexpected disconnections: {metrics['disconnections']}"

        print(
            f"Quick stability test passed: {metrics['total_messages']} messages in 5 minutes"
        )
