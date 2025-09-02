"""
Integration tests for WebSocket load testing framework.

Validates WebSocket connection handling, message delivery, and performance metrics.
"""

import asyncio
import json
import time
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import websockets

from tests.load.websocket_load_test import (
    ChaosEngineering,
    ConnectionMetrics,
    ConnectionState,
    WebSocketConnectionManager,
    WebSocketMetricsCollector,
)


class MockWebSocketServer:
    """Mock WebSocket server for testing."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.connections: List = []
        self.messages_received: List[Dict] = []
        self.running = False
    
    async def handler(self, websocket, path: str):
        """Handle WebSocket connections."""
        self.connections.append(websocket)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                self.messages_received.append(data)
                
                # Echo message back with server timestamp
                response = {
                    "type": "response",
                    "original": data,
                    "server_timestamp": time.time()
                }
                
                # If original message has timestamp, include it for latency measurement
                if "timestamp" in data:
                    response["timestamp"] = data["timestamp"]
                
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.remove(websocket)
    
    async def start(self):
        """Start the mock server."""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port
        )
        self.running = True
    
    async def stop(self):
        """Stop the mock server."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()


@pytest_asyncio.fixture
async def mock_websocket_server():
    """Fixture for mock WebSocket server."""
    server = MockWebSocketServer()
    await server.start()
    yield server
    await server.stop()


@pytest_asyncio.fixture
def metrics_collector():
    """Fixture for metrics collector."""
    with patch('tests.load.websocket_load_test.start_http_server'):
        collector = WebSocketMetricsCollector()
        yield collector


class TestWebSocketConnectionManager:
    """Test WebSocket connection manager functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_establishment(self, mock_websocket_server, metrics_collector):
        """Test successful WebSocket connection establishment."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_001",
            metrics_collector=metrics_collector
        )
        
        # Test connection
        connected = await manager.connect()
        assert connected is True
        assert manager.metrics.state == ConnectionState.CONNECTED
        assert manager.websocket is not None
        assert metrics_collector.connections_active._value.get() == 1
        
        # Disconnect
        await manager.disconnect()
        assert manager.metrics.state == ConnectionState.DISCONNECTED
        assert metrics_collector.connections_active._value.get() == 0
    
    @pytest.mark.asyncio
    async def test_connection_timeout(self, metrics_collector):
        """Test connection timeout handling."""
        manager = WebSocketConnectionManager(
            url="ws://nonexistent:9999",
            connection_id="test_timeout",
            metrics_collector=metrics_collector
        )
        manager.connection_timeout = 1  # Short timeout for testing
        
        connected = await manager.connect()
        assert connected is False
        assert manager.metrics.state == ConnectionState.FAILED
        assert len(manager.metrics.errors) > 0
        assert metrics_collector.connections_failed._value.get() == 1
    
    @pytest.mark.asyncio
    async def test_message_sending_and_receiving(self, mock_websocket_server, metrics_collector):
        """Test message sending and latency measurement."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_messages",
            metrics_collector=metrics_collector
        )
        
        await manager.connect()
        
        # Start receiving messages in background
        manager.running = True
        receive_task = asyncio.create_task(manager.receive_messages())
        
        # Send test message
        test_message = {
            "action": "subscribe",
            "channel": "ticker",
            "symbol": "BTC/USDT"
        }
        
        send_time = await manager.send_message(test_message)
        assert send_time is not None
        
        # Wait for response
        await asyncio.sleep(0.1)
        
        # Check metrics
        assert manager.metrics.messages_sent == 1
        assert manager.metrics.messages_received > 0
        assert len(manager.metrics.latency_samples) > 0
        assert metrics_collector.messages_sent._value.get() == 1
        
        # Clean up
        manager.running = False
        receive_task.cancel()
        await manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_reconnection_logic(self, mock_websocket_server, metrics_collector):
        """Test automatic reconnection with exponential backoff."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_reconnect",
            metrics_collector=metrics_collector
        )
        manager.reconnect_delays = [0.1, 0.2]  # Short delays for testing
        
        # Connect initially
        await manager.connect()
        initial_connect_time = manager.metrics.connect_time
        
        # Simulate connection loss
        await manager.websocket.close()
        manager.metrics.state = ConnectionState.DISCONNECTED
        
        # Attempt reconnection
        reconnected = await manager.reconnect()
        assert reconnected is True
        assert manager.metrics.reconnect_count == 1
        assert manager.metrics.connect_time > initial_connect_time
        
        await manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, mock_websocket_server, metrics_collector):
        """Test heartbeat sending functionality."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_heartbeat",
            metrics_collector=metrics_collector
        )
        manager.heartbeat_interval = 0.5  # Short interval for testing
        
        await manager.connect()
        manager.running = True
        
        # Run heartbeat for a short time
        heartbeat_task = asyncio.create_task(manager.send_heartbeat())
        await asyncio.sleep(1.5)
        
        # Should have sent at least 2 heartbeats
        assert manager.metrics.messages_sent >= 2
        assert manager.metrics.last_heartbeat is not None
        
        # Clean up
        manager.running = False
        heartbeat_task.cancel()
        await manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_metrics_summary(self, mock_websocket_server, metrics_collector):
        """Test metrics summary generation."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_metrics",
            metrics_collector=metrics_collector
        )
        
        await manager.connect()
        
        # Send some messages
        for i in range(5):
            await manager.send_message({"test": i})
            # Simulate some latency samples
            manager.metrics.latency_samples.append(0.01 * (i + 1))
        
        await manager.disconnect()
        
        summary = manager.get_metrics_summary()
        
        assert summary["connection_id"] == "test_metrics"
        assert summary["messages_sent"] == 5
        assert summary["latency_p50_ms"] > 0
        assert summary["latency_p99_ms"] > summary["latency_p50_ms"]
        assert summary["uptime_seconds"] > 0


class TestChaosEngineering:
    """Test chaos engineering capabilities."""
    
    @pytest.mark.asyncio
    async def test_network_interruption_simulation(self, mock_websocket_server, metrics_collector):
        """Test network interruption simulation."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_chaos",
            metrics_collector=metrics_collector
        )
        
        await manager.connect()
        assert manager.metrics.state == ConnectionState.CONNECTED
        
        # Simulate network interruption
        await ChaosEngineering.simulate_network_interruption(manager, duration=0.5)
        
        # Connection should be closed
        assert manager.websocket.closed
        
        await manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_packet_loss_simulation(self, mock_websocket_server, metrics_collector):
        """Test packet loss simulation."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_packet_loss",
            metrics_collector=metrics_collector
        )
        
        await manager.connect()
        
        # Apply packet loss
        await ChaosEngineering.simulate_packet_loss(manager, loss_rate=0.5)
        
        # Send multiple messages
        successful_sends = 0
        for i in range(10):
            result = await manager.send_message({"test": i})
            if result is not None:
                successful_sends += 1
        
        # With 50% loss rate, we expect roughly half to succeed
        assert 2 <= successful_sends <= 8  # Allow for randomness
        
        await manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_high_latency_simulation(self, mock_websocket_server, metrics_collector):
        """Test high latency simulation."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="test_latency",
            metrics_collector=metrics_collector
        )
        
        await manager.connect()
        
        # Measure baseline latency
        start = time.time()
        await manager.send_message({"test": "baseline"})
        baseline_latency = time.time() - start
        
        # Apply artificial latency
        await ChaosEngineering.simulate_high_latency(manager, added_latency=0.1)
        
        # Measure with added latency
        start = time.time()
        await manager.send_message({"test": "delayed"})
        delayed_latency = time.time() - start
        
        # Delayed should be at least 0.1s slower
        assert delayed_latency >= baseline_latency + 0.1
        
        await manager.disconnect()


class TestLoadScenarios:
    """Test various load scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self, mock_websocket_server, metrics_collector):
        """Test handling multiple concurrent connections."""
        num_connections = 10
        managers = []
        
        # Create and connect multiple managers
        for i in range(num_connections):
            manager = WebSocketConnectionManager(
                url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
                connection_id=f"concurrent_{i}",
                metrics_collector=metrics_collector
            )
            managers.append(manager)
        
        # Connect all concurrently
        connect_tasks = [manager.connect() for manager in managers]
        results = await asyncio.gather(*connect_tasks)
        
        assert all(results)  # All should connect successfully
        assert metrics_collector.connections_active._value.get() == num_connections
        assert len(mock_websocket_server.connections) == num_connections
        
        # Send messages from all connections
        send_tasks = []
        for i, manager in enumerate(managers):
            send_tasks.append(manager.send_message({"connection": i}))
        
        await asyncio.gather(*send_tasks)
        
        # Disconnect all
        disconnect_tasks = [manager.disconnect() for manager in managers]
        await asyncio.gather(*disconnect_tasks)
        
        assert metrics_collector.connections_active._value.get() == 0
    
    @pytest.mark.asyncio
    async def test_connection_ramp_up(self, mock_websocket_server, metrics_collector):
        """Test gradual connection ramp-up."""
        target_connections = 20
        ramp_up_rate = 5  # connections per second
        
        managers = []
        semaphore = asyncio.Semaphore(ramp_up_rate)
        
        async def connect_with_rate_limit(manager):
            async with semaphore:
                await manager.connect()
                await asyncio.sleep(1.0 / ramp_up_rate)
        
        # Create managers
        for i in range(target_connections):
            manager = WebSocketConnectionManager(
                url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
                connection_id=f"rampup_{i}",
                metrics_collector=metrics_collector
            )
            managers.append(manager)
        
        # Connect with rate limiting
        start_time = time.time()
        connect_tasks = [connect_with_rate_limit(manager) for manager in managers]
        await asyncio.gather(*connect_tasks)
        ramp_up_duration = time.time() - start_time
        
        # Should take approximately (target_connections / ramp_up_rate) seconds
        expected_duration = target_connections / ramp_up_rate
        assert abs(ramp_up_duration - expected_duration) < 1.0
        
        # All should be connected
        assert metrics_collector.connections_active._value.get() == target_connections
        
        # Cleanup
        disconnect_tasks = [manager.disconnect() for manager in managers]
        await asyncio.gather(*disconnect_tasks)
    
    @pytest.mark.asyncio
    async def test_sustained_message_throughput(self, mock_websocket_server, metrics_collector):
        """Test sustained message throughput over time."""
        manager = WebSocketConnectionManager(
            url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
            connection_id="throughput_test",
            metrics_collector=metrics_collector
        )
        
        await manager.connect()
        manager.running = True
        
        # Start receiving in background
        receive_task = asyncio.create_task(manager.receive_messages())
        
        # Send messages at high rate for 5 seconds
        target_rate = 100  # messages per second
        duration = 5
        total_messages = target_rate * duration
        
        start_time = time.time()
        for i in range(total_messages):
            await manager.send_message({"sequence": i, "timestamp": time.time()})
            
            # Rate limiting
            elapsed = time.time() - start_time
            expected_messages = int(elapsed * target_rate)
            if i + 1 > expected_messages:
                await asyncio.sleep((i + 1 - expected_messages) / target_rate)
        
        # Wait for all responses
        await asyncio.sleep(1)
        
        # Verify throughput
        actual_duration = time.time() - start_time
        actual_rate = manager.metrics.messages_sent / actual_duration
        
        assert actual_rate >= target_rate * 0.9  # Allow 10% tolerance
        assert manager.metrics.messages_received >= total_messages * 0.95  # Allow 5% loss
        
        # Check latency remains acceptable
        if manager.metrics.latency_samples:
            latencies = sorted(manager.metrics.latency_samples)
            p99_latency = latencies[int(len(latencies) * 0.99)]
            assert p99_latency < 0.1  # P99 should be under 100ms
        
        # Cleanup
        manager.running = False
        receive_task.cancel()
        await manager.disconnect()


class TestMemoryStability:
    """Test memory usage and leak detection."""
    
    @pytest.mark.asyncio
    async def test_memory_growth_under_load(self, mock_websocket_server):
        """Test that memory doesn't grow excessively under load."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create metrics collector without Prometheus (to reduce memory overhead)
        with patch('tests.load.websocket_load_test.start_http_server'):
            metrics_collector = WebSocketMetricsCollector()
        
        # Create and destroy many connections
        for iteration in range(5):
            managers = []
            
            # Create 20 connections
            for i in range(20):
                manager = WebSocketConnectionManager(
                    url=f"ws://{mock_websocket_server.host}:{mock_websocket_server.port}",
                    connection_id=f"memory_test_{iteration}_{i}",
                    metrics_collector=metrics_collector
                )
                managers.append(manager)
            
            # Connect all
            connect_tasks = [m.connect() for m in managers]
            await asyncio.gather(*connect_tasks)
            
            # Send some messages
            for manager in managers:
                for j in range(10):
                    await manager.send_message({"data": "x" * 1000})  # 1KB message
            
            # Disconnect all
            disconnect_tasks = [m.disconnect() for m in managers]
            await asyncio.gather(*disconnect_tasks)
            
            # Force garbage collection
            import gc
            gc.collect()
            await asyncio.sleep(0.5)
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (less than 50MB for this test)
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.2f}MB"