"""
Unit tests for WebSocket load testing framework components.

Tests individual components without requiring actual WebSocket connections.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.load.websocket_load_test import (
    ConnectionMetrics,
    ConnectionState,
    WebSocketConnectionManager,
    WebSocketMetricsCollector,
)


class TestConnectionMetrics:
    """Test ConnectionMetrics data class."""
    
    def test_metrics_initialization(self):
        """Test ConnectionMetrics initialization."""
        metrics = ConnectionMetrics(connection_id="test_001")
        
        assert metrics.connection_id == "test_001"
        assert metrics.state == ConnectionState.DISCONNECTED
        assert metrics.messages_sent == 0
        assert metrics.messages_received == 0
        assert metrics.bytes_sent == 0
        assert metrics.bytes_received == 0
        assert metrics.reconnect_count == 0
        assert len(metrics.latency_samples) == 0
        assert len(metrics.errors) == 0
    
    def test_metrics_update(self):
        """Test updating metrics values."""
        metrics = ConnectionMetrics(connection_id="test_002")
        
        # Update various metrics
        metrics.state = ConnectionState.CONNECTED
        metrics.messages_sent = 10
        metrics.messages_received = 8
        metrics.bytes_sent = 1024
        metrics.bytes_received = 2048
        metrics.reconnect_count = 2
        metrics.latency_samples.extend([0.01, 0.02, 0.015])
        metrics.errors.append("Test error")
        
        assert metrics.state == ConnectionState.CONNECTED
        assert metrics.messages_sent == 10
        assert metrics.messages_received == 8
        assert metrics.bytes_sent == 1024
        assert metrics.bytes_received == 2048
        assert metrics.reconnect_count == 2
        assert len(metrics.latency_samples) == 3
        assert len(metrics.errors) == 1


class TestWebSocketMetricsCollector:
    """Test Prometheus metrics collector."""
    
    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        with patch('tests.load.websocket_load_test.start_http_server'):
            collector = WebSocketMetricsCollector()
            
            # Check all metrics are initialized
            assert collector.connections_total is not None
            assert collector.connections_active is not None
            assert collector.connections_failed is not None
            assert collector.messages_sent is not None
            assert collector.messages_received is not None
            assert collector.message_latency is not None
            assert collector.bytes_sent is not None
            assert collector.bytes_received is not None
            assert collector.memory_usage is not None
            assert collector.cpu_usage is not None
    
    def test_metric_updates(self):
        """Test updating metric values."""
        # Clear Prometheus registry to avoid duplicates
        from prometheus_client import REGISTRY
        REGISTRY._collector_to_names.clear()
        REGISTRY._names_to_collectors.clear()
        
        with patch('tests.load.websocket_load_test.start_http_server'):
            collector = WebSocketMetricsCollector()
            
            # Update connection metrics
            collector.connections_total.inc()
            collector.connections_active.inc()
            collector.connections_active.inc()
            collector.connections_active.dec()
            collector.connections_failed.inc()
            
            # Update message metrics
            collector.messages_sent.inc()
            collector.messages_sent.inc()
            collector.messages_received.inc()
            
            # Update bandwidth metrics
            collector.bytes_sent.inc(100)
            collector.bytes_received.inc(200)
            
            # Verify updates (accessing internal values for testing)
            assert collector.connections_total._value.get() == 1
            assert collector.connections_active._value.get() == 1
            assert collector.connections_failed._value.get() == 1
            assert collector.messages_sent._value.get() == 2
            assert collector.messages_received._value.get() == 1
            assert collector.bytes_sent._value.get() == 100
            assert collector.bytes_received._value.get() == 200


class TestWebSocketConnectionManager:
    """Test WebSocket connection manager logic."""
    
    def test_manager_initialization(self):
        """Test connection manager initialization."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="test_manager_001"
        )
        
        assert manager.url == "ws://localhost:8000/ws"
        assert manager.connection_id == "test_manager_001"
        assert manager.metrics.connection_id == "test_manager_001"
        assert manager.metrics.state == ConnectionState.DISCONNECTED
        assert manager.websocket is None
        assert manager.running is False
        assert len(manager.reconnect_delays) == 6
    
    @pytest.mark.asyncio
    async def test_connection_with_mock_websocket(self):
        """Test connection with mocked websocket."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="test_mock_001"
        )
        
        # Mock websocket connect
        mock_ws = AsyncMock()
        mock_ws.closed = False
        
        with patch('websockets.connect', new=AsyncMock(return_value=mock_ws)) as mock_connect:
            result = await manager.connect()
            
            assert result is True
            assert manager.metrics.state == ConnectionState.CONNECTED
            assert manager.websocket == mock_ws
            assert manager.metrics.connect_time is not None
            
            # Verify connect was called
            mock_connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending messages."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="test_send_001"
        )
        
        # Setup mock websocket
        mock_ws = AsyncMock()
        mock_ws.closed = False
        manager.websocket = mock_ws
        manager.metrics.state = ConnectionState.CONNECTED
        
        # Send a test message
        test_message = {"action": "subscribe", "channel": "ticker"}
        send_time = await manager.send_message(test_message)
        
        assert send_time is not None
        assert manager.metrics.messages_sent == 1
        assert manager.metrics.bytes_sent > 0
        
        # Verify websocket send was called
        mock_ws.send.assert_called_once()
        sent_data = mock_ws.send.call_args[0][0]
        sent_json = json.loads(sent_data)
        assert sent_json["action"] == "subscribe"
        assert sent_json["channel"] == "ticker"
        assert "timestamp" in sent_json
        assert sent_json["connection_id"] == "test_send_001"
    
    @pytest.mark.asyncio
    async def test_send_message_when_disconnected(self):
        """Test sending message when disconnected returns None."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="test_disconnected_001"
        )
        
        # Manager is disconnected
        assert manager.metrics.state == ConnectionState.DISCONNECTED
        
        result = await manager.send_message({"test": "data"})
        assert result is None
        assert manager.metrics.messages_sent == 0
    
    def test_metrics_summary_generation(self):
        """Test metrics summary generation."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="test_summary_001"
        )
        
        # Set up some test data
        manager.metrics.state = ConnectionState.CONNECTED
        manager.metrics.connect_time = time.time() - 60  # Connected 60 seconds ago
        manager.metrics.messages_sent = 100
        manager.metrics.messages_received = 95
        manager.metrics.bytes_sent = 10240
        manager.metrics.bytes_received = 20480
        manager.metrics.reconnect_count = 2
        manager.metrics.latency_samples = [0.01, 0.02, 0.015, 0.025, 0.03]
        manager.metrics.errors = ["Error 1", "Error 2"]
        
        summary = manager.get_metrics_summary()
        
        assert summary["connection_id"] == "test_summary_001"
        assert summary["state"] == "connected"
        assert summary["uptime_seconds"] >= 60
        assert summary["messages_sent"] == 100
        assert summary["messages_received"] == 95
        assert summary["bytes_sent"] == 10240
        assert summary["bytes_received"] == 20480
        assert summary["reconnect_count"] == 2
        assert summary["error_count"] == 2
        assert summary["latency_p50_ms"] > 0
        assert summary["latency_p95_ms"] > 0
        assert summary["latency_p99_ms"] > 0
        assert summary["latency_max_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_reconnect_delays(self):
        """Test reconnection with exponential backoff."""
        manager = WebSocketConnectionManager(
            url="ws://invalid-host:9999",
            connection_id="test_reconnect_001"
        )
        
        # Use very short delays for testing
        manager.reconnect_delays = [0.01, 0.02, 0.04]
        manager.connection_timeout = 0.1  # Very short timeout
        
        start_time = time.time()
        result = await manager.reconnect()
        duration = time.time() - start_time
        
        # Should have tried all delays and failed
        assert result is False
        assert manager.metrics.reconnect_count == 1
        
        # Total duration should be at least sum of delays
        total_delays = sum(manager.reconnect_delays)
        assert duration >= total_delays


class TestConnectionStates:
    """Test connection state transitions."""
    
    def test_state_transitions(self):
        """Test valid state transitions."""
        metrics = ConnectionMetrics(connection_id="state_test")
        
        # Initial state
        assert metrics.state == ConnectionState.DISCONNECTED
        
        # Transition to connecting
        metrics.state = ConnectionState.CONNECTING
        assert metrics.state == ConnectionState.CONNECTING
        
        # Transition to connected
        metrics.state = ConnectionState.CONNECTED
        assert metrics.state == ConnectionState.CONNECTED
        
        # Transition to reconnecting
        metrics.state = ConnectionState.RECONNECTING
        assert metrics.state == ConnectionState.RECONNECTING
        
        # Transition to failed
        metrics.state = ConnectionState.FAILED
        assert metrics.state == ConnectionState.FAILED
        
        # Back to disconnected
        metrics.state = ConnectionState.DISCONNECTED
        assert metrics.state == ConnectionState.DISCONNECTED


class TestLatencyCalculations:
    """Test latency calculation functions."""
    
    def test_percentile_calculations(self):
        """Test percentile calculations from latency samples."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="latency_test"
        )
        
        # Add latency samples (in seconds)
        samples = [0.001, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.100]
        manager.metrics.latency_samples = samples
        
        summary = manager.get_metrics_summary()
        
        # P50 (median) should be between 0.020 and 0.025 (index 4-5)
        assert 20 <= summary["latency_p50_ms"] <= 30
        
        # P95 should be around index 9 (0.100s = 100ms)
        assert 90 <= summary["latency_p95_ms"] <= 105
        
        # P99 should be close to max (100ms)
        assert 95 <= summary["latency_p99_ms"] <= 105
        
        # Max should be exactly 100ms
        assert summary["latency_max_ms"] == 100
    
    def test_empty_latency_samples(self):
        """Test handling of empty latency samples."""
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws",
            connection_id="empty_latency_test"
        )
        
        # No latency samples
        assert len(manager.metrics.latency_samples) == 0
        
        summary = manager.get_metrics_summary()
        
        # All percentiles should be 0
        assert summary["latency_p50_ms"] == 0
        assert summary["latency_p95_ms"] == 0
        assert summary["latency_p99_ms"] == 0
        assert summary["latency_max_ms"] == 0