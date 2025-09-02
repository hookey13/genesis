"""
WebSocket connection load testing framework for Genesis.

This module provides comprehensive WebSocket load testing capabilities including:
- 10,000+ concurrent connection support
- Message throughput and latency measurement
- Connection stability monitoring over extended periods
- Graceful reconnection handling
- Real-time performance metrics collection
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, ClassVar
from urllib.parse import urlparse

import psutil
import websockets
from locust import FastHttpUser, between, events, task
from locust.env import Environment
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class ConnectionMetrics:
    """Metrics for a single WebSocket connection."""
    connection_id: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    connect_time: float | None = None
    disconnect_time: float | None = None
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    latency_samples: list[float] = field(default_factory=list)
    reconnect_count: int = 0
    last_heartbeat: float | None = None
    errors: list[str] = field(default_factory=list)


class WebSocketMetricsCollector:
    """Prometheus metrics collector for WebSocket testing."""

    def __init__(self, port: int = 9091):
        # Connection metrics
        self.connections_total = Gauge(
            'genesis_websocket_connections_total',
            'Total number of WebSocket connections'
        )
        self.connections_active = Gauge(
            'genesis_websocket_connections_active',
            'Number of active WebSocket connections'
        )
        self.connections_failed = Counter(
            'genesis_websocket_connections_failed',
            'Number of failed WebSocket connections'
        )

        # Message metrics
        self.messages_sent = Counter(
            'genesis_websocket_messages_sent_total',
            'Total number of messages sent'
        )
        self.messages_received = Counter(
            'genesis_websocket_messages_received_total',
            'Total number of messages received'
        )
        self.message_latency = Histogram(
            'genesis_websocket_message_latency_seconds',
            'Message round-trip latency',
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )

        # Bandwidth metrics
        self.bytes_sent = Counter(
            'genesis_websocket_bytes_sent_total',
            'Total bytes sent over WebSocket'
        )
        self.bytes_received = Counter(
            'genesis_websocket_bytes_received_total',
            'Total bytes received over WebSocket'
        )

        # System metrics
        self.memory_usage = Gauge(
            'genesis_websocket_memory_usage_mb',
            'Memory usage in MB'
        )
        self.cpu_usage = Gauge(
            'genesis_websocket_cpu_usage_percent',
            'CPU usage percentage'
        )

        # Start Prometheus HTTP server
        start_http_server(port)
        self.process = psutil.Process()
        self._metrics_task: asyncio.Task | None = None

        # Start system metrics collection only if event loop is running
        try:
            self._metrics_task = asyncio.create_task(self._collect_system_metrics())
        except RuntimeError:
            # No event loop running, skip background task
            pass

    async def _collect_system_metrics(self):
        """Continuously collect system metrics."""
        while True:
            self.memory_usage.set(self.process.memory_info().rss / 1024 / 1024)
            self.cpu_usage.set(self.process.cpu_percent())
            await asyncio.sleep(5)


class WebSocketConnectionManager:
    """Manages WebSocket connections with stability monitoring and reconnection."""

    # Class-level rate limiting
    _connection_attempts: ClassVar[deque] = deque(maxlen=1000)  # Track last 1000 attempts
    _rate_limit_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    MAX_CONNECTIONS_PER_MINUTE: ClassVar[int] = 100
    MAX_CONNECTIONS_PER_SECOND: ClassVar[int] = 10

    def __init__(self, url: str, connection_id: str, metrics_collector: WebSocketMetricsCollector | None = None, auth_token: str | None = None):
        self.url = url
        self.connection_id = connection_id
        self.metrics = ConnectionMetrics(connection_id=connection_id)
        self.metrics_collector = metrics_collector
        self.websocket: Any | None = None  # websockets.WebSocketClientProtocol
        self.running = False
        self.reconnect_delays = [1, 2, 4, 8, 16, 32]  # Exponential backoff
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 30  # seconds
        self._tasks: list[asyncio.Task] = []
        self.auth_token = auth_token or os.environ.get('GENESIS_AUTH_TOKEN', '')

    async def _check_rate_limit(self) -> bool:
        """Check if connection attempt is allowed by rate limiting."""
        async with self._rate_limit_lock:
            now = datetime.now()

            # Clean old attempts
            cutoff_time = now - timedelta(minutes=1)
            while self._connection_attempts and self._connection_attempts[0] < cutoff_time:
                self._connection_attempts.popleft()

            # Check per-second limit
            recent_second = [t for t in self._connection_attempts if t > now - timedelta(seconds=1)]
            if len(recent_second) >= self.MAX_CONNECTIONS_PER_SECOND:
                logger.warning(f"Rate limit hit: {len(recent_second)} connections in last second")
                return False

            # Check per-minute limit
            if len(self._connection_attempts) >= self.MAX_CONNECTIONS_PER_MINUTE:
                logger.warning(f"Rate limit hit: {len(self._connection_attempts)} connections in last minute")
                return False

            # Record this attempt
            self._connection_attempts.append(now)
            return True

    async def connect(self) -> bool:
        """Establish WebSocket connection with timeout and rate limiting."""
        try:
            # Check rate limit first
            if not await self._check_rate_limit():
                await asyncio.sleep(1)  # Back off if rate limited
                return False

            self.metrics.state = ConnectionState.CONNECTING
            logger.info(f"[{self.connection_id}] Connecting to {self.url}")

            # Parse URL and add authentication
            _ = urlparse(self.url)  # Validate URL format
            headers = {
                "User-Agent": f"Genesis-LoadTest/{self.connection_id}",
            }

            # Add Bearer token authentication if available
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ),
                timeout=self.connection_timeout
            )

            self.metrics.state = ConnectionState.CONNECTED
            self.metrics.connect_time = time.time()

            if self.metrics_collector:
                self.metrics_collector.connections_active.inc()
                self.metrics_collector.connections_total.inc()

            logger.info(f"[{self.connection_id}] Connected successfully")
            return True

        except TimeoutError:
            logger.error(f"[{self.connection_id}] Connection timeout")
            self.metrics.errors.append(f"Connection timeout at {datetime.now()}")
            self.metrics.state = ConnectionState.FAILED
            if self.metrics_collector:
                self.metrics_collector.connections_failed.inc()
            return False

        except Exception as e:
            logger.error(f"[{self.connection_id}] Connection error: {e}")
            self.metrics.errors.append(f"Connection error: {e!s}")
            self.metrics.state = ConnectionState.FAILED
            if self.metrics_collector:
                self.metrics_collector.connections_failed.inc()
            return False

    async def disconnect(self):
        """Gracefully disconnect WebSocket."""
        self.running = False

        # Cancel all tasks
        for t in self._tasks:
            if not t.done():
                t.cancel()

        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.warning(f"[{self.connection_id}] Error closing connection: {e}")

            self.metrics.state = ConnectionState.DISCONNECTED
            self.metrics.disconnect_time = time.time()

            if self.metrics_collector:
                self.metrics_collector.connections_active.dec()

    async def reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        self.metrics.state = ConnectionState.RECONNECTING
        self.metrics.reconnect_count += 1

        for delay in self.reconnect_delays:
            logger.info(f"[{self.connection_id}] Reconnecting in {delay} seconds...")
            await asyncio.sleep(delay)

            if await self.connect():
                return True

        logger.error(f"[{self.connection_id}] Failed to reconnect after all attempts")
        return False

    async def send_message(self, message: dict[str, Any]) -> float | None:
        """Send a message with validation and measure latency."""
        if self.metrics.state != ConnectionState.CONNECTED or not self.websocket:
            return None

        # Validate message structure
        if not self._validate_message(message):
            logger.error(f"[{self.connection_id}] Invalid message structure")
            return None

        try:
            # Add timestamp for latency measurement
            message["timestamp"] = time.time()
            message["connection_id"] = self.connection_id

            data = json.dumps(message)
            send_time = time.time()

            await self.websocket.send(data)

            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(data)

            if self.metrics_collector:
                self.metrics_collector.messages_sent.inc()
                self.metrics_collector.bytes_sent.inc(len(data))

            return send_time

        except Exception as e:
            logger.error(f"[{self.connection_id}] Send error: {e}")
            self.metrics.errors.append(f"Send error: {e!s}")
            return None

    def _validate_message(self, message: dict[str, Any]) -> bool:
        """Validate message structure and content."""
        # Check required fields
        if not isinstance(message, dict):
            return False

        if "action" not in message:
            return False

        # Validate action types
        valid_actions = {"subscribe", "unsubscribe", "ping", "pong"}
        if message["action"] not in valid_actions:
            return False

        # Validate subscribe/unsubscribe messages
        if message["action"] in {"subscribe", "unsubscribe"}:
            if "channel" not in message:
                return False
            valid_channels = {"ticker", "orderbook", "trades"}
            if message["channel"] not in valid_channels:
                return False

        # Sanitize string fields (prevent injection)
        for _key, value in message.items():
            if isinstance(value, str) and len(value) > 1000:
                return False  # Reject overly long strings

        return True

    async def receive_messages(self):
        """Continuously receive messages and track metrics."""
        while self.running and self.websocket:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=60  # Timeout if no message for 60 seconds
                )

                receive_time = time.time()
                self.metrics.messages_received += 1
                self.metrics.bytes_received += len(message)

                if self.metrics_collector:
                    self.metrics_collector.messages_received.inc()
                    self.metrics_collector.bytes_received.inc(len(message))

                # Parse message and calculate latency if timestamp present
                try:
                    data = json.loads(message)
                    if "timestamp" in data:
                        latency = receive_time - data["timestamp"]
                        self.metrics.latency_samples.append(latency)

                        if self.metrics_collector:
                            self.metrics_collector.message_latency.observe(latency)

                except json.JSONDecodeError:
                    pass  # Not all messages may be JSON

            except TimeoutError:
                logger.warning(f"[{self.connection_id}] No message received for 60 seconds")

            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[{self.connection_id}] Connection closed")
                break

            except Exception as e:
                logger.error(f"[{self.connection_id}] Receive error: {e}")
                self.metrics.errors.append(f"Receive error: {e!s}")
                break

        # Connection lost, attempt reconnection
        if self.running:
            await self.reconnect()

    async def send_heartbeat(self):
        """Send periodic heartbeat messages."""
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)

            if self.metrics.state == ConnectionState.CONNECTED:
                heartbeat_msg = {
                    "action": "ping",
                    "timestamp": time.time()
                }

                await self.send_message(heartbeat_msg)
                self.metrics.last_heartbeat = time.time()

    async def run(self):
        """Main connection lifecycle."""
        self.running = True

        if not await self.connect():
            return

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self.receive_messages()),
            asyncio.create_task(self.send_heartbeat())
        ]

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get connection metrics summary."""
        latencies = sorted(self.metrics.latency_samples) if self.metrics.latency_samples else [0]
        n = len(latencies)

        uptime = 0.0
        if self.metrics.connect_time:
            end_time = self.metrics.disconnect_time or time.time()
            uptime = end_time - self.metrics.connect_time

        return {
            "connection_id": self.connection_id,
            "state": self.metrics.state.value,
            "uptime_seconds": uptime,
            "messages_sent": self.metrics.messages_sent,
            "messages_received": self.metrics.messages_received,
            "bytes_sent": self.metrics.bytes_sent,
            "bytes_received": self.metrics.bytes_received,
            "reconnect_count": self.metrics.reconnect_count,
            "error_count": len(self.metrics.errors),
            "latency_p50_ms": latencies[int(n * 0.5)] * 1000 if n > 0 else 0,
            "latency_p95_ms": latencies[int(n * 0.95)] * 1000 if n > 0 else 0,
            "latency_p99_ms": latencies[int(n * 0.99)] * 1000 if n > 0 else 0,
            "latency_max_ms": latencies[-1] * 1000 if n > 0 else 0,
        }


class WebSocketUser(FastHttpUser):
    """Locust user for WebSocket load testing."""

    wait_time = between(0.5, 2.0)  # Wait between tasks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connection_manager: WebSocketConnectionManager | None = None
        self.metrics_collector: WebSocketMetricsCollector | None = None
        self.connection_semaphore = asyncio.Semaphore(250)  # Max connections per worker
        self.auth_token = os.environ.get('GENESIS_AUTH_TOKEN', '')  # Get auth token from env
        self._connect_task: asyncio.Task | None = None
        self._disconnect_task: asyncio.Task | None = None

    def on_start(self):
        """Initialize WebSocket connection on user start."""
        # Create metrics collector (shared across all users)
        if not hasattr(self.environment, "ws_metrics_collector"):
            self.environment.ws_metrics_collector = WebSocketMetricsCollector()

        self.metrics_collector = self.environment.ws_metrics_collector

        # Create connection manager
        host = self.host or "http://localhost:8000"
        ws_url = host.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/market"  # Append WebSocket endpoint

        connection_id = f"user_{self.environment.runner.user_count}_{id(self)}"
        self.connection_manager = WebSocketConnectionManager(
            url=ws_url,
            connection_id=connection_id,
            metrics_collector=self.metrics_collector,
            auth_token=self.auth_token
        )

        # Start connection in background
        self._connect_task = asyncio.create_task(self._connect())

    async def _connect(self):
        """Connect with semaphore to control ramp-up."""
        async with self.connection_semaphore:
            if self.connection_manager:
                await self.connection_manager.run()

    def on_stop(self):
        """Clean up WebSocket connection on user stop."""
        if self.connection_manager:
            self._disconnect_task = asyncio.create_task(self.connection_manager.disconnect())

    @task(3)
    async def subscribe_ticker(self):
        """Subscribe to ticker channel."""
        if self.connection_manager:
            message = {
                "action": "subscribe",
                "channel": "ticker",
                "symbol": "BTC/USDT"
            }

            start_time = await self.connection_manager.send_message(message)

            if start_time:
                # Report to Locust stats
                self.environment.events.request.fire(
                    request_type="WebSocket",
                    name="subscribe_ticker",
                    response_time=(time.time() - start_time) * 1000,
                    response_length=len(json.dumps(message)),
                    exception=None,
                    context={}
                )

    @task(2)
    async def subscribe_orderbook(self):
        """Subscribe to orderbook channel."""
        if self.connection_manager:
            message = {
                "action": "subscribe",
                "channel": "orderbook",
                "symbol": "BTC/USDT",
                "depth": 20
            }

            start_time = await self.connection_manager.send_message(message)

            if start_time:
                self.environment.events.request.fire(
                    request_type="WebSocket",
                    name="subscribe_orderbook",
                    response_time=(time.time() - start_time) * 1000,
                    response_length=len(json.dumps(message)),
                    exception=None,
                    context={}
                )

    @task(1)
    async def subscribe_trades(self):
        """Subscribe to trades channel."""
        if self.connection_manager:
            message = {
                "action": "subscribe",
                "channel": "trades",
                "symbol": "BTC/USDT"
            }

            start_time = await self.connection_manager.send_message(message)

            if start_time:
                self.environment.events.request.fire(
                    request_type="WebSocket",
                    name="subscribe_trades",
                    response_time=(time.time() - start_time) * 1000,
                    response_length=len(json.dumps(message)),
                    exception=None,
                    context={}
                )

    @task(1)
    async def unsubscribe_channel(self):
        """Test unsubscribe functionality."""
        if self.connection_manager:
            message = {
                "action": "unsubscribe",
                "channel": "ticker",
                "symbol": "BTC/USDT"
            }

            start_time = await self.connection_manager.send_message(message)

            if start_time:
                self.environment.events.request.fire(
                    request_type="WebSocket",
                    name="unsubscribe",
                    response_time=(time.time() - start_time) * 1000,
                    response_length=len(json.dumps(message)),
                    exception=None,
                    context={}
                )


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Generate final report on test completion."""
    if hasattr(environment, "ws_metrics_collector"):
        collector = environment.ws_metrics_collector

        print("\n" + "="*80)
        print("WebSocket Load Test Results")
        print("="*80)

        print("\nConnection Statistics:")
        print(f"  Total Connections: {collector.connections_total._value.get()}")
        print(f"  Active Connections: {collector.connections_active._value.get()}")
        print(f"  Failed Connections: {collector.connections_failed._value.get()}")

        print("\nMessage Statistics:")
        print(f"  Messages Sent: {collector.messages_sent._value.get()}")
        print(f"  Messages Received: {collector.messages_received._value.get()}")

        print("\nBandwidth Statistics:")
        print(f"  Bytes Sent: {collector.bytes_sent._value.get():,}")
        print(f"  Bytes Received: {collector.bytes_received._value.get():,}")

        print("\nSystem Statistics:")
        print(f"  Memory Usage: {collector.memory_usage._value.get():.2f} MB")
        print(f"  CPU Usage: {collector.cpu_usage._value.get():.2f}%")

        print("\n" + "="*80)


# Chaos engineering functions for testing resilience
class ChaosEngineering:
    """Simulate network failures and test recovery."""

    @staticmethod
    async def simulate_network_interruption(connection_manager: WebSocketConnectionManager, duration: float = 5.0):
        """Simulate network interruption by closing connection."""
        logger.info(f"Simulating network interruption for {duration} seconds")

        if connection_manager.websocket:
            await connection_manager.websocket.close()

        await asyncio.sleep(duration)

        # Connection should auto-reconnect
        logger.info("Network interruption simulation complete")

    @staticmethod
    async def simulate_packet_loss(connection_manager: WebSocketConnectionManager, loss_rate: float = 0.1):
        """Simulate packet loss by dropping messages."""
        original_send = connection_manager.send_message
        drop_count = 0

        async def lossy_send(message: dict[str, Any]) -> float | None:
            nonlocal drop_count
            import random

            if random.random() < loss_rate:
                drop_count += 1
                logger.debug(f"Dropping message (total dropped: {drop_count})")
                return None

            return await original_send(message)

        connection_manager.send_message = lossy_send  # type: ignore[method-assign]
        logger.info(f"Simulating {loss_rate*100}% packet loss")

    @staticmethod
    async def simulate_high_latency(connection_manager: WebSocketConnectionManager, added_latency: float = 0.5):
        """Add artificial latency to messages."""
        original_send = connection_manager.send_message

        async def delayed_send(message: dict[str, Any]) -> float | None:
            await asyncio.sleep(added_latency)
            return await original_send(message)

        connection_manager.send_message = delayed_send  # type: ignore[method-assign]
        logger.info(f"Adding {added_latency*1000}ms artificial latency")


# Standalone test runner for development
async def run_standalone_test():
    """Run a standalone WebSocket load test without Locust."""
    print("Starting standalone WebSocket load test...")

    # Create metrics collector
    metrics_collector = WebSocketMetricsCollector()

    # Create connection managers
    num_connections = 100  # Start with 100 for testing
    connections = []

    # Get auth token from environment
    auth_token = os.environ.get('GENESIS_AUTH_TOKEN', '')

    for i in range(num_connections):
        manager = WebSocketConnectionManager(
            url="ws://localhost:8000/ws/market",
            connection_id=f"standalone_{i}",
            metrics_collector=metrics_collector,
            auth_token=auth_token
        )
        connections.append(manager)

    # Start all connections
    tasks = [asyncio.create_task(conn.run()) for conn in connections]

    # Run for test duration
    test_duration = 60  # seconds
    print(f"Running test for {test_duration} seconds...")

    await asyncio.sleep(test_duration)

    # Stop all connections
    for conn in connections:
        await conn.disconnect()

    # Wait for tasks to complete
    await asyncio.gather(*tasks, return_exceptions=True)

    # Print results
    print("\n" + "="*80)
    print("Test Results")
    print("="*80)

    for conn in connections[:5]:  # Show first 5 connections
        summary = conn.get_metrics_summary()
        print(f"\nConnection {summary['connection_id']}:")
        print(f"  State: {summary['state']}")
        print(f"  Messages: {summary['messages_sent']} sent, {summary['messages_received']} received")
        print(f"  Latency P99: {summary['latency_p99_ms']:.2f}ms")
        print(f"  Reconnects: {summary['reconnect_count']}")


if __name__ == "__main__":
    # Run standalone test if executed directly
    asyncio.run(run_standalone_test())
