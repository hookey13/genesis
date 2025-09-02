"""Application-specific metrics for WebSocket, API, and circuit breaker monitoring."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import structlog

from .prometheus_exporter import MetricsRegistry

logger = structlog.get_logger(__name__)


class WebSocketState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, not allowing requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class WebSocketMetrics:
    """WebSocket connection metrics."""
    connections_active: int = 0
    connections_total: int = 0
    reconnections_total: int = 0
    messages_received: int = 0
    messages_sent: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    errors_total: int = 0
    last_error: Optional[str] = None
    connection_duration_seconds: float = 0.0
    state: WebSocketState = WebSocketState.DISCONNECTED
    last_heartbeat: Optional[datetime] = None


@dataclass
class APIMetrics:
    """API request metrics."""
    requests_total: Dict[str, int] = field(default_factory=dict)
    requests_success: Dict[str, int] = field(default_factory=dict)
    requests_failed: Dict[str, int] = field(default_factory=dict)
    request_duration_seconds: Dict[str, list] = field(default_factory=dict)
    rate_limit_hits: int = 0
    rate_limit_remaining: Dict[str, int] = field(default_factory=dict)
    rate_limit_reset: Dict[str, datetime] = field(default_factory=dict)
    last_error_by_endpoint: Dict[str, str] = field(default_factory=dict)


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failures_count: int = 0
    success_count: int = 0
    half_open_attempts: int = 0
    state_changes: int = 0
    last_state_change: Optional[datetime] = None
    last_failure_reason: Optional[str] = None
    open_duration_seconds: float = 0.0


class ApplicationMetricsCollector:
    """Collects application-specific metrics for WebSocket, API, and circuit breaker."""
    
    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self.websocket_metrics: Dict[str, WebSocketMetrics] = {}
        self.api_metrics = APIMetrics()
        self.circuit_breakers: Dict[str, CircuitBreakerMetrics] = {}
        self._start_time = time.time()
        
        # Register metrics
        asyncio.create_task(self._register_metrics())
        
        # Register collector with registry
        self.registry.register_collector(self._update_metrics)
        
    async def _register_metrics(self) -> None:
        """Register application metrics with Prometheus."""
        from .prometheus_exporter import Metric, MetricType
        
        # WebSocket metrics
        await self.registry.register(Metric(
            name="genesis_websocket_connections_active",
            type=MetricType.GAUGE,
            help="Number of active WebSocket connections by exchange and stream"
        ))
        
        await self.registry.register(Metric(
            name="genesis_websocket_connections_total",
            type=MetricType.COUNTER,
            help="Total WebSocket connections established"
        ))
        
        await self.registry.register(Metric(
            name="genesis_websocket_reconnections_total",
            type=MetricType.COUNTER,
            help="Total WebSocket reconnection attempts"
        ))
        
        await self.registry.register(Metric(
            name="genesis_websocket_messages_total",
            type=MetricType.COUNTER,
            help="Total WebSocket messages by direction (sent/received)"
        ))
        
        await self.registry.register(Metric(
            name="genesis_websocket_bytes_total",
            type=MetricType.COUNTER,
            help="Total WebSocket bytes transferred by direction"
        ))
        
        await self.registry.register(Metric(
            name="genesis_websocket_errors_total",
            type=MetricType.COUNTER,
            help="Total WebSocket errors by exchange"
        ))
        
        await self.registry.register(Metric(
            name="genesis_websocket_state",
            type=MetricType.GAUGE,
            help="WebSocket connection state (1=connected, 0=disconnected)"
        ))
        
        # API metrics
        await self.registry.register(Metric(
            name="genesis_api_requests_total",
            type=MetricType.COUNTER,
            help="Total API requests by endpoint and status"
        ))
        
        await self.registry.register(Metric(
            name="genesis_api_request_duration_seconds",
            type=MetricType.HISTOGRAM,
            help="API request duration by endpoint",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ))
        
        await self.registry.register(Metric(
            name="genesis_api_rate_limit_hits_total",
            type=MetricType.COUNTER,
            help="Total rate limit hits"
        ))
        
        await self.registry.register(Metric(
            name="genesis_api_rate_limit_remaining",
            type=MetricType.GAUGE,
            help="Remaining rate limit by endpoint"
        ))
        
        # Circuit breaker metrics
        await self.registry.register(Metric(
            name="genesis_circuit_breaker_state",
            type=MetricType.GAUGE,
            help="Circuit breaker state (0=closed, 1=open, 2=half-open)"
        ))
        
        await self.registry.register(Metric(
            name="genesis_circuit_breaker_failures_total",
            type=MetricType.COUNTER,
            help="Total circuit breaker failures by service"
        ))
        
        await self.registry.register(Metric(
            name="genesis_circuit_breaker_state_changes_total",
            type=MetricType.COUNTER,
            help="Total circuit breaker state changes"
        ))
        
        await self.registry.register(Metric(
            name="genesis_circuit_breaker_open_duration_seconds",
            type=MetricType.GAUGE,
            help="Duration circuit breaker has been open"
        ))
        
        logger.info("Registered application metrics")
        
    async def record_websocket_connection(self, exchange: str, stream: str, 
                                         state: WebSocketState) -> None:
        """Record WebSocket connection state change."""
        key = f"{exchange}_{stream}"
        
        if key not in self.websocket_metrics:
            self.websocket_metrics[key] = WebSocketMetrics()
            
        metrics = self.websocket_metrics[key]
        old_state = metrics.state
        metrics.state = state
        
        if state == WebSocketState.CONNECTED:
            metrics.connections_total += 1
            metrics.connections_active = 1
            metrics.connection_duration_seconds = time.time()
        elif state == WebSocketState.DISCONNECTED:
            metrics.connections_active = 0
            if metrics.connection_duration_seconds > 0:
                metrics.connection_duration_seconds = time.time() - metrics.connection_duration_seconds
        elif state == WebSocketState.RECONNECTING:
            metrics.reconnections_total += 1
        elif state == WebSocketState.ERROR:
            metrics.errors_total += 1
            
        # Update Prometheus metrics
        await self.registry.set_gauge(
            "genesis_websocket_connections_active",
            float(metrics.connections_active),
            {"exchange": exchange, "stream": stream}
        )
        
        await self.registry.set_gauge(
            "genesis_websocket_state",
            1.0 if state == WebSocketState.CONNECTED else 0.0,
            {"exchange": exchange, "stream": stream}
        )
        
        logger.debug("WebSocket state updated",
                    exchange=exchange,
                    stream=stream,
                    old_state=old_state.value,
                    new_state=state.value)
                    
    async def record_websocket_message(self, exchange: str, stream: str,
                                      direction: str, bytes_count: int) -> None:
        """Record WebSocket message."""
        key = f"{exchange}_{stream}"
        
        if key not in self.websocket_metrics:
            self.websocket_metrics[key] = WebSocketMetrics()
            
        metrics = self.websocket_metrics[key]
        
        if direction == "received":
            metrics.messages_received += 1
            metrics.bytes_received += bytes_count
        else:
            metrics.messages_sent += 1
            metrics.bytes_sent += bytes_count
            
        # Update heartbeat
        metrics.last_heartbeat = datetime.now()
        
        # Update Prometheus metrics
        await self.registry.increment_counter(
            "genesis_websocket_messages_total",
            1.0,
            {"exchange": exchange, "stream": stream, "direction": direction}
        )
        
        await self.registry.increment_counter(
            "genesis_websocket_bytes_total",
            float(bytes_count),
            {"exchange": exchange, "stream": stream, "direction": direction}
        )
        
    async def record_api_request(self, endpoint: str, duration_seconds: float,
                                success: bool, status_code: int = 200) -> None:
        """Record API request metrics."""
        # Update counters
        if endpoint not in self.api_metrics.requests_total:
            self.api_metrics.requests_total[endpoint] = 0
            self.api_metrics.requests_success[endpoint] = 0
            self.api_metrics.requests_failed[endpoint] = 0
            self.api_metrics.request_duration_seconds[endpoint] = []
            
        self.api_metrics.requests_total[endpoint] += 1
        
        if success:
            self.api_metrics.requests_success[endpoint] += 1
            status = "success"
        else:
            self.api_metrics.requests_failed[endpoint] += 1
            status = "failed"
            
        # Track duration
        self.api_metrics.request_duration_seconds[endpoint].append(duration_seconds)
        if len(self.api_metrics.request_duration_seconds[endpoint]) > 1000:
            self.api_metrics.request_duration_seconds[endpoint].pop(0)
            
        # Update Prometheus metrics
        await self.registry.increment_counter(
            "genesis_api_requests_total",
            1.0,
            {"endpoint": endpoint, "status": status, "code": str(status_code)}
        )
        
        await self.registry.observe_histogram(
            "genesis_api_request_duration_seconds",
            duration_seconds,
            {"endpoint": endpoint}
        )
        
        # Check for rate limit response
        if status_code == 429:
            self.api_metrics.rate_limit_hits += 1
            await self.registry.increment_counter(
                "genesis_api_rate_limit_hits_total",
                1.0,
                {"endpoint": endpoint}
            )
            
    async def update_rate_limit(self, endpoint: str, remaining: int, 
                               reset_time: datetime) -> None:
        """Update API rate limit metrics."""
        self.api_metrics.rate_limit_remaining[endpoint] = remaining
        self.api_metrics.rate_limit_reset[endpoint] = reset_time
        
        await self.registry.set_gauge(
            "genesis_api_rate_limit_remaining",
            float(remaining),
            {"endpoint": endpoint}
        )
        
    async def record_circuit_breaker_event(self, service: str, 
                                          state: CircuitBreakerState,
                                          failure_reason: Optional[str] = None) -> None:
        """Record circuit breaker state change."""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreakerMetrics()
            
        cb = self.circuit_breakers[service]
        old_state = cb.state
        cb.state = state
        
        if old_state != state:
            cb.state_changes += 1
            cb.last_state_change = datetime.now()
            
            if state == CircuitBreakerState.OPEN:
                cb.open_duration_seconds = time.time()
            elif old_state == CircuitBreakerState.OPEN:
                cb.open_duration_seconds = time.time() - cb.open_duration_seconds
                
        if failure_reason:
            cb.failures_count += 1
            cb.last_failure_reason = failure_reason
            
        # Map state to numeric value
        state_value = {
            CircuitBreakerState.CLOSED: 0,
            CircuitBreakerState.OPEN: 1,
            CircuitBreakerState.HALF_OPEN: 2
        }[state]
        
        # Update Prometheus metrics
        await self.registry.set_gauge(
            "genesis_circuit_breaker_state",
            float(state_value),
            {"service": service}
        )
        
        if failure_reason:
            await self.registry.increment_counter(
                "genesis_circuit_breaker_failures_total",
                1.0,
                {"service": service, "reason": failure_reason[:50]}
            )
            
        if old_state != state:
            await self.registry.increment_counter(
                "genesis_circuit_breaker_state_changes_total",
                1.0,
                {"service": service, "from": old_state.value, "to": state.value}
            )
            
        logger.info("Circuit breaker state updated",
                   service=service,
                   old_state=old_state.value,
                   new_state=state.value,
                   failure_reason=failure_reason)
                   
    async def _update_metrics(self) -> None:
        """Update metrics (called by registry during collection)."""
        # Update WebSocket metrics
        for key, metrics in self.websocket_metrics.items():
            exchange, stream = key.split("_", 1)
            
            await self.registry.set_gauge(
                "genesis_websocket_connections_active",
                float(metrics.connections_active),
                {"exchange": exchange, "stream": stream}
            )
            
            # Check for stale connections
            if metrics.last_heartbeat:
                time_since_heartbeat = (datetime.now() - metrics.last_heartbeat).total_seconds()
                if time_since_heartbeat > 60 and metrics.state == WebSocketState.CONNECTED:
                    logger.warning("Stale WebSocket connection detected",
                                 exchange=exchange,
                                 stream=stream,
                                 seconds_since_heartbeat=time_since_heartbeat)
                                 
        # Update circuit breaker open duration
        for service, cb in self.circuit_breakers.items():
            if cb.state == CircuitBreakerState.OPEN and cb.open_duration_seconds > 0:
                current_open_duration = time.time() - cb.open_duration_seconds
                await self.registry.set_gauge(
                    "genesis_circuit_breaker_open_duration_seconds",
                    current_open_duration,
                    {"service": service}
                )
                
    def get_websocket_summary(self) -> dict:
        """Get WebSocket metrics summary."""
        summary = {
            "total_connections": sum(m.connections_total for m in self.websocket_metrics.values()),
            "active_connections": sum(m.connections_active for m in self.websocket_metrics.values()),
            "total_reconnections": sum(m.reconnections_total for m in self.websocket_metrics.values()),
            "total_errors": sum(m.errors_total for m in self.websocket_metrics.values()),
            "connections": {}
        }
        
        for key, metrics in self.websocket_metrics.items():
            summary["connections"][key] = {
                "state": metrics.state.value,
                "messages_received": metrics.messages_received,
                "messages_sent": metrics.messages_sent,
                "bytes_received": metrics.bytes_received,
                "bytes_sent": metrics.bytes_sent,
                "errors": metrics.errors_total
            }
            
        return summary
        
    def get_api_summary(self) -> dict:
        """Get API metrics summary."""
        return {
            "total_requests": sum(self.api_metrics.requests_total.values()),
            "successful_requests": sum(self.api_metrics.requests_success.values()),
            "failed_requests": sum(self.api_metrics.requests_failed.values()),
            "rate_limit_hits": self.api_metrics.rate_limit_hits,
            "endpoints": {
                endpoint: {
                    "total": self.api_metrics.requests_total.get(endpoint, 0),
                    "success": self.api_metrics.requests_success.get(endpoint, 0),
                    "failed": self.api_metrics.requests_failed.get(endpoint, 0),
                    "rate_limit_remaining": self.api_metrics.rate_limit_remaining.get(endpoint, 0)
                }
                for endpoint in self.api_metrics.requests_total
            }
        }
        
    def get_circuit_breaker_summary(self) -> dict:
        """Get circuit breaker metrics summary."""
        return {
            service: {
                "state": cb.state.value,
                "failures": cb.failures_count,
                "successes": cb.success_count,
                "state_changes": cb.state_changes,
                "last_failure": cb.last_failure_reason
            }
            for service, cb in self.circuit_breakers.items()
        }