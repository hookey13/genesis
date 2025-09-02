"""Unit tests for Prometheus metrics implementation."""

import asyncio
import re
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.monitoring.prometheus_exporter import (
    APIKeyRotator,
    Metric,
    MetricType,
    MetricsRegistry,
    PrometheusExporter,
)
from genesis.monitoring.metrics_collector import MetricsCollector, TradingMetrics
from genesis.monitoring.application_metrics import (
    ApplicationMetricsCollector,
    CircuitBreakerState,
    WebSocketState,
)


class TestMetricsRegistry:
    """Test MetricsRegistry functionality."""
    
    @pytest.fixture
    async def registry(self):
        """Create a test registry."""
        return MetricsRegistry()
    
    async def test_register_metric(self, registry):
        """Test metric registration."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            help="Test metric"
        )
        await registry.register(metric)
        
        assert "test_metric" in registry._metrics
        assert registry._metrics["test_metric"].type == MetricType.COUNTER
    
    async def test_increment_counter(self, registry):
        """Test counter increment with labels."""
        # Register counter
        await registry.register(Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            help="Test counter"
        ))
        
        # Increment without labels
        await registry.increment_counter("test_counter", 1.0)
        assert registry._metrics["test_counter"].value == 1.0
        
        # Increment with labels
        await registry.increment_counter(
            "test_counter", 
            2.0,
            {"exchange": "binance", "symbol": "BTC"}
        )
        
        label_key = "exchange=binance,symbol=BTC"
        assert label_key in registry._labeled_metrics["test_counter"]
        assert registry._labeled_metrics["test_counter"][label_key].value == 2.0
    
    async def test_set_gauge(self, registry):
        """Test gauge setting with labels."""
        # Register gauge
        await registry.register(Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            help="Test gauge"
        ))
        
        # Set without labels
        await registry.set_gauge("test_gauge", 42.0)
        assert registry._metrics["test_gauge"].value == 42.0
        
        # Set with labels
        await registry.set_gauge(
            "test_gauge",
            100.0,
            {"type": "memory", "unit": "bytes"}
        )
        
        label_key = "type=memory,unit=bytes"
        assert label_key in registry._labeled_metrics["test_gauge"]
        assert registry._labeled_metrics["test_gauge"][label_key].value == 100.0
    
    async def test_observe_histogram(self, registry):
        """Test histogram observation."""
        # Register histogram
        await registry.register(Metric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            help="Test histogram",
            buckets=[0.1, 0.5, 1.0, 5.0]
        ))
        
        # Observe values
        await registry.observe_histogram("test_histogram", 0.25)
        assert registry._metrics["test_histogram"].value == 0.25
        
        # Observe with labels
        await registry.observe_histogram(
            "test_histogram",
            0.75,
            {"endpoint": "/api/orders"}
        )
        
        label_key = "endpoint=/api/orders"
        assert label_key in registry._labeled_metrics["test_histogram"]
    
    async def test_metric_name_validation(self, registry):
        """Test metric name validation."""
        await registry.register(Metric(
            name="valid_metric_name",
            type=MetricType.COUNTER,
            help="Valid metric"
        ))
        
        # Invalid metric name
        with pytest.raises(ValueError, match="Invalid metric name"):
            await registry.set_gauge("invalid-metric-name", 1.0)
        
        with pytest.raises(ValueError, match="Invalid metric name"):
            await registry.increment_counter("123_starts_with_number", 1.0)
    
    async def test_label_validation(self, registry):
        """Test label name and value validation."""
        await registry.register(Metric(
            name="test_metric",
            type=MetricType.GAUGE,
            help="Test metric"
        ))
        
        # Invalid label name
        with pytest.raises(ValueError, match="Invalid label name"):
            await registry.set_gauge(
                "test_metric",
                1.0,
                {"invalid-label": "value"}
            )
        
        # Invalid label value
        with pytest.raises(ValueError, match="Invalid label value"):
            await registry.set_gauge(
                "test_metric",
                1.0,
                {"label": "invalid\nvalue"}
            )
    
    async def test_collect_prometheus_format(self, registry):
        """Test Prometheus format output."""
        # Register metrics
        await registry.register(Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            help="Test counter metric"
        ))
        await registry.register(Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            help="Test gauge metric"
        ))
        
        # Set values
        await registry.increment_counter("test_counter", 5.0)
        await registry.set_gauge("test_gauge", 42.0)
        await registry.set_gauge(
            "test_gauge",
            100.0,
            {"instance": "node1"}
        )
        
        # Collect metrics
        output = await registry.collect()
        
        # Verify format
        assert "# HELP test_counter Test counter metric" in output
        assert "# TYPE test_counter counter" in output
        assert "test_counter 5.0" in output
        
        assert "# HELP test_gauge Test gauge metric" in output
        assert "# TYPE test_gauge gauge" in output
        assert "test_gauge 42.0" in output
        assert 'test_gauge{instance="node1"} 100.0' in output
    
    async def test_collector_registration(self, registry):
        """Test metric collector registration."""
        called = False
        
        async def test_collector():
            nonlocal called
            called = True
        
        registry.register_collector(test_collector)
        await registry.collect()
        
        assert called


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    @pytest.fixture
    async def collector(self):
        """Create a test metrics collector."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry, enable_memory_profiling=False)
        return collector
    
    async def test_record_order_metrics(self, collector):
        """Test order metrics recording."""
        from genesis.core.models import Order, OrderStatus
        
        # Create test order
        order = MagicMock(spec=Order)
        order.client_order_id = "test_order_123"
        order.status = OrderStatus.FILLED
        order.exchange = "binance"
        order.symbol = "BTC/USDT"
        order.side = "BUY"
        order.order_type = "MARKET"
        order.executed_qty = 0.1
        order.price = 50000
        
        # Record order
        await collector.record_order(order)
        
        assert collector.metrics.orders_placed == 1
        assert collector.metrics.orders_filled == 1
        assert collector.metrics.trades_executed == 1
        assert collector.metrics.total_volume == Decimal("5000")
    
    async def test_record_execution_time(self, collector):
        """Test execution time recording."""
        await collector.record_execution_time(0.025, "binance", "market")
        
        assert len(collector._latency_samples) == 1
        assert collector._latency_samples[0] == 0.025
        
        # Test bounds checking
        await collector.record_execution_time(-1.0)  # Should be rejected
        assert len(collector._latency_samples) == 1
        
        await collector.record_execution_time(100.0)  # Should be capped at 60
        assert len(collector._latency_samples) == 2
        assert collector._latency_samples[1] == 60
    
    async def test_update_pnl_metrics(self, collector):
        """Test P&L metrics update."""
        await collector.update_pnl(
            Decimal("1000.50"),
            Decimal("-250.25"),
            "arbitrage",
            "BTC/USDT"
        )
        
        assert collector.metrics.realized_pnl == Decimal("1000.50")
        assert collector.metrics.unrealized_pnl == Decimal("-250.25")
    
    async def test_update_position_metrics(self, collector):
        """Test position metrics update."""
        from genesis.core.models import Position
        
        # Create test positions
        positions = [
            MagicMock(
                spec=Position,
                symbol="BTC/USDT",
                side="BUY",
                unrealized_pnl=Decimal("100")
            ),
            MagicMock(
                spec=Position,
                symbol="ETH/USDT",
                side="SELL",
                unrealized_pnl=Decimal("-50")
            ),
            MagicMock(
                spec=Position,
                symbol="BTC/USDT",
                side="BUY",
                unrealized_pnl=Decimal("75")
            ),
        ]
        
        await collector.update_position_metrics(positions)
        
        assert collector.metrics.current_positions == 3
        assert collector.metrics.unrealized_pnl == Decimal("125")
    
    async def test_update_tilt_score(self, collector):
        """Test tilt score update."""
        await collector.update_tilt_score(
            45.5,
            {
                "click_speed": 30.0,
                "cancel_rate": 50.0,
                "revenge_trading": 60.0
            }
        )
        
        assert collector.metrics.tilt_score == 45.5
        assert "click_speed" in collector.metrics.tilt_indicators
        assert collector.metrics.tilt_indicators["click_speed"] == 30.0
        
        # Test bounds checking
        await collector.update_tilt_score(150.0, {})  # Should be clamped
        assert collector.metrics.tilt_score == 100.0
    
    async def test_system_metrics_collection(self, collector):
        """Test system metrics collection."""
        await collector._collect_system_metrics()
        
        # Verify metrics are collected
        assert collector.metrics.cpu_usage >= 0
        assert collector.metrics.memory_usage > 0
        assert collector.metrics.disk_usage_percent >= 0
        assert collector.metrics.health_score >= 0
        assert collector.metrics.health_score <= 100
    
    async def test_metrics_summary(self, collector):
        """Test metrics summary generation."""
        # Set some metrics
        collector.metrics.orders_placed = 100
        collector.metrics.orders_filled = 80
        collector.metrics.realized_pnl = Decimal("500.00")
        collector.metrics.tilt_score = 25.0
        
        summary = collector.get_metrics_summary()
        
        assert summary["orders"]["placed"] == 100
        assert summary["orders"]["filled"] == 80
        assert summary["pnl"]["realized"] == 500.0
        assert summary["tilt"]["score"] == 25.0


class TestApplicationMetrics:
    """Test application-specific metrics."""
    
    @pytest.fixture
    async def app_metrics(self):
        """Create application metrics collector."""
        registry = MetricsRegistry()
        return ApplicationMetricsCollector(registry)
    
    async def test_websocket_connection_metrics(self, app_metrics):
        """Test WebSocket connection tracking."""
        # Record connection
        await app_metrics.record_websocket_connection(
            "binance",
            "trades",
            WebSocketState.CONNECTED
        )
        
        key = "binance_trades"
        assert key in app_metrics.websocket_metrics
        assert app_metrics.websocket_metrics[key].state == WebSocketState.CONNECTED
        assert app_metrics.websocket_metrics[key].connections_total == 1
        
        # Record reconnection
        await app_metrics.record_websocket_connection(
            "binance",
            "trades",
            WebSocketState.RECONNECTING
        )
        assert app_metrics.websocket_metrics[key].reconnections_total == 1
    
    async def test_websocket_message_metrics(self, app_metrics):
        """Test WebSocket message tracking."""
        await app_metrics.record_websocket_message(
            "binance",
            "orderbook",
            "received",
            1024
        )
        
        key = "binance_orderbook"
        assert app_metrics.websocket_metrics[key].messages_received == 1
        assert app_metrics.websocket_metrics[key].bytes_received == 1024
    
    async def test_api_request_metrics(self, app_metrics):
        """Test API request tracking."""
        # Record successful request
        await app_metrics.record_api_request(
            "/api/orders",
            0.150,
            True,
            200
        )
        
        assert app_metrics.api_metrics.requests_total["/api/orders"] == 1
        assert app_metrics.api_metrics.requests_success["/api/orders"] == 1
        
        # Record failed request
        await app_metrics.record_api_request(
            "/api/orders",
            0.050,
            False,
            500
        )
        
        assert app_metrics.api_metrics.requests_total["/api/orders"] == 2
        assert app_metrics.api_metrics.requests_failed["/api/orders"] == 1
        
        # Record rate limit hit
        await app_metrics.record_api_request(
            "/api/orders",
            0.010,
            False,
            429
        )
        
        assert app_metrics.api_metrics.rate_limit_hits == 1
    
    async def test_circuit_breaker_metrics(self, app_metrics):
        """Test circuit breaker state tracking."""
        # Record circuit breaker opening
        await app_metrics.record_circuit_breaker_event(
            "exchange_api",
            CircuitBreakerState.OPEN,
            "Too many failures"
        )
        
        cb = app_metrics.circuit_breakers["exchange_api"]
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failures_count == 1
        assert cb.last_failure_reason == "Too many failures"
        
        # Record state change
        await app_metrics.record_circuit_breaker_event(
            "exchange_api",
            CircuitBreakerState.HALF_OPEN
        )
        
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.state_changes == 1
    
    async def test_metrics_summaries(self, app_metrics):
        """Test metrics summary generation."""
        # Set up some metrics
        await app_metrics.record_websocket_connection(
            "binance", "trades", WebSocketState.CONNECTED
        )
        await app_metrics.record_api_request("/api/orders", 0.1, True, 200)
        await app_metrics.record_circuit_breaker_event(
            "api", CircuitBreakerState.CLOSED
        )
        
        # Get summaries
        ws_summary = app_metrics.get_websocket_summary()
        assert ws_summary["total_connections"] == 1
        assert ws_summary["active_connections"] == 1
        
        api_summary = app_metrics.get_api_summary()
        assert api_summary["total_requests"] == 1
        assert api_summary["successful_requests"] == 1
        
        cb_summary = app_metrics.get_circuit_breaker_summary()
        assert "api" in cb_summary
        assert cb_summary["api"]["state"] == "closed"


class TestPrometheusExporter:
    """Test PrometheusExporter functionality."""
    
    @pytest.fixture
    async def exporter(self):
        """Create test exporter."""
        registry = MetricsRegistry()
        exporter = PrometheusExporter(
            registry=registry,
            port=9090,
            use_https=False,
            production_mode=False
        )
        return exporter
    
    async def test_api_key_rotation(self, exporter):
        """Test API key rotation."""
        rotator = exporter.api_key_rotator
        
        initial_key = rotator.current_key
        assert rotator.validate_key(initial_key)
        
        # Invalid key should fail
        assert not rotator.validate_key("invalid_key")
        
        # Force rotation
        from datetime import datetime, timedelta
        rotator.rotation_time = datetime.utcnow() - timedelta(seconds=1)
        rotator.rotate_if_needed()
        
        # Old key should still work (grace period)
        assert rotator.validate_key(initial_key)
        
        # New key should work
        assert rotator.validate_key(rotator.current_key)
        assert rotator.current_key != initial_key
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_auth(self, exporter):
        """Test metrics endpoint authentication."""
        from fastapi.testclient import TestClient
        
        client = TestClient(exporter.app)
        
        # No auth should fail
        response = client.get("/metrics")
        assert response.status_code == 403
        
        # Invalid auth should fail
        response = client.get(
            "/metrics",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 403
        
        # Valid auth should succeed
        valid_token = exporter.api_key_rotator.current_key
        response = client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {valid_token}"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4"
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, exporter):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        
        client = TestClient(exporter.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "api_key_rotation" in data
    
    async def test_default_metrics_registration(self, exporter):
        """Test default metrics are registered."""
        registry = exporter.registry
        
        # Wait for async registration
        await asyncio.sleep(0.1)
        
        # Check some default metrics exist
        assert "genesis_up" in registry._metrics
        assert "genesis_orders_total" in registry._metrics
        assert "genesis_position_count" in registry._metrics
        assert "genesis_websocket_latency_ms" in registry._metrics


@pytest.mark.asyncio
async def test_metric_name_format():
    """Test metric names follow Prometheus conventions."""
    pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    test_names = [
        ("genesis_orders_total", True),
        ("genesis_pnl_dollars", True),
        ("invalid-metric-name", False),
        ("123_starts_with_number", False),
        ("has spaces", False),
        ("_valid_underscore", True),
    ]
    
    for name, should_match in test_names:
        assert bool(pattern.match(name)) == should_match


@pytest.mark.asyncio
async def test_label_validation():
    """Test label name and value validation."""
    label_name_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    label_value_pattern = re.compile(r'^[\w\s\-\.]+$')
    
    # Test label names
    assert label_name_pattern.match("exchange")
    assert label_name_pattern.match("order_type")
    assert not label_name_pattern.match("order-type")
    assert not label_name_pattern.match("123_order")
    
    # Test label values  
    assert label_value_pattern.match("binance")
    assert label_value_pattern.match("BTC-USDT")
    assert label_value_pattern.match("market_order")
    assert not label_value_pattern.match("has\nnewline")
    assert not label_value_pattern.match("has\"quote")