"""Unit tests for Prometheus exporter."""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock

from genesis.monitoring.prometheus_exporter import (
    Metric,
    MetricType,
    MetricsRegistry,
    PrometheusExporter
)


class TestMetric:
    """Test Metric class."""
    
    def test_metric_creation(self):
        """Test creating a metric."""
        metric = Metric(
            name="test_metric",
            type=MetricType.GAUGE,
            help="Test metric",
            value=42.0
        )
        
        assert metric.name == "test_metric"
        assert metric.type == MetricType.GAUGE
        assert metric.value == 42.0
    
    def test_metric_format_prometheus_no_labels(self):
        """Test formatting metric without labels."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            help="Test counter",
            value=100.0
        )
        
        output = metric.format_prometheus()
        assert "# HELP test_metric Test counter" in output
        assert "# TYPE test_metric counter" in output
        assert "test_metric 100.0" in output
    
    def test_metric_format_prometheus_with_labels(self):
        """Test formatting metric with labels."""
        metric = Metric(
            name="test_metric",
            type=MetricType.GAUGE,
            help="Test gauge",
            value=3.14,
            labels={"tier": "sniper", "exchange": "binance"}
        )
        
        output = metric.format_prometheus()
        assert "# HELP test_metric Test gauge" in output
        assert "# TYPE test_metric gauge" in output
        assert 'test_metric{tier="sniper",exchange="binance"} 3.14' in output


@pytest.mark.asyncio
class TestMetricsRegistry:
    """Test MetricsRegistry class."""
    
    async def test_register_metric(self):
        """Test registering a metric."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            help="Test counter"
        )
        
        await registry.register(metric)
        assert "test_counter" in registry._metrics
        assert registry._metrics["test_counter"] == metric
    
    async def test_unregister_metric(self):
        """Test unregistering a metric."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            help="Test gauge"
        )
        
        await registry.register(metric)
        assert "test_gauge" in registry._metrics
        
        await registry.unregister("test_gauge")
        assert "test_gauge" not in registry._metrics
    
    async def test_set_gauge(self):
        """Test setting gauge value."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            help="Test gauge"
        )
        
        await registry.register(metric)
        await registry.set_gauge("test_gauge", 123.45)
        
        assert registry._metrics["test_gauge"].value == 123.45
    
    async def test_set_gauge_with_labels(self):
        """Test setting gauge with labels."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            help="Test gauge"
        )
        
        await registry.register(metric)
        await registry.set_gauge("test_gauge", 99.9, {"status": "active"})
        
        assert registry._metrics["test_gauge"].value == 99.9
        assert registry._metrics["test_gauge"].labels == {"status": "active"}
    
    async def test_increment_counter(self):
        """Test incrementing counter."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            help="Test counter",
            value=10.0
        )
        
        await registry.register(metric)
        await registry.increment_counter("test_counter", 5.0)
        
        assert registry._metrics["test_counter"].value == 15.0
    
    async def test_increment_counter_default(self):
        """Test incrementing counter with default value."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            help="Test counter",
            value=0.0
        )
        
        await registry.register(metric)
        await registry.increment_counter("test_counter")
        
        assert registry._metrics["test_counter"].value == 1.0
    
    async def test_observe_histogram(self):
        """Test observing histogram value."""
        registry = MetricsRegistry()
        metric = Metric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            help="Test histogram",
            buckets=[0.1, 0.5, 1.0, 5.0]
        )
        
        await registry.register(metric)
        await registry.observe_histogram("test_histogram", 0.75)
        
        # For now, just stores the latest value
        assert registry._metrics["test_histogram"].value == 0.75
    
    async def test_collect(self):
        """Test collecting all metrics."""
        registry = MetricsRegistry()
        
        # Register multiple metrics
        await registry.register(Metric(
            name="metric1",
            type=MetricType.GAUGE,
            help="First metric",
            value=1.0
        ))
        
        await registry.register(Metric(
            name="metric2",
            type=MetricType.COUNTER,
            help="Second metric",
            value=2.0
        ))
        
        output = await registry.collect()
        
        assert "# HELP metric1 First metric" in output
        assert "# TYPE metric1 gauge" in output
        assert "metric1 1.0" in output
        
        assert "# HELP metric2 Second metric" in output
        assert "# TYPE metric2 counter" in output
        assert "metric2 2.0" in output
    
    async def test_register_collector(self):
        """Test registering a collector function."""
        registry = MetricsRegistry()
        
        # Create a mock collector
        collector_called = False
        
        async def test_collector():
            nonlocal collector_called
            collector_called = True
        
        registry.register_collector(test_collector)
        
        # Collect should call the collector
        await registry.collect()
        assert collector_called


@pytest.mark.asyncio
class TestPrometheusExporter:
    """Test PrometheusExporter class."""
    
    async def test_exporter_creation(self):
        """Test creating exporter."""
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9091)
        
        assert exporter.registry == registry
        assert exporter.port == 9091
        assert exporter.app is not None
    
    async def test_exporter_with_auth(self):
        """Test exporter with authentication."""
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9092, auth_token="secret123")
        
        assert exporter.auth_token == "secret123"
    
    @patch('genesis.monitoring.prometheus_exporter.uvicorn')
    async def test_start_exporter(self, mock_uvicorn):
        """Test starting the exporter."""
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9093)
        
        # Mock uvicorn
        mock_server = AsyncMock()
        mock_uvicorn.Server.return_value = mock_server
        
        # Start exporter
        task = exporter.run_in_background()
        assert isinstance(task, asyncio.Task)
        
        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    async def test_default_metrics_registered(self):
        """Test that default metrics are registered."""
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9094)
        
        # Wait for default metrics to be registered
        await asyncio.sleep(0.1)
        
        # Check some default metrics
        assert "genesis_up" in registry._metrics
        assert "genesis_info" in registry._metrics
        assert "genesis_position_count" in registry._metrics
        assert "genesis_pnl_dollars" in registry._metrics
        assert "genesis_connection_status" in registry._metrics
        assert "genesis_orders_total" in registry._metrics
        assert "genesis_tilt_score" in registry._metrics
    
    async def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        from fastapi.testclient import TestClient
        
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9095)
        
        # Add a test metric
        await registry.register(Metric(
            name="test_metric",
            type=MetricType.GAUGE,
            help="Test metric",
            value=42.0
        ))
        
        # Test the endpoint
        client = TestClient(exporter.app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "test_metric" in response.text
    
    async def test_metrics_endpoint_with_auth(self):
        """Test metrics endpoint with authentication."""
        from fastapi.testclient import TestClient
        
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9096, auth_token="secret123")
        
        client = TestClient(exporter.app)
        
        # Test without auth
        response = client.get("/metrics")
        assert response.status_code == 401
        
        # Test with wrong auth
        response = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 403
        
        # Test with correct auth
        response = client.get("/metrics", headers={"Authorization": "Bearer secret123"})
        assert response.status_code == 200
    
    async def test_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        
        registry = MetricsRegistry()
        exporter = PrometheusExporter(registry, port=9097)
        
        client = TestClient(exporter.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data