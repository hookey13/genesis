"""Integration tests for Prometheus metrics endpoint."""

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from genesis.core.models import Order, OrderStatus, Position
from genesis.monitoring.application_metrics import (
    ApplicationMetricsCollector,
    CircuitBreakerState,
    WebSocketState,
)
from genesis.monitoring.metrics_collector import MetricsCollector
from genesis.monitoring.prometheus_exporter import (
    Metric,
    MetricType,
    MetricsRegistry,
    PrometheusExporter,
)


@pytest.fixture
async def full_metrics_stack():
    """Create a full metrics collection stack."""
    registry = MetricsRegistry()
    
    # Initialize all collectors
    metrics_collector = MetricsCollector(registry, enable_memory_profiling=False)
    app_metrics = ApplicationMetricsCollector(registry)
    
    # Initialize exporter
    exporter = PrometheusExporter(
        registry=registry,
        port=9091,  # Use different port for tests
        use_https=False,
        production_mode=False
    )
    
    # Start collectors
    await metrics_collector.start()
    
    yield {
        "registry": registry,
        "metrics_collector": metrics_collector,
        "app_metrics": app_metrics,
        "exporter": exporter,
        "client": TestClient(exporter.app),
        "api_key": exporter.api_key_rotator.current_key
    }
    
    # Cleanup
    await metrics_collector.stop()


@pytest.mark.asyncio
async def test_metrics_endpoint_auth(full_metrics_stack):
    """Test /metrics endpoint authentication."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    
    # Test without auth
    response = client.get("/metrics")
    assert response.status_code == 403
    
    # Test with invalid auth
    response = client.get(
        "/metrics",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 403
    
    # Test with valid auth
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_metrics_format(full_metrics_stack):
    """Test Prometheus exposition format."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    registry = full_metrics_stack["registry"]
    
    # Register and set some metrics
    await registry.register(Metric(
        name="test_gauge",
        type=MetricType.GAUGE,
        help="Test gauge metric"
    ))
    await registry.set_gauge("test_gauge", 42.0)
    
    # Get metrics
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    content = response.text
    
    # Verify Prometheus format
    assert "# HELP test_gauge Test gauge metric" in content
    assert "# TYPE test_gauge gauge" in content
    assert "test_gauge 42.0" in content
    
    # Verify default metrics are present
    assert "# HELP genesis_up" in content
    assert "# TYPE genesis_up gauge" in content


@pytest.mark.asyncio
async def test_metrics_performance(full_metrics_stack):
    """Test metrics collection doesn't exceed 1% overhead."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    metrics_collector = full_metrics_stack["metrics_collector"]
    
    # Measure baseline operation time
    baseline_times = []
    for _ in range(100):
        start = time.perf_counter()
        # Simulate order processing
        order = MagicMock()
        order.client_order_id = "test"
        order.status = OrderStatus.FILLED
        baseline_times.append(time.perf_counter() - start)
    
    baseline_avg = sum(baseline_times) / len(baseline_times)
    
    # Measure with metrics collection
    metrics_times = []
    for i in range(100):
        start = time.perf_counter()
        # Process order with metrics
        order = MagicMock()
        order.client_order_id = f"test_{i}"
        order.status = OrderStatus.FILLED
        order.exchange = "binance"
        order.symbol = "BTC/USDT"
        order.side = "BUY"
        order.order_type = "MARKET"
        await metrics_collector.record_order(order)
        metrics_times.append(time.perf_counter() - start)
    
    metrics_avg = sum(metrics_times) / len(metrics_times)
    
    # Calculate overhead
    if baseline_avg > 0:
        overhead_percent = ((metrics_avg - baseline_avg) / baseline_avg) * 100
    else:
        overhead_percent = 0
    
    # Assert overhead is less than 1%
    assert overhead_percent < 1.0, f"Metrics overhead {overhead_percent:.2f}% exceeds 1%"


@pytest.mark.asyncio
async def test_trading_metrics_integration(full_metrics_stack):
    """Test trading metrics collection and exposure."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    metrics_collector = full_metrics_stack["metrics_collector"]
    
    # Record trading activity
    order = MagicMock(spec=Order)
    order.client_order_id = "order_123"
    order.status = OrderStatus.FILLED
    order.exchange = "binance"
    order.symbol = "ETH/USDT"
    order.side = "SELL"
    order.order_type = "LIMIT"
    order.executed_qty = 2.5
    order.price = 2000
    
    await metrics_collector.record_order(order)
    await metrics_collector.record_execution_time(0.125, "binance", "limit")
    await metrics_collector.update_pnl(
        Decimal("1500.00"),
        Decimal("-200.00"),
        "momentum",
        "ETH/USDT"
    )
    
    # Update positions
    positions = [
        MagicMock(
            spec=Position,
            symbol="ETH/USDT",
            side="SELL",
            unrealized_pnl=Decimal("-200.00")
        )
    ]
    await metrics_collector.update_position_metrics(positions)
    
    # Collect metrics
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    content = response.text
    
    # Verify trading metrics are present
    assert "genesis_orders_total" in content
    assert 'exchange="binance"' in content
    assert 'symbol="ETH/USDT"' in content
    assert 'status="FILLED"' in content
    assert "genesis_trading_volume_usdt" in content
    assert "genesis_trading_pnl_usdt" in content
    assert "genesis_positions_count" in content


@pytest.mark.asyncio
async def test_websocket_metrics_integration(full_metrics_stack):
    """Test WebSocket metrics collection."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    app_metrics = full_metrics_stack["app_metrics"]
    
    # Record WebSocket activity
    await app_metrics.record_websocket_connection(
        "binance",
        "trades",
        WebSocketState.CONNECTED
    )
    
    for i in range(10):
        await app_metrics.record_websocket_message(
            "binance",
            "trades",
            "received",
            512
        )
    
    await app_metrics.record_websocket_message(
        "binance",
        "trades",
        "sent",
        64
    )
    
    # Collect metrics
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    content = response.text
    
    # Verify WebSocket metrics
    assert "genesis_websocket_connections_active" in content
    assert "genesis_websocket_messages_total" in content
    assert "genesis_websocket_bytes_total" in content
    assert 'exchange="binance"' in content
    assert 'stream="trades"' in content


@pytest.mark.asyncio
async def test_api_metrics_integration(full_metrics_stack):
    """Test API metrics collection."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    app_metrics = full_metrics_stack["app_metrics"]
    
    # Record API requests
    endpoints = ["/api/orders", "/api/positions", "/api/balance"]
    
    for endpoint in endpoints:
        # Successful requests
        for _ in range(5):
            await app_metrics.record_api_request(
                endpoint,
                0.050,
                True,
                200
            )
        
        # Failed request
        await app_metrics.record_api_request(
            endpoint,
            0.100,
            False,
            500
        )
    
    # Rate limit hit
    await app_metrics.record_api_request(
        "/api/orders",
        0.001,
        False,
        429
    )
    
    # Collect metrics
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    content = response.text
    
    # Verify API metrics
    assert "genesis_api_requests_total" in content
    assert "genesis_api_request_duration_seconds" in content
    assert "genesis_api_rate_limit_hits_total" in content
    assert 'endpoint="/api/orders"' in content
    assert 'status="success"' in content
    assert 'status="failed"' in content


@pytest.mark.asyncio
async def test_circuit_breaker_metrics_integration(full_metrics_stack):
    """Test circuit breaker metrics."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    app_metrics = full_metrics_stack["app_metrics"]
    
    # Record circuit breaker events
    services = ["exchange_api", "database", "websocket"]
    
    for service in services:
        # Start closed
        await app_metrics.record_circuit_breaker_event(
            service,
            CircuitBreakerState.CLOSED
        )
        
        # Failures cause opening
        for i in range(3):
            await app_metrics.record_circuit_breaker_event(
                service,
                CircuitBreakerState.CLOSED,
                f"Failure {i+1}"
            )
        
        # Open circuit
        await app_metrics.record_circuit_breaker_event(
            service,
            CircuitBreakerState.OPEN
        )
    
    # Collect metrics
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    content = response.text
    
    # Verify circuit breaker metrics
    assert "genesis_circuit_breaker_state" in content
    assert "genesis_circuit_breaker_failures_total" in content
    assert "genesis_circuit_breaker_state_changes_total" in content
    assert 'service="exchange_api"' in content


@pytest.mark.asyncio
async def test_system_metrics_integration(full_metrics_stack):
    """Test system metrics collection."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    metrics_collector = full_metrics_stack["metrics_collector"]
    
    # Trigger system metrics collection
    await metrics_collector._collect_system_metrics()
    
    # Collect metrics
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    content = response.text
    
    # Verify system metrics
    assert "genesis_cpu_usage_percent" in content
    assert "genesis_memory_usage_bytes" in content
    assert "genesis_disk_usage_percent" in content
    assert "genesis_health_score" in content
    assert "genesis_process_uptime_seconds" in content


@pytest.mark.asyncio
async def test_grafana_dashboard_import(full_metrics_stack):
    """Test Grafana dashboard JSON is valid."""
    import json
    from pathlib import Path
    
    dashboard_dir = Path("genesis/config/grafana")
    
    # Check dashboard files exist and are valid JSON
    dashboard_files = [
        "trading_operations.json",
        "system_performance.json",
        "alerts.json"
    ]
    
    for dashboard_file in dashboard_files:
        filepath = dashboard_dir / dashboard_file
        assert filepath.exists(), f"Dashboard file {dashboard_file} not found"
        
        # Validate JSON
        with open(filepath, "r") as f:
            try:
                dashboard_json = json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {dashboard_file}: {e}")
        
        # Check required fields
        assert "dashboard" in dashboard_json
        assert "title" in dashboard_json["dashboard"]
        assert "panels" in dashboard_json["dashboard"]
        assert len(dashboard_json["dashboard"]["panels"]) > 0


@pytest.mark.asyncio
async def test_prometheus_config_valid():
    """Test Prometheus configuration is valid."""
    import yaml
    from pathlib import Path
    
    config_file = Path("genesis/config/prometheus.yml")
    assert config_file.exists(), "Prometheus config not found"
    
    # Validate YAML
    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in prometheus.yml: {e}")
    
    # Check required sections
    assert "global" in config
    assert "scrape_configs" in config
    
    # Check scrape config
    scrape_configs = config["scrape_configs"]
    assert len(scrape_configs) > 0
    
    genesis_config = next(
        (c for c in scrape_configs if c["job_name"] == "genesis"),
        None
    )
    assert genesis_config is not None
    assert genesis_config["metrics_path"] == "/metrics"
    assert genesis_config["scheme"] == "https"


@pytest.mark.asyncio
async def test_alert_rules_valid():
    """Test Prometheus alert rules are valid."""
    import yaml
    from pathlib import Path
    
    alerts_dir = Path("genesis/config/alerts")
    assert alerts_dir.exists(), "Alerts directory not found"
    
    alert_files = ["trading_alerts.yml", "system_alerts.yml"]
    
    for alert_file in alert_files:
        filepath = alerts_dir / alert_file
        assert filepath.exists(), f"Alert file {alert_file} not found"
        
        # Validate YAML
        with open(filepath, "r") as f:
            try:
                alerts = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {alert_file}: {e}")
        
        # Check structure
        assert "groups" in alerts
        for group in alerts["groups"]:
            assert "name" in group
            assert "rules" in group
            
            for rule in group["rules"]:
                assert "alert" in rule
                assert "expr" in rule
                assert "labels" in rule
                assert "annotations" in rule


@pytest.mark.asyncio
async def test_metrics_rate_limiting(full_metrics_stack):
    """Test metrics endpoint rate limiting."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    
    # Make many rapid requests
    responses = []
    for _ in range(150):  # Exceed 100/minute limit
        response = client.get(
            "/metrics",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        responses.append(response.status_code)
        
        # Stop if rate limited
        if response.status_code == 429:
            break
    
    # Should hit rate limit
    assert 429 in responses, "Rate limiting not working"


@pytest.mark.asyncio
async def test_metrics_security_headers(full_metrics_stack):
    """Test security headers on metrics endpoint."""
    client = full_metrics_stack["client"]
    api_key = full_metrics_stack["api_key"]
    
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert response.status_code == 200
    
    # Check security headers
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-xss-protection"] == "1; mode=block"
    assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert response.headers["pragma"] == "no-cache"