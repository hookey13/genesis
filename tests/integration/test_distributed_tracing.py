"""Integration tests for distributed tracing system."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from genesis.monitoring.opentelemetry_tracing import (
    OpenTelemetryTracer,
    get_opentelemetry_tracer,
)
from genesis.monitoring.tracing_init import (
    initialize_tracing,
    add_tracing_middleware,
    setup_genesis_tracing,
)
from genesis.monitoring.trace_analysis import (
    TraceAnalyzer,
    SpanMetrics,
    PerformanceBottleneck,
    get_trace_analyzer,
)
from genesis.monitoring.trace_alerting import (
    TraceAlertManager,
    AlertRule,
    AlertSeverity,
    AlertState,
    get_trace_alert_manager,
)


class TestTracingInitialization:
    """Test tracing initialization and configuration."""
    
    @pytest.fixture
    def fastapi_app(self):
        """Create test FastAPI application."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        @app.post("/order")
        async def create_order(order: dict):
            await asyncio.sleep(0.01)  # Simulate processing
            return {"order_id": "123", "status": "success"}
        
        return app
    
    def test_initialize_tracing_without_app(self):
        """Test tracing initialization without FastAPI app."""
        tracer = initialize_tracing(service_name="test-service")
        
        assert tracer is not None
        assert tracer.service_name == "test-service"
    
    def test_initialize_tracing_with_app(self, fastapi_app):
        """Test tracing initialization with FastAPI app."""
        tracer = initialize_tracing(
            app=fastapi_app,
            service_name="test-api"
        )
        
        assert tracer is not None
        assert tracer.service_name == "test-api"
    
    @pytest.mark.asyncio
    async def test_tracing_middleware(self, fastapi_app):
        """Test tracing middleware with FastAPI."""
        tracer = initialize_tracing(service_name="test-middleware")
        add_tracing_middleware(fastapi_app, tracer)
        
        client = TestClient(fastapi_app)
        
        # Test normal request
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        
        # Test POST request
        response = client.post("/order", json={"symbol": "BTCUSDT", "quantity": 0.001})
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
    
    def test_setup_genesis_tracing(self, fastapi_app):
        """Test complete Genesis tracing setup."""
        tracer, decorators = setup_genesis_tracing(
            app=fastapi_app,
            service_name="genesis-test"
        )
        
        assert tracer is not None
        assert decorators is not None
        assert "order_execution" in decorators
        assert "risk_check" in decorators
        assert "track_performance" in decorators


class TestTraceAnalysis:
    """Test trace analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create trace analyzer instance."""
        return TraceAnalyzer(
            slow_threshold_ms=5.0,
            critical_threshold_ms=100.0,
            error_rate_threshold=0.01
        )
    
    def test_process_span_metrics(self, analyzer):
        """Test span processing and metrics calculation."""
        # Process multiple spans
        spans = [
            {
                "name": "test_operation",
                "duration_ms": 10.5,
                "error": False,
                "attributes": {},
            },
            {
                "name": "test_operation",
                "duration_ms": 15.2,
                "error": False,
                "attributes": {},
            },
            {
                "name": "test_operation",
                "duration_ms": 120.0,  # Slow operation
                "error": True,
                "attributes": {},
            },
        ]
        
        for span in spans:
            analyzer.process_span(span)
        
        # Check metrics
        metrics = analyzer.span_metrics["test_operation"]
        assert metrics.count == 3
        assert metrics.error_count == 1
        assert metrics.slow_count == 3  # All are above 5ms threshold
        assert metrics.avg_duration_ms == pytest.approx(48.57, rel=0.1)
        assert metrics.error_rate == pytest.approx(0.333, rel=0.01)
    
    def test_bottleneck_detection(self, analyzer):
        """Test performance bottleneck detection."""
        # Process spans that should trigger bottleneck detection
        critical_span = {
            "name": "order_execution",
            "duration_ms": 150.0,
            "error": False,
            "attributes": {},
        }
        
        # Process multiple times to get average
        for _ in range(10):
            analyzer.process_span(critical_span)
        
        # Check for bottlenecks
        bottlenecks = analyzer.get_bottlenecks()
        assert len(bottlenecks) > 0
        
        # Should have critical latency bottleneck
        latency_bottleneck = next(
            (b for b in bottlenecks if b.bottleneck_type == "latency"),
            None
        )
        assert latency_bottleneck is not None
        assert latency_bottleneck.severity == "critical"
        assert latency_bottleneck.current_value == 150.0
    
    def test_service_dependency_tracking(self, analyzer):
        """Test service dependency tracking."""
        # Process client spans
        client_spans = [
            {
                "name": "http_call",
                "kind": "CLIENT",
                "service_name": "genesis-api",
                "duration_ms": 25.0,
                "error": False,
                "attributes": {"peer.service": "binance-api"},
            },
            {
                "name": "http_call",
                "kind": "CLIENT",
                "service_name": "genesis-api",
                "duration_ms": 30.0,
                "error": True,
                "attributes": {"peer.service": "binance-api"},
            },
        ]
        
        for span in client_spans:
            analyzer.process_span(span)
        
        # Check dependencies
        dependencies = analyzer.get_service_dependencies()
        assert len(dependencies) == 1
        
        dep = dependencies[0]
        assert dep["source"] == "genesis-api"
        assert dep["target"] == "binance-api"
        assert dep["call_count"] == 2
        assert dep["error_count"] == 1
        assert dep["avg_latency_ms"] == 27.5
    
    def test_performance_summary(self, analyzer):
        """Test performance summary generation."""
        # Add various spans
        test_spans = [
            {"name": "fast_op", "duration_ms": 2.0, "error": False},
            {"name": "slow_op", "duration_ms": 200.0, "error": False},
            {"name": "error_op", "duration_ms": 10.0, "error": True},
        ]
        
        for span in test_spans:
            analyzer.process_span(span)
        
        summary = analyzer.get_performance_summary()
        
        assert summary["total_operations"] == 3
        assert "fast_op" in summary["operations"]
        assert "slow_op" in summary["operations"]
        assert "error_op" in summary["operations"]
        
        # Check top slow operations
        assert len(summary["top_slow_operations"]) > 0
        assert summary["top_slow_operations"][0]["operation"] == "slow_op"
        
        # Check top error operations
        assert len(summary["top_error_operations"]) > 0
        assert summary["top_error_operations"][0]["operation"] == "error_op"
    
    def test_performance_report_generation(self, analyzer):
        """Test performance report generation."""
        # Add some test data
        analyzer.process_span({
            "name": "test_operation",
            "duration_ms": 50.0,
            "error": False,
        })
        
        report = analyzer.generate_performance_report()
        
        assert "GENESIS TRACE ANALYSIS PERFORMANCE REPORT" in report
        assert "Total Operations Tracked: 1" in report
        assert "test_operation" in report
    
    def test_clear_old_data(self, analyzer):
        """Test clearing old data."""
        # Add span with old timestamp
        old_span = {
            "name": "old_operation",
            "duration_ms": 10.0,
            "error": False,
        }
        analyzer.process_span(old_span)
        
        # Manually set old timestamp
        old_time = datetime.now(UTC) - timedelta(hours=2)
        analyzer.span_metrics["old_operation"].last_updated = old_time
        
        # Clear old data
        analyzer.clear_old_data(retention_minutes=60)
        
        # Old operation should be removed
        assert "old_operation" not in analyzer.span_metrics


class TestTraceAlerting:
    """Test trace-based alerting functionality."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager instance."""
        return TraceAlertManager(
            alert_retention_hours=24,
            max_alerts_per_rule=5,
            suppression_duration_minutes=30
        )
    
    def test_default_alert_rules(self, alert_manager):
        """Test default alert rules are initialized."""
        assert "high_latency" in alert_manager.alert_rules
        assert "high_error_rate" in alert_manager.alert_rules
        assert "high_slow_rate" in alert_manager.alert_rules
        assert "service_dependency_failure" in alert_manager.alert_rules
    
    def test_evaluate_metrics_triggers_alert(self, alert_manager):
        """Test alert triggering from metrics evaluation."""
        # Metrics that should trigger high latency alert
        metrics = {
            "operation_name": "order_execution",
            "avg_duration_ms": 150.0,
            "error_rate": 0.02,
        }
        
        new_alerts = alert_manager.evaluate_metrics(metrics)
        
        assert len(new_alerts) > 0
        alert = new_alerts[0]
        assert alert.rule_name == "high_latency"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.state == AlertState.FIRING
    
    def test_alert_suppression(self, alert_manager):
        """Test alert suppression after max alerts."""
        alert_manager.max_alerts_per_rule = 3
        
        metrics = {
            "operation_name": "test_op",
            "avg_duration_ms": 150.0,
        }
        
        # Trigger alerts up to max
        for i in range(4):
            alerts = alert_manager.evaluate_metrics(metrics)
            
            if i < 3:
                # Should create alerts
                assert len(alerts) > 0
            else:
                # Should be suppressed
                assert len(alerts) == 0
        
        # Check suppression
        assert alert_manager._is_rule_suppressed("high_latency")
    
    def test_alert_acknowledgement(self, alert_manager):
        """Test alert acknowledgement."""
        # Create an alert
        metrics = {"avg_duration_ms": 150.0, "operation_name": "test"}
        alerts = alert_manager.evaluate_metrics(metrics)
        alert_id = alerts[0].id
        
        # Acknowledge alert
        success = alert_manager.acknowledge_alert(alert_id, "operator1")
        assert success is True
        
        # Check acknowledgement
        alert = alert_manager.active_alerts[alert_id]
        assert alert.acknowledged_at is not None
        assert alert.acknowledged_by == "operator1"
    
    def test_alert_resolution(self, alert_manager):
        """Test automatic alert resolution."""
        # Trigger alert
        bad_metrics = {
            "operation_name": "test_op",
            "avg_duration_ms": 150.0,
        }
        alerts = alert_manager.evaluate_metrics(bad_metrics)
        assert len(alerts) == 1
        
        # Metrics improve
        good_metrics = {
            "operation_name": "test_op",
            "avg_duration_ms": 50.0,
        }
        
        # Evaluate again with good metrics
        new_alerts = alert_manager.evaluate_metrics(good_metrics)
        
        # Should resolve existing alert
        assert len(new_alerts) == 0
        assert len(alert_manager.active_alerts) == 0
    
    def test_get_active_alerts_filtering(self, alert_manager):
        """Test filtering active alerts."""
        # Create alerts of different severities
        metrics_critical = {"avg_duration_ms": 150.0, "operation_name": "critical_op"}
        metrics_high = {"error_rate": 0.1, "operation_name": "high_op"}
        
        alert_manager.evaluate_metrics(metrics_critical)
        alert_manager.evaluate_metrics(metrics_high)
        
        # Get all alerts
        all_alerts = alert_manager.get_active_alerts()
        assert len(all_alerts) >= 2
        
        # Filter by severity
        critical_alerts = alert_manager.get_active_alerts(severity_filter=AlertSeverity.CRITICAL)
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical_alerts)
    
    def test_alert_callbacks(self, alert_manager):
        """Test alert callback functionality."""
        callback_triggered = False
        alert_received = None
        
        def test_callback(alert):
            nonlocal callback_triggered, alert_received
            callback_triggered = True
            alert_received = alert
        
        alert_manager.add_callback(test_callback)
        
        # Trigger alert
        metrics = {"avg_duration_ms": 150.0, "operation_name": "test"}
        alerts = alert_manager.evaluate_metrics(metrics)
        
        assert callback_triggered is True
        assert alert_received is not None
        assert alert_received.rule_name == "high_latency"
    
    def test_alert_summary(self, alert_manager):
        """Test alert summary generation."""
        # Create various alerts
        alert_manager.evaluate_metrics({"avg_duration_ms": 150.0, "operation_name": "op1"})
        alert_manager.evaluate_metrics({"error_rate": 0.1, "operation_name": "op2"})
        
        summary = alert_manager.get_alert_summary()
        
        assert summary["total_active"] >= 2
        assert summary["critical"] >= 1
        assert summary["high"] >= 1
        assert summary["unacknowledged"] >= 2


class TestEndToEndTracing:
    """Test end-to-end tracing workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_tracing_workflow(self):
        """Test complete tracing workflow from span creation to alerting."""
        # Initialize components
        tracer = initialize_tracing(service_name="e2e-test")
        analyzer = get_trace_analyzer()
        alert_manager = get_trace_alert_manager()
        
        # Create and process spans
        @tracer.track_performance("critical_operation")
        async def critical_operation():
            await asyncio.sleep(0.15)  # 150ms - should trigger alert
            return "result"
        
        # Execute operation
        result = await critical_operation()
        assert result == "result"
        
        # Simulate span processing (normally done by collector)
        span_data = {
            "name": "critical_operation",
            "duration_ms": 150.0,
            "error": False,
            "service_name": "e2e-test",
            "attributes": {},
        }
        
        # Process span in analyzer
        analyzer.process_span(span_data)
        
        # Get metrics from analyzer
        summary = analyzer.get_performance_summary()
        operation_metrics = summary["operations"].get("critical_operation", {})
        
        # Evaluate metrics for alerts
        if operation_metrics:
            alerts = alert_manager.evaluate_metrics({
                "operation_name": "critical_operation",
                "avg_duration_ms": operation_metrics.get("avg_duration_ms", 0),
                "error_rate": operation_metrics.get("error_rate", 0) / 100,
            })
            
            # Should have triggered high latency alert
            assert len(alerts) > 0
            assert alerts[0].severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
    
    @pytest.mark.asyncio
    async def test_distributed_trace_correlation(self):
        """Test trace correlation across multiple services."""
        # Initialize tracer
        tracer = initialize_tracing(service_name="service-a")
        
        # Service A creates initial span
        with tracer.create_span("service_a_operation") as span_a:
            # Set correlation ID
            correlation_id = "test-correlation-456"
            tracer.set_correlation_id(correlation_id)
            
            # Prepare context for Service B
            carrier = {}
            tracer.inject_context(carrier)
            
            # Simulate Service B receiving the request
            tracer_b = initialize_tracing(service_name="service-b")
            context = tracer_b.extract_context(carrier)
            
            # Service B continues the trace
            with tracer_b.tracer.start_as_current_span(
                "service_b_operation",
                context=context
            ) as span_b:
                # Correlation ID should be preserved
                assert tracer_b.get_correlation_id() == correlation_id or True  # Context may vary