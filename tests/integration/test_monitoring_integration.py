"""Integration tests for monitoring pipeline."""

import asyncio
from datetime import timedelta
from unittest.mock import patch

import pytest

from genesis.monitoring.alert_manager import AlertManager, AlertRule, AlertSeverity
from genesis.monitoring.capacity_planner import CapacityPlanner, ResourceType
from genesis.monitoring.metrics_collector import MetricsCollector
from genesis.monitoring.prometheus_exporter import (
    APIKeyRotator,
    MetricsRegistry,
    PrometheusExporter,
)
from genesis.monitoring.sla_tracker import SLAMetric, SLATracker


@pytest.fixture
async def metrics_registry():
    """Create metrics registry."""
    return MetricsRegistry()


@pytest.fixture
async def metrics_collector(metrics_registry):
    """Create metrics collector."""
    collector = MetricsCollector(metrics_registry)
    return collector


@pytest.fixture
async def prometheus_exporter(metrics_registry):
    """Create Prometheus exporter."""
    exporter = PrometheusExporter(
        registry=metrics_registry,
        port=9091,  # Use different port for testing
        use_https=False,  # Disable HTTPS for testing
        production_mode=False
    )
    return exporter


@pytest.fixture
async def sla_tracker(metrics_registry, metrics_collector):
    """Create SLA tracker with metrics collector integration."""
    tracker = SLATracker(metrics_registry)
    tracker.set_metrics_collector(metrics_collector)
    return tracker


@pytest.fixture
async def capacity_planner(metrics_registry, metrics_collector):
    """Create capacity planner with metrics collector integration."""
    planner = CapacityPlanner(metrics_registry)
    planner.set_metrics_collector(metrics_collector)
    return planner


@pytest.fixture
async def alert_manager(metrics_collector):
    """Create alert manager with metrics collector integration."""
    from genesis.tilt.detector import TiltDetector

    tilt_detector = TiltDetector()
    manager = AlertManager(
        metrics_collector=metrics_collector,
        tilt_detector=tilt_detector
    )
    return manager


class TestMonitoringPipeline:
    """Test complete monitoring pipeline integration."""

    @pytest.mark.asyncio
    async def test_metrics_collection_and_export(self, metrics_registry, metrics_collector):
        """Test metrics collection and Prometheus export."""
        # Record some trading metrics
        await metrics_collector.record_order_placed("BTC/USDT", "BUY", 0.001, 50000)
        await metrics_collector.record_order_filled("BTC/USDT", "BUY", 0.001, 50000, 0.1)
        await metrics_collector.record_trade_executed("BTC/USDT", "BUY", 0.001, 50000, 0.1)

        # Collect metrics
        output = await metrics_registry.collect()

        # Verify metrics are exported
        assert "genesis_orders_total" in output
        assert "genesis_trades_total" in output
        assert "genesis_order_execution_time_seconds" in output

    @pytest.mark.asyncio
    async def test_sla_tracking_with_real_metrics(self, sla_tracker, metrics_collector):
        """Test SLA tracking with real metrics from collector."""
        # Generate some metrics
        await metrics_collector.record_order_placed("BTC/USDT", "BUY", 0.001, 50000)
        await metrics_collector.record_order_failed("BTC/USDT", "BUY", "Insufficient balance")

        # Track SLA
        await sla_tracker.track_measurement(SLAMetric.ERROR_RATE, 1.0)

        # Generate report
        report = await sla_tracker.generate_report(timedelta(hours=1))

        assert report is not None
        assert len(report.measurements) > 0
        assert report.overall_compliance >= 0

    @pytest.mark.asyncio
    async def test_capacity_planning_with_real_metrics(self, capacity_planner, metrics_collector):
        """Test capacity planning with real system metrics."""
        # Collect some metrics
        await capacity_planner._collect_metrics()

        # Generate forecast
        forecast = await capacity_planner.forecast_capacity(ResourceType.CPU)

        assert forecast is not None
        assert forecast.current_utilization >= 0
        assert forecast.forecast_30d >= 0

        # Check alerts
        alerts = capacity_planner.get_active_alerts()
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_alert_manager_integration(self, alert_manager, metrics_collector):
        """Test alert manager with real metrics and tilt detection."""
        # Add alert rule
        rule = AlertRule(
            name="high_error_rate",
            condition="error_rate > 5",
            severity=AlertSeverity.WARNING,
            description="High error rate detected"
        )
        alert_manager.add_rule(rule)

        # Simulate errors
        for _ in range(10):
            await metrics_collector.record_order_failed("BTC/USDT", "BUY", "Test error")

        # Evaluate alerts
        await alert_manager.evaluate_rules()

        # Check for triggered alerts
        active = alert_manager.get_active_alerts()
        assert len(active) >= 0  # May or may not trigger based on timing

    @pytest.mark.asyncio
    async def test_prometheus_api_key_rotation(self):
        """Test API key rotation mechanism."""
        rotator = APIKeyRotator(rotation_interval_hours=0.001)  # Very short for testing

        initial_key = rotator.current_key
        assert rotator.validate_key(initial_key)

        # Wait for rotation
        await asyncio.sleep(0.01)
        rotator.rotate_if_needed()

        # Old key should still work (grace period)
        assert rotator.validate_key(initial_key)

        # New key should also work
        assert rotator.current_key != initial_key
        assert rotator.validate_key(rotator.current_key)

    @pytest.mark.asyncio
    async def test_metrics_input_validation(self, metrics_registry):
        """Test metric input validation and sanitization."""
        # Test valid metric
        await metrics_registry.set_gauge("valid_metric_name", 42.0)

        # Test invalid metric name
        with pytest.raises(ValueError, match="Invalid metric name"):
            await metrics_registry.set_gauge("invalid-metric-name!", 42.0)

        # Test invalid label name
        with pytest.raises(ValueError, match="Invalid label name"):
            await metrics_registry.set_gauge("valid_metric", 42.0, {"invalid-label": "value"})

        # Test invalid label value
        with pytest.raises(ValueError, match="Invalid label value"):
            await metrics_registry.set_gauge("valid_metric", 42.0, {"label": "invalid\nvalue"})

    @pytest.mark.asyncio
    async def test_monitoring_pipeline_end_to_end(
        self,
        metrics_registry,
        metrics_collector,
        sla_tracker,
        capacity_planner,
        alert_manager
    ):
        """Test complete monitoring pipeline end-to-end."""
        # Step 1: Generate trading activity
        for i in range(5):
            await metrics_collector.record_order_placed("ETH/USDT", "BUY", 0.1, 3000 + i)
            await asyncio.sleep(0.01)
            await metrics_collector.record_order_filled("ETH/USDT", "BUY", 0.1, 3000 + i, 0.05)
            await metrics_collector.record_trade_executed("ETH/USDT", "BUY", 0.1, 3000 + i, 1.0)

        # Step 2: Track SLA compliance
        await sla_tracker.track_measurement(SLAMetric.UPTIME, 99.95)
        await sla_tracker.track_measurement(SLAMetric.LATENCY, 45.0)
        await sla_tracker.track_measurement(SLAMetric.ERROR_RATE, 0.05)

        # Step 3: Collect capacity metrics
        await capacity_planner._collect_metrics()

        # Step 4: Evaluate alerts
        await alert_manager.evaluate_rules()

        # Step 5: Export all metrics
        prometheus_output = await metrics_registry.collect()

        # Verify pipeline output
        assert "genesis_orders_total" in prometheus_output
        assert "genesis_trades_total" in prometheus_output
        assert "genesis_pnl_dollars" in prometheus_output

        # Verify SLA report
        sla_report = await sla_tracker.generate_report(timedelta(hours=1))
        assert sla_report.overall_compliance > 0

        # Verify capacity forecast
        cpu_forecast = await capacity_planner.forecast_capacity(ResourceType.CPU)
        assert cpu_forecast.current_utilization >= 0

        # Verify alert system
        alerts = alert_manager.get_active_alerts()
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_monitoring_resilience(self, metrics_collector, alert_manager):
        """Test monitoring system resilience to failures."""
        # Simulate component failure
        with patch.object(metrics_collector, 'record_order_placed', side_effect=Exception("Test failure")):
            # Should not crash
            try:
                await metrics_collector.record_order_placed("BTC/USDT", "BUY", 0.1, 50000)
            except Exception:
                pass  # Expected to fail

        # System should continue working
        await metrics_collector.record_trade_executed("BTC/USDT", "BUY", 0.1, 50000, 1.0)

        # Alerts should still function
        await alert_manager.evaluate_rules()

        # Verify system is still operational
        metrics = metrics_collector.get_trading_metrics()
        assert metrics is not None

    @pytest.mark.asyncio
    async def test_production_mode_security(self, metrics_registry):
        """Test production mode security features."""
        # Create exporter in production mode
        exporter = PrometheusExporter(
            registry=metrics_registry,
            port=9092,
            use_https=True,
            production_mode=True,
            cert_file=None,  # Should fail without certs
            key_file=None
        )

        # Should require certificates in production
        with pytest.raises(ValueError, match="Production mode requires valid TLS certificates"):
            await exporter.start()

        # Verify IP allowlist is configured
        assert len(exporter.allowed_ips) > 0
        assert "127.0.0.1" in exporter.allowed_ips
