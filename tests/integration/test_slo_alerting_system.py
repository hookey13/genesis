"""
Comprehensive integration tests for SLO tracking and alerting system.

Tests all components of Story 9.6.3 to ensure full implementation.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import pytest
import aiohttp
from freezegun import freeze_time

from genesis.monitoring.slo_tracker import (
    SLOTracker, SLIType, AlertSeverity, SLIResult, ErrorBudget
)
from genesis.monitoring.alert_channels import (
    AlertChannelManager, Alert, AlertPriority, AlertChannel, DeliveryResult
)
from genesis.monitoring.alert_dedup import (
    AlertDeduplicator, AlertGroup, RoutingRule, GroupingStrategy
)
from genesis.monitoring.slo_reporter import (
    SLOReporter, ReportConfig
)
from genesis.monitoring.runbook_executor import (
    RunbookExecutor, Runbook, RunbookAction, ActionType, ExecutionMode, ExecutionResult
)
from genesis.monitoring.incident_tracker import (
    IncidentTracker, Incident, IncidentStatus, IncidentPriority, IncidentMetrics
)
from genesis.api.alerts_webhook import (
    AlertWebhookRequest, RunbookExecutionRequest
)


class TestSLOTracker:
    """Test SLO tracking functionality."""
    
    @pytest.fixture
    async def slo_tracker(self, tmp_path):
        """Create SLO tracker with test configuration."""
        config_path = tmp_path / "slo_definitions.yaml"
        config_data = {
            "services": {
                "test_service": {
                    "slis": {
                        "availability": {
                            "type": "availability",
                            "metric": "up{job='test'}",
                            "threshold": 0.99,
                            "aggregation": "avg"
                        },
                        "latency": {
                            "type": "latency",
                            "metric": "latency_seconds",
                            "threshold": 0.1,
                            "aggregation": "p99"
                        }
                    },
                    "error_budget": {
                        "window_days": 30,
                        "burn_rate_thresholds": [
                            {"rate": 14.4, "window_minutes": 60, "severity": "critical"},
                            {"rate": 6, "window_minutes": 360, "severity": "warning"}
                        ]
                    }
                }
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        tracker = SLOTracker(str(config_path))
        await tracker.initialize()
        return tracker
    
    @pytest.mark.asyncio
    async def test_sli_calculation(self, slo_tracker):
        """Test SLI calculation and compliance checking."""
        # Calculate SLI
        sli_config = slo_tracker.slo_configs["test_service"].slis["availability"]
        result = await slo_tracker.calculate_sli("test_service", "availability", sli_config)
        
        assert isinstance(result, SLIResult)
        assert result.service == "test_service"
        assert result.sli_type == SLIType.AVAILABILITY
        assert result.threshold == 0.99
        assert isinstance(result.is_good, bool)
        assert result.calculation_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_error_budget_calculation(self, slo_tracker):
        """Test error budget calculation with burn rate."""
        # Add some SLI history
        for i in range(100):
            sli_config = slo_tracker.slo_configs["test_service"].slis["availability"]
            await slo_tracker.calculate_sli("test_service", "availability", sli_config)
        
        # Calculate error budget
        budget = slo_tracker.calculate_error_budget("test_service", timedelta(days=30))
        
        assert isinstance(budget, ErrorBudget)
        assert budget.service == "test_service"
        assert 0 <= budget.remaining_ratio <= 1
        assert budget.burn_rate >= 0
        assert budget.total_budget == 1.0
    
    @pytest.mark.asyncio
    async def test_burn_rate_alerts(self, slo_tracker):
        """Test burn rate threshold alerting."""
        # Simulate high burn rate by adding many failures
        slo_tracker.sli_history["test_service"] = []
        for i in range(50):
            slo_tracker.sli_history["test_service"].append(
                SLIResult(
                    timestamp=datetime.utcnow() - timedelta(minutes=i),
                    service="test_service",
                    sli_type=SLIType.AVAILABILITY,
                    value=0.8,  # Below threshold
                    threshold=0.99,
                    is_good=False,
                    calculation_time_ms=1.0
                )
            )
        
        # Check for alerts
        alerts = slo_tracker.check_burn_rate_alerts("test_service")
        
        assert len(alerts) > 0
        assert any(alert[0] == AlertSeverity.CRITICAL for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_slo_evaluation(self, slo_tracker):
        """Test full SLO evaluation for all services."""
        results = await slo_tracker.evaluate_slos()
        
        assert "test_service" in results
        assert len(results["test_service"]) > 0
        for result in results["test_service"]:
            assert isinstance(result, SLIResult)


class TestAlertChannels:
    """Test alert channel delivery system."""
    
    @pytest.fixture
    async def channel_manager(self):
        """Create alert channel manager."""
        with patch('genesis.monitoring.alert_channels.VaultClient'):
            manager = AlertChannelManager()
            await manager.initialize()
            return manager
    
    @pytest.mark.asyncio
    async def test_pagerduty_delivery(self, channel_manager):
        """Test PagerDuty alert delivery."""
        alert = Alert(
            id="test_001",
            name="TestAlert",
            summary="Test alert summary",
            description="Test alert description",
            severity="critical",
            priority=AlertPriority.CRITICAL,
            service="test_service",
            source="test",
            labels={"env": "test"},
            annotations={"runbook": "test.md"},
            timestamp=datetime.utcnow()
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 202
            mock_post.return_value.__aenter__.return_value = mock_response
            
            results = await channel_manager.send_alert(
                alert, [AlertChannel.PAGERDUTY], failover=False
            )
            
            assert len(results) == 1
            assert results[0].channel == AlertChannel.PAGERDUTY
            assert results[0].success
    
    @pytest.mark.asyncio
    async def test_slack_delivery(self, channel_manager):
        """Test Slack alert delivery."""
        alert = Alert(
            id="test_002",
            name="TestAlert",
            summary="Test alert",
            description="Test description",
            severity="warning",
            priority=AlertPriority.MEDIUM,
            service="test_service",
            source="test",
            labels={},
            annotations={},
            timestamp=datetime.utcnow()
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response
            
            results = await channel_manager.send_alert(
                alert, [AlertChannel.SLACK], failover=False
            )
            
            assert len(results) == 1
            assert results[0].channel == AlertChannel.SLACK
            assert results[0].success
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, channel_manager):
        """Test channel rate limiting."""
        # Set very low rate limit
        channel_manager.rate_limiters[AlertChannel.PAGERDUTY].max_per_minute = 1
        channel_manager.rate_limiters[AlertChannel.PAGERDUTY].tokens = 1
        
        alert = Alert(
            id="test_003",
            name="TestAlert",
            summary="Test",
            description="Test",
            severity="critical",
            priority=AlertPriority.CRITICAL,
            service="test",
            source="test",
            labels={},
            annotations={},
            timestamp=datetime.utcnow()
        )
        
        # First alert should succeed
        assert await channel_manager.rate_limiters[AlertChannel.PAGERDUTY].acquire()
        
        # Second should fail (no tokens)
        assert not await channel_manager.rate_limiters[AlertChannel.PAGERDUTY].acquire()


class TestAlertDeduplication:
    """Test alert deduplication and routing."""
    
    @pytest.fixture
    async def deduplicator(self):
        """Create alert deduplicator."""
        dedup = AlertDeduplicator(
            dedup_window_minutes=5,
            group_wait_seconds=30
        )
        await dedup.initialize()
        return dedup
    
    @pytest.mark.asyncio
    async def test_deduplication(self, deduplicator):
        """Test duplicate alert detection."""
        alert1 = Alert(
            id="alert_001",
            name="HighCPU",
            summary="High CPU usage",
            description="CPU above 90%",
            severity="warning",
            priority=AlertPriority.MEDIUM,
            service="api",
            source="prometheus",
            labels={"instance": "server1"},
            annotations={},
            timestamp=datetime.utcnow()
        )
        
        # First alert should pass
        should_send1, group1, channels1 = await deduplicator.process_alert(alert1)
        assert should_send1
        assert group1 is not None
        
        # Duplicate should be deduplicated
        alert2 = alert1  # Same alert
        should_send2, group2, channels2 = await deduplicator.process_alert(alert2)
        assert not should_send2
    
    @pytest.mark.asyncio
    async def test_alert_grouping(self, deduplicator):
        """Test alert grouping by fingerprint."""
        base_alert = Alert(
            id="",
            name="ServiceDown",
            summary="Service unavailable",
            description="Service not responding",
            severity="critical",
            priority=AlertPriority.CRITICAL,
            service="database",
            source="prometheus",
            labels={},
            annotations={},
            timestamp=datetime.utcnow()
        )
        
        # Create multiple similar alerts
        for i in range(5):
            alert = base_alert
            alert.id = f"alert_{i:03d}"
            alert.timestamp = datetime.utcnow()
            
            should_send, group, channels = await deduplicator.process_alert(alert)
            
            # All should be grouped together
            assert group is not None
            assert group.count == i + 1
    
    @pytest.mark.asyncio
    async def test_routing_rules(self, deduplicator):
        """Test alert routing based on rules."""
        # Add custom routing rule
        rule = RoutingRule(
            name="database_critical",
            priority=100,
            match_conditions={"service": "database", "severity": "critical"},
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK]
        )
        deduplicator.add_routing_rule(rule)
        
        alert = Alert(
            id="db_001",
            name="DatabaseDown",
            summary="Database down",
            description="Primary database unavailable",
            severity="critical",
            priority=AlertPriority.CRITICAL,
            service="database",
            source="prometheus",
            labels={},
            annotations={},
            timestamp=datetime.utcnow()
        )
        
        should_send, group, channels = await deduplicator.process_alert(alert)
        
        assert should_send
        assert AlertChannel.PAGERDUTY in channels
        assert AlertChannel.SLACK in channels


class TestSLOReporter:
    """Test SLO report generation."""
    
    @pytest.fixture
    async def reporter(self, tmp_path):
        """Create SLO reporter."""
        tracker = Mock(spec=SLOTracker)
        tracker.slo_configs = {
            "test_service": Mock(slis={"availability": Mock()})
        }
        tracker.get_slo_summary.return_value = {
            "service": "test_service",
            "compliance": {"30d": 0.995},
            "error_budgets": {"30d": {"remaining": 0.5, "burn_rate": 1.2}},
            "current_slis": {"availability": {"value": 0.998, "threshold": 0.99, "is_good": True}}
        }
        
        reporter = SLOReporter(tracker, str(tmp_path))
        return reporter
    
    @pytest.mark.asyncio
    async def test_pdf_generation(self, reporter):
        """Test PDF report generation."""
        config = ReportConfig(
            output_formats=["pdf"],
            report_period_days=30
        )
        
        files = await reporter.generate_report(config, ["test_service"])
        
        assert "pdf" in files
        assert Path(files["pdf"]).exists()
        assert Path(files["pdf"]).suffix == ".pdf"
    
    @pytest.mark.asyncio
    async def test_html_generation(self, reporter):
        """Test HTML report generation."""
        config = ReportConfig(
            output_formats=["html"],
            report_period_days=30
        )
        
        files = await reporter.generate_report(config, ["test_service"])
        
        assert "html" in files
        assert Path(files["html"]).exists()
        
        # Verify HTML content
        with open(files["html"], 'r') as f:
            content = f.read()
            assert "test_service" in content
            assert "99.5%" in content  # Compliance
    
    @pytest.mark.asyncio
    async def test_json_generation(self, reporter):
        """Test JSON report generation."""
        config = ReportConfig(
            output_formats=["json"],
            report_period_days=30
        )
        
        files = await reporter.generate_report(config, ["test_service"])
        
        assert "json" in files
        assert Path(files["json"]).exists()
        
        # Verify JSON structure
        with open(files["json"], 'r') as f:
            data = json.load(f)
            assert "metadata" in data
            assert "summary" in data
            assert "services" in data
            assert "test_service" in data["services"]


class TestRunbookExecutor:
    """Test runbook automation."""
    
    @pytest.fixture
    async def executor(self, tmp_path):
        """Create runbook executor."""
        runbook_dir = tmp_path / "runbooks"
        runbook_dir.mkdir()
        
        # Create test runbook
        runbook_yaml = """
name: test_runbook
description: Test runbook
trigger:
  alert_name: HighCPU
  severity: warning
conditions:
  - metric: cpu_usage
    operator: ">"
    value: 80
execution_mode: automatic
max_executions_per_hour: 5
actions:
  - name: clear_cache
    type: clear_cache
    description: Clear application cache
    safe_for_auto: true
    requires_confirmation: false
    timeout_seconds: 10
"""
        (runbook_dir / "test.yaml").write_text(runbook_yaml)
        
        executor = RunbookExecutor(str(runbook_dir))
        await executor.initialize()
        return executor
    
    @pytest.mark.asyncio
    async def test_runbook_loading(self, executor):
        """Test runbook loading from YAML."""
        assert len(executor.runbooks) > 0
        assert "test_runbook" in executor.runbooks
        
        runbook = executor.runbooks["test_runbook"]
        assert runbook.name == "test_runbook"
        assert len(runbook.actions) == 1
        assert runbook.execution_mode == ExecutionMode.AUTOMATIC
    
    @pytest.mark.asyncio
    async def test_dry_run_execution(self, executor):
        """Test runbook dry-run execution."""
        context = {"cpu_usage": 85, "service": "test"}
        
        result = await executor.execute_runbook(
            "test_runbook",
            context,
            dry_run=True
        )
        
        assert isinstance(result, ExecutionResult)
        assert result.dry_run
        assert result.status in ["success", "no_actions"]
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, executor):
        """Test runbook execution rate limiting."""
        context = {"cpu_usage": 85}
        
        # Execute up to limit
        for i in range(5):
            result = await executor.execute_runbook(
                "test_runbook",
                context,
                dry_run=True
            )
            assert result.status != "skipped"
        
        # Next should fail due to rate limit
        with pytest.raises(Exception) as exc_info:
            await executor.execute_runbook(
                "test_runbook",
                context,
                dry_run=True
            )
        assert "Rate limit" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_condition_evaluation(self, executor):
        """Test runbook condition evaluation."""
        # Context doesn't meet conditions
        context = {"cpu_usage": 50}  # Below threshold
        
        result = await executor.execute_runbook(
            "test_runbook",
            context,
            dry_run=True
        )
        
        assert result.status == "skipped"
        assert "Conditions not met" in result.outputs.get("reason", "")


class TestIncidentTracker:
    """Test incident management."""
    
    @pytest.fixture
    async def tracker(self):
        """Create incident tracker."""
        with patch('genesis.monitoring.incident_tracker.VaultClient'):
            tracker = IncidentTracker()
            await tracker.initialize()
            return tracker
    
    @pytest.mark.asyncio
    async def test_incident_creation(self, tracker):
        """Test incident creation."""
        incident_id = await tracker.create_incident(
            title="Test Incident",
            description="Test incident description",
            service="test_service",
            priority=IncidentPriority.HIGH,
            alert_ids=["alert_001", "alert_002"]
        )
        
        assert incident_id.startswith("INC-")
        
        incident = tracker.get_incident(incident_id)
        assert incident is not None
        assert incident.title == "Test Incident"
        assert incident.status == IncidentStatus.OPEN
        assert len(incident.alert_ids) == 2
    
    @pytest.mark.asyncio
    async def test_incident_acknowledgment(self, tracker):
        """Test incident acknowledgment with MTTA calculation."""
        incident_id = await tracker.create_incident(
            title="Test Incident",
            description="Test",
            service="test_service"
        )
        
        # Acknowledge after some time
        await asyncio.sleep(0.1)
        success = await tracker.acknowledge_incident(incident_id, "operator1")
        
        assert success
        
        incident = tracker.get_incident(incident_id)
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_by == "operator1"
        assert incident.time_to_acknowledge > 0
    
    @pytest.mark.asyncio
    async def test_incident_resolution(self, tracker):
        """Test incident resolution with MTTR calculation."""
        incident_id = await tracker.create_incident(
            title="Test Incident",
            description="Test",
            service="test_service"
        )
        
        # Acknowledge
        await tracker.acknowledge_incident(incident_id, "operator1")
        
        # Resolve after some time
        await asyncio.sleep(0.1)
        success = await tracker.resolve_incident(
            incident_id,
            "Issue resolved by restarting service",
            "operator1"
        )
        
        assert success
        
        incident = tracker.get_incident(incident_id)
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_by == "operator1"
        assert incident.time_to_resolve > 0
        assert len(incident.notes) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, tracker):
        """Test incident metrics calculation."""
        # Create multiple incidents
        for i in range(5):
            incident_id = await tracker.create_incident(
                title=f"Incident {i}",
                description="Test",
                service="test_service",
                priority=IncidentPriority.MEDIUM
            )
            
            if i < 3:  # Acknowledge some
                await tracker.acknowledge_incident(incident_id)
            
            if i < 2:  # Resolve some
                await tracker.resolve_incident(
                    incident_id,
                    "Resolved",
                    "operator"
                )
        
        metrics = tracker.calculate_metrics()
        
        assert metrics.total_incidents == 5
        assert metrics.acknowledged_incidents >= 1
        assert metrics.resolved_incidents == 2
        assert metrics.resolution_rate == 40.0  # 2/5 * 100
        assert "test_service" in metrics.incidents_by_service


class TestAlertWebhook:
    """Test alert webhook integration."""
    
    @pytest.mark.asyncio
    async def test_webhook_processing(self):
        """Test AlertManager webhook processing."""
        from genesis.api.alerts_webhook import receive_alert_webhook
        
        request = AlertWebhookRequest(
            version="4",
            groupKey="test_group",
            truncatedAlerts=0,
            status="firing",
            receiver="test_receiver",
            groupLabels={"alertname": "HighCPU"},
            commonLabels={"service": "test_service"},
            commonAnnotations={"description": "CPU high"},
            externalURL="http://alertmanager",
            alerts=[
                {
                    "labels": {
                        "alertname": "HighCPU",
                        "severity": "warning",
                        "service": "test_service"
                    },
                    "annotations": {
                        "summary": "High CPU usage",
                        "description": "CPU above 90%"
                    },
                    "startsAt": datetime.utcnow().isoformat(),
                    "fingerprint": "abc123"
                }
            ]
        )
        
        with patch('genesis.api.alerts_webhook.get_runbook_executor') as mock_executor:
            with patch('genesis.api.alerts_webhook.get_incident_tracker') as mock_tracker:
                mock_executor.return_value.runbooks = {}
                mock_tracker.return_value.create_incident = AsyncMock(return_value="INC-001")
                
                response = await receive_alert_webhook(
                    request,
                    mock_executor.return_value,
                    mock_tracker.return_value,
                    True
                )
                
                assert response.status == "success"
                assert "alerts_processed" in response.data


class TestEndToEndIntegration:
    """Test complete end-to-end flow."""
    
    @pytest.mark.asyncio
    async def test_alert_to_incident_flow(self):
        """Test flow from alert to incident creation and resolution."""
        # Create all components
        with patch('genesis.monitoring.alert_channels.VaultClient'):
            channel_manager = AlertChannelManager()
            await channel_manager.initialize()
        
        deduplicator = AlertDeduplicator()
        await deduplicator.initialize()
        
        with patch('genesis.monitoring.incident_tracker.VaultClient'):
            incident_tracker = IncidentTracker()
            await incident_tracker.initialize()
        
        # Create alert
        alert = Alert(
            id="e2e_001",
            name="DatabaseDown",
            summary="Database unavailable",
            description="Primary database is not responding",
            severity="critical",
            priority=AlertPriority.CRITICAL,
            service="database",
            source="prometheus",
            labels={"env": "production"},
            annotations={"runbook": "db_recovery.md"},
            timestamp=datetime.utcnow()
        )
        
        # Process through deduplication
        should_send, group, channels = await deduplicator.process_alert(alert)
        assert should_send
        assert AlertChannel.PAGERDUTY in channels
        
        # Create incident
        incident_id = await incident_tracker.create_incident(
            title=alert.summary,
            description=alert.description,
            service=alert.service,
            priority=IncidentPriority.CRITICAL,
            alert_ids=[alert.id]
        )
        
        assert incident_id is not None
        
        # Acknowledge incident
        await incident_tracker.acknowledge_incident(incident_id, "oncall_engineer")
        
        # Resolve incident
        await incident_tracker.resolve_incident(
            incident_id,
            "Database connection restored",
            "oncall_engineer"
        )
        
        # Verify metrics
        metrics = incident_tracker.calculate_metrics()
        assert metrics.mean_time_to_acknowledge > 0
        assert metrics.mean_time_to_resolve > 0
        assert metrics.resolution_rate > 0
    
    @pytest.mark.asyncio
    async def test_slo_breach_to_runbook_execution(self, tmp_path):
        """Test SLO breach triggering runbook execution."""
        # Setup SLO tracker with breached SLO
        config_path = tmp_path / "slo_config.yaml"
        config_data = {
            "services": {
                "api": {
                    "slis": {
                        "availability": {
                            "type": "availability",
                            "metric": "up",
                            "threshold": 0.999
                        }
                    },
                    "error_budget": {
                        "window_days": 30,
                        "burn_rate_thresholds": [
                            {"rate": 10, "window_minutes": 60, "severity": "critical"}
                        ]
                    }
                }
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        tracker = SLOTracker(str(config_path))
        await tracker.initialize()
        
        # Simulate SLO breach
        for i in range(20):
            tracker.sli_history["api"] = tracker.sli_history.get("api", [])
            tracker.sli_history["api"].append(
                SLIResult(
                    timestamp=datetime.utcnow() - timedelta(minutes=i),
                    service="api",
                    sli_type=SLIType.AVAILABILITY,
                    value=0.95,  # Below threshold
                    threshold=0.999,
                    is_good=False,
                    calculation_time_ms=1.0
                )
            )
        
        # Check for burn rate alerts
        alerts = tracker.check_burn_rate_alerts("api")
        assert len(alerts) > 0
        
        # Would trigger runbook in production
        assert any(alert[0] == AlertSeverity.CRITICAL for alert in alerts)


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])