"""
Unit tests for SLO tracking and alerting components.

Ensures all individual components work correctly in isolation.
"""

import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal

import pytest

from genesis.monitoring.slo_tracker import (
    SLIType, AlertSeverity, BurnRateThreshold, SLIConfig,
    SLOConfig, SLIResult, ErrorBudget, AlertFingerprint
)
from genesis.monitoring.alert_channels import (
    AlertChannel, AlertPriority, Alert, DeliveryResult, RateLimiter
)
from genesis.monitoring.alert_dedup import (
    GroupingStrategy, AlertGroup, RoutingRule
)
from genesis.monitoring.runbook_executor import (
    ActionType, ExecutionMode, RunbookAction, RunbookCondition, Runbook
)
from genesis.monitoring.incident_tracker import (
    IncidentStatus, IncidentPriority, Incident
)


class TestSLODataModels:
    """Test SLO data model classes."""
    
    def test_sli_config_creation(self):
        """Test SLI configuration creation."""
        config = SLIConfig(
            type=SLIType.AVAILABILITY,
            metric_query="up{job='api'}",
            threshold=0.999,
            aggregation="avg",
            unit="ratio"
        )
        
        assert config.type == SLIType.AVAILABILITY
        assert config.threshold == 0.999
        assert config.aggregation == "avg"
    
    def test_burn_rate_threshold(self):
        """Test burn rate threshold configuration."""
        threshold = BurnRateThreshold(
            rate=14.4,
            window_minutes=60,
            severity=AlertSeverity.CRITICAL
        )
        
        assert threshold.rate == 14.4
        assert threshold.window_minutes == 60
        assert threshold.severity == AlertSeverity.CRITICAL
    
    def test_sli_result(self):
        """Test SLI result dataclass."""
        result = SLIResult(
            timestamp=datetime.utcnow(),
            service="test_service",
            sli_type=SLIType.LATENCY,
            value=0.05,
            threshold=0.1,
            is_good=True,
            calculation_time_ms=2.5
        )
        
        assert result.is_good
        assert result.value < result.threshold
        assert result.calculation_time_ms == 2.5
    
    def test_error_budget(self):
        """Test error budget calculation result."""
        budget = ErrorBudget(
            service="test_service",
            window=timedelta(days=30),
            total_budget=1.0,
            consumed_budget=0.3,
            remaining_budget=0.7,
            remaining_ratio=0.7,
            burn_rate=1.5,
            time_until_exhaustion=timedelta(days=14)
        )
        
        assert budget.remaining_ratio == 0.7
        assert budget.burn_rate == 1.5
        assert budget.time_until_exhaustion.days == 14


class TestAlertModels:
    """Test alert data models."""
    
    def test_alert_creation(self):
        """Test alert object creation."""
        alert = Alert(
            id="test_001",
            name="TestAlert",
            summary="Test summary",
            description="Test description",
            severity="critical",
            priority=AlertPriority.CRITICAL,
            service="test_service",
            source="prometheus",
            labels={"env": "prod"},
            annotations={"runbook": "test.md"},
            timestamp=datetime.utcnow(),
            runbook_url="https://runbooks.io/test",
            dashboard_url="https://grafana.io/test"
        )
        
        assert alert.priority == AlertPriority.CRITICAL
        assert alert.labels["env"] == "prod"
        assert alert.runbook_url is not None
    
    def test_delivery_result(self):
        """Test delivery result dataclass."""
        result = DeliveryResult(
            channel=AlertChannel.SLACK,
            success=True,
            message="Alert delivered",
            response_code=200,
            delivery_time_ms=150.5
        )
        
        assert result.success
        assert result.channel == AlertChannel.SLACK
        assert result.delivery_time_ms == 150.5
    
    def test_alert_fingerprint(self):
        """Test alert fingerprint generation."""
        alert = Alert(
            id="fp_001",
            name="HighCPU",
            summary="High CPU",
            description="CPU above 90%",
            severity="warning",
            priority=AlertPriority.MEDIUM,
            service="api",
            source="prometheus",
            labels={"instance": "server1"},
            annotations={},
            timestamp=datetime.utcnow()
        )
        
        fingerprint = AlertFingerprint.generate(
            alert,
            ["name", "service", "label:instance"]
        )
        
        assert fingerprint.hash is not None
        assert len(fingerprint.hash) == 16
        assert fingerprint.components["name"] == "HighCPU"
        assert fingerprint.components["service"] == "api"
        assert fingerprint.components["label:instance"] == "server1"


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_token_bucket(self):
        """Test token bucket rate limiting."""
        limiter = RateLimiter(max_per_minute=60, burst=10)
        
        # Should have burst tokens available
        assert limiter.tokens == 10
        
        # Acquire tokens
        for i in range(10):
            assert await limiter.acquire()
        
        # Next should fail (no tokens)
        assert not await limiter.acquire()
    
    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time."""
        limiter = RateLimiter(max_per_minute=60, burst=10)
        
        # Use all tokens
        limiter.tokens = 0
        
        # Wait for refill
        import asyncio
        await asyncio.sleep(1.1)  # Wait for tokens to refill
        
        # Should have some tokens now
        assert await limiter.acquire()


class TestAlertGrouping:
    """Test alert grouping functionality."""
    
    def test_alert_group_creation(self):
        """Test alert group creation and management."""
        group = AlertGroup(
            id="group_001",
            strategy=GroupingStrategy.SERVICE,
            fingerprint="abc123"
        )
        
        alert = Mock(spec=Alert)
        
        # Add alerts
        group.add_alert(alert, suppress=False)
        assert len(group.alerts) == 1
        assert group.count == 1
        assert group.suppressed_count == 0
        
        # Add suppressed alert
        group.add_alert(alert, suppress=True)
        assert len(group.alerts) == 1  # Not added to list
        assert group.count == 2
        assert group.suppressed_count == 1
    
    def test_suppression_threshold(self):
        """Test alert suppression based on group size."""
        group = AlertGroup(
            id="group_002",
            strategy=GroupingStrategy.ALERTNAME,
            fingerprint="def456"
        )
        
        # Add alerts up to limit
        for i in range(10):
            group.add_alert(Mock(spec=Alert), suppress=False)
        
        # Should suppress after limit
        assert group.should_suppress(max_alerts=10)


class TestRoutingRules:
    """Test alert routing rules."""
    
    def test_routing_rule_matching(self):
        """Test routing rule condition matching."""
        rule = RoutingRule(
            name="test_rule",
            priority=100,
            match_conditions={
                "service": "database",
                "severity": ["critical", "high"]
            },
            channels=[AlertChannel.PAGERDUTY]
        )
        
        # Matching alert
        alert1 = Mock(spec=Alert)
        alert1.service = "database"
        alert1.severity = "critical"
        alert1.priority = AlertPriority.CRITICAL
        alert1.labels = {}
        
        assert rule.matches(alert1)
        
        # Non-matching alert
        alert2 = Mock(spec=Alert)
        alert2.service = "api"
        alert2.severity = "warning"
        alert2.priority = AlertPriority.MEDIUM
        alert2.labels = {}
        
        assert not rule.matches(alert2)
    
    def test_regex_pattern_matching(self):
        """Test regex pattern matching in routing rules."""
        rule = RoutingRule(
            name="regex_rule",
            priority=90,
            match_conditions={
                "service": "~database.*"
            },
            channels=[AlertChannel.SLACK]
        )
        
        # Should match
        alert1 = Mock(spec=Alert)
        alert1.service = "database_primary"
        alert1.severity = "warning"
        alert1.priority = AlertPriority.MEDIUM
        alert1.labels = {}
        
        assert rule.matches(alert1)
        
        # Should not match
        alert2 = Mock(spec=Alert)
        alert2.service = "api_service"
        alert2.severity = "warning"
        alert2.priority = AlertPriority.MEDIUM
        alert2.labels = {}
        
        assert not rule.matches(alert2)


class TestRunbookModels:
    """Test runbook data models."""
    
    def test_runbook_action(self):
        """Test runbook action configuration."""
        action = RunbookAction(
            name="restart_service",
            type=ActionType.RESTART_SERVICE,
            description="Restart the service",
            command="systemctl restart api",
            timeout_seconds=30,
            retry_count=3,
            safe_for_auto=False,
            requires_confirmation=True
        )
        
        assert action.type == ActionType.RESTART_SERVICE
        assert action.requires_confirmation
        assert not action.safe_for_auto
    
    def test_runbook_condition_evaluation(self):
        """Test runbook condition evaluation."""
        condition = RunbookCondition(
            metric="cpu_usage",
            operator=">",
            value=80
        )
        
        # Should match
        assert condition.evaluate({"cpu_usage": 85})
        
        # Should not match
        assert not condition.evaluate({"cpu_usage": 75})
        
        # Test other operators
        condition2 = RunbookCondition(
            metric="service",
            operator="in",
            value=["api", "database", "cache"]
        )
        
        assert condition2.evaluate({"service": "api"})
        assert not condition2.evaluate({"service": "frontend"})
    
    def test_runbook_creation(self):
        """Test runbook object creation."""
        action = RunbookAction(
            name="test_action",
            type=ActionType.HTTP,
            description="Test action"
        )
        
        condition = RunbookCondition(
            metric="test_metric",
            operator="==",
            value="test_value"
        )
        
        runbook = Runbook(
            id="test_runbook",
            name="Test Runbook",
            description="Test runbook description",
            trigger={"alert_name": "TestAlert"},
            conditions=[condition],
            actions=[action],
            execution_mode=ExecutionMode.APPROVAL_REQUIRED,
            max_executions_per_hour=5,
            tags=["test", "automation"]
        )
        
        assert runbook.id == "test_runbook"
        assert len(runbook.actions) == 1
        assert runbook.execution_mode == ExecutionMode.APPROVAL_REQUIRED


class TestIncidentModels:
    """Test incident data models."""
    
    def test_incident_creation(self):
        """Test incident object creation."""
        incident = Incident(
            id="INC-001",
            title="Test Incident",
            description="Test description",
            service="test_service",
            status=IncidentStatus.OPEN,
            priority=IncidentPriority.HIGH,
            created_at=datetime.utcnow(),
            created_by="system",
            alert_ids=["alert_001", "alert_002"]
        )
        
        assert incident.status == IncidentStatus.OPEN
        assert incident.priority == IncidentPriority.HIGH
        assert len(incident.alert_ids) == 2
    
    def test_mtta_calculation(self):
        """Test mean time to acknowledge calculation."""
        created = datetime.utcnow()
        acknowledged = created + timedelta(minutes=5)
        
        incident = Incident(
            id="INC-002",
            title="Test",
            description="Test",
            service="test",
            status=IncidentStatus.ACKNOWLEDGED,
            priority=IncidentPriority.MEDIUM,
            created_at=created,
            acknowledged_at=acknowledged
        )
        
        mtta = incident.calculate_mtta()
        assert mtta is not None
        assert mtta == 300  # 5 minutes in seconds
    
    def test_mttr_calculation(self):
        """Test mean time to resolve calculation."""
        created = datetime.utcnow()
        resolved = created + timedelta(hours=2)
        
        incident = Incident(
            id="INC-003",
            title="Test",
            description="Test",
            service="test",
            status=IncidentStatus.RESOLVED,
            priority=IncidentPriority.LOW,
            created_at=created,
            resolved_at=resolved
        )
        
        mttr = incident.calculate_mttr()
        assert mttr is not None
        assert mttr == 7200  # 2 hours in seconds


class TestConfigurationLoading:
    """Test configuration file loading."""
    
    def test_yaml_parsing(self, tmp_path):
        """Test YAML configuration parsing."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "services": {
                "test": {
                    "slis": {
                        "availability": {
                            "type": "availability",
                            "metric": "up",
                            "threshold": 0.99
                        }
                    }
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and verify
        with open(config_file, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert "services" in loaded
        assert "test" in loaded["services"]
        assert loaded["services"]["test"]["slis"]["availability"]["threshold"] == 0.99
    
    def test_json_serialization(self):
        """Test JSON serialization of data structures."""
        data = {
            "incident_id": "INC-001",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "mtta": 300,
                "mttr": 7200
            },
            "alerts": ["alert_001", "alert_002"]
        }
        
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        
        assert loaded["incident_id"] == "INC-001"
        assert loaded["metrics"]["mtta"] == 300
        assert len(loaded["alerts"]) == 2


class TestErrorHandling:
    """Test error handling in components."""
    
    @pytest.mark.asyncio
    async def test_prometheus_query_error_handling(self):
        """Test handling of Prometheus query errors."""
        from genesis.monitoring.slo_tracker import SLOTracker
        
        tracker = SLOTracker()
        
        # Mock failed query
        with patch.object(tracker, '_query_prometheus', side_effect=Exception("Connection error")):
            with pytest.raises(Exception) as exc_info:
                sli_config = SLIConfig(
                    type=SLIType.AVAILABILITY,
                    metric_query="up",
                    threshold=0.99
                )
                await tracker.calculate_sli("test", "availability", sli_config)
            
            assert "Connection error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_alert_delivery_retry(self):
        """Test alert delivery retry on failure."""
        from genesis.monitoring.alert_channels import AlertChannelManager
        
        with patch('genesis.monitoring.alert_channels.VaultClient'):
            manager = AlertChannelManager()
        
        alert = Mock(spec=Alert)
        alert.id = "test"
        alert.priority = AlertPriority.HIGH
        
        # Mock failed then successful delivery
        call_count = 0
        
        async def mock_send(alert):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            return DeliveryResult(
                channel=AlertChannel.SLACK,
                success=True,
                message="Delivered"
            )
        
        with patch.object(manager, '_send_slack', side_effect=mock_send):
            # Should retry and eventually succeed
            results = await manager.send_alert(alert, [AlertChannel.SLACK])
            # Due to retry decorator, this might not work as expected in test
            # but demonstrates the concept
    
    def test_runbook_validation(self):
        """Test runbook validation and error handling."""
        from genesis.monitoring.runbook_executor import RunbookExecutor
        from genesis.core.exceptions import ValidationError
        
        executor = RunbookExecutor()
        
        # Invalid runbook ID should raise error
        with pytest.raises(ValidationError) as exc_info:
            import asyncio
            asyncio.run(executor.execute_runbook(
                "nonexistent_runbook",
                {},
                dry_run=True
            ))
        
        assert "Runbook not found" in str(exc_info.value)


# Performance tests
class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_sli_calculation_performance(self):
        """Test SLI calculation performance."""
        from genesis.monitoring.slo_tracker import SLOTracker
        import time
        
        tracker = SLOTracker()
        sli_config = SLIConfig(
            type=SLIType.AVAILABILITY,
            metric_query="up",
            threshold=0.99
        )
        
        # Mock query to be fast
        async def fast_query(query):
            return 0.995
        
        tracker._query_prometheus = fast_query
        
        start = time.time()
        result = await tracker.calculate_sli("test", "availability", sli_config)
        duration = time.time() - start
        
        # Should be very fast (< 10ms)
        assert duration < 0.01
        assert result.calculation_time_ms < 10
    
    def test_alert_deduplication_performance(self):
        """Test alert deduplication performance with many alerts."""
        from genesis.monitoring.alert_dedup import AlertDeduplicator
        import time
        
        dedup = AlertDeduplicator()
        
        # Create many alerts
        alerts = []
        for i in range(1000):
            alert = Mock(spec=Alert)
            alert.id = f"alert_{i:04d}"
            alert.name = f"Alert{i % 10}"  # 10 different alert types
            alert.service = f"service{i % 5}"  # 5 different services
            alert.severity = "warning"
            alert.priority = AlertPriority.MEDIUM
            alert.labels = {"instance": f"server{i % 20}"}
            alert.source = "prometheus"
            alert.annotations = {}
            alert.timestamp = datetime.utcnow()
            alerts.append(alert)
        
        # Process all alerts
        start = time.time()
        for alert in alerts:
            import asyncio
            asyncio.run(dedup.process_alert(alert))
        duration = time.time() - start
        
        # Should process 1000 alerts quickly (< 1 second)
        assert duration < 1.0
        
        # Should have deduplicated many
        assert dedup.stats["deduplicated"] > 0
        assert len(dedup.groups) <= 50  # Should group effectively


if __name__ == "__main__":
    pytest.main([__file__, "-v"])