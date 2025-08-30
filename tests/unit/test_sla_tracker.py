"""Unit tests for SLA Tracker."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock

from genesis.monitoring.sla_tracker import (
    SLAMetric,
    SLATarget,
    SLAStatus,
    SLAReport,
    SLATracker,
    IncidentSeverity,
    Incident
)


class TestSLAMetric:
    """Test SLAMetric class."""
    
    def test_sla_metric_creation(self):
        """Test creating an SLA metric."""
        metric = SLAMetric(
            name="uptime",
            current_value=99.95,
            target_value=99.9,
            unit="percent"
        )
        
        assert metric.name == "uptime"
        assert metric.current_value == 99.95
        assert metric.target_value == 99.9
        assert metric.unit == "percent"
        assert metric.is_met is True
    
    def test_sla_metric_not_met(self):
        """Test SLA metric not meeting target."""
        metric = SLAMetric(
            name="latency",
            current_value=150,
            target_value=100,
            unit="ms",
            lower_is_better=True
        )
        
        assert metric.is_met is False
    
    def test_sla_metric_calculate_breach_duration(self):
        """Test calculating breach duration."""
        metric = SLAMetric(
            name="uptime",
            current_value=99.5,
            target_value=99.9,
            unit="percent"
        )
        
        metric.breach_start = datetime.now() - timedelta(minutes=30)
        duration = metric.get_breach_duration()
        
        assert duration is not None
        assert duration.total_seconds() >= 1800  # 30 minutes


class TestSLATarget:
    """Test SLATarget class."""
    
    def test_sla_target_creation(self):
        """Test creating an SLA target."""
        target = SLATarget(
            metric_name="uptime",
            target_value=99.9,
            measurement_period=timedelta(days=30),
            description="99.9% uptime over 30 days"
        )
        
        assert target.metric_name == "uptime"
        assert target.target_value == 99.9
        assert target.measurement_period == timedelta(days=30)
    
    def test_sla_target_with_thresholds(self):
        """Test SLA target with warning thresholds."""
        target = SLATarget(
            metric_name="latency",
            target_value=100,
            measurement_period=timedelta(hours=1),
            warning_threshold=80,
            critical_threshold=90
        )
        
        assert target.warning_threshold == 80
        assert target.critical_threshold == 90


class TestIncident:
    """Test Incident class."""
    
    def test_incident_creation(self):
        """Test creating an incident."""
        incident = Incident(
            id="INC-001",
            title="WebSocket disconnection",
            severity=IncidentSeverity.MAJOR,
            start_time=datetime.now()
        )
        
        assert incident.id == "INC-001"
        assert incident.title == "WebSocket disconnection"
        assert incident.severity == IncidentSeverity.MAJOR
        assert incident.end_time is None
        assert incident.is_resolved is False
    
    def test_incident_resolution(self):
        """Test resolving an incident."""
        incident = Incident(
            id="INC-001",
            title="High latency",
            severity=IncidentSeverity.MINOR,
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        incident.resolve("Latency returned to normal")
        
        assert incident.is_resolved is True
        assert incident.end_time is not None
        assert incident.resolution == "Latency returned to normal"
    
    def test_incident_duration(self):
        """Test calculating incident duration."""
        start = datetime.now() - timedelta(hours=2)
        incident = Incident(
            id="INC-001",
            title="Service outage",
            severity=IncidentSeverity.CRITICAL,
            start_time=start
        )
        
        incident.end_time = datetime.now()
        duration = incident.get_duration()
        
        assert duration is not None
        assert duration.total_seconds() >= 7200  # 2 hours
    
    def test_incident_mttr(self):
        """Test mean time to recovery."""
        incident = Incident(
            id="INC-001",
            title="Database connection lost",
            severity=IncidentSeverity.MAJOR,
            start_time=datetime.now() - timedelta(minutes=45)
        )
        
        incident.detection_time = incident.start_time + timedelta(minutes=5)
        incident.response_time = incident.start_time + timedelta(minutes=10)
        incident.end_time = datetime.now()
        
        mttr = incident.get_mttr()
        assert mttr is not None
        assert mttr.total_seconds() >= 2700  # 45 minutes


class TestSLATracker:
    """Test SLATracker class."""
    
    def test_sla_tracker_initialization(self):
        """Test SLA tracker initialization."""
        tracker = SLATracker()
        
        assert len(tracker.targets) == 0
        assert len(tracker.metrics) == 0
        assert len(tracker.incidents) == 0
    
    def test_add_target(self):
        """Test adding SLA target."""
        tracker = SLATracker()
        
        target = SLATarget(
            metric_name="uptime",
            target_value=99.9,
            measurement_period=timedelta(days=30)
        )
        
        tracker.add_target(target)
        assert "uptime" in tracker.targets
        assert tracker.targets["uptime"] == target
    
    def test_record_metric(self):
        """Test recording SLA metric."""
        tracker = SLATracker()
        
        # Add target first
        tracker.add_target(SLATarget(
            metric_name="uptime",
            target_value=99.9,
            measurement_period=timedelta(days=30)
        ))
        
        # Record metric
        tracker.record_metric("uptime", 99.95)
        
        assert "uptime" in tracker.metrics
        assert len(tracker.metrics["uptime"]) == 1
        assert tracker.metrics["uptime"][0].value == 99.95
    
    def test_calculate_uptime(self):
        """Test calculating uptime percentage."""
        tracker = SLATracker()
        
        # Simulate some downtime
        total_time = timedelta(days=30)
        downtime = timedelta(hours=1)
        
        uptime = tracker.calculate_uptime(total_time, downtime)
        
        expected = 100 * (1 - downtime.total_seconds() / total_time.total_seconds())
        assert abs(uptime - expected) < 0.01
    
    @pytest.mark.asyncio
    async def test_calculate_availability(self):
        """Test calculating availability."""
        tracker = SLATracker()
        
        # Record some health checks
        for i in range(100):
            success = i % 10 != 0  # 10% failure rate
            await tracker.record_health_check(success)
        
        availability = tracker.calculate_availability()
        assert availability == 90.0
    
    def test_calculate_error_rate(self):
        """Test calculating error rate."""
        tracker = SLATracker()
        
        # Record requests
        tracker.record_request(success=True)
        tracker.record_request(success=True)
        tracker.record_request(success=False)
        tracker.record_request(success=True)
        tracker.record_request(success=False)
        
        error_rate = tracker.calculate_error_rate()
        assert error_rate == 40.0  # 2 errors out of 5 requests
    
    def test_calculate_latency_percentiles(self):
        """Test calculating latency percentiles."""
        tracker = SLATracker()
        
        # Record latencies
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for latency in latencies:
            tracker.record_latency(latency)
        
        p50 = tracker.get_latency_percentile(50)
        p95 = tracker.get_latency_percentile(95)
        p99 = tracker.get_latency_percentile(99)
        
        assert p50 == 55  # median
        assert p95 == 95.5
        assert p99 == 99.1
    
    def test_create_incident(self):
        """Test creating an incident."""
        tracker = SLATracker()
        
        incident = tracker.create_incident(
            title="High error rate",
            severity=IncidentSeverity.MAJOR,
            description="Error rate exceeded 5%"
        )
        
        assert incident.id.startswith("INC-")
        assert incident.title == "High error rate"
        assert len(tracker.incidents) == 1
    
    def test_resolve_incident(self):
        """Test resolving an incident."""
        tracker = SLATracker()
        
        # Create incident
        incident = tracker.create_incident(
            title="Service degradation",
            severity=IncidentSeverity.MINOR
        )
        
        # Resolve it
        tracker.resolve_incident(incident.id, "Service restored")
        
        assert incident.is_resolved is True
        assert incident.resolution == "Service restored"
    
    def test_get_active_incidents(self):
        """Test getting active incidents."""
        tracker = SLATracker()
        
        # Create multiple incidents
        inc1 = tracker.create_incident("Issue 1", IncidentSeverity.MINOR)
        inc2 = tracker.create_incident("Issue 2", IncidentSeverity.MAJOR)
        inc3 = tracker.create_incident("Issue 3", IncidentSeverity.CRITICAL)
        
        # Resolve one
        tracker.resolve_incident(inc2.id, "Fixed")
        
        active = tracker.get_active_incidents()
        assert len(active) == 2
        assert inc1 in active
        assert inc3 in active
        assert inc2 not in active
    
    def test_calculate_mttr(self):
        """Test calculating mean time to recovery."""
        tracker = SLATracker()
        
        # Create and resolve incidents with different durations
        for i in range(3):
            incident = tracker.create_incident(
                f"Issue {i}",
                IncidentSeverity.MAJOR
            )
            incident.start_time = datetime.now() - timedelta(hours=i+1)
            incident.end_time = datetime.now()
        
        mttr = tracker.calculate_mttr(period=timedelta(days=1))
        
        # Average of 1, 2, and 3 hours = 2 hours
        assert mttr.total_seconds() == 7200
    
    def test_calculate_mtbf(self):
        """Test calculating mean time between failures."""
        tracker = SLATracker()
        
        now = datetime.now()
        
        # Create incidents at specific intervals
        inc1 = tracker.create_incident("Issue 1", IncidentSeverity.MAJOR)
        inc1.start_time = now - timedelta(hours=10)
        
        inc2 = tracker.create_incident("Issue 2", IncidentSeverity.MAJOR)
        inc2.start_time = now - timedelta(hours=6)
        
        inc3 = tracker.create_incident("Issue 3", IncidentSeverity.MAJOR)
        inc3.start_time = now - timedelta(hours=2)
        
        mtbf = tracker.calculate_mtbf(period=timedelta(days=1))
        
        # Average of 4 hours and 4 hours = 4 hours
        assert mtbf.total_seconds() == 14400
    
    def test_generate_sla_report(self):
        """Test generating SLA report."""
        tracker = SLATracker()
        
        # Set up targets
        tracker.add_target(SLATarget(
            metric_name="uptime",
            target_value=99.9,
            measurement_period=timedelta(days=30)
        ))
        
        tracker.add_target(SLATarget(
            metric_name="latency_p95",
            target_value=100,
            measurement_period=timedelta(days=30)
        ))
        
        # Record some metrics
        tracker.record_metric("uptime", 99.95)
        tracker.record_metric("latency_p95", 85)
        
        # Create an incident
        tracker.create_incident("Test incident", IncidentSeverity.MINOR)
        
        # Generate report
        report = tracker.generate_report(period=timedelta(days=30))
        
        assert report.period == timedelta(days=30)
        assert len(report.metrics) == 2
        assert report.metrics[0].name == "uptime"
        assert report.metrics[0].is_met is True
        assert report.incident_count == 1
    
    def test_check_sla_breaches(self):
        """Test checking for SLA breaches."""
        tracker = SLATracker()
        
        # Add targets
        tracker.add_target(SLATarget(
            metric_name="uptime",
            target_value=99.9,
            measurement_period=timedelta(days=30)
        ))
        
        tracker.add_target(SLATarget(
            metric_name="error_rate",
            target_value=1.0,  # Max 1% error rate
            measurement_period=timedelta(hours=1),
            lower_is_better=True
        ))
        
        # Record metrics that breach SLA
        tracker.record_metric("uptime", 99.5)  # Below target
        tracker.record_metric("error_rate", 2.5)  # Above target
        
        breaches = tracker.check_breaches()
        
        assert len(breaches) == 2
        assert any(b.name == "uptime" for b in breaches)
        assert any(b.name == "error_rate" for b in breaches)
    
    def test_export_metrics_prometheus(self):
        """Test exporting metrics in Prometheus format."""
        tracker = SLATracker()
        
        # Record various metrics
        tracker.record_metric("uptime", 99.95)
        tracker.record_latency(50)
        tracker.record_request(success=True)
        tracker.record_request(success=False)
        
        prometheus_metrics = tracker.export_prometheus_format()
        
        assert "# HELP sla_uptime_percent" in prometheus_metrics
        assert "# TYPE sla_uptime_percent gauge" in prometheus_metrics
        assert "sla_uptime_percent 99.95" in prometheus_metrics
    
    def test_historical_sla_tracking(self):
        """Test tracking SLA over time."""
        tracker = SLATracker()
        
        # Add target
        tracker.add_target(SLATarget(
            metric_name="uptime",
            target_value=99.9,
            measurement_period=timedelta(days=30)
        ))
        
        # Record metrics over time
        base_time = datetime.now() - timedelta(days=7)
        for day in range(7):
            timestamp = base_time + timedelta(days=day)
            value = 99.9 + (0.05 if day % 2 == 0 else -0.05)
            tracker.record_metric("uptime", value, timestamp=timestamp)
        
        # Get historical data
        history = tracker.get_metric_history("uptime", days=7)
        
        assert len(history) == 7
        
        # Calculate compliance rate
        compliance = tracker.calculate_compliance_rate("uptime", days=7)
        assert compliance == 50.0  # 3.5 days out of 7 met target