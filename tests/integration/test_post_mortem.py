"""
Integration tests for post-mortem generation system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from genesis.operations.post_mortem import (
    PostMortemGenerator, PostMortemReport, TimelineEvent,
    RootCauseAnalysis, ActionItem, ImpactLevel
)
from genesis.operations.incident_manager import (
    Incident, Alert, IncidentSeverity, IncidentStatus, AlertType
)


@pytest.fixture
def post_mortem_generator():
    """Create post-mortem generator."""
    return PostMortemGenerator()


@pytest.fixture
def sample_incident():
    """Create sample incident with full data."""
    incident = Incident(
        incident_id="INC-20240101-0001",
        title="Database Connection Lost",
        description="Database connection pool exhausted, unable to execute queries",
        severity=IncidentSeverity.HIGH,
        status=IncidentStatus.RESOLVED,
        created_at=datetime.utcnow() - timedelta(hours=2),
        updated_at=datetime.utcnow(),
        resolved_at=datetime.utcnow() - timedelta(minutes=30),
        alerts=[
            Alert(
                alert_id="ALT-001",
                alert_type=AlertType.DATABASE,
                severity=IncidentSeverity.HIGH,
                message="Database connection pool exhausted",
                source="db_monitor",
                timestamp=datetime.utcnow() - timedelta(hours=2)
            )
        ],
        responders=["john.doe", "jane.smith"],
        runbook_url="docs/runbooks/incident-response.md#scenario-4",
        mitigation_steps=[
            "Increased connection pool size",
            "Restarted application servers",
            "Cleared stale connections"
        ],
        root_cause="Connection leak in order processing module"
    )
    
    # Add timeline entries
    incident.add_timeline_entry(
        action="incident_created",
        actor="system",
        details="Incident created from alert ALT-001"
    )
    incident.add_timeline_entry(
        action="acknowledged",
        actor="john.doe",
        details="Acknowledged and investigating"
    )
    incident.add_timeline_entry(
        action="investigating",
        actor="john.doe",
        details="Found connection pool at maximum"
    )
    incident.add_timeline_entry(
        action="mitigation_started",
        actor="jane.smith",
        details="Increasing pool size and restarting services"
    )
    incident.add_timeline_entry(
        action="resolved",
        actor="jane.smith",
        details="Services restored, monitoring for stability"
    )
    
    return incident


@pytest.fixture
def sample_logs():
    """Create sample log entries."""
    base_time = datetime.utcnow() - timedelta(hours=2)
    return [
        {
            "timestamp": base_time.isoformat(),
            "level": "error",
            "message": "Database connection pool exhausted",
            "service": "db_pool_manager"
        },
        {
            "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
            "level": "error",
            "message": "Failed to acquire database connection",
            "service": "order_processor"
        },
        {
            "timestamp": (base_time + timedelta(minutes=30)).isoformat(),
            "level": "info",
            "message": "Restarting application server",
            "service": "supervisor"
        },
        {
            "timestamp": (base_time + timedelta(minutes=45)).isoformat(),
            "level": "info",
            "message": "Connection pool size increased to 100",
            "service": "db_pool_manager"
        }
    ]


@pytest.fixture
def sample_metrics():
    """Create sample metrics during incident."""
    return {
        "failed_orders": 127,
        "affected_positions": 15,
        "estimated_loss": 2340.50,
        "api_errors": 523,
        "response_time_increase": 450  # ms
    }


class TestPostMortemGenerator:
    """Test PostMortemGenerator class."""
    
    @pytest.mark.asyncio
    async def test_generate_report(self, post_mortem_generator, sample_incident,
                                  sample_logs, sample_metrics):
        """Test generating a complete post-mortem report."""
        report = await post_mortem_generator.generate_report(
            sample_incident,
            sample_logs,
            sample_metrics
        )
        
        assert report is not None
        assert report.incident_id == sample_incident.incident_id
        assert report.severity == sample_incident.severity
        assert report.duration.total_seconds() > 0
        assert len(report.timeline) > 0
        assert report.root_cause_analysis is not None
        assert len(report.action_items) > 0
    
    @pytest.mark.asyncio
    async def test_timeline_reconstruction(self, post_mortem_generator, 
                                          sample_incident, sample_logs):
        """Test timeline reconstruction from incident and logs."""
        timeline = await post_mortem_generator._reconstruct_timeline(
            sample_incident,
            sample_logs
        )
        
        assert len(timeline) > 0
        
        # Check timeline is sorted
        timestamps = [event.timestamp for event in timeline]
        assert timestamps == sorted(timestamps)
        
        # Check event types
        event_types = [event.event_type for event in timeline]
        assert "detection" in event_types
        assert "action" in event_types
        assert "resolution" in event_types
    
    @pytest.mark.asyncio
    async def test_root_cause_analysis(self, post_mortem_generator, 
                                      sample_incident, sample_logs):
        """Test root cause analysis."""
        timeline = await post_mortem_generator._reconstruct_timeline(
            sample_incident,
            sample_logs
        )
        
        analysis = await post_mortem_generator._analyze_root_cause(
            sample_incident,
            sample_logs,
            timeline
        )
        
        assert analysis is not None
        assert analysis.what_happened is not None
        assert len(analysis.five_whys) > 0
        assert analysis.root_cause == sample_incident.root_cause
    
    @pytest.mark.asyncio
    async def test_impact_calculation(self, post_mortem_generator,
                                     sample_incident, sample_metrics):
        """Test impact calculation."""
        quantitative = await post_mortem_generator._calculate_quantitative_impact(
            sample_incident,
            sample_metrics
        )
        
        assert quantitative["duration_minutes"] > 0
        assert quantitative["orders_affected"] == 127
        assert quantitative["estimated_loss"] == 2340.50
        
        qualitative = post_mortem_generator._assess_qualitative_impact(sample_incident)
        
        assert qualitative["reputation"] in [ImpactLevel.MODERATE, ImpactLevel.SEVERE]
        assert qualitative["regulatory"] == ImpactLevel.NONE
    
    def test_detection_analysis(self, post_mortem_generator, sample_incident):
        """Test detection analysis."""
        timeline = [
            TimelineEvent(
                timestamp=sample_incident.created_at + timedelta(seconds=30),
                event="Alert triggered",
                actor="monitoring",
                notes="Database alert",
                event_type="detection"
            )
        ]
        
        analysis = post_mortem_generator._analyze_detection(sample_incident, timeline)
        
        assert analysis["time_to_detection_seconds"] == 30
        assert analysis["detection_method"] == "monitoring"
        assert analysis["could_detect_sooner"] is False
    
    def test_response_analysis(self, post_mortem_generator, sample_incident):
        """Test response analysis."""
        timeline = [
            TimelineEvent(
                timestamp=sample_incident.created_at + timedelta(minutes=2),
                event="Incident acknowledged",
                actor="john.doe",
                notes="",
                event_type="action"
            ),
            TimelineEvent(
                timestamp=sample_incident.created_at + timedelta(minutes=5),
                event="Investigation started",
                actor="john.doe",
                notes="",
                event_type="action"
            )
        ]
        
        analysis = post_mortem_generator._analyze_response(sample_incident, timeline)
        
        assert analysis["time_to_acknowledgment_seconds"] == 120
        assert analysis["response_efficiency"] == "Good"
        assert len(analysis["responders"]) == 2
    
    def test_identify_positives(self, post_mortem_generator, sample_incident):
        """Test identifying what went well."""
        timeline = [
            TimelineEvent(
                timestamp=sample_incident.created_at + timedelta(seconds=15),
                event="Alert triggered",
                actor="monitoring",
                notes="",
                event_type="detection"
            ),
            TimelineEvent(
                timestamp=sample_incident.created_at + timedelta(minutes=2),
                event="Acknowledged",
                actor="john.doe",
                notes="",
                event_type="action"
            )
        ]
        
        positives = post_mortem_generator._identify_positives(
            sample_incident,
            timeline,
            None
        )
        
        assert len(positives) > 0
        assert any("detected within" in p for p in positives)
        assert any("Runbook" in p for p in positives)
    
    def test_identify_improvements(self, post_mortem_generator):
        """Test identifying areas for improvement."""
        incident = Incident(
            incident_id="TEST",
            title="Test",
            description="Test",
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.utcnow() - timedelta(hours=2),
            updated_at=datetime.utcnow(),
            resolved_at=datetime.utcnow(),
            responders=["single_person"]
        )
        
        detection_analysis = {"could_detect_sooner": True}
        response_analysis = {"response_efficiency": "Poor"}
        
        improvements = post_mortem_generator._identify_improvements(
            incident,
            [],
            detection_analysis,
            response_analysis
        )
        
        assert len(improvements) > 0
        assert any("Detection" in i for i in improvements)
        assert any("Response" in i for i in improvements)
        assert any("one responder" in i for i in improvements)
    
    def test_generate_action_items(self, post_mortem_generator, sample_incident):
        """Test action item generation."""
        root_cause_analysis = RootCauseAnalysis(
            what_happened="Database failure",
            five_whys=[],
            contributing_factors={
                "monitoring_gap": "No connection pool monitoring",
                "documentation": "Runbook outdated"
            },
            root_cause="Connection leak"
        )
        
        improvements = ["Detection time could be improved"]
        
        action_items = post_mortem_generator._generate_action_items(
            sample_incident,
            root_cause_analysis,
            improvements
        )
        
        assert len(action_items) > 0
        
        # Check priorities
        priorities = [item.priority for item in action_items]
        assert "immediate" in priorities
        assert "short-term" in priorities
        
        # Check due dates
        for item in action_items:
            assert item.due_date > datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_export_to_markdown(self, post_mortem_generator,
                                     sample_incident, sample_logs, sample_metrics):
        """Test exporting report to markdown."""
        report = await post_mortem_generator.generate_report(
            sample_incident,
            sample_logs,
            sample_metrics
        )
        
        markdown = post_mortem_generator.export_to_markdown(report)
        
        assert markdown is not None
        assert report.incident_id in markdown
        assert "Post-Mortem Report" in markdown
        assert "Timeline" in markdown
        assert "Root Cause Analysis" in markdown
        assert "Action Items" in markdown
    
    @pytest.mark.asyncio
    async def test_report_persistence(self, post_mortem_generator,
                                     sample_incident):
        """Test that reports are stored in generator."""
        report = await post_mortem_generator.generate_report(
            sample_incident,
            logs=None,
            metrics=None
        )
        
        assert sample_incident.incident_id in post_mortem_generator.reports
        assert post_mortem_generator.reports[sample_incident.incident_id] == report
    
    @pytest.mark.asyncio
    async def test_minimal_incident(self, post_mortem_generator):
        """Test generating report with minimal incident data."""
        minimal_incident = Incident(
            incident_id="MIN-001",
            title="Minimal Incident",
            description="Test incident with minimal data",
            severity=IncidentSeverity.LOW,
            status=IncidentStatus.DETECTED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        report = await post_mortem_generator.generate_report(
            minimal_incident,
            logs=None,
            metrics=None
        )
        
        assert report is not None
        assert report.incident_id == "MIN-001"
        assert len(report.timeline) > 0
        assert report.root_cause_analysis is not None