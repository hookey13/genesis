"""
Unit tests for incident management system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from genesis.operations.incident_manager import (
    IncidentManager, Alert, Incident, IncidentSeverity, 
    IncidentStatus, AlertType
)
from genesis.operations.pagerduty_client import PagerDutyClient


@pytest.fixture
def mock_pagerduty():
    """Create mock PagerDuty client."""
    client = Mock(spec=PagerDutyClient)
    client.trigger_incident = AsyncMock(return_value={"dedup_key": "test-key"})
    client.acknowledge_incident = AsyncMock(return_value={"status": "acknowledged"})
    client.resolve_incident = AsyncMock(return_value={"status": "resolved"})
    return client


@pytest.fixture
def incident_manager(mock_pagerduty):
    """Create incident manager with mock PagerDuty."""
    return IncidentManager(pagerduty_client=mock_pagerduty)


@pytest.fixture
def sample_alert():
    """Create sample alert."""
    return Alert(
        alert_id="ALT-001",
        alert_type=AlertType.SYSTEM_DOWN,
        severity=IncidentSeverity.CRITICAL,
        message="System is not responding",
        source="health_monitor",
        timestamp=datetime.utcnow(),
        metadata={"host": "prod-01"}
    )


class TestAlert:
    """Test Alert class."""
    
    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            alert_id="ALT-001",
            alert_type=AlertType.ORDER_FAILURE,
            severity=IncidentSeverity.HIGH,
            message="Orders failing",
            source="executor",
            timestamp=datetime.utcnow()
        )
        
        assert alert.alert_id == "ALT-001"
        assert alert.alert_type == AlertType.ORDER_FAILURE
        assert alert.severity == IncidentSeverity.HIGH
    
    def test_alert_fingerprint(self):
        """Test alert fingerprint generation."""
        alert1 = Alert(
            alert_id="ALT-001",
            alert_type=AlertType.DATABASE,
            severity=IncidentSeverity.HIGH,
            message="Database connection lost",
            source="db_monitor",
            timestamp=datetime.utcnow()
        )
        
        alert2 = Alert(
            alert_id="ALT-002",
            alert_type=AlertType.DATABASE,
            severity=IncidentSeverity.HIGH,
            message="Database connection lost",
            source="db_monitor",
            timestamp=datetime.utcnow()
        )
        
        # Same type, source, message = same fingerprint
        assert alert1.fingerprint() == alert2.fingerprint()
        
        alert3 = Alert(
            alert_id="ALT-003",
            alert_type=AlertType.DATABASE,
            severity=IncidentSeverity.HIGH,
            message="Different message",
            source="db_monitor",
            timestamp=datetime.utcnow()
        )
        
        # Different message = different fingerprint
        assert alert1.fingerprint() != alert3.fingerprint()


class TestIncident:
    """Test Incident class."""
    
    def test_incident_creation(self):
        """Test incident creation."""
        incident = Incident(
            incident_id="INC-001",
            title="System Outage",
            description="Complete system failure",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.DETECTED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert incident.incident_id == "INC-001"
        assert incident.status == IncidentStatus.DETECTED
        assert incident.resolved_at is None
    
    def test_timeline_entry(self):
        """Test adding timeline entries."""
        incident = Incident(
            incident_id="INC-001",
            title="Test Incident",
            description="Test",
            severity=IncidentSeverity.MEDIUM,
            status=IncidentStatus.DETECTED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        incident.add_timeline_entry(
            action="acknowledged",
            actor="john.doe",
            details="Looking into it"
        )
        
        assert len(incident.timeline) == 1
        assert incident.timeline[0]["action"] == "acknowledged"
        assert incident.timeline[0]["actor"] == "john.doe"
    
    def test_incident_duration(self):
        """Test incident duration calculation."""
        created = datetime.utcnow() - timedelta(hours=2)
        resolved = datetime.utcnow()
        
        incident = Incident(
            incident_id="INC-001",
            title="Test",
            description="Test",
            severity=IncidentSeverity.LOW,
            status=IncidentStatus.RESOLVED,
            created_at=created,
            updated_at=datetime.utcnow(),
            resolved_at=resolved
        )
        
        duration = incident.duration()
        assert duration is not None
        assert duration.total_seconds() >= 7200  # At least 2 hours


class TestIncidentManager:
    """Test IncidentManager class."""
    
    @pytest.mark.asyncio
    async def test_process_new_alert(self, incident_manager, sample_alert):
        """Test processing a new alert creates incident."""
        incident = await incident_manager.process_alert(sample_alert)
        
        assert incident is not None
        assert incident.incident_id.startswith("INC-")
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.status == IncidentStatus.DETECTED
        assert len(incident.alerts) == 1
        assert incident.alerts[0].alert_id == sample_alert.alert_id
    
    @pytest.mark.asyncio
    async def test_alert_deduplication(self, incident_manager, sample_alert):
        """Test duplicate alerts are suppressed."""
        # Process first alert
        incident1 = await incident_manager.process_alert(sample_alert)
        assert incident1 is not None
        
        # Process duplicate immediately
        incident2 = await incident_manager.process_alert(sample_alert)
        assert incident2 is None  # Suppressed
    
    @pytest.mark.asyncio
    async def test_alert_correlation(self, incident_manager):
        """Test related alerts are correlated."""
        # Create database alert
        db_alert = Alert(
            alert_id="ALT-001",
            alert_type=AlertType.DATABASE,
            severity=IncidentSeverity.HIGH,
            message="Database connection lost",
            source="db_monitor",
            timestamp=datetime.utcnow()
        )
        
        incident = await incident_manager.process_alert(db_alert)
        assert incident is not None
        initial_id = incident.incident_id
        
        # Create related order failure alert
        order_alert = Alert(
            alert_id="ALT-002",
            alert_type=AlertType.ORDER_FAILURE,
            severity=IncidentSeverity.HIGH,
            message="Orders failing due to DB",
            source="executor",
            timestamp=datetime.utcnow()
        )
        
        incident2 = await incident_manager.process_alert(order_alert)
        
        # Should correlate to same incident
        assert incident2.incident_id == initial_id
        assert len(incident2.alerts) == 2
    
    @pytest.mark.asyncio
    async def test_severity_escalation(self, incident_manager):
        """Test incident severity escalation."""
        # Create medium severity alert
        alert1 = Alert(
            alert_id="ALT-001",
            alert_type=AlertType.PERFORMANCE,
            severity=IncidentSeverity.MEDIUM,
            message="Performance degraded",
            source="monitor",
            timestamp=datetime.utcnow()
        )
        
        incident = await incident_manager.process_alert(alert1)
        assert incident.severity == IncidentSeverity.MEDIUM
        
        # Add critical alert to same incident
        alert2 = Alert(
            alert_id="ALT-002",
            alert_type=AlertType.PERFORMANCE,
            severity=IncidentSeverity.CRITICAL,
            message="Performance critical",
            source="monitor",
            timestamp=datetime.utcnow()
        )
        
        incident2 = await incident_manager.process_alert(alert2)
        
        # Severity should escalate
        assert incident2.incident_id == incident.incident_id
        assert incident2.severity == IncidentSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_pagerduty_integration(self, incident_manager, mock_pagerduty):
        """Test PagerDuty integration for critical alerts."""
        alert = Alert(
            alert_id="ALT-001",
            alert_type=AlertType.SYSTEM_DOWN,
            severity=IncidentSeverity.CRITICAL,
            message="System down",
            source="monitor",
            timestamp=datetime.utcnow()
        )
        
        await incident_manager.process_alert(alert)
        
        # Should trigger PagerDuty for critical system down
        mock_pagerduty.trigger_incident.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acknowledge_incident(self, incident_manager, sample_alert):
        """Test acknowledging incident."""
        incident = await incident_manager.process_alert(sample_alert)
        
        success = await incident_manager.acknowledge_incident(
            incident.incident_id,
            "john.doe"
        )
        
        assert success
        updated = incident_manager.get_incident(incident.incident_id)
        assert updated.status == IncidentStatus.ACKNOWLEDGED
        assert "john.doe" in updated.responders
    
    @pytest.mark.asyncio
    async def test_resolve_incident(self, incident_manager, sample_alert):
        """Test resolving incident."""
        incident = await incident_manager.process_alert(sample_alert)
        
        success = await incident_manager.resolve_incident(
            incident.incident_id,
            resolver="jane.doe",
            resolution="Restarted service",
            root_cause="Memory leak"
        )
        
        assert success
        updated = incident_manager.get_incident(incident.incident_id)
        assert updated.status == IncidentStatus.RESOLVED
        assert updated.resolved_at is not None
        assert updated.root_cause == "Memory leak"
        assert incident.incident_id not in incident_manager.active_incidents
    
    @pytest.mark.asyncio
    async def test_update_status(self, incident_manager, sample_alert):
        """Test updating incident status."""
        incident = await incident_manager.process_alert(sample_alert)
        
        success = await incident_manager.update_status(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            "ops.team",
            "Running diagnostics"
        )
        
        assert success
        updated = incident_manager.get_incident(incident.incident_id)
        assert updated.status == IncidentStatus.INVESTIGATING
    
    def test_get_active_incidents(self, incident_manager):
        """Test getting active incidents."""
        # Create some test incidents
        incident_manager.incidents["INC-001"] = Incident(
            incident_id="INC-001",
            title="Active 1",
            description="Test",
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.ACKNOWLEDGED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        incident_manager.active_incidents.add("INC-001")
        
        incident_manager.incidents["INC-002"] = Incident(
            incident_id="INC-002",
            title="Resolved",
            description="Test",
            severity=IncidentSeverity.LOW,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_at=datetime.utcnow()
        )
        
        active = incident_manager.get_active_incidents()
        assert len(active) == 1
        assert active[0].incident_id == "INC-001"
    
    def test_get_incident_stats(self, incident_manager):
        """Test getting incident statistics."""
        # Create test incidents
        created = datetime.utcnow() - timedelta(hours=1)
        
        incident_manager.incidents["INC-001"] = Incident(
            incident_id="INC-001",
            title="Test",
            description="Test",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.RESOLVED,
            created_at=created,
            updated_at=datetime.utcnow(),
            resolved_at=datetime.utcnow()
        )
        
        incident_manager.incidents["INC-002"] = Incident(
            incident_id="INC-002",
            title="Test",
            description="Test",
            severity=IncidentSeverity.HIGH,
            status=IncidentStatus.ACKNOWLEDGED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        incident_manager.active_incidents.add("INC-002")
        
        stats = incident_manager.get_incident_stats()
        
        assert stats["total_incidents"] == 2
        assert stats["active_incidents"] == 1
        assert stats["resolved_incidents"] == 1
        assert "critical" in stats["severity_breakdown"]
        assert "high" in stats["severity_breakdown"]
    
    @pytest.mark.asyncio
    async def test_cleanup_old_incidents(self, incident_manager):
        """Test cleaning up old incidents."""
        # Create old resolved incident
        old_date = datetime.utcnow() - timedelta(days=35)
        incident_manager.incidents["INC-001"] = Incident(
            incident_id="INC-001",
            title="Old",
            description="Test",
            severity=IncidentSeverity.LOW,
            status=IncidentStatus.RESOLVED,
            created_at=old_date,
            updated_at=old_date,
            resolved_at=old_date
        )
        
        # Create recent incident
        incident_manager.incidents["INC-002"] = Incident(
            incident_id="INC-002",
            title="Recent",
            description="Test",
            severity=IncidentSeverity.LOW,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            resolved_at=datetime.utcnow()
        )
        
        removed = await incident_manager.cleanup_old_incidents(days=30)
        
        assert removed == 1
        assert "INC-001" not in incident_manager.incidents
        assert "INC-002" in incident_manager.incidents
    
    def test_mitigation_steps(self, incident_manager):
        """Test mitigation steps are included."""
        # Test known alert types have mitigation steps
        for alert_type in [AlertType.SYSTEM_DOWN, AlertType.ORDER_FAILURE, 
                          AlertType.DATA_LOSS, AlertType.TILT]:
            steps = incident_manager._get_mitigation_steps(alert_type)
            assert len(steps) > 0
            assert all(isinstance(step, str) for step in steps)