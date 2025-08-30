"""
Integration tests for complete incident management flow.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import tempfile

from genesis.operations.incident_manager import (
    IncidentManager, Alert, AlertType, IncidentSeverity, IncidentStatus
)
from genesis.operations.pagerduty_client import PagerDutyClient
from genesis.operations.chatops import (
    ChatOpsManager, CommandContext, WebhookType
)
from genesis.operations.contact_manager import (
    ContactManager, Contact, ContactRole, OnCallRotation, RotationSchedule
)
from genesis.operations.post_mortem import PostMortemGenerator
from cryptography.fernet import Fernet


class TestIncidentManagementFlow:
    """Test complete incident management flow."""
    
    @pytest.mark.asyncio
    async def test_full_incident_lifecycle(self):
        """Test full incident lifecycle from alert to post-mortem."""
        
        # Setup components
        pagerduty_client = Mock(spec=PagerDutyClient)
        pagerduty_client.trigger_incident = AsyncMock(return_value={"dedup_key": "test"})
        pagerduty_client.resolve_incident = AsyncMock(return_value={"status": "resolved"})
        
        incident_manager = IncidentManager(pagerduty_client=pagerduty_client)
        post_mortem_generator = PostMortemGenerator()
        
        # Step 1: Alert triggers incident
        alert = Alert(
            alert_id="ALT-TEST-001",
            alert_type=AlertType.DATABASE,
            severity=IncidentSeverity.HIGH,
            message="Database connection pool exhausted",
            source="monitoring",
            timestamp=datetime.utcnow()
        )
        
        incident = await incident_manager.process_alert(alert)
        
        assert incident is not None
        assert incident.severity == IncidentSeverity.HIGH
        assert len(incident.alerts) == 1
        
        # Verify PagerDuty was called for high severity
        pagerduty_client.trigger_incident.assert_called_once()
        
        # Step 2: Acknowledge incident
        success = await incident_manager.acknowledge_incident(
            incident.incident_id,
            "john.doe"
        )
        
        assert success
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert "john.doe" in incident.responders
        
        # Step 3: Update status during investigation
        await incident_manager.update_status(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            "john.doe",
            "Checking database connections"
        )
        
        await incident_manager.update_status(
            incident.incident_id,
            IncidentStatus.MITIGATING,
            "jane.smith",
            "Increasing connection pool size"
        )
        
        # Step 4: Resolve incident
        success = await incident_manager.resolve_incident(
            incident.incident_id,
            resolver="jane.smith",
            resolution="Increased connection pool and restarted services",
            root_cause="Connection leak in order processing module"
        )
        
        assert success
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.root_cause is not None
        
        # Verify PagerDuty resolution
        pagerduty_client.resolve_incident.assert_called_once()
        
        # Step 5: Generate post-mortem
        report = await post_mortem_generator.generate_report(
            incident,
            logs=[
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "error",
                    "message": "Connection pool exhausted",
                    "service": "database"
                }
            ],
            metrics={
                "failed_orders": 50,
                "affected_positions": 10
            }
        )
        
        assert report is not None
        assert report.incident_id == incident.incident_id
        assert len(report.action_items) > 0
        assert report.root_cause_analysis.root_cause == incident.root_cause
    
    @pytest.mark.asyncio
    async def test_chatops_incident_interaction(self):
        """Test ChatOps commands for incident management."""
        
        # Setup ChatOps
        chatops = ChatOpsManager(
            webhook_url="https://test.webhook.com",
            webhook_type=WebhookType.SLACK
        )
        
        # Mock incident manager
        incident_manager = IncidentManager()
        
        # Create test incident
        incident = await incident_manager.process_alert(Alert(
            alert_id="ALT-CHAT-001",
            alert_type=AlertType.SYSTEM_DOWN,
            severity=IncidentSeverity.CRITICAL,
            message="System is down",
            source="healthcheck",
            timestamp=datetime.utcnow()
        ))
        
        # Test incident status command
        context = CommandContext(
            user="operator",
            channel="#ops",
            timestamp=datetime.utcnow(),
            raw_message="alerts",
            webhook_type=WebhookType.SLACK,
            metadata={}
        )
        
        # Test health check during incident
        result = await chatops.process_message("health", context)
        assert result.success
        
        # Test emergency stop command
        context.user = "admin"  # Need admin for emergency
        result = await chatops.process_message("emergency", context)
        assert result.success
        assert "EMERGENCY STOP" in result.message
    
    @pytest.mark.asyncio
    async def test_contact_escalation_during_incident(self):
        """Test contact escalation during incident."""
        
        # Setup contact manager
        with tempfile.NamedTemporaryFile(suffix='.enc', delete=False) as f:
            temp_path = f.name
        
        key = Fernet.generate_key()
        contact_manager = ContactManager(
            encryption_key=key.decode(),
            storage_path=temp_path
        )
        
        # Add contacts
        primary = Contact(
            contact_id="PRIMARY",
            name="Primary On-Call",
            role=ContactRole.PRIMARY_ONCALL,
            email="primary@example.com",
            phone="+1234567890"
        )
        secondary = Contact(
            contact_id="SECONDARY",
            name="Secondary On-Call",
            role=ContactRole.SECONDARY_ONCALL,
            email="secondary@example.com"
        )
        lead = Contact(
            contact_id="LEAD",
            name="Engineering Lead",
            role=ContactRole.ENGINEERING_LEAD,
            email="lead@example.com"
        )
        
        contact_manager.add_contact(primary)
        contact_manager.add_contact(secondary)
        contact_manager.add_contact(lead)
        
        # Create rotation
        rotation = OnCallRotation(
            rotation_id="PRIMARY_ROT",
            name="Primary Rotation",
            schedule_type=RotationSchedule.WEEKLY,
            contacts=["PRIMARY", "SECONDARY"],
            start_date=datetime.utcnow()
        )
        contact_manager.create_rotation(rotation)
        
        # Test escalation for critical incident
        notification_list = contact_manager.get_notification_list("critical")
        
        assert len(notification_list) >= 2
        contacts_notified = [contact for contact, _ in notification_list]
        assert any(c.role == ContactRole.PRIMARY_ONCALL for c in contacts_notified)
        assert any(c.role == ContactRole.ENGINEERING_LEAD for c in contacts_notified)
        
        # Test current on-call
        current = contact_manager.get_current_on_call("PRIMARY_ROT")
        assert current is not None
        assert current.contact_id == "PRIMARY"
        
        # Clean up
        import os
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_alert_correlation(self):
        """Test alert correlation into single incident."""
        
        incident_manager = IncidentManager()
        
        # First alert creates incident
        alert1 = Alert(
            alert_id="ALT-CORR-001",
            alert_type=AlertType.DATABASE,
            severity=IncidentSeverity.HIGH,
            message="Database connection issues",
            source="db_monitor",
            timestamp=datetime.utcnow()
        )
        
        incident1 = await incident_manager.process_alert(alert1)
        assert incident1 is not None
        initial_id = incident1.incident_id
        
        # Related alert within correlation window
        alert2 = Alert(
            alert_id="ALT-CORR-002",
            alert_type=AlertType.ORDER_FAILURE,
            severity=IncidentSeverity.HIGH,
            message="Orders failing due to database",
            source="order_processor",
            timestamp=datetime.utcnow() + timedelta(minutes=1)
        )
        
        incident2 = await incident_manager.process_alert(alert2)
        
        # Should correlate to same incident
        assert incident2.incident_id == initial_id
        assert len(incident2.alerts) == 2
        
        # Unrelated alert creates new incident
        alert3 = Alert(
            alert_id="ALT-CORR-003",
            alert_type=AlertType.MEMORY,
            severity=IncidentSeverity.MEDIUM,
            message="High memory usage",
            source="system_monitor",
            timestamp=datetime.utcnow() + timedelta(minutes=2)
        )
        
        incident3 = await incident_manager.process_alert(alert3)
        
        # Should be different incident
        assert incident3.incident_id != initial_id
        assert len(incident3.alerts) == 1
    
    @pytest.mark.asyncio
    async def test_incident_with_runbook(self):
        """Test incident handling with runbook."""
        
        incident_manager = IncidentManager()
        
        # Alert for scenario with runbook
        alert = Alert(
            alert_id="ALT-RB-001",
            alert_type=AlertType.SYSTEM_DOWN,
            severity=IncidentSeverity.CRITICAL,
            message="Complete system outage",
            source="healthcheck",
            timestamp=datetime.utcnow()
        )
        
        incident = await incident_manager.process_alert(alert)
        
        # Check runbook is attached
        assert incident.runbook_url is not None
        assert "scenario-1" in incident.runbook_url.lower()
        
        # Check mitigation steps are provided
        assert len(incident.mitigation_steps) > 0
        assert any("supervisorctl" in step for step in incident.mitigation_steps)
    
    @pytest.mark.asyncio
    async def test_performance_incident_flow(self):
        """Test performance degradation incident flow."""
        
        incident_manager = IncidentManager()
        post_mortem_generator = PostMortemGenerator()
        
        # Performance alert
        alert = Alert(
            alert_id="ALT-PERF-001",
            alert_type=AlertType.PERFORMANCE,
            severity=IncidentSeverity.MEDIUM,
            message="Response times exceeding 1 second",
            source="performance_monitor",
            timestamp=datetime.utcnow(),
            metadata={
                "avg_response_time": 1250,
                "p99_response_time": 3500,
                "cpu_usage": 85
            }
        )
        
        incident = await incident_manager.process_alert(alert)
        
        assert incident.severity == IncidentSeverity.MEDIUM
        assert incident.runbook_url is not None
        
        # Simulate investigation and resolution
        await incident_manager.update_status(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            "sre_team",
            "Running profiler to identify bottlenecks"
        )
        
        await incident_manager.resolve_incident(
            incident.incident_id,
            "sre_team",
            "Optimized database queries and added caching",
            "Inefficient N+1 queries in order listing"
        )
        
        # Generate post-mortem with performance metrics
        report = await post_mortem_generator.generate_report(
            incident,
            metrics={
                "response_time_increase": 450,
                "api_errors": 0,
                "orders_affected": 0
            }
        )
        
        # Check lessons learned include performance
        assert "technical" in report.lessons_learned
        assert any(
            "performance" in lesson.lower() or "optimization" in lesson.lower()
            for lesson in report.lessons_learned.get("technical", []) +
                         report.lessons_learned.get("process", [])
        )
    
    @pytest.mark.asyncio
    async def test_incident_statistics(self):
        """Test incident statistics generation."""
        
        incident_manager = IncidentManager()
        
        # Create multiple incidents
        alert_types = [
            (AlertType.SYSTEM_DOWN, IncidentSeverity.CRITICAL),
            (AlertType.DATABASE, IncidentSeverity.HIGH),
            (AlertType.PERFORMANCE, IncidentSeverity.MEDIUM),
            (AlertType.RATE_LIMIT, IncidentSeverity.MEDIUM),
            (AlertType.CONFIG_DRIFT, IncidentSeverity.LOW)
        ]
        
        for i, (alert_type, severity) in enumerate(alert_types):
            alert = Alert(
                alert_id=f"ALT-STAT-{i:03d}",
                alert_type=alert_type,
                severity=severity,
                message=f"Test alert {i}",
                source="test",
                timestamp=datetime.utcnow() + timedelta(minutes=i)
            )
            
            incident = await incident_manager.process_alert(alert)
            
            # Resolve first 3 incidents
            if i < 3:
                await incident_manager.resolve_incident(
                    incident.incident_id,
                    "test_resolver",
                    f"Resolved test incident {i}",
                    f"Test root cause {i}"
                )
        
        # Get statistics
        stats = incident_manager.get_incident_stats()
        
        assert stats["total_incidents"] == 5
        assert stats["active_incidents"] == 2
        assert stats["resolved_incidents"] == 3
        assert "critical" in stats["severity_breakdown"]
        assert stats["severity_breakdown"]["critical"] == 1
        assert len(stats["active_incident_ids"]) == 2