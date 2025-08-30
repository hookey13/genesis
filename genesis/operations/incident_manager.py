"""
Automated incident management system with alert processing and incident creation.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import structlog

from genesis.core.events import Event, EventType

logger = structlog.get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    CRITICAL = "critical"  # <5 min response, money at risk
    HIGH = "high"  # <15 min response, service degraded
    MEDIUM = "medium"  # <1 hour response, performance issue
    LOW = "low"  # <4 hours response, minor issue


class IncidentStatus(Enum):
    """Incident lifecycle status."""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    POST_MORTEM = "post_mortem"


class AlertType(Enum):
    """Types of alerts that can trigger incidents."""
    SYSTEM_DOWN = "system_down"
    ORDER_FAILURE = "order_failure"
    DATA_LOSS = "data_loss"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TILT = "tilt"
    RATE_LIMIT = "rate_limit"
    POSITION_VIOLATION = "position_violation"
    NETWORK = "network"
    DATABASE = "database"
    DISK_SPACE = "disk_space"
    MEMORY = "memory"
    AUTHENTICATION = "authentication"
    BACKUP_FAILURE = "backup_failure"
    CONFIG_DRIFT = "config_drift"
    MARKET_EMERGENCY = "market_emergency"


@dataclass
class Alert:
    """Alert that may trigger an incident."""
    alert_id: str
    alert_type: AlertType
    severity: IncidentSeverity
    message: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        data = f"{self.alert_type.value}:{self.source}:{self.message}"
        return hashlib.md5(data.encode()).hexdigest()


@dataclass
class Incident:
    """Incident requiring response."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    alerts: List[Alert] = field(default_factory=list)
    responders: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    runbook_url: Optional[str] = None
    mitigation_steps: List[str] = field(default_factory=list)
    impact: Optional[str] = None
    root_cause: Optional[str] = None
    
    def add_timeline_entry(self, action: str, actor: str, details: Optional[str] = None):
        """Add entry to incident timeline."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "actor": actor,
            "details": details
        }
        self.timeline.append(entry)
        self.updated_at = datetime.utcnow()
    
    def duration(self) -> Optional[timedelta]:
        """Calculate incident duration."""
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return datetime.utcnow() - self.created_at


class IncidentManager:
    """Manages incident lifecycle from alert to resolution."""
    
    # Alert to incident mapping rules
    ALERT_RULES = {
        AlertType.SYSTEM_DOWN: {
            "severity": IncidentSeverity.CRITICAL,
            "title": "Complete System Outage",
            "runbook": "docs/runbooks/incident-response.md#scenario-1-complete-system-outage",
            "auto_page": True
        },
        AlertType.ORDER_FAILURE: {
            "severity": IncidentSeverity.HIGH,
            "title": "Order Execution Failure",
            "runbook": "docs/runbooks/incident-response.md#scenario-2-order-execution-failure",
            "auto_page": True
        },
        AlertType.DATA_LOSS: {
            "severity": IncidentSeverity.HIGH,
            "title": "Market Data Feed Loss",
            "runbook": "docs/runbooks/incident-response.md#scenario-3-market-data-feed-loss",
            "auto_page": True
        },
        AlertType.DATABASE: {
            "severity": IncidentSeverity.HIGH,
            "title": "Database Connection Lost",
            "runbook": "docs/runbooks/incident-response.md#scenario-4-database-connection-lost",
            "auto_page": True
        },
        AlertType.MEMORY: {
            "severity": IncidentSeverity.MEDIUM,
            "title": "High Memory Usage Detected",
            "runbook": "docs/runbooks/incident-response.md#scenario-5-memory-leak--high-memory-usage",
            "auto_page": False
        },
        AlertType.TILT: {
            "severity": IncidentSeverity.HIGH,
            "title": "Tilt Detection Triggered",
            "runbook": "docs/runbooks/incident-response.md#scenario-6-tilt-detection-triggered",
            "auto_page": True
        },
        AlertType.RATE_LIMIT: {
            "severity": IncidentSeverity.MEDIUM,
            "title": "API Rate Limit Exceeded",
            "runbook": "docs/runbooks/incident-response.md#scenario-7-api-rate-limit-exceeded",
            "auto_page": False
        },
        AlertType.POSITION_VIOLATION: {
            "severity": IncidentSeverity.HIGH,
            "title": "Position Size Violation",
            "runbook": "docs/runbooks/incident-response.md#scenario-8-position-size-violation",
            "auto_page": True
        },
        AlertType.DISK_SPACE: {
            "severity": IncidentSeverity.MEDIUM,
            "title": "Disk Space Exhaustion",
            "runbook": "docs/runbooks/incident-response.md#scenario-10-disk-space-exhaustion",
            "auto_page": False
        },
        AlertType.NETWORK: {
            "severity": IncidentSeverity.HIGH,
            "title": "Network Partition Detected",
            "runbook": "docs/runbooks/incident-response.md#scenario-11-network-partition",
            "auto_page": True
        },
        AlertType.AUTHENTICATION: {
            "severity": IncidentSeverity.HIGH,
            "title": "Authentication Failure",
            "runbook": "docs/runbooks/incident-response.md#scenario-14-authentication-failure",
            "auto_page": True
        },
        AlertType.BACKUP_FAILURE: {
            "severity": IncidentSeverity.MEDIUM,
            "title": "Backup Failure Detected",
            "runbook": "docs/runbooks/incident-response.md#scenario-16-backup-failure",
            "auto_page": False
        },
        AlertType.PERFORMANCE: {
            "severity": IncidentSeverity.MEDIUM,
            "title": "Performance Degradation",
            "runbook": "docs/runbooks/incident-response.md#scenario-17-performance-degradation",
            "auto_page": False
        },
        AlertType.CONFIG_DRIFT: {
            "severity": IncidentSeverity.LOW,
            "title": "Configuration Drift Detected",
            "runbook": "docs/runbooks/incident-response.md#scenario-19-configuration-drift",
            "auto_page": False
        },
        AlertType.MARKET_EMERGENCY: {
            "severity": IncidentSeverity.CRITICAL,
            "title": "Emergency Market Conditions",
            "runbook": "docs/runbooks/incident-response.md#scenario-20-emergency-market-conditions",
            "auto_page": True
        }
    }
    
    def __init__(self, pagerduty_client=None):
        """Initialize incident manager."""
        self.incidents: Dict[str, Incident] = {}
        self.active_incidents: Set[str] = set()
        self.alert_dedup: Dict[str, datetime] = {}  # Fingerprint -> last seen
        self.pagerduty_client = pagerduty_client
        self.correlation_window = timedelta(minutes=5)
        self._incident_counter = 0
        
        logger.info("Incident manager initialized")
    
    async def process_alert(self, alert: Alert) -> Optional[Incident]:
        """Process incoming alert and create/update incident if needed."""
        try:
            # Check for duplicate alerts
            fingerprint = alert.fingerprint()
            if fingerprint in self.alert_dedup:
                last_seen = self.alert_dedup[fingerprint]
                if datetime.utcnow() - last_seen < timedelta(minutes=1):
                    logger.debug("Suppressing duplicate alert", 
                               alert_id=alert.alert_id,
                               fingerprint=fingerprint)
                    return None
            
            self.alert_dedup[fingerprint] = datetime.utcnow()
            
            # Check if alert correlates with existing incident
            incident = self._correlate_alert(alert)
            
            if incident:
                # Add alert to existing incident
                incident.alerts.append(alert)
                incident.add_timeline_entry(
                    action="alert_correlated",
                    actor="system",
                    details=f"Alert {alert.alert_id} correlated to incident"
                )
                
                # Potentially escalate severity
                if alert.severity.value < incident.severity.value:
                    incident.severity = alert.severity
                    incident.add_timeline_entry(
                        action="severity_escalated",
                        actor="system",
                        details=f"Severity escalated to {alert.severity.value}"
                    )
                
                logger.info("Alert correlated to existing incident",
                          alert_id=alert.alert_id,
                          incident_id=incident.incident_id)
            else:
                # Create new incident
                incident = await self._create_incident(alert)
                logger.info("New incident created",
                          incident_id=incident.incident_id,
                          alert_id=alert.alert_id)
            
            # Send to PagerDuty if needed
            if self._should_page(alert) and self.pagerduty_client:
                await self._send_page(incident)
            
            return incident
            
        except Exception as e:
            logger.error("Failed to process alert",
                        alert_id=alert.alert_id,
                        error=str(e))
            raise
    
    def _correlate_alert(self, alert: Alert) -> Optional[Incident]:
        """Check if alert correlates with existing incident."""
        for incident_id in self.active_incidents:
            incident = self.incidents[incident_id]
            
            # Same alert type within correlation window
            for existing_alert in incident.alerts:
                if (existing_alert.alert_type == alert.alert_type and
                    alert.timestamp - existing_alert.timestamp < self.correlation_window):
                    return incident
            
            # Related alert types (e.g., database issues often cause order failures)
            if self._are_alerts_related(alert, incident.alerts):
                return incident
        
        return None
    
    def _are_alerts_related(self, new_alert: Alert, existing_alerts: List[Alert]) -> bool:
        """Determine if alerts are related."""
        related_groups = [
            {AlertType.DATABASE, AlertType.ORDER_FAILURE, AlertType.DATA_LOSS},
            {AlertType.NETWORK, AlertType.AUTHENTICATION, AlertType.DATA_LOSS},
            {AlertType.MEMORY, AlertType.PERFORMANCE, AlertType.SYSTEM_DOWN},
            {AlertType.RATE_LIMIT, AlertType.ORDER_FAILURE},
            {AlertType.TILT, AlertType.POSITION_VIOLATION}
        ]
        
        for group in related_groups:
            if new_alert.alert_type in group:
                for existing in existing_alerts:
                    if existing.alert_type in group:
                        time_diff = abs(new_alert.timestamp - existing.timestamp)
                        if time_diff < self.correlation_window:
                            return True
        
        return False
    
    async def _create_incident(self, alert: Alert) -> Incident:
        """Create new incident from alert."""
        self._incident_counter += 1
        incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{self._incident_counter:04d}"
        
        rule = self.ALERT_RULES.get(alert.alert_type, {})
        
        incident = Incident(
            incident_id=incident_id,
            title=rule.get("title", f"Incident: {alert.alert_type.value}"),
            description=alert.message,
            severity=rule.get("severity", alert.severity),
            status=IncidentStatus.DETECTED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            alerts=[alert],
            runbook_url=rule.get("runbook"),
            mitigation_steps=self._get_mitigation_steps(alert.alert_type)
        )
        
        incident.add_timeline_entry(
            action="incident_created",
            actor="system",
            details=f"Incident created from alert {alert.alert_id}"
        )
        
        self.incidents[incident_id] = incident
        self.active_incidents.add(incident_id)
        
        # Emit incident created event
        await self._emit_incident_event(incident, "created")
        
        return incident
    
    def _get_mitigation_steps(self, alert_type: AlertType) -> List[str]:
        """Get mitigation steps for alert type."""
        steps_map = {
            AlertType.SYSTEM_DOWN: [
                "Check process status: supervisorctl status genesis",
                "Review logs: tail -n 100 /var/log/genesis/trading.log",
                "Restart if needed: supervisorctl restart genesis"
            ],
            AlertType.ORDER_FAILURE: [
                "Check exchange connectivity",
                "Verify API credentials",
                "Check rate limits",
                "Reset circuit breaker if needed"
            ],
            AlertType.DATA_LOSS: [
                "Check WebSocket status",
                "Test network connectivity",
                "Restart WebSocket connections",
                "Switch to REST fallback if needed"
            ],
            AlertType.DATABASE: [
                "Check database status",
                "Verify connection string",
                "Restart database if needed",
                "Restore from backup if corrupted"
            ],
            AlertType.MEMORY: [
                "Check memory usage",
                "Generate heap dump for analysis",
                "Perform graceful restart",
                "Analyze heap dump post-restart"
            ],
            AlertType.TILT: [
                "Check tilt status",
                "Review recent trades",
                "Apply cooling period",
                "Reduce position limits",
                "Notify trader"
            ],
            AlertType.MARKET_EMERGENCY: [
                "Activate emergency mode",
                "Close all positions",
                "Cancel pending orders",
                "Enable safe mode",
                "Wait for market normalization"
            ]
        }
        
        return steps_map.get(alert_type, ["Review runbook for detailed steps"])
    
    def _should_page(self, alert: Alert) -> bool:
        """Determine if alert should trigger paging."""
        rule = self.ALERT_RULES.get(alert.alert_type, {})
        return rule.get("auto_page", False)
    
    async def _send_page(self, incident: Incident):
        """Send page via PagerDuty."""
        if not self.pagerduty_client:
            logger.warning("PagerDuty client not configured, skipping page")
            return
        
        try:
            await self.pagerduty_client.trigger_incident(
                incident_key=incident.incident_id,
                summary=incident.title,
                severity=incident.severity.value,
                details={
                    "description": incident.description,
                    "runbook_url": incident.runbook_url,
                    "mitigation_steps": incident.mitigation_steps,
                    "alerts": [
                        {
                            "id": a.alert_id,
                            "type": a.alert_type.value,
                            "message": a.message,
                            "timestamp": a.timestamp.isoformat()
                        }
                        for a in incident.alerts
                    ]
                }
            )
            
            incident.add_timeline_entry(
                action="pagerduty_triggered",
                actor="system",
                details="PagerDuty incident triggered"
            )
            
        except Exception as e:
            logger.error("Failed to send PagerDuty page",
                        incident_id=incident.incident_id,
                        error=str(e))
    
    async def acknowledge_incident(self, incident_id: str, responder: str) -> bool:
        """Acknowledge incident."""
        if incident_id not in self.incidents:
            logger.error("Incident not found", incident_id=incident_id)
            return False
        
        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.ACKNOWLEDGED
        incident.responders.append(responder)
        incident.add_timeline_entry(
            action="acknowledged",
            actor=responder,
            details="Incident acknowledged"
        )
        
        await self._emit_incident_event(incident, "acknowledged")
        
        logger.info("Incident acknowledged",
                   incident_id=incident_id,
                   responder=responder)
        return True
    
    async def update_status(self, incident_id: str, status: IncidentStatus, 
                           actor: str, notes: Optional[str] = None) -> bool:
        """Update incident status."""
        if incident_id not in self.incidents:
            logger.error("Incident not found", incident_id=incident_id)
            return False
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = status
        
        if status == IncidentStatus.RESOLVED:
            incident.resolved_at = datetime.utcnow()
            self.active_incidents.discard(incident_id)
        
        incident.add_timeline_entry(
            action=f"status_changed",
            actor=actor,
            details=f"Status changed from {old_status.value} to {status.value}. {notes or ''}"
        )
        
        await self._emit_incident_event(incident, "status_changed")
        
        logger.info("Incident status updated",
                   incident_id=incident_id,
                   old_status=old_status.value,
                   new_status=status.value)
        return True
    
    async def resolve_incident(self, incident_id: str, resolver: str, 
                              resolution: str, root_cause: Optional[str] = None) -> bool:
        """Resolve incident."""
        if incident_id not in self.incidents:
            logger.error("Incident not found", incident_id=incident_id)
            return False
        
        incident = self.incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.utcnow()
        incident.root_cause = root_cause
        incident.add_timeline_entry(
            action="resolved",
            actor=resolver,
            details=f"Resolution: {resolution}"
        )
        
        self.active_incidents.discard(incident_id)
        
        # Resolve in PagerDuty if configured
        if self.pagerduty_client:
            await self.pagerduty_client.resolve_incident(incident_id)
        
        await self._emit_incident_event(incident, "resolved")
        
        logger.info("Incident resolved",
                   incident_id=incident_id,
                   resolver=resolver,
                   duration=str(incident.duration()))
        return True
    
    async def _emit_incident_event(self, incident: Incident, action: str):
        """Emit incident event for audit trail."""
        event = Event(
            event_type=EventType.AUDIT_LOG_CREATED,
            event_data={
                "incident_id": incident.incident_id,
                "action": action,
                "title": incident.title,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "alert_count": len(incident.alerts)
            }
        )
        # Event would be published to event bus here
        logger.info(f"Incident event: {action}", incident_id=incident.incident_id)
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self.incidents.get(incident_id)
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents."""
        return [self.incidents[id] for id in self.active_incidents]
    
    def get_incident_stats(self) -> Dict[str, Any]:
        """Get incident statistics."""
        total = len(self.incidents)
        active = len(self.active_incidents)
        resolved = total - active
        
        if resolved > 0:
            mttr_sum = timedelta()
            for incident in self.incidents.values():
                if incident.resolved_at:
                    mttr_sum += incident.duration()
            mttr = mttr_sum / resolved
        else:
            mttr = timedelta()
        
        severity_counts = {}
        for incident in self.incidents.values():
            sev = incident.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            "total_incidents": total,
            "active_incidents": active,
            "resolved_incidents": resolved,
            "mean_time_to_resolve": str(mttr),
            "severity_breakdown": severity_counts,
            "active_incident_ids": list(self.active_incidents)
        }
    
    async def cleanup_old_incidents(self, days: int = 30):
        """Clean up old resolved incidents."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        to_remove = []
        
        for incident_id, incident in self.incidents.items():
            if (incident.status == IncidentStatus.RESOLVED and
                incident.resolved_at and
                incident.resolved_at < cutoff):
                to_remove.append(incident_id)
        
        for incident_id in to_remove:
            del self.incidents[incident_id]
            logger.info("Cleaned up old incident", incident_id=incident_id)
        
        return len(to_remove)