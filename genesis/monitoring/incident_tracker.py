"""
Incident tracking and management system with PagerDuty integration.

This module provides comprehensive incident lifecycle management including
creation, tracking, acknowledgment, and resolution with MTTA/MTTR metrics.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

import aiohttp
import structlog
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from genesis.core.exceptions import ValidationError
from genesis.security.vault_client import VaultClient

logger = structlog.get_logger(__name__)

Base = declarative_base()


class IncidentStatus(Enum):
    """Incident status states."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class IncidentPriority(Enum):
    """Incident priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentModel(Base):
    """Database model for incidents."""
    __tablename__ = "incidents"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    service = Column(String, nullable=False)
    status = Column(SQLEnum(IncidentStatus), default=IncidentStatus.OPEN)
    priority = Column(SQLEnum(IncidentPriority), default=IncidentPriority.MEDIUM)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    
    created_by = Column(String, default="system")
    acknowledged_by = Column(String, nullable=True)
    resolved_by = Column(String, nullable=True)
    
    pagerduty_id = Column(String, nullable=True)
    alert_ids = Column(Text, default="[]")  # JSON array
    runbook_executions = Column(Text, default="[]")  # JSON array
    
    time_to_acknowledge = Column(Float, nullable=True)  # seconds
    time_to_resolve = Column(Float, nullable=True)  # seconds
    
    notes = Column(Text, default="[]")  # JSON array of notes
    tags = Column(Text, default="[]")  # JSON array


@dataclass
class Incident:
    """In-memory incident representation."""
    id: str
    title: str
    description: str
    service: str
    status: IncidentStatus
    priority: IncidentPriority
    
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    created_by: str = "system"
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    
    pagerduty_id: Optional[str] = None
    alert_ids: List[str] = field(default_factory=list)
    runbook_executions: List[str] = field(default_factory=list)
    
    time_to_acknowledge: Optional[float] = None  # seconds
    time_to_resolve: Optional[float] = None  # seconds
    
    notes: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def calculate_mtta(self) -> Optional[float]:
        """Calculate mean time to acknowledge."""
        if self.acknowledged_at and self.created_at:
            return (self.acknowledged_at - self.created_at).total_seconds()
        return None
    
    def calculate_mttr(self) -> Optional[float]:
        """Calculate mean time to resolve."""
        if self.resolved_at and self.created_at:
            return (self.resolved_at - self.created_at).total_seconds()
        return None


@dataclass
class IncidentMetrics:
    """Incident metrics and statistics."""
    total_incidents: int = 0
    open_incidents: int = 0
    acknowledged_incidents: int = 0
    resolved_incidents: int = 0
    
    incidents_by_priority: Dict[str, int] = field(default_factory=dict)
    incidents_by_service: Dict[str, int] = field(default_factory=dict)
    
    mean_time_to_acknowledge: float = 0.0  # seconds
    mean_time_to_resolve: float = 0.0  # seconds
    
    incidents_last_hour: int = 0
    incidents_last_24h: int = 0
    incidents_last_7d: int = 0
    
    resolution_rate: float = 0.0  # percentage


class IncidentTracker:
    """
    Manages incident lifecycle and integrates with PagerDuty.
    
    Features:
    - Incident creation and tracking
    - PagerDuty synchronization
    - MTTA/MTTR calculation
    - Incident timeline and notes
    - Automated escalation
    """
    
    def __init__(
        self,
        database_session: Optional[Session] = None,
        vault_client: Optional[VaultClient] = None
    ):
        self.db_session = database_session
        self.vault_client = vault_client or VaultClient()
        self.incidents: Dict[str, Incident] = {}
        self._pagerduty_config: Dict[str, Any] = {}
        self._sync_task = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize the incident tracker."""
        await self._load_pagerduty_config()
        await self._load_incidents_from_db()
        
        # Start PagerDuty sync task
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info("Incident tracker initialized", incident_count=len(self.incidents))
    
    async def shutdown(self) -> None:
        """Shutdown the incident tracker."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
    
    async def _load_pagerduty_config(self) -> None:
        """Load PagerDuty configuration from Vault."""
        try:
            self._pagerduty_config = {
                'api_key': await self.vault_client.get_secret('pagerduty/api_key'),
                'routing_key': await self.vault_client.get_secret('pagerduty/routing_key'),
                'service_id': await self.vault_client.get_secret('pagerduty/service_id'),
                'escalation_policy_id': await self.vault_client.get_secret('pagerduty/escalation_policy_id')
            }
        except Exception as e:
            logger.error("Failed to load PagerDuty config", error=str(e))
            # Use mock config for testing
            self._pagerduty_config = {
                'api_key': 'mock_api_key',
                'routing_key': 'mock_routing_key',
                'service_id': 'mock_service_id',
                'escalation_policy_id': 'mock_policy_id'
            }
    
    async def _load_incidents_from_db(self) -> None:
        """Load active incidents from database."""
        if not self.db_session:
            return
        
        try:
            # Load open and acknowledged incidents
            active_incidents = self.db_session.query(IncidentModel).filter(
                IncidentModel.status.in_([IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED, IncidentStatus.IN_PROGRESS])
            ).all()
            
            for db_incident in active_incidents:
                incident = self._db_to_incident(db_incident)
                self.incidents[incident.id] = incident
                
        except Exception as e:
            logger.error("Failed to load incidents from database", error=str(e))
    
    def _db_to_incident(self, db_incident: IncidentModel) -> Incident:
        """Convert database model to incident object."""
        return Incident(
            id=db_incident.id,
            title=db_incident.title,
            description=db_incident.description,
            service=db_incident.service,
            status=db_incident.status,
            priority=db_incident.priority,
            created_at=db_incident.created_at,
            acknowledged_at=db_incident.acknowledged_at,
            resolved_at=db_incident.resolved_at,
            closed_at=db_incident.closed_at,
            created_by=db_incident.created_by,
            acknowledged_by=db_incident.acknowledged_by,
            resolved_by=db_incident.resolved_by,
            pagerduty_id=db_incident.pagerduty_id,
            alert_ids=json.loads(db_incident.alert_ids) if db_incident.alert_ids else [],
            runbook_executions=json.loads(db_incident.runbook_executions) if db_incident.runbook_executions else [],
            time_to_acknowledge=db_incident.time_to_acknowledge,
            time_to_resolve=db_incident.time_to_resolve,
            notes=json.loads(db_incident.notes) if db_incident.notes else [],
            tags=json.loads(db_incident.tags) if db_incident.tags else []
        )
    
    async def create_incident(
        self,
        title: str,
        description: str,
        service: str,
        priority: IncidentPriority = IncidentPriority.MEDIUM,
        alert_ids: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> str:
        """
        Create a new incident.
        
        Args:
            title: Incident title
            description: Incident description
            service: Affected service
            priority: Incident priority
            alert_ids: Related alert IDs
            created_by: Creator identifier
            
        Returns:
            Incident ID
        """
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        
        incident = Incident(
            id=incident_id,
            title=title,
            description=description,
            service=service,
            status=IncidentStatus.OPEN,
            priority=priority,
            created_at=datetime.utcnow(),
            created_by=created_by,
            alert_ids=alert_ids or []
        )
        
        # Store in memory
        self.incidents[incident_id] = incident
        
        # Persist to database
        if self.db_session:
            await self._save_to_db(incident)
        
        # Create PagerDuty incident
        pagerduty_id = await self._create_pagerduty_incident(incident)
        if pagerduty_id:
            incident.pagerduty_id = pagerduty_id
            if self.db_session:
                await self._update_db(incident)
        
        logger.info(
            "Created incident",
            incident_id=incident_id,
            service=service,
            priority=priority.value
        )
        
        return incident_id
    
    async def acknowledge_incident(
        self,
        incident_id: str,
        acknowledged_by: str = "operator"
    ) -> bool:
        """
        Acknowledge an incident.
        
        Args:
            incident_id: Incident ID
            acknowledged_by: Person acknowledging
            
        Returns:
            Success status
        """
        if incident_id not in self.incidents:
            raise ValidationError(f"Incident not found: {incident_id}")
        
        incident = self.incidents[incident_id]
        
        if incident.status != IncidentStatus.OPEN:
            logger.warning(
                "Cannot acknowledge incident in current state",
                incident_id=incident_id,
                status=incident.status.value
            )
            return False
        
        now = datetime.utcnow()
        incident.status = IncidentStatus.ACKNOWLEDGED
        incident.acknowledged_at = now
        incident.acknowledged_by = acknowledged_by
        incident.time_to_acknowledge = (now - incident.created_at).total_seconds()
        
        # Add note
        incident.notes.append({
            "timestamp": now.isoformat(),
            "author": acknowledged_by,
            "action": "acknowledged",
            "note": f"Incident acknowledged by {acknowledged_by}"
        })
        
        # Update database
        if self.db_session:
            await self._update_db(incident)
        
        # Update PagerDuty
        if incident.pagerduty_id:
            await self._acknowledge_pagerduty_incident(incident.pagerduty_id, acknowledged_by)
        
        logger.info(
            "Incident acknowledged",
            incident_id=incident_id,
            acknowledged_by=acknowledged_by,
            time_to_acknowledge=incident.time_to_acknowledge
        )
        
        return True
    
    async def resolve_incident(
        self,
        incident_id: str,
        resolution_notes: str,
        resolved_by: str = "operator"
    ) -> bool:
        """
        Resolve an incident.
        
        Args:
            incident_id: Incident ID
            resolution_notes: Resolution description
            resolved_by: Person resolving
            
        Returns:
            Success status
        """
        if incident_id not in self.incidents:
            raise ValidationError(f"Incident not found: {incident_id}")
        
        incident = self.incidents[incident_id]
        
        if incident.status == IncidentStatus.RESOLVED:
            logger.warning("Incident already resolved", incident_id=incident_id)
            return False
        
        now = datetime.utcnow()
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = now
        incident.resolved_by = resolved_by
        incident.time_to_resolve = (now - incident.created_at).total_seconds()
        
        # Add resolution note
        incident.notes.append({
            "timestamp": now.isoformat(),
            "author": resolved_by,
            "action": "resolved",
            "note": resolution_notes
        })
        
        # Update database
        if self.db_session:
            await self._update_db(incident)
        
        # Resolve in PagerDuty
        if incident.pagerduty_id:
            await self._resolve_pagerduty_incident(incident.pagerduty_id, resolved_by)
        
        logger.info(
            "Incident resolved",
            incident_id=incident_id,
            resolved_by=resolved_by,
            time_to_resolve=incident.time_to_resolve
        )
        
        return True
    
    async def add_note(
        self,
        incident_id: str,
        note: str,
        author: str = "operator"
    ) -> bool:
        """Add a note to an incident."""
        if incident_id not in self.incidents:
            raise ValidationError(f"Incident not found: {incident_id}")
        
        incident = self.incidents[incident_id]
        
        incident.notes.append({
            "timestamp": datetime.utcnow().isoformat(),
            "author": author,
            "action": "note",
            "note": note
        })
        
        # Update database
        if self.db_session:
            await self._update_db(incident)
        
        return True
    
    async def add_runbook_execution(
        self,
        incident_id: str,
        runbook_id: str,
        execution_id: str
    ) -> bool:
        """Link a runbook execution to an incident."""
        if incident_id not in self.incidents:
            raise ValidationError(f"Incident not found: {incident_id}")
        
        incident = self.incidents[incident_id]
        incident.runbook_executions.append(execution_id)
        
        # Add note
        incident.notes.append({
            "timestamp": datetime.utcnow().isoformat(),
            "author": "system",
            "action": "runbook_executed",
            "note": f"Executed runbook: {runbook_id} (execution: {execution_id})"
        })
        
        # Update database
        if self.db_session:
            await self._update_db(incident)
        
        return True
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self.incidents.get(incident_id)
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active (non-closed) incidents."""
        return [
            incident for incident in self.incidents.values()
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED, IncidentStatus.CANCELLED]
        ]
    
    def get_incidents_by_service(self, service: str) -> List[Incident]:
        """Get incidents for a specific service."""
        return [
            incident for incident in self.incidents.values()
            if incident.service == service
        ]
    
    def calculate_metrics(self) -> IncidentMetrics:
        """Calculate incident metrics."""
        metrics = IncidentMetrics()
        
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        mtta_times = []
        mttr_times = []
        
        for incident in self.incidents.values():
            # Count by status
            if incident.status == IncidentStatus.OPEN:
                metrics.open_incidents += 1
            elif incident.status == IncidentStatus.ACKNOWLEDGED:
                metrics.acknowledged_incidents += 1
            elif incident.status == IncidentStatus.RESOLVED:
                metrics.resolved_incidents += 1
            
            # Count by priority
            priority_key = incident.priority.value
            metrics.incidents_by_priority[priority_key] = metrics.incidents_by_priority.get(priority_key, 0) + 1
            
            # Count by service
            metrics.incidents_by_service[incident.service] = metrics.incidents_by_service.get(incident.service, 0) + 1
            
            # Time-based counts
            if incident.created_at > hour_ago:
                metrics.incidents_last_hour += 1
            if incident.created_at > day_ago:
                metrics.incidents_last_24h += 1
            if incident.created_at > week_ago:
                metrics.incidents_last_7d += 1
            
            # MTTA/MTTR
            if incident.time_to_acknowledge:
                mtta_times.append(incident.time_to_acknowledge)
            if incident.time_to_resolve:
                mttr_times.append(incident.time_to_resolve)
        
        metrics.total_incidents = len(self.incidents)
        
        # Calculate averages
        if mtta_times:
            metrics.mean_time_to_acknowledge = sum(mtta_times) / len(mtta_times)
        if mttr_times:
            metrics.mean_time_to_resolve = sum(mttr_times) / len(mttr_times)
        
        # Resolution rate
        if metrics.total_incidents > 0:
            metrics.resolution_rate = (metrics.resolved_incidents / metrics.total_incidents) * 100
        
        return metrics
    
    # PagerDuty Integration Methods
    
    async def _create_pagerduty_incident(self, incident: Incident) -> Optional[str]:
        """Create incident in PagerDuty."""
        try:
            payload = {
                "incident": {
                    "type": "incident",
                    "title": incident.title,
                    "service": {
                        "id": self._pagerduty_config.get('service_id'),
                        "type": "service_reference"
                    },
                    "urgency": self._map_priority_to_urgency(incident.priority),
                    "body": {
                        "type": "incident_body",
                        "details": incident.description
                    },
                    "escalation_policy": {
                        "id": self._pagerduty_config.get('escalation_policy_id'),
                        "type": "escalation_policy_reference"
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token token={self._pagerduty_config.get('api_key')}",
                    "Accept": "application/vnd.pagerduty+json;version=2",
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    "https://api.pagerduty.com/incidents",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 201:
                        data = await response.json()
                        return data["incident"]["id"]
                    else:
                        logger.error(
                            "Failed to create PagerDuty incident",
                            status=response.status,
                            response=await response.text()
                        )
                        
        except Exception as e:
            logger.error("Error creating PagerDuty incident", error=str(e))
        
        return None
    
    async def _acknowledge_pagerduty_incident(self, pagerduty_id: str, acknowledged_by: str) -> bool:
        """Acknowledge incident in PagerDuty."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token token={self._pagerduty_config.get('api_key')}",
                    "Accept": "application/vnd.pagerduty+json;version=2",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "incident": {
                        "type": "incident",
                        "status": "acknowledged"
                    }
                }
                
                async with session.put(
                    f"https://api.pagerduty.com/incidents/{pagerduty_id}",
                    json=payload,
                    headers=headers
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error("Error acknowledging PagerDuty incident", error=str(e))
        
        return False
    
    async def _resolve_pagerduty_incident(self, pagerduty_id: str, resolved_by: str) -> bool:
        """Resolve incident in PagerDuty."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Token token={self._pagerduty_config.get('api_key')}",
                    "Accept": "application/vnd.pagerduty+json;version=2",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "incident": {
                        "type": "incident",
                        "status": "resolved"
                    }
                }
                
                async with session.put(
                    f"https://api.pagerduty.com/incidents/{pagerduty_id}",
                    json=payload,
                    headers=headers
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error("Error resolving PagerDuty incident", error=str(e))
        
        return False
    
    def _map_priority_to_urgency(self, priority: IncidentPriority) -> str:
        """Map incident priority to PagerDuty urgency."""
        mapping = {
            IncidentPriority.CRITICAL: "high",
            IncidentPriority.HIGH: "high",
            IncidentPriority.MEDIUM: "low",
            IncidentPriority.LOW: "low",
            IncidentPriority.INFO: "low"
        }
        return mapping.get(priority, "low")
    
    async def _sync_loop(self) -> None:
        """Background task to sync with PagerDuty."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Sync every minute
                await self._sync_with_pagerduty()
            except Exception as e:
                logger.error("Error in PagerDuty sync loop", error=str(e))
    
    async def _sync_with_pagerduty(self) -> None:
        """Sync incident status with PagerDuty."""
        # This would fetch incidents from PagerDuty API and update local state
        # For now, this is a placeholder
        pass
    
    async def _save_to_db(self, incident: Incident) -> None:
        """Save incident to database."""
        if not self.db_session:
            return
        
        try:
            db_incident = IncidentModel(
                id=incident.id,
                title=incident.title,
                description=incident.description,
                service=incident.service,
                status=incident.status,
                priority=incident.priority,
                created_at=incident.created_at,
                created_by=incident.created_by,
                pagerduty_id=incident.pagerduty_id,
                alert_ids=json.dumps(incident.alert_ids),
                runbook_executions=json.dumps(incident.runbook_executions),
                notes=json.dumps(incident.notes),
                tags=json.dumps(incident.tags)
            )
            
            self.db_session.add(db_incident)
            self.db_session.commit()
            
        except Exception as e:
            logger.error("Failed to save incident to database", error=str(e))
            self.db_session.rollback()
    
    async def _update_db(self, incident: Incident) -> None:
        """Update incident in database."""
        if not self.db_session:
            return
        
        try:
            db_incident = self.db_session.query(IncidentModel).filter_by(id=incident.id).first()
            if db_incident:
                db_incident.status = incident.status
                db_incident.acknowledged_at = incident.acknowledged_at
                db_incident.resolved_at = incident.resolved_at
                db_incident.closed_at = incident.closed_at
                db_incident.acknowledged_by = incident.acknowledged_by
                db_incident.resolved_by = incident.resolved_by
                db_incident.time_to_acknowledge = incident.time_to_acknowledge
                db_incident.time_to_resolve = incident.time_to_resolve
                db_incident.pagerduty_id = incident.pagerduty_id
                db_incident.alert_ids = json.dumps(incident.alert_ids)
                db_incident.runbook_executions = json.dumps(incident.runbook_executions)
                db_incident.notes = json.dumps(incident.notes)
                db_incident.tags = json.dumps(incident.tags)
                
                self.db_session.commit()
                
        except Exception as e:
            logger.error("Failed to update incident in database", error=str(e))
            self.db_session.rollback()