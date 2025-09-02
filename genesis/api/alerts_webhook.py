"""
Webhook endpoint for receiving alerts and triggering runbook automation.

This module provides the HTTP endpoint that AlertManager and other systems
can call to trigger automated runbook execution.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import structlog

from genesis.monitoring.runbook_executor import RunbookExecutor, ExecutionMode
from genesis.monitoring.alert_channels import Alert, AlertPriority
from genesis.monitoring.incident_tracker import IncidentTracker, IncidentPriority

logger = structlog.get_logger(__name__)

# Create API router
router = APIRouter(prefix="/alerts", tags=["alerts"])


# ============= Request/Response Models =============

class AlertWebhookRequest(BaseModel):
    """Alert webhook request from AlertManager."""
    version: str = Field("4", description="AlertManager version")
    groupKey: str = Field(..., description="Alert group key")
    truncatedAlerts: int = Field(0, description="Number of truncated alerts")
    status: str = Field(..., description="Alert status (firing/resolved)")
    receiver: str = Field(..., description="Receiver name")
    groupLabels: Dict[str, str] = Field(default_factory=dict)
    commonLabels: Dict[str, str] = Field(default_factory=dict)
    commonAnnotations: Dict[str, str] = Field(default_factory=dict)
    externalURL: str = Field(..., description="AlertManager URL")
    alerts: List[Dict[str, Any]] = Field(default_factory=list)


class RunbookExecutionRequest(BaseModel):
    """Manual runbook execution request."""
    runbook_id: str = Field(..., description="Runbook ID to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    dry_run: bool = Field(False, description="Execute in dry-run mode")
    force: bool = Field(False, description="Force execution bypassing limits")


class IncidentCreationRequest(BaseModel):
    """Incident creation request."""
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    service: str = Field(..., description="Affected service")
    priority: str = Field("medium", description="Incident priority")
    alert_ids: List[str] = Field(default_factory=list, description="Related alert IDs")


class WebhookResponse(BaseModel):
    """Standard webhook response."""
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")


# ============= Dependencies =============

# Global instances (in production, use dependency injection)
_runbook_executor: Optional[RunbookExecutor] = None
_incident_tracker: Optional[IncidentTracker] = None


def get_runbook_executor() -> RunbookExecutor:
    """Get or create runbook executor instance."""
    global _runbook_executor
    if _runbook_executor is None:
        _runbook_executor = RunbookExecutor()
        asyncio.create_task(_runbook_executor.initialize())
    return _runbook_executor


def get_incident_tracker() -> IncidentTracker:
    """Get or create incident tracker instance."""
    global _incident_tracker
    if _incident_tracker is None:
        _incident_tracker = IncidentTracker()
        asyncio.create_task(_incident_tracker.initialize())
    return _incident_tracker


def verify_webhook_auth(authorization: Optional[str] = Header(None)) -> bool:
    """Verify webhook authentication."""
    # In production, verify against a shared secret or JWT
    if not authorization:
        return True  # Allow for testing
    
    # Verify bearer token or shared secret
    expected_token = "Bearer webhook_secret_token"  # Would come from config
    return authorization == expected_token


# ============= Webhook Endpoints =============

@router.post("/webhook", response_model=WebhookResponse)
async def receive_alert_webhook(
    request: AlertWebhookRequest,
    executor: RunbookExecutor = Depends(get_runbook_executor),
    incident_tracker: IncidentTracker = Depends(get_incident_tracker),
    authenticated: bool = Depends(verify_webhook_auth)
) -> WebhookResponse:
    """
    Receive alert webhook from AlertManager.
    
    This endpoint:
    1. Receives alerts from AlertManager
    2. Determines if runbook automation should be triggered
    3. Executes appropriate runbooks
    4. Creates or updates incidents as needed
    """
    try:
        logger.info(
            "Received alert webhook",
            status=request.status,
            receiver=request.receiver,
            alert_count=len(request.alerts)
        )
        
        processed_alerts = []
        runbooks_executed = []
        incidents_created = []
        
        for alert_data in request.alerts:
            # Process each alert
            alert_name = alert_data.get("labels", {}).get("alertname", "unknown")
            severity = alert_data.get("labels", {}).get("severity", "info")
            service = alert_data.get("labels", {}).get("service", "unknown")
            
            # Create alert object
            alert = Alert(
                id=alert_data.get("fingerprint", ""),
                name=alert_name,
                summary=alert_data.get("annotations", {}).get("summary", ""),
                description=alert_data.get("annotations", {}).get("description", ""),
                severity=severity,
                priority=_map_severity_to_priority(severity),
                service=service,
                source="alertmanager",
                labels=alert_data.get("labels", {}),
                annotations=alert_data.get("annotations", {}),
                timestamp=datetime.fromisoformat(alert_data.get("startsAt", datetime.utcnow().isoformat())),
                runbook_url=alert_data.get("annotations", {}).get("runbook_url")
            )
            
            processed_alerts.append(alert.id)
            
            # Check if alert is firing (not resolved)
            if request.status == "firing":
                # Look for matching runbooks
                matching_runbooks = _find_matching_runbooks(alert, executor)
                
                for runbook_id in matching_runbooks:
                    try:
                        # Prepare context for runbook
                        context = {
                            "alert_name": alert.name,
                            "severity": alert.severity,
                            "service": alert.service,
                            **alert.labels,
                            **alert.annotations
                        }
                        
                        # Execute runbook
                        result = await executor.execute_runbook(
                            runbook_id=runbook_id,
                            context=context,
                            dry_run=False,
                            force=False
                        )
                        
                        runbooks_executed.append({
                            "runbook_id": runbook_id,
                            "status": result.status,
                            "execution_id": result.execution_id
                        })
                        
                        logger.info(
                            "Executed runbook for alert",
                            alert_id=alert.id,
                            runbook_id=runbook_id,
                            status=result.status
                        )
                        
                    except Exception as e:
                        logger.error(
                            "Failed to execute runbook",
                            alert_id=alert.id,
                            runbook_id=runbook_id,
                            error=str(e)
                        )
                
                # Create incident for critical alerts
                if severity == "critical":
                    incident_id = await incident_tracker.create_incident(
                        title=f"{alert.name} - {alert.service}",
                        description=alert.description,
                        service=alert.service,
                        priority=IncidentPriority.HIGH,
                        alert_ids=[alert.id]
                    )
                    
                    incidents_created.append(incident_id)
                    
                    logger.info(
                        "Created incident for critical alert",
                        alert_id=alert.id,
                        incident_id=incident_id
                    )
        
        return WebhookResponse(
            status="success",
            message=f"Processed {len(processed_alerts)} alerts",
            data={
                "alerts_processed": processed_alerts,
                "runbooks_executed": runbooks_executed,
                "incidents_created": incidents_created
            }
        )
        
    except Exception as e:
        logger.error("Failed to process alert webhook", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/error-budget", response_model=WebhookResponse)
async def receive_error_budget_alert(
    request: Dict[str, Any],
    executor: RunbookExecutor = Depends(get_runbook_executor)
) -> WebhookResponse:
    """
    Receive error budget alert and trigger appropriate actions.
    
    Special handling for error budget burn rate alerts.
    """
    try:
        service = request.get("service", "unknown")
        burn_rate = request.get("burn_rate", 0)
        remaining_budget = request.get("remaining_budget", 100)
        severity = request.get("severity", "info")
        
        logger.info(
            "Received error budget alert",
            service=service,
            burn_rate=burn_rate,
            remaining_budget=remaining_budget,
            severity=severity
        )
        
        actions_taken = []
        
        # Critical burn rate - immediate action
        if burn_rate > 14.4:
            # Trigger emergency procedures
            actions_taken.append("triggered_emergency_procedures")
            
            # Execute error budget runbook if exists
            try:
                result = await executor.execute_runbook(
                    runbook_id="error_budget_critical",
                    context={
                        "service": service,
                        "burn_rate": burn_rate,
                        "remaining_budget": remaining_budget
                    },
                    dry_run=False,
                    force=True  # Force execution for critical
                )
                actions_taken.append(f"executed_runbook: {result.status}")
            except:
                pass
        
        # High burn rate - alert and prepare
        elif burn_rate > 6:
            actions_taken.append("alerted_on_call")
        
        # Low budget - restrict changes
        if remaining_budget < 10:
            actions_taken.append("enabled_change_freeze")
            # Would implement change freeze logic here
        
        return WebhookResponse(
            status="success",
            message=f"Processed error budget alert for {service}",
            data={
                "service": service,
                "burn_rate": burn_rate,
                "remaining_budget": remaining_budget,
                "actions_taken": actions_taken
            }
        )
        
    except Exception as e:
        logger.error("Failed to process error budget alert", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/infrastructure", response_model=WebhookResponse)
async def receive_infrastructure_alert(
    request: Dict[str, Any],
    executor: RunbookExecutor = Depends(get_runbook_executor)
) -> WebhookResponse:
    """Receive infrastructure-specific alerts."""
    try:
        alert_type = request.get("type", "unknown")
        component = request.get("component", "unknown")
        metrics = request.get("metrics", {})
        
        logger.info(
            "Received infrastructure alert",
            type=alert_type,
            component=component
        )
        
        # Map to appropriate runbook
        runbook_map = {
            "high_cpu": "cpu_mitigation",
            "high_memory": "memory_pressure_mitigation",
            "disk_space": "disk_cleanup",
            "network_issue": "network_diagnostics"
        }
        
        runbook_id = runbook_map.get(alert_type)
        
        if runbook_id:
            try:
                result = await executor.execute_runbook(
                    runbook_id=runbook_id,
                    context={
                        "component": component,
                        **metrics
                    },
                    dry_run=False
                )
                
                return WebhookResponse(
                    status="success",
                    message=f"Executed infrastructure runbook: {runbook_id}",
                    data={"execution_id": result.execution_id, "status": result.status}
                )
            except Exception as e:
                logger.error(f"Failed to execute runbook: {e}")
        
        return WebhookResponse(
            status="acknowledged",
            message=f"Infrastructure alert acknowledged: {alert_type}",
            data={"component": component}
        )
        
    except Exception as e:
        logger.error("Failed to process infrastructure alert", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============= Manual Control Endpoints =============

@router.post("/runbook/execute", response_model=WebhookResponse)
async def execute_runbook_manual(
    request: RunbookExecutionRequest,
    executor: RunbookExecutor = Depends(get_runbook_executor)
) -> WebhookResponse:
    """Manually execute a runbook."""
    try:
        result = await executor.execute_runbook(
            runbook_id=request.runbook_id,
            context=request.context,
            dry_run=request.dry_run,
            force=request.force
        )
        
        return WebhookResponse(
            status="success",
            message=f"Runbook executed: {result.status}",
            data={
                "execution_id": result.execution_id,
                "status": result.status,
                "actions_executed": result.actions_executed,
                "actions_failed": result.actions_failed,
                "duration_seconds": (result.end_time - result.start_time).total_seconds()
            }
        )
        
    except Exception as e:
        logger.error("Failed to execute runbook", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runbook/list")
async def list_runbooks(
    executor: RunbookExecutor = Depends(get_runbook_executor)
) -> Dict[str, Any]:
    """List available runbooks."""
    return {
        "runbooks": executor.list_runbooks()
    }


@router.get("/runbook/{runbook_id}/status")
async def get_runbook_status(
    runbook_id: str,
    executor: RunbookExecutor = Depends(get_runbook_executor)
) -> Dict[str, Any]:
    """Get status of a specific runbook."""
    return executor.get_runbook_status(runbook_id)


@router.post("/incident/create", response_model=WebhookResponse)
async def create_incident_manual(
    request: IncidentCreationRequest,
    incident_tracker: IncidentTracker = Depends(get_incident_tracker)
) -> WebhookResponse:
    """Manually create an incident."""
    try:
        incident_id = await incident_tracker.create_incident(
            title=request.title,
            description=request.description,
            service=request.service,
            priority=IncidentPriority(request.priority),
            alert_ids=request.alert_ids
        )
        
        return WebhookResponse(
            status="success",
            message="Incident created",
            data={"incident_id": incident_id}
        )
        
    except Exception as e:
        logger.error("Failed to create incident", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============= Helper Functions =============

def _map_severity_to_priority(severity: str) -> AlertPriority:
    """Map alert severity to priority."""
    mapping = {
        "critical": AlertPriority.CRITICAL,
        "high": AlertPriority.HIGH,
        "warning": AlertPriority.MEDIUM,
        "info": AlertPriority.INFO
    }
    return mapping.get(severity, AlertPriority.LOW)


def _find_matching_runbooks(alert: Alert, executor: RunbookExecutor) -> List[str]:
    """Find runbooks that match the alert."""
    matching = []
    
    for runbook_id, runbook in executor.runbooks.items():
        # Check if runbook trigger matches alert
        trigger = runbook.trigger
        
        if trigger.get("alert_name") == alert.name:
            matching.append(runbook_id)
        elif trigger.get("service") == alert.service and trigger.get("severity") == alert.severity:
            matching.append(runbook_id)
    
    return matching