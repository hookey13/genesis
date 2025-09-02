"""FastAPI endpoints for distributed tracing metrics and analysis."""

from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from genesis.monitoring.trace_analysis import (
    TraceAnalyzer,
    PerformanceBottleneck,
    get_trace_analyzer,
)
from genesis.monitoring.trace_alerting import (
    TraceAlertManager,
    Alert,
    AlertSeverity,
    get_trace_alert_manager,
)
from genesis.monitoring.opentelemetry_tracing import get_opentelemetry_tracer

logger = structlog.get_logger(__name__)

# Create API router
router = APIRouter(prefix="/tracing", tags=["tracing"])


# ============= Response Models =============

class TracingStatusResponse(BaseModel):
    """Tracing system status response."""
    enabled: bool = Field(..., description="Whether tracing is enabled")
    otlp_endpoint: Optional[str] = Field(None, description="OTLP collector endpoint")
    sampling_rate: float = Field(..., description="Base sampling rate")
    production_mode: bool = Field(..., description="Whether in production mode")
    traces_collected: int = Field(..., description="Total traces collected")
    spans_processed: int = Field(..., description="Total spans processed")
    
    
class PerformanceBottleneckResponse(BaseModel):
    """Performance bottleneck response."""
    operation_name: str
    bottleneck_type: str
    severity: str
    current_value: float
    threshold_value: float
    impact_score: float
    recommendations: List[str]
    detected_at: datetime
    

class ServiceDependencyResponse(BaseModel):
    """Service dependency response."""
    source: str
    target: str
    call_count: int
    avg_latency_ms: float
    error_rate: float
    error_count: int


class TraceAnalysisResponse(BaseModel):
    """Trace analysis summary response."""
    timestamp: datetime
    analysis_window_minutes: int
    total_operations: int
    critical_bottlenecks: int
    high_bottlenecks: int
    operations: Dict[str, Dict[str, Any]]
    top_slow_operations: List[Dict[str, Any]]
    top_error_operations: List[Dict[str, Any]]
    

class TraceAlertResponse(BaseModel):
    """Trace alert response."""
    id: str
    rule_name: str
    severity: str
    state: str
    message: str
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    

class TraceAlertSummaryResponse(BaseModel):
    """Trace alert summary response."""
    total_active: int
    critical: int
    high: int
    medium: int
    low: int
    info: int
    unacknowledged: int
    suppressed_rules: List[Dict[str, Any]]


# ============= Endpoints =============

@router.get("/status", response_model=TracingStatusResponse)
async def get_tracing_status() -> TracingStatusResponse:
    """Get current tracing system status."""
    tracer = get_opentelemetry_tracer()
    analyzer = get_trace_analyzer()
    
    # Calculate metrics
    total_spans = sum(m.count for m in analyzer.span_metrics.values())
    total_operations = len(analyzer.span_metrics)
    
    return TracingStatusResponse(
        enabled=tracer.otlp_endpoint is not None,
        otlp_endpoint=tracer.otlp_endpoint,
        sampling_rate=tracer.sampling_rate,
        production_mode=tracer.production_mode,
        traces_collected=total_operations,  # Approximation
        spans_processed=total_spans,
    )


@router.get("/analysis/summary", response_model=TraceAnalysisResponse)
async def get_trace_analysis_summary(
    window_minutes: int = Query(5, description="Analysis window in minutes", ge=1, le=60)
) -> TraceAnalysisResponse:
    """Get trace analysis summary."""
    analyzer = get_trace_analyzer()
    summary = analyzer.get_performance_summary()
    
    return TraceAnalysisResponse(**summary)


@router.get("/analysis/bottlenecks", response_model=List[PerformanceBottleneckResponse])
async def get_performance_bottlenecks(
    severity: Optional[str] = Query(None, description="Filter by severity (critical/high/medium/low)")
) -> List[PerformanceBottleneckResponse]:
    """Get identified performance bottlenecks."""
    analyzer = get_trace_analyzer()
    bottlenecks = analyzer.get_bottlenecks(severity_filter=severity)
    
    return [
        PerformanceBottleneckResponse(
            operation_name=b.operation_name,
            bottleneck_type=b.bottleneck_type,
            severity=b.severity,
            current_value=b.current_value,
            threshold_value=b.threshold_value,
            impact_score=b.impact_score,
            recommendations=b.recommendations,
            detected_at=b.detected_at,
        )
        for b in bottlenecks
    ]


@router.get("/analysis/dependencies", response_model=List[ServiceDependencyResponse])
async def get_service_dependencies() -> List[ServiceDependencyResponse]:
    """Get service dependency analysis."""
    analyzer = get_trace_analyzer()
    dependencies = analyzer.get_service_dependencies()
    
    return [
        ServiceDependencyResponse(**dep)
        for dep in dependencies
    ]


@router.get("/analysis/report")
async def get_performance_report() -> Dict[str, str]:
    """Get detailed performance analysis report."""
    analyzer = get_trace_analyzer()
    report = analyzer.generate_performance_report()
    
    return {
        "report": report,
        "generated_at": datetime.now(UTC).isoformat(),
    }


@router.post("/analysis/process-span")
async def process_span(span_data: Dict[str, Any]) -> Dict[str, str]:
    """Process a span for analysis (used by collectors)."""
    analyzer = get_trace_analyzer()
    analyzer.process_span(span_data)
    
    # Check for alerts
    alert_manager = get_trace_alert_manager()
    
    # Get span metrics for alerting
    operation_name = span_data.get("name", "unknown")
    if operation_name in analyzer.span_metrics:
        metrics = analyzer.span_metrics[operation_name]
        alert_data = {
            "operation_name": operation_name,
            "avg_duration_ms": metrics.avg_duration_ms,
            "error_rate": metrics.error_rate,
            "slow_rate": metrics.slow_rate,
        }
        
        # Evaluate alerts
        new_alerts = alert_manager.evaluate_metrics(alert_data)
        
        if new_alerts:
            logger.warning(
                "Trace alerts triggered",
                operation=operation_name,
                alert_count=len(new_alerts),
            )
    
    return {"status": "processed"}


@router.get("/alerts/active", response_model=List[TraceAlertResponse])
async def get_active_trace_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    unacknowledged_only: bool = Query(False, description="Only unacknowledged alerts")
) -> List[TraceAlertResponse]:
    """Get active trace-based alerts."""
    alert_manager = get_trace_alert_manager()
    
    severity_filter = None
    if severity:
        try:
            severity_filter = AlertSeverity(severity.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    
    alerts = alert_manager.get_active_alerts(
        severity_filter=severity_filter,
        unacknowledged_only=unacknowledged_only
    )
    
    return [
        TraceAlertResponse(
            id=alert.id,
            rule_name=alert.rule_name,
            severity=alert.severity.value,
            state=alert.state.value,
            message=alert.message,
            fired_at=alert.fired_at,
            resolved_at=alert.resolved_at,
            acknowledged_at=alert.acknowledged_at,
            acknowledged_by=alert.acknowledged_by,
        )
        for alert in alerts
    ]


@router.get("/alerts/summary", response_model=TraceAlertSummaryResponse)
async def get_trace_alert_summary() -> TraceAlertSummaryResponse:
    """Get trace alert summary."""
    alert_manager = get_trace_alert_manager()
    summary = alert_manager.get_alert_summary()
    
    return TraceAlertSummaryResponse(**summary)


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_trace_alert(
    alert_id: str,
    acknowledged_by: str = "operator"
) -> Dict[str, str]:
    """Acknowledge a trace alert."""
    alert_manager = get_trace_alert_manager()
    
    if alert_manager.acknowledge_alert(alert_id, acknowledged_by):
        return {
            "status": "success",
            "message": f"Alert {alert_id} acknowledged by {acknowledged_by}",
        }
    else:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@router.post("/sampling/update")
async def update_sampling_rate(
    sampling_rate: float = Query(..., description="New sampling rate (0.0-1.0)", ge=0.0, le=1.0)
) -> Dict[str, Any]:
    """Update trace sampling rate (requires restart to take effect)."""
    # This would typically update configuration
    # For now, just return what would be done
    return {
        "status": "pending",
        "message": "Sampling rate update requires service restart",
        "new_rate": sampling_rate,
        "restart_required": True,
    }


@router.get("/correlation/{correlation_id}")
async def get_traces_by_correlation_id(
    correlation_id: str
) -> Dict[str, Any]:
    """Get all traces for a correlation ID."""
    # This would query Jaeger for traces with the correlation ID
    # For now, return a placeholder
    return {
        "correlation_id": correlation_id,
        "traces": [],
        "message": "Query Jaeger UI for detailed trace information",
        "jaeger_url": f"http://localhost:16686/search?tags=%7B%22correlation_id%22%3A%22{correlation_id}%22%7D",
    }


@router.post("/analysis/cleanup")
async def cleanup_old_trace_data(
    retention_minutes: int = Query(60, description="Data retention in minutes", ge=10)
) -> Dict[str, str]:
    """Clean up old trace analysis data."""
    analyzer = get_trace_analyzer()
    analyzer.clear_old_data(retention_minutes)
    
    alert_manager = get_trace_alert_manager()
    alert_manager.cleanup_old_alerts()
    
    return {
        "status": "success",
        "message": f"Cleaned up data older than {retention_minutes} minutes",
    }


@router.get("/health")
async def trace_system_health_check() -> Dict[str, Any]:
    """Check trace system health."""
    tracer = get_opentelemetry_tracer()
    analyzer = get_trace_analyzer()
    alert_manager = get_trace_alert_manager()
    
    # Basic health checks
    health = {
        "status": "healthy",
        "components": {
            "tracer": {
                "status": "healthy" if tracer else "unhealthy",
                "otlp_connected": tracer.otlp_endpoint is not None if tracer else False,
            },
            "analyzer": {
                "status": "healthy" if analyzer else "unhealthy",
                "operations_tracked": len(analyzer.span_metrics) if analyzer else 0,
            },
            "alert_manager": {
                "status": "healthy" if alert_manager else "unhealthy",
                "active_alerts": len(alert_manager.active_alerts) if alert_manager else 0,
            },
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }
    
    # Determine overall health
    if any(c["status"] == "unhealthy" for c in health["components"].values()):
        health["status"] = "degraded"
    
    return health