"""FastAPI metrics endpoints for operational dashboard data."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
import structlog

from genesis.monitoring.deployment_tracker import (
    DeploymentTracker,
    DeploymentInfo,
    DeploymentMetrics,
    DeploymentStatus,
)
from genesis.monitoring.error_budget import ErrorBudget, ErrorBudgetStatus
from genesis.monitoring.metrics_collector import MetricsCollector, TradingMetrics
from genesis.monitoring.rate_limit_metrics import RateLimitMetricsCollector

logger = structlog.get_logger(__name__)

# Create API router
router = APIRouter(prefix="/metrics", tags=["metrics"])


# ============= Response Models =============

class SystemHealthResponse(BaseModel):
    """System health metrics response."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage in bytes")
    memory_percent: float = Field(..., description="Memory usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    network_bytes_sent: float = Field(..., description="Network bytes sent per second")
    network_bytes_recv: float = Field(..., description="Network bytes received per second")
    connection_count: int = Field(..., description="Number of network connections")
    thread_count: int = Field(..., description="Number of threads")
    open_files: int = Field(..., description="Number of open file descriptors")
    system_uptime: float = Field(..., description="System uptime in seconds")
    process_uptime: float = Field(..., description="Process uptime in seconds")
    health_score: float = Field(..., description="Overall health score (0-100)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PnLMetricsResponse(BaseModel):
    """P&L metrics response."""
    current_pnl: float = Field(..., description="Current total P&L")
    daily_pnl: float = Field(..., description="Daily P&L")
    daily_pnl_percent: float = Field(..., description="Daily P&L percentage")
    realized_pnl: float = Field(..., description="Realized P&L")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    win_rate: float = Field(..., description="Win rate percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    total_trades: int = Field(..., description="Total number of trades")
    account_balance: float = Field(..., description="Account balance")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response."""
    
    class LatencyMetrics(BaseModel):
        p50: float
        p95: float
        p99: float
        max: float
        
    order_latency: Optional[LatencyMetrics] = None
    api_latency: Optional[LatencyMetrics] = None
    websocket_latency: Optional[LatencyMetrics] = None
    database_latency: Optional[LatencyMetrics] = None
    
    orders_per_second: float = Field(..., description="Orders processed per second")
    events_per_second: float = Field(..., description="Events processed per second")
    messages_per_second: float = Field(..., description="Messages processed per second")
    
    trading_volume_24h: float = Field(..., description="24-hour trading volume")
    total_trades_24h: int = Field(..., description="Total trades in 24 hours")
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RateLimitMetricsResponse(BaseModel):
    """Rate limit metrics response."""
    
    class EndpointLimit(BaseModel):
        name: str
        tokens_used: int
        tokens_capacity: int
        utilization_percent: float
        requests_allowed: int
        requests_rejected: int
    
    endpoints: List[EndpointLimit] = Field(default_factory=list)
    overall_utilization: float = Field(..., description="Overall rate limit utilization")
    
    window_1m_weight: int = Field(..., description="1-minute window weight")
    window_1m_limit: int = Field(..., description="1-minute window limit")
    window_10s_orders: int = Field(..., description="10-second window orders")
    window_10s_limit: int = Field(..., description="10-second window limit")
    
    critical_overrides: int = Field(0, description="Number of critical overrides")
    coalesced_requests: int = Field(0, description="Number of coalesced requests")
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorBudgetResponse(BaseModel):
    """Error budget metrics response."""
    
    class SLOMetrics(BaseModel):
        name: str
        target: float
        current_success_rate: float
        error_budget_remaining: float
        burn_rate: float
        is_exhausted: bool
    
    slos: List[SLOMetrics] = Field(default_factory=list)
    overall_success_rate: float = Field(..., description="Overall success rate")
    total_errors: int = Field(..., description="Total errors")
    total_requests: int = Field(..., description="Total requests")
    error_rate_per_minute: float = Field(..., description="Errors per minute")
    error_rate_per_hour: float = Field(..., description="Errors per hour")
    critical_slos_at_risk: int = Field(..., description="Number of critical SLOs at risk")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AlertMetricsResponse(BaseModel):
    """Alert metrics response."""
    
    class AlertInfo(BaseModel):
        id: str
        name: str
        severity: str
        state: str
        message: str
        source: str
        timestamp: datetime
        acknowledged: bool
        
    active_alerts: List[AlertInfo] = Field(default_factory=list)
    critical_count: int = Field(0, description="Number of critical alerts")
    high_count: int = Field(0, description="Number of high alerts")
    medium_count: int = Field(0, description="Number of medium alerts")
    unacknowledged_count: int = Field(0, description="Number of unacknowledged alerts")
    
    alerts_24h: int = Field(0, description="Total alerts in 24 hours")
    resolution_rate: float = Field(0, description="Alert resolution rate")
    mean_time_to_acknowledge: float = Field(0, description="Mean time to acknowledge in seconds")
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DeploymentMetricsResponse(BaseModel):
    """Deployment metrics response."""
    
    class DeploymentSummary(BaseModel):
        id: str
        version: str
        type: str
        status: str
        timestamp: datetime
        deployed_by: str
        duration_seconds: Optional[float] = None
    
    current_deployment: Optional[DeploymentSummary] = None
    recent_deployments: List[DeploymentSummary] = Field(default_factory=list)
    
    total_deployments: int = Field(0, description="Total deployments")
    success_rate: float = Field(0, description="Deployment success rate")
    deployments_today: int = Field(0, description="Deployments today")
    deployments_this_week: int = Field(0, description="Deployments this week")
    
    last_deployment: Optional[datetime] = None
    last_successful_deployment: Optional[datetime] = None
    mean_deployment_time_seconds: float = Field(0, description="Mean deployment time in seconds")
    
    can_deploy: bool = Field(True, description="Whether new deployment is allowed")
    can_deploy_reason: str = Field("", description="Reason for deployment status")
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============= Dependencies =============

# These would be properly injected in production
_metrics_collector: Optional[MetricsCollector] = None
_rate_limit_collector: Optional[RateLimitMetricsCollector] = None
_error_budget: Optional[ErrorBudget] = None
_deployment_tracker: Optional[DeploymentTracker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    global _metrics_collector
    if not _metrics_collector:
        from genesis.monitoring.prometheus_exporter import MetricsRegistry
        registry = MetricsRegistry()
        _metrics_collector = MetricsCollector(registry)
    return _metrics_collector


def get_rate_limit_collector() -> RateLimitMetricsCollector:
    """Get rate limit metrics collector."""
    global _rate_limit_collector
    if not _rate_limit_collector:
        _rate_limit_collector = RateLimitMetricsCollector()
    return _rate_limit_collector


def get_error_budget() -> ErrorBudget:
    """Get error budget instance."""
    global _error_budget
    if not _error_budget:
        _error_budget = ErrorBudget()
    return _error_budget


def get_deployment_tracker() -> DeploymentTracker:
    """Get deployment tracker instance."""
    global _deployment_tracker
    if not _deployment_tracker:
        _deployment_tracker = DeploymentTracker()
    return _deployment_tracker


# ============= Endpoints =============

@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health(
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> SystemHealthResponse:
    """Get current system health metrics."""
    metrics = collector.metrics
    
    return SystemHealthResponse(
        cpu_usage=metrics.cpu_usage,
        memory_usage=metrics.memory_usage,
        memory_percent=metrics.memory_percent,
        disk_usage_percent=metrics.disk_usage_percent,
        network_bytes_sent=metrics.network_bytes_sent,
        network_bytes_recv=metrics.network_bytes_recv,
        connection_count=metrics.connection_count,
        thread_count=metrics.thread_count,
        open_files=metrics.open_files,
        system_uptime=metrics.system_uptime,
        process_uptime=metrics.process_uptime,
        health_score=metrics.health_score,
    )


@router.get("/pnl", response_model=PnLMetricsResponse)
async def get_pnl_metrics(
    collector: MetricsCollector = Depends(get_metrics_collector)
) -> PnLMetricsResponse:
    """Get P&L metrics."""
    metrics = collector.metrics
    
    # Calculate account balance (mock for now)
    account_balance = 50000.0  # Would come from account manager
    
    return PnLMetricsResponse(
        current_pnl=float(metrics.realized_pnl + metrics.unrealized_pnl),
        daily_pnl=float(metrics.realized_pnl),  # Simplified
        daily_pnl_percent=2.5,  # Mock
        realized_pnl=float(metrics.realized_pnl),
        unrealized_pnl=float(metrics.unrealized_pnl),
        win_rate=metrics.win_rate * 100,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        total_trades=metrics.trades_executed,
        account_balance=account_balance,
    )


@router.get("/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    collector: MetricsCollector = Depends(get_metrics_collector),
    time_window: str = Query("1h", description="Time window (1h, 24h, 7d)"),
) -> PerformanceMetricsResponse:
    """Get performance metrics."""
    # Mock data for demonstration
    # In production, this would aggregate from actual metrics
    
    response = PerformanceMetricsResponse(
        orders_per_second=125.5,
        events_per_second=850.3,
        messages_per_second=2150.7,
        trading_volume_24h=5250000.0,
        total_trades_24h=12500,
    )
    
    # Add latency metrics
    response.order_latency = PerformanceMetricsResponse.LatencyMetrics(
        p50=25.5, p95=45.2, p99=89.3, max=250.0
    )
    response.api_latency = PerformanceMetricsResponse.LatencyMetrics(
        p50=15.3, p95=28.9, p99=55.2, max=150.0
    )
    response.websocket_latency = PerformanceMetricsResponse.LatencyMetrics(
        p50=2.1, p95=4.5, p99=8.9, max=25.0
    )
    
    return response


@router.get("/rate-limits", response_model=RateLimitMetricsResponse)
async def get_rate_limit_metrics(
    collector: RateLimitMetricsCollector = Depends(get_rate_limit_collector)
) -> RateLimitMetricsResponse:
    """Get rate limit metrics."""
    # Mock data for demonstration
    response = RateLimitMetricsResponse(
        overall_utilization=45.5,
        window_1m_weight=600,
        window_1m_limit=1200,
        window_10s_orders=15,
        window_10s_limit=50,
        critical_overrides=0,
        coalesced_requests=25,
    )
    
    # Add endpoint limits
    response.endpoints = [
        RateLimitMetricsResponse.EndpointLimit(
            name="POST /api/v3/order",
            tokens_used=750,
            tokens_capacity=1000,
            utilization_percent=75.0,
            requests_allowed=750,
            requests_rejected=10,
        ),
        RateLimitMetricsResponse.EndpointLimit(
            name="GET /api/v3/ticker",
            tokens_used=200,
            tokens_capacity=1000,
            utilization_percent=20.0,
            requests_allowed=200,
            requests_rejected=0,
        ),
    ]
    
    return response


@router.get("/error-budget", response_model=ErrorBudgetResponse)
async def get_error_budget_metrics(
    error_budget: ErrorBudget = Depends(get_error_budget)
) -> ErrorBudgetResponse:
    """Get error budget metrics."""
    response = ErrorBudgetResponse(
        overall_success_rate=99.697,
        total_errors=940,
        total_requests=310000,
        error_rate_per_minute=3.5,
        error_rate_per_hour=210.0,
        critical_slos_at_risk=1,
    )
    
    # Add SLO metrics
    for slo_name in ["overall_availability", "order_execution", "data_accuracy"]:
        status = error_budget.get_budget_status(slo_name)
        if status:
            response.slos.append(
                ErrorBudgetResponse.SLOMetrics(
                    name=slo_name,
                    target=status.target,
                    current_success_rate=status.current_success_rate,
                    error_budget_remaining=status.error_budget_remaining,
                    burn_rate=status.burn_rate,
                    is_exhausted=status.is_exhausted,
                )
            )
    
    return response


@router.get("/alerts", response_model=AlertMetricsResponse)
async def get_alert_metrics(
    active_only: bool = Query(False, description="Show only active alerts"),
) -> AlertMetricsResponse:
    """Get alert metrics."""
    # Mock data for demonstration
    response = AlertMetricsResponse(
        critical_count=1,
        high_count=2,
        medium_count=3,
        unacknowledged_count=2,
        alerts_24h=15,
        resolution_rate=85.5,
        mean_time_to_acknowledge=180.0,  # 3 minutes
    )
    
    # Add sample alerts
    now = datetime.now(UTC)
    response.active_alerts = [
        AlertMetricsResponse.AlertInfo(
            id="alert_001",
            name="High CPU Usage",
            severity="high",
            state="firing",
            message="CPU usage above 90% for 5 minutes",
            source="system_monitor",
            timestamp=now - timedelta(minutes=15),
            acknowledged=False,
        ),
        AlertMetricsResponse.AlertInfo(
            id="alert_002",
            name="Order Execution Failed",
            severity="critical",
            state="firing",
            message="Failed to execute market order",
            source="order_executor",
            timestamp=now - timedelta(minutes=2),
            acknowledged=True,
        ),
    ]
    
    return response


@router.get("/deployments", response_model=DeploymentMetricsResponse)
async def get_deployment_metrics(
    tracker: DeploymentTracker = Depends(get_deployment_tracker),
    limit: int = Query(10, description="Number of recent deployments to return"),
) -> DeploymentMetricsResponse:
    """Get deployment metrics."""
    metrics = tracker.get_metrics()
    history = tracker.get_deployment_history(limit=limit)
    can_deploy, reason = tracker.can_deploy()
    
    response = DeploymentMetricsResponse(
        total_deployments=metrics.total_deployments,
        success_rate=metrics.success_rate,
        deployments_today=metrics.deployments_today,
        deployments_this_week=metrics.deployments_this_week,
        last_deployment=metrics.last_deployment,
        last_successful_deployment=metrics.last_successful_deployment,
        mean_deployment_time_seconds=metrics.mean_deployment_time.total_seconds() if metrics.mean_deployment_time else 0,
        can_deploy=can_deploy,
        can_deploy_reason=reason,
    )
    
    # Add current deployment if any
    if tracker.current_deployment:
        response.current_deployment = DeploymentMetricsResponse.DeploymentSummary(
            id=tracker.current_deployment.id,
            version=tracker.current_deployment.version,
            type=tracker.current_deployment.type.value,
            status=tracker.current_deployment.status.value,
            timestamp=tracker.current_deployment.timestamp,
            deployed_by=tracker.current_deployment.deployed_by,
        )
    
    # Add recent deployments
    for deployment in history:
        response.recent_deployments.append(
            DeploymentMetricsResponse.DeploymentSummary(
                id=deployment.id,
                version=deployment.version,
                type=deployment.type.value,
                status=deployment.status.value,
                timestamp=deployment.timestamp,
                deployed_by=deployment.deployed_by,
                duration_seconds=deployment.duration.total_seconds() if deployment.duration else None,
            )
        )
    
    return response


@router.post("/deployments/rollback")
async def rollback_deployment(
    target_version: Optional[str] = None,
    reason: str = "Manual rollback",
    tracker: DeploymentTracker = Depends(get_deployment_tracker),
) -> Dict[str, str]:
    """Rollback to a previous deployment."""
    success = tracker.rollback(target_version=target_version, reason=reason)
    
    if not success:
        raise HTTPException(status_code=400, detail="Rollback failed")
    
    return {
        "status": "success",
        "message": f"Rolled back to version {target_version or 'last successful'}",
    }


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = "operator",
) -> Dict[str, str]:
    """Acknowledge an alert."""
    # In production, this would interact with the alert manager
    return {
        "status": "success",
        "message": f"Alert {alert_id} acknowledged by {acknowledged_by}",
    }


@router.get("/dashboard/summary")
async def get_dashboard_summary() -> Dict[str, Any]:
    """Get comprehensive dashboard summary."""
    # Aggregate all metrics for dashboard
    collector = get_metrics_collector()
    error_budget = get_error_budget()
    deployment_tracker = get_deployment_tracker()
    
    metrics = collector.metrics
    deployment_metrics = deployment_tracker.get_metrics()
    
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "health": {
            "score": metrics.health_score,
            "cpu": metrics.cpu_usage,
            "memory": metrics.memory_percent,
            "status": "healthy" if metrics.health_score > 80 else "degraded",
        },
        "trading": {
            "pnl": float(metrics.realized_pnl + metrics.unrealized_pnl),
            "win_rate": metrics.win_rate * 100,
            "trades_today": metrics.trades_executed,
            "positions_open": metrics.current_positions,
        },
        "errors": {
            "success_rate": 99.7,
            "errors_per_hour": 210,
            "critical_slos_at_risk": 1,
        },
        "deployments": {
            "last_deployment": deployment_metrics.last_deployment.isoformat() if deployment_metrics.last_deployment else None,
            "success_rate": deployment_metrics.success_rate * 100,
            "deployments_today": deployment_metrics.deployments_today,
        },
        "alerts": {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "unacknowledged": 2,
        },
    }