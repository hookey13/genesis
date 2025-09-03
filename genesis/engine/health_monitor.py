"""
Health monitoring system for Project GENESIS trading engine.

Monitors component health, performance metrics, and system resources.
Provides alerting and automatic recovery mechanisms.
"""

import asyncio
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TextIO

import psutil
import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Component health status levels."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_count: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def update_heartbeat(self) -> None:
        """Update the last heartbeat time."""
        self.last_heartbeat = datetime.now(UTC)
        if self.status == HealthStatus.UNKNOWN:
            self.status = HealthStatus.HEALTHY

    def record_error(self, error: str) -> None:
        """Record an error for this component."""
        self.error_count += 1
        self.consecutive_failures += 1
        self.last_error = error

        # Update status based on consecutive failures
        if self.consecutive_failures >= 5:
            self.status = HealthStatus.FAILED
        elif self.consecutive_failures >= 3:
            self.status = HealthStatus.CRITICAL
        elif self.consecutive_failures >= 1:
            self.status = HealthStatus.DEGRADED

    def record_success(self) -> None:
        """Record a successful health check."""
        self.consecutive_failures = 0
        if self.status != HealthStatus.HEALTHY:
            self.status = HealthStatus.HEALTHY
        self.update_heartbeat()

    def is_stale(self, timeout_seconds: int = 60) -> bool:
        """Check if heartbeat is stale."""
        age = datetime.now(UTC) - self.last_heartbeat
        return age.total_seconds() > timeout_seconds


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_percent: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    open_files: int = 0
    active_threads: int = 0

    # Trading metrics
    events_per_second: float = 0.0
    orders_per_minute: float = 0.0
    avg_latency_ms: float = 0.0

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Alert:
    """System alert."""

    severity: AlertSeverity
    component: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    alert_id: str = ""

    def __post_init__(self):
        """Generate alert ID if not provided."""
        if not self.alert_id:
            self.alert_id = f"{self.component}_{self.severity}_{int(self.timestamp.timestamp())}"


class HealthMonitor:
    """
    Central health monitoring system.

    Monitors:
    - Component heartbeats and health status
    - System resource utilization
    - Performance metrics
    - Trading-specific health indicators
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        check_interval: int = 10,
        heartbeat_timeout: int = 60,
        alert_callback: Callable[[Alert], None] | None = None
    ):
        """
        Initialize health monitor.

        Args:
            event_bus: Optional event bus for publishing alerts
            check_interval: Health check interval in seconds
            heartbeat_timeout: Heartbeat timeout in seconds
            alert_callback: Optional callback for alerts
        """
        self.event_bus = event_bus
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.alert_callback = alert_callback

        self.components: dict[str, ComponentHealth] = {}
        self.system_metrics = SystemMetrics()
        self.alerts: deque[Alert] = deque(maxlen=1000)

        self.running = False
        self.monitor_task: asyncio.Task | None = None

        # Performance tracking
        self.event_timestamps: deque[float] = deque(maxlen=1000)
        self.order_timestamps: deque[float] = deque(maxlen=100)
        self.latency_samples: deque[float] = deque(maxlen=1000)

        # Resource baselines
        self.cpu_baseline: float | None = None
        self.memory_baseline: float | None = None

        logger.info("HealthMonitor initialized")

    def register_component(self, name: str) -> None:
        """
        Register a component for health monitoring.

        Args:
            name: Component name
        """
        if name not in self.components:
            self.components[name] = ComponentHealth(name=name)
            logger.info("Component registered for health monitoring", component=name)

    def update_component_health(
        self,
        name: str,
        status: HealthStatus | None = None,
        metrics: dict[str, Any] | None = None
    ) -> None:
        """
        Update component health status.

        Args:
            name: Component name
            status: Optional health status
            metrics: Optional component-specific metrics
        """
        if name not in self.components:
            self.register_component(name)

        component = self.components[name]
        component.update_heartbeat()

        if status:
            component.status = status

        if metrics:
            component.metrics.update(metrics)

        # Record success if healthy
        if status == HealthStatus.HEALTHY:
            component.record_success()

    def record_component_error(self, name: str, error: str) -> None:
        """
        Record a component error.

        Args:
            name: Component name
            error: Error message
        """
        if name not in self.components:
            self.register_component(name)

        component = self.components[name]
        component.record_error(error)

        # Generate alert for critical or failed status
        if component.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            alert = Alert(
                severity=AlertSeverity.CRITICAL if component.status == HealthStatus.CRITICAL else AlertSeverity.ERROR,
                component=name,
                message=f"Component {name} is {component.status}",
                details={
                    "error": error,
                    "consecutive_failures": component.consecutive_failures,
                    "total_errors": component.error_count
                }
            )
            self._raise_alert(alert)

    def record_event(self) -> None:
        """Record an event for metrics calculation."""
        self.event_timestamps.append(time.time())

    def record_order(self) -> None:
        """Record an order for metrics calculation."""
        self.order_timestamps.append(time.time())

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self.latency_samples.append(latency_ms)

    async def start(self) -> None:
        """Start the health monitoring system."""
        if self.running:
            logger.warning("Health monitor already running")
            return

        self.running = True

        # Set resource baselines
        await self._establish_baselines()

        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop the health monitoring system."""
        if not self.running:
            return

        self.running = False

        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Check component health
                await self._check_component_health()

                # Calculate performance metrics
                self._calculate_performance_metrics()

                # Check for anomalies
                await self._check_anomalies()

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(self.check_interval)

    async def _establish_baselines(self) -> None:
        """Establish resource usage baselines."""
        try:
            # Sample CPU and memory over 5 seconds
            samples = []
            for _ in range(5):
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory().percent
                samples.append((cpu, memory))
                await asyncio.sleep(1)

            # Calculate baselines as averages
            self.cpu_baseline = sum(s[0] for s in samples) / len(samples)
            self.memory_baseline = sum(s[1] for s in samples) / len(samples)

            logger.info(
                "Resource baselines established",
                cpu_baseline=round(self.cpu_baseline, 2),
                memory_baseline=round(self.memory_baseline, 2)
            )

        except Exception as e:
            logger.error("Failed to establish baselines", error=str(e))
            self.cpu_baseline = 50.0  # Default baseline
            self.memory_baseline = 50.0

    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # CPU and Memory
            self.system_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.system_metrics.memory_percent = memory.percent
            self.system_metrics.memory_mb = memory.used / (1024 * 1024)

            # Disk
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_percent = disk.percent

            # Network (calculate delta)
            net_io = psutil.net_io_counters()
            self.system_metrics.network_sent_mb = net_io.bytes_sent / (1024 * 1024)
            self.system_metrics.network_recv_mb = net_io.bytes_recv / (1024 * 1024)

            # Process info
            process = psutil.Process(os.getpid())
            self.system_metrics.open_files = len(process.open_files())
            self.system_metrics.active_threads = process.num_threads()

            self.system_metrics.timestamp = datetime.now(UTC)

        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))

    async def _check_component_health(self) -> None:
        """Check health of all registered components."""
        for name, component in self.components.items():
            # Check for stale heartbeats
            if component.is_stale(self.heartbeat_timeout) and component.status != HealthStatus.FAILED:
                component.status = HealthStatus.FAILED

                alert = Alert(
                    severity=AlertSeverity.CRITICAL,
                    component=name,
                    message=f"Component {name} heartbeat timeout",
                    details={
                        "last_heartbeat": component.last_heartbeat.isoformat(),
                        "timeout_seconds": self.heartbeat_timeout
                    }
                )
                self._raise_alert(alert)

    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics from collected data."""
        now = time.time()

        # Events per second (last 10 seconds)
        recent_events = [t for t in self.event_timestamps if now - t <= 10]
        self.system_metrics.events_per_second = len(recent_events) / 10.0 if recent_events else 0

        # Orders per minute (last 60 seconds)
        recent_orders = [t for t in self.order_timestamps if now - t <= 60]
        self.system_metrics.orders_per_minute = len(recent_orders)

        # Average latency
        if self.latency_samples:
            self.system_metrics.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)

    async def _check_anomalies(self) -> None:
        """Check for system anomalies."""
        # Check CPU usage
        if self.cpu_baseline and self.system_metrics.cpu_percent > self.cpu_baseline * 2:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                component="system",
                message="High CPU usage detected",
                details={
                    "current": round(self.system_metrics.cpu_percent, 2),
                    "baseline": round(self.cpu_baseline, 2),
                    "threshold": round(self.cpu_baseline * 2, 2)
                }
            )
            self._raise_alert(alert)

        # Check memory usage
        if self.system_metrics.memory_percent > 80:
            alert = Alert(
                severity=AlertSeverity.ERROR if self.system_metrics.memory_percent > 90 else AlertSeverity.WARNING,
                component="system",
                message="High memory usage detected",
                details={
                    "current_percent": round(self.system_metrics.memory_percent, 2),
                    "current_mb": round(self.system_metrics.memory_mb, 2)
                }
            )
            self._raise_alert(alert)

        # Check disk usage
        if self.system_metrics.disk_percent > 85:
            alert = Alert(
                severity=AlertSeverity.ERROR if self.system_metrics.disk_percent > 95 else AlertSeverity.WARNING,
                component="system",
                message="High disk usage detected",
                details={
                    "current_percent": round(self.system_metrics.disk_percent, 2)
                }
            )
            self._raise_alert(alert)

        # Check latency
        if self.system_metrics.avg_latency_ms > 100:  # >100ms average latency
            alert = Alert(
                severity=AlertSeverity.WARNING,
                component="performance",
                message="High latency detected",
                details={
                    "avg_latency_ms": round(self.system_metrics.avg_latency_ms, 2),
                    "threshold_ms": 100
                }
            )
            self._raise_alert(alert)

    def _raise_alert(self, alert: Alert) -> None:
        """
        Raise an alert.

        Args:
            alert: Alert to raise
        """
        # Avoid duplicate alerts within 5 minutes
        recent_alerts = [
            a for a in self.alerts
            if a.component == alert.component
            and a.severity == alert.severity
            and (alert.timestamp - a.timestamp).total_seconds() < 300
        ]

        if recent_alerts:
            return  # Skip duplicate alert

        self.alerts.append(alert)

        logger.warning(
            "Health alert raised",
            severity=alert.severity,
            component=alert.component,
            message=alert.message,
            details=alert.details
        )

        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))

        # Publish to event bus if available
        if self.event_bus:
            event = Event(
                type=EventType.HEALTH_ALERT,
                data=alert,
                priority=EventPriority.CRITICAL if alert.severity == AlertSeverity.CRITICAL else EventPriority.HIGH
            )
            asyncio.create_task(self.event_bus.publish(event))

    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall health status
        """
        if not self.components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in self.components.values()]

        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        metrics_lines = []
        
        # System metrics
        metrics_lines.append("# HELP genesis_cpu_usage_percent CPU usage percentage")
        metrics_lines.append("# TYPE genesis_cpu_usage_percent gauge")
        metrics_lines.append(f"genesis_cpu_usage_percent {self.system_metrics.cpu_percent}")
        
        metrics_lines.append("# HELP genesis_memory_usage_percent Memory usage percentage")
        metrics_lines.append("# TYPE genesis_memory_usage_percent gauge")
        metrics_lines.append(f"genesis_memory_usage_percent {self.system_metrics.memory_percent}")
        
        metrics_lines.append("# HELP genesis_memory_usage_mb Memory usage in MB")
        metrics_lines.append("# TYPE genesis_memory_usage_mb gauge")
        metrics_lines.append(f"genesis_memory_usage_mb {self.system_metrics.memory_mb}")
        
        metrics_lines.append("# HELP genesis_disk_usage_percent Disk usage percentage")
        metrics_lines.append("# TYPE genesis_disk_usage_percent gauge")
        metrics_lines.append(f"genesis_disk_usage_percent {self.system_metrics.disk_percent}")
        
        metrics_lines.append("# HELP genesis_open_files Number of open files")
        metrics_lines.append("# TYPE genesis_open_files gauge")
        metrics_lines.append(f"genesis_open_files {self.system_metrics.open_files}")
        
        metrics_lines.append("# HELP genesis_active_threads Number of active threads")
        metrics_lines.append("# TYPE genesis_active_threads gauge")
        metrics_lines.append(f"genesis_active_threads {self.system_metrics.active_threads}")
        
        # Trading metrics
        metrics_lines.append("# HELP genesis_events_per_second Events processed per second")
        metrics_lines.append("# TYPE genesis_events_per_second gauge")
        metrics_lines.append(f"genesis_events_per_second {self.system_metrics.events_per_second}")
        
        metrics_lines.append("# HELP genesis_orders_per_minute Orders processed per minute")
        metrics_lines.append("# TYPE genesis_orders_per_minute gauge")
        metrics_lines.append(f"genesis_orders_per_minute {self.system_metrics.orders_per_minute}")
        
        metrics_lines.append("# HELP genesis_avg_latency_ms Average processing latency in milliseconds")
        metrics_lines.append("# TYPE genesis_avg_latency_ms gauge")
        metrics_lines.append(f"genesis_avg_latency_ms {self.system_metrics.avg_latency_ms}")
        
        # Component health (0=healthy, 1=degraded, 2=critical, 3=failed, 4=unknown)
        health_values = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.CRITICAL: 2,
            HealthStatus.FAILED: 3,
            HealthStatus.UNKNOWN: 4
        }
        
        metrics_lines.append("# HELP genesis_component_health Component health status (0=healthy, 1=degraded, 2=critical, 3=failed, 4=unknown)")
        metrics_lines.append("# TYPE genesis_component_health gauge")
        
        for name, component in self.components.items():
            value = health_values.get(component.status, 4)
            metrics_lines.append(f'genesis_component_health{{component="{name}"}} {value}')
        
        # Component error counts
        metrics_lines.append("# HELP genesis_component_errors_total Total number of errors per component")
        metrics_lines.append("# TYPE genesis_component_errors_total counter")
        
        for name, component in self.components.items():
            metrics_lines.append(f'genesis_component_errors_total{{component="{name}"}} {component.error_count}')
        
        # Alert counts by severity
        alert_counts = {severity: 0 for severity in AlertSeverity}
        for alert in self.alerts:
            alert_counts[alert.severity] += 1
        
        metrics_lines.append("# HELP genesis_alerts_total Total number of alerts by severity")
        metrics_lines.append("# TYPE genesis_alerts_total counter")
        
        for severity, count in alert_counts.items():
            metrics_lines.append(f'genesis_alerts_total{{severity="{severity}"}} {count}')
        
        # Overall health status
        overall_health_value = health_values.get(self.get_overall_health(), 4)
        metrics_lines.append("# HELP genesis_overall_health Overall system health (0=healthy, 1=degraded, 2=critical, 3=failed, 4=unknown)")
        metrics_lines.append("# TYPE genesis_overall_health gauge")
        metrics_lines.append(f"genesis_overall_health {overall_health_value}")
        
        return "\n".join(metrics_lines)
    
    async def write_prometheus_metrics(self, file_path: str) -> None:
        """
        Write Prometheus metrics to a file.
        
        Args:
            file_path: Path to write metrics file
        """
        try:
            metrics = self.export_prometheus_metrics()
            with open(file_path, 'w') as f:
                f.write(metrics)
            logger.debug("Prometheus metrics written", path=file_path)
        except Exception as e:
            logger.error("Failed to write Prometheus metrics", error=str(e))
    
    def get_health_report(self) -> dict[str, Any]:
        """
        Generate comprehensive health report.

        Returns:
            Health report dictionary
        """
        return {
            "overall_status": self.get_overall_health(),
            "timestamp": datetime.now(UTC).isoformat(),
            "components": {
                name: {
                    "status": comp.status,
                    "last_heartbeat": comp.last_heartbeat.isoformat(),
                    "error_count": comp.error_count,
                    "consecutive_failures": comp.consecutive_failures,
                    "last_error": comp.last_error,
                    "metrics": comp.metrics
                }
                for name, comp in self.components.items()
            },
            "system_metrics": {
                "cpu_percent": round(self.system_metrics.cpu_percent, 2),
                "memory_percent": round(self.system_metrics.memory_percent, 2),
                "memory_mb": round(self.system_metrics.memory_mb, 2),
                "disk_percent": round(self.system_metrics.disk_percent, 2),
                "open_files": self.system_metrics.open_files,
                "active_threads": self.system_metrics.active_threads,
                "events_per_second": round(self.system_metrics.events_per_second, 2),
                "orders_per_minute": self.system_metrics.orders_per_minute,
                "avg_latency_ms": round(self.system_metrics.avg_latency_ms, 2)
            },
            "recent_alerts": [
                {
                    "severity": alert.severity,
                    "component": alert.component,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ]
        }

    async def perform_recovery(self, component_name: str) -> bool:
        """
        Attempt to recover a failed component.

        Args:
            component_name: Name of component to recover

        Returns:
            True if recovery successful
        """
        if component_name not in self.components:
            logger.error("Unknown component for recovery", component=component_name)
            return False

        component = self.components[component_name]

        logger.info(
            "Attempting component recovery",
            component=component_name,
            current_status=component.status
        )

        # Reset error counters
        component.consecutive_failures = 0
        component.status = HealthStatus.DEGRADED

        # Publish recovery event
        if self.event_bus:
            event = Event(
                type=EventType.COMPONENT_RECOVERY,
                data={
                    "component": component_name,
                    "previous_status": component.status,
                    "action": "reset"
                },
                priority=EventPriority.HIGH
            )
            await self.event_bus.publish(event)

        # Wait for component to recover
        await asyncio.sleep(5)

        # Check if recovery was successful
        if component.status == HealthStatus.HEALTHY:
            logger.info("Component recovery successful", component=component_name)
            return True
        else:
            logger.warning(
                "Component recovery failed",
                component=component_name,
                status=component.status
            )
            return False
