"""
Health monitoring for exchange connections.

Monitors API response times, success rates, and connection states
to provide early warning of degraded performance.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Single health metric measurement."""

    timestamp: float
    response_time_ms: float | None = None
    success: bool = True
    error: str | None = None


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    last_check: float
    success_rate: float
    avg_response_time_ms: float
    error_count: int
    last_error: str | None = None
    details: dict = field(default_factory=dict)


class HealthMonitor:
    """
    Monitors health of exchange connections and services.

    Tracks response times, success rates, and provides
    health check endpoints.
    """

    def __init__(
        self,
        check_interval_seconds: int = 60,
        window_size: int = 100,
        degraded_threshold: float = 0.95,
        unhealthy_threshold: float = 0.80,
        response_time_threshold_ms: float = 1000,
    ):
        """
        Initialize health monitor.

        Args:
            check_interval_seconds: Interval between health checks
            window_size: Number of metrics to keep in rolling window
            degraded_threshold: Success rate below this is degraded
            unhealthy_threshold: Success rate below this is unhealthy
            response_time_threshold_ms: Response time above this is slow
        """
        self.check_interval = check_interval_seconds
        self.window_size = window_size
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold
        self.response_time_threshold_ms = response_time_threshold_ms

        # Metrics storage
        self.metrics: dict[str, deque] = {}

        # Components to monitor
        self.components = {}

        # Monitoring state
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

        # Callbacks for health changes
        self.health_change_callbacks = []

        logger.info(
            "HealthMonitor initialized",
            check_interval=check_interval_seconds,
            degraded_threshold=degraded_threshold,
            unhealthy_threshold=unhealthy_threshold,
        )

    def register_component(self, name: str, check_func: callable) -> None:
        """
        Register a component for health monitoring.

        Args:
            name: Component name
            check_func: Async function that returns health data
        """
        self.components[name] = check_func
        self.metrics[name] = deque(maxlen=self.window_size)
        logger.info(f"Registered component for health monitoring: {name}")

    def register_health_change_callback(self, callback: callable) -> None:
        """
        Register callback for health status changes.

        Args:
            callback: Function to call on health change
        """
        self.health_change_callbacks.append(callback)

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring = False

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Health monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Run health checks
                await self.check_all_components()

                # Wait for next interval
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
                await asyncio.sleep(10)  # Brief pause on error

    async def check_all_components(self) -> dict[str, ComponentHealth]:
        """
        Check health of all registered components.

        Returns:
            Dictionary of component health statuses
        """
        results = {}

        for name, check_func in self.components.items():
            health = await self.check_component(name, check_func)
            results[name] = health

            # Check for status changes
            self._check_health_change(name, health)

        return results

    async def check_component(self, name: str, check_func: callable) -> ComponentHealth:
        """
        Check health of a single component.

        Args:
            name: Component name
            check_func: Health check function

        Returns:
            Component health status
        """
        start_time = time.time()
        metric = HealthMetric(timestamp=start_time)

        try:
            # Run health check
            result = await asyncio.wait_for(check_func(), timeout=10)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            metric.response_time_ms = response_time_ms
            metric.success = result.get("success", True)

            if not metric.success:
                metric.error = result.get("error", "Health check failed")

        except TimeoutError:
            metric.success = False
            metric.error = "Health check timeout"
            metric.response_time_ms = 10000  # Timeout value

        except Exception as e:
            metric.success = False
            metric.error = str(e)
            metric.response_time_ms = (time.time() - start_time) * 1000

        # Store metric
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
        self.metrics[name].append(metric)

        # Calculate health status
        return self._calculate_health(name)

    def _calculate_health(self, component_name: str) -> ComponentHealth:
        """
        Calculate health status for a component.

        Args:
            component_name: Component name

        Returns:
            Component health status
        """
        metrics = self.metrics.get(component_name, [])

        if not metrics:
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNKNOWN,
                last_check=time.time(),
                success_rate=0.0,
                avg_response_time_ms=0.0,
                error_count=0,
            )

        # Calculate success rate
        successful = sum(1 for m in metrics if m.success)
        success_rate = successful / len(metrics)

        # Calculate average response time
        response_times = [m.response_time_ms for m in metrics if m.response_time_ms]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Count errors
        error_count = len(metrics) - successful

        # Get last error
        last_error = None
        for metric in reversed(metrics):
            if not metric.success and metric.error:
                last_error = metric.error
                break

        # Determine health status
        if success_rate < self.unhealthy_threshold:
            status = HealthStatus.UNHEALTHY
        elif (
            success_rate < self.degraded_threshold
            or avg_response_time > self.response_time_threshold_ms
        ):
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return ComponentHealth(
            name=component_name,
            status=status,
            last_check=metrics[-1].timestamp,
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            error_count=error_count,
            last_error=last_error,
            details={
                "total_checks": len(metrics),
                "successful_checks": successful,
                "window_size": self.window_size,
            },
        )

    def _check_health_change(self, name: str, health: ComponentHealth) -> None:
        """
        Check for health status changes and notify callbacks.

        Args:
            name: Component name
            health: Current health status
        """
        # Store previous status
        previous_key = f"{name}_previous_status"
        previous_status = getattr(self, previous_key, None)

        if previous_status and previous_status != health.status:
            # Status changed
            logger.warning(
                f"Health status changed for {name}",
                previous=previous_status,
                current=health.status,
                success_rate=health.success_rate,
            )

            # Notify callbacks
            for callback in self.health_change_callbacks:
                try:
                    asyncio.create_task(callback(name, previous_status, health))
                except Exception as e:
                    logger.error("Health callback error", error=str(e))

        # Update stored status
        setattr(self, previous_key, health.status)

    def get_component_health(self, name: str) -> ComponentHealth | None:
        """
        Get current health status for a component.

        Args:
            name: Component name

        Returns:
            Component health or None if not found
        """
        if name in self.metrics:
            return self._calculate_health(name)
        return None

    def get_all_health(self) -> dict[str, ComponentHealth]:
        """
        Get health status for all components.

        Returns:
            Dictionary of all component health statuses
        """
        return {name: self._calculate_health(name) for name in self.metrics.keys()}

    def is_healthy(self) -> bool:
        """
        Check if all components are healthy.

        Returns:
            True if all components are healthy
        """
        all_health = self.get_all_health()
        return all(
            health.status == HealthStatus.HEALTHY for health in all_health.values()
        )

    def get_statistics(self) -> dict:
        """
        Get health monitoring statistics.

        Returns:
            Dictionary of statistics
        """
        all_health = self.get_all_health()

        healthy_count = sum(
            1 for h in all_health.values() if h.status == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for h in all_health.values() if h.status == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for h in all_health.values() if h.status == HealthStatus.UNHEALTHY
        )

        return {
            "monitoring": self._monitoring,
            "total_components": len(self.components),
            "healthy_components": healthy_count,
            "degraded_components": degraded_count,
            "unhealthy_components": unhealthy_count,
            "overall_status": self._get_overall_status(),
            "components": {
                name: {
                    "status": health.status,
                    "success_rate": health.success_rate,
                    "avg_response_time_ms": health.avg_response_time_ms,
                    "error_count": health.error_count,
                }
                for name, health in all_health.items()
            },
        }

    def _get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall health status
        """
        all_health = self.get_all_health()

        if not all_health:
            return HealthStatus.UNKNOWN

        # If any component is unhealthy, system is unhealthy
        if any(h.status == HealthStatus.UNHEALTHY for h in all_health.values()):
            return HealthStatus.UNHEALTHY

        # If any component is degraded, system is degraded
        if any(h.status == HealthStatus.DEGRADED for h in all_health.values()):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY
