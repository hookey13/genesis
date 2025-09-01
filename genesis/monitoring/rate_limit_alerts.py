"""
Rate limit alerting integration for monitoring and alerting on rate limit thresholds.
Integrates with Prometheus metrics from Story 8.5 for comprehensive monitoring.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when Prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass

from genesis.core.circuit_breaker import CircuitBreaker, CircuitState
from genesis.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    PROMETHEUS = "prometheus"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    threshold: float
    current_value: float
    timestamp: datetime
    metadata: dict[str, Any]


@dataclass
class AlertThreshold:
    """Configuration for alert thresholds."""
    metric: str
    warning_threshold: float
    critical_threshold: float
    evaluation_window: int  # seconds
    cooldown_period: int  # seconds to wait before re-alerting


class RateLimitAlerter:
    """
    Monitors rate limiter and circuit breaker metrics, generating alerts
    when thresholds are exceeded. Integrates with Prometheus for metrics.
    """

    # Prometheus metrics (only created if Prometheus is available)
    if PROMETHEUS_AVAILABLE:
        rate_limit_usage = Gauge(
            'genesis_rate_limit_usage_ratio',
            'Current rate limit usage as a ratio of capacity',
            ['endpoint', 'priority']
        )

        rate_limit_rejections = Counter(
            'genesis_rate_limit_rejections_total',
            'Total number of rate limit rejections',
            ['endpoint', 'priority', 'reason']
        )

        circuit_breaker_state = Gauge(
            'genesis_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['component']
        )

        alert_triggered = Counter(
            'genesis_alerts_triggered_total',
            'Total number of alerts triggered',
            ['severity', 'component', 'alert_type']
        )

        rate_limit_wait_time = Histogram(
            'genesis_rate_limit_wait_seconds',
            'Time spent waiting for rate limit availability',
            ['endpoint', 'priority']
        )

    def __init__(
        self,
        rate_limiter: RateLimiter,
        circuit_breaker: CircuitBreaker,
        thresholds: list[AlertThreshold] | None = None
    ):
        """
        Initialize the rate limit alerter.
        
        Args:
            rate_limiter: Rate limiter instance to monitor
            circuit_breaker: Circuit breaker instance to monitor
            thresholds: Custom alert thresholds
        """
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.thresholds = thresholds or self._default_thresholds()
        self.alert_history: dict[str, datetime] = {}
        self.alert_handlers: dict[AlertChannel, Callable] = {}
        self._monitoring_task: asyncio.Task | None = None
        self._running = False

        # Register default handlers
        self._register_default_handlers()

    def _default_thresholds(self) -> list[AlertThreshold]:
        """Get default alert thresholds."""
        return [
            AlertThreshold(
                metric="rate_limit_usage",
                warning_threshold=0.8,  # 80% usage
                critical_threshold=0.95,  # 95% usage
                evaluation_window=60,
                cooldown_period=300
            ),
            AlertThreshold(
                metric="circuit_breaker_open",
                warning_threshold=1,  # Any circuit open
                critical_threshold=3,  # Multiple circuits open
                evaluation_window=30,
                cooldown_period=180
            ),
            AlertThreshold(
                metric="rejection_rate",
                warning_threshold=0.05,  # 5% rejection rate
                critical_threshold=0.15,  # 15% rejection rate
                evaluation_window=120,
                cooldown_period=600
            ),
            AlertThreshold(
                metric="average_wait_time",
                warning_threshold=1.0,  # 1 second average wait
                critical_threshold=5.0,  # 5 seconds average wait
                evaluation_window=60,
                cooldown_period=300
            )
        ]

    def _register_default_handlers(self):
        """Register default alert handlers."""
        self.alert_handlers[AlertChannel.LOG] = self._handle_log_alert
        if PROMETHEUS_AVAILABLE:
            self.alert_handlers[AlertChannel.PROMETHEUS] = self._handle_prometheus_alert

    async def start_monitoring(self, interval: int = 10):
        """
        Start the monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._running:
            logger.warning("Monitoring already running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )
        logger.info("Rate limit monitoring started", interval=interval)

    async def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Rate limit monitoring stopped")

    async def _monitoring_loop(self, interval: int):
        """
        Main monitoring loop that checks metrics and generates alerts.
        
        Args:
            interval: Check interval in seconds
        """
        while self._running:
            try:
                await self._check_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(interval)

    async def _check_metrics(self):
        """Check all metrics against thresholds and generate alerts."""
        # Get current metrics
        rate_metrics = self.rate_limiter.get_metrics()
        circuit_metrics = self.circuit_breaker.get_statistics()

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._update_prometheus_metrics(rate_metrics, circuit_metrics)

        # Check rate limit usage
        await self._check_rate_limit_usage(rate_metrics)

        # Check circuit breaker state
        await self._check_circuit_breaker_state(circuit_metrics)

        # Check rejection rate
        await self._check_rejection_rate(rate_metrics)

        # Check average wait time
        await self._check_wait_time(rate_metrics)

    def _update_prometheus_metrics(
        self,
        rate_metrics: dict[str, Any],
        circuit_metrics: dict[str, Any]
    ):
        """Update Prometheus metrics with current values."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Update rate limit usage
        usage_ratio = 1.0 - (rate_metrics['available'] / rate_metrics['capacity'])
        self.rate_limit_usage.labels(
            endpoint='default',
            priority='all'
        ).set(usage_ratio)

        # Update circuit breaker state
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 1,
            CircuitState.HALF_OPEN: 2
        }.get(self.circuit_breaker.state, -1)

        self.circuit_breaker_state.labels(
            component='exchange_gateway'
        ).set(state_value)

    async def _check_rate_limit_usage(self, metrics: dict[str, Any]):
        """Check rate limit usage against thresholds."""
        threshold = next(
            (t for t in self.thresholds if t.metric == "rate_limit_usage"),
            None
        )
        if not threshold:
            return

        usage_ratio = 1.0 - (metrics['available'] / metrics['capacity'])

        if usage_ratio >= threshold.critical_threshold:
            await self._trigger_alert(
                Alert(
                    id=f"rate_limit_critical_{datetime.now().timestamp()}",
                    severity=AlertSeverity.CRITICAL,
                    title="Critical Rate Limit Usage",
                    message=f"Rate limit usage at {usage_ratio:.1%} of capacity",
                    component="rate_limiter",
                    threshold=threshold.critical_threshold,
                    current_value=usage_ratio,
                    timestamp=datetime.now(),
                    metadata={
                        "capacity": metrics['capacity'],
                        "available": metrics['available'],
                        "refill_rate": metrics['refill_rate']
                    }
                ),
                threshold
            )
        elif usage_ratio >= threshold.warning_threshold:
            await self._trigger_alert(
                Alert(
                    id=f"rate_limit_warning_{datetime.now().timestamp()}",
                    severity=AlertSeverity.WARNING,
                    title="High Rate Limit Usage",
                    message=f"Rate limit usage at {usage_ratio:.1%} of capacity",
                    component="rate_limiter",
                    threshold=threshold.warning_threshold,
                    current_value=usage_ratio,
                    timestamp=datetime.now(),
                    metadata={
                        "capacity": metrics['capacity'],
                        "available": metrics['available'],
                        "refill_rate": metrics['refill_rate']
                    }
                ),
                threshold
            )

    async def _check_circuit_breaker_state(self, metrics: dict[str, Any]):
        """Check circuit breaker state and alert if open."""
        threshold = next(
            (t for t in self.thresholds if t.metric == "circuit_breaker_open"),
            None
        )
        if not threshold:
            return

        if self.circuit_breaker.state == CircuitState.OPEN:
            await self._trigger_alert(
                Alert(
                    id=f"circuit_open_{datetime.now().timestamp()}",
                    severity=AlertSeverity.CRITICAL,
                    title="Circuit Breaker Open",
                    message="Circuit breaker is open, external API calls are blocked",
                    component="circuit_breaker",
                    threshold=1,
                    current_value=1,
                    timestamp=datetime.now(),
                    metadata={
                        "failure_count": metrics['failure_count'],
                        "success_count": metrics['success_count'],
                        "state": str(self.circuit_breaker.state)
                    }
                ),
                threshold
            )
        elif self.circuit_breaker.state == CircuitState.HALF_OPEN:
            await self._trigger_alert(
                Alert(
                    id=f"circuit_half_open_{datetime.now().timestamp()}",
                    severity=AlertSeverity.WARNING,
                    title="Circuit Breaker Half-Open",
                    message="Circuit breaker is testing recovery",
                    component="circuit_breaker",
                    threshold=0.5,
                    current_value=0.5,
                    timestamp=datetime.now(),
                    metadata={
                        "half_open_calls": metrics.get('half_open_calls', 0),
                        "state": str(self.circuit_breaker.state)
                    }
                ),
                threshold
            )

    async def _check_rejection_rate(self, metrics: dict[str, Any]):
        """Check request rejection rate."""
        threshold = next(
            (t for t in self.thresholds if t.metric == "rejection_rate"),
            None
        )
        if not threshold:
            return

        # Calculate rejection rate from metrics
        total_requests = metrics.get('total_requests', 0)
        rejected_requests = metrics.get('rejected_requests', 0)

        if total_requests > 0:
            rejection_rate = rejected_requests / total_requests

            if rejection_rate >= threshold.critical_threshold:
                await self._trigger_alert(
                    Alert(
                        id=f"rejection_critical_{datetime.now().timestamp()}",
                        severity=AlertSeverity.CRITICAL,
                        title="Critical Request Rejection Rate",
                        message=f"Rejection rate at {rejection_rate:.1%}",
                        component="rate_limiter",
                        threshold=threshold.critical_threshold,
                        current_value=rejection_rate,
                        timestamp=datetime.now(),
                        metadata={
                            "total_requests": total_requests,
                            "rejected_requests": rejected_requests
                        }
                    ),
                    threshold
                )
            elif rejection_rate >= threshold.warning_threshold:
                await self._trigger_alert(
                    Alert(
                        id=f"rejection_warning_{datetime.now().timestamp()}",
                        severity=AlertSeverity.WARNING,
                        title="High Request Rejection Rate",
                        message=f"Rejection rate at {rejection_rate:.1%}",
                        component="rate_limiter",
                        threshold=threshold.warning_threshold,
                        current_value=rejection_rate,
                        timestamp=datetime.now(),
                        metadata={
                            "total_requests": total_requests,
                            "rejected_requests": rejected_requests
                        }
                    ),
                    threshold
                )

    async def _check_wait_time(self, metrics: dict[str, Any]):
        """Check average wait time for rate limited requests."""
        threshold = next(
            (t for t in self.thresholds if t.metric == "average_wait_time"),
            None
        )
        if not threshold:
            return

        avg_wait_time = metrics.get('average_wait_time', 0)

        if avg_wait_time >= threshold.critical_threshold:
            await self._trigger_alert(
                Alert(
                    id=f"wait_time_critical_{datetime.now().timestamp()}",
                    severity=AlertSeverity.CRITICAL,
                    title="Critical Rate Limit Wait Time",
                    message=f"Average wait time is {avg_wait_time:.1f} seconds",
                    component="rate_limiter",
                    threshold=threshold.critical_threshold,
                    current_value=avg_wait_time,
                    timestamp=datetime.now(),
                    metadata={
                        "max_wait_time": metrics.get('max_wait_time', 0)
                    }
                ),
                threshold
            )
        elif avg_wait_time >= threshold.warning_threshold:
            await self._trigger_alert(
                Alert(
                    id=f"wait_time_warning_{datetime.now().timestamp()}",
                    severity=AlertSeverity.WARNING,
                    title="High Rate Limit Wait Time",
                    message=f"Average wait time is {avg_wait_time:.1f} seconds",
                    component="rate_limiter",
                    threshold=threshold.warning_threshold,
                    current_value=avg_wait_time,
                    timestamp=datetime.now(),
                    metadata={
                        "max_wait_time": metrics.get('max_wait_time', 0)
                    }
                ),
                threshold
            )

    async def _trigger_alert(self, alert: Alert, threshold: AlertThreshold):
        """
        Trigger an alert if cooldown period has passed.
        
        Args:
            alert: Alert to trigger
            threshold: Threshold configuration
        """
        # Check cooldown
        alert_key = f"{alert.component}_{threshold.metric}"
        last_alert = self.alert_history.get(alert_key)

        if last_alert:
            cooldown_elapsed = (datetime.now() - last_alert).total_seconds()
            if cooldown_elapsed < threshold.cooldown_period:
                return  # Still in cooldown

        # Update alert history
        self.alert_history[alert_key] = datetime.now()

        # Send to all registered handlers
        for channel, handler in self.alert_handlers.items():
            try:
                await handler(alert)
            except Exception as e:
                logger.error(
                    "Failed to send alert",
                    channel=channel.value,
                    error=str(e)
                )

        # Update Prometheus counter
        if PROMETHEUS_AVAILABLE:
            self.alert_triggered.labels(
                severity=alert.severity.value,
                component=alert.component,
                alert_type=threshold.metric
            ).inc()

    async def _handle_log_alert(self, alert: Alert):
        """Handle alert by logging."""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.error)

        log_method(
            alert.title,
            message=alert.message,
            component=alert.component,
            threshold=alert.threshold,
            current_value=alert.current_value,
            metadata=alert.metadata
        )

    async def _handle_prometheus_alert(self, alert: Alert):
        """Handle alert by updating Prometheus alert metrics."""
        # Prometheus alerting is handled by the alert_triggered counter
        # and the continuous metric updates
        pass

    def register_handler(
        self,
        channel: AlertChannel,
        handler: Callable[[Alert], asyncio.Awaitable[None]]
    ):
        """
        Register a custom alert handler.
        
        Args:
            channel: Alert channel
            handler: Async function to handle alerts
        """
        self.alert_handlers[channel] = handler
        logger.info("Registered alert handler", channel=channel.value)

    def update_threshold(self, threshold: AlertThreshold):
        """
        Update or add an alert threshold.
        
        Args:
            threshold: New threshold configuration
        """
        # Remove existing threshold for this metric
        self.thresholds = [
            t for t in self.thresholds
            if t.metric != threshold.metric
        ]
        # Add new threshold
        self.thresholds.append(threshold)
        logger.info(
            "Updated alert threshold",
            metric=threshold.metric,
            warning=threshold.warning_threshold,
            critical=threshold.critical_threshold
        )
