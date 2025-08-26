"""Observability module for monitoring, metrics, and alerting."""

import asyncio
import functools
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    INFO = "info"


class ObservabilityManager:
    """Central manager for all observability features."""

    def __init__(self):
        """Initialize observability components."""
        self.registry = CollectorRegistry()
        self._metrics: dict[str, Any] = {}
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default application metrics."""
        # Request metrics
        self._metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        # Trading metrics
        self._metrics['trades_total'] = Counter(
            'trades_total',
            'Total number of trades executed',
            ['exchange', 'symbol', 'side', 'strategy'],
            registry=self.registry
        )

        self._metrics['trade_latency'] = Histogram(
            'trade_latency_seconds',
            'Trade execution latency',
            ['exchange', 'operation'],
            registry=self.registry
        )

        # Portfolio metrics
        self._metrics['portfolio_value'] = Gauge(
            'portfolio_value_usd',
            'Current portfolio value in USD',
            ['tier'],
            registry=self.registry
        )

        self._metrics['pnl_total'] = Gauge(
            'pnl_total_usd',
            'Total profit and loss in USD',
            ['strategy'],
            registry=self.registry
        )

        # System metrics
        self._metrics['system_health'] = Gauge(
            'system_health_score',
            'Overall system health score (0-100)',
            registry=self.registry
        )

        # Application info
        self._metrics['app_info'] = Info(
            'genesis_app',
            'Genesis application information',
            registry=self.registry
        )
        self._metrics['app_info'].info({
            'version': '1.0.0',
            'tier': 'sniper',
            'environment': 'development'
        })

    def record_metric(self, name: str, value: float, labels: dict | None = None):
        """Record a metric value."""
        if name not in self._metrics:
            logger.warning(f"Metric {name} not found")
            return

        metric = self._metrics[name]
        labels = labels or {}

        if isinstance(metric, Counter):
            metric.labels(**labels).inc(value)
        elif isinstance(metric, Gauge):
            metric.labels(**labels).set(value)
        elif isinstance(metric, Histogram):
            metric.labels(**labels).observe(value)

    def get_metrics(self) -> bytes:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry)

    def create_metric(self, name: str, metric_type: MetricType,
                     description: str, labels: list | None = None):
        """Create a new metric."""
        labels = labels or []

        if metric_type == MetricType.COUNTER:
            self._metrics[name] = Counter(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.GAUGE:
            self._metrics[name] = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.HISTOGRAM:
            self._metrics[name] = Histogram(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.INFO:
            self._metrics[name] = Info(name, description, registry=self.registry)


class HealthCheck:
    """Health check system for monitoring component status."""

    def __init__(self):
        """Initialize health check system."""
        self.checks: dict[str, Callable] = {}
        self.status_cache: dict[str, dict] = {}
        self.cache_ttl = timedelta(seconds=30)

    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func

    async def run_checks(self) -> dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }

        for name, check_func in self.checks.items():
            try:
                # Check cache first
                if name in self.status_cache:
                    cached = self.status_cache[name]
                    if datetime.utcnow() - cached['timestamp'] < self.cache_ttl:
                        results['checks'][name] = cached['result']
                        continue

                # Run the check
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                # Cache the result
                self.status_cache[name] = {
                    'timestamp': datetime.utcnow(),
                    'result': result
                }

                results['checks'][name] = result

                if not result.get('healthy', False):
                    results['status'] = 'unhealthy'

            except Exception as e:
                logger.error(f"Health check {name} failed", error=str(e))
                results['checks'][name] = {
                    'healthy': False,
                    'error': str(e)
                }
                results['status'] = 'unhealthy'

        return results

    def check_database(self) -> dict[str, Any]:
        """Check database connectivity."""
        try:
            # Placeholder for actual database check
            return {'healthy': True, 'latency_ms': 5}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    def check_exchange_connectivity(self) -> dict[str, Any]:
        """Check exchange API connectivity."""
        try:
            # Placeholder for actual exchange check
            return {'healthy': True, 'connected_exchanges': ['binance', 'coinbase']}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}


class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        """Initialize alert manager."""
        self.alert_handlers: dict[str, Callable] = {}
        self.alert_history: list = []
        self.max_history = 1000

    def register_handler(self, name: str, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers[name] = handler

    async def send_alert(self, level: str, title: str, message: str,
                         metadata: dict | None = None):
        """Send an alert through all registered handlers."""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'title': title,
            'message': message,
            'metadata': metadata or {}
        }

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Send through handlers
        for name, handler in self.alert_handlers.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler {name} failed", error=str(e))

    def get_recent_alerts(self, count: int = 10) -> list:
        """Get recent alerts from history."""
        return self.alert_history[-count:]


def monitor_performance(func):
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"Function {func.__name__} completed",
                       duration_seconds=duration)
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Function {func.__name__} failed",
                        duration_seconds=duration,
                        error=str(e))
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"Function {func.__name__} completed",
                       duration_seconds=duration)
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Function {func.__name__} failed",
                        duration_seconds=duration,
                        error=str(e))
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# Global instances
observability = ObservabilityManager()
health_check = HealthCheck()
alert_manager = AlertManager()

# Register default health checks
health_check.register_check('database', health_check.check_database)
health_check.register_check('exchange', health_check.check_exchange_connectivity)

__all__ = [
    'AlertManager',
    'HealthCheck',
    'MetricType',
    'ObservabilityManager',
    'alert_manager',
    'health_check',
    'monitor_performance',
    'observability'
]
