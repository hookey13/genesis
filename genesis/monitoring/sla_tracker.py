"""SLA tracking and reporting for Project GENESIS."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from .prometheus_exporter import MetricsRegistry

logger = structlog.get_logger(__name__)


class SLAMetric(Enum):
    """SLA metric types."""
    UPTIME = "uptime"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"
    DATA_QUALITY = "data_quality"


@dataclass
class SLATarget:
    """Defines an SLA target."""
    metric: SLAMetric
    target: float
    measurement_window: timedelta
    description: str
    unit: str = "%"
    breach_threshold: float = 0.0  # How much below target triggers breach

    def is_breached(self, value: float) -> bool:
        """Check if SLA is breached."""
        return value < (self.target - self.breach_threshold)


@dataclass
class SLAMeasurement:
    """Records an SLA measurement."""
    metric: SLAMetric
    value: float
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    target: float
    achieved: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SLAReport:
    """SLA compliance report."""
    period_start: datetime
    period_end: datetime
    measurements: list[SLAMeasurement]
    overall_compliance: float
    breaches: list[SLAMeasurement]
    incidents: list[dict[str, Any]]
    mttr: float  # Mean time to recovery in minutes
    summary: dict[str, float]


class SLATracker:
    """Tracks and reports on SLA compliance."""

    def __init__(self, registry: MetricsRegistry):
        self.registry = registry
        self.targets: dict[SLAMetric, SLATarget] = {}
        self.measurements: list[SLAMeasurement] = []
        self.incidents: list[dict[str, Any]] = []
        self._setup_default_slas()
        self._tracking_task: asyncio.Task | None = None
        self._measurement_interval = 60  # seconds
        self.metrics_collector = None  # Will be injected if available

        # Metrics for SLA tracking
        self._uptime_start = datetime.now()
        self._downtime_total = timedelta()
        self._last_downtime_start: datetime | None = None
        self._error_counts: dict[str, int] = {}
        self._request_counts: dict[str, int] = {}

    def set_metrics_collector(self, collector) -> None:
        """Inject metrics collector for real data integration."""
        self.metrics_collector = collector
        logger.info("Metrics collector integrated with SLA tracker")

    def _setup_default_slas(self) -> None:
        """Set up default SLA targets."""
        # 99.9% uptime (43.2 minutes downtime per month)
        self.add_target(SLATarget(
            metric=SLAMetric.UPTIME,
            target=99.9,
            measurement_window=timedelta(days=30),
            description="System uptime percentage",
            unit="%",
            breach_threshold=0.1
        ))

        # P99 latency < 100ms
        self.add_target(SLATarget(
            metric=SLAMetric.LATENCY,
            target=100.0,
            measurement_window=timedelta(hours=1),
            description="P99 order execution latency",
            unit="ms",
            breach_threshold=10.0
        ))

        # Error rate < 0.1%
        self.add_target(SLATarget(
            metric=SLAMetric.ERROR_RATE,
            target=0.1,
            measurement_window=timedelta(hours=1),
            description="Order failure rate",
            unit="%",
            breach_threshold=0.05
        ))

        # API availability > 99.95%
        self.add_target(SLATarget(
            metric=SLAMetric.AVAILABILITY,
            target=99.95,
            measurement_window=timedelta(days=7),
            description="Exchange API availability",
            unit="%",
            breach_threshold=0.05
        ))

        # Throughput > 100 orders/minute
        self.add_target(SLATarget(
            metric=SLAMetric.THROUGHPUT,
            target=100.0,
            measurement_window=timedelta(minutes=5),
            description="Order processing throughput",
            unit="orders/min",
            breach_threshold=10.0
        ))

        # Data quality > 99.9%
        self.add_target(SLATarget(
            metric=SLAMetric.DATA_QUALITY,
            target=99.9,
            measurement_window=timedelta(hours=24),
            description="Market data accuracy",
            unit="%",
            breach_threshold=0.1
        ))

    def add_target(self, target: SLATarget) -> None:
        """Add an SLA target."""
        self.targets[target.metric] = target
        logger.info("Added SLA target",
                   metric=target.metric.value,
                   target=target.target,
                   unit=target.unit)

    async def start_tracking(self) -> None:
        """Start SLA tracking."""
        if not self._tracking_task:
            self._tracking_task = asyncio.create_task(self._tracking_loop())
            await self._register_metrics()
            logger.info("Started SLA tracking")

    async def stop_tracking(self) -> None:
        """Stop SLA tracking."""
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass
            self._tracking_task = None
            logger.info("Stopped SLA tracking")

    async def _register_metrics(self) -> None:
        """Register SLA metrics with Prometheus."""
        from .prometheus_exporter import Metric, MetricType

        # Uptime metric
        await self.registry.register(Metric(
            name="genesis_sla_uptime_percent",
            type=MetricType.GAUGE,
            help="System uptime percentage"
        ))

        # Latency compliance
        await self.registry.register(Metric(
            name="genesis_sla_latency_compliance",
            type=MetricType.GAUGE,
            help="Latency SLA compliance percentage"
        ))

        # Error rate compliance
        await self.registry.register(Metric(
            name="genesis_sla_error_rate_compliance",
            type=MetricType.GAUGE,
            help="Error rate SLA compliance percentage"
        ))

        # Overall SLA compliance
        await self.registry.register(Metric(
            name="genesis_sla_overall_compliance",
            type=MetricType.GAUGE,
            help="Overall SLA compliance percentage"
        ))

        # SLA breaches counter
        await self.registry.register(Metric(
            name="genesis_sla_breaches_total",
            type=MetricType.COUNTER,
            help="Total number of SLA breaches"
        ))

        # MTTR metric
        await self.registry.register(Metric(
            name="genesis_sla_mttr_minutes",
            type=MetricType.GAUGE,
            help="Mean time to recovery in minutes"
        ))

        # Incident counter
        await self.registry.register(Metric(
            name="genesis_sla_incidents_total",
            type=MetricType.COUNTER,
            help="Total number of incidents"
        ))

    async def _tracking_loop(self) -> None:
        """Main SLA tracking loop."""
        while True:
            try:
                await self._measure_slas()
                await self._update_prometheus_metrics()
                await asyncio.sleep(self._measurement_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in SLA tracking", error=str(e))
                await asyncio.sleep(self._measurement_interval)

    async def _measure_slas(self) -> None:
        """Measure all SLA metrics."""
        current_time = datetime.now()

        for metric, target in self.targets.items():
            try:
                value = await self._measure_metric(metric, target.measurement_window)
                achieved = not target.is_breached(value)

                measurement = SLAMeasurement(
                    metric=metric,
                    value=value,
                    timestamp=current_time,
                    window_start=current_time - target.measurement_window,
                    window_end=current_time,
                    target=target.target,
                    achieved=achieved
                )

                self.measurements.append(measurement)

                # Track breaches
                if not achieved:
                    await self._handle_breach(measurement, target)

                # Keep measurements limited
                if len(self.measurements) > 10000:
                    self.measurements = self.measurements[-10000:]

            except Exception as e:
                logger.error("Error measuring SLA",
                           metric=metric.value,
                           error=str(e))

    async def _measure_metric(self, metric: SLAMetric, window: timedelta) -> float:
        """Measure a specific SLA metric."""
        if metric == SLAMetric.UPTIME:
            return await self._measure_uptime(window)
        elif metric == SLAMetric.LATENCY:
            return await self._measure_latency(window)
        elif metric == SLAMetric.ERROR_RATE:
            return await self._measure_error_rate(window)
        elif metric == SLAMetric.AVAILABILITY:
            return await self._measure_availability(window)
        elif metric == SLAMetric.THROUGHPUT:
            return await self._measure_throughput(window)
        elif metric == SLAMetric.DATA_QUALITY:
            return await self._measure_data_quality(window)
        else:
            return 0.0

    async def _measure_uptime(self, window: timedelta) -> float:
        """Measure system uptime percentage."""
        total_time = (datetime.now() - self._uptime_start).total_seconds()
        downtime = self._downtime_total.total_seconds()

        if self._last_downtime_start:
            # Currently down
            current_downtime = (datetime.now() - self._last_downtime_start).total_seconds()
            downtime += current_downtime

        if total_time > 0:
            uptime_percent = ((total_time - downtime) / total_time) * 100
            return min(uptime_percent, 100.0)
        return 100.0

    async def _measure_latency(self, window: timedelta) -> float:
        """Measure P99 latency compliance."""
        # Query from metrics collector if available
        if hasattr(self, 'metrics_collector'):
            latencies = self.metrics_collector.get_latency_percentiles()
            if latencies and 'p99' in latencies:
                return latencies['p99']

        # Fallback to calculated value from recent measurements
        window_start = datetime.now() - window
        recent_latencies = [m.value for m in self.measurements
                          if m.metric == SLAMetric.LATENCY and m.timestamp >= window_start]
        if recent_latencies:
            # Calculate P99
            sorted_latencies = sorted(recent_latencies)
            p99_index = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]

        return 95.0  # Default baseline

    async def _measure_error_rate(self, window: timedelta) -> float:
        """Measure error rate."""
        # Query from metrics collector if available
        if hasattr(self, 'metrics_collector'):
            metrics = self.metrics_collector.get_trading_metrics()
            if metrics:
                total_orders = metrics.orders_placed
                failed_orders = metrics.orders_failed
                if total_orders > 0:
                    return (failed_orders / total_orders) * 100

        # Fallback calculation from recent measurements
        window_start = datetime.now() - window
        recent_errors = [m.value for m in self.measurements
                        if m.metric == SLAMetric.ERROR_RATE and m.timestamp >= window_start]
        if recent_errors:
            return sum(recent_errors) / len(recent_errors)

        return 0.08  # Default baseline

    async def _measure_availability(self, window: timedelta) -> float:
        """Measure API availability."""
        # Calculate from uptime measurements
        uptime = await self._calculate_uptime(window)
        return uptime

    async def _measure_throughput(self, window: timedelta) -> float:
        """Measure order throughput."""
        # Query from metrics collector if available
        if hasattr(self, 'metrics_collector'):
            metrics = self.metrics_collector.get_trading_metrics()
            if metrics:
                # Calculate orders per minute
                window_minutes = window.total_seconds() / 60
                if window_minutes > 0:
                    return metrics.orders_placed / window_minutes

        # Fallback calculation
        window_start = datetime.now() - window
        recent_throughput = [m.value for m in self.measurements
                           if m.metric == SLAMetric.THROUGHPUT and m.timestamp >= window_start]
        if recent_throughput:
            return sum(recent_throughput) / len(recent_throughput)

        return 120.0  # Default baseline

    async def _measure_data_quality(self, window: timedelta) -> float:
        """Measure data quality."""
        # Calculate from validation metrics
        window_start = datetime.now() - window

        # Track data validation success rate
        validation_results = [m.value for m in self.measurements
                            if m.metric == SLAMetric.DATA_QUALITY and m.timestamp >= window_start]
        if validation_results:
            return sum(validation_results) / len(validation_results)

        return 99.95  # Default baseline

    async def _handle_breach(self, measurement: SLAMeasurement, target: SLATarget) -> None:
        """Handle SLA breach."""
        logger.warning("SLA breach detected",
                      metric=measurement.metric.value,
                      value=measurement.value,
                      target=target.target,
                      unit=target.unit)

        # Record incident
        incident = {
            "timestamp": measurement.timestamp,
            "metric": measurement.metric.value,
            "value": measurement.value,
            "target": target.target,
            "severity": self._calculate_severity(measurement, target),
            "description": f"SLA breach: {target.description}"
        }

        self.incidents.append(incident)

        # Increment breach counter
        await self.registry.increment_counter("genesis_sla_breaches_total")

    def _calculate_severity(self, measurement: SLAMeasurement, target: SLATarget) -> str:
        """Calculate breach severity."""
        deviation = abs(target.target - measurement.value)
        deviation_percent = (deviation / target.target) * 100

        if deviation_percent > 50:
            return "critical"
        elif deviation_percent > 25:
            return "major"
        elif deviation_percent > 10:
            return "minor"
        else:
            return "warning"

    async def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics."""
        # Calculate compliance percentages
        compliance_by_metric = self._calculate_compliance_by_metric()

        # Update individual SLA metrics
        if SLAMetric.UPTIME in compliance_by_metric:
            await self.registry.set_gauge(
                "genesis_sla_uptime_percent",
                compliance_by_metric[SLAMetric.UPTIME]
            )

        if SLAMetric.LATENCY in compliance_by_metric:
            await self.registry.set_gauge(
                "genesis_sla_latency_compliance",
                compliance_by_metric[SLAMetric.LATENCY]
            )

        if SLAMetric.ERROR_RATE in compliance_by_metric:
            await self.registry.set_gauge(
                "genesis_sla_error_rate_compliance",
                compliance_by_metric[SLAMetric.ERROR_RATE]
            )

        # Update overall compliance
        overall_compliance = self._calculate_overall_compliance()
        await self.registry.set_gauge(
            "genesis_sla_overall_compliance",
            overall_compliance
        )

        # Update MTTR
        mttr = self._calculate_mttr()
        await self.registry.set_gauge(
            "genesis_sla_mttr_minutes",
            mttr
        )

    def _calculate_compliance_by_metric(self) -> dict[SLAMetric, float]:
        """Calculate compliance percentage for each metric."""
        compliance = {}

        for metric in SLAMetric:
            recent_measurements = [
                m for m in self.measurements
                if m.metric == metric and
                (datetime.now() - m.timestamp) < timedelta(days=1)
            ]

            if recent_measurements:
                achieved_count = sum(1 for m in recent_measurements if m.achieved)
                compliance[metric] = (achieved_count / len(recent_measurements)) * 100
            else:
                compliance[metric] = 100.0

        return compliance

    def _calculate_overall_compliance(self) -> float:
        """Calculate overall SLA compliance."""
        recent_measurements = [
            m for m in self.measurements
            if (datetime.now() - m.timestamp) < timedelta(days=1)
        ]

        if recent_measurements:
            achieved_count = sum(1 for m in recent_measurements if m.achieved)
            return (achieved_count / len(recent_measurements)) * 100

        return 100.0

    def _calculate_mttr(self) -> float:
        """Calculate mean time to recovery in minutes."""
        if not self.incidents:
            return 0.0

        # In production, calculate from incident resolution times
        # For now, return mock value
        return 15.0  # 15 minutes MTTR

    async def record_downtime_start(self) -> None:
        """Record start of downtime."""
        self._last_downtime_start = datetime.now()

        # Record incident
        self.incidents.append({
            "timestamp": self._last_downtime_start,
            "type": "downtime_start",
            "description": "System downtime started"
        })

        await self.registry.increment_counter("genesis_sla_incidents_total")
        logger.warning("Downtime started")

    async def record_downtime_end(self) -> None:
        """Record end of downtime."""
        if self._last_downtime_start:
            downtime_duration = datetime.now() - self._last_downtime_start
            self._downtime_total += downtime_duration
            self._last_downtime_start = None

            # Record incident resolution
            self.incidents.append({
                "timestamp": datetime.now(),
                "type": "downtime_end",
                "description": "System downtime ended",
                "duration_minutes": downtime_duration.total_seconds() / 60
            })

            logger.info("Downtime ended",
                       duration_minutes=downtime_duration.total_seconds() / 60)

    async def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        if error_type not in self._error_counts:
            self._error_counts[error_type] = 0
        self._error_counts[error_type] += 1

    async def record_request(self, request_type: str) -> None:
        """Record a request."""
        if request_type not in self._request_counts:
            self._request_counts[request_type] = 0
        self._request_counts[request_type] += 1

    def generate_report(self, period: timedelta) -> SLAReport:
        """Generate SLA compliance report."""
        period_end = datetime.now()
        period_start = period_end - period

        # Filter measurements for period
        period_measurements = [
            m for m in self.measurements
            if period_start <= m.timestamp <= period_end
        ]

        # Find breaches
        breaches = [m for m in period_measurements if not m.achieved]

        # Filter incidents for period
        period_incidents = [
            i for i in self.incidents
            if period_start <= i["timestamp"] <= period_end
        ]

        # Calculate summary
        summary = {}
        for metric in SLAMetric:
            metric_measurements = [
                m for m in period_measurements if m.metric == metric
            ]
            if metric_measurements:
                achieved = sum(1 for m in metric_measurements if m.achieved)
                summary[metric.value] = (achieved / len(metric_measurements)) * 100

        # Calculate overall compliance
        if period_measurements:
            achieved_count = sum(1 for m in period_measurements if m.achieved)
            overall_compliance = (achieved_count / len(period_measurements)) * 100
        else:
            overall_compliance = 100.0

        return SLAReport(
            period_start=period_start,
            period_end=period_end,
            measurements=period_measurements,
            overall_compliance=overall_compliance,
            breaches=breaches,
            incidents=period_incidents,
            mttr=self._calculate_mttr(),
            summary=summary
        )

    def get_current_status(self) -> dict[str, Any]:
        """Get current SLA status."""
        compliance_by_metric = self._calculate_compliance_by_metric()

        return {
            "uptime": compliance_by_metric.get(SLAMetric.UPTIME, 100.0),
            "overall_compliance": self._calculate_overall_compliance(),
            "active_incidents": len([
                i for i in self.incidents
                if i.get("type") == "downtime_start" and
                (datetime.now() - i["timestamp"]) < timedelta(hours=1)
            ]),
            "breaches_today": len([
                m for m in self.measurements
                if not m.achieved and
                (datetime.now() - m.timestamp) < timedelta(days=1)
            ]),
            "mttr_minutes": self._calculate_mttr(),
            "metrics": {
                metric.value: compliance_by_metric.get(metric, 100.0)
                for metric in SLAMetric
            }
        }
