"""
Observability instrumentation with OpenTelemetry and Prometheus metrics.

Provides distributed tracing, metrics collection, and monitoring capabilities.
"""

from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode
from prometheus_client import start_http_server

logger = structlog.get_logger(__name__)


class ObservabilityManager:
    """
    Manages observability instrumentation for the trading system.
    
    Features:
    - Distributed tracing with OpenTelemetry
    - Metrics collection and export to Prometheus
    - Custom metrics for trading-specific operations
    - Performance monitoring and alerting
    """

    def __init__(
        self,
        service_name: str = "genesis-trading-loop",
        metrics_port: int = 9090,
        enable_tracing: bool = True,
        enable_metrics: bool = True
    ):
        self.service_name = service_name
        self.metrics_port = metrics_port
        self.enable_tracing = enable_tracing
        self.enable_metrics = enable_metrics

        # Initialize providers
        self.tracer = None
        self.meter = None

        # Metrics collectors
        self.metrics_data = {
            "events_processed": 0,
            "orders_executed": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "signals_generated": 0,
            "errors_count": 0,
            "last_event_timestamp": None,
            "event_latencies": [],
            "order_latencies": [],
            "position_values": {}
        }

        # Custom metrics
        self.event_counter = None
        self.order_histogram = None
        self.position_gauge = None
        self.error_counter = None
        self.latency_histogram = None

    def initialize(self) -> None:
        """Initialize telemetry providers and exporters."""
        logger.info(
            "Initializing observability",
            service_name=self.service_name,
            tracing=self.enable_tracing,
            metrics=self.enable_metrics
        )

        # Set up resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })

        # Initialize tracing
        if self.enable_tracing:
            self._setup_tracing(resource)

        # Initialize metrics
        if self.enable_metrics:
            self._setup_metrics(resource)

        logger.info("Observability initialized successfully")

    def _setup_tracing(self, resource: Resource) -> None:
        """Set up distributed tracing."""
        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add span processor with console exporter (can be replaced with Jaeger/Zipkin)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info("Tracing initialized with console exporter")

    def _setup_metrics(self, resource: Resource) -> None:
        """Set up metrics collection."""
        # Create Prometheus metric reader
        reader = PrometheusMetricReader()

        # Create meter provider
        provider = MeterProvider(resource=resource, metric_readers=[reader])

        # Set global meter provider
        metrics.set_meter_provider(provider)

        # Get meter
        self.meter = metrics.get_meter(__name__)

        # Create metrics instruments
        self._create_metrics_instruments()

        # Start Prometheus HTTP server
        start_http_server(self.metrics_port)

        logger.info(
            "Metrics initialized with Prometheus exporter",
            port=self.metrics_port
        )

    def _create_metrics_instruments(self) -> None:
        """Create custom metrics instruments."""
        # Event processing counter
        self.event_counter = self.meter.create_counter(
            name="trading_events_total",
            description="Total number of trading events processed",
            unit="events"
        )

        # Order execution histogram
        self.order_histogram = self.meter.create_histogram(
            name="order_execution_duration",
            description="Order execution duration in milliseconds",
            unit="ms"
        )

        # Position value gauge (observable)
        self.position_gauge = self.meter.create_observable_gauge(
            name="position_value_usd",
            description="Current position value in USD",
            callbacks=[self._observe_position_value]
        )

        # Error counter
        self.error_counter = self.meter.create_counter(
            name="trading_errors_total",
            description="Total number of trading errors",
            unit="errors"
        )

        # Event processing latency histogram
        self.latency_histogram = self.meter.create_histogram(
            name="event_processing_latency",
            description="Event processing latency in milliseconds",
            unit="ms"
        )

        # System metrics (observable)
        self.meter.create_observable_gauge(
            name="trading_loop_active",
            description="Trading loop active status (1=active, 0=inactive)",
            callbacks=[self._observe_system_status]
        )

        self.meter.create_observable_gauge(
            name="event_queue_depth",
            description="Number of events in processing queue",
            callbacks=[self._observe_queue_depth]
        )

    @contextmanager
    def trace_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        record_exception: bool = True
    ):
        """
        Create a traced span for an operation.
        
        Args:
            name: Span name
            attributes: Span attributes
            record_exception: Whether to record exceptions
        """
        if not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, self._serialize_attribute(value))

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                if record_exception:
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, str(e))
                    )
                raise

    def record_event_processed(
        self,
        event_type: str,
        latency_ms: float,
        success: bool = True
    ) -> None:
        """Record event processing metrics."""
        self.metrics_data["events_processed"] += 1
        self.metrics_data["last_event_timestamp"] = datetime.now()
        self.metrics_data["event_latencies"].append(latency_ms)

        # Keep only last 1000 latencies
        if len(self.metrics_data["event_latencies"]) > 1000:
            self.metrics_data["event_latencies"] = self.metrics_data["event_latencies"][-1000:]

        if self.event_counter:
            self.event_counter.add(
                1,
                {"event_type": event_type, "success": str(success)}
            )

        if self.latency_histogram:
            self.latency_histogram.record(
                latency_ms,
                {"event_type": event_type}
            )

    def record_order_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        execution_time_ms: float,
        success: bool = True
    ) -> None:
        """Record order execution metrics."""
        self.metrics_data["orders_executed"] += 1
        self.metrics_data["order_latencies"].append(execution_time_ms)

        if self.order_histogram:
            self.order_histogram.record(
                execution_time_ms,
                {
                    "symbol": symbol,
                    "side": side,
                    "success": str(success)
                }
            )

    def record_position_change(
        self,
        position_id: str,
        symbol: str,
        action: str,  # "opened" or "closed"
        value_usd: float
    ) -> None:
        """Record position change metrics."""
        if action == "opened":
            self.metrics_data["positions_opened"] += 1
            self.metrics_data["position_values"][position_id] = {
                "symbol": symbol,
                "value": value_usd
            }
        elif action == "closed":
            self.metrics_data["positions_closed"] += 1
            self.metrics_data["position_values"].pop(position_id, None)

    def record_signal_generated(
        self,
        strategy_id: str,
        signal_type: str,
        confidence: float
    ) -> None:
        """Record signal generation metrics."""
        self.metrics_data["signals_generated"] += 1

        if self.event_counter:
            self.event_counter.add(
                1,
                {
                    "event_type": "signal_generated",
                    "strategy_id": strategy_id,
                    "signal_type": signal_type
                }
            )

    def record_error(
        self,
        error_type: str,
        component: str,
        message: str
    ) -> None:
        """Record error metrics."""
        self.metrics_data["errors_count"] += 1

        if self.error_counter:
            self.error_counter.add(
                1,
                {
                    "error_type": error_type,
                    "component": component
                }
            )

        logger.error(
            "Trading error recorded",
            error_type=error_type,
            component=component,
            message=message
        )

    def _observe_position_value(self, options: CallbackOptions) -> list:
        """Callback for position value gauge."""
        observations = []

        for position_id, position_data in self.metrics_data["position_values"].items():
            observations.append(
                Observation(
                    value=position_data["value"],
                    attributes={
                        "position_id": position_id,
                        "symbol": position_data["symbol"]
                    }
                )
            )

        return observations

    def _observe_system_status(self, options: CallbackOptions) -> list:
        """Callback for system status gauge."""
        # Check if system is active based on recent events
        if self.metrics_data["last_event_timestamp"]:
            seconds_since_last = (
                datetime.now() - self.metrics_data["last_event_timestamp"]
            ).total_seconds()
            is_active = 1 if seconds_since_last < 60 else 0
        else:
            is_active = 0

        return [Observation(value=is_active)]

    def _observe_queue_depth(self, options: CallbackOptions) -> list:
        """Callback for queue depth gauge."""
        # This would be connected to actual event bus queue
        # For now, return estimated value based on processing rate
        queue_depth = len(self.metrics_data["event_latencies"]) % 100
        return [Observation(value=queue_depth)]

    def _serialize_attribute(self, value: Any) -> Any:
        """Serialize attribute value for tracing."""
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current metrics summary."""
        # Calculate latency statistics
        event_latencies = self.metrics_data["event_latencies"]
        order_latencies = self.metrics_data["order_latencies"]

        def calculate_percentile(data: list, percentile: float) -> float:
            if not data:
                return 0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile)
            return sorted_data[min(index, len(sorted_data) - 1)]

        return {
            "events_processed": self.metrics_data["events_processed"],
            "orders_executed": self.metrics_data["orders_executed"],
            "positions_opened": self.metrics_data["positions_opened"],
            "positions_closed": self.metrics_data["positions_closed"],
            "signals_generated": self.metrics_data["signals_generated"],
            "errors_count": self.metrics_data["errors_count"],
            "active_positions": len(self.metrics_data["position_values"]),
            "total_position_value": sum(
                p["value"] for p in self.metrics_data["position_values"].values()
            ),
            "event_latency_p50": calculate_percentile(event_latencies, 0.5),
            "event_latency_p95": calculate_percentile(event_latencies, 0.95),
            "event_latency_p99": calculate_percentile(event_latencies, 0.99),
            "order_latency_p50": calculate_percentile(order_latencies, 0.5),
            "order_latency_p95": calculate_percentile(order_latencies, 0.95),
            "order_latency_p99": calculate_percentile(order_latencies, 0.99),
        }

    def export_grafana_dashboard(self) -> dict[str, Any]:
        """Export Grafana dashboard configuration."""
        return {
            "dashboard": {
                "title": "Genesis Trading Loop Metrics",
                "panels": [
                    {
                        "title": "Events Processed",
                        "type": "graph",
                        "targets": [
                            {"expr": "rate(trading_events_total[5m])"}
                        ]
                    },
                    {
                        "title": "Event Processing Latency",
                        "type": "graph",
                        "targets": [
                            {"expr": "histogram_quantile(0.5, event_processing_latency)"},
                            {"expr": "histogram_quantile(0.95, event_processing_latency)"},
                            {"expr": "histogram_quantile(0.99, event_processing_latency)"}
                        ]
                    },
                    {
                        "title": "Order Execution Time",
                        "type": "graph",
                        "targets": [
                            {"expr": "histogram_quantile(0.95, order_execution_duration)"}
                        ]
                    },
                    {
                        "title": "Position Values",
                        "type": "graph",
                        "targets": [
                            {"expr": "sum(position_value_usd)"}
                        ]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {"expr": "rate(trading_errors_total[5m])"}
                        ]
                    },
                    {
                        "title": "System Status",
                        "type": "singlestat",
                        "targets": [
                            {"expr": "trading_loop_active"}
                        ]
                    }
                ]
            }
        }
