"""Distributed tracing with OpenTelemetry for Project GENESIS."""

import asyncio
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Optional, TypeVar
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

# Context variable for trace propagation
current_trace: ContextVar[Optional['TraceContext']] = ContextVar('current_trace', default=None)

T = TypeVar('T')


class SpanStatus(Enum):
    """Span status codes."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """Represents a span in a distributed trace."""
    span_id: str
    trace_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def end(self, status: SpanStatus = SpanStatus.OK) -> None:
        """End the span and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        self.events.append(event)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events
        }


@dataclass
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    spans: list[Span] = field(default_factory=list)
    current_span: Span | None = None
    correlation_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    def start_span(self, operation_name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Start a new span."""
        span = Span(
            span_id=str(uuid4()),
            trace_id=self.trace_id,
            parent_span_id=self.current_span.span_id if self.current_span else None,
            operation_name=operation_name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        self.spans.append(span)
        return span

    def end_span(self, span: Span, status: SpanStatus = SpanStatus.OK) -> None:
        """End a span."""
        span.end(status)

        # Log span completion
        logger.debug("Span completed",
                    trace_id=self.trace_id,
                    span_id=span.span_id,
                    operation=span.operation_name,
                    duration_ms=span.duration_ms,
                    status=status.value)

    def set_baggage(self, key: str, value: str) -> None:
        """Set baggage item for propagation."""
        self.baggage[key] = value

    def get_baggage(self, key: str) -> str | None:
        """Get baggage item."""
        return self.baggage.get(key)

    def inject_headers(self) -> dict[str, str]:
        """Inject trace context into HTTP headers."""
        headers = {
            "X-Trace-Id": self.trace_id,
            "X-Correlation-Id": self.correlation_id or self.trace_id
        }

        if self.current_span:
            headers["X-Parent-Span-Id"] = self.current_span.span_id

        # Inject baggage
        for key, value in self.baggage.items():
            headers[f"X-Baggage-{key}"] = value

        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> 'TraceContext':
        """Extract trace context from HTTP headers."""
        trace_id = headers.get("X-Trace-Id", str(uuid4()))
        correlation_id = headers.get("X-Correlation-Id")

        # Extract baggage
        baggage = {}
        for key, value in headers.items():
            if key.startswith("X-Baggage-"):
                baggage_key = key[len("X-Baggage-"):]
                baggage[baggage_key] = value

        return cls(
            trace_id=trace_id,
            correlation_id=correlation_id,
            baggage=baggage
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "spans": [span.to_dict() for span in self.spans],
            "baggage": self.baggage
        }


class TracingManager:
    """Manages distributed tracing for the application."""

    def __init__(self, service_name: str = "genesis", sampling_rate: float = 0.1):
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.traces: dict[str, TraceContext] = {}
        self._exporters: list[Callable] = []
        self._export_interval = 10  # seconds
        self._export_task: asyncio.Task | None = None

    def should_sample(self, is_error: bool = False) -> bool:
        """Determine if a trace should be sampled."""
        # Always sample errors
        if is_error:
            return True

        # Sample based on rate
        import random
        return random.random() < self.sampling_rate

    def start_trace(self, operation_name: str, correlation_id: str | None = None) -> TraceContext | None:
        """Start a new trace."""
        # Check sampling decision
        if not self.should_sample():
            return None

        trace = TraceContext(correlation_id=correlation_id)
        trace.set_baggage("service", self.service_name)

        # Store trace
        self.traces[trace.trace_id] = trace

        # Set as current trace
        current_trace.set(trace)

        # Start root span
        span = trace.start_span(operation_name)
        trace.current_span = span

        logger.debug("Started trace",
                    trace_id=trace.trace_id,
                    operation=operation_name)

        return trace

    def get_current_trace(self) -> TraceContext | None:
        """Get the current trace context."""
        return current_trace.get()

    def add_exporter(self, exporter: Callable) -> None:
        """Add a trace exporter."""
        self._exporters.append(exporter)

    async def start_export(self) -> None:
        """Start trace export loop."""
        if not self._export_task:
            self._export_task = asyncio.create_task(self._export_loop())
            logger.info("Started trace export")

    async def stop_export(self) -> None:
        """Stop trace export loop."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
            self._export_task = None
            logger.info("Stopped trace export")

    async def _export_loop(self) -> None:
        """Export traces periodically."""
        while True:
            try:
                await self._export_traces()
                await asyncio.sleep(self._export_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error exporting traces", error=str(e))
                await asyncio.sleep(self._export_interval)

    async def _export_traces(self) -> None:
        """Export completed traces."""
        # Find completed traces
        completed = []
        for trace_id, trace in list(self.traces.items()):
            if all(span.end_time is not None for span in trace.spans):
                completed.append(trace_id)

        # Export and remove completed traces
        for trace_id in completed:
            trace = self.traces.pop(trace_id)

            # Call exporters
            for exporter in self._exporters:
                try:
                    if asyncio.iscoroutinefunction(exporter):
                        await exporter(trace)
                    else:
                        exporter(trace)
                except Exception as e:
                    logger.error("Error in trace exporter",
                               trace_id=trace_id,
                               error=str(e))

    def export_to_jaeger(self, trace: TraceContext) -> None:
        """Export trace to Jaeger (placeholder for actual implementation)."""
        # In production, this would send to Jaeger collector
        logger.info("Exporting trace to Jaeger",
                   trace_id=trace.trace_id,
                   span_count=len(trace.spans))


# Global tracing manager instance
tracing_manager = TracingManager()


def trace_operation(operation_name: str = None):
    """Decorator for tracing function execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            trace = tracing_manager.get_current_trace()
            if not trace:
                # Start new trace if none exists
                trace = tracing_manager.start_trace(op_name)
                if not trace:
                    # Not sampled, execute without tracing
                    return await func(*args, **kwargs)

                span = trace.current_span
                try:
                    # Add function arguments as attributes
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("module", func.__module__)

                    result = await func(*args, **kwargs)
                    trace.end_span(span, SpanStatus.OK)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.add_event("exception", {"message": str(e)})
                    trace.end_span(span, SpanStatus.ERROR)
                    raise
                finally:
                    current_trace.set(None)
            else:
                # Create child span
                span = trace.start_span(op_name)
                prev_span = trace.current_span
                trace.current_span = span

                try:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("module", func.__module__)

                    result = await func(*args, **kwargs)
                    trace.end_span(span, SpanStatus.OK)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.add_event("exception", {"message": str(e)})
                    trace.end_span(span, SpanStatus.ERROR)
                    raise
                finally:
                    trace.current_span = prev_span

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            trace = tracing_manager.get_current_trace()
            if not trace:
                # Start new trace if none exists
                trace = tracing_manager.start_trace(op_name)
                if not trace:
                    # Not sampled, execute without tracing
                    return func(*args, **kwargs)

                span = trace.current_span
                try:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("module", func.__module__)

                    result = func(*args, **kwargs)
                    trace.end_span(span, SpanStatus.OK)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.add_event("exception", {"message": str(e)})
                    trace.end_span(span, SpanStatus.ERROR)
                    raise
                finally:
                    current_trace.set(None)
            else:
                # Create child span
                span = trace.start_span(op_name)
                prev_span = trace.current_span
                trace.current_span = span

                try:
                    span.set_attribute("function", func.__name__)
                    span.set_attribute("module", func.__module__)

                    result = func(*args, **kwargs)
                    trace.end_span(span, SpanStatus.OK)
                    return result
                except Exception as e:
                    span.set_attribute("error", str(e))
                    span.add_event("exception", {"message": str(e)})
                    trace.end_span(span, SpanStatus.ERROR)
                    raise
                finally:
                    trace.current_span = prev_span

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
