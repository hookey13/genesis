"""OpenTelemetry distributed tracing implementation for Project GENESIS."""

import asyncio
import functools
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from opentelemetry import baggage, context, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
try:
    from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
    AIOHTTP_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    AIOHTTP_INSTRUMENTATION_AVAILABLE = False

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = structlog.get_logger(__name__)

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class OpenTelemetryTracer:
    """OpenTelemetry tracer for distributed tracing in GENESIS."""
    
    def __init__(
        self,
        service_name: str = "genesis-trading",
        otlp_endpoint: str | None = None,
        sampling_rate: float = 1.0,
        export_to_console: bool = False,
        production_mode: bool = False
    ):
        """Initialize OpenTelemetry tracer with OTLP exporter.
        
        Args:
            service_name: Name of the service
            otlp_endpoint: OTLP collector endpoint (e.g., "localhost:4317")
            sampling_rate: Trace sampling rate (0.0 to 1.0)
            export_to_console: Whether to export spans to console (for debugging)
            production_mode: Whether running in production mode
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.sampling_rate = sampling_rate
        self.export_to_console = export_to_console
        self.production_mode = production_mode
        
        # Initialize resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production" if production_mode else "development",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
        })
        
        # Configure sampler based on mode
        if production_mode:
            # In production, use sampling to reduce overhead
            sampler = TraceIdRatioBased(sampling_rate)
        else:
            # In development, trace everything
            sampler = TraceIdRatioBased(1.0)
        
        # Create tracer provider
        self.tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler
        )
        
        # Add span processors
        if otlp_endpoint:
            # Add OTLP exporter for sending to collector
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=not production_mode  # Use TLS in production
            )
            # Use batch processor for efficiency
            batch_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                max_export_interval_millis=5000
            )
            self.tracer_provider.add_span_processor(batch_processor)
        
        if export_to_console and not production_mode:
            # Add console exporter for debugging
            console_exporter = ConsoleSpanExporter()
            simple_processor = SimpleSpanProcessor(console_exporter)
            self.tracer_provider.add_span_processor(simple_processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(self.tracer_provider)
        
        # Get tracer instance
        self.tracer = trace.get_tracer(
            __name__,
            "1.0.0",
            tracer_provider=self.tracer_provider
        )
        
        # Initialize propagator for context propagation
        self.propagator = TraceContextTextMapPropagator()
        
        logger.info(
            "OpenTelemetry tracer initialized",
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            sampling_rate=sampling_rate,
            production_mode=production_mode
        )
    
    def instrument_libraries(self) -> None:
        """Instrument common libraries for automatic tracing."""
        if AIOHTTP_INSTRUMENTATION_AVAILABLE:
            try:
                # Instrument aiohttp for HTTP client tracing
                AioHttpClientInstrumentor().instrument()
                logger.debug("Instrumented aiohttp client")
            except Exception as e:
                logger.warning("Failed to instrument aiohttp", error=str(e))
        else:
            logger.debug("aiohttp instrumentation not available")
        
        try:
            # Instrument SQLAlchemy for database tracing
            SQLAlchemyInstrumentor().instrument()
            logger.debug("Instrumented SQLAlchemy")
        except Exception as e:
            logger.warning("Failed to instrument SQLAlchemy", error=str(e))
    
    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None
    ) -> trace.Span:
        """Create a new span.
        
        Args:
            name: Span name
            kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)
            attributes: Span attributes
        
        Returns:
            Created span
        """
        span = self.tracer.start_span(
            name,
            kind=kind,
            attributes=attributes or {}
        )
        
        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            span.set_attribute("correlation_id", corr_id)
        
        return span
    
    def track_performance(
        self,
        operation_name: str | None = None,
        span_kind: SpanKind = SpanKind.INTERNAL,
        capture_args: bool = False,
        capture_result: bool = False
    ) -> Callable:
        """Decorator for automatic span creation and performance tracking.
        
        Args:
            operation_name: Name for the span (defaults to function name)
            span_kind: Kind of span
            capture_args: Whether to capture function arguments as span attributes
            capture_result: Whether to capture function result as span attribute
        
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            name = operation_name or func.__name__
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                # Create span with function context
                with self.tracer.start_as_current_span(
                    name,
                    kind=span_kind
                ) as span:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Add correlation ID if available
                    corr_id = correlation_id.get()
                    if corr_id:
                        span.set_attribute("correlation_id", corr_id)
                    
                    # Capture arguments if requested
                    if capture_args:
                        try:
                            span.set_attribute("function.args", str(args)[:1000])
                            span.set_attribute("function.kwargs", str(kwargs)[:1000])
                        except Exception:
                            pass
                    
                    # Track execution time
                    start_time = time.perf_counter()
                    
                    try:
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record success
                        span.set_status(Status(StatusCode.OK))
                        
                        # Capture result if requested
                        if capture_result:
                            try:
                                span.set_attribute("function.result", str(result)[:1000])
                            except Exception:
                                pass
                        
                        return result
                    
                    except Exception as e:
                        # Record error
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
                    
                    finally:
                        # Record execution time
                        execution_time = time.perf_counter() - start_time
                        span.set_attribute("execution_time_seconds", execution_time)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                # Create span with function context
                with self.tracer.start_as_current_span(
                    name,
                    kind=span_kind
                ) as span:
                    # Add function metadata
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Add correlation ID if available
                    corr_id = correlation_id.get()
                    if corr_id:
                        span.set_attribute("correlation_id", corr_id)
                    
                    # Capture arguments if requested
                    if capture_args:
                        try:
                            span.set_attribute("function.args", str(args)[:1000])
                            span.set_attribute("function.kwargs", str(kwargs)[:1000])
                        except Exception:
                            pass
                    
                    # Track execution time
                    start_time = time.perf_counter()
                    
                    try:
                        # Execute function
                        result = func(*args, **kwargs)
                        
                        # Record success
                        span.set_status(Status(StatusCode.OK))
                        
                        # Capture result if requested
                        if capture_result:
                            try:
                                span.set_attribute("function.result", str(result)[:1000])
                            except Exception:
                                pass
                        
                        return result
                    
                    except Exception as e:
                        # Record error
                        span.set_status(
                            Status(StatusCode.ERROR, str(e))
                        )
                        span.record_exception(e)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
                    
                    finally:
                        # Record execution time
                        execution_time = time.perf_counter() - start_time
                        span.set_attribute("execution_time_seconds", execution_time)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def inject_context(self, carrier: dict) -> None:
        """Inject trace context into carrier for propagation.
        
        Args:
            carrier: Dictionary to inject context into
        """
        inject(carrier)
    
    def extract_context(self, carrier: dict) -> context.Context:
        """Extract trace context from carrier.
        
        Args:
            carrier: Dictionary to extract context from
        
        Returns:
            Extracted context
        """
        return extract(carrier)
    
    def set_correlation_id(self, corr_id: str) -> None:
        """Set correlation ID for current context.
        
        Args:
            corr_id: Correlation ID to set
        """
        correlation_id.set(corr_id)
    
    def get_correlation_id(self) -> str | None:
        """Get current correlation ID.
        
        Returns:
            Current correlation ID or None
        """
        return correlation_id.get()
    
    def shutdown(self) -> None:
        """Shutdown tracer and flush all pending spans."""
        if hasattr(self, 'tracer_provider'):
            self.tracer_provider.shutdown()
            logger.info("OpenTelemetry tracer shutdown complete")


# Critical path instrumentation functions
def instrument_order_execution(tracer: OpenTelemetryTracer) -> Callable:
    """Instrument order execution with detailed span attributes."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(order, *args, **kwargs) -> Any:
            with tracer.create_span(
                f"order_execution_{order.order_type}",
                kind=SpanKind.CLIENT,
                attributes={
                    "order.id": order.id,
                    "order.type": order.order_type,
                    "order.symbol": order.symbol,
                    "order.side": order.side,
                    "order.quantity": str(order.quantity),
                    "order.price": str(order.price) if hasattr(order, 'price') else None,
                    "order.tier": order.tier if hasattr(order, 'tier') else None,
                }
            ) as span:
                try:
                    result = await func(order, *args, **kwargs)
                    span.set_attribute("order.status", "success")
                    span.set_attribute("order.execution_id", result.get('id', ''))
                    return result
                except Exception as e:
                    span.set_attribute("order.status", "failed")
                    span.set_attribute("order.error", str(e))
                    raise
        
        return wrapper
    
    return decorator


def instrument_risk_check(tracer: OpenTelemetryTracer) -> Callable:
    """Instrument risk check with detailed span attributes."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            check_type = kwargs.get('check_type', 'unknown')
            
            with tracer.create_span(
                f"risk_check_{check_type}",
                kind=SpanKind.INTERNAL,
                attributes={
                    "risk.check_type": check_type,
                    "risk.tier": kwargs.get('tier', 'unknown'),
                }
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("risk.passed", result)
                    
                    if not result:
                        span.add_event("risk_limit_breach", {
                            "breach_type": check_type,
                            "timestamp": time.time()
                        })
                    
                    return result
                except Exception as e:
                    span.set_attribute("risk.error", str(e))
                    raise
        
        return wrapper
    
    return decorator


def instrument_tilt_detection(tracer: OpenTelemetryTracer) -> Callable:
    """Instrument tilt detection with detailed span attributes."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            with tracer.create_span(
                "tilt_detection",
                kind=SpanKind.INTERNAL
            ) as span:
                try:
                    result = await func(*args, **kwargs)
                    
                    if isinstance(result, dict):
                        span.set_attribute("tilt.score", result.get('score', 0))
                        span.set_attribute("tilt.level", result.get('level', 'normal'))
                        
                        indicators = result.get('indicators', {})
                        for key, value in indicators.items():
                            span.set_attribute(f"tilt.indicator.{key}", value)
                        
                        if result.get('intervention_required'):
                            span.add_event("tilt_intervention_triggered", {
                                "intervention_type": result.get('intervention_type'),
                                "timestamp": time.time()
                            })
                    
                    return result
                except Exception as e:
                    span.set_attribute("tilt.error", str(e))
                    raise
        
        return wrapper
    
    return decorator


# Global tracer instance
_tracer_instance: OpenTelemetryTracer | None = None


def get_opentelemetry_tracer(
    service_name: str = "genesis-trading",
    otlp_endpoint: str | None = None,
    **kwargs
) -> OpenTelemetryTracer:
    """Get or create the global OpenTelemetry tracer instance.
    
    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP collector endpoint
        **kwargs: Additional arguments for tracer initialization
    
    Returns:
        OpenTelemetry tracer instance
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = OpenTelemetryTracer(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            **kwargs
        )
        # Instrument common libraries
        _tracer_instance.instrument_libraries()
    return _tracer_instance