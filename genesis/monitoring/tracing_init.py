"""Initialize and configure distributed tracing for Genesis services."""

import os

import structlog
from fastapi import FastAPI, Request

from genesis.monitoring.opentelemetry_tracing import (
    OpenTelemetryTracer,
    get_opentelemetry_tracer,
    instrument_exchange_api_call,
    instrument_market_data_processing,
    instrument_order_execution,
    instrument_risk_check,
    instrument_tilt_detection,
    instrument_websocket_connection,
)

logger = structlog.get_logger(__name__)


def initialize_tracing(
    app: FastAPI | None = None,
    service_name: str = "genesis-trading",
) -> OpenTelemetryTracer:
    """Initialize OpenTelemetry tracing for the Genesis application.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service for tracing
    
    Returns:
        Configured OpenTelemetry tracer instance
    """
    # Get configuration from environment
    otlp_endpoint = os.getenv("OTLP_ENDPOINT", "localhost:4317")
    otlp_secure = os.getenv("OTLP_SECURE", "false").lower() == "true"
    trace_sampling_rate = float(os.getenv("TRACE_SAMPLING_RATE", "0.01"))
    production_mode = os.getenv("ENVIRONMENT", "development") == "production"
    export_to_console = os.getenv("TRACE_EXPORT_CONSOLE", "false").lower() == "true"

    # Initialize tracer
    tracer = get_opentelemetry_tracer(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint if otlp_endpoint != "disabled" else None,
        sampling_rate=trace_sampling_rate,
        export_to_console=export_to_console,
        production_mode=production_mode,
    )

    # Instrument libraries with FastAPI app if provided
    tracer.instrument_libraries(app=app)

    logger.info(
        "Distributed tracing initialized",
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        sampling_rate=trace_sampling_rate,
        production_mode=production_mode,
    )

    return tracer


def add_tracing_middleware(app: FastAPI, tracer: OpenTelemetryTracer) -> None:
    """Add tracing middleware to FastAPI application.
    
    Args:
        app: FastAPI application instance
        tracer: OpenTelemetry tracer instance
    """

    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next):
        """Add correlation ID to each request for distributed tracing."""
        # Extract or generate correlation ID
        corr_id = request.headers.get("X-Correlation-ID")
        if not corr_id:
            import uuid
            corr_id = str(uuid.uuid4())

        # Set correlation ID in context
        tracer.set_correlation_id(corr_id)

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = corr_id

        return response

    @app.middleware("http")
    async def trace_request(request: Request, call_next):
        """Create span for each HTTP request."""
        # Skip tracing for health and metrics endpoints
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Create span for request
        with tracer.create_span(
            f"http_{request.method}_{request.url.path}",
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname,
                "http.target": request.url.path,
                "http.user_agent": request.headers.get("user-agent", ""),
                "client.address": request.client.host if request.client else None,
            }
        ) as span:
            try:
                # Process request
                response = await call_next(request)

                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)

                # Mark error if status code indicates failure
                if response.status_code >= 400:
                    span.set_attribute("error", True)
                    span.set_attribute("http.status_text", response.reason_phrase)

                return response

            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise

    logger.info("Tracing middleware added to FastAPI application")


def create_trading_span_decorators(tracer: OpenTelemetryTracer) -> dict:
    """Create span decorators for trading operations.
    
    Args:
        tracer: OpenTelemetry tracer instance
    
    Returns:
        Dictionary of decorator functions
    """
    return {
        "order_execution": instrument_order_execution(tracer),
        "risk_check": instrument_risk_check(tracer),
        "tilt_detection": instrument_tilt_detection(tracer),
        "websocket_connection": instrument_websocket_connection(tracer),
        "market_data_processing": instrument_market_data_processing(tracer),
        "exchange_api_call": instrument_exchange_api_call(tracer),
        "track_performance": tracer.track_performance,
    }


def configure_span_processors(tracer: OpenTelemetryTracer) -> None:
    """Configure additional span processors for enhanced tracing.
    
    Args:
        tracer: OpenTelemetry tracer instance
    """
    # This is already handled in the OpenTelemetryTracer initialization
    # But we can add custom processors here if needed
    pass


def setup_trace_context_propagation(tracer: OpenTelemetryTracer) -> None:
    """Setup trace context propagation for cross-service communication.
    
    Args:
        tracer: OpenTelemetry tracer instance
    """
    # Context propagation is already set up in OpenTelemetryTracer
    # This function can be extended for custom propagation needs
    pass


# Export convenience function for quick setup
def setup_genesis_tracing(
    app: FastAPI | None = None,
    service_name: str = "genesis-trading",
) -> tuple[OpenTelemetryTracer, dict]:
    """Complete tracing setup for Genesis application.
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service
    
    Returns:
        Tuple of (tracer instance, decorator dictionary)
    """
    # Initialize tracing
    tracer = initialize_tracing(app, service_name)

    # Add middleware if FastAPI app provided
    if app:
        add_tracing_middleware(app, tracer)

    # Configure span processors
    configure_span_processors(tracer)

    # Setup context propagation
    setup_trace_context_propagation(tracer)

    # Create decorators
    decorators = create_trading_span_decorators(tracer)

    logger.info(
        "Genesis tracing setup complete",
        service_name=service_name,
        decorators_available=list(decorators.keys()),
    )

    return tracer, decorators
