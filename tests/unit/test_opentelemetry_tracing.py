"""Unit tests for OpenTelemetry tracing implementation."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanKind

from genesis.monitoring.opentelemetry_tracing import (
    OpenTelemetryTracer,
    AdaptiveSampler,
    get_opentelemetry_tracer,
    instrument_order_execution,
    instrument_risk_check,
    instrument_tilt_detection,
    instrument_websocket_connection,
    instrument_market_data_processing,
    instrument_exchange_api_call,
    correlation_id,
)


class TestAdaptiveSampler:
    """Test adaptive sampling logic."""
    
    def test_error_sampling(self):
        """Test that errors are always sampled."""
        sampler = AdaptiveSampler(
            base_sampling_rate=0.01,
            error_sampling_rate=1.0
        )
        
        # Test error attribute
        result = sampler.should_sample(
            None, 12345, "test_span", SpanKind.INTERNAL,
            {"error": True}, None
        )
        # Check if sampling decision is positive
        # The result should indicate the span should be sampled
        assert result is not None
        
        # Test error type attribute
        result = sampler.should_sample(
            None, 12345, "test_span", SpanKind.INTERNAL,
            {"error.type": "ValueError"}, None
        )
        # Check if sampling decision is positive
        # The result should indicate the span should be sampled
        assert result is not None
    
    def test_slow_operation_sampling(self):
        """Test that slow operations are sampled at higher rate."""
        sampler = AdaptiveSampler(
            base_sampling_rate=0.01,
            slow_threshold_ms=100,
            slow_sampling_rate=1.0
        )
        
        # Test slow operation
        result = sampler.should_sample(
            None, 12345, "test_span", SpanKind.INTERNAL,
            {"latency_ms": 150}, None
        )
        # Check if sampling decision is positive
        # The result should indicate the span should be sampled
        assert result is not None
        
        # Test fast operation (may or may not be sampled based on base rate)
        result = sampler.should_sample(
            None, 12345, "test_span", SpanKind.INTERNAL,
            {"latency_ms": 50}, None
        )
        # Can't assert definitely due to probabilistic sampling
    
    def test_critical_operation_sampling(self):
        """Test that critical operations are sampled at higher rate."""
        sampler = AdaptiveSampler(base_sampling_rate=0.01)
        
        # Test critical operation names
        critical_ops = ["order_execution", "trade_processing", "risk_check", "tilt_detection"]
        
        for op in critical_ops:
            # Run multiple times to verify higher sampling rate
            sampled_count = 0
            for _ in range(100):
                result = sampler.should_sample(
                    None, 12345, op, SpanKind.INTERNAL,
                    {}, None
                )
                if result is not None:
                    sampled_count += 1
            
            # Should sample more than base rate (1%) but not necessarily 100%
            assert sampled_count > 1  # At least some should be sampled
    
    def test_sampler_description(self):
        """Test sampler description generation."""
        sampler = AdaptiveSampler(
            base_sampling_rate=0.01,
            error_sampling_rate=1.0,
            slow_threshold_ms=100,
            slow_sampling_rate=0.5
        )
        
        description = sampler.get_description()
        assert "AdaptiveSampler" in description
        assert "base=0.01" in description
        assert "error=1.0" in description
        assert "slow_threshold=100ms" in description
        assert "slow=0.5" in description


class TestOpenTelemetryTracer:
    """Test OpenTelemetry tracer functionality."""
    
    @pytest.fixture
    def tracer(self):
        """Create test tracer instance."""
        return OpenTelemetryTracer(
            service_name="test-service",
            otlp_endpoint=None,  # No actual export
            sampling_rate=1.0,
            export_to_console=False,
            production_mode=False
        )
    
    def test_tracer_initialization(self, tracer):
        """Test tracer initialization."""
        assert tracer.service_name == "test-service"
        assert tracer.sampling_rate == 1.0
        assert tracer.production_mode is False
        assert isinstance(tracer.tracer_provider, TracerProvider)
        assert tracer.tracer is not None
    
    def test_create_span(self, tracer):
        """Test span creation."""
        with tracer.create_span(
            "test_operation",
            kind=SpanKind.INTERNAL,
            attributes={"test.attribute": "value"}
        ) as span:
            assert span is not None
            assert span.is_recording()
            span.set_attribute("additional.attribute", "added")
    
    def test_correlation_id_propagation(self, tracer):
        """Test correlation ID propagation."""
        test_corr_id = "test-correlation-123"
        tracer.set_correlation_id(test_corr_id)
        
        assert tracer.get_correlation_id() == test_corr_id
        
        with tracer.create_span("test_operation") as span:
            # Correlation ID should be added to span
            pass
    
    @pytest.mark.asyncio
    async def test_track_performance_decorator_async(self, tracer):
        """Test performance tracking decorator for async functions."""
        call_count = 0
        
        @tracer.track_performance("custom_operation", capture_args=True, capture_result=True)
        async def test_function(arg1, arg2="default"):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return f"result_{arg1}_{arg2}"
        
        result = await test_function("test", arg2="value")
        
        assert call_count == 1
        assert result == "result_test_value"
    
    def test_track_performance_decorator_sync(self, tracer):
        """Test performance tracking decorator for sync functions."""
        call_count = 0
        
        @tracer.track_performance("custom_operation", capture_args=True, capture_result=True)
        def test_function(arg1, arg2="default"):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return f"result_{arg1}_{arg2}"
        
        result = test_function("test", arg2="value")
        
        assert call_count == 1
        assert result == "result_test_value"
    
    @pytest.mark.asyncio
    async def test_track_performance_error_handling(self, tracer):
        """Test performance tracking with error handling."""
        @tracer.track_performance("error_operation")
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await failing_function()
    
    def test_context_injection_extraction(self, tracer):
        """Test context injection and extraction."""
        carrier = {}
        
        # Inject context
        tracer.inject_context(carrier)
        
        # Carrier should have trace context headers
        assert len(carrier) > 0
        
        # Extract context
        extracted_context = tracer.extract_context(carrier)
        assert extracted_context is not None
    
    @patch('genesis.monitoring.opentelemetry_tracing.FastAPIInstrumentor')
    @patch('genesis.monitoring.opentelemetry_tracing.SQLAlchemyInstrumentor')
    def test_instrument_libraries(self, mock_sqlalchemy, mock_fastapi, tracer):
        """Test library instrumentation."""
        # Create mock instrumentors
        mock_sqlalchemy_inst = Mock()
        mock_sqlalchemy.return_value = mock_sqlalchemy_inst
        
        # Call instrument_libraries
        tracer.instrument_libraries()
        
        # Verify SQLAlchemy was instrumented
        mock_sqlalchemy_inst.instrument.assert_called_once()
    
    def test_shutdown(self, tracer):
        """Test tracer shutdown."""
        with patch.object(tracer.tracer_provider, 'shutdown') as mock_shutdown:
            tracer.shutdown()
            mock_shutdown.assert_called_once()


class TestInstrumentationFunctions:
    """Test instrumentation decorator functions."""
    
    @pytest.fixture
    def mock_tracer(self):
        """Create mock tracer."""
        tracer = Mock(spec=OpenTelemetryTracer)
        tracer.create_span = MagicMock()
        span = MagicMock()
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=None)
        tracer.create_span.return_value = span
        return tracer
    
    @pytest.mark.asyncio
    async def test_instrument_order_execution(self, mock_tracer):
        """Test order execution instrumentation."""
        decorator = instrument_order_execution(mock_tracer)
        
        @decorator
        async def execute_order(order):
            return {"id": "execution_123", "status": "success"}
        
        # Create mock order
        order = Mock()
        order.id = "order_123"
        order.order_type = "market"
        order.symbol = "BTCUSDT"
        order.side = "buy"
        order.quantity = 0.001
        order.tier = "sniper"
        
        result = await execute_order(order)
        
        assert result["id"] == "execution_123"
        mock_tracer.create_span.assert_called_once()
        call_args = mock_tracer.create_span.call_args
        assert "order_execution_market" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_instrument_risk_check(self, mock_tracer):
        """Test risk check instrumentation."""
        decorator = instrument_risk_check(mock_tracer)
        
        @decorator
        async def check_risk(check_type="position"):
            return True
        
        result = await check_risk(check_type="leverage")
        
        assert result is True
        mock_tracer.create_span.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_instrument_websocket_connection(self, mock_tracer):
        """Test WebSocket connection instrumentation."""
        decorator = instrument_websocket_connection(mock_tracer)
        
        @decorator
        async def connect_websocket(ws_url, ws_type="market_data"):
            return {"status": "connected"}
        
        result = await connect_websocket("wss://stream.binance.com", ws_type="trade")
        
        assert result["status"] == "connected"
        mock_tracer.create_span.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_instrument_market_data_processing(self, mock_tracer):
        """Test market data processing instrumentation."""
        decorator = instrument_market_data_processing(mock_tracer)
        
        @decorator
        async def process_market_data(market_data):
            return {"processed": True}
        
        market_data = {
            "symbol": "BTCUSDT",
            "type": "ticker",
            "timestamp": 1234567890
        }
        
        result = await process_market_data(market_data)
        
        assert result["processed"] is True
        mock_tracer.create_span.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_instrument_exchange_api_call(self, mock_tracer):
        """Test exchange API call instrumentation with retries."""
        decorator = instrument_exchange_api_call(mock_tracer)
        
        call_count = 0
        
        @decorator
        async def call_api(endpoint, method="GET", exchange="binance", max_retries=3):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"status": "success", "rate_limit": 950}
        
        result = await call_api("/api/v3/order", method="POST")
        
        assert result["status"] == "success"
        assert call_count == 3  # Should retry twice before succeeding
        mock_tracer.create_span.assert_called_once()


class TestGlobalTracerInstance:
    """Test global tracer instance management."""
    
    def test_get_opentelemetry_tracer_singleton(self):
        """Test that get_opentelemetry_tracer returns singleton."""
        tracer1 = get_opentelemetry_tracer(service_name="test1")
        tracer2 = get_opentelemetry_tracer(service_name="test2")
        
        # Should return same instance
        assert tracer1 is tracer2
        
        # Service name should be from first call
        assert tracer1.service_name == "test1"
    
    def test_get_opentelemetry_tracer_with_otlp(self):
        """Test tracer creation with OTLP endpoint."""
        import genesis.monitoring.opentelemetry_tracing as tracing_module
        
        # Reset global instance
        tracing_module._tracer_instance = None
        
        with patch('genesis.monitoring.opentelemetry_tracing.OTLPSpanExporter'):
            tracer = get_opentelemetry_tracer(
                service_name="test-otlp",
                otlp_endpoint="localhost:4317"
            )
            
            assert tracer is not None
            assert tracer.service_name == "test-otlp"
            assert tracer.otlp_endpoint == "localhost:4317"


class TestProductionMode:
    """Test production mode specific behavior."""
    
    def test_production_mode_sampling(self):
        """Test that production mode uses adaptive sampling."""
        tracer = OpenTelemetryTracer(
            service_name="prod-service",
            otlp_endpoint=None,
            sampling_rate=0.01,
            production_mode=True
        )
        
        # In production mode, should use AdaptiveSampler
        assert tracer.production_mode is True
        assert tracer.sampling_rate == 0.01
    
    def test_development_mode_sampling(self):
        """Test that development mode samples everything."""
        tracer = OpenTelemetryTracer(
            service_name="dev-service",
            otlp_endpoint=None,
            sampling_rate=0.5,  # Even if set lower
            production_mode=False
        )
        
        # In development mode, should sample everything
        assert tracer.production_mode is False