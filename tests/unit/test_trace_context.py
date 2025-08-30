"""Unit tests for Trace Context (distributed tracing)."""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import UUID, uuid4

from genesis.monitoring.trace_context import (
    TraceContext,
    Span,
    SpanKind,
    SpanStatus,
    TraceManager,
    OpenTelemetryExporter
)


class TestSpan:
    """Test Span class."""
    
    def test_span_creation(self):
        """Test creating a span."""
        trace_id = uuid4()
        span_id = uuid4()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name="test_span",
            kind=SpanKind.SERVER
        )
        
        assert span.trace_id == trace_id
        assert span.span_id == span_id
        assert span.name == "test_span"
        assert span.kind == SpanKind.SERVER
        assert span.status == SpanStatus.UNSET
        assert span.parent_id is None
    
    def test_span_with_parent(self):
        """Test creating a span with parent."""
        trace_id = uuid4()
        span_id = uuid4()
        parent_id = uuid4()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name="child_span",
            kind=SpanKind.INTERNAL,
            parent_id=parent_id
        )
        
        assert span.parent_id == parent_id
    
    def test_span_set_status(self):
        """Test setting span status."""
        span = Span(
            trace_id=uuid4(),
            span_id=uuid4(),
            name="test_span",
            kind=SpanKind.CLIENT
        )
        
        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK
        
        span.set_status(SpanStatus.ERROR, "Something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "Something went wrong"
    
    def test_span_add_attributes(self):
        """Test adding attributes to span."""
        span = Span(
            trace_id=uuid4(),
            span_id=uuid4(),
            name="test_span",
            kind=SpanKind.SERVER
        )
        
        span.add_attribute("http.method", "GET")
        span.add_attribute("http.status_code", 200)
        span.add_attribute("user.id", "user123")
        
        assert span.attributes["http.method"] == "GET"
        assert span.attributes["http.status_code"] == 200
        assert span.attributes["user.id"] == "user123"
    
    def test_span_add_event(self):
        """Test adding events to span."""
        span = Span(
            trace_id=uuid4(),
            span_id=uuid4(),
            name="test_span",
            kind=SpanKind.SERVER
        )
        
        span.add_event("request_received", {"size": 1024})
        span.add_event("processing_started")
        span.add_event("response_sent", {"size": 2048})
        
        assert len(span.events) == 3
        assert span.events[0]["name"] == "request_received"
        assert span.events[0]["attributes"]["size"] == 1024
        assert span.events[1]["name"] == "processing_started"
        assert span.events[2]["attributes"]["size"] == 2048
    
    def test_span_end(self):
        """Test ending a span."""
        span = Span(
            trace_id=uuid4(),
            span_id=uuid4(),
            name="test_span",
            kind=SpanKind.SERVER
        )
        
        assert span.end_time is None
        assert span.duration_ms is None
        
        span.end()
        
        assert span.end_time is not None
        assert span.duration_ms > 0
    
    def test_span_to_dict(self):
        """Test converting span to dictionary."""
        trace_id = uuid4()
        span_id = uuid4()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            name="test_span",
            kind=SpanKind.CLIENT
        )
        
        span.add_attribute("service.name", "trading")
        span.set_status(SpanStatus.OK)
        span.end()
        
        data = span.to_dict()
        
        assert data["trace_id"] == str(trace_id)
        assert data["span_id"] == str(span_id)
        assert data["name"] == "test_span"
        assert data["kind"] == "CLIENT"
        assert data["status"] == "OK"
        assert data["attributes"]["service.name"] == "trading"
        assert "start_time" in data
        assert "end_time" in data
        assert "duration_ms" in data


class TestTraceContext:
    """Test TraceContext class."""
    
    def test_trace_context_creation(self):
        """Test creating a trace context."""
        context = TraceContext()
        
        assert context.trace_id is not None
        assert isinstance(context.trace_id, UUID)
        assert context.span_id is None
        assert context.baggage == {}
    
    def test_trace_context_with_ids(self):
        """Test creating trace context with specific IDs."""
        trace_id = uuid4()
        span_id = uuid4()
        
        context = TraceContext(trace_id=trace_id, span_id=span_id)
        
        assert context.trace_id == trace_id
        assert context.span_id == span_id
    
    def test_trace_context_add_baggage(self):
        """Test adding baggage to trace context."""
        context = TraceContext()
        
        context.add_baggage("user_id", "12345")
        context.add_baggage("session_id", "abc-def")
        context.add_baggage("tier", "sniper")
        
        assert context.baggage["user_id"] == "12345"
        assert context.baggage["session_id"] == "abc-def"
        assert context.baggage["tier"] == "sniper"
    
    def test_trace_context_to_headers(self):
        """Test converting trace context to HTTP headers."""
        trace_id = uuid4()
        span_id = uuid4()
        
        context = TraceContext(trace_id=trace_id, span_id=span_id)
        context.add_baggage("user_id", "12345")
        
        headers = context.to_headers()
        
        assert headers["X-Trace-Id"] == str(trace_id)
        assert headers["X-Span-Id"] == str(span_id)
        assert "X-Baggage" in headers
        
        baggage = json.loads(headers["X-Baggage"])
        assert baggage["user_id"] == "12345"
    
    def test_trace_context_from_headers(self):
        """Test creating trace context from HTTP headers."""
        trace_id = uuid4()
        span_id = uuid4()
        
        headers = {
            "X-Trace-Id": str(trace_id),
            "X-Span-Id": str(span_id),
            "X-Baggage": json.dumps({"user_id": "12345", "tier": "hunter"})
        }
        
        context = TraceContext.from_headers(headers)
        
        assert context.trace_id == trace_id
        assert context.span_id == span_id
        assert context.baggage["user_id"] == "12345"
        assert context.baggage["tier"] == "hunter"


class TestTraceManager:
    """Test TraceManager class."""
    
    def test_trace_manager_initialization(self):
        """Test trace manager initialization."""
        manager = TraceManager()
        
        assert manager.current_trace is None
        assert len(manager.active_spans) == 0
        assert len(manager.completed_spans) == 0
    
    def test_start_trace(self):
        """Test starting a new trace."""
        manager = TraceManager()
        
        context = manager.start_trace("order_execution")
        
        assert manager.current_trace is not None
        assert manager.current_trace.trace_id == context.trace_id
        assert len(manager.active_spans) == 1
        
        # Check root span
        root_span = list(manager.active_spans.values())[0]
        assert root_span.name == "order_execution"
        assert root_span.parent_id is None
    
    def test_start_span(self):
        """Test starting a child span."""
        manager = TraceManager()
        
        # Start trace
        context = manager.start_trace("order_execution")
        
        # Start child span
        span = manager.start_span("risk_check", SpanKind.INTERNAL)
        
        assert span.trace_id == context.trace_id
        assert span.parent_id is not None
        assert len(manager.active_spans) == 2
    
    def test_end_span(self):
        """Test ending a span."""
        manager = TraceManager()
        
        # Start trace and span
        manager.start_trace("order_execution")
        span = manager.start_span("risk_check", SpanKind.INTERNAL)
        span_id = span.span_id
        
        assert span_id in manager.active_spans
        
        # End span
        manager.end_span(span_id)
        
        assert span_id not in manager.active_spans
        assert any(s.span_id == span_id for s in manager.completed_spans)
    
    def test_add_span_attribute(self):
        """Test adding attribute to active span."""
        manager = TraceManager()
        
        manager.start_trace("order_execution")
        span = manager.start_span("risk_check", SpanKind.INTERNAL)
        
        manager.add_span_attribute(span.span_id, "risk.score", 0.75)
        manager.add_span_attribute(span.span_id, "risk.passed", True)
        
        assert span.attributes["risk.score"] == 0.75
        assert span.attributes["risk.passed"] is True
    
    def test_add_span_event(self):
        """Test adding event to active span."""
        manager = TraceManager()
        
        manager.start_trace("order_execution")
        span = manager.start_span("risk_check", SpanKind.INTERNAL)
        
        manager.add_span_event(span.span_id, "position_limit_check", {"limit": 10000})
        manager.add_span_event(span.span_id, "drawdown_check", {"current": 0.02})
        
        assert len(span.events) == 2
        assert span.events[0]["name"] == "position_limit_check"
        assert span.events[1]["attributes"]["current"] == 0.02
    
    def test_set_span_status(self):
        """Test setting span status."""
        manager = TraceManager()
        
        manager.start_trace("order_execution")
        span = manager.start_span("risk_check", SpanKind.INTERNAL)
        
        manager.set_span_status(span.span_id, SpanStatus.ERROR, "Risk limit exceeded")
        
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "Risk limit exceeded"
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using trace manager as context manager."""
        manager = TraceManager()
        
        async with manager.trace("order_execution") as span:
            span.add_attribute("order.id", "12345")
            span.add_attribute("order.symbol", "BTC/USDT")
            
            # Simulate some work
            await asyncio.sleep(0.01)
            
            span.set_status(SpanStatus.OK)
        
        # Span should be completed
        assert len(manager.completed_spans) == 1
        completed = manager.completed_spans[0]
        assert completed.name == "order_execution"
        assert completed.attributes["order.id"] == "12345"
        assert completed.end_time is not None
    
    def test_get_trace_summary(self):
        """Test getting trace summary."""
        manager = TraceManager()
        
        # Create a trace with multiple spans
        context = manager.start_trace("order_execution")
        
        risk_span = manager.start_span("risk_check", SpanKind.INTERNAL)
        risk_span.add_attribute("risk.score", 0.5)
        manager.end_span(risk_span.span_id)
        
        exec_span = manager.start_span("exchange_execution", SpanKind.CLIENT)
        exec_span.add_attribute("exchange", "binance")
        manager.end_span(exec_span.span_id)
        
        summary = manager.get_trace_summary()
        
        assert summary["trace_id"] == str(context.trace_id)
        assert summary["span_count"] == 3  # root + 2 children
        assert summary["active_spans"] == 1  # only root still active
        assert summary["completed_spans"] == 2
    
    def test_export_spans(self):
        """Test exporting completed spans."""
        manager = TraceManager()
        
        # Create and complete some spans
        manager.start_trace("order_execution")
        
        for i in range(3):
            span = manager.start_span(f"step_{i}", SpanKind.INTERNAL)
            span.add_attribute("step", i)
            manager.end_span(span.span_id)
        
        exported = manager.export_spans(limit=2)
        
        assert len(exported) == 2
        assert all(isinstance(s, dict) for s in exported)
    
    def test_clear_completed_spans(self):
        """Test clearing completed spans."""
        manager = TraceManager()
        
        # Create and complete some spans
        manager.start_trace("order_execution")
        
        for i in range(5):
            span = manager.start_span(f"step_{i}", SpanKind.INTERNAL)
            manager.end_span(span.span_id)
        
        assert len(manager.completed_spans) == 5
        
        manager.clear_completed_spans()
        
        assert len(manager.completed_spans) == 0
    
    def test_span_limit(self):
        """Test span storage limit."""
        manager = TraceManager(max_completed_spans=10)
        
        manager.start_trace("test")
        
        # Create more spans than the limit
        for i in range(15):
            span = manager.start_span(f"span_{i}", SpanKind.INTERNAL)
            manager.end_span(span.span_id)
        
        # Should only keep the most recent 10
        assert len(manager.completed_spans) == 10
        
        # Check that we have the most recent spans
        span_names = [s.name for s in manager.completed_spans]
        assert "span_14" in span_names
        assert "span_5" in span_names
        assert "span_4" not in span_names


class TestOpenTelemetryExporter:
    """Test OpenTelemetry exporter."""
    
    @pytest.mark.asyncio
    async def test_exporter_initialization(self):
        """Test exporter initialization."""
        exporter = OpenTelemetryExporter(
            endpoint="http://localhost:4318",
            service_name="genesis-trading"
        )
        
        assert exporter.endpoint == "http://localhost:4318"
        assert exporter.service_name == "genesis-trading"
        assert exporter.export_interval == 10
    
    @pytest.mark.asyncio
    async def test_export_spans_format(self):
        """Test span export format."""
        exporter = OpenTelemetryExporter(
            endpoint="http://localhost:4318",
            service_name="genesis-trading"
        )
        
        # Create test spans
        trace_id = uuid4()
        spans = [
            Span(
                trace_id=trace_id,
                span_id=uuid4(),
                name="test_span",
                kind=SpanKind.SERVER
            )
        ]
        
        spans[0].add_attribute("test.attribute", "value")
        spans[0].set_status(SpanStatus.OK)
        spans[0].end()
        
        # Format for export
        formatted = exporter._format_spans(spans)
        
        assert "resourceSpans" in formatted
        assert formatted["resourceSpans"][0]["resource"]["attributes"][0]["key"] == "service.name"
        assert formatted["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"] == "genesis-trading"
    
    @pytest.mark.asyncio
    async def test_export_with_circuit_breaker(self):
        """Test export with circuit breaker."""
        exporter = OpenTelemetryExporter(
            endpoint="http://localhost:4318",
            service_name="genesis-trading"
        )
        
        # Open circuit breaker
        for _ in range(5):
            exporter.circuit_breaker.call_failed()
        
        spans = [
            Span(
                trace_id=uuid4(),
                span_id=uuid4(),
                name="test_span",
                kind=SpanKind.SERVER
            )
        ]
        
        # Should not attempt export
        result = await exporter.export(spans)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_batch_export(self):
        """Test batch span export."""
        exporter = OpenTelemetryExporter(
            endpoint="http://localhost:4318",
            service_name="genesis-trading",
            batch_size=5
        )
        
        # Add spans to batch
        for i in range(7):
            span = Span(
                trace_id=uuid4(),
                span_id=uuid4(),
                name=f"span_{i}",
                kind=SpanKind.INTERNAL
            )
            span.end()
            exporter.add_to_batch(span)
        
        # Should have 2 spans remaining after batch export
        assert len(exporter.span_batch) == 2
    
    @pytest.mark.asyncio
    async def test_exporter_cleanup(self):
        """Test exporter cleanup."""
        exporter = OpenTelemetryExporter(
            endpoint="http://localhost:4318",
            service_name="genesis-trading"
        )
        
        # Start exporter (would normally create background task)
        # await exporter.start()
        
        # Stop and cleanup
        await exporter.stop()
        
        # Check cleanup completed
        assert exporter._export_task is None