"""
Unit tests for the global error handler infrastructure.
"""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import structlog

from genesis.core.error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    GlobalErrorHandler,
    RemediationStep,
    get_error_handler,
)


class TestErrorContext:
    """Test ErrorContext class."""
    
    def test_error_context_initialization(self):
        """Test ErrorContext initializes correctly."""
        error = ValueError("Test error")
        context = ErrorContext(
            correlation_id="test-123",
            error=error,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            component="test_component",
            function="test_function",
            line_number=42,
            additional_context={"key": "value"},
        )
        
        assert context.correlation_id == "test-123"
        assert context.error == error
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.VALIDATION
        assert context.component == "test_component"
        assert context.function == "test_function"
        assert context.line_number == 42
        assert context.additional_context == {"key": "value"}
        assert isinstance(context.timestamp, datetime)
        assert context.error_code.startswith("GENESIS-")
    
    def test_error_code_generation(self):
        """Test error code generation follows GENESIS-XXXX format."""
        error = ValueError("Test")
        context = ErrorContext(
            correlation_id="test",
            error=error,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.EXCHANGE,
            component="test",
            function="test",
            line_number=1,
        )
        
        # Should be GENESIS-19XX where 1=EXCHANGE, 9=CRITICAL, XX=hash
        assert context.error_code.startswith("GENESIS-19")
        assert len(context.error_code) >= 11  # GENESIS-XXXX format
    
    def test_error_context_to_dict(self):
        """Test converting ErrorContext to dictionary."""
        error = RuntimeError("Test error")
        context = ErrorContext(
            correlation_id="test-456",
            error=error,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            component="component",
            function="function",
            line_number=100,
            additional_context={"extra": "data"},
        )
        
        result = context.to_dict()
        
        assert result["correlation_id"] == "test-456"
        assert result["error_type"] == "RuntimeError"
        assert result["error_message"] == "Test error"
        assert result["severity"] == "high"
        assert result["category"] == "system"
        assert result["component"] == "component"
        assert result["function"] == "function"
        assert result["line_number"] == 100
        assert result["additional_context"] == {"extra": "data"}
        assert "traceback" in result
        assert "timestamp" in result
        assert "error_code" in result


class TestGlobalErrorHandler:
    """Test GlobalErrorHandler class."""
    
    @pytest.fixture
    def handler(self):
        """Create a GlobalErrorHandler instance."""
        return GlobalErrorHandler()
    
    @pytest.fixture
    def mock_logger(self, handler):
        """Mock the logger for testing."""
        handler.logger = MagicMock()
        return handler.logger
    
    def test_handler_initialization(self, handler):
        """Test GlobalErrorHandler initializes correctly."""
        assert handler.logger is not None
        assert len(handler._error_handlers) == 0
        assert len(handler._remediation_registry) > 0  # Has default remediations
        assert len(handler._error_counts) == 0
        assert len(handler._critical_error_callbacks) == 0
    
    def test_generate_correlation_id(self, handler):
        """Test correlation ID generation."""
        correlation_id = handler.generate_correlation_id()
        
        assert isinstance(correlation_id, str)
        assert len(correlation_id) == 36  # UUID4 format
        
        # Verify it's a valid UUID
        uuid.UUID(correlation_id)
        
        # Each call should generate unique ID
        id2 = handler.generate_correlation_id()
        assert correlation_id != id2
    
    def test_handle_error_basic(self, handler, mock_logger):
        """Test basic error handling."""
        error = ValueError("Test error")
        
        context = handler.handle_error(
            error=error,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            component="test_comp",
            function="test_func",
            line_number=50,
        )
        
        assert isinstance(context, ErrorContext)
        assert context.error == error
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.VALIDATION
        assert context.component == "test_comp"
        assert context.function == "test_func"
        assert context.line_number == 50
        
        # Verify logging
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "Error handled"
    
    def test_handle_error_with_correlation_id(self, handler, mock_logger):
        """Test error handling with provided correlation ID."""
        error = RuntimeError("Test")
        correlation_id = "custom-id-123"
        
        context = handler.handle_error(
            error=error,
            correlation_id=correlation_id,
        )
        
        assert context.correlation_id == correlation_id
    
    def test_handle_critical_error(self, handler, mock_logger):
        """Test critical error handling triggers escalation."""
        error = Exception("Critical failure")
        callback_called = False
        
        def critical_callback(context):
            nonlocal callback_called
            callback_called = True
            
        handler.register_critical_error_callback(critical_callback)
        
        context = handler.handle_error(
            error=error,
            severity=ErrorSeverity.CRITICAL,
        )
        
        # Verify critical logging
        mock_logger.critical.assert_called()
        assert callback_called
    
    @pytest.mark.asyncio
    async def test_handle_async_error(self, handler, mock_logger):
        """Test async error handling."""
        error = ValueError("Async error")
        
        context = await handler.handle_async_error(
            error=error,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.NETWORK,
        )
        
        assert isinstance(context, ErrorContext)
        assert context.error == error
        assert context.severity == ErrorSeverity.LOW
        assert context.category == ErrorCategory.NETWORK
        
        mock_logger.info.assert_called_once()
    
    def test_error_counting(self, handler, mock_logger):
        """Test error counting and statistics."""
        # Generate different types of errors
        handler.handle_error(
            ValueError("Test"),
            category=ErrorCategory.VALIDATION,
        )
        handler.handle_error(
            ValueError("Test2"),
            category=ErrorCategory.VALIDATION,
        )
        handler.handle_error(
            RuntimeError("Test"),
            category=ErrorCategory.SYSTEM,
        )
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert "validation:ValueError" in stats["error_counts"]
        assert stats["error_counts"]["validation:ValueError"] == 2
        assert stats["error_counts"]["system:RuntimeError"] == 1
        assert stats["category_statistics"]["validation"] == 2
        assert stats["category_statistics"]["system"] == 1
    
    def test_clear_error_counts(self, handler, mock_logger):
        """Test clearing error statistics."""
        handler.handle_error(ValueError("Test"))
        assert handler.get_error_statistics()["total_errors"] == 1
        
        handler.clear_error_counts()
        assert handler.get_error_statistics()["total_errors"] == 0
    
    def test_remediation_steps_retrieval(self, handler):
        """Test getting remediation steps for errors."""
        # Test with a registered error type
        class RateLimitError(Exception):
            pass
            
        handler.register_remediation_steps(
            "RateLimitError",
            [
                RemediationStep(
                    "wait",
                    "Wait for rate limit reset",
                    automated=True,
                    retry_after_seconds=60,
                ),
            ],
        )
        
        error = RateLimitError("Rate limited")
        steps = handler.get_remediation_steps(error)
        
        assert steps is not None
        assert len(steps) == 1
        assert steps[0].action == "wait"
        assert steps[0].automated is True
        assert steps[0].retry_after_seconds == 60
    
    def test_register_error_handler(self, handler):
        """Test registering custom error handlers."""
        def custom_handler(error):
            return "handled"
            
        handler.register_error_handler(ValueError, custom_handler)
        
        assert ValueError in handler._error_handlers
        assert handler._error_handlers[ValueError] == custom_handler
    
    def test_severity_based_logging(self, handler, mock_logger):
        """Test different severity levels use appropriate log methods."""
        severities = [
            (ErrorSeverity.CRITICAL, mock_logger.critical),
            (ErrorSeverity.HIGH, mock_logger.error),
            (ErrorSeverity.MEDIUM, mock_logger.warning),
            (ErrorSeverity.LOW, mock_logger.info),
        ]
        
        for severity, expected_method in severities:
            handler.handle_error(
                Exception("Test"),
                severity=severity,
            )
            expected_method.assert_called()
    
    def test_additional_context_included(self, handler, mock_logger):
        """Test additional context is included in error handling."""
        error = ValueError("Test")
        additional_context = {
            "user_id": "12345",
            "order_id": "ABC-123",
            "amount": 100.50,
        }
        
        context = handler.handle_error(
            error=error,
            additional_context=additional_context,
        )
        
        assert context.additional_context == additional_context
        
        # Verify context is in log call
        call_args = mock_logger.warning.call_args
        logged_dict = call_args[1]
        assert logged_dict["additional_context"] == additional_context
    
    @pytest.mark.asyncio
    async def test_async_critical_callback(self, handler, mock_logger):
        """Test async critical error callbacks."""
        callback_executed = False
        
        async def async_callback(context):
            nonlocal callback_executed
            await asyncio.sleep(0.01)
            callback_executed = True
            
        handler.register_critical_error_callback(async_callback)
        
        handler.handle_error(
            Exception("Critical"),
            severity=ErrorSeverity.CRITICAL,
        )
        
        # Allow async callback to execute
        await asyncio.sleep(0.02)
        
        assert callback_executed


class TestGlobalErrorHandlerSingleton:
    """Test global error handler singleton pattern."""
    
    def test_get_error_handler_returns_singleton(self):
        """Test get_error_handler returns the same instance."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2
        assert isinstance(handler1, GlobalErrorHandler)
    
    def test_singleton_persists_state(self):
        """Test singleton maintains state between calls."""
        handler1 = get_error_handler()
        
        # Add some state
        handler1.handle_error(ValueError("Test"))
        stats1 = handler1.get_error_statistics()
        
        # Get handler again
        handler2 = get_error_handler()
        stats2 = handler2.get_error_statistics()
        
        assert stats1 == stats2
        assert stats2["total_errors"] > 0


class TestRemediationStep:
    """Test RemediationStep class."""
    
    def test_remediation_step_creation(self):
        """Test creating remediation steps."""
        step = RemediationStep(
            action="retry",
            description="Retry the operation",
            automated=True,
            retry_after_seconds=30,
        )
        
        assert step.action == "retry"
        assert step.description == "Retry the operation"
        assert step.automated is True
        assert step.retry_after_seconds == 30
    
    def test_remediation_step_defaults(self):
        """Test remediation step default values."""
        step = RemediationStep(
            action="manual_fix",
            description="Manual intervention required",
        )
        
        assert step.action == "manual_fix"
        assert step.description == "Manual intervention required"
        assert step.automated is False
        assert step.retry_after_seconds is None


class TestDefaultRemediationRegistry:
    """Test default remediation registry entries."""
    
    def test_default_remediations_loaded(self):
        """Test default remediation steps are loaded."""
        handler = GlobalErrorHandler()
        
        # Check some default entries exist
        expected_errors = [
            "RateLimitError",
            "ConnectionTimeout",
            "DatabaseLocked",
            "TierViolation",
            "RiskLimitExceeded",
        ]
        
        for error_name in expected_errors:
            assert error_name in handler._remediation_registry
            steps = handler._remediation_registry[error_name]
            assert len(steps) > 0
            assert all(isinstance(step, RemediationStep) for step in steps)