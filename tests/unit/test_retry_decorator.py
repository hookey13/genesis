"""
Unit tests for the retry decorator with exponential backoff.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from genesis.core.exceptions import (
    NetworkError,
    ConnectionTimeout,
    RateLimitError,
    DatabaseLocked,
    ValidationError,
)
from genesis.utils.decorators import with_retry


class TestRetryDecorator:
    """Test the with_retry decorator."""
    
    def test_successful_execution_no_retry_needed(self):
        """Test function succeeds on first attempt."""
        mock_func = MagicMock(return_value="success")
        
        @with_retry(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_successful_execution_no_retry_needed(self):
        """Test async function succeeds on first attempt."""
        mock_func = AsyncMock(return_value="success")
        
        @with_retry(max_attempts=3)
        async def test_func():
            return await mock_func()
        
        result = await test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_on_failure_then_success(self):
        """Test function retries on failure and eventually succeeds."""
        mock_func = MagicMock(side_effect=[
            NetworkError("Connection failed"),
            NetworkError("Connection failed"),
            "success"
        ])
        
        @with_retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_on_failure_then_success(self):
        """Test async function retries on failure and eventually succeeds."""
        mock_func = AsyncMock(side_effect=[
            ConnectionTimeout("Timeout", timeout_seconds=30),
            RateLimitError("Rate limited"),
            "success"
        ])
        
        @with_retry(max_attempts=3, initial_delay=0.01)
        async def test_func():
            return await mock_func()
        
        result = await test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_max_attempts_exhausted(self):
        """Test function raises exception after max attempts."""
        mock_func = MagicMock(side_effect=NetworkError("Connection failed"))
        
        @with_retry(max_attempts=3, initial_delay=0.01)
        def test_func():
            return mock_func()
        
        with pytest.raises(NetworkError) as exc_info:
            test_func()
        
        assert str(exc_info.value) == "Connection failed"
        assert mock_func.call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_max_attempts_exhausted(self):
        """Test async function raises exception after max attempts."""
        mock_func = AsyncMock(side_effect=DatabaseLocked("Database is locked"))
        
        @with_retry(max_attempts=2, initial_delay=0.01)
        async def test_func():
            return await mock_func()
        
        with pytest.raises(DatabaseLocked) as exc_info:
            await test_func()
        
        assert str(exc_info.value) == "Database is locked"
        assert mock_func.call_count == 2
    
    def test_non_retryable_exception_raised_immediately(self):
        """Test non-retryable exceptions are raised immediately."""
        mock_func = MagicMock(side_effect=ValidationError("Invalid input"))
        
        @with_retry(
            max_attempts=3,
            retryable_exceptions=(NetworkError, ConnectionTimeout)
        )
        def test_func():
            return mock_func()
        
        with pytest.raises(ValidationError) as exc_info:
            test_func()
        
        assert str(exc_info.value) == "Invalid input"
        assert mock_func.call_count == 1  # No retries
    
    @pytest.mark.asyncio
    async def test_async_non_retryable_exception_raised_immediately(self):
        """Test async non-retryable exceptions are raised immediately."""
        mock_func = AsyncMock(side_effect=ValueError("Invalid value"))
        
        @with_retry(
            max_attempts=3,
            retryable_exceptions=(NetworkError,)
        )
        async def test_func():
            return await mock_func()
        
        with pytest.raises(ValueError) as exc_info:
            await test_func()
        
        assert str(exc_info.value) == "Invalid value"
        assert mock_func.call_count == 1  # No retries
    
    def test_exponential_backoff_timing(self):
        """Test exponential backoff increases delay between retries."""
        mock_func = MagicMock(side_effect=NetworkError("Connection failed"))
        
        with patch("time.sleep") as mock_sleep:
            @with_retry(
                max_attempts=4,
                initial_delay=1.0,
                backoff_factor=2.0,
                jitter=False  # Disable jitter for predictable timing
            )
            def test_func():
                return mock_func()
            
            with pytest.raises(NetworkError):
                test_func()
            
            # Verify delays: 1s, 2s, 4s
            assert mock_sleep.call_count == 3
            delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert delays[0] == 1.0
            assert delays[1] == 2.0
            assert delays[2] == 4.0
    
    @pytest.mark.asyncio
    async def test_async_exponential_backoff_timing(self):
        """Test async exponential backoff increases delay between retries."""
        mock_func = AsyncMock(side_effect=ConnectionTimeout("Timeout", timeout_seconds=30))
        
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = asyncio.sleep(0)  # Make it fast
            
            @with_retry(
                max_attempts=4,
                initial_delay=1.0,
                backoff_factor=2.0,
                jitter=False  # Disable jitter for predictable timing
            )
            async def test_func():
                return await mock_func()
            
            with pytest.raises(ConnectionTimeout):
                await test_func()
            
            # Verify delays: 1s, 2s, 4s
            assert mock_sleep.call_count == 3
            delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert delays[0] == 1.0
            assert delays[1] == 2.0
            assert delays[2] == 4.0
    
    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        mock_func = MagicMock(side_effect=NetworkError("Connection failed"))
        
        with patch("time.sleep") as mock_sleep:
            @with_retry(
                max_attempts=5,
                initial_delay=1.0,
                max_delay=5.0,
                backoff_factor=3.0,
                jitter=False
            )
            def test_func():
                return mock_func()
            
            with pytest.raises(NetworkError):
                test_func()
            
            # Verify delays: 1s, 3s, 5s (capped), 5s (capped)
            assert mock_sleep.call_count == 4
            delays = [call[0][0] for call in mock_sleep.call_args_list]
            assert delays[0] == 1.0
            assert delays[1] == 3.0
            assert delays[2] == 5.0  # Capped at max_delay
            assert delays[3] == 5.0  # Still capped
    
    def test_jitter_adds_randomness(self):
        """Test jitter adds randomness to delays."""
        mock_func = MagicMock(side_effect=NetworkError("Connection failed"))
        
        with patch("time.sleep") as mock_sleep:
            @with_retry(
                max_attempts=3,
                initial_delay=1.0,
                jitter=True
            )
            def test_func():
                return mock_func()
            
            with pytest.raises(NetworkError):
                test_func()
            
            # Verify delays have jitter (not exactly 1.0 and 2.0)
            assert mock_sleep.call_count == 2
            delays = [call[0][0] for call in mock_sleep.call_args_list]
            
            # With jitter, delays should be within ±25% of expected
            assert 0.75 <= delays[0] <= 1.25  # 1.0 ± 25%
            assert 1.5 <= delays[1] <= 2.5    # 2.0 ± 25%
    
    def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions configuration."""
        mock_func = MagicMock(side_effect=[
            ConnectionTimeout("Timeout", timeout_seconds=30),
            ValueError("Not retryable"),
            "success"
        ])
        
        @with_retry(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionTimeout,)
        )
        def test_func():
            return mock_func()
        
        # Should retry on ConnectionTimeout but not on ValueError
        with pytest.raises(ValueError):
            test_func()
        
        assert mock_func.call_count == 2  # First attempt + one retry
    
    @pytest.mark.asyncio
    async def test_async_mixed_exceptions(self):
        """Test async function with mix of retryable and non-retryable exceptions."""
        mock_func = AsyncMock(side_effect=[
            RateLimitError("Rate limited"),
            DatabaseLocked("Database locked"),
            ValidationError("Invalid input")
        ])
        
        @with_retry(
            max_attempts=5,
            initial_delay=0.01,
            retryable_exceptions=(RateLimitError, DatabaseLocked)
        )
        async def test_func():
            return await mock_func()
        
        with pytest.raises(ValidationError):
            await test_func()
        
        # Should retry twice before hitting non-retryable exception
        assert mock_func.call_count == 3
    
    def test_logging_on_retry(self):
        """Test logging occurs on retry attempts."""
        mock_func = MagicMock(side_effect=[
            NetworkError("Connection failed"),
            "success"
        ])
        
        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @with_retry(max_attempts=2, initial_delay=0.01)
            def test_func():
                return mock_func()
            
            result = test_func()
            
            assert result == "success"
            # Should log warning for retry
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args[1]
            assert call_kwargs["function"] == "test_func"
            assert call_kwargs["attempt"] == 1
            assert call_kwargs["max_attempts"] == 2
            assert call_kwargs["error_type"] == "NetworkError"
    
    def test_logging_on_exhausted_attempts(self):
        """Test logging when max attempts exhausted."""
        mock_func = MagicMock(side_effect=NetworkError("Connection failed"))
        
        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @with_retry(max_attempts=2, initial_delay=0.01)
            def test_func():
                return mock_func()
            
            with pytest.raises(NetworkError):
                test_func()
            
            # Should log error for exhausted attempts
            mock_logger.error.assert_called()
            error_calls = [call for call in mock_logger.error.call_args_list
                          if "Max retry attempts exhausted" in call[0][0]]
            assert len(error_calls) == 1
    
    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves original function metadata."""
        @with_retry(max_attempts=2)
        def original_function():
            """Original function docstring."""
            return "result"
        
        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original function docstring."
    
    @pytest.mark.asyncio
    async def test_async_decorator_preserves_function_metadata(self):
        """Test async decorator preserves original function metadata."""
        @with_retry(max_attempts=2)
        async def original_async_function():
            """Original async function docstring."""
            return "result"
        
        assert original_async_function.__name__ == "original_async_function"
        assert original_async_function.__doc__ == "Original async function docstring."
        assert asyncio.iscoroutinefunction(original_async_function)