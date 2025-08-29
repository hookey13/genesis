"""
Unit tests for Binance gateway connection pooling enhancements.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import aiohttp

from genesis.exchange.gateway import BinanceGateway


@pytest.fixture
async def gateway():
    """Create a test gateway instance."""
    with patch("genesis.exchange.gateway.get_settings") as mock_settings:
        mock_settings.return_value.exchange.binance_api_key.get_secret_value.return_value = "test_key"
        mock_settings.return_value.exchange.binance_api_secret.get_secret_value.return_value = "test_secret"
        mock_settings.return_value.exchange.binance_testnet = True
        mock_settings.return_value.exchange.exchange_rate_limit = 50
        mock_settings.return_value.development.use_mock_exchange = False
        
        gateway = BinanceGateway(mock_mode=True)
        yield gateway
        await gateway.close()


class TestConnectionPooling:
    """Test connection pooling functionality."""
    
    async def test_connection_pool_initialization(self, gateway):
        """Test that connection pool is properly initialized."""
        assert gateway._connection_pool_size == 10
        assert gateway._keep_alive_timeout == 30
        assert gateway._connection_metrics["total_requests"] == 0
        assert gateway._connection_metrics["failed_requests"] == 0
        assert gateway._connection_metrics["connection_reuses"] == 0
    
    async def test_persistent_session_creation(self):
        """Test that persistent session is created with correct configuration."""
        with patch("genesis.exchange.gateway.get_settings") as mock_settings:
            mock_settings.return_value.exchange.binance_api_key.get_secret_value.return_value = "test_key"
            mock_settings.return_value.exchange.binance_api_secret.get_secret_value.return_value = "test_secret"
            mock_settings.return_value.exchange.binance_testnet = True
            mock_settings.return_value.exchange.exchange_rate_limit = 50
            mock_settings.return_value.development.use_mock_exchange = False
            
            gateway = BinanceGateway(mock_mode=False)
            
            with patch("genesis.exchange.gateway.ccxt.binance") as mock_ccxt:
                mock_exchange = MagicMock()
                mock_exchange.load_markets = AsyncMock()
                mock_ccxt.return_value = mock_exchange
                
                await gateway.initialize()
                
                # Verify session was created
                assert gateway._session is not None
                assert isinstance(gateway._session, aiohttp.ClientSession)
                
                # Verify connector configuration
                connector = gateway._session.connector
                assert connector.limit == 10
                assert connector.limit_per_host == 10
                assert connector._keepalive_timeout == 30
                
                await gateway.close()
    
    async def test_connection_metrics_tracking(self, gateway):
        """Test that connection metrics are properly tracked."""
        # Initialize metrics
        assert gateway.get_connection_metrics()["total_requests"] == 0
        
        # Simulate requests
        gateway._connection_metrics["total_requests"] = 5
        gateway._connection_metrics["failed_requests"] = 1
        gateway._connection_metrics["connection_reuses"] = 3
        
        metrics = gateway.get_connection_metrics()
        assert metrics["total_requests"] == 5
        assert metrics["failed_requests"] == 1
        assert metrics["connection_reuses"] == 3
    
    async def test_retry_with_exponential_backoff(self, gateway):
        """Test retry logic with exponential backoff."""
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise aiohttp.ClientError("Connection timeout")
            return "success"
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await gateway._execute_with_retry(failing_func)
            
            assert result == "success"
            assert call_count == 3
            # Verify exponential backoff delays
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)  # First retry: base_delay
            mock_sleep.assert_any_call(2)  # Second retry: base_delay * 2
    
    async def test_max_retries_exceeded(self, gateway):
        """Test that max retries are respected."""
        async def always_failing_func():
            raise aiohttp.ClientError("Connection refused")
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(aiohttp.ClientError):
                await gateway._execute_with_retry(always_failing_func)
            
            # Verify failure metrics
            assert gateway._connection_metrics["failed_requests"] == 3
    
    async def test_connection_health_monitoring(self, gateway):
        """Test connection health monitoring."""
        # First health check should pass
        result = await gateway.monitor_connection_health()
        assert result is True
        
        # Update last check time to force new check
        gateway._connection_metrics["last_health_check"] = 0
        
        # Mock session with connector
        mock_connector = MagicMock()
        mock_connector._acquired = ["conn1", "conn2"]
        mock_connector._available = ["conn3", "conn4", "conn5"]
        mock_connector.limit = 10
        mock_connector.limit_per_host = 10
        
        gateway._session = MagicMock()
        gateway._session.connector = mock_connector
        
        result = await gateway.monitor_connection_health()
        assert result is True
        
        # Verify metrics were updated
        metrics = gateway.get_connection_metrics()
        assert metrics["active_connections"] == 2
        assert metrics["available_connections"] == 3
    
    async def test_connection_reuse_tracking(self, gateway):
        """Test that connection reuses are properly tracked."""
        # Successful on second attempt (reuse)
        attempt = 0
        
        async def func_succeeds_on_retry():
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise aiohttp.ClientError("Temporary failure")
            return "success"
        
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gateway._execute_with_retry(func_succeeds_on_retry)
            
            assert result == "success"
            assert gateway._connection_metrics["connection_reuses"] == 1
            assert gateway._connection_metrics["total_requests"] == 2
            assert gateway._connection_metrics["failed_requests"] == 1
    
    async def test_non_retryable_errors(self, gateway):
        """Test that non-retryable errors are not retried."""
        async def auth_error_func():
            raise Exception("Invalid API key")
        
        with pytest.raises(Exception, match="Invalid API key"):
            await gateway._execute_with_retry(auth_error_func)
        
        # Should fail immediately without retries
        assert gateway._connection_metrics["total_requests"] == 1
        assert gateway._connection_metrics["failed_requests"] == 1
    
    async def test_cleanup_on_close(self, gateway):
        """Test that resources are properly cleaned up on close."""
        # Create mock session
        gateway._session = MagicMock()
        gateway._session.close = AsyncMock()
        
        gateway._exchange = MagicMock()
        gateway._exchange.close = AsyncMock()
        
        gateway._initialized = True
        
        await gateway.close()
        
        # Verify cleanup
        assert gateway._session is None
        assert gateway._initialized is False
        gateway._exchange.close.assert_called_once()