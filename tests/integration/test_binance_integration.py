"""
Integration tests for Binance API integration.
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import patch, AsyncMock

from genesis.exchange.gateway import BinanceGateway, OrderRequest
from genesis.exchange.websocket_manager import WebSocketManager
from genesis.exchange.circuit_breaker import CircuitBreakerManager
from genesis.exchange.time_sync import TimeSync
from genesis.exchange.health_monitor import HealthMonitor, HealthStatus


class TestBinanceIntegration:
    """Integration tests for complete Binance integration."""
    
    @pytest.mark.asyncio
    async def test_full_order_flow(self, mock_settings):
        """Test complete order flow from placement to status check."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            # Place order
            order_request = OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                type="limit",
                quantity=Decimal("0.001"),
                price=Decimal("49000"),
                client_order_id="integration_test_001"
            )
            
            order_response = await gateway.place_order(order_request)
            assert order_response.order_id is not None
            assert order_response.status in ["open", "filled"]
            
            # Check order status
            status = await gateway.get_order_status(order_response.order_id, "BTC/USDT")
            assert status.order_id == order_response.order_id
            
            # Cancel order (if still open)
            if status.status == "open":
                cancelled = await gateway.cancel_order(order_response.order_id, "BTC/USDT")
                assert cancelled is True
            
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self, mock_settings):
        """Test WebSocket automatic reconnection."""
        with patch("genesis.exchange.websocket_manager.get_settings", return_value=mock_settings):
            manager = WebSocketManager()
            
            # Mock WebSocket connection
            with patch("genesis.exchange.websocket_manager.websockets.connect") as mock_connect:
                mock_ws = AsyncMock()
                mock_ws.recv = AsyncMock(side_effect=[
                    '{"stream": "btcusdt@trade", "data": {"price": "50000"}}',
                    asyncio.CancelledError()  # Simulate disconnect
                ])
                mock_connect.return_value = mock_ws
                
                await manager.start()
                
                # Give it time to connect
                await asyncio.sleep(0.1)
                
                # Check that at least one connection was established
                assert len(manager.connections) > 0
                
                await manager.stop()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_settings):
        """Test circuit breaker protection during failures."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            # Create circuit breaker manager
            cb_manager = CircuitBreakerManager()
            api_breaker = cb_manager.get_breaker("api")
            
            # Simulate API failures
            async def failing_api_call():
                raise Exception("API Error")
            
            # Fail multiple times to trip the circuit
            for _ in range(3):
                with pytest.raises(Exception):
                    await api_breaker.call(failing_api_call)
            
            # Circuit should be open
            assert api_breaker.state == "open"
            
            # Next call should be rejected immediately
            from genesis.exchange.circuit_breaker import CircuitOpenError
            with pytest.raises(CircuitOpenError):
                await api_breaker.call(failing_api_call)
            
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, mock_settings):
        """Test rate limit enforcement across multiple requests."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            # Make multiple rapid requests
            tasks = []
            for i in range(5):
                order_request = OrderRequest(
                    symbol="BTC/USDT",
                    side="buy",
                    type="market",
                    quantity=Decimal("0.001")
                )
                tasks.append(gateway.place_order(order_request))
            
            # All should succeed without rate limit errors
            responses = await asyncio.gather(*tasks)
            assert len(responses) == 5
            assert all(r.order_id is not None for r in responses)
            
            # Check rate limiter statistics
            stats = gateway.rate_limiter.get_statistics()
            assert stats["total_requests"] >= 5
            assert stats["current_utilization_percent"] < 80  # Should stay below threshold
            
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_time_synchronization(self, mock_settings):
        """Test time synchronization with server."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            time_sync = TimeSync()
            await time_sync.start(gateway)
            
            # Perform sync
            await time_sync.sync_time()
            
            # Check synchronization
            assert time_sync.is_synchronized()
            stats = time_sync.get_statistics()
            assert stats["sync_count"] >= 1
            assert stats["is_synchronized"] is True
            
            # Get synchronized timestamp
            sync_timestamp = time_sync.get_synchronized_timestamp()
            assert sync_timestamp > 0
            
            await time_sync.stop()
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_settings):
        """Test health monitoring of exchange components."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            monitor = HealthMonitor()
            
            # Register gateway health check
            async def check_gateway():
                try:
                    await gateway.get_server_time()
                    return {"success": True}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            monitor.register_component("gateway", check_gateway)
            
            # Perform health check
            health = await monitor.check_component("gateway", check_gateway)
            
            assert health.name == "gateway"
            assert health.status == HealthStatus.HEALTHY
            assert health.success_rate == 1.0
            
            # Get overall health
            all_health = monitor.get_all_health()
            assert "gateway" in all_health
            assert monitor.is_healthy()
            
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_failover_between_websocket_connections(self, mock_settings):
        """Test failover between WebSocket connection pools."""
        with patch("genesis.exchange.websocket_manager.get_settings", return_value=mock_settings):
            manager = WebSocketManager()
            
            # Mock WebSocket connections
            with patch("genesis.exchange.websocket_manager.websockets.connect") as mock_connect:
                mock_ws = AsyncMock()
                mock_ws.recv = AsyncMock(return_value='{"stream": "test", "data": {}}')
                mock_ws.send = AsyncMock()
                mock_ws.close = AsyncMock()
                mock_connect.return_value = mock_ws
                
                await manager.start()
                
                # Check connection states
                states = manager.get_connection_states()
                
                # Should have multiple connections
                assert len(states) > 0
                
                # Get statistics
                stats = manager.get_statistics()
                assert stats["running"] is True
                assert "connections" in stats
                
                await manager.stop()
    
    @pytest.mark.asyncio
    async def test_mock_mode_functionality(self, mock_settings):
        """Test complete functionality in mock mode."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            # Test all major operations
            
            # 1. Get balance
            balances = await gateway.get_account_balance()
            assert "USDT" in balances
            
            # 2. Get ticker
            ticker = await gateway.get_ticker("BTC/USDT")
            assert ticker.symbol == "BTC/USDT"
            
            # 3. Get order book
            orderbook = await gateway.get_order_book("BTC/USDT")
            assert len(orderbook.bids) > 0
            assert len(orderbook.asks) > 0
            
            # 4. Get klines
            klines = await gateway.get_klines("BTC/USDT", "1m", 10)
            assert len(klines) == 10
            
            # 5. Place and manage order
            order = OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                type="limit",
                quantity=Decimal("0.001"),
                price=Decimal("49000")
            )
            response = await gateway.place_order(order)
            assert response.order_id is not None
            
            # 6. Check order status
            status = await gateway.get_order_status(response.order_id, "BTC/USDT")
            assert status.order_id == response.order_id
            
            # 7. Cancel order
            cancelled = await gateway.cancel_order(response.order_id, "BTC/USDT")
            assert cancelled is True
            
            await gateway.close()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_settings):
        """Test error handling and recovery mechanisms."""
        with patch("genesis.exchange.gateway.get_settings", return_value=mock_settings):
            gateway = BinanceGateway(mock_mode=True)
            await gateway.initialize()
            
            # Set mock exchange to fail
            gateway.mock_exchange.set_failure_rate(1.0)  # 100% failure
            
            # Attempt operations that should fail
            with pytest.raises(Exception, match="Mock API error"):
                await gateway.get_account_balance()
            
            # Reset failure rate
            gateway.mock_exchange.set_failure_rate(0.0)
            
            # Operations should work again
            balances = await gateway.get_account_balance()
            assert "USDT" in balances
            
            await gateway.close()