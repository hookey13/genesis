"""Market data feed integration tests."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
import websockets
from datetime import datetime, timedelta

from genesis.exchange.websocket_manager import WebSocketManager
from genesis.core.models import MarketData, OrderBook

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestMarketDataFeed:
    """Test market data feed functionality and resilience."""

    async def test_websocket_connection_resilience(self, trading_system):
        """Test WebSocket connection resilience and auto-reconnect."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws",
            reconnect_interval=1,
            max_reconnect_attempts=5
        )
        
        connection_states = []
        
        async def monitor_connection():
            for _ in range(10):
                state = ws_manager.is_connected()
                connection_states.append(state)
                await asyncio.sleep(0.5)
        
        await ws_manager.connect()
        
        monitor_task = asyncio.create_task(monitor_connection())
        
        await asyncio.sleep(2)
        await ws_manager.disconnect()  # Simulate disconnection
        
        await asyncio.sleep(2)
        await ws_manager.connect()  # Should auto-reconnect
        
        await monitor_task
        
        assert True in connection_states  # Was connected
        assert False in connection_states  # Was disconnected
        assert connection_states[-1] == True  # Reconnected

    async def test_data_normalization(self, trading_system):
        """Test market data normalization across different formats."""
        raw_data_formats = [
            {
                "e": "trade",
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "b": 123456,
                "a": 123457
            },
            {
                "event": "trade",
                "symbol": "BTC/USDT",
                "price": 50000.00,
                "amount": 0.1,
                "bid_id": 123456,
                "ask_id": 123457
            },
            {
                "type": "match",
                "product_id": "BTC-USDT",
                "price": "50000.00",
                "size": "0.1",
                "maker_order_id": "123456",
                "taker_order_id": "123457"
            }
        ]
        
        normalized_data = []
        
        for raw_data in raw_data_formats:
            normalized = await trading_system.engine.normalize_market_data(raw_data)
            normalized_data.append(normalized)
        
        for data in normalized_data:
            assert "symbol" in data
            assert "price" in data
            assert "quantity" in data
            assert isinstance(data["price"], Decimal)
            assert isinstance(data["quantity"], Decimal)

    async def test_reconnection_logic(self, trading_system):
        """Test automatic reconnection after network issues."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws",
            reconnect_interval=1,
            max_reconnect_attempts=3
        )
        
        reconnect_count = 0
        
        async def count_reconnects():
            nonlocal reconnect_count
            reconnect_count += 1
        
        ws_manager.on_reconnect = count_reconnects
        
        await ws_manager.connect()
        
        for _ in range(3):
            await ws_manager.simulate_network_error()
            await asyncio.sleep(2)
        
        assert reconnect_count >= 3
        assert ws_manager.is_connected()

    async def test_data_integrity_validation(self, trading_system):
        """Test validation of incoming market data integrity."""
        valid_data = {
            "symbol": "BTC/USDT",
            "bid": Decimal("50000.00"),
            "ask": Decimal("50001.00"),
            "last": Decimal("50000.50"),
            "volume": Decimal("1000.00"),
            "timestamp": datetime.now().timestamp()
        }
        
        invalid_data_samples = [
            {"symbol": "BTC/USDT"},  # Missing price data
            {"symbol": "BTC/USDT", "bid": "invalid"},  # Invalid price format
            {"symbol": "BTC/USDT", "bid": Decimal("-100")},  # Negative price
            {"symbol": "BTC/USDT", "bid": Decimal("50001"), "ask": Decimal("50000")},  # Inverted spread
            {"symbol": "", "bid": Decimal("50000")},  # Empty symbol
        ]
        
        is_valid = await trading_system.engine.validate_market_data(valid_data)
        assert is_valid
        
        for invalid_data in invalid_data_samples:
            is_valid = await trading_system.engine.validate_market_data(invalid_data)
            assert not is_valid

    async def test_orderbook_updates(self, trading_system):
        """Test order book update processing."""
        orderbook = OrderBook(symbol="BTC/USDT")
        
        updates = [
            {"side": "bid", "price": Decimal("50000"), "quantity": Decimal("1.0")},
            {"side": "bid", "price": Decimal("49999"), "quantity": Decimal("2.0")},
            {"side": "ask", "price": Decimal("50001"), "quantity": Decimal("1.5")},
            {"side": "ask", "price": Decimal("50002"), "quantity": Decimal("2.5")},
        ]
        
        for update in updates:
            await orderbook.update(update)
        
        best_bid = orderbook.get_best_bid()
        best_ask = orderbook.get_best_ask()
        
        assert best_bid["price"] == Decimal("50000")
        assert best_ask["price"] == Decimal("50001")
        assert orderbook.get_spread() == Decimal("1")

    async def test_multiple_symbol_subscriptions(self, trading_system):
        """Test subscribing to multiple symbol feeds simultaneously."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws"
        )
        
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
        
        await ws_manager.connect()
        
        for symbol in symbols:
            await ws_manager.subscribe(symbol)
        
        subscriptions = ws_manager.get_active_subscriptions()
        assert len(subscriptions) == len(symbols)
        assert all(s in subscriptions for s in symbols)
        
        await ws_manager.unsubscribe("BTC/USDT")
        subscriptions = ws_manager.get_active_subscriptions()
        assert len(subscriptions) == len(symbols) - 1
        assert "BTC/USDT" not in subscriptions

    async def test_data_buffer_overflow_handling(self, trading_system):
        """Test handling of data buffer overflow scenarios."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws",
            buffer_size=100
        )
        
        await ws_manager.connect()
        
        overflow_handled = False
        
        async def send_rapid_data():
            nonlocal overflow_handled
            for i in range(200):  # Send more than buffer size
                data = {
                    "symbol": "BTC/USDT",
                    "price": Decimal("50000") + Decimal(i),
                    "timestamp": datetime.now().timestamp()
                }
                try:
                    await ws_manager.process_data(data)
                except BufferError:
                    overflow_handled = True
                    break
        
        await send_rapid_data()
        
        assert overflow_handled or ws_manager.get_buffer_size() == 100

    async def test_latency_monitoring(self, trading_system):
        """Test WebSocket message latency monitoring."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws"
        )
        
        await ws_manager.connect()
        
        latencies = []
        
        for _ in range(100):
            start_time = datetime.now()
            
            data = {
                "symbol": "BTC/USDT",
                "price": Decimal("50000"),
                "server_time": start_time.timestamp()
            }
            
            await ws_manager.process_data(data)
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 5  # Average should be under 5ms
        assert max_latency < 10  # Max should be under 10ms

    async def test_heartbeat_mechanism(self, trading_system):
        """Test WebSocket heartbeat/ping-pong mechanism."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws",
            heartbeat_interval=5
        )
        
        await ws_manager.connect()
        
        heartbeats_sent = 0
        pongs_received = 0
        
        async def monitor_heartbeats():
            nonlocal heartbeats_sent, pongs_received
            for _ in range(20):
                if ws_manager.last_ping_time:
                    heartbeats_sent += 1
                if ws_manager.last_pong_time:
                    pongs_received += 1
                await asyncio.sleep(1)
        
        await monitor_heartbeats()
        
        assert heartbeats_sent > 0
        assert pongs_received > 0
        assert abs(heartbeats_sent - pongs_received) <= 1

    async def test_rate_limit_handling(self, trading_system):
        """Test handling of rate limits on WebSocket connections."""
        ws_manager = WebSocketManager(
            url="wss://stream.binance.com:9443/ws",
            rate_limit=10  # 10 messages per second
        )
        
        await ws_manager.connect()
        
        messages_sent = 0
        rate_limited = False
        
        for i in range(20):
            try:
                await ws_manager.send_message({"action": "subscribe", "symbol": f"TEST{i}"})
                messages_sent += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    rate_limited = True
                    break
            
            if i == 10:  # After 10 messages
                await asyncio.sleep(0.1)  # Small delay
        
        assert messages_sent <= 15 or rate_limited