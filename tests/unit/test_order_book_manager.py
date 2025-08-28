"""Unit tests for Order Book Manager."""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets

from genesis.exchange.order_book_manager import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderBookManager
)
from genesis.engine.event_bus import EventBus
from genesis.core.events import Event


class TestOrderBookLevel:
    """Test OrderBookLevel dataclass."""
    
    def test_order_book_level_creation(self):
        """Test creating an order book level."""
        level = OrderBookLevel(
            price=Decimal("50000.00"),
            quantity=Decimal("1.5"),
            order_count=3
        )
        
        assert level.price == Decimal("50000.00")
        assert level.quantity == Decimal("1.5")
        assert level.order_count == 3
        assert level.notional == Decimal("75000.00")


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot dataclass."""
    
    def test_empty_snapshot(self):
        """Test empty order book snapshot."""
        snapshot = OrderBookSnapshot(symbol="BTCUSDT")
        
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.mid_price is None
        assert snapshot.spread is None
        assert snapshot.spread_bps is None
    
    def test_snapshot_with_data(self):
        """Test order book snapshot with bid/ask data."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("1.0")),
                OrderBookLevel(Decimal("49999"), Decimal("2.0")),
                OrderBookLevel(Decimal("49998"), Decimal("1.5"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("1.0")),
                OrderBookLevel(Decimal("50002"), Decimal("1.5")),
                OrderBookLevel(Decimal("50003"), Decimal("2.0"))
            ]
        )
        
        assert snapshot.best_bid == Decimal("50000")
        assert snapshot.best_ask == Decimal("50001")
        assert snapshot.mid_price == Decimal("50000.5")
        assert snapshot.spread == Decimal("1")
        assert snapshot.spread_bps == 1  # 0.01%
    
    def test_volume_calculations(self):
        """Test volume calculation methods."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("1.0")),
                OrderBookLevel(Decimal("49999"), Decimal("2.0")),
                OrderBookLevel(Decimal("49998"), Decimal("1.5"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("0.5")),
                OrderBookLevel(Decimal("50002"), Decimal("1.5")),
                OrderBookLevel(Decimal("50003"), Decimal("2.0"))
            ]
        )
        
        assert snapshot.get_bid_volume(2) == Decimal("3.0")
        assert snapshot.get_ask_volume(2) == Decimal("2.0")
        assert snapshot.get_bid_volume() == Decimal("4.5")
        assert snapshot.get_ask_volume() == Decimal("4.0")
    
    def test_imbalance_ratio(self):
        """Test order book imbalance ratio calculation."""
        # Balanced book
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1.0"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1.0"))]
        )
        assert snapshot.get_imbalance_ratio() == Decimal("0")
        
        # Buy pressure
        snapshot.bids = [OrderBookLevel(Decimal("50000"), Decimal("2.0"))]
        snapshot.asks = [OrderBookLevel(Decimal("50001"), Decimal("1.0"))]
        imbalance = snapshot.get_imbalance_ratio()
        assert imbalance > 0
        assert abs(imbalance - Decimal("0.3333")) < Decimal("0.01")
        
        # Sell pressure
        snapshot.bids = [OrderBookLevel(Decimal("50000"), Decimal("1.0"))]
        snapshot.asks = [OrderBookLevel(Decimal("50001"), Decimal("3.0"))]
        imbalance = snapshot.get_imbalance_ratio()
        assert imbalance < 0
        assert abs(imbalance + Decimal("0.5")) < Decimal("0.01")
    
    def test_weighted_mid_price(self):
        """Test volume-weighted mid price calculation."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("2.0")),
                OrderBookLevel(Decimal("49999"), Decimal("1.0"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("1.0")),
                OrderBookLevel(Decimal("50002"), Decimal("2.0"))
            ]
        )
        
        weighted_mid = snapshot.get_weighted_mid_price(levels=2)
        assert weighted_mid is not None
        # Should be weighted towards the larger volumes
        assert Decimal("50000") < weighted_mid < Decimal("50001")


@pytest.mark.asyncio
class TestOrderBookManager:
    """Test OrderBookManager class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock(spec=EventBus)
        return bus
    
    @pytest.fixture
    def manager(self, event_bus):
        """Create order book manager instance."""
        return OrderBookManager(event_bus, depth_levels=20, update_frequency_ms=100)
    
    async def test_initialization(self, manager, event_bus):
        """Test order book manager initialization."""
        assert manager.event_bus == event_bus
        assert manager.depth_levels == 20
        assert manager.update_frequency_ms == 100
        assert manager.order_books == {}
        assert not manager.running
    
    async def test_start_stop(self, manager):
        """Test starting and stopping the manager."""
        with patch.object(manager, '_connect_and_subscribe', new_callable=AsyncMock):
            await manager.start(["BTCUSDT", "ETHUSDT"])
            assert manager.running
            assert manager.heartbeat_task is not None
            
            await manager.stop()
            assert not manager.running
    
    def test_parse_depth_update(self, manager):
        """Test parsing Binance depth update."""
        data = {
            "lastUpdateId": 12345,
            "bids": [
                ["50000.00", "1.5"],
                ["49999.00", "2.0"],
                ["49998.00", "1.0"]
            ],
            "asks": [
                ["50001.00", "1.0"],
                ["50002.00", "1.5"],
                ["50003.00", "2.0"]
            ]
        }
        
        snapshot = manager._parse_depth_update("BTCUSDT", data)
        
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.sequence_number == 12345
        assert len(snapshot.bids) == 3
        assert len(snapshot.asks) == 3
        assert snapshot.bids[0].price == Decimal("50000.00")
        assert snapshot.asks[0].price == Decimal("50001.00")
        # Check sorting
        assert snapshot.bids[0].price > snapshot.bids[1].price
        assert snapshot.asks[0].price < snapshot.asks[1].price
    
    async def test_process_order_book_updates(self, manager, event_bus):
        """Test processing order book updates from websocket."""
        manager.running = True
        
        # Create mock websocket with test data
        mock_ws = AsyncMock()
        test_messages = [
            json.dumps({
                "lastUpdateId": 1,
                "bids": [["50000.00", "1.0"]],
                "asks": [["50001.00", "1.0"]]
            }),
            json.dumps({
                "lastUpdateId": 2,
                "bids": [["50000.00", "3.0"]],  # Increased bid volume
                "asks": [["50001.00", "1.0"]]
            })
        ]
        
        # Make the mock async iterable
        async def mock_iter():
            for msg in test_messages:
                yield msg
            manager.running = False  # Stop after messages
        
        mock_ws.__aiter__ = mock_iter
        
        await manager._process_order_book_updates("BTCUSDT", mock_ws)
        
        # Check events were published
        assert event_bus.publish.call_count >= 2
        
        # Check imbalance detection
        calls = event_bus.publish.call_args_list
        imbalance_detected = False
        for call in calls:
            event = call[0][0]
            if event.type == "order_book_imbalance_detected":
                imbalance_detected = True
                assert event.data["direction"] == "buy"
        
        assert imbalance_detected
    
    def test_get_order_book(self, manager):
        """Test retrieving order book snapshot."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1.0"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1.0"))]
        )
        
        manager.order_books["BTCUSDT"] = snapshot
        
        retrieved = manager.get_order_book("BTCUSDT")
        assert retrieved == snapshot
        
        assert manager.get_order_book("ETHUSDT") is None
    
    def test_get_mid_price(self, manager):
        """Test getting mid price."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1.0"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1.0"))]
        )
        
        manager.order_books["BTCUSDT"] = snapshot
        
        mid_price = manager.get_mid_price("BTCUSDT")
        assert mid_price == Decimal("50000.5")
        
        assert manager.get_mid_price("ETHUSDT") is None
    
    def test_get_spread(self, manager):
        """Test getting spread."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1.0"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1.0"))]
        )
        
        manager.order_books["BTCUSDT"] = snapshot
        
        spread = manager.get_spread("BTCUSDT")
        assert spread == Decimal("1")
        
        assert manager.get_spread("ETHUSDT") is None
    
    def test_get_imbalance(self, manager):
        """Test getting order book imbalance."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("3.0")),
                OrderBookLevel(Decimal("49999"), Decimal("2.0"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("1.0")),
                OrderBookLevel(Decimal("50002"), Decimal("1.0"))
            ]
        )
        
        manager.order_books["BTCUSDT"] = snapshot
        
        imbalance = manager.get_imbalance("BTCUSDT", levels=2)
        assert imbalance > 0  # Buy pressure
        assert abs(imbalance - Decimal("0.4286")) < Decimal("0.01")
        
        assert manager.get_imbalance("ETHUSDT") is None
    
    def test_calculate_price_impact(self, manager):
        """Test calculating price impact for orders."""
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("1.0")),
                OrderBookLevel(Decimal("49999"), Decimal("2.0")),
                OrderBookLevel(Decimal("49998"), Decimal("3.0"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("1.0")),
                OrderBookLevel(Decimal("50002"), Decimal("2.0")),
                OrderBookLevel(Decimal("50003"), Decimal("3.0"))
            ]
        )
        
        manager.order_books["BTCUSDT"] = snapshot
        
        # Test buy order impact
        avg_price, impact = manager.calculate_price_impact("BTCUSDT", "buy", Decimal("2.5"))
        assert avg_price == Decimal("50001.8")  # Weighted average
        assert impact > 0
        
        # Test sell order impact
        avg_price, impact = manager.calculate_price_impact("BTCUSDT", "sell", Decimal("2.5"))
        assert avg_price < Decimal("50000")
        assert impact > 0
        
        # Test insufficient liquidity
        result = manager.calculate_price_impact("BTCUSDT", "buy", Decimal("100"))
        assert result is None
        
        # Test missing order book
        result = manager.calculate_price_impact("ETHUSDT", "buy", Decimal("1"))
        assert result is None
    
    async def test_websocket_reconnection(self, manager):
        """Test websocket reconnection with exponential backoff."""
        manager.running = True
        
        with patch('websockets.connect', side_effect=ConnectionError("Test error")):
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                # Run for a limited time
                async def stop_after_attempts():
                    await asyncio.sleep(0.1)
                    manager.running = False
                
                task = asyncio.create_task(manager._connect_and_subscribe("BTCUSDT"))
                stop_task = asyncio.create_task(stop_after_attempts())
                
                await asyncio.gather(task, stop_task, return_exceptions=True)
                
                # Check exponential backoff was applied
                sleep_calls = mock_sleep.call_args_list
                if len(sleep_calls) >= 2:
                    assert sleep_calls[1][0][0] > sleep_calls[0][0][0]
    
    async def test_heartbeat_loop(self, manager):
        """Test heartbeat loop for keeping connections alive."""
        manager.running = True
        
        # Create mock websocket connections
        mock_ws1 = AsyncMock()
        mock_ws1.closed = False
        mock_ws1.ping.return_value = asyncio.create_future()
        mock_ws1.ping.return_value.set_result(None)
        
        mock_ws2 = AsyncMock()
        mock_ws2.closed = False
        mock_ws2.ping.return_value = asyncio.create_future()
        mock_ws2.ping.return_value.set_result(None)
        
        manager.websocket_connections = {
            "BTCUSDT": mock_ws1,
            "ETHUSDT": mock_ws2
        }
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Run one iteration of heartbeat
            async def run_one_heartbeat():
                await manager._heartbeat_loop()
            
            # Stop after one iteration
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            
            with pytest.raises(asyncio.CancelledError):
                await run_one_heartbeat()
            
            # Check pings were sent
            mock_ws1.ping.assert_called_once()
            mock_ws2.ping.assert_called_once()