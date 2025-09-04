"""Test system fixtures for integration and E2E tests."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, MagicMock
import structlog
from datetime import datetime, timedelta

from genesis.engine.engine import TradingEngine
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import StateMachine, TierState
from genesis.exchange.gateway import ExchangeGateway
from genesis.exchange.websocket_manager import WebSocketManager
from genesis.data.repository import DatabaseConnection, StateRepository
from genesis.cache.manager import CacheManager
from genesis.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


class MockExchangeGateway:
    """Mock exchange gateway for testing."""
    
    def __init__(self):
        self.orders = {}
        self.balances = {
            "USDT": Decimal("10000.00"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("10.0")
        }
        self.market_data = {}
        self.order_counter = 0
        self.connection_status = True
        
    async def connect(self):
        """Connect to exchange."""
        if not self.connection_status:
            raise ConnectionError("Exchange connection failed")
        return True
    
    async def disconnect(self):
        """Disconnect from exchange."""
        self.connection_status = False
        return True
    
    async def get_balance(self, asset: Optional[str] = None):
        """Get account balance."""
        if asset:
            return self.balances.get(asset, Decimal("0"))
        return self.balances
    
    async def get_ticker(self, symbol: str):
        """Get current ticker for symbol."""
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                "bid": Decimal("50000.00"),
                "ask": Decimal("50001.00"),
                "last": Decimal("50000.50"),
                "volume": Decimal("1000.00")
            }
        return self.market_data[symbol]
    
    async def place_order(self, order: Dict[str, Any]):
        """Place an order."""
        self.order_counter += 1
        order_id = f"mock_order_{self.order_counter}"
        
        self.orders[order_id] = {
            **order,
            "order_id": order_id,
            "status": "filled",
            "filled_qty": order.get("quantity"),
            "avg_price": order.get("price", Decimal("50000.00")),
            "timestamp": datetime.now()
        }
        
        return self.orders[order_id]
    
    async def cancel_order(self, order_id: str):
        """Cancel an order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False
    
    async def get_order_status(self, order_id: str):
        """Get order status."""
        return self.orders.get(order_id)
    
    async def health_check(self):
        """Check exchange health."""
        return {
            "status": "healthy" if self.connection_status else "unhealthy",
            "latency_ms": 5,
            "timestamp": datetime.now()
        }


class MockWebSocketManager:
    """Mock WebSocket manager for testing."""
    
    def __init__(self, url: str):
        self.url = url
        self.connected = False
        self.subscriptions = set()
        self.message_handlers = []
        self.reconnect_count = 0
        
    async def connect(self):
        """Connect to WebSocket."""
        self.connected = True
        return True
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self.connected = False
        return True
    
    async def subscribe(self, symbol: str):
        """Subscribe to symbol updates."""
        self.subscriptions.add(symbol)
        return True
    
    async def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol updates."""
        self.subscriptions.discard(symbol)
        return True
    
    def is_connected(self):
        """Check connection status."""
        return self.connected
    
    def get_active_subscriptions(self):
        """Get active subscriptions."""
        return list(self.subscriptions)
    
    async def process_message(self, message: Dict[str, Any]):
        """Process incoming message."""
        for handler in self.message_handlers:
            await handler(message)
        return True
    
    async def ping(self, message: Dict[str, Any]):
        """Send ping and wait for pong."""
        if self.connected:
            return {"type": "pong", "id": message.get("id")}
        return None


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.trades = {}
        self.positions = {}
        self.orders = {}
        self.state = {}
        self.connected = True
        
    async def connect(self):
        """Connect to database."""
        if not self.connected:
            raise ConnectionError("Database connection failed")
        return True
    
    async def disconnect(self):
        """Disconnect from database."""
        self.connected = False
        return True
    
    async def save_trade(self, trade: Dict[str, Any]):
        """Save trade to database."""
        trade_id = trade.get("order_id", f"trade_{len(self.trades)}")
        self.trades[trade_id] = trade
        return trade_id
    
    async def get_trade(self, trade_id: str):
        """Get trade by ID."""
        return self.trades.get(trade_id)
    
    async def save_position(self, position: Dict[str, Any]):
        """Save position to database."""
        symbol = position.get("symbol")
        self.positions[symbol] = position
        return symbol
    
    async def get_position(self, symbol: str):
        """Get position by symbol."""
        return self.positions.get(symbol)
    
    async def save_order(self, order: Dict[str, Any]):
        """Save order to database."""
        order_id = order.get("order_id", f"order_{len(self.orders)}")
        self.orders[order_id] = order
        return order_id
    
    async def get_order(self, order_id: str):
        """Get order by ID."""
        return self.orders.get(order_id)
    
    async def save_state(self, key: str, value: Any):
        """Save state to database."""
        self.state[key] = value
        return True
    
    async def get_state(self, key: str):
        """Get state from database."""
        return self.state.get(key)
    
    async def health_check(self):
        """Check database health."""
        return {
            "status": "healthy" if self.connected else "unhealthy",
            "latency_ms": 2,
            "timestamp": datetime.now()
        }


class MockCache:
    """Mock cache for testing."""
    
    def __init__(self):
        self.data = {}
        self.connected = True
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if not self.connected:
            raise ConnectionError("Cache connection failed")
        self.data[key] = {"value": value, "ttl": ttl, "timestamp": datetime.now()}
        return True
    
    async def get(self, key: str):
        """Get value from cache."""
        if not self.connected:
            return None
        item = self.data.get(key)
        if item:
            return item["value"]
        return None
    
    async def delete(self, key: str):
        """Delete key from cache."""
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    async def clear(self):
        """Clear all cache data."""
        self.data.clear()
        return True
    
    async def health_check(self):
        """Check cache health."""
        return {
            "status": "healthy" if self.connected else "unhealthy",
            "latency_ms": 1,
            "timestamp": datetime.now()
        }


@pytest.fixture
async def mock_exchange_gateway():
    """Create mock exchange gateway."""
    return MockExchangeGateway()


@pytest.fixture
async def mock_websocket_manager():
    """Create mock WebSocket manager."""
    return MockWebSocketManager("wss://mock.exchange.com")


@pytest.fixture
async def mock_database():
    """Create mock database."""
    return MockDatabase()


@pytest.fixture
async def mock_cache():
    """Create mock cache."""
    return MockCache()


@pytest.fixture
async def trading_system(mock_exchange_gateway, mock_database, mock_cache):
    """Create complete trading system with mocks."""
    state_machine = StateMachine()
    risk_engine = RiskEngine(state_machine)
    
    engine = TradingEngine(
        exchange_gateway=mock_exchange_gateway,
        risk_engine=risk_engine,
        state_machine=state_machine
    )
    
    engine.database = mock_database
    engine.cache = mock_cache
    
    await engine.initialize()
    
    yield engine
    
    await engine.shutdown()


@pytest.fixture
def market_data_samples():
    """Sample market data for testing."""
    return {
        "BTC/USDT": {
            "bid": Decimal("50000.00"),
            "ask": Decimal("50001.00"),
            "last": Decimal("50000.50"),
            "volume": Decimal("1000.00"),
            "high": Decimal("51000.00"),
            "low": Decimal("49000.00")
        },
        "ETH/USDT": {
            "bid": Decimal("3000.00"),
            "ask": Decimal("3001.00"),
            "last": Decimal("3000.50"),
            "volume": Decimal("5000.00"),
            "high": Decimal("3100.00"),
            "low": Decimal("2900.00")
        }
    }


@pytest.fixture
def order_samples():
    """Sample orders for testing."""
    return [
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "market",
            "quantity": Decimal("0.1")
        },
        {
            "symbol": "ETH/USDT",
            "side": "sell",
            "type": "limit",
            "quantity": Decimal("1.0"),
            "price": Decimal("3100.00")
        }
    ]


@pytest.fixture
def position_samples():
    """Sample positions for testing."""
    return [
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": Decimal("0.5"),
            "entry_price": Decimal("49500.00"),
            "current_price": Decimal("50000.00"),
            "unrealized_pnl": Decimal("250.00")
        },
        {
            "symbol": "ETH/USDT",
            "side": "sell",
            "quantity": Decimal("2.0"),
            "entry_price": Decimal("3050.00"),
            "current_price": Decimal("3000.00"),
            "unrealized_pnl": Decimal("100.00")
        }
    ]