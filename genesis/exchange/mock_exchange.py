"""
Mock exchange implementation for testing.

Provides a simulated exchange environment for testing without
real funds or API calls.
"""

import asyncio
import random
import time
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

from genesis.exchange.models import (
    AccountBalance,
    MarketTicker,
    OrderBook,
    OrderRequest,
    OrderResponse,
    KlineData
)


logger = structlog.get_logger(__name__)


class MockExchange:
    """
    Mock exchange for testing and development.
    
    Simulates Binance API responses with configurable behavior.
    """
    
    def __init__(self, initial_balance: Dict[str, Decimal] = None):
        """
        Initialize the mock exchange.
        
        Args:
            initial_balance: Initial account balances
        """
        # Default balances
        self.balances = initial_balance or {
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("5.0")
        }
        
        # Orders storage
        self.orders: Dict[str, OrderResponse] = {}
        self.order_counter = 0
        
        # Market data
        self.market_prices = {
            "BTC/USDT": Decimal("50000"),
            "ETH/USDT": Decimal("3000"),
            "BNB/USDT": Decimal("400")
        }
        
        # Configuration
        self.latency_ms = 50  # Simulated network latency
        self.failure_rate = 0.0  # Probability of request failure
        self.partial_fill_rate = 0.2  # Probability of partial fills
        
        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        
        logger.info(
            "MockExchange initialized",
            initial_balances=self.balances,
            market_prices=self.market_prices
        )
    
    async def _simulate_latency(self) -> None:
        """Simulate network latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
    
    def _should_fail(self) -> bool:
        """Determine if request should fail based on failure rate."""
        return random.random() < self.failure_rate
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self.order_counter += 1
        return f"MOCK_{self.order_counter:06d}"
    
    async def fetch_balance(self) -> Dict[str, AccountBalance]:
        """
        Fetch account balances.
        
        Returns:
            Dictionary of account balances
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to fetch balance")
        
        result = {}
        for asset, balance in self.balances.items():
            result[asset] = AccountBalance(
                asset=asset,
                free=balance,
                locked=Decimal("0"),
                total=balance
            )
        
        return result
    
    async def create_order(self, request: OrderRequest) -> OrderResponse:
        """
        Create a new order.
        
        Args:
            request: Order request
            
        Returns:
            Order response
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to create order")
        
        # Generate order ID
        order_id = self._generate_order_id()
        exchange_order_id = f"EX_{uuid.uuid4().hex[:8]}"
        
        # Determine order status
        if request.type == "market":
            status = "filled"
            filled_quantity = request.quantity
        else:
            # Simulate partial fills for limit orders
            if random.random() < self.partial_fill_rate:
                status = "partially_filled"
                filled_quantity = request.quantity * Decimal(str(random.uniform(0.1, 0.9)))
            else:
                status = "open"
                filled_quantity = Decimal("0")
        
        # Create order response
        from datetime import datetime
        response = OrderResponse(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            type=request.type,
            status=status,
            price=request.price or self.market_prices.get(request.symbol, Decimal("50000")),
            quantity=request.quantity,
            filled_quantity=filled_quantity,
            created_at=datetime.now(),
            updated_at=None
        )
        
        # Store order
        self.orders[order_id] = response
        
        # Update balance for market orders
        if status == "filled" and request.type == "market":
            self._update_balance_for_order(request, response)
        
        logger.info(
            "Mock order created",
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            status=status
        )
        
        return response
    
    def _update_balance_for_order(self, request: OrderRequest, response: OrderResponse) -> None:
        """Update balances based on filled order."""
        base, quote = request.symbol.split("/")
        
        if request.side == "buy":
            # Buying base currency with quote currency
            cost = response.filled_quantity * response.price
            self.balances[quote] -= cost
            self.balances[base] = self.balances.get(base, Decimal("0")) + response.filled_quantity
        else:
            # Selling base currency for quote currency
            revenue = response.filled_quantity * response.price
            self.balances[base] -= response.filled_quantity
            self.balances[quote] = self.balances.get(quote, Decimal("0")) + revenue
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            True if cancellation successful
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to cancel order")
        
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in ["open", "partially_filled"]:
                order.status = "canceled"
                logger.info("Mock order cancelled", order_id=order_id)
                return True
        
        return False
    
    async def fetch_order(self, order_id: str, symbol: str) -> OrderResponse:
        """
        Fetch order status.
        
        Args:
            order_id: Order ID
            symbol: Trading pair
            
        Returns:
            Order response
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to fetch order")
        
        if order_id in self.orders:
            return self.orders[order_id]
        
        # Return a default filled order if not found
        from datetime import datetime
        return OrderResponse(
            order_id=order_id,
            client_order_id=None,
            symbol=symbol,
            side="buy",
            type="limit",
            status="filled",
            price=self.market_prices.get(symbol, Decimal("50000")),
            quantity=Decimal("0.001"),
            filled_quantity=Decimal("0.001"),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Fetch order book.
        
        Args:
            symbol: Trading pair
            limit: Number of levels
            
        Returns:
            Order book data
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to fetch order book")
        
        # Generate mock order book around market price
        base_price = self.market_prices.get(symbol, Decimal("50000"))
        
        bids = []
        asks = []
        
        for i in range(limit):
            # Bids below market price
            bid_price = base_price * (Decimal("1") - Decimal(str(0.0001 * (i + 1))))
            bid_quantity = Decimal(str(random.uniform(0.1, 5.0)))
            bids.append((bid_price, bid_quantity))
            
            # Asks above market price
            ask_price = base_price * (Decimal("1") + Decimal(str(0.0001 * (i + 1))))
            ask_quantity = Decimal(str(random.uniform(0.1, 5.0)))
            asks.append((ask_price, ask_quantity))
        
        from datetime import datetime
        return OrderBook(
            symbol=symbol,
            bids=[[price, qty] for price, qty in bids],
            asks=[[price, qty] for price, qty in asks],
            timestamp=datetime.now()
        )
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch kline data.
        
        Args:
            symbol: Trading pair
            interval: Time interval
            limit: Number of klines
            
        Returns:
            List of kline data
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to fetch klines")
        
        base_price = self.market_prices.get(symbol, Decimal("50000"))
        klines = []
        
        current_time = int(time.time() * 1000)
        interval_ms = 60000  # 1 minute in milliseconds
        
        for i in range(limit):
            # Generate random OHLC data
            variation = Decimal(str(random.uniform(0.98, 1.02)))
            open_price = base_price * variation
            high_price = open_price * Decimal(str(random.uniform(1.0, 1.01)))
            low_price = open_price * Decimal(str(random.uniform(0.99, 1.0)))
            close_price = open_price * Decimal(str(random.uniform(0.995, 1.005)))
            
            klines.append({
                "timestamp": current_time - (i * interval_ms),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": Decimal(str(random.uniform(10, 1000)))
            })
        
        return klines
    
    async def fetch_ticker(self, symbol: str) -> MarketTicker:
        """
        Fetch ticker data.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Market ticker data
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to fetch ticker")
        
        base_price = self.market_prices.get(symbol, Decimal("50000"))
        
        return MarketTicker(
            symbol=symbol,
            last_price=base_price,
            bid_price=base_price * Decimal("0.9999"),
            ask_price=base_price * Decimal("1.0001"),
            volume_24h=Decimal(str(random.uniform(1000, 10000))),
            quote_volume_24h=base_price * Decimal(str(random.uniform(1000, 10000))),
            price_change_percent=Decimal(str(random.uniform(-5, 5))),
            high_24h=base_price * Decimal("1.05"),
            low_24h=base_price * Decimal("0.95")
        )
    
    async def fetch_time(self) -> int:
        """
        Fetch server time.
        
        Returns:
            Server timestamp in milliseconds
        """
        await self._simulate_latency()
        self.total_requests += 1
        
        if self._should_fail():
            self.failed_requests += 1
            raise Exception("Mock API error: Failed to fetch time")
        
        return int(time.time() * 1000)
    
    def set_failure_rate(self, rate: float) -> None:
        """
        Set the failure rate for simulating errors.
        
        Args:
            rate: Failure probability (0.0 to 1.0)
        """
        self.failure_rate = max(0.0, min(1.0, rate))
        logger.info(f"Mock exchange failure rate set to {self.failure_rate}")
    
    def set_latency(self, latency_ms: int) -> None:
        """
        Set simulated network latency.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.latency_ms = max(0, latency_ms)
        logger.info(f"Mock exchange latency set to {self.latency_ms}ms")
    
    def update_market_price(self, symbol: str, price: Decimal) -> None:
        """
        Update market price for a symbol.
        
        Args:
            symbol: Trading pair
            price: New market price
        """
        self.market_prices[symbol] = price
        logger.info(f"Market price updated: {symbol} = {price}")
    
    def get_statistics(self) -> Dict:
        """Get mock exchange statistics."""
        success_rate = (
            ((self.total_requests - self.failed_requests) / self.total_requests * 100)
            if self.total_requests > 0 else 100
        )
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "active_orders": len([o for o in self.orders.values() if o.status == "open"]),
            "total_orders": len(self.orders),
            "balances": self.balances,
            "market_prices": self.market_prices,
            "latency_ms": self.latency_ms,
            "failure_rate": self.failure_rate
        }
    
    def reset(self) -> None:
        """Reset the mock exchange to initial state."""
        self.orders.clear()
        self.order_counter = 0
        self.total_requests = 0
        self.failed_requests = 0
        
        # Reset balances to default
        self.balances = {
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("5.0")
        }
        
        logger.info("Mock exchange reset to initial state")