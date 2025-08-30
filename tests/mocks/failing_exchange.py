"""
Mock exchange with configurable failure modes for testing.

Simulates various exchange API failure scenarios.
"""

import asyncio
import random
import time
from typing import Dict, Any, Optional
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class FailureMode(Enum):
    """Exchange failure modes."""
    
    NONE = "none"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    ORDER_REJECTION = "order_rejection"
    MAINTENANCE = "maintenance"
    CONNECTION_ERROR = "connection_error"


class FailingExchange:
    """Mock exchange with failure injection capabilities."""
    
    def __init__(self):
        self.failure_mode = FailureMode.NONE
        self.failure_rate = 0.0
        self.request_count = 0
        self.rate_limit_threshold = 10
        
    def set_failure_mode(self, mode: FailureMode, rate: float = 1.0):
        """Set the failure mode and rate."""
        self.failure_mode = mode
        self.failure_rate = rate
        logger.info(f"Failure mode set to {mode.value} with {rate*100}% rate")
        
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker with potential failures."""
        self.request_count += 1
        
        if await self._should_fail():
            await self._inject_failure()
        
        return {
            "symbol": symbol,
            "bid": 50000.0,
            "ask": 50010.0,
            "last": 50005.0,
            "timestamp": int(time.time() * 1000)
        }
        
    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """Create order with potential failures."""
        self.request_count += 1
        
        if await self._should_fail():
            await self._inject_failure()
        
        if self.failure_mode == FailureMode.ORDER_REJECTION:
            raise Exception("Order rejected: Insufficient balance")
        
        return {
            "id": f"order_{int(time.time() * 1000000)}",
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "status": "open",
            "timestamp": int(time.time() * 1000)
        }
        
    async def _should_fail(self) -> bool:
        """Determine if request should fail."""
        if self.failure_mode == FailureMode.NONE:
            return False
        return random.random() < self.failure_rate
        
    async def _inject_failure(self):
        """Inject the configured failure."""
        if self.failure_mode == FailureMode.RATE_LIMIT:
            if self.request_count > self.rate_limit_threshold:
                raise Exception("Rate limit exceeded")
                
        elif self.failure_mode == FailureMode.TIMEOUT:
            await asyncio.sleep(30)  # Simulate timeout
            raise asyncio.TimeoutError("Request timeout")
            
        elif self.failure_mode == FailureMode.INVALID_RESPONSE:
            raise ValueError("Invalid JSON response")
            
        elif self.failure_mode == FailureMode.MAINTENANCE:
            raise Exception("Exchange under maintenance")
            
        elif self.failure_mode == FailureMode.CONNECTION_ERROR:
            raise ConnectionError("Connection refused")


async def test_rate_limit_handling():
    """Test rate limit handling."""
    exchange = FailingExchange()
    exchange.set_failure_mode(FailureMode.RATE_LIMIT, 0.5)
    
    errors = 0
    for i in range(20):
        try:
            await exchange.fetch_ticker("BTC/USDT")
        except Exception as e:
            logger.warning(f"Rate limit error: {e}")
            errors += 1
            await asyncio.sleep(1)  # Back off
    
    logger.info(f"Rate limit test completed with {errors} errors")
    return errors < 15  # Should handle some requests


async def test_order_rejection_handling():
    """Test order rejection handling."""
    exchange = FailingExchange()
    exchange.set_failure_mode(FailureMode.ORDER_REJECTION, 1.0)
    
    try:
        await exchange.create_order("BTC/USDT", "limit", "buy", 0.001, 50000)
        return False  # Should have failed
    except Exception as e:
        logger.info(f"Order rejected as expected: {e}")
        return True