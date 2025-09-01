"""Rate limiter implementation with token bucket and sliding window algorithms."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


class Priority(Enum):
    """Request priority levels."""
    CRITICAL = 1  # Emergency operations (stop loss, emergency close)
    HIGH = 2      # Important operations (order placement)
    NORMAL = 3    # Regular operations (market data)
    LOW = 4       # Background operations (analytics)


@dataclass
class RateLimitConfig:
    """Rate limiter configuration."""
    requests_per_second: int
    burst_size: int
    window_size_seconds: int = 60
    critical_reserve_percent: Decimal = Decimal("0.2")  # Reserve 20% for critical
    
    def __post_init__(self):
        """Validate configuration."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size < self.requests_per_second:
            raise ValueError("burst_size must be >= requests_per_second")
        if self.window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")
        if not (Decimal("0") <= self.critical_reserve_percent <= Decimal("1")):
            raise ValueError("critical_reserve_percent must be between 0 and 1")


@dataclass
class TokenBucket:
    """Token bucket implementation for rate limiting."""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    def __post_init__(self):
        """Initialize token bucket."""
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
    
    async def acquire(self, tokens: int = 1, priority: Priority = Priority.NORMAL) -> bool:
        """Acquire tokens from bucket."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            # For critical priority, allow overdraft up to 10% of capacity
            if priority == Priority.CRITICAL and self.tokens >= -self.capacity * 0.1:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens to become available."""
        async with self.lock:
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            return wait_time


@dataclass
class SlidingWindowLimiter:
    """Sliding window rate limiter for burst protection."""
    window_size_seconds: int
    max_requests: int
    requests: deque = field(default_factory=deque, init=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    
    async def check_and_add(self, priority: Priority = Priority.NORMAL) -> bool:
        """Check if request is allowed and add to window."""
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_size_seconds
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            # Allow critical requests to exceed by 10%
            if priority == Priority.CRITICAL and len(self.requests) < self.max_requests * 1.1:
                self.requests.append(now)
                return True
            
            return False
    
    async def get_window_usage(self) -> tuple[int, int]:
        """Get current window usage (used, total)."""
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_size_seconds
            
            # Clean old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            return len(self.requests), self.max_requests


class RateLimiter:
    """Advanced rate limiter with multiple algorithms and priority support."""
    
    def __init__(self, config: RateLimitConfig, redis_client: Optional[Any] = None):
        """Initialize rate limiter."""
        self.config = config
        self.redis_client = redis_client
        
        # Token bucket for steady-state rate limiting
        self.token_bucket = TokenBucket(
            capacity=config.burst_size,
            refill_rate=config.requests_per_second
        )
        
        # Sliding window for burst protection
        self.sliding_window = SlidingWindowLimiter(
            window_size_seconds=config.window_size_seconds,
            max_requests=config.requests_per_second * config.window_size_seconds
        )
        
        # Priority queues for pending requests
        self.priority_queues: Dict[Priority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in Priority
        }
        
        # Metrics
        self.metrics = {
            "requests_allowed": 0,
            "requests_rejected": 0,
            "requests_queued": 0,
            "critical_overrides": 0
        }
        
        # Request coalescing
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.coalesce_lock = asyncio.Lock()
        
        logger.info("Rate limiter initialized", config=config)
    
    async def acquire(self, 
                     tokens: int = 1, 
                     priority: Priority = Priority.NORMAL,
                     wait: bool = True) -> bool:
        """Acquire permission to make a request."""
        # Check both token bucket and sliding window
        token_allowed = await self.token_bucket.acquire(tokens, priority)
        window_allowed = await self.sliding_window.check_and_add(priority)
        
        if token_allowed and window_allowed:
            self.metrics["requests_allowed"] += 1
            if priority == Priority.CRITICAL and not (token_allowed and window_allowed):
                self.metrics["critical_overrides"] += 1
            return True
        
        if not wait:
            self.metrics["requests_rejected"] += 1
            return False
        
        # Queue the request if waiting is enabled
        self.metrics["requests_queued"] += 1
        future = asyncio.Future()
        await self.priority_queues[priority].put((tokens, future))
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self.metrics["requests_rejected"] += 1
            return False
    
    async def release(self, tokens: int = 1):
        """Release tokens back to the bucket (for cancelled operations)."""
        async with self.token_bucket.lock:
            self.token_bucket.tokens = min(
                self.token_bucket.capacity,
                self.token_bucket.tokens + tokens
            )
    
    async def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens."""
        return await self.token_bucket.wait_for_tokens(tokens)
    
    async def coalesce_request(self, key: str, coroutine) -> Any:
        """Coalesce similar requests to reduce API calls."""
        async with self.coalesce_lock:
            # Check if request is already pending
            if key in self.pending_requests:
                logger.debug("Coalescing request", key=key)
                return await self.pending_requests[key]
            
            # Create new future for this request
            future = asyncio.create_task(coroutine)
            self.pending_requests[key] = future
        
        try:
            result = await future
            return result
        finally:
            # Clean up after completion
            async with self.coalesce_lock:
                self.pending_requests.pop(key, None)
    
    async def process_queues(self):
        """Background task to process priority queues."""
        while True:
            try:
                # Process queues in priority order
                for priority in Priority:
                    if not self.priority_queues[priority].empty():
                        tokens, future = await self.priority_queues[priority].get()
                        
                        # Try to acquire tokens
                        if await self.acquire(tokens, priority, wait=False):
                            future.set_result(True)
                        else:
                            # Re-queue if not available
                            await self.priority_queues[priority].put((tokens, future))
                
                await asyncio.sleep(0.1)  # Small delay between checks
                
            except Exception as e:
                logger.error("Error processing rate limit queues", error=str(e))
                await asyncio.sleep(1.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            **self.metrics,
            "token_bucket_tokens": self.token_bucket.tokens,
            "token_bucket_capacity": self.token_bucket.capacity,
            "sliding_window_usage": asyncio.create_task(
                self.sliding_window.get_window_usage()
            )
        }
    
    async def update_from_headers(self, headers: Dict[str, str]):
        """Update rate limits based on response headers (Binance specific)."""
        # Parse Binance rate limit headers
        used_weight = headers.get("X-MBX-USED-WEIGHT")
        used_weight_1m = headers.get("X-MBX-USED-WEIGHT-1M") 
        order_count_1m = headers.get("X-MBX-ORDER-COUNT-1M")
        order_count_10s = headers.get("X-MBX-ORDER-COUNT-10S")
        
        if used_weight_1m:
            # Adjust token bucket based on actual usage
            weight_limit = 1200  # Binance default
            usage_percent = int(used_weight_1m) / weight_limit
            
            if usage_percent > 0.8:
                # Slow down if approaching limit
                logger.warning("Approaching rate limit", usage_percent=usage_percent)
                self.token_bucket.refill_rate *= 0.5
            elif usage_percent < 0.5:
                # Speed up if well under limit
                self.token_bucket.refill_rate = min(
                    self.config.requests_per_second,
                    self.token_bucket.refill_rate * 1.2
                )
        
        logger.debug("Rate limit headers", 
                    used_weight=used_weight,
                    used_weight_1m=used_weight_1m,
                    order_count_1m=order_count_1m,
                    order_count_10s=order_count_10s)


class DistributedRateLimiter(RateLimiter):
    """Rate limiter with Redis support for distributed rate limiting."""
    
    def __init__(self, config: RateLimitConfig, redis_client: Any):
        """Initialize distributed rate limiter."""
        super().__init__(config, redis_client)
        self.instance_id = f"rl_{id(self)}"
        
    async def acquire_distributed(self, 
                                 key: str,
                                 tokens: int = 1,
                                 priority: Priority = Priority.NORMAL) -> bool:
        """Acquire tokens using distributed counter in Redis."""
        if not self.redis_client:
            # Fallback to local rate limiting
            return await self.acquire(tokens, priority)
        
        try:
            # Lua script for atomic rate limit check
            lua_script = """
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local current = redis.call('INCR', key)
            
            if current == 1 then
                redis.call('EXPIRE', key, window)
            end
            
            if current <= limit then
                return 1
            else
                redis.call('DECR', key)
                return 0
            end
            """
            
            result = await self.redis_client.eval(
                lua_script,
                keys=[f"ratelimit:{key}"],
                args=[self.config.requests_per_second * self.config.window_size_seconds,
                     self.config.window_size_seconds]
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error("Redis rate limit failed, using local", error=str(e))
            return await self.acquire(tokens, priority)