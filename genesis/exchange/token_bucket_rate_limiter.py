"""
Token bucket rate limiting module for Binance API requests.

Implements a token bucket algorithm with configurable capacity and refill rate
for more efficient burst handling and smoother rate limiting.
"""

import asyncio
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    The bucket starts with a certain capacity and tokens are refilled at a
    constant rate. Requests consume tokens, and if not enough tokens are
    available, the request must wait.
    """

    capacity: int  # Maximum number of tokens
    refill_rate: float  # Tokens added per second
    tokens: float  # Current number of tokens
    last_refill: float  # Timestamp of last refill

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time

    def consume(self, amount: int) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            amount: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if not enough available
        """
        self.refill()

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def time_until_tokens_available(self, amount: int) -> float:
        """
        Calculate time to wait until enough tokens are available.
        
        Args:
            amount: Number of tokens needed
            
        Returns:
            Seconds to wait (0 if tokens available now)
        """
        self.refill()

        if self.tokens >= amount:
            return 0

        tokens_needed = amount - self.tokens
        return tokens_needed / self.refill_rate


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for Binance API.
    
    Uses token bucket algorithm for smoother rate limiting with burst capacity.
    Default configuration: 1200 tokens capacity, 20 tokens/second refill rate
    (equivalent to 1200 tokens per minute).
    """

    # Binance API endpoint weights (same as before)
    ENDPOINT_WEIGHTS = {
        # Account endpoints
        ("GET", "/api/v3/account"): 10,
        ("GET", "/api/v3/myTrades"): 10,
        # Market data endpoints
        ("GET", "/api/v3/depth"): {
            "limit <= 100": 1,
            "limit <= 500": 5,
            "limit <= 1000": 10,
            "limit > 1000": 50,
        },
        ("GET", "/api/v3/trades"): 1,
        ("GET", "/api/v3/historicalTrades"): 5,
        ("GET", "/api/v3/aggTrades"): 1,
        ("GET", "/api/v3/klines"): 1,
        ("GET", "/api/v3/ticker/24hr"): {"symbol": 1, "no symbol": 40},
        ("GET", "/api/v3/ticker/price"): {"symbol": 1, "no symbol": 2},
        ("GET", "/api/v3/ticker/bookTicker"): {"symbol": 1, "no symbol": 2},
        # Trading endpoints
        ("POST", "/api/v3/order"): 1,
        ("DELETE", "/api/v3/order"): 1,
        ("GET", "/api/v3/order"): 2,
        ("DELETE", "/api/v3/openOrders"): 1,
        ("GET", "/api/v3/openOrders"): {"symbol": 3, "no symbol": 40},
        ("GET", "/api/v3/allOrders"): 10,
        # System endpoints
        ("GET", "/api/v3/ping"): 1,
        ("GET", "/api/v3/time"): 1,
        ("GET", "/api/v3/exchangeInfo"): 10,
    }

    def __init__(
        self,
        bucket_capacity: int = 1200,
        refill_rate: float = 20.0,  # 20 tokens per second = 1200 per minute
        burst_reserve: int = 100,  # Reserve tokens for burst operations
    ):
        """
        Initialize the token bucket rate limiter.
        
        Args:
            bucket_capacity: Maximum tokens in bucket (default: 1200)
            refill_rate: Tokens added per second (default: 20)
            burst_reserve: Tokens reserved for burst operations (default: 100)
        """
        self.bucket = TokenBucket(
            capacity=bucket_capacity,
            refill_rate=refill_rate,
            tokens=bucket_capacity,  # Start with full bucket
            last_refill=time.time()
        )

        self.burst_reserve = burst_reserve
        self.effective_capacity = bucket_capacity - burst_reserve

        # Order-specific token buckets
        self.order_bucket_10s = TokenBucket(
            capacity=50,
            refill_rate=5.0,  # 50 orders per 10 seconds
            tokens=50,
            last_refill=time.time()
        )

        self.order_bucket_daily = TokenBucket(
            capacity=160000,
            refill_rate=1.85,  # ~160k per day
            tokens=160000,
            last_refill=time.time()
        )

        # Statistics
        self.total_requests = 0
        self.total_weight_consumed = 0
        self.total_orders_placed = 0
        self.burst_activations = 0
        self.wait_times = []

        # Dynamic weight tracking from response headers
        self.last_response_weight = {}
        self.weight_adjustments = {}

        logger.info(
            "TokenBucketRateLimiter initialized",
            capacity=bucket_capacity,
            refill_rate=refill_rate,
            burst_reserve=burst_reserve,
        )

    def _get_endpoint_weight(
        self, method: str, endpoint: str, params: dict | None = None
    ) -> int:
        """
        Get the weight for a specific endpoint with dynamic adjustment.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Request parameters
            
        Returns:
            Weight value for the endpoint
        """
        key = (method.upper(), endpoint)

        # Check if we have a dynamic weight adjustment from response headers
        if key in self.weight_adjustments:
            return self.weight_adjustments[key]

        weight_config = self.ENDPOINT_WEIGHTS.get(key, 1)

        # Handle conditional weights
        if isinstance(weight_config, dict):
            if "limit" in str(weight_config):
                limit = params.get("limit", 100) if params else 100
                for condition, weight in weight_config.items():
                    if "limit <=" in condition:
                        threshold = int(condition.split("<=")[1].strip())
                        if limit <= threshold:
                            return weight
                    elif "limit >" in condition:
                        threshold = int(condition.split(">")[1].strip())
                        if limit > threshold:
                            return weight
            elif "symbol" in str(weight_config):
                has_symbol = params and "symbol" in params
                return weight_config["symbol" if has_symbol else "no symbol"]

        return weight_config if isinstance(weight_config, int) else 1

    def update_weight_from_headers(self, headers: dict) -> None:
        """
        Update weight consumption based on API response headers.
        
        Binance returns actual weight consumed in response headers:
        - X-MBX-USED-WEIGHT: Weight used by this request
        - X-MBX-USED-WEIGHT-1M: Total weight used in current minute
        
        Args:
            headers: Response headers from Binance API
        """
        if "X-MBX-USED-WEIGHT" in headers:
            try:
                used_weight = int(headers["X-MBX-USED-WEIGHT"])
                # Adjust our token count based on actual consumption
                actual_vs_expected = used_weight - self.last_request_weight
                if actual_vs_expected > 0:
                    # We underestimated, consume additional tokens
                    self.bucket.consume(actual_vs_expected)
                    logger.debug(
                        "Weight adjustment from headers",
                        expected=self.last_request_weight,
                        actual=used_weight,
                        adjustment=actual_vs_expected,
                    )
            except (ValueError, KeyError):
                pass

        if "X-MBX-USED-WEIGHT-1M" in headers:
            try:
                total_used = int(headers["X-MBX-USED-WEIGHT-1M"])
                # Sync our bucket state with server state
                tokens_used = total_used
                expected_tokens = self.bucket.capacity - self.bucket.tokens

                if abs(tokens_used - expected_tokens) > 10:
                    # Significant discrepancy, adjust our bucket
                    self.bucket.tokens = max(0, self.bucket.capacity - tokens_used)
                    logger.warning(
                        "Token bucket synced with server",
                        server_used=tokens_used,
                        local_used=expected_tokens,
                        new_tokens=self.bucket.tokens,
                    )
            except (ValueError, KeyError):
                pass

    async def check_and_wait(
        self, method: str, endpoint: str, params: dict | None = None
    ) -> None:
        """
        Check rate limit and wait if necessary using token bucket.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
        """
        # Calculate weight for this request
        weight = self._get_endpoint_weight(method, endpoint, params)
        self.last_request_weight = weight

        # Check if this is an order placement
        is_order = method.upper() == "POST" and endpoint == "/api/v3/order"

        # Handle order-specific limits
        if is_order:
            # Check 10-second order limit
            wait_time_10s = self.order_bucket_10s.time_until_tokens_available(1)
            if wait_time_10s > 0:
                logger.info(
                    "Waiting for order rate limit (10s window)",
                    wait_seconds=wait_time_10s,
                    current_tokens=self.order_bucket_10s.tokens,
                )
                await asyncio.sleep(wait_time_10s)

            # Check daily order limit
            wait_time_daily = self.order_bucket_daily.time_until_tokens_available(1)
            if wait_time_daily > 3600:  # More than 1 hour wait
                raise Exception(
                    f"Daily order limit exhausted. Wait time: {wait_time_daily/3600:.1f} hours"
                )
            elif wait_time_daily > 0:
                logger.info(
                    "Waiting for order rate limit (daily)",
                    wait_seconds=wait_time_daily,
                    current_tokens=self.order_bucket_daily.tokens,
                )
                await asyncio.sleep(wait_time_daily)

            # Consume order tokens
            self.order_bucket_10s.consume(1)
            self.order_bucket_daily.consume(1)
            self.total_orders_placed += 1

        # Check main weight limit
        wait_time = self.bucket.time_until_tokens_available(weight)

        # Check if we need to use burst capacity
        if wait_time > 0 and weight <= self.burst_reserve:
            # Try to use burst capacity for important operations
            if self.bucket.tokens + self.burst_reserve >= weight:
                logger.info(
                    "Using burst capacity for request",
                    weight=weight,
                    tokens_available=self.bucket.tokens,
                    burst_reserve=self.burst_reserve,
                )
                self.burst_activations += 1
                # Allow the request to proceed even with low tokens
                wait_time = 0

        if wait_time > 0:
            logger.info(
                "Token bucket rate limit: waiting",
                wait_seconds=wait_time,
                weight_needed=weight,
                tokens_available=self.bucket.tokens,
                endpoint=endpoint,
            )
            self.wait_times.append(wait_time)
            await asyncio.sleep(wait_time)

        # Consume tokens
        success = self.bucket.consume(weight)
        if not success:
            # This shouldn't happen if wait_time calculation is correct
            logger.error(
                "Failed to consume tokens after waiting",
                weight=weight,
                tokens=self.bucket.tokens,
            )
            # Emergency wait
            await asyncio.sleep(1)
            self.bucket.refill()
            self.bucket.consume(weight)

        # Update statistics
        self.total_requests += 1
        self.total_weight_consumed += weight

        # Log utilization if high
        utilization = ((self.bucket.capacity - self.bucket.tokens) / self.bucket.capacity) * 100
        if utilization > 70:
            logger.info(
                "High token bucket utilization",
                utilization_percent=f"{utilization:.1f}%",
                tokens_remaining=self.bucket.tokens,
                capacity=self.bucket.capacity,
            )

    def get_current_utilization(self) -> float:
        """
        Get current bucket utilization as a percentage.
        
        Returns:
            Utilization percentage (0-100)
        """
        self.bucket.refill()
        used = self.bucket.capacity - self.bucket.tokens
        return (used / self.bucket.capacity) * 100

    def get_remaining_capacity(self) -> int:
        """
        Get remaining token capacity.
        
        Returns:
            Number of tokens available
        """
        self.bucket.refill()
        return int(self.bucket.tokens)

    def reset(self) -> None:
        """Reset all buckets to full capacity (for testing)."""
        self.bucket.tokens = self.bucket.capacity
        self.bucket.last_refill = time.time()

        self.order_bucket_10s.tokens = self.order_bucket_10s.capacity
        self.order_bucket_10s.last_refill = time.time()

        self.order_bucket_daily.tokens = self.order_bucket_daily.capacity
        self.order_bucket_daily.last_refill = time.time()

        logger.info("All token buckets reset to full capacity")

    def get_statistics(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        self.bucket.refill()
        self.order_bucket_10s.refill()
        self.order_bucket_daily.refill()

        avg_wait = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0

        return {
            "total_requests": self.total_requests,
            "total_weight_consumed": self.total_weight_consumed,
            "burst_activations": self.burst_activations,
            "average_wait_time": avg_wait,
            "main_bucket": {
                "tokens_available": int(self.bucket.tokens),
                "capacity": self.bucket.capacity,
                "utilization_percent": self.get_current_utilization(),
                "refill_rate": self.bucket.refill_rate,
            },
            "order_buckets": {
                "10s": {
                    "tokens_available": int(self.order_bucket_10s.tokens),
                    "capacity": self.order_bucket_10s.capacity,
                    "orders_remaining": int(self.order_bucket_10s.tokens),
                },
                "daily": {
                    "tokens_available": int(self.order_bucket_daily.tokens),
                    "capacity": self.order_bucket_daily.capacity,
                    "orders_remaining": int(self.order_bucket_daily.tokens),
                },
                "total_orders_placed": self.total_orders_placed,
            },
        }


# Maintain backward compatibility
RateLimiter = TokenBucketRateLimiter
