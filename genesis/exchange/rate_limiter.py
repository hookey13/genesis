"""
Rate limiting module for Binance API requests.

Tracks API weight consumption and implements automatic backoff to stay
within rate limits.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WeightWindow:
    """Tracks weight usage within a time window."""

    timestamp: float
    weight: int


@dataclass
class EndpointWeight:
    """API endpoint weight configuration."""

    endpoint: str
    method: str
    weight: int


class RateLimiter:
    """
    Manages API rate limiting for Binance.

    Binance uses a weight-based system where different endpoints consume
    different amounts of weight. The limit is 1200 weight per minute.
    """

    # Binance API endpoint weights (as of API v3)
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
        max_weight: int = 1200,
        window_seconds: int = 60,
        threshold_percent: float = 80.0,
    ):
        """
        Initialize the rate limiter.

        Args:
            max_weight: Maximum weight allowed per window (default: 1200)
            window_seconds: Time window in seconds (default: 60)
            threshold_percent: Warning threshold as percentage (default: 80%)
        """
        self.max_weight = max_weight
        self.window_seconds = window_seconds
        self.threshold_percent = threshold_percent
        self.threshold_weight = int(max_weight * threshold_percent / 100)

        # Track weight usage with timestamps
        self.weight_history: deque = deque()
        self.current_weight = 0

        # Backoff state
        self.backoff_until: Optional[float] = None
        self.consecutive_threshold_hits = 0

        # Statistics
        self.total_requests = 0
        self.total_weight_consumed = 0
        self.threshold_hits = 0

        logger.info(
            "RateLimiter initialized",
            max_weight=self.max_weight,
            window_seconds=self.window_seconds,
            threshold_weight=self.threshold_weight,
        )

    def _clean_old_weights(self) -> None:
        """Remove weight entries outside the current window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        while self.weight_history and self.weight_history[0].timestamp < cutoff_time:
            old_weight = self.weight_history.popleft()
            self.current_weight -= old_weight.weight

    def _get_endpoint_weight(
        self, method: str, endpoint: str, params: Optional[dict] = None
    ) -> int:
        """
        Get the weight for a specific endpoint.

        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Request parameters (for conditional weights)

        Returns:
            Weight value for the endpoint
        """
        key = (method.upper(), endpoint)
        weight_config = self.ENDPOINT_WEIGHTS.get(key, 1)  # Default to 1 if unknown

        # Handle conditional weights based on parameters
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
                    elif "limit" in condition:
                        return weight
            elif "symbol" in str(weight_config):
                has_symbol = params and "symbol" in params
                return weight_config["symbol" if has_symbol else "no symbol"]

        return weight_config if isinstance(weight_config, int) else 1

    async def check_and_wait(
        self, method: str, endpoint: str, params: Optional[dict] = None
    ) -> None:
        """
        Check rate limit and wait if necessary.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
        """
        # Clean old weight entries
        self._clean_old_weights()

        # Calculate weight for this request
        weight = self._get_endpoint_weight(method, endpoint, params)

        # Check if we're in backoff
        current_time = time.time()
        if self.backoff_until and current_time < self.backoff_until:
            wait_time = self.backoff_until - current_time
            logger.warning(
                "Rate limiter in backoff",
                wait_seconds=wait_time,
                current_weight=self.current_weight,
                max_weight=self.max_weight,
            )
            await asyncio.sleep(wait_time)
            self.backoff_until = None

        # Check if adding this request would exceed threshold
        if self.current_weight + weight >= self.threshold_weight:
            self.threshold_hits += 1
            self.consecutive_threshold_hits += 1

            # Calculate backoff time based on consecutive hits
            backoff_seconds = min(
                2**self.consecutive_threshold_hits, 30
            )  # Max 30 seconds

            logger.warning(
                "Rate limit threshold reached, applying backoff",
                current_weight=self.current_weight,
                threshold_weight=self.threshold_weight,
                max_weight=self.max_weight,
                backoff_seconds=backoff_seconds,
                consecutive_hits=self.consecutive_threshold_hits,
            )

            self.backoff_until = current_time + backoff_seconds
            await asyncio.sleep(backoff_seconds)

            # Clean weights again after backoff
            self._clean_old_weights()
        else:
            # Reset consecutive hits if we're below threshold
            if (
                self.current_weight + weight < self.threshold_weight * 0.5
            ):  # Below 50% of threshold
                self.consecutive_threshold_hits = 0

        # Add weight to history
        self.weight_history.append(WeightWindow(current_time, weight))
        self.current_weight += weight
        self.total_weight_consumed += weight
        self.total_requests += 1

        # Log if we're approaching the limit
        utilization = (self.current_weight / self.max_weight) * 100
        if utilization > 60:
            logger.info(
                "Rate limit utilization",
                current_weight=self.current_weight,
                max_weight=self.max_weight,
                utilization_percent=utilization,
                endpoint=endpoint,
            )

    def get_current_utilization(self) -> float:
        """
        Get current rate limit utilization as a percentage.

        Returns:
            Utilization percentage (0-100)
        """
        self._clean_old_weights()
        return (self.current_weight / self.max_weight) * 100

    def get_remaining_weight(self) -> int:
        """
        Get remaining weight in current window.

        Returns:
            Remaining weight capacity
        """
        self._clean_old_weights()
        return self.max_weight - self.current_weight

    def reset_window(self) -> None:
        """Force reset the rate limit window (for testing)."""
        self.weight_history.clear()
        self.current_weight = 0
        self.backoff_until = None
        self.consecutive_threshold_hits = 0
        logger.info("Rate limit window reset")

    def get_statistics(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with usage statistics
        """
        self._clean_old_weights()

        return {
            "total_requests": self.total_requests,
            "total_weight_consumed": self.total_weight_consumed,
            "current_weight": self.current_weight,
            "current_utilization_percent": self.get_current_utilization(),
            "remaining_weight": self.get_remaining_weight(),
            "threshold_hits": self.threshold_hits,
            "consecutive_threshold_hits": self.consecutive_threshold_hits,
            "in_backoff": self.backoff_until is not None,
            "backoff_remaining": (
                max(0, self.backoff_until - time.time()) if self.backoff_until else 0
            ),
        }
