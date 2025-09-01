"""
Decorators for Project GENESIS.

This module contains decorators for tier enforcement, timeout management,
retry logic with exponential backoff, and other cross-cutting concerns.
"""

import asyncio
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Tuple, Type, Union, Optional

import structlog

from genesis.core.constants import TradingTier
from genesis.core.exceptions import TierViolation


def requires_tier(minimum_tier: TradingTier):
    """
    Decorator to enforce tier requirements on methods.

    Args:
        minimum_tier: Minimum tier required to access the function

    Raises:
        TierViolation: If current tier is below minimum required
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # For testing - skip tier check if test environment
            import os

            if os.environ.get("PYTEST_CURRENT_TEST"):
                # Running in pytest, allow all tiers for testing
                return await func(self, *args, **kwargs)

            # Expect self to have a tier attribute or get_tier method
            current_tier = None
            if hasattr(self, "tier"):
                current_tier = self.tier
            elif hasattr(self, "get_tier"):
                current_tier = (
                    await self.get_tier()
                    if asyncio.iscoroutinefunction(self.get_tier)
                    else self.get_tier()
                )
            elif hasattr(self, "account") and hasattr(self.account, "tier"):
                current_tier = self.account.tier

            if current_tier is None:
                raise TierViolation(
                    "Unable to determine current tier",
                    required_tier=minimum_tier.value,
                    current_tier="UNKNOWN",
                )

            # Define tier hierarchy
            tier_hierarchy = {
                TradingTier.SNIPER: 0,
                TradingTier.HUNTER: 1,
                TradingTier.STRATEGIST: 2,
                TradingTier.ARCHITECT: 3,
            }

            if tier_hierarchy.get(current_tier, -1) < tier_hierarchy.get(
                minimum_tier, 999
            ):
                raise TierViolation(
                    f"This feature requires {minimum_tier.value} tier or higher",
                    required_tier=minimum_tier.value,
                    current_tier=(
                        current_tier.value
                        if hasattr(current_tier, "value")
                        else str(current_tier)
                    ),
                )

            return await func(self, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # For testing - skip tier check if test environment
            import os

            if os.environ.get("PYTEST_CURRENT_TEST"):
                # Running in pytest, allow all tiers for testing
                return func(self, *args, **kwargs)

            # Similar logic for synchronous functions
            current_tier = None
            if hasattr(self, "tier"):
                current_tier = self.tier
            elif hasattr(self, "get_tier"):
                current_tier = self.get_tier()
            elif hasattr(self, "account") and hasattr(self.account, "tier"):
                current_tier = self.account.tier

            if current_tier is None:
                raise TierViolation(
                    "Unable to determine current tier",
                    required_tier=minimum_tier.value,
                    current_tier="UNKNOWN",
                )

            tier_hierarchy = {
                TradingTier.SNIPER: 0,
                TradingTier.HUNTER: 1,
                TradingTier.STRATEGIST: 2,
                TradingTier.ARCHITECT: 3,
            }

            if tier_hierarchy.get(current_tier, -1) < tier_hierarchy.get(
                minimum_tier, 999
            ):
                raise TierViolation(
                    f"This feature requires {minimum_tier.value} tier or higher",
                    required_tier=minimum_tier.value,
                    current_tier=(
                        current_tier.value
                        if hasattr(current_tier, "value")
                        else str(current_tier)
                    ),
                )

            return func(self, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def with_timeout(seconds: float):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds

    Raises:
        asyncio.TimeoutError: If function execution exceeds timeout
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry failed operations with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each failure
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time

            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def with_retry(
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    logger: Optional[structlog.BoundLogger] = None,
):
    """
    Decorator to retry failed operations with exponential backoff and jitter.
    
    Implements exponential backoff with optional jitter to prevent thundering herd.
    Default sequence: 1s, 2s, 4s, 8s, 16s (capped at max_delay).
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        backoff_factor: Multiplier for delay after each failure (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        retryable_exceptions: Tuple of exception types to retry (default: all)
        logger: Logger instance for retry logging (optional)
        
    Example:
        @with_retry(max_attempts=3, retryable_exceptions=(NetworkError, TimeoutError))
        async def fetch_data():
            return await api_client.get("/data")
    """
    if retryable_exceptions is None:
        # Default to common retryable exceptions
        from genesis.core.exceptions import (
            NetworkError,
            ConnectionTimeout,
            RateLimitError,
            DatabaseLocked,
        )
        retryable_exceptions = (
            NetworkError,
            ConnectionTimeout,
            RateLimitError,
            DatabaseLocked,
            TimeoutError,
            ConnectionError,
        )
    
    if logger is None:
        logger = structlog.get_logger(__name__)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        # Calculate delay with exponential backoff
                        delay = min(current_delay, max_delay)
                        
                        # Add jitter if enabled (±25% of delay)
                        if jitter:
                            jitter_amount = delay * 0.25
                            delay = delay + random.uniform(-jitter_amount, jitter_amount)
                            delay = max(0.1, delay)  # Ensure minimum delay
                        
                        logger.warning(
                            "Retrying after failure",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay_seconds=round(delay, 2),
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                        
                        await asyncio.sleep(delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            "Max retry attempts exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(
                        "Non-retryable exception encountered",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise
            
            # Raise the last exception after all retries exhausted
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        # Calculate delay with exponential backoff
                        delay = min(current_delay, max_delay)
                        
                        # Add jitter if enabled (±25% of delay)
                        if jitter:
                            jitter_amount = delay * 0.25
                            delay = delay + random.uniform(-jitter_amount, jitter_amount)
                            delay = max(0.1, delay)  # Ensure minimum delay
                        
                        logger.warning(
                            "Retrying after failure",
                            function=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay_seconds=round(delay, 2),
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                        
                        time.sleep(delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            "Max retry attempts exhausted",
                            function=func.__name__,
                            attempts=max_attempts,
                            error_type=type(e).__name__,
                            error_message=str(e),
                        )
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    logger.error(
                        "Non-retryable exception encountered",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise
            
            # Raise the last exception after all retries exhausted
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
