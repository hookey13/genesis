"""
Runtime integration tests to ensure method signatures remain consistent
between components, preventing runtime errors from mismatched interfaces.
"""

import asyncio
import inspect
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.circuit_breaker import CircuitBreaker, CircuitState
from genesis.core.rate_limiter import Priority, RateLimiter
from genesis.exchange.gateway import ExchangeGateway


class TestMethodSignatures:
    """Test that method signatures match between interacting components."""

    @pytest.fixture
    async def rate_limiter(self):
        """Create a rate limiter instance."""
        from genesis.core.rate_limiter import RateLimitConfig
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=100,
            window_size_seconds=60
        )
        return RateLimiter(config)

    @pytest.fixture
    async def circuit_breaker(self):
        """Create a circuit breaker instance."""
        return CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            half_open_max_calls=3
        )

    @pytest.fixture
    async def gateway(self, rate_limiter, circuit_breaker):
        """Create a gateway instance with mocked exchange."""
        with patch('genesis.exchange.gateway.ccxt.binance') as mock_exchange:
            gateway = ExchangeGateway()
            gateway.rate_limiter = rate_limiter
            gateway.circuit_breaker = circuit_breaker
            gateway.exchange = mock_exchange()
            gateway.initialized = True
            return gateway

    async def test_rate_limiter_method_signatures(self, gateway, rate_limiter):
        """
        Test that gateway calls rate limiter methods with correct signatures.
        This test validates that the integration between gateway and rate limiter
        is correct at runtime.
        """
        # Test 1: Verify _make_request_with_priority method exists and has correct signature
        assert hasattr(gateway, '_make_request_with_priority')

        # Get the method signature
        sig = inspect.signature(gateway._make_request_with_priority)
        params = list(sig.parameters.keys())

        # Expected parameters: func, weight, priority, *args, **kwargs
        assert 'func' in params
        assert 'weight' in params
        assert 'priority' in params

        # Test 2: Verify rate limiter has required methods
        assert hasattr(rate_limiter, 'acquire')
        assert hasattr(rate_limiter, 'update_from_headers')
        assert hasattr(rate_limiter, 'coalesce_request')
        assert hasattr(rate_limiter, 'get_metrics')

        # Test 3: Test actual method call with mock function
        mock_func = AsyncMock(return_value={'test': 'data'})

        # This should not raise any exceptions
        result = await gateway._make_request_with_priority(
            mock_func,
            5,  # weight
            Priority.NORMAL,  # priority
            'arg1',
            kwarg1='value1'
        )

        # Verify the mock was called correctly
        mock_func.assert_called_once_with('arg1', kwarg1='value1')
        assert result == {'test': 'data'}
        
    async def test_circuit_breaker_method_signatures(self, gateway, circuit_breaker):
        """
        Test that gateway calls circuit breaker methods with correct signatures.
        """
        # Test 1: Verify circuit breaker has required methods
        assert hasattr(circuit_breaker, 'call')
        assert hasattr(circuit_breaker, 'state')
        assert hasattr(circuit_breaker, 'get_statistics')
        assert hasattr(circuit_breaker, 'reset')
        
        # Test 2: Verify call method signature
        sig = inspect.signature(circuit_breaker.call)
        params = list(sig.parameters.keys())
        
        # Expected parameters: func, *args, fallback, degradation_strategy, **kwargs
        assert 'func' in params
        assert 'fallback' in params
        assert 'degradation_strategy' in params
        
        # Test 3: Test actual method call
        mock_func = AsyncMock(return_value={'test': 'data'})
        
        result = await circuit_breaker.call(
            mock_func,
            'arg1',
            fallback=lambda: {'fallback': 'data'},
            degradation_strategy='fail_fast',
            kwarg1='value1'
        )
        
        # Should succeed when circuit is closed
        assert result == {'test': 'data'}
        mock_func.assert_called_once_with('arg1', kwarg1='value1')
        
    async def test_gateway_rate_limiting_integration(self, gateway):
        """
        Test that gateway correctly integrates rate limiting for all API calls.
        This ensures method signatures remain consistent across the integration.
        """
        # Mock the exchange methods
        gateway.exchange.fetch_balance = AsyncMock(return_value={
            'USDT': {'free': 1000.0, 'total': 1000.0},
            'BTC': {'free': 0.1, 'total': 0.1}
        })
        
        gateway.exchange.create_order = AsyncMock(return_value={
            'id': 'test-order-123',
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 50000,
            'amount': 0.01
        })
        
        gateway.exchange.cancel_order = AsyncMock(return_value={
            'id': 'test-order-123',
            'status': 'canceled'
        })
        
        gateway.exchange.fetch_open_orders = AsyncMock(return_value=[])
        
        # Test 1: Get balance with rate limiting
        balance = await gateway.get_balance()
        assert balance is not None
        gateway.exchange.fetch_balance.assert_called_once()
        
        # Test 2: Place order with rate limiting
        from genesis.core.models import Order, OrderSide, OrderType
        from decimal import Decimal
        
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal('0.01'),
            price=Decimal('50000')
        )
        
        result = await gateway.place_order(order)
        assert result is not None
        gateway.exchange.create_order.assert_called_once()
        
        # Test 3: Cancel order with rate limiting
        cancel_result = await gateway.cancel_order('test-order-123', 'BTC/USDT')
        assert cancel_result is True
        gateway.exchange.cancel_order.assert_called_once()
        
        # Test 4: Get open orders with rate limiting
        orders = await gateway.get_open_orders('BTC/USDT')
        assert orders == []
        gateway.exchange.fetch_open_orders.assert_called_once()
        
    async def test_priority_handling_consistency(self, gateway):
        """
        Test that priority handling is consistent across all gateway methods.
        """
        # Mock exchange methods
        gateway.exchange.create_order = AsyncMock(return_value={'id': 'test-123'})
        gateway.exchange.cancel_order = AsyncMock(return_value={'status': 'canceled'})
        
        # Test critical priority for emergency close
        from genesis.core.models import Order, OrderSide, OrderType
        from decimal import Decimal
        
        # Create emergency close order
        emergency_order = Order(
            symbol='BTC/USDT',
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal('0.01'),
            client_order_id='emergency-close-001'
        )
        
        # This should use CRITICAL priority internally
        result = await gateway.place_order(emergency_order)
        assert result is not None
        
        # Test that cancel order for emergency uses HIGH priority
        await gateway.cancel_order('emergency-order-123', 'BTC/USDT')
        
        # Verify rate limiter metrics show priority usage
        metrics = gateway.rate_limiter.get_metrics()
        assert 'requests_allowed' in metrics
        assert 'requests_rejected' in metrics
        
    async def test_error_propagation_consistency(self, gateway):
        """
        Test that errors propagate correctly through rate limiter and circuit breaker.
        """
        # Test 1: Rate limit exhaustion
        gateway.rate_limiter.token_bucket.available = 0  # Exhaust capacity
        gateway.exchange.fetch_balance = AsyncMock(return_value={})
        
        # Should raise appropriate exception
        with pytest.raises(Exception) as exc_info:
            await gateway.get_balance()
        
        # Test 2: Circuit breaker open
        # Force circuit breaker to open state
        gateway.circuit_breaker.state = CircuitState.OPEN
        gateway.circuit_breaker.last_failure_time = asyncio.get_event_loop().time()
        
        # Reset rate limiter
        gateway.rate_limiter.token_bucket.capacity = 100
        gateway.rate_limiter.token_bucket.available = 100
        
        # Should handle gracefully with fallback or error
        with pytest.raises(Exception) as exc_info:
            await gateway.get_balance()
        
    async def test_coalescing_integration(self, gateway):
        """
        Test that request coalescing works correctly with gateway integration.
        """
        # Mock exchange method with delay
        call_count = 0
        
        async def mock_fetch_ticker(symbol):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate API delay
            return {'symbol': symbol, 'last': 50000}
        
        gateway.exchange.fetch_ticker = mock_fetch_ticker
        
        # Make multiple concurrent requests for same data
        tasks = [
            gateway._make_request_with_priority(
                gateway.exchange.fetch_ticker,
                1,  # weight
                Priority.NORMAL,  # priority
                'BTC/USDT'
            )
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All results should be the same (coalesced)
        for result in results:
            assert result == {'symbol': 'BTC/USDT', 'last': 50000}
        
        # Should only have made 1 actual API call due to coalescing
        assert call_count == 1
        
    async def test_adaptive_rate_limiting_integration(self, gateway):
        """
        Test that adaptive rate limiting correctly updates from response headers.
        """
        # Mock exchange method that returns headers
        async def mock_fetch_with_headers(*args, **kwargs):
            # Simulate Binance rate limit headers
            gateway.rate_limiter.update_from_headers({
                'X-MBX-USED-WEIGHT': '50',
                'X-MBX-USED-WEIGHT-1M': '500', 
                'X-MBX-ORDER-COUNT-1M': '10',
                'X-MBX-ORDER-COUNT-10S': '5'
            })
            return {'test': 'data'}
        
        gateway.exchange.fetch_balance = mock_fetch_with_headers
        
        # Get initial metrics
        initial_metrics = gateway.rate_limiter.get_metrics()
        initial_capacity = initial_metrics['capacity']
        
        # Make request that updates rate limits
        await gateway.get_balance()
        
        # Check if rate limiter was updated
        updated_metrics = gateway.rate_limiter.get_metrics()

        # Metrics should reflect the update
        assert 'requests_allowed' in updated_metrics
        assert 'requests_rejected' in updated_metrics
        assert 'adaptive_capacity' in updated_metrics


class TestRuntimeValidation:
    """
    Additional runtime validation tests to catch integration issues early.
    """

    async def test_all_gateway_methods_use_rate_limiting(self):
        """
        Verify that all public gateway methods that make API calls use rate limiting.
        """
        gateway = ExchangeGateway()

        # List of methods that should use rate limiting
        api_methods = [
            'place_order',
            'cancel_order',
            'get_balance',
            'get_open_orders',
            'get_order_status',
            'get_ticker',
            'get_orderbook',
            'get_recent_trades'
        ]

        for method_name in api_methods:
            assert hasattr(gateway, method_name), f"Gateway missing method: {method_name}"

            # Get method source code
            method = getattr(gateway, method_name)
            source = inspect.getsource(method)

            # Verify it uses _make_request_with_priority for rate limiting
            assert '_make_request_with_priority' in source or 'rate_limiter' in source, \
                f"Method {method_name} doesn't use rate limiting"

    async def test_all_external_calls_use_circuit_breaker(self):
        """
        Verify that all external API calls are wrapped with circuit breaker.
        """
        gateway = ExchangeGateway()

        # Get all methods that make external calls
        external_methods = []
        for name, method in inspect.getmembers(gateway, predicate=inspect.ismethod):
            if name.startswith('_'):
                continue
            source = inspect.getsource(method)
            if 'exchange.' in source:  # Calls exchange API
                external_methods.append(name)

        # Each should use circuit breaker
        for method_name in external_methods:
            method = getattr(gateway, method_name)
            source = inspect.getsource(method)

            # Should either use circuit_breaker directly or via _make_request_with_priority
            assert 'circuit_breaker' in source or '_make_request_with_priority' in source, \
                f"Method {method_name} doesn't use circuit breaker for external calls"
