# Error Handling Best Practices Guide

## Overview

This guide provides comprehensive best practices for implementing and using the error handling system in Project GENESIS. It covers patterns, anti-patterns, and practical examples for robust error management.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             Error Handling Integration               │  │
│  │  - Correlation ID Tracking                          │  │
│  │  - Circuit Breaker Protection                       │  │
│  │  - Retry Logic with Backoff                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                             ↓                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Core Error Components                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │  Global  │  │ Circuit  │  │   Dead Letter    │ │  │
│  │  │  Handler │  │ Breakers │  │      Queue       │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │
│  │  │ Recovery │  │  Error   │  │     Feature      │ │  │
│  │  │ Manager  │  │  Budget  │  │      Flags       │ │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Fail Fast, Recover Gracefully

```python
# ✅ Good: Fail fast with clear error
async def place_order(order: OrderRequest):
    # Validate immediately
    if order.quantity <= 0:
        raise ValidationError("Order quantity must be positive", field="quantity")
    
    # Then attempt with recovery
    try:
        return await exchange.place_order(order)
    except NetworkError as e:
        # Graceful recovery attempt
        await recovery_manager.attempt_recovery(e)
        raise

# ❌ Bad: Silent failure
async def place_order(order: OrderRequest):
    try:
        return await exchange.place_order(order)
    except:
        return None  # Never do this!
```

### 2. Use Specific Exception Types

```python
# ✅ Good: Specific exceptions with context
raise OrderRejected(
    "Insufficient balance for order",
    order_id=order.id,
    reason="INSUFFICIENT_BALANCE"
)

# ❌ Bad: Generic exceptions
raise Exception("Order failed")
```

### 3. Always Include Correlation IDs

```python
# ✅ Good: Correlation ID propagation
from genesis.core.correlation import CorrelationContext

async def process_trade():
    with CorrelationContext.with_correlation_id() as cid:
        logger.info("Processing trade", correlation_id=cid)
        
        # All nested calls will have same correlation ID
        await validate_order()
        await execute_order()
        await update_position()

# ❌ Bad: No correlation tracking
async def process_trade():
    await validate_order()
    await execute_order()
    # Can't trace the request flow!
```

## Implementation Patterns

### Pattern 1: Wrapped Service Calls

Use the error integration wrapper for all external service calls:

```python
from genesis.core.error_integration import get_error_integration

class TradingService:
    def __init__(self):
        self.integration = get_error_integration()
        
    async def execute_trade(self, order):
        # Wrap the actual exchange call
        wrapped_call = self.integration.wrap_exchange_call(
            self._raw_execute_trade
        )
        return await wrapped_call(order)
    
    async def _raw_execute_trade(self, order):
        # Actual implementation
        return await self.exchange.place_order(order)
```

### Pattern 2: Circuit Breaker for External Services

```python
from genesis.core.circuit_breaker import get_circuit_breaker_registry

class MarketDataService:
    def __init__(self):
        self.breaker = get_circuit_breaker_registry().get_or_create(
            "market_data",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                success_threshold=2,
            )
        )
    
    async def get_ticker(self, symbol: str):
        return await self.breaker.call(
            self._fetch_ticker,
            symbol
        )
    
    async def _fetch_ticker(self, symbol: str):
        # Actual API call
        return await self.api.get_ticker(symbol)
```

### Pattern 3: Retry with Exponential Backoff

```python
from genesis.utils.decorators import with_retry
from genesis.core.exceptions import NetworkError, RateLimitError

@with_retry(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(NetworkError, RateLimitError)
)
async def fetch_market_data():
    # This will automatically retry with backoff
    return await api.get_market_data()
```

### Pattern 4: Dead Letter Queue for Critical Operations

```python
from genesis.core.dead_letter_queue import DeadLetterQueue

class OrderProcessor:
    def __init__(self):
        self.dlq = DeadLetterQueue(name="orders")
        
        # Register retry handler
        self.dlq.register_retry_handler(
            "order_execution",
            self.retry_order
        )
    
    async def process_order(self, order):
        try:
            return await self.execute_order(order)
        except Exception as e:
            # Add to DLQ for later retry
            await self.dlq.add(
                operation_type="order_execution",
                payload=order.to_dict(),
                error=e,
                max_retries=3,
                correlation_id=CorrelationContext.get_current()
            )
            raise
    
    async def retry_order(self, payload):
        # Called by DLQ retry worker
        order = Order.from_dict(payload)
        return await self.execute_order(order)
```

### Pattern 5: Feature Degradation

```python
from genesis.core.feature_flags import FeatureManager

class TradingEngine:
    def __init__(self):
        self.features = FeatureManager()
    
    async def execute_strategy(self):
        # Check feature availability
        if self.features.is_enabled("advanced_analytics"):
            await self.run_advanced_analytics()
        else:
            # Fallback to basic analytics
            await self.run_basic_analytics()
        
        # Critical features always run
        await self.execute_core_trading()
```

### Pattern 6: Error Budget Tracking

```python
from genesis.monitoring.error_budget import ErrorBudget

class OrderService:
    def __init__(self):
        self.error_budget = ErrorBudget()
    
    async def place_order(self, order):
        try:
            result = await self.exchange.place_order(order)
            
            # Record success
            self.error_budget.record_success("order_execution")
            
            return result
            
        except Exception as e:
            # Record error with appropriate category
            self.error_budget.record_error(
                "order_execution",
                category="EXCHANGE",
                severity=self._determine_severity(e)
            )
            raise
```

## Error Handling Decision Tree

```
Exception Occurs
     |
     v
Is it retryable?
     |
    Yes ──→ Apply retry logic
     |           |
     No          v
     |      Success? ──→ Continue
     |           |
     |          No
     |           |
     v           v
Is it critical?  |
     |           |
    Yes ─────────┘
     |
     v
Add to DLQ
     |
     v
Check circuit breaker
     |
     v
Should degrade?
     |
    Yes ──→ Adjust feature flags
     |
     v
Log and alert
```

## Common Anti-Patterns to Avoid

### 1. Catching and Ignoring Exceptions

```python
# ❌ Never do this
try:
    result = await risky_operation()
except:
    pass  # Silent failure!

# ✅ Always handle appropriately
try:
    result = await risky_operation()
except SpecificError as e:
    logger.error("Operation failed", error=str(e))
    # Take appropriate action
    raise
```

### 2. Using Bare Except Clauses

```python
# ❌ Bad: Catches everything including system exits
try:
    process_data()
except:
    handle_error()

# ✅ Good: Catch specific exceptions
try:
    process_data()
except (ValueError, TypeError) as e:
    handle_error(e)
```

### 3. Losing Error Context

```python
# ❌ Bad: Original context lost
try:
    await process()
except Exception as e:
    raise ValueError("Processing failed")  # Lost original error!

# ✅ Good: Preserve context
try:
    await process()
except Exception as e:
    raise ValueError("Processing failed") from e  # Chain exceptions
```

### 4. Inconsistent Error Handling

```python
# ❌ Bad: Different patterns everywhere
def function_a():
    if error:
        return None
        
def function_b():
    if error:
        return -1
        
def function_c():
    if error:
        raise Exception("Error")

# ✅ Good: Consistent error handling
def function_a():
    if error:
        raise DomainError("Specific error message")
```

## Testing Error Scenarios

### Unit Testing with Error Simulation

```python
import pytest
from genesis.testing.error_simulator import ErrorSimulator

@pytest.fixture
def error_simulator():
    simulator = ErrorSimulator()
    simulator.set_mode(SimulationMode.DETERMINISTIC)
    return simulator

async def test_order_with_network_failure(error_simulator):
    # Activate network failure scenario
    error_simulator.activate_scenario("network_timeout")
    
    service = OrderService()
    
    # Should handle network timeout gracefully
    with pytest.raises(ConnectionTimeout):
        await service.place_order(test_order)
    
    # Verify recovery was attempted
    assert service.recovery_attempts > 0
```

### Integration Testing with Chaos Mode

```python
async def test_system_resilience():
    simulator = ErrorSimulator()
    harness = simulator.create_test_harness()
    
    # Run chaos test
    results = await harness.run_chaos_test(
        test_function=execute_trading_cycle,
        duration_seconds=60,
        intensity=0.1  # 10% failure rate
    )
    
    # System should maintain > 90% success rate
    assert results["failure_rate"] < 0.2
```

## Monitoring and Alerting

### Key Metrics to Track

```python
from genesis.core.error_integration import get_error_integration

# Get comprehensive health metrics
integration = get_error_integration()
health = integration.get_system_health()

# Check specific components
if health["circuit_breakers"]["open"] > 0:
    alert("Circuit breakers are open")

if health["error_budget"]["exhausted_budgets"] > 0:
    alert("Error budgets exhausted")

if health["dlq"]["queue_size"] > 100:
    alert("Dead letter queue building up")
```

### Setting Up Alerts

```python
# Configure error budget alerts
error_budget = ErrorBudget()
error_budget.set_alert_threshold("order_execution", 0.8)  # Alert at 80%

# Configure circuit breaker monitoring
def on_circuit_open(breaker_name):
    send_alert(f"Circuit breaker {breaker_name} opened")
    
registry = get_circuit_breaker_registry()
for breaker in registry._breakers.values():
    breaker.on_open_callback = on_circuit_open
```

## Performance Considerations

### 1. Minimize Error Handler Overhead

```python
# ✅ Good: Lightweight error handling
async def fast_path():
    if not validated:
        raise ValidationError("Quick fail")
    return await process()

# ❌ Bad: Heavy error handling
async def slow_path():
    try:
        # Complex validation
        validate_everything()
        log_everything()
        check_everything()
        return await process()
    except Exception as e:
        # Heavy error processing
        analyze_error(e)
        generate_report(e)
        send_notifications(e)
        raise
```

### 2. Use Async Error Handling

```python
# ✅ Good: Non-blocking error handling
async def handle_error_async(error):
    await error_handler.handle_async_error(error)
    
# ❌ Bad: Blocking error handling
def handle_error_sync(error):
    time.sleep(1)  # Blocks event loop!
    error_handler.handle_error(error)
```

### 3. Batch Error Reporting

```python
# ✅ Good: Batch errors for efficiency
class ErrorBatcher:
    def __init__(self):
        self.errors = []
        
    async def add_error(self, error):
        self.errors.append(error)
        if len(self.errors) >= 10:
            await self.flush()
    
    async def flush(self):
        if self.errors:
            await self.send_batch(self.errors)
            self.errors.clear()
```

## Migration Guide

### Integrating with Existing Code

1. **Step 1: Wrap External Calls**
```python
# Before
result = await exchange.place_order(order)

# After
from genesis.core.error_integration import get_error_integration
integration = get_error_integration()
wrapped_call = integration.wrap_exchange_call(exchange.place_order)
result = await wrapped_call(order)
```

2. **Step 2: Add Correlation Tracking**
```python
# Before
async def process_request(request):
    await handle_request(request)

# After
async def process_request(request):
    with CorrelationContext.with_correlation_id() as cid:
        logger.info("Processing request", correlation_id=cid)
        await handle_request(request)
```

3. **Step 3: Implement Circuit Breakers**
```python
# Before
data = await fetch_external_data()

# After
breaker = get_circuit_breaker_registry().get_or_create("external_api")
data = await breaker.call(fetch_external_data)
```

## Troubleshooting Common Issues

### Issue: Circuit Breaker Stuck Open

**Symptoms**: All requests failing with `CircuitBreakerError`

**Solution**:
```python
# Check breaker status
breaker = registry.get("service_name")
status = breaker.get_status()

# If service recovered, manually reset
if service_is_healthy():
    await breaker.reset()
```

### Issue: DLQ Growing Unbounded

**Symptoms**: Memory usage increasing, DLQ stats show high queue depth

**Solution**:
```python
# Check failed items
failed = dlq.get_items(status=DLQItemStatus.FAILED)

# Clear permanently failed items
await dlq.clear(status=DLQItemStatus.FAILED)

# Adjust retry strategy
dlq.max_retries = 3  # Reduce retry attempts
```

### Issue: Error Budget Exhausted

**Symptoms**: System in degraded mode, features disabled

**Solution**:
```python
# Check budget status
status = error_budget.get_budget_status("slo_name")

# If false positives, reset budget
if issue_resolved:
    error_budget.reset_budget("slo_name")
    
# Adjust SLO if too strict
slo.target = 0.999  # From 0.9999
```

## Summary

Effective error handling in GENESIS requires:

1. **Proactive Design**: Build error handling in from the start
2. **Consistent Patterns**: Use the same patterns throughout
3. **Comprehensive Testing**: Test failure scenarios thoroughly
4. **Active Monitoring**: Watch error metrics continuously
5. **Continuous Improvement**: Learn from failures and improve

Remember: **Every error is an opportunity to make the system more resilient.**

---

*Last Updated: 2025-08-30*
*Version: 1.0*