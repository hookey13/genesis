# Exchange Failures Documentation

## Overview

Exchange failure simulation provides comprehensive testing of system behavior when interacting with unreliable exchange APIs. This framework simulates various failure modes including rate limiting, connection issues, invalid responses, order rejections, and maintenance windows to ensure robust error handling and recovery mechanisms.

## Purpose

Exchange failure testing addresses critical trading system concerns:
- Validates rate limit handling and backoff strategies
- Tests connection timeout and retry mechanisms
- Verifies order rejection handling
- Tests partial fill scenarios
- Validates price slippage handling
- Ensures balance mismatch detection
- Tests maintenance window behavior
- Validates data consistency during failures

## Architecture

### Core Components

#### 1. FailingExchange Class
Mock exchange that simulates failures:
- Configurable failure modes
- Probabilistic failure injection
- Stateful failure simulation
- Realistic response generation
- Failure metrics tracking

#### 2. FailureMode Enum
Types of failures that can be simulated:
- RATE_LIMIT: API rate limiting
- CONNECTION_ERROR: Network connection failures
- TIMEOUT: Request timeouts
- INVALID_ORDER: Order validation failures
- INSUFFICIENT_BALANCE: Balance-related rejections
- PARTIAL_FILL: Incomplete order execution
- PRICE_SLIPPAGE: Unfavorable price execution
- INVALID_RESPONSE: Malformed API responses
- MAINTENANCE: Exchange maintenance mode

#### 3. Failure Injection
Controls how failures are introduced:
- Probability-based injection
- Time-based patterns
- Sequence-based triggers
- Load-based thresholds
- Correlated failures

## Configuration

### Basic Setup

```python
from tests.mocks.failing_exchange import FailingExchange, FailureMode

# Create failing exchange
exchange = FailingExchange()

# Configure failure mode
exchange.set_failure_mode(
    mode=FailureMode.RATE_LIMIT,
    probability=0.1  # 10% chance of rate limiting
)

# Use like normal exchange
try:
    balance = await exchange.fetch_balance()
except RateLimitError:
    # Handle rate limit
    await asyncio.sleep(60)
```

### Advanced Configuration

```python
exchange = FailingExchange(
    base_failure_rate=0.05,
    rate_limit_threshold=10,
    rate_limit_window=60,
    connection_timeout=5,
    max_retries=3,
    maintenance_schedule={
        "daily": {"start": "02:00", "duration": 30},
        "weekly": {"day": "Sunday", "start": "00:00", "duration": 120}
    }
)

# Configure multiple failure modes
exchange.add_failure_mode(FailureMode.RATE_LIMIT, probability=0.1)
exchange.add_failure_mode(FailureMode.CONNECTION_ERROR, probability=0.05)
exchange.add_failure_mode(FailureMode.PARTIAL_FILL, probability=0.2)
```

## Failure Scenarios

### 1. Rate Limiting

```python
async def test_rate_limiting():
    """Test rate limit handling"""
    exchange = FailingExchange()
    exchange.set_rate_limit(calls_per_minute=60)
    
    # Burst requests
    requests_made = 0
    rate_limited = False
    
    for _ in range(100):
        try:
            await exchange.fetch_ticker("BTC/USDT")
            requests_made += 1
        except RateLimitError as e:
            rate_limited = True
            wait_time = e.retry_after
            print(f"Rate limited, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
    
    assert rate_limited, "Should have hit rate limit"
    assert requests_made <= 60, "Should respect rate limit"
```

### 2. Connection Failures

```python
async def test_connection_resilience():
    """Test connection failure handling"""
    exchange = FailingExchange()
    exchange.set_failure_mode(FailureMode.CONNECTION_ERROR, probability=0.3)
    
    # Implement retry logic
    async def fetch_with_retry(symbol, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await exchange.fetch_ticker(symbol)
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
    
    # Test resilience
    success_count = 0
    for _ in range(100):
        try:
            await fetch_with_retry("BTC/USDT")
            success_count += 1
        except ConnectionError:
            pass
    
    # Should succeed most of the time with retries
    assert success_count > 90
```

### 3. Order Rejections

```python
async def test_order_rejection_handling():
    """Test order rejection scenarios"""
    exchange = FailingExchange()
    
    # Test various rejection reasons
    rejection_scenarios = [
        (FailureMode.INVALID_ORDER, "Invalid price"),
        (FailureMode.INSUFFICIENT_BALANCE, "Insufficient funds"),
        (FailureMode.INVALID_SYMBOL, "Symbol not found")
    ]
    
    for mode, expected_error in rejection_scenarios:
        exchange.set_failure_mode(mode, probability=1.0)
        
        try:
            await exchange.create_order(
                symbol="BTC/USDT",
                type="limit",
                side="buy",
                amount=1.0,
                price=50000
            )
            assert False, f"Should have raised {expected_error}"
        except OrderRejectedError as e:
            assert expected_error.lower() in str(e).lower()
            print(f"Correctly handled: {e}")
```

### 4. Partial Fills

```python
async def test_partial_fill_handling():
    """Test partial order fill scenarios"""
    exchange = FailingExchange()
    exchange.set_failure_mode(
        FailureMode.PARTIAL_FILL,
        probability=1.0,
        params={"fill_ratio": 0.3}  # Fill only 30%
    )
    
    # Place order
    order = await exchange.create_order(
        symbol="BTC/USDT",
        type="limit",
        side="buy",
        amount=1.0,
        price=50000
    )
    
    assert order['status'] == 'open'
    assert order['filled'] == 0.3
    assert order['remaining'] == 0.7
    
    # Handle partial fill
    while order['remaining'] > 0:
        # Wait for fill or timeout
        await asyncio.sleep(5)
        
        # Check order status
        updated = await exchange.fetch_order(order['id'])
        
        if updated['status'] == 'closed':
            break
        elif time.time() - order['timestamp'] > 300:
            # Cancel after 5 minutes
            await exchange.cancel_order(order['id'])
            break
```

### 5. Price Slippage

```python
async def test_price_slippage():
    """Test price slippage in market orders"""
    exchange = FailingExchange()
    exchange.set_failure_mode(
        FailureMode.PRICE_SLIPPAGE,
        probability=1.0,
        params={"max_slippage": 0.02}  # 2% max slippage
    )
    
    # Get expected price
    ticker = await exchange.fetch_ticker("BTC/USDT")
    expected_price = ticker['ask']
    
    # Place market order
    order = await exchange.create_order(
        symbol="BTC/USDT",
        type="market",
        side="buy",
        amount=10.0  # Large order
    )
    
    # Check slippage
    actual_price = order['average']
    slippage = (actual_price - expected_price) / expected_price
    
    print(f"Expected: ${expected_price:.2f}")
    print(f"Actual: ${actual_price:.2f}")
    print(f"Slippage: {slippage:.2%}")
    
    assert slippage <= 0.02, "Excessive slippage"
```

### 6. Maintenance Windows

```python
async def test_maintenance_handling():
    """Test exchange maintenance window handling"""
    exchange = FailingExchange()
    
    # Schedule maintenance
    exchange.schedule_maintenance(
        start_time=datetime.now() + timedelta(seconds=5),
        duration_minutes=10,
        message="System upgrade in progress"
    )
    
    # Wait for maintenance to start
    await asyncio.sleep(6)
    
    # All requests should fail during maintenance
    with pytest.raises(ExchangeMaintenanceError) as exc:
        await exchange.fetch_balance()
    
    assert "System upgrade" in str(exc.value)
    
    # System should queue or defer operations
    queued_orders = []
    
    async def queue_order(order):
        queued_orders.append(order)
        print(f"Order queued during maintenance: {order['id']}")
    
    # Queue orders during maintenance
    await queue_order({
        "id": "123",
        "symbol": "BTC/USDT",
        "side": "buy",
        "amount": 0.1
    })
    
    # Process queue after maintenance
    await asyncio.sleep(600)  # Wait for maintenance to end
    
    for order in queued_orders:
        await exchange.create_order(**order)
```

## Usage Examples

### Comprehensive Failure Testing

```python
import asyncio
from tests.mocks.failing_exchange import FailingExchange, FailureMode

async def test_exchange_resilience():
    """Test system resilience to various exchange failures"""
    
    exchange = FailingExchange()
    
    # Configure realistic failure mix
    failure_config = {
        FailureMode.RATE_LIMIT: 0.05,
        FailureMode.CONNECTION_ERROR: 0.02,
        FailureMode.TIMEOUT: 0.01,
        FailureMode.INVALID_ORDER: 0.01,
        FailureMode.PARTIAL_FILL: 0.1,
        FailureMode.PRICE_SLIPPAGE: 0.05
    }
    
    for mode, probability in failure_config.items():
        exchange.add_failure_mode(mode, probability)
    
    # Run trading simulation
    stats = {
        "orders_attempted": 0,
        "orders_successful": 0,
        "orders_failed": 0,
        "rate_limits_hit": 0,
        "connection_errors": 0,
        "partial_fills": 0
    }
    
    for _ in range(1000):
        stats["orders_attempted"] += 1
        
        try:
            order = await place_order_with_retry(exchange)
            
            if order['filled'] < order['amount']:
                stats["partial_fills"] += 1
            
            stats["orders_successful"] += 1
            
        except RateLimitError:
            stats["rate_limits_hit"] += 1
            await asyncio.sleep(60)
            
        except ConnectionError:
            stats["connection_errors"] += 1
            stats["orders_failed"] += 1
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            stats["orders_failed"] += 1
    
    # Analyze results
    success_rate = stats["orders_successful"] / stats["orders_attempted"]
    print(f"Success rate: {success_rate:.2%}")
    print(f"Rate limits hit: {stats['rate_limits_hit']}")
    print(f"Connection errors: {stats['connection_errors']}")
    print(f"Partial fills: {stats['partial_fills']}")
    
    # System should maintain high success rate despite failures
    assert success_rate > 0.95

asyncio.run(test_exchange_resilience())
```

### Stress Testing with Failures

```python
async def stress_test_with_failures():
    """Stress test system with exchange failures"""
    
    exchange = FailingExchange()
    
    # Gradually increase failure rate
    for failure_rate in [0.01, 0.05, 0.1, 0.2, 0.5]:
        exchange.set_base_failure_rate(failure_rate)
        
        print(f"\nTesting with {failure_rate:.0%} failure rate")
        
        # Run load test
        start_time = time.time()
        successful_orders = 0
        failed_orders = 0
        
        async def place_orders():
            nonlocal successful_orders, failed_orders
            
            for _ in range(100):
                try:
                    await exchange.create_order(
                        symbol="BTC/USDT",
                        type="limit",
                        side="buy",
                        amount=0.01,
                        price=50000
                    )
                    successful_orders += 1
                except Exception:
                    failed_orders += 1
        
        # Run concurrent orders
        await asyncio.gather(*[place_orders() for _ in range(10)])
        
        duration = time.time() - start_time
        throughput = successful_orders / duration
        
        print(f"Successful: {successful_orders}")
        print(f"Failed: {failed_orders}")
        print(f"Throughput: {throughput:.2f} orders/sec")
        
        # System should degrade gracefully
        if failure_rate <= 0.1:
            assert successful_orders > failed_orders

asyncio.run(stress_test_with_failures())
```

## Command-Line Usage

```bash
# Basic failure simulation
python -m tests.mocks.failing_exchange --duration 3600

# Specific failure mode
python -m tests.mocks.failing_exchange \
    --mode RATE_LIMIT \
    --probability 0.1 \
    --duration 1800

# Multiple failure modes
python -m tests.mocks.failing_exchange \
    --modes RATE_LIMIT CONNECTION_ERROR PARTIAL_FILL \
    --probabilities 0.1 0.05 0.2 \
    --duration 3600

# Maintenance window simulation
python -m tests.mocks.failing_exchange \
    --maintenance \
    --maintenance-start "02:00" \
    --maintenance-duration 30
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --duration | 3600 | Test duration in seconds |
| --mode | RANDOM | Failure mode to simulate |
| --probability | 0.1 | Failure probability |
| --rate-limit | 60 | Rate limit (requests/minute) |
| --timeout | 5 | Connection timeout (seconds) |
| --maintenance | false | Enable maintenance simulation |
| --output | exchange_test.json | Output file for results |

## Best Practices

### 1. Realistic Failure Patterns

```python
# Use realistic failure patterns
def get_time_based_failure_rate(hour):
    """Higher failure rates during peak hours"""
    if 9 <= hour <= 11 or 14 <= hour <= 16:
        return 0.05  # 5% during peak
    elif 2 <= hour <= 4:
        return 0.01  # 1% during maintenance window
    else:
        return 0.02  # 2% normal

exchange.set_dynamic_failure_rate(get_time_based_failure_rate)
```

### 2. Comprehensive Error Handling

```python
async def robust_order_placement(exchange, **kwargs):
    """Place order with comprehensive error handling"""
    
    max_retries = 3
    retry_delays = [1, 5, 30]  # Escalating delays
    
    for attempt in range(max_retries):
        try:
            return await exchange.create_order(**kwargs)
            
        except RateLimitError as e:
            wait_time = e.retry_after or 60
            logger.warning(f"Rate limited, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
            
        except ConnectionError:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logger.warning(f"Connection error, retry in {delay}s")
                await asyncio.sleep(delay)
            else:
                raise
                
        except InsufficientBalanceError:
            logger.error("Insufficient balance")
            raise  # Don't retry
            
        except OrderRejectedError as e:
            if "price" in str(e).lower():
                # Adjust price and retry
                kwargs['price'] *= 1.01
            else:
                raise
```

### 3. State Consistency

```python
async def maintain_consistency():
    """Maintain state consistency during failures"""
    
    # Track local state
    local_orders = {}
    
    # Sync with exchange periodically
    async def sync_orders():
        while True:
            try:
                exchange_orders = await exchange.fetch_open_orders()
                
                # Reconcile differences
                for order in exchange_orders:
                    if order['id'] not in local_orders:
                        logger.warning(f"Unknown order found: {order['id']}")
                        local_orders[order['id']] = order
                
                # Check for missing orders
                for order_id in list(local_orders.keys()):
                    if order_id not in [o['id'] for o in exchange_orders]:
                        logger.warning(f"Order disappeared: {order_id}")
                        del local_orders[order_id]
                        
            except Exception as e:
                logger.error(f"Sync failed: {e}")
            
            await asyncio.sleep(30)
    
    asyncio.create_task(sync_orders())
```

### 4. Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Circuit breaker for exchange calls"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error("Circuit breaker opened")
            
            raise

# Use circuit breaker
breaker = CircuitBreaker()
try:
    balance = await breaker.call(exchange.fetch_balance)
except CircuitOpenError:
    logger.error("Exchange unavailable")
```

## Troubleshooting

### Debugging Failures

```python
# Enable detailed logging
exchange.enable_debug_logging()

# Track failure patterns
failures = exchange.get_failure_statistics()
print(f"Total failures: {failures['total']}")
print(f"By type: {failures['by_type']}")
print(f"Failure rate: {failures['rate']:.2%}")

# Get failure timeline
timeline = exchange.get_failure_timeline()
for event in timeline[-10:]:
    print(f"{event['timestamp']}: {event['type']} - {event['details']}")
```

### Performance Analysis

```python
# Analyze performance impact
metrics = exchange.get_performance_metrics()
print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"P99 latency: {metrics['p99_latency_ms']:.2f}ms")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Retry rate: {metrics['retry_rate']:.2%}")
```

## Integration Examples

### Unit Testing

```python
@pytest.fixture
def failing_exchange():
    """Fixture for failing exchange"""
    exchange = FailingExchange()
    exchange.set_failure_mode(FailureMode.RANDOM, probability=0.1)
    return exchange

async def test_order_retry_logic(failing_exchange):
    """Test order placement with retries"""
    order = await place_order_with_retry(
        failing_exchange,
        symbol="BTC/USDT",
        type="limit",
        side="buy",
        amount=0.1,
        price=50000
    )
    assert order is not None
    assert order['status'] in ['open', 'closed']
```

### Integration Testing

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  failing-exchange:
    image: genesis/failing-exchange
    environment:
      FAILURE_MODE: "MIXED"
      FAILURE_RATE: "0.1"
    ports:
      - "8080:8080"
  
  trading-system:
    image: genesis/trading-system
    environment:
      EXCHANGE_URL: "http://failing-exchange:8080"
      ENABLE_RETRIES: "true"
      MAX_RETRIES: "3"
    depends_on:
      - failing-exchange
  
  test-runner:
    image: genesis/test-runner
    command: pytest tests/integration/test_exchange_failures.py
    depends_on:
      - trading-system
```

## Related Documentation

- [Chaos Engineering](chaos_engineering.md)
- [Network Simulation](network_simulation.md)
- [Continuous Operation](continuous_operation.md)
- [Load Generator](load_generator.md)

---
*Last Updated: 2025-08-30*
*Version: 1.0.0*