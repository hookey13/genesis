# Database Stress Testing Documentation

## Overview

Database stress testing validates database performance, reliability, and scalability under extreme load conditions. This framework tests database behavior with 1M+ records, complex queries, concurrent transactions, and various failure scenarios to ensure production readiness.

## Purpose

Database stress testing addresses critical concerns:
- Validates query performance at scale
- Tests index effectiveness with large datasets
- Verifies transaction isolation and consistency
- Measures connection pool efficiency
- Tests backup/restore with production-sized data
- Validates migration scripts under load
- Identifies deadlock and contention issues

## Architecture

### Core Components

#### 1. DatabaseStressor Class
Main orchestrator for database stress testing:
- Generates realistic test data
- Executes concurrent transactions
- Simulates various load patterns
- Measures performance metrics
- Validates data integrity

#### 2. DataGenerator
Creates realistic test data:
- Orders, trades, positions
- Market data time series
- User accounts and balances
- Audit logs and events
- Configurable data distributions

#### 3. QueryBenchmark
Benchmarks query performance:
- Single query latency
- Concurrent query throughput
- Complex join performance
- Aggregation efficiency
- Index utilization

#### 4. TransactionSimulator
Simulates production transactions:
- Order placement/cancellation
- Trade execution
- Balance updates
- Position calculations
- Audit trail generation

## Configuration

### Basic Setup

```python
from tests.stress.db_stress import DatabaseStressor

# Initialize with default settings
stressor = DatabaseStressor(
    connection_string="postgresql://localhost/genesis",
    target_records=1_000_000,
    concurrent_connections=50
)

# Run stress test
await stressor.run_stress_test()
```

### Advanced Configuration

```python
stressor = DatabaseStressor(
    connection_string="postgresql://localhost/genesis",
    target_records=10_000_000,
    concurrent_connections=100,
    connection_pool_size=200,
    query_timeout_seconds=30,
    transaction_isolation="REPEATABLE READ",
    enable_query_logging=True,
    slow_query_threshold_ms=100,
    deadlock_retry_attempts=3,
    batch_size=10000
)
```

### Database Types

Supports multiple database systems:

| Database | Configuration | Notes |
|----------|--------------|-------|
| PostgreSQL | `postgresql://host/db` | Primary production DB |
| SQLite | `sqlite:///path/to/db` | Development/testing |
| MySQL | `mysql://host/db` | Alternative deployment |
| TimescaleDB | `postgresql+timescale://` | Time-series optimization |

## Test Scenarios

### 1. Bulk Data Generation

```python
async def test_bulk_insert():
    stressor = DatabaseStressor()
    
    # Generate 1M orders
    await stressor.generate_orders(1_000_000)
    
    # Generate 10M trades
    await stressor.generate_trades(10_000_000)
    
    # Verify data integrity
    assert await stressor.verify_data_integrity()
```

### 2. Query Performance

```python
async def test_query_performance():
    stressor = DatabaseStressor()
    
    # Benchmark queries
    results = await stressor.benchmark_queries([
        "SELECT * FROM orders WHERE status = 'OPEN'",
        "SELECT SUM(quantity) FROM trades WHERE symbol = 'BTC/USDT'",
        "SELECT o.*, t.* FROM orders o JOIN trades t ON o.id = t.order_id"
    ])
    
    for query, metrics in results.items():
        print(f"Query: {query[:50]}...")
        print(f"  Avg latency: {metrics['avg_ms']:.2f}ms")
        print(f"  P99 latency: {metrics['p99_ms']:.2f}ms")
```

### 3. Concurrent Load

```python
async def test_concurrent_load():
    stressor = DatabaseStressor(concurrent_connections=100)
    
    # Simulate concurrent operations
    await stressor.simulate_concurrent_load(
        duration_seconds=300,
        operations_per_second=1000,
        operation_mix={
            "insert": 0.3,
            "update": 0.3,
            "select": 0.3,
            "delete": 0.1
        }
    )
    
    metrics = stressor.get_metrics()
    print(f"Total operations: {metrics['total_operations']}")
    print(f"Successful: {metrics['successful_operations']}")
    print(f"Failed: {metrics['failed_operations']}")
    print(f"Deadlocks: {metrics['deadlock_count']}")
```

### 4. Index Effectiveness

```python
async def test_index_effectiveness():
    stressor = DatabaseStressor()
    
    # Test without indexes
    await stressor.drop_indexes()
    without_indexes = await stressor.benchmark_queries(test_queries)
    
    # Test with indexes
    await stressor.create_indexes()
    with_indexes = await stressor.benchmark_queries(test_queries)
    
    # Compare performance
    for query in test_queries:
        speedup = without_indexes[query]['avg_ms'] / with_indexes[query]['avg_ms']
        print(f"Query speedup with index: {speedup:.2f}x")
```

### 5. Backup and Restore

```python
async def test_backup_restore():
    stressor = DatabaseStressor()
    
    # Generate test data
    await stressor.generate_data(1_000_000)
    original_checksum = await stressor.calculate_checksum()
    
    # Perform backup
    backup_file = await stressor.backup_database()
    print(f"Backup size: {os.path.getsize(backup_file) / 1024 / 1024:.2f} MB")
    
    # Clear database
    await stressor.clear_database()
    
    # Restore from backup
    await stressor.restore_database(backup_file)
    
    # Verify data integrity
    restored_checksum = await stressor.calculate_checksum()
    assert original_checksum == restored_checksum
```

## Usage Examples

### Production Simulation

```python
import asyncio
from tests.stress.db_stress import DatabaseStressor

async def simulate_production():
    """Simulate production database load"""
    
    stressor = DatabaseStressor(
        connection_string="postgresql://localhost/genesis",
        target_records=5_000_000,
        concurrent_connections=75
    )
    
    # Generate production-like data
    await stressor.generate_production_dataset()
    
    # Simulate trading day
    await stressor.simulate_trading_day(
        duration_hours=8,
        peak_hours=[9, 10, 14, 15],  # Market open/close
        orders_per_second=100,
        trades_per_second=50
    )
    
    # Generate report
    report = stressor.generate_report()
    print(f"Average query latency: {report['avg_latency_ms']:.2f}ms")
    print(f"Peak connections: {report['peak_connections']}")
    print(f"Slow queries: {report['slow_query_count']}")

asyncio.run(simulate_production())
```

### Migration Testing

```python
async def test_migration():
    """Test database migration with production data"""
    
    stressor = DatabaseStressor()
    
    # Load production snapshot
    await stressor.load_production_snapshot()
    
    # Record pre-migration state
    pre_migration = await stressor.capture_state()
    
    # Run migration
    start_time = time.time()
    await run_migration_script("migrations/v2.0.0.sql")
    migration_time = time.time() - start_time
    
    print(f"Migration completed in {migration_time:.2f} seconds")
    
    # Verify data integrity
    post_migration = await stressor.capture_state()
    assert await verify_migration_integrity(pre_migration, post_migration)
    
    # Test performance after migration
    await stressor.benchmark_critical_queries()

asyncio.run(test_migration())
```

### Deadlock Testing

```python
async def test_deadlock_handling():
    """Test deadlock detection and recovery"""
    
    stressor = DatabaseStressor()
    
    # Create deadlock-prone scenario
    async def transaction_a():
        async with stressor.get_connection() as conn:
            await conn.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
            await asyncio.sleep(0.1)
            await conn.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
    
    async def transaction_b():
        async with stressor.get_connection() as conn:
            await conn.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 2")
            await asyncio.sleep(0.1)
            await conn.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 1")
    
    # Run concurrent transactions
    results = await asyncio.gather(
        transaction_a(),
        transaction_b(),
        return_exceptions=True
    )
    
    # Verify deadlock handling
    deadlocks = [r for r in results if "deadlock" in str(r).lower()]
    assert len(deadlocks) >= 1, "Deadlock should be detected"

asyncio.run(test_deadlock_handling())
```

## Command-Line Usage

```bash
# Basic stress test
python -m tests.stress.db_stress --records 1000000 --duration 3600

# Concurrent load test
python -m tests.stress.db_stress \
    --records 5000000 \
    --connections 100 \
    --operations-per-second 1000 \
    --duration 7200

# Benchmark specific queries
python -m tests.stress.db_stress \
    --benchmark \
    --query-file queries.sql \
    --iterations 1000

# Migration test
python -m tests.stress.db_stress \
    --migration-test \
    --snapshot production_backup.sql \
    --migration migrations/v2.0.0.sql
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --records | 1000000 | Number of records to generate |
| --connections | 50 | Concurrent database connections |
| --duration | 3600 | Test duration in seconds |
| --operations-per-second | 100 | Target operations per second |
| --database-url | env:DATABASE_URL | Database connection string |
| --benchmark | false | Run query benchmarks |
| --migration-test | false | Test migration scripts |
| --output | db_stress_report.json | Output file for results |

## Performance Metrics

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Query Latency P50 | Median query time | <10ms |
| Query Latency P99 | 99th percentile query time | <100ms |
| Throughput | Queries per second | >1000 |
| Connection Pool Efficiency | Active/total connections | >0.7 |
| Deadlock Rate | Deadlocks per 1000 transactions | <1 |
| Index Hit Rate | Index usage percentage | >95% |

### Query Analysis

```python
# Analyze slow queries
slow_queries = stressor.get_slow_queries(threshold_ms=100)
for query in slow_queries:
    print(f"Slow query ({query['duration_ms']}ms): {query['sql']}")
    print(f"Execution plan: {query['explain']}")

# Get query statistics
stats = stressor.get_query_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

### Connection Pool Monitoring

```python
# Monitor connection pool
pool_stats = stressor.get_connection_pool_stats()
print(f"Active connections: {pool_stats['active']}")
print(f"Idle connections: {pool_stats['idle']}")
print(f"Waiting requests: {pool_stats['waiting']}")
print(f"Pool efficiency: {pool_stats['efficiency']:.2%}")
```

## Best Practices

### 1. Realistic Data Distribution

```python
# Generate realistic data distributions
await stressor.generate_data_with_distribution({
    "orders": {
        "count": 1_000_000,
        "distribution": "normal",
        "peak_hours": [9, 10, 14, 15]
    },
    "trades": {
        "count": 5_000_000,
        "distribution": "poisson",
        "lambda": 100
    }
})
```

### 2. Incremental Load Testing

```python
# Gradually increase load
for connections in [10, 25, 50, 100, 200]:
    stressor = DatabaseStressor(concurrent_connections=connections)
    metrics = await stressor.run_stress_test(duration_seconds=300)
    
    if metrics['error_rate'] > 0.01:
        print(f"Database degraded at {connections} connections")
        break
```

### 3. Query Optimization

```python
# Identify and optimize slow queries
slow_queries = await stressor.identify_slow_queries()
for query in slow_queries:
    # Analyze execution plan
    plan = await stressor.explain_query(query)
    
    # Suggest optimizations
    suggestions = stressor.suggest_optimizations(plan)
    for suggestion in suggestions:
        print(f"Optimization: {suggestion}")
```

### 4. Backup Testing

```python
# Regular backup testing
async def test_backup_performance():
    sizes = [100_000, 1_000_000, 10_000_000]
    
    for size in sizes:
        await stressor.generate_data(size)
        
        start = time.time()
        backup_file = await stressor.backup_database()
        backup_time = time.time() - start
        
        backup_size = os.path.getsize(backup_file)
        print(f"{size} records: {backup_time:.2f}s, {backup_size/1024/1024:.2f}MB")
```

## Troubleshooting

### High Query Latency

1. **Check Execution Plans**
```python
plan = await stressor.explain_query(slow_query)
if "Seq Scan" in plan:
    print("Missing index detected")
```

2. **Analyze Lock Contention**
```python
locks = await stressor.get_lock_statistics()
for lock in locks:
    if lock['wait_time_ms'] > 100:
        print(f"Lock contention on {lock['table']}")
```

3. **Review Connection Pool**
```python
if pool_stats['waiting'] > 0:
    print("Connection pool exhausted, increase pool size")
```

### Deadlock Issues

1. **Identify Deadlock Patterns**
```python
deadlocks = await stressor.analyze_deadlocks()
for pattern in deadlocks['patterns']:
    print(f"Deadlock pattern: {pattern}")
```

2. **Implement Retry Logic**
```python
@retry_on_deadlock(max_attempts=3)
async def safe_transaction():
    async with stressor.get_connection() as conn:
        # Transaction logic
        pass
```

### Data Integrity Issues

1. **Verify Constraints**
```python
violations = await stressor.check_constraint_violations()
for violation in violations:
    print(f"Constraint violation: {violation}")
```

2. **Check Referential Integrity**
```python
orphans = await stressor.find_orphaned_records()
for table, count in orphans.items():
    print(f"Orphaned records in {table}: {count}")
```

## Optimization Techniques

### Index Optimization

```python
# Analyze index usage
index_stats = await stressor.analyze_index_usage()
for index in index_stats:
    if index['usage_count'] == 0:
        print(f"Unused index: {index['name']}")
    elif index['hit_rate'] < 0.5:
        print(f"Inefficient index: {index['name']}")
```

### Query Optimization

```python
# Optimize common queries
common_queries = await stressor.get_most_frequent_queries()
for query in common_queries:
    optimized = await stressor.optimize_query(query)
    print(f"Original: {query}")
    print(f"Optimized: {optimized}")
```

### Connection Pool Tuning

```python
# Find optimal pool size
optimal_size = await stressor.find_optimal_pool_size(
    min_size=10,
    max_size=200,
    target_latency_ms=50
)
print(f"Optimal pool size: {optimal_size}")
```

## Integration Examples

### CI/CD Pipeline

```yaml
name: Database Stress Test
on: [push, pull_request]

jobs:
  db-stress:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v2
      - name: Run database stress test
        run: |
          python -m tests.stress.db_stress \
            --records 100000 \
            --connections 50 \
            --duration 600
      - name: Check performance
        run: |
          python -c "
          import json
          with open('db_stress_report.json') as f:
              report = json.load(f)
          assert report['avg_latency_ms'] < 100
          assert report['error_rate'] < 0.01
          "
```

### Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: genesis
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  stress-test:
    build: .
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres/genesis
    command: >
      python -m tests.stress.db_stress
      --records 1000000
      --connections 100
      --duration 3600

volumes:
  postgres_data:
```

## Related Documentation

- [Load Generator](load_generator.md)
- [Memory Profiling](memory_profiling.md)
- [Continuous Operation](continuous_operation.md)
- [Performance Baseline](../qa/performance_baseline.md)

---
*Last Updated: 2025-08-30*
*Version: 1.0.0*