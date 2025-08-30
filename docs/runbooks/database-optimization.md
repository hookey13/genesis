# Database Query Optimization Playbook

## Slow Query Identification Procedures

### PostgreSQL Slow Query Detection

#### Enable Query Logging
```sql
-- Enable logging of slow queries
ALTER SYSTEM SET log_min_duration_statement = 50;  -- Log queries taking >50ms
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET log_temp_files = 0;
ALTER SYSTEM SET log_autovacuum_min_duration = 0;

-- Apply changes
SELECT pg_reload_conf();

-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
-- Note: Requires restart
```

#### Query Performance Statistics
```sql
-- Top 10 slowest queries by mean time
SELECT 
    queryid,
    round(mean_exec_time::numeric, 2) AS mean_ms,
    round(stddev_exec_time::numeric, 2) AS stddev_ms,
    round(max_exec_time::numeric, 2) AS max_ms,
    calls,
    round(total_exec_time::numeric, 2) AS total_ms,
    left(query, 100) AS query_preview
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Queries consuming most total time
SELECT 
    round(100.0 * total_exec_time / sum(total_exec_time) OVER (), 2) AS percent_total,
    round(total_exec_time::numeric, 2) AS total_ms,
    calls,
    round(mean_exec_time::numeric, 2) AS mean_ms,
    left(query, 100) AS query_preview
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_%'
ORDER BY total_exec_time DESC
LIMIT 10;

-- Frequently called queries
SELECT 
    calls,
    round(mean_exec_time::numeric, 2) AS mean_ms,
    round(calls * mean_exec_time::numeric, 2) AS total_impact_ms,
    left(query, 100) AS query_preview
FROM pg_stat_statements
WHERE calls > 1000
ORDER BY calls DESC
LIMIT 10;
```

#### Real-time Query Monitoring
```sql
-- Currently running queries
SELECT 
    pid,
    now() - query_start AS duration,
    state,
    query,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE state != 'idle'
    AND query NOT LIKE '%pg_stat_activity%'
ORDER BY duration DESC;

-- Blocking queries
SELECT 
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query,
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    now() - blocked.query_start AS blocked_duration
FROM pg_stat_activity AS blocked
JOIN pg_stat_activity AS blocking 
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.wait_event_type = 'Lock';

-- Kill long-running query
SELECT pg_cancel_backend(pid)  -- Gentle cancel
-- Or SELECT pg_terminate_backend(pid)  -- Force terminate
FROM pg_stat_activity
WHERE pid = <pid> AND state != 'idle';
```

### SQLite Slow Query Detection

#### Enable Query Profiling
```python
import sqlite3
import time
import logging
from contextlib import contextmanager

class QueryProfiler:
    def __init__(self, threshold_ms=50):
        self.threshold_ms = threshold_ms
        self.slow_queries = []
    
    @contextmanager
    def profile_query(self, sql, params=None):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            if duration_ms > self.threshold_ms:
                self.slow_queries.append({
                    'sql': sql,
                    'params': params,
                    'duration_ms': duration_ms,
                    'timestamp': time.time()
                })
                logging.warning(f"Slow query ({duration_ms:.1f}ms): {sql[:100]}")
    
    def get_report(self):
        return sorted(self.slow_queries, key=lambda x: x['duration_ms'], reverse=True)

# Usage
profiler = QueryProfiler(threshold_ms=10)

conn = sqlite3.connect('genesis.db')
cursor = conn.cursor()

with profiler.profile_query("SELECT * FROM trades WHERE symbol = ?", ('BTC/USDT',)):
    cursor.execute("SELECT * FROM trades WHERE symbol = ?", ('BTC/USDT',))
    results = cursor.fetchall()
```

#### SQLite Query Analysis
```sql
-- Enable query plan logging
PRAGMA query_only = 1;  -- Read-only mode for safety

-- Analyze query plan
EXPLAIN QUERY PLAN
SELECT t.*, o.*
FROM trades t
JOIN orders o ON t.order_id = o.id
WHERE t.timestamp > datetime('now', '-1 hour');

-- Get table statistics
SELECT name, COUNT(*) as row_count
FROM sqlite_master 
JOIN (
    SELECT 'orders' as tbl_name UNION ALL
    SELECT 'trades' UNION ALL
    SELECT 'positions'
) tables ON sqlite_master.name = tables.tbl_name
WHERE type = 'table';

-- Check index usage
SELECT name, tbl_name, sql
FROM sqlite_master
WHERE type = 'index';
```

## Index Optimization Strategies

### Finding Missing Indexes

#### PostgreSQL Index Analysis
```sql
-- Find missing indexes based on usage patterns
WITH table_stats AS (
    SELECT 
        schemaname,
        tablename,
        n_live_tup AS row_count,
        n_tup_ins + n_tup_upd + n_tup_del AS write_activity,
        seq_scan,
        seq_tup_read,
        idx_scan,
        idx_tup_fetch
    FROM pg_stat_user_tables
),
index_usage AS (
    SELECT 
        schemaname,
        tablename,
        round(100.0 * idx_scan / NULLIF(seq_scan + idx_scan, 0), 2) AS index_usage_percent,
        seq_scan,
        idx_scan,
        row_count,
        write_activity
    FROM table_stats
    WHERE row_count > 1000  -- Focus on larger tables
)
SELECT 
    schemaname,
    tablename,
    row_count,
    seq_scan AS sequential_scans,
    idx_scan AS index_scans,
    index_usage_percent,
    write_activity,
    CASE 
        WHEN index_usage_percent < 50 AND seq_scan > 100 THEN 'NEEDS INDEX'
        WHEN index_usage_percent > 90 AND write_activity > 10000 THEN 'OVER-INDEXED'
        ELSE 'OK'
    END AS recommendation
FROM index_usage
ORDER BY 
    CASE 
        WHEN index_usage_percent < 50 THEN 0
        ELSE 1
    END,
    seq_scan DESC;

-- Find columns that need indexes
SELECT 
    schemaname,
    tablename,
    attname AS column_name,
    n_distinct,
    correlation,
    CASE 
        WHEN n_distinct > 100 AND abs(correlation) < 0.1 THEN 'GOOD CANDIDATE'
        WHEN n_distinct BETWEEN 10 AND 100 THEN 'MAYBE'
        ELSE 'POOR CANDIDATE'
    END AS index_recommendation
FROM pg_stats
WHERE schemaname = 'public'
    AND n_distinct > 0
ORDER BY 
    CASE 
        WHEN n_distinct > 100 AND abs(correlation) < 0.1 THEN 0
        ELSE 1
    END,
    n_distinct DESC;

-- Duplicate indexes
SELECT 
    pg_size_pretty(SUM(pg_relation_size(idx))::BIGINT) AS size,
    (array_agg(idx))[1] AS idx1,
    (array_agg(idx))[2] AS idx2,
    (array_agg(idx))[3] AS idx3,
    (array_agg(idx))[4] AS idx4
FROM (
    SELECT 
        indexrelid::regclass AS idx,
        (indrelid::text ||E'\n'|| indclass::text ||E'\n'|| 
         indkey::text ||E'\n'|| COALESCE(indexprs::text,'') ||E'\n'|| 
         COALESCE(indpred::text,'')) AS KEY
    FROM pg_index
) sub
GROUP BY KEY 
HAVING COUNT(*) > 1
ORDER BY SUM(pg_relation_size(idx)) DESC;
```

### Creating Optimal Indexes

#### Index Creation Strategies
```sql
-- Basic B-tree index for equality and range queries
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);

-- Composite index for multi-column queries
CREATE INDEX idx_orders_symbol_status_created 
ON orders(symbol, status, created_at DESC);

-- Partial index for filtered queries
CREATE INDEX idx_orders_pending 
ON orders(status, created_at) 
WHERE status = 'PENDING';

-- Expression index for computed values
CREATE INDEX idx_trades_date 
ON trades(DATE(timestamp));

-- BRIN index for large time-series data
CREATE INDEX idx_trades_timestamp_brin 
ON trades USING BRIN(timestamp);

-- GIN index for JSONB columns
CREATE INDEX idx_orders_metadata 
ON orders USING GIN(metadata);

-- Covering index (PostgreSQL 11+)
CREATE INDEX idx_positions_symbol_include 
ON positions(symbol) 
INCLUDE (size, entry_price, current_price);

-- Concurrent index creation (no table lock)
CREATE INDEX CONCURRENTLY idx_large_table ON large_table(column);
```

#### SQLite Index Creation
```sql
-- Basic indexes
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_trades_order_id ON trades(order_id);

-- Composite index
CREATE INDEX idx_orders_symbol_status 
ON orders(symbol, status);

-- Expression index (SQLite 3.9.0+)
CREATE INDEX idx_trades_date 
ON trades(date(timestamp));

-- Analyze after creating indexes
ANALYZE;
```

### Index Maintenance

#### PostgreSQL Index Maintenance
```sql
-- Rebuild bloated indexes
REINDEX INDEX idx_orders_created_at;
REINDEX TABLE orders;
REINDEX DATABASE genesis;

-- Find bloated indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    indexrelid::regclass AS index,
    100 * (1 - (index_bytes / NULLIF(index_bytes_expected, 0))) AS bloat_percent
FROM (
    SELECT 
        schemaname,
        tablename,
        indexname,
        indexrelid,
        pg_relation_size(indexrelid) AS index_bytes,
        (SELECT (pg_relation_size(indexrelid::regclass) * 0.9)::bigint) AS index_bytes_expected
    FROM pg_stat_user_indexes
) AS stats
WHERE index_bytes > 10485760  -- Only indexes > 10MB
ORDER BY bloat_percent DESC;

-- Monitor index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Drop unused indexes
SELECT 
    schemaname || '.' || indexname AS index,
    'DROP INDEX ' || schemaname || '.' || indexname || ';' AS drop_statement
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND indexrelid NOT IN (
        SELECT conindid FROM pg_constraint  -- Keep constraint indexes
    );
```

## EXPLAIN ANALYZE Interpretation Guide

### PostgreSQL EXPLAIN Output

#### Understanding Query Plans
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT o.*, t.*
FROM orders o
JOIN trades t ON o.id = t.order_id
WHERE o.created_at > NOW() - INTERVAL '1 hour';
```

#### Key Metrics to Review
| Metric | Good | Concerning | Critical |
|--------|------|------------|----------|
| Planning Time | <5ms | 5-50ms | >50ms |
| Execution Time | <100ms | 100-1000ms | >1000ms |
| Rows Estimated vs Actual | Within 10x | 10-100x off | >100x off |
| Shared Buffers Hit | >99% | 90-99% | <90% |
| Temp Buffers | 0 | <1MB | >1MB |
| Disk Sorts | 0 | 1-5 | >5 |

#### Node Types and Performance
```
Good Nodes:
- Index Scan: Direct index lookup (fast)
- Index Only Scan: Even better, no table access
- Hash Join: Good for large result sets
- Merge Join: Good for pre-sorted data

Concerning Nodes:
- Bitmap Heap Scan: Multiple index lookups
- Nested Loop: OK for small sets, bad for large
- Sort: In-memory sorting

Bad Nodes:
- Seq Scan: Full table scan (slow on large tables)
- Nested Loop with Seq Scan: Very slow
- Materialize: Storing intermediate results
```

#### Example Analysis
```sql
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM orders 
WHERE symbol = 'BTC/USDT' 
  AND created_at > '2024-01-01';

-- Output Analysis:
/*
Bitmap Heap Scan on orders (cost=12.50..516.50 rows=150 width=128) 
    (actual time=0.123..2.456 rows=142 loops=1)
  Recheck Cond: ((symbol = 'BTC/USDT') AND (created_at > '2024-01-01'))
  Heap Blocks: exact=45
  Buffers: shared hit=48
  ->  Bitmap Index Scan on idx_orders_symbol_created (cost=0.00..12.46 rows=150 width=0)
        (actual time=0.089..0.089 rows=142 loops=1)
        Index Cond: ((symbol = 'BTC/USDT') AND (created_at > '2024-01-01'))
        Buffers: shared hit=3

Analysis:
- Good: Using index (idx_orders_symbol_created)
- Good: Shared buffer hit rate 100% (48/48)
- Good: Actual rows (142) close to estimate (150)
- Good: Fast execution (2.456ms)
*/
```

### SQLite EXPLAIN Output

#### Query Plan Analysis
```sql
EXPLAIN QUERY PLAN
SELECT o.*, t.*
FROM orders o
JOIN trades t ON o.id = t.order_id
WHERE o.created_at > datetime('now', '-1 hour');

-- Output:
/*
QUERY PLAN
|--SEARCH TABLE orders AS o USING INDEX idx_orders_created_at (created_at>?)
`--SEARCH TABLE trades AS t USING INDEX idx_trades_order_id (order_id=?)

Good: Both tables using indexes
*/

-- Detailed bytecode (advanced)
EXPLAIN SELECT * FROM orders WHERE symbol = 'BTC/USDT';
```

## Vacuum and Analyze Procedures

### PostgreSQL Maintenance

#### VACUUM Operations
```sql
-- Manual VACUUM (reclaim space)
VACUUM orders;
VACUUM ANALYZE orders;  -- Also update statistics
VACUUM FULL orders;  -- Aggressive, locks table

-- Monitor VACUUM progress
SELECT 
    pid,
    datname,
    phase,
    heap_blks_total,
    heap_blks_scanned,
    heap_blks_vacuumed,
    index_vacuum_count,
    max_dead_tuples,
    num_dead_tuples
FROM pg_stat_progress_vacuum;

-- Auto-vacuum settings
ALTER TABLE orders SET (
    autovacuum_vacuum_scale_factor = 0.1,  -- Vacuum when 10% dead
    autovacuum_analyze_scale_factor = 0.05,  -- Analyze when 5% changed
    autovacuum_vacuum_cost_delay = 10  -- Less aggressive
);

-- Check last vacuum/analyze time
SELECT 
    schemaname,
    tablename,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    n_dead_tup,
    n_live_tup,
    round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_percent
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;
```

#### ANALYZE Operations
```sql
-- Update table statistics
ANALYZE orders;
ANALYZE;  -- All tables

-- Check statistics accuracy
SELECT 
    tablename,
    attname,
    n_distinct,
    correlation,
    null_frac,
    avg_width
FROM pg_stats
WHERE tablename = 'orders';

-- Force statistics update after bulk operations
BEGIN;
-- Bulk insert/update/delete
COMMIT;
ANALYZE orders;
```

### SQLite Maintenance

#### VACUUM and ANALYZE
```sql
-- VACUUM (reclaim space)
VACUUM;

-- ANALYZE (update statistics)
ANALYZE;
ANALYZE orders;

-- Check database integrity
PRAGMA integrity_check;

-- Optimize database
PRAGMA optimize;

-- Auto-vacuum settings
PRAGMA auto_vacuum = FULL;  -- Set before any tables created
PRAGMA wal_autocheckpoint = 1000;  -- Checkpoint every 1000 pages
```

## Connection Pool Tuning Guidelines

### PostgreSQL Connection Pooling

#### PgBouncer Configuration
```ini
[databases]
genesis = host=localhost port=5432 dbname=genesis

[pgbouncer]
listen_addr = *
listen_port = 6432
auth_type = md5
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
pool_mode = transaction  # Most efficient for short transactions

# Performance tuning
server_lifetime = 3600
server_idle_timeout = 600
query_wait_timeout = 120
client_idle_timeout = 0
client_login_timeout = 60

# Logging
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
stats_period = 60
```

#### Application Pool Settings
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimal pool configuration
engine = create_engine(
    'postgresql://user:pass@localhost/genesis',
    poolclass=QueuePool,
    pool_size=20,  # Number of persistent connections
    max_overflow=10,  # Maximum overflow connections
    pool_timeout=30,  # Timeout waiting for connection
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Test connections before use
    echo_pool=True  # Log pool checkouts/checkins
)

# Monitor pool status
def log_pool_status():
    pool = engine.pool
    print(f"Pool size: {pool.size()}")
    print(f"Checked in: {pool.checkedin()}")
    print(f"Overflow: {pool.overflow()}")
    print(f"Total: {pool.checkedout()}")
```

### SQLite Connection Management

#### Connection Pool for SQLite
```python
import sqlite3
from queue import Queue
from contextlib import contextmanager
import threading

class SQLitePool:
    def __init__(self, database, max_connections=10):
        self.database = database
        self.max_connections = max_connections
        self.pool = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(max_connections):
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _create_connection(self):
        conn = sqlite3.connect(self.database, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)
    
    def close_all(self):
        while not self.pool.empty():
            conn = self.pool.get()
            conn.close()

# Usage
pool = SQLitePool('genesis.db', max_connections=5)

with pool.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders")
    results = cursor.fetchall()
```

## Query Optimization Patterns

### Common Optimization Techniques

#### 1. Replace Subqueries with JOINs
```sql
-- Before (slow subquery)
SELECT * FROM orders o
WHERE o.user_id IN (
    SELECT u.id FROM users u 
    WHERE u.tier = 'STRATEGIST'
);

-- After (fast join)
SELECT o.* FROM orders o
JOIN users u ON o.user_id = u.id
WHERE u.tier = 'STRATEGIST';
```

#### 2. Use EXISTS Instead of IN
```sql
-- Before (slow IN)
SELECT * FROM orders
WHERE symbol IN (
    SELECT symbol FROM positions 
    WHERE size > 0
);

-- After (fast EXISTS)
SELECT * FROM orders o
WHERE EXISTS (
    SELECT 1 FROM positions p 
    WHERE p.symbol = o.symbol AND p.size > 0
);
```

#### 3. Optimize Pagination
```sql
-- Before (slow OFFSET)
SELECT * FROM trades
ORDER BY timestamp DESC
LIMIT 20 OFFSET 10000;

-- After (fast keyset pagination)
SELECT * FROM trades
WHERE timestamp < '2024-01-01 12:00:00'
ORDER BY timestamp DESC
LIMIT 20;
```

#### 4. Batch Operations
```python
# Before (slow individual inserts)
for trade in trades:
    cursor.execute("INSERT INTO trades VALUES (?, ?, ?)", trade)

# After (fast batch insert)
cursor.executemany("INSERT INTO trades VALUES (?, ?, ?)", trades)

# Or with PostgreSQL COPY
with open('trades.csv', 'r') as f:
    cursor.copy_expert(
        "COPY trades FROM STDIN WITH CSV HEADER",
        f
    )
```

#### 5. Denormalization for Read Performance
```sql
-- Add calculated columns to avoid joins
ALTER TABLE positions ADD COLUMN current_value DECIMAL;
ALTER TABLE positions ADD COLUMN unrealized_pnl DECIMAL;

-- Update via trigger or batch job
CREATE OR REPLACE FUNCTION update_position_values()
RETURNS TRIGGER AS $$
BEGIN
    NEW.current_value = NEW.size * NEW.current_price;
    NEW.unrealized_pnl = (NEW.current_price - NEW.entry_price) * NEW.size;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER position_values_trigger
BEFORE INSERT OR UPDATE ON positions
FOR EACH ROW
EXECUTE FUNCTION update_position_values();
```

## Emergency Query Fixes

### Kill Runaway Queries
```sql
-- PostgreSQL
SELECT pg_cancel_backend(pid)  -- Gentle
FROM pg_stat_activity
WHERE state != 'idle' 
  AND query_start < NOW() - INTERVAL '5 minutes';

SELECT pg_terminate_backend(pid)  -- Force
FROM pg_stat_activity
WHERE state != 'idle' 
  AND query_start < NOW() - INTERVAL '10 minutes';
```

### Emergency Index Creation
```sql
-- Create index without blocking
CREATE INDEX CONCURRENTLY idx_emergency ON orders(created_at);

-- If that fails, create partial indexes progressively
CREATE INDEX idx_orders_2024 ON orders(created_at) 
WHERE created_at >= '2024-01-01';
```

### Clear Query Cache
```sql
-- PostgreSQL
DISCARD PLANS;
DISCARD ALL;

-- SQLite
PRAGMA shrink_memory;
```

### Reset Statistics
```sql
-- PostgreSQL
SELECT pg_stat_reset();
SELECT pg_stat_statements_reset();

-- SQLite
ANALYZE;
PRAGMA optimize;
```