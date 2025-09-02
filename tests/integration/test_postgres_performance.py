"""Performance benchmarks for PostgreSQL migration.

Tests read/write performance comparing SQLite vs PostgreSQL to ensure
performance meets or exceeds baseline requirements.
"""

import asyncio
import pytest
import time
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

import asyncpg
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from genesis.data.models_db import Base, AccountDB, PositionDB, OrderDB


class PerformanceBenchmark:
    """Performance benchmark utilities for database comparison."""
    
    def __init__(self):
        self.results = {
            'sqlite': {},
            'postgresql': {}
        }
    
    async def benchmark_postgres_reads(self, pool: asyncpg.Pool, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark PostgreSQL read operations.
        
        Args:
            pool: PostgreSQL connection pool
            iterations: Number of iterations for each test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Simple SELECT
        start = time.perf_counter()
        for _ in range(iterations):
            async with pool.acquire() as conn:
                await conn.fetch("SELECT * FROM accounts LIMIT 100")
        results['simple_select'] = (time.perf_counter() - start) / iterations
        
        # JOIN query
        start = time.perf_counter()
        for _ in range(iterations):
            async with pool.acquire() as conn:
                await conn.fetch("""
                    SELECT p.*, a.balance_usdt 
                    FROM positions p 
                    JOIN accounts a ON p.account_id = a.account_id 
                    WHERE p.status = 'OPEN' 
                    LIMIT 100
                """)
        results['join_query'] = (time.perf_counter() - start) / iterations
        
        # Aggregation query
        start = time.perf_counter()
        for _ in range(iterations):
            async with pool.acquire() as conn:
                await conn.fetch("""
                    SELECT account_id, COUNT(*) as position_count, 
                           SUM(CAST(dollar_value AS NUMERIC)) as total_value
                    FROM positions
                    WHERE status = 'OPEN'
                    GROUP BY account_id
                """)
        results['aggregation'] = (time.perf_counter() - start) / iterations
        
        # Complex query with subquery
        start = time.perf_counter()
        for _ in range(iterations // 10):  # Fewer iterations for complex query
            async with pool.acquire() as conn:
                await conn.fetch("""
                    SELECT a.*, 
                           (SELECT COUNT(*) FROM positions WHERE account_id = a.account_id) as position_count,
                           (SELECT SUM(CAST(pnl_dollars AS NUMERIC)) 
                            FROM positions 
                            WHERE account_id = a.account_id AND status = 'CLOSED') as total_pnl
                    FROM accounts a
                    WHERE a.tier = 'SNIPER'
                """)
        results['complex_query'] = (time.perf_counter() - start) / (iterations // 10)
        
        return results
    
    def benchmark_sqlite_reads(self, conn: sqlite3.Connection, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark SQLite read operations.
        
        Args:
            conn: SQLite connection
            iterations: Number of iterations for each test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        cursor = conn.cursor()
        
        # Simple SELECT
        start = time.perf_counter()
        for _ in range(iterations):
            cursor.execute("SELECT * FROM accounts LIMIT 100")
            cursor.fetchall()
        results['simple_select'] = (time.perf_counter() - start) / iterations
        
        # JOIN query
        start = time.perf_counter()
        for _ in range(iterations):
            cursor.execute("""
                SELECT p.*, a.balance_usdt 
                FROM positions p 
                JOIN accounts a ON p.account_id = a.account_id 
                WHERE p.status = 'OPEN' 
                LIMIT 100
            """)
            cursor.fetchall()
        results['join_query'] = (time.perf_counter() - start) / iterations
        
        # Aggregation query
        start = time.perf_counter()
        for _ in range(iterations):
            cursor.execute("""
                SELECT account_id, COUNT(*) as position_count, 
                       SUM(CAST(dollar_value AS REAL)) as total_value
                FROM positions
                WHERE status = 'OPEN'
                GROUP BY account_id
            """)
            cursor.fetchall()
        results['aggregation'] = (time.perf_counter() - start) / iterations
        
        # Complex query with subquery
        start = time.perf_counter()
        for _ in range(iterations // 10):
            cursor.execute("""
                SELECT a.*, 
                       (SELECT COUNT(*) FROM positions WHERE account_id = a.account_id) as position_count,
                       (SELECT SUM(CAST(pnl_dollars AS REAL)) 
                        FROM positions 
                        WHERE account_id = a.account_id AND status = 'CLOSED') as total_pnl
                FROM accounts a
                WHERE a.tier = 'SNIPER'
            """)
            cursor.fetchall()
        results['complex_query'] = (time.perf_counter() - start) / (iterations // 10)
        
        return results
    
    async def benchmark_postgres_writes(self, pool: asyncpg.Pool, iterations: int = 100) -> Dict[str, float]:
        """Benchmark PostgreSQL write operations.
        
        Args:
            pool: PostgreSQL connection pool
            iterations: Number of iterations for each test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Single INSERT
        start = time.perf_counter()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i in range(iterations):
                    await conn.execute("""
                        INSERT INTO orders 
                        (order_id, account_id, client_order_id, symbol, side, type, quantity, status, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (order_id) DO NOTHING
                    """, 
                    f"bench_order_{i}", "test_account", f"client_{i}", 
                    "BTC/USDT", "BUY", "MARKET", "0.001", "PENDING", datetime.utcnow())
        results['single_insert'] = (time.perf_counter() - start) / iterations
        
        # Batch INSERT
        start = time.perf_counter()
        batch_data = [
            (f"bench_batch_{i}", "test_account", f"batch_client_{i}", 
             "ETH/USDT", "SELL", "LIMIT", "0.1", "PENDING", datetime.utcnow())
            for i in range(iterations)
        ]
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany("""
                    INSERT INTO orders 
                    (order_id, account_id, client_order_id, symbol, side, type, quantity, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (order_id) DO NOTHING
                """, batch_data)
        results['batch_insert'] = (time.perf_counter() - start) / iterations
        
        # UPDATE operations
        start = time.perf_counter()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i in range(iterations):
                    await conn.execute("""
                        UPDATE orders 
                        SET status = 'FILLED', executed_at = $1
                        WHERE order_id = $2
                    """, datetime.utcnow(), f"bench_order_{i}")
        results['update'] = (time.perf_counter() - start) / iterations
        
        # DELETE operations
        start = time.perf_counter()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("""
                    DELETE FROM orders 
                    WHERE order_id LIKE 'bench_%'
                """)
        results['delete'] = time.perf_counter() - start
        
        return results
    
    def benchmark_sqlite_writes(self, conn: sqlite3.Connection, iterations: int = 100) -> Dict[str, float]:
        """Benchmark SQLite write operations.
        
        Args:
            conn: SQLite connection
            iterations: Number of iterations for each test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        cursor = conn.cursor()
        
        # Single INSERT
        start = time.perf_counter()
        for i in range(iterations):
            cursor.execute("""
                INSERT OR IGNORE INTO orders 
                (order_id, account_id, client_order_id, symbol, side, type, quantity, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            (f"bench_order_{i}", "test_account", f"client_{i}", 
             "BTC/USDT", "BUY", "MARKET", "0.001", "PENDING", datetime.utcnow()))
        conn.commit()
        results['single_insert'] = (time.perf_counter() - start) / iterations
        
        # Batch INSERT
        start = time.perf_counter()
        batch_data = [
            (f"bench_batch_{i}", "test_account", f"batch_client_{i}", 
             "ETH/USDT", "SELL", "LIMIT", "0.1", "PENDING", datetime.utcnow())
            for i in range(iterations)
        ]
        cursor.executemany("""
            INSERT OR IGNORE INTO orders 
            (order_id, account_id, client_order_id, symbol, side, type, quantity, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_data)
        conn.commit()
        results['batch_insert'] = (time.perf_counter() - start) / iterations
        
        # UPDATE operations
        start = time.perf_counter()
        for i in range(iterations):
            cursor.execute("""
                UPDATE orders 
                SET status = 'FILLED', executed_at = ?
                WHERE order_id = ?
            """, (datetime.utcnow(), f"bench_order_{i}"))
        conn.commit()
        results['update'] = (time.perf_counter() - start) / iterations
        
        # DELETE operations
        start = time.perf_counter()
        cursor.execute("DELETE FROM orders WHERE order_id LIKE 'bench_%'")
        conn.commit()
        results['delete'] = time.perf_counter() - start
        
        return results
    
    async def benchmark_concurrent_operations(self, pool: asyncpg.Pool, concurrent_tasks: int = 50) -> Dict[str, float]:
        """Benchmark concurrent database operations.
        
        Args:
            pool: PostgreSQL connection pool
            concurrent_tasks: Number of concurrent tasks
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        # Concurrent reads
        async def read_task(task_id: int):
            async with pool.acquire() as conn:
                await conn.fetch(f"""
                    SELECT * FROM positions 
                    WHERE account_id = 'test_account_{task_id % 10}'
                    LIMIT 10
                """)
        
        start = time.perf_counter()
        await asyncio.gather(*[read_task(i) for i in range(concurrent_tasks)])
        results['concurrent_reads'] = (time.perf_counter() - start) / concurrent_tasks
        
        # Concurrent writes
        async def write_task(task_id: int):
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO orders 
                    (order_id, account_id, client_order_id, symbol, side, type, quantity, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (order_id) DO NOTHING
                """,
                f"concurrent_{task_id}", f"test_account_{task_id % 10}", f"client_concurrent_{task_id}",
                "BTC/USDT", "BUY", "MARKET", "0.001", "PENDING", datetime.utcnow())
        
        start = time.perf_counter()
        await asyncio.gather(*[write_task(i) for i in range(concurrent_tasks)])
        results['concurrent_writes'] = (time.perf_counter() - start) / concurrent_tasks
        
        # Mixed operations
        async def mixed_task(task_id: int):
            async with pool.acquire() as conn:
                if task_id % 2 == 0:
                    # Read operation
                    await conn.fetch("SELECT * FROM accounts LIMIT 1")
                else:
                    # Write operation
                    await conn.execute("""
                        UPDATE positions 
                        SET current_price = $1, updated_at = $2
                        WHERE position_id = $3
                    """, "50000.00", datetime.utcnow(), f"pos_{task_id}")
        
        start = time.perf_counter()
        await asyncio.gather(*[mixed_task(i) for i in range(concurrent_tasks)])
        results['mixed_operations'] = (time.perf_counter() - start) / concurrent_tasks
        
        # Cleanup
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM orders WHERE order_id LIKE 'concurrent_%'")
        
        return results
    
    def compare_results(self, sqlite_results: Dict, postgres_results: Dict) -> Dict[str, Any]:
        """Compare SQLite and PostgreSQL benchmark results.
        
        Args:
            sqlite_results: SQLite benchmark results
            postgres_results: PostgreSQL benchmark results
            
        Returns:
            Comparison summary
        """
        comparison = {}
        
        for operation in sqlite_results:
            if operation in postgres_results:
                sqlite_time = sqlite_results[operation]
                postgres_time = postgres_results[operation]
                
                if sqlite_time > 0:
                    improvement = ((sqlite_time - postgres_time) / sqlite_time) * 100
                else:
                    improvement = 0
                
                comparison[operation] = {
                    'sqlite_ms': sqlite_time * 1000,
                    'postgres_ms': postgres_time * 1000,
                    'improvement_percent': improvement,
                    'faster': 'PostgreSQL' if improvement > 0 else 'SQLite'
                }
        
        # Calculate overall improvement
        total_sqlite = sum(r['sqlite_ms'] for r in comparison.values())
        total_postgres = sum(r['postgres_ms'] for r in comparison.values())
        
        comparison['overall'] = {
            'total_sqlite_ms': total_sqlite,
            'total_postgres_ms': total_postgres,
            'improvement_percent': ((total_sqlite - total_postgres) / total_sqlite) * 100 if total_sqlite > 0 else 0
        }
        
        return comparison


@pytest.fixture
async def postgres_pool():
    """Create PostgreSQL connection pool for testing."""
    pool = await asyncpg.create_pool(
        host='localhost',
        port=5432,
        database='genesis_trading_test',
        user='genesis',
        password='test_password',
        min_size=5,
        max_size=20
    )
    yield pool
    await pool.close()


@pytest.fixture
def sqlite_conn():
    """Create SQLite connection for testing."""
    conn = sqlite3.connect('test_genesis.db')
    yield conn
    conn.close()


@pytest.fixture
def benchmark():
    """Create benchmark instance."""
    return PerformanceBenchmark()


@pytest.mark.asyncio
async def test_read_performance(postgres_pool, sqlite_conn, benchmark):
    """Test and compare read performance."""
    # Run PostgreSQL benchmarks
    postgres_results = await benchmark.benchmark_postgres_reads(postgres_pool, iterations=100)
    
    # Run SQLite benchmarks
    sqlite_results = benchmark.benchmark_sqlite_reads(sqlite_conn, iterations=100)
    
    # Compare results
    comparison = benchmark.compare_results(sqlite_results, postgres_results)
    
    # Assert PostgreSQL is faster for complex queries
    assert comparison['complex_query']['improvement_percent'] > 20, \
        "PostgreSQL should be >20% faster for complex queries"
    
    # Log results
    print("\n=== READ PERFORMANCE COMPARISON ===")
    for op, result in comparison.items():
        if op != 'overall':
            print(f"{op}: SQLite={result['sqlite_ms']:.2f}ms, "
                  f"PostgreSQL={result['postgres_ms']:.2f}ms, "
                  f"Improvement={result['improvement_percent']:.1f}%")


@pytest.mark.asyncio
async def test_write_performance(postgres_pool, sqlite_conn, benchmark):
    """Test and compare write performance."""
    # Run PostgreSQL benchmarks
    postgres_results = await benchmark.benchmark_postgres_writes(postgres_pool, iterations=50)
    
    # Run SQLite benchmarks
    sqlite_results = benchmark.benchmark_sqlite_writes(sqlite_conn, iterations=50)
    
    # Compare results
    comparison = benchmark.compare_results(sqlite_results, postgres_results)
    
    # Assert PostgreSQL is faster for batch operations
    assert comparison['batch_insert']['improvement_percent'] > 30, \
        "PostgreSQL should be >30% faster for batch inserts"
    
    # Log results
    print("\n=== WRITE PERFORMANCE COMPARISON ===")
    for op, result in comparison.items():
        if op != 'overall':
            print(f"{op}: SQLite={result['sqlite_ms']:.2f}ms, "
                  f"PostgreSQL={result['postgres_ms']:.2f}ms, "
                  f"Improvement={result['improvement_percent']:.1f}%")


@pytest.mark.asyncio
async def test_concurrent_performance(postgres_pool, benchmark):
    """Test PostgreSQL concurrent operation performance."""
    results = await benchmark.benchmark_concurrent_operations(postgres_pool, concurrent_tasks=100)
    
    # Assert concurrent operations are performant
    assert results['concurrent_reads'] < 0.01, \
        "Concurrent reads should average <10ms"
    assert results['concurrent_writes'] < 0.02, \
        "Concurrent writes should average <20ms"
    
    # Log results
    print("\n=== CONCURRENT OPERATIONS PERFORMANCE ===")
    for op, time_sec in results.items():
        print(f"{op}: {time_sec * 1000:.2f}ms average")


@pytest.mark.asyncio
async def test_transaction_performance(postgres_pool):
    """Test PostgreSQL transaction performance."""
    iterations = 100
    
    # Test transaction commit time
    start = time.perf_counter()
    for _ in range(iterations):
        async with postgres_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT 1")
    
    avg_transaction_time = (time.perf_counter() - start) / iterations
    
    assert avg_transaction_time < 0.005, \
        f"Transaction overhead should be <5ms, got {avg_transaction_time * 1000:.2f}ms"
    
    print(f"\nAverage transaction time: {avg_transaction_time * 1000:.2f}ms")


@pytest.mark.asyncio
async def test_index_performance(postgres_pool):
    """Test PostgreSQL index effectiveness."""
    async with postgres_pool.acquire() as conn:
        # Test indexed query
        start = time.perf_counter()
        await conn.fetch("""
            SELECT * FROM positions 
            WHERE account_id = 'test_account' AND status = 'OPEN'
        """)
        indexed_time = time.perf_counter() - start
        
        # Force sequential scan for comparison
        await conn.execute("SET enable_indexscan = OFF")
        start = time.perf_counter()
        await conn.fetch("""
            SELECT * FROM positions 
            WHERE account_id = 'test_account' AND status = 'OPEN'
        """)
        sequential_time = time.perf_counter() - start
        await conn.execute("SET enable_indexscan = ON")
        
        # Index should be significantly faster
        improvement = ((sequential_time - indexed_time) / sequential_time) * 100 if sequential_time > 0 else 0
        
        assert improvement > 50, \
            f"Indexed query should be >50% faster, got {improvement:.1f}%"
        
        print(f"\nIndex performance improvement: {improvement:.1f}%")


if __name__ == "__main__":
    # Run benchmarks directly
    async def main():
        benchmark = PerformanceBenchmark()
        
        # Setup test connections
        pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            database='genesis_trading_test',
            user='genesis',
            password='test_password'
        )
        
        conn = sqlite3.connect('test_genesis.db')
        
        try:
            # Run all benchmarks
            print("Running performance benchmarks...")
            
            postgres_reads = await benchmark.benchmark_postgres_reads(pool)
            sqlite_reads = benchmark.benchmark_sqlite_reads(conn)
            
            postgres_writes = await benchmark.benchmark_postgres_writes(pool)
            sqlite_writes = benchmark.benchmark_sqlite_writes(conn)
            
            concurrent_results = await benchmark.benchmark_concurrent_operations(pool)
            
            # Compare and display results
            read_comparison = benchmark.compare_results(sqlite_reads, postgres_reads)
            write_comparison = benchmark.compare_results(sqlite_writes, postgres_writes)
            
            print("\n" + "=" * 60)
            print("PERFORMANCE BENCHMARK RESULTS")
            print("=" * 60)
            
            print("\nREAD OPERATIONS:")
            for op, result in read_comparison.items():
                if op != 'overall':
                    print(f"  {op}: {result['improvement_percent']:.1f}% improvement")
            
            print("\nWRITE OPERATIONS:")
            for op, result in write_comparison.items():
                if op != 'overall':
                    print(f"  {op}: {result['improvement_percent']:.1f}% improvement")
            
            print("\nCONCURRENT OPERATIONS:")
            for op, time_sec in concurrent_results.items():
                print(f"  {op}: {time_sec * 1000:.2f}ms average")
            
            print("\n" + "=" * 60)
            print(f"Overall Read Improvement: {read_comparison['overall']['improvement_percent']:.1f}%")
            print(f"Overall Write Improvement: {write_comparison['overall']['improvement_percent']:.1f}%")
            print("=" * 60)
            
        finally:
            await pool.close()
            conn.close()
    
    asyncio.run(main())