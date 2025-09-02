"""
Performance benchmarking for database operations.
Validates <5ms query targets and partition effectiveness.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional
import uuid
import statistics

from .postgres_manager import PostgresManager
from .partition_manager import PartitionManager

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Benchmarks database performance to ensure <5ms query targets.
    """
    
    def __init__(self, db_manager: PostgresManager):
        """
        Initialize performance benchmark.
        
        Args:
            db_manager: PostgreSQL connection manager
        """
        self.db = db_manager
        self.results: Dict[str, List[float]] = {}
        
    async def run_query_benchmark(
        self,
        query: str,
        params: Optional[tuple] = None,
        iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark a query's performance.
        
        Args:
            query: SQL query to benchmark
            params: Query parameters
            iterations: Number of iterations to run
            warmup: Number of warmup iterations
            
        Returns:
            Performance statistics in milliseconds
        """
        timings = []
        
        # Warmup runs
        for _ in range(warmup):
            if params:
                await self.db.fetch(query, *params)
            else:
                await self.db.fetch(query)
                
        # Actual benchmark runs
        for _ in range(iterations):
            start = time.perf_counter()
            
            if params:
                await self.db.fetch(query, *params)
            else:
                await self.db.fetch(query)
                
            elapsed_ms = (time.perf_counter() - start) * 1000
            timings.append(elapsed_ms)
            
        return {
            'min_ms': min(timings),
            'max_ms': max(timings),
            'avg_ms': statistics.mean(timings),
            'median_ms': statistics.median(timings),
            'p95_ms': statistics.quantiles(timings, n=20)[18],  # 95th percentile
            'p99_ms': statistics.quantiles(timings, n=100)[98],  # 99th percentile
            'stdev_ms': statistics.stdev(timings) if len(timings) > 1 else 0,
            'iterations': iterations,
            'meets_target': statistics.quantiles(timings, n=100)[98] < 5.0  # p99 < 5ms
        }
        
    async def benchmark_partition_pruning(self) -> Dict[str, Any]:
        """
        Benchmark partition pruning effectiveness.
        
        Returns:
            Comparison of query performance with and without partition pruning
        """
        results = {}
        
        # Test query WITH partition pruning (uses partition key)
        query_with_pruning = """
        SELECT COUNT(*) as count, AVG(quantity) as avg_quantity
        FROM orders
        WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
        AND created_at < CURRENT_DATE
        AND symbol = 'BTC/USDT'
        """
        
        # Test query WITHOUT effective pruning (no partition key in WHERE)
        query_without_pruning = """
        SELECT COUNT(*) as count, AVG(quantity) as avg_quantity
        FROM orders
        WHERE symbol = 'BTC/USDT'
        AND status = 'filled'
        """
        
        # Benchmark with pruning
        logger.info("Benchmarking query WITH partition pruning...")
        with_pruning = await self.run_query_benchmark(query_with_pruning)
        results['with_pruning'] = with_pruning
        
        # Benchmark without pruning
        logger.info("Benchmarking query WITHOUT partition pruning...")
        without_pruning = await self.run_query_benchmark(query_without_pruning)
        results['without_pruning'] = without_pruning
        
        # Calculate improvement
        if without_pruning['avg_ms'] > 0:
            improvement = (
                (without_pruning['avg_ms'] - with_pruning['avg_ms']) / 
                without_pruning['avg_ms'] * 100
            )
        else:
            improvement = 0
            
        results['improvement_percent'] = improvement
        results['pruning_effective'] = improvement > 50  # At least 50% improvement
        
        # Get EXPLAIN plans to verify pruning
        explain_with = await self.db.fetch(f"EXPLAIN (ANALYZE, BUFFERS) {query_with_pruning}")
        explain_without = await self.db.fetch(f"EXPLAIN (ANALYZE, BUFFERS) {query_without_pruning}")
        
        results['explain_with_pruning'] = str(explain_with[0]['QUERY PLAN']) if explain_with else ""
        results['explain_without_pruning'] = str(explain_without[0]['QUERY PLAN']) if explain_without else ""
        
        return results
        
    async def benchmark_connection_pool(self) -> Dict[str, Any]:
        """
        Benchmark connection pool performance.
        
        Returns:
            Connection pool performance metrics
        """
        results = {}
        
        # Test single connection performance
        single_timings = []
        for _ in range(50):
            start = time.perf_counter()
            async with self.db.acquire() as conn:
                await conn.fetchval("SELECT 1")
            elapsed_ms = (time.perf_counter() - start) * 1000
            single_timings.append(elapsed_ms)
            
        results['single_connection'] = {
            'avg_ms': statistics.mean(single_timings),
            'p99_ms': statistics.quantiles(single_timings, n=100)[98]
        }
        
        # Test concurrent connections
        async def concurrent_query(query_id: int):
            start = time.perf_counter()
            async with self.db.acquire() as conn:
                await conn.fetchval("SELECT $1::int", query_id)
            return (time.perf_counter() - start) * 1000
            
        # Run 100 concurrent queries
        tasks = [concurrent_query(i) for i in range(100)]
        concurrent_timings = await asyncio.gather(*tasks)
        
        results['concurrent_connections'] = {
            'avg_ms': statistics.mean(concurrent_timings),
            'p99_ms': statistics.quantiles(concurrent_timings, n=100)[98],
            'max_ms': max(concurrent_timings),
            'total_queries': len(concurrent_timings)
        }
        
        # Calculate pool efficiency
        results['pool_efficiency'] = {
            'meets_latency_target': results['concurrent_connections']['p99_ms'] < 5.0,
            'connection_overhead_ms': results['single_connection']['avg_ms'],
            'concurrent_speedup': (
                results['single_connection']['avg_ms'] * 100 / 
                sum(concurrent_timings)
            ) if sum(concurrent_timings) > 0 else 0
        }
        
        return results
        
    async def benchmark_trading_queries(self) -> Dict[str, Dict]:
        """
        Benchmark common trading queries.
        
        Returns:
            Performance metrics for each query type
        """
        results = {}
        
        # Prepare test data
        test_symbol = 'BENCH/USDT'
        test_order_id = str(uuid.uuid4())
        
        # Insert test order
        await self.db.execute("""
            INSERT INTO orders (
                created_at, symbol, side, type, quantity, price,
                status, client_order_id, tier, strategy_name
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            ) ON CONFLICT (client_order_id) DO NOTHING
        """, datetime.utcnow(), test_symbol, 'buy', 'limit', 
        Decimal('1.0'), Decimal('50000'), 'pending', test_order_id,
        'sniper', 'benchmark_test')
        
        # Benchmark different query patterns
        queries = {
            'recent_orders': (
                """
                SELECT * FROM orders
                WHERE created_at >= CURRENT_DATE - INTERVAL '1 hour'
                ORDER BY created_at DESC
                LIMIT 100
                """,
                None
            ),
            'pending_orders': (
                """
                SELECT * FROM orders
                WHERE status = 'pending'
                AND created_at >= CURRENT_DATE - INTERVAL '1 day'
                ORDER BY created_at DESC
                """,
                None
            ),
            'order_by_client_id': (
                """
                SELECT * FROM orders
                WHERE client_order_id = $1
                """,
                (test_order_id,)
            ),
            'symbol_aggregation': (
                """
                SELECT 
                    symbol,
                    COUNT(*) as order_count,
                    SUM(quantity) as total_quantity
                FROM orders
                WHERE created_at >= CURRENT_DATE
                GROUP BY symbol
                """,
                None
            ),
            'trade_performance': (
                """
                SELECT 
                    DATE(executed_at) as trade_date,
                    COUNT(*) as trade_count,
                    SUM(realized_pnl) as total_pnl
                FROM trades
                WHERE executed_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(executed_at)
                ORDER BY trade_date DESC
                """,
                None
            )
        }
        
        for query_name, (query, params) in queries.items():
            logger.info(f"Benchmarking {query_name}...")
            results[query_name] = await self.run_query_benchmark(
                query, params, iterations=50, warmup=5
            )
            
        # Summary
        all_meet_target = all(r['meets_target'] for r in results.values())
        avg_p99 = statistics.mean(r['p99_ms'] for r in results.values())
        
        results['summary'] = {
            'all_queries_meet_5ms_target': all_meet_target,
            'average_p99_ms': avg_p99,
            'queries_tested': len(queries)
        }
        
        return results
        
    async def benchmark_insert_performance(self) -> Dict[str, Any]:
        """
        Benchmark insert performance with partitioning.
        
        Returns:
            Insert performance metrics
        """
        results = {}
        
        # Single insert benchmark
        single_timings = []
        for i in range(100):
            order_data = {
                'created_at': datetime.utcnow(),
                'symbol': f'TEST{i}/USDT',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'type': 'limit',
                'quantity': Decimal('1.0'),
                'price': Decimal('50000') + Decimal(i),
                'status': 'pending',
                'client_order_id': str(uuid.uuid4()),
                'tier': 'sniper',
                'strategy_name': 'benchmark'
            }
            
            start = time.perf_counter()
            await self.db.execute("""
                INSERT INTO orders (
                    created_at, symbol, side, type, quantity, price,
                    status, client_order_id, tier, strategy_name
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, *order_data.values())
            elapsed_ms = (time.perf_counter() - start) * 1000
            single_timings.append(elapsed_ms)
            
        results['single_insert'] = {
            'avg_ms': statistics.mean(single_timings),
            'p99_ms': statistics.quantiles(single_timings, n=100)[98],
            'meets_target': statistics.quantiles(single_timings, n=100)[98] < 5.0
        }
        
        # Batch insert benchmark
        batch_data = []
        for i in range(1000):
            batch_data.append((
                datetime.utcnow(),
                f'BATCH{i}/USDT',
                'buy' if i % 2 == 0 else 'sell',
                'market',
                Decimal('0.5'),
                None,  # market orders have no price
                'filled',
                str(uuid.uuid4()),
                'hunter',
                'batch_test'
            ))
            
        start = time.perf_counter()
        await self.db.execute_many("""
            INSERT INTO orders (
                created_at, symbol, side, type, quantity, price,
                status, client_order_id, tier, strategy_name
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, batch_data)
        batch_time_ms = (time.perf_counter() - start) * 1000
        
        results['batch_insert'] = {
            'total_ms': batch_time_ms,
            'records': len(batch_data),
            'avg_per_record_ms': batch_time_ms / len(batch_data),
            'records_per_second': len(batch_data) / (batch_time_ms / 1000)
        }
        
        return results
        
    async def validate_partition_effectiveness(self) -> Dict[str, Any]:
        """
        Validate that partition pruning is working effectively.
        
        Returns:
            Partition effectiveness validation results
        """
        results = {}
        
        # Check partition elimination for date-range queries
        test_queries = [
            {
                'name': 'single_day_query',
                'query': """
                    EXPLAIN (ANALYZE, BUFFERS) 
                    SELECT COUNT(*) FROM orders 
                    WHERE created_at >= '2025-09-02'::date 
                    AND created_at < '2025-09-03'::date
                """
            },
            {
                'name': 'current_month_query',
                'query': """
                    EXPLAIN (ANALYZE, BUFFERS)
                    SELECT COUNT(*) FROM orders
                    WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
                    AND created_at < DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
                """
            },
            {
                'name': 'cross_partition_query',
                'query': """
                    EXPLAIN (ANALYZE, BUFFERS)
                    SELECT COUNT(*) FROM orders
                    WHERE created_at >= CURRENT_DATE - INTERVAL '3 months'
                """
            }
        ]
        
        for test in test_queries:
            explain_result = await self.db.fetch(test['query'])
            explain_text = str(explain_result) if explain_result else ""
            
            # Check for partition pruning indicators
            has_partition_pruning = (
                'partitions' in explain_text.lower() or
                'Append' in explain_text or
                'Parallel Append' in explain_text
            )
            
            # Extract execution time from EXPLAIN ANALYZE
            execution_time = 0
            if 'Execution Time:' in explain_text:
                try:
                    time_str = explain_text.split('Execution Time:')[1].split('ms')[0].strip()
                    execution_time = float(time_str)
                except:
                    pass
                    
            results[test['name']] = {
                'has_partition_pruning': has_partition_pruning,
                'execution_time_ms': execution_time,
                'meets_5ms_target': execution_time < 5.0 if execution_time > 0 else False,
                'explain_snippet': explain_text[:500] if explain_text else "No explain output"
            }
            
        # Overall validation
        results['validation_summary'] = {
            'all_queries_use_pruning': all(
                r['has_partition_pruning'] for r in results.values() 
                if isinstance(r, dict)
            ),
            'all_meet_performance_target': all(
                r['meets_5ms_target'] for r in results.values()
                if isinstance(r, dict) and 'meets_5ms_target' in r
            )
        }
        
        return results
        
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete performance benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        logger.info("Starting full performance benchmark...")
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'benchmarks': {}
        }
        
        # Run all benchmarks
        logger.info("1. Benchmarking connection pool...")
        results['benchmarks']['connection_pool'] = await self.benchmark_connection_pool()
        
        logger.info("2. Benchmarking partition pruning...")
        results['benchmarks']['partition_pruning'] = await self.benchmark_partition_pruning()
        
        logger.info("3. Benchmarking trading queries...")
        results['benchmarks']['trading_queries'] = await self.benchmark_trading_queries()
        
        logger.info("4. Benchmarking insert performance...")
        results['benchmarks']['insert_performance'] = await self.benchmark_insert_performance()
        
        logger.info("5. Validating partition effectiveness...")
        results['benchmarks']['partition_validation'] = await self.validate_partition_effectiveness()
        
        # Overall summary
        all_targets_met = (
            results['benchmarks']['connection_pool']['pool_efficiency']['meets_latency_target'] and
            results['benchmarks']['trading_queries']['summary']['all_queries_meet_5ms_target'] and
            results['benchmarks']['insert_performance']['single_insert']['meets_target']
        )
        
        results['summary'] = {
            'all_performance_targets_met': all_targets_met,
            'connection_pool_p99_ms': results['benchmarks']['connection_pool']['concurrent_connections']['p99_ms'],
            'trading_queries_avg_p99_ms': results['benchmarks']['trading_queries']['summary']['average_p99_ms'],
            'insert_p99_ms': results['benchmarks']['insert_performance']['single_insert']['p99_ms'],
            'partition_pruning_improvement': results['benchmarks']['partition_pruning'].get('improvement_percent', 0),
            'recommendation': "Performance targets MET - Ready for production" if all_targets_met else "Performance optimization needed"
        }
        
        logger.info(f"Benchmark complete. Overall result: {results['summary']['recommendation']}")
        
        return results