"""
Full integration test for complete database stack.
Validates PgBouncer, partitioning, and performance targets.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
import time

from genesis.database import (
    DatabaseIntegration,
    PerformanceBenchmark,
    PgBouncerAdmin
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestFullDatabaseStack:
    """Complete integration test for database infrastructure."""
    
    async def test_complete_database_setup(self):
        """Test complete database setup and initialization."""
        # Initialize database integration
        db_integration = DatabaseIntegration()
        
        try:
            # Initialize all components
            await db_integration.initialize()
            
            # Verify connection
            assert db_integration._initialized
            
            # Check pool health
            pool_health = await db_integration.get_pool_health()
            assert pool_health['is_healthy']
            
            # Check partition statistics
            partition_stats = await db_integration.get_partition_stats()
            assert 'orders' in partition_stats
            assert partition_stats['orders']['total_partitions'] > 0
            
        finally:
            await db_integration.shutdown()
            
    async def test_pgbouncer_admin_interface(self):
        """Test PgBouncer admin interface functionality."""
        admin = PgBouncerAdmin()
        
        try:
            await admin.connect()
            
            # Get stats
            stats = await admin.get_stats()
            assert isinstance(stats, list)
            
            # Get pools
            pools = await admin.get_pools()
            assert isinstance(pools, list)
            
            # Get configuration
            config = await admin.get_config()
            assert isinstance(config, dict)
            assert 'pool_mode' in config
            
            # Monitor health
            health = await admin.monitor_pool_health()
            assert health['status'] in ['healthy', 'warning', 'critical']
            
        except Exception as e:
            # PgBouncer might not be running in test environment
            pytest.skip(f"PgBouncer not available: {e}")
        finally:
            await admin.disconnect()
            
    async def test_performance_benchmarks(self):
        """Test that performance meets <5ms targets."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            benchmark = PerformanceBenchmark(db_integration.db_manager)
            
            # Run connection pool benchmark
            pool_results = await benchmark.benchmark_connection_pool()
            assert pool_results['pool_efficiency']['meets_latency_target']
            assert pool_results['concurrent_connections']['p99_ms'] < 5.0
            
            # Run trading query benchmarks
            query_results = await benchmark.benchmark_trading_queries()
            assert query_results['summary']['all_queries_meet_5ms_target']
            
            # Run insert performance benchmark
            insert_results = await benchmark.benchmark_insert_performance()
            assert insert_results['single_insert']['meets_target']
            
        finally:
            await db_integration.shutdown()
            
    async def test_partition_pruning_effectiveness(self):
        """Test that partition pruning is working effectively."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            benchmark = PerformanceBenchmark(db_integration.db_manager)
            
            # Validate partition effectiveness
            validation = await benchmark.validate_partition_effectiveness()
            assert validation['validation_summary']['all_queries_use_pruning']
            
            # Benchmark pruning improvement
            pruning_results = await benchmark.benchmark_partition_pruning()
            assert pruning_results['pruning_effective']
            assert pruning_results['improvement_percent'] > 50
            
        finally:
            await db_integration.shutdown()
            
    async def test_concurrent_load_handling(self):
        """Test system under concurrent load."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            # Generate concurrent load
            async def create_order(index: int):
                order_data = {
                    'created_at': datetime.utcnow(),
                    'symbol': f'LOAD{index}/USDT',
                    'side': 'buy' if index % 2 == 0 else 'sell',
                    'type': 'limit',
                    'quantity': Decimal('1.0'),
                    'price': Decimal('100') + Decimal(index),
                    'status': 'pending',
                    'client_order_id': f'load-test-{uuid.uuid4()}',
                    'tier': 'sniper',
                    'strategy_name': 'load_test'
                }
                
                start = time.perf_counter()
                order_id = await db_integration.insert_order(order_data)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                return order_id, elapsed_ms
                
            # Create 500 concurrent orders
            tasks = [create_order(i) for i in range(500)]
            results = await asyncio.gather(*tasks)
            
            # Verify all succeeded
            order_ids = [r[0] for r in results]
            latencies = [r[1] for r in results]
            
            assert all(oid > 0 for oid in order_ids)
            
            # Check p99 latency
            latencies.sort()
            p99_index = int(len(latencies) * 0.99)
            p99_latency = latencies[p99_index]
            
            # Should handle load with reasonable latency
            assert p99_latency < 100  # 100ms is reasonable under heavy load
            
            # Verify pool stayed healthy
            pool_health = await db_integration.get_pool_health()
            assert pool_health['is_healthy']
            
        finally:
            await db_integration.shutdown()
            
    async def test_partition_management_automation(self):
        """Test automated partition management."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            partition_mgr = db_integration.partition_manager
            
            # Check current partitions
            initial_stats = await partition_mgr.get_partition_statistics('orders')
            initial_count = len(initial_stats)
            
            # Create future partitions
            await partition_mgr.create_future_partitions(months_ahead=6)
            
            # Verify partitions were created
            new_stats = await partition_mgr.get_partition_statistics('orders')
            assert len(new_stats) >= initial_count
            
            # Test partition for future date exists
            future_date = datetime.now() + timedelta(days=150)
            partition_name = f"orders_{future_date.year:04d}_{future_date.month:02d}"
            
            exists = await db_integration.db_manager.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_class 
                    WHERE relname = $1
                )
                """,
                partition_name
            )
            assert exists
            
        finally:
            await db_integration.shutdown()
            
    async def test_monitoring_and_metrics(self):
        """Test monitoring and metrics collection."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            # Let monitoring run for a bit
            await asyncio.sleep(2)
            
            # Get pool metrics
            pool_monitor = db_integration.pool_monitor
            metrics = await pool_monitor.get_pool_metrics()
            
            assert metrics.total_connections > 0
            
            # Get metrics summary
            summary = pool_monitor.get_metrics_summary(hours=1)
            if summary:  # May be empty if just started
                assert 'avg_active_connections' in summary
                
            # Export Prometheus metrics
            prometheus_output = await pool_monitor.export_metrics(format="prometheus")
            assert "pgbouncer_total_connections" in prometheus_output
            
            # Check slow queries
            slow_queries = await pool_monitor.get_slow_queries(threshold_ms=1000)
            assert isinstance(slow_queries, list)
            
        finally:
            await db_integration.shutdown()
            
    async def test_failover_and_recovery(self):
        """Test connection recovery after failure."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            # Simulate query failure
            with pytest.raises(Exception):
                await db_integration.db_manager.execute(
                    "SELECT * FROM nonexistent_table_xyz"
                )
                
            # Verify connection recovers
            health = await db_integration.db_manager.health_check()
            assert health
            
            # Verify can still execute queries
            result = await db_integration.db_manager.fetchval("SELECT 1")
            assert result == 1
            
            # Test retry logic
            async def flaky_operation():
                # This should succeed with retries
                return await db_integration.db_manager.fetchval("SELECT 2")
                
            result = await db_integration.db_manager.retry_operation(flaky_operation)
            assert result == 2
            
        finally:
            await db_integration.shutdown()
            
    async def test_end_to_end_trading_workflow(self):
        """Test complete trading workflow with all components."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            # Create order
            order_data = {
                'created_at': datetime.utcnow(),
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'type': 'limit',
                'quantity': Decimal('0.01'),
                'price': Decimal('50000'),
                'status': 'pending',
                'client_order_id': str(uuid.uuid4()),
                'tier': 'sniper',
                'strategy_name': 'test_strategy'
            }
            
            order_id = await db_integration.insert_order(order_data)
            assert order_id > 0
            
            # Update order status
            await db_integration.db_manager.execute(
                """
                UPDATE orders 
                SET status = 'filled', updated_at = NOW()
                WHERE id = $1 AND created_at >= CURRENT_DATE
                """,
                order_id
            )
            
            # Create trade
            trade_data = {
                'executed_at': datetime.utcnow(),
                'order_id': order_id,
                'symbol': 'BTC/USDT',
                'side': 'buy',
                'quantity': Decimal('0.01'),
                'price': Decimal('49999'),
                'commission': Decimal('0.00001'),
                'commission_asset': 'BTC',
                'exchange_trade_id': str(uuid.uuid4()),
                'is_maker': True,
                'realized_pnl': Decimal('-10'),
                'tier': 'sniper'
            }
            
            trade_id = await db_integration.insert_trade(trade_data)
            assert trade_id > 0
            
            # Query recent orders
            orders = await db_integration.get_recent_orders(
                symbol='BTC/USDT',
                status='filled'
            )
            assert len(orders) > 0
            
            # Get performance metrics
            today = datetime.utcnow().strftime('%Y-%m-%d')
            performance = await db_integration.get_trading_performance(
                start_date=today,
                end_date=today,
                symbol='BTC/USDT'
            )
            
            assert performance['total_trades'] >= 1
            assert performance['total_volume'] >= Decimal('0.01')
            
            # Verify queries are fast
            start = time.perf_counter()
            await db_integration.get_recent_orders(limit=100)
            query_time_ms = (time.perf_counter() - start) * 1000
            
            assert query_time_ms < 50  # Should be much faster than 50ms
            
        finally:
            await db_integration.shutdown()
            
    async def test_full_benchmark_suite(self):
        """Run and validate full benchmark suite."""
        db_integration = DatabaseIntegration()
        
        try:
            await db_integration.initialize()
            
            benchmark = PerformanceBenchmark(db_integration.db_manager)
            
            # Run complete benchmark
            results = await benchmark.run_full_benchmark()
            
            # Validate all targets met
            assert results['summary']['all_performance_targets_met']
            assert results['summary']['connection_pool_p99_ms'] < 5.0
            assert results['summary']['trading_queries_avg_p99_ms'] < 5.0
            assert results['summary']['insert_p99_ms'] < 5.0
            assert results['summary']['partition_pruning_improvement'] > 0
            
            # Check recommendation
            assert "Ready for production" in results['summary']['recommendation']
            
        finally:
            await db_integration.shutdown()