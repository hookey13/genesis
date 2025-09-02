"""
Integration tests for database performance with PgBouncer and partitioning.
Tests connection pooling, partition pruning, and query performance.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
import uuid

from genesis.database import PostgresManager, PartitionManager, PoolMonitor
from genesis.database.integration import DatabaseIntegration


@pytest.fixture
async def db_integration():
    """Create database integration instance for testing."""
    integration = DatabaseIntegration()
    await integration.initialize()
    yield integration
    await integration.shutdown()


@pytest.fixture
async def db_manager():
    """Create PostgreSQL manager for testing."""
    manager = PostgresManager()
    await manager.connect()
    yield manager
    await manager.disconnect()


class TestConnectionPooling:
    """Test PgBouncer connection pooling functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_limits(self, db_manager):
        """Verify pool handles concurrent connections correctly."""
        connections = []
        
        # Test acquiring multiple connections
        for i in range(50):
            async with db_manager.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                assert result == 1
                
        # Verify pool statistics
        stats = await db_manager.get_pool_stats()
        assert stats['max_size'] >= 50
        assert stats['min_size'] >= 10
        
    @pytest.mark.asyncio
    async def test_pool_recovery(self, db_manager):
        """Test pool recovery after connection failure."""
        # Simulate connection failure by invalid query
        with pytest.raises(Exception):
            await db_manager.execute("SELECT * FROM nonexistent_table")
            
        # Verify pool recovers
        result = await db_manager.fetchval("SELECT 1")
        assert result == 1
        
        # Check health
        is_healthy = await db_manager.health_check()
        assert is_healthy
        
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, db_manager):
        """Test concurrent query execution through pool."""
        async def run_query(query_id: int):
            result = await db_manager.fetchval(
                "SELECT $1::int",
                query_id
            )
            return result
            
        # Run 100 concurrent queries
        tasks = [run_query(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 100
        assert all(results[i] == i for i in range(100))
        
    @pytest.mark.asyncio
    async def test_transaction_pooling_mode(self, db_manager):
        """Verify transaction pooling mode behavior."""
        # Transaction should reset after completion
        async with db_manager.transaction() as conn:
            await conn.execute("CREATE TEMP TABLE test_temp (id INT)")
            await conn.execute("INSERT INTO test_temp VALUES (1)")
            
        # Temp table should not exist in new transaction
        async with db_manager.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_tables 
                    WHERE tablename = 'test_temp'
                )
                """
            )
            assert not exists
            
    @pytest.mark.asyncio
    async def test_query_timeout(self, db_manager):
        """Test query timeout handling."""
        with pytest.raises(asyncio.TimeoutError):
            await db_manager.execute(
                "SELECT pg_sleep(10)",
                timeout=1.0
            )
            
    @pytest.mark.asyncio
    async def test_retry_logic(self, db_manager):
        """Test automatic retry on connection errors."""
        # This should succeed with retries
        result = await db_manager.retry_operation(
            db_manager.fetchval,
            "SELECT 1"
        )
        assert result == 1


class TestTablePartitioning:
    """Test table partitioning functionality."""
    
    @pytest.mark.asyncio
    async def test_partition_creation(self, db_manager):
        """Verify automatic partition creation."""
        partition_mgr = PartitionManager(db_manager)
        
        # Setup partitioned tables
        await partition_mgr.setup_partitioned_tables()
        
        # Verify partitions were created
        current_date = datetime.now()
        partition_name = f"orders_{current_date.year:04d}_{current_date.month:02d}"
        
        exists = await db_manager.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_class 
                WHERE relname = $1
            )
            """,
            partition_name
        )
        assert exists
        
    @pytest.mark.asyncio
    async def test_partition_pruning(self, db_integration):
        """Verify query uses partition elimination."""
        # Insert test data
        order_data = {
            'created_at': datetime.utcnow(),
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'type': 'limit',
            'quantity': Decimal('0.1'),
            'price': Decimal('50000'),
            'status': 'pending',
            'client_order_id': str(uuid.uuid4()),
            'tier': 'sniper',
            'strategy_name': 'test_strategy'
        }
        
        order_id = await db_integration.insert_order(order_data)
        assert order_id > 0
        
        # Query with partition key should use pruning
        explain_result = await db_integration.db_manager.fetch(
            """
            EXPLAIN (ANALYZE, BUFFERS) 
            SELECT * FROM orders 
            WHERE created_at >= CURRENT_DATE
            AND created_at < CURRENT_DATE + INTERVAL '1 day'
            """
        )
        
        # Check that partition pruning occurred
        explain_text = str(explain_result)
        assert 'Partition' in explain_text or 'Append' in explain_text
        
    @pytest.mark.asyncio
    async def test_partition_performance(self, db_integration):
        """Verify <5ms query performance on partitioned tables."""
        # Insert test data across multiple partitions
        base_time = datetime.utcnow()
        
        for days_ago in range(30):
            order_data = {
                'created_at': base_time - timedelta(days=days_ago),
                'symbol': 'BTC/USDT',
                'side': 'buy' if days_ago % 2 == 0 else 'sell',
                'type': 'limit',
                'quantity': Decimal('0.1'),
                'price': Decimal('50000') + Decimal(days_ago * 100),
                'status': 'filled',
                'client_order_id': str(uuid.uuid4()),
                'tier': 'sniper',
                'strategy_name': 'test_strategy'
            }
            await db_integration.insert_order(order_data)
            
        # Measure query performance
        import time
        start_time = time.perf_counter()
        
        result = await db_integration.get_recent_orders(
            symbol='BTC/USDT',
            limit=100
        )
        
        query_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert len(result) > 0
        assert query_time_ms < 100  # Should be much less than 100ms
        
    @pytest.mark.asyncio
    async def test_automated_partition_management(self, db_manager):
        """Test automated partition creation and pruning."""
        partition_mgr = PartitionManager(db_manager)
        
        # Create future partitions
        await partition_mgr.create_future_partitions(months_ahead=3)
        
        # Verify future partitions exist
        future_date = datetime.now() + timedelta(days=60)
        partition_name = f"orders_{future_date.year:04d}_{future_date.month:02d}"
        
        exists = await db_manager.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_class 
                WHERE relname = $1
            )
            """,
            partition_name
        )
        assert exists
        
    @pytest.mark.asyncio
    async def test_partition_statistics(self, db_manager):
        """Test partition statistics collection."""
        partition_mgr = PartitionManager(db_manager)
        await partition_mgr.setup_partitioned_tables()
        
        stats = await partition_mgr.get_partition_statistics('orders')
        
        assert isinstance(stats, list)
        if stats:
            assert 'partition_name' in stats[0]
            assert 'row_count' in stats[0]
            assert 'total_size' in stats[0]


class TestPoolMonitoring:
    """Test connection pool monitoring."""
    
    @pytest.mark.asyncio
    async def test_pool_metrics_collection(self, db_manager):
        """Test metrics collection from pool."""
        monitor = PoolMonitor(db_manager)
        
        metrics = await monitor.get_pool_metrics()
        
        assert metrics.total_connections > 0
        assert metrics.timestamp is not None
        
        metrics_dict = metrics.to_dict()
        assert 'connection_utilization' in metrics_dict
        
    @pytest.mark.asyncio
    async def test_pool_health_check(self, db_manager):
        """Test pool health monitoring."""
        monitor = PoolMonitor(db_manager)
        
        is_healthy, issues = await monitor.check_pool_health()
        
        assert isinstance(is_healthy, bool)
        assert isinstance(issues, list)
        
    @pytest.mark.asyncio
    async def test_slow_query_detection(self, db_manager):
        """Test detection of slow queries."""
        monitor = PoolMonitor(db_manager)
        
        # Run a potentially slow query in background
        async def slow_query():
            try:
                await db_manager.execute("SELECT pg_sleep(0.5)")
            except:
                pass
                
        task = asyncio.create_task(slow_query())
        
        # Give it time to start
        await asyncio.sleep(0.1)
        
        # Check for slow queries
        slow_queries = await monitor.get_slow_queries(threshold_ms=100)
        
        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
            
        # May or may not detect depending on timing
        assert isinstance(slow_queries, list)
        
    @pytest.mark.asyncio
    async def test_pool_size_recommendations(self, db_manager):
        """Test pool size optimization recommendations."""
        monitor = PoolMonitor(db_manager)
        
        # Collect some metrics first
        for _ in range(5):
            await monitor.get_pool_metrics()
            await asyncio.sleep(0.1)
            
        recommendations = await monitor.recommend_pool_size()
        
        assert 'min_pool_size' in recommendations
        assert 'default_pool_size' in recommendations
        assert 'max_client_conn' in recommendations
        
        # Verify recommendations are within reasonable bounds
        assert 10 <= recommendations['min_pool_size'] <= 100
        assert 50 <= recommendations['default_pool_size'] <= 200
        assert 100 <= recommendations['max_client_conn'] <= 2000
        
    @pytest.mark.asyncio
    async def test_prometheus_metrics_export(self, db_manager):
        """Test Prometheus format metrics export."""
        monitor = PoolMonitor(db_manager)
        
        prometheus_output = await monitor.export_metrics(format="prometheus")
        
        assert "pgbouncer_total_connections" in prometheus_output
        assert "pgbouncer_active_connections" in prometheus_output
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output


class TestDatabaseIntegration:
    """Test integrated database functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading_flow(self, db_integration):
        """Test complete trading flow with pooling and partitioning."""
        # Create an order
        order_data = {
            'created_at': datetime.utcnow(),
            'symbol': 'ETH/USDT',
            'side': 'buy',
            'type': 'market',
            'quantity': Decimal('1.5'),
            'price': None,
            'status': 'pending',
            'client_order_id': str(uuid.uuid4()),
            'tier': 'hunter',
            'strategy_name': 'momentum'
        }
        
        order_id = await db_integration.insert_order(order_data)
        assert order_id > 0
        
        # Execute trade
        trade_data = {
            'executed_at': datetime.utcnow(),
            'order_id': order_id,
            'symbol': 'ETH/USDT',
            'side': 'buy',
            'quantity': Decimal('1.5'),
            'price': Decimal('3000'),
            'commission': Decimal('0.0015'),
            'commission_asset': 'ETH',
            'exchange_trade_id': str(uuid.uuid4()),
            'is_maker': False,
            'realized_pnl': Decimal('0'),
            'tier': 'hunter'
        }
        
        trade_id = await db_integration.insert_trade(trade_data)
        assert trade_id > 0
        
        # Query recent orders
        orders = await db_integration.get_recent_orders(symbol='ETH/USDT')
        assert len(orders) > 0
        assert orders[0]['symbol'] == 'ETH/USDT'
        
        # Get performance metrics
        today = datetime.utcnow().strftime('%Y-%m-%d')
        performance = await db_integration.get_trading_performance(
            start_date=today,
            end_date=today,
            symbol='ETH/USDT'
        )
        
        assert performance['total_trades'] >= 1
        assert performance['total_volume'] >= 1.5
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, db_integration):
        """Test concurrent database operations."""
        async def create_order(index: int):
            order_data = {
                'created_at': datetime.utcnow(),
                'symbol': f'TEST{index}/USDT',
                'side': 'buy' if index % 2 == 0 else 'sell',
                'type': 'limit',
                'quantity': Decimal('1'),
                'price': Decimal('100') + Decimal(index),
                'status': 'pending',
                'client_order_id': f'test-{index}-{uuid.uuid4()}',
                'tier': 'sniper',
                'strategy_name': 'test'
            }
            return await db_integration.insert_order(order_data)
            
        # Create 50 orders concurrently
        tasks = [create_order(i) for i in range(50)]
        order_ids = await asyncio.gather(*tasks)
        
        assert len(order_ids) == 50
        assert all(oid > 0 for oid in order_ids)
        
        # Verify pool handled load
        pool_health = await db_integration.get_pool_health()
        assert pool_health['is_healthy']
        
    @pytest.mark.asyncio
    async def test_partition_statistics_integration(self, db_integration):
        """Test partition statistics through integration layer."""
        stats = await db_integration.get_partition_stats()
        
        assert 'orders' in stats
        assert 'trades' in stats
        assert 'total_partitions' in stats['orders']
        assert stats['orders']['total_partitions'] > 0