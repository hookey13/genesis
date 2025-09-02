"""
Unit tests for PgBouncer connection pooling functionality.
Tests pool limits, recovery, and performance characteristics.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from genesis.database.postgres_manager import PostgresManager, DatabaseConfig
from genesis.database.pool_monitor import PoolMonitor, PoolMetrics


class TestPostgresManager:
    """Test PostgreSQL manager with connection pooling."""
    
    @pytest.fixture
    def db_config(self):
        """Create test database configuration."""
        return DatabaseConfig(
            host="localhost",
            port=6432,
            database="test_db",
            user="test_user",
            password="test_pass",
            min_pool_size=5,
            max_pool_size=20
        )
        
    @pytest.fixture
    def db_manager(self, db_config):
        """Create PostgreSQL manager instance."""
        return PostgresManager(db_config)
        
    @pytest.mark.asyncio
    async def test_connection_initialization(self, db_manager):
        """Test connection pool initialization."""
        with patch('asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool.return_value = Mock()
            
            await db_manager.connect()
            
            assert db_manager._is_connected
            mock_pool.assert_called_once()
            
            # Verify connection parameters
            call_kwargs = mock_pool.call_args.kwargs
            assert call_kwargs['host'] == 'localhost'
            assert call_kwargs['port'] == 6432
            assert call_kwargs['min_size'] == 5
            assert call_kwargs['max_size'] == 20
            
    @pytest.mark.asyncio
    async def test_connection_acquire(self, db_manager):
        """Test acquiring connection from pool."""
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        db_manager._pool = mock_pool
        db_manager._is_connected = True
        
        async with db_manager.acquire() as conn:
            assert conn == mock_conn
            
        mock_pool.acquire.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_execute_query(self, db_manager):
        """Test query execution through pool."""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "INSERT 0 1"
        
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        db_manager._pool = mock_pool
        db_manager._is_connected = True
        
        result = await db_manager.execute("INSERT INTO test VALUES (1)")
        
        assert result == "INSERT 0 1"
        mock_conn.execute.assert_called_once_with("INSERT INTO test VALUES (1)", timeout=None)
        
    @pytest.mark.asyncio
    async def test_fetch_query(self, db_manager):
        """Test fetching results through pool."""
        mock_records = [{'id': 1, 'value': 'test'}]
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_records
        
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        db_manager._pool = mock_pool
        db_manager._is_connected = True
        
        results = await db_manager.fetch("SELECT * FROM test")
        
        assert results == mock_records
        mock_conn.fetch.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_transaction_context(self, db_manager):
        """Test transaction context manager."""
        mock_transaction = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.transaction.return_value = mock_transaction
        
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        db_manager._pool = mock_pool
        db_manager._is_connected = True
        
        async with db_manager.transaction():
            pass
            
        mock_conn.transaction.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_health_check(self, db_manager):
        """Test database health check."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        db_manager._pool = mock_pool
        db_manager._is_connected = True
        
        is_healthy = await db_manager.health_check()
        
        assert is_healthy
        mock_conn.fetchval.assert_called_once_with("SELECT 1", timeout=5.0)
        
    @pytest.mark.asyncio
    async def test_retry_operation(self, db_manager):
        """Test retry logic for failed operations."""
        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            "Success"
        ]
        
        db_manager.config.max_retries = 3
        db_manager.config.retry_delay = 0.01
        
        with patch.object(db_manager, 'health_check', return_value=True):
            with patch.object(db_manager, 'disconnect', new_callable=AsyncMock):
                with patch.object(db_manager, 'connect', new_callable=AsyncMock):
                    result = await db_manager.retry_operation(mock_operation)
                    
        assert result == "Success"
        assert mock_operation.call_count == 3
        
    @pytest.mark.asyncio
    async def test_pool_stats(self, db_manager):
        """Test getting pool statistics."""
        mock_pool = Mock()
        mock_pool.get_size.return_value = 10
        mock_pool.get_free_size.return_value = 7
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 20
        
        db_manager._pool = mock_pool
        
        stats = await db_manager.get_pool_stats()
        
        assert stats['size'] == 10
        assert stats['free_size'] == 7
        assert stats['min_size'] == 5
        assert stats['max_size'] == 20
        assert stats['total_connections'] == 10


class TestPoolMonitor:
    """Test connection pool monitoring."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        manager = AsyncMock(spec=PostgresManager)
        manager.get_pool_stats.return_value = {
            'max_size': 50,
            'size': 20,
            'free_size': 15,
            'min_size': 10
        }
        return manager
        
    @pytest.fixture
    def pool_monitor(self, mock_db_manager):
        """Create pool monitor instance."""
        return PoolMonitor(mock_db_manager)
        
    def test_pool_metrics_initialization(self):
        """Test PoolMetrics initialization."""
        metrics = PoolMetrics()
        
        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.timestamp is not None
        
        metrics_dict = metrics.to_dict()
        assert 'timestamp' in metrics_dict
        assert 'connection_utilization' in metrics_dict
        
    @pytest.mark.asyncio
    async def test_get_pool_metrics(self, pool_monitor, mock_db_manager):
        """Test collecting pool metrics."""
        mock_db_manager.fetchrow.return_value = None  # No PgBouncer stats
        
        metrics = await pool_monitor.get_pool_metrics()
        
        assert metrics.total_connections == 50
        assert metrics.active_connections == 5  # 20 - 15
        assert metrics.idle_connections == 15
        
    @pytest.mark.asyncio
    async def test_check_pool_health_healthy(self, pool_monitor):
        """Test pool health check when healthy."""
        # Create healthy metrics
        metrics = PoolMetrics()
        metrics.total_connections = 50
        metrics.active_connections = 20
        metrics.waiting_clients = 2
        metrics.avg_query_time = 50
        metrics.max_query_time = 200
        metrics.pool_hit_rate = 98
        
        with patch.object(pool_monitor, 'get_pool_metrics', return_value=metrics):
            is_healthy, issues = await pool_monitor.check_pool_health()
            
        assert is_healthy
        assert len(issues) == 0
        
    @pytest.mark.asyncio
    async def test_check_pool_health_unhealthy(self, pool_monitor):
        """Test pool health check when unhealthy."""
        # Create unhealthy metrics
        metrics = PoolMetrics()
        metrics.total_connections = 50
        metrics.active_connections = 45  # 90% utilization
        metrics.waiting_clients = 20
        metrics.avg_query_time = 200
        metrics.max_query_time = 10000
        metrics.pool_hit_rate = 80
        
        with patch.object(pool_monitor, 'get_pool_metrics', return_value=metrics):
            is_healthy, issues = await pool_monitor.check_pool_health()
            
        assert not is_healthy
        assert len(issues) > 0
        assert any("High connection utilization" in issue for issue in issues)
        assert any("waiting clients" in issue for issue in issues)
        
    @pytest.mark.asyncio
    async def test_get_slow_queries(self, pool_monitor, mock_db_manager):
        """Test slow query detection."""
        mock_db_manager.fetch.return_value = [
            {
                'pid': 123,
                'usename': 'test_user',
                'duration_ms': 1500,
                'query': 'SELECT * FROM large_table'
            }
        ]
        
        slow_queries = await pool_monitor.get_slow_queries(threshold_ms=1000)
        
        assert len(slow_queries) == 1
        assert slow_queries[0]['duration_ms'] == 1500
        
    @pytest.mark.asyncio
    async def test_recommend_pool_size_no_history(self, pool_monitor):
        """Test pool size recommendations without history."""
        recommendations = await pool_monitor.recommend_pool_size()
        
        assert recommendations['min_pool_size'] == 10
        assert recommendations['default_pool_size'] == 50
        assert recommendations['max_client_conn'] == 1000
        
    @pytest.mark.asyncio
    async def test_recommend_pool_size_with_history(self, pool_monitor):
        """Test pool size recommendations with metrics history."""
        # Add metrics history
        for i in range(60):
            metrics = PoolMetrics()
            metrics.active_connections = 10 + i % 20
            metrics.waiting_clients = i % 5
            pool_monitor._metrics_history.append(metrics)
            
        recommendations = await pool_monitor.recommend_pool_size()
        
        assert 10 <= recommendations['min_pool_size'] <= 100
        assert 50 <= recommendations['default_pool_size'] <= 200
        assert 100 <= recommendations['max_client_conn'] <= 2000
        
    def test_metrics_summary(self, pool_monitor):
        """Test metrics summary generation."""
        # Add sample metrics
        for i in range(10):
            metrics = PoolMetrics()
            metrics.active_connections = 10 + i
            metrics.waiting_clients = i % 3
            metrics.avg_query_time = 50 + i * 10
            pool_monitor._metrics_history.append(metrics)
            
        summary = pool_monitor.get_metrics_summary(hours=1)
        
        assert summary['samples'] == 10
        assert summary['avg_active_connections'] > 0
        assert summary['max_active_connections'] >= summary['avg_active_connections']
        
    @pytest.mark.asyncio
    async def test_prometheus_export(self, pool_monitor):
        """Test Prometheus format export."""
        metrics = PoolMetrics()
        metrics.total_connections = 50
        metrics.active_connections = 20
        metrics.pool_hit_rate = 95.5
        
        with patch.object(pool_monitor, 'get_pool_metrics', return_value=metrics):
            prometheus_output = await pool_monitor.export_metrics(format="prometheus")
            
        assert "pgbouncer_total_connections 50" in prometheus_output
        assert "pgbouncer_active_connections 20" in prometheus_output
        assert "pgbouncer_pool_hit_rate 95.5" in prometheus_output
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output