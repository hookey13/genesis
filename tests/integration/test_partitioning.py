"""
Unit tests for table partitioning functionality.
Tests partition creation, pruning, and query optimization.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from genesis.database.partition_manager import PartitionManager
from genesis.database.postgres_manager import PostgresManager


class TestPartitionManager:
    """Test partition management functionality."""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        manager = AsyncMock(spec=PostgresManager)
        return manager
        
    @pytest.fixture
    def partition_manager(self, mock_db_manager):
        """Create partition manager instance."""
        return PartitionManager(mock_db_manager)
        
    @pytest.mark.asyncio
    async def test_setup_partitioned_tables(self, partition_manager, mock_db_manager):
        """Test initial setup of partitioned tables."""
        mock_db_manager.execute.return_value = None
        
        await partition_manager.setup_partitioned_tables()
        
        # Verify tables were created
        assert mock_db_manager.execute.called
        
        # Check that all expected tables were created
        calls = mock_db_manager.execute.call_args_list
        call_queries = [call[0][0] for call in calls]
        
        # Verify orders table creation
        assert any("CREATE TABLE IF NOT EXISTS orders" in query for query in call_queries)
        
        # Verify trades table creation
        assert any("CREATE TABLE IF NOT EXISTS trades" in query for query in call_queries)
        
        # Verify market_data table creation
        assert any("CREATE TABLE IF NOT EXISTS market_data" in query for query in call_queries)
        
        # Verify tilt_events table creation
        assert any("CREATE TABLE IF NOT EXISTS tilt_events" in query for query in call_queries)
        
    @pytest.mark.asyncio
    async def test_create_monthly_partition(self, partition_manager, mock_db_manager):
        """Test creating a monthly partition."""
        mock_db_manager.fetchval.return_value = False  # Partition doesn't exist
        mock_db_manager.execute.return_value = None
        
        await partition_manager.create_monthly_partition('orders', 2025, 9)
        
        # Verify partition creation query was executed
        mock_db_manager.execute.assert_called()
        
        # Check the partition creation call
        calls = mock_db_manager.execute.call_args_list
        assert any(
            "create_monthly_partition" in str(call)
            for call in calls
        )
        
    @pytest.mark.asyncio
    async def test_partition_already_exists(self, partition_manager, mock_db_manager):
        """Test handling when partition already exists."""
        mock_db_manager.fetchval.return_value = True  # Partition exists
        
        await partition_manager.create_monthly_partition('orders', 2025, 9)
        
        # Should check existence but not create
        mock_db_manager.fetchval.assert_called_once()
        
        # Execute should not be called for partition creation
        assert not mock_db_manager.execute.called
        
    @pytest.mark.asyncio
    async def test_create_future_partitions(self, partition_manager, mock_db_manager):
        """Test creating partitions for future months."""
        mock_db_manager.fetchval.return_value = False
        mock_db_manager.execute.return_value = None
        
        await partition_manager.create_future_partitions(months_ahead=3)
        
        # Should create partitions for current + 3 future months
        # For each table (4 tables) * 4 months = 16 partitions
        expected_tables = len(PartitionManager.PARTITIONED_TABLES)
        expected_months = 4  # Current + 3 future
        
        # Each partition checks existence then creates
        assert mock_db_manager.fetchval.call_count == expected_tables * expected_months
        
    @pytest.mark.asyncio
    async def test_prune_old_partitions(self, partition_manager, mock_db_manager):
        """Test pruning old partitions."""
        mock_db_manager.execute.return_value = None
        
        await partition_manager.prune_old_partitions()
        
        # Should call drop_old_partition for each table
        calls = mock_db_manager.execute.call_args_list
        
        for table_name in PartitionManager.PARTITIONED_TABLES:
            assert any(
                f"drop_old_partition(${table_name}" in str(call) or
                f"drop_old_partition('{table_name}'" in str(call) or
                f'drop_old_partition("{table_name}"' in str(call) or
                f"drop_old_partition($1" in str(call)
                for call in calls
            )
            
    @pytest.mark.asyncio
    async def test_get_partition_statistics(self, partition_manager, mock_db_manager):
        """Test getting partition statistics."""
        mock_stats = [
            {
                'partition_name': 'orders_2025_09',
                'row_count': 1000,
                'total_size': '10 MB',
                'index_size': '2 MB'
            },
            {
                'partition_name': 'orders_2025_08',
                'row_count': 5000,
                'total_size': '50 MB',
                'index_size': '10 MB'
            }
        ]
        
        mock_db_manager.fetch.return_value = [Mock(**stat) for stat in mock_stats]
        
        stats = await partition_manager.get_partition_statistics('orders')
        
        assert len(stats) == 2
        assert stats[0]['partition_name'] == 'orders_2025_09'
        assert stats[0]['row_count'] == 1000
        
    @pytest.mark.asyncio
    async def test_analyze_partitions(self, partition_manager, mock_db_manager):
        """Test analyzing partitions for query optimization."""
        mock_db_manager.execute.return_value = None
        
        await partition_manager.analyze_partitions()
        
        # Should analyze each table
        calls = mock_db_manager.execute.call_args_list
        
        for table_name in PartitionManager.PARTITIONED_TABLES:
            assert any(
                f"ANALYZE {table_name}" in str(call)
                for call in calls
            )
            
    @pytest.mark.asyncio
    async def test_optimize_partition_queries(self, partition_manager, mock_db_manager):
        """Test creating optimized views."""
        mock_db_manager.execute.return_value = None
        
        await partition_manager.optimize_partition_queries()
        
        calls = mock_db_manager.execute.call_args_list
        call_queries = [str(call) for call in calls]
        
        # Verify views were created
        assert any("CREATE OR REPLACE VIEW recent_orders" in query for query in call_queries)
        assert any("CREATE OR REPLACE VIEW todays_trades" in query for query in call_queries)
        assert any("CREATE OR REPLACE VIEW pending_orders" in query for query in call_queries)
        assert any("CREATE OR REPLACE VIEW performance_summary" in query for query in call_queries)
        
    @pytest.mark.asyncio
    async def test_maintenance_task_start(self, partition_manager, mock_db_manager):
        """Test starting automated maintenance task."""
        mock_db_manager.execute.return_value = None
        mock_db_manager.fetchval.return_value = False
        
        await partition_manager.start_maintenance_task(interval_hours=0.0001)
        
        # Give task time to run once
        await asyncio.sleep(0.01)
        
        # Stop the task
        await partition_manager.stop_maintenance_task()
        
        # Verify maintenance operations were called
        assert mock_db_manager.execute.called
        
    @pytest.mark.asyncio
    async def test_maintenance_task_error_handling(self, partition_manager, mock_db_manager):
        """Test maintenance task handles errors gracefully."""
        mock_db_manager.execute.side_effect = Exception("Database error")
        
        await partition_manager.start_maintenance_task(interval_hours=0.0001)
        
        # Give task time to run and fail
        await asyncio.sleep(0.01)
        
        # Task should continue despite error
        assert partition_manager._maintenance_task is not None
        assert not partition_manager._maintenance_task.done()
        
        # Clean up
        await partition_manager.stop_maintenance_task()
        
    @pytest.mark.asyncio
    async def test_create_partition_indexes(self, partition_manager, mock_db_manager):
        """Test index creation for partitions."""
        mock_db_manager.execute.return_value = None
        
        await partition_manager._create_partition_indexes('orders_2025_09', 'orders')
        
        # Should create indexes based on configuration
        calls = mock_db_manager.execute.call_args_list
        
        # Verify indexes were created
        assert any("CREATE INDEX IF NOT EXISTS" in str(call) for call in calls)
        
        # Check for specific index types
        call_strings = [str(call) for call in calls]
        
        # Should have symbol+time index
        assert any("symbol" in s and "created_at" in s for s in call_strings)
        
        # Should have partial index for pending orders
        assert any("WHERE status = 'pending'" in s for s in call_strings)
        
    def test_partitioned_tables_configuration(self):
        """Test that partitioned tables configuration is valid."""
        tables = PartitionManager.PARTITIONED_TABLES
        
        # Verify required tables are configured
        assert 'orders' in tables
        assert 'trades' in tables
        assert 'market_data' in tables
        assert 'tilt_events' in tables
        
        # Verify each table has required configuration
        for table_name, config in tables.items():
            assert 'partition_key' in config
            assert 'retention_months' in config
            assert 'indexes' in config
            
            # Verify retention is reasonable
            assert 6 <= config['retention_months'] <= 24
            
            # Verify indexes are properly formatted
            for index in config['indexes']:
                assert isinstance(index, tuple)
                assert len(index) in [2, 3]  # (columns, method) or (columns, method, where)