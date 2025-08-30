"""Integration tests for database optimization system."""

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiosqlite

from genesis.operations.db_optimizer import (
    DBOptimizer,
    MaintenanceConfig,
    QueryPerformanceStats,
    IndexUsageStats
)


@pytest.fixture
async def test_database():
    """Create test SQLite database with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Create database with test schema
        async with aiosqlite.connect(db_path) as db:
            # Create tables
            await db.execute("""
                CREATE TABLE positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    price REAL,
                    amount REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (position_id) REFERENCES positions(id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await db.execute(
                "CREATE INDEX idx_positions_pair ON positions(pair)"
            )
            await db.execute(
                "CREATE INDEX idx_orders_status ON orders(status)"
            )
            await db.execute(
                "CREATE INDEX idx_orders_position ON orders(position_id)"
            )
            await db.execute(
                "CREATE INDEX idx_events_type ON events(event_type)"
            )
            
            # Insert test data
            for i in range(100):
                await db.execute(
                    """
                    INSERT INTO positions (pair, side, amount, entry_price)
                    VALUES (?, ?, ?, ?)
                    """,
                    (f"BTC/USDT", "buy" if i % 2 == 0 else "sell", 0.01 * i, 50000 + i * 100)
                )
                
                await db.execute(
                    """
                    INSERT INTO orders (position_id, order_type, status, price, amount)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (i + 1, "market", "filled" if i % 3 == 0 else "pending", 50000 + i * 100, 0.01 * i)
                )
                
                await db.execute(
                    """
                    INSERT INTO events (event_type, event_data)
                    VALUES (?, ?)
                    """,
                    (f"event_type_{i % 10}", f'{{"data": "test_{i}"}}')
                )
            
            await db.commit()
        
        yield db_path


@pytest.fixture
def maintenance_config():
    """Create test maintenance configuration."""
    return MaintenanceConfig(
        vacuum_enabled=True,
        analyze_enabled=True,
        index_optimization_enabled=True,
        maintenance_window_start=0,  # Always in window for testing
        maintenance_window_duration=24,  # 24 hours
        backup_before_maintenance=True,
        min_days_between_vacuum=0,  # Allow immediate vacuum for testing
        query_log_retention_days=30,
        slow_query_threshold_ms=10.0,  # Low threshold for testing
        index_usage_threshold=5
    )


@pytest.fixture
async def db_optimizer(maintenance_config, test_database):
    """Create database optimizer instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backup_dir = Path(tmpdir) / "backups"
        backup_dir.mkdir()
        
        optimizer = DBOptimizer(
            config=maintenance_config,
            db_path=test_database,
            backup_dir=backup_dir
        )
        
        yield optimizer


class TestDBOptimizer:
    """Test cases for database optimizer."""
    
    @pytest.mark.asyncio
    async def test_vacuum_operation(self, db_optimizer, test_database):
        """Test VACUUM operation reduces database size."""
        # Get initial size
        initial_size = test_database.stat().st_size
        
        # Delete some data to create fragmentation
        async with aiosqlite.connect(test_database) as db:
            await db.execute("DELETE FROM events WHERE id % 2 = 0")
            await db.commit()
        
        # Perform vacuum
        result = await db_optimizer.perform_vacuum(force=True)
        
        assert result is True
        
        # Check size reduction
        final_size = test_database.stat().st_size
        assert final_size <= initial_size
        
        # Verify backup was created
        backups = list(db_optimizer.backup_dir.glob("genesis_backup_*.db"))
        assert len(backups) > 0
        
        # Verify state updated
        assert db_optimizer.last_vacuum_time is not None
    
    @pytest.mark.asyncio
    async def test_vacuum_respects_minimum_interval(self, db_optimizer):
        """Test VACUUM respects minimum interval between runs."""
        # Set last vacuum time to recent
        db_optimizer.last_vacuum_time = datetime.utcnow() - timedelta(days=1)
        db_optimizer.config.min_days_between_vacuum = 7
        
        # Try vacuum without force
        result = await db_optimizer.perform_vacuum(force=False)
        
        assert result is False  # Should not vacuum
    
    @pytest.mark.asyncio
    async def test_analyze_operation(self, db_optimizer):
        """Test ANALYZE operation updates statistics."""
        result = await db_optimizer.perform_analyze()
        
        assert result is True
        assert db_optimizer.last_analyze_time is not None
    
    @pytest.mark.asyncio
    async def test_index_analysis(self, db_optimizer, test_database):
        """Test index usage analysis and recommendations."""
        index_stats = await db_optimizer.analyze_indexes()
        
        assert len(index_stats) > 0
        
        # Check index statistics
        for stats in index_stats:
            assert stats.table_name is not None
            assert stats.index_name is not None
            assert stats.columns is not None
            assert stats.selectivity >= 0.0
            assert stats.selectivity <= 1.0
        
        # Check for any recommendations
        recommendations = [s for s in index_stats if s.recommendation]
        # Some indexes might have recommendations based on selectivity
        assert isinstance(recommendations, list)
    
    @pytest.mark.asyncio
    async def test_query_performance_tracking(self, db_optimizer):
        """Test query performance tracking."""
        # Track several query executions
        queries = [
            ("SELECT * FROM positions WHERE pair = ?", 15.5, 10),
            ("SELECT * FROM positions WHERE pair = ?", 12.3, 10),
            ("SELECT * FROM orders WHERE status = ?", 150.0, 50),  # Slow query
            ("INSERT INTO events (event_type) VALUES (?)", 5.0, 1),
        ]
        
        for query, time_ms, rows in queries:
            await db_optimizer.track_query_performance(query, time_ms, rows)
        
        # Check statistics
        assert len(db_optimizer.query_stats) > 0
        
        # Check slow query detection
        slow_queries = [
            stats for stats in db_optimizer.query_stats.values()
            if stats.avg_time_ms > db_optimizer.config.slow_query_threshold_ms
        ]
        assert len(slow_queries) > 0
    
    @pytest.mark.asyncio
    async def test_performance_baseline_calculation(self, db_optimizer):
        """Test performance baseline calculation."""
        # Track queries first
        for i in range(20):
            query = f"SELECT * FROM positions WHERE id = {i}"
            time_ms = 5.0 + (i * 0.5)  # Varying times
            await db_optimizer.track_query_performance(query, time_ms, 1)
        
        baseline = await db_optimizer.get_performance_baseline()
        
        assert "total_queries_tracked" in baseline
        assert "total_executions" in baseline
        assert "avg_execution_time_ms" in baseline
        assert "p50_execution_time_ms" in baseline
        assert "p95_execution_time_ms" in baseline
        assert "p99_execution_time_ms" in baseline
        
        # Check percentiles make sense
        assert baseline["p50_execution_time_ms"] <= baseline["p95_execution_time_ms"]
        assert baseline["p95_execution_time_ms"] <= baseline["p99_execution_time_ms"]
    
    @pytest.mark.asyncio
    async def test_database_backup(self, db_optimizer, test_database):
        """Test database backup creation."""
        backup_path = await db_optimizer._backup_database()
        
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0
        
        # Verify backup is valid SQLite database
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM positions")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count > 0  # Should have data
    
    @pytest.mark.asyncio
    async def test_backup_cleanup(self, db_optimizer):
        """Test old backup cleanup."""
        # Create multiple backup files
        for i in range(10):
            backup_file = db_optimizer.backup_dir / f"genesis_backup_2024010{i}_000000.db"
            backup_file.touch()
        
        # Run cleanup
        await db_optimizer._cleanup_old_backups(keep_count=5)
        
        # Check only 5 remain
        backups = list(db_optimizer.backup_dir.glob("genesis_backup_*.db"))
        assert len(backups) == 5
    
    def test_maintenance_window_check(self, db_optimizer):
        """Test maintenance window detection."""
        # Set window from 2 AM to 4 AM
        db_optimizer.config.maintenance_window_start = 2
        db_optimizer.config.maintenance_window_duration = 2
        
        # Mock current time
        with patch('genesis.operations.db_optimizer.datetime') as mock_datetime:
            # Test inside window
            mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 3, 0, 0)
            assert db_optimizer.is_maintenance_window() is True
            
            # Test outside window
            mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 5, 0, 0)
            assert db_optimizer.is_maintenance_window() is False
            
            # Test window crossing midnight
            db_optimizer.config.maintenance_window_start = 23
            db_optimizer.config.maintenance_window_duration = 3
            
            mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 0, 30, 0)
            assert db_optimizer.is_maintenance_window() is True
            
            mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)
            assert db_optimizer.is_maintenance_window() is False
    
    @pytest.mark.asyncio
    async def test_scheduled_maintenance(self, db_optimizer):
        """Test scheduled maintenance execution."""
        results = await db_optimizer.run_scheduled_maintenance()
        
        assert "vacuum" in results
        assert "analyze" in results
        assert "index_optimization" in results
        
        # At least some operations should succeed
        assert any(results.values())
    
    @pytest.mark.asyncio
    async def test_postgres_migration_preparation(self, db_optimizer, test_database):
        """Test PostgreSQL migration preparation."""
        report = await db_optimizer.prepare_for_postgres_migration()
        
        assert "database_size" in report
        assert "table_count" in report
        assert "index_count" in report
        assert "incompatible_features" in report
        assert "migration_notes" in report
        
        # Should detect our test tables
        assert report["table_count"] >= 3  # positions, orders, events
        assert report["index_count"] >= 4  # Our test indexes
        
        # Should have migration notes
        assert len(report["migration_notes"]) > 0
    
    def test_query_normalization(self, db_optimizer):
        """Test query normalization for grouping."""
        queries = [
            ("SELECT * FROM positions WHERE id = 123", "SELECT * FROM POSITIONS WHERE ID = ?"),
            ("SELECT * FROM orders WHERE price > 50000.5", "SELECT * FROM ORDERS WHERE PRICE > ?"),
            ("INSERT INTO events (type, data) VALUES ('test', 'data')", 
             "INSERT INTO EVENTS (TYPE, DATA) VALUES (?, ?)"),
        ]
        
        for original, expected in queries:
            normalized = db_optimizer._normalize_query(original)
            assert normalized == expected
    
    @pytest.mark.asyncio
    async def test_index_selectivity_calculation(self, db_optimizer, test_database):
        """Test index selectivity calculation."""
        async with aiosqlite.connect(test_database) as db:
            # Calculate selectivity for pair column
            selectivity = await db_optimizer._calculate_selectivity(
                db, "positions", ["pair"]
            )
            
            # Since we inserted all with same pair, selectivity should be low
            assert selectivity > 0.0
            assert selectivity <= 1.0
    
    @pytest.mark.asyncio
    async def test_maintenance_state_persistence(self, db_optimizer):
        """Test maintenance state save/load."""
        # Set maintenance times
        db_optimizer.last_vacuum_time = datetime.utcnow()
        db_optimizer.last_analyze_time = datetime.utcnow() - timedelta(hours=1)
        
        # Save state
        db_optimizer._save_maintenance_state()
        
        # Create new optimizer with same backup dir
        new_optimizer = DBOptimizer(
            config=db_optimizer.config,
            db_path=db_optimizer.db_path,
            backup_dir=db_optimizer.backup_dir
        )
        
        # Check state loaded
        assert new_optimizer.last_vacuum_time is not None
        assert new_optimizer.last_analyze_time is not None
        
        # Times should match (within reason due to serialization)
        time_diff = abs(
            (new_optimizer.last_vacuum_time - db_optimizer.last_vacuum_time).total_seconds()
        )
        assert time_diff < 1.0
    
    @pytest.mark.asyncio
    async def test_vacuum_with_backup_failure(self, db_optimizer):
        """Test vacuum handling when backup fails."""
        # Mock backup to fail
        with patch.object(db_optimizer, '_backup_database', side_effect=Exception("Backup failed")):
            result = await db_optimizer.perform_vacuum(force=True)
            
            # Vacuum should fail if backup fails
            assert result is False
    
    @pytest.mark.asyncio
    async def test_concurrent_maintenance_operations(self, db_optimizer):
        """Test concurrent maintenance operations don't interfere."""
        # Run multiple operations concurrently
        tasks = [
            db_optimizer.perform_analyze(),
            db_optimizer.analyze_indexes(),
            db_optimizer.get_performance_baseline()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)


@pytest.mark.asyncio
async def test_setup_db_maintenance():
    """Test database maintenance setup."""
    with patch('genesis.operations.db_optimizer.DBOptimizer') as MockOptimizer:
        mock_optimizer = AsyncMock()
        MockOptimizer.return_value = mock_optimizer
        
        from genesis.operations.db_optimizer import setup_db_maintenance
        
        # Setup maintenance
        optimizer = await setup_db_maintenance()
        
        # Verify optimizer created
        assert optimizer is not None
        
        # Let background task start
        await asyncio.sleep(0.1)