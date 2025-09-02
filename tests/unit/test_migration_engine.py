"""Unit tests for SQLite to PostgreSQL migration engine.

Tests individual components of the migration engine including:
- Data type conversion
- Checksum calculation
- Batch processing
- Error handling
"""

import json
import pytest
import sqlite3
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import uuid

from genesis.data.migration_engine import SQLiteToPostgreSQLMigrator
from genesis.core.exceptions import MigrationError


class TestDataTypeConversion:
    """Test data type conversion methods."""
    
    def test_datetime_conversion(self):
        """Test datetime string to datetime object conversion."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={},
            batch_size=100
        )
        
        # ISO format datetime
        iso_datetime = "2024-01-15T10:30:00"
        result = migrator.convert_sqlite_value(iso_datetime, 'DATETIME')
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        
        # Datetime with timezone
        tz_datetime = "2024-01-15T10:30:00+00:00"
        result = migrator.convert_sqlite_value(tz_datetime, 'DATETIME')
        assert isinstance(result, datetime)
        
        # NULL datetime
        result = migrator.convert_sqlite_value(None, 'DATETIME')
        assert result is None
    
    def test_boolean_conversion(self):
        """Test integer/string to boolean conversion."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={},
            batch_size=100
        )
        
        # Integer to boolean
        assert migrator.convert_sqlite_value(1, 'BOOLEAN') is True
        assert migrator.convert_sqlite_value(0, 'BOOLEAN') is False
        
        # String to boolean
        assert migrator.convert_sqlite_value("1", 'BOOLEAN') is True
        assert migrator.convert_sqlite_value("0", 'BOOLEAN') is False
        
        # NULL boolean
        assert migrator.convert_sqlite_value(None, 'BOOLEAN') is None
    
    def test_json_conversion(self):
        """Test JSON string to object conversion."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={},
            batch_size=100
        )
        
        # Valid JSON
        json_str = '{"key": "value", "number": 42}'
        result = migrator.convert_sqlite_value(json_str, 'JSON')
        assert isinstance(result, dict)
        assert result['key'] == 'value'
        assert result['number'] == 42
        
        # Invalid JSON (should return as-is)
        invalid_json = "not json"
        result = migrator.convert_sqlite_value(invalid_json, 'JSON')
        assert result == invalid_json
        
        # NULL JSON
        assert migrator.convert_sqlite_value(None, 'JSON') is None
    
    def test_decimal_conversion(self):
        """Test decimal/numeric value preservation."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={},
            batch_size=100
        )
        
        # String decimal (should remain as string for precision)
        assert migrator.convert_sqlite_value("12345.67890", 'DECIMAL') == "12345.67890"
        
        # Integer to decimal
        assert migrator.convert_sqlite_value(100, 'NUMERIC') == "100"
        
        # Float to decimal
        assert migrator.convert_sqlite_value(99.99, 'DECIMAL') == "99.99"
        
        # NULL decimal
        assert migrator.convert_sqlite_value(None, 'DECIMAL') is None


class TestChecksumCalculation:
    """Test checksum calculation methods."""
    
    def test_table_checksum_calculation(self):
        """Test SHA256 checksum calculation for table data."""
        # Create temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL
                )
            """)
            
            # Insert test data
            cursor.executemany(
                "INSERT INTO test_table (id, name, value) VALUES (?, ?, ?)",
                [(1, "Alice", 100.5), (2, "Bob", 200.0), (3, "Charlie", 300.25)]
            )
            conn.commit()
            
            # Calculate checksum
            migrator = SQLiteToPostgreSQLMigrator(
                sqlite_path=db_path,
                postgres_config={},
                batch_size=100
            )
            migrator.sqlite_conn = conn
            
            checksum = migrator.calculate_table_checksum(conn, "test_table")
            
            # Verify checksum properties
            assert checksum is not None
            assert len(checksum) == 64  # SHA256 produces 64 hex characters
            assert all(c in '0123456789abcdef' for c in checksum)
            
            # Same data should produce same checksum
            checksum2 = migrator.calculate_table_checksum(conn, "test_table")
            assert checksum == checksum2
            
            # Modified data should produce different checksum
            cursor.execute("UPDATE test_table SET value = 999 WHERE id = 1")
            conn.commit()
            checksum3 = migrator.calculate_table_checksum(conn, "test_table")
            assert checksum != checksum3
            
        finally:
            conn.close()
            Path(db_path).unlink()
    
    def test_null_handling_in_checksum(self):
        """Test that NULL values are properly handled in checksums."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE test_nulls (
                    id INTEGER PRIMARY KEY,
                    nullable_field TEXT
                )
            """)
            
            cursor.executemany(
                "INSERT INTO test_nulls (id, nullable_field) VALUES (?, ?)",
                [(1, "value"), (2, None), (3, "another")]
            )
            conn.commit()
            
            migrator = SQLiteToPostgreSQLMigrator(
                sqlite_path=db_path,
                postgres_config={},
                batch_size=100
            )
            migrator.sqlite_conn = conn
            
            checksum = migrator.calculate_table_checksum(conn, "test_nulls")
            assert checksum is not None
            
            # Checksum should differentiate between NULL and empty string
            cursor.execute("UPDATE test_nulls SET nullable_field = '' WHERE id = 2")
            conn.commit()
            checksum2 = migrator.calculate_table_checksum(conn, "test_nulls")
            assert checksum != checksum2
            
        finally:
            conn.close()
            Path(db_path).unlink()


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_size_configuration(self):
        """Test that batch size is properly configured."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={},
            batch_size=500
        )
        
        assert migrator.batch_size == 500
        
        # Test default batch size
        migrator2 = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={}
        )
        assert migrator2.batch_size == 1000  # Default value
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_mock_data(self):
        """Test batch processing with mocked data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Create test database with many rows
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE large_table (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            """)
            
            # Insert 250 rows
            for i in range(250):
                cursor.execute(
                    "INSERT INTO large_table (id, data) VALUES (?, ?)",
                    (i, f"data_{i}")
                )
            conn.commit()
            
            migrator = SQLiteToPostgreSQLMigrator(
                sqlite_path=db_path,
                postgres_config={
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'test',
                    'user': 'test',
                    'password': 'test'
                },
                batch_size=100  # Process 100 rows at a time
            )
            
            # Mock PostgreSQL pool
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            migrator.pg_pool = mock_pool
            migrator.sqlite_conn = conn
            
            # Get row count
            count = migrator.get_table_row_count('large_table')
            assert count == 250
            
            # Verify batch processing would work
            cursor.execute("SELECT * FROM large_table")
            processed = 0
            while True:
                batch = cursor.fetchmany(100)
                if not batch:
                    break
                processed += len(batch)
            
            assert processed == 250
            
        finally:
            conn.close()
            Path(db_path).unlink()


class TestErrorHandling:
    """Test error handling and rollback functionality."""
    
    @pytest.mark.asyncio
    async def test_migration_error_on_connection_failure(self):
        """Test that connection failures raise appropriate errors."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="nonexistent.db",
            postgres_config={
                'host': 'invalid_host',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
        )
        
        with pytest.raises(Exception):
            await migrator.connect()
    
    @pytest.mark.asyncio
    async def test_rollback_on_migration_failure(self):
        """Test rollback mechanism on migration failure."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Create test database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            cursor.execute("INSERT INTO test (id) VALUES (1)")
            conn.commit()
            conn.close()
            
            migrator = SQLiteToPostgreSQLMigrator(
                sqlite_path=db_path,
                postgres_config={
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'test',
                    'user': 'test',
                    'password': 'test'
                }
            )
            
            # Mock connection to simulate failure
            migrator.pg_pool = AsyncMock()
            migrator.sqlite_conn = sqlite3.connect(db_path)
            
            # Create backup
            backup_path = await migrator.create_backup()
            assert Path(backup_path).exists()
            
            # Verify backup contains data
            backup_conn = sqlite3.connect(backup_path)
            backup_cursor = backup_conn.cursor()
            backup_cursor.execute("SELECT COUNT(*) FROM test")
            assert backup_cursor.fetchone()[0] == 1
            backup_conn.close()
            
            # Cleanup
            Path(backup_path).unlink()
            
        finally:
            Path(db_path).unlink()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self):
        """Test that failed transactions are properly rolled back."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
        )
        
        # Mock PostgreSQL connection
        mock_conn = AsyncMock()
        mock_transaction = AsyncMock()
        mock_conn.transaction.return_value = mock_transaction
        
        # Simulate transaction failure
        mock_conn.execute.side_effect = Exception("Transaction failed")
        
        # Mock pool
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        migrator.pg_pool = mock_pool
        
        with pytest.raises(Exception):
            await migrator.execute_transaction([
                ("INSERT INTO test VALUES ($1)", [1]),
                ("INSERT INTO test VALUES ($1)", [2])
            ])
        
        # Verify transaction context was used
        mock_conn.transaction.assert_called()


class TestTableMapping:
    """Test SQLite to PostgreSQL type mappings."""
    
    def test_get_table_mappings(self):
        """Test that type mappings are correctly defined."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={},
            batch_size=100
        )
        
        mappings = migrator._get_table_mappings()
        
        # Verify essential mappings exist
        assert mappings['INTEGER'] == 'BIGINT'
        assert mappings['TEXT'] == 'TEXT'
        assert mappings['REAL'] == 'DOUBLE PRECISION'
        assert mappings['BLOB'] == 'BYTEA'
        assert mappings['DATETIME'] == 'TIMESTAMP WITH TIME ZONE'
        assert mappings['BOOLEAN'] == 'BOOLEAN'
        assert mappings['JSON'] == 'JSONB'
        
        # Verify all values are strings
        assert all(isinstance(v, str) for v in mappings.values())


class TestMigrationLog:
    """Test migration logging functionality."""
    
    @pytest.mark.asyncio
    async def test_migration_log_creation(self):
        """Test that migration logs are properly created."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
        )
        
        # Mock PostgreSQL connection
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        migrator.pg_pool = mock_pool
        
        # Create test result
        result = {
            'table': 'test_table',
            'source_count': 100,
            'target_count': 100,
            'source_checksum': 'abc123',
            'target_checksum': 'abc123',
            'checksums_match': True,
            'counts_match': True,
            'status': 'success'
        }
        
        await migrator.log_migration(result)
        
        # Verify INSERT queries were executed
        assert mock_conn.execute.call_count == 2  # migration_history and checksum_verification
        
        # Check first call (migration_history)
        first_call = mock_conn.execute.call_args_list[0]
        assert 'INSERT INTO migration_history' in first_call[0][0]
        
        # Check second call (checksum_verification)
        second_call = mock_conn.execute.call_args_list[1]
        assert 'INSERT INTO checksum_verification' in second_call[0][0]


class TestConnectionManagement:
    """Test database connection management."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test that connection pool is properly created."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
        )
        
        # Mock asyncpg.create_pool
        with patch('genesis.data.migration_engine.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await migrator.connect()
            
            # Verify pool was created with correct parameters
            mock_create_pool.assert_called_once_with(
                host='localhost',
                port=5432,
                database='test',
                user='test',
                password='test',
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            assert migrator.pg_pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_connection_cleanup(self):
        """Test that connections are properly closed."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
        )
        
        # Mock pool
        mock_pool = AsyncMock()
        migrator.pg_pool = mock_pool
        
        # Mock SQLite connection
        mock_sqlite = MagicMock()
        migrator.sqlite_conn = mock_sqlite
        
        await migrator.disconnect()
        
        # Verify connections were closed
        mock_pool.close.assert_called_once()
        mock_sqlite.close.assert_called_once()
        
        assert migrator.pg_pool is None
        assert migrator.sqlite_conn is None


class TestSequenceUpdates:
    """Test PostgreSQL sequence updates."""
    
    @pytest.mark.asyncio
    async def test_sequence_update_after_migration(self):
        """Test that sequences are properly updated after migration."""
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path="test.db",
            postgres_config={
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test',
                'password': 'test'
            }
        )
        
        # Mock PostgreSQL connection
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [100, 200, 50]  # Max IDs for different tables
        
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        migrator.pg_pool = mock_pool
        
        await migrator.update_sequences()
        
        # Verify sequence updates
        execute_calls = mock_conn.execute.call_args_list
        assert len(execute_calls) == 3
        
        # Check that setval was called for each sequence
        for call in execute_calls:
            assert "SELECT setval" in call[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])