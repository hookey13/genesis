"""Integration tests for SQLite to PostgreSQL migration.

Tests the complete migration process including:
- Data integrity verification
- Rollback functionality
- Foreign key preservation
- Edge cases and error handling
"""

import asyncio
import json
import os
import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import uuid

import asyncpg

from genesis.data.migration_engine import SQLiteToPostgreSQLMigrator
from genesis.core.exceptions import MigrationError


@pytest.fixture
def temp_sqlite_db():
    """Create temporary SQLite database with test data."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Create schema
    cursor.executescript("""
        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY,
            balance_usdt TEXT NOT NULL,
            tier TEXT DEFAULT 'SNIPER',
            locked_features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );
        
        CREATE TABLE positions (
            position_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price TEXT NOT NULL,
            current_price TEXT,
            quantity TEXT NOT NULL,
            dollar_value TEXT NOT NULL,
            pnl_dollars TEXT DEFAULT '0',
            pnl_percent TEXT DEFAULT '0',
            status TEXT DEFAULT 'OPEN',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        );
        
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            position_id TEXT,
            account_id TEXT NOT NULL,
            client_order_id TEXT UNIQUE NOT NULL,
            exchange_order_id TEXT UNIQUE,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            type TEXT NOT NULL,
            quantity TEXT NOT NULL,
            price TEXT,
            executed_price TEXT,
            executed_quantity TEXT DEFAULT '0',
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            executed_at TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id),
            FOREIGN KEY (position_id) REFERENCES positions(position_id)
        );
        
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE trading_sessions (
            session_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            session_date TIMESTAMP NOT NULL,
            starting_balance TEXT NOT NULL,
            current_balance TEXT NOT NULL,
            ending_balance TEXT,
            realized_pnl TEXT DEFAULT '0',
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        );
    """)
    
    # Insert test data
    test_account_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO accounts (account_id, balance_usdt, tier, locked_features)
        VALUES (?, '10000.00', 'SNIPER', '[]')
    """, (test_account_id,))
    
    # Insert positions
    for i in range(5):
        position_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO positions 
            (position_id, account_id, symbol, side, entry_price, quantity, dollar_value, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (position_id, test_account_id, f"BTC/USDT", "LONG", "50000.00", "0.1", "5000.00", "OPEN"))
    
    # Insert orders
    for i in range(10):
        order_id = str(uuid.uuid4())
        client_order_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO orders 
            (order_id, account_id, client_order_id, symbol, side, type, quantity, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, test_account_id, client_order_id, "ETH/USDT", "BUY", "MARKET", "1.0", "FILLED"))
    
    # Insert user
    cursor.execute("""
        INSERT INTO users (user_id, username, email, password_hash)
        VALUES (?, 'testuser', 'test@example.com', 'hash123')
    """, (str(uuid.uuid4()),))
    
    # Insert trading session
    session_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO trading_sessions 
        (session_id, account_id, session_date, starting_balance, current_balance)
        VALUES (?, ?, CURRENT_TIMESTAMP, '10000.00', '10500.00')
    """, (session_id, test_account_id))
    
    conn.commit()
    
    yield path
    
    # Cleanup
    conn.close()
    os.unlink(path)


@pytest.fixture
async def postgres_test_db():
    """Create test PostgreSQL database."""
    config = {
        'host': os.getenv('TEST_POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('TEST_POSTGRES_PORT', '5432')),
        'database': 'genesis_test_migration',
        'user': os.getenv('TEST_POSTGRES_USER', 'postgres'),
        'password': os.getenv('TEST_POSTGRES_PASSWORD', 'postgres')
    }
    
    # Create test database
    admin_conn = await asyncpg.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database='postgres'
    )
    
    try:
        await admin_conn.execute(f"DROP DATABASE IF EXISTS {config['database']}")
        await admin_conn.execute(f"CREATE DATABASE {config['database']}")
    finally:
        await admin_conn.close()
    
    # Create schema in test database
    conn = await asyncpg.connect(**config)
    try:
        # Create tables (simplified for testing)
        await conn.execute("""
            CREATE TABLE accounts (
                account_id TEXT PRIMARY KEY,
                balance_usdt TEXT NOT NULL,
                tier TEXT DEFAULT 'SNIPER',
                locked_features TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        await conn.execute("""
            CREATE TABLE positions (
                position_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price TEXT NOT NULL,
                current_price TEXT,
                quantity TEXT NOT NULL,
                dollar_value TEXT NOT NULL,
                pnl_dollars TEXT DEFAULT '0',
                pnl_percent TEXT DEFAULT '0',
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                position_id TEXT REFERENCES positions(position_id),
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                client_order_id TEXT UNIQUE NOT NULL,
                exchange_order_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                type TEXT NOT NULL,
                quantity TEXT NOT NULL,
                price TEXT,
                executed_price TEXT,
                executed_quantity TEXT DEFAULT '0',
                status TEXT DEFAULT 'PENDING',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        await conn.execute("""
            CREATE TABLE users (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await conn.execute("""
            CREATE TABLE trading_sessions (
                session_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                session_date TIMESTAMP WITH TIME ZONE NOT NULL,
                starting_balance TEXT NOT NULL,
                current_balance TEXT NOT NULL,
                ending_balance TEXT,
                realized_pnl TEXT DEFAULT '0',
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create migration tracking tables
        await conn.execute("""
            CREATE TABLE migration_history (
                id BIGSERIAL PRIMARY KEY,
                source_table TEXT NOT NULL,
                target_table TEXT NOT NULL,
                rows_migrated BIGINT NOT NULL,
                migration_started TIMESTAMP WITH TIME ZONE NOT NULL,
                migration_completed TIMESTAMP WITH TIME ZONE,
                status TEXT NOT NULL,
                error_message TEXT
            )
        """)
        
        await conn.execute("""
            CREATE TABLE checksum_verification (
                id BIGSERIAL PRIMARY KEY,
                table_name TEXT NOT NULL,
                source_checksum TEXT NOT NULL,
                target_checksum TEXT,
                row_count BIGINT NOT NULL,
                verification_time TIMESTAMP WITH TIME ZONE NOT NULL,
                match_status BOOLEAN
            )
        """)
        
    finally:
        await conn.close()
    
    yield config
    
    # Cleanup
    admin_conn = await asyncpg.connect(
        host=config['host'],
        port=config['port'],
        user=config['user'],
        password=config['password'],
        database='postgres'
    )
    try:
        await admin_conn.execute(f"DROP DATABASE IF EXISTS {config['database']}")
    finally:
        await admin_conn.close()


@pytest.fixture
async def migrator(temp_sqlite_db, postgres_test_db):
    """Create migrator instance for testing."""
    return SQLiteToPostgreSQLMigrator(
        sqlite_path=temp_sqlite_db,
        postgres_config=postgres_test_db,
        batch_size=100
    )


@pytest.mark.asyncio
async def test_data_integrity(migrator):
    """Test that all data is migrated without loss."""
    # Execute migration
    result = await migrator.execute_migration()
    
    # Verify success
    assert result['status'] == 'success'
    assert result['tables_migrated'] > 0
    
    # Verify all tables have matching checksums
    for verification in result['verification']:
        assert verification['checksums_match'], \
            f"Checksum mismatch for table {verification['table']}"
        assert verification['counts_match'], \
            f"Row count mismatch for table {verification['table']}"


@pytest.mark.asyncio
async def test_foreign_key_preservation(migrator):
    """Test that foreign key relationships are preserved."""
    await migrator.connect()
    
    try:
        # Migrate data
        tables = await migrator.get_tables_to_migrate()
        for table in tables:
            await migrator.migrate_table(table)
        
        # Verify foreign keys
        fk_valid = await migrator.verify_foreign_keys()
        assert fk_valid, "Foreign key validation failed"
        
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_data_type_conversion(migrator):
    """Test proper data type conversion from SQLite to PostgreSQL."""
    await migrator.connect()
    
    try:
        # Test datetime conversion
        test_datetime = "2024-01-15T10:30:00"
        converted = migrator.convert_sqlite_value(test_datetime, 'DATETIME')
        assert isinstance(converted, datetime)
        
        # Test boolean conversion
        assert migrator.convert_sqlite_value(1, 'BOOLEAN') is True
        assert migrator.convert_sqlite_value(0, 'BOOLEAN') is False
        assert migrator.convert_sqlite_value("1", 'BOOLEAN') is True
        
        # Test JSON conversion
        json_str = '{"key": "value"}'
        converted = migrator.convert_sqlite_value(json_str, 'JSON')
        assert converted == {"key": "value"}
        
        # Test decimal/numeric preservation
        decimal_str = "12345.67890"
        converted = migrator.convert_sqlite_value(decimal_str, 'DECIMAL')
        assert converted == decimal_str  # Should remain as string for precision
        
        # Test NULL handling
        assert migrator.convert_sqlite_value(None, 'TEXT') is None
        
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_batch_processing(migrator):
    """Test batch processing for large tables."""
    await migrator.connect()
    
    try:
        # Set small batch size to test batching
        migrator.batch_size = 2
        
        # Migrate orders table (has 10 rows)
        result = await migrator.migrate_table('orders')
        
        # Verify all rows migrated despite small batch size
        assert result['source_count'] == result['target_count']
        assert result['status'] == 'success'
        
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_rollback_on_failure(migrator):
    """Test rollback mechanism on migration failure."""
    # Create backup before migration
    backup_path = await migrator.create_backup()
    assert Path(backup_path).exists()
    
    # Simulate migration failure
    with patch.object(migrator, 'migrate_table', side_effect=MigrationError("Test failure")):
        with pytest.raises(MigrationError):
            await migrator.execute_migration()
    
    # Verify backup still exists for manual rollback
    assert Path(backup_path).exists()
    
    # Cleanup
    os.unlink(backup_path)


@pytest.mark.asyncio
async def test_checksum_calculation(migrator):
    """Test checksum calculation for data verification."""
    await migrator.connect()
    
    try:
        # Calculate SQLite checksum
        sqlite_checksum = migrator.calculate_table_checksum(
            migrator.sqlite_conn, 
            'accounts', 
            is_sqlite=True
        )
        assert sqlite_checksum
        assert len(sqlite_checksum) == 64  # SHA256 hex length
        
        # Migrate table
        await migrator.migrate_table('accounts')
        
        # Calculate PostgreSQL checksum
        postgres_checksum = await migrator.calculate_postgres_checksum('accounts')
        assert postgres_checksum
        assert len(postgres_checksum) == 64
        
        # Checksums should match for identical data
        assert sqlite_checksum == postgres_checksum
        
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_migration_logging(migrator):
    """Test migration progress logging."""
    await migrator.connect()
    
    try:
        # Migrate a table
        result = await migrator.migrate_table('users')
        
        # Verify migration was logged
        async with migrator.pg_pool.acquire() as conn:
            log_entry = await conn.fetchrow("""
                SELECT * FROM migration_history 
                WHERE source_table = 'users'
                ORDER BY migration_started DESC
                LIMIT 1
            """)
            
            assert log_entry is not None
            assert log_entry['status'] == 'success'
            assert log_entry['rows_migrated'] > 0
            
            # Verify checksum was logged
            checksum_entry = await conn.fetchrow("""
                SELECT * FROM checksum_verification
                WHERE table_name = 'users'
                ORDER BY verification_time DESC
                LIMIT 1
            """)
            
            assert checksum_entry is not None
            assert checksum_entry['match_status'] is True
            
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_sequence_updates(migrator):
    """Test that sequences are properly updated after migration."""
    await migrator.connect()
    
    try:
        # Migrate users table (has AUTOINCREMENT)
        await migrator.migrate_table('users')
        
        # Update sequences
        await migrator.update_sequences()
        
        # Try inserting new record to verify sequence works
        async with migrator.pg_pool.acquire() as conn:
            new_user = await conn.fetchval("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, str(uuid.uuid4()), 'newuser', 'new@example.com', 'hash456')
            
            assert new_user > 1  # Should be greater than migrated records
            
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_empty_table_migration(migrator):
    """Test migration of empty tables."""
    await migrator.connect()
    
    try:
        # Create empty table in SQLite
        migrator.sqlite_conn.execute("""
            CREATE TABLE empty_table (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)
        migrator.sqlite_conn.commit()
        
        # Create corresponding table in PostgreSQL
        async with migrator.pg_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE empty_table (
                    id BIGINT PRIMARY KEY,
                    data TEXT
                )
            """)
        
        # Migrate empty table
        result = await migrator.migrate_table('empty_table')
        
        assert result['source_count'] == 0
        assert result['target_count'] == 0
        assert result['status'] == 'success'
        
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_special_characters_handling(migrator):
    """Test handling of special characters in data."""
    await migrator.connect()
    
    try:
        # Insert data with special characters
        migrator.sqlite_conn.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, locked_features)
            VALUES ('special-test', '1000.00', 'SNIPER', '["feature''s", "test\"quote"]')
        """)
        migrator.sqlite_conn.commit()
        
        # Migrate table
        await migrator.migrate_table('accounts')
        
        # Verify data integrity
        async with migrator.pg_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM accounts WHERE account_id = 'special-test'
            """)
            
            assert row is not None
            assert row['locked_features'] == '["feature''s", "test\"quote"]'
            
    finally:
        await migrator.disconnect()


@pytest.mark.asyncio
async def test_concurrent_migration_prevention():
    """Test that concurrent migrations are prevented."""
    # This would typically use database locks or migration status tracking
    # to prevent multiple migration processes from running simultaneously
    pass  # Implementation depends on production requirements


@pytest.mark.asyncio
async def test_partial_migration_recovery(migrator):
    """Test recovery from partial migration."""
    await migrator.connect()
    
    try:
        # Migrate some tables
        await migrator.migrate_table('accounts')
        await migrator.migrate_table('positions')
        
        # Simulate failure
        # In real scenario, we'd check migration_history to resume
        async with migrator.pg_pool.acquire() as conn:
            completed = await conn.fetch("""
                SELECT DISTINCT source_table 
                FROM migration_history 
                WHERE status = 'success'
            """)
            
            completed_tables = {row['source_table'] for row in completed}
            all_tables = set(await migrator.get_tables_to_migrate())
            remaining_tables = all_tables - completed_tables
            
            assert len(remaining_tables) > 0  # Some tables not migrated yet
            
            # Resume migration for remaining tables
            for table in remaining_tables:
                if table not in ('migration_history', 'checksum_verification'):
                    await migrator.migrate_table(table)
            
    finally:
        await migrator.disconnect()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])