"""Integration tests for database performance."""

import pytest
import sqlite3
from pathlib import Path
from tests.stress.db_stress import DatabaseStressTest


@pytest.mark.asyncio
async def test_database_setup():
    """Test database schema setup."""
    test = DatabaseStressTest(db_path=":memory:")
    await test.setup_database()
    
    # Verify tables created
    cursor = test.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    assert "trades" in tables
    assert "positions" in tables
    assert "orders" in tables


@pytest.mark.asyncio
async def test_database_population():
    """Test database population with test data."""
    test = DatabaseStressTest(db_path=":memory:")
    await test.setup_database()
    await test.populate_database(1000)
    
    # Verify data inserted
    cursor = test.conn.execute("SELECT COUNT(*) FROM trades")
    count = cursor.fetchone()[0]
    
    assert count == 1000
    assert test.metrics.total_records == 1000


@pytest.mark.asyncio
async def test_query_performance():
    """Test query performance benchmarks."""
    test = DatabaseStressTest(db_path=":memory:")
    await test.setup_database()
    await test.populate_database(1000)
    await test.benchmark_queries()
    
    # All queries should complete
    assert len(test.metrics.query_times_ms) > 0
    
    # Queries should be reasonably fast
    for query_time in test.metrics.query_times_ms:
        assert query_time < 1000  # Less than 1 second


@pytest.mark.asyncio
async def test_index_effectiveness():
    """Test database index effectiveness."""
    test = DatabaseStressTest(db_path=":memory:")
    await test.setup_database()
    await test.populate_database(5000)
    await test.test_index_effectiveness()
    
    # Indexes should provide some improvement
    assert test.metrics.index_effectiveness >= 0