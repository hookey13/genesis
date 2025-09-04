"""Database validation for Genesis trading system."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
import psycopg2
from decimal import Decimal

from . import BaseValidator, ValidationIssue, ValidationSeverity


class DatabaseValidator(BaseValidator):
    """Validates database connectivity, performance, and migrations."""
    
    @property
    def name(self) -> str:
        return "database"
    
    @property
    def description(self) -> str:
        return "Validates database connectivity, query performance, and migration status"
    
    async def _validate(self, mode: str):
        """Perform database validation."""
        # Test database connectivity
        await self._test_connectivity()
        
        # Measure query performance
        await self._measure_query_performance()
        
        # Verify migration status
        await self._verify_migrations()
        
        # Check index optimization
        if mode in ["standard", "thorough"]:
            await self._check_indexes()
        
        # Test connection pooling
        if mode == "thorough":
            await self._test_connection_pooling()
    
    async def _test_connectivity(self):
        """Test database connectivity."""
        # Test SQLite connectivity
        sqlite_path = Path(".genesis/data/genesis.db")
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            conn.close()
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"SQLite connected: version {version}",
                details={"path": str(sqlite_path), "version": version}
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message="SQLite connection failed",
                details={"error": str(e)},
                recommendation="Ensure SQLite is installed and accessible"
            ))
        
        # Test PostgreSQL connectivity (if configured)
        try:
            from genesis.config.settings import Settings
            settings = Settings()
            
            if hasattr(settings, "postgres_url") and settings.postgres_url:
                conn = psycopg2.connect(settings.postgres_url)
                cursor = conn.cursor()
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                conn.close()
                
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="PostgreSQL connected",
                    details={"version": version}
                ))
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="PostgreSQL not configured (expected for MVP)",
                recommendation="PostgreSQL will be added in Hunter tier ($2k+)"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="PostgreSQL connection failed",
                details={"error": str(e)}
            ))
    
    async def _measure_query_performance(self):
        """Measure database query performance."""
        sqlite_path = Path(".genesis/data/genesis.db")
        
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Create test table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS perf_test (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)
            
            # Insert test data
            test_data = []
            for i in range(1000):
                test_data.append((
                    f"BTC/USDT",
                    50000.0 + i,
                    100.0 + i,
                    int(time.time() * 1000) + i
                ))
            
            cursor.executemany(
                "INSERT INTO perf_test (symbol, price, volume, timestamp) VALUES (?, ?, ?, ?)",
                test_data
            )
            conn.commit()
            
            # Test query performance
            queries = [
                ("Simple SELECT", "SELECT * FROM perf_test LIMIT 100"),
                ("Aggregation", "SELECT symbol, AVG(price), SUM(volume) FROM perf_test GROUP BY symbol"),
                ("Range query", "SELECT * FROM perf_test WHERE price > 50500 AND price < 50600"),
                ("Order by", "SELECT * FROM perf_test ORDER BY timestamp DESC LIMIT 50"),
            ]
            
            for query_name, query in queries:
                start = time.perf_counter()
                cursor.execute(query)
                cursor.fetchall()
                latency_ms = (time.perf_counter() - start) * 1000
                
                self.check_threshold(
                    latency_ms,
                    10,
                    "<",
                    f"Query '{query_name}'",
                    "ms",
                    ValidationSeverity.WARNING if latency_ms < 20 else ValidationSeverity.ERROR
                )
            
            # Cleanup
            cursor.execute("DROP TABLE IF EXISTS perf_test")
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Query performance test failed",
                details={"error": str(e)}
            ))
    
    async def _verify_migrations(self):
        """Verify database migration status."""
        alembic_path = Path("alembic/versions")
        
        if alembic_path.exists():
            migration_files = list(alembic_path.glob("*.py"))
            
            self.check_condition(
                len(migration_files) > 0,
                f"Found {len(migration_files)} migration files",
                "No migration files found",
                ValidationSeverity.WARNING,
                details={"count": len(migration_files)},
                recommendation="Create database migrations using Alembic"
            )
            
            # Check for critical migrations
            expected_migrations = [
                "001_initial_schema.py",
                "002_add_correlation_view.py",
                "003_sqlite_to_postgres.py"
            ]
            
            for migration in expected_migrations:
                migration_path = alembic_path / migration
                if migration_path.exists():
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Migration present: {migration}"
                    ))
                else:
                    severity = ValidationSeverity.WARNING if "postgres" in migration else ValidationSeverity.ERROR
                    self.result.add_issue(ValidationIssue(
                        severity=severity,
                        message=f"Missing migration: {migration}",
                        recommendation=f"Create migration: alembic revision -m '{migration.replace('.py', '')}'"
                    ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Alembic migrations directory not found",
                recommendation="Initialize Alembic: alembic init alembic"
            ))
    
    async def _check_indexes(self):
        """Check database index optimization."""
        sqlite_path = Path(".genesis/data/genesis.db")
        
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Check for important indexes
            cursor.execute("""
                SELECT name, tbl_name, sql 
                FROM sqlite_master 
                WHERE type = 'index' AND sql IS NOT NULL
            """)
            
            indexes = cursor.fetchall()
            
            if len(indexes) > 0:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Found {len(indexes)} indexes",
                    details={"indexes": [idx[0] for idx in indexes]}
                ))
            else:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="No indexes found",
                    recommendation="Add indexes for frequently queried columns"
                ))
            
            conn.close()
            
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Index check failed",
                details={"error": str(e)}
            ))
    
    async def _test_connection_pooling(self):
        """Test database connection pooling."""
        try:
            from genesis.data.repository import RepositoryFactory
            
            factory = RepositoryFactory()
            
            # Test concurrent connections
            tasks = []
            for i in range(10):
                repo = factory.get_repository()
                tasks.append(repo.health_check())
            
            start = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration_ms = (time.perf_counter() - start) * 1000
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            
            self.check_condition(
                successful == len(tasks),
                f"Connection pool handled {successful}/{len(tasks)} concurrent connections",
                f"Connection pool failures: {len(tasks) - successful}/{len(tasks)}",
                ValidationSeverity.ERROR if successful < len(tasks) * 0.8 else ValidationSeverity.WARNING,
                details={
                    "total": len(tasks),
                    "successful": successful,
                    "duration_ms": duration_ms
                }
            )
            
            self.check_threshold(
                duration_ms,
                1000,
                "<",
                "Connection pool response time",
                "ms",
                ValidationSeverity.WARNING
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Repository factory not implemented",
                recommendation="Implement RepositoryFactory for database abstraction"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Connection pooling test failed",
                details={"error": str(e)}
            ))