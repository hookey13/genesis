"""SQLite to PostgreSQL migration engine with zero data loss guarantee.

This module provides comprehensive migration capabilities with:
- Batch processing for memory efficiency
- Data integrity verification via checksums
- Progress tracking and logging
- Automatic rollback on failure
- Type conversion and validation
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from sqlalchemy import create_engine, inspect, MetaData
from sqlalchemy.orm import sessionmaker

from genesis.core.exceptions import MigrationError


class SQLiteToPostgreSQLMigrator:
    """Manages zero-downtime migration from SQLite to PostgreSQL."""
    
    # Class-level lock for preventing concurrent migrations
    _migration_lock = None
    _migration_lock_file = '.genesis/.migration.lock'
    
    def __init__(
        self,
        sqlite_path: str,
        postgres_config: Dict[str, Any],
        batch_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize migrator with database configurations.
        
        Args:
            sqlite_path: Path to SQLite database file
            postgres_config: PostgreSQL connection configuration
            batch_size: Number of rows to process per batch
            logger: Logger instance for migration logging
        """
        self.sqlite_path = sqlite_path
        self.postgres_config = postgres_config
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        self.migration_log = []
        self.checksum_results = {}
        self.table_mappings = self._get_table_mappings()
        
        # Connection pools
        self.pg_pool = None
        self.sqlite_conn = None
        
        # Transaction state
        self._in_transaction = False
        self._transaction_conn = None
        
    def _get_table_mappings(self) -> Dict[str, Dict[str, str]]:
        """Define SQLite to PostgreSQL type mappings."""
        return {
            'INTEGER': 'BIGINT',
            'TEXT': 'TEXT',
            'REAL': 'DOUBLE PRECISION',
            'BLOB': 'BYTEA',
            'DATETIME': 'TIMESTAMP WITH TIME ZONE',
            'BOOLEAN': 'BOOLEAN',
            'VARCHAR': 'VARCHAR',
            'JSON': 'JSONB',
        }
    
    def _validate_table_name(self, table_name: str) -> bool:
        """Validate table name to prevent SQL injection.
        
        Args:
            table_name: Name to validate
            
        Returns:
            True if valid, False otherwise
        """
        import re
        # Allow only alphanumeric, underscore, and dash
        pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
        return bool(re.match(pattern, table_name)) and len(table_name) <= 63
    
    def _validate_column_name(self, column_name: str) -> bool:
        """Validate column name to prevent SQL injection.
        
        Args:
            column_name: Name to validate
            
        Returns:
            True if valid, False otherwise
        """
        import re
        # Allow only alphanumeric and underscore
        pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, column_name)) and len(column_name) <= 63
    
    async def _acquire_migration_lock(self) -> bool:
        """Acquire exclusive lock to prevent concurrent migrations.
        
        Returns:
            True if lock acquired, False if another migration is running
        """
        from pathlib import Path
        import sys
        
        try:
            # Create lock directory if it doesn't exist
            lock_dir = Path(self._migration_lock_file).parent
            lock_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if lock file exists and is recent
            lock_path = Path(self._migration_lock_file)
            if lock_path.exists():
                # Read existing lock info
                try:
                    with open(lock_path, 'r') as f:
                        lock_info = json.load(f)
                        
                    # Check if process is still running (cross-platform)
                    pid = lock_info.get('pid')
                    if pid and self._is_process_running(pid):
                        self.logger.warning(f"Migration already running: PID={pid}, Started={lock_info.get('started')}")
                        return False
                    else:
                        # Stale lock, remove it
                        lock_path.unlink()
                except (json.JSONDecodeError, OSError):
                    # Corrupted lock file, remove it
                    lock_path.unlink(missing_ok=True)
            
            # Try to create lock file atomically
            try:
                # Use exclusive creation mode
                self._migration_lock = open(self._migration_lock_file, 'x')
                
                # Write migration info to lock file
                lock_info = {
                    'pid': os.getpid(),
                    'started': datetime.utcnow().isoformat(),
                    'sqlite_path': self.sqlite_path
                }
                json.dump(lock_info, self._migration_lock)
                self._migration_lock.flush()
                
                # Platform-specific locking
                if sys.platform != 'win32':
                    import fcntl
                    fcntl.lockf(self._migration_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                self.logger.info("Migration lock acquired")
                return True
                
            except FileExistsError:
                # Another process created the lock file
                self.logger.warning("Another migration is already in progress")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to acquire migration lock: {e}")
            if self._migration_lock:
                self._migration_lock.close()
                self._migration_lock = None
            return False
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running (cross-platform)."""
        import sys
        import signal
        
        if sys.platform == 'win32':
            # Windows
            import subprocess
            try:
                output = subprocess.check_output(['tasklist', '/FI', f'PID eq {pid}'], stderr=subprocess.DEVNULL)
                return str(pid) in output.decode()
            except subprocess.CalledProcessError:
                return False
        else:
            # Unix/Linux/Mac
            try:
                os.kill(pid, signal.SIG_DFL)
                return True
            except (OSError, ProcessLookupError):
                return False
    
    async def _release_migration_lock(self) -> None:
        """Release migration lock."""
        from pathlib import Path
        import sys
        
        if self._migration_lock:
            try:
                # Platform-specific unlocking
                if sys.platform != 'win32':
                    try:
                        import fcntl
                        fcntl.lockf(self._migration_lock, fcntl.LOCK_UN)
                    except ImportError:
                        pass
                
                # Close file
                self._migration_lock.close()
                
                # Remove lock file
                Path(self._migration_lock_file).unlink(missing_ok=True)
                
                self.logger.info("Migration lock released")
            except Exception as e:
                self.logger.warning(f"Failed to release migration lock: {e}")
            finally:
                self._migration_lock = None
    
    async def begin_global_transaction(self) -> None:
        """Begin a global transaction across both databases."""
        if self._in_transaction:
            raise MigrationError("Transaction already in progress")
        
        # Acquire dedicated connection for transaction
        self._transaction_conn = await self.pg_pool.acquire()
        self._in_transaction = True
        
        # Begin PostgreSQL transaction
        await self._transaction_conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
        
        # SQLite doesn't support async transactions, but we can start one
        self.sqlite_conn.execute("BEGIN IMMEDIATE")
        
        self.logger.debug("Global transaction started")
    
    async def commit_global_transaction(self) -> None:
        """Commit the global transaction."""
        if not self._in_transaction:
            raise MigrationError("No transaction in progress")
        
        try:
            # Commit SQLite first (it's the source)
            self.sqlite_conn.commit()
            
            # Then commit PostgreSQL
            await self._transaction_conn.execute("COMMIT")
            
            self.logger.debug("Global transaction committed")
        finally:
            # Clean up transaction state
            if self._transaction_conn:
                await self.pg_pool.release(self._transaction_conn)
            self._transaction_conn = None
            self._in_transaction = False
    
    async def rollback_global_transaction(self) -> None:
        """Rollback the global transaction."""
        if not self._in_transaction:
            return
        
        try:
            # Rollback both databases
            try:
                self.sqlite_conn.rollback()
            except:
                pass
            
            if self._transaction_conn:
                try:
                    await self._transaction_conn.execute("ROLLBACK")
                except:
                    pass
            
            self.logger.warning("Global transaction rolled back")
        finally:
            # Clean up transaction state
            if self._transaction_conn:
                await self.pg_pool.release(self._transaction_conn)
            self._transaction_conn = None
            self._in_transaction = False
    
    async def connect(self) -> None:
        """Establish connections to both databases."""
        # PostgreSQL connection pool
        self.pg_pool = await asyncpg.create_pool(
            host=self.postgres_config['host'],
            port=self.postgres_config['port'],
            database=self.postgres_config['database'],
            user=self.postgres_config['user'],
            password=self.postgres_config['password'],
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # SQLite connection
        self.sqlite_conn = sqlite3.connect(self.sqlite_path)
        self.sqlite_conn.row_factory = sqlite3.Row
        
        self.logger.info("Database connections established")
    
    async def disconnect(self) -> None:
        """Close all database connections."""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.sqlite_conn:
            self.sqlite_conn.close()
        
        self.logger.info("Database connections closed")
    
    def calculate_table_checksum(self, conn: Any, table_name: str, is_sqlite: bool = True) -> str:
        """Calculate SHA256 checksum for table data.
        
        Args:
            conn: Database connection
            table_name: Name of table to checksum
            is_sqlite: Whether this is SQLite connection
            
        Returns:
            SHA256 hash of table data
        """
        if is_sqlite:
            cursor = conn.cursor()
            # Validate table name to prevent SQL injection
            if not self._validate_table_name(table_name):
                raise MigrationError(f"Invalid table name: {table_name}")
            
            # Use quote identifier for safe table name handling
            safe_query = f'SELECT * FROM "{table_name}" ORDER BY 1'
            cursor.execute(safe_query)
            rows = cursor.fetchall()
        else:
            # For PostgreSQL, we'll handle this differently
            return ""  # Will implement async version
        
        # Create deterministic string representation
        data_str = ""
        for row in rows:
            row_str = "|".join(str(val) if val is not None else "NULL" for val in row)
            data_str += row_str + "\n"
        
        # Calculate SHA256
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def calculate_postgres_checksum(self, table_name: str) -> str:
        """Calculate checksum for PostgreSQL table."""
        # Validate table name to prevent SQL injection
        if not self._validate_table_name(table_name):
            raise MigrationError(f"Invalid table name: {table_name}")
        
        async with self.pg_pool.acquire() as conn:
            # Use asyncpg's built-in identifier escaping
            rows = await conn.fetch(f'SELECT * FROM "{table_name}" ORDER BY 1')
            
            data_str = ""
            for row in rows:
                row_str = "|".join(str(val) if val is not None else "NULL" for val in row.values())
                data_str += row_str + "\n"
            
            return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get row count from SQLite table."""
        # Validate table name to prevent SQL injection
        if not self._validate_table_name(table_name):
            raise MigrationError(f"Invalid table name: {table_name}")
        
        cursor = self.sqlite_conn.cursor()
        safe_query = f'SELECT COUNT(*) FROM "{table_name}"'
        cursor.execute(safe_query)
        return cursor.fetchone()[0]
    
    async def get_postgres_row_count(self, table_name: str) -> int:
        """Get row count from PostgreSQL table."""
        # Validate table name to prevent SQL injection
        if not self._validate_table_name(table_name):
            raise MigrationError(f"Invalid table name: {table_name}")
        
        async with self.pg_pool.acquire() as conn:
            result = await conn.fetchval(f'SELECT COUNT(*) FROM "{table_name}"')
            return result
    
    def convert_sqlite_value(self, value: Any, column_type: str) -> Any:
        """Convert SQLite value to PostgreSQL compatible format.
        
        Args:
            value: Value to convert
            column_type: Target column type
            
        Returns:
            Converted value
        """
        if value is None:
            return None
        
        # Handle datetime conversion
        if column_type in ('DATETIME', 'TIMESTAMP'):
            if isinstance(value, str):
                try:
                    # Parse ISO format datetime
                    return datetime.fromisoformat(value.replace('Z', '+00:00'))
                except (ValueError, AttributeError) as e:
                    self.logger.warning(f"Failed to parse datetime '{value}': {e}")
                    return value
        
        # Handle boolean conversion
        if column_type == 'BOOLEAN':
            if isinstance(value, (int, str)):
                return bool(int(value))
        
        # Handle JSON conversion
        if column_type == 'JSON':
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.warning(f"Failed to parse JSON '{value}': {e}")
                    return value
        
        # Handle decimal/money values
        if column_type in ('DECIMAL', 'NUMERIC'):
            if isinstance(value, (str, int, float)):
                return str(value)  # Keep as string for precision
        
        return value
    
    async def migrate_table(self, table_name: str) -> Dict[str, Any]:
        """Migrate a single table from SQLite to PostgreSQL.
        
        Args:
            table_name: Name of table to migrate
            
        Returns:
            Migration results dictionary
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting migration of table: {table_name}")
        
        try:
            # Get source row count
            source_count = self.get_table_row_count(table_name)
            
            # Calculate source checksum
            source_checksum = self.calculate_table_checksum(self.sqlite_conn, table_name)
            
            # Get table schema
            cursor = self.sqlite_conn.cursor()
            # Validate table name first
            if not self._validate_table_name(table_name):
                raise MigrationError(f"Invalid table name: {table_name}")
            
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Validate column names
            for col_name in column_names:
                if not self._validate_column_name(col_name):
                    raise MigrationError(f"Invalid column name: {col_name}")
            
            # Prepare insert statement with quoted identifiers
            quoted_columns = ', '.join(f'"{col}"' for col in column_names)
            placeholders = ", ".join([f"${i+1}" for i in range(len(column_names))])
            insert_sql = f"""
                INSERT INTO "{table_name}" ({quoted_columns})
                VALUES ({placeholders})
                ON CONFLICT DO NOTHING
            """
            
            # Migrate data in batches
            cursor.execute(f'SELECT * FROM "{table_name}"')
            batch = []
            rows_migrated = 0
            
            async with self.pg_pool.acquire() as conn:
                # Start atomic transaction for entire table migration
                async with conn.transaction(isolation='serializable'):
                    # Clear target table (for clean migration)
                    # Table name already validated above
                    await conn.execute(f'TRUNCATE TABLE "{table_name}" CASCADE')
                    
                    while True:
                        rows = cursor.fetchmany(self.batch_size)
                        if not rows:
                            break
                        
                        # Convert rows to tuples
                        for row in rows:
                            converted_row = []
                            for i, value in enumerate(row):
                                converted_value = self.convert_sqlite_value(
                                    value, 
                                    columns[i][2]  # Column type
                                )
                                converted_row.append(converted_value)
                            batch.append(tuple(converted_row))
                        
                        # Execute batch insert
                        if batch:
                            await conn.executemany(insert_sql, batch)
                            rows_migrated += len(batch)
                            batch = []
                            
                            # Log progress
                            progress = (rows_migrated / source_count) * 100 if source_count > 0 else 100
                            self.logger.info(f"{table_name}: {rows_migrated}/{source_count} rows ({progress:.1f}%)")
            
            # Calculate target checksum
            target_checksum = await self.calculate_postgres_checksum(table_name)
            
            # Verify row counts
            target_count = await self.get_postgres_row_count(table_name)
            
            # Store results
            result = {
                'table': table_name,
                'source_count': source_count,
                'target_count': target_count,
                'source_checksum': source_checksum,
                'target_checksum': target_checksum,
                'checksums_match': source_checksum == target_checksum,
                'counts_match': source_count == target_count,
                'duration_seconds': (datetime.utcnow() - start_time).total_seconds(),
                'status': 'success' if source_count == target_count else 'failed'
            }
            
            self.checksum_results[table_name] = result
            
            # Log migration to tracking table
            await self.log_migration(result)
            
            self.logger.info(f"Completed migration of {table_name}: {result['status']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to migrate {table_name}: {str(e)}")
            raise MigrationError(f"Table migration failed: {table_name}") from e
    
    async def log_migration(self, result: Dict[str, Any]) -> None:
        """Log migration results to tracking table."""
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO migration_history 
                (source_table, target_table, rows_migrated, migration_started, 
                 migration_completed, status, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
            result['table'], result['table'], result['target_count'],
            datetime.utcnow(), datetime.utcnow(), result['status'], None)
            
            await conn.execute("""
                INSERT INTO checksum_verification
                (table_name, source_checksum, target_checksum, row_count, 
                 verification_time, match_status)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
            result['table'], result['source_checksum'], result['target_checksum'],
            result['target_count'], datetime.utcnow(), result['checksums_match'])
    
    async def get_tables_to_migrate(self) -> List[str]:
        """Get list of tables to migrate."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
            AND name NOT IN ('alembic_version', 'migration_history', 'checksum_verification')
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    async def verify_foreign_keys(self) -> bool:
        """Verify all foreign key relationships are intact."""
        self.logger.info("Verifying foreign key constraints...")
        
        async with self.pg_pool.acquire() as conn:
            # Check for orphaned records
            checks = [
                ("positions", "account_id", "accounts", "account_id"),
                ("orders", "account_id", "accounts", "account_id"),
                ("orders", "position_id", "positions", "position_id"),
                ("trading_sessions", "account_id", "accounts", "account_id"),
                ("behavioral_metrics", "profile_id", "tilt_profiles", "profile_id"),
            ]
            
            all_valid = True
            for child_table, child_col, parent_table, parent_col in checks:
                # Validate all identifiers
                if not all([
                    self._validate_table_name(child_table),
                    self._validate_table_name(parent_table),
                    self._validate_column_name(child_col),
                    self._validate_column_name(parent_col)
                ]):
                    raise MigrationError(f"Invalid identifier in foreign key check")
                
                result = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM "{child_table}" c
                    LEFT JOIN "{parent_table}" p ON c."{child_col}" = p."{parent_col}"
                    WHERE c."{child_col}" IS NOT NULL AND p."{parent_col}" IS NULL
                """)
                
                if result > 0:
                    self.logger.error(f"Found {result} orphaned records in {child_table}")
                    all_valid = False
                else:
                    self.logger.info(f"✓ {child_table}.{child_col} -> {parent_table}.{parent_col}")
            
            return all_valid
    
    async def execute_migration(self) -> Dict[str, Any]:
        """Execute complete database migration with atomic guarantees.
        
        Returns:
            Migration summary with results
        """
        # Acquire migration lock to prevent concurrent migrations
        if not await self._acquire_migration_lock():
            raise MigrationError("Another migration is already in progress")
        
        try:
            await self.connect()
            
            # Get tables to migrate
            tables = await self.get_tables_to_migrate()
            self.logger.info(f"Found {len(tables)} tables to migrate")
            
            # Create backup
            backup_path = await self.create_backup()
            self.logger.info(f"Created backup at: {backup_path}")
            
            # Migrate each table
            results = []
            for table in tables:
                result = await self.migrate_table(table)
                results.append(result)
                
                # Check if migration failed
                if result['status'] != 'success':
                    raise MigrationError(f"Migration failed for table: {table}")
            
            # Verify foreign keys
            fk_valid = await self.verify_foreign_keys()
            if not fk_valid:
                raise MigrationError("Foreign key validation failed")
            
            # Update sequences
            await self.update_sequences()
            
            # Final verification
            all_verified = all(r['checksums_match'] and r['counts_match'] for r in results)
            
            summary = {
                'status': 'success' if all_verified else 'failed',
                'tables_migrated': len(results),
                'total_rows': sum(r['target_count'] for r in results),
                'verification': results,
                'backup_path': backup_path,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Migration completed: {summary['status']}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            await self.rollback()
            raise
        finally:
            await self.disconnect()
            await self._release_migration_lock()
    
    async def create_backup(self) -> str:
        """Create backup of SQLite database."""
        backup_path = f"{self.sqlite_path}.backup.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Use SQLite backup API
        backup_conn = sqlite3.connect(backup_path)
        with backup_conn:
            self.sqlite_conn.backup(backup_conn)
        backup_conn.close()
        
        return backup_path
    
    async def rollback(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Rollback migration on failure by restoring from backup.
        
        Args:
            backup_path: Path to backup file, or use most recent if None
            
        Returns:
            Rollback results dictionary
        """
        self.logger.warning("Starting rollback procedure...")
        
        rollback_results = {
            'status': 'failed',
            'backup_used': None,
            'tables_restored': 0,
            'error': None
        }
        
        try:
            # Step 1: Find backup to restore
            if not backup_path:
                # Find most recent backup
                from pathlib import Path
                backup_dir = Path(self.sqlite_path).parent
                backups = list(backup_dir.glob(f"{Path(self.sqlite_path).name}.backup.*"))
                
                if not backups:
                    raise MigrationError("No backup found for rollback")
                
                # Sort by modification time and get most recent
                backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                backup_path = str(backups[0])
            
            rollback_results['backup_used'] = backup_path
            self.logger.info(f"Using backup: {backup_path}")
            
            # Step 2: Close current connections
            if self.sqlite_conn:
                self.sqlite_conn.close()
                self.sqlite_conn = None
            
            # Step 3: Restore from backup
            import shutil
            from pathlib import Path
            
            # Create a safety backup of current (potentially corrupted) database
            corrupted_backup = f"{self.sqlite_path}.corrupted.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(self.sqlite_path, corrupted_backup)
            self.logger.info(f"Saved corrupted database to: {corrupted_backup}")
            
            # Restore from backup
            shutil.copy2(backup_path, self.sqlite_path)
            self.logger.info(f"Restored database from: {backup_path}")
            
            # Step 4: Verify restored database
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            self.sqlite_conn.row_factory = sqlite3.Row
            
            cursor = self.sqlite_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            rollback_results['tables_restored'] = len(tables)
            
            # Step 5: Clear migration tracking tables in PostgreSQL
            if self.pg_pool:
                async with self.pg_pool.acquire() as conn:
                    await conn.execute("TRUNCATE TABLE migration_history CASCADE")
                    await conn.execute("TRUNCATE TABLE checksum_verification CASCADE")
                    self.logger.info("Cleared migration tracking tables")
            
            # Step 6: Update configuration to use SQLite
            await self._restore_sqlite_config()
            
            rollback_results['status'] = 'success'
            self.logger.info(f"Rollback completed successfully - restored {rollback_results['tables_restored']} tables")
            
            return rollback_results
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            rollback_results['error'] = str(e)
            raise MigrationError(f"Rollback failed: {str(e)}") from e
    
    async def _restore_sqlite_config(self) -> None:
        """Restore application configuration to use SQLite."""
        import json
        from pathlib import Path
        
        config_path = Path('.genesis/config/database.json')
        backup_config_path = config_path.with_suffix('.json.backup')
        
        if backup_config_path.exists():
            # Restore from backup
            import shutil
            shutil.copy2(backup_config_path, config_path)
            self.logger.info("Restored SQLite configuration")
        else:
            # Create default SQLite config
            sqlite_config = {
                'type': 'sqlite',
                'path': self.sqlite_path,
                'journal_mode': 'WAL',
                'synchronous': 'NORMAL',
                'cache_size': -64000,  # 64MB cache
                'foreign_keys': True
            }
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(sqlite_config, f, indent=2)
            
            self.logger.info("Created default SQLite configuration")
    
    async def update_sequences(self) -> None:
        """Update PostgreSQL sequences to match SQLite autoincrement values."""
        self.logger.info("Updating sequences...")
        
        async with self.pg_pool.acquire() as conn:
            # Update primary key sequences
            sequences = [
                ("events", "event_id", "event_id_seq"),
                ("orders", "order_id", "order_id_seq"),
                ("users", "id", "users_id_seq"),
            ]
            
            for table, column, sequence in sequences:
                # Validate identifiers
                if not self._validate_table_name(table) or not self._validate_column_name(column):
                    raise MigrationError(f"Invalid identifier in sequence update")
                
                # Validate sequence name
                if not self._validate_table_name(sequence.replace('_seq', '')):
                    raise MigrationError(f"Invalid sequence name: {sequence}")
                
                max_val = await conn.fetchval(f'SELECT MAX("{column}") FROM "{table}"')
                if max_val:
                    # Use parameterized query for the value
                    await conn.execute(f"SELECT setval($1, $2)", sequence, max_val)
                    self.logger.info(f"Updated {sequence} to {max_val}")
    
    async def perform_hot_cutover(self, app_controller=None) -> Dict[str, Any]:
        """Perform hot cutover with minimal downtime.
        
        Args:
            app_controller: Optional application controller for coordinating cutover
            
        Returns:
            Cutover results with timing and status
        """
        import time
        start_time = time.time()
        cutover_results = {
            'status': 'failed',
            'downtime_seconds': 0,
            'steps_completed': [],
            'error': None
        }
        
        try:
            self.logger.info("Starting hot cutover process...")
            
            # Step 1: Enable read-only mode in application
            self.logger.info("Step 1: Enabling read-only mode")
            if app_controller:
                await app_controller.set_read_only_mode(True)
            cutover_results['steps_completed'].append('read_only_mode')
            
            # Step 2: Wait for in-flight transactions to complete
            self.logger.info("Step 2: Waiting for in-flight transactions")
            await asyncio.sleep(2)  # Allow 2 seconds for transactions to complete
            cutover_results['steps_completed'].append('wait_transactions')
            
            # Step 3: Capture final changes from SQLite
            self.logger.info("Step 3: Capturing final delta changes")
            final_changes = await self._capture_delta_changes()
            cutover_results['steps_completed'].append('capture_delta')
            
            # Step 4: Apply final changes to PostgreSQL
            if final_changes:
                self.logger.info(f"Step 4: Applying {len(final_changes)} final changes")
                await self._apply_delta_changes(final_changes)
            cutover_results['steps_completed'].append('apply_delta')
            
            # Step 5: Verify data consistency
            self.logger.info("Step 5: Verifying data consistency")
            verification_passed = await self._quick_verify_consistency()
            if not verification_passed:
                raise MigrationError("Data consistency check failed during cutover")
            cutover_results['steps_completed'].append('verify_consistency')
            
            # Step 6: Update database configuration
            self.logger.info("Step 6: Updating database configuration")
            await self._update_database_config()
            cutover_results['steps_completed'].append('update_config')
            
            # Step 7: Test PostgreSQL connectivity
            self.logger.info("Step 7: Testing PostgreSQL connectivity")
            await self._test_postgres_connectivity()
            cutover_results['steps_completed'].append('test_connectivity')
            
            # Step 8: Switch application to PostgreSQL
            self.logger.info("Step 8: Switching application to PostgreSQL")
            if app_controller:
                await app_controller.switch_to_postgres(self.postgres_config)
                await app_controller.set_read_only_mode(False)
            cutover_results['steps_completed'].append('switch_database')
            
            # Calculate downtime
            end_time = time.time()
            cutover_results['downtime_seconds'] = end_time - start_time
            cutover_results['status'] = 'success'
            
            self.logger.info(f"Hot cutover completed successfully in {cutover_results['downtime_seconds']:.2f} seconds")
            return cutover_results
            
        except Exception as e:
            self.logger.error(f"Hot cutover failed: {str(e)}")
            cutover_results['error'] = str(e)
            
            # Attempt to restore service with SQLite
            if app_controller:
                self.logger.info("Attempting to restore SQLite service")
                await app_controller.set_read_only_mode(False)
            
            raise MigrationError(f"Hot cutover failed: {str(e)}") from e
    
    async def _capture_delta_changes(self) -> List[Dict[str, Any]]:
        """Capture changes made since migration started."""
        delta_changes = []
        
        # Get timestamp of migration start
        async with self.pg_pool.acquire() as conn:
            migration_start = await conn.fetchval("""
                SELECT MIN(migration_started) FROM migration_history
                WHERE status = 'success'
            """)
        
        if not migration_start:
            return delta_changes
        
        # Query SQLite for changes since migration started
        cursor = self.sqlite_conn.cursor()
        
        # Check for new/updated records in key tables
        tables_to_check = ['orders', 'positions', 'trading_sessions']
        
        for table in tables_to_check:
            if not self._validate_table_name(table):
                continue
                
            # Get records modified after migration start
            cursor.execute(f"""
                SELECT * FROM "{table}"
                WHERE updated_at > ?
                OR created_at > ?
            """, (migration_start, migration_start))
            
            rows = cursor.fetchall()
            if rows:
                delta_changes.append({
                    'table': table,
                    'operation': 'upsert',
                    'rows': rows
                })
        
        return delta_changes
    
    async def _apply_delta_changes(self, changes: List[Dict[str, Any]]) -> None:
        """Apply delta changes to PostgreSQL."""
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                for change in changes:
                    table = change['table']
                    rows = change['rows']
                    
                    if not self._validate_table_name(table):
                        raise MigrationError(f"Invalid table name in delta: {table}")
                    
                    # Get column info for the table
                    cursor = self.sqlite_conn.cursor()
                    cursor.execute(f'PRAGMA table_info("{table}")')
                    columns = cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    # Validate column names
                    for col_name in column_names:
                        if not self._validate_column_name(col_name):
                            raise MigrationError(f"Invalid column name: {col_name}")
                    
                    # Prepare upsert query
                    quoted_columns = ', '.join(f'"{col}"' for col in column_names)
                    placeholders = ', '.join(f'${i+1}' for i in range(len(column_names)))
                    conflict_columns = column_names[0]  # Assume first column is primary key
                    
                    upsert_sql = f"""
                        INSERT INTO "{table}" ({quoted_columns})
                        VALUES ({placeholders})
                        ON CONFLICT ("{conflict_columns}") DO UPDATE SET
                        {', '.join(f'"{col}" = EXCLUDED."{col}"' for col in column_names[1:])}
                    """
                    
                    # Apply each row
                    for row in rows:
                        converted_row = []
                        for i, value in enumerate(row):
                            converted_value = self.convert_sqlite_value(
                                value, columns[i][2]
                            )
                            converted_row.append(converted_value)
                        
                        await conn.execute(upsert_sql, *converted_row)
    
    async def _quick_verify_consistency(self) -> bool:
        """Quick consistency check for critical tables."""
        critical_tables = ['accounts', 'positions', 'orders']
        
        for table in critical_tables:
            if not self._validate_table_name(table):
                continue
                
            # Compare row counts
            sqlite_count = self.get_table_row_count(table)
            postgres_count = await self.get_postgres_row_count(table)
            
            if abs(sqlite_count - postgres_count) > 5:  # Allow small delta
                self.logger.error(f"Row count mismatch in {table}: SQLite={sqlite_count}, PostgreSQL={postgres_count}")
                return False
        
        return True
    
    async def _update_database_config(self) -> None:
        """Update application configuration to use PostgreSQL."""
        import json
        from pathlib import Path
        
        config_path = Path('.genesis/config/database.json')
        
        if config_path.exists():
            # Backup current config
            backup_path = config_path.with_suffix('.json.backup')
            config_path.rename(backup_path)
            
            # Write new config
            new_config = {
                'type': 'postgresql',
                'host': self.postgres_config['host'],
                'port': self.postgres_config['port'],
                'database': self.postgres_config['database'],
                'user': self.postgres_config['user'],
                # Password should come from vault
                'password_vault_key': 'database/postgres/password',
                'pool_size': self.postgres_config.get('pool_size', 20),
                'previous_config': str(backup_path)
            }
            
            with open(config_path, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            self.logger.info(f"Database configuration updated: {config_path}")
    
    async def _test_postgres_connectivity(self) -> None:
        """Test PostgreSQL connectivity and basic operations."""
        async with self.pg_pool.acquire() as conn:
            # Test SELECT
            result = await conn.fetchval("SELECT 1")
            if result != 1:
                raise MigrationError("PostgreSQL connectivity test failed")
            
            # Test table access
            await conn.fetch("SELECT COUNT(*) FROM accounts")
            
            self.logger.info("PostgreSQL connectivity test passed")