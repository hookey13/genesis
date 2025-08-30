"""
Database Optimization and Maintenance System for Project GENESIS.

Provides automated database maintenance including VACUUM, index optimization,
query performance tracking, and scheduled maintenance windows.
"""

import asyncio
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

import aiosqlite
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import StaticPool

from genesis.utils.logger import get_logger, LoggerType, PerformanceLogger


class QueryPerformanceStats(BaseModel):
    """Statistics for query performance."""
    
    query_hash: str = Field(description="Hash of query for identification")
    query_template: str = Field(description="Query template with placeholders")
    execution_count: int = Field(description="Number of executions")
    total_time_ms: float = Field(description="Total execution time")
    avg_time_ms: float = Field(description="Average execution time")
    min_time_ms: float = Field(description="Minimum execution time")
    max_time_ms: float = Field(description="Maximum execution time")
    p50_time_ms: float = Field(description="50th percentile time")
    p95_time_ms: float = Field(description="95th percentile time")
    p99_time_ms: float = Field(description="99th percentile time")
    last_executed: datetime = Field(description="Last execution timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }


class IndexUsageStats(BaseModel):
    """Statistics for index usage."""
    
    table_name: str = Field(description="Table name")
    index_name: str = Field(description="Index name")
    columns: List[str] = Field(description="Indexed columns")
    usage_count: int = Field(description="Number of times used")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    size_bytes: int = Field(description="Index size in bytes")
    selectivity: float = Field(description="Index selectivity ratio")
    recommendation: Optional[str] = Field(None, description="Optimization recommendation")


class MaintenanceConfig(BaseModel):
    """Configuration for database maintenance."""
    
    vacuum_enabled: bool = Field(True, description="Enable VACUUM operations")
    analyze_enabled: bool = Field(True, description="Enable ANALYZE operations")
    index_optimization_enabled: bool = Field(True, description="Enable index optimization")
    maintenance_window_start: int = Field(2, description="Start hour (0-23)")
    maintenance_window_duration: int = Field(2, description="Duration in hours")
    backup_before_maintenance: bool = Field(True, description="Backup before maintenance")
    min_days_between_vacuum: int = Field(7, description="Minimum days between VACUUM")
    query_log_retention_days: int = Field(30, description="Query log retention")
    slow_query_threshold_ms: float = Field(100.0, description="Slow query threshold")
    index_usage_threshold: int = Field(10, description="Minimum usage for keeping index")


class DBOptimizer:
    """
    Database optimization and maintenance system.
    
    Features:
        - Automated VACUUM with online backup
        - Index usage analysis and recommendations
        - Query performance tracking
        - Scheduled maintenance windows
        - PostgreSQL migration support
    """
    
    def __init__(
        self,
        config: MaintenanceConfig,
        db_path: Path = Path(".genesis/data/genesis.db"),
        backup_dir: Path = Path(".genesis/backups/db")
    ):
        """
        Initialize database optimizer.
        
        Args:
            config: Maintenance configuration
            db_path: Path to database file
            backup_dir: Directory for database backups
        """
        self.config = config
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.logger = get_logger(__name__, LoggerType.SYSTEM)
        
        # Create directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.query_stats: Dict[str, QueryPerformanceStats] = {}
        self.index_stats: Dict[str, IndexUsageStats] = {}
        
        # Maintenance state
        self.last_vacuum_time: Optional[datetime] = None
        self.last_analyze_time: Optional[datetime] = None
        self._load_maintenance_state()
    
    def _load_maintenance_state(self) -> None:
        """Load maintenance state from file."""
        state_file = self.backup_dir / "maintenance_state.json"
        if state_file.exists():
            try:
                import json
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    if 'last_vacuum' in state:
                        self.last_vacuum_time = datetime.fromisoformat(state['last_vacuum'])
                    if 'last_analyze' in state:
                        self.last_analyze_time = datetime.fromisoformat(state['last_analyze'])
            except Exception as e:
                self.logger.error("failed_to_load_maintenance_state", error=str(e))
    
    def _save_maintenance_state(self) -> None:
        """Save maintenance state to file."""
        state_file = self.backup_dir / "maintenance_state.json"
        try:
            import json
            state = {}
            if self.last_vacuum_time:
                state['last_vacuum'] = self.last_vacuum_time.isoformat()
            if self.last_analyze_time:
                state['last_analyze'] = self.last_analyze_time.isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error("failed_to_save_maintenance_state", error=str(e))
    
    async def perform_vacuum(self, force: bool = False) -> bool:
        """
        Perform VACUUM operation on SQLite database.
        
        Args:
            force: Force vacuum even if not due
            
        Returns:
            True if vacuum performed successfully
        """
        if not self.config.vacuum_enabled and not force:
            return False
        
        # Check if vacuum is due
        if not force and self.last_vacuum_time:
            days_since = (datetime.utcnow() - self.last_vacuum_time).days
            if days_since < self.config.min_days_between_vacuum:
                self.logger.info(
                    "vacuum_not_due",
                    days_since=days_since,
                    min_days=self.config.min_days_between_vacuum
                )
                return False
        
        try:
            # Backup database first
            if self.config.backup_before_maintenance:
                backup_path = await self._backup_database()
                self.logger.info("database_backed_up", backup_path=str(backup_path))
            
            # Get database size before
            size_before = self.db_path.stat().st_size
            
            # Perform VACUUM
            self.logger.info("vacuum_started", database=str(self.db_path))
            
            async with aiosqlite.connect(self.db_path) as db:
                # Set pragmas for optimization
                await db.execute("PRAGMA journal_mode = WAL")
                await db.execute("PRAGMA synchronous = NORMAL")
                await db.execute("PRAGMA foreign_keys = ON")
                
                # Perform VACUUM
                with PerformanceLogger(self.logger, "vacuum_operation"):
                    await db.execute("VACUUM")
                    await db.commit()
            
            # Get database size after
            size_after = self.db_path.stat().st_size
            size_reduction = size_before - size_after
            reduction_percent = (size_reduction / size_before) * 100 if size_before > 0 else 0
            
            self.logger.info(
                "vacuum_completed",
                size_before=size_before,
                size_after=size_after,
                reduction_bytes=size_reduction,
                reduction_percent=round(reduction_percent, 2)
            )
            
            # Update state
            self.last_vacuum_time = datetime.utcnow()
            self._save_maintenance_state()
            
            return True
            
        except Exception as e:
            self.logger.error("vacuum_failed", error=str(e))
            return False
    
    async def perform_analyze(self) -> bool:
        """
        Perform ANALYZE operation to update SQLite statistics.
        
        Returns:
            True if analyze performed successfully
        """
        if not self.config.analyze_enabled:
            return False
        
        try:
            self.logger.info("analyze_started", database=str(self.db_path))
            
            async with aiosqlite.connect(self.db_path) as db:
                with PerformanceLogger(self.logger, "analyze_operation"):
                    await db.execute("ANALYZE")
                    await db.commit()
            
            self.logger.info("analyze_completed")
            
            # Update state
            self.last_analyze_time = datetime.utcnow()
            self._save_maintenance_state()
            
            return True
            
        except Exception as e:
            self.logger.error("analyze_failed", error=str(e))
            return False
    
    async def analyze_indexes(self) -> List[IndexUsageStats]:
        """
        Analyze index usage and provide recommendations.
        
        Returns:
            List of index usage statistics with recommendations
        """
        index_stats = []
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get all indexes
                cursor = await db.execute("""
                    SELECT name, tbl_name, sql 
                    FROM sqlite_master 
                    WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
                """)
                indexes = await cursor.fetchall()
                
                for index_name, table_name, index_sql in indexes:
                    # Get index columns
                    columns = self._extract_index_columns(index_sql) if index_sql else []
                    
                    # Analyze index statistics
                    stats = IndexUsageStats(
                        table_name=table_name,
                        index_name=index_name,
                        columns=columns,
                        usage_count=0,  # Would need query log analysis
                        last_used=None,
                        size_bytes=await self._get_index_size(db, index_name),
                        selectivity=await self._calculate_selectivity(db, table_name, columns),
                        recommendation=None
                    )
                    
                    # Generate recommendations
                    if stats.selectivity < 0.1:
                        stats.recommendation = "Low selectivity - consider removing"
                    elif stats.usage_count < self.config.index_usage_threshold:
                        stats.recommendation = "Rarely used - consider removing"
                    elif stats.size_bytes > 10_000_000:  # 10MB
                        stats.recommendation = "Large index - monitor performance impact"
                    
                    index_stats.append(stats)
                    self.index_stats[index_name] = stats
            
            # Log recommendations
            recommendations = [s for s in index_stats if s.recommendation]
            if recommendations:
                self.logger.info(
                    "index_recommendations",
                    count=len(recommendations),
                    recommendations=[
                        {
                            "index": r.index_name,
                            "recommendation": r.recommendation
                        }
                        for r in recommendations
                    ]
                )
            
            return index_stats
            
        except Exception as e:
            self.logger.error("index_analysis_failed", error=str(e))
            return []
    
    def _extract_index_columns(self, index_sql: str) -> List[str]:
        """Extract column names from index SQL."""
        if not index_sql:
            return []
        
        try:
            # Simple extraction - would need more robust parsing
            import re
            match = re.search(r'\((.*?)\)', index_sql)
            if match:
                columns = match.group(1).split(',')
                return [col.strip().split()[0] for col in columns]
        except:
            pass
        
        return []
    
    async def _get_index_size(self, db: aiosqlite.Connection, index_name: str) -> int:
        """Get approximate size of an index."""
        try:
            # This is an approximation for SQLite
            cursor = await db.execute(
                "SELECT COUNT(*) * 20 FROM sqlite_master WHERE name = ?",
                (index_name,)
            )
            result = await cursor.fetchone()
            return result[0] if result else 0
        except:
            return 0
    
    async def _calculate_selectivity(
        self,
        db: aiosqlite.Connection,
        table_name: str,
        columns: List[str]
    ) -> float:
        """Calculate index selectivity (distinct values / total rows)."""
        if not columns:
            return 0.0
        
        try:
            # Get total row count
            cursor = await db.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_rows = (await cursor.fetchone())[0]
            
            if total_rows == 0:
                return 0.0
            
            # Get distinct value count for first column
            column = columns[0]
            cursor = await db.execute(
                f"SELECT COUNT(DISTINCT {column}) FROM {table_name}"
            )
            distinct_values = (await cursor.fetchone())[0]
            
            return distinct_values / total_rows
            
        except Exception:
            return 0.0
    
    async def track_query_performance(
        self,
        query: str,
        execution_time_ms: float,
        rows_affected: int = 0
    ) -> None:
        """
        Track query performance for analysis.
        
        Args:
            query: SQL query executed
            execution_time_ms: Execution time in milliseconds
            rows_affected: Number of rows affected
        """
        # Normalize query for grouping
        import hashlib
        query_template = self._normalize_query(query)
        query_hash = hashlib.md5(query_template.encode()).hexdigest()
        
        # Update or create stats
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryPerformanceStats(
                query_hash=query_hash,
                query_template=query_template[:200],  # Truncate for storage
                execution_count=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=float('inf'),
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                last_executed=datetime.utcnow()
            )
        
        stats = self.query_stats[query_hash]
        stats.execution_count += 1
        stats.total_time_ms += execution_time_ms
        stats.avg_time_ms = stats.total_time_ms / stats.execution_count
        stats.min_time_ms = min(stats.min_time_ms, execution_time_ms)
        stats.max_time_ms = max(stats.max_time_ms, execution_time_ms)
        stats.last_executed = datetime.utcnow()
        
        # Log slow queries
        if execution_time_ms > self.config.slow_query_threshold_ms:
            self.logger.warning(
                "slow_query_detected",
                query_template=query_template[:100],
                execution_time_ms=round(execution_time_ms, 2),
                rows_affected=rows_affected
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for grouping similar queries."""
        import re
        
        # Remove whitespace
        normalized = ' '.join(query.split())
        
        # Replace values with placeholders
        normalized = re.sub(r'\b\d+\b', '?', normalized)  # Numbers
        normalized = re.sub(r"'[^']*'", '?', normalized)  # String literals
        
        return normalized.upper()
    
    async def get_performance_baseline(self) -> Dict[str, Any]:
        """
        Calculate performance baseline for all tracked queries.
        
        Returns:
            Performance baseline statistics
        """
        if not self.query_stats:
            return {}
        
        # Calculate percentiles across all queries
        all_times = []
        for stats in self.query_stats.values():
            all_times.extend([stats.avg_time_ms] * min(stats.execution_count, 100))
        
        if not all_times:
            return {}
        
        all_times.sort()
        
        def percentile(data: List[float], p: float) -> float:
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]
        
        baseline = {
            "total_queries_tracked": len(self.query_stats),
            "total_executions": sum(s.execution_count for s in self.query_stats.values()),
            "avg_execution_time_ms": sum(all_times) / len(all_times),
            "p50_execution_time_ms": percentile(all_times, 50),
            "p95_execution_time_ms": percentile(all_times, 95),
            "p99_execution_time_ms": percentile(all_times, 99),
            "slow_queries": [
                {
                    "template": stats.query_template,
                    "avg_time_ms": round(stats.avg_time_ms, 2),
                    "executions": stats.execution_count
                }
                for stats in self.query_stats.values()
                if stats.avg_time_ms > self.config.slow_query_threshold_ms
            ]
        }
        
        return baseline
    
    async def _backup_database(self) -> Path:
        """Create database backup before maintenance."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"genesis_backup_{timestamp}.db"
        
        try:
            # Use SQLite backup API for consistency
            async with aiosqlite.connect(self.db_path) as source:
                async with aiosqlite.connect(backup_path) as backup:
                    await source.backup(backup)
            
            self.logger.info(
                "database_backup_created",
                backup_path=str(backup_path),
                size=backup_path.stat().st_size
            )
            
            # Clean old backups (keep last 5)
            await self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            self.logger.error("backup_failed", error=str(e))
            raise
    
    async def _cleanup_old_backups(self, keep_count: int = 5) -> None:
        """Clean up old database backups."""
        try:
            backups = sorted(
                self.backup_dir.glob("genesis_backup_*.db"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for backup in backups[keep_count:]:
                backup.unlink()
                self.logger.info("old_backup_deleted", path=str(backup))
                
        except Exception as e:
            self.logger.error("backup_cleanup_failed", error=str(e))
    
    def is_maintenance_window(self) -> bool:
        """Check if current time is within maintenance window."""
        now = datetime.utcnow()
        start_hour = self.config.maintenance_window_start
        end_hour = (start_hour + self.config.maintenance_window_duration) % 24
        
        current_hour = now.hour
        
        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:  # Window crosses midnight
            return current_hour >= start_hour or current_hour < end_hour
    
    async def run_scheduled_maintenance(self) -> Dict[str, bool]:
        """
        Run scheduled maintenance tasks.
        
        Returns:
            Dictionary of task results
        """
        results = {
            "vacuum": False,
            "analyze": False,
            "index_optimization": False
        }
        
        # Check if in maintenance window
        if not self.is_maintenance_window():
            self.logger.info("not_in_maintenance_window")
            return results
        
        self.logger.info("scheduled_maintenance_started")
        
        try:
            # Run VACUUM if due
            if self.config.vacuum_enabled:
                results["vacuum"] = await self.perform_vacuum()
            
            # Run ANALYZE
            if self.config.analyze_enabled:
                results["analyze"] = await self.perform_analyze()
            
            # Analyze indexes
            if self.config.index_optimization_enabled:
                index_stats = await self.analyze_indexes()
                results["index_optimization"] = len(index_stats) > 0
            
            # Get and log performance baseline
            baseline = await self.get_performance_baseline()
            if baseline:
                self.logger.info(
                    "performance_baseline_updated",
                    p50_ms=baseline.get("p50_execution_time_ms"),
                    p95_ms=baseline.get("p95_execution_time_ms"),
                    slow_query_count=len(baseline.get("slow_queries", []))
                )
            
            self.logger.info(
                "scheduled_maintenance_completed",
                results=results
            )
            
        except Exception as e:
            self.logger.error("scheduled_maintenance_failed", error=str(e))
        
        return results
    
    async def prepare_for_postgres_migration(self) -> Dict[str, Any]:
        """
        Prepare database for PostgreSQL migration.
        
        Returns:
            Migration readiness report
        """
        report = {
            "database_size": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "table_count": 0,
            "index_count": 0,
            "incompatible_features": [],
            "migration_notes": []
        }
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Count tables
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                )
                report["table_count"] = (await cursor.fetchone())[0]
                
                # Count indexes
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"
                )
                report["index_count"] = (await cursor.fetchone())[0]
                
                # Check for SQLite-specific features
                cursor = await db.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table'"
                )
                table_definitions = await cursor.fetchall()
                
                for (sql,) in table_definitions:
                    if sql and 'AUTOINCREMENT' in sql.upper():
                        report["incompatible_features"].append(
                            "AUTOINCREMENT - use SERIAL in PostgreSQL"
                        )
                    if sql and 'DATETIME' in sql.upper():
                        report["migration_notes"].append(
                            "DATETIME columns - convert to TIMESTAMP in PostgreSQL"
                        )
            
            # Add general migration notes
            report["migration_notes"].extend([
                "Update connection strings in configuration",
                "Test all queries for PostgreSQL compatibility",
                "Update any SQLite-specific pragmas",
                "Consider adding connection pooling"
            ])
            
            self.logger.info(
                "postgres_migration_report",
                table_count=report["table_count"],
                index_count=report["index_count"],
                issues=len(report["incompatible_features"])
            )
            
        except Exception as e:
            self.logger.error("migration_preparation_failed", error=str(e))
            report["error"] = str(e)
        
        return report


async def setup_db_maintenance() -> DBOptimizer:
    """Setup and configure database maintenance system."""
    config = MaintenanceConfig(
        vacuum_enabled=True,
        analyze_enabled=True,
        index_optimization_enabled=True,
        maintenance_window_start=2,  # 2 AM
        maintenance_window_duration=2,  # 2 hours
        backup_before_maintenance=True,
        min_days_between_vacuum=7,
        query_log_retention_days=30,
        slow_query_threshold_ms=100.0,
        index_usage_threshold=10
    )
    
    optimizer = DBOptimizer(config)
    
    # Setup scheduled maintenance
    async def scheduled_maintenance():
        while True:
            try:
                # Run maintenance check every hour
                await asyncio.sleep(3600)
                
                # Check if maintenance should run
                if optimizer.is_maintenance_window():
                    await optimizer.run_scheduled_maintenance()
                    
            except Exception as e:
                logger = get_logger(__name__, LoggerType.SYSTEM)
                logger.error("scheduled_maintenance_error", error=str(e))
    
    # Start background task
    asyncio.create_task(scheduled_maintenance())
    
    return optimizer