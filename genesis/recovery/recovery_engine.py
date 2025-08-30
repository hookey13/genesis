"""Point-in-time recovery engine for database restoration."""

import asyncio
import json
import shutil
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from genesis.backup.backup_manager import BackupManager
from genesis.backup.s3_client import BackupMetadata
from genesis.core.exceptions import BackupError
from genesis.recovery.event_replayer import EventReplayer
from genesis.recovery.state_reconstructor import StateReconstructor

logger = structlog.get_logger(__name__)


class RecoveryEngine:
    """Manages point-in-time database recovery."""
    
    def __init__(
        self,
        backup_manager: BackupManager,
        database_path: Path,
        recovery_staging_dir: Path
    ):
        """Initialize recovery engine.
        
        Args:
            backup_manager: Backup manager instance
            database_path: Path to main database
            recovery_staging_dir: Directory for recovery staging
        """
        self.backup_manager = backup_manager
        self.database_path = database_path
        self.recovery_staging_dir = recovery_staging_dir
        
        # Create staging directory
        self.recovery_staging_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.event_replayer = EventReplayer()
        self.state_reconstructor = StateReconstructor()
        
        # Track recovery state
        self.recovery_in_progress = False
        self.last_recovery_timestamp: Optional[datetime] = None
        self.recovery_metrics: Dict[str, Any] = {}
    
    async def recover_to_timestamp(
        self,
        target_timestamp: datetime,
        validate: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Recover database to specific point in time.
        
        Args:
            target_timestamp: Target recovery timestamp
            validate: Whether to validate recovered state
            dry_run: If True, perform recovery without replacing main database
            
        Returns:
            Recovery results dictionary
        """
        if self.recovery_in_progress:
            raise BackupError("Recovery already in progress")
        
        self.recovery_in_progress = True
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting point-in-time recovery",
            target=target_timestamp.isoformat(),
            dry_run=dry_run
        )
        
        try:
            # Find required backups
            full_backup, incremental_backups = await self.backup_manager.get_backup_for_timestamp(
                target_timestamp
            )
            
            if not full_backup:
                raise BackupError(f"No backup found before {target_timestamp}")
            
            # Stage recovery
            recovered_db_path = await self._stage_recovery(
                full_backup,
                incremental_backups,
                target_timestamp
            )
            
            # Validate if requested
            if validate:
                validation_results = await self._validate_recovery(
                    recovered_db_path,
                    target_timestamp
                )
                
                if not validation_results["is_valid"]:
                    raise BackupError(f"Recovery validation failed: {validation_results['errors']}")
            
            # Apply recovery if not dry run
            if not dry_run:
                await self._apply_recovery(recovered_db_path)
            
            # Calculate metrics
            end_time = datetime.utcnow()
            recovery_time = (end_time - start_time).total_seconds()
            
            results = {
                "success": True,
                "target_timestamp": target_timestamp.isoformat(),
                "full_backup_used": full_backup.backup_id,
                "incremental_count": len(incremental_backups),
                "recovery_time_seconds": recovery_time,
                "dry_run": dry_run,
                "recovered_db_path": str(recovered_db_path) if dry_run else None
            }
            
            self.last_recovery_timestamp = target_timestamp
            self.recovery_metrics = results
            
            logger.info(
                "Recovery completed successfully",
                recovery_time=recovery_time,
                target=target_timestamp.isoformat()
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Recovery failed",
                error=str(e),
                target=target_timestamp.isoformat()
            )
            raise BackupError(f"Recovery failed: {e}")
            
        finally:
            self.recovery_in_progress = False
    
    async def _stage_recovery(
        self,
        full_backup: BackupMetadata,
        incremental_backups: List[BackupMetadata],
        target_timestamp: datetime
    ) -> Path:
        """Stage recovery in temporary location.
        
        Args:
            full_backup: Full backup metadata
            incremental_backups: List of incremental backups
            target_timestamp: Target recovery timestamp
            
        Returns:
            Path to recovered database
        """
        # Create staging file
        staging_db = self.recovery_staging_dir / f"recovery_{target_timestamp.strftime('%Y%m%d_%H%M%S')}.db"
        
        # Download and restore full backup
        logger.info("Downloading full backup", backup_id=full_backup.backup_id)
        
        full_backup_path = self.recovery_staging_dir / f"full_{full_backup.backup_id}.db"
        await self.backup_manager.s3_client.download_backup(
            key=full_backup.destination_key,
            destination_path=full_backup_path,
            verify_checksum=True
        )
        
        # Copy to staging location
        shutil.copy2(full_backup_path, staging_db)
        
        # Apply incremental backups
        for inc_backup in incremental_backups:
            logger.info(
                "Applying incremental backup",
                backup_id=inc_backup.backup_id,
                timestamp=inc_backup.timestamp.isoformat()
            )
            
            # Download WAL file
            wal_path = self.recovery_staging_dir / f"wal_{inc_backup.backup_id}.wal"
            await self.backup_manager.s3_client.download_backup(
                key=inc_backup.destination_key,
                destination_path=wal_path,
                verify_checksum=True
            )
            
            # Apply WAL to database
            await self._apply_wal(staging_db, wal_path)
            
            # Clean up WAL file
            wal_path.unlink()
        
        # Clean up full backup file
        full_backup_path.unlink()
        
        # Replay events to exact timestamp
        await self._replay_events_to_timestamp(staging_db, target_timestamp)
        
        return staging_db
    
    async def _apply_wal(self, database_path: Path, wal_path: Path) -> None:
        """Apply WAL file to database.
        
        Args:
            database_path: Database file path
            wal_path: WAL file path
        """
        loop = asyncio.get_event_loop()
        
        def apply():
            # Open database with WAL mode
            conn = sqlite3.connect(str(database_path))
            conn.execute("PRAGMA journal_mode=WAL")
            
            try:
                # Copy WAL file to database location
                db_wal_path = Path(str(database_path) + "-wal")
                shutil.copy2(wal_path, db_wal_path)
                
                # Checkpoint to apply WAL
                conn.execute("PRAGMA wal_checkpoint(RESTART)")
                conn.commit()
                
                logger.debug("WAL applied successfully", database=str(database_path))
                
                # Clean up WAL file
                if db_wal_path.exists():
                    db_wal_path.unlink()
                    
            finally:
                conn.close()
        
        await loop.run_in_executor(None, apply)
    
    async def _replay_events_to_timestamp(
        self,
        database_path: Path,
        target_timestamp: datetime
    ) -> None:
        """Replay events to reach exact timestamp.
        
        Args:
            database_path: Database file path
            target_timestamp: Target timestamp
        """
        loop = asyncio.get_event_loop()
        
        def replay():
            conn = sqlite3.connect(str(database_path))
            conn.row_factory = sqlite3.Row
            
            try:
                # Get events after last checkpoint
                cursor = conn.execute("""
                    SELECT event_id, event_type, aggregate_id, event_data, created_at
                    FROM events
                    WHERE created_at > (
                        SELECT MAX(created_at) 
                        FROM events 
                        WHERE event_type = 'CHECKPOINT'
                    )
                    AND created_at <= ?
                    ORDER BY sequence_number
                """, (target_timestamp.isoformat(),))
                
                events = cursor.fetchall()
                
                if events:
                    logger.info(f"Replaying {len(events)} events to reach target timestamp")
                    
                    for event in events:
                        # Parse and apply event
                        event_data = json.loads(event["event_data"])
                        self.event_replayer.replay_event(
                            conn,
                            event["event_type"],
                            event["aggregate_id"],
                            event_data
                        )
                
                conn.commit()
                
            finally:
                conn.close()
        
        await loop.run_in_executor(None, replay)
    
    async def _validate_recovery(
        self,
        recovered_db_path: Path,
        target_timestamp: datetime
    ) -> Dict[str, Any]:
        """Validate recovered database.
        
        Args:
            recovered_db_path: Path to recovered database
            target_timestamp: Expected recovery point
            
        Returns:
            Validation results
        """
        loop = asyncio.get_event_loop()
        
        def validate():
            conn = sqlite3.connect(str(recovered_db_path))
            conn.row_factory = sqlite3.Row
            
            errors = []
            warnings = []
            
            try:
                # Check database integrity
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                
                if integrity_result != "ok":
                    errors.append(f"Database integrity check failed: {integrity_result}")
                
                # Verify schema version
                cursor = conn.execute("PRAGMA user_version")
                version = cursor.fetchone()[0]
                
                if version == 0:
                    warnings.append("Database schema version is 0")
                
                # Check event sequence
                cursor = conn.execute("""
                    SELECT COUNT(*) as gap_count
                    FROM (
                        SELECT sequence_number,
                               LAG(sequence_number) OVER (ORDER BY sequence_number) as prev_seq
                        FROM events
                    )
                    WHERE sequence_number - prev_seq > 1
                """)
                
                gap_count = cursor.fetchone()["gap_count"]
                if gap_count > 0:
                    errors.append(f"Found {gap_count} gaps in event sequence")
                
                # Verify last event timestamp
                cursor = conn.execute("""
                    SELECT MAX(created_at) as last_event_time
                    FROM events
                """)
                
                last_event = cursor.fetchone()
                if last_event and last_event["last_event_time"]:
                    last_time = datetime.fromisoformat(last_event["last_event_time"])
                    
                    if last_time > target_timestamp:
                        errors.append(f"Events exist after target timestamp: {last_time}")
                
                # Validate state consistency
                state_validation = self.state_reconstructor.validate_state(conn)
                
                if not state_validation["is_consistent"]:
                    errors.extend(state_validation["errors"])
                
                warnings.extend(state_validation.get("warnings", []))
                
            finally:
                conn.close()
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "database_path": str(recovered_db_path)
            }
        
        return await loop.run_in_executor(None, validate)
    
    async def _apply_recovery(self, recovered_db_path: Path) -> None:
        """Apply recovered database as main database.
        
        Args:
            recovered_db_path: Path to recovered database
        """
        # Create backup of current database
        backup_path = self.database_path.parent / f"{self.database_path.stem}_pre_recovery_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db"
        
        logger.info(
            "Creating pre-recovery backup",
            current_db=str(self.database_path),
            backup_path=str(backup_path)
        )
        
        # Stop any database connections
        # Note: In production, coordinate with application shutdown
        
        # Move current database to backup
        shutil.move(self.database_path, backup_path)
        
        # Move recovered database to main location
        shutil.move(recovered_db_path, self.database_path)
        
        logger.info(
            "Recovery applied successfully",
            new_db=str(self.database_path),
            backup_available=str(backup_path)
        )
    
    async def recover_positions(
        self,
        target_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Recover trading positions from event sourcing.
        
        Args:
            target_timestamp: Optional target timestamp
            
        Returns:
            Recovered positions
        """
        loop = asyncio.get_event_loop()
        
        def recover():
            conn = sqlite3.connect(str(self.database_path))
            conn.row_factory = sqlite3.Row
            
            try:
                # Query for position events
                query = """
                    SELECT event_type, aggregate_id, event_data, created_at
                    FROM events
                    WHERE event_type IN ('POSITION_OPENED', 'POSITION_CLOSED', 'POSITION_UPDATED')
                """
                
                params = []
                if target_timestamp:
                    query += " AND created_at <= ?"
                    params.append(target_timestamp.isoformat())
                
                query += " ORDER BY sequence_number"
                
                cursor = conn.execute(query, params)
                events = cursor.fetchall()
                
                # Reconstruct positions
                positions = {}
                
                for event in events:
                    event_data = json.loads(event["event_data"])
                    position_id = event["aggregate_id"]
                    
                    if event["event_type"] == "POSITION_OPENED":
                        positions[position_id] = {
                            "id": position_id,
                            "status": "open",
                            **event_data
                        }
                    elif event["event_type"] == "POSITION_UPDATED":
                        if position_id in positions:
                            positions[position_id].update(event_data)
                    elif event["event_type"] == "POSITION_CLOSED":
                        if position_id in positions:
                            positions[position_id]["status"] = "closed"
                            positions[position_id].update(event_data)
                
                # Filter for open positions
                open_positions = {
                    pid: pos for pid, pos in positions.items()
                    if pos.get("status") == "open"
                }
                
                return {
                    "total_positions": len(positions),
                    "open_positions": len(open_positions),
                    "positions": open_positions,
                    "recovery_timestamp": target_timestamp.isoformat() if target_timestamp else "current"
                }
                
            finally:
                conn.close()
        
        return await loop.run_in_executor(None, recover)
    
    async def verify_order_reconciliation(
        self,
        exchange_orders: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Reconcile recovered orders with exchange state.
        
        Args:
            exchange_orders: Current orders from exchange
            
        Returns:
            Reconciliation results
        """
        # Get orders from recovered database
        loop = asyncio.get_event_loop()
        
        def get_db_orders():
            conn = sqlite3.connect(str(self.database_path))
            conn.row_factory = sqlite3.Row
            
            try:
                cursor = conn.execute("""
                    SELECT order_id, client_order_id, status, symbol, side, quantity, price
                    FROM orders
                    WHERE status IN ('new', 'partially_filled')
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
            finally:
                conn.close()
        
        db_orders = await loop.run_in_executor(None, get_db_orders)
        
        # Create lookup maps
        db_order_map = {o["client_order_id"]: o for o in db_orders}
        exchange_order_map = {o["clientOrderId"]: o for o in exchange_orders}
        
        # Find discrepancies
        missing_in_db = []
        missing_on_exchange = []
        status_mismatches = []
        
        for client_id, exchange_order in exchange_order_map.items():
            if client_id not in db_order_map:
                missing_in_db.append(exchange_order)
            else:
                db_order = db_order_map[client_id]
                if db_order["status"] != exchange_order["status"].lower():
                    status_mismatches.append({
                        "client_order_id": client_id,
                        "db_status": db_order["status"],
                        "exchange_status": exchange_order["status"]
                    })
        
        for client_id, db_order in db_order_map.items():
            if client_id not in exchange_order_map:
                missing_on_exchange.append(db_order)
        
        return {
            "is_reconciled": len(missing_in_db) == 0 and len(missing_on_exchange) == 0,
            "missing_in_db": missing_in_db,
            "missing_on_exchange": missing_on_exchange,
            "status_mismatches": status_mismatches,
            "db_order_count": len(db_orders),
            "exchange_order_count": len(exchange_orders)
        }
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery status.
        
        Returns:
            Status dictionary
        """
        return {
            "recovery_in_progress": self.recovery_in_progress,
            "last_recovery_timestamp": self.last_recovery_timestamp.isoformat() if self.last_recovery_timestamp else None,
            "recovery_metrics": self.recovery_metrics,
            "staging_directory": str(self.recovery_staging_dir)
        }