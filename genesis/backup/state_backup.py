"""Trading state and position backup management."""

import asyncio
import json
import pickle
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from genesis.backup.s3_client import BackupMetadata, S3Client
from genesis.core.exceptions import BackupError
from genesis.utils.decorators import with_retry

logger = structlog.get_logger(__name__)


class TradingStateSnapshot(BaseModel):
    """Snapshot of trading system state."""

    timestamp: datetime
    tier: str
    positions: list[dict[str, Any]]
    open_orders: list[dict[str, Any]]
    balances: dict[str, Decimal]
    risk_metrics: dict[str, Any]
    tilt_status: dict[str, Any]
    session_id: str
    pnl_total: Decimal
    pnl_today: Decimal

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str
        }


class StateBackupManager:
    """Manages trading state and position backups."""

    def __init__(
        self,
        s3_client: S3Client | None = None,
        backup_dir: Path | None = None,
        enable_continuous_backup: bool = True
    ):
        """Initialize state backup manager.
        
        Args:
            s3_client: S3 client for remote storage
            backup_dir: Local backup directory
            enable_continuous_backup: Enable real-time state backup
        """
        self.s3_client = s3_client or S3Client()
        self.backup_dir = backup_dir or Path("/tmp/state_backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.enable_continuous_backup = enable_continuous_backup

        # State tracking
        self.last_snapshot: TradingStateSnapshot | None = None
        self.snapshot_history: list[TradingStateSnapshot] = []
        self.recovery_point: datetime | None = None

    @with_retry(max_attempts=3, backoff_factor=2)
    async def create_state_snapshot(
        self,
        trading_state: dict[str, Any]
    ) -> BackupMetadata:
        """Create atomic snapshot of trading state.
        
        Args:
            trading_state: Current trading system state
            
        Returns:
            Backup metadata
        """
        timestamp = datetime.utcnow()
        snapshot_id = f"state_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.info("Creating trading state snapshot", snapshot_id=snapshot_id)

        try:
            # Create snapshot model
            snapshot = TradingStateSnapshot(
                timestamp=timestamp,
                tier=trading_state.get('tier', 'sniper'),
                positions=await self._serialize_positions(
                    trading_state.get('positions', [])
                ),
                open_orders=await self._serialize_orders(
                    trading_state.get('open_orders', [])
                ),
                balances=trading_state.get('balances', {}),
                risk_metrics=trading_state.get('risk_metrics', {}),
                tilt_status=trading_state.get('tilt_status', {}),
                session_id=trading_state.get('session_id', ''),
                pnl_total=Decimal(str(trading_state.get('pnl_total', 0))),
                pnl_today=Decimal(str(trading_state.get('pnl_today', 0)))
            )

            # Save snapshot to file
            snapshot_path = self.backup_dir / f"{snapshot_id}.json"
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot.dict(), f, indent=2, default=str)

            # Also create binary pickle for faster recovery
            pickle_path = self.backup_dir / f"{snapshot_id}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Calculate metadata
            file_stats = snapshot_path.stat()

            # Create metadata
            metadata = BackupMetadata(
                backup_id=snapshot_id,
                timestamp=timestamp,
                size_bytes=file_stats.st_size,
                checksum=await self._calculate_checksum(snapshot_path),
                database_version=trading_state.get('tier', 'sniper'),
                backup_type="trading_state",
                retention_policy="hourly",
                source_path="trading_engine",
                destination_key=""
            )

            # Upload both formats to S3
            if self.s3_client:
                # Upload JSON version
                json_key, _ = await self.s3_client.upload_backup_with_replication(
                    file_path=snapshot_path,
                    key_prefix="state/json/",
                    metadata=metadata
                )

                # Upload pickle version
                pickle_key, _ = await self.s3_client.upload_backup_with_replication(
                    file_path=pickle_path,
                    key_prefix="state/pickle/",
                    metadata=metadata
                )

                metadata.destination_key = json_key

            # Update tracking
            self.last_snapshot = snapshot
            self.snapshot_history.append(snapshot)

            # Clean up local files after successful upload
            snapshot_path.unlink()
            pickle_path.unlink()

            logger.info(
                "Trading state snapshot created",
                snapshot_id=snapshot_id,
                positions=len(snapshot.positions),
                orders=len(snapshot.open_orders),
                pnl_total=str(snapshot.pnl_total)
            )

            return metadata

        except Exception as e:
            logger.error("State snapshot failed", error=str(e))
            raise BackupError(f"State snapshot failed: {e}")

    async def _serialize_positions(
        self,
        positions: list[Any]
    ) -> list[dict[str, Any]]:
        """Serialize position objects for backup.
        
        Args:
            positions: List of position objects
            
        Returns:
            List of serialized positions
        """
        serialized = []

        for pos in positions:
            if hasattr(pos, 'dict'):
                # Pydantic model
                serialized.append(pos.dict())
            elif hasattr(pos, '__dict__'):
                # Regular object
                serialized.append(pos.__dict__)
            else:
                # Already a dict
                serialized.append(pos)

        return serialized

    async def _serialize_orders(
        self,
        orders: list[Any]
    ) -> list[dict[str, Any]]:
        """Serialize order objects for backup.
        
        Args:
            orders: List of order objects
            
        Returns:
            List of serialized orders
        """
        serialized = []

        for order in orders:
            if hasattr(order, 'dict'):
                # Pydantic model
                serialized.append(order.dict())
            elif hasattr(order, '__dict__'):
                # Regular object
                serialized.append(order.__dict__)
            else:
                # Already a dict
                serialized.append(order)

        return serialized

    async def recover_state(
        self,
        snapshot_id: str,
        use_pickle: bool = True
    ) -> TradingStateSnapshot:
        """Recover trading state from a snapshot.
        
        Args:
            snapshot_id: Snapshot ID to recover
            use_pickle: Use pickle format for faster recovery
            
        Returns:
            Recovered trading state snapshot
        """
        logger.info(f"Recovering trading state from {snapshot_id}")

        try:
            # Download snapshot from S3
            if use_pickle:
                key_prefix = "state/pickle/"
                file_ext = ".pkl"
            else:
                key_prefix = "state/json/"
                file_ext = ".json"

            local_path = self.backup_dir / f"recover_{snapshot_id}{file_ext}"

            await self.s3_client.download_backup(
                key=f"{key_prefix}{snapshot_id}{file_ext}",
                destination_path=local_path,
                verify_checksum=True
            )

            # Load snapshot
            if use_pickle:
                with open(local_path, 'rb') as f:
                    snapshot = pickle.load(f)
            else:
                with open(local_path) as f:
                    data = json.load(f)
                    snapshot = TradingStateSnapshot(**data)

            # Clean up
            local_path.unlink()

            # Set recovery point
            self.recovery_point = snapshot.timestamp

            logger.info(
                "Trading state recovered",
                snapshot_id=snapshot_id,
                timestamp=snapshot.timestamp.isoformat(),
                positions=len(snapshot.positions),
                pnl_total=str(snapshot.pnl_total)
            )

            return snapshot

        except Exception as e:
            logger.error("State recovery failed", error=str(e))
            raise BackupError(f"State recovery failed: {e}")

    async def reconcile_positions(
        self,
        snapshot: TradingStateSnapshot,
        current_positions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reconcile recovered positions with current state.
        
        Args:
            snapshot: Recovered snapshot
            current_positions: Current position list
            
        Returns:
            Reconciliation report
        """
        logger.info("Reconciling positions")

        # Create position maps
        snapshot_map = {
            pos.get('symbol'): pos for pos in snapshot.positions
        }
        current_map = {
            pos.get('symbol'): pos for pos in current_positions
        }

        # Find differences
        missing_in_current = set(snapshot_map.keys()) - set(current_map.keys())
        extra_in_current = set(current_map.keys()) - set(snapshot_map.keys())

        # Check for size mismatches
        size_mismatches = []
        for symbol in set(snapshot_map.keys()) & set(current_map.keys()):
            snapshot_size = Decimal(str(snapshot_map[symbol].get('size', 0)))
            current_size = Decimal(str(current_map[symbol].get('size', 0)))

            if abs(snapshot_size - current_size) > Decimal('0.00001'):
                size_mismatches.append({
                    'symbol': symbol,
                    'snapshot_size': str(snapshot_size),
                    'current_size': str(current_size),
                    'difference': str(current_size - snapshot_size)
                })

        reconciliation = {
            'timestamp': datetime.utcnow().isoformat(),
            'snapshot_timestamp': snapshot.timestamp.isoformat(),
            'positions_in_snapshot': len(snapshot.positions),
            'positions_current': len(current_positions),
            'missing_in_current': list(missing_in_current),
            'extra_in_current': list(extra_in_current),
            'size_mismatches': size_mismatches,
            'needs_adjustment': bool(missing_in_current or size_mismatches)
        }

        logger.info(
            "Position reconciliation complete",
            missing=len(missing_in_current),
            extra=len(extra_in_current),
            mismatches=len(size_mismatches)
        )

        return reconciliation

    async def create_continuous_backup(
        self,
        trading_state: dict[str, Any],
        interval_seconds: int = 60
    ):
        """Start continuous state backup.
        
        Args:
            trading_state: Trading state to backup
            interval_seconds: Backup interval
        """
        if not self.enable_continuous_backup:
            return

        logger.info(f"Starting continuous backup every {interval_seconds}s")

        while self.enable_continuous_backup:
            try:
                await self.create_state_snapshot(trading_state)
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Continuous backup error: {e}")
                await asyncio.sleep(interval_seconds)

    async def get_state_at_timestamp(
        self,
        target_time: datetime
    ) -> TradingStateSnapshot | None:
        """Get the closest state snapshot to a target timestamp.
        
        Args:
            target_time: Target timestamp
            
        Returns:
            Closest snapshot or None
        """
        # List available snapshots
        backups = await self.s3_client.list_backups(prefix="state/json/")

        # Filter to trading state backups
        state_backups = [
            b for b in backups
            if b.backup_type == "trading_state"
        ]

        if not state_backups:
            return None

        # Find closest backup before target time
        valid_backups = [
            b for b in state_backups
            if b.timestamp <= target_time
        ]

        if not valid_backups:
            return None

        # Sort by timestamp and get most recent
        valid_backups.sort(key=lambda x: x.timestamp, reverse=True)
        closest = valid_backups[0]

        # Recover the snapshot
        snapshot_id = closest.backup_id
        return await self.recover_state(snapshot_id)

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of checksum
        """
        import hashlib
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def stop_continuous_backup(self):
        """Stop continuous backup."""
        self.enable_continuous_backup = False
        logger.info("Continuous backup stopped")
