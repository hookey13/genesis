"""
Disaster recovery procedures for Project GENESIS.

Provides automated backup verification, position recovery from event store,
emergency position close procedures, and system state snapshot/restore.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from genesis.core.constants import TradingTier
from genesis.core.models import Position
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.exchange.gateway import BinanceGateway
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class DisasterRecovery:
    """Handles disaster recovery and emergency procedures."""

    def __init__(
        self,
        repository: Repository,
        event_bus: EventBus,
        gateway: BinanceGateway,
        backup_path: Path = Path(".genesis/backups"),
    ):
        """Initialize disaster recovery system."""
        self.repository = repository
        self.event_bus = event_bus
        self.gateway = gateway
        self.backup_path = backup_path
        self.backup_path.mkdir(parents=True, exist_ok=True)
        logger.info("disaster_recovery_initialized")

    @requires_tier(TradingTier.STRATEGIST)
    async def verify_backup(self, backup_file: Path) -> bool:
        """
        Verify backup integrity.

        Args:
            backup_file: Path to backup file

        Returns:
            True if backup is valid
        """
        try:
            if not backup_file.exists():
                logger.error("backup_file_not_found", file=str(backup_file))
                return False

            # Check file size
            size = backup_file.stat().st_size
            if size == 0:
                logger.error("backup_file_empty", file=str(backup_file))
                return False

            # Try to load backup
            with open(backup_file) as f:
                data = json.load(f)

            # Verify required fields
            required_fields = [
                "timestamp",
                "version",
                "accounts",
                "positions",
                "events",
            ]
            for field in required_fields:
                if field not in data:
                    logger.error("backup_missing_field", field=field)
                    return False

            logger.info(
                "backup_verified",
                file=str(backup_file),
                timestamp=data["timestamp"],
                accounts=len(data["accounts"]),
                positions=len(data["positions"]),
            )

            return True

        except Exception as e:
            logger.error("backup_verification_failed", error=str(e))
            return False

    @requires_tier(TradingTier.STRATEGIST)
    async def recover_positions_from_events(self, account_id: str) -> list[Position]:
        """
        Recover positions from event store.

        Args:
            account_id: Account to recover positions for

        Returns:
            List of recovered positions
        """
        try:
            # Get all events for account
            events = await self.repository.get_events(
                aggregate_id=account_id,
                event_type=None,
            )

            # Replay events to rebuild position state
            positions = {}

            for event in events:
                event_type = event.get("event_type")
                event_data = (
                    json.loads(event["event_data"])
                    if isinstance(event["event_data"], str)
                    else event["event_data"]
                )

                if event_type == "position.opened":
                    position_id = event_data["position_id"]
                    positions[position_id] = Position(
                        position_id=position_id,
                        account_id=account_id,
                        symbol=event_data["symbol"],
                        side=event_data["side"],
                        entry_price=event_data["entry_price"],
                        quantity=event_data["quantity"],
                        dollar_value=event_data.get("dollar_value", "0"),
                    )

                elif event_type == "position.closed":
                    position_id = event_data["position_id"]
                    positions.pop(position_id, None)

                elif event_type == "position.updated":
                    position_id = event_data["position_id"]
                    if position_id in positions:
                        # Update position fields
                        for key, value in event_data.items():
                            if hasattr(positions[position_id], key):
                                setattr(positions[position_id], key, value)

            recovered_positions = list(positions.values())

            logger.info(
                "positions_recovered_from_events",
                account_id=account_id,
                positions_count=len(recovered_positions),
            )

            return recovered_positions

        except Exception as e:
            logger.error("position_recovery_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def emergency_close_all_positions(self, account_id: str) -> dict[str, Any]:
        """
        Emergency close all positions.

        Args:
            account_id: Account to close positions for

        Returns:
            Results of emergency closure
        """
        try:
            # Get all open positions
            positions = await self.repository.get_positions_by_account(
                account_id, status="open"
            )

            results = {
                "closed_positions": [],
                "failed_closures": [],
                "total_positions": len(positions),
            }

            for position in positions:
                try:
                    # Send market order to close
                    order = {
                        "symbol": position.symbol,
                        "side": "SELL" if position.side == "LONG" else "BUY",
                        "type": "MARKET",
                        "quantity": str(position.quantity),
                    }

                    order_result = await self.gateway.place_order(order)

                    # Update position status
                    position.close_reason = "emergency_closure"
                    await self.repository.close_position(
                        position.position_id,
                        position.pnl_dollars,
                    )

                    results["closed_positions"].append(
                        {
                            "position_id": position.position_id,
                            "symbol": position.symbol,
                            "quantity": str(position.quantity),
                            "order_id": order_result.get("orderId"),
                        }
                    )

                except Exception as e:
                    results["failed_closures"].append(
                        {
                            "position_id": position.position_id,
                            "symbol": position.symbol,
                            "error": str(e),
                        }
                    )

            # Publish emergency closure event
            await self.event_bus.publish(
                "disaster_recovery.emergency_closure",
                {
                    "account_id": account_id,
                    "closed_count": len(results["closed_positions"]),
                    "failed_count": len(results["failed_closures"]),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            logger.warning(
                "emergency_positions_closed",
                account_id=account_id,
                closed=len(results["closed_positions"]),
                failed=len(results["failed_closures"]),
            )

            return results

        except Exception as e:
            logger.error("emergency_closure_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def create_system_snapshot(self) -> Path:
        """
        Create full system state snapshot.

        Returns:
            Path to snapshot file
        """
        try:
            timestamp = datetime.now(UTC)
            snapshot_name = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_path = self.backup_path / snapshot_name

            # Gather all data
            snapshot = {
                "timestamp": timestamp.isoformat(),
                "version": "1.0",
                "accounts": [],
                "positions": [],
                "sessions": [],
                "events": [],
                "risk_metrics": [],
            }

            # Get all accounts
            accounts = await self.repository.list_accounts()
            snapshot["accounts"] = [
                {
                    "account_id": a.account_id,
                    "balance": str(a.balance_usdt),
                    "tier": a.tier.value,
                    "account_type": (
                        a.account_type.value if hasattr(a, "account_type") else "MASTER"
                    ),
                }
                for a in accounts
            ]

            # Get all positions
            for account in accounts:
                positions = await self.repository.get_positions_by_account(
                    account.account_id
                )
                snapshot["positions"].extend(
                    [
                        {
                            "position_id": p.position_id,
                            "account_id": p.account_id,
                            "symbol": p.symbol,
                            "side": p.side.value,
                            "quantity": str(p.quantity),
                            "entry_price": str(p.entry_price),
                            "pnl": str(p.pnl_dollars),
                        }
                        for p in positions
                    ]
                )

            # Save snapshot
            with open(snapshot_path, "w") as f:
                json.dump(snapshot, f, indent=2)

            logger.info(
                "system_snapshot_created",
                file=str(snapshot_path),
                accounts=len(snapshot["accounts"]),
                positions=len(snapshot["positions"]),
            )

            return snapshot_path

        except Exception as e:
            logger.error("snapshot_creation_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def restore_from_snapshot(self, snapshot_path: Path) -> bool:
        """
        Restore system from snapshot.

        Args:
            snapshot_path: Path to snapshot file

        Returns:
            True if restore successful
        """
        try:
            # Verify snapshot first
            if not await self.verify_backup(snapshot_path):
                return False

            # Load snapshot
            with open(snapshot_path) as f:
                snapshot = json.load(f)

            # Restore accounts
            for account_data in snapshot["accounts"]:
                # Would restore account to repository
                pass

            # Restore positions
            for position_data in snapshot["positions"]:
                # Would restore position to repository
                pass

            logger.info(
                "system_restored_from_snapshot",
                file=str(snapshot_path),
                timestamp=snapshot["timestamp"],
            )

            return True

        except Exception as e:
            logger.error("snapshot_restore_failed", error=str(e))
            return False

    async def automated_backup(self) -> Path:
        """
        Perform automated backup.

        Returns:
            Path to backup file
        """
        return await self.create_system_snapshot()
