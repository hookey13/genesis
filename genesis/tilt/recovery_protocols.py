"""Recovery protocol management for gradual position size restoration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import IntEnum
from typing import Any
from uuid import uuid4

import structlog

from genesis.core.events import EventType
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class RecoveryStage(IntEnum):
    """Recovery stages with position size multipliers."""

    STAGE_0 = 0  # 25% of normal position size
    STAGE_1 = 1  # 50% of normal position size
    STAGE_2 = 2  # 75% of normal position size
    STAGE_3 = 3  # 100% - fully recovered


@dataclass
class RecoveryProtocol:
    """Represents a recovery protocol session."""

    protocol_id: str = field(default_factory=lambda: str(uuid4()))
    profile_id: str = ""
    initiated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    lockout_duration_minutes: int = 0
    initial_debt_amount: Decimal = Decimal("0")
    current_debt_amount: Decimal = Decimal("0")
    recovery_stage: RecoveryStage = RecoveryStage.STAGE_0
    profitable_trades_count: int = 0
    loss_trades_count: int = 0
    total_profit: Decimal = Decimal("0")
    total_loss: Decimal = Decimal("0")
    recovery_completed_at: datetime | None = None
    is_active: bool = True
    is_drawdown_recovery: bool = False
    drawdown_percentage: Decimal = Decimal("0")
    recovery_milestones: list[Decimal] = field(default_factory=list)


class RecoveryProtocolManager:
    """Manages recovery protocols with graduated position restoration."""

    # Position size multipliers by recovery stage
    STAGE_MULTIPLIERS = {
        RecoveryStage.STAGE_0: Decimal("0.25"),  # 25% of normal
        RecoveryStage.STAGE_1: Decimal("0.50"),  # 50% of normal
        RecoveryStage.STAGE_2: Decimal("0.75"),  # 75% of normal
        RecoveryStage.STAGE_3: Decimal("1.00"),  # 100% - fully recovered
    }

    # Requirements to advance to next stage
    STAGE_ADVANCEMENT_REQUIREMENTS = {
        RecoveryStage.STAGE_0: {
            "min_profitable_trades": 3,
            "min_profit_ratio": Decimal("1.5"),  # Profit > 1.5x losses
            "min_debt_paid_ratio": Decimal("0.25"),  # 25% of debt paid
        },
        RecoveryStage.STAGE_1: {
            "min_profitable_trades": 5,
            "min_profit_ratio": Decimal("2.0"),  # Profit > 2x losses
            "min_debt_paid_ratio": Decimal("0.50"),  # 50% of debt paid
        },
        RecoveryStage.STAGE_2: {
            "min_profitable_trades": 7,
            "min_profit_ratio": Decimal("2.5"),  # Profit > 2.5x losses
            "min_debt_paid_ratio": Decimal("0.75"),  # 75% of debt paid
        },
    }

    def __init__(
        self,
        repository: SQLiteRepository | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize recovery protocol manager.

        Args:
            repository: Database repository for persistence
            event_bus: Event bus for publishing recovery events
        """
        self.repository = repository
        self.event_bus = event_bus

        # Active recovery protocols cache
        self.active_protocols: dict[str, RecoveryProtocol] = {}

    async def initiate_drawdown_recovery(
        self,
        account_id: str,
        drawdown_pct: Decimal,
    ) -> RecoveryProtocol:
        """Initiate recovery protocol for drawdown breach.

        Args:
            account_id: Account identifier
            drawdown_pct: Current drawdown percentage

        Returns:
            Created recovery protocol for drawdown
        """
        # Check if already has active protocol
        if account_id in self.active_protocols:
            existing = self.active_protocols[account_id]
            if existing.is_active and existing.is_drawdown_recovery:
                logger.warning(
                    "Drawdown recovery already active",
                    account_id=account_id,
                    protocol_id=existing.protocol_id,
                )
                return existing

        # Create new drawdown recovery protocol
        protocol = RecoveryProtocol(
            protocol_id=str(uuid4()),
            profile_id=account_id,
            initiated_at=datetime.now(UTC),
            lockout_duration_minutes=0,
            initial_debt_amount=Decimal("0"),
            current_debt_amount=Decimal("0"),
            recovery_stage=RecoveryStage.STAGE_1,  # Start at 50% for drawdown
            is_active=True,
            is_drawdown_recovery=True,
            drawdown_percentage=drawdown_pct,
            recovery_milestones=[],
        )

        # Store in cache
        self.active_protocols[account_id] = protocol

        # Persist to database
        if self.repository:
            await self._persist_protocol(protocol)

        # Publish event
        if self.event_bus:
            await self.event_bus.publish(
                EventType.DRAWDOWN_RECOVERY_INITIATED,
                {
                    "account_id": account_id,
                    "protocol_id": protocol.protocol_id,
                    "drawdown_pct": str(drawdown_pct),
                    "recovery_stage": protocol.recovery_stage.name,
                    "position_size_multiplier": str(
                        self.STAGE_MULTIPLIERS[protocol.recovery_stage]
                    ),
                    "timestamp": protocol.initiated_at.isoformat(),
                },
            )

        logger.info(
            "Drawdown recovery protocol initiated",
            account_id=account_id,
            protocol_id=protocol.protocol_id,
            drawdown_pct=float(drawdown_pct * 100),
            initial_stage=protocol.recovery_stage.name,
        )

        return protocol

    def update_recovery_progress(
        self,
        protocol_id: str,
        trade_result: dict,
    ) -> RecoveryStage:
        """Update recovery progress based on trade results.

        Args:
            protocol_id: Protocol identifier
            trade_result: Trade result dictionary

        Returns:
            Updated recovery stage
        """
        protocol = None
        for p in self.active_protocols.values():
            if p.protocol_id == protocol_id:
                protocol = p
                break

        if not protocol:
            logger.warning("Protocol not found", protocol_id=protocol_id)
            return RecoveryStage.STAGE_0

        # Update trade statistics
        is_profitable = trade_result.get("profit_loss", Decimal("0")) > 0
        if is_profitable:
            protocol.profitable_trades_count += 1
            protocol.total_profit += abs(trade_result["profit_loss"])
        else:
            protocol.loss_trades_count += 1
            protocol.total_loss += abs(trade_result["profit_loss"])

        # Check for milestone achievement (25%, 50%, 75% recovery)
        if protocol.is_drawdown_recovery:
            current_recovery = (
                protocol.total_profit - protocol.total_loss
            ) / protocol.initial_debt_amount

            if (
                current_recovery >= Decimal("0.25")
                and Decimal("0.25") not in protocol.recovery_milestones
            ):
                protocol.recovery_milestones.append(Decimal("0.25"))
                logger.info("Recovery milestone reached: 25%", protocol_id=protocol_id)

            if (
                current_recovery >= Decimal("0.50")
                and Decimal("0.50") not in protocol.recovery_milestones
            ):
                protocol.recovery_milestones.append(Decimal("0.50"))
                protocol.recovery_stage = RecoveryStage.STAGE_2
                logger.info("Recovery milestone reached: 50%", protocol_id=protocol_id)

            if (
                current_recovery >= Decimal("0.75")
                and Decimal("0.75") not in protocol.recovery_milestones
            ):
                protocol.recovery_milestones.append(Decimal("0.75"))
                protocol.recovery_stage = RecoveryStage.STAGE_3
                logger.info("Recovery milestone reached: 75%", protocol_id=protocol_id)

            if current_recovery >= Decimal("1.00"):
                protocol.recovery_stage = RecoveryStage.STAGE_3
                protocol.is_active = False
                protocol.recovery_completed_at = datetime.now(UTC)
                logger.info("Drawdown recovery completed", protocol_id=protocol_id)

        return protocol.recovery_stage

    async def initiate_recovery_protocol(
        self,
        profile_id: str,
        lockout_duration_minutes: int,
        initial_debt_amount: Decimal,
    ) -> RecoveryProtocol:
        """Initiate a new recovery protocol for a profile.

        Args:
            profile_id: Profile identifier
            lockout_duration_minutes: Duration of initial lockout
            initial_debt_amount: Starting tilt debt amount

        Returns:
            Created recovery protocol
        """
        # Check if already has active protocol
        if profile_id in self.active_protocols:
            existing = self.active_protocols[profile_id]
            if existing.is_active:
                logger.warning(
                    "Recovery protocol already active",
                    profile_id=profile_id,
                    protocol_id=existing.protocol_id,
                )
                return existing

        # Create new protocol
        protocol = RecoveryProtocol(
            protocol_id=str(uuid4()),
            profile_id=profile_id,
            initiated_at=datetime.now(UTC),
            lockout_duration_minutes=lockout_duration_minutes,
            initial_debt_amount=initial_debt_amount,
            current_debt_amount=initial_debt_amount,
            recovery_stage=RecoveryStage.STAGE_0,
            is_active=True,
        )

        # Store in cache
        self.active_protocols[profile_id] = protocol

        # Persist to database
        if self.repository:
            await self._persist_protocol(protocol)

        # Publish event
        if self.event_bus:
            await self._publish_protocol_initiated_event(protocol)

        logger.info(
            "Recovery protocol initiated",
            profile_id=profile_id,
            protocol_id=protocol.protocol_id,
            initial_debt=str(initial_debt_amount),
            initial_stage=protocol.recovery_stage.name,
        )

        return protocol

    def calculate_recovery_position_size(
        self,
        base_size: Decimal,
        recovery_stage: int,
    ) -> Decimal:
        """Calculate position size based on recovery stage.

        Args:
            base_size: Normal position size
            recovery_stage: Current recovery stage (0-3)

        Returns:
            Adjusted position size
        """
        # Ensure recovery stage is valid
        try:
            stage = RecoveryStage(recovery_stage)
        except ValueError:
            logger.warning(
                "Invalid recovery stage, using Stage 0",
                recovery_stage=recovery_stage,
            )
            stage = RecoveryStage.STAGE_0

        multiplier = self.STAGE_MULTIPLIERS.get(stage, Decimal("0.25"))
        adjusted_size = base_size * multiplier

        logger.debug(
            "Position size calculated for recovery",
            base_size=str(base_size),
            recovery_stage=stage.name,
            multiplier=str(multiplier),
            adjusted_size=str(adjusted_size),
        )

        return adjusted_size

    def get_position_size_multiplier(self, profile_id: str) -> Decimal:
        """Get current position size multiplier for profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Position size multiplier (1.0 if no recovery protocol)
        """
        protocol = self.get_active_protocol(profile_id)

        if not protocol:
            return Decimal("1.0")

        return self.STAGE_MULTIPLIERS.get(
            protocol.recovery_stage,
            Decimal("0.25"),
        )

    async def record_trade_result(
        self,
        profile_id: str,
        profit_loss: Decimal,
        is_profitable: bool,
    ) -> None:
        """Record a trade result during recovery.

        Args:
            profile_id: Profile identifier
            profit_loss: Profit or loss amount (positive for profit)
            is_profitable: Whether trade was profitable
        """
        protocol = self.get_active_protocol(profile_id)

        if not protocol:
            return

        # Update trade statistics
        if is_profitable:
            protocol.profitable_trades_count += 1
            protocol.total_profit += abs(profit_loss)
        else:
            protocol.loss_trades_count += 1
            protocol.total_loss += abs(profit_loss)

        # Update debt if profitable
        if is_profitable and protocol.current_debt_amount > Decimal("0"):
            debt_reduction = min(abs(profit_loss), protocol.current_debt_amount)
            protocol.current_debt_amount -= debt_reduction

            if self.event_bus:
                await self.event_bus.publish(
                    EventType.TILT_DEBT_REDUCED,
                    {
                        "profile_id": profile_id,
                        "amount_reduced": str(debt_reduction),
                        "remaining_debt": str(protocol.current_debt_amount),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

        # Persist changes
        if self.repository:
            await self._update_protocol(protocol)

        logger.info(
            "Trade result recorded for recovery",
            profile_id=profile_id,
            is_profitable=is_profitable,
            profit_loss=str(profit_loss),
            current_debt=str(protocol.current_debt_amount),
            profitable_trades=protocol.profitable_trades_count,
        )

    async def advance_recovery_stage(self, profile_id: str) -> bool:
        """Try to advance to the next recovery stage.

        Args:
            profile_id: Profile identifier

        Returns:
            True if stage was advanced
        """
        protocol = self.get_active_protocol(profile_id)

        if not protocol:
            return False

        # Check if already fully recovered
        if protocol.recovery_stage == RecoveryStage.STAGE_3:
            logger.info(
                "Already at maximum recovery stage",
                profile_id=profile_id,
            )
            return False

        # Check advancement requirements
        if self._check_advancement_requirements(protocol):
            old_stage = protocol.recovery_stage
            protocol.recovery_stage = RecoveryStage(protocol.recovery_stage + 1)

            # Check if fully recovered
            if protocol.recovery_stage == RecoveryStage.STAGE_3:
                await self._complete_recovery(protocol)

            # Persist changes
            if self.repository:
                await self._update_protocol(protocol)

            # Publish event
            if self.event_bus:
                await self._publish_stage_advanced_event(protocol, old_stage)

            logger.info(
                "Recovery stage advanced",
                profile_id=profile_id,
                old_stage=old_stage.name,
                new_stage=protocol.recovery_stage.name,
            )

            return True

        return False

    def _check_advancement_requirements(self, protocol: RecoveryProtocol) -> bool:
        """Check if protocol meets requirements for stage advancement.

        Args:
            protocol: Recovery protocol to check

        Returns:
            True if requirements are met
        """
        requirements = self.STAGE_ADVANCEMENT_REQUIREMENTS.get(protocol.recovery_stage)

        if not requirements:
            return False

        # Check profitable trades count
        if protocol.profitable_trades_count < requirements["min_profitable_trades"]:
            return False

        # Check profit ratio
        if protocol.total_loss > Decimal("0"):
            profit_ratio = protocol.total_profit / protocol.total_loss
            if profit_ratio < requirements["min_profit_ratio"]:
                return False

        # Check debt payment ratio
        if protocol.initial_debt_amount > Decimal("0"):
            debt_paid = protocol.initial_debt_amount - protocol.current_debt_amount
            debt_paid_ratio = debt_paid / protocol.initial_debt_amount
            if debt_paid_ratio < requirements["min_debt_paid_ratio"]:
                return False

        return True

    async def _complete_recovery(self, protocol: RecoveryProtocol) -> None:
        """Mark recovery protocol as complete.

        Args:
            protocol: Protocol to complete
        """
        protocol.recovery_completed_at = datetime.now(UTC)
        protocol.is_active = False

        # Remove from active cache
        if protocol.profile_id in self.active_protocols:
            del self.active_protocols[protocol.profile_id]

        # Publish completion event
        if self.event_bus:
            await self.event_bus.publish(
                EventType.RECOVERY_COMPLETED,
                {
                    "profile_id": protocol.profile_id,
                    "protocol_id": protocol.protocol_id,
                    "total_duration_minutes": int(
                        (
                            protocol.recovery_completed_at - protocol.initiated_at
                        ).total_seconds()
                        / 60
                    ),
                    "final_debt": str(protocol.current_debt_amount),
                    "total_profit": str(protocol.total_profit),
                    "total_loss": str(protocol.total_loss),
                    "timestamp": protocol.recovery_completed_at.isoformat(),
                },
            )

        logger.info(
            "Recovery protocol completed",
            profile_id=protocol.profile_id,
            protocol_id=protocol.protocol_id,
            duration_hours=round(
                (protocol.recovery_completed_at - protocol.initiated_at).total_seconds()
                / 3600,
                2,
            ),
        )

    def get_active_protocol(self, profile_id: str) -> RecoveryProtocol | None:
        """Get active recovery protocol for profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Active protocol or None
        """
        protocol = self.active_protocols.get(profile_id)

        if protocol and protocol.is_active:
            return protocol

        return None

    def get_recovery_statistics(self, profile_id: str) -> dict[str, Any]:
        """Get recovery statistics for a profile.

        Args:
            profile_id: Profile identifier

        Returns:
            Dictionary of recovery statistics
        """
        protocol = self.get_active_protocol(profile_id)

        if not protocol:
            return {
                "has_active_protocol": False,
                "recovery_stage": None,
                "position_size_multiplier": "1.0",
            }

        return {
            "has_active_protocol": True,
            "protocol_id": protocol.protocol_id,
            "recovery_stage": protocol.recovery_stage.name,
            "recovery_stage_number": protocol.recovery_stage,
            "position_size_multiplier": str(
                self.STAGE_MULTIPLIERS[protocol.recovery_stage]
            ),
            "initiated_at": protocol.initiated_at.isoformat(),
            "initial_debt": str(protocol.initial_debt_amount),
            "current_debt": str(protocol.current_debt_amount),
            "debt_paid": str(
                protocol.initial_debt_amount - protocol.current_debt_amount
            ),
            "profitable_trades": protocol.profitable_trades_count,
            "loss_trades": protocol.loss_trades_count,
            "total_profit": str(protocol.total_profit),
            "total_loss": str(protocol.total_loss),
            "can_advance": self._check_advancement_requirements(protocol),
        }

    async def force_complete_recovery(self, profile_id: str) -> bool:
        """Force complete a recovery protocol (emergency override).

        Args:
            profile_id: Profile identifier

        Returns:
            True if protocol was completed
        """
        protocol = self.get_active_protocol(profile_id)

        if not protocol:
            return False

        await self._complete_recovery(protocol)

        # Persist changes
        if self.repository:
            await self._update_protocol(protocol)

        logger.warning(
            "Recovery protocol force completed",
            profile_id=profile_id,
            protocol_id=protocol.protocol_id,
        )

        return True

    async def load_active_protocols(self) -> None:
        """Load active recovery protocols from database on startup."""
        if not self.repository:
            return

        try:
            protocols_data = await self.repository.get_active_recovery_protocols()

            for data in protocols_data:
                protocol = self._protocol_from_dict(data)
                if protocol.is_active:
                    self.active_protocols[protocol.profile_id] = protocol

            logger.info(
                "Active recovery protocols loaded",
                count=len(self.active_protocols),
            )
        except Exception as e:
            logger.error(
                "Failed to load active recovery protocols",
                error=str(e),
            )

    async def _persist_protocol(self, protocol: RecoveryProtocol) -> None:
        """Persist recovery protocol to database.

        Args:
            protocol: Protocol to persist
        """
        if not self.repository:
            return

        try:
            await self.repository.save_recovery_protocol(
                self._protocol_to_dict(protocol)
            )
        except Exception as e:
            logger.error(
                "Failed to persist recovery protocol",
                protocol_id=protocol.protocol_id,
                error=str(e),
            )

    async def _update_protocol(self, protocol: RecoveryProtocol) -> None:
        """Update recovery protocol in database.

        Args:
            protocol: Protocol to update
        """
        if not self.repository:
            return

        try:
            await self.repository.update_recovery_protocol(
                protocol.protocol_id,
                self._protocol_to_dict(protocol),
            )
        except Exception as e:
            logger.error(
                "Failed to update recovery protocol",
                protocol_id=protocol.protocol_id,
                error=str(e),
            )

    async def _publish_protocol_initiated_event(
        self,
        protocol: RecoveryProtocol,
    ) -> None:
        """Publish recovery protocol initiated event.

        Args:
            protocol: Initiated protocol
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.RECOVERY_PROTOCOL_INITIATED,
            {
                "profile_id": protocol.profile_id,
                "protocol_id": protocol.protocol_id,
                "lockout_duration_minutes": protocol.lockout_duration_minutes,
                "initial_debt": str(protocol.initial_debt_amount),
                "recovery_stage": protocol.recovery_stage.name,
                "position_size_multiplier": str(
                    self.STAGE_MULTIPLIERS[protocol.recovery_stage]
                ),
                "timestamp": protocol.initiated_at.isoformat(),
            },
        )

    async def _publish_stage_advanced_event(
        self,
        protocol: RecoveryProtocol,
        old_stage: RecoveryStage,
    ) -> None:
        """Publish recovery stage advanced event.

        Args:
            protocol: Protocol with advanced stage
            old_stage: Previous stage
        """
        if not self.event_bus:
            return

        await self.event_bus.publish(
            EventType.RECOVERY_STAGE_ADVANCED,
            {
                "profile_id": protocol.profile_id,
                "protocol_id": protocol.protocol_id,
                "old_stage": old_stage.name,
                "new_stage": protocol.recovery_stage.name,
                "new_position_size_multiplier": str(
                    self.STAGE_MULTIPLIERS[protocol.recovery_stage]
                ),
                "current_debt": str(protocol.current_debt_amount),
                "profitable_trades": protocol.profitable_trades_count,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def _protocol_to_dict(self, protocol: RecoveryProtocol) -> dict[str, Any]:
        """Convert protocol to dictionary.

        Args:
            protocol: Protocol to convert

        Returns:
            Protocol data dictionary
        """
        return {
            "protocol_id": protocol.protocol_id,
            "profile_id": protocol.profile_id,
            "initiated_at": protocol.initiated_at.isoformat(),
            "lockout_duration_minutes": protocol.lockout_duration_minutes,
            "initial_debt_amount": str(protocol.initial_debt_amount),
            "current_debt_amount": str(protocol.current_debt_amount),
            "recovery_stage": protocol.recovery_stage,
            "profitable_trades_count": protocol.profitable_trades_count,
            "loss_trades_count": protocol.loss_trades_count,
            "total_profit": str(protocol.total_profit),
            "total_loss": str(protocol.total_loss),
            "recovery_completed_at": (
                protocol.recovery_completed_at.isoformat()
                if protocol.recovery_completed_at
                else None
            ),
            "is_active": protocol.is_active,
            "is_drawdown_recovery": protocol.is_drawdown_recovery,
            "drawdown_percentage": str(protocol.drawdown_percentage),
            "recovery_milestones": [str(m) for m in protocol.recovery_milestones],
        }

    def _protocol_from_dict(self, data: dict[str, Any]) -> RecoveryProtocol:
        """Create protocol from dictionary.

        Args:
            data: Protocol data

        Returns:
            Recovery protocol object
        """
        return RecoveryProtocol(
            protocol_id=data["protocol_id"],
            profile_id=data["profile_id"],
            initiated_at=datetime.fromisoformat(data["initiated_at"]),
            lockout_duration_minutes=data["lockout_duration_minutes"],
            initial_debt_amount=Decimal(data["initial_debt_amount"]),
            current_debt_amount=Decimal(data["current_debt_amount"]),
            recovery_stage=RecoveryStage(data["recovery_stage"]),
            profitable_trades_count=data.get("profitable_trades_count", 0),
            loss_trades_count=data.get("loss_trades_count", 0),
            total_profit=Decimal(data.get("total_profit", "0")),
            total_loss=Decimal(data.get("total_loss", "0")),
            recovery_completed_at=(
                datetime.fromisoformat(data["recovery_completed_at"])
                if data.get("recovery_completed_at")
                else None
            ),
            is_active=data.get("is_active", True),
            is_drawdown_recovery=data.get("is_drawdown_recovery", False),
            drawdown_percentage=Decimal(data.get("drawdown_percentage", "0")),
            recovery_milestones=[
                Decimal(m) for m in data.get("recovery_milestones", [])
            ],
        )
