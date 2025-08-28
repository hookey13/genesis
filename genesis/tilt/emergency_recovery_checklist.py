"""
Emergency recovery checklist for post-crisis recovery.

Provides a structured, step-by-step recovery process after
emergency events to ensure safe return to normal trading operations.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class ChecklistItemStatus(str, Enum):
    """Status of checklist items."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class RecoveryPhase(str, Enum):
    """Phases of emergency recovery."""

    IMMEDIATE = "immediate"  # First 5 minutes
    ASSESSMENT = "assessment"  # 5-30 minutes
    STABILIZATION = "stabilization"  # 30 minutes - 2 hours
    RECOVERY = "recovery"  # 2-24 hours
    NORMALIZATION = "normalization"  # 24+ hours


@dataclass
class ChecklistItem:
    """Individual checklist item."""

    item_id: str
    phase: RecoveryPhase
    title: str
    description: str
    critical: bool  # Must be completed before proceeding
    validation_required: bool  # Requires system validation
    estimated_duration_minutes: int
    status: ChecklistItemStatus = ChecklistItemStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completed_by: Optional[str] = None
    notes: Optional[str] = None
    validation_result: Optional[dict[str, Any]] = None


@dataclass
class RecoveryChecklist:
    """Complete recovery checklist."""

    checklist_id: str
    emergency_type: str
    created_at: datetime
    items: list[ChecklistItem] = field(default_factory=list)
    current_phase: RecoveryPhase = RecoveryPhase.IMMEDIATE
    completed_items: int = 0
    total_items: int = 0

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all critical items are complete."""
        critical_items = [i for i in self.items if i.critical]
        return all(
            i.status in [ChecklistItemStatus.COMPLETED, ChecklistItemStatus.SKIPPED]
            for i in critical_items
        )


class EmergencyRecoveryChecklist:
    """
    Manages recovery checklists for post-emergency procedures.

    Ensures systematic recovery with:
    - Phase-based recovery steps
    - Critical path validation
    - Progress tracking
    - Audit trail
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize recovery checklist manager.

        Args:
            event_bus: Event bus for publishing updates
        """
        self.event_bus = event_bus

        # Active checklists
        self.active_checklists: dict[str, RecoveryChecklist] = {}

        # Completed checklists
        self.completed_checklists: list[RecoveryChecklist] = []

        # Statistics
        self.checklists_created = 0
        self.checklists_completed = 0
        self.items_completed = 0
        self.items_failed = 0

        logger.info("EmergencyRecoveryChecklist initialized")

    def create_recovery_checklist(
        self, emergency_type: str, severity: str = "HIGH"
    ) -> RecoveryChecklist:
        """
        Create a recovery checklist for an emergency type.

        Args:
            emergency_type: Type of emergency
            severity: Emergency severity level

        Returns:
            Created recovery checklist
        """
        checklist_id = (
            f"recovery_{emergency_type}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        )

        checklist = RecoveryChecklist(
            checklist_id=checklist_id,
            emergency_type=emergency_type,
            created_at=datetime.now(UTC),
        )

        # Generate checklist items based on emergency type
        if emergency_type == "daily_loss_halt":
            checklist.items = self._create_daily_loss_checklist()
        elif emergency_type == "flash_crash":
            checklist.items = self._create_flash_crash_checklist()
        elif emergency_type == "correlation_spike":
            checklist.items = self._create_correlation_checklist()
        elif emergency_type == "liquidity_crisis":
            checklist.items = self._create_liquidity_crisis_checklist()
        else:
            checklist.items = self._create_generic_checklist()

        # Add severity-specific items
        if severity == "CRITICAL":
            checklist.items.extend(self._create_critical_severity_items())

        checklist.total_items = len(checklist.items)

        # Store checklist
        self.active_checklists[checklist_id] = checklist
        self.checklists_created += 1

        logger.info(
            "Recovery checklist created",
            checklist_id=checklist_id,
            emergency_type=emergency_type,
            items_count=checklist.total_items,
        )

        return checklist

    def _create_daily_loss_checklist(self) -> list[ChecklistItem]:
        """Create checklist for daily loss halt recovery."""
        return [
            # IMMEDIATE Phase (0-5 minutes)
            ChecklistItem(
                item_id="dl_001",
                phase=RecoveryPhase.IMMEDIATE,
                title="Verify all positions closed",
                description="Confirm all positions have been closed or secured",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=2,
            ),
            ChecklistItem(
                item_id="dl_002",
                phase=RecoveryPhase.IMMEDIATE,
                title="Cancel all pending orders",
                description="Ensure no pending orders remain in the system",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=1,
            ),
            ChecklistItem(
                item_id="dl_003",
                phase=RecoveryPhase.IMMEDIATE,
                title="Document final P&L",
                description="Record exact loss amount and percentage",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=2,
            ),
            # ASSESSMENT Phase (5-30 minutes)
            ChecklistItem(
                item_id="dl_004",
                phase=RecoveryPhase.ASSESSMENT,
                title="Review trade history",
                description="Analyze trades that led to the loss",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=10,
            ),
            ChecklistItem(
                item_id="dl_005",
                phase=RecoveryPhase.ASSESSMENT,
                title="Identify loss triggers",
                description="Document specific events or decisions that caused losses",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=10,
            ),
            ChecklistItem(
                item_id="dl_006",
                phase=RecoveryPhase.ASSESSMENT,
                title="Check system logs",
                description="Review system logs for anomalies or errors",
                critical=False,
                validation_required=True,
                estimated_duration_minutes=5,
            ),
            # STABILIZATION Phase (30 min - 2 hours)
            ChecklistItem(
                item_id="dl_007",
                phase=RecoveryPhase.STABILIZATION,
                title="Review risk parameters",
                description="Evaluate and adjust risk limits if necessary",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=15,
            ),
            ChecklistItem(
                item_id="dl_008",
                phase=RecoveryPhase.STABILIZATION,
                title="Validate account balance",
                description="Confirm account balance matches expected value",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=5,
            ),
            ChecklistItem(
                item_id="dl_009",
                phase=RecoveryPhase.STABILIZATION,
                title="Test system connectivity",
                description="Verify all system connections are stable",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=10,
            ),
            # RECOVERY Phase (2-24 hours)
            ChecklistItem(
                item_id="dl_010",
                phase=RecoveryPhase.RECOVERY,
                title="Complete journal entry",
                description="Write detailed journal entry about the loss event",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=30,
            ),
            ChecklistItem(
                item_id="dl_011",
                phase=RecoveryPhase.RECOVERY,
                title="Review trading strategy",
                description="Analyze if strategy modifications are needed",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=60,
            ),
            ChecklistItem(
                item_id="dl_012",
                phase=RecoveryPhase.RECOVERY,
                title="Plan next trading session",
                description="Create plan for returning to trading with reduced size",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=30,
            ),
            # NORMALIZATION Phase (24+ hours)
            ChecklistItem(
                item_id="dl_013",
                phase=RecoveryPhase.NORMALIZATION,
                title="Resume paper trading",
                description="Start with paper trading to rebuild confidence",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=120,
            ),
            ChecklistItem(
                item_id="dl_014",
                phase=RecoveryPhase.NORMALIZATION,
                title="Gradual position sizing",
                description="Return to live trading with 25% normal size",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=60,
            ),
        ]

    def _create_flash_crash_checklist(self) -> list[ChecklistItem]:
        """Create checklist for flash crash recovery."""
        return [
            # IMMEDIATE Phase
            ChecklistItem(
                item_id="fc_001",
                phase=RecoveryPhase.IMMEDIATE,
                title="Verify order cancellation",
                description="Confirm all orders cancelled for affected symbols",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=2,
            ),
            ChecklistItem(
                item_id="fc_002",
                phase=RecoveryPhase.IMMEDIATE,
                title="Check market status",
                description="Verify if trading is halted on exchange",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=1,
            ),
            ChecklistItem(
                item_id="fc_003",
                phase=RecoveryPhase.IMMEDIATE,
                title="Document price levels",
                description="Record crash start/end prices and times",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=3,
            ),
            # ASSESSMENT Phase
            ChecklistItem(
                item_id="fc_004",
                phase=RecoveryPhase.ASSESSMENT,
                title="Analyze price recovery",
                description="Monitor if prices are recovering or stabilizing",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=15,
            ),
            ChecklistItem(
                item_id="fc_005",
                phase=RecoveryPhase.ASSESSMENT,
                title="Check news sources",
                description="Look for news explaining the crash",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=10,
            ),
            # STABILIZATION Phase
            ChecklistItem(
                item_id="fc_006",
                phase=RecoveryPhase.STABILIZATION,
                title="Wait for volatility reduction",
                description="Ensure volatility has decreased before trading",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=60,
            ),
            ChecklistItem(
                item_id="fc_007",
                phase=RecoveryPhase.STABILIZATION,
                title="Update price feeds",
                description="Verify price feeds are accurate and stable",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=5,
            ),
            # RECOVERY Phase
            ChecklistItem(
                item_id="fc_008",
                phase=RecoveryPhase.RECOVERY,
                title="Review stop-loss levels",
                description="Adjust stop-loss levels for new volatility",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=20,
            ),
        ]

    def _create_correlation_checklist(self) -> list[ChecklistItem]:
        """Create checklist for correlation spike recovery."""
        return [
            # IMMEDIATE Phase
            ChecklistItem(
                item_id="cs_001",
                phase=RecoveryPhase.IMMEDIATE,
                title="Identify correlated positions",
                description="List all positions with high correlation",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=3,
            ),
            ChecklistItem(
                item_id="cs_002",
                phase=RecoveryPhase.IMMEDIATE,
                title="Reduce correlated exposure",
                description="Close or reduce highly correlated positions",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=5,
            ),
            # ASSESSMENT Phase
            ChecklistItem(
                item_id="cs_003",
                phase=RecoveryPhase.ASSESSMENT,
                title="Calculate portfolio correlation",
                description="Compute overall portfolio correlation matrix",
                critical=False,
                validation_required=True,
                estimated_duration_minutes=10,
            ),
            ChecklistItem(
                item_id="cs_004",
                phase=RecoveryPhase.ASSESSMENT,
                title="Review diversification",
                description="Assess portfolio diversification adequacy",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=15,
            ),
            # RECOVERY Phase
            ChecklistItem(
                item_id="cs_005",
                phase=RecoveryPhase.RECOVERY,
                title="Rebalance portfolio",
                description="Adjust positions to reduce correlation risk",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=30,
            ),
        ]

    def _create_liquidity_crisis_checklist(self) -> list[ChecklistItem]:
        """Create checklist for liquidity crisis recovery."""
        return [
            # IMMEDIATE Phase
            ChecklistItem(
                item_id="lc_001",
                phase=RecoveryPhase.IMMEDIATE,
                title="Switch to market orders",
                description="Use only market orders for urgent executions",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=1,
            ),
            ChecklistItem(
                item_id="lc_002",
                phase=RecoveryPhase.IMMEDIATE,
                title="Cancel limit orders",
                description="Remove all limit orders from order book",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=2,
            ),
            # ASSESSMENT Phase
            ChecklistItem(
                item_id="lc_003",
                phase=RecoveryPhase.ASSESSMENT,
                title="Monitor spread widening",
                description="Track bid-ask spread changes",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=10,
            ),
            ChecklistItem(
                item_id="lc_004",
                phase=RecoveryPhase.ASSESSMENT,
                title="Check alternative venues",
                description="Assess liquidity on other exchanges",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=15,
            ),
            # STABILIZATION Phase
            ChecklistItem(
                item_id="lc_005",
                phase=RecoveryPhase.STABILIZATION,
                title="Reduce position sizes",
                description="Lower max position size for low liquidity",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=10,
            ),
        ]

    def _create_generic_checklist(self) -> list[ChecklistItem]:
        """Create generic emergency recovery checklist."""
        return [
            ChecklistItem(
                item_id="gen_001",
                phase=RecoveryPhase.IMMEDIATE,
                title="Stop all trading",
                description="Halt all trading activities immediately",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=1,
            ),
            ChecklistItem(
                item_id="gen_002",
                phase=RecoveryPhase.IMMEDIATE,
                title="Document emergency",
                description="Record emergency type and trigger conditions",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=5,
            ),
            ChecklistItem(
                item_id="gen_003",
                phase=RecoveryPhase.ASSESSMENT,
                title="Assess damage",
                description="Evaluate financial and operational impact",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=20,
            ),
            ChecklistItem(
                item_id="gen_004",
                phase=RecoveryPhase.STABILIZATION,
                title="System health check",
                description="Verify all systems functioning normally",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=15,
            ),
            ChecklistItem(
                item_id="gen_005",
                phase=RecoveryPhase.RECOVERY,
                title="Create recovery plan",
                description="Develop plan for returning to normal operations",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=30,
            ),
        ]

    def _create_critical_severity_items(self) -> list[ChecklistItem]:
        """Additional items for critical severity emergencies."""
        return [
            ChecklistItem(
                item_id="crit_001",
                phase=RecoveryPhase.IMMEDIATE,
                title="Notify risk management",
                description="Alert risk management team or supervisor",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=2,
            ),
            ChecklistItem(
                item_id="crit_002",
                phase=RecoveryPhase.ASSESSMENT,
                title="Preserve forensic data",
                description="Save all logs and data for investigation",
                critical=True,
                validation_required=True,
                estimated_duration_minutes=10,
            ),
            ChecklistItem(
                item_id="crit_003",
                phase=RecoveryPhase.RECOVERY,
                title="Post-mortem analysis",
                description="Conduct thorough post-mortem of the event",
                critical=True,
                validation_required=False,
                estimated_duration_minutes=120,
            ),
            ChecklistItem(
                item_id="crit_004",
                phase=RecoveryPhase.NORMALIZATION,
                title="Update emergency procedures",
                description="Revise procedures based on lessons learned",
                critical=False,
                validation_required=False,
                estimated_duration_minutes=60,
            ),
        ]

    async def start_item(
        self, checklist_id: str, item_id: str, user_id: Optional[str] = None
    ) -> bool:
        """
        Mark a checklist item as started.

        Args:
            checklist_id: Checklist identifier
            item_id: Item identifier
            user_id: User starting the item

        Returns:
            True if successful
        """
        if checklist_id not in self.active_checklists:
            logger.warning("Checklist not found", checklist_id=checklist_id)
            return False

        checklist = self.active_checklists[checklist_id]

        for item in checklist.items:
            if item.item_id == item_id:
                if item.status != ChecklistItemStatus.PENDING:
                    logger.warning(
                        "Item not in pending state",
                        item_id=item_id,
                        current_status=item.status.value,
                    )
                    return False

                item.status = ChecklistItemStatus.IN_PROGRESS
                item.started_at = datetime.now(UTC)

                # Publish update event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.RECOVERY_CHECKLIST_UPDATED,
                        aggregate_id=checklist_id,
                        event_data={
                            "checklist_id": checklist_id,
                            "item_id": item_id,
                            "action": "started",
                            "phase": item.phase.value,
                            "title": item.title,
                        },
                    ),
                    priority=EventPriority.NORMAL,
                )

                logger.info(
                    "Checklist item started",
                    checklist_id=checklist_id,
                    item_id=item_id,
                    title=item.title,
                )

                return True

        logger.warning("Item not found", checklist_id=checklist_id, item_id=item_id)
        return False

    async def complete_item(
        self,
        checklist_id: str,
        item_id: str,
        notes: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a checklist item as completed.

        Args:
            checklist_id: Checklist identifier
            item_id: Item identifier
            notes: Completion notes
            user_id: User completing the item

        Returns:
            True if successful
        """
        if checklist_id not in self.active_checklists:
            logger.warning("Checklist not found", checklist_id=checklist_id)
            return False

        checklist = self.active_checklists[checklist_id]

        for item in checklist.items:
            if item.item_id == item_id:
                if item.status not in [
                    ChecklistItemStatus.PENDING,
                    ChecklistItemStatus.IN_PROGRESS,
                ]:
                    logger.warning(
                        "Item cannot be completed",
                        item_id=item_id,
                        current_status=item.status.value,
                    )
                    return False

                # Validate if required
                if item.validation_required:
                    validation_result = await self._validate_item(item)
                    if not validation_result.get("valid", False):
                        logger.warning(
                            "Item validation failed",
                            item_id=item_id,
                            validation_result=validation_result,
                        )
                        item.status = ChecklistItemStatus.FAILED
                        item.validation_result = validation_result
                        self.items_failed += 1
                        return False
                    item.validation_result = validation_result

                item.status = ChecklistItemStatus.COMPLETED
                item.completed_at = datetime.now(UTC)
                item.completed_by = user_id
                item.notes = notes

                checklist.completed_items += 1
                self.items_completed += 1

                # Check if phase complete
                phase_items = [i for i in checklist.items if i.phase == item.phase]
                phase_complete = all(
                    i.status
                    in [ChecklistItemStatus.COMPLETED, ChecklistItemStatus.SKIPPED]
                    for i in phase_items
                )

                if phase_complete:
                    # Advance to next phase
                    next_phase = self._get_next_phase(item.phase)
                    if next_phase:
                        checklist.current_phase = next_phase

                # Publish update event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.RECOVERY_CHECKLIST_UPDATED,
                        aggregate_id=checklist_id,
                        event_data={
                            "checklist_id": checklist_id,
                            "item_id": item_id,
                            "action": "completed",
                            "phase": item.phase.value,
                            "title": item.title,
                            "progress_percentage": checklist.progress_percentage,
                            "phase_complete": phase_complete,
                        },
                    ),
                    priority=EventPriority.NORMAL,
                )

                # Check if checklist complete
                if checklist.is_complete:
                    await self._complete_checklist(checklist_id)

                logger.info(
                    "Checklist item completed",
                    checklist_id=checklist_id,
                    item_id=item_id,
                    title=item.title,
                    progress=checklist.progress_percentage,
                )

                return True

        logger.warning("Item not found", checklist_id=checklist_id, item_id=item_id)
        return False

    async def _validate_item(self, item: ChecklistItem) -> dict[str, Any]:
        """
        Validate a checklist item.

        Args:
            item: Item to validate

        Returns:
            Validation result
        """
        # Implement actual validation logic here
        # This would check system state, positions, etc.

        # For now, return mock validation
        return {
            "valid": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "checks_performed": [
                "system_connectivity",
                "position_status",
                "order_status",
            ],
        }

    def _get_next_phase(self, current_phase: RecoveryPhase) -> Optional[RecoveryPhase]:
        """Get next recovery phase."""
        phases = [
            RecoveryPhase.IMMEDIATE,
            RecoveryPhase.ASSESSMENT,
            RecoveryPhase.STABILIZATION,
            RecoveryPhase.RECOVERY,
            RecoveryPhase.NORMALIZATION,
        ]

        try:
            current_index = phases.index(current_phase)
            if current_index < len(phases) - 1:
                return phases[current_index + 1]
        except ValueError:
            pass

        return None

    async def _complete_checklist(self, checklist_id: str) -> None:
        """
        Mark checklist as complete.

        Args:
            checklist_id: Checklist identifier
        """
        if checklist_id not in self.active_checklists:
            return

        checklist = self.active_checklists[checklist_id]

        # Move to completed
        self.completed_checklists.append(checklist)
        del self.active_checklists[checklist_id]
        self.checklists_completed += 1

        # Publish completion event
        await self.event_bus.publish(
            Event(
                event_type=EventType.RECOVERY_COMPLETED,
                aggregate_id=checklist_id,
                event_data={
                    "checklist_id": checklist_id,
                    "emergency_type": checklist.emergency_type,
                    "total_items": checklist.total_items,
                    "completed_items": checklist.completed_items,
                    "duration_minutes": (
                        datetime.now(UTC) - checklist.created_at
                    ).total_seconds()
                    / 60,
                },
            ),
            priority=EventPriority.HIGH,
        )

        logger.info(
            "Recovery checklist completed",
            checklist_id=checklist_id,
            emergency_type=checklist.emergency_type,
        )

    def get_checklist_status(self, checklist_id: str) -> Optional[dict[str, Any]]:
        """
        Get current status of a checklist.

        Args:
            checklist_id: Checklist identifier

        Returns:
            Status dictionary or None
        """
        checklist = self.active_checklists.get(checklist_id)

        if not checklist:
            # Check completed checklists
            for completed in self.completed_checklists:
                if completed.checklist_id == checklist_id:
                    checklist = completed
                    break

        if not checklist:
            return None

        return {
            "checklist_id": checklist_id,
            "emergency_type": checklist.emergency_type,
            "current_phase": checklist.current_phase.value,
            "progress_percentage": checklist.progress_percentage,
            "is_complete": checklist.is_complete,
            "created_at": checklist.created_at.isoformat(),
            "total_items": checklist.total_items,
            "completed_items": checklist.completed_items,
            "phases": {
                phase.value: {
                    "items": [
                        {
                            "id": item.item_id,
                            "title": item.title,
                            "status": item.status.value,
                            "critical": item.critical,
                            "estimated_minutes": item.estimated_duration_minutes,
                        }
                        for item in checklist.items
                        if item.phase == phase
                    ]
                }
                for phase in RecoveryPhase
            },
        }

    def get_active_checklists(self) -> list[dict[str, Any]]:
        """Get all active checklists."""
        return [
            self.get_checklist_status(checklist_id)
            for checklist_id in self.active_checklists
        ]

    def reset(self) -> None:
        """Reset checklist manager (useful for testing)."""
        self.active_checklists.clear()
        self.completed_checklists.clear()
        self.checklists_created = 0
        self.checklists_completed = 0
        self.items_completed = 0
        self.items_failed = 0

        logger.info("Emergency recovery checklist reset")
