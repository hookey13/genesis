"""Tier state machine with transition celebration.

Manages tier transitions and celebrates achievements with
appropriate, muted recognition to maintain psychological balance.
"""

import functools
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from genesis.core.events import (
    EventPriority,
    GateCompletedEvent,
    TierDemotionEvent,
    TierProgressionEvent,
)
from genesis.core.exceptions import ValidationError
from genesis.data.models_db import AccountDB, Session, TierTransition, get_session
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class Tier(Enum):
    """Trading tier levels."""

    SNIPER = "SNIPER"
    HUNTER = "HUNTER"
    STRATEGIST = "STRATEGIST"
    ARCHITECT = "ARCHITECT"
    EMPEROR = "EMPEROR"


# Tier progression order
TIER_ORDER = [Tier.SNIPER, Tier.HUNTER, Tier.STRATEGIST, Tier.ARCHITECT, Tier.EMPEROR]

# Grace period for new tier adjustment (48 hours)
GRACE_PERIOD_HOURS = 48


class TransitionResult(Enum):
    """Result of tier transition attempt."""

    SUCCESS = "SUCCESS"
    BLOCKED_BY_PROTECTION = "BLOCKED_BY_PROTECTION"
    REQUIREMENTS_NOT_MET = "REQUIREMENTS_NOT_MET"
    IN_GRACE_PERIOD = "IN_GRACE_PERIOD"
    FAILED = "FAILED"


@dataclass
class AchievementReport:
    """Report summarizing tier achievement."""

    account_id: str
    previous_tier: str
    new_tier: str
    transition_date: datetime
    days_at_previous_tier: int
    total_trades: int
    win_rate: Decimal
    total_pnl: Decimal
    best_trade: Decimal
    worst_trade: Decimal
    journey_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "previous_tier": self.previous_tier,
            "new_tier": self.new_tier,
            "transition_date": self.transition_date.isoformat(),
            "days_at_previous_tier": self.days_at_previous_tier,
            "total_trades": self.total_trades,
            "win_rate": float(self.win_rate),
            "total_pnl": float(self.total_pnl),
            "best_trade": float(self.best_trade),
            "worst_trade": float(self.worst_trade),
            "journey_summary": self.journey_summary,
        }


def prevent_manual_tier_change(func: Callable) -> Callable:
    """Decorator to prevent manual tier changes outside state machine.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with protection
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if this is coming from an authorized source
        # In production, would check call stack or use authentication
        if kwargs.get("manual_override"):
            logger.error(
                "Manual tier change blocked",
                function=func.__name__,
                args=args,
                kwargs=kwargs,
            )
            raise ValidationError(
                "Manual tier changes are prohibited. "
                "All tier transitions must go through the state machine."
            )
        return await func(self, *args, **kwargs)

    return wrapper


class TierStateMachine:
    """Manages tier transitions and progression."""

    def __init__(
        self, session: Session | None = None, event_bus: EventBus | None = None
    ):
        """Initialize tier state machine.

        Args:
            session: Optional database session
            event_bus: Optional event bus for publishing events
        """
        self.session = session or get_session()
        self.event_bus = event_bus or EventBus()

    async def check_tier_requirement(self, account_id: str, required: Tier) -> bool:
        """Check if account meets tier requirement.

        Args:
            account_id: Account to check
            required: Required tier level

        Returns:
            True if account tier >= required tier
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        if not account:
            return False

        try:
            current_tier = Tier[account.current_tier]
            current_index = TIER_ORDER.index(current_tier)
            required_index = TIER_ORDER.index(required)

            return current_index >= required_index

        except (KeyError, ValueError):
            logger.error(
                "Invalid tier comparison",
                account_id=account_id,
                current=account.current_tier,
                required=required.value,
            )
            return False

    async def evaluate_progression(self, account_id: str) -> TierTransition | None:
        """Evaluate if account is ready for tier progression.

        Args:
            account_id: Account to evaluate

        Returns:
            TierTransition if progression available, None otherwise
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        if not account:
            return None

        # Check for existing transition
        existing = (
            self.session.query(TierTransition)
            .filter_by(account_id=account_id, transition_status="READY")
            .first()
        )

        return existing

    async def request_tier_change(
        self, account_id: str, new_tier: str, reason: str
    ) -> bool:
        """Request a tier change (promotion or demotion).

        Args:
            account_id: Account to change
            new_tier: Target tier
            reason: Reason for change

        Returns:
            True if change approved and executed
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        if not account:
            raise ValidationError(f"Account not found: {account_id}")

        try:
            current_tier = Tier[account.current_tier]
            target_tier = Tier[new_tier]
        except KeyError:
            raise ValidationError(f"Invalid tier: {new_tier}")

        current_index = TIER_ORDER.index(current_tier)
        target_index = TIER_ORDER.index(target_tier)

        # Determine if promotion or demotion
        is_promotion = target_index > current_index

        if is_promotion:
            # Check transition requirements are met
            transition = (
                self.session.query(TierTransition)
                .filter_by(
                    account_id=account_id,
                    from_tier=current_tier.value,
                    to_tier=target_tier.value,
                    transition_status="READY",
                )
                .first()
            )

            if not transition:
                logger.warning(
                    "Tier promotion denied - requirements not met",
                    account_id=account_id,
                    current_tier=current_tier.value,
                    target_tier=target_tier.value,
                )
                return False

            # Check all prerequisites
            if not all(
                [
                    transition.checklist_completed,
                    transition.funeral_completed,
                    transition.paper_trading_completed,
                ]
            ):
                logger.warning(
                    "Tier promotion denied - prerequisites incomplete",
                    account_id=account_id,
                    checklist=transition.checklist_completed,
                    funeral=transition.funeral_completed,
                    paper_trading=transition.paper_trading_completed,
                )
                return False

            # Execute promotion
            old_tier = account.current_tier
            account.current_tier = target_tier.value
            account.tier_started_at = datetime.utcnow()

            # Update transition
            transition.transition_status = "COMPLETED"

            self.session.commit()

            # Celebrate achievement
            await self.celebrate_tier_achievement(account_id, new_tier)

            # Publish tier progression event
            progression_event = TierProgressionEvent(
                account_id=account_id,
                from_tier=old_tier,
                to_tier=target_tier.value,
                reason=reason,
                gates_passed=(
                    transition.gates_passed
                    if hasattr(transition, "gates_passed")
                    else []
                ),
            )
            await self.event_bus.publish(progression_event, EventPriority.HIGH)

            logger.info(
                "Tier promotion completed",
                account_id=account_id,
                old_tier=old_tier,
                new_tier=target_tier.value,
                reason=reason,
            )

            return True

        else:
            # Emergency demotion
            await self.force_demotion(account_id, new_tier, reason)
            return True

    async def force_demotion(self, account_id: str, new_tier: str, reason: str) -> None:
        """Force emergency tier demotion.

        Args:
            account_id: Account to demote
            new_tier: Target tier (must be lower)
            reason: Reason for demotion
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        if not account:
            raise ValidationError(f"Account not found: {account_id}")

        old_tier = account.current_tier
        account.current_tier = new_tier
        account.tier_started_at = datetime.utcnow()

        # Create demotion record
        demotion = TierTransition(
            transition_id=str(uuid.uuid4()),
            account_id=account_id,
            from_tier=old_tier,
            to_tier=new_tier,
            transition_status="COMPLETED",
        )

        self.session.add(demotion)
        self.session.commit()

        # Publish tier demotion event
        demotion_event = TierDemotionEvent(
            account_id=account_id,
            from_tier=old_tier,
            to_tier=new_tier,
            reason=reason,
            triggers=[reason],  # Can be expanded to include multiple triggers
        )
        await self.event_bus.publish(demotion_event, EventPriority.HIGH)

        logger.warning(
            "Emergency tier demotion executed",
            account_id=account_id,
            old_tier=old_tier,
            new_tier=new_tier,
            reason=reason,
        )

    async def celebrate_tier_achievement(self, account_id: str, new_tier: str) -> None:
        """Celebrate tier achievement with muted recognition.

        Args:
            account_id: Account that achieved new tier
            new_tier: New tier achieved
        """
        try:
            # Generate achievement report
            report = await self._generate_achievement_report(account_id, new_tier)

            # Log achievement (muted, professional)
            logger.info(
                "Tier milestone achieved",
                account_id=account_id,
                new_tier=new_tier,
                days_at_previous=report.days_at_previous_tier,
                win_rate=float(report.win_rate),
                total_pnl=float(report.total_pnl),
            )

            # Create achievement message (no fanfare)
            message = self._create_achievement_message(report)

            # Store achievement
            await self._store_achievement(report, message)

            # Update UI badge (would be implemented in UI layer)
            await self._update_tier_badge(account_id, new_tier)

        except Exception as e:
            logger.error(
                "Failed to celebrate achievement",
                account_id=account_id,
                new_tier=new_tier,
                error=str(e),
            )

    async def _generate_achievement_report(
        self, account_id: str, new_tier: str
    ) -> AchievementReport:
        """Generate achievement report for tier transition.

        Args:
            account_id: Account ID
            new_tier: New tier achieved

        Returns:
            AchievementReport with journey summary
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        # Get most recent completed transition
        transition = (
            self.session.query(TierTransition)
            .filter_by(
                account_id=account_id, to_tier=new_tier, transition_status="COMPLETED"
            )
            .order_by(TierTransition.updated_at.desc())
            .first()
        )

        if not transition:
            # Fallback data
            return AchievementReport(
                account_id=account_id,
                previous_tier="UNKNOWN",
                new_tier=new_tier,
                transition_date=datetime.utcnow(),
                days_at_previous_tier=0,
                total_trades=0,
                win_rate=Decimal("0"),
                total_pnl=Decimal("0"),
                best_trade=Decimal("0"),
                worst_trade=Decimal("0"),
                journey_summary="Tier transition completed.",
            )

        # Calculate metrics from trading history
        # (Simplified - would query Trade table in real implementation)
        days_at_previous = (
            datetime.utcnow() - (account.tier_started_at or account.created_at)
        ).days

        # Generate journey summary
        journey_summary = self._generate_journey_summary(
            transition.from_tier, new_tier, days_at_previous
        )

        return AchievementReport(
            account_id=account_id,
            previous_tier=transition.from_tier,
            new_tier=new_tier,
            transition_date=datetime.utcnow(),
            days_at_previous_tier=days_at_previous,
            total_trades=100,  # Placeholder
            win_rate=Decimal("0.65"),  # Placeholder
            total_pnl=Decimal("5000"),  # Placeholder
            best_trade=Decimal("500"),  # Placeholder
            worst_trade=Decimal("-200"),  # Placeholder
            journey_summary=journey_summary,
        )

    def _create_achievement_message(self, report: AchievementReport) -> str:
        """Create muted achievement message.

        Args:
            report: Achievement report

        Returns:
            Professional, muted message
        """
        tier_messages = {
            "HUNTER": "Hunter tier unlocked. Iceberg orders now available.",
            "STRATEGIST": "Strategist tier reached. Statistical arbitrage enabled.",
            "ARCHITECT": "Architect tier achieved. Full strategy suite accessible.",
            "EMPEROR": "Emperor tier attained. Maximum capabilities unlocked.",
        }

        message = tier_messages.get(report.new_tier, f"{report.new_tier} tier reached.")

        # Add brief stats (no excessive celebration)
        message += f"\n{report.days_at_previous_tier} days at {report.previous_tier}. "
        message += f"Win rate: {float(report.win_rate):.1%}."

        return message

    def _generate_journey_summary(self, from_tier: str, to_tier: str, days: int) -> str:
        """Generate journey summary text.

        Args:
            from_tier: Previous tier
            to_tier: New tier
            days: Days at previous tier

        Returns:
            Journey summary
        """
        return (
            f"Progressed from {from_tier} to {to_tier} after {days} days of "
            f"disciplined trading and successful completion of all transition requirements."
        )

    async def _store_achievement(self, report: AchievementReport, message: str) -> None:
        """Store achievement record.

        Args:
            report: Achievement report
            message: Achievement message
        """
        # In real implementation, would store in achievements table
        logger.debug(
            "Achievement stored",
            account_id=report.account_id,
            new_tier=report.new_tier,
            message=message,
        )

    async def _update_tier_badge(self, account_id: str, new_tier: str) -> None:
        """Update UI tier badge.

        Args:
            account_id: Account ID
            new_tier: New tier
        """
        # This would trigger UI update in real implementation
        logger.debug("Tier badge updated", account_id=account_id, new_tier=new_tier)

    def get_next_tier(self, current_tier: str) -> str | None:
        """Get the next tier in progression.

        Args:
            current_tier: Current tier name

        Returns:
            Next tier name or None if at highest
        """
        try:
            tier = Tier[current_tier]
            current_index = TIER_ORDER.index(tier)

            if current_index < len(TIER_ORDER) - 1:
                return TIER_ORDER[current_index + 1].value

        except (KeyError, ValueError):
            pass

        return None

    def get_tier_requirements(self, tier: str) -> dict[str, Any]:
        """Get requirements for a tier.

        Args:
            tier: Tier name

        Returns:
            Dictionary of requirements
        """
        requirements = {
            "HUNTER": {
                "min_balance": 2000,
                "min_trades": 50,
                "max_tilt_events": 2,
                "paper_trading_required": True,
            },
            "STRATEGIST": {
                "min_balance": 10000,
                "min_trades": 200,
                "max_tilt_events": 1,
                "paper_trading_required": True,
            },
            "ARCHITECT": {
                "min_balance": 50000,
                "min_trades": 500,
                "max_tilt_events": 0,
                "paper_trading_required": True,
            },
            "EMPEROR": {
                "min_balance": 250000,
                "min_trades": 1000,
                "max_tilt_events": 0,
                "paper_trading_required": False,
            },
        }

        return requirements.get(tier, {})

    @prevent_manual_tier_change
    async def enforce_tier_transition(
        self, account: AccountDB, new_tier: Tier
    ) -> TransitionResult:
        """Enforce automatic tier transition with validation.

        Args:
            account: Account to transition
            new_tier: Target tier

        Returns:
            TransitionResult indicating outcome
        """
        try:
            current_tier = Tier[account.current_tier]
        except KeyError:
            logger.error(
                "Invalid current tier",
                account_id=account.account_id,
                tier=account.current_tier,
            )
            return TransitionResult.FAILED

        # Check if in grace period
        if await self.is_in_grace_period(account):
            logger.info(
                "Account in grace period",
                account_id=account.account_id,
                tier=account.current_tier,
            )
            return TransitionResult.IN_GRACE_PERIOD

        # Validate progression order
        current_index = TIER_ORDER.index(current_tier)
        target_index = TIER_ORDER.index(new_tier)

        if abs(target_index - current_index) != 1:
            logger.warning(
                "Invalid tier jump attempted",
                account_id=account.account_id,
                current=current_tier.value,
                target=new_tier.value,
            )
            return TransitionResult.BLOCKED_BY_PROTECTION

        # Check requirements
        requirements_met = await self.validate_requirements(account, new_tier)
        if not requirements_met:
            return TransitionResult.REQUIREMENTS_NOT_MET

        # Execute transition
        old_tier = account.current_tier
        account.current_tier = new_tier.value
        account.tier_started_at = datetime.utcnow()

        # Create transition record
        transition = TierTransition(
            transition_id=str(uuid.uuid4()),
            account_id=account.account_id,
            from_tier=old_tier,
            to_tier=new_tier.value,
            transition_status="COMPLETED",
        )

        self.session.add(transition)
        self.session.commit()

        # Publish appropriate event
        if target_index > current_index:
            # Promotion
            event = TierProgressionEvent(
                account_id=account.account_id,
                from_tier=old_tier,
                to_tier=new_tier.value,
                reason="Automatic progression - all gates passed",
                gates_passed=[],  # Would be populated from actual gate checks
            )
            await self.celebrate_tier_achievement(account.account_id, new_tier.value)
        else:
            # Demotion
            event = TierDemotionEvent(
                account_id=account.account_id,
                from_tier=old_tier,
                to_tier=new_tier.value,
                reason="Automatic demotion - requirements not maintained",
                triggers=[],  # Would be populated from actual trigger checks
            )

        await self.event_bus.publish(event, EventPriority.HIGH)

        logger.info(
            "Automatic tier transition completed",
            account_id=account.account_id,
            old_tier=old_tier,
            new_tier=new_tier.value,
        )

        return TransitionResult.SUCCESS

    async def apply_grace_period(
        self, account_id: str, hours: int = GRACE_PERIOD_HOURS
    ) -> None:
        """Apply grace period after tier transition.

        Args:
            account_id: Account to apply grace period
            hours: Grace period duration in hours
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        if not account:
            raise ValidationError(f"Account not found: {account_id}")

        # Store grace period end time in account metadata
        # In production, would store in dedicated field
        grace_end = datetime.utcnow() + timedelta(hours=hours)

        logger.info(
            "Grace period applied",
            account_id=account_id,
            tier=account.current_tier,
            grace_end=grace_end.isoformat(),
            hours=hours,
        )

    async def is_in_grace_period(self, account: AccountDB) -> bool:
        """Check if account is in grace period.

        Args:
            account: Account to check

        Returns:
            True if in grace period
        """
        if not account.tier_started_at:
            return False

        grace_end = account.tier_started_at + timedelta(hours=GRACE_PERIOD_HOURS)
        return datetime.utcnow() < grace_end

    async def validate_requirements(
        self, account: AccountDB, target_tier: Tier
    ) -> bool:
        """Validate all requirements for tier transition.

        Args:
            account: Account to validate
            target_tier: Target tier

        Returns:
            True if all requirements met
        """
        requirements = self.get_tier_requirements(target_tier.value)

        if not requirements:
            return False

        # Check minimum balance
        if account.balance < requirements.get("min_balance", 0):
            logger.debug(
                "Balance requirement not met",
                account_id=account.account_id,
                balance=float(account.balance),
                required=requirements.get("min_balance"),
            )
            return False

        # Check minimum trades
        # In production, would query trade history
        min_trades = requirements.get("min_trades", 0)
        # Placeholder check

        # Check tilt events
        # In production, would query tilt history
        max_tilt = requirements.get("max_tilt_events", float("inf"))
        # Placeholder check

        # Check paper trading completion
        if requirements.get("paper_trading_required", False):
            # Check if paper trading completed
            transition = (
                self.session.query(TierTransition)
                .filter_by(
                    account_id=account.account_id,
                    to_tier=target_tier.value,
                    paper_trading_completed=True,
                )
                .first()
            )

            if not transition:
                logger.debug(
                    "Paper trading not completed",
                    account_id=account.account_id,
                    target_tier=target_tier.value,
                )
                return False

        return True

    def get_available_features(self, tier: str) -> list[str]:
        """Get available features for a tier.

        Args:
            tier: Tier name

        Returns:
            List of available features
        """
        features_by_tier = {
            "SNIPER": ["market_orders", "basic_analytics", "single_pair_trading"],
            "HUNTER": [
                "market_orders",
                "basic_analytics",
                "single_pair_trading",
                "iceberg_orders",
                "multi_pair_trading",
                "spread_analysis",
            ],
            "STRATEGIST": [
                "market_orders",
                "basic_analytics",
                "single_pair_trading",
                "iceberg_orders",
                "multi_pair_trading",
                "spread_analysis",
                "twap_execution",
                "statistical_arbitrage",
                "advanced_analytics",
            ],
            "ARCHITECT": [
                "market_orders",
                "basic_analytics",
                "single_pair_trading",
                "iceberg_orders",
                "multi_pair_trading",
                "spread_analysis",
                "twap_execution",
                "statistical_arbitrage",
                "advanced_analytics",
                "market_making",
                "custom_strategies",
                "api_access",
            ],
            "EMPEROR": ["all_features"],
        }

        return features_by_tier.get(tier, [])

    async def handle_gate_completion(
        self, account_id: str, gate_name: str, completion_value: Any
    ) -> None:
        """Handle gate completion and check for tier eligibility.

        Args:
            account_id: Account ID
            gate_name: Name of the completed gate
            completion_value: Value that satisfied the gate
        """
        account = self.session.query(AccountDB).filter_by(account_id=account_id).first()

        if not account:
            logger.error("Account not found", account_id=account_id)
            return

        current_tier = account.current_tier
        next_tier = self.get_next_tier(current_tier)

        if not next_tier:
            return  # Already at max tier

        # Publish gate completed event
        gate_event = GateCompletedEvent(
            account_id=account_id,
            gate_name=gate_name,
            current_tier=current_tier,
            target_tier=next_tier,
            completion_value=completion_value,
        )
        await self.event_bus.publish(gate_event, EventPriority.NORMAL)

        # Check if all gates are now complete for progression
        try:
            target_tier = Tier[next_tier]
            if await self.validate_requirements(account, target_tier):
                # All requirements met, initiate automatic transition
                result = await self.enforce_tier_transition(account, target_tier)

                if result == TransitionResult.SUCCESS:
                    logger.info(
                        "Automatic tier progression triggered",
                        account_id=account_id,
                        from_tier=current_tier,
                        to_tier=next_tier,
                        trigger_gate=gate_name,
                    )
        except Exception as e:
            logger.error(
                "Error checking tier progression after gate completion",
                account_id=account_id,
                gate=gate_name,
                error=str(e),
            )
