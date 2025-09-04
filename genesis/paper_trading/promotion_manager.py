"""Strategy promotion manager for paper trading to live trading transition."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from genesis.paper_trading.validation_criteria import ValidationCriteria

logger = structlog.get_logger(__name__)


class PromotionStatus(Enum):
    """Strategy promotion status."""

    PAPER = "paper"
    PENDING_PROMOTION = "pending_promotion"
    GRADUAL_ALLOCATION = "gradual_allocation"
    FULL_ALLOCATION = "full_allocation"
    DEMOTED = "demoted"
    FAILED = "failed"


class ABTestVariant(Enum):
    """A/B test variant types."""

    CONTROL = "control"
    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    VARIANT_C = "variant_c"


class AllocationStrategy(Enum):
    """Capital allocation strategies."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    CUSTOM = "custom"


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""

    control_metrics: dict[str, Any]
    variant_metrics: dict[str, Any]
    winner: str
    confidence: float
    improvement: float
    test_duration_days: int


@dataclass
class PromotionConfig:
    """Configuration for strategy promotion."""

    auto_promote: bool = True
    initial_allocation: Decimal = Decimal("0.1")
    allocation_increment: Decimal = Decimal("0.1")
    max_allocation: Decimal = Decimal("1.0")
    regression_threshold: Decimal = Decimal("0.2")
    demotion_threshold: Decimal = Decimal("0.3")
    ab_testing_enabled: bool = True
    max_ab_variants: int = 3


@dataclass
class PromotionDecision:
    """Decision record for strategy promotion."""

    strategy_id: str
    decision: str
    reason: str
    timestamp: datetime
    metrics: dict[str, Any]
    approved_by: str = "auto"


@dataclass
class StrategyPromotion:
    """Represents a strategy promotion record."""

    strategy_id: str
    status: PromotionStatus
    current_allocation: Decimal
    target_allocation: Decimal
    promoted_at: datetime | None = None
    demoted_at: datetime | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    promotion_history: list[dict[str, Any]] = field(default_factory=list)
    ab_test_variant: ABTestVariant | None = None
    audit_trail: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    enabled: bool = True
    max_variants: int = 3
    control_allocation: Decimal = Decimal("0.4")
    variant_allocation: Decimal = Decimal("0.2")
    min_sample_size: int = 100
    confidence_level: float = 0.95


@dataclass
class AllocationConfig:
    """Configuration for gradual allocation."""

    auto_promote: bool = True
    initial_allocation: Decimal = Decimal("0.10")
    allocation_increment: Decimal = Decimal("0.10")
    max_allocation: Decimal = Decimal("1.00")
    increment_interval_days: int = 7
    performance_check_interval_hours: int = 24


class StrategyPromotionManager:
    """Manages strategy promotion from paper to live trading."""

    def __init__(
        self,
        validation_criteria: ValidationCriteria,
        allocation_config: AllocationConfig = None,
        ab_test_config: ABTestConfig = None,
    ):
        """Initialize promotion manager.

        Args:
            validation_criteria: Criteria for strategy validation
            allocation_config: Configuration for gradual allocation
            ab_test_config: Configuration for A/B testing
        """
        self.validation_criteria = validation_criteria
        self.allocation_config = allocation_config or AllocationConfig()
        self.ab_test_config = ab_test_config or ABTestConfig()
        self.strategies: dict[str, StrategyPromotion] = {}
        self.ab_tests: dict[str, dict[ABTestVariant, str]] = {}
        self._monitoring_tasks: dict[str, asyncio.Task] = {}

    def register_strategy(
        self,
        strategy_id: str,
        ab_test_variant: ABTestVariant | None = None,
        ab_test_group: str | None = None,
    ) -> StrategyPromotion:
        """Register a strategy for promotion tracking.

        Args:
            strategy_id: Unique strategy identifier
            ab_test_variant: A/B test variant if applicable
            ab_test_group: A/B test group identifier

        Returns:
            Strategy promotion record
        """
        if strategy_id in self.strategies:
            logger.warning("Strategy already registered", strategy_id=strategy_id)
            return self.strategies[strategy_id]

        promotion = StrategyPromotion(
            strategy_id=strategy_id,
            status=PromotionStatus.PAPER,
            current_allocation=Decimal("0"),
            target_allocation=Decimal("0"),
            ab_test_variant=ab_test_variant,
        )

        self.strategies[strategy_id] = promotion

        if ab_test_group and ab_test_variant:
            if ab_test_group not in self.ab_tests:
                self.ab_tests[ab_test_group] = {}
            self.ab_tests[ab_test_group][ab_test_variant] = strategy_id

        self._add_audit_entry(
            strategy_id,
            "REGISTERED",
            {
                "ab_test_variant": ab_test_variant.value if ab_test_variant else None,
                "ab_test_group": ab_test_group,
            },
        )

        logger.info(
            "Strategy registered for promotion tracking",
            strategy_id=strategy_id,
            ab_test_variant=ab_test_variant.value if ab_test_variant else None,
        )

        return promotion

    async def check_promotion_eligibility(
        self, strategy_id: str, metrics: dict[str, Any]
    ) -> bool:
        """Check if a strategy is eligible for promotion.

        Args:
            strategy_id: Strategy to check
            metrics: Current performance metrics

        Returns:
            True if eligible for promotion
        """
        if strategy_id not in self.strategies:
            logger.error("Strategy not registered", strategy_id=strategy_id)
            return False

        promotion = self.strategies[strategy_id]

        if promotion.status != PromotionStatus.PAPER:
            logger.info(
                "Strategy already promoted or failed",
                strategy_id=strategy_id,
                status=promotion.status.value,
            )
            return False

        is_eligible = self.validation_criteria.is_eligible(metrics)

        if is_eligible:
            self._add_audit_entry(strategy_id, "ELIGIBLE", {"metrics": metrics})

        return is_eligible

    async def promote_strategy(
        self, strategy_id: str, metrics: dict[str, Any], force: bool = False
    ) -> bool:
        """Promote a strategy to live trading.

        Args:
            strategy_id: Strategy to promote
            metrics: Current performance metrics
            force: Force promotion even if criteria not met

        Returns:
            True if promotion successful
        """
        if strategy_id not in self.strategies:
            logger.error("Strategy not registered", strategy_id=strategy_id)
            return False

        promotion = self.strategies[strategy_id]

        if not force and not await self.check_promotion_eligibility(
            strategy_id, metrics
        ):
            logger.warning(
                "Strategy not eligible for promotion", strategy_id=strategy_id
            )
            return False

        promotion.status = PromotionStatus.PENDING_PROMOTION
        promotion.baseline_metrics = metrics.copy()
        promotion.promoted_at = datetime.now()

        if self.allocation_config.auto_promote:
            promotion.status = PromotionStatus.GRADUAL_ALLOCATION
            promotion.current_allocation = self.allocation_config.initial_allocation
            promotion.target_allocation = self.allocation_config.max_allocation

            task = asyncio.create_task(self._monitor_promoted_strategy(strategy_id))
            self._monitoring_tasks[strategy_id] = task

        promotion.promotion_history.append(
            {
                "timestamp": promotion.promoted_at.isoformat(),
                "from_status": PromotionStatus.PAPER.value,
                "to_status": promotion.status.value,
                "metrics": metrics,
                "forced": force,
            }
        )

        self._add_audit_entry(
            strategy_id,
            "PROMOTED",
            {
                "initial_allocation": str(promotion.current_allocation),
                "forced": force,
                "metrics": metrics,
            },
        )

        logger.info(
            "Strategy promoted to live trading",
            strategy_id=strategy_id,
            status=promotion.status.value,
            allocation=str(promotion.current_allocation),
        )

        return True

    async def adjust_allocation(
        self, strategy_id: str, new_allocation: Decimal
    ) -> bool:
        """Manually adjust strategy allocation.

        Args:
            strategy_id: Strategy to adjust
            new_allocation: New allocation percentage

        Returns:
            True if adjustment successful
        """
        if strategy_id not in self.strategies:
            logger.error("Strategy not registered", strategy_id=strategy_id)
            return False

        promotion = self.strategies[strategy_id]

        if promotion.status not in [
            PromotionStatus.GRADUAL_ALLOCATION,
            PromotionStatus.FULL_ALLOCATION,
        ]:
            logger.warning(
                "Cannot adjust allocation for non-promoted strategy",
                strategy_id=strategy_id,
                status=promotion.status.value,
            )
            return False

        old_allocation = promotion.current_allocation
        promotion.current_allocation = new_allocation

        if new_allocation >= self.allocation_config.max_allocation:
            promotion.status = PromotionStatus.FULL_ALLOCATION
        else:
            promotion.status = PromotionStatus.GRADUAL_ALLOCATION

        self._add_audit_entry(
            strategy_id,
            "ALLOCATION_ADJUSTED",
            {
                "old_allocation": str(old_allocation),
                "new_allocation": str(new_allocation),
                "status": promotion.status.value,
            },
        )

        logger.info(
            "Strategy allocation adjusted",
            strategy_id=strategy_id,
            old_allocation=str(old_allocation),
            new_allocation=str(new_allocation),
        )

        return True

    async def demote_strategy(
        self, strategy_id: str, reason: str, metrics: dict[str, Any] | None = None
    ) -> bool:
        """Demote a strategy from live trading.

        Args:
            strategy_id: Strategy to demote
            reason: Reason for demotion
            metrics: Current performance metrics

        Returns:
            True if demotion successful
        """
        if strategy_id not in self.strategies:
            logger.error("Strategy not registered", strategy_id=strategy_id)
            return False

        promotion = self.strategies[strategy_id]

        if promotion.status == PromotionStatus.PAPER:
            logger.warning(
                "Cannot demote paper trading strategy", strategy_id=strategy_id
            )
            return False

        old_status = promotion.status
        promotion.status = PromotionStatus.DEMOTED
        promotion.current_allocation = Decimal("0")
        promotion.demoted_at = datetime.now()

        if strategy_id in self._monitoring_tasks:
            self._monitoring_tasks[strategy_id].cancel()
            del self._monitoring_tasks[strategy_id]

        promotion.promotion_history.append(
            {
                "timestamp": promotion.demoted_at.isoformat(),
                "from_status": old_status.value,
                "to_status": promotion.status.value,
                "reason": reason,
                "metrics": metrics,
            }
        )

        self._add_audit_entry(
            strategy_id,
            "DEMOTED",
            {"reason": reason, "old_status": old_status.value, "metrics": metrics},
        )

        logger.warning(
            "Strategy demoted from live trading", strategy_id=strategy_id, reason=reason
        )

        return True

    async def _monitor_promoted_strategy(self, strategy_id: str) -> None:
        """Monitor a promoted strategy for performance and gradual allocation.

        Args:
            strategy_id: Strategy to monitor
        """
        promotion = self.strategies[strategy_id]
        check_interval = timedelta(
            hours=self.allocation_config.performance_check_interval_hours
        )
        increment_interval = timedelta(
            days=self.allocation_config.increment_interval_days
        )
        last_increment = datetime.now()

        while promotion.status in [
            PromotionStatus.GRADUAL_ALLOCATION,
            PromotionStatus.FULL_ALLOCATION,
        ]:
            try:
                await asyncio.sleep(check_interval.total_seconds())

                now = datetime.now()

                if (
                    promotion.status == PromotionStatus.GRADUAL_ALLOCATION
                    and now - last_increment >= increment_interval
                ):
                    new_allocation = min(
                        promotion.current_allocation
                        + self.allocation_config.allocation_increment,
                        self.allocation_config.max_allocation,
                    )

                    await self.adjust_allocation(strategy_id, new_allocation)
                    last_increment = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error monitoring promoted strategy",
                    strategy_id=strategy_id,
                    error=str(e),
                )

    def check_regression(
        self, strategy_id: str, current_metrics: dict[str, Any]
    ) -> bool:
        """Check if a promoted strategy has regressed.

        Args:
            strategy_id: Strategy to check
            current_metrics: Current performance metrics

        Returns:
            True if regression detected
        """
        if strategy_id not in self.strategies:
            return False

        promotion = self.strategies[strategy_id]

        if not promotion.baseline_metrics:
            return False

        regression_detected = self.validation_criteria.check_regression(
            current_metrics, promotion.baseline_metrics
        )

        if regression_detected:
            self._add_audit_entry(
                strategy_id,
                "REGRESSION_DETECTED",
                {
                    "baseline_metrics": promotion.baseline_metrics,
                    "current_metrics": current_metrics,
                },
            )

        return regression_detected

    def get_ab_test_results(self, ab_test_group: str) -> dict[str, Any]:
        """Get A/B test results for a group.

        Args:
            ab_test_group: A/B test group identifier

        Returns:
            A/B test results and statistics
        """
        if ab_test_group not in self.ab_tests:
            return {}

        results = {}

        for variant, strategy_id in self.ab_tests[ab_test_group].items():
            if strategy_id in self.strategies:
                promotion = self.strategies[strategy_id]
                results[variant.value] = {
                    "strategy_id": strategy_id,
                    "status": promotion.status.value,
                    "allocation": str(promotion.current_allocation),
                    "baseline_metrics": promotion.baseline_metrics,
                    "promotion_history": promotion.promotion_history,
                }

        return {
            "ab_test_group": ab_test_group,
            "variants": results,
            "timestamp": datetime.now().isoformat(),
        }

    def _add_audit_entry(
        self, strategy_id: str, action: str, details: dict[str, Any]
    ) -> None:
        """Add entry to audit trail.

        Args:
            strategy_id: Strategy identifier
            action: Action performed
            details: Action details
        """
        if strategy_id not in self.strategies:
            return

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }

        self.strategies[strategy_id].audit_trail.append(audit_entry)

        logger.info("Audit trail entry added", strategy_id=strategy_id, action=action)

    def export_promotion_report(self, strategy_id: str) -> dict[str, Any]:
        """Export complete promotion report for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Complete promotion report
        """
        if strategy_id not in self.strategies:
            return {}

        promotion = self.strategies[strategy_id]

        return {
            "strategy_id": strategy_id,
            "status": promotion.status.value,
            "current_allocation": str(promotion.current_allocation),
            "target_allocation": str(promotion.target_allocation),
            "promoted_at": (
                promotion.promoted_at.isoformat() if promotion.promoted_at else None
            ),
            "demoted_at": (
                promotion.demoted_at.isoformat() if promotion.demoted_at else None
            ),
            "baseline_metrics": promotion.baseline_metrics,
            "promotion_history": promotion.promotion_history,
            "ab_test_variant": (
                promotion.ab_test_variant.value if promotion.ab_test_variant else None
            ),
            "audit_trail": promotion.audit_trail,
            "validation_criteria": self.validation_criteria.to_dict(),
            "allocation_config": {
                "auto_promote": self.allocation_config.auto_promote,
                "initial_allocation": str(self.allocation_config.initial_allocation),
                "allocation_increment": str(
                    self.allocation_config.allocation_increment
                ),
                "max_allocation": str(self.allocation_config.max_allocation),
                "increment_interval_days": self.allocation_config.increment_interval_days,
            },
            "export_timestamp": datetime.now().isoformat(),
        }

    async def cleanup(self) -> None:
        """Cleanup monitoring tasks."""
        for task in self._monitoring_tasks.values():
            task.cancel()

        if self._monitoring_tasks:
            await asyncio.gather(
                *self._monitoring_tasks.values(), return_exceptions=True
            )

        self._monitoring_tasks.clear()
        logger.info("Promotion manager cleanup complete")
