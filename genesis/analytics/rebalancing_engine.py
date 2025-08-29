"""
Rebalancing Engine for Portfolio Optimization

Manages portfolio rebalancing with transaction cost analysis.
Implements both threshold-based and scheduled rebalancing strategies.
Part of the Hunter+ tier portfolio optimization suite.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

import structlog

from genesis.core.constants import TradingTier
from genesis.core.events import Event, EventType
from genesis.core.exceptions import (
    DataError as InvalidDataError,
)
from genesis.core.exceptions import (
    GenesisException as CalculationError,
)
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier, with_timeout

logger = structlog.get_logger(__name__)


class RebalanceTrigger(Enum):
    """Types of rebalancing triggers"""

    THRESHOLD = "threshold"  # Deviation from target
    SCHEDULED = "scheduled"  # Time-based
    EMERGENCY = "emergency"  # Risk event
    MANUAL = "manual"  # User-initiated


@dataclass
class RebalanceAction:
    """Represents a single rebalancing action"""

    strategy: str
    current_weight: Decimal
    target_weight: Decimal
    delta_weight: Decimal
    current_value_usdt: Decimal
    target_value_usdt: Decimal
    delta_value_usdt: Decimal
    estimated_cost_usdt: Decimal  # Transaction cost

    @property
    def is_buy(self) -> bool:
        """Whether this action is a buy (increase allocation)"""
        return self.delta_value_usdt > Decimal("0")

    @property
    def is_sell(self) -> bool:
        """Whether this action is a sell (decrease allocation)"""
        return self.delta_value_usdt < Decimal("0")


@dataclass
class TransactionCosts:
    """Transaction cost breakdown"""

    maker_fee: Decimal  # Percentage (e.g., 0.001 for 0.1%)
    taker_fee: Decimal  # Percentage
    estimated_slippage: Decimal  # Percentage
    total_cost_usdt: Decimal  # Total estimated cost in USDT

    @property
    def total_percentage(self) -> Decimal:
        """Total cost as percentage"""
        return self.maker_fee + self.taker_fee + self.estimated_slippage


@dataclass
class RebalanceRecommendation:
    """Recommendation for portfolio rebalancing"""

    trigger: RebalanceTrigger
    actions: list[RebalanceAction]
    total_cost_usdt: Decimal
    expected_improvement: Decimal  # Expected Sharpe ratio improvement
    cost_benefit_ratio: Decimal  # Benefit / Cost
    should_execute: bool
    rationale: str
    generated_at: datetime = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now(UTC)

    @property
    def total_turnover_usdt(self) -> Decimal:
        """Total value of assets to be traded"""
        return sum(abs(action.delta_value_usdt) for action in self.actions) / Decimal(
            "2"
        )  # Divide by 2 to avoid double counting


class RebalancingEngine:
    """
    Manages portfolio rebalancing with cost-benefit analysis.

    Implements intelligent rebalancing that considers transaction costs
    and expected improvements to avoid unnecessary churn.
    """

    # Default configuration
    DEFAULT_THRESHOLD_PERCENT = Decimal("5.0")  # 5% deviation triggers rebalance
    DEFAULT_MIN_IMPROVEMENT = Decimal("0.01")  # Minimum 0.01 Sharpe improvement
    DEFAULT_MAKER_FEE = Decimal("0.001")  # 0.1% maker fee
    DEFAULT_TAKER_FEE = Decimal("0.001")  # 0.1% taker fee
    DEFAULT_SLIPPAGE = Decimal("0.0005")  # 0.05% slippage estimate

    def __init__(
        self, event_bus: EventBus | None = None, config: dict | None = None
    ):
        """
        Initialize rebalancing engine.

        Args:
            event_bus: Optional event bus for notifications
            config: Optional configuration overrides
        """
        self.event_bus = event_bus

        # Load configuration
        self.config = config or {}
        self.threshold_percent = Decimal(
            str(self.config.get("threshold_percent", self.DEFAULT_THRESHOLD_PERCENT))
        )
        self.min_improvement = Decimal(
            str(self.config.get("min_improvement", self.DEFAULT_MIN_IMPROVEMENT))
        )
        self.maker_fee = Decimal(
            str(self.config.get("maker_fee", self.DEFAULT_MAKER_FEE))
        )
        self.taker_fee = Decimal(
            str(self.config.get("taker_fee", self.DEFAULT_TAKER_FEE))
        )
        self.slippage = Decimal(str(self.config.get("slippage", self.DEFAULT_SLIPPAGE)))

        # Scheduling state
        self._last_rebalance = datetime.now(UTC)
        self._rebalance_schedule = self.config.get("schedule", "weekly")

        logger.info(
            "rebalancing_engine_initialized",
            threshold=float(self.threshold_percent),
            schedule=self._rebalance_schedule,
        )

    @requires_tier(TradingTier.HUNTER)
    @with_timeout(5.0)
    async def check_rebalance_triggers(
        self,
        current_weights: dict[str, Decimal],
        target_weights: dict[str, Decimal],
        portfolio_value_usdt: Decimal,
        expected_sharpe_improvement: Decimal | None = None,
    ) -> RebalanceRecommendation:
        """
        Check if rebalancing should be triggered and generate recommendation.

        Args:
            current_weights: Current portfolio weights by strategy
            target_weights: Target portfolio weights from optimization
            portfolio_value_usdt: Total portfolio value in USDT
            expected_sharpe_improvement: Expected Sharpe ratio improvement

        Returns:
            RebalanceRecommendation with actions and cost-benefit analysis
        """
        try:
            # Validate inputs
            if not current_weights or not target_weights:
                raise InvalidDataError("Weights cannot be empty")

            # Ensure all strategies are in both dicts
            all_strategies = set(current_weights.keys()) | set(target_weights.keys())
            for strategy in all_strategies:
                current_weights.setdefault(strategy, Decimal("0"))
                target_weights.setdefault(strategy, Decimal("0"))

            # Check triggers
            trigger = await self._determine_trigger(current_weights, target_weights)

            if trigger is None:
                # No trigger, return empty recommendation
                return RebalanceRecommendation(
                    trigger=RebalanceTrigger.THRESHOLD,
                    actions=[],
                    total_cost_usdt=Decimal("0"),
                    expected_improvement=Decimal("0"),
                    cost_benefit_ratio=Decimal("0"),
                    should_execute=False,
                    rationale="No rebalancing trigger met",
                )

            # Calculate rebalancing actions
            actions = await self._calculate_actions(
                current_weights, target_weights, portfolio_value_usdt
            )

            # Calculate transaction costs
            total_cost = await self._calculate_total_costs(actions)

            # Perform cost-benefit analysis
            recommendation = await self._analyze_cost_benefit(
                trigger,
                actions,
                total_cost,
                expected_sharpe_improvement or Decimal("0"),
            )

            # Send event if recommendation generated
            if self.event_bus and recommendation.should_execute:
                await self._send_rebalance_event(recommendation)

            logger.info(
                "rebalance_recommendation_generated",
                trigger=trigger.value,
                num_actions=len(actions),
                total_cost=float(total_cost.total_cost_usdt),
                should_execute=recommendation.should_execute,
            )

            return recommendation

        except Exception as e:
            logger.error("rebalance_check_failed", error=str(e))
            raise CalculationError(f"Failed to check rebalance triggers: {e}")

    async def _determine_trigger(
        self, current_weights: dict[str, Decimal], target_weights: dict[str, Decimal]
    ) -> RebalanceTrigger | None:
        """Determine if any rebalancing trigger is met"""
        # Check threshold trigger
        max_deviation = Decimal("0")
        for strategy in current_weights:
            current = current_weights[strategy]
            target = target_weights.get(strategy, Decimal("0"))
            deviation = abs(current - target)
            max_deviation = max(max_deviation, deviation)

        if max_deviation * Decimal("100") > self.threshold_percent:
            return RebalanceTrigger.THRESHOLD

        # Check scheduled trigger
        if await self._is_scheduled_rebalance_due():
            return RebalanceTrigger.SCHEDULED

        return None

    async def _is_scheduled_rebalance_due(self) -> bool:
        """Check if scheduled rebalancing is due"""
        now = datetime.now(UTC)
        time_since_last = now - self._last_rebalance

        if self._rebalance_schedule == "daily":
            return time_since_last > timedelta(days=1)
        elif self._rebalance_schedule == "weekly":
            return time_since_last > timedelta(weeks=1)
        elif self._rebalance_schedule == "monthly":
            return time_since_last > timedelta(days=30)

        return False

    async def _calculate_actions(
        self,
        current_weights: dict[str, Decimal],
        target_weights: dict[str, Decimal],
        portfolio_value_usdt: Decimal,
    ) -> list[RebalanceAction]:
        """Calculate specific rebalancing actions"""
        actions = []

        for strategy in current_weights:
            current = current_weights[strategy]
            target = target_weights.get(strategy, Decimal("0"))
            delta = target - current

            if abs(delta) < Decimal("0.0001"):  # Skip tiny changes
                continue

            current_value = current * portfolio_value_usdt
            target_value = target * portfolio_value_usdt
            delta_value = delta * portfolio_value_usdt

            # Estimate transaction cost for this action
            cost = await self._estimate_transaction_cost(abs(delta_value))

            action = RebalanceAction(
                strategy=strategy,
                current_weight=current,
                target_weight=target,
                delta_weight=delta,
                current_value_usdt=current_value.quantize(
                    Decimal("0.01"), ROUND_HALF_UP
                ),
                target_value_usdt=target_value.quantize(Decimal("0.01"), ROUND_HALF_UP),
                delta_value_usdt=delta_value.quantize(Decimal("0.01"), ROUND_HALF_UP),
                estimated_cost_usdt=cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
            )

            actions.append(action)

        return actions

    async def _estimate_transaction_cost(self, trade_value_usdt: Decimal) -> Decimal:
        """Estimate transaction cost for a single trade"""
        # Assume mix of maker and taker orders
        avg_fee = (self.maker_fee + self.taker_fee) / Decimal("2")

        # Add slippage
        total_cost_percent = avg_fee + self.slippage

        return trade_value_usdt * total_cost_percent

    async def _calculate_total_costs(
        self, actions: list[RebalanceAction]
    ) -> TransactionCosts:
        """Calculate total transaction costs for all actions"""
        total_turnover = sum(abs(a.delta_value_usdt) for a in actions)

        # Calculate weighted average fees
        avg_fee = (self.maker_fee + self.taker_fee) / Decimal("2")

        # Total costs
        fee_cost = total_turnover * avg_fee
        slippage_cost = total_turnover * self.slippage
        total_cost = fee_cost + slippage_cost

        return TransactionCosts(
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            estimated_slippage=self.slippage,
            total_cost_usdt=total_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
        )

    async def _analyze_cost_benefit(
        self,
        trigger: RebalanceTrigger,
        actions: list[RebalanceAction],
        costs: TransactionCosts,
        expected_improvement: Decimal,
    ) -> RebalanceRecommendation:
        """Analyze cost-benefit of rebalancing"""
        # Calculate expected annual benefit (simplified)
        # Assume Sharpe improvement translates to return improvement
        portfolio_value = sum(a.current_value_usdt for a in actions)
        annual_benefit = (
            portfolio_value * expected_improvement * Decimal("0.1")
        )  # Rough estimate

        # Cost-benefit ratio
        if costs.total_cost_usdt > Decimal("0"):
            cost_benefit_ratio = annual_benefit / costs.total_cost_usdt
        else:
            cost_benefit_ratio = Decimal("999")

        # Decision logic
        should_execute = False
        rationale = ""

        if trigger == RebalanceTrigger.EMERGENCY:
            should_execute = True
            rationale = "Emergency rebalancing triggered due to risk event"
        elif trigger == RebalanceTrigger.MANUAL:
            should_execute = True
            rationale = "Manual rebalancing requested by user"
        elif cost_benefit_ratio > Decimal("2"):  # Benefits > 2x costs
            should_execute = True
            rationale = (
                f"Cost-benefit ratio of {cost_benefit_ratio:.2f} exceeds threshold"
            )
        elif expected_improvement > self.min_improvement:
            should_execute = True
            rationale = f"Expected Sharpe improvement of {expected_improvement:.4f} exceeds minimum"
        else:
            rationale = f"Costs outweigh benefits (ratio: {cost_benefit_ratio:.2f})"

        return RebalanceRecommendation(
            trigger=trigger,
            actions=actions,
            total_cost_usdt=costs.total_cost_usdt,
            expected_improvement=expected_improvement,
            cost_benefit_ratio=cost_benefit_ratio.quantize(
                Decimal("0.01"), ROUND_HALF_UP
            ),
            should_execute=should_execute,
            rationale=rationale,
        )

    async def _send_rebalance_event(self, recommendation: RebalanceRecommendation):
        """Send rebalancing event to event bus"""
        if not self.event_bus:
            return

        event = Event(
            type=EventType.REBALANCE_RECOMMENDED,
            data={
                "trigger": recommendation.trigger.value,
                "num_actions": len(recommendation.actions),
                "total_cost": float(recommendation.total_cost_usdt),
                "expected_improvement": float(recommendation.expected_improvement),
                "should_execute": recommendation.should_execute,
                "rationale": recommendation.rationale,
            },
        )

        await self.event_bus.publish(event)

    async def execute_rebalance(self, recommendation: RebalanceRecommendation) -> bool:
        """
        Execute rebalancing based on recommendation.

        Note: This is a placeholder - actual execution would interface
        with order management system.

        Args:
            recommendation: Rebalancing recommendation to execute

        Returns:
            True if execution successful
        """
        if not recommendation.should_execute:
            logger.warning("attempted_to_execute_non_recommended_rebalance")
            return False

        logger.info(
            "executing_rebalance",
            num_actions=len(recommendation.actions),
            total_turnover=float(recommendation.total_turnover_usdt),
        )

        # Update last rebalance time
        self._last_rebalance = datetime.now(UTC)

        # In production, this would:
        # 1. Create orders for each action
        # 2. Execute orders via exchange
        # 3. Monitor execution
        # 4. Update portfolio state

        return True

    @requires_tier(TradingTier.HUNTER)
    async def generate_weekly_recommendation(
        self,
        current_weights: dict[str, Decimal],
        optimal_weights: dict[str, Decimal],
        portfolio_value_usdt: Decimal,
        historical_performance: dict,
    ) -> dict:
        """
        Generate weekly rebalancing recommendation report.

        Args:
            current_weights: Current portfolio allocations
            optimal_weights: Optimal allocations from optimizer
            portfolio_value_usdt: Total portfolio value
            historical_performance: Historical performance metrics

        Returns:
            Detailed recommendation report
        """
        # Force scheduled trigger for weekly recommendation
        recommendation = await self.check_rebalance_triggers(
            current_weights,
            optimal_weights,
            portfolio_value_usdt,
            expected_sharpe_improvement=Decimal("0.05"),  # Estimate
        )

        # Generate detailed report
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "portfolio_value_usdt": float(portfolio_value_usdt),
            "recommendation": {
                "should_execute": recommendation.should_execute,
                "rationale": recommendation.rationale,
                "expected_improvement": float(recommendation.expected_improvement),
                "total_cost_usdt": float(recommendation.total_cost_usdt),
                "cost_benefit_ratio": float(recommendation.cost_benefit_ratio),
            },
            "current_allocation": {k: float(v) for k, v in current_weights.items()},
            "target_allocation": {k: float(v) for k, v in optimal_weights.items()},
            "actions": [
                {
                    "strategy": action.strategy,
                    "action": "buy" if action.is_buy else "sell",
                    "amount_usdt": float(abs(action.delta_value_usdt)),
                    "cost_usdt": float(action.estimated_cost_usdt),
                }
                for action in recommendation.actions
            ],
            "performance_context": historical_performance,
        }

        logger.info(
            "weekly_recommendation_generated",
            should_execute=recommendation.should_execute,
        )

        return report
