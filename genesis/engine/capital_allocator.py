"""
Capital allocation system for multi-strategy portfolio management.

Manages dynamic capital allocation between strategies based on performance,
risk metrics, and configurable allocation rules.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from enum import Enum

import structlog
import yaml
from pydantic import BaseModel, Field, field_validator

from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class AllocationMethod(str, Enum):
    """Capital allocation methods."""
    EQUAL_WEIGHT = "equal_weight"  # Equal allocation across strategies
    RISK_PARITY = "risk_parity"  # Inverse volatility weighting
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Based on historical performance
    KELLY_CRITERION = "kelly_criterion"  # Kelly-based sizing
    CUSTOM = "custom"  # User-defined allocation


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency options."""
    CONTINUOUS = "continuous"  # Rebalance on every update
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    THRESHOLD = "threshold"  # When drift exceeds threshold


@dataclass
class AllocationRule:
    """Rule for capital allocation."""
    rule_id: str
    name: str
    condition: str  # e.g., "performance_score > 0.7"
    action: str  # e.g., "increase_allocation"
    adjustment_percent: Decimal = Decimal("5")
    priority: int = 1
    enabled: bool = True

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.adjustment_percent, Decimal):
            self.adjustment_percent = Decimal(str(self.adjustment_percent))


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy."""
    strategy_id: str
    strategy_name: str
    current_allocation: Decimal = Decimal("0")
    target_allocation: Decimal = Decimal("0")
    min_allocation: Decimal = Decimal("0")
    max_allocation: Decimal = Decimal("100")
    locked_capital: Decimal = Decimal("0")  # Capital in open positions
    available_capital: Decimal = Decimal("0")
    performance_score: Decimal = Decimal("1.0")
    risk_score: Decimal = Decimal("1.0")
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Ensure Decimal types."""
        for attr in ['current_allocation', 'target_allocation', 'min_allocation',
                    'max_allocation', 'locked_capital', 'available_capital',
                    'performance_score', 'risk_score']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))


class AllocationConfig(BaseModel):
    """Configuration for capital allocation."""

    method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.DAILY
    max_strategies: int = 10
    min_allocation_percent: Decimal = Field(default=Decimal("5"))
    max_allocation_percent: Decimal = Field(default=Decimal("40"))
    reserve_percent: Decimal = Field(default=Decimal("10"))  # Keep in reserve
    rebalance_threshold_percent: Decimal = Field(default=Decimal("5"))
    use_kelly_sizing: bool = False
    kelly_fraction: Decimal = Field(default=Decimal("0.25"))  # Conservative Kelly

    @field_validator("min_allocation_percent", "max_allocation_percent",
                     "reserve_percent", "rebalance_threshold_percent", "kelly_fraction")
    @classmethod
    def ensure_decimal(cls, v):
        """Convert to Decimal."""
        return Decimal(str(v)) if not isinstance(v, Decimal) else v

    @field_validator("max_allocation_percent")
    @classmethod
    def validate_max_allocation(cls, v, values):
        """Ensure max > min allocation."""
        min_alloc = values.data.get("min_allocation_percent", Decimal("5"))
        if v <= min_alloc:
            raise ValueError("Max allocation must be greater than min allocation")
        return v


class CapitalAllocator:
    """
    Manages capital allocation across multiple trading strategies.

    Implements various allocation methods and enforces risk limits.
    """

    def __init__(
        self,
        event_bus: EventBus,
        total_capital: Decimal,
        config_path: str | None = None
    ):
        """
        Initialize capital allocator.

        Args:
            event_bus: Event bus for allocation events
            total_capital: Total available capital
            config_path: Optional path to allocation config file
        """
        self.event_bus = event_bus
        self.total_capital = Decimal(str(total_capital))
        self.config = self._load_config(config_path)
        self.allocations: dict[str, StrategyAllocation] = {}
        self.allocation_rules: list[AllocationRule] = []
        self.last_rebalance: datetime = datetime.now(UTC)
        self._lock = asyncio.Lock()

    def _load_config(self, config_path: str | None) -> AllocationConfig:
        """
        Load allocation configuration.

        Args:
            config_path: Path to config file

        Returns:
            AllocationConfig instance
        """
        if config_path:
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                    return AllocationConfig(**config_data.get('allocation', {}))
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return AllocationConfig()

    async def allocate_capital(
        self,
        strategy_allocations: list[StrategyAllocation]
    ) -> dict[str, Decimal]:
        """
        Allocate capital across strategies.

        Args:
            strategy_allocations: List of strategy allocation requests

        Returns:
            Dictionary of strategy_id -> allocated capital

        Raises:
            ValueError: If allocation constraints cannot be satisfied
        """
        async with self._lock:
            # Update internal allocations
            for alloc in strategy_allocations:
                self.allocations[alloc.strategy_id] = alloc

            # Calculate allocations based on method
            if self.config.method == AllocationMethod.EQUAL_WEIGHT:
                allocations = await self._allocate_equal_weight()
            elif self.config.method == AllocationMethod.PERFORMANCE_WEIGHTED:
                allocations = await self._allocate_performance_weighted()
            elif self.config.method == AllocationMethod.RISK_PARITY:
                allocations = await self._allocate_risk_parity()
            elif self.config.method == AllocationMethod.KELLY_CRITERION:
                allocations = await self._allocate_kelly()
            else:
                allocations = await self._allocate_custom()

            # Apply constraints
            allocations = self._apply_constraints(allocations)

            # Update allocations
            for strategy_id, capital in allocations.items():
                if strategy_id in self.allocations:
                    self.allocations[strategy_id].current_allocation = capital
                    self.allocations[strategy_id].available_capital = (
                        capital - self.allocations[strategy_id].locked_capital
                    )
                    self.allocations[strategy_id].last_updated = datetime.now(UTC)

            # Publish allocation event
            await self.event_bus.publish(Event(
                event_type=EventType.STRATEGY_CAPITAL_ADJUSTED,
                event_data={
                    "allocations": {k: str(v) for k, v in allocations.items()},
                    "total_allocated": str(sum(allocations.values())),
                    "method": self.config.method.value
                }
            ))

            logger.info(
                "Capital allocated",
                method=self.config.method.value,
                num_strategies=len(allocations),
                total_allocated=sum(allocations.values())
            )

            return allocations

    async def _allocate_equal_weight(self) -> dict[str, Decimal]:
        """
        Allocate capital equally across strategies.

        Returns:
            Dictionary of allocations
        """
        if not self.allocations:
            return {}

        # Calculate available capital (excluding reserve)
        available = self.total_capital * (Decimal("1") - self.config.reserve_percent / Decimal("100"))

        # Equal allocation per strategy
        num_strategies = len(self.allocations)
        per_strategy = available / Decimal(str(num_strategies))

        allocations = {}
        for strategy_id in self.allocations:
            allocations[strategy_id] = per_strategy.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        return allocations

    async def _allocate_performance_weighted(self) -> dict[str, Decimal]:
        """
        Allocate based on performance scores.

        Returns:
            Dictionary of allocations
        """
        if not self.allocations:
            return {}

        # Calculate total performance score
        total_score = sum(
            max(alloc.performance_score, Decimal("0.1"))  # Minimum score to avoid zero allocation
            for alloc in self.allocations.values()
        )

        if total_score == 0:
            # Fall back to equal weight
            return await self._allocate_equal_weight()

        # Calculate available capital
        available = self.total_capital * (Decimal("1") - self.config.reserve_percent / Decimal("100"))

        # Allocate proportional to performance
        allocations = {}
        for strategy_id, alloc in self.allocations.items():
            score = max(alloc.performance_score, Decimal("0.1"))
            weight = score / total_score
            allocations[strategy_id] = (available * weight).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        return allocations

    async def _allocate_risk_parity(self) -> dict[str, Decimal]:
        """
        Allocate based on risk parity (inverse volatility).

        Returns:
            Dictionary of allocations
        """
        if not self.allocations:
            return {}

        # Calculate inverse risk scores
        inverse_risks = {}
        total_inverse = Decimal("0")

        for strategy_id, alloc in self.allocations.items():
            # Higher risk score = lower allocation
            inverse_risk = Decimal("1") / max(alloc.risk_score, Decimal("0.1"))
            inverse_risks[strategy_id] = inverse_risk
            total_inverse += inverse_risk

        if total_inverse == 0:
            return await self._allocate_equal_weight()

        # Calculate available capital
        available = self.total_capital * (Decimal("1") - self.config.reserve_percent / Decimal("100"))

        # Allocate inversely proportional to risk
        allocations = {}
        for strategy_id, inverse_risk in inverse_risks.items():
            weight = inverse_risk / total_inverse
            allocations[strategy_id] = (available * weight).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        return allocations

    async def _allocate_kelly(self) -> dict[str, Decimal]:
        """
        Allocate using Kelly Criterion.

        Returns:
            Dictionary of allocations
        """
        if not self.allocations:
            return {}

        # Calculate Kelly fractions for each strategy
        kelly_allocations = {}

        for strategy_id, alloc in self.allocations.items():
            # Simplified Kelly: f = (p*b - q) / b
            # where p = win probability, b = win/loss ratio, q = 1-p
            # Using performance_score as proxy for edge

            edge = alloc.performance_score - Decimal("0.5")  # Edge over random
            if edge <= 0:
                kelly_allocations[strategy_id] = Decimal("0")
            else:
                # Conservative Kelly with fraction
                kelly_percent = edge * self.config.kelly_fraction * Decimal("100")
                kelly_allocations[strategy_id] = min(
                    kelly_percent,
                    self.config.max_allocation_percent
                )

        # Normalize if total exceeds 100%
        total_kelly = sum(kelly_allocations.values())
        if total_kelly > Decimal("100") - self.config.reserve_percent:
            scale = (Decimal("100") - self.config.reserve_percent) / total_kelly
            for strategy_id in kelly_allocations:
                kelly_allocations[strategy_id] *= scale

        # Convert percentages to capital amounts
        allocations = {}
        for strategy_id, percent in kelly_allocations.items():
            allocations[strategy_id] = (
                self.total_capital * percent / Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        return allocations

    async def _allocate_custom(self) -> dict[str, Decimal]:
        """
        Apply custom allocation rules.

        Returns:
            Dictionary of allocations
        """
        # Start with equal weight as base
        allocations = await self._allocate_equal_weight()

        # Apply custom rules
        for rule in self.allocation_rules:
            if not rule.enabled:
                continue

            for strategy_id, alloc in self.allocations.items():
                # Evaluate rule condition (simplified)
                if self._evaluate_rule(rule, alloc):
                    # Apply adjustment
                    adjustment = allocations[strategy_id] * rule.adjustment_percent / Decimal("100")

                    if "increase" in rule.action:
                        allocations[strategy_id] += adjustment
                    elif "decrease" in rule.action:
                        allocations[strategy_id] = max(
                            allocations[strategy_id] - adjustment,
                            Decimal("0")
                        )

        return allocations

    def _evaluate_rule(self, rule: AllocationRule, allocation: StrategyAllocation) -> bool:
        """
        Evaluate if a rule applies to an allocation.

        Args:
            rule: Allocation rule
            allocation: Strategy allocation

        Returns:
            True if rule applies
        """
        # Simplified rule evaluation
        # In production, would use a proper expression evaluator
        if "performance_score >" in rule.condition:
            threshold = Decimal(rule.condition.split(">")[1].strip())
            return allocation.performance_score > threshold
        elif "risk_score <" in rule.condition:
            threshold = Decimal(rule.condition.split("<")[1].strip())
            return allocation.risk_score < threshold

        return False

    def _apply_constraints(self, allocations: dict[str, Decimal]) -> dict[str, Decimal]:
        """
        Apply min/max constraints to allocations.

        Args:
            allocations: Proposed allocations

        Returns:
            Constrained allocations
        """
        constrained = {}
        total_allocated = Decimal("0")
        available = self.total_capital * (Decimal("1") - self.config.reserve_percent / Decimal("100"))

        for strategy_id, amount in allocations.items():
            # Apply min/max percentages
            min_amount = available * self.config.min_allocation_percent / Decimal("100")
            max_amount = available * self.config.max_allocation_percent / Decimal("100")

            # Apply strategy-specific constraints
            if strategy_id in self.allocations:
                alloc = self.allocations[strategy_id]
                min_amount = max(min_amount, available * alloc.min_allocation / Decimal("100"))
                max_amount = min(max_amount, available * alloc.max_allocation / Decimal("100"))

            # Constrain amount
            constrained_amount = max(min(amount, max_amount), min_amount)
            constrained[strategy_id] = constrained_amount
            total_allocated += constrained_amount

        # Ensure we don't exceed available capital
        if total_allocated > available:
            scale = available / total_allocated
            for strategy_id in constrained:
                constrained[strategy_id] = (
                    constrained[strategy_id] * scale
                ).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

        return constrained

    async def rebalance(self, force: bool = False) -> bool:
        """
        Rebalance portfolio allocations.

        Args:
            force: Force rebalance regardless of schedule

        Returns:
            True if rebalanced
        """
        async with self._lock:
            # Check if rebalance is needed
            if not force and not self._should_rebalance():
                return False

            # Calculate new allocations
            strategy_list = list(self.allocations.values())
            new_allocations = await self.allocate_capital(strategy_list)

            self.last_rebalance = datetime.now(UTC)

            logger.info(
                "Portfolio rebalanced",
                num_strategies=len(new_allocations),
                total_allocated=sum(new_allocations.values())
            )

            return True

    def _should_rebalance(self) -> bool:
        """
        Check if rebalancing is needed.

        Returns:
            True if rebalance should occur
        """
        now = datetime.now(UTC)

        if self.config.rebalance_frequency == RebalanceFrequency.CONTINUOUS:
            return True

        # Check time-based rebalancing
        time_since_last = (now - self.last_rebalance).total_seconds()

        if self.config.rebalance_frequency == RebalanceFrequency.DAILY:
            return time_since_last >= 86400
        elif self.config.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return time_since_last >= 604800
        elif self.config.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return time_since_last >= 2592000

        # Check threshold-based rebalancing
        if self.config.rebalance_frequency == RebalanceFrequency.THRESHOLD:
            for alloc in self.allocations.values():
                drift = abs(alloc.current_allocation - alloc.target_allocation)
                if drift > self.config.rebalance_threshold_percent:
                    return True

        return False

    def update_strategy_performance(
        self,
        strategy_id: str,
        performance_score: Decimal,
        risk_score: Decimal | None = None
    ) -> None:
        """
        Update strategy performance metrics.

        Args:
            strategy_id: Strategy ID
            performance_score: New performance score (0-1+)
            risk_score: Optional new risk score (0-1+)
        """
        if strategy_id in self.allocations:
            self.allocations[strategy_id].performance_score = Decimal(str(performance_score))
            if risk_score is not None:
                self.allocations[strategy_id].risk_score = Decimal(str(risk_score))

    def lock_capital(self, strategy_id: str, amount: Decimal) -> bool:
        """
        Lock capital for a strategy (e.g., for open positions).

        Args:
            strategy_id: Strategy ID
            amount: Amount to lock

        Returns:
            True if successful
        """
        if strategy_id not in self.allocations:
            return False

        alloc = self.allocations[strategy_id]

        if amount > alloc.available_capital:
            logger.warning(
                "Insufficient available capital",
                strategy_id=strategy_id,
                requested=amount,
                available=alloc.available_capital
            )
            return False

        alloc.locked_capital += amount
        alloc.available_capital -= amount

        return True

    def unlock_capital(self, strategy_id: str, amount: Decimal) -> bool:
        """
        Unlock capital for a strategy.

        Args:
            strategy_id: Strategy ID
            amount: Amount to unlock

        Returns:
            True if successful
        """
        if strategy_id not in self.allocations:
            return False

        alloc = self.allocations[strategy_id]

        unlock_amount = min(amount, alloc.locked_capital)
        alloc.locked_capital -= unlock_amount
        alloc.available_capital += unlock_amount

        return True

    def get_available_capital(self, strategy_id: str) -> Decimal:
        """
        Get available capital for a strategy.

        Args:
            strategy_id: Strategy ID

        Returns:
            Available capital amount
        """
        if strategy_id in self.allocations:
            return self.allocations[strategy_id].available_capital
        return Decimal("0")

    def get_allocation_summary(self) -> dict[str, any]:
        """
        Get summary of current allocations.

        Returns:
            Allocation summary dictionary
        """
        total_allocated = sum(
            alloc.current_allocation for alloc in self.allocations.values()
        )
        total_locked = sum(
            alloc.locked_capital for alloc in self.allocations.values()
        )

        return {
            "total_capital": str(self.total_capital),
            "total_allocated": str(total_allocated),
            "total_locked": str(total_locked),
            "reserve": str(self.total_capital - total_allocated),
            "num_strategies": len(self.allocations),
            "allocation_method": self.config.method.value,
            "strategies": {
                sid: {
                    "name": alloc.strategy_name,
                    "current": str(alloc.current_allocation),
                    "target": str(alloc.target_allocation),
                    "locked": str(alloc.locked_capital),
                    "available": str(alloc.available_capital),
                    "performance": str(alloc.performance_score)
                }
                for sid, alloc in self.allocations.items()
            }
        }

    def add_rule(self, rule: AllocationRule) -> None:
        """
        Add a custom allocation rule.

        Args:
            rule: Allocation rule to add
        """
        self.allocation_rules.append(rule)
        self.allocation_rules.sort(key=lambda r: r.priority)

        logger.info(
            "Allocation rule added",
            rule_id=rule.rule_id,
            rule_name=rule.name
        )
