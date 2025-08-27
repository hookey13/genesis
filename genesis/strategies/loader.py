"""Strategy loader with automatic migration enforcement.

Manages strategy availability by tier and enforces automatic
migration during tier transitions to prevent dangerous strategy misuse.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import structlog

from genesis.data.models_db import AdjustmentPeriod, Session, get_session

logger = structlog.get_logger(__name__)


class StrategyStatus(Enum):
    """Strategy availability status."""
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    LOCKED = "LOCKED"  # Tier-locked
    MIGRATION = "MIGRATION"  # In migration period


# Strategy availability by tier
TIER_STRATEGIES = {
    'SNIPER': [
        'simple_arbitrage',
        'spread_capture',
        'market_orders_only'
    ],
    'HUNTER': [
        'simple_arbitrage',
        'spread_capture',
        'market_orders_only',
        'iceberg_orders',
        'multi_pair_trading',
        'mean_reversion'
    ],
    'STRATEGIST': [
        'simple_arbitrage',
        'spread_capture',
        'market_orders_only',
        'iceberg_orders',
        'multi_pair_trading',
        'mean_reversion',
        'statistical_arbitrage',
        'vwap_execution',
        'market_making'
    ],
    'ARCHITECT': [
        # All previous strategies plus:
        'cross_exchange_arbitrage',
        'options_strategies',
        'complex_portfolio_management'
    ],
    'EMPEROR': [
        # All strategies available
        'unlimited_strategy_access'
    ]
}

# Position size adjustments during migration
MIGRATION_POSITION_MULTIPLIERS = {
    'new_strategy': Decimal('0.25'),  # Start at 25% for new strategies
    'existing_strategy': Decimal('0.75'),  # Reduce to 75% for existing
    'adjustment_period': Decimal('0.50')  # 50% during adjustment period
}


@dataclass
class TierStrategy:
    """Strategy configuration tied to a specific tier."""
    name: str
    tier: str
    enabled: bool = True


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    tier_required: str
    status: StrategyStatus
    position_multiplier: Decimal
    max_position_usdt: Decimal
    enabled_at: Optional[datetime] = None
    disabled_at: Optional[datetime] = None
    migration_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'tier_required': self.tier_required,
            'status': self.status.value,
            'position_multiplier': float(self.position_multiplier),
            'max_position_usdt': float(self.max_position_usdt),
            'enabled_at': self.enabled_at.isoformat() if self.enabled_at else None,
            'disabled_at': self.disabled_at.isoformat() if self.disabled_at else None,
            'migration_notes': self.migration_notes
        }


@dataclass
class MigrationPlan:
    """Plan for strategy migration during tier transition."""
    account_id: str
    old_tier: str
    new_tier: str
    strategies_to_disable: list[str]
    strategies_to_enable: list[str]
    strategies_to_adjust: list[str]
    migration_timestamp: datetime
    rollback_available: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'account_id': self.account_id,
            'old_tier': self.old_tier,
            'new_tier': self.new_tier,
            'strategies_to_disable': self.strategies_to_disable,
            'strategies_to_enable': self.strategies_to_enable,
            'strategies_to_adjust': self.strategies_to_adjust,
            'migration_timestamp': self.migration_timestamp.isoformat(),
            'rollback_available': self.rollback_available
        }


class StrategyLoader:
    """Loads and manages strategies based on tier."""

    def __init__(self, session: Optional[Session] = None):
        """Initialize strategy loader.
        
        Args:
            session: Optional database session
        """
        self.session = session or get_session()
        self._loaded_strategies: dict[str, Any] = {}
        self._active_configs: dict[str, dict[str, StrategyConfig]] = {}
        self._migration_history: list[MigrationPlan] = []

    async def load_strategies(self, account_id: str, tier: str) -> list[StrategyConfig]:
        """Load strategies available for a tier.
        
        Args:
            account_id: Account ID
            tier: Tier name
            
        Returns:
            List of available strategy configurations
        """
        # Get base strategies for tier
        available_strategies = self._get_tier_strategies(tier)

        # Check for adjustment period
        adjustment_period = self._get_active_adjustment_period(account_id)

        # Create configurations
        configs = []
        for strategy_name in available_strategies:
            config = await self._create_strategy_config(
                account_id=account_id,
                strategy_name=strategy_name,
                tier=tier,
                adjustment_period=adjustment_period
            )
            configs.append(config)

        # Store active configs
        self._active_configs[account_id] = {
            config.name: config for config in configs
        }

        logger.info(
            "Strategies loaded for tier",
            account_id=account_id,
            tier=tier,
            strategy_count=len(configs),
            in_adjustment=adjustment_period is not None
        )

        return configs

    async def migrate_strategies(
        self,
        account_id: str,
        old_tier: str,
        new_tier: str
    ) -> MigrationPlan:
        """Migrate strategies during tier transition.
        
        Args:
            account_id: Account ID
            old_tier: Previous tier
            new_tier: New tier
            
        Returns:
            MigrationPlan with details
            
        Raises:
            ValidationError: If migration invalid
        """
        # Get strategy sets
        old_strategies = set(self._get_tier_strategies(old_tier))
        new_strategies = set(self._get_tier_strategies(new_tier))

        # Determine changes
        strategies_to_disable = list(old_strategies - new_strategies) if len(old_strategies) > len(new_strategies) else []
        strategies_to_enable = list(new_strategies - old_strategies)
        strategies_to_adjust = list(old_strategies & new_strategies)

        # Create migration plan
        plan = MigrationPlan(
            account_id=account_id,
            old_tier=old_tier,
            new_tier=new_tier,
            strategies_to_disable=strategies_to_disable,
            strategies_to_enable=strategies_to_enable,
            strategies_to_adjust=strategies_to_adjust,
            migration_timestamp=datetime.utcnow(),
            rollback_available=True
        )

        # Execute migration
        await self._execute_migration(account_id, plan)

        # Store history
        self._migration_history.append(plan)

        logger.info(
            "Strategy migration completed",
            account_id=account_id,
            old_tier=old_tier,
            new_tier=new_tier,
            disabled=len(strategies_to_disable),
            enabled=len(strategies_to_enable),
            adjusted=len(strategies_to_adjust)
        )

        return plan

    async def rollback_migration(
        self,
        account_id: str
    ) -> bool:
        """Rollback the last strategy migration.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if rollback successful
        """
        # Find last migration for account
        account_migrations = [
            m for m in self._migration_history
            if m.account_id == account_id
        ]

        if not account_migrations:
            logger.warning(
                "No migrations to rollback",
                account_id=account_id
            )
            return False

        last_migration = account_migrations[-1]

        if not last_migration.rollback_available:
            logger.warning(
                "Rollback not available",
                account_id=account_id,
                reason="Rollback window expired or already used"
            )
            return False

        # Reverse the migration
        reverse_plan = MigrationPlan(
            account_id=account_id,
            old_tier=last_migration.new_tier,
            new_tier=last_migration.old_tier,
            strategies_to_disable=last_migration.strategies_to_enable,
            strategies_to_enable=last_migration.strategies_to_disable,
            strategies_to_adjust=last_migration.strategies_to_adjust,
            migration_timestamp=datetime.utcnow(),
            rollback_available=False
        )

        await self._execute_migration(account_id, reverse_plan)

        # Mark original as rolled back
        last_migration.rollback_available = False

        logger.info(
            "Strategy migration rolled back",
            account_id=account_id,
            from_tier=last_migration.new_tier,
            to_tier=last_migration.old_tier
        )

        return True

    def get_strategy_config(
        self,
        account_id: str,
        strategy_name: str
    ) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy.
        
        Args:
            account_id: Account ID
            strategy_name: Strategy name
            
        Returns:
            StrategyConfig or None if not found
        """
        account_configs = self._active_configs.get(account_id, {})
        return account_configs.get(strategy_name)

    def is_strategy_enabled(
        self,
        account_id: str,
        strategy_name: str
    ) -> bool:
        """Check if a strategy is enabled for an account.
        
        Args:
            account_id: Account ID
            strategy_name: Strategy name
            
        Returns:
            True if strategy is enabled
        """
        config = self.get_strategy_config(account_id, strategy_name)
        return config is not None and config.status == StrategyStatus.ENABLED

    def get_position_multiplier(
        self,
        account_id: str,
        strategy_name: str
    ) -> Decimal:
        """Get position size multiplier for a strategy.
        
        Args:
            account_id: Account ID
            strategy_name: Strategy name
            
        Returns:
            Position multiplier (0-1)
        """
        config = self.get_strategy_config(account_id, strategy_name)
        if config:
            return config.position_multiplier
        return Decimal('0')

    def _get_tier_strategies(self, tier: str) -> list[str]:
        """Get available strategies for a tier.
        
        Args:
            tier: Tier name
            
        Returns:
            List of strategy names
        """
        strategies = TIER_STRATEGIES.get(tier, [])

        # Higher tiers include all lower tier strategies
        if tier == 'ARCHITECT':
            strategies = (
                TIER_STRATEGIES['SNIPER'] +
                TIER_STRATEGIES['HUNTER'] +
                TIER_STRATEGIES['STRATEGIST'] +
                TIER_STRATEGIES.get('ARCHITECT', [])
            )
        elif tier == 'EMPEROR':
            # All strategies available
            all_strategies = []
            for tier_strats in TIER_STRATEGIES.values():
                all_strategies.extend(tier_strats)
            strategies = list(set(all_strategies))

        return strategies

    async def _create_strategy_config(
        self,
        account_id: str,
        strategy_name: str,
        tier: str,
        adjustment_period: Optional[Any]
    ) -> StrategyConfig:
        """Create strategy configuration.
        
        Args:
            account_id: Account ID
            strategy_name: Strategy name
            tier: Current tier
            adjustment_period: Active adjustment period if any
            
        Returns:
            StrategyConfig
        """
        # Determine tier requirement
        tier_required = 'SNIPER'  # Default
        for t, strategies in TIER_STRATEGIES.items():
            if strategy_name in strategies:
                tier_required = t
                break

        # Determine position multiplier
        position_multiplier = Decimal('1.0')

        if adjustment_period:
            # During adjustment period
            position_multiplier = MIGRATION_POSITION_MULTIPLIERS['adjustment_period']
        elif strategy_name in TIER_STRATEGIES.get(tier, [])[:3]:
            # New strategy for this tier
            position_multiplier = MIGRATION_POSITION_MULTIPLIERS['new_strategy']

        # Determine max position based on tier
        max_positions = {
            'SNIPER': Decimal('500'),
            'HUNTER': Decimal('2000'),
            'STRATEGIST': Decimal('10000'),
            'ARCHITECT': Decimal('50000'),
            'EMPEROR': Decimal('250000')
        }

        max_position = max_positions.get(tier, Decimal('500'))

        return StrategyConfig(
            name=strategy_name,
            tier_required=tier_required,
            status=StrategyStatus.ENABLED,
            position_multiplier=position_multiplier,
            max_position_usdt=max_position * position_multiplier,
            enabled_at=datetime.utcnow()
        )

    async def _execute_migration(
        self,
        account_id: str,
        plan: MigrationPlan
    ) -> None:
        """Execute a strategy migration plan.
        
        Args:
            account_id: Account ID
            plan: Migration plan to execute
        """
        current_configs = self._active_configs.get(account_id, {})

        # Disable strategies
        for strategy_name in plan.strategies_to_disable:
            if strategy_name in current_configs:
                config = current_configs[strategy_name]
                config.status = StrategyStatus.DISABLED
                config.disabled_at = datetime.utcnow()
                config.migration_notes = f"Disabled during {plan.old_tier} to {plan.new_tier} migration"

        # Enable new strategies with reduced position sizes
        for strategy_name in plan.strategies_to_enable:
            config = await self._create_strategy_config(
                account_id=account_id,
                strategy_name=strategy_name,
                tier=plan.new_tier,
                adjustment_period=self._get_active_adjustment_period(account_id)
            )
            config.position_multiplier = MIGRATION_POSITION_MULTIPLIERS['new_strategy']
            config.migration_notes = f"Enabled during {plan.old_tier} to {plan.new_tier} migration"
            current_configs[strategy_name] = config

        # Adjust existing strategies
        for strategy_name in plan.strategies_to_adjust:
            if strategy_name in current_configs:
                config = current_configs[strategy_name]
                config.position_multiplier = MIGRATION_POSITION_MULTIPLIERS['existing_strategy']
                config.status = StrategyStatus.MIGRATION
                config.migration_notes = f"Adjusted during {plan.old_tier} to {plan.new_tier} migration"

        # Update active configs
        self._active_configs[account_id] = current_configs

        # Create audit trail in database
        await self._store_migration_audit(account_id, plan)

    async def _store_migration_audit(
        self,
        account_id: str,
        plan: MigrationPlan
    ) -> None:
        """Store migration audit trail.
        
        Args:
            account_id: Account ID
            plan: Migration plan
        """
        # In real implementation, would store in migrations table
        logger.debug(
            "Migration audit stored",
            account_id=account_id,
            migration_id=datetime.utcnow().isoformat()
        )

    def _get_active_adjustment_period(
        self,
        account_id: str
    ) -> Optional[AdjustmentPeriod]:
        """Get active adjustment period for account.
        
        Args:
            account_id: Account ID
            
        Returns:
            AdjustmentPeriod or None
        """
        now = datetime.utcnow()
        period = self.session.query(AdjustmentPeriod).filter(
            AdjustmentPeriod.account_id == account_id,
            AdjustmentPeriod.start_time <= now,
            AdjustmentPeriod.end_time > now
        ).first()

        return period

    def prevent_manual_override(self, account_id: str) -> bool:
        """Prevent manual override of strategy migration.
        
        Args:
            account_id: Account ID
            
        Returns:
            True if override prevented
        """
        # Check for recent migration
        recent_migrations = [
            m for m in self._migration_history
            if m.account_id == account_id and
            (datetime.utcnow() - m.migration_timestamp).total_seconds() < 3600
        ]

        if recent_migrations:
            logger.warning(
                "Manual strategy override prevented",
                account_id=account_id,
                reason="Recent automatic migration in progress"
            )
            return True

        return False
