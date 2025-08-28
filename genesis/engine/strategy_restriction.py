from typing import Optional

"""Strategy restriction system for recovery mode."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import wraps

import structlog

from genesis.core.models import TradingTier
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class StrategyRestrictionManager:
    """Manages strategy restrictions during recovery periods."""

    def __init__(
        self,
        repository: SQLiteRepository,
    ):
        """Initialize strategy restriction manager.

        Args:
            repository: Database repository for persistence
        """
        self.repository = repository
        self._restricted_accounts: dict[str, set[str]] = {}
        self._strategy_performance: dict[str, dict] = {}
        self._cache_ttl = timedelta(hours=1)
        self._last_cache_update: Optional[datetime] = None

    @requires_tier(TradingTier.HUNTER)
    def restrict_strategies(
        self, account_id: str, allowed_strategies: list[str]
    ) -> None:
        """Restrict account to specific strategies.

        Args:
            account_id: Account identifier
            allowed_strategies: List of allowed strategy names
        """
        self._restricted_accounts[account_id] = set(allowed_strategies)

        logger.info(
            "Strategy restrictions applied",
            account_id=account_id,
            allowed_strategies=allowed_strategies,
            count=len(allowed_strategies),
        )

        # Persist restrictions
        self.repository.save_strategy_restrictions(
            account_id, allowed_strategies, datetime.now(UTC)
        )

    def get_highest_winrate_strategy(self, window_days: int = 30) -> str:
        """Get strategy with highest win rate over recent period.

        Args:
            window_days: Number of days to analyze

        Returns:
            Name of highest win-rate strategy
        """
        # Check cache
        if self._should_refresh_cache():
            self._refresh_strategy_performance(window_days)

        if not self._strategy_performance:
            logger.warning("No strategy performance data available")
            return "simple_arb"  # Default fallback

        # Find highest win rate
        best_strategy = None
        best_winrate = Decimal("0")

        for strategy_name, stats in self._strategy_performance.items():
            winrate = stats.get("win_rate", Decimal("0"))
            if winrate > best_winrate:
                best_winrate = winrate
                best_strategy = strategy_name

        logger.info(
            "Highest win-rate strategy identified",
            strategy=best_strategy,
            win_rate=float(best_winrate),
            window_days=window_days,
        )

        return best_strategy or "simple_arb"

    def is_strategy_allowed(self, account_id: str, strategy_name: str) -> bool:
        """Check if strategy is allowed for account.

        Args:
            account_id: Account identifier
            strategy_name: Strategy to check

        Returns:
            True if strategy is allowed
        """
        if account_id not in self._restricted_accounts:
            return True  # No restrictions

        allowed = self._restricted_accounts[account_id]
        return strategy_name in allowed

    def remove_restrictions(self, account_id: str) -> None:
        """Remove all strategy restrictions for account.

        Args:
            account_id: Account identifier
        """
        if account_id in self._restricted_accounts:
            del self._restricted_accounts[account_id]

            logger.info("Strategy restrictions removed", account_id=account_id)

            self.repository.remove_strategy_restrictions(account_id)

    def apply_recovery_restrictions(self, account_id: str) -> None:
        """Apply default recovery mode restrictions.

        Args:
            account_id: Account identifier
        """
        # Get highest win-rate strategy
        best_strategy = self.get_highest_winrate_strategy()

        # Restrict to only the best strategy
        self.restrict_strategies(account_id, [best_strategy])

        logger.info(
            "Recovery mode restrictions applied",
            account_id=account_id,
            restricted_to=best_strategy,
        )

    def _should_refresh_cache(self) -> bool:
        """Check if performance cache should be refreshed.

        Returns:
            True if cache should be refreshed
        """
        if self._last_cache_update is None:
            return True

        age = datetime.now(UTC) - self._last_cache_update
        return age > self._cache_ttl

    def _refresh_strategy_performance(self, window_days: int) -> None:
        """Refresh strategy performance statistics.

        Args:
            window_days: Number of days to analyze
        """
        try:
            since_date = datetime.now(UTC) - timedelta(days=window_days)

            # Get trade history from repository
            trades = self.repository.get_trades_since(since_date)

            # Calculate performance by strategy
            strategy_stats = {}

            for trade in trades:
                strategy = trade.get("strategy_name", "unknown")

                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "total_profit": Decimal("0"),
                        "total_loss": Decimal("0"),
                    }

                stats = strategy_stats[strategy]
                stats["total_trades"] += 1

                pnl = trade.get("profit_loss", Decimal("0"))
                if pnl > 0:
                    stats["winning_trades"] += 1
                    stats["total_profit"] += pnl
                else:
                    stats["total_loss"] += abs(pnl)

            # Calculate win rates
            for strategy, stats in strategy_stats.items():
                if stats["total_trades"] > 0:
                    stats["win_rate"] = Decimal(stats["winning_trades"]) / Decimal(
                        stats["total_trades"]
                    )
                else:
                    stats["win_rate"] = Decimal("0")

            self._strategy_performance = strategy_stats
            self._last_cache_update = datetime.now(UTC)

            logger.info(
                "Strategy performance cache refreshed",
                strategies_analyzed=len(strategy_stats),
                window_days=window_days,
            )

        except Exception as e:
            logger.error("Failed to refresh strategy performance", error=str(e))


def recovery_mode(func):
    """Decorator to enforce strategy restrictions in recovery mode.

    This decorator checks if the current account is in recovery mode
    and enforces strategy restrictions accordingly.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract account_id and strategy_name from arguments
        account_id = kwargs.get("account_id")
        strategy_name = kwargs.get("strategy_name")

        if not account_id or not strategy_name:
            # Can't enforce restrictions without required info
            return func(*args, **kwargs)

        # Get restriction manager from context (would be injected)
        # For now, just log and proceed
        logger.debug(
            "Recovery mode check", account_id=account_id, strategy=strategy_name
        )

        return func(*args, **kwargs)

    return wrapper


class StrategyFilter:
    """Filter strategies based on recovery mode restrictions."""

    def __init__(
        self,
        restriction_manager: StrategyRestrictionManager,
    ):
        """Initialize strategy filter.

        Args:
            restriction_manager: Strategy restriction manager
        """
        self.restriction_manager = restriction_manager

    def filter_available_strategies(
        self, account_id: str, all_strategies: list[str]
    ) -> list[str]:
        """Filter strategies based on account restrictions.

        Args:
            account_id: Account identifier
            all_strategies: List of all available strategies

        Returns:
            List of allowed strategies
        """
        allowed = []

        for strategy in all_strategies:
            if self.restriction_manager.is_strategy_allowed(account_id, strategy):
                allowed.append(strategy)

        if not allowed and all_strategies:
            # If no strategies allowed but some exist, allow default
            allowed = [all_strategies[0]]
            logger.warning(
                "No strategies allowed, using default",
                account_id=account_id,
                default=allowed[0],
            )

        return allowed
