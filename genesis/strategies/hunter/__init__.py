"""Hunter Tier Trading Strategies ($2k-$10k capital range).

This package contains advanced trading strategies for the Hunter tier,
supporting multi-pair concurrent execution with sophisticated risk management.
"""

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .mean_reversion import MeanReversionStrategy
    from .portfolio_manager import HunterPortfolioManager

logger = structlog.get_logger(__name__)

__all__ = [
    "HunterPortfolioManager",
    "MeanReversionStrategy",
]


def initialize_hunter_strategies():
    """Initialize Hunter tier strategies with proper logging."""
    logger.info("Hunter strategies package initialized", tier="hunter", capital_range="$2k-$10k")
