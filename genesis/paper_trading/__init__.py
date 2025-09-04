"""Paper Trading Module - Strategy validation with simulated execution.

This module provides paper trading capabilities for validating strategies
in live market conditions without capital risk.
"""

from genesis.paper_trading.persistence import PersistenceConfig, StatePersistence
from genesis.paper_trading.promotion_manager import (
    ABTestResult,
    StrategyPromotionManager,
)
from genesis.paper_trading.simulator import PaperTradingSimulator
from genesis.paper_trading.validation_criteria import CriteriaResult, ValidationCriteria
from genesis.paper_trading.virtual_portfolio import Trade, VirtualPortfolio

__all__ = [
    "ABTestResult",
    "CriteriaResult",
    "PaperTradingSimulator",
    "PersistenceConfig",
    "StatePersistence",
    "StrategyPromotionManager",
    "Trade",
    "ValidationCriteria",
    "VirtualPortfolio",
]
