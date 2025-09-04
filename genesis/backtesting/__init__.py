"""
Backtesting Engine for Strategy Validation

Provides comprehensive historical replay and simulation capabilities
for validating trading strategies before live deployment.
"""

from .performance_metrics import (
    PerformanceCalculator,
    PerformanceMetrics,
    DrawdownInfo,
    TradeStatistics,
    RollingMetrics
)

__all__ = [
    'PerformanceCalculator',
    'PerformanceMetrics',
    'DrawdownInfo',
    'TradeStatistics',
    'RollingMetrics'
]