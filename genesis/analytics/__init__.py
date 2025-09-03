"""Analytics module for performance analysis and arbitrage detection."""

from genesis.analytics.market_analyzer import (
    MarketAnalyzer,
    MarketAnalyzerConfig,
    ArbitrageOpportunity,
    OpportunityType,
    OrderBook,
    OrderBookLevel
)

from genesis.analytics.spread_tracker_enhanced import (
    EnhancedSpreadTracker,
    SpreadTrackerConfig,
    SpreadMetricsEnhanced
)

from genesis.analytics.liquidity_analyzer import (
    LiquidityAnalyzer,
    LiquidityAnalyzerConfig,
    LiquidityDepthMetrics,
    MarketImpactEstimate,
    OrderBookImbalance,
    MicrostructureAnomaly,
    LiquidityScore
)

__version__ = "1.0.0"

__all__ = [
    # MarketAnalyzer exports
    "MarketAnalyzer",
    "MarketAnalyzerConfig",
    "ArbitrageOpportunity",
    "OpportunityType",
    "OrderBook",
    "OrderBookLevel",
    
    # Enhanced Spread Tracker exports
    "EnhancedSpreadTracker",
    "SpreadTrackerConfig",
    "SpreadMetricsEnhanced",
    
    # Liquidity Analyzer exports
    "LiquidityAnalyzer",
    "LiquidityAnalyzerConfig",
    "LiquidityDepthMetrics",
    "MarketImpactEstimate",
    "OrderBookImbalance",
    "MicrostructureAnomaly",
    "LiquidityScore"
]