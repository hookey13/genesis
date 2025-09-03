"""Opportunity data models for arbitrage detection."""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any


class OpportunityType(Enum):
    """Types of arbitrage opportunities."""
    DIRECT = "direct"
    TRIANGULAR = "triangular"
    STATISTICAL = "statistical"


class OpportunityStatus(Enum):
    """Status of an arbitrage opportunity."""
    ACTIVE = "active"
    EXECUTED = "executed"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


@dataclass
class ExchangePair:
    """Represents a trading pair on an exchange."""
    exchange: str
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    bid_volume: Decimal
    ask_volume: Decimal
    timestamp: datetime
    fee_rate: Decimal


@dataclass
class ArbitrageOpportunity:
    """Base class for arbitrage opportunities."""
    id: str
    type: OpportunityType
    profit_pct: Decimal
    profit_amount: Decimal
    confidence_score: float
    created_at: datetime
    expires_at: datetime
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectArbitrageOpportunity:
    """Direct arbitrage between two exchanges."""
    id: str
    type: OpportunityType
    profit_pct: Decimal
    profit_amount: Decimal
    confidence_score: float
    created_at: datetime
    expires_at: datetime
    buy_exchange: str
    sell_exchange: str
    symbol: str
    buy_price: Decimal
    sell_price: Decimal
    max_volume: Decimal
    buy_fee: Decimal
    sell_fee: Decimal
    net_profit_pct: Decimal
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriangularArbitrageOpportunity:
    """Triangular arbitrage across multiple pairs."""
    id: str
    type: OpportunityType
    profit_pct: Decimal
    profit_amount: Decimal
    confidence_score: float
    created_at: datetime
    expires_at: datetime
    path: List[ExchangePair]
    exchange: str
    start_currency: str
    end_currency: str
    path_description: str
    cumulative_fees: Decimal
    execution_order: List[Dict[str, Any]]
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalArbitrageOpportunity:
    """Statistical arbitrage based on pair correlations."""
    id: str
    type: OpportunityType
    profit_pct: Decimal
    profit_amount: Decimal
    confidence_score: float
    created_at: datetime
    expires_at: datetime
    pair_a: ExchangePair
    pair_b: ExchangePair
    correlation: float
    z_score: float
    mean_spread: Decimal
    current_spread: Decimal
    std_spread: Decimal
    mean_reversion_probability: float
    historical_success_rate: float
    entry_threshold: float
    exit_threshold: float
    status: OpportunityStatus = OpportunityStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPath:
    """Optimized execution path for an opportunity."""
    opportunity_id: str
    steps: List[Dict[str, Any]]
    estimated_time_ms: int
    estimated_slippage: Decimal
    fallback_paths: List[List[Dict[str, Any]]]
    risk_score: float