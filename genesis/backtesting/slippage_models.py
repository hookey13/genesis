"""
Slippage Models for Realistic Market Impact Simulation

Implements various slippage models for accurate backtesting.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum

import structlog

logger = structlog.get_logger()


class SlippageType(Enum):
    """Types of slippage models."""
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    ALMGREN_CHRISS = "almgren_chriss"
    CUSTOM = "custom"


@dataclass
class MarketConditions:
    """Market conditions for slippage calculation."""
    bid_price: Decimal
    ask_price: Decimal
    bid_volume: Decimal
    ask_volume: Decimal
    average_volume: Decimal
    volatility: Decimal
    spread: Decimal
    liquidity_score: float  # 0 to 1, higher is more liquid


class BaseSlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    def __init__(self, base_slippage: Decimal = Decimal("0.001")):
        """Initialize base slippage model.
        
        Args:
            base_slippage: Base slippage percentage
        """
        self.base_slippage = base_slippage
    
    @abstractmethod
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """Calculate slippage amount.
        
        Args:
            price: Base execution price
            quantity: Order quantity
            side: "buy" or "sell"
            market_conditions: Current market conditions
            
        Returns:
            Slippage amount (positive for adverse, negative for favorable)
        """
        pass
    
    def _get_side_multiplier(self, side: str) -> int:
        """Get multiplier based on order side.
        
        Buy orders experience positive slippage (pay more),
        Sell orders experience negative slippage (receive less).
        """
        return 1 if side.lower() == "buy" else -1


class LinearSlippageModel(BaseSlippageModel):
    """Linear slippage model based on order size."""
    
    def __init__(
        self,
        base_slippage: Decimal = Decimal("0.001"),
        volume_impact_factor: Decimal = Decimal("10")
    ):
        """Initialize linear slippage model.
        
        Args:
            base_slippage: Base slippage percentage
            volume_impact_factor: Multiplier for volume-based impact
        """
        super().__init__(base_slippage)
        self.volume_impact_factor = volume_impact_factor
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """Calculate linear slippage.
        
        Slippage = base_slippage * (1 + volume_impact * quantity/avg_volume)
        """
        slippage_pct = self.base_slippage
        
        if market_conditions and market_conditions.average_volume > 0:
            # Volume impact: larger orders relative to average volume have more slippage
            volume_ratio = quantity / market_conditions.average_volume
            volume_impact = min(volume_ratio * self.volume_impact_factor, Decimal("0.05"))
            slippage_pct *= (1 + volume_impact)
            
            # Adjust for liquidity
            if market_conditions.liquidity_score < 0.5:
                liquidity_penalty = Decimal(str(1 + (0.5 - market_conditions.liquidity_score)))
                slippage_pct *= liquidity_penalty
        
        slippage = price * slippage_pct * self._get_side_multiplier(side)
        
        logger.debug(
            "linear_slippage_calculated",
            price=float(price),
            quantity=float(quantity),
            side=side,
            slippage=float(slippage),
            slippage_pct=float(slippage_pct)
        )
        
        return slippage


class SquareRootSlippageModel(BaseSlippageModel):
    """Square-root market impact model."""
    
    def __init__(
        self,
        impact_coefficient: Decimal = Decimal("0.0001"),
        daily_volume_fraction: Decimal = Decimal("0.1")
    ):
        """Initialize square-root slippage model.
        
        Args:
            impact_coefficient: Market impact coefficient
            daily_volume_fraction: Fraction of daily volume for normalization
        """
        super().__init__()
        self.impact_coefficient = impact_coefficient
        self.daily_volume_fraction = daily_volume_fraction
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """Calculate square-root market impact.
        
        Slippage = impact_coefficient * price * sqrt(quantity / ADV)
        """
        if market_conditions and market_conditions.average_volume > 0:
            # Estimate average daily volume (ADV)
            adv = market_conditions.average_volume * 1440  # Convert minute volume to daily
            
            # Square root impact
            participation_rate = quantity / (adv * self.daily_volume_fraction)
            impact = self.impact_coefficient * Decimal(str(math.sqrt(float(participation_rate))))
            
            # Volatility adjustment
            if market_conditions.volatility > 0:
                volatility_adj = 1 + market_conditions.volatility
                impact *= volatility_adj
        else:
            # Fallback to simple square root of quantity
            impact = self.impact_coefficient * Decimal(str(math.sqrt(float(quantity))))
        
        slippage = price * impact * self._get_side_multiplier(side)
        
        logger.debug(
            "sqrt_slippage_calculated",
            price=float(price),
            quantity=float(quantity),
            side=side,
            slippage=float(slippage),
            impact=float(impact)
        )
        
        return slippage


class LogarithmicSlippageModel(BaseSlippageModel):
    """Logarithmic slippage model for large orders."""
    
    def __init__(
        self,
        base_slippage: Decimal = Decimal("0.001"),
        log_coefficient: Decimal = Decimal("0.01")
    ):
        """Initialize logarithmic slippage model.
        
        Args:
            base_slippage: Base slippage percentage
            log_coefficient: Coefficient for logarithmic impact
        """
        super().__init__(base_slippage)
        self.log_coefficient = log_coefficient
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """Calculate logarithmic slippage.
        
        Slippage = base + log_coefficient * log(1 + quantity/avg_volume)
        """
        base_slippage = self.base_slippage
        
        if market_conditions and market_conditions.average_volume > 0:
            volume_ratio = quantity / market_conditions.average_volume
            log_impact = self.log_coefficient * Decimal(str(math.log(1 + float(volume_ratio))))
            total_impact = base_slippage + log_impact
            
            # Cap at reasonable maximum
            total_impact = min(total_impact, Decimal("0.1"))  # 10% max
        else:
            total_impact = base_slippage
        
        slippage = price * total_impact * self._get_side_multiplier(side)
        
        logger.debug(
            "log_slippage_calculated",
            price=float(price),
            quantity=float(quantity),
            side=side,
            slippage=float(slippage)
        )
        
        return slippage


class AlmgrenChrissModel(BaseSlippageModel):
    """Almgren-Chriss optimal execution model."""
    
    def __init__(
        self,
        permanent_impact: Decimal = Decimal("0.0001"),
        temporary_impact: Decimal = Decimal("0.001"),
        risk_aversion: Decimal = Decimal("1.0")
    ):
        """Initialize Almgren-Chriss model.
        
        Args:
            permanent_impact: Permanent market impact coefficient
            temporary_impact: Temporary market impact coefficient
            risk_aversion: Risk aversion parameter
        """
        super().__init__()
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
        self.risk_aversion = risk_aversion
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """Calculate Almgren-Chriss market impact.
        
        This model separates permanent and temporary impact components.
        """
        if market_conditions and market_conditions.average_volume > 0:
            # Participation rate
            participation = quantity / market_conditions.average_volume
            
            # Permanent impact (information leakage)
            perm_impact = self.permanent_impact * participation
            
            # Temporary impact (liquidity consumption)
            temp_impact = self.temporary_impact * Decimal(str(math.sqrt(float(participation))))
            
            # Volatility adjustment
            if market_conditions.volatility > 0:
                volatility_factor = Decimal(str(math.sqrt(float(market_conditions.volatility))))
                temp_impact *= volatility_factor
            
            # Total impact
            total_impact = perm_impact + temp_impact
            
            # Risk adjustment
            risk_adjustment = 1 + (self.risk_aversion * market_conditions.volatility)
            total_impact *= risk_adjustment
        else:
            # Fallback to simple impact
            total_impact = self.permanent_impact + self.temporary_impact
        
        slippage = price * total_impact * self._get_side_multiplier(side)
        
        logger.debug(
            "almgren_chriss_calculated",
            price=float(price),
            quantity=float(quantity),
            side=side,
            slippage=float(slippage)
        )
        
        return slippage


class AdaptiveSlippageModel(BaseSlippageModel):
    """Adaptive slippage model that adjusts based on market conditions."""
    
    def __init__(self):
        """Initialize adaptive slippage model."""
        super().__init__()
        self.models = {
            'linear': LinearSlippageModel(),
            'sqrt': SquareRootSlippageModel(),
            'log': LogarithmicSlippageModel(),
            'almgren': AlmgrenChrissModel()
        }
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market_conditions: Optional[MarketConditions] = None
    ) -> Decimal:
        """Adaptively select and apply the best slippage model.
        
        Selection based on:
        - Order size relative to average volume
        - Market liquidity
        - Current volatility
        """
        if not market_conditions:
            # Default to linear model
            return self.models['linear'].calculate(price, quantity, side, market_conditions)
        
        # Determine best model based on conditions
        if market_conditions.average_volume > 0:
            size_ratio = float(quantity / market_conditions.average_volume)
        else:
            size_ratio = 1.0
        
        # Model selection logic
        if size_ratio < 0.01:  # Small orders
            model = self.models['linear']
            model_name = 'linear'
        elif size_ratio < 0.1:  # Medium orders
            model = self.models['sqrt']
            model_name = 'sqrt'
        elif market_conditions.liquidity_score > 0.7:  # Large orders in liquid markets
            model = self.models['log']
            model_name = 'log'
        else:  # Large orders in illiquid markets
            model = self.models['almgren']
            model_name = 'almgren'
        
        slippage = model.calculate(price, quantity, side, market_conditions)
        
        logger.debug(
            "adaptive_model_selected",
            model=model_name,
            size_ratio=size_ratio,
            liquidity=market_conditions.liquidity_score if market_conditions else None
        )
        
        return slippage


class SlippageModelFactory:
    """Factory for creating slippage models."""
    
    @staticmethod
    def create(
        model_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseSlippageModel:
        """Create a slippage model.
        
        Args:
            model_type: Type of slippage model
            config: Configuration parameters
            
        Returns:
            Slippage model instance
        """
        config = config or {}
        
        if model_type == SlippageType.LINEAR.value:
            return LinearSlippageModel(**config)
        elif model_type == SlippageType.SQUARE_ROOT.value:
            return SquareRootSlippageModel(**config)
        elif model_type == SlippageType.LOGARITHMIC.value:
            return LogarithmicSlippageModel(**config)
        elif model_type == SlippageType.ALMGREN_CHRISS.value:
            return AlmgrenChrissModel(**config)
        elif model_type == "adaptive":
            return AdaptiveSlippageModel()
        else:
            logger.warning(f"Unknown slippage model: {model_type}, using linear")
            return LinearSlippageModel(**config)