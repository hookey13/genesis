"""Performance attribution analysis for trading strategies.

Analyzes and attributes performance to various factors including
market movements, timing, selection, and strategy-specific components.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AttributionResult:
    """Performance attribution analysis results."""
    
    # Total attribution
    total_return: Decimal
    market_return: Decimal
    excess_return: Decimal
    
    # Factor attribution
    timing_effect: Decimal
    selection_effect: Decimal
    interaction_effect: Decimal
    
    # Detailed attribution
    asset_attribution: Dict[str, Decimal]
    sector_attribution: Optional[Dict[str, Decimal]] = None
    strategy_attribution: Optional[Dict[str, Decimal]] = None
    
    # Time-based attribution
    daily_attribution: Optional[List[Tuple[datetime, Decimal]]] = None
    monthly_attribution: Optional[List[Tuple[datetime, Decimal]]] = None
    
    # Risk attribution
    risk_contribution: Optional[Dict[str, float]] = None
    var_contribution: Optional[Dict[str, float]] = None


class PerformanceAttributor:
    """Analyze and attribute performance to various factors."""
    
    def __init__(self, benchmark_returns: Optional[List[float]] = None):
        """Initialize performance attributor.
        
        Args:
            benchmark_returns: Optional benchmark returns for comparison
        """
        self.benchmark_returns = benchmark_returns
        logger.info("Performance attributor initialized")
    
    def attribute_performance(
        self,
        portfolio_returns: List[float],
        portfolio_weights: List[Dict[str, float]],
        asset_returns: Dict[str, List[float]],
        timestamps: List[datetime]
    ) -> AttributionResult:
        """Perform comprehensive performance attribution analysis.
        
        Args:
            portfolio_returns: List of portfolio returns
            portfolio_weights: List of weight dictionaries by period
            asset_returns: Dictionary of asset returns
            timestamps: List of timestamps for returns
            
        Returns:
            AttributionResult with detailed attribution analysis
        """
        # Calculate total and market returns
        total_return = self._calculate_total_return(portfolio_returns)
        market_return = self._calculate_market_return()
        excess_return = total_return - market_return
        
        # Brinson attribution (timing, selection, interaction)
        timing, selection, interaction = self._brinson_attribution(
            portfolio_weights, asset_returns
        )
        
        # Asset-level attribution
        asset_attribution = self._asset_level_attribution(
            portfolio_weights, asset_returns, portfolio_returns
        )
        
        # Time-based attribution
        daily_attr = self._daily_attribution(portfolio_returns, timestamps)
        monthly_attr = self._monthly_attribution(portfolio_returns, timestamps)
        
        # Risk attribution
        risk_contrib = self._risk_attribution(portfolio_weights, asset_returns)
        
        return AttributionResult(
            total_return=total_return,
            market_return=market_return,
            excess_return=excess_return,
            timing_effect=timing,
            selection_effect=selection,
            interaction_effect=interaction,
            asset_attribution=asset_attribution,
            daily_attribution=daily_attr,
            monthly_attribution=monthly_attr,
            risk_contribution=risk_contrib
        )
    
    def _calculate_total_return(self, returns: List[float]) -> Decimal:
        """Calculate total cumulative return."""
        if not returns:
            return Decimal('0')
        
        cumulative = np.prod([1 + r for r in returns]) - 1
        return Decimal(str(cumulative * 100))
    
    def _calculate_market_return(self) -> Decimal:
        """Calculate benchmark/market return."""
        if not self.benchmark_returns:
            return Decimal('0')
        
        cumulative = np.prod([1 + r for r in self.benchmark_returns]) - 1
        return Decimal(str(cumulative * 100))
    
    def _brinson_attribution(
        self,
        portfolio_weights: List[Dict[str, float]],
        asset_returns: Dict[str, List[float]]
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Perform Brinson attribution analysis.
        
        Returns:
            Tuple of (timing_effect, selection_effect, interaction_effect)
        """
        if not portfolio_weights or not asset_returns:
            return Decimal('0'), Decimal('0'), Decimal('0')
        
        # Calculate benchmark weights (equal weight for simplicity)
        assets = list(asset_returns.keys())
        benchmark_weight = 1.0 / len(assets) if assets else 0
        
        timing_effect = Decimal('0')
        selection_effect = Decimal('0')
        interaction_effect = Decimal('0')
        
        for period_idx in range(min(len(portfolio_weights), 
                                   min(len(returns) for returns in asset_returns.values()))):
            period_weights = portfolio_weights[period_idx]
            
            for asset in assets:
                if asset in period_weights and asset in asset_returns:
                    # Portfolio weight and return
                    wp = period_weights[asset]
                    rp = asset_returns[asset][period_idx] if period_idx < len(asset_returns[asset]) else 0
                    
                    # Benchmark weight and return (simplified)
                    wb = benchmark_weight
                    rb = np.mean([r[period_idx] for r in asset_returns.values() 
                                if period_idx < len(r)])
                    
                    # Attribution components
                    timing_effect += Decimal(str((wp - wb) * rb))
                    selection_effect += Decimal(str(wb * (rp - rb)))
                    interaction_effect += Decimal(str((wp - wb) * (rp - rb)))
        
        # Convert to percentage
        timing_effect *= 100
        selection_effect *= 100
        interaction_effect *= 100
        
        return timing_effect, selection_effect, interaction_effect
    
    def _asset_level_attribution(
        self,
        portfolio_weights: List[Dict[str, float]],
        asset_returns: Dict[str, List[float]],
        portfolio_returns: List[float]
    ) -> Dict[str, Decimal]:
        """Calculate contribution of each asset to total return."""
        asset_contributions = {}
        
        for asset in asset_returns:
            contribution = Decimal('0')
            
            for period_idx in range(min(len(portfolio_weights), len(asset_returns[asset]))):
                if asset in portfolio_weights[period_idx]:
                    weight = portfolio_weights[period_idx][asset]
                    ret = asset_returns[asset][period_idx]
                    contribution += Decimal(str(weight * ret))
            
            asset_contributions[asset] = contribution * 100
        
        return asset_contributions
    
    def _daily_attribution(
        self,
        returns: List[float],
        timestamps: List[datetime]
    ) -> List[Tuple[datetime, Decimal]]:
        """Calculate daily performance attribution."""
        if len(returns) != len(timestamps):
            return []
        
        daily_attr = []
        current_date = timestamps[0].date() if timestamps else None
        daily_return = 0
        
        for i, (ret, ts) in enumerate(zip(returns, timestamps)):
            if ts.date() != current_date:
                # New day - record previous day's return
                daily_attr.append((
                    datetime.combine(current_date, datetime.min.time()),
                    Decimal(str(daily_return * 100))
                ))
                current_date = ts.date()
                daily_return = ret
            else:
                # Compound within same day
                daily_return = (1 + daily_return) * (1 + ret) - 1
        
        # Add last day
        if current_date:
            daily_attr.append((
                datetime.combine(current_date, datetime.min.time()),
                Decimal(str(daily_return * 100))
            ))
        
        return daily_attr
    
    def _monthly_attribution(
        self,
        returns: List[float],
        timestamps: List[datetime]
    ) -> List[Tuple[datetime, Decimal]]:
        """Calculate monthly performance attribution."""
        if len(returns) != len(timestamps):
            return []
        
        monthly_attr = []
        current_month = (timestamps[0].year, timestamps[0].month) if timestamps else None
        monthly_return = 0
        
        for i, (ret, ts) in enumerate(zip(returns, timestamps)):
            month = (ts.year, ts.month)
            if month != current_month:
                # New month - record previous month's return
                if current_month:
                    monthly_attr.append((
                        datetime(current_month[0], current_month[1], 1),
                        Decimal(str(monthly_return * 100))
                    ))
                current_month = month
                monthly_return = ret
            else:
                # Compound within same month
                monthly_return = (1 + monthly_return) * (1 + ret) - 1
        
        # Add last month
        if current_month:
            monthly_attr.append((
                datetime(current_month[0], current_month[1], 1),
                Decimal(str(monthly_return * 100))
            ))
        
        return monthly_attr
    
    def _risk_attribution(
        self,
        portfolio_weights: List[Dict[str, float]],
        asset_returns: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate risk contribution of each asset."""
        if not portfolio_weights or not asset_returns:
            return {}
        
        # Get average weights
        avg_weights = {}
        for asset in asset_returns:
            weights = [w.get(asset, 0) for w in portfolio_weights]
            avg_weights[asset] = np.mean(weights) if weights else 0
        
        # Calculate covariance matrix
        returns_matrix = []
        assets = list(asset_returns.keys())
        
        for asset in assets:
            returns_matrix.append(asset_returns[asset])
        
        if not returns_matrix:
            return {}
        
        # Ensure all return series have same length
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = [r[:min_len] for r in returns_matrix]
        
        cov_matrix = np.cov(returns_matrix)
        
        # Calculate marginal risk contributions
        risk_contributions = {}
        portfolio_variance = 0
        
        for i, asset in enumerate(assets):
            weight = avg_weights.get(asset, 0)
            
            # Marginal contribution to variance
            marginal_var = 0
            for j, other_asset in enumerate(assets):
                other_weight = avg_weights.get(other_asset, 0)
                if i < len(cov_matrix) and j < len(cov_matrix[i]):
                    marginal_var += other_weight * cov_matrix[i][j]
            
            contribution = weight * marginal_var
            risk_contributions[asset] = float(contribution)
            portfolio_variance += contribution
        
        # Normalize to percentage contributions
        if portfolio_variance > 0:
            for asset in risk_contributions:
                risk_contributions[asset] = (risk_contributions[asset] / portfolio_variance) * 100
        
        return risk_contributions
    
    def calculate_factor_attribution(
        self,
        returns: List[float],
        factor_exposures: Dict[str, List[float]],
        factor_returns: Dict[str, List[float]]
    ) -> Dict[str, Decimal]:
        """Attribute returns to specific factors (e.g., momentum, value).
        
        Args:
            returns: Portfolio returns
            factor_exposures: Exposure to each factor over time
            factor_returns: Returns of each factor
            
        Returns:
            Dictionary of factor contributions to return
        """
        factor_contributions = {}
        
        for factor in factor_exposures:
            if factor in factor_returns:
                exposures = factor_exposures[factor]
                f_returns = factor_returns[factor]
                
                # Calculate contribution
                contribution = 0
                min_len = min(len(exposures), len(f_returns))
                
                for i in range(min_len):
                    contribution += exposures[i] * f_returns[i]
                
                factor_contributions[factor] = Decimal(str(contribution * 100))
        
        # Calculate unexplained (alpha)
        total_explained = sum(factor_contributions.values())
        total_return = self._calculate_total_return(returns)
        factor_contributions['alpha'] = total_return - total_explained
        
        return factor_contributions
    
    def calculate_strategy_attribution(
        self,
        strategy_returns: Dict[str, List[float]],
        strategy_weights: Dict[str, List[float]]
    ) -> Dict[str, Decimal]:
        """Attribute returns to different strategies.
        
        Args:
            strategy_returns: Returns by strategy
            strategy_weights: Weights allocated to each strategy
            
        Returns:
            Dictionary of strategy contributions
        """
        strategy_contributions = {}
        
        for strategy in strategy_returns:
            if strategy in strategy_weights:
                returns = strategy_returns[strategy]
                weights = strategy_weights[strategy]
                
                # Calculate weighted contribution
                contribution = 0
                min_len = min(len(returns), len(weights))
                
                for i in range(min_len):
                    contribution += weights[i] * returns[i]
                
                strategy_contributions[strategy] = Decimal(str(contribution * 100))
        
        return strategy_contributions
    
    def generate_attribution_report(self, result: AttributionResult) -> str:
        """Generate text report of attribution analysis.
        
        Args:
            result: AttributionResult to report
            
        Returns:
            Formatted text report
        """
        lines = [
            "PERFORMANCE ATTRIBUTION ANALYSIS",
            "=" * 50,
            "",
            "SUMMARY",
            "-" * 30,
            f"Total Return: {result.total_return:.2f}%",
            f"Market Return: {result.market_return:.2f}%",
            f"Excess Return: {result.excess_return:.2f}%",
            "",
            "BRINSON ATTRIBUTION",
            "-" * 30,
            f"Timing Effect: {result.timing_effect:.2f}%",
            f"Selection Effect: {result.selection_effect:.2f}%",
            f"Interaction Effect: {result.interaction_effect:.2f}%",
            ""
        ]
        
        if result.asset_attribution:
            lines.extend([
                "ASSET CONTRIBUTION",
                "-" * 30
            ])
            for asset, contribution in sorted(result.asset_attribution.items(), 
                                            key=lambda x: x[1], reverse=True):
                lines.append(f"{asset}: {contribution:.2f}%")
            lines.append("")
        
        if result.risk_contribution:
            lines.extend([
                "RISK CONTRIBUTION",
                "-" * 30
            ])
            for asset, contribution in sorted(result.risk_contribution.items(),
                                            key=lambda x: x[1], reverse=True):
                lines.append(f"{asset}: {contribution:.1f}%")
            lines.append("")
        
        return "\n".join(lines)