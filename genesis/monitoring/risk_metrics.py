"""Risk metrics calculation for strategy monitoring."""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    
    var_95: Decimal  # Value at Risk at 95% confidence
    var_99: Decimal  # Value at Risk at 99% confidence
    cvar_95: Decimal  # Conditional Value at Risk at 95%
    cvar_99: Decimal  # Conditional Value at Risk at 99%
    beta: Decimal  # Market beta
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    volatility: Decimal  # Annualized volatility
    downside_deviation: Decimal
    correlation_matrix: Dict[str, Dict[str, Decimal]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RiskMetricsCalculator:
    """Calculate risk metrics for trading strategies."""
    
    def __init__(self, risk_free_rate: Decimal = Decimal("0.02")):
        """Initialize risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
        self.returns_history: Dict[str, deque] = {}
        self.benchmark_returns: deque = deque(maxlen=1000)
        self.price_history: Dict[str, deque] = {}
        
    def add_return(self, strategy_id: str, return_value: Decimal) -> None:
        """Add a return observation for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            return_value: Return value (as decimal, e.g., 0.01 for 1%)
        """
        if strategy_id not in self.returns_history:
            self.returns_history[strategy_id] = deque(maxlen=1000)
        self.returns_history[strategy_id].append(float(return_value))
        
    def add_benchmark_return(self, return_value: Decimal) -> None:
        """Add a benchmark return observation.
        
        Args:
            return_value: Benchmark return value
        """
        self.benchmark_returns.append(float(return_value))
        
    def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> Decimal:
        """Calculate Value at Risk.
        
        Args:
            returns: List of returns
            confidence_level: Confidence level (0.95 for 95%)
            
        Returns:
            VaR at specified confidence level
        """
        if not returns:
            return Decimal("0")
            
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
        elif index < 0:
            index = 0
            
        return Decimal(str(-sorted_returns[index]))
        
    def calculate_cvar(self, returns: List[float], confidence_level: float = 0.95) -> Decimal:
        """Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: List of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR at specified confidence level
        """
        if not returns:
            return Decimal("0")
            
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        
        if var_index == 0:
            return Decimal(str(-sorted_returns[0]))
            
        tail_returns = sorted_returns[:var_index]
        if not tail_returns:
            return Decimal("0")
            
        return Decimal(str(-np.mean(tail_returns)))
        
    def calculate_beta(self, strategy_returns: List[float], benchmark_returns: List[float]) -> Decimal:
        """Calculate beta relative to benchmark.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Beta coefficient
        """
        if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
            return Decimal("1")
            
        # Align lengths
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        # Calculate covariance and variance
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return Decimal("1")
            
        return Decimal(str(covariance / benchmark_variance))
        
    def calculate_sharpe_ratio(self, returns: List[float], periods_per_year: int = 252) -> Decimal:
        """Calculate Sharpe ratio.
        
        Args:
            returns: List of returns
            periods_per_year: Number of trading periods per year
            
        Returns:
            Annualized Sharpe ratio
        """
        if not returns or len(returns) < 2:
            return Decimal("0")
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return Decimal("0")
            
        # Annualize
        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)
        
        # Risk-free rate per period
        rf_per_period = float(self.risk_free_rate) / periods_per_year
        
        sharpe = (annual_return - float(self.risk_free_rate)) / annual_std
        return Decimal(str(sharpe))
        
    def calculate_sortino_ratio(self, returns: List[float], periods_per_year: int = 252) -> Decimal:
        """Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: List of returns
            periods_per_year: Number of trading periods per year
            
        Returns:
            Annualized Sortino ratio
        """
        if not returns or len(returns) < 2:
            return Decimal("0")
            
        mean_return = np.mean(returns)
        
        # Calculate downside deviation
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return Decimal("999")  # No downside risk
            
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return Decimal("999")
            
        # Annualize
        annual_return = mean_return * periods_per_year
        annual_downside_std = downside_std * np.sqrt(periods_per_year)
        
        sortino = (annual_return - float(self.risk_free_rate)) / annual_downside_std
        return Decimal(str(sortino))
        
    def calculate_volatility(self, returns: List[float], periods_per_year: int = 252) -> Decimal:
        """Calculate annualized volatility.
        
        Args:
            returns: List of returns
            periods_per_year: Number of trading periods per year
            
        Returns:
            Annualized volatility
        """
        if not returns or len(returns) < 2:
            return Decimal("0")
            
        std_return = np.std(returns)
        annual_vol = std_return * np.sqrt(periods_per_year)
        return Decimal(str(annual_vol))
        
    def calculate_downside_deviation(self, returns: List[float], threshold: float = 0, periods_per_year: int = 252) -> Decimal:
        """Calculate downside deviation.
        
        Args:
            returns: List of returns
            threshold: Threshold for downside (usually 0 or risk-free rate)
            periods_per_year: Number of trading periods per year
            
        Returns:
            Annualized downside deviation
        """
        if not returns:
            return Decimal("0")
            
        downside_returns = [min(0, r - threshold) for r in returns]
        downside_std = np.sqrt(np.mean([r**2 for r in downside_returns]))
        annual_downside = downside_std * np.sqrt(periods_per_year)
        return Decimal(str(annual_downside))
        
    def calculate_correlation_matrix(self, strategies: Dict[str, List[float]]) -> Dict[str, Dict[str, Decimal]]:
        """Calculate correlation matrix between strategies.
        
        Args:
            strategies: Dictionary of strategy_id to returns list
            
        Returns:
            Correlation matrix as nested dictionary
        """
        correlation_matrix = {}
        strategy_ids = list(strategies.keys())
        
        if len(strategy_ids) < 2:
            # Single strategy, perfect correlation with itself
            if strategy_ids:
                correlation_matrix[strategy_ids[0]] = {strategy_ids[0]: Decimal("1.0")}
            return correlation_matrix
            
        # Create returns matrix
        returns_matrix = []
        valid_strategies = []
        
        for strategy_id in strategy_ids:
            returns = strategies[strategy_id]
            if len(returns) >= 2:
                returns_matrix.append(returns)
                valid_strategies.append(strategy_id)
                
        if len(valid_strategies) < 2:
            # Not enough data for correlation
            for strategy_id in valid_strategies:
                correlation_matrix[strategy_id] = {strategy_id: Decimal("1.0")}
            return correlation_matrix
            
        # Align lengths
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_len:] for r in returns_matrix]
        
        # Calculate correlation
        corr_array = np.corrcoef(returns_matrix)
        
        # Convert to dictionary format
        for i, strategy_i in enumerate(valid_strategies):
            correlation_matrix[strategy_i] = {}
            for j, strategy_j in enumerate(valid_strategies):
                if i < len(corr_array) and j < len(corr_array[i]):
                    correlation_matrix[strategy_i][strategy_j] = Decimal(str(corr_array[i, j]))
                else:
                    correlation_matrix[strategy_i][strategy_j] = Decimal("0")
                    
        return correlation_matrix
        
    def calculate_max_drawdown(self, prices: List[float]) -> Tuple[Decimal, Decimal]:
        """Calculate maximum drawdown from price series.
        
        Args:
            prices: List of prices or cumulative returns
            
        Returns:
            Tuple of (max_drawdown, current_drawdown)
        """
        if not prices:
            return Decimal("0"), Decimal("0")
            
        peak = prices[0]
        max_dd = 0
        current_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd
            current_dd = dd
            
        return Decimal(str(max_dd)), Decimal(str(current_dd))
        
    def calculate_risk_metrics(self, strategy_id: str) -> Optional[RiskMetrics]:
        """Calculate all risk metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            RiskMetrics object or None if insufficient data
        """
        if strategy_id not in self.returns_history:
            return None
            
        returns = list(self.returns_history[strategy_id])
        
        if len(returns) < 10:  # Need minimum data for meaningful metrics
            logger.warning("insufficient_data_for_risk_metrics", 
                         strategy_id=strategy_id, 
                         data_points=len(returns))
            return None
            
        # Calculate VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Calculate beta if benchmark data available
        beta = Decimal("1")
        if self.benchmark_returns:
            benchmark = list(self.benchmark_returns)
            beta = self.calculate_beta(returns, benchmark)
            
        # Calculate Sharpe and Sortino ratios
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        
        # Calculate volatility
        volatility = self.calculate_volatility(returns)
        downside_deviation = self.calculate_downside_deviation(returns)
        
        # Calculate drawdown if price history available
        max_drawdown = Decimal("0")
        current_drawdown = Decimal("0")
        if strategy_id in self.price_history:
            prices = list(self.price_history[strategy_id])
            if prices:
                max_drawdown, current_drawdown = self.calculate_max_drawdown(prices)
                
        # Calculate correlation with other strategies
        all_strategies = {sid: list(ret_hist) 
                         for sid, ret_hist in self.returns_history.items()
                         if len(ret_hist) >= 10}
        correlation_matrix = self.calculate_correlation_matrix(all_strategies)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            downside_deviation=downside_deviation,
            correlation_matrix=correlation_matrix.get(strategy_id, {}),
            timestamp=datetime.utcnow()
        )
        
    def update_price_series(self, strategy_id: str, price: Decimal) -> None:
        """Update price series for drawdown calculation.
        
        Args:
            strategy_id: Strategy identifier
            price: Current portfolio value or cumulative return
        """
        if strategy_id not in self.price_history:
            self.price_history[strategy_id] = deque(maxlen=1000)
        self.price_history[strategy_id].append(float(price))
        
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary across all strategies.
        
        Returns:
            Dictionary with aggregate risk metrics
        """
        all_metrics = {}
        for strategy_id in self.returns_history:
            metrics = self.calculate_risk_metrics(strategy_id)
            if metrics:
                all_metrics[strategy_id] = metrics
                
        if not all_metrics:
            return {"strategies": 0, "message": "No risk metrics available"}
            
        # Calculate aggregate metrics
        avg_var_95 = sum(m.var_95 for m in all_metrics.values()) / len(all_metrics)
        avg_sharpe = sum(m.sharpe_ratio for m in all_metrics.values()) / len(all_metrics)
        max_drawdown = max(m.max_drawdown for m in all_metrics.values())
        
        return {
            "strategies": len(all_metrics),
            "average_var_95": float(avg_var_95),
            "average_sharpe_ratio": float(avg_sharpe),
            "max_drawdown_across_strategies": float(max_drawdown),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def check_risk_limits(self, strategy_id: str, limits: Dict[str, Decimal]) -> List[str]:
        """Check if strategy exceeds risk limits.
        
        Args:
            strategy_id: Strategy identifier
            limits: Dictionary of risk limits
            
        Returns:
            List of limit violations
        """
        violations = []
        metrics = self.calculate_risk_metrics(strategy_id)
        
        if not metrics:
            return ["Insufficient data for risk assessment"]
            
        # Check each limit
        if "max_var_95" in limits and metrics.var_95 > limits["max_var_95"]:
            violations.append(f"VaR 95% exceeds limit: {metrics.var_95:.2f}% > {limits['max_var_95']:.2f}%")
            
        if "max_drawdown" in limits and metrics.max_drawdown > limits["max_drawdown"]:
            violations.append(f"Max drawdown exceeds limit: {metrics.max_drawdown:.2f}% > {limits['max_drawdown']:.2f}%")
            
        if "min_sharpe" in limits and metrics.sharpe_ratio < limits["min_sharpe"]:
            violations.append(f"Sharpe ratio below minimum: {metrics.sharpe_ratio:.2f} < {limits['min_sharpe']:.2f}")
            
        if "max_volatility" in limits and metrics.volatility > limits["max_volatility"]:
            violations.append(f"Volatility exceeds limit: {metrics.volatility:.2f}% > {limits['max_volatility']:.2f}%")
            
        return violations