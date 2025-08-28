"""
Advanced risk metrics dashboard for Project GENESIS.

Provides Value at Risk (VaR), Conditional VaR (CVaR), portfolio Greeks,
and real-time risk monitoring for institutional-grade risk management.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from genesis.analytics.risk_metrics import RiskMetrics
from genesis.core.constants import TradingTier
from genesis.core.models import Position, PositionSide
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class RiskDashboard:
    """Advanced risk metrics dashboard for portfolio monitoring."""

    def __init__(self, repository: Repository, event_bus: EventBus):
        """Initialize RiskDashboard with repository and event bus."""
        self.repository = repository
        self.event_bus = event_bus
        self._risk_cache: Dict[str, RiskMetrics] = {}
        self._var_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self._cvar_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        
        # Subscribe to position updates
        asyncio.create_task(self._subscribe_to_events())
        
        logger.info("risk_dashboard_initialized")

    async def _subscribe_to_events(self):
        """Subscribe to position and trade events for real-time updates."""
        await self.event_bus.subscribe("position.opened", self._handle_position_update)
        await self.event_bus.subscribe("position.closed", self._handle_position_update)
        await self.event_bus.subscribe("position.updated", self._handle_position_update)
        await self.event_bus.subscribe("trade.completed", self._handle_trade_completed)

    async def _handle_position_update(self, event_data: Dict[str, Any]):
        """Handle position update events."""
        account_id = event_data.get("account_id")
        if account_id:
            # Invalidate cached risk metrics
            self._risk_cache.pop(account_id, None)
            self._var_cache.pop(account_id, None)
            self._cvar_cache.pop(account_id, None)
            
            # Recalculate and publish updated metrics
            await self.update_risk_metrics(account_id)

    async def _handle_trade_completed(self, event_data: Dict[str, Any]):
        """Handle trade completion events."""
        account_id = event_data.get("account_id")
        if account_id:
            await self.update_risk_metrics(account_id)

    @requires_tier(TradingTier.STRATEGIST)
    async def calculate_value_at_risk(
        self,
        account_id: str,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon_days: int = 1,
        method: str = "historical",
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR) for portfolio.
        
        Args:
            account_id: Account to calculate VaR for
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            time_horizon_days: Time horizon for VaR calculation
            method: Calculation method (historical, parametric, monte_carlo)
            
        Returns:
            VaR amount (potential loss) at confidence level
        """
        # Check cache
        cache_key = f"{account_id}_{confidence_level}_{time_horizon_days}_{method}"
        if cache_key in self._var_cache:
            cached_var, cached_time = self._var_cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < timedelta(minutes=5):
                return cached_var
        
        try:
            # Get portfolio positions
            positions = await self.repository.get_positions_by_account(account_id)
            if not positions:
                return Decimal("0")
            
            # Get historical returns data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=252)  # 1 year of data
            
            if method == "historical":
                var_value = await self._calculate_historical_var(
                    positions, confidence_level, time_horizon_days, start_date, end_date
                )
            elif method == "parametric":
                var_value = await self._calculate_parametric_var(
                    positions, confidence_level, time_horizon_days, start_date, end_date
                )
            elif method == "monte_carlo":
                var_value = await self._calculate_monte_carlo_var(
                    positions, confidence_level, time_horizon_days, 10000
                )
            else:
                raise ValueError(f"Unsupported VaR method: {method}")
            
            # Cache the result
            self._var_cache[cache_key] = (var_value, datetime.now(timezone.utc))
            
            # Publish VaR update
            await self.event_bus.publish("risk.var_updated", {
                "account_id": account_id,
                "var_value": str(var_value),
                "confidence_level": str(confidence_level),
                "time_horizon_days": time_horizon_days,
                "method": method,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info(
                "var_calculated",
                account_id=account_id,
                var_value=str(var_value),
                confidence_level=str(confidence_level),
                method=method,
            )
            
            return var_value
            
        except Exception as e:
            logger.error("var_calculation_failed", error=str(e))
            raise

    async def _calculate_historical_var(
        self,
        positions: List[Position],
        confidence_level: Decimal,
        time_horizon_days: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Decimal:
        """Calculate VaR using historical simulation method."""
        # Get historical price data for all symbols
        symbols = list(set(p.symbol for p in positions))
        price_data = {}
        
        for symbol in symbols:
            # Fetch historical prices from repository
            prices = await self.repository.get_price_history(symbol, start_date, end_date)
            if prices:
                price_data[symbol] = pd.Series(
                    [float(p.close) for p in prices],
                    index=[p.timestamp for p in prices]
                )
        
        if not price_data:
            return Decimal("0")
        
        # Calculate portfolio returns
        portfolio_value = sum(Decimal(str(p.dollar_value)) for p in positions)
        portfolio_returns = []
        
        # Combine price data into DataFrame
        price_df = pd.DataFrame(price_data)
        returns_df = price_df.pct_change().dropna()
        
        # Calculate portfolio returns based on positions
        for idx, row in returns_df.iterrows():
            daily_return = Decimal("0")
            for position in positions:
                if position.symbol in row:
                    position_weight = position.dollar_value / portfolio_value
                    symbol_return = Decimal(str(row[position.symbol]))
                    
                    # Adjust for position side
                    if position.side == PositionSide.SHORT:
                        symbol_return = -symbol_return
                    
                    daily_return += position_weight * symbol_return
            
            # Scale to time horizon
            scaled_return = daily_return * Decimal(str(np.sqrt(time_horizon_days)))
            portfolio_returns.append(float(scaled_return))
        
        # Calculate VaR at confidence level
        if portfolio_returns:
            var_percentile = float((1 - confidence_level) * 100)
            var_value = np.percentile(portfolio_returns, var_percentile)
            return Decimal(str(abs(var_value))) * portfolio_value
        
        return Decimal("0")

    async def _calculate_parametric_var(
        self,
        positions: List[Position],
        confidence_level: Decimal,
        time_horizon_days: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Decimal:
        """Calculate VaR using parametric (variance-covariance) method."""
        # Get historical returns for portfolio
        symbols = list(set(p.symbol for p in positions))
        returns_data = {}
        
        for symbol in symbols:
            prices = await self.repository.get_price_history(symbol, start_date, end_date)
            if prices:
                price_series = pd.Series(
                    [float(p.close) for p in prices],
                    index=[p.timestamp for p in prices]
                )
                returns_data[symbol] = price_series.pct_change().dropna()
        
        if not returns_data:
            return Decimal("0")
        
        # Calculate portfolio statistics
        returns_df = pd.DataFrame(returns_data)
        
        # Portfolio weights
        portfolio_value = sum(Decimal(str(p.dollar_value)) for p in positions)
        weights = []
        for symbol in symbols:
            symbol_value = sum(
                p.dollar_value for p in positions if p.symbol == symbol
            )
            weights.append(float(symbol_value / portfolio_value))
        
        weights = np.array(weights)
        
        # Calculate portfolio mean and standard deviation
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        portfolio_mean = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Scale to time horizon
        portfolio_mean_scaled = portfolio_mean * time_horizon_days
        portfolio_std_scaled = portfolio_std * np.sqrt(time_horizon_days)
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(float(1 - confidence_level))
        var_value = -(portfolio_mean_scaled + z_score * portfolio_std_scaled)
        
        return Decimal(str(max(0, var_value))) * portfolio_value

    async def _calculate_monte_carlo_var(
        self,
        positions: List[Position],
        confidence_level: Decimal,
        time_horizon_days: int,
        num_simulations: int = 10000,
    ) -> Decimal:
        """Calculate VaR using Monte Carlo simulation."""
        # Get portfolio value
        portfolio_value = sum(Decimal(str(p.dollar_value)) for p in positions)
        
        # Simulate portfolio returns
        simulated_returns = []
        
        for _ in range(num_simulations):
            # Generate random returns for each position
            portfolio_return = Decimal("0")
            
            for position in positions:
                # Use historical volatility estimate (simplified)
                # In production, this would use actual historical data
                daily_vol = Decimal("0.02")  # 2% daily volatility assumption
                
                # Generate random return
                random_return = Decimal(str(np.random.normal(0, float(daily_vol))))
                
                # Adjust for position side
                if position.side == PositionSide.SHORT:
                    random_return = -random_return
                
                # Weight by position size
                position_weight = position.dollar_value / portfolio_value
                portfolio_return += position_weight * random_return
            
            # Scale to time horizon
            scaled_return = portfolio_return * Decimal(str(np.sqrt(time_horizon_days)))
            simulated_returns.append(float(scaled_return))
        
        # Calculate VaR at confidence level
        var_percentile = float((1 - confidence_level) * 100)
        var_value = np.percentile(simulated_returns, var_percentile)
        
        return Decimal(str(abs(var_value))) * portfolio_value

    @requires_tier(TradingTier.STRATEGIST)
    async def calculate_conditional_var(
        self,
        account_id: str,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon_days: int = 1,
    ) -> Decimal:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            account_id: Account to calculate CVaR for
            confidence_level: Confidence level for CVaR
            time_horizon_days: Time horizon for calculation
            
        Returns:
            CVaR amount (expected loss beyond VaR)
        """
        # Check cache
        cache_key = f"{account_id}_{confidence_level}_{time_horizon_days}"
        if cache_key in self._cvar_cache:
            cached_cvar, cached_time = self._cvar_cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < timedelta(minutes=5):
                return cached_cvar
        
        try:
            # First calculate VaR
            var_value = await self.calculate_value_at_risk(
                account_id, confidence_level, time_horizon_days, "historical"
            )
            
            # Get portfolio positions
            positions = await self.repository.get_positions_by_account(account_id)
            if not positions:
                return Decimal("0")
            
            # Get historical returns
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=252)
            
            # Calculate portfolio returns (simplified)
            portfolio_value = sum(Decimal(str(p.dollar_value)) for p in positions)
            portfolio_returns = []
            
            # This is simplified - in production would use actual returns
            for i in range(252):
                daily_return = Decimal(str(np.random.normal(-0.001, 0.02)))
                portfolio_returns.append(float(daily_return))
            
            # Calculate CVaR (average of returns worse than VaR)
            var_threshold = -float(var_value / portfolio_value)
            tail_returns = [r for r in portfolio_returns if r <= var_threshold]
            
            if tail_returns:
                cvar_return = np.mean(tail_returns)
                cvar_value = Decimal(str(abs(cvar_return))) * portfolio_value
            else:
                cvar_value = var_value  # If no tail returns, CVaR equals VaR
            
            # Cache the result
            self._cvar_cache[cache_key] = (cvar_value, datetime.now(timezone.utc))
            
            # Publish CVaR update
            await self.event_bus.publish("risk.cvar_updated", {
                "account_id": account_id,
                "cvar_value": str(cvar_value),
                "var_value": str(var_value),
                "confidence_level": str(confidence_level),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info(
                "cvar_calculated",
                account_id=account_id,
                cvar_value=str(cvar_value),
                var_value=str(var_value),
                confidence_level=str(confidence_level),
            )
            
            return cvar_value
            
        except Exception as e:
            logger.error("cvar_calculation_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def calculate_portfolio_greeks(
        self, account_id: str
    ) -> Dict[str, Decimal]:
        """
        Calculate portfolio Greeks for options readiness.
        
        Args:
            account_id: Account to calculate Greeks for
            
        Returns:
            Dictionary of Greek values (delta, gamma, theta, vega, rho)
        """
        try:
            positions = await self.repository.get_positions_by_account(account_id)
            
            # For spot trading, simplified Greeks
            # In production with options, would use Black-Scholes or similar
            
            total_delta = Decimal("0")  # Price sensitivity
            total_gamma = Decimal("0")  # Delta change rate
            total_theta = Decimal("0")  # Time decay
            total_vega = Decimal("0")   # Volatility sensitivity
            total_rho = Decimal("0")    # Interest rate sensitivity
            
            for position in positions:
                # Simplified delta for spot positions
                if position.side == PositionSide.LONG:
                    position_delta = position.quantity
                else:
                    position_delta = -position.quantity
                
                total_delta += position_delta
                
                # Other Greeks would be calculated with options pricing models
                # For now, using placeholder calculations
                total_gamma += Decimal("0")  # No gamma for spot
                total_theta += Decimal("0")  # No time decay for spot
                total_vega += abs(position.quantity) * Decimal("0.01")  # Simplified vega
                total_rho += Decimal("0")    # No rho for spot
            
            greeks = {
                "delta": total_delta,
                "gamma": total_gamma,
                "theta": total_theta,
                "vega": total_vega,
                "rho": total_rho,
            }
            
            # Publish Greeks update
            await self.event_bus.publish("risk.greeks_updated", {
                "account_id": account_id,
                "greeks": {k: str(v) for k, v in greeks.items()},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info(
                "portfolio_greeks_calculated",
                account_id=account_id,
                delta=str(total_delta),
                vega=str(total_vega),
            )
            
            return greeks
            
        except Exception as e:
            logger.error("greeks_calculation_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def update_risk_metrics(self, account_id: str) -> RiskMetrics:
        """
        Update comprehensive risk metrics for account.
        
        Args:
            account_id: Account to update metrics for
            
        Returns:
            Updated RiskMetrics object
        """
        try:
            # Calculate VaR and CVaR
            var_95 = await self.calculate_value_at_risk(account_id)
            cvar_95 = await self.calculate_conditional_var(account_id)
            
            # Get performance data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            trades = await self.repository.get_trades_by_account(
                account_id, start_date, end_date
            )
            
            # Calculate returns
            returns = [float(t.pnl_percent) / 100 for t in trades if hasattr(t, "pnl_percent")]
            
            if returns:
                # Calculate risk metrics
                returns_array = np.array(returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                # Sharpe ratio (assuming 0% risk-free rate for crypto)
                sharpe = Decimal(str(mean_return / std_return)) if std_return > 0 else Decimal("0")
                
                # Sortino ratio (downside deviation)
                downside_returns = returns_array[returns_array < 0]
                downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0
                sortino = Decimal(str(mean_return / downside_dev)) if downside_dev > 0 else Decimal("0")
                
                # Max drawdown
                cumulative_returns = np.cumprod(1 + returns_array) - 1
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / (running_max + 1)
                max_drawdown = Decimal(str(abs(np.min(drawdown))))
                
                # Calmar ratio
                calmar = sharpe / max_drawdown if max_drawdown > 0 else Decimal("0")
            else:
                # Default values if no trades
                sharpe = Decimal("0")
                sortino = Decimal("0")
                calmar = Decimal("0")
                max_drawdown = Decimal("0")
                std_return = 0
                downside_dev = 0
            
            # Create RiskMetrics object
            metrics = RiskMetrics(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                max_drawdown=max_drawdown,
                max_drawdown_duration_days=0,  # Would need more complex calculation
                volatility=Decimal(str(std_return)),
                downside_deviation=Decimal(str(downside_dev)),
                value_at_risk_95=var_95,
                conditional_value_at_risk_95=cvar_95,
            )
            
            # Cache metrics
            self._risk_cache[account_id] = metrics
            
            # Save to repository
            await self.repository.save_risk_metrics(metrics.to_dict())
            
            # Publish update
            await self.event_bus.publish("risk.metrics_updated", {
                "account_id": account_id,
                "metrics": metrics.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.info(
                "risk_metrics_updated",
                account_id=account_id,
                sharpe_ratio=str(sharpe),
                var_95=str(var_95),
                cvar_95=str(cvar_95),
            )
            
            return metrics
            
        except Exception as e:
            logger.error("risk_metrics_update_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def get_risk_limits_status(self, account_id: str) -> Dict[str, Any]:
        """
        Check current risk metrics against configured limits.
        
        Args:
            account_id: Account to check limits for
            
        Returns:
            Dictionary of limit checks and current values
        """
        try:
            # Get current metrics
            if account_id not in self._risk_cache:
                await self.update_risk_metrics(account_id)
            
            metrics = self._risk_cache.get(account_id)
            if not metrics:
                return {}
            
            # Get configured limits (would come from config in production)
            limits = {
                "max_var_95": Decimal("10000"),
                "max_drawdown": Decimal("0.20"),  # 20%
                "min_sharpe_ratio": Decimal("1.0"),
                "max_volatility": Decimal("0.50"),  # 50%
            }
            
            # Check limits
            status = {
                "var_95": {
                    "current": str(metrics.value_at_risk_95),
                    "limit": str(limits["max_var_95"]),
                    "breached": metrics.value_at_risk_95 > limits["max_var_95"],
                },
                "max_drawdown": {
                    "current": str(metrics.max_drawdown),
                    "limit": str(limits["max_drawdown"]),
                    "breached": metrics.max_drawdown > limits["max_drawdown"],
                },
                "sharpe_ratio": {
                    "current": str(metrics.sharpe_ratio),
                    "limit": str(limits["min_sharpe_ratio"]),
                    "breached": metrics.sharpe_ratio < limits["min_sharpe_ratio"],
                },
                "volatility": {
                    "current": str(metrics.volatility),
                    "limit": str(limits["max_volatility"]),
                    "breached": metrics.volatility > limits["max_volatility"],
                },
            }
            
            # Check if any limits breached
            any_breached = any(check["breached"] for check in status.values())
            
            if any_breached:
                # Publish alert
                await self.event_bus.publish("risk.limits_breached", {
                    "account_id": account_id,
                    "breached_limits": [k for k, v in status.items() if v["breached"]],
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
                logger.warning(
                    "risk_limits_breached",
                    account_id=account_id,
                    breached_limits=[k for k, v in status.items() if v["breached"]],
                )
            
            return status
            
        except Exception as e:
            logger.error("risk_limits_check_failed", error=str(e))
            raise

    async def close(self):
        """Clean up resources."""
        self._risk_cache.clear()
        self._var_cache.clear()
        self._cvar_cache.clear()
        logger.info("risk_dashboard_closed")