"""Portfolio Manager for Hunter Tier Strategies.

This module manages multi-pair portfolio tracking, correlation-based risk limits,
and position allocation for Hunter tier strategies.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

from genesis.core.models import Order, Position, Signal, SignalType

logger = structlog.get_logger(__name__)


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio management."""
    
    max_concurrent_positions: int = 5
    max_correlation: Decimal = Decimal("0.7")
    max_position_percent: Decimal = Decimal("0.05")  # 5% per position
    max_portfolio_exposure: Decimal = Decimal("0.25")  # 25% total exposure
    min_position_value: Decimal = Decimal("100")  # Minimum $100 per position
    max_position_value: Decimal = Decimal("5000")  # Maximum $5000 per position


@dataclass
class PortfolioMetrics:
    """Real-time portfolio metrics."""
    
    total_exposure: Decimal = Decimal("0")
    position_count: int = 0
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    correlation_matrix: pd.DataFrame | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PositionAllocation:
    """Position allocation recommendation."""
    
    symbol: str
    recommended_size: Decimal
    risk_adjusted_size: Decimal
    correlation_adjustment: Decimal
    final_size: Decimal
    allocation_percent: Decimal
    risk_score: Decimal
    metadata: dict[str, Any] = field(default_factory=dict)


class HunterPortfolioManager:
    """Portfolio manager for Hunter tier strategies."""
    
    def __init__(
        self,
        account_balance: Decimal,
        constraints: PortfolioConstraints | None = None
    ):
        """Initialize the portfolio manager.
        
        Args:
            account_balance: Total account balance
            constraints: Portfolio constraints
        """
        self.account_balance = account_balance
        self.constraints = constraints or PortfolioConstraints()
        
        # Portfolio tracking
        self.positions: dict[str, Position] = {}
        self.pending_signals: list[Signal] = []
        self.historical_returns: dict[str, list[Decimal]] = {}
        self.correlation_cache: dict[str, pd.DataFrame] = {}
        self.cache_timestamp: datetime | None = None
        
        # Metrics
        self.metrics = PortfolioMetrics()
        
        logger.info(
            "HunterPortfolioManager initialized",
            account_balance=float(account_balance),
            max_positions=self.constraints.max_concurrent_positions
        )
    
    def can_add_position(self, symbol: str, signal: Signal) -> bool:
        """Check if a new position can be added to the portfolio.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            
        Returns:
            True if position can be added, False otherwise
        """
        try:
            # Check position count limit
            if len(self.positions) >= self.constraints.max_concurrent_positions:
                logger.debug(f"Max concurrent positions reached: {len(self.positions)}")
                return False
            
            # Check if symbol already has position
            if symbol in self.positions:
                logger.debug(f"Position already exists for {symbol}")
                return False
            
            # Check total exposure limit
            current_exposure = self._calculate_total_exposure()
            max_exposure = self.account_balance * self.constraints.max_portfolio_exposure
            
            if current_exposure >= max_exposure:
                logger.debug(
                    f"Max portfolio exposure reached",
                    current=float(current_exposure),
                    max=float(max_exposure)
                )
                return False
            
            # Check correlation limits
            if not self._check_correlation_limits(symbol):
                logger.debug(f"Correlation limits exceeded for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if position can be added: {e}")
            return False
    
    def allocate_position(
        self,
        symbol: str,
        signal: Signal,
        current_price: Decimal
    ) -> PositionAllocation:
        """Calculate position allocation for a signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            PositionAllocation with recommended sizing
        """
        try:
            # Calculate base position size
            base_size = self._calculate_base_position_size(signal)
            
            # Adjust for risk
            risk_adjusted_size = self._adjust_for_risk(base_size, signal)
            
            # Adjust for correlation
            correlation_adjustment = self._calculate_correlation_adjustment(symbol)
            correlated_size = risk_adjusted_size * correlation_adjustment
            
            # Apply portfolio constraints
            final_size = self._apply_constraints(correlated_size, current_price)
            
            # Calculate allocation percentage
            position_value = final_size * current_price
            allocation_percent = (position_value / self.account_balance) * Decimal("100")
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(symbol, signal, final_size)
            
            allocation = PositionAllocation(
                symbol=symbol,
                recommended_size=base_size,
                risk_adjusted_size=risk_adjusted_size,
                correlation_adjustment=correlation_adjustment,
                final_size=final_size,
                allocation_percent=allocation_percent,
                risk_score=risk_score,
                metadata={
                    "signal_confidence": float(signal.confidence),
                    "position_value": float(position_value),
                    "timestamp": datetime.now(UTC).isoformat()
                }
            )
            
            logger.debug(
                f"Position allocated for {symbol}",
                final_size=float(final_size),
                allocation_percent=float(allocation_percent),
                risk_score=float(risk_score)
            )
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error allocating position: {e}")
            # Return minimal allocation on error
            return PositionAllocation(
                symbol=symbol,
                recommended_size=Decimal("0"),
                risk_adjusted_size=Decimal("0"),
                correlation_adjustment=Decimal("1"),
                final_size=Decimal("0"),
                allocation_percent=Decimal("0"),
                risk_score=Decimal("1")
            )
    
    def add_position(self, position: Position) -> bool:
        """Add a position to the portfolio.
        
        Args:
            position: Position to add
            
        Returns:
            True if position was added successfully
        """
        try:
            if position.symbol in self.positions:
                logger.warning(f"Position already exists for {position.symbol}")
                return False
            
            self.positions[position.symbol] = position
            self._update_metrics()
            
            logger.info(
                f"Position added to portfolio",
                symbol=position.symbol,
                size=float(position.quantity),
                value=float(position.dollar_value)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """Remove a position from the portfolio.
        
        Args:
            symbol: Symbol to remove
            
        Returns:
            True if position was removed successfully
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions.pop(symbol)
            self._update_metrics()
            
            logger.info(
                f"Position removed from portfolio",
                symbol=symbol,
                pnl=float(position.pnl_dollars) if position.pnl_dollars else 0
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: Decimal) -> None:
        """Update position with current market price.
        
        Args:
            symbol: Symbol to update
            current_price: Current market price
        """
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            position.current_price = current_price
            position.update_pnl(current_price)
            position.updated_at = datetime.now(UTC)
            
            self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def get_portfolio_status(self) -> dict[str, Any]:
        """Get current portfolio status.
        
        Returns:
            Dictionary with portfolio status information
        """
        try:
            return {
                "position_count": len(self.positions),
                "max_positions": self.constraints.max_concurrent_positions,
                "total_exposure": float(self.metrics.total_exposure),
                "exposure_percent": float(
                    (self.metrics.total_exposure / self.account_balance * Decimal("100"))
                    if self.account_balance > 0 else Decimal("0")
                ),
                "unrealized_pnl": float(self.metrics.unrealized_pnl),
                "realized_pnl": float(self.metrics.realized_pnl),
                "sharpe_ratio": float(self.metrics.sharpe_ratio),
                "max_drawdown": float(self.metrics.max_drawdown),
                "positions": {
                    symbol: {
                        "entry_price": float(pos.entry_price),
                        "current_price": float(pos.current_price) if pos.current_price else None,
                        "quantity": float(pos.quantity),
                        "pnl": float(pos.pnl_dollars) if pos.pnl_dollars else 0,
                        "pnl_percent": float(pos.pnl_percent) if pos.pnl_percent else 0
                    }
                    for symbol, pos in self.positions.items()
                },
                "last_update": self.metrics.last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
    
    def calculate_correlation_matrix(self, price_data: dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix for symbols.
        
        Args:
            price_data: Dictionary of symbol -> price series
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            # Create DataFrame from price data
            df = pd.DataFrame(price_data)
            
            # Calculate returns
            returns = df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Update cache
            self.correlation_cache["latest"] = correlation_matrix
            self.cache_timestamp = datetime.now(UTC)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def _calculate_base_position_size(self, signal: Signal) -> Decimal:
        """Calculate base position size from signal."""
        try:
            # Use signal's recommended quantity if available
            if signal.quantity:
                return signal.quantity
            
            # Otherwise calculate based on confidence and account balance
            max_position = self.account_balance * self.constraints.max_position_percent
            base_size = max_position * signal.confidence
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating base position size: {e}")
            return Decimal("0")
    
    def _adjust_for_risk(self, base_size: Decimal, signal: Signal) -> Decimal:
        """Adjust position size for risk factors."""
        try:
            risk_multiplier = Decimal("1")
            
            # Adjust based on signal confidence
            if signal.confidence < Decimal("0.3"):
                risk_multiplier *= Decimal("0.5")
            elif signal.confidence > Decimal("0.7"):
                risk_multiplier *= Decimal("1.2")
            
            # Adjust based on stop loss distance
            if signal.stop_loss and signal.price_target:
                risk_reward_ratio = abs(signal.price_target - signal.stop_loss)
                if risk_reward_ratio > Decimal("0"):
                    # Better risk/reward = larger position
                    risk_multiplier *= min(Decimal("1.5"), Decimal("1") + (risk_reward_ratio / Decimal("10")))
            
            return base_size * risk_multiplier
            
        except Exception as e:
            logger.error(f"Error adjusting for risk: {e}")
            return base_size
    
    def _calculate_correlation_adjustment(self, symbol: str) -> Decimal:
        """Calculate correlation-based size adjustment."""
        try:
            if not self.positions:
                return Decimal("1")  # No adjustment needed
            
            # Check cached correlation matrix
            if "latest" not in self.correlation_cache:
                return Decimal("1")  # No correlation data available
            
            corr_matrix = self.correlation_cache["latest"]
            
            if symbol not in corr_matrix.columns:
                return Decimal("1")  # Symbol not in correlation matrix
            
            # Calculate average correlation with existing positions
            existing_symbols = [s for s in self.positions.keys() if s in corr_matrix.columns]
            
            if not existing_symbols:
                return Decimal("1")
            
            avg_correlation = Decimal("0")
            for existing_symbol in existing_symbols:
                correlation = abs(corr_matrix.loc[symbol, existing_symbol])
                avg_correlation += Decimal(str(correlation))
            
            avg_correlation /= len(existing_symbols)
            
            # Reduce size based on correlation
            if avg_correlation > self.constraints.max_correlation:
                # High correlation - reduce size significantly
                adjustment = Decimal("0.5")
            elif avg_correlation > Decimal("0.5"):
                # Moderate correlation - reduce size moderately
                adjustment = Decimal("1") - (avg_correlation - Decimal("0.5")) / Decimal("2")
            else:
                # Low correlation - no reduction
                adjustment = Decimal("1")
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return Decimal("1")
    
    def _apply_constraints(self, size: Decimal, price: Decimal) -> Decimal:
        """Apply portfolio constraints to position size."""
        try:
            position_value = size * price
            
            # Apply minimum position value
            if position_value < self.constraints.min_position_value:
                size = self.constraints.min_position_value / price
            
            # Apply maximum position value
            if position_value > self.constraints.max_position_value:
                size = self.constraints.max_position_value / price
            
            # Apply maximum position percent
            max_size_by_percent = (self.account_balance * self.constraints.max_position_percent) / price
            size = min(size, max_size_by_percent)
            
            # Ensure we don't exceed total exposure limit
            current_exposure = self._calculate_total_exposure()
            remaining_exposure = (self.account_balance * self.constraints.max_portfolio_exposure) - current_exposure
            
            if remaining_exposure > 0:
                max_size_by_exposure = remaining_exposure / price
                size = min(size, max_size_by_exposure)
            else:
                size = Decimal("0")
            
            return size.quantize(Decimal("0.001"))  # Round to 3 decimal places
            
        except Exception as e:
            logger.error(f"Error applying constraints: {e}")
            return Decimal("0")
    
    def _calculate_risk_score(self, symbol: str, signal: Signal, size: Decimal) -> Decimal:
        """Calculate risk score for position."""
        try:
            risk_score = Decimal("0")
            
            # Risk from position size
            position_percent = (size * signal.price_target if signal.price_target else Decimal("0")) / self.account_balance
            risk_score += position_percent * Decimal("10")  # Scale to 0-10
            
            # Risk from correlation
            correlation_adjustment = self._calculate_correlation_adjustment(symbol)
            risk_score += (Decimal("1") - correlation_adjustment) * Decimal("5")
            
            # Risk from confidence
            risk_score += (Decimal("1") - signal.confidence) * Decimal("3")
            
            # Normalize to 0-1 scale
            risk_score = min(Decimal("1"), risk_score / Decimal("10"))
            
            return risk_score.quantize(Decimal("0.01"))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return Decimal("0.5")
    
    def _check_correlation_limits(self, symbol: str) -> bool:
        """Check if adding symbol would violate correlation limits."""
        try:
            if not self.positions:
                return True  # No existing positions
            
            # Check cached correlation matrix
            if "latest" not in self.correlation_cache:
                return True  # No correlation data available, allow position
            
            corr_matrix = self.correlation_cache["latest"]
            
            if symbol not in corr_matrix.columns:
                return True  # Symbol not in correlation matrix, allow position
            
            # Check correlation with each existing position
            for existing_symbol in self.positions.keys():
                if existing_symbol in corr_matrix.columns:
                    correlation = abs(corr_matrix.loc[symbol, existing_symbol])
                    if correlation > float(self.constraints.max_correlation):
                        logger.debug(
                            f"Correlation limit exceeded",
                            symbol=symbol,
                            existing=existing_symbol,
                            correlation=correlation
                        )
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking correlation limits: {e}")
            return True  # Allow on error
    
    def _calculate_total_exposure(self) -> Decimal:
        """Calculate total portfolio exposure."""
        try:
            total = Decimal("0")
            for position in self.positions.values():
                total += position.dollar_value
            return total
            
        except Exception as e:
            logger.error(f"Error calculating total exposure: {e}")
            return Decimal("0")
    
    def _update_metrics(self) -> None:
        """Update portfolio metrics."""
        try:
            # Update basic metrics
            self.metrics.total_exposure = self._calculate_total_exposure()
            self.metrics.position_count = len(self.positions)
            
            # Calculate unrealized P&L
            unrealized_pnl = Decimal("0")
            for position in self.positions.values():
                if position.pnl_dollars:
                    unrealized_pnl += position.pnl_dollars
            self.metrics.unrealized_pnl = unrealized_pnl
            
            # Update timestamp
            self.metrics.last_update = datetime.now(UTC)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")