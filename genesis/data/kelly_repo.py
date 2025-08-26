"""
Repository for Kelly Criterion parameters and calculations.

Provides database persistence for Kelly parameters, historical calculations,
and strategy performance metrics.
"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from uuid import UUID
import json
import logging

from sqlalchemy import Column, String, Numeric, DateTime, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from genesis.core.constants import TradingTier, ConvictionLevel
from genesis.analytics.kelly_sizing import StrategyEdge, KellyParams, SimulationResult
from genesis.analytics.strategy_metrics import StrategyMetrics

logger = logging.getLogger(__name__)

Base = declarative_base()


class KellyParameterDB(Base):
    """Database model for Kelly parameters."""
    
    __tablename__ = "kelly_parameters"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String, nullable=False, index=True)
    kelly_fraction = Column(Numeric(10, 4), nullable=False)
    fractional_multiplier = Column(Numeric(10, 4), nullable=False)
    win_rate = Column(Numeric(10, 4), nullable=False)
    win_loss_ratio = Column(Numeric(10, 4), nullable=False)
    sample_size = Column(Integer, nullable=False)
    confidence_lower = Column(Numeric(10, 4))
    confidence_upper = Column(Numeric(10, 4))
    calculated_at = Column(DateTime, nullable=False)
    
    def to_strategy_edge(self) -> StrategyEdge:
        """Convert database model to domain model."""
        return StrategyEdge(
            strategy_id=self.strategy_id,
            win_rate=Decimal(str(self.win_rate)),
            win_loss_ratio=Decimal(str(self.win_loss_ratio)),
            sample_size=self.sample_size,
            confidence_interval=(
                Decimal(str(self.confidence_lower)) if self.confidence_lower else Decimal("0"),
                Decimal(str(self.confidence_upper)) if self.confidence_upper else Decimal("1")
            ),
            last_calculated=self.calculated_at
        )


class KellyCalculationDB(Base):
    """Database model for historical Kelly calculations."""
    
    __tablename__ = "kelly_calculations"
    
    id = Column(Integer, primary_key=True)
    calculation_id = Column(String, nullable=False, unique=True)
    strategy_id = Column(String, nullable=False, index=True)
    account_balance = Column(Numeric(20, 2), nullable=False)
    kelly_fraction = Column(Numeric(10, 4), nullable=False)
    position_size = Column(Numeric(20, 2), nullable=False)
    conviction_level = Column(String, nullable=False)
    volatility_multiplier = Column(Numeric(10, 4))
    final_size = Column(Numeric(20, 2), nullable=False)
    tier = Column(String, nullable=False)
    calculated_at = Column(DateTime, nullable=False)
    metadata = Column(JSON)  # Additional data like volatility regime, adjustments


class StrategyPerformanceDB(Base):
    """Database model for strategy performance metrics."""
    
    __tablename__ = "strategy_performance"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String, nullable=False, unique=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Numeric(20, 2), default=0)
    total_win_amount = Column(Numeric(20, 2), default=0)
    total_loss_amount = Column(Numeric(20, 2), default=0)
    max_drawdown = Column(Numeric(10, 4), default=0)
    sharpe_ratio = Column(Numeric(10, 4), default=0)
    win_rate = Column(Numeric(10, 4), default=0)
    profit_factor = Column(Numeric(10, 4), default=0)
    average_win = Column(Numeric(20, 2), default=0)
    average_loss = Column(Numeric(20, 2), default=0)
    last_updated = Column(DateTime, nullable=False)
    
    def to_strategy_metrics(self) -> StrategyMetrics:
        """Convert database model to domain model."""
        return StrategyMetrics(
            strategy_id=self.strategy_id,
            total_trades=self.total_trades,
            winning_trades=self.winning_trades,
            losing_trades=self.losing_trades,
            total_pnl=Decimal(str(self.total_pnl)),
            total_win_amount=Decimal(str(self.total_win_amount)),
            total_loss_amount=Decimal(str(self.total_loss_amount)),
            max_drawdown=Decimal(str(self.max_drawdown)),
            sharpe_ratio=Decimal(str(self.sharpe_ratio)),
            win_rate=Decimal(str(self.win_rate)),
            profit_factor=Decimal(str(self.profit_factor)),
            average_win=Decimal(str(self.average_win)),
            average_loss=Decimal(str(self.average_loss)),
            last_updated=self.last_updated
        )


class MonteCarloSimulationDB(Base):
    """Database model for Monte Carlo simulation results."""
    
    __tablename__ = "monte_carlo_simulations"
    
    id = Column(Integer, primary_key=True)
    simulation_id = Column(String, nullable=False, unique=True)
    strategy_id = Column(String, nullable=False, index=True)
    win_rate = Column(Numeric(10, 4), nullable=False)
    win_loss_ratio = Column(Numeric(10, 4), nullable=False)
    kelly_fraction = Column(Numeric(10, 4), nullable=False)
    optimal_kelly = Column(Numeric(10, 4), nullable=False)
    risk_of_ruin = Column(Numeric(10, 4), nullable=False)
    expected_growth_rate = Column(Numeric(10, 4), nullable=False)
    median_final_balance = Column(Numeric(20, 2), nullable=False)
    percentile_5 = Column(Numeric(20, 2), nullable=False)
    percentile_95 = Column(Numeric(20, 2), nullable=False)
    iterations = Column(Integer, nullable=False)
    trades_per_iteration = Column(Integer, nullable=False)
    simulated_at = Column(DateTime, nullable=False)


class KellyRepository:
    """Repository for Kelly Criterion data persistence."""
    
    def __init__(self, session: Session):
        """
        Initialize Kelly repository.
        
        Args:
            session: SQLAlchemy session
        """
        self.session = session
    
    def save_kelly_parameters(self, edge: StrategyEdge) -> None:
        """
        Save Kelly parameters to database.
        
        Args:
            edge: StrategyEdge with calculated parameters
        """
        try:
            params = KellyParameterDB(
                strategy_id=edge.strategy_id,
                kelly_fraction=float(edge.win_rate * edge.win_loss_ratio - (1 - edge.win_rate)) / float(edge.win_loss_ratio) if edge.win_loss_ratio > 0 else 0,
                fractional_multiplier=0.25,  # Default fractional Kelly
                win_rate=float(edge.win_rate),
                win_loss_ratio=float(edge.win_loss_ratio),
                sample_size=edge.sample_size,
                confidence_lower=float(edge.confidence_interval[0]),
                confidence_upper=float(edge.confidence_interval[1]),
                calculated_at=edge.last_calculated
            )
            
            self.session.merge(params)
            self.session.commit()
            
            logger.info("Kelly parameters saved", strategy_id=edge.strategy_id)
        except Exception as e:
            logger.error("Failed to save Kelly parameters", error=str(e))
            self.session.rollback()
            raise
    
    def get_kelly_parameters(self, strategy_id: str) -> Optional[StrategyEdge]:
        """
        Retrieve Kelly parameters for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            StrategyEdge or None if not found
        """
        try:
            params = self.session.query(KellyParameterDB)\
                .filter_by(strategy_id=strategy_id)\
                .order_by(KellyParameterDB.calculated_at.desc())\
                .first()
            
            if params:
                return params.to_strategy_edge()
            return None
        except Exception as e:
            logger.error("Failed to retrieve Kelly parameters", error=str(e))
            return None
    
    def save_kelly_calculation(
        self,
        calculation_id: str,
        strategy_id: str,
        balance: Decimal,
        kelly_params: KellyParams,
        tier: TradingTier,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save a Kelly calculation for audit trail.
        
        Args:
            calculation_id: Unique calculation ID
            strategy_id: Strategy identifier
            balance: Account balance at calculation
            kelly_params: Kelly calculation parameters
            tier: Trading tier
            metadata: Additional metadata
        """
        try:
            calculation = KellyCalculationDB(
                calculation_id=calculation_id,
                strategy_id=strategy_id,
                account_balance=float(balance),
                kelly_fraction=float(kelly_params.kelly_fraction),
                position_size=float(kelly_params.position_size),
                conviction_level="MEDIUM",  # Default if not in metadata
                volatility_multiplier=metadata.get("volatility_multiplier") if metadata else None,
                final_size=float(kelly_params.position_size),
                tier=tier.value,
                calculated_at=datetime.now(timezone.utc),
                metadata=metadata
            )
            
            self.session.add(calculation)
            self.session.commit()
            
            logger.debug("Kelly calculation saved", calculation_id=calculation_id)
        except Exception as e:
            logger.error("Failed to save Kelly calculation", error=str(e))
            self.session.rollback()
    
    def save_strategy_performance(self, metrics: StrategyMetrics) -> None:
        """
        Save or update strategy performance metrics.
        
        Args:
            metrics: Strategy performance metrics
        """
        try:
            perf = StrategyPerformanceDB(
                strategy_id=metrics.strategy_id,
                total_trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                total_pnl=float(metrics.total_pnl),
                total_win_amount=float(metrics.total_win_amount),
                total_loss_amount=float(metrics.total_loss_amount),
                max_drawdown=float(metrics.max_drawdown),
                sharpe_ratio=float(metrics.sharpe_ratio),
                win_rate=float(metrics.win_rate),
                profit_factor=float(metrics.profit_factor),
                average_win=float(metrics.average_win),
                average_loss=float(metrics.average_loss),
                last_updated=metrics.last_updated
            )
            
            # Use merge to update if exists
            self.session.merge(perf)
            self.session.commit()
            
            logger.info("Strategy performance saved", strategy_id=metrics.strategy_id)
        except Exception as e:
            logger.error("Failed to save strategy performance", error=str(e))
            self.session.rollback()
            raise
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[StrategyMetrics]:
        """
        Retrieve strategy performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            StrategyMetrics or None if not found
        """
        try:
            perf = self.session.query(StrategyPerformanceDB)\
                .filter_by(strategy_id=strategy_id)\
                .first()
            
            if perf:
                return perf.to_strategy_metrics()
            return None
        except Exception as e:
            logger.error("Failed to retrieve strategy performance", error=str(e))
            return None
    
    def save_simulation_result(
        self,
        simulation_id: str,
        strategy_id: str,
        win_rate: Decimal,
        win_loss_ratio: Decimal,
        kelly_fraction: Decimal,
        result: SimulationResult,
        iterations: int,
        trades_per_iteration: int
    ) -> None:
        """
        Save Monte Carlo simulation results.
        
        Args:
            simulation_id: Unique simulation ID
            strategy_id: Strategy identifier
            win_rate: Win rate used in simulation
            win_loss_ratio: Win/loss ratio used
            kelly_fraction: Kelly fraction tested
            result: Simulation results
            iterations: Number of iterations run
            trades_per_iteration: Trades per iteration
        """
        try:
            simulation = MonteCarloSimulationDB(
                simulation_id=simulation_id,
                strategy_id=strategy_id,
                win_rate=float(win_rate),
                win_loss_ratio=float(win_loss_ratio),
                kelly_fraction=float(kelly_fraction),
                optimal_kelly=float(result.optimal_kelly),
                risk_of_ruin=float(result.risk_of_ruin),
                expected_growth_rate=float(result.expected_growth_rate),
                median_final_balance=float(result.median_final_balance),
                percentile_5=float(result.percentile_5),
                percentile_95=float(result.percentile_95),
                iterations=iterations,
                trades_per_iteration=trades_per_iteration,
                simulated_at=datetime.now(timezone.utc)
            )
            
            self.session.add(simulation)
            self.session.commit()
            
            logger.info("Simulation result saved", simulation_id=simulation_id)
        except Exception as e:
            logger.error("Failed to save simulation result", error=str(e))
            self.session.rollback()
    
    def get_recent_calculations(
        self,
        strategy_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent Kelly calculations for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of results
            
        Returns:
            List of calculation dictionaries
        """
        try:
            calcs = self.session.query(KellyCalculationDB)\
                .filter_by(strategy_id=strategy_id)\
                .order_by(KellyCalculationDB.calculated_at.desc())\
                .limit(limit)\
                .all()
            
            return [
                {
                    "calculation_id": c.calculation_id,
                    "kelly_fraction": Decimal(str(c.kelly_fraction)),
                    "position_size": Decimal(str(c.position_size)),
                    "conviction_level": c.conviction_level,
                    "tier": c.tier,
                    "calculated_at": c.calculated_at,
                    "metadata": c.metadata
                }
                for c in calcs
            ]
        except Exception as e:
            logger.error("Failed to retrieve calculations", error=str(e))
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old Kelly data.
        
        Args:
            days_to_keep: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Delete old calculations
            calc_count = self.session.query(KellyCalculationDB)\
                .filter(KellyCalculationDB.calculated_at < cutoff_date)\
                .delete()
            
            # Delete old simulations
            sim_count = self.session.query(MonteCarloSimulationDB)\
                .filter(MonteCarloSimulationDB.simulated_at < cutoff_date)\
                .delete()
            
            self.session.commit()
            
            total_deleted = calc_count + sim_count
            logger.info("Cleaned up old Kelly data", records_deleted=total_deleted)
            
            return total_deleted
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
            self.session.rollback()
            return 0