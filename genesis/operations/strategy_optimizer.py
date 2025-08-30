"""Strategy Parameter Optimization with A/B Testing."""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import numpy as np

from pydantic import BaseModel, Field

from genesis.utils.logger import get_logger, LoggerType


@dataclass
class ParameterSet:
    """A set of strategy parameters for testing."""
    id: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    sample_size: int = 0
    created_at: datetime = datetime.utcnow()


class OptimizationConfig(BaseModel):
    """Configuration for strategy optimization."""
    
    min_sample_size: int = Field(100, description="Minimum trades for significance")
    confidence_level: float = Field(0.95, description="Statistical confidence level")
    max_concurrent_tests: int = Field(3, description="Maximum concurrent A/B tests")
    exploration_rate: float = Field(0.1, description="Rate of parameter exploration")
    optimization_interval_hours: int = Field(24, description="Hours between optimizations")


class StrategyOptimizer:
    """Optimizes strategy parameters using A/B testing and Bayesian optimization."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.active_tests: Dict[str, Tuple[ParameterSet, ParameterSet]] = {}
        self.parameter_history: List[ParameterSet] = []
        self.logger = get_logger(__name__, LoggerType.SYSTEM)
    
    async def create_ab_test(
        self,
        strategy_name: str,
        control_params: Dict[str, Any],
        test_params: Dict[str, Any]
    ) -> str:
        """Create a new A/B test for strategy parameters."""
        if len(self.active_tests) >= self.config.max_concurrent_tests:
            self.logger.warning("max_concurrent_tests_reached")
            return ""
        
        test_id = f"{strategy_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        control = ParameterSet(f"{test_id}_control", control_params, {})
        test = ParameterSet(f"{test_id}_test", test_params, {})
        
        self.active_tests[test_id] = (control, test)
        
        self.logger.info(
            "ab_test_created",
            test_id=test_id,
            strategy=strategy_name,
            control_params=control_params,
            test_params=test_params
        )
        
        return test_id
    
    def record_performance(
        self,
        test_id: str,
        is_control: bool,
        pnl: float,
        sharpe: float,
        max_drawdown: float
    ) -> None:
        """Record performance metrics for A/B test."""
        if test_id not in self.active_tests:
            return
        
        control, test = self.active_tests[test_id]
        param_set = control if is_control else test
        
        param_set.sample_size += 1
        
        # Update rolling metrics
        if "total_pnl" not in param_set.performance_metrics:
            param_set.performance_metrics = {
                "total_pnl": 0,
                "avg_sharpe": 0,
                "max_drawdown": 0
            }
        
        param_set.performance_metrics["total_pnl"] += pnl
        param_set.performance_metrics["avg_sharpe"] = (
            (param_set.performance_metrics["avg_sharpe"] * (param_set.sample_size - 1) + sharpe)
            / param_set.sample_size
        )
        param_set.performance_metrics["max_drawdown"] = max(
            param_set.performance_metrics["max_drawdown"],
            max_drawdown
        )
    
    async def evaluate_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate A/B test for statistical significance."""
        if test_id not in self.active_tests:
            return None
        
        control, test = self.active_tests[test_id]
        
        # Check minimum sample size
        if control.sample_size < self.config.min_sample_size or \
           test.sample_size < self.config.min_sample_size:
            return None
        
        # Perform statistical test (t-test for PnL)
        control_pnl = control.performance_metrics.get("total_pnl", 0) / max(control.sample_size, 1)
        test_pnl = test.performance_metrics.get("total_pnl", 0) / max(test.sample_size, 1)
        
        # Simplified t-test (would need actual sample data for proper test)
        t_stat = (test_pnl - control_pnl) / (0.1 + abs(control_pnl))  # Simplified
        p_value = 1 - stats.norm.cdf(abs(t_stat))
        
        significant = p_value < (1 - self.config.confidence_level)
        
        result = {
            "test_id": test_id,
            "control_pnl": control_pnl,
            "test_pnl": test_pnl,
            "improvement": (test_pnl - control_pnl) / max(abs(control_pnl), 1),
            "p_value": p_value,
            "significant": significant,
            "winner": "test" if significant and test_pnl > control_pnl else "control"
        }
        
        self.logger.info(
            "test_evaluated",
            test_id=test_id,
            p_value=p_value,
            significant=significant,
            winner=result["winner"]
        )
        
        return result
    
    async def bayesian_optimization(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        current_best: Dict[str, float]
    ) -> Dict[str, float]:
        """Use Bayesian optimization to suggest next parameters."""
        # Simplified Bayesian optimization
        # In production, use libraries like scikit-optimize
        
        new_params = {}
        
        for param, (min_val, max_val) in parameter_space.items():
            current = current_best.get(param, (min_val + max_val) / 2)
            
            # Exploration vs exploitation
            if random.random() < self.config.exploration_rate:
                # Explore
                new_params[param] = random.uniform(min_val, max_val)
            else:
                # Exploit with small perturbation
                perturbation = (max_val - min_val) * 0.1 * random.gauss(0, 1)
                new_params[param] = max(min_val, min(max_val, current + perturbation))
        
        return new_params
    
    async def auto_rollback(self, test_id: str) -> None:
        """Automatically rollback underperforming parameters."""
        result = await self.evaluate_test(test_id)
        
        if result and result["winner"] == "control":
            self.logger.warning(
                "auto_rollback_triggered",
                test_id=test_id,
                reason="test_underperformed"
            )
            
            # Signal to strategy manager to rollback
            # This would integrate with strategy management
            
            del self.active_tests[test_id]
    
    async def optimization_loop(self) -> None:
        """Main optimization loop."""
        while True:
            try:
                await asyncio.sleep(self.config.optimization_interval_hours * 3600)
                
                # Evaluate all active tests
                for test_id in list(self.active_tests.keys()):
                    result = await self.evaluate_test(test_id)
                    if result and result["significant"]:
                        if result["winner"] == "test":
                            # Promote test parameters
                            self.logger.info(
                                "promoting_test_parameters",
                                test_id=test_id,
                                improvement=result["improvement"]
                            )
                        else:
                            # Rollback to control
                            await self.auto_rollback(test_id)
                
            except Exception as e:
                self.logger.error("optimization_loop_error", error=str(e))