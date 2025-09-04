"""
Monte Carlo Simulation for Strategy Robustness Testing

Provides statistical analysis through parameter randomization.
"""

import asyncio
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import structlog

logger = structlog.get_logger()


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    num_simulations: int = 1000
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    confidence_intervals: List[float] = field(default_factory=lambda: [0.95, 0.99])
    seed: Optional[int] = None
    parallel_jobs: int = 4
    perturbation_type: str = "uniform"  # uniform, normal, lognormal
    save_all_results: bool = False


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    mean_return: float
    std_return: float
    median_return: float
    min_return: float
    max_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    confidence_intervals: Dict[float, Tuple[float, float]]
    percentiles: Dict[int, float]
    parameter_sensitivity: Dict[str, float]
    convergence_metrics: Dict[str, List[float]]
    all_results: Optional[List[Dict[str, Any]]] = None


class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtesting robustness.
    
    Tests strategy performance across parameter variations.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """Initialize Monte Carlo simulator.
        
        Args:
            config: Monte Carlo configuration
        """
        self.config = config
        self.results = []
        self.convergence_history = {
            'mean': [],
            'std': [],
            'sharpe': []
        }
        
        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
    
    async def run(
        self,
        backtest_engine: Any,
        strategy_class: type,
        base_params: Dict[str, Any],
        market_data: Any = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            backtest_engine: Backtesting engine instance
            strategy_class: Strategy class to test
            base_params: Base strategy parameters
            market_data: Optional market data to use
            
        Returns:
            MonteCarloResult with statistical analysis
        """
        logger.info(
            "monte_carlo_started",
            simulations=self.config.num_simulations,
            parameters=list(self.config.parameter_ranges.keys())
        )
        
        # Generate parameter sets
        parameter_sets = self._generate_parameter_sets(base_params)
        
        # Run simulations
        results = await self._run_simulations(
            backtest_engine,
            strategy_class,
            parameter_sets,
            market_data
        )
        
        # Analyze results
        analysis = self._analyze_results(results, parameter_sets)
        
        logger.info(
            "monte_carlo_completed",
            simulations=len(results),
            mean_return=analysis.mean_return,
            std_return=analysis.std_return,
            sharpe_ratio=analysis.sharpe_ratio
        )
        
        return analysis
    
    def _generate_parameter_sets(
        self,
        base_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate randomized parameter sets.
        
        Args:
            base_params: Base parameters to vary
            
        Returns:
            List of parameter dictionaries
        """
        parameter_sets = []
        
        for i in range(self.config.num_simulations):
            params = base_params.copy()
            
            # Apply perturbations
            for param_name, (min_val, max_val) in self.config.parameter_ranges.items():
                if param_name in params:
                    base_value = params[param_name]
                    
                    if self.config.perturbation_type == "uniform":
                        # Uniform distribution
                        value = random.uniform(min_val, max_val)
                    elif self.config.perturbation_type == "normal":
                        # Normal distribution around base value
                        mean = base_value
                        std = (max_val - min_val) / 6  # 99.7% within range
                        value = np.random.normal(mean, std)
                        value = max(min_val, min(max_val, value))  # Clip to range
                    elif self.config.perturbation_type == "lognormal":
                        # Lognormal distribution
                        mean = np.log(base_value) if base_value > 0 else 0
                        std = 0.2
                        value = np.exp(np.random.normal(mean, std))
                        value = max(min_val, min(max_val, value))
                    else:
                        value = base_value
                    
                    params[param_name] = type(base_value)(value)
            
            parameter_sets.append(params)
        
        return parameter_sets
    
    async def _run_simulations(
        self,
        backtest_engine: Any,
        strategy_class: type,
        parameter_sets: List[Dict[str, Any]],
        market_data: Any
    ) -> List[Dict[str, Any]]:
        """
        Run parallel backtest simulations.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            parameter_sets: List of parameter sets
            market_data: Market data
            
        Returns:
            List of simulation results
        """
        results = []
        
        # Create progress callback
        async def progress_callback(sim_num: int):
            if sim_num % 100 == 0:
                logger.info(f"Monte Carlo progress: {sim_num}/{self.config.num_simulations}")
        
        # Run simulations in batches for better parallelism
        batch_size = self.config.parallel_jobs
        
        for batch_start in range(0, len(parameter_sets), batch_size):
            batch_end = min(batch_start + batch_size, len(parameter_sets))
            batch_params = parameter_sets[batch_start:batch_end]
            
            # Create tasks for parallel execution
            tasks = []
            for i, params in enumerate(batch_params):
                sim_num = batch_start + i
                task = self._run_single_simulation(
                    backtest_engine,
                    strategy_class,
                    params,
                    market_data,
                    sim_num
                )
                tasks.append(task)
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "simulation_failed",
                        simulation=batch_start + i,
                        error=str(result)
                    )
                else:
                    results.append(result)
                    # Update convergence metrics
                    self._update_convergence(results)
            
            # Progress update
            await progress_callback(batch_end)
        
        return results
    
    async def _run_single_simulation(
        self,
        backtest_engine: Any,
        strategy_class: type,
        params: Dict[str, Any],
        market_data: Any,
        sim_num: int
    ) -> Dict[str, Any]:
        """
        Run a single backtest simulation.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            params: Strategy parameters
            market_data: Market data
            sim_num: Simulation number
            
        Returns:
            Simulation results
        """
        try:
            # Create strategy instance with parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            result = await backtest_engine.run_backtest(strategy)
            
            # Extract key metrics
            metrics = {
                'simulation': sim_num,
                'parameters': params,
                'total_return': float((result.final_capital - result.initial_capital) / result.initial_capital),
                'final_capital': float(result.final_capital),
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'max_drawdown': float(result.max_drawdown),
                'profit_factor': result.profit_factor,
                'avg_win': float(result.avg_win),
                'avg_loss': float(result.avg_loss)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(
                "single_simulation_failed",
                simulation=sim_num,
                error=str(e)
            )
            raise
    
    def _update_convergence(self, results: List[Dict[str, Any]]) -> None:
        """Update convergence metrics.
        
        Args:
            results: Current results list
        """
        if not results:
            return
        
        returns = [r['total_return'] for r in results]
        sharpes = [r['sharpe_ratio'] for r in results]
        
        self.convergence_history['mean'].append(np.mean(returns))
        self.convergence_history['std'].append(np.std(returns))
        self.convergence_history['sharpe'].append(np.mean(sharpes))
    
    def _analyze_results(
        self,
        results: List[Dict[str, Any]],
        parameter_sets: List[Dict[str, Any]]
    ) -> MonteCarloResult:
        """
        Analyze Monte Carlo simulation results.
        
        Args:
            results: List of simulation results
            parameter_sets: Parameter sets used
            
        Returns:
            MonteCarloResult with statistical analysis
        """
        # Extract metrics
        returns = np.array([r['total_return'] for r in results])
        sharpes = np.array([r['sharpe_ratio'] for r in results])
        win_rates = np.array([r['win_rate'] for r in results])
        drawdowns = np.array([r['max_drawdown'] for r in results])
        
        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        median_return = np.median(returns)
        
        # Confidence intervals
        confidence_intervals = {}
        for ci_level in self.config.confidence_intervals:
            lower = np.percentile(returns, (1 - ci_level) * 50)
            upper = np.percentile(returns, 100 - (1 - ci_level) * 50)
            confidence_intervals[ci_level] = (lower, upper)
        
        # Percentiles
        percentiles = {
            5: np.percentile(returns, 5),
            25: np.percentile(returns, 25),
            50: np.percentile(returns, 50),
            75: np.percentile(returns, 75),
            95: np.percentile(returns, 95)
        }
        
        # Parameter sensitivity analysis
        sensitivity = self._calculate_parameter_sensitivity(
            results,
            parameter_sets
        )
        
        return MonteCarloResult(
            mean_return=mean_return,
            std_return=std_return,
            median_return=median_return,
            min_return=np.min(returns),
            max_return=np.max(returns),
            sharpe_ratio=np.mean(sharpes),
            win_rate=np.mean(win_rates),
            max_drawdown=np.mean(drawdowns),
            confidence_intervals=confidence_intervals,
            percentiles=percentiles,
            parameter_sensitivity=sensitivity,
            convergence_metrics=self.convergence_history,
            all_results=results if self.config.save_all_results else None
        )
    
    def _calculate_parameter_sensitivity(
        self,
        results: List[Dict[str, Any]],
        parameter_sets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate parameter sensitivity using correlation analysis.
        
        Args:
            results: Simulation results
            parameter_sets: Parameter sets
            
        Returns:
            Dictionary of parameter sensitivities
        """
        sensitivity = {}
        returns = [r['total_return'] for r in results]
        
        for param_name in self.config.parameter_ranges.keys():
            if param_name in parameter_sets[0]:
                param_values = [p[param_name] for p in parameter_sets]
                
                # Calculate correlation
                if len(set(param_values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(param_values, returns)[0, 1]
                    sensitivity[param_name] = abs(correlation)
                else:
                    sensitivity[param_name] = 0.0
        
        return sensitivity


def plot_monte_carlo_results(result: MonteCarloResult) -> None:
    """
    Plot Monte Carlo simulation results.
    
    Args:
        result: Monte Carlo results
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Returns distribution
        if result.all_results:
            returns = [r['total_return'] for r in result.all_results]
            axes[0, 0].hist(returns, bins=50, edgecolor='black')
            axes[0, 0].axvline(result.mean_return, color='red', linestyle='--', label='Mean')
            axes[0, 0].axvline(result.median_return, color='green', linestyle='--', label='Median')
            axes[0, 0].set_title('Returns Distribution')
            axes[0, 0].set_xlabel('Return')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # Convergence plot
        axes[0, 1].plot(result.convergence_metrics['mean'], label='Mean Return')
        axes[0, 1].set_title('Convergence of Mean Return')
        axes[0, 1].set_xlabel('Simulation Number')
        axes[0, 1].set_ylabel('Mean Return')
        axes[0, 1].grid(True)
        
        # Parameter sensitivity
        if result.parameter_sensitivity:
            params = list(result.parameter_sensitivity.keys())
            sensitivities = list(result.parameter_sensitivity.values())
            axes[1, 0].bar(params, sensitivities)
            axes[1, 0].set_title('Parameter Sensitivity')
            axes[1, 0].set_xlabel('Parameter')
            axes[1, 0].set_ylabel('Correlation with Return')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Confidence intervals
        ci_levels = list(result.confidence_intervals.keys())
        ci_ranges = [ci[1] - ci[0] for ci in result.confidence_intervals.values()]
        axes[1, 1].bar(ci_levels, ci_ranges)
        axes[1, 1].set_title('Confidence Interval Ranges')
        axes[1, 1].set_xlabel('Confidence Level')
        axes[1, 1].set_ylabel('Range Width')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")