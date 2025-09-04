"""
Walk-Forward Optimization for Strategy Parameter Tuning

Provides rolling window optimization to prevent overfitting.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Tuple, Optional, Callable
from enum import Enum

import structlog

logger = structlog.get_logger()


class OptimizationMethod(Enum):
    """Optimization methods for parameter tuning."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    total_period_days: int
    in_sample_days: int
    out_sample_days: int
    step_days: Optional[int] = None  # If None, uses out_sample_days
    optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    parameter_grid: Dict[str, List[Any]] = field(default_factory=dict)
    optimization_metric: str = "sharpe_ratio"
    min_trades_required: int = 10
    anchored_walk: bool = False  # If True, in-sample always starts from beginning


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_number: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    optimal_params: Dict[str, Any]
    in_sample_performance: Dict[str, float]
    out_sample_performance: Dict[str, float]
    performance_degradation: float
    overfitting_score: float


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""
    windows: List[WindowResult]
    overall_performance: Dict[str, float]
    parameter_stability: Dict[str, float]
    overfitting_metrics: Dict[str, float]
    best_stable_params: Dict[str, Any]
    degradation_pattern: List[float]
    success_rate: float


class WalkForwardOptimizer:
    """
    Walk-forward optimizer for robust parameter selection.
    
    Implements rolling window optimization to avoid overfitting.
    """
    
    def __init__(self, config: WalkForwardConfig):
        """Initialize walk-forward optimizer.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.windows = []
        self.parameter_history = []
        
    async def run(
        self,
        backtest_engine: Any,
        strategy_class: type,
        start_date: datetime,
        end_date: datetime
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class to optimize
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            WalkForwardResult with analysis
        """
        logger.info(
            "walk_forward_started",
            start_date=start_date,
            end_date=end_date,
            in_sample_days=self.config.in_sample_days,
            out_sample_days=self.config.out_sample_days,
            method=self.config.optimization_method.value
        )
        
        # Generate windows
        windows = self._generate_windows(start_date, end_date)
        
        # Process each window
        window_results = []
        for i, (is_start, is_end, os_start, os_end) in enumerate(windows):
            logger.info(
                "processing_window",
                window=i+1,
                total_windows=len(windows),
                in_sample=(is_start, is_end),
                out_sample=(os_start, os_end)
            )
            
            result = await self._process_window(
                backtest_engine,
                strategy_class,
                i,
                is_start,
                is_end,
                os_start,
                os_end
            )
            
            window_results.append(result)
            self.windows.append(result)
        
        # Analyze overall results
        analysis = self._analyze_results(window_results)
        
        logger.info(
            "walk_forward_completed",
            windows_processed=len(window_results),
            success_rate=analysis.success_rate,
            avg_degradation=np.mean(analysis.degradation_pattern)
        )
        
        return analysis
    
    def _generate_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows.
        
        Args:
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            List of (in_sample_start, in_sample_end, out_sample_start, out_sample_end)
        """
        windows = []
        
        step_days = self.config.step_days or self.config.out_sample_days
        current_start = start_date
        
        while current_start < end_date:
            if self.config.anchored_walk:
                # Anchored walk: in-sample always starts from beginning
                is_start = start_date
            else:
                # Rolling walk: in-sample moves forward
                is_start = current_start
            
            is_end = is_start + timedelta(days=self.config.in_sample_days)
            os_start = is_end
            os_end = os_start + timedelta(days=self.config.out_sample_days)
            
            # Check if we have enough data
            if os_end > end_date:
                # Adjust last window if needed
                if (end_date - os_start).days >= self.config.out_sample_days // 2:
                    os_end = end_date
                else:
                    break
            
            windows.append((is_start, is_end, os_start, os_end))
            
            # Move to next window
            current_start += timedelta(days=step_days)
        
        return windows
    
    async def _process_window(
        self,
        backtest_engine: Any,
        strategy_class: type,
        window_num: int,
        is_start: datetime,
        is_end: datetime,
        os_start: datetime,
        os_end: datetime
    ) -> WindowResult:
        """
        Process a single walk-forward window.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            window_num: Window number
            is_start: In-sample start date
            is_end: In-sample end date
            os_start: Out-of-sample start date
            os_end: Out-of-sample end date
            
        Returns:
            WindowResult for this window
        """
        # Optimize on in-sample period
        optimal_params, in_sample_perf = await self._optimize_parameters(
            backtest_engine,
            strategy_class,
            is_start,
            is_end
        )
        
        # Test on out-of-sample period
        out_sample_perf = await self._test_parameters(
            backtest_engine,
            strategy_class,
            optimal_params,
            os_start,
            os_end
        )
        
        # Calculate degradation and overfitting metrics
        degradation = self._calculate_degradation(
            in_sample_perf,
            out_sample_perf
        )
        
        overfitting_score = self._calculate_overfitting_score(
            in_sample_perf,
            out_sample_perf
        )
        
        return WindowResult(
            window_number=window_num,
            in_sample_start=is_start,
            in_sample_end=is_end,
            out_sample_start=os_start,
            out_sample_end=os_end,
            optimal_params=optimal_params,
            in_sample_performance=in_sample_perf,
            out_sample_performance=out_sample_perf,
            performance_degradation=degradation,
            overfitting_score=overfitting_score
        )
    
    async def _optimize_parameters(
        self,
        backtest_engine: Any,
        strategy_class: type,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize parameters on in-sample data.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            start_date: Period start
            end_date: Period end
            
        Returns:
            Tuple of (optimal_parameters, performance_metrics)
        """
        if self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
            return await self._grid_search(
                backtest_engine,
                strategy_class,
                start_date,
                end_date
            )
        elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            return await self._random_search(
                backtest_engine,
                strategy_class,
                start_date,
                end_date
            )
        else:
            # Default to grid search
            return await self._grid_search(
                backtest_engine,
                strategy_class,
                start_date,
                end_date
            )
    
    async def _grid_search(
        self,
        backtest_engine: Any,
        strategy_class: type,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Perform grid search optimization.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            start_date: Period start
            end_date: Period end
            
        Returns:
            Tuple of (optimal_parameters, performance_metrics)
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(self.config.parameter_grid.keys())
        param_values = list(self.config.parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        best_score = -float('inf')
        best_params = None
        best_performance = None
        
        # Test each combination
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            # Create strategy with parameters
            strategy = strategy_class(**params)
            
            # Configure backtest engine for this period
            backtest_engine.config.start_date = start_date
            backtest_engine.config.end_date = end_date
            
            # Run backtest
            try:
                result = await backtest_engine.run_backtest(strategy)
                
                # Extract optimization metric
                if self.config.optimization_metric == "sharpe_ratio":
                    score = result.sharpe_ratio
                elif self.config.optimization_metric == "total_return":
                    score = float((result.final_capital - result.initial_capital) / result.initial_capital)
                elif self.config.optimization_metric == "calmar_ratio":
                    score = result.calmar_ratio
                else:
                    score = result.sharpe_ratio
                
                # Check if this is the best so far
                if score > best_score and result.total_trades >= self.config.min_trades_required:
                    best_score = score
                    best_params = params
                    best_performance = {
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_return': float((result.final_capital - result.initial_capital) / result.initial_capital),
                        'max_drawdown': float(result.max_drawdown),
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades
                    }
                    
            except Exception as e:
                logger.error(
                    "optimization_iteration_failed",
                    params=params,
                    error=str(e)
                )
                continue
        
        if best_params is None:
            # Use default parameters if optimization failed
            best_params = dict(zip(param_names, [v[0] for v in param_values]))
            best_performance = {'sharpe_ratio': 0, 'total_return': 0}
        
        return best_params, best_performance
    
    async def _random_search(
        self,
        backtest_engine: Any,
        strategy_class: type,
        start_date: datetime,
        end_date: datetime,
        n_iterations: int = 100
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Perform random search optimization.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            start_date: Period start
            end_date: Period end
            n_iterations: Number of random samples
            
        Returns:
            Tuple of (optimal_parameters, performance_metrics)
        """
        import random
        
        best_score = -float('inf')
        best_params = None
        best_performance = None
        
        for _ in range(n_iterations):
            # Random sample from parameter space
            params = {}
            for param_name, param_values in self.config.parameter_grid.items():
                params[param_name] = random.choice(param_values)
            
            # Create strategy
            strategy = strategy_class(**params)
            
            # Configure backtest
            backtest_engine.config.start_date = start_date
            backtest_engine.config.end_date = end_date
            
            # Run backtest
            try:
                result = await backtest_engine.run_backtest(strategy)
                
                # Extract optimization metric
                if self.config.optimization_metric == "sharpe_ratio":
                    score = result.sharpe_ratio
                elif self.config.optimization_metric == "total_return":
                    score = float((result.final_capital - result.initial_capital) / result.initial_capital)
                else:
                    score = result.sharpe_ratio
                
                # Update best if improved
                if score > best_score and result.total_trades >= self.config.min_trades_required:
                    best_score = score
                    best_params = params
                    best_performance = {
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_return': float((result.final_capital - result.initial_capital) / result.initial_capital),
                        'max_drawdown': float(result.max_drawdown),
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades
                    }
                    
            except Exception as e:
                logger.error(
                    "random_search_iteration_failed",
                    params=params,
                    error=str(e)
                )
                continue
        
        return best_params, best_performance
    
    async def _test_parameters(
        self,
        backtest_engine: Any,
        strategy_class: type,
        params: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """
        Test parameters on out-of-sample data.
        
        Args:
            backtest_engine: Backtesting engine
            strategy_class: Strategy class
            params: Parameters to test
            start_date: Period start
            end_date: Period end
            
        Returns:
            Performance metrics
        """
        # Create strategy with parameters
        strategy = strategy_class(**params)
        
        # Configure backtest
        backtest_engine.config.start_date = start_date
        backtest_engine.config.end_date = end_date
        
        # Run backtest
        try:
            result = await backtest_engine.run_backtest(strategy)
            
            return {
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': float((result.final_capital - result.initial_capital) / result.initial_capital),
                'max_drawdown': float(result.max_drawdown),
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor
            }
        except Exception as e:
            logger.error(
                "out_sample_test_failed",
                params=params,
                error=str(e)
            )
            return {'sharpe_ratio': 0, 'total_return': -1}
    
    def _calculate_degradation(
        self,
        in_sample: Dict[str, float],
        out_sample: Dict[str, float]
    ) -> float:
        """
        Calculate performance degradation from in-sample to out-of-sample.
        
        Args:
            in_sample: In-sample performance
            out_sample: Out-of-sample performance
            
        Returns:
            Degradation score (0 = no degradation, 1 = complete degradation)
        """
        metric = self.config.optimization_metric
        
        if metric in in_sample and metric in out_sample:
            in_value = in_sample[metric]
            out_value = out_sample[metric]
            
            if in_value != 0:
                degradation = (in_value - out_value) / abs(in_value)
                return max(0, min(1, degradation))
        
        return 0.5  # Default to 50% degradation if can't calculate
    
    def _calculate_overfitting_score(
        self,
        in_sample: Dict[str, float],
        out_sample: Dict[str, float]
    ) -> float:
        """
        Calculate overfitting score.
        
        Args:
            in_sample: In-sample performance
            out_sample: Out-of-sample performance
            
        Returns:
            Overfitting score (0 = no overfitting, 1 = severe overfitting)
        """
        scores = []
        
        # Compare multiple metrics
        for metric in ['sharpe_ratio', 'total_return', 'win_rate']:
            if metric in in_sample and metric in out_sample:
                in_val = in_sample[metric]
                out_val = out_sample[metric]
                
                if in_val > 0:
                    ratio = out_val / in_val
                    # Score based on how much worse out-sample is
                    score = max(0, 1 - ratio)
                    scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_results(self, window_results: List[WindowResult]) -> WalkForwardResult:
        """
        Analyze walk-forward results.
        
        Args:
            window_results: List of window results
            
        Returns:
            WalkForwardResult with complete analysis
        """
        # Aggregate out-of-sample performance
        overall_performance = {}
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']
        
        for metric in metrics:
            values = [w.out_sample_performance.get(metric, 0) for w in window_results]
            overall_performance[f'mean_{metric}'] = np.mean(values)
            overall_performance[f'std_{metric}'] = np.std(values)
            overall_performance[f'median_{metric}'] = np.median(values)
        
        # Parameter stability analysis
        parameter_stability = self._analyze_parameter_stability(window_results)
        
        # Overfitting metrics
        overfitting_scores = [w.overfitting_score for w in window_results]
        overfitting_metrics = {
            'mean_overfitting': np.mean(overfitting_scores),
            'max_overfitting': np.max(overfitting_scores),
            'windows_overfitted': sum(1 for s in overfitting_scores if s > 0.5)
        }
        
        # Find best stable parameters
        best_stable_params = self._find_stable_parameters(window_results)
        
        # Degradation pattern
        degradation_pattern = [w.performance_degradation for w in window_results]
        
        # Success rate (windows where out-sample was profitable)
        profitable_windows = sum(
            1 for w in window_results 
            if w.out_sample_performance.get('total_return', 0) > 0
        )
        success_rate = profitable_windows / len(window_results) if window_results else 0
        
        return WalkForwardResult(
            windows=window_results,
            overall_performance=overall_performance,
            parameter_stability=parameter_stability,
            overfitting_metrics=overfitting_metrics,
            best_stable_params=best_stable_params,
            degradation_pattern=degradation_pattern,
            success_rate=success_rate
        )
    
    def _analyze_parameter_stability(
        self,
        window_results: List[WindowResult]
    ) -> Dict[str, float]:
        """
        Analyze parameter stability across windows.
        
        Args:
            window_results: List of window results
            
        Returns:
            Parameter stability scores
        """
        stability = {}
        
        if not window_results:
            return stability
        
        # Get all parameter names
        param_names = list(window_results[0].optimal_params.keys())
        
        for param_name in param_names:
            values = [w.optimal_params.get(param_name) for w in window_results]
            
            # Calculate stability as inverse of coefficient of variation
            if all(isinstance(v, (int, float)) for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    stability[param_name] = 1 / (1 + cv)  # Higher is more stable
                else:
                    stability[param_name] = 0
            else:
                # For non-numeric parameters, use frequency of mode
                from collections import Counter
                counts = Counter(values)
                mode_freq = counts.most_common(1)[0][1]
                stability[param_name] = mode_freq / len(values)
        
        return stability
    
    def _find_stable_parameters(
        self,
        window_results: List[WindowResult]
    ) -> Dict[str, Any]:
        """
        Find most stable parameters across windows.
        
        Args:
            window_results: List of window results
            
        Returns:
            Most stable parameter set
        """
        if not window_results:
            return {}
        
        # Weight parameters by out-of-sample performance
        weighted_params = {}
        param_names = list(window_results[0].optimal_params.keys())
        
        for param_name in param_names:
            values = []
            weights = []
            
            for w in window_results:
                value = w.optimal_params.get(param_name)
                weight = w.out_sample_performance.get('sharpe_ratio', 0) + 1  # Avoid negative weights
                
                values.append(value)
                weights.append(weight)
            
            # Calculate weighted average for numeric parameters
            if all(isinstance(v, (int, float)) for v in values):
                weighted_avg = np.average(values, weights=weights)
                weighted_params[param_name] = type(values[0])(weighted_avg)
            else:
                # For non-numeric, use mode
                from collections import Counter
                counts = Counter(values)
                weighted_params[param_name] = counts.most_common(1)[0][0]
        
        return weighted_params