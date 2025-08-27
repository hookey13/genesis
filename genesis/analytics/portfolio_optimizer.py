"""
Portfolio Optimizer - Main Integration Module

Integrates Sharpe ratio, efficient frontier, and rebalancing components.
Implements complete portfolio optimization workflow with validation.
Part of the Hunter+ tier advanced trading features.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import structlog
import yaml

from genesis.analytics.efficient_frontier import EfficientFrontierAnalyzer
from typing import Optional

from genesis.analytics.rebalancing_engine import RebalancingEngine
from genesis.analytics.sharpe_ratio import SharpeRatioCalculator
from genesis.core.constants import TradingTier
from genesis.core.events import Event, EventType
from genesis.core.exceptions import (
    DataError as InvalidDataError,
    GenesisException as CalculationError,
)
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier, with_timeout

logger = structlog.get_logger(__name__)


@dataclass
class Strategy:
    """Represents a trading strategy"""

    name: str
    returns: list[Decimal]
    current_allocation: Decimal
    min_allocation: Optional[Decimal] = None
    max_allocation: Optional[Decimal] = None
    is_active: bool = True


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization"""

    min_allocation: Decimal = Decimal("0.01")  # 1% minimum
    max_allocation: Decimal = Decimal("0.40")  # 40% maximum
    max_correlation: Decimal = Decimal("0.60")  # 60% max correlation
    min_strategies: int = 2  # Minimum active strategies
    tier_limits: Optional[dict] = None  # Tier-based position limits


@dataclass
class ValidationResult:
    """Result of out-of-sample validation"""

    in_sample_sharpe: Decimal
    out_sample_sharpe: Decimal
    performance_degradation: Decimal  # Percentage degradation
    is_robust: bool  # Passes robustness threshold
    walk_forward_results: list[dict]


@dataclass
class OptimizationResult:
    """Complete portfolio optimization result"""

    optimal_weights: dict[str, Decimal]
    expected_return: Decimal
    expected_risk: Decimal
    sharpe_ratio: Decimal
    correlation_matrix: dict[tuple[str, str], Decimal]
    validation: Optional[ValidationResult]
    rebalance_recommendation: Optional[dict]
    optimization_time_ms: int
    calculated_at: datetime = None

    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now(UTC)


class PortfolioOptimizer:
    """
    Main portfolio optimization engine.

    Coordinates all optimization components and enforces constraints.
    Hunter+ tier feature for advanced portfolio management.
    """

    # Performance requirements
    MAX_OPTIMIZATION_TIME_MS = 1000  # 1 second for 10 strategies

    # Validation parameters
    OUT_OF_SAMPLE_RATIO = 0.3  # 30% for out-of-sample
    MAX_DEGRADATION_THRESHOLD = Decimal("0.20")  # 20% max degradation
    WALK_FORWARD_WINDOWS = 5

    def __init__(
        self, event_bus: Optional[EventBus] = None, config_path: Optional[str] = None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            event_bus: Optional event bus for real-time updates
            config_path: Path to configuration file
        """
        self.event_bus = event_bus

        # Initialize components
        self.sharpe_calculator = SharpeRatioCalculator()
        self.frontier_analyzer = EfficientFrontierAnalyzer()

        # Load configuration
        self.config = self._load_configuration(config_path)
        self.constraints = self._parse_constraints(self.config)

        # Initialize rebalancing engine with config
        rebalance_config = self.config.get("rebalancing", {})
        self.rebalancing_engine = RebalancingEngine(event_bus, rebalance_config)

        # Cache for performance
        self._correlation_cache: dict[str, np.ndarray] = {}
        self._cache_lock = asyncio.Lock()

        logger.info(
            "portfolio_optimizer_initialized",
            min_allocation=float(self.constraints.min_allocation),
            max_allocation=float(self.constraints.max_allocation),
        )

    @requires_tier(TradingTier.HUNTER)
    @with_timeout(10.0)
    async def optimize_portfolio(
        self,
        strategies: list[Strategy],
        portfolio_value_usdt: Decimal,
        validate: bool = True,
        rebalance_check: bool = True,
    ) -> OptimizationResult:
        """
        Perform complete portfolio optimization.

        Args:
            strategies: List of strategies with returns data
            portfolio_value_usdt: Total portfolio value in USDT
            validate: Whether to perform out-of-sample validation
            rebalance_check: Whether to check rebalancing triggers

        Returns:
            OptimizationResult with optimal allocations and analysis

        Raises:
            InvalidDataError: If input data is invalid
            CalculationError: If optimization fails
        """
        start_time = datetime.now(UTC)

        try:
            # Validate inputs
            await self._validate_strategies(strategies)

            # Filter active strategies and apply minimum allocation
            active_strategies = await self._apply_minimum_allocations(
                strategies, portfolio_value_usdt
            )

            if len(active_strategies) < self.constraints.min_strategies:
                raise InvalidDataError(
                    f"Need at least {self.constraints.min_strategies} active strategies"
                )

            # Calculate correlations
            correlation_matrix = await self._calculate_correlations(active_strategies)

            # Check correlation constraints
            await self._check_correlation_constraints(
                correlation_matrix, active_strategies
            )

            # Prepare returns data
            strategy_returns = {s.name: s.returns for s in active_strategies}

            # Calculate efficient frontier
            frontier_result = await self.frontier_analyzer.calculate_efficient_frontier(
                strategy_returns,
                risk_free_rate=Decimal(str(self.config.get("risk_free_rate", 0.02))),
                constraints=self._get_optimization_constraints(),
            )

            # Get optimal portfolio (max Sharpe)
            optimal_portfolio = frontier_result.max_sharpe_portfolio

            # Adjust weights for minimum allocations
            adjusted_weights = await self._adjust_for_minimums(
                optimal_portfolio.weights, portfolio_value_usdt
            )

            # Perform out-of-sample validation if requested
            validation_result = None
            if validate:
                validation_result = await self._validate_out_of_sample(
                    active_strategies, adjusted_weights
                )

            # Check rebalancing if requested
            rebalance_recommendation = None
            if rebalance_check:
                current_weights = {s.name: s.current_allocation for s in strategies}
                recommendation = await self.rebalancing_engine.check_rebalance_triggers(
                    current_weights,
                    adjusted_weights,
                    portfolio_value_usdt,
                    optimal_portfolio.sharpe_ratio,
                )
                rebalance_recommendation = (
                    await self.rebalancing_engine.generate_weekly_recommendation(
                        current_weights,
                        adjusted_weights,
                        portfolio_value_usdt,
                        {"current_sharpe": float(optimal_portfolio.sharpe_ratio)},
                    )
                )

            # Calculate optimization time
            optimization_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )

            # Check performance requirement
            if optimization_time_ms > self.MAX_OPTIMIZATION_TIME_MS:
                logger.warning(
                    "optimization_time_exceeded",
                    time_ms=optimization_time_ms,
                    limit_ms=self.MAX_OPTIMIZATION_TIME_MS,
                )

            # Create result
            result = OptimizationResult(
                optimal_weights=adjusted_weights,
                expected_return=optimal_portfolio.expected_return,
                expected_risk=optimal_portfolio.risk,
                sharpe_ratio=optimal_portfolio.sharpe_ratio,
                correlation_matrix=self._format_correlation_matrix(
                    correlation_matrix, active_strategies
                ),
                validation=validation_result,
                rebalance_recommendation=rebalance_recommendation,
                optimization_time_ms=optimization_time_ms,
            )

            # Send event if connected
            if self.event_bus:
                await self._send_optimization_event(result)

            logger.info(
                "portfolio_optimization_complete",
                num_strategies=len(active_strategies),
                sharpe_ratio=float(result.sharpe_ratio),
                optimization_time_ms=optimization_time_ms,
            )

            return result

        except Exception as e:
            logger.error("portfolio_optimization_failed", error=str(e))
            raise CalculationError(f"Portfolio optimization failed: {e}")

    async def _validate_strategies(self, strategies: list[Strategy]):
        """Validate strategy data"""
        for strategy in strategies:
            if not strategy.returns or len(strategy.returns) < 30:
                raise InvalidDataError(
                    f"Strategy {strategy.name} needs at least 30 return periods"
                )

            # Check for NaN or infinite values
            for ret in strategy.returns:
                if ret is None or abs(ret) > Decimal("100"):
                    raise InvalidDataError(
                        f"Invalid return value in strategy {strategy.name}"
                    )

    async def _apply_minimum_allocations(
        self, strategies: list[Strategy], portfolio_value_usdt: Decimal
    ) -> list[Strategy]:
        """Filter strategies that meet minimum allocation requirements"""
        active_strategies = []
        min_value = self.constraints.min_allocation * portfolio_value_usdt

        for strategy in strategies:
            if not strategy.is_active:
                continue

            # Check if strategy meets minimum
            current_value = strategy.current_allocation * portfolio_value_usdt

            # Use strategy-specific minimum if set
            strategy_min = strategy.min_allocation or self.constraints.min_allocation
            strategy_min_value = strategy_min * portfolio_value_usdt

            if current_value >= strategy_min_value or current_value == Decimal("0"):
                # Include if above minimum or new strategy
                active_strategies.append(strategy)
            else:
                logger.info(
                    "strategy_below_minimum",
                    strategy=strategy.name,
                    current_value=float(current_value),
                    min_value=float(strategy_min_value),
                )

        return active_strategies

    async def _calculate_correlations(self, strategies: list[Strategy]) -> np.ndarray:
        """Calculate correlation matrix between strategies"""
        # Create returns matrix
        returns_matrix = np.array([[float(r) for r in s.returns] for s in strategies])

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns_matrix)

        return correlation_matrix

    async def _check_correlation_constraints(
        self, correlation_matrix: np.ndarray, strategies: list[Strategy]
    ):
        """Check if correlations exceed maximum threshold"""
        n = len(strategies)
        max_corr = float(self.constraints.max_correlation)

        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(correlation_matrix[i, j])
                if corr > max_corr:
                    logger.warning(
                        "high_correlation_detected",
                        strategy1=strategies[i].name,
                        strategy2=strategies[j].name,
                        correlation=corr,
                    )

    def _format_correlation_matrix(
        self, correlation_matrix: np.ndarray, strategies: list[Strategy]
    ) -> dict[tuple[str, str], Decimal]:
        """Format correlation matrix for output"""
        result = {}
        n = len(strategies)

        for i in range(n):
            for j in range(n):
                key = (strategies[i].name, strategies[j].name)
                value = Decimal(str(correlation_matrix[i, j])).quantize(
                    Decimal("0.0001"), ROUND_HALF_UP
                )
                result[key] = value

        return result

    async def _adjust_for_minimums(
        self, weights: dict[str, Decimal], portfolio_value_usdt: Decimal
    ) -> dict[str, Decimal]:
        """Adjust weights to respect minimum allocation constraints"""
        adjusted_weights = {}
        total_weight = Decimal("0")

        # First pass: apply minimums and round to exchange requirements
        for strategy, weight in weights.items():
            # Round to 4 decimal places (0.01% precision)
            rounded_weight = weight.quantize(Decimal("0.0001"), ROUND_HALF_UP)

            # Apply minimum if non-zero
            if (
                rounded_weight > Decimal("0")
                and rounded_weight < self.constraints.min_allocation
            ):
                rounded_weight = self.constraints.min_allocation

            adjusted_weights[strategy] = rounded_weight
            total_weight += rounded_weight

        # Second pass: normalize to sum to 1
        if total_weight > Decimal("0"):
            for strategy in adjusted_weights:
                adjusted_weights[strategy] = (
                    adjusted_weights[strategy] / total_weight
                ).quantize(Decimal("0.0001"), ROUND_HALF_UP)

        return adjusted_weights

    async def _validate_out_of_sample(
        self, strategies: list[Strategy], optimal_weights: dict[str, Decimal]
    ) -> ValidationResult:
        """Perform out-of-sample validation"""
        # Split data
        split_point = int(len(strategies[0].returns) * (1 - self.OUT_OF_SAMPLE_RATIO))

        # In-sample optimization
        in_sample_returns = {s.name: s.returns[:split_point] for s in strategies}

        in_sample_frontier = await self.frontier_analyzer.calculate_efficient_frontier(
            in_sample_returns
        )
        in_sample_sharpe = in_sample_frontier.max_sharpe_portfolio.sharpe_ratio

        # Out-of-sample testing
        out_sample_returns = []
        for i in range(split_point, len(strategies[0].returns)):
            period_return = sum(
                strategies[j].returns[i]
                * optimal_weights.get(strategies[j].name, Decimal("0"))
                for j in range(len(strategies))
            )
            out_sample_returns.append(period_return)

        # Calculate out-of-sample Sharpe
        out_sample_result = await self.sharpe_calculator.calculate_sharpe_ratio(
            out_sample_returns, confidence_level=0  # Skip CI for validation
        )

        # Calculate degradation
        degradation = (
            (in_sample_sharpe - out_sample_result.sharpe_ratio) / in_sample_sharpe
            if in_sample_sharpe > Decimal("0")
            else Decimal("0")
        )

        # Walk-forward analysis
        walk_forward_results = await self._walk_forward_analysis(
            strategies, self.WALK_FORWARD_WINDOWS
        )

        return ValidationResult(
            in_sample_sharpe=in_sample_sharpe,
            out_sample_sharpe=out_sample_result.sharpe_ratio,
            performance_degradation=degradation.quantize(
                Decimal("0.0001"), ROUND_HALF_UP
            ),
            is_robust=degradation <= self.MAX_DEGRADATION_THRESHOLD,
            walk_forward_results=walk_forward_results,
        )

    async def _walk_forward_analysis(
        self, strategies: list[Strategy], num_windows: int
    ) -> list[dict]:
        """Perform walk-forward analysis for robustness testing"""
        results = []
        window_size = len(strategies[0].returns) // (num_windows + 1)

        for window in range(num_windows):
            # Training window
            train_start = window * (window_size // 2)
            train_end = train_start + window_size

            # Test window
            test_start = train_end
            test_end = min(test_start + (window_size // 2), len(strategies[0].returns))

            # Skip if not enough data
            if test_end <= test_start:
                break

            # Optimize on training data
            train_returns = {
                s.name: s.returns[train_start:train_end] for s in strategies
            }

            train_frontier = await self.frontier_analyzer.calculate_efficient_frontier(
                train_returns
            )
            optimal_weights = train_frontier.max_sharpe_portfolio.weights

            # Test on out-of-sample data
            test_portfolio_returns = []
            for i in range(test_start, test_end):
                period_return = sum(
                    strategies[j].returns[i]
                    * optimal_weights.get(strategies[j].name, Decimal("0"))
                    for j in range(len(strategies))
                )
                test_portfolio_returns.append(period_return)

            if test_portfolio_returns:
                test_sharpe_result = (
                    await self.sharpe_calculator.calculate_sharpe_ratio(
                        test_portfolio_returns, confidence_level=0
                    )
                )

                results.append(
                    {
                        "window": window,
                        "train_periods": train_end - train_start,
                        "test_periods": test_end - test_start,
                        "train_sharpe": float(
                            train_frontier.max_sharpe_portfolio.sharpe_ratio
                        ),
                        "test_sharpe": float(test_sharpe_result.sharpe_ratio),
                    }
                )

        return results

    def _load_configuration(self, config_path: Optional[str]) -> dict:
        """Load configuration from file"""
        if not config_path:
            config_path = "config/trading_rules.yaml"

        path = Path(config_path)
        if not path.exists():
            logger.warning("config_file_not_found", path=config_path)
            return {}

        try:
            with open(path) as f:
                config = yaml.safe_load(f)
                return config.get("portfolio_optimization", {})
        except Exception as e:
            logger.error("config_load_failed", error=str(e))
            return {}

    def _parse_constraints(self, config: dict) -> OptimizationConstraints:
        """Parse constraints from configuration"""
        constraints_config = config.get("constraints", {})

        return OptimizationConstraints(
            min_allocation=Decimal(str(constraints_config.get("min_allocation", 0.01))),
            max_allocation=Decimal(str(constraints_config.get("max_allocation", 0.40))),
            max_correlation=Decimal(
                str(constraints_config.get("max_correlation", 0.60))
            ),
            min_strategies=constraints_config.get("min_strategies", 2),
            tier_limits=constraints_config.get("tier_limits"),
        )

    def _get_optimization_constraints(self) -> dict:
        """Get constraints in format for optimizer"""
        return {
            "min_allocation": float(self.constraints.min_allocation),
            "max_allocation": float(self.constraints.max_allocation),
            "max_correlation": float(self.constraints.max_correlation),
        }

    async def _send_optimization_event(self, result: OptimizationResult):
        """Send optimization event to event bus"""
        if not self.event_bus:
            return

        event = Event(
            type=EventType.PORTFOLIO_OPTIMIZED,
            data={
                "sharpe_ratio": float(result.sharpe_ratio),
                "expected_return": float(result.expected_return),
                "expected_risk": float(result.expected_risk),
                "num_strategies": len(result.optimal_weights),
                "optimization_time_ms": result.optimization_time_ms,
                "is_robust": result.validation.is_robust if result.validation else None,
            },
        )

        await self.event_bus.publish(event)
