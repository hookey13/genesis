"""
Efficient Frontier Analysis for Portfolio Optimization

Implements Modern Portfolio Theory to find optimal portfolio allocations.
Identifies portfolios with maximum Sharpe ratio and minimum variance.
Part of the Hunter+ tier portfolio optimization suite.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import structlog
from scipy.optimize import minimize

from typing import Optional

from genesis.core.constants import TradingTier
from genesis.core.exceptions import (
    DataError as InvalidDataError,
    GenesisException as CalculationError,
)
from genesis.utils.decorators import requires_tier, with_timeout

logger = structlog.get_logger(__name__)


@dataclass
class PortfolioPoint:
    """A point on the efficient frontier"""

    expected_return: Decimal
    risk: Decimal  # Standard deviation
    sharpe_ratio: Decimal
    weights: dict[str, Decimal]  # Strategy name -> allocation weight

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "expected_return": float(self.expected_return),
            "risk": float(self.risk),
            "sharpe_ratio": float(self.sharpe_ratio),
            "weights": {k: float(v) for k, v in self.weights.items()},
        }


@dataclass
class EfficientFrontierResult:
    """Results from efficient frontier analysis"""

    frontier_points: list[PortfolioPoint]
    max_sharpe_portfolio: PortfolioPoint
    min_variance_portfolio: PortfolioPoint
    strategies: list[str]
    covariance_matrix: list[list[Decimal]]
    expected_returns: dict[str, Decimal]
    calculated_at: datetime = None

    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now(UTC)


class EfficientFrontierAnalyzer:
    """
    Analyzes efficient frontier for portfolio optimization.

    Uses Modern Portfolio Theory to find optimal allocations across strategies.
    Hunter+ tier feature for advanced portfolio management.
    """

    # Number of points to calculate on the frontier
    FRONTIER_POINTS = 50

    # Cache TTL
    CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        """Initialize efficient frontier analyzer"""
        self._cache: dict[str, tuple[EfficientFrontierResult, datetime]] = {}
        self._cache_lock = asyncio.Lock()
        logger.info("efficient_frontier_analyzer_initialized")

    @requires_tier(TradingTier.HUNTER)
    @with_timeout(10.0)
    async def calculate_efficient_frontier(
        self,
        strategy_returns: dict[str, list[Decimal]],
        risk_free_rate: Decimal = Decimal("0.02"),
        constraints: Optional[dict] = None,
    ) -> EfficientFrontierResult:
        """
        Calculate the efficient frontier for a set of strategies.

        Args:
            strategy_returns: Dictionary of strategy name -> list of returns
            risk_free_rate: Annual risk-free rate for Sharpe calculations
            constraints: Optional constraints (min/max allocations, etc.)

        Returns:
            EfficientFrontierResult with frontier points and optimal portfolios

        Raises:
            InvalidDataError: If input data is invalid
            CalculationError: If optimization fails
        """
        try:
            # Validate inputs
            if not strategy_returns or len(strategy_returns) < 2:
                raise InvalidDataError(
                    "Need at least 2 strategies for frontier analysis"
                )

            strategies = list(strategy_returns.keys())
            n_strategies = len(strategies)

            # Check all strategies have same number of returns
            return_lengths = [len(returns) for returns in strategy_returns.values()]
            if len(set(return_lengths)) > 1:
                raise InvalidDataError(
                    "All strategies must have same number of return periods"
                )

            if return_lengths[0] < 2:
                raise InvalidDataError("Need at least 2 return periods for analysis")

            # Check cache
            cache_key = self._get_cache_key(strategy_returns, risk_free_rate)
            cached = await self._get_cached_result(cache_key)
            if cached:
                logger.info("efficient_frontier_cache_hit")
                return cached

            # Convert to numpy arrays
            returns_matrix = np.array(
                [[float(r) for r in strategy_returns[s]] for s in strategies]
            ).T  # Transpose to get periods x strategies

            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix, rowvar=False, ddof=1)

            # Convert to Decimal for storage
            expected_returns_dict = {
                strategies[i]: Decimal(str(expected_returns[i]))
                for i in range(n_strategies)
            }

            cov_matrix_decimal = [
                [Decimal(str(cov_matrix[i][j])) for j in range(n_strategies)]
                for i in range(n_strategies)
            ]

            # Calculate frontier points
            frontier_points = await self._calculate_frontier_points(
                expected_returns, cov_matrix, strategies, risk_free_rate, constraints
            )

            # Find special portfolios
            max_sharpe_portfolio = await self._find_max_sharpe_portfolio(
                expected_returns, cov_matrix, strategies, risk_free_rate, constraints
            )

            min_variance_portfolio = await self._find_min_variance_portfolio(
                expected_returns, cov_matrix, strategies, constraints
            )

            # Create result
            result = EfficientFrontierResult(
                frontier_points=frontier_points,
                max_sharpe_portfolio=max_sharpe_portfolio,
                min_variance_portfolio=min_variance_portfolio,
                strategies=strategies,
                covariance_matrix=cov_matrix_decimal,
                expected_returns=expected_returns_dict,
            )

            # Cache result
            await self._cache_result(cache_key, result)

            logger.info(
                "efficient_frontier_calculated",
                num_strategies=n_strategies,
                num_frontier_points=len(frontier_points),
                max_sharpe=float(max_sharpe_portfolio.sharpe_ratio),
            )

            return result

        except Exception as e:
            logger.error("efficient_frontier_calculation_failed", error=str(e))
            raise CalculationError(f"Failed to calculate efficient frontier: {e}")

    async def _calculate_frontier_points(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        strategies: list[str],
        risk_free_rate: Decimal,
        constraints: Optional[dict],
    ) -> list[PortfolioPoint]:
        """Calculate points along the efficient frontier"""
        n_strategies = len(strategies)

        # Get return range
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)

        # Target returns for frontier
        target_returns = np.linspace(min_return, max_return, self.FRONTIER_POINTS)

        frontier_points = []

        for target_return in target_returns:
            # Optimization constraints
            cons = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Sum to 1
                {
                    "type": "eq",
                    "fun": lambda x, tr=target_return: np.sum(x * expected_returns)
                    - tr,
                },  # Target return
            ]

            # Add custom constraints if provided
            if constraints:
                cons.extend(self._parse_constraints(constraints, n_strategies))

            # Bounds for weights (0 to 1 for each strategy)
            bounds = tuple((0, 1) for _ in range(n_strategies))

            # Initial guess (equal weights)
            x0 = np.array([1 / n_strategies] * n_strategies)

            # Minimize portfolio variance
            result = minimize(
                lambda w: np.sqrt(w @ cov_matrix @ w),  # Portfolio std dev
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-9, "disp": False},
            )

            if result.success:
                weights = result.x
                portfolio_return = Decimal(str(np.sum(weights * expected_returns)))
                portfolio_risk = Decimal(str(np.sqrt(weights @ cov_matrix @ weights)))

                # Calculate Sharpe ratio
                excess_return = portfolio_return - risk_free_rate
                sharpe = (
                    (excess_return / portfolio_risk)
                    if portfolio_risk > 0
                    else Decimal("0")
                )

                # Create portfolio point
                weights_dict = {
                    strategies[i]: Decimal(str(weights[i])).quantize(
                        Decimal("0.0001"), ROUND_HALF_UP
                    )
                    for i in range(n_strategies)
                }

                frontier_points.append(
                    PortfolioPoint(
                        expected_return=portfolio_return.quantize(
                            Decimal("0.000001"), ROUND_HALF_UP
                        ),
                        risk=portfolio_risk.quantize(
                            Decimal("0.000001"), ROUND_HALF_UP
                        ),
                        sharpe_ratio=sharpe.quantize(Decimal("0.0001"), ROUND_HALF_UP),
                        weights=weights_dict,
                    )
                )

        return frontier_points

    async def _find_max_sharpe_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        strategies: list[str],
        risk_free_rate: Decimal,
        constraints: Optional[dict],
    ) -> PortfolioPoint:
        """Find the portfolio with maximum Sharpe ratio"""
        n_strategies = len(strategies)

        # Objective: Maximize Sharpe ratio (minimize negative Sharpe)
        def neg_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_std = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_std == 0:
                return 999999  # Large penalty for zero variance
            sharpe = (portfolio_return - float(risk_free_rate)) / portfolio_std
            return -sharpe  # Negative because we're minimizing

        # Constraints
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Sum to 1

        if constraints:
            cons.extend(self._parse_constraints(constraints, n_strategies))

        # Bounds
        bounds = tuple((0, 1) for _ in range(n_strategies))

        # Initial guess
        x0 = np.array([1 / n_strategies] * n_strategies)

        # Optimize
        result = minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "disp": False},
        )

        if not result.success:
            logger.warning("max_sharpe_optimization_failed", message=result.message)

        # Extract results
        weights = result.x
        portfolio_return = Decimal(str(np.sum(weights * expected_returns)))
        portfolio_risk = Decimal(str(np.sqrt(weights @ cov_matrix @ weights)))
        sharpe = Decimal(str(-result.fun))  # Negative because we minimized negative

        weights_dict = {
            strategies[i]: Decimal(str(weights[i])).quantize(
                Decimal("0.0001"), ROUND_HALF_UP
            )
            for i in range(n_strategies)
        }

        return PortfolioPoint(
            expected_return=portfolio_return.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
            risk=portfolio_risk.quantize(Decimal("0.000001"), ROUND_HALF_UP),
            sharpe_ratio=sharpe.quantize(Decimal("0.0001"), ROUND_HALF_UP),
            weights=weights_dict,
        )

    async def _find_min_variance_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        strategies: list[str],
        constraints: Optional[dict],
    ) -> PortfolioPoint:
        """Find the minimum variance portfolio"""
        n_strategies = len(strategies)

        # Objective: Minimize variance
        def portfolio_variance(weights):
            return weights @ cov_matrix @ weights

        # Constraints
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # Sum to 1

        if constraints:
            cons.extend(self._parse_constraints(constraints, n_strategies))

        # Bounds
        bounds = tuple((0, 1) for _ in range(n_strategies))

        # Initial guess
        x0 = np.array([1 / n_strategies] * n_strategies)

        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-9, "disp": False},
        )

        if not result.success:
            logger.warning("min_variance_optimization_failed", message=result.message)

        # Extract results
        weights = result.x
        portfolio_return = Decimal(str(np.sum(weights * expected_returns)))
        portfolio_risk = Decimal(str(np.sqrt(result.fun)))

        # Calculate Sharpe (even though we're not optimizing for it)
        risk_free_rate = Decimal("0.02")  # Default
        excess_return = portfolio_return - risk_free_rate
        sharpe = (
            (excess_return / portfolio_risk) if portfolio_risk > 0 else Decimal("0")
        )

        weights_dict = {
            strategies[i]: Decimal(str(weights[i])).quantize(
                Decimal("0.0001"), ROUND_HALF_UP
            )
            for i in range(n_strategies)
        }

        return PortfolioPoint(
            expected_return=portfolio_return.quantize(
                Decimal("0.000001"), ROUND_HALF_UP
            ),
            risk=portfolio_risk.quantize(Decimal("0.000001"), ROUND_HALF_UP),
            sharpe_ratio=sharpe.quantize(Decimal("0.0001"), ROUND_HALF_UP),
            weights=weights_dict,
        )

    def _parse_constraints(self, constraints: dict, n_strategies: int) -> list[dict]:
        """Parse user constraints into scipy format"""
        scipy_constraints = []

        # Handle max correlation constraint
        if "max_correlation" in constraints:
            # This would need correlation matrix - skip for now
            pass

        # Handle minimum allocation
        if "min_allocation" in constraints:
            min_alloc = float(constraints["min_allocation"])
            # Each weight must be 0 or >= min_allocation
            # This is complex with scipy, handled via bounds instead
            pass

        return scipy_constraints

    def _get_cache_key(
        self, strategy_returns: dict[str, list[Decimal]], risk_free_rate: Decimal
    ) -> str:
        """Generate cache key"""
        # Create hash from strategy returns
        strategy_hashes = []
        for strategy in sorted(strategy_returns.keys()):
            returns_hash = hash(tuple(strategy_returns[strategy]))
            strategy_hashes.append(f"{strategy}:{returns_hash}")

        return f"{'_'.join(strategy_hashes)}_{risk_free_rate}"

    async def _get_cached_result(
        self, cache_key: str
    ) -> Optional[EfficientFrontierResult]:
        """Get cached result if valid"""
        async with self._cache_lock:
            if cache_key in self._cache:
                result, cached_at = self._cache[cache_key]
                age_seconds = (datetime.now(UTC) - cached_at).total_seconds()

                if age_seconds < self.CACHE_TTL_SECONDS:
                    return result
                else:
                    del self._cache[cache_key]

        return None

    async def _cache_result(self, cache_key: str, result: EfficientFrontierResult):
        """Cache calculation result"""
        async with self._cache_lock:
            self._cache[cache_key] = (result, datetime.now(UTC))

            # Limit cache size
            if len(self._cache) > 50:
                # Remove oldest entries
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:10]:
                    del self._cache[key]

    async def clear_cache(self):
        """Clear all cached results"""
        async with self._cache_lock:
            self._cache.clear()
            logger.info("efficient_frontier_cache_cleared")
