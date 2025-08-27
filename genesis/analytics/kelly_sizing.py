"""
Kelly Criterion-based position sizing calculator.

Implements fractional Kelly sizing for optimal portfolio growth while managing risk.
Hunter+ tier feature for sophisticated position sizing.
"""
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from enum import Enum

import numpy as np
from scipy import stats

from genesis.core.constants import ConvictionLevel, TradingTier
from genesis.core.models import Trade
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyEdge:
    """Edge calculation for a specific strategy."""

    strategy_id: str
    win_rate: Decimal
    win_loss_ratio: Decimal
    sample_size: int
    confidence_interval: tuple[Decimal, Decimal]
    last_calculated: datetime


@dataclass
class KellyParams:
    """Parameters for Kelly Criterion calculation."""

    kelly_fraction: Decimal
    fractional_multiplier: Decimal
    final_fraction: Decimal
    position_size: Decimal
    confidence_level: Decimal


@dataclass
class SimulationResult:
    """Result from Monte Carlo simulation."""

    optimal_kelly: Decimal
    risk_of_ruin: Decimal
    expected_growth_rate: Decimal
    median_final_balance: Decimal
    percentile_5: Decimal
    percentile_95: Decimal
    paths: Optional[np.ndarray] = None


class VolatilityRegime(Enum):
    """Market volatility regimes."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class KellyCalculator:
    """
    Kelly Criterion-based position sizing calculator.
    
    Provides optimal position sizing based on historical performance,
    with fractional Kelly implementation for safety.
    """

    def __init__(
        self,
        default_fraction: Decimal = Decimal("0.25"),
        min_trades: int = 20,
        lookback_days: int = 30,
        max_kelly: Decimal = Decimal("0.5")
    ):
        """
        Initialize Kelly calculator.
        
        Args:
            default_fraction: Default fractional Kelly multiplier (safety factor)
            min_trades: Minimum trades required for Kelly calculation
            lookback_days: Days to look back for performance metrics
            max_kelly: Maximum allowed Kelly fraction (safety cap)
        """
        self.default_fraction = default_fraction
        self.min_trades = min_trades
        self.lookback_days = lookback_days
        self.max_kelly = max_kelly
        self._strategy_edges: dict[str, StrategyEdge] = {}
        self._cache_ttl = timedelta(minutes=1)

    def calculate_kelly_fraction(
        self,
        win_rate: Decimal,
        win_loss_ratio: Decimal
    ) -> Decimal:
        """
        Calculate raw Kelly fraction.
        
        Kelly formula: f* = (p * b - q) / b
        Where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio
        
        Args:
            win_rate: Probability of winning (0 to 1)
            win_loss_ratio: Average win / average loss ratio
            
        Returns:
            Kelly fraction (capped at max_kelly)
        """
        if win_loss_ratio <= 0:
            logger.warning("Invalid win/loss ratio: %s", win_loss_ratio)
            return Decimal("0")

        # Kelly formula
        p = win_rate
        q = Decimal("1") - p
        b = win_loss_ratio

        # Handle edge cases
        if b == 0:
            return Decimal("0")

        kelly_f = (p * b - q) / b

        # Cap Kelly fraction for safety
        if kelly_f < 0:
            return Decimal("0")
        elif kelly_f > self.max_kelly:
            logger.info("Capping Kelly fraction from %s to %s", kelly_f, self.max_kelly)
            return self.max_kelly

        return kelly_f.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)

    def calculate_position_size(
        self,
        kelly_f: Decimal,
        balance: Decimal,
        fraction: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate position size using fractional Kelly.
        
        Args:
            kelly_f: Raw Kelly fraction
            balance: Account balance
            fraction: Fractional Kelly multiplier (default: 0.25)
            
        Returns:
            Position size in base currency
        """
        if fraction is None:
            fraction = self.default_fraction

        # Apply fractional Kelly for safety
        fractional_kelly = kelly_f * fraction

        # Calculate position size
        position_size = balance * fractional_kelly

        # Round down to avoid exceeding limits
        return position_size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def estimate_edge(
        self,
        trades: list[Trade],
        confidence_level: Decimal = Decimal("0.95")
    ) -> dict[str, Decimal]:
        """
        Estimate trading edge from historical trades.
        
        Args:
            trades: List of completed trades
            confidence_level: Confidence level for interval estimation
            
        Returns:
            Dictionary with win_rate, win_loss_ratio, and confidence metrics
        """
        if not trades:
            return {
                "win_rate": Decimal("0"),
                "win_loss_ratio": Decimal("0"),
                "sample_size": 0,
                "confidence": Decimal("0")
            }

        wins = []
        losses = []

        for trade in trades:
            pnl = trade.pnl_dollars
            if pnl > 0:
                wins.append(float(pnl))
            elif pnl < 0:
                losses.append(abs(float(pnl)))

        # Calculate win rate
        total_trades = len(trades)
        win_rate = Decimal(str(len(wins) / total_trades)) if total_trades > 0 else Decimal("0")

        # Calculate win/loss ratio
        avg_win = Decimal(str(np.mean(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(np.mean(losses))) if losses else Decimal("1")
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal("0")

        # Calculate confidence interval for win rate
        confidence = self._calculate_confidence(
            win_rate,
            total_trades,
            confidence_level
        )

        return {
            "win_rate": win_rate.quantize(Decimal("0.0001")),
            "win_loss_ratio": win_loss_ratio.quantize(Decimal("0.01")),
            "sample_size": total_trades,
            "confidence": confidence
        }

    def _calculate_confidence(
        self,
        win_rate: Decimal,
        sample_size: int,
        confidence_level: Decimal
    ) -> Decimal:
        """
        Calculate confidence interval for win rate.
        
        Uses Wilson score interval for binomial proportion.
        """
        if sample_size < self.min_trades:
            return Decimal("0")

        p = float(win_rate)
        n = sample_size
        z = stats.norm.ppf(float((1 + confidence_level) / 2))

        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

        lower = max(0, center - margin)
        upper = min(1, center + margin)

        # Return confidence as width of interval (narrower = more confident)
        confidence_score = 1 - (upper - lower)
        return Decimal(str(confidence_score)).quantize(Decimal("0.0001"))

    def calculate_strategy_edge(
        self,
        strategy_id: str,
        trades: list[Trade],
        window_days: Optional[int] = None
    ) -> StrategyEdge:
        """
        Calculate edge for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            trades: Historical trades for the strategy
            window_days: Days to look back (default: lookback_days)
            
        Returns:
            StrategyEdge with calculated metrics
        """
        if window_days is None:
            window_days = self.lookback_days

        # Check cache
        cached_edge = self._strategy_edges.get(strategy_id)
        if cached_edge:
            age = datetime.now(UTC) - cached_edge.last_calculated
            if age < self._cache_ttl:
                return cached_edge

        # Filter trades by window
        cutoff_date = datetime.now(UTC) - timedelta(days=window_days)
        recent_trades = [
            t for t in trades
            if t.timestamp >= cutoff_date
        ]

        # Estimate edge
        edge_metrics = self.estimate_edge(recent_trades)

        # Calculate confidence interval
        win_rate = edge_metrics["win_rate"]
        sample_size = edge_metrics["sample_size"]

        if sample_size >= self.min_trades:
            z = 1.96  # 95% confidence
            se = float(np.sqrt(float(win_rate * (1 - win_rate)) / sample_size))
            lower = Decimal(str(max(0, float(win_rate) - z * se)))
            upper = Decimal(str(min(1, float(win_rate) + z * se)))
            confidence_interval = (lower.quantize(Decimal("0.0001")),
                                 upper.quantize(Decimal("0.0001")))
        else:
            confidence_interval = (Decimal("0"), Decimal("1"))

        # Create and cache edge
        strategy_edge = StrategyEdge(
            strategy_id=strategy_id,
            win_rate=edge_metrics["win_rate"],
            win_loss_ratio=edge_metrics["win_loss_ratio"],
            sample_size=edge_metrics["sample_size"],
            confidence_interval=confidence_interval,
            last_calculated=datetime.now(UTC)
        )

        self._strategy_edges[strategy_id] = strategy_edge
        return strategy_edge

    def adjust_kelly_for_performance(
        self,
        base_kelly: Decimal,
        recent_trades: list[Trade],
        window_size: int = 20
    ) -> Decimal:
        """
        Adjust Kelly fraction based on recent performance.
        
        Reduces Kelly during drawdowns and increases during winning streaks.
        
        Args:
            base_kelly: Base Kelly fraction
            recent_trades: Recent trade history
            window_size: Number of trades to consider
            
        Returns:
            Adjusted Kelly fraction
        """
        if len(recent_trades) < window_size // 2:
            return base_kelly

        # Get last N trades
        window_trades = recent_trades[-window_size:] if len(recent_trades) >= window_size else recent_trades

        # Calculate recent performance metrics
        returns = [float(t.pnl_percent) for t in window_trades]

        # Check for drawdown
        cumulative_return = np.cumprod([1 + r/100 for r in returns])
        peak = np.maximum.accumulate(cumulative_return)
        drawdown = (peak - cumulative_return) / peak
        max_drawdown = np.max(drawdown)

        # Adjust Kelly based on drawdown
        if max_drawdown > 0.2:  # 20% drawdown
            adjustment = Decimal(str(1 - max_drawdown))
            adjusted_kelly = base_kelly * adjustment
            logger.info("Reducing Kelly from %s to %s due to %.1f%% drawdown",
                       base_kelly, adjusted_kelly, max_drawdown * 100)
        else:
            # Check winning streak
            recent_wins = sum(1 for t in window_trades[-5:] if t.pnl_dollars > 0)
            if recent_wins >= 4:  # 4+ wins in last 5 trades
                adjustment = Decimal("1.1")  # Slight increase
                adjusted_kelly = min(base_kelly * adjustment, self.max_kelly)
                logger.info("Increasing Kelly from %s to %s due to winning streak",
                           base_kelly, adjusted_kelly)
            else:
                adjusted_kelly = base_kelly

        return adjusted_kelly.quantize(Decimal("0.0001"))

    def apply_conviction_multiplier(
        self,
        kelly_size: Decimal,
        conviction: ConvictionLevel,
        multipliers: Optional[dict[ConvictionLevel, Decimal]] = None
    ) -> Decimal:
        """
        Apply conviction multiplier to Kelly-based position size.
        
        Strategist+ feature for high-conviction trade overrides.
        
        Args:
            kelly_size: Base Kelly position size
            conviction: Trade conviction level
            multipliers: Custom multipliers (optional)
            
        Returns:
            Adjusted position size
        """
        default_multipliers = {
            ConvictionLevel.LOW: Decimal("0.5"),
            ConvictionLevel.MEDIUM: Decimal("1.0"),
            ConvictionLevel.HIGH: Decimal("1.5")
        }

        if multipliers:
            default_multipliers.update(multipliers)

        multiplier = default_multipliers.get(conviction, Decimal("1.0"))
        adjusted_size = kelly_size * multiplier

        logger.info("Applied %s conviction multiplier: %s -> %s",
                   conviction.value, kelly_size, adjusted_size)

        return adjusted_size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def enforce_position_boundaries(
        self,
        calculated_size: Decimal,
        balance: Decimal,
        tier: TradingTier,
        boundaries: Optional[dict[str, Decimal]] = None
    ) -> Decimal:
        """
        Enforce minimum and maximum position size boundaries.
        
        Args:
            calculated_size: Calculated position size
            balance: Account balance
            tier: Current trading tier
            boundaries: Custom boundaries (min_pct, max_pct)
            
        Returns:
            Position size within boundaries
        """
        # Default boundaries by tier
        default_boundaries = {
            TradingTier.SNIPER: {"min_pct": Decimal("2.0"), "max_pct": Decimal("5.0")},
            TradingTier.HUNTER: {"min_pct": Decimal("1.0"), "max_pct": Decimal("10.0")},
            TradingTier.STRATEGIST: {"min_pct": Decimal("0.5"), "max_pct": Decimal("15.0")}
        }

        tier_boundaries = boundaries or default_boundaries.get(tier, default_boundaries[TradingTier.HUNTER])

        min_size = balance * tier_boundaries["min_pct"] / 100
        max_size = balance * tier_boundaries["max_pct"] / 100

        # Apply boundaries
        bounded_size = max(min_size, min(calculated_size, max_size))

        if bounded_size != calculated_size:
            logger.info("Applied position boundaries: %s -> %s (tier: %s)",
                       calculated_size, bounded_size, tier.value)

        return bounded_size.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def calculate_volatility_multiplier(
        self,
        returns: list[float],
        lookback: int = 14,
        max_reduction: Decimal = Decimal("0.5")
    ) -> tuple[Decimal, VolatilityRegime]:
        """
        Calculate position size multiplier based on volatility.
        
        Args:
            returns: Recent returns for volatility calculation
            lookback: Number of periods for volatility calculation
            max_reduction: Maximum reduction factor in high volatility
            
        Returns:
            Tuple of (multiplier, volatility regime)
        """
        if len(returns) < lookback:
            return Decimal("1.0"), VolatilityRegime.NORMAL

        # Calculate volatility (standard deviation of returns)
        recent_returns = returns[-lookback:]
        volatility = np.std(recent_returns)

        # Define volatility regimes (annualized)
        annual_vol = volatility * np.sqrt(365)

        if annual_vol < 0.15:  # < 15% annualized
            regime = VolatilityRegime.LOW
            multiplier = Decimal("1.1")  # Slight increase in low vol
        elif annual_vol < 0.30:  # 15-30% annualized
            regime = VolatilityRegime.NORMAL
            multiplier = Decimal("1.0")
        else:  # > 30% annualized
            regime = VolatilityRegime.HIGH
            # Scale reduction based on how high volatility is
            reduction_factor = min(1.0, (annual_vol - 0.30) / 0.30)
            multiplier = Decimal(str(1.0 - float(max_reduction) * reduction_factor))

        logger.debug("Volatility regime: %s (%.1f%% annual), multiplier: %s",
                    regime.value, annual_vol * 100, multiplier)

        return multiplier.quantize(Decimal("0.01")), regime

    def run_monte_carlo_simulation(
        self,
        win_rate: Decimal,
        win_loss_ratio: Decimal,
        kelly_fraction: Decimal,
        initial_balance: Decimal = Decimal("10000"),
        iterations: int = 10000,
        trades_per_iteration: int = 100
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation to validate Kelly parameters.
        
        Args:
            win_rate: Historical win rate
            win_loss_ratio: Average win/loss ratio
            kelly_fraction: Kelly fraction to test
            initial_balance: Starting balance for simulation
            iterations: Number of simulation runs
            trades_per_iteration: Trades per simulation run
            
        Returns:
            SimulationResult with statistics
        """
        p = float(win_rate)
        b = float(win_loss_ratio)
        f = float(kelly_fraction)

        # Run simulations
        final_balances = []
        ruined_count = 0

        for _ in range(iterations):
            balance = float(initial_balance)

            for _ in range(trades_per_iteration):
                if balance <= 0:
                    ruined_count += 1
                    break

                bet_size = balance * f

                # Simulate trade outcome
                if np.random.random() < p:
                    # Win
                    balance += bet_size * b
                else:
                    # Loss
                    balance -= bet_size

            final_balances.append(balance)

        # Calculate statistics
        final_balances = np.array(final_balances)

        # Risk of ruin
        risk_of_ruin = Decimal(str(ruined_count / iterations))

        # Expected growth rate (geometric mean)
        positive_balances = final_balances[final_balances > 0]
        if len(positive_balances) > 0:
            growth_rate = Decimal(str(
                (np.mean(positive_balances) / float(initial_balance)) ** (1/trades_per_iteration) - 1
            ))
        else:
            growth_rate = Decimal("-1")

        # Calculate percentiles
        median = Decimal(str(np.median(final_balances)))
        p5 = Decimal(str(np.percentile(final_balances, 5)))
        p95 = Decimal(str(np.percentile(final_balances, 95)))

        # Find optimal Kelly through simulation
        optimal_kelly = self._find_optimal_kelly_simulation(
            p, b, initial_balance, trades_per_iteration, iterations=1000
        )

        return SimulationResult(
            optimal_kelly=Decimal(str(optimal_kelly)).quantize(Decimal("0.0001")),
            risk_of_ruin=risk_of_ruin.quantize(Decimal("0.0001")),
            expected_growth_rate=growth_rate.quantize(Decimal("0.0001")),
            median_final_balance=median.quantize(Decimal("0.01")),
            percentile_5=p5.quantize(Decimal("0.01")),
            percentile_95=p95.quantize(Decimal("0.01"))
        )

    def _find_optimal_kelly_simulation(
        self,
        win_rate: float,
        win_loss_ratio: float,
        initial_balance: float,
        trades: int,
        iterations: int = 1000
    ) -> float:
        """Find optimal Kelly fraction through simulation."""
        best_kelly = 0
        best_median = 0

        # Test different Kelly fractions
        for f in np.linspace(0.01, min(0.5, win_rate * win_loss_ratio), 50):
            balances = []

            for _ in range(iterations):
                balance = float(initial_balance) if isinstance(initial_balance, Decimal) else initial_balance

                for _ in range(trades):
                    if balance <= 0:
                        break

                    bet = balance * f
                    if np.random.random() < win_rate:
                        balance += bet * win_loss_ratio
                    else:
                        balance -= bet

                balances.append(balance)

            median_balance = np.median(balances)
            if median_balance > best_median:
                best_median = median_balance
                best_kelly = f

        return best_kelly
