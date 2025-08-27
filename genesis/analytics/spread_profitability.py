"""
Spread Profitability Calculator

Calculates profit potential, break-even points, and optimal position sizing
based on spread analysis and market conditions.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from genesis.core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class ProfitabilityMetrics:
    """Profitability metrics for spread trading"""

    symbol: str
    spread_bps: Decimal
    gross_profit_bps: Decimal
    net_profit_bps: Decimal
    profit_usdt: Decimal
    roi_percent: Decimal
    break_even_spread_bps: Decimal
    is_profitable: bool
    optimal_position_size: Decimal
    estimated_slippage_bps: Decimal


@dataclass
class TradingCosts:
    """Trading cost structure"""

    maker_fee_bps: Decimal = Decimal("10")  # 0.10% default Binance maker fee
    taker_fee_bps: Decimal = Decimal("10")  # 0.10% default Binance taker fee
    slippage_factor: Decimal = Decimal("0.1")  # 10% of spread as slippage estimate


class SpreadProfitabilityCalculator:
    """
    Calculates profitability metrics for spread trading opportunities
    """

    def __init__(
        self,
        trading_costs: Optional[TradingCosts] = None,
        min_profit_bps: Decimal = Decimal("5"),
    ):
        """
        Initialize profitability calculator

        Args:
            trading_costs: Trading cost structure
            min_profit_bps: Minimum profit in bps to consider profitable
        """
        self.costs = trading_costs or TradingCosts()
        self.min_profit_bps = min_profit_bps
        self._logger = logger.bind(component="SpreadProfitabilityCalculator")

    def calculate_profit_potential(
        self,
        symbol: str,
        spread_bps: Decimal,
        volume: Decimal,
        fee_bps: Optional[Decimal] = None,
        custom_slippage: Optional[Decimal] = None,
    ) -> ProfitabilityMetrics:
        """
        Calculate profit potential for a spread opportunity

        Args:
            symbol: Trading pair symbol
            spread_bps: Current spread in basis points
            volume: Trade volume in USDT
            fee_bps: Override fee in basis points
            custom_slippage: Override slippage estimate

        Returns:
            ProfitabilityMetrics with calculations
        """
        if spread_bps <= 0:
            raise ValidationError(f"Invalid spread: {spread_bps}")
        if volume <= 0:
            raise ValidationError(f"Invalid volume: {volume}")

        # Use provided fee or default
        effective_fee = fee_bps if fee_bps is not None else self.costs.taker_fee_bps

        # Calculate slippage
        slippage_bps = (
            custom_slippage
            if custom_slippage is not None
            else self._estimate_slippage(spread_bps, volume)
        )

        # Calculate costs (entry + exit fees + slippage)
        total_cost_bps = (effective_fee * 2) + slippage_bps

        # Calculate profit
        gross_profit_bps = spread_bps
        net_profit_bps = gross_profit_bps - total_cost_bps

        # Calculate profit in USDT
        profit_usdt = (net_profit_bps / Decimal("10000")) * volume

        # Calculate ROI
        roi_percent = (
            (net_profit_bps / total_cost_bps * Decimal("100"))
            if total_cost_bps > 0
            else Decimal("0")
        )

        # Calculate break-even spread
        break_even_spread = total_cost_bps

        # Determine if profitable
        is_profitable = net_profit_bps >= self.min_profit_bps

        # Calculate optimal position size
        optimal_size = self._calculate_optimal_position(
            spread_bps, volume, net_profit_bps
        )

        metrics = ProfitabilityMetrics(
            symbol=symbol,
            spread_bps=spread_bps,
            gross_profit_bps=gross_profit_bps,
            net_profit_bps=net_profit_bps,
            profit_usdt=profit_usdt,
            roi_percent=roi_percent,
            break_even_spread_bps=break_even_spread,
            is_profitable=is_profitable,
            optimal_position_size=optimal_size,
            estimated_slippage_bps=slippage_bps,
        )

        self._logger.debug(
            "Profitability calculated",
            symbol=symbol,
            spread_bps=float(spread_bps),
            net_profit_bps=float(net_profit_bps),
            profit_usdt=float(profit_usdt),
            is_profitable=is_profitable,
        )

        return metrics

    def calculate_break_even_spread(
        self, fee_bps: Optional[Decimal] = None, slippage_bps: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate minimum spread needed to break even

        Args:
            fee_bps: Trading fee in basis points
            slippage_bps: Expected slippage in basis points

        Returns:
            Break-even spread in basis points
        """
        effective_fee = fee_bps if fee_bps is not None else self.costs.taker_fee_bps
        effective_slippage = slippage_bps if slippage_bps is not None else Decimal("2")

        # Break-even = 2 * fees + slippage
        break_even = (effective_fee * 2) + effective_slippage

        return break_even

    def optimize_position_size(
        self,
        spread_bps: Decimal,
        available_liquidity: Decimal,
        max_position: Decimal,
        target_profit_usdt: Decimal,
    ) -> Decimal:
        """
        Optimize position size based on spread and constraints

        Args:
            spread_bps: Current spread in basis points
            available_liquidity: Available liquidity in order book
            max_position: Maximum allowed position size
            target_profit_usdt: Target profit in USDT

        Returns:
            Optimal position size in USDT
        """
        # Calculate size needed for target profit
        if spread_bps <= 0:
            return Decimal("0")

        # Estimate net profit after costs
        net_profit_bps = (
            spread_bps - (self.costs.taker_fee_bps * 2) - Decimal("2")
        )  # Basic slippage

        if net_profit_bps <= 0:
            return Decimal("0")

        # Size needed for target profit
        size_for_target = (target_profit_usdt * Decimal("10000")) / net_profit_bps

        # Apply constraints
        optimal_size = min(
            size_for_target,
            available_liquidity
            * Decimal("0.1"),  # Don't take more than 10% of liquidity
            max_position,
        )

        return optimal_size

    def _estimate_slippage(self, spread_bps: Decimal, volume: Decimal) -> Decimal:
        """
        Estimate slippage based on spread and volume

        Args:
            spread_bps: Current spread
            volume: Trade volume

        Returns:
            Estimated slippage in basis points
        """
        # Basic model: slippage increases with volume and tight spreads
        base_slippage = spread_bps * self.costs.slippage_factor

        # Volume impact (simplified)
        if volume > Decimal("100000"):
            volume_factor = Decimal("1.5")
        elif volume > Decimal("50000"):
            volume_factor = Decimal("1.2")
        else:
            volume_factor = Decimal("1.0")

        estimated_slippage = base_slippage * volume_factor

        return estimated_slippage

    def _calculate_optimal_position(
        self, spread_bps: Decimal, volume: Decimal, net_profit_bps: Decimal
    ) -> Decimal:
        """
        Calculate optimal position size

        Args:
            spread_bps: Current spread
            volume: Available volume
            net_profit_bps: Net profit in bps

        Returns:
            Optimal position size
        """
        if net_profit_bps <= 0:
            return Decimal("0")

        # Kelly Criterion simplified
        # f = (p * b - q) / b
        # where p = win probability, b = profit/loss ratio, q = loss probability

        # Simplified: use profit ratio as position sizing factor
        profit_ratio = net_profit_bps / spread_bps
        kelly_fraction = profit_ratio * Decimal("0.25")  # Conservative Kelly (1/4)

        # Apply to available volume
        optimal_size = volume * kelly_fraction

        # Apply min/max constraints
        min_size = Decimal("100")  # Minimum $100 position
        max_size = volume * Decimal("0.2")  # Max 20% of available volume

        return max(min_size, min(optimal_size, max_size))

    def batch_calculate(self, opportunities: list[dict]) -> list[ProfitabilityMetrics]:
        """
        Calculate profitability for multiple opportunities

        Args:
            opportunities: List of dicts with symbol, spread_bps, volume

        Returns:
            List of ProfitabilityMetrics
        """
        results = []

        for opp in opportunities:
            try:
                metrics = self.calculate_profit_potential(
                    symbol=opp["symbol"],
                    spread_bps=opp["spread_bps"],
                    volume=opp["volume"],
                    fee_bps=opp.get("fee_bps"),
                    custom_slippage=opp.get("slippage"),
                )
                results.append(metrics)
            except Exception as e:
                self._logger.error(
                    "Failed to calculate profitability",
                    symbol=opp.get("symbol"),
                    error=str(e),
                )

        # Sort by profitability
        results.sort(key=lambda x: x.net_profit_bps, reverse=True)

        return results

    def compare_exchanges(
        self,
        symbol: str,
        spreads: dict[str, Decimal],
        volumes: dict[str, Decimal],
        fees: dict[str, Decimal],
    ) -> dict[str, ProfitabilityMetrics]:
        """
        Compare profitability across exchanges (future-ready)

        Args:
            symbol: Trading pair symbol
            spreads: Exchange to spread mapping
            volumes: Exchange to volume mapping
            fees: Exchange to fee mapping

        Returns:
            Dictionary of exchange to metrics
        """
        results = {}

        for exchange in spreads:
            if exchange not in volumes or exchange not in fees:
                continue

            try:
                metrics = self.calculate_profit_potential(
                    symbol=f"{symbol}_{exchange}",
                    spread_bps=spreads[exchange],
                    volume=volumes[exchange],
                    fee_bps=fees[exchange],
                )
                results[exchange] = metrics
            except Exception as e:
                self._logger.error(
                    "Failed to calculate exchange profitability",
                    exchange=exchange,
                    symbol=symbol,
                    error=str(e),
                )

        return results
