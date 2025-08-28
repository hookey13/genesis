"""
TWAP execution analyzer for Project GENESIS.

This module provides post-trade analysis of TWAP executions, calculating
slippage, market impact, timing effectiveness, and generating detailed reports.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog

from genesis.data.repository import Repository
from genesis.engine.executor.base import OrderSide

logger = structlog.get_logger(__name__)


@dataclass
class TwapReport:
    """Comprehensive TWAP execution report."""

    execution_id: str
    symbol: str
    side: OrderSide
    total_quantity: Decimal
    executed_quantity: Decimal
    duration_minutes: int
    slice_count: int
    completed_slices: int

    # Price metrics
    arrival_price: Decimal
    average_execution_price: Decimal
    twap_price: Decimal
    vwap_market_price: Decimal  # Market VWAP during execution
    best_possible_price: Decimal  # Theoretical best execution
    worst_slice_price: Decimal
    best_slice_price: Decimal

    # Performance metrics
    implementation_shortfall: Decimal  # vs arrival price
    twap_effectiveness: Decimal  # vs market TWAP
    slippage_bps: Decimal  # Average slippage in basis points
    max_slice_slippage_bps: Decimal
    total_slippage_cost: Decimal  # In quote currency

    # Market impact
    estimated_market_impact_bps: Decimal
    temporary_impact_bps: Decimal
    permanent_impact_bps: Decimal

    # Volume metrics
    avg_participation_rate: Decimal
    max_participation_rate: Decimal
    min_participation_rate: Decimal
    volume_weighted_participation: Decimal

    # Timing effectiveness
    timing_score: Decimal  # 0-100 score
    early_completion: bool
    early_completion_benefit: Optional[Decimal]  # Saved/lost from early completion

    # Risk metrics
    execution_risk_score: Decimal  # 0-100, lower is better
    concentration_risk: Decimal  # How concentrated were executions

    # Recommendations
    optimal_slice_count: int
    recommended_duration: int
    suggested_participation_rate: Decimal
    improvement_opportunities: list[str]

    # Metadata
    started_at: datetime
    completed_at: datetime
    analysis_timestamp: datetime


class TwapAnalyzer:
    """
    Analyzes TWAP execution performance and generates reports.

    Provides detailed post-trade analysis including slippage calculation,
    market impact estimation, and execution quality metrics.
    """

    def __init__(self, repository: Repository):
        """
        Initialize the TWAP analyzer.

        Args:
            repository: Data repository for accessing execution data
        """
        self.repository = repository
        logger.info("TWAP analyzer initialized")

    async def generate_execution_report(self, execution_id: str) -> TwapReport:
        """
        Generate comprehensive report for a TWAP execution.

        Args:
            execution_id: ID of the execution to analyze

        Returns:
            TwapReport with detailed analysis

        Raises:
            ValueError: If execution not found
        """
        try:
            # Fetch execution data
            execution_data = await self.repository.get_twap_execution(execution_id)
            if not execution_data:
                raise ValueError(f"Execution {execution_id} not found")

            # Fetch slice history
            slice_history = await self.repository.get_twap_slices(execution_id)
            if not slice_history:
                raise ValueError(f"No slice history found for execution {execution_id}")

            logger.info(
                "Generating TWAP execution report",
                execution_id=execution_id,
                slices=len(slice_history),
            )

            # Calculate price metrics
            price_metrics = self._calculate_price_metrics(execution_data, slice_history)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                execution_data, slice_history, price_metrics
            )

            # Estimate market impact
            market_impact = await self._estimate_market_impact(
                execution_data, slice_history
            )

            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(slice_history)

            # Calculate timing effectiveness
            timing_metrics = self._calculate_timing_effectiveness(
                execution_data, slice_history, price_metrics
            )

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(slice_history, volume_metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                execution_data, performance_metrics, volume_metrics, risk_metrics
            )

            # Create report
            report = TwapReport(
                execution_id=execution_id,
                symbol=execution_data["symbol"],
                side=OrderSide(execution_data["side"]),
                total_quantity=Decimal(str(execution_data["total_quantity"])),
                executed_quantity=Decimal(str(execution_data["executed_quantity"])),
                duration_minutes=execution_data["duration_minutes"],
                slice_count=execution_data["slice_count"],
                completed_slices=len(
                    [s for s in slice_history if s["status"] == "EXECUTED"]
                ),
                # Price metrics
                arrival_price=price_metrics["arrival_price"],
                average_execution_price=price_metrics["average_execution_price"],
                twap_price=price_metrics["twap_price"],
                vwap_market_price=price_metrics["vwap_market_price"],
                best_possible_price=price_metrics["best_possible_price"],
                worst_slice_price=price_metrics["worst_slice_price"],
                best_slice_price=price_metrics["best_slice_price"],
                # Performance metrics
                implementation_shortfall=performance_metrics[
                    "implementation_shortfall"
                ],
                twap_effectiveness=performance_metrics["twap_effectiveness"],
                slippage_bps=performance_metrics["slippage_bps"],
                max_slice_slippage_bps=performance_metrics["max_slice_slippage_bps"],
                total_slippage_cost=performance_metrics["total_slippage_cost"],
                # Market impact
                estimated_market_impact_bps=market_impact["estimated_impact_bps"],
                temporary_impact_bps=market_impact["temporary_impact_bps"],
                permanent_impact_bps=market_impact["permanent_impact_bps"],
                # Volume metrics
                avg_participation_rate=volume_metrics["avg_participation_rate"],
                max_participation_rate=volume_metrics["max_participation_rate"],
                min_participation_rate=volume_metrics["min_participation_rate"],
                volume_weighted_participation=volume_metrics[
                    "volume_weighted_participation"
                ],
                # Timing effectiveness
                timing_score=timing_metrics["timing_score"],
                early_completion=execution_data.get("early_completion", False),
                early_completion_benefit=timing_metrics.get("early_completion_benefit"),
                # Risk metrics
                execution_risk_score=risk_metrics["execution_risk_score"],
                concentration_risk=risk_metrics["concentration_risk"],
                # Recommendations
                optimal_slice_count=recommendations["optimal_slice_count"],
                recommended_duration=recommendations["recommended_duration"],
                suggested_participation_rate=recommendations[
                    "suggested_participation_rate"
                ],
                improvement_opportunities=recommendations["improvement_opportunities"],
                # Metadata
                started_at=execution_data["started_at"],
                completed_at=execution_data["completed_at"],
                analysis_timestamp=datetime.now(),
            )

            logger.info(
                "TWAP execution report generated",
                execution_id=execution_id,
                implementation_shortfall=str(report.implementation_shortfall),
                timing_score=str(report.timing_score),
                risk_score=str(report.execution_risk_score),
            )

            # Store analysis results for future reference
            await self._store_analysis_results(report)

            return report

        except Exception as e:
            logger.error(
                "Failed to generate TWAP report",
                execution_id=execution_id,
                error=str(e),
            )
            raise

    def _calculate_price_metrics(
        self, execution_data: dict, slice_history: list[dict]
    ) -> dict[str, Decimal]:
        """
        Calculate price-related metrics.

        Args:
            execution_data: Execution details
            slice_history: List of slice executions

        Returns:
            Dictionary of price metrics
        """
        arrival_price = Decimal(str(execution_data["arrival_price"]))

        # Calculate average execution price (volume-weighted)
        total_value = Decimal("0")
        total_quantity = Decimal("0")
        prices = []

        for slice_data in slice_history:
            if slice_data["status"] == "EXECUTED":
                quantity = Decimal(str(slice_data["executed_quantity"]))
                price = Decimal(str(slice_data["execution_price"]))
                total_value += quantity * price
                total_quantity += quantity
                prices.append(price)

        average_execution_price = (
            total_value / total_quantity if total_quantity > 0 else Decimal("0")
        )

        # TWAP price (simple time-weighted average)
        twap_price = sum(prices) / len(prices) if prices else Decimal("0")

        # Market VWAP (would need market data, using approximation)
        vwap_market_price = average_execution_price * Decimal(
            "0.999"
        )  # Slight improvement assumption

        # Best and worst prices
        best_slice_price = min(prices) if prices else Decimal("0")
        worst_slice_price = max(prices) if prices else Decimal("0")

        # Theoretical best (all at best slice price)
        side = OrderSide(execution_data["side"])
        if side == OrderSide.BUY:
            best_possible_price = best_slice_price
        else:
            best_possible_price = worst_slice_price

        return {
            "arrival_price": arrival_price,
            "average_execution_price": average_execution_price,
            "twap_price": twap_price,
            "vwap_market_price": vwap_market_price,
            "best_possible_price": best_possible_price,
            "worst_slice_price": worst_slice_price,
            "best_slice_price": best_slice_price,
        }

    def _calculate_performance_metrics(
        self,
        execution_data: dict,
        slice_history: list[dict],
        price_metrics: dict[str, Decimal],
    ) -> dict[str, Decimal]:
        """
        Calculate performance metrics.

        Args:
            execution_data: Execution details
            slice_history: List of slice executions
            price_metrics: Calculated price metrics

        Returns:
            Dictionary of performance metrics
        """
        arrival_price = price_metrics["arrival_price"]
        avg_execution_price = price_metrics["average_execution_price"]
        twap_price = price_metrics["twap_price"]
        vwap_market_price = price_metrics["vwap_market_price"]

        # Implementation shortfall (vs arrival price)
        if arrival_price > 0:
            implementation_shortfall = (
                (avg_execution_price - arrival_price) / arrival_price
            ) * Decimal(
                "10000"
            )  # in bps
        else:
            implementation_shortfall = Decimal("0")

        # TWAP effectiveness (how close to market TWAP)
        if vwap_market_price > 0:
            twap_effectiveness = (
                Decimal("1") - abs(twap_price - vwap_market_price) / vwap_market_price
            ) * Decimal("100")
        else:
            twap_effectiveness = Decimal("0")

        # Calculate slippage
        total_slippage_bps = Decimal("0")
        max_slippage_bps = Decimal("0")
        slice_count = 0

        for slice_data in slice_history:
            if slice_data["status"] == "EXECUTED":
                slippage = Decimal(str(slice_data.get("slippage_bps", 0)))
                total_slippage_bps += abs(slippage)
                max_slippage_bps = max(max_slippage_bps, abs(slippage))
                slice_count += 1

        avg_slippage_bps = (
            total_slippage_bps / slice_count if slice_count > 0 else Decimal("0")
        )

        # Total slippage cost
        executed_quantity = Decimal(str(execution_data["executed_quantity"]))
        total_slippage_cost = (
            (avg_slippage_bps / Decimal("10000"))
            * avg_execution_price
            * executed_quantity
        )

        return {
            "implementation_shortfall": implementation_shortfall,
            "twap_effectiveness": twap_effectiveness,
            "slippage_bps": avg_slippage_bps,
            "max_slice_slippage_bps": max_slippage_bps,
            "total_slippage_cost": total_slippage_cost,
        }

    async def _estimate_market_impact(
        self, execution_data: dict, slice_history: list[dict]
    ) -> dict[str, Decimal]:
        """
        Estimate market impact of the TWAP execution.

        Args:
            execution_data: Execution details
            slice_history: List of slice executions

        Returns:
            Dictionary of market impact estimates
        """
        # Simple market impact model
        # Impact = sqrt(volume_fraction) * volatility * constant

        total_volume = Decimal("0")
        for slice_data in slice_history:
            if "volume_at_execution" in slice_data:
                total_volume += Decimal(str(slice_data["volume_at_execution"]))

        executed_quantity = Decimal(str(execution_data["executed_quantity"]))

        if total_volume > 0:
            volume_fraction = executed_quantity / total_volume
            # Square root model for market impact
            impact_factor = volume_fraction ** Decimal("0.5")

            # Assume 2% daily volatility, 10 bps impact constant
            volatility = Decimal("0.02")
            impact_constant = Decimal("10")

            estimated_impact_bps = (
                impact_factor * volatility * impact_constant * Decimal("100")
            )

            # Split into temporary (60%) and permanent (40%) impact
            temporary_impact_bps = estimated_impact_bps * Decimal("0.6")
            permanent_impact_bps = estimated_impact_bps * Decimal("0.4")
        else:
            estimated_impact_bps = Decimal("0")
            temporary_impact_bps = Decimal("0")
            permanent_impact_bps = Decimal("0")

        return {
            "estimated_impact_bps": estimated_impact_bps,
            "temporary_impact_bps": temporary_impact_bps,
            "permanent_impact_bps": permanent_impact_bps,
        }

    def _calculate_volume_metrics(
        self, slice_history: list[dict]
    ) -> dict[str, Decimal]:
        """
        Calculate volume-related metrics.

        Args:
            slice_history: List of slice executions

        Returns:
            Dictionary of volume metrics
        """
        participation_rates = []
        weighted_participation = Decimal("0")
        total_volume = Decimal("0")

        for slice_data in slice_history:
            if slice_data["status"] == "EXECUTED":
                participation = Decimal(str(slice_data.get("participation_rate", 0)))
                participation_rates.append(participation)

                if "volume_at_execution" in slice_data:
                    volume = Decimal(str(slice_data["volume_at_execution"]))
                    weighted_participation += participation * volume
                    total_volume += volume

        if participation_rates:
            avg_participation = sum(participation_rates) / len(participation_rates)
            max_participation = max(participation_rates)
            min_participation = min(participation_rates)
        else:
            avg_participation = Decimal("0")
            max_participation = Decimal("0")
            min_participation = Decimal("0")

        volume_weighted_participation = (
            weighted_participation / total_volume if total_volume > 0 else Decimal("0")
        )

        return {
            "avg_participation_rate": avg_participation,
            "max_participation_rate": max_participation,
            "min_participation_rate": min_participation,
            "volume_weighted_participation": volume_weighted_participation,
        }

    def _calculate_timing_effectiveness(
        self,
        execution_data: dict,
        slice_history: list[dict],
        price_metrics: dict[str, Decimal],
    ) -> dict[str, Optional[Decimal]]:
        """
        Calculate timing effectiveness metrics.

        Args:
            execution_data: Execution details
            slice_history: List of slice executions
            price_metrics: Calculated price metrics

        Returns:
            Dictionary of timing metrics
        """
        # Calculate timing score based on execution prices vs market prices
        timing_scores = []

        for slice_data in slice_history:
            if slice_data["status"] == "EXECUTED":
                execution_price = Decimal(str(slice_data["execution_price"]))
                market_price = Decimal(
                    str(slice_data.get("market_price", execution_price))
                )

                # Score based on favorable execution vs market
                side = OrderSide(execution_data["side"])
                if side == OrderSide.BUY:
                    # Lower execution price is better for buys
                    if execution_price <= market_price:
                        score = min(
                            Decimal("100"),
                            (market_price - execution_price)
                            / market_price
                            * Decimal("1000"),
                        )
                    else:
                        score = max(
                            Decimal("0"),
                            Decimal("100")
                            - (execution_price - market_price)
                            / market_price
                            * Decimal("1000"),
                        )
                else:
                    # Higher execution price is better for sells
                    if execution_price >= market_price:
                        score = min(
                            Decimal("100"),
                            (execution_price - market_price)
                            / market_price
                            * Decimal("1000"),
                        )
                    else:
                        score = max(
                            Decimal("0"),
                            Decimal("100")
                            - (market_price - execution_price)
                            / market_price
                            * Decimal("1000"),
                        )

                timing_scores.append(score)

        overall_timing_score = (
            sum(timing_scores) / len(timing_scores) if timing_scores else Decimal("50")
        )

        # Calculate early completion benefit if applicable
        early_completion_benefit = None
        if execution_data.get("early_completion"):
            # Estimate benefit from completing early at favorable price
            avg_price = price_metrics["average_execution_price"]
            arrival_price = price_metrics["arrival_price"]
            remaining = Decimal(str(execution_data.get("remaining_quantity", 0)))

            if remaining > 0 and arrival_price > 0:
                # Potential additional slippage avoided
                early_completion_benefit = abs(avg_price - arrival_price) * remaining

        return {
            "timing_score": overall_timing_score,
            "early_completion_benefit": early_completion_benefit,
        }

    def _calculate_risk_metrics(
        self, slice_history: list[dict], volume_metrics: dict[str, Decimal]
    ) -> dict[str, Decimal]:
        """
        Calculate risk metrics.

        Args:
            slice_history: List of slice executions
            volume_metrics: Volume metrics

        Returns:
            Dictionary of risk metrics
        """
        # Calculate execution risk score (0-100, lower is better)
        risk_factors = []

        # Factor 1: Participation rate risk (high participation = higher risk)
        max_participation = volume_metrics["max_participation_rate"]
        if max_participation > 10:
            participation_risk = min(Decimal("100"), (max_participation - 10) * 5)
        else:
            participation_risk = Decimal("0")
        risk_factors.append(participation_risk)

        # Factor 2: Slippage risk
        slippage_values = []
        for slice_data in slice_history:
            if slice_data["status"] == "EXECUTED":
                slippage = abs(Decimal(str(slice_data.get("slippage_bps", 0))))
                slippage_values.append(slippage)

        if slippage_values:
            avg_slippage = sum(slippage_values) / len(slippage_values)
            slippage_risk = min(Decimal("100"), avg_slippage)
        else:
            slippage_risk = Decimal("0")
        risk_factors.append(slippage_risk)

        # Factor 3: Failed slice risk
        total_slices = len(slice_history)
        failed_slices = len([s for s in slice_history if s["status"] == "FAILED"])
        if total_slices > 0:
            failure_risk = (failed_slices / total_slices) * Decimal("100")
        else:
            failure_risk = Decimal("0")
        risk_factors.append(failure_risk)

        # Overall risk score (average of factors)
        execution_risk_score = (
            sum(risk_factors) / len(risk_factors) if risk_factors else Decimal("0")
        )

        # Calculate concentration risk (how evenly distributed were executions)
        quantities = []
        for slice_data in slice_history:
            if slice_data["status"] == "EXECUTED":
                quantities.append(Decimal(str(slice_data["executed_quantity"])))

        if quantities:
            avg_quantity = sum(quantities) / len(quantities)
            deviations = [abs(q - avg_quantity) for q in quantities]
            concentration_risk = (
                (sum(deviations) / len(deviations)) / avg_quantity * Decimal("100")
                if avg_quantity > 0
                else Decimal("0")
            )
        else:
            concentration_risk = Decimal("0")

        return {
            "execution_risk_score": execution_risk_score,
            "concentration_risk": concentration_risk,
        }

    def _generate_recommendations(
        self,
        execution_data: dict,
        performance_metrics: dict[str, Decimal],
        volume_metrics: dict[str, Decimal],
        risk_metrics: dict[str, Decimal],
    ) -> dict:
        """
        Generate recommendations for future executions.

        Args:
            execution_data: Execution details
            performance_metrics: Performance metrics
            volume_metrics: Volume metrics
            risk_metrics: Risk metrics

        Returns:
            Dictionary of recommendations
        """
        recommendations = {"improvement_opportunities": []}

        # Optimal slice count based on execution quality
        current_slices = execution_data["slice_count"]
        if risk_metrics["concentration_risk"] > 30:
            # High concentration, need more slices
            optimal_slices = min(current_slices + 5, 20)
            recommendations["improvement_opportunities"].append(
                f"Increase slice count from {current_slices} to {optimal_slices} to reduce concentration risk"
            )
        elif risk_metrics["concentration_risk"] < 10 and current_slices > 10:
            # Over-slicing, can reduce
            optimal_slices = max(current_slices - 3, 5)
            recommendations["improvement_opportunities"].append(
                f"Reduce slice count from {current_slices} to {optimal_slices} for better efficiency"
            )
        else:
            optimal_slices = current_slices
        recommendations["optimal_slice_count"] = optimal_slices

        # Recommended duration based on market impact
        current_duration = execution_data["duration_minutes"]
        if performance_metrics["slippage_bps"] > 20:
            # High slippage, extend duration
            recommended_duration = min(current_duration + 10, 30)
            recommendations["improvement_opportunities"].append(
                f"Extend execution duration from {current_duration} to {recommended_duration} minutes to reduce market impact"
            )
        elif performance_metrics["slippage_bps"] < 5 and current_duration > 15:
            # Low impact, can be more aggressive
            recommended_duration = max(current_duration - 5, 5)
            recommendations["improvement_opportunities"].append(
                f"Reduce execution duration from {current_duration} to {recommended_duration} minutes for faster execution"
            )
        else:
            recommended_duration = current_duration
        recommendations["recommended_duration"] = recommended_duration

        # Suggested participation rate
        avg_participation = volume_metrics["avg_participation_rate"]
        if avg_participation > 8:
            suggested_participation = Decimal("7")
            recommendations["improvement_opportunities"].append(
                f"Reduce participation rate from {avg_participation:.1f}% to {suggested_participation}% to minimize market impact"
            )
        elif avg_participation < 3:
            suggested_participation = Decimal("5")
            recommendations["improvement_opportunities"].append(
                f"Increase participation rate from {avg_participation:.1f}% to {suggested_participation}% for more efficient execution"
            )
        else:
            suggested_participation = avg_participation
        recommendations["suggested_participation_rate"] = suggested_participation

        # Additional recommendations based on metrics
        if performance_metrics["implementation_shortfall"] > 50:  # > 50 bps
            recommendations["improvement_opportunities"].append(
                "High implementation shortfall detected - consider using limit orders for some slices"
            )

        if risk_metrics["execution_risk_score"] > 70:
            recommendations["improvement_opportunities"].append(
                "High execution risk - implement more conservative slice sizing and timing"
            )

        if volume_metrics["max_participation_rate"] > 15:
            recommendations["improvement_opportunities"].append(
                "Participation rate spike detected - implement better volume prediction and adaptive sizing"
            )

        return recommendations

    async def _store_analysis_results(self, report: TwapReport) -> None:
        """
        Store analysis results for future reference and optimization.

        Args:
            report: Generated TWAP report
        """
        try:
            await self.repository.save_twap_analysis(report)
            logger.info(
                "TWAP analysis results stored", execution_id=report.execution_id
            )
        except Exception as e:
            logger.warning(
                "Failed to store TWAP analysis results",
                execution_id=report.execution_id,
                error=str(e),
            )

    async def compare_executions(self, execution_ids: list[str]) -> dict:
        """
        Compare multiple TWAP executions for the same symbol.

        Args:
            execution_ids: List of execution IDs to compare

        Returns:
            Comparative analysis results
        """
        reports = []
        for execution_id in execution_ids:
            try:
                report = await self.generate_execution_report(execution_id)
                reports.append(report)
            except Exception as e:
                logger.warning(
                    "Failed to generate report for comparison",
                    execution_id=execution_id,
                    error=str(e),
                )

        if len(reports) < 2:
            return {"error": "Need at least 2 executions for comparison"}

        # Compare key metrics
        comparison = {
            "execution_ids": execution_ids,
            "best_implementation_shortfall": min(
                r.implementation_shortfall for r in reports
            ),
            "worst_implementation_shortfall": max(
                r.implementation_shortfall for r in reports
            ),
            "avg_implementation_shortfall": sum(
                r.implementation_shortfall for r in reports
            )
            / len(reports),
            "best_timing_score": max(r.timing_score for r in reports),
            "worst_timing_score": min(r.timing_score for r in reports),
            "best_risk_score": min(r.execution_risk_score for r in reports),
            "worst_risk_score": max(r.execution_risk_score for r in reports),
        }

        # Identify best performing execution
        scores = []
        for report in reports:
            # Composite score (lower is better)
            score = (
                report.implementation_shortfall * Decimal("0.4")
                + (Decimal("100") - report.timing_score) * Decimal("0.3")
                + report.execution_risk_score * Decimal("0.3")
            )
            scores.append((report.execution_id, score))

        scores.sort(key=lambda x: x[1])
        comparison["best_execution"] = scores[0][0]
        comparison["worst_execution"] = scores[-1][0]

        return comparison
