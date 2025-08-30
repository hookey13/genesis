"""Position unwinding logic with correlation consideration."""

from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class PositionUnwinder:
    """Manages intelligent position unwinding."""

    def __init__(self, correlation_threshold: Decimal = Decimal("0.7")):
        """Initialize position unwinder.
        
        Args:
            correlation_threshold: Threshold for considering positions correlated
        """
        self.correlation_threshold = correlation_threshold

        # Hardcoded correlations for major pairs
        self.known_correlations = {
            ("BTC/USDT", "ETH/USDT"): Decimal("0.85"),
            ("BTC/USDT", "BNB/USDT"): Decimal("0.75"),
            ("ETH/USDT", "BNB/USDT"): Decimal("0.70"),
            ("BTC/USDT", "SOL/USDT"): Decimal("0.80"),
            ("ETH/USDT", "SOL/USDT"): Decimal("0.75"),
            ("USDT/USD", "BUSD/USD"): Decimal("0.95"),
        }

    def prioritize_positions(
        self,
        positions: list[dict],
        risk_first: bool = True
    ) -> list[dict]:
        """Prioritize positions for unwinding.
        
        Args:
            positions: List of open positions
            risk_first: If True, close highest risk positions first
            
        Returns:
            Prioritized list of positions
        """
        # Calculate risk scores
        for position in positions:
            position["risk_score"] = self._calculate_risk_score(position)
            position["correlation_group"] = self._get_correlation_group(
                position,
                positions
            )

        # Group correlated positions
        groups = self._group_correlated_positions(positions)

        # Sort groups by total exposure
        sorted_groups = sorted(
            groups,
            key=lambda g: sum(p["exposure"] for p in g),
            reverse=risk_first
        )

        # Flatten groups maintaining order
        prioritized = []
        for group in sorted_groups:
            # Within group, sort by individual risk
            group_sorted = sorted(
                group,
                key=lambda p: p["risk_score"],
                reverse=risk_first
            )
            prioritized.extend(group_sorted)

        logger.info(
            "Positions prioritized",
            count=len(prioritized),
            groups=len(groups)
        )

        return prioritized

    def _calculate_risk_score(self, position: dict) -> Decimal:
        """Calculate risk score for a position.
        
        Args:
            position: Position dictionary
            
        Returns:
            Risk score (higher = riskier)
        """
        # Base risk is exposure
        risk = position["exposure"]

        # Adjust for unrealized loss
        if position["unrealized_pnl"] < 0:
            loss_factor = abs(position["unrealized_pnl"]) / position["exposure"]
            risk *= (Decimal("1") + loss_factor)

        # Adjust for volatility (simplified - would use actual volatility in production)
        volatility_multipliers = {
            "BTC/USDT": Decimal("1.2"),
            "ETH/USDT": Decimal("1.3"),
            "SOL/USDT": Decimal("1.5"),
            "DOGE/USDT": Decimal("2.0"),
        }

        symbol = position["symbol"]
        if symbol in volatility_multipliers:
            risk *= volatility_multipliers[symbol]

        return risk

    def _get_correlation_group(
        self,
        position: dict,
        all_positions: list[dict]
    ) -> int:
        """Get correlation group for position.
        
        Args:
            position: Position to check
            all_positions: All positions
            
        Returns:
            Group ID
        """
        # Simple grouping by base asset
        symbol = position["symbol"]
        base_asset = symbol.split("/")[0] if "/" in symbol else symbol

        # Map base assets to groups
        asset_groups = {
            "BTC": 1,
            "ETH": 2,
            "BNB": 3,
            "SOL": 4,
            "USDT": 5,
            "BUSD": 5,  # Stablecoins grouped together
        }

        return asset_groups.get(base_asset, 99)  # 99 for ungrouped

    def _group_correlated_positions(
        self,
        positions: list[dict]
    ) -> list[list[dict]]:
        """Group positions by correlation.
        
        Args:
            positions: List of positions
            
        Returns:
            List of position groups
        """
        groups = {}

        for position in positions:
            group_id = position["correlation_group"]

            if group_id not in groups:
                groups[group_id] = []

            groups[group_id].append(position)

        return list(groups.values())

    def calculate_unwinding_schedule(
        self,
        positions: list[dict],
        max_orders_per_minute: int = 60,
        market_impact_threshold: Decimal = Decimal("0.1")
    ) -> list[tuple[dict, int]]:
        """Calculate optimal unwinding schedule.
        
        Args:
            positions: Prioritized positions
            max_orders_per_minute: Rate limit
            market_impact_threshold: Maximum market impact per order
            
        Returns:
            List of (position, delay_seconds) tuples
        """
        schedule = []

        # Calculate inter-order delay
        min_delay = 60 / max_orders_per_minute

        for i, position in enumerate(positions):
            # Calculate market impact (simplified)
            impact = self._estimate_market_impact(position)

            # Adjust delay based on impact
            if impact > market_impact_threshold:
                # Split large positions
                chunks = self._split_position(position, market_impact_threshold)

                for j, chunk in enumerate(chunks):
                    delay = min_delay * (j + 1)
                    schedule.append((chunk, int(delay)))
            else:
                delay = min_delay * (i + 1)
                schedule.append((position, int(delay)))

        return schedule

    def _estimate_market_impact(self, position: dict) -> Decimal:
        """Estimate market impact of closing position.
        
        Args:
            position: Position to close
            
        Returns:
            Estimated impact (0-1 scale)
        """
        # Simplified impact model
        # In production, would use order book depth

        size_thresholds = {
            "BTC/USDT": Decimal("10"),
            "ETH/USDT": Decimal("100"),
            "BNB/USDT": Decimal("500"),
            "SOL/USDT": Decimal("1000"),
        }

        symbol = position["symbol"]
        threshold = size_thresholds.get(symbol, Decimal("1000"))

        impact = position["quantity"] / threshold

        return min(impact, Decimal("1"))  # Cap at 1

    def _split_position(
        self,
        position: dict,
        max_impact: Decimal
    ) -> list[dict]:
        """Split large position into smaller chunks.
        
        Args:
            position: Position to split
            max_impact: Maximum impact per chunk
            
        Returns:
            List of position chunks
        """
        impact = self._estimate_market_impact(position)

        if impact <= max_impact:
            return [position]

        # Calculate number of chunks needed
        num_chunks = int(impact / max_impact) + 1
        chunk_size = position["quantity"] / num_chunks

        chunks = []
        for i in range(num_chunks):
            chunk = position.copy()
            chunk["quantity"] = chunk_size
            chunk["exposure"] = chunk_size * position["current_price"]
            chunk["chunk_number"] = i + 1
            chunk["total_chunks"] = num_chunks
            chunks.append(chunk)

        logger.info(
            f"Position split into {num_chunks} chunks",
            symbol=position["symbol"],
            original_size=position["quantity"],
            chunk_size=chunk_size
        )

        return chunks

    def suggest_hedge_positions(
        self,
        positions: list[dict]
    ) -> list[dict]:
        """Suggest hedge positions to reduce risk during unwinding.
        
        Args:
            positions: Current positions
            
        Returns:
            List of suggested hedges
        """
        hedges = []

        # Calculate net exposure by asset
        net_exposure = {}

        for position in positions:
            symbol = position["symbol"]
            base_asset = symbol.split("/")[0] if "/" in symbol else symbol

            exposure = position["exposure"]
            if position["side"] in ["sell", "short"]:
                exposure = -exposure

            if base_asset not in net_exposure:
                net_exposure[base_asset] = Decimal("0")

            net_exposure[base_asset] += exposure

        # Suggest hedges for large exposures
        for asset, exposure in net_exposure.items():
            if abs(exposure) > Decimal("10000"):  # $10k threshold
                hedge = {
                    "symbol": f"{asset}/USDT",
                    "side": "sell" if exposure > 0 else "buy",
                    "suggested_quantity": abs(exposure) / Decimal("2"),  # 50% hedge
                    "reason": f"Hedge {asset} exposure of {exposure}"
                }
                hedges.append(hedge)

        return hedges
