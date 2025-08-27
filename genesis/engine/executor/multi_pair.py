from typing import Optional
"""Multi-pair concurrent trading manager for Hunter tier and above."""

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from genesis.core.exceptions import TierLockedException
from genesis.core.models import Position, Signal, Tier
from genesis.data.repository import Repository
from genesis.engine.state_machine import StateManager

logger = structlog.get_logger(__name__)


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""
    total_exposure_dollars: Decimal
    position_count: int
    max_drawdown_dollars: Decimal
    correlation_risk: Decimal  # 0-1, higher = more correlated
    concentration_risk: Decimal  # 0-1, higher = more concentrated
    available_capital: Decimal
    risk_score: Decimal  # Overall risk score 0-100
    warnings: list[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        """Check if portfolio is in high risk state."""
        return self.risk_score > 75 or self.correlation_risk > 0.8


@dataclass
class PortfolioLimits:
    """Portfolio and per-pair trading limits."""
    max_positions_global: int
    max_exposure_global_dollars: Decimal
    per_pair_limits: dict[str, tuple[Decimal, Decimal]]  # symbol -> (max_size, max_dollars)
    default_pair_limit_size: Decimal
    default_pair_limit_dollars: Decimal


class MultiPairManager:
    """Manages concurrent trading across multiple pairs with portfolio-level risk controls."""

    MINIMUM_TIER = Tier.HUNTER
    MAX_CORRELATION_WARNING = Decimal("0.6")
    MAX_CORRELATION_RISK_ADJUST = Decimal("0.8")

    def __init__(
        self,
        repository: Repository,
        state_manager: StateManager,
        account_id: str
    ):
        """Initialize multi-pair manager.
        
        Args:
            repository: Data repository for persistence
            state_manager: State machine for tier validation
            account_id: Account identifier
        """
        self.repository = repository
        self.state_manager = state_manager
        self.account_id = account_id
        self._active_positions: dict[str, Position] = {}
        self._position_correlations: dict[tuple[str, str], Decimal] = {}
        self._limits: Optional[PortfolioLimits] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize manager and validate tier requirements."""
        # Check tier gate
        current_tier = await self.state_manager.get_current_tier(self.account_id)
        if current_tier < self.MINIMUM_TIER:
            raise TierLockedException(
                f"Multi-pair trading requires {self.MINIMUM_TIER.value} tier or above. "
                f"Current tier: {current_tier.value}"
            )

        # Load portfolio limits
        await self._load_portfolio_limits()

        # Load active positions
        await self._load_active_positions()

        logger.info(
            "multi_pair_manager_initialized",
            account_id=self.account_id,
            tier=current_tier.value,
            max_positions=self._limits.max_positions_global if self._limits else 0
        )

    async def can_open_position(self, symbol: str, size: Decimal) -> bool:
        """Check if a new position can be opened.
        
        Args:
            symbol: Trading pair symbol
            size: Position size in base currency
            
        Returns:
            True if position can be opened, False otherwise
        """
        async with self._lock:
            if not self._limits:
                return False

            # Check global position count
            if len(self._active_positions) >= self._limits.max_positions_global:
                logger.warning(
                    "max_positions_reached",
                    current=len(self._active_positions),
                    max=self._limits.max_positions_global
                )
                return False

            # Get price for dollar value calculation
            current_price = await self._get_current_price(symbol)
            dollar_value = size * current_price

            # Check per-pair limits
            pair_limit_size, pair_limit_dollars = self._get_pair_limits(symbol)

            # Check if symbol already has position
            existing_position = self._active_positions.get(symbol)
            if existing_position:
                total_size = existing_position.quantity + size
                total_dollars = existing_position.dollar_value + dollar_value

                if total_size > pair_limit_size or total_dollars > pair_limit_dollars:
                    logger.warning(
                        "pair_limit_exceeded",
                        symbol=symbol,
                        existing_size=existing_position.quantity,
                        new_size=size,
                        limit_size=pair_limit_size,
                        limit_dollars=pair_limit_dollars
                    )
                    return False
            else:
                if size > pair_limit_size or dollar_value > pair_limit_dollars:
                    logger.warning(
                        "pair_limit_exceeded_new",
                        symbol=symbol,
                        size=size,
                        dollar_value=dollar_value,
                        limit_size=pair_limit_size,
                        limit_dollars=pair_limit_dollars
                    )
                    return False

            # Check global exposure limit
            total_exposure = sum(p.dollar_value for p in self._active_positions.values())
            if total_exposure + dollar_value > self._limits.max_exposure_global_dollars:
                logger.warning(
                    "global_exposure_exceeded",
                    current_exposure=total_exposure,
                    new_value=dollar_value,
                    max_exposure=self._limits.max_exposure_global_dollars
                )
                return False

            return True

    async def allocate_capital(self, signals: list[Signal]) -> dict[str, Decimal]:
        """Allocate capital across multiple signals intelligently.
        
        Args:
            signals: List of trading signals to allocate capital to
            
        Returns:
            Dictionary mapping symbol to allocated capital amount
        """
        if not signals:
            return {}

        async with self._lock:
            allocations = {}

            # Get available capital
            account = await self.repository.get_account(self.account_id)
            available_capital = account.balance

            # Calculate total exposure
            total_exposure = sum(p.dollar_value for p in self._active_positions.values())
            remaining_capital = max(
                Decimal("0"),
                self._limits.max_exposure_global_dollars - total_exposure
            )

            # Use the lesser of account balance and exposure room
            allocatable = min(available_capital, remaining_capital)

            if allocatable <= Decimal("0"):
                logger.warning(
                    "no_capital_available",
                    account_balance=available_capital,
                    exposure_room=remaining_capital
                )
                return {}

            # Sort signals by confidence/priority
            sorted_signals = sorted(
                signals,
                key=lambda s: (s.priority, s.confidence_score),
                reverse=True
            )

            # Apply Kelly Criterion-inspired allocation with correlation adjustment
            total_weight = Decimal("0")
            signal_weights = {}

            for signal in sorted_signals:
                # Base weight from confidence and priority
                base_weight = (signal.confidence_score * Decimal("0.6") +
                              Decimal(signal.priority) / Decimal("100") * Decimal("0.4"))

                # Adjust for correlation with existing positions
                correlation_penalty = await self._calculate_correlation_penalty(signal.symbol)
                adjusted_weight = base_weight * (Decimal("1") - correlation_penalty)

                signal_weights[signal.symbol] = adjusted_weight
                total_weight += adjusted_weight

            # Allocate capital proportionally
            if total_weight > Decimal("0"):
                for signal in sorted_signals:
                    symbol = signal.symbol
                    weight = signal_weights[symbol]

                    # Calculate allocation
                    allocation = (weight / total_weight) * allocatable

                    # Apply per-pair limits
                    _, pair_limit_dollars = self._get_pair_limits(symbol)
                    existing_exposure = self._active_positions.get(symbol, Position()).dollar_value
                    max_additional = pair_limit_dollars - existing_exposure

                    allocation = min(allocation, max_additional)

                    # Apply minimum trade size (e.g., $10)
                    min_trade_size = Decimal("10")
                    if allocation >= min_trade_size:
                        allocations[symbol] = allocation

                    logger.info(
                        "capital_allocated",
                        symbol=symbol,
                        amount=allocation,
                        weight=weight,
                        confidence=signal.confidence_score,
                        priority=signal.priority
                    )

            return allocations

    async def get_active_positions(self) -> list[Position]:
        """Get all active positions.
        
        Returns:
            List of active Position objects
        """
        async with self._lock:
            return list(self._active_positions.values())

    async def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics.
        
        Returns:
            PortfolioRisk object with current risk assessment
        """
        async with self._lock:
            positions = list(self._active_positions.values())

            if not positions:
                return PortfolioRisk(
                    total_exposure_dollars=Decimal("0"),
                    position_count=0,
                    max_drawdown_dollars=Decimal("0"),
                    correlation_risk=Decimal("0"),
                    concentration_risk=Decimal("0"),
                    available_capital=Decimal("0"),
                    risk_score=Decimal("0")
                )

            # Calculate basic metrics
            total_exposure = sum(p.dollar_value for p in positions)
            position_count = len(positions)

            # Calculate drawdown
            max_drawdown = await self._calculate_max_drawdown(positions)

            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk(positions)

            # Calculate concentration risk (Herfindahl index)
            if total_exposure > Decimal("0"):
                concentration_scores = [
                    (p.dollar_value / total_exposure) ** 2
                    for p in positions
                ]
                concentration_risk = sum(concentration_scores)
            else:
                concentration_risk = Decimal("0")

            # Get available capital
            account = await self.repository.get_account(self.account_id)
            available_capital = account.balance - total_exposure

            # Calculate overall risk score (0-100)
            risk_components = []

            # Position count risk (more positions = higher risk)
            position_risk = min(Decimal("30"), position_count * Decimal("5"))
            risk_components.append(position_risk)

            # Correlation risk (0-30 points)
            corr_risk_points = correlation_risk * Decimal("30")
            risk_components.append(corr_risk_points)

            # Concentration risk (0-20 points)
            conc_risk_points = concentration_risk * Decimal("20")
            risk_components.append(conc_risk_points)

            # Drawdown risk (0-20 points)
            if total_exposure > Decimal("0"):
                drawdown_ratio = max_drawdown / total_exposure
                drawdown_points = min(Decimal("20"), drawdown_ratio * Decimal("100"))
            else:
                drawdown_points = Decimal("0")
            risk_components.append(drawdown_points)

            risk_score = sum(risk_components)

            # Generate warnings
            warnings = []
            if correlation_risk > self.MAX_CORRELATION_WARNING:
                warnings.append(f"High correlation risk: {correlation_risk:.2%}")
            if concentration_risk > Decimal("0.3"):
                warnings.append(f"High concentration in few positions: {concentration_risk:.2%}")
            if position_count >= self._limits.max_positions_global * Decimal("0.8"):
                warnings.append(f"Approaching max position limit: {position_count}/{self._limits.max_positions_global}")
            if available_capital < total_exposure * Decimal("0.1"):
                warnings.append(f"Low available capital: ${available_capital:.2f}")

            return PortfolioRisk(
                total_exposure_dollars=total_exposure,
                position_count=position_count,
                max_drawdown_dollars=max_drawdown,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                available_capital=available_capital,
                risk_score=risk_score,
                warnings=warnings
            )

    async def add_position(self, position: Position) -> None:
        """Add a new position to tracking.
        
        Args:
            position: Position object to add
        """
        async with self._lock:
            self._active_positions[position.symbol] = position
            logger.info(
                "position_added",
                symbol=position.symbol,
                size=position.quantity,
                value=position.dollar_value
            )

    async def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking.
        
        Args:
            symbol: Symbol of position to remove
        """
        async with self._lock:
            if symbol in self._active_positions:
                del self._active_positions[symbol]
                logger.info("position_removed", symbol=symbol)

    async def update_correlations(self, correlations: dict[tuple[str, str], Decimal]) -> None:
        """Update position correlation matrix.
        
        Args:
            correlations: Dictionary of (symbol1, symbol2) -> correlation coefficient
        """
        async with self._lock:
            self._position_correlations = correlations

            # Check for high correlations
            for (sym1, sym2), corr in correlations.items():
                if abs(corr) > self.MAX_CORRELATION_WARNING:
                    logger.warning(
                        "high_correlation_detected",
                        symbol1=sym1,
                        symbol2=sym2,
                        correlation=corr
                    )

    # Private methods

    async def _load_portfolio_limits(self) -> None:
        """Load portfolio limits from database."""
        limits = await self.repository.get_portfolio_limits(self.account_id)

        # Get tier-specific defaults if no custom limits
        if not limits:
            tier = await self.state_manager.get_current_tier(self.account_id)
            limits = self._get_default_limits(tier)

        self._limits = limits

    async def _load_active_positions(self) -> None:
        """Load active positions from database."""
        positions = await self.repository.get_open_positions(self.account_id)
        self._active_positions = {p.symbol: p for p in positions}

    def _get_default_limits(self, tier: Tier) -> PortfolioLimits:
        """Get default limits based on tier.
        
        Args:
            tier: Current account tier
            
        Returns:
            Default PortfolioLimits for the tier
        """
        tier_defaults = {
            Tier.HUNTER: {
                "max_positions": 5,
                "max_exposure": Decimal("10000"),
                "per_pair_size": Decimal("2"),
                "per_pair_dollars": Decimal("3000")
            },
            Tier.STRATEGIST: {
                "max_positions": 10,
                "max_exposure": Decimal("50000"),
                "per_pair_size": Decimal("5"),
                "per_pair_dollars": Decimal("10000")
            },
            Tier.ARCHITECT: {
                "max_positions": 20,
                "max_exposure": Decimal("200000"),
                "per_pair_size": Decimal("10"),
                "per_pair_dollars": Decimal("30000")
            }
        }

        defaults = tier_defaults.get(tier, tier_defaults[Tier.HUNTER])

        return PortfolioLimits(
            max_positions_global=defaults["max_positions"],
            max_exposure_global_dollars=defaults["max_exposure"],
            per_pair_limits={},
            default_pair_limit_size=defaults["per_pair_size"],
            default_pair_limit_dollars=defaults["per_pair_dollars"]
        )

    def _get_pair_limits(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Get limits for a specific pair.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (max_size, max_dollars)
        """
        if symbol in self._limits.per_pair_limits:
            return self._limits.per_pair_limits[symbol]
        return (
            self._limits.default_pair_limit_size,
            self._limits.default_pair_limit_dollars
        )

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price
        """
        # This would normally call the market data service
        # For now, return a placeholder
        return await self.repository.get_latest_price(symbol)

    async def _calculate_correlation_penalty(self, symbol: str) -> Decimal:
        """Calculate correlation penalty for a symbol.
        
        Args:
            symbol: Trading pair to check
            
        Returns:
            Penalty factor (0-1) based on correlation with existing positions
        """
        if not self._active_positions:
            return Decimal("0")

        max_correlation = Decimal("0")
        for existing_symbol in self._active_positions.keys():
            if existing_symbol == symbol:
                continue

            # Check both orderings
            corr1 = self._position_correlations.get((symbol, existing_symbol), Decimal("0"))
            corr2 = self._position_correlations.get((existing_symbol, symbol), Decimal("0"))
            correlation = max(abs(corr1), abs(corr2))

            max_correlation = max(max_correlation, correlation)

        # Apply penalty curve
        if max_correlation < self.MAX_CORRELATION_WARNING:
            return Decimal("0")
        elif max_correlation < self.MAX_CORRELATION_RISK_ADJUST:
            # Linear penalty from 0 to 0.5
            range_size = self.MAX_CORRELATION_RISK_ADJUST - self.MAX_CORRELATION_WARNING
            penalty_range = max_correlation - self.MAX_CORRELATION_WARNING
            return (penalty_range / range_size) * Decimal("0.5")
        else:
            # Heavy penalty above 0.8 correlation
            return Decimal("0.75")

    async def _calculate_max_drawdown(self, positions: list[Position]) -> Decimal:
        """Calculate maximum drawdown across positions.
        
        Args:
            positions: List of positions
            
        Returns:
            Maximum drawdown in dollars
        """
        if not positions:
            return Decimal("0")

        # Sum of negative P&Ls
        losses = [p.pnl_dollars for p in positions if p.pnl_dollars < Decimal("0")]
        return abs(sum(losses)) if losses else Decimal("0")

    async def _calculate_correlation_risk(self, positions: list[Position]) -> Decimal:
        """Calculate overall correlation risk.
        
        Args:
            positions: List of positions
            
        Returns:
            Correlation risk score (0-1)
        """
        if len(positions) < 2:
            return Decimal("0")

        correlations = []
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                corr1 = self._position_correlations.get((pos1.symbol, pos2.symbol), Decimal("0"))
                corr2 = self._position_correlations.get((pos2.symbol, pos1.symbol), Decimal("0"))
                correlation = max(abs(corr1), abs(corr2))

                # Weight by position sizes
                weight = (pos1.dollar_value + pos2.dollar_value) / Decimal("2")
                correlations.append((correlation, weight))

        if not correlations:
            return Decimal("0")

        # Weighted average correlation
        total_weight = sum(w for _, w in correlations)
        if total_weight > Decimal("0"):
            weighted_corr = sum(c * w for c, w in correlations) / total_weight
            return min(Decimal("1"), weighted_corr)

        return Decimal("0")
