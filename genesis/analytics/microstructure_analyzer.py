"""Main Microstructure Analysis Module - Integrates all components."""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd
import structlog

from genesis.analytics.large_trader_detection import LargeTraderDetector
from genesis.analytics.market_manipulation import MarketManipulationDetector
from genesis.analytics.order_flow_analysis import OrderFlowAnalyzer
from genesis.analytics.price_impact_model import PriceImpactModel
from genesis.core.events import Event
from genesis.engine.event_bus import EventBus
from genesis.exchange.order_book_manager import OrderBookManager, OrderBookSnapshot

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market microstructure regimes."""

    NORMAL = "normal"
    STRESSED = "stressed"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    TOXIC = "toxic"


@dataclass
class ExecutionMetrics:
    """Optimal execution timing metrics."""

    symbol: str
    timestamp: datetime
    optimal_time: datetime  # Best time to execute
    participation_rate: Decimal  # Recommended rate
    expected_spread: Decimal  # Expected spread at optimal time
    liquidity_score: Decimal  # Expected liquidity (0-100)
    urgency_score: Decimal  # How urgent is execution (0-100)
    confidence: Decimal


@dataclass
class MarketMakerProfile:
    """Market maker behavior profile."""

    symbol: str
    maker_id: str  # Identified market maker
    quote_frequency: Decimal  # Quotes per minute
    spread_preference: Decimal  # Typical spread maintained
    inventory_bias: str  # 'long', 'short', 'neutral'
    withdrawal_triggers: list[str]  # Conditions causing withdrawal
    activity_periods: list[tuple[datetime, datetime]]  # Active time periods


@dataclass
class ToxicityScore:
    """Pair toxicity assessment."""

    symbol: str
    timestamp: datetime
    toxicity_score: Decimal  # 0-100 (higher = more toxic)
    adverse_selection: Decimal  # Adverse selection component
    informed_trading_probability: Decimal  # PIN score
    manipulation_frequency: Decimal  # Manipulation events per hour
    effective_spread: Decimal  # Actual trading cost
    realized_spread: Decimal  # Post-trade price movement
    recommendation: str  # 'avoid', 'caution', 'safe'


@dataclass
class MicrostructureState:
    """Complete microstructure state."""

    symbol: str
    timestamp: datetime
    regime: MarketRegime
    regime_confidence: Decimal
    transition_probability: dict[MarketRegime, Decimal]
    flow_imbalance: Decimal
    whale_activity: bool
    manipulation_detected: bool
    toxicity: Decimal
    execution_quality: Decimal


class ExecutionOptimizer:
    """Optimal execution timing engine."""

    def __init__(self):
        """Initialize execution optimizer."""
        self.volume_patterns: dict[str, pd.DataFrame] = {}
        self.spread_patterns: dict[str, pd.DataFrame] = {}
        self.participation_models: dict[str, dict] = {}

    def analyze_intraday_patterns(
        self, symbol: str, historical_data: pd.DataFrame
    ) -> dict[str, any]:
        """Analyze intraday volume and spread patterns.

        Args:
            symbol: Trading symbol
            historical_data: Historical trade data

        Returns:
            Pattern analysis results
        """
        if historical_data.empty:
            return {}

        # Group by hour of day
        historical_data["hour"] = pd.to_datetime(historical_data["timestamp"]).dt.hour

        # Volume patterns
        volume_by_hour = historical_data.groupby("hour")["volume"].agg(["mean", "std"])

        # Spread patterns
        spread_by_hour = historical_data.groupby("hour")["spread"].agg(["mean", "std"])

        # Find optimal execution windows
        optimal_hours = volume_by_hour[
            (volume_by_hour["mean"] > volume_by_hour["mean"].median())
            & (spread_by_hour["mean"] < spread_by_hour["mean"].median())
        ].index.tolist()

        return {
            "volume_pattern": volume_by_hour.to_dict(),
            "spread_pattern": spread_by_hour.to_dict(),
            "optimal_hours": optimal_hours,
            "high_liquidity_periods": volume_by_hour.nlargest(3, "mean").index.tolist(),
            "low_spread_periods": spread_by_hour.nsmallest(3, "mean").index.tolist(),
        }

    def calculate_optimal_participation(
        self, total_size: Decimal, time_horizon: timedelta, urgency: Decimal
    ) -> Decimal:
        """Calculate optimal participation rate.

        Args:
            total_size: Total order size
            time_horizon: Available time for execution
            urgency: Urgency factor (0-1)

        Returns:
            Optimal participation rate
        """
        # Almgren-Chriss inspired calculation
        base_rate = total_size / Decimal(str(time_horizon.total_seconds() / 60))

        # Adjust for urgency
        urgency_multiplier = Decimal("1") + urgency * Decimal("2")

        optimal_rate = base_rate * urgency_multiplier

        # Cap at 30% participation
        return min(optimal_rate, Decimal("0.3"))

    def get_execution_schedule(
        self, symbol: str, total_quantity: Decimal, duration_minutes: int
    ) -> list[tuple[datetime, Decimal]]:
        """Create execution schedule for large orders.

        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to execute
            duration_minutes: Execution duration in minutes

        Returns:
            List of (timestamp, quantity) tuples
        """
        schedule = []
        now = datetime.now(UTC)

        # Simple TWAP with randomization
        slices = max(10, duration_minutes // 5)
        slice_size = total_quantity / Decimal(str(slices))

        for i in range(slices):
            # Add some randomness to avoid detection
            random_factor = Decimal(str(0.8 + np.random.random() * 0.4))
            adjusted_size = slice_size * random_factor

            timestamp = now + timedelta(minutes=i * duration_minutes / slices)
            schedule.append((timestamp, adjusted_size))

        return schedule


class MarketMakerAnalyzer:
    """Analyzes market maker behavior patterns."""

    def __init__(self):
        """Initialize market maker analyzer."""
        self.maker_profiles: dict[str, list[MarketMakerProfile]] = {}
        self.quote_history: dict[str, deque] = {}

    def identify_market_makers(
        self, symbol: str, order_book_history: list[OrderBookSnapshot]
    ) -> list[str]:
        """Identify potential market makers.

        Args:
            symbol: Trading symbol
            order_book_history: Historical order book snapshots

        Returns:
            List of identified market maker IDs
        """
        if len(order_book_history) < 100:
            return []

        # Track persistent quotes at multiple levels
        quote_persistence = {}

        for snapshot in order_book_history:
            # Look for orders that appear consistently
            for bid in snapshot.bids[:5]:
                key = f"bid_{bid.price}"
                quote_persistence[key] = quote_persistence.get(key, 0) + 1

            for ask in snapshot.asks[:5]:
                key = f"ask_{ask.price}"
                quote_persistence[key] = quote_persistence.get(key, 0) + 1

        # Market makers have persistent quotes
        threshold = len(order_book_history) * 0.5
        persistent_quotes = [k for k, v in quote_persistence.items() if v > threshold]

        # Group by price patterns to identify entities
        maker_ids = []
        for i, quote_group in enumerate(self._group_quotes(persistent_quotes)):
            if len(quote_group) >= 3:  # Multiple price levels
                maker_ids.append(f"mm_{symbol}_{i}")

        return maker_ids

    def _group_quotes(self, quotes: list[str]) -> list[list[str]]:
        """Group quotes that likely belong to same market maker.

        Args:
            quotes: List of quote identifiers

        Returns:
            Grouped quotes
        """
        # Simple clustering by price proximity
        groups = []
        used = set()

        for quote in quotes:
            if quote not in used:
                group = [quote]
                used.add(quote)
                # Find related quotes (simplified)
                for other in quotes:
                    if other not in used and self._are_related(quote, other):
                        group.append(other)
                        used.add(other)
                if group:
                    groups.append(group)

        return groups

    def _are_related(self, quote1: str, quote2: str) -> bool:
        """Check if two quotes are likely from same market maker.

        Args:
            quote1: First quote
            quote2: Second quote

        Returns:
            True if related
        """
        # Simplified: check if on same side and close in price
        side1 = quote1.split("_")[0]
        side2 = quote2.split("_")[0]

        return side1 == side2  # Same side

    def detect_withdrawal(
        self,
        symbol: str,
        before_snapshot: OrderBookSnapshot,
        after_snapshot: OrderBookSnapshot,
    ) -> bool:
        """Detect market maker withdrawal.

        Args:
            symbol: Trading symbol
            before_snapshot: Order book before
            after_snapshot: Order book after

        Returns:
            True if withdrawal detected
        """
        # Check for significant reduction in depth
        before_depth = len(before_snapshot.bids) + len(before_snapshot.asks)
        after_depth = len(after_snapshot.bids) + len(after_snapshot.asks)

        # Check for spread widening
        before_spread = before_snapshot.spread_bps or 0
        after_spread = after_snapshot.spread_bps or 0

        # Withdrawal indicators
        depth_reduction = after_depth < before_depth * 0.5
        spread_widening = after_spread > before_spread * 2

        return depth_reduction or spread_widening


class ToxicityScorer:
    """Scores trading pair toxicity."""

    def __init__(self):
        """Initialize toxicity scorer."""
        self.adverse_selection_history: dict[str, deque] = {}
        self.pin_scores: dict[str, deque] = {}

    def calculate_toxicity(
        self,
        symbol: str,
        trades: list[dict],
        manipulation_events: int,
        vpin_score: Decimal,
    ) -> ToxicityScore:
        """Calculate comprehensive toxicity score.

        Args:
            symbol: Trading symbol
            trades: Recent trades
            manipulation_events: Number of manipulation events
            vpin_score: Current VPIN score

        Returns:
            Toxicity score
        """
        # Calculate adverse selection
        adverse_selection = self._calculate_adverse_selection(trades)

        # Calculate effective vs realized spread
        effective_spread, realized_spread = self._calculate_spreads(trades)

        # Combine components
        toxicity = (
            adverse_selection * Decimal("0.3")
            + vpin_score * Decimal("100") * Decimal("0.3")
            + Decimal(str(manipulation_events)) * Decimal("5") * Decimal("0.2")
            + (effective_spread - realized_spread).max(Decimal("0")) * Decimal("0.2")
        )

        # Determine recommendation
        if toxicity > Decimal("70"):
            recommendation = "avoid"
        elif toxicity > Decimal("40"):
            recommendation = "caution"
        else:
            recommendation = "safe"

        return ToxicityScore(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            toxicity_score=min(toxicity, Decimal("100")),
            adverse_selection=adverse_selection,
            informed_trading_probability=vpin_score,
            manipulation_frequency=Decimal(str(manipulation_events)),
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            recommendation=recommendation,
        )

    def _calculate_adverse_selection(self, trades: list[dict]) -> Decimal:
        """Calculate adverse selection component.

        Args:
            trades: Trade data

        Returns:
            Adverse selection score
        """
        if len(trades) < 10:
            return Decimal("0")

        # Check post-trade price movement
        adverse_moves = 0
        for i in range(len(trades) - 1):
            current_price = Decimal(str(trades[i].get("price", 0)))
            next_price = Decimal(str(trades[i + 1].get("price", 0)))
            side = trades[i].get("side", "buy")

            # Adverse selection: price moves against liquidity provider
            if (side == "buy" and next_price > current_price) or (side == "sell" and next_price < current_price):
                adverse_moves += 1

        return Decimal(str(adverse_moves * 100 / len(trades)))

    def _calculate_spreads(self, trades: list[dict]) -> tuple[Decimal, Decimal]:
        """Calculate effective and realized spreads.

        Args:
            trades: Trade data

        Returns:
            Tuple of (effective_spread, realized_spread)
        """
        if not trades:
            return Decimal("0"), Decimal("0")

        # Simplified calculation
        prices = [Decimal(str(t.get("price", 0))) for t in trades]

        if len(prices) < 2:
            return Decimal("0"), Decimal("0")

        # Effective spread: immediate cost
        effective = abs(prices[-1] - prices[-2]) / prices[-2] * Decimal("10000")

        # Realized spread: cost after price movement
        if len(prices) >= 10:
            realized = abs(prices[-1] - prices[-10]) / prices[-10] * Decimal("10000")
        else:
            realized = effective

        return effective, realized


class MicrostructureAnalyzer:
    """Main microstructure analysis coordinator."""

    def __init__(self, event_bus: EventBus):
        """Initialize microstructure analyzer.

        Args:
            event_bus: Event bus for publishing events
        """
        self.event_bus = event_bus

        # Initialize components
        self.order_book_manager = OrderBookManager(event_bus)
        self.flow_analyzer = OrderFlowAnalyzer(event_bus)
        self.whale_detector = LargeTraderDetector(event_bus)
        self.manipulation_detector = MarketManipulationDetector(event_bus)
        self.impact_model = PriceImpactModel()
        self.execution_optimizer = ExecutionOptimizer()
        self.market_maker_analyzer = MarketMakerAnalyzer()
        self.toxicity_scorer = ToxicityScorer()

        # State tracking
        self.current_states: dict[str, MicrostructureState] = {}
        self.regime_history: dict[str, deque] = {}

        # Hidden Markov Model parameters for regime detection
        self.transition_matrix = self._initialize_transition_matrix()

    def _initialize_transition_matrix(
        self,
    ) -> dict[MarketRegime, dict[MarketRegime, Decimal]]:
        """Initialize regime transition probabilities.

        Returns:
            Transition probability matrix
        """
        return {
            MarketRegime.NORMAL: {
                MarketRegime.NORMAL: Decimal("0.8"),
                MarketRegime.STRESSED: Decimal("0.1"),
                MarketRegime.TRENDING: Decimal("0.05"),
                MarketRegime.RANGE_BOUND: Decimal("0.04"),
                MarketRegime.TOXIC: Decimal("0.01"),
            },
            MarketRegime.STRESSED: {
                MarketRegime.NORMAL: Decimal("0.3"),
                MarketRegime.STRESSED: Decimal("0.5"),
                MarketRegime.TRENDING: Decimal("0.1"),
                MarketRegime.RANGE_BOUND: Decimal("0.05"),
                MarketRegime.TOXIC: Decimal("0.05"),
            },
            MarketRegime.TRENDING: {
                MarketRegime.NORMAL: Decimal("0.2"),
                MarketRegime.STRESSED: Decimal("0.05"),
                MarketRegime.TRENDING: Decimal("0.6"),
                MarketRegime.RANGE_BOUND: Decimal("0.1"),
                MarketRegime.TOXIC: Decimal("0.05"),
            },
            MarketRegime.RANGE_BOUND: {
                MarketRegime.NORMAL: Decimal("0.3"),
                MarketRegime.STRESSED: Decimal("0.05"),
                MarketRegime.TRENDING: Decimal("0.15"),
                MarketRegime.RANGE_BOUND: Decimal("0.5"),
                MarketRegime.TOXIC: Decimal("0"),
            },
            MarketRegime.TOXIC: {
                MarketRegime.NORMAL: Decimal("0.1"),
                MarketRegime.STRESSED: Decimal("0.3"),
                MarketRegime.TRENDING: Decimal("0.1"),
                MarketRegime.RANGE_BOUND: Decimal("0"),
                MarketRegime.TOXIC: Decimal("0.5"),
            },
        }

    async def analyze_market(
        self, symbol: str, order_book: OrderBookSnapshot, recent_trades: list[dict]
    ) -> MicrostructureState:
        """Perform comprehensive market microstructure analysis.

        Args:
            symbol: Trading symbol
            order_book: Current order book
            recent_trades: Recent trade data

        Returns:
            Complete microstructure state
        """
        # Update order book
        self.flow_analyzer.update_order_book(order_book)
        self.manipulation_detector.update_order_book(order_book)

        # Analyze flow imbalance
        flow_imbalance = (
            order_book.get_imbalance_ratio() if order_book else Decimal("0")
        )

        # Check for whale activity
        whale_activity = len(self.whale_detector.get_active_whales(symbol)) > 0

        # Check for manipulation
        manip_stats = self.manipulation_detector.get_manipulation_statistics(symbol)
        manipulation_detected = manip_stats.get("active_patterns", 0) > 0

        # Calculate toxicity
        vpin = self.whale_detector.vpin_history.get(symbol, [])
        current_vpin = vpin[-1].vpin if vpin else Decimal("0")
        toxicity_score = self.toxicity_scorer.calculate_toxicity(
            symbol, recent_trades, manip_stats.get("active_patterns", 0), current_vpin
        )

        # Detect regime
        regime, confidence = self._detect_regime(
            symbol,
            flow_imbalance,
            whale_activity,
            manipulation_detected,
            toxicity_score.toxicity_score,
        )

        # Calculate transition probabilities
        current_regime = self.current_states.get(
            symbol,
            MicrostructureState(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                regime=MarketRegime.NORMAL,
                regime_confidence=Decimal("0.5"),
                transition_probability={},
                flow_imbalance=Decimal("0"),
                whale_activity=False,
                manipulation_detected=False,
                toxicity=Decimal("0"),
                execution_quality=Decimal("50"),
            ),
        ).regime

        transition_probs = self.transition_matrix[current_regime]

        # Create state
        state = MicrostructureState(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            regime=regime,
            regime_confidence=confidence,
            transition_probability=transition_probs,
            flow_imbalance=flow_imbalance,
            whale_activity=whale_activity,
            manipulation_detected=manipulation_detected,
            toxicity=toxicity_score.toxicity_score,
            execution_quality=self._calculate_execution_quality(
                order_book, toxicity_score.toxicity_score
            ),
        )

        # Update current state
        self.current_states[symbol] = state

        # Store history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = deque(maxlen=100)
        self.regime_history[symbol].append(state)

        # Publish state change event
        await self._publish_state_change(state)

        return state

    def _detect_regime(
        self,
        symbol: str,
        flow_imbalance: Decimal,
        whale_activity: bool,
        manipulation_detected: bool,
        toxicity: Decimal,
    ) -> tuple[MarketRegime, Decimal]:
        """Detect current market regime using HMM.

        Args:
            symbol: Trading symbol
            flow_imbalance: Current flow imbalance
            whale_activity: Whale activity detected
            manipulation_detected: Manipulation detected
            toxicity: Toxicity score

        Returns:
            Tuple of (regime, confidence)
        """
        # Feature-based regime detection
        scores = {
            MarketRegime.NORMAL: Decimal("0.5"),
            MarketRegime.STRESSED: Decimal("0"),
            MarketRegime.TRENDING: Decimal("0"),
            MarketRegime.RANGE_BOUND: Decimal("0"),
            MarketRegime.TOXIC: Decimal("0"),
        }

        # Toxic regime
        if toxicity > Decimal("70"):
            scores[MarketRegime.TOXIC] += Decimal("0.8")
        elif manipulation_detected:
            scores[MarketRegime.TOXIC] += Decimal("0.4")

        # Stressed regime
        if whale_activity and abs(flow_imbalance) > Decimal("0.5"):
            scores[MarketRegime.STRESSED] += Decimal("0.6")

        # Trending regime
        if abs(flow_imbalance) > Decimal("0.3"):
            scores[MarketRegime.TRENDING] += Decimal("0.5")
            # Check historical flow
            flow_trend = self.flow_analyzer.get_flow_trend(symbol)
            if flow_trend in ["bullish", "bearish"]:
                scores[MarketRegime.TRENDING] += Decimal("0.3")

        # Range-bound regime
        if abs(flow_imbalance) < Decimal("0.1") and not whale_activity:
            scores[MarketRegime.RANGE_BOUND] += Decimal("0.4")

        # Select regime with highest score
        regime = max(scores, key=scores.get)
        confidence = scores[regime] / sum(scores.values())

        return regime, confidence

    def _calculate_execution_quality(
        self, order_book: OrderBookSnapshot | None, toxicity: Decimal
    ) -> Decimal:
        """Calculate execution quality score.

        Args:
            order_book: Order book snapshot
            toxicity: Toxicity score

        Returns:
            Execution quality (0-100)
        """
        quality = Decimal("50")  # Base quality

        if order_book:
            # Tight spread improves quality
            if order_book.spread_bps and order_book.spread_bps < 10:
                quality += Decimal("20")
            elif order_book.spread_bps and order_book.spread_bps < 20:
                quality += Decimal("10")

            # Good depth improves quality
            bid_depth = order_book.get_bid_volume(5)
            ask_depth = order_book.get_ask_volume(5)
            if bid_depth > Decimal("100") and ask_depth > Decimal("100"):
                quality += Decimal("20")

        # Toxicity reduces quality
        quality -= toxicity * Decimal("0.5")

        return max(Decimal("0"), min(Decimal("100"), quality))

    async def _publish_state_change(self, state: MicrostructureState) -> None:
        """Publish microstructure state change event.

        Args:
            state: Current microstructure state
        """
        await self.event_bus.publish(
            Event(
                type="microstructure_state_changed",
                data={
                    "symbol": state.symbol,
                    "regime": state.regime.value,
                    "regime_confidence": float(state.regime_confidence),
                    "flow_imbalance": float(state.flow_imbalance),
                    "whale_activity": state.whale_activity,
                    "manipulation_detected": state.manipulation_detected,
                    "toxicity": float(state.toxicity),
                    "execution_quality": float(state.execution_quality),
                    "timestamp": state.timestamp.isoformat(),
                },
            )
        )

    async def start(self, symbols: list[str]) -> None:
        """Start microstructure analysis.

        Args:
            symbols: List of symbols to analyze
        """
        await self.order_book_manager.start(symbols)
        logger.info("microstructure_analyzer_started", symbols=symbols)

    async def stop(self) -> None:
        """Stop microstructure analysis."""
        await self.order_book_manager.stop()
        logger.info("microstructure_analyzer_stopped")
