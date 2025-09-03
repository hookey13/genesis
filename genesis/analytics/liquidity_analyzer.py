"""
Liquidity Analysis System

Provides comprehensive liquidity depth analysis, market impact estimation,
and microstructure anomaly detection for position sizing and risk management.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from genesis.core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class LiquidityAnalyzerConfig:
    """Configuration for liquidity analyzer"""

    depth_levels: List[int] = field(default_factory=lambda: [5, 10, 20])
    imbalance_threshold: Decimal = Decimal("2.0")
    anomaly_detection_window: int = 100
    liquidity_score_weights: Dict[str, Decimal] = field(default_factory=lambda: {
        "depth": Decimal("0.4"),
        "spread": Decimal("0.4"),
        "volatility": Decimal("0.2")
    })
    market_impact_eta: Decimal = Decimal("0.1")  # Temporary impact coefficient
    market_impact_gamma: Decimal = Decimal("0.05")  # Permanent impact coefficient
    quote_stuffing_threshold: int = 50  # Updates per second
    layering_detection_levels: int = 5


@dataclass
class LiquidityDepthMetrics:
    """Metrics for liquidity depth at multiple levels"""

    symbol: str
    bid_depth: Dict[int, Decimal]  # Level -> cumulative volume
    ask_depth: Dict[int, Decimal]  # Level -> cumulative volume
    total_bid_volume: Decimal
    total_ask_volume: Decimal
    bid_liquidity_gaps: List[Tuple[Decimal, Decimal]]  # (price, gap_size)
    ask_liquidity_gaps: List[Tuple[Decimal, Decimal]]  # (price, gap_size)
    weighted_mid_price: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MarketImpactEstimate:
    """Market impact estimation for an order"""

    symbol: str
    order_size: Decimal
    side: str  # 'buy' or 'sell'
    temporary_impact_bps: Decimal
    permanent_impact_bps: Decimal
    total_impact_bps: Decimal
    expected_slippage: Decimal
    execution_price: Decimal
    confidence_interval: Tuple[Decimal, Decimal]  # (lower, upper)


@dataclass
class OrderBookImbalance:
    """Order book imbalance metrics"""

    symbol: str
    imbalance_ratio: Decimal
    bid_pressure: Decimal
    ask_pressure: Decimal
    flow_imbalance: Decimal
    is_one_sided: bool
    direction: str  # 'bid_heavy', 'ask_heavy', 'balanced'
    severity: str  # 'low', 'medium', 'high', 'extreme'


@dataclass
class MicrostructureAnomaly:
    """Detected microstructure anomaly"""

    symbol: str
    anomaly_type: str  # 'quote_stuffing', 'layering', 'spoofing', 'unusual_spread'
    severity: str  # 'low', 'medium', 'high', 'critical'
    details: Dict[str, any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class LiquidityScore:
    """Composite liquidity score"""

    symbol: str
    overall_score: Decimal  # 0-100
    depth_score: Decimal
    spread_score: Decimal
    stability_score: Decimal
    time_of_day_adjustment: Decimal
    final_score: Decimal  # After adjustments
    liquidity_grade: str  # 'A', 'B', 'C', 'D', 'F'


class LiquidityAnalyzer:
    """
    Comprehensive liquidity analysis for market microstructure
    """

    def __init__(self, config: Optional[LiquidityAnalyzerConfig] = None):
        """
        Initialize liquidity analyzer
        
        Args:
            config: Configuration parameters
        """
        self.config = config or LiquidityAnalyzerConfig()
        
        # Thread-safe collections
        self._lock = asyncio.Lock()
        
        # Orderbook tracking
        self._orderbook_cache: Dict[str, dict] = {}
        self._depth_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.anomaly_detection_window)
        )
        
        # Market impact models
        self._impact_coefficients: Dict[str, Dict[str, Decimal]] = {}
        self._average_daily_volumes: Dict[str, Decimal] = {}
        
        # Imbalance tracking
        self._imbalance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Anomaly detection
        self._update_rates: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._anomaly_scores: Dict[str, List[MicrostructureAnomaly]] = defaultdict(list)
        
        # Liquidity scores
        self._liquidity_scores: Dict[str, LiquidityScore] = {}
        self._score_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        self._logger = logger.bind(component="LiquidityAnalyzer")

    async def assess_depth(self, symbol: str, orderbook: dict) -> LiquidityDepthMetrics:
        """
        Assess liquidity depth at multiple levels
        
        Args:
            symbol: Trading pair symbol
            orderbook: Order book data
            
        Returns:
            Liquidity depth metrics
        """
        async with self._lock:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                raise ValidationError(f"Invalid orderbook data for {symbol}")
            
            # Cache orderbook
            self._orderbook_cache[symbol] = orderbook
            
            # Calculate cumulative depth at each level
            bid_depth = {}
            ask_depth = {}
            
            for level in self.config.depth_levels:
                bid_volume = Decimal("0")
                ask_volume = Decimal("0")
                
                # Sum volumes up to level
                for i in range(min(level, len(bids))):
                    bid_volume += Decimal(str(bids[i][1]))
                
                for i in range(min(level, len(asks))):
                    ask_volume += Decimal(str(asks[i][1]))
                
                bid_depth[level] = bid_volume
                ask_depth[level] = ask_volume
            
            # Calculate total volumes
            total_bid = sum(Decimal(str(b[1])) for b in bids)
            total_ask = sum(Decimal(str(a[1])) for a in asks)
            
            # Identify liquidity gaps
            bid_gaps = self._identify_liquidity_gaps(bids, "bid")
            ask_gaps = self._identify_liquidity_gaps(asks, "ask")
            
            # Calculate weighted mid price
            if total_bid + total_ask > 0:
                bid_price = Decimal(str(bids[0][0]))
                ask_price = Decimal(str(asks[0][0]))
                weighted_mid = (bid_price * total_ask + ask_price * total_bid) / (total_bid + total_ask)
            else:
                weighted_mid = (Decimal(str(bids[0][0])) + Decimal(str(asks[0][0]))) / Decimal("2")
            
            metrics = LiquidityDepthMetrics(
                symbol=symbol,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                total_bid_volume=total_bid,
                total_ask_volume=total_ask,
                bid_liquidity_gaps=bid_gaps,
                ask_liquidity_gaps=ask_gaps,
                weighted_mid_price=weighted_mid
            )
            
            # Store in history
            self._depth_history[symbol].append(metrics)
            
            # Track depth changes
            if len(self._depth_history[symbol]) > 1:
                prev_metrics = self._depth_history[symbol][-2]
                depth_change = abs(total_bid - prev_metrics.total_bid_volume) + \
                               abs(total_ask - prev_metrics.total_ask_volume)
                
                if depth_change > total_bid + total_ask:
                    self._logger.warning(
                        "Large depth change detected",
                        symbol=symbol,
                        change=float(depth_change),
                        total_depth=float(total_bid + total_ask)
                    )
            
            return metrics

    def _identify_liquidity_gaps(
        self,
        levels: List[List],
        side: str
    ) -> List[Tuple[Decimal, Decimal]]:
        """Identify gaps in liquidity"""
        gaps = []
        
        if len(levels) < 2:
            return gaps
        
        for i in range(1, min(20, len(levels))):
            curr_price = Decimal(str(levels[i][0]))
            prev_price = Decimal(str(levels[i-1][0]))
            
            # Calculate price gap
            if side == "bid":
                gap = prev_price - curr_price
            else:
                gap = curr_price - prev_price
            
            # Check if gap is significant (> 0.1% of price)
            if gap > prev_price * Decimal("0.001"):
                gaps.append((curr_price, gap))
        
        return gaps

    async def estimate_impact(
        self,
        symbol: str,
        order_size: Decimal,
        side: str,
        average_daily_volume: Optional[Decimal] = None
    ) -> MarketImpactEstimate:
        """
        Estimate market impact using square-root model
        
        Args:
            symbol: Trading pair symbol
            order_size: Size of the order
            side: 'buy' or 'sell'
            average_daily_volume: ADV for the pair
            
        Returns:
            Market impact estimate
        """
        async with self._lock:
            if symbol not in self._orderbook_cache:
                raise ValidationError(f"No orderbook data for {symbol}")
            
            orderbook = self._orderbook_cache[symbol]
            
            # Get or estimate ADV
            adv = average_daily_volume or self._average_daily_volumes.get(
                symbol, Decimal("1000000")  # Default 1M units
            )
            
            # Update stored ADV if provided
            if average_daily_volume:
                self._average_daily_volumes[symbol] = average_daily_volume
            
            # Get current spread and volatility
            best_bid = Decimal(str(orderbook["bids"][0][0]))
            best_ask = Decimal(str(orderbook["asks"][0][0]))
            mid_price = (best_bid + best_ask) / Decimal("2")
            spread_bps = ((best_ask - best_bid) / mid_price) * Decimal("10000")
            
            # Estimate volatility (simplified - use historical if available)
            volatility = spread_bps / Decimal("10")  # Rough estimate
            
            # Calculate participation rate
            participation = order_size / adv
            
            # Square-root market impact model (Almgren-Chriss)
            eta = self.config.market_impact_eta
            gamma = self.config.market_impact_gamma
            
            # Temporary impact (square-root of participation)
            temp_impact = eta * participation.sqrt() * volatility
            
            # Permanent impact (linear in participation)
            perm_impact = gamma * participation * volatility
            
            # Total impact
            total_impact = temp_impact + perm_impact
            
            # Calculate expected execution price
            if side == "buy":
                execution_price = mid_price * (Decimal("1") + total_impact / Decimal("10000"))
                slippage = execution_price - best_ask
            else:
                execution_price = mid_price * (Decimal("1") - total_impact / Decimal("10000"))
                slippage = best_bid - execution_price
            
            # Estimate confidence interval (simplified)
            lower_bound = total_impact * Decimal("0.7")
            upper_bound = total_impact * Decimal("1.3")
            
            estimate = MarketImpactEstimate(
                symbol=symbol,
                order_size=order_size,
                side=side,
                temporary_impact_bps=temp_impact,
                permanent_impact_bps=perm_impact,
                total_impact_bps=total_impact,
                expected_slippage=slippage,
                execution_price=execution_price,
                confidence_interval=(lower_bound, upper_bound)
            )
            
            self._logger.debug(
                "Market impact estimated",
                symbol=symbol,
                order_size=float(order_size),
                side=side,
                total_impact_bps=float(total_impact),
                execution_price=float(execution_price)
            )
            
            return estimate

    async def calculate_imbalance(self, symbol: str, orderbook: dict) -> OrderBookImbalance:
        """
        Calculate bid-ask imbalance
        
        Args:
            symbol: Trading pair symbol
            orderbook: Order book data
            
        Returns:
            Order book imbalance metrics
        """
        async with self._lock:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if not bids or not asks:
                return OrderBookImbalance(
                    symbol=symbol,
                    imbalance_ratio=Decimal("1.0"),
                    bid_pressure=Decimal("0"),
                    ask_pressure=Decimal("0"),
                    flow_imbalance=Decimal("0"),
                    is_one_sided=False,
                    direction="balanced",
                    severity="low"
                )
            
            # Calculate weighted volumes (closer levels have more weight)
            bid_pressure = Decimal("0")
            ask_pressure = Decimal("0")
            
            for i in range(min(10, len(bids))):
                weight = Decimal("1") / Decimal(str(i + 1))
                bid_pressure += Decimal(str(bids[i][1])) * weight
            
            for i in range(min(10, len(asks))):
                weight = Decimal("1") / Decimal(str(i + 1))
                ask_pressure += Decimal(str(asks[i][1])) * weight
            
            # Calculate imbalance ratio
            total_pressure = bid_pressure + ask_pressure
            if total_pressure > 0:
                imbalance_ratio = bid_pressure / ask_pressure if ask_pressure > 0 else Decimal("999")
                flow_imbalance = (bid_pressure - ask_pressure) / total_pressure
            else:
                imbalance_ratio = Decimal("1.0")
                flow_imbalance = Decimal("0")
            
            # Determine direction and severity
            if imbalance_ratio > self.config.imbalance_threshold:
                direction = "bid_heavy"
                is_one_sided = True
            elif imbalance_ratio < Decimal("1") / self.config.imbalance_threshold:
                direction = "ask_heavy"
                is_one_sided = True
            else:
                direction = "balanced"
                is_one_sided = False
            
            # Determine severity
            if abs(flow_imbalance) < Decimal("0.2"):
                severity = "low"
            elif abs(flow_imbalance) < Decimal("0.4"):
                severity = "medium"
            elif abs(flow_imbalance) < Decimal("0.6"):
                severity = "high"
            else:
                severity = "extreme"
            
            imbalance = OrderBookImbalance(
                symbol=symbol,
                imbalance_ratio=imbalance_ratio,
                bid_pressure=bid_pressure,
                ask_pressure=ask_pressure,
                flow_imbalance=flow_imbalance,
                is_one_sided=is_one_sided,
                direction=direction,
                severity=severity
            )
            
            # Store in history
            self._imbalance_history[symbol].append(imbalance)
            
            # Check for persistent imbalance
            if len(self._imbalance_history[symbol]) >= 10:
                recent_imbalances = list(self._imbalance_history[symbol])[-10:]
                one_sided_count = sum(1 for i in recent_imbalances if i.is_one_sided)
                
                if one_sided_count >= 8:
                    self._logger.warning(
                        "Persistent order imbalance detected",
                        symbol=symbol,
                        direction=direction,
                        one_sided_count=one_sided_count
                    )
            
            return imbalance

    async def detect_anomalies(self, symbol: str, orderbook: dict) -> List[MicrostructureAnomaly]:
        """
        Detect microstructure anomalies
        
        Args:
            symbol: Trading pair symbol
            orderbook: Order book data
            
        Returns:
            List of detected anomalies
        """
        async with self._lock:
            anomalies = []
            now = datetime.now(UTC)
            
            # Track update rate
            self._update_rates[symbol].append(now)
            
            # Clean old updates
            cutoff = now - timedelta(seconds=1)
            self._update_rates[symbol] = deque(
                [t for t in self._update_rates[symbol] if t > cutoff],
                maxlen=100
            )
            
            # Check for quote stuffing
            updates_per_second = len(self._update_rates[symbol])
            if updates_per_second > self.config.quote_stuffing_threshold:
                anomaly = MicrostructureAnomaly(
                    symbol=symbol,
                    anomaly_type="quote_stuffing",
                    severity="high" if updates_per_second > 100 else "medium",
                    details={
                        "updates_per_second": updates_per_second,
                        "threshold": self.config.quote_stuffing_threshold
                    }
                )
                anomalies.append(anomaly)
            
            # Check for layering/spoofing
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            if len(bids) >= self.config.layering_detection_levels:
                layering_detected = self._detect_layering(bids, "bid")
                if layering_detected:
                    anomaly = MicrostructureAnomaly(
                        symbol=symbol,
                        anomaly_type="layering",
                        severity=layering_detected["severity"],
                        details=layering_detected
                    )
                    anomalies.append(anomaly)
            
            if len(asks) >= self.config.layering_detection_levels:
                layering_detected = self._detect_layering(asks, "ask")
                if layering_detected:
                    anomaly = MicrostructureAnomaly(
                        symbol=symbol,
                        anomaly_type="layering",
                        severity=layering_detected["severity"],
                        details=layering_detected
                    )
                    anomalies.append(anomaly)
            
            # Check for unusual spread patterns
            if bids and asks:
                best_bid = Decimal(str(bids[0][0]))
                best_ask = Decimal(str(asks[0][0]))
                spread = best_ask - best_bid
                
                # Check for inverted or zero spread
                if spread <= 0:
                    anomaly = MicrostructureAnomaly(
                        symbol=symbol,
                        anomaly_type="unusual_spread",
                        severity="critical",
                        details={
                            "type": "inverted_spread" if spread < 0 else "zero_spread",
                            "bid": float(best_bid),
                            "ask": float(best_ask)
                        }
                    )
                    anomalies.append(anomaly)
                
                # Check for extremely wide spread
                mid_price = (best_bid + best_ask) / Decimal("2")
                spread_pct = (spread / mid_price) * Decimal("100")
                
                if spread_pct > Decimal("1.0"):  # > 1% spread
                    anomaly = MicrostructureAnomaly(
                        symbol=symbol,
                        anomaly_type="unusual_spread",
                        severity="high" if spread_pct > Decimal("2.0") else "medium",
                        details={
                            "type": "wide_spread",
                            "spread_pct": float(spread_pct),
                            "bid": float(best_bid),
                            "ask": float(best_ask)
                        }
                    )
                    anomalies.append(anomaly)
            
            # Store anomalies
            if anomalies:
                self._anomaly_scores[symbol].extend(anomalies)
                
                # Keep only recent anomalies (last hour)
                cutoff_time = now - timedelta(hours=1)
                self._anomaly_scores[symbol] = [
                    a for a in self._anomaly_scores[symbol]
                    if a.timestamp > cutoff_time
                ]
                
                for anomaly in anomalies:
                    self._logger.warning(
                        "Microstructure anomaly detected",
                        symbol=symbol,
                        anomaly_type=anomaly.anomaly_type,
                        severity=anomaly.severity,
                        details=anomaly.details
                    )
            
            return anomalies

    def _detect_layering(self, levels: List[List], side: str) -> Optional[Dict]:
        """Detect potential layering/spoofing patterns"""
        if len(levels) < self.config.layering_detection_levels:
            return None
        
        volumes = [Decimal(str(level[1])) for level in levels[:self.config.layering_detection_levels]]
        
        # Check for suspicious volume pattern (large orders away from best price)
        best_volume = volumes[0]
        distant_volumes = volumes[2:]  # Levels 3-5
        
        # Calculate average distant volume
        avg_distant = sum(distant_volumes) / len(distant_volumes)
        
        # Layering pattern: distant orders significantly larger than best
        if avg_distant > best_volume * Decimal("3"):
            severity = "high" if avg_distant > best_volume * Decimal("5") else "medium"
            
            return {
                "side": side,
                "best_volume": float(best_volume),
                "avg_distant_volume": float(avg_distant),
                "ratio": float(avg_distant / best_volume),
                "severity": severity
            }
        
        return None

    async def calculate_vwap_spread(
        self,
        symbol: str,
        orderbook: dict,
        lookback_seconds: int = 300
    ) -> Decimal:
        """
        Calculate volume-weighted average spread
        
        Args:
            symbol: Trading pair symbol
            orderbook: Current orderbook
            lookback_seconds: Time window for calculation
            
        Returns:
            VWAP spread in basis points
        """
        async with self._lock:
            # Get depth history
            history = list(self._depth_history[symbol])
            
            if not history:
                # Calculate from current orderbook only
                best_bid = Decimal(str(orderbook["bids"][0][0]))
                best_ask = Decimal(str(orderbook["asks"][0][0]))
                mid_price = (best_bid + best_ask) / Decimal("2")
                return ((best_ask - best_bid) / mid_price) * Decimal("10000")
            
            # Filter by time
            cutoff = datetime.now(UTC) - timedelta(seconds=lookback_seconds)
            recent_history = [h for h in history if h.timestamp >= cutoff]
            
            if not recent_history:
                recent_history = history[-1:]  # Use most recent if no data in window
            
            # Calculate VWAP spread
            total_volume = Decimal("0")
            weighted_spread = Decimal("0")
            
            for metrics in recent_history:
                # Use current orderbook for spread calculation
                if symbol in self._orderbook_cache:
                    ob = self._orderbook_cache[symbol]
                    best_bid = Decimal(str(ob["bids"][0][0]))
                    best_ask = Decimal(str(ob["asks"][0][0]))
                    mid_price = (best_bid + best_ask) / Decimal("2")
                    spread_bps = ((best_ask - best_bid) / mid_price) * Decimal("10000")
                    
                    volume = metrics.total_bid_volume + metrics.total_ask_volume
                    weighted_spread += spread_bps * volume
                    total_volume += volume
            
            if total_volume > 0:
                vwap_spread = weighted_spread / total_volume
            else:
                # Fallback to simple spread
                best_bid = Decimal(str(orderbook["bids"][0][0]))
                best_ask = Decimal(str(orderbook["asks"][0][0]))
                mid_price = (best_bid + best_ask) / Decimal("2")
                vwap_spread = ((best_ask - best_bid) / mid_price) * Decimal("10000")
            
            return vwap_spread

    async def calculate_liquidity_score(
        self,
        symbol: str,
        orderbook: dict,
        spread_bps: Decimal,
        volatility: Optional[Decimal] = None
    ) -> LiquidityScore:
        """
        Calculate composite liquidity score
        
        Args:
            symbol: Trading pair symbol
            orderbook: Order book data
            spread_bps: Current spread in basis points
            volatility: Current volatility (optional)
            
        Returns:
            Liquidity score (0-100)
        """
        async with self._lock:
            # Assess depth
            depth_metrics = await self.assess_depth(symbol, orderbook)
            
            # Calculate depth score (based on total volume at key levels)
            target_depth = Decimal("10000")  # Target depth in base units
            actual_depth = depth_metrics.bid_depth.get(10, Decimal("0")) + \
                          depth_metrics.ask_depth.get(10, Decimal("0"))
            
            depth_score = min(Decimal("100"), (actual_depth / target_depth) * Decimal("100"))
            
            # Calculate spread score (tighter spread = higher score)
            if spread_bps < Decimal("10"):
                spread_score = Decimal("100")
            elif spread_bps < Decimal("50"):
                spread_score = Decimal("100") - (spread_bps - Decimal("10")) * Decimal("2")
            else:
                spread_score = max(Decimal("0"), Decimal("100") - spread_bps)
            
            # Calculate stability score (based on volatility if provided)
            if volatility is not None:
                if volatility < Decimal("0.01"):
                    stability_score = Decimal("100")
                elif volatility < Decimal("0.05"):
                    stability_score = Decimal("80")
                elif volatility < Decimal("0.1"):
                    stability_score = Decimal("60")
                else:
                    stability_score = max(Decimal("0"), Decimal("100") - volatility * Decimal("500"))
            else:
                # Use depth stability as proxy
                if len(self._depth_history[symbol]) > 1:
                    prev = self._depth_history[symbol][-2]
                    depth_change = abs(depth_metrics.total_bid_volume - prev.total_bid_volume) + \
                                  abs(depth_metrics.total_ask_volume - prev.total_ask_volume)
                    
                    change_pct = depth_change / (depth_metrics.total_bid_volume + 
                                                 depth_metrics.total_ask_volume + Decimal("0.001"))
                    
                    if change_pct < Decimal("0.1"):
                        stability_score = Decimal("100")
                    elif change_pct < Decimal("0.3"):
                        stability_score = Decimal("80")
                    else:
                        stability_score = Decimal("60")
                else:
                    stability_score = Decimal("80")  # Default
            
            # Calculate weighted overall score
            weights = self.config.liquidity_score_weights
            overall_score = (
                depth_score * weights["depth"] +
                spread_score * weights["spread"] +
                stability_score * weights["volatility"]
            )
            
            # Time of day adjustment (simplified - would use actual patterns)
            hour = datetime.now(UTC).hour
            if 8 <= hour <= 16:  # Peak hours
                time_adjustment = Decimal("1.0")
            elif 6 <= hour <= 8 or 16 <= hour <= 20:  # Moderate hours
                time_adjustment = Decimal("0.9")
            else:  # Off-peak
                time_adjustment = Decimal("0.8")
            
            # Calculate final score
            final_score = overall_score * time_adjustment
            
            # Determine grade
            if final_score >= Decimal("90"):
                grade = "A"
            elif final_score >= Decimal("80"):
                grade = "B"
            elif final_score >= Decimal("70"):
                grade = "C"
            elif final_score >= Decimal("60"):
                grade = "D"
            else:
                grade = "F"
            
            score = LiquidityScore(
                symbol=symbol,
                overall_score=overall_score,
                depth_score=depth_score,
                spread_score=spread_score,
                stability_score=stability_score,
                time_of_day_adjustment=time_adjustment,
                final_score=final_score,
                liquidity_grade=grade
            )
            
            # Cache score
            self._liquidity_scores[symbol] = score
            self._score_history[symbol].append(score)
            
            self._logger.debug(
                "Liquidity score calculated",
                symbol=symbol,
                final_score=float(final_score),
                grade=grade,
                depth_score=float(depth_score),
                spread_score=float(spread_score)
            )
            
            return score

    async def save_metrics(self, symbol: str, filepath: str):
        """Save metrics to file for persistence"""
        async with self._lock:
            import json
            
            metrics = {
                "symbol": symbol,
                "timestamp": datetime.now(UTC).isoformat(),
                "depth_history_size": len(self._depth_history[symbol]),
                "imbalance_history_size": len(self._imbalance_history[symbol]),
                "liquidity_score": None,
                "recent_anomalies": []
            }
            
            # Add latest liquidity score
            if symbol in self._liquidity_scores:
                score = self._liquidity_scores[symbol]
                metrics["liquidity_score"] = {
                    "final_score": float(score.final_score),
                    "grade": score.liquidity_grade,
                    "depth_score": float(score.depth_score),
                    "spread_score": float(score.spread_score)
                }
            
            # Add recent anomalies
            if symbol in self._anomaly_scores:
                recent = self._anomaly_scores[symbol][-10:]  # Last 10 anomalies
                metrics["recent_anomalies"] = [
                    {
                        "type": a.anomaly_type,
                        "severity": a.severity,
                        "timestamp": a.timestamp.isoformat()
                    }
                    for a in recent
                ]
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self._logger.info(
                "Metrics saved",
                symbol=symbol,
                filepath=filepath
            )

    async def load_historical(self, symbol: str, filepath: str):
        """Load historical metrics from file"""
        async with self._lock:
            import json
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self._logger.info(
                    "Historical metrics loaded",
                    symbol=symbol,
                    filepath=filepath,
                    data_points=len(data) if isinstance(data, list) else 1
                )
                
                return data
                
            except Exception as e:
                self._logger.error(
                    "Failed to load historical metrics",
                    symbol=symbol,
                    filepath=filepath,
                    error=str(e)
                )
                return None