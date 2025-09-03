"""
Enhanced Spread Tracking System

Implements comprehensive spread tracking with historical baselines, volatility measurement,
and weighted spread calculations for market condition assessment.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import structlog

from genesis.core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class SpreadTrackerConfig:
    """Configuration for spread tracker"""

    spread_window: int = 1000  # Rolling window size
    baseline_period: int = 3600  # Seconds for baseline calculation
    volatility_halflife: int = 300  # Seconds for EWMA volatility
    anomaly_threshold: Decimal = Decimal("3.0")  # Z-score for anomalies
    cache_ttl: int = 60  # Seconds for metric cache
    max_memory_mb: int = 50  # Memory limit
    ema_alpha: Decimal = Decimal("0.1")  # EMA smoothing factor


@dataclass
class SpreadMetricsEnhanced:
    """Enhanced spread metrics with additional calculations"""

    symbol: str
    current_spread_bps: Decimal
    bid_price: Decimal
    ask_price: Decimal
    bid_volume: Decimal
    ask_volume: Decimal

    # Baseline metrics
    ema_spread: Decimal
    percentile_25: Decimal
    percentile_50: Decimal  # Median
    percentile_75: Decimal

    # Volatility metrics
    std_deviation: Decimal
    ewma_volatility: Decimal
    realized_volatility: Decimal
    volatility_regime: str  # 'low', 'normal', 'high', 'extreme'

    # Weighted spreads
    vwap_spread: Decimal
    twap_spread: Decimal
    effective_spread: Decimal

    # Additional metrics
    spread_change_rate: Decimal
    is_anomaly: bool
    anomaly_score: Decimal

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EnhancedSpreadTracker:
    """
    Enhanced spread tracker with comprehensive metrics and analysis
    """

    def __init__(self, config: SpreadTrackerConfig | None = None):
        """
        Initialize enhanced spread tracker

        Args:
            config: Configuration parameters
        """
        self.config = config or SpreadTrackerConfig()

        # Thread-safe collections using asyncio locks
        self._lock = asyncio.Lock()

        # Data structures for spread history
        self._spread_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.spread_window)
        )
        self._timestamp_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.spread_window)
        )
        self._volume_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.spread_window)
        )

        # Baseline tracking
        self._ema_spreads: dict[str, Decimal] = {}
        self._hourly_baselines: dict[str, dict[int, list[Decimal]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._daily_baselines: dict[str, dict[int, list[Decimal]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Volatility tracking
        self._ewma_volatility: dict[str, Decimal] = {}
        self._volatility_regimes: dict[str, str] = {}

        # Weighted spread tracking
        self._vwap_spreads: dict[str, Decimal] = {}
        self._twap_spreads: dict[str, Decimal] = {}

        # Cache for recent metrics
        self._metrics_cache: dict[str, SpreadMetricsEnhanced] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        self._logger = logger.bind(component="EnhancedSpreadTracker")
        self._memory_usage = 0
        self._last_cleanup = datetime.now(UTC)

    async def update_spread(
        self,
        symbol: str,
        bid_price: Decimal,
        ask_price: Decimal,
        bid_volume: Decimal,
        ask_volume: Decimal
    ) -> SpreadMetricsEnhanced:
        """
        Update spread tracking with new market data

        Args:
            symbol: Trading pair symbol
            bid_price: Best bid price
            ask_price: Best ask price
            bid_volume: Bid volume at best price
            ask_volume: Ask volume at best price

        Returns:
            Enhanced spread metrics
        """
        async with self._lock:
            # Validate inputs
            if bid_price <= 0 or ask_price <= 0:
                raise ValidationError(f"Invalid prices: bid={bid_price}, ask={ask_price}")

            if ask_price <= bid_price:
                raise ValidationError(f"Ask must be > bid: bid={bid_price}, ask={ask_price}")

            # Calculate spread in basis points
            mid_price = (ask_price + bid_price) / Decimal("2")
            spread_bps = ((ask_price - bid_price) / mid_price) * Decimal("10000")

            # Store in rolling windows
            now = datetime.now(UTC)
            self._spread_history[symbol].append(spread_bps)
            self._timestamp_history[symbol].append(now)
            self._volume_history[symbol].append((bid_volume, ask_volume))

            # Update spread change detection
            spread_change_rate = self._calculate_spread_change_rate(symbol)

            # Calculate baseline metrics
            ema_spread = self._update_ema_spread(symbol, spread_bps)
            percentiles = self._calculate_percentiles(symbol)

            # Update baseline storage
            hour = now.hour
            day = now.weekday()
            self._hourly_baselines[symbol][hour].append(spread_bps)
            self._daily_baselines[symbol][day].append(spread_bps)

            # Calculate volatility metrics
            std_dev = self._calculate_std_deviation(symbol)
            ewma_vol = self._update_ewma_volatility(symbol, spread_bps)
            realized_vol = self._calculate_realized_volatility(symbol)
            vol_regime = self._detect_volatility_regime(symbol, ewma_vol)

            # Calculate weighted spreads
            vwap_spread = self._calculate_vwap_spread(symbol)
            twap_spread = self._calculate_twap_spread(symbol)
            effective_spread = self._calculate_effective_spread(
                bid_price, ask_price, mid_price
            )

            # Detect anomalies
            is_anomaly, anomaly_score = self._detect_anomaly(symbol, spread_bps)

            # Create metrics object
            metrics = SpreadMetricsEnhanced(
                symbol=symbol,
                current_spread_bps=spread_bps,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                ema_spread=ema_spread,
                percentile_25=percentiles[0],
                percentile_50=percentiles[1],
                percentile_75=percentiles[2],
                std_deviation=std_dev,
                ewma_volatility=ewma_vol,
                realized_volatility=realized_vol,
                volatility_regime=vol_regime,
                vwap_spread=vwap_spread,
                twap_spread=twap_spread,
                effective_spread=effective_spread,
                spread_change_rate=spread_change_rate,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score
            )

            # Cache metrics
            self._metrics_cache[symbol] = metrics
            self._cache_timestamps[symbol] = now

            # Check memory usage periodically
            if (now - self._last_cleanup).total_seconds() > 300:
                await self._cleanup_old_data()

            self._logger.debug(
                "Spread updated",
                symbol=symbol,
                spread_bps=float(spread_bps),
                volatility_regime=vol_regime,
                is_anomaly=is_anomaly
            )

            return metrics

    def _calculate_spread_change_rate(self, symbol: str) -> Decimal:
        """Calculate rate of spread change"""
        spreads = list(self._spread_history[symbol])
        if len(spreads) < 2:
            return Decimal("0")

        # Calculate percentage change from previous spread
        current = spreads[-1]
        previous = spreads[-2]

        if previous == 0:
            return Decimal("0")

        return ((current - previous) / previous) * Decimal("100")

    def _update_ema_spread(self, symbol: str, current_spread: Decimal) -> Decimal:
        """Update exponential moving average of spread"""
        alpha = self.config.ema_alpha

        if symbol not in self._ema_spreads:
            self._ema_spreads[symbol] = current_spread
        else:
            prev_ema = self._ema_spreads[symbol]
            self._ema_spreads[symbol] = alpha * current_spread + (1 - alpha) * prev_ema

        return self._ema_spreads[symbol]

    def _calculate_percentiles(self, symbol: str) -> tuple[Decimal, Decimal, Decimal]:
        """Calculate percentile-based baselines"""
        spreads = list(self._spread_history[symbol])

        if not spreads:
            return (Decimal("0"), Decimal("0"), Decimal("0"))

        # Convert to numpy for percentile calculation
        spreads_array = np.array([float(s) for s in spreads])

        p25 = Decimal(str(np.percentile(spreads_array, 25)))
        p50 = Decimal(str(np.percentile(spreads_array, 50)))  # Median
        p75 = Decimal(str(np.percentile(spreads_array, 75)))

        return (p25, p50, p75)

    def _calculate_std_deviation(self, symbol: str) -> Decimal:
        """Calculate standard deviation of spreads"""
        spreads = list(self._spread_history[symbol])

        if len(spreads) < 2:
            return Decimal("0")

        mean = sum(spreads) / len(spreads)
        variance = sum((x - mean) ** 2 for x in spreads) / len(spreads)

        return variance.sqrt()

    def _update_ewma_volatility(self, symbol: str, current_spread: Decimal) -> Decimal:  # noqa: ARG002
        """Update exponentially weighted moving average volatility"""
        spreads = list(self._spread_history[symbol])

        if len(spreads) < 2:
            return Decimal("0")

        # Calculate returns
        returns = []
        for i in range(1, min(20, len(spreads))):
            if spreads[i-1] != 0:
                ret = (spreads[i] - spreads[i-1]) / spreads[i-1]
                returns.append(ret)

        if not returns:
            return Decimal("0")

        # Calculate EWMA volatility
        lambda_param = Decimal("0.94")  # Standard EWMA decay factor

        weights = []
        for i in range(len(returns)):
            weight = (1 - lambda_param) * lambda_param ** i
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Calculate weighted variance
        mean_return = sum(r * w for r, w in zip(returns, weights, strict=False))
        variance = sum(w * (r - mean_return) ** 2 for r, w in zip(returns, weights, strict=False))

        ewma_vol = variance.sqrt() if variance > 0 else Decimal("0")
        self._ewma_volatility[symbol] = ewma_vol

        return ewma_vol

    def _calculate_realized_volatility(self, symbol: str) -> Decimal:
        """Calculate realized volatility"""
        spreads = list(self._spread_history[symbol])
        timestamps = list(self._timestamp_history[symbol])

        if len(spreads) < 2:
            return Decimal("0")

        # Calculate time-weighted returns
        returns = []
        for i in range(1, len(spreads)):
            if spreads[i-1] != 0:
                ret = (spreads[i] - spreads[i-1]) / spreads[i-1]
                time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()

                # Annualize based on time difference
                if time_diff > 0:
                    annualized_ret = ret * Decimal(str(np.sqrt(365 * 24 * 3600 / time_diff)))
                    returns.append(annualized_ret)

        if not returns:
            return Decimal("0")

        # Calculate standard deviation of returns
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)

        return variance.sqrt()

    def _detect_volatility_regime(self, symbol: str, current_vol: Decimal) -> str:
        """Detect current volatility regime"""
        # Define thresholds based on historical volatility
        spreads = list(self._spread_history[symbol])

        if len(spreads) < 20:
            return "normal"

        # Calculate historical volatility percentiles
        vols = []
        for i in range(20, len(spreads)):
            window = spreads[i-20:i]
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            vols.append(var.sqrt())

        if not vols:
            return "normal"

        vols_array = np.array([float(v) for v in vols])
        p25 = Decimal(str(np.percentile(vols_array, 25)))
        p50 = Decimal(str(np.percentile(vols_array, 50)))
        p75 = Decimal(str(np.percentile(vols_array, 75)))
        # p90 = Decimal(str(np.percentile(vols_array, 90)))  # Not currently used

        # Classify regime
        if current_vol < p25:
            regime = "low"
        elif current_vol < p50:
            regime = "normal"
        elif current_vol < p75:
            regime = "high"
        else:
            regime = "extreme"

        # Update regime tracking
        self._volatility_regimes[symbol] = regime

        # Log regime changes
        if symbol in self._volatility_regimes:
            prev_regime = self._volatility_regimes.get(symbol)
            if prev_regime != regime:
                self._logger.info(
                    "Volatility regime change",
                    symbol=symbol,
                    from_regime=prev_regime,
                    to_regime=regime,
                    current_vol=float(current_vol)
                )

        return regime

    def _calculate_vwap_spread(self, symbol: str) -> Decimal:
        """Calculate volume-weighted average spread"""
        spreads = list(self._spread_history[symbol])
        volumes = list(self._volume_history[symbol])

        if not spreads or not volumes:
            return Decimal("0")

        # Use last N periods for VWAP
        n = min(20, len(spreads))
        recent_spreads = spreads[-n:]
        recent_volumes = volumes[-n:]

        # Calculate total volume for each spread
        total_volumes = [bid_vol + ask_vol for bid_vol, ask_vol in recent_volumes]

        # Calculate VWAP
        if sum(total_volumes) > 0:
            vwap = sum(s * v for s, v in zip(recent_spreads, total_volumes, strict=False)) / sum(total_volumes)
        else:
            vwap = sum(recent_spreads) / len(recent_spreads)

        self._vwap_spreads[symbol] = vwap
        return vwap

    def _calculate_twap_spread(self, symbol: str) -> Decimal:
        """Calculate time-weighted average spread"""
        spreads = list(self._spread_history[symbol])
        timestamps = list(self._timestamp_history[symbol])

        if len(spreads) < 2:
            return spreads[0] if spreads else Decimal("0")

        # Calculate time-weighted average
        total_time = Decimal("0")
        weighted_sum = Decimal("0")

        for i in range(1, len(spreads)):
            time_diff = Decimal(str((timestamps[i] - timestamps[i-1]).total_seconds()))
            weighted_sum += spreads[i-1] * time_diff
            total_time += time_diff

        if total_time > 0:
            twap = weighted_sum / total_time
        else:
            twap = sum(spreads) / len(spreads)

        self._twap_spreads[symbol] = twap
        return twap

    def _calculate_effective_spread(
        self,
        bid_price: Decimal,
        ask_price: Decimal,
        mid_price: Decimal
    ) -> Decimal:
        """
        Calculate effective spread (includes market impact)

        Effective spread = 2 * |execution_price - mid_price| / mid_price * 10000
        For this calculation, we estimate execution price based on typical fills
        """
        # Estimate execution price (simplified - in reality would use actual trades)
        # Assume buyer pays slightly above mid, seller receives slightly below
        typical_slippage = (ask_price - bid_price) * Decimal("0.1")  # 10% of spread as slippage

        # For a buy order
        buy_execution = mid_price + typical_slippage
        # For a sell order
        sell_execution = mid_price - typical_slippage

        # Average effective spread
        buy_effective = 2 * abs(buy_execution - mid_price) / mid_price * Decimal("10000")
        sell_effective = 2 * abs(sell_execution - mid_price) / mid_price * Decimal("10000")

        return (buy_effective + sell_effective) / Decimal("2")

    def _detect_anomaly(self, symbol: str, current_spread: Decimal) -> tuple[bool, Decimal]:
        """Detect if current spread is anomalous"""
        spreads = list(self._spread_history[symbol])

        if len(spreads) < 20:
            return False, Decimal("0")

        # Calculate z-score
        mean = sum(spreads[:-1]) / (len(spreads) - 1)
        variance = sum((x - mean) ** 2 for x in spreads[:-1]) / (len(spreads) - 1)
        std_dev = variance.sqrt()

        # Handle zero variance case - when all historical spreads are identical
        if std_dev == 0:
            # If current spread differs significantly from the constant baseline,
            # consider it anomalous based on percentage change
            if mean > 0:
                pct_change = abs((current_spread - mean) / mean)
                # Consider >300% change as anomalous when baseline is constant
                if pct_change > Decimal("3"):
                    self._logger.warning(
                        "Spread anomaly detected (zero variance baseline)",
                        symbol=symbol,
                        current_spread=float(current_spread),
                        mean_spread=float(mean),
                        pct_change=float(pct_change * 100)
                    )
                    # Return high z-score equivalent for anomaly
                    return True, Decimal("10")
            return False, Decimal("0")

        z_score = abs((current_spread - mean) / std_dev)

        is_anomaly = z_score > self.config.anomaly_threshold

        if is_anomaly:
            self._logger.warning(
                "Spread anomaly detected",
                symbol=symbol,
                current_spread=float(current_spread),
                mean_spread=float(mean),
                z_score=float(z_score)
            )

        return is_anomaly, z_score

    async def calculate_baseline(
        self,
        symbol: str,
        period_seconds: int | None = None
    ) -> dict[str, Decimal]:
        """
        Calculate historical baseline for a symbol

        Args:
            symbol: Trading pair symbol
            period_seconds: Period for baseline calculation

        Returns:
            Dictionary of baseline metrics
        """
        async with self._lock:
            period = period_seconds or self.config.baseline_period

            spreads = list(self._spread_history[symbol])
            timestamps = list(self._timestamp_history[symbol])

            if not spreads:
                return {}

            # Filter by time period
            cutoff_time = datetime.now(UTC) - timedelta(seconds=period)
            filtered_data = [
                (s, t) for s, t in zip(spreads, timestamps, strict=False) if t >= cutoff_time
            ]

            if not filtered_data:
                return {}

            filtered_spreads = [s for s, _ in filtered_data]

            # Calculate baseline metrics
            baseline = {
                "ema": self._ema_spreads.get(symbol, Decimal("0")),
                "mean": sum(filtered_spreads) / len(filtered_spreads),
                "median": Decimal(str(np.median([float(s) for s in filtered_spreads]))),
                "std_dev": self._calculate_std_deviation(symbol),
                "min": min(filtered_spreads),
                "max": max(filtered_spreads),
                "percentile_10": Decimal(str(np.percentile([float(s) for s in filtered_spreads], 10))),
                "percentile_90": Decimal(str(np.percentile([float(s) for s in filtered_spreads], 90))),
            }

            self._logger.debug(
                "Baseline calculated",
                symbol=symbol,
                period_seconds=period,
                mean=float(baseline["mean"]),
                median=float(baseline["median"])
            )

            return baseline

    async def get_cached_metrics(self, symbol: str) -> SpreadMetricsEnhanced | None:
        """Get cached metrics if still valid"""
        async with self._lock:
            if symbol not in self._metrics_cache:
                return None

            # Check cache TTL
            cache_time = self._cache_timestamps.get(symbol)
            if cache_time:
                age = (datetime.now(UTC) - cache_time).total_seconds()
                if age > self.config.cache_ttl:
                    return None

            return self._metrics_cache.get(symbol)

    async def _cleanup_old_data(self):
        """Clean up old data to manage memory usage"""
        now = datetime.now(UTC)

        # Clean hourly baselines older than 24 hours
        for symbol in list(self._hourly_baselines.keys()):
            for hour in list(self._hourly_baselines[symbol].keys()):
                # Keep only last 100 values per hour
                if len(self._hourly_baselines[symbol][hour]) > 100:
                    self._hourly_baselines[symbol][hour] = (
                        self._hourly_baselines[symbol][hour][-100:]
                    )

        # Clean daily baselines older than 7 days
        for symbol in list(self._daily_baselines.keys()):
            for day in list(self._daily_baselines[symbol].keys()):
                # Keep only last 500 values per day
                if len(self._daily_baselines[symbol][day]) > 500:
                    self._daily_baselines[symbol][day] = (
                        self._daily_baselines[symbol][day][-500:]
                    )

        # Estimate memory usage (simplified)
        estimated_mb = (
            len(self._spread_history) * self.config.spread_window * 32  # Decimal size
            + len(self._hourly_baselines) * 24 * 100 * 32
            + len(self._daily_baselines) * 7 * 500 * 32
        ) / (1024 * 1024)

        self._memory_usage = estimated_mb

        if estimated_mb > self.config.max_memory_mb:
            self._logger.warning(
                "Memory limit exceeded",
                usage_mb=estimated_mb,
                limit_mb=self.config.max_memory_mb
            )

        self._last_cleanup = now

        self._logger.debug(
            "Data cleanup completed",
            memory_usage_mb=estimated_mb,
            symbols_tracked=len(self._spread_history)
        )

