"""
Market State Classifier Module

Provides real-time market regime detection and classification for risk management.
Classifies market states: DEAD, NORMAL, VOLATILE, PANIC, MAINTENANCE.
"""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import aiohttp
import structlog

from genesis.core.exceptions import DataError
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class VolumeProfile:
    """Historical volume profile for pattern detection."""

    symbol: str
    hourly_averages: dict[int, Decimal]  # hour -> average volume
    daily_averages: dict[int, Decimal]  # day_of_week -> average volume
    rolling_mean: Decimal
    rolling_std: Decimal
    last_updated: datetime


class MarketState(Enum):
    """Market state classifications."""

    DEAD = "DEAD"
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    PANIC = "PANIC"
    MAINTENANCE = "MAINTENANCE"


@dataclass
class Candle:
    """OHLC candle data."""

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_volume: Decimal
    trades: int


@dataclass
class MarketStateContext:
    """Context for market state classification."""

    symbol: str
    current_state: MarketState
    previous_state: Optional[MarketState]
    volatility_atr: Decimal
    realized_volatility: Decimal
    volatility_percentile: int
    volume_24h: Decimal
    volume_mean: Decimal
    volume_std: Decimal
    volume_zscore: Decimal
    spread_basis_points: int
    liquidity_score: Decimal
    correlation_spike: bool
    detected_at: datetime
    state_duration_seconds: int
    reason: str


class MarketStateClassifier:
    """
    Market state classifier for real-time regime detection.

    Analyzes volatility, volume, and market conditions to classify
    the current market state for risk-adjusted trading.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the market state classifier.

        Args:
            event_bus: Optional event bus for publishing state changes
        """
        self.event_bus = event_bus
        self._state_history: dict[str, deque] = {}  # symbol -> deque of states
        self._state_timestamps: dict[str, datetime] = {}  # symbol -> last state change
        self._current_states: dict[str, MarketState] = {}  # symbol -> current state
        self._volatility_history: dict[str, deque] = (
            {}
        )  # symbol -> deque of volatilities
        self._volume_history: dict[str, deque] = {}  # symbol -> deque of volumes
        self._volume_profiles: dict[str, VolumeProfile] = {}  # symbol -> volume profile

        # Hysteresis thresholds to prevent state flapping
        self.hysteresis_factor = Decimal("0.1")  # 10% buffer

        # State classification thresholds
        self.dead_volume_threshold = Decimal("0.2")  # 20% of average
        self.dead_volatility_percentile = 5
        self.normal_volume_std_dev = Decimal("1.0")
        self.normal_volatility_range = (25, 75)  # percentile range
        self.volatile_atr_percentile = 75
        self.volatile_realized_multiplier = Decimal("2.0")
        self.panic_correlation_threshold = Decimal("0.8")
        self.panic_volume_multiplier = Decimal("3.0")
        self.panic_volatility_percentile = 90

        logger.info("MarketStateClassifier initialized")

    async def classify_market_state(self, symbol: str) -> MarketState:
        """
        Classify the current market state for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current market state classification
        """
        try:
            # This will be integrated with market data service
            # For now, return NORMAL as default
            current_state = self._current_states.get(symbol, MarketState.NORMAL)
            logger.debug(f"Market state for {symbol}: {current_state.value}")
            return current_state

        except Exception as e:
            logger.error(f"Error classifying market state for {symbol}: {e}")
            return MarketState.NORMAL

    def calculate_volatility_atr(
        self, candles: list[Candle], period: int = 14
    ) -> Decimal:
        """
        Calculate Average True Range (ATR) volatility.

        Args:
            candles: List of OHLC candles
            period: ATR period (default: 14)

        Returns:
            ATR value as Decimal
        """
        if len(candles) < period + 1:
            raise DataError(
                f"Insufficient candles for ATR calculation: {len(candles)} < {period + 1}"
            )

        true_ranges = []
        for i in range(1, len(candles)):
            high_low = candles[i].high - candles[i].low
            high_close = abs(candles[i].high - candles[i - 1].close)
            low_close = abs(candles[i].low - candles[i - 1].close)
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        # Calculate initial ATR as simple average
        atr = sum(true_ranges[:period]) / Decimal(period)

        # Apply exponential smoothing for remaining values
        for i in range(period, len(true_ranges)):
            atr = (atr * Decimal(period - 1) + true_ranges[i]) / Decimal(period)

        return atr

    def calculate_realized_volatility(
        self, prices: list[Decimal], window: int = 20
    ) -> Decimal:
        """
        Calculate realized volatility using standard deviation of log returns.

        Args:
            prices: List of prices
            window: Rolling window size (default: 20)

        Returns:
            Realized volatility as Decimal
        """
        if len(prices) < window:
            raise DataError(
                f"Insufficient prices for volatility calculation: {len(prices)} < {window}"
            )

        # Calculate log returns
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                log_return = (prices[i] / prices[i - 1]).ln()
                log_returns.append(log_return)

        # Use most recent window
        recent_returns = log_returns[-window:]

        # Calculate mean return
        mean_return = sum(recent_returns) / Decimal(len(recent_returns))

        # Calculate variance
        variance = sum((r - mean_return) ** 2 for r in recent_returns) / Decimal(
            len(recent_returns) - 1
        )

        # Calculate standard deviation (realized volatility)
        volatility = variance.sqrt()

        # Annualize (assuming daily returns, 365 days)
        annualized_volatility = volatility * Decimal(365).sqrt()

        return annualized_volatility

    def _determine_state(
        self,
        symbol: str,
        volatility_atr: Decimal,
        realized_volatility: Decimal,
        volume_24h: Decimal,
        spread_bps: int,
        correlation_spike: bool = False,
        maintenance_detected: bool = False,
    ) -> tuple[MarketState, str]:
        """
        Determine market state based on multiple indicators.

        Args:
            symbol: Trading pair
            volatility_atr: ATR volatility
            realized_volatility: Realized volatility
            volume_24h: 24-hour volume
            spread_bps: Spread in basis points
            correlation_spike: Whether correlation spike detected
            maintenance_detected: Whether maintenance detected

        Returns:
            Tuple of (MarketState, reason for classification)
        """
        # Check for maintenance first (highest priority)
        if maintenance_detected:
            return MarketState.MAINTENANCE, "Exchange maintenance detected"

        # Get historical data for percentile calculations
        vol_history = self._volatility_history.get(symbol, deque(maxlen=30))
        volume_history = self._volume_history.get(symbol, deque(maxlen=20))

        # Calculate percentiles and statistics
        volatility_percentile = self._calculate_percentile(volatility_atr, vol_history)
        volume_mean, volume_std = self._calculate_statistics(volume_history)
        volume_zscore = self._calculate_zscore(volume_24h, volume_mean, volume_std)

        # Check for PANIC state
        if (
            correlation_spike
            and volume_zscore > 3
            and volatility_percentile > self.panic_volatility_percentile
        ):
            return (
                MarketState.PANIC,
                f"Market panic: correlation spike, volume Z-score {volume_zscore:.2f}, volatility percentile {volatility_percentile}",
            )

        # Check for DEAD state
        if (
            volume_24h < volume_mean * self.dead_volume_threshold
            and volatility_percentile < self.dead_volatility_percentile
        ):
            return (
                MarketState.DEAD,
                f"Low activity: volume {volume_24h:.2f} < {volume_mean * self.dead_volume_threshold:.2f}, volatility percentile {volatility_percentile}",
            )

        # Check for VOLATILE state
        if (
            volatility_percentile > self.volatile_atr_percentile
            or realized_volatility > volume_mean * self.volatile_realized_multiplier
        ):
            return (
                MarketState.VOLATILE,
                f"High volatility: ATR percentile {volatility_percentile}, realized vol {realized_volatility:.4f}",
            )

        # Default to NORMAL
        return (
            MarketState.NORMAL,
            f"Normal market conditions: volatility percentile {volatility_percentile}, volume Z-score {volume_zscore:.2f}",
        )

    def _calculate_percentile(self, value: Decimal, history: deque) -> int:
        """Calculate percentile rank of value in history."""
        if not history:
            return 50  # Default to median if no history

        sorted_history = sorted(history)
        count_below = sum(1 for h in sorted_history if h < value)
        percentile = int((count_below / len(sorted_history)) * 100)
        return percentile

    def _calculate_statistics(self, data: deque) -> tuple[Decimal, Decimal]:
        """Calculate mean and standard deviation."""
        if not data:
            return Decimal("0"), Decimal("0")

        mean = sum(data) / Decimal(len(data))

        if len(data) < 2:
            return mean, Decimal("0")

        variance = sum((x - mean) ** 2 for x in data) / Decimal(len(data) - 1)
        std_dev = variance.sqrt()

        return mean, std_dev

    def _calculate_zscore(
        self, value: Decimal, mean: Decimal, std_dev: Decimal
    ) -> Decimal:
        """Calculate Z-score."""
        if std_dev == 0:
            return Decimal("0")
        return (value - mean) / std_dev

    def _apply_hysteresis(
        self, current_state: MarketState, new_state: MarketState, confidence: Decimal
    ) -> bool:
        """
        Apply hysteresis to prevent state flapping.

        Args:
            current_state: Current market state
            new_state: Proposed new state
            confidence: Confidence in state change (0-1)

        Returns:
            Whether to transition to new state
        """
        # Always allow transition to/from MAINTENANCE
        if (
            new_state == MarketState.MAINTENANCE
            or current_state == MarketState.MAINTENANCE
        ):
            return True

        # Always allow transition to PANIC (high priority)
        if new_state == MarketState.PANIC:
            return True

        # Require higher confidence for other transitions
        required_confidence = Decimal("0.5") + self.hysteresis_factor

        # Transitions from PANIC require even higher confidence
        if current_state == MarketState.PANIC:
            required_confidence = Decimal("0.7")

        return confidence >= required_confidence

    async def update_state(
        self,
        symbol: str,
        volatility_atr: Decimal,
        realized_volatility: Decimal,
        volume_24h: Decimal,
        spread_bps: int,
        liquidity_score: Decimal,
        correlation_spike: bool = False,
        maintenance_detected: bool = False,
    ) -> MarketStateContext:
        """
        Update market state with new data.

        Args:
            symbol: Trading pair
            volatility_atr: ATR volatility
            realized_volatility: Realized volatility
            volume_24h: 24-hour volume
            spread_bps: Spread in basis points
            liquidity_score: Liquidity score
            correlation_spike: Whether correlation spike detected
            maintenance_detected: Whether maintenance detected

        Returns:
            Updated market state context
        """
        # Update history
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=30)
            self._volume_history[symbol] = deque(maxlen=20)

        self._volatility_history[symbol].append(volatility_atr)
        self._volume_history[symbol].append(volume_24h)

        # Get current state
        current_state = self._current_states.get(symbol, MarketState.NORMAL)

        # Determine new state
        new_state, reason = self._determine_state(
            symbol,
            volatility_atr,
            realized_volatility,
            volume_24h,
            spread_bps,
            correlation_spike,
            maintenance_detected,
        )

        # Calculate state duration
        state_duration = 0
        if symbol in self._state_timestamps:
            state_duration = int(
                (datetime.now(UTC) - self._state_timestamps[symbol]).total_seconds()
            )

        # Apply hysteresis
        confidence = Decimal("0.6")  # Base confidence
        if new_state != current_state:
            if self._apply_hysteresis(current_state, new_state, confidence):
                # Transition to new state
                self._current_states[symbol] = new_state
                self._state_timestamps[symbol] = datetime.now(UTC)

                # Update history
                if symbol not in self._state_history:
                    self._state_history[symbol] = deque(maxlen=100)
                self._state_history[symbol].append(new_state)

                # Publish event if event bus available
                if self.event_bus:
                    await self._publish_state_change(
                        symbol, current_state, new_state, reason
                    )

                logger.info(
                    f"Market state transition for {symbol}: {current_state.value} -> {new_state.value}, reason: {reason}"
                )
                state_duration = 0  # Reset duration after transition
            else:
                # Keep current state due to hysteresis
                new_state = current_state
                reason = f"Hysteresis: maintaining {current_state.value} state"

        # Calculate statistics
        volume_mean, volume_std = self._calculate_statistics(
            self._volume_history[symbol]
        )
        volume_zscore = self._calculate_zscore(volume_24h, volume_mean, volume_std)
        volatility_percentile = self._calculate_percentile(
            volatility_atr, self._volatility_history[symbol]
        )

        # Create context
        context = MarketStateContext(
            symbol=symbol,
            current_state=new_state,
            previous_state=current_state if new_state != current_state else None,
            volatility_atr=volatility_atr,
            realized_volatility=realized_volatility,
            volatility_percentile=volatility_percentile,
            volume_24h=volume_24h,
            volume_mean=volume_mean,
            volume_std=volume_std,
            volume_zscore=volume_zscore,
            spread_basis_points=spread_bps,
            liquidity_score=liquidity_score,
            correlation_spike=correlation_spike,
            detected_at=datetime.now(UTC),
            state_duration_seconds=state_duration,
            reason=reason,
        )

        return context

    async def _publish_state_change(
        self, symbol: str, old_state: MarketState, new_state: MarketState, reason: str
    ):
        """Publish market state change event."""
        if self.event_bus:
            event_data = {
                "symbol": symbol,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await self.event_bus.publish("MarketStateChangeEvent", event_data)

    def get_state_history(self, symbol: str, limit: int = 100) -> list[MarketState]:
        """
        Get state history for a symbol.

        Args:
            symbol: Trading pair
            limit: Maximum number of states to return

        Returns:
            List of historical states
        """
        if symbol not in self._state_history:
            return []

        history = list(self._state_history[symbol])
        return history[-limit:] if len(history) > limit else history

    def detect_volume_anomaly(
        self,
        current_volume: Decimal,
        historical_volumes: list[Decimal],
        threshold_std: Decimal = Decimal("2.0"),
    ) -> bool:
        """
        Detect volume anomalies using statistical analysis.

        Args:
            current_volume: Current trading volume
            historical_volumes: Historical volume data
            threshold_std: Standard deviation threshold (default: 2.0)

        Returns:
            True if volume is anomalous
        """
        if len(historical_volumes) < 20:
            logger.warning("Insufficient historical data for volume anomaly detection")
            return False

        # Calculate statistics
        mean, std_dev = self._calculate_statistics(deque(historical_volumes))

        if std_dev == 0:
            return False

        # Calculate Z-score
        z_score = self._calculate_zscore(current_volume, mean, std_dev)

        # Detect anomaly
        is_anomaly = abs(z_score) > threshold_std

        if is_anomaly:
            logger.info(
                f"Volume anomaly detected: current={current_volume:.2f}, "
                f"mean={mean:.2f}, std={std_dev:.2f}, z_score={z_score:.2f}"
            )

        return is_anomaly

    def update_volume_profile(self, symbol: str, volume: Decimal, timestamp: datetime):
        """
        Update volume profile with new data.

        Args:
            symbol: Trading pair
            volume: Trading volume
            timestamp: Time of volume data
        """
        if symbol not in self._volume_profiles:
            self._volume_profiles[symbol] = VolumeProfile(
                symbol=symbol,
                hourly_averages={},
                daily_averages={},
                rolling_mean=Decimal("0"),
                rolling_std=Decimal("0"),
                last_updated=timestamp,
            )

        profile = self._volume_profiles[symbol]

        # Update hourly average
        hour = timestamp.hour
        if hour not in profile.hourly_averages:
            profile.hourly_averages[hour] = volume
        else:
            # Exponential moving average
            alpha = Decimal("0.1")
            profile.hourly_averages[hour] = (
                alpha * volume + (Decimal("1") - alpha) * profile.hourly_averages[hour]
            )

        # Update daily average
        day_of_week = timestamp.weekday()
        if day_of_week not in profile.daily_averages:
            profile.daily_averages[day_of_week] = volume
        else:
            # Exponential moving average
            alpha = Decimal("0.1")
            profile.daily_averages[day_of_week] = (
                alpha * volume
                + (Decimal("1") - alpha) * profile.daily_averages[day_of_week]
            )

        # Update rolling statistics
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=20)

        self._volume_history[symbol].append(volume)

        if len(self._volume_history[symbol]) >= 2:
            profile.rolling_mean, profile.rolling_std = self._calculate_statistics(
                self._volume_history[symbol]
            )

        profile.last_updated = timestamp

    def detect_volume_pattern_anomaly(
        self, symbol: str, current_volume: Decimal, timestamp: datetime
    ) -> tuple[bool, str]:
        """
        Detect volume anomalies considering time patterns.

        Args:
            symbol: Trading pair
            current_volume: Current volume
            timestamp: Current timestamp

        Returns:
            Tuple of (is_anomaly, reason)
        """
        if symbol not in self._volume_profiles:
            return False, "No volume profile available"

        profile = self._volume_profiles[symbol]

        # Check against hourly pattern
        hour = timestamp.hour
        if hour in profile.hourly_averages:
            expected_hourly = profile.hourly_averages[hour]
            hourly_deviation = (
                abs(current_volume - expected_hourly) / expected_hourly
                if expected_hourly > 0
                else Decimal("0")
            )

            if hourly_deviation > Decimal("3.0"):  # 300% deviation
                return (
                    True,
                    f"Volume {current_volume:.2f} deviates {hourly_deviation:.1f}x from hourly average {expected_hourly:.2f}",
                )

        # Check against daily pattern
        day_of_week = timestamp.weekday()
        if day_of_week in profile.daily_averages:
            expected_daily = profile.daily_averages[day_of_week]
            daily_deviation = (
                abs(current_volume - expected_daily) / expected_daily
                if expected_daily > 0
                else Decimal("0")
            )

            if daily_deviation > Decimal("2.5"):  # 250% deviation
                return (
                    True,
                    f"Volume {current_volume:.2f} deviates {daily_deviation:.1f}x from daily average {expected_daily:.2f}",
                )

        # Check against rolling statistics
        if profile.rolling_std > 0:
            z_score = (current_volume - profile.rolling_mean) / profile.rolling_std
            if abs(z_score) > Decimal("2.5"):
                return True, f"Volume Z-score {z_score:.2f} exceeds threshold"

        return False, "Volume within normal patterns"

    async def check_maintenance_status(
        self, api_key: Optional[str] = None, base_url: str = "https://api.binance.com"
    ) -> Optional[tuple[bool, dict[str, Any]]]:
        """
        Check Binance system maintenance status.

        Args:
            api_key: Optional API key for authenticated endpoint
            base_url: Binance API base URL

        Returns:
            Tuple of (is_maintenance, status_data)
        """
        try:
            headers = {}
            if api_key:
                headers["X-MBX-APIKEY"] = api_key

            async with aiohttp.ClientSession() as session:
                # Check system status endpoint
                url = f"{base_url}/sapi/v1/system/status"

                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Binance returns status: 0 (normal) or 1 (maintenance)
                        is_maintenance = data.get("status", 0) == 1

                        if is_maintenance:
                            logger.warning(
                                f"Binance system maintenance detected: {data}"
                            )

                        return is_maintenance, data
                    else:
                        logger.warning(
                            f"Failed to check system status: HTTP {response.status}"
                        )
                        return False, None

        except TimeoutError:
            logger.warning("Timeout checking Binance system status")
            return False, None
        except Exception as e:
            logger.error(f"Error checking maintenance status: {e}")
            return False, None

    def detect_maintenance_keywords(self, text: str) -> bool:
        """
        Detect maintenance keywords in text.

        Args:
            text: Text to check for maintenance keywords

        Returns:
            True if maintenance keywords detected
        """
        maintenance_keywords = [
            "maintenance",
            "system upgrade",
            "wallet maintenance",
            "network upgrade",
            "trading suspended",
            "deposits suspended",
            "withdrawals suspended",
            "temporarily unavailable",
            "service disruption",
        ]

        text_lower = text.lower()
        for keyword in maintenance_keywords:
            if keyword in text_lower:
                logger.info(f"Maintenance keyword detected: '{keyword}'")
                return True

        return False

    async def schedule_maintenance_check(
        self, symbol: str, scheduled_time: datetime, buffer_minutes: int = 30
    ) -> bool:
        """
        Check if approaching scheduled maintenance.

        Args:
            symbol: Trading pair
            scheduled_time: Scheduled maintenance time
            buffer_minutes: Buffer time before maintenance (default: 30)

        Returns:
            True if within maintenance buffer window
        """
        now = datetime.now(UTC)
        time_until_maintenance = (scheduled_time - now).total_seconds() / 60

        if 0 <= time_until_maintenance <= buffer_minutes:
            logger.warning(
                f"Approaching scheduled maintenance for {symbol} in "
                f"{time_until_maintenance:.1f} minutes"
            )
            return True

        return False

    async def handle_maintenance_detected(
        self, symbol: str, reason: str = "Maintenance detected"
    ):
        """
        Handle maintenance detection.

        Args:
            symbol: Trading pair
            reason: Reason for maintenance detection
        """
        # Update state to MAINTENANCE
        self._current_states[symbol] = MarketState.MAINTENANCE
        self._state_timestamps[symbol] = datetime.now(UTC)

        # Log maintenance
        logger.warning(f"Maintenance mode activated for {symbol}: {reason}")

        # Publish maintenance event
        if self.event_bus:
            event_data = {
                "symbol": symbol,
                "state": MarketState.MAINTENANCE.value,
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "close_positions",
            }
            await self.event_bus.publish("MaintenanceDetectedEvent", event_data)


class MaintenanceMonitor:
    """
    Monitors for exchange maintenance and system issues.

    Polls system status and detects maintenance windows to
    prevent trading during system downtime.
    """

    def __init__(
        self,
        classifier: MarketStateClassifier,
        poll_interval_seconds: int = 300,  # 5 minutes
        pre_maintenance_buffer_minutes: int = 30,
    ):
        """
        Initialize maintenance monitor.

        Args:
            classifier: Market state classifier instance
            poll_interval_seconds: Polling interval in seconds
            pre_maintenance_buffer_minutes: Buffer before maintenance
        """
        self.classifier = classifier
        self.poll_interval = poll_interval_seconds
        self.buffer_minutes = pre_maintenance_buffer_minutes
        self._monitoring_task: Optional[asyncio.Task] = None
        self._scheduled_maintenances: dict[str, datetime] = {}

        logger.info(
            f"MaintenanceMonitor initialized with poll_interval={poll_interval_seconds}s, "
            f"buffer={pre_maintenance_buffer_minutes}min"
        )

    async def start_monitoring(self, api_key: Optional[str] = None):
        """
        Start maintenance monitoring.

        Args:
            api_key: Optional API key for authenticated endpoints
        """
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Maintenance monitoring already running")
            return

        self._monitoring_task = asyncio.create_task(self._monitor_loop(api_key))
        logger.info("Maintenance monitoring started")

    async def stop_monitoring(self):
        """Stop maintenance monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Maintenance monitoring stopped")

    async def _monitor_loop(self, api_key: Optional[str]):
        """
        Main monitoring loop.

        Args:
            api_key: Optional API key
        """
        while True:
            try:
                # Check system status
                is_maintenance, status_data = (
                    await self.classifier.check_maintenance_status(api_key=api_key)
                )

                if is_maintenance:
                    # Handle maintenance for all symbols
                    await self._handle_system_maintenance(status_data)

                # Check scheduled maintenances
                await self._check_scheduled_maintenances()

                # Wait for next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance monitor loop: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _handle_system_maintenance(self, status_data: Optional[dict[str, Any]]):
        """
        Handle system-wide maintenance.

        Args:
            status_data: System status data from API
        """
        msg = (
            status_data.get("msg", "System maintenance")
            if status_data
            else "System maintenance"
        )

        # Update all tracked symbols to maintenance state
        for symbol in self.classifier._current_states.keys():
            await self.classifier.handle_maintenance_detected(symbol, msg)

    async def _check_scheduled_maintenances(self):
        """Check for approaching scheduled maintenances."""
        now = datetime.now(UTC)

        for symbol, scheduled_time in self._scheduled_maintenances.items():
            if scheduled_time > now:
                is_approaching = await self.classifier.schedule_maintenance_check(
                    symbol, scheduled_time, self.buffer_minutes
                )

                if is_approaching:
                    await self.classifier.handle_maintenance_detected(
                        symbol, f"Scheduled maintenance at {scheduled_time.isoformat()}"
                    )

    def schedule_maintenance(self, symbol: str, scheduled_time: datetime):
        """
        Schedule a maintenance window.

        Args:
            symbol: Trading pair
            scheduled_time: Scheduled maintenance time
        """
        self._scheduled_maintenances[symbol] = scheduled_time
        logger.info(
            f"Scheduled maintenance for {symbol} at {scheduled_time.isoformat()}"
        )

    def cancel_scheduled_maintenance(self, symbol: str):
        """
        Cancel a scheduled maintenance.

        Args:
            symbol: Trading pair
        """
        if symbol in self._scheduled_maintenances:
            del self._scheduled_maintenances[symbol]
            logger.info(f"Cancelled scheduled maintenance for {symbol}")


class GlobalMarketState(Enum):
    """Global market state classifications."""

    BULL = "BULL"
    BEAR = "BEAR"
    CRAB = "CRAB"  # Sideways market
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"


@dataclass
class GlobalMarketContext:
    """Context for global market state."""

    btc_price: Decimal
    total_market_cap: Optional[Decimal]
    fear_greed_index: Optional[int]
    correlation_spike: bool
    state: GlobalMarketState
    vix_crypto: Optional[Decimal]
    detected_at: datetime
    major_pairs_correlation: Decimal
    panic_indicators: int  # Count of panic signals


class GlobalMarketStateClassifier:
    """
    Global market state classifier for overall market conditions.

    Tracks BTC as market leader, correlation spikes, and overall
    market sentiment to classify global crypto market regimes.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize global market state classifier.

        Args:
            event_bus: Optional event bus for publishing events
        """
        self.event_bus = event_bus
        self._current_state = GlobalMarketState.CRAB
        self._btc_price_history: deque = deque(maxlen=100)
        self._correlation_matrix: dict[tuple[str, str], Decimal] = {}
        self._state_history: deque = deque(maxlen=100)
        self._last_state_change = datetime.now(UTC)

        # Thresholds for state classification
        self.crash_threshold = Decimal("-0.15")  # 15% drop
        self.bull_threshold = Decimal("0.10")  # 10% rise
        self.bear_threshold = Decimal("-0.10")  # 10% drop
        self.recovery_threshold = Decimal("0.05")  # 5% rise after crash
        self.correlation_panic_threshold = Decimal("0.8")  # 80% correlation

        logger.info("GlobalMarketStateClassifier initialized")

    async def classify_global_state(
        self,
        btc_price: Decimal,
        major_pairs: list[dict[str, Decimal]],
        fear_greed_index: Optional[int] = None,
    ) -> GlobalMarketState:
        """
        Classify global market state.

        Args:
            btc_price: Current BTC price
            major_pairs: List of major pair prices and volumes
            fear_greed_index: Optional fear & greed index (0-100)

        Returns:
            Global market state classification
        """
        # Update BTC price history
        self._btc_price_history.append(btc_price)

        # Calculate BTC price change
        btc_change = self._calculate_price_change()

        # Calculate correlation among major pairs
        correlation = await self._calculate_correlation(major_pairs)

        # Count panic indicators
        panic_count = 0
        if correlation > self.correlation_panic_threshold:
            panic_count += 1
        if fear_greed_index and fear_greed_index < 20:  # Extreme fear
            panic_count += 1
        if btc_change < self.crash_threshold:
            panic_count += 2

        # Determine state
        new_state = self._determine_global_state(
            btc_change, correlation, panic_count, fear_greed_index
        )

        # Update state if changed
        if new_state != self._current_state:
            await self._handle_state_transition(new_state, btc_price)

        return new_state

    def _calculate_price_change(self) -> Decimal:
        """Calculate BTC price change percentage."""
        if len(self._btc_price_history) < 2:
            return Decimal("0")

        current = self._btc_price_history[-1]
        # Compare to 24h ago (assuming hourly updates)
        past_index = min(24, len(self._btc_price_history) - 1)
        past = self._btc_price_history[-past_index] if past_index > 0 else current

        if past == 0:
            return Decimal("0")

        return (current - past) / past

    async def _calculate_correlation(
        self, major_pairs: list[dict[str, Decimal]]
    ) -> Decimal:
        """
        Calculate average correlation among major pairs.

        Args:
            major_pairs: List of pair data with prices

        Returns:
            Average correlation coefficient
        """
        if len(major_pairs) < 2:
            return Decimal("0")

        # Simplified correlation calculation
        # In production, would use proper statistical methods
        correlations = []

        for i in range(len(major_pairs)):
            for j in range(i + 1, len(major_pairs)):
                # Calculate price movements
                pair1_change = major_pairs[i].get("change_percent", Decimal("0"))
                pair2_change = major_pairs[j].get("change_percent", Decimal("0"))

                # Simple correlation proxy based on direction
                if pair1_change * pair2_change > 0:  # Same direction
                    correlation = min(abs(pair1_change), abs(pair2_change)) / Decimal(
                        "100"
                    )
                    correlations.append(correlation)

        if correlations:
            avg_correlation = sum(correlations) / len(correlations)
            return avg_correlation

        return Decimal("0")

    def _determine_global_state(
        self,
        btc_change: Decimal,
        correlation: Decimal,
        panic_count: int,
        fear_greed_index: Optional[int],
    ) -> GlobalMarketState:
        """
        Determine global market state based on indicators.

        Args:
            btc_change: BTC price change percentage
            correlation: Average correlation among pairs
            panic_count: Number of panic indicators
            fear_greed_index: Fear & greed index

        Returns:
            Global market state
        """
        # CRASH: Multiple panic indicators
        if panic_count >= 2 or btc_change < self.crash_threshold:
            return GlobalMarketState.CRASH

        # RECOVERY: Rising from recent crash
        if (
            self._current_state == GlobalMarketState.CRASH
            and btc_change > self.recovery_threshold
        ):
            return GlobalMarketState.RECOVERY

        # BULL: Strong upward movement
        if btc_change > self.bull_threshold:
            if (
                fear_greed_index and fear_greed_index > 70
            ) or btc_change > self.bull_threshold * Decimal(
                "1.5"
            ):  # Greed
                return GlobalMarketState.BULL

        # BEAR: Sustained downward movement
        if btc_change < self.bear_threshold:
            if (
                fear_greed_index and fear_greed_index < 30
            ) or btc_change < self.bear_threshold * Decimal(
                "1.5"
            ):  # Fear
                return GlobalMarketState.BEAR

        # CRAB: Sideways/neutral market
        return GlobalMarketState.CRAB

    async def _handle_state_transition(
        self, new_state: GlobalMarketState, btc_price: Decimal
    ):
        """Handle global state transition."""
        old_state = self._current_state
        self._current_state = new_state
        self._state_history.append(new_state)
        self._last_state_change = datetime.now(UTC)

        logger.warning(
            f"GLOBAL MARKET STATE CHANGE: {old_state.value} -> {new_state.value}, "
            f"BTC: ${btc_price:.2f}"
        )

        # Publish event
        if self.event_bus:
            await self.event_bus.publish(
                "GlobalMarketStateChangeEvent",
                {
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "btc_price": str(btc_price),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

    def get_current_state(self) -> GlobalMarketState:
        """Get current global market state."""
        return self._current_state

    def get_state_duration(self) -> timedelta:
        """Get duration in current state."""
        return datetime.now(UTC) - self._last_state_change


class StateTransitionManager:
    """
    Manages state transitions and associated actions.

    Handles position sizing adjustments and strategy activation
    based on market state changes.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the state transition manager.

        Args:
            event_bus: Optional event bus for publishing events
        """
        self.event_bus = event_bus
        self._transition_history: list[dict[str, Any]] = []

        # Position size multipliers by state
        self.position_multipliers = {
            MarketState.DEAD: Decimal("0.5"),
            MarketState.NORMAL: Decimal("1.0"),
            MarketState.VOLATILE: Decimal("0.75"),
            MarketState.PANIC: Decimal("0.25"),
            MarketState.MAINTENANCE: Decimal("0"),
        }

        logger.info("StateTransitionManager initialized")

    async def transition_to_state(
        self,
        symbol: str,
        new_state: MarketState,
        reason: str,
        context: Optional[MarketStateContext] = None,
    ):
        """
        Execute transition to new market state.

        Args:
            symbol: Trading pair
            new_state: New market state
            reason: Reason for transition
            context: Optional market state context
        """
        transition_data = {
            "symbol": symbol,
            "state": new_state.value,
            "reason": reason,
            "timestamp": datetime.now(UTC),
            "context": context,
        }

        self._transition_history.append(transition_data)

        # Apply position sizing adjustment
        multiplier = self.position_multipliers[new_state]
        await self._adjust_position_sizing(symbol, multiplier, reason)

        # Apply strategy adjustments
        await self._adjust_strategies(symbol, new_state)

        logger.info(
            f"State transition completed for {symbol}: {new_state.value}, multiplier: {multiplier}"
        )

    async def _adjust_position_sizing(
        self, symbol: str, multiplier: Decimal, reason: str
    ):
        """Adjust position sizing based on state."""
        if self.event_bus:
            event_data = {
                "symbol": symbol,
                "multiplier": str(multiplier),
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await self.event_bus.publish("PositionSizeAdjustmentEvent", event_data)

    async def _adjust_strategies(self, symbol: str, state: MarketState):
        """Adjust strategy activation based on state."""
        # Strategy adjustment logic will be implemented with strategy manager
        # For now, log the intended adjustments
        adjustments = {
            MarketState.DEAD: ["disable_grid_trading", "reduce_frequency"],
            MarketState.NORMAL: ["enable_all_strategies"],
            MarketState.VOLATILE: ["disable_mean_reversion", "enable_momentum"],
            MarketState.PANIC: ["disable_all_except_safety"],
            MarketState.MAINTENANCE: ["disable_all_strategies", "close_positions"],
        }

        actions = adjustments.get(state, [])
        logger.info(f"Strategy adjustments for {symbol} in {state.value}: {actions}")

    def get_transition_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Get transition history.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of transitions

        Returns:
            List of transition records
        """
        history = self._transition_history

        if symbol:
            history = [t for t in history if t["symbol"] == symbol]

        return history[-limit:] if len(history) > limit else history


class PositionSizeAdjuster:
    """
    Adjusts position sizes based on market state.

    Implements dynamic position sizing based on current market
    conditions to manage risk appropriately.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize position size adjuster.

        Args:
            event_bus: Optional event bus for publishing events
        """
        self.event_bus = event_bus

        # Position size multipliers by market state
        self.state_multipliers = {
            MarketState.DEAD: Decimal("0.5"),  # 50% reduction
            MarketState.NORMAL: Decimal("1.0"),  # Normal sizing
            MarketState.VOLATILE: Decimal("0.75"),  # 25% reduction
            MarketState.PANIC: Decimal("0.25"),  # 75% reduction
            MarketState.MAINTENANCE: Decimal("0"),  # No new positions
        }

        # Global state multipliers
        self.global_state_multipliers = {
            GlobalMarketState.BULL: Decimal("1.1"),  # 10% increase
            GlobalMarketState.BEAR: Decimal("0.8"),  # 20% reduction
            GlobalMarketState.CRAB: Decimal("1.0"),  # Normal
            GlobalMarketState.CRASH: Decimal("0.3"),  # 70% reduction
            GlobalMarketState.RECOVERY: Decimal("0.7"),  # 30% reduction
        }

        # Current adjustments
        self._current_adjustments: dict[str, Decimal] = {}
        self._adjustment_history: list[dict[str, Any]] = []

        logger.info("PositionSizeAdjuster initialized")

    async def calculate_position_size(
        self,
        symbol: str,
        base_size: Decimal,
        market_state: MarketState,
        global_state: Optional[GlobalMarketState] = None,
        volatility_percentile: Optional[int] = None,
    ) -> tuple[Decimal, str]:
        """
        Calculate adjusted position size.

        Args:
            symbol: Trading pair
            base_size: Base position size
            market_state: Current market state
            global_state: Optional global market state
            volatility_percentile: Optional volatility percentile

        Returns:
            Tuple of (adjusted_size, adjustment_reason)
        """
        # Start with base size
        adjusted_size = base_size
        reasons = []

        # Apply market state multiplier
        state_mult = self.state_multipliers[market_state]
        adjusted_size *= state_mult
        if state_mult != Decimal("1.0"):
            reasons.append(f"{market_state.value}: {state_mult}x")

        # Apply global state multiplier if available
        if global_state:
            global_mult = self.global_state_multipliers[global_state]
            adjusted_size *= global_mult
            if global_mult != Decimal("1.0"):
                reasons.append(f"Global {global_state.value}: {global_mult}x")

        # Apply volatility-based adjustment
        if volatility_percentile is not None:
            vol_mult = self._calculate_volatility_multiplier(volatility_percentile)
            adjusted_size *= vol_mult
            if vol_mult != Decimal("1.0"):
                reasons.append(f"Volatility P{volatility_percentile}: {vol_mult}x")

        # Ensure minimum position size
        min_size = base_size * Decimal("0.1")  # Minimum 10% of base
        if adjusted_size < min_size and market_state != MarketState.MAINTENANCE:
            adjusted_size = min_size
            reasons.append("Minimum size enforced")

        # Round to reasonable precision
        adjusted_size = adjusted_size.quantize(Decimal("0.01"))

        # Create adjustment reason
        reason = " | ".join(reasons) if reasons else "No adjustment"

        # Record adjustment
        await self._record_adjustment(symbol, base_size, adjusted_size, reason)

        return adjusted_size, reason

    def _calculate_volatility_multiplier(self, percentile: int) -> Decimal:
        """
        Calculate position size multiplier based on volatility percentile.

        Args:
            percentile: Volatility percentile (0-100)

        Returns:
            Position size multiplier
        """
        if percentile >= 90:
            return Decimal("0.5")  # Extreme volatility: 50% reduction
        elif percentile >= 75:
            return Decimal("0.75")  # High volatility: 25% reduction
        elif percentile >= 25:
            return Decimal("1.0")  # Normal volatility: no change
        else:
            return Decimal("1.1")  # Low volatility: 10% increase

    async def _record_adjustment(
        self, symbol: str, base_size: Decimal, adjusted_size: Decimal, reason: str
    ):
        """Record position size adjustment."""
        adjustment = {
            "symbol": symbol,
            "base_size": base_size,
            "adjusted_size": adjusted_size,
            "multiplier": adjusted_size / base_size if base_size > 0 else Decimal("0"),
            "reason": reason,
            "timestamp": datetime.now(UTC),
        }

        self._adjustment_history.append(adjustment)
        self._current_adjustments[symbol] = (
            adjusted_size / base_size if base_size > 0 else Decimal("0")
        )

        # Publish event
        if self.event_bus:
            await self.event_bus.publish(
                "PositionSizeAdjustedEvent",
                {
                    "symbol": symbol,
                    "multiplier": str(adjustment["multiplier"]),
                    "reason": reason,
                    "timestamp": adjustment["timestamp"].isoformat(),
                },
            )

        logger.info(
            f"Position size adjusted for {symbol}: "
            f"{base_size:.2f} -> {adjusted_size:.2f} ({reason})"
        )

    def get_current_adjustment(self, symbol: str) -> Decimal:
        """Get current adjustment multiplier for a symbol."""
        return self._current_adjustments.get(symbol, Decimal("1.0"))

    def get_adjustment_history(
        self, symbol: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get adjustment history."""
        history = self._adjustment_history

        if symbol:
            history = [a for a in history if a["symbol"] == symbol]

        return history[-limit:] if len(history) > limit else history


class StrategyStateManager:
    """
    Manages strategy activation based on market states.

    Enables or disables trading strategies based on current
    market conditions to optimize performance and risk.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize strategy state manager.

        Args:
            event_bus: Optional event bus for publishing events
        """
        self.event_bus = event_bus

        # Strategy compatibility matrix
        # True = strategy enabled, False = strategy disabled
        self.strategy_matrix = {
            MarketState.DEAD: {
                "arbitrage": False,
                "grid_trading": False,
                "mean_reversion": False,
                "momentum": False,
                "market_making": False,
            },
            MarketState.NORMAL: {
                "arbitrage": True,
                "grid_trading": True,
                "mean_reversion": True,
                "momentum": True,
                "market_making": True,
            },
            MarketState.VOLATILE: {
                "arbitrage": True,
                "grid_trading": False,  # Disable in volatile markets
                "mean_reversion": False,  # Disable in trending markets
                "momentum": True,  # Enable momentum strategies
                "market_making": False,  # Too risky
            },
            MarketState.PANIC: {
                "arbitrage": True,  # May still work
                "grid_trading": False,
                "mean_reversion": False,
                "momentum": False,
                "market_making": False,
            },
            MarketState.MAINTENANCE: {
                "arbitrage": False,
                "grid_trading": False,
                "mean_reversion": False,
                "momentum": False,
                "market_making": False,
            },
        }

        # Current strategy states
        self._active_strategies: dict[str, bool] = {}
        self._strategy_performance: dict[str, dict[str, Any]] = {}

        logger.info("StrategyStateManager initialized")

    async def update_strategy_states(
        self, market_state: MarketState, symbol: Optional[str] = None
    ) -> dict[str, bool]:
        """
        Update strategy activation states based on market state.

        Args:
            market_state: Current market state
            symbol: Optional symbol for symbol-specific strategies

        Returns:
            Dict of strategy states
        """
        # Get strategy states for market condition
        new_states = self.strategy_matrix[market_state].copy()

        # Check for changes
        changes = []
        for strategy, enabled in new_states.items():
            if self._active_strategies.get(strategy) != enabled:
                old_state = self._active_strategies.get(strategy, False)
                changes.append(
                    {
                        "strategy": strategy,
                        "old_state": old_state,
                        "new_state": enabled,
                        "reason": f"Market state: {market_state.value}",
                    }
                )

        # Update active strategies
        self._active_strategies = new_states

        # Publish changes
        if changes and self.event_bus:
            for change in changes:
                await self.event_bus.publish(
                    "StrategyStateChangeEvent",
                    {
                        "strategy": change["strategy"],
                        "enabled": change["new_state"],
                        "reason": change["reason"],
                        "symbol": symbol,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                action = "Enabled" if change["new_state"] else "Disabled"
                logger.info(
                    f"{action} {change['strategy']} strategy: {change['reason']}"
                )

        return self._active_strategies

    def is_strategy_enabled(self, strategy: str) -> bool:
        """Check if a strategy is currently enabled."""
        return self._active_strategies.get(strategy, False)

    def get_enabled_strategies(self) -> list[str]:
        """Get list of currently enabled strategies."""
        return [s for s, enabled in self._active_strategies.items() if enabled]

    async def emergency_disable_all(self, reason: str = "Emergency stop"):
        """Emergency disable all strategies."""
        for strategy in self._active_strategies:
            self._active_strategies[strategy] = False

        if self.event_bus:
            await self.event_bus.publish(
                "EmergencyStrategyStopEvent",
                {"reason": reason, "timestamp": datetime.now(UTC).isoformat()},
            )

        logger.warning(f"EMERGENCY: All strategies disabled - {reason}")
