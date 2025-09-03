"""Market Regime Detection Module.

This module provides functionality to detect market regimes (trending vs ranging)
and volatility classifications for trading strategies.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

from genesis.analytics.technical_indicators import (
    calculate_adx,
)

logger = structlog.get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime types."""

    STRONG_TREND_UP = "STRONG_TREND_UP"
    TREND_UP = "TREND_UP"
    RANGING = "RANGING"
    TREND_DOWN = "TREND_DOWN"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    UNKNOWN = "UNKNOWN"


class VolatilityRegime(str, Enum):
    """Volatility regime classifications."""

    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class RegimeAnalysis:
    """Result of regime detection analysis."""

    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_strength: Decimal  # 0-100 scale
    volatility_percentile: Decimal  # 0-100 percentile
    confidence: Decimal  # 0-1 confidence in analysis
    timestamp: datetime
    metadata: dict[str, Any]


class MarketRegimeDetector:
    """Detects market regimes for trading strategy adaptation."""

    def __init__(
        self,
        adx_threshold_trending: float = 25.0,
        adx_threshold_strong: float = 50.0,
        lookback_periods: int = 100,
        cache_ttl_seconds: int = 300
    ):
        """Initialize the market regime detector.
        
        Args:
            adx_threshold_trending: ADX value above which market is considered trending
            adx_threshold_strong: ADX value above which trend is considered strong
            lookback_periods: Number of periods to analyze for regime detection
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.adx_threshold_trending = Decimal(str(adx_threshold_trending))
        self.adx_threshold_strong = Decimal(str(adx_threshold_strong))
        self.lookback_periods = lookback_periods
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cache for regime analysis
        self.cache: dict[str, RegimeAnalysis] = {}
        self.cache_timestamps: dict[str, datetime] = {}

        logger.info(
            "MarketRegimeDetector initialized",
            adx_threshold_trending=adx_threshold_trending,
            adx_threshold_strong=adx_threshold_strong,
            lookback_periods=lookback_periods
        )

    def detect_regime(
        self,
        symbol: str,
        prices: pd.DataFrame,
        use_cache: bool = True
    ) -> RegimeAnalysis | None:
        """Detect the current market regime for a symbol.
        
        Args:
            symbol: Trading symbol
            prices: DataFrame with columns: timestamp, high, low, close, volume
            use_cache: Whether to use cached results
            
        Returns:
            RegimeAnalysis object or None if insufficient data
        """
        try:
            # Check cache
            if use_cache and symbol in self.cache:
                cache_time = self.cache_timestamps.get(symbol)
                if cache_time and (datetime.now(UTC) - cache_time).seconds < self.cache_ttl_seconds:
                    return self.cache[symbol]

            # Validate data
            if len(prices) < self.lookback_periods:
                logger.warning(
                    "Insufficient data for regime detection",
                    symbol=symbol,
                    data_points=len(prices),
                    required=self.lookback_periods
                )
                return None

            # Use recent data for analysis
            recent_data = prices.tail(self.lookback_periods).copy()

            # Detect market regime
            market_regime = self._detect_market_regime(recent_data)

            # Detect volatility regime
            volatility_regime, volatility_percentile = self._detect_volatility_regime(recent_data)

            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(recent_data)

            # Calculate confidence
            confidence = self._calculate_confidence(recent_data, market_regime, volatility_regime)

            # Create analysis result
            analysis = RegimeAnalysis(
                market_regime=market_regime,
                volatility_regime=volatility_regime,
                trend_strength=trend_strength,
                volatility_percentile=volatility_percentile,
                confidence=confidence,
                timestamp=datetime.now(UTC),
                metadata={
                    "symbol": symbol,
                    "data_points": len(recent_data),
                    "lookback_periods": self.lookback_periods
                }
            )

            # Update cache
            if use_cache:
                self.cache[symbol] = analysis
                self.cache_timestamps[symbol] = datetime.now(UTC)

            logger.debug(
                f"Regime detected for {symbol}",
                market_regime=market_regime.value,
                volatility_regime=volatility_regime.value,
                trend_strength=float(trend_strength),
                confidence=float(confidence)
            )

            return analysis

        except Exception as e:
            logger.error(f"Error detecting regime for {symbol}: {e}", exc_info=True)
            return None

    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect market regime based on price action and indicators."""
        try:
            # Convert to Decimal lists for indicator calculations
            highs = [Decimal(str(x)) for x in data["high"].values]
            lows = [Decimal(str(x)) for x in data["low"].values]
            closes = [Decimal(str(x)) for x in data["close"].values]

            # Calculate ADX for trend strength
            adx, plus_di, minus_di = calculate_adx(highs, lows, closes)

            # Calculate price trend using linear regression
            prices_array = np.array(data["close"].values, dtype=float)
            x = np.arange(len(prices_array))

            # Fit linear regression
            slope, intercept = np.polyfit(x, prices_array, 1)

            # Normalize slope by average price
            avg_price = np.mean(prices_array)
            normalized_slope = (slope / avg_price) * 100  # Percentage change per period

            # Determine regime based on ADX and slope
            if adx < self.adx_threshold_trending:
                # Low ADX indicates ranging market
                return MarketRegime.RANGING

            elif adx >= self.adx_threshold_strong:
                # Strong trend
                if normalized_slope > 0.5:  # Strong uptrend
                    return MarketRegime.STRONG_TREND_UP
                elif normalized_slope < -0.5:  # Strong downtrend
                    return MarketRegime.STRONG_TREND_DOWN
                else:
                    # Strong ADX but neutral slope - likely volatile ranging
                    return MarketRegime.RANGING

            else:
                # Moderate trend
                if normalized_slope > 0.1:  # Uptrend
                    return MarketRegime.TREND_UP
                elif normalized_slope < -0.1:  # Downtrend
                    return MarketRegime.TREND_DOWN
                else:
                    return MarketRegime.RANGING

        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return MarketRegime.UNKNOWN

    def _detect_volatility_regime(self, data: pd.DataFrame) -> tuple[VolatilityRegime, Decimal]:
        """Detect volatility regime based on historical volatility."""
        try:
            # Calculate returns
            returns = data["close"].pct_change().dropna()

            # Calculate current volatility (standard deviation of returns)
            current_volatility = returns.std()

            # Calculate historical volatility percentiles
            rolling_volatility = returns.rolling(window=20).std()
            rolling_volatility = rolling_volatility.dropna()

            if len(rolling_volatility) == 0:
                return VolatilityRegime.NORMAL, Decimal("50")

            # Calculate percentile
            percentile = (rolling_volatility < current_volatility).mean() * 100
            percentile_decimal = Decimal(str(percentile))

            # Classify volatility regime
            if percentile < 20:
                regime = VolatilityRegime.VERY_LOW
            elif percentile < 40:
                regime = VolatilityRegime.LOW
            elif percentile < 60:
                regime = VolatilityRegime.NORMAL
            elif percentile < 80:
                regime = VolatilityRegime.HIGH
            else:
                regime = VolatilityRegime.EXTREME

            return regime, percentile_decimal

        except Exception as e:
            logger.error(f"Error in volatility regime detection: {e}")
            return VolatilityRegime.NORMAL, Decimal("50")

    def _calculate_trend_strength(self, data: pd.DataFrame) -> Decimal:
        """Calculate trend strength on a 0-100 scale."""
        try:
            # Convert to Decimal lists
            highs = [Decimal(str(x)) for x in data["high"].values]
            lows = [Decimal(str(x)) for x in data["low"].values]
            closes = [Decimal(str(x)) for x in data["close"].values]

            # Calculate ADX
            adx, _, _ = calculate_adx(highs, lows, closes)

            # ADX is already on a 0-100 scale
            return adx.quantize(Decimal("0.01"))

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return Decimal("0")

    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        market_regime: MarketRegime,
        volatility_regime: VolatilityRegime
    ) -> Decimal:
        """Calculate confidence in regime detection."""
        try:
            confidence = Decimal("0.5")  # Base confidence

            # Increase confidence if we have more data
            if len(data) >= self.lookback_periods * 2:
                confidence += Decimal("0.1")

            # Increase confidence for clear regimes
            if market_regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
                confidence += Decimal("0.2")
            elif market_regime == MarketRegime.RANGING:
                confidence += Decimal("0.15")

            # Adjust for volatility
            if volatility_regime in [VolatilityRegime.NORMAL, VolatilityRegime.LOW]:
                confidence += Decimal("0.1")
            elif volatility_regime == VolatilityRegime.EXTREME:
                confidence -= Decimal("0.1")

            # Cap confidence between 0 and 1
            confidence = max(Decimal("0"), min(Decimal("1"), confidence))

            return confidence.quantize(Decimal("0.01"))

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return Decimal("0.5")

    def is_ranging_market(self, analysis: RegimeAnalysis) -> bool:
        """Check if market is in ranging regime.
        
        Args:
            analysis: RegimeAnalysis object
            
        Returns:
            True if market is ranging, False otherwise
        """
        return analysis.market_regime == MarketRegime.RANGING

    def is_trending_market(self, analysis: RegimeAnalysis) -> bool:
        """Check if market is in trending regime.
        
        Args:
            analysis: RegimeAnalysis object
            
        Returns:
            True if market is trending, False otherwise
        """
        return analysis.market_regime in [
            MarketRegime.TREND_UP,
            MarketRegime.TREND_DOWN,
            MarketRegime.STRONG_TREND_UP,
            MarketRegime.STRONG_TREND_DOWN
        ]

    def get_regime_parameters(self, analysis: RegimeAnalysis) -> dict[str, Any]:
        """Get recommended parameters based on regime.
        
        Args:
            analysis: RegimeAnalysis object
            
        Returns:
            Dictionary of recommended strategy parameters
        """
        params = {}

        # Adjust parameters based on market regime
        if analysis.market_regime == MarketRegime.RANGING:
            # Ranging market - tighter bands, more aggressive mean reversion
            params["bb_std_dev"] = 1.5  # Tighter bands
            params["rsi_oversold"] = 35  # Less extreme RSI
            params["rsi_overbought"] = 65
            params["position_multiplier"] = 1.2  # Slightly larger positions

        elif analysis.market_regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            # Strong trend - avoid mean reversion
            params["bb_std_dev"] = 3.0  # Wider bands
            params["rsi_oversold"] = 20  # More extreme RSI required
            params["rsi_overbought"] = 80
            params["position_multiplier"] = 0.5  # Smaller positions

        else:
            # Normal trending - standard parameters
            params["bb_std_dev"] = 2.0
            params["rsi_oversold"] = 30
            params["rsi_overbought"] = 70
            params["position_multiplier"] = 1.0

        # Adjust for volatility
        if analysis.volatility_regime == VolatilityRegime.EXTREME:
            params["stop_loss_multiplier"] = 3.0  # Wider stops
            params["position_multiplier"] *= 0.5  # Halve position size

        elif analysis.volatility_regime == VolatilityRegime.VERY_LOW:
            params["stop_loss_multiplier"] = 1.5  # Tighter stops
            params["position_multiplier"] *= 1.5  # Increase position size

        else:
            params["stop_loss_multiplier"] = 2.0  # Normal stops

        return params

    def clear_cache(self, symbol: str | None = None):
        """Clear cached regime analysis.
        
        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self.cache.pop(symbol, None)
            self.cache_timestamps.pop(symbol, None)
            logger.debug(f"Cleared cache for {symbol}")
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.debug("Cleared all regime cache")
