"""
Spread Pattern Analyzer

Analyzes historical spread patterns to identify recurring behaviors,
optimal trading times, and volatility patterns.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from genesis.analytics.spread_analyzer import SpreadAnalyzer

logger = structlog.get_logger(__name__)


@dataclass
class SpreadVolatilityScore:
    """Volatility scoring for spread analysis"""

    symbol: str
    score: Decimal  # 0-100 scale
    category: str  # 'low', 'medium', 'high', 'extreme'
    percentile: Decimal
    recent_volatility: Decimal
    historical_volatility: Decimal


@dataclass
class RecurringPattern:
    """Recurring spread pattern detection"""

    symbol: str
    pattern_type: str  # 'compression', 'expansion', 'cyclical'
    frequency_hours: Decimal
    confidence: Decimal  # 0-1 confidence score
    next_occurrence: Optional[datetime]
    description: str


@dataclass
class SpreadAnomalyEvent:
    """Anomaly detection in spread behavior"""

    symbol: str
    anomaly_type: str  # 'spike', 'compression', 'unusual_stability'
    severity: Decimal  # 1-10 scale
    timestamp: datetime
    current_value: Decimal
    expected_value: Decimal
    std_deviations: Decimal


class SpreadPatternAnalyzer:
    """
    Advanced pattern analysis for spread behaviors including
    volatility scoring, pattern recognition, and anomaly detection
    """

    def __init__(
        self, spread_analyzer: SpreadAnalyzer, lookback_hours: int = 168  # 7 days
    ):
        """
        Initialize pattern analyzer

        Args:
            spread_analyzer: SpreadAnalyzer instance
            lookback_hours: Hours of history to analyze
        """
        self.analyzer = spread_analyzer
        self.lookback_hours = lookback_hours

        # Pattern storage
        self._hourly_patterns: dict[str, pd.DataFrame] = {}
        self._daily_patterns: dict[str, pd.DataFrame] = {}
        self._volatility_scores: dict[str, SpreadVolatilityScore] = {}
        self._recurring_patterns: dict[str, list[RecurringPattern]] = defaultdict(list)
        self._anomalies: dict[str, list[SpreadAnomalyEvent]] = defaultdict(list)

        self._logger = logger.bind(component="SpreadPatternAnalyzer")

    def analyze_hourly_patterns(
        self, symbol: str, spread_history: list[tuple[datetime, Decimal]]
    ) -> pd.DataFrame:
        """
        Analyze spread patterns by hour of day

        Args:
            symbol: Trading pair symbol
            spread_history: List of (timestamp, spread_bps) tuples

        Returns:
            DataFrame with hourly statistics
        """
        if not spread_history:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(spread_history, columns=["timestamp", "spread_bps"])
        df["spread_bps"] = df["spread_bps"].astype(float)
        df["hour"] = df["timestamp"].dt.hour

        # Calculate hourly statistics
        hourly_stats = (
            df.groupby("hour")["spread_bps"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .round(4)
        )

        # Add percentiles
        hourly_stats["p25"] = df.groupby("hour")["spread_bps"].quantile(0.25)
        hourly_stats["p75"] = df.groupby("hour")["spread_bps"].quantile(0.75)

        # Cache results
        self._hourly_patterns[symbol] = hourly_stats

        self._logger.debug(
            "Hourly patterns analyzed", symbol=symbol, hours_with_data=len(hourly_stats)
        )

        return hourly_stats

    def analyze_daily_patterns(
        self, symbol: str, spread_history: list[tuple[datetime, Decimal]]
    ) -> pd.DataFrame:
        """
        Analyze spread patterns by day of week

        Args:
            symbol: Trading pair symbol
            spread_history: List of (timestamp, spread_bps) tuples

        Returns:
            DataFrame with daily statistics
        """
        if not spread_history:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(spread_history, columns=["timestamp", "spread_bps"])
        df["spread_bps"] = df["spread_bps"].astype(float)
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_name"] = df["timestamp"].dt.day_name()

        # Calculate daily statistics
        daily_stats = (
            df.groupby(["day_of_week", "day_name"])["spread_bps"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .round(4)
        )

        # Cache results
        self._daily_patterns[symbol] = daily_stats

        self._logger.debug(
            "Daily patterns analyzed", symbol=symbol, days_with_data=len(daily_stats)
        )

        return daily_stats

    def calculate_volatility_score(
        self, symbol: str, spread_history: list[tuple[datetime, Decimal]]
    ) -> SpreadVolatilityScore:
        """
        Calculate volatility score on 0-100 scale

        Args:
            symbol: Trading pair symbol
            spread_history: Historical spread data

        Returns:
            SpreadVolatilityScore with categorization
        """
        if len(spread_history) < 20:
            return SpreadVolatilityScore(
                symbol=symbol,
                score=Decimal("0"),
                category="insufficient_data",
                percentile=Decimal("0"),
                recent_volatility=Decimal("0"),
                historical_volatility=Decimal("0"),
            )

        # Convert to numpy array for calculations
        spreads = np.array([float(spread) for _, spread in spread_history])

        # Calculate volatilities
        historical_vol = np.std(spreads)
        recent_vol = np.std(spreads[-20:])  # Last 20 periods

        # Calculate relative volatility (coefficient of variation)
        mean_spread = np.mean(spreads)
        cv = (historical_vol / mean_spread * 100) if mean_spread > 0 else 0

        # Score based on CV (0-100 scale)
        # CV < 10%: low vol, CV > 50%: extreme vol
        if cv < 10:
            score = cv * 2  # 0-20 range for low vol
            category = "low"
        elif cv < 25:
            score = 20 + (cv - 10) * 2  # 20-50 range for medium
            category = "medium"
        elif cv < 50:
            score = 50 + (cv - 25) * 1.2  # 50-80 range for high
            category = "high"
        else:
            score = min(80 + (cv - 50) * 0.4, 100)  # 80-100 for extreme
            category = "extreme"

        # Calculate percentile rank
        all_vols = [abs(spreads[i] - spreads[i - 1]) for i in range(1, len(spreads))]
        percentile = np.percentile(all_vols, score) if all_vols else 0

        volatility_score = SpreadVolatilityScore(
            symbol=symbol,
            score=Decimal(str(score)),
            category=category,
            percentile=Decimal(str(percentile)),
            recent_volatility=Decimal(str(recent_vol)),
            historical_volatility=Decimal(str(historical_vol)),
        )

        # Cache result
        self._volatility_scores[symbol] = volatility_score

        self._logger.info(
            "Volatility score calculated",
            symbol=symbol,
            score=float(volatility_score.score),
            category=category,
        )

        return volatility_score

    def detect_recurring_patterns(
        self,
        symbol: str,
        spread_history: list[tuple[datetime, Decimal]],
        min_confidence: Decimal = Decimal("0.7"),
    ) -> list[RecurringPattern]:
        """
        Detect recurring patterns in spread behavior

        Args:
            symbol: Trading pair symbol
            spread_history: Historical spread data
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected recurring patterns
        """
        if len(spread_history) < 48:  # Need at least 2 days of data
            return []

        patterns = []

        # Convert to DataFrame for analysis
        df = pd.DataFrame(spread_history, columns=["timestamp", "spread_bps"])
        df["spread_bps"] = df["spread_bps"].astype(float)

        # Detect compression patterns (spread < 80% of rolling mean)
        df["rolling_mean"] = df["spread_bps"].rolling(window=20, min_periods=10).mean()
        df["compression"] = df["spread_bps"] < (df["rolling_mean"] * 0.8)

        # Find compression cycles
        compression_times = df[df["compression"]]["timestamp"].tolist()
        if len(compression_times) > 2:
            # Calculate intervals between compressions
            intervals = [
                (compression_times[i + 1] - compression_times[i]).total_seconds() / 3600
                for i in range(len(compression_times) - 1)
            ]

            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)

                # Low std relative to mean indicates regular pattern
                if std_interval / avg_interval < 0.3:  # Less than 30% variation
                    confidence = Decimal(str(1 - (std_interval / avg_interval)))

                    if confidence >= min_confidence:
                        next_occurrence = compression_times[-1] + timedelta(
                            hours=avg_interval
                        )

                        patterns.append(
                            RecurringPattern(
                                symbol=symbol,
                                pattern_type="compression",
                                frequency_hours=Decimal(str(avg_interval)),
                                confidence=confidence,
                                next_occurrence=next_occurrence,
                                description=f"Spread compression every {avg_interval:.1f} hours",
                            )
                        )

        # Detect cyclical patterns using FFT
        try:
            spreads = df["spread_bps"].values
            if len(spreads) > 24:
                # Apply FFT to find dominant frequencies
                fft = np.fft.fft(spreads)
                frequencies = np.fft.fftfreq(len(spreads))

                # Find dominant frequency (excluding DC component)
                magnitudes = np.abs(fft[1 : len(fft) // 2])
                dominant_idx = np.argmax(magnitudes) + 1
                dominant_freq = frequencies[dominant_idx]

                if dominant_freq > 0:
                    period_hours = 1 / (
                        dominant_freq
                        * len(spreads)
                        / (
                            (
                                df["timestamp"].max() - df["timestamp"].min()
                            ).total_seconds()
                            / 3600
                        )
                    )

                    if 2 <= period_hours <= 168:  # Between 2 hours and 1 week
                        # Calculate confidence based on signal strength
                        signal_strength = magnitudes[dominant_idx - 1] / np.mean(
                            magnitudes
                        )
                        confidence = Decimal(
                            str(min(signal_strength / 10, 1))
                        )  # Normalize to 0-1

                        if confidence >= min_confidence:
                            patterns.append(
                                RecurringPattern(
                                    symbol=symbol,
                                    pattern_type="cyclical",
                                    frequency_hours=Decimal(str(period_hours)),
                                    confidence=confidence,
                                    next_occurrence=None,
                                    description=f"Cyclical pattern with {period_hours:.1f} hour period",
                                )
                            )
        except Exception as e:
            self._logger.debug("FFT analysis failed", error=str(e))

        # Cache patterns
        self._recurring_patterns[symbol] = patterns

        return patterns

    def detect_anomalies(
        self,
        symbol: str,
        current_spread: Decimal,
        spread_history: list[tuple[datetime, Decimal]],
        z_threshold: Decimal = Decimal("3"),
    ) -> Optional[SpreadAnomalyEvent]:
        """
        Detect anomalies in spread behavior

        Args:
            symbol: Trading pair symbol
            current_spread: Current spread value
            spread_history: Historical spread data
            z_threshold: Z-score threshold for anomaly detection

        Returns:
            SpreadAnomalyEvent if anomaly detected, None otherwise
        """
        if len(spread_history) < 20:
            return None

        # Calculate statistics
        spreads = [float(spread) for _, spread in spread_history]
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)

        if std_spread == 0:
            return None

        # Calculate z-score
        z_score = abs((float(current_spread) - mean_spread) / std_spread)

        if z_score >= float(z_threshold):
            # Determine anomaly type
            if float(current_spread) > mean_spread + (3 * std_spread):
                anomaly_type = "spike"
                severity = min(z_score * 2, 10)  # Scale to 1-10
            elif float(current_spread) < mean_spread - (3 * std_spread):
                anomaly_type = "compression"
                severity = min(z_score * 2, 10)
            else:
                anomaly_type = "unusual_stability"
                severity = min(z_score, 10)

            anomaly = SpreadAnomalyEvent(
                symbol=symbol,
                anomaly_type=anomaly_type,
                severity=Decimal(str(severity)),
                timestamp=datetime.now(UTC),
                current_value=current_spread,
                expected_value=Decimal(str(mean_spread)),
                std_deviations=Decimal(str(z_score)),
            )

            # Cache anomaly
            self._anomalies[symbol].append(anomaly)

            # Keep only recent anomalies (last 100)
            self._anomalies[symbol] = self._anomalies[symbol][-100:]

            self._logger.warning(
                "Spread anomaly detected",
                symbol=symbol,
                anomaly_type=anomaly_type,
                z_score=z_score,
                severity=float(severity),
            )

            return anomaly

        return None

    def get_best_trading_hours(
        self, symbol: str, metric: str = "tightness"
    ) -> list[tuple[int, float]]:
        """
        Get best hours for trading based on spread patterns

        Args:
            symbol: Trading pair symbol
            metric: Metric to optimize ('tightness', 'stability', 'volume')

        Returns:
            List of (hour, score) tuples
        """
        if symbol not in self._hourly_patterns:
            return []

        hourly_stats = self._hourly_patterns[symbol]

        if metric == "tightness":
            # Sort by lowest mean spread
            scores = [(hour, -row["mean"]) for hour, row in hourly_stats.iterrows()]
        elif metric == "stability":
            # Sort by lowest std deviation
            scores = [(hour, -row["std"]) for hour, row in hourly_stats.iterrows()]
        else:
            # Default to tightness
            scores = [(hour, -row["mean"]) for hour, row in hourly_stats.iterrows()]

        # Sort by score (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def get_pattern_summary(self, symbol: str) -> dict:
        """
        Get comprehensive pattern summary for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with pattern analysis summary
        """
        summary = {
            "symbol": symbol,
            "volatility_score": self._volatility_scores.get(symbol),
            "recurring_patterns": self._recurring_patterns.get(symbol, []),
            "recent_anomalies": self._anomalies.get(symbol, [])[-5:],  # Last 5
            "best_hours": self.get_best_trading_hours(symbol)[:3],  # Top 3
            "has_hourly_data": symbol in self._hourly_patterns,
            "has_daily_data": symbol in self._daily_patterns,
        }

        return summary
