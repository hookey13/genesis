"""Volume curve estimation for VWAP execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VolumeProfile:
    """Intraday volume profile."""

    symbol: str
    intervals: List[datetime]
    volumes: List[Decimal]
    normalized_volumes: List[Decimal]
    total_volume: Decimal
    date: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalVolumeData:
    """Historical volume data for analysis."""

    symbol: str
    start_date: datetime
    end_date: datetime
    daily_profiles: List[VolumeProfile]
    average_profile: VolumeProfile | None = None
    volatility_profile: List[Decimal] = field(default_factory=list)


class VolumeCurveEstimator:
    """Estimates intraday volume curves from historical data."""

    def __init__(self, lookback_days: int = 20, intervals_per_day: int = 48):
        """Initialize volume curve estimator.

        Args:
            lookback_days: Number of historical days to analyze.
            intervals_per_day: Number of time intervals per trading day.
        """
        self.lookback_days = lookback_days
        self.intervals_per_day = intervals_per_day
        self.interval_minutes = 1440 // intervals_per_day  # Minutes per interval
        self.cached_curves: Dict[str, VolumeProfile] = {}
        self.historical_data: Dict[str, HistoricalVolumeData] = {}

    async def estimate_volume_curve(
        self,
        symbol: str,
        date: Optional[datetime] = None,
        special_events: Optional[List[str]] = None,
    ) -> VolumeProfile:
        """Estimate volume curve for a symbol.

        Args:
            symbol: Trading symbol.
            date: Target date for estimation.
            special_events: List of special events affecting volume.

        Returns:
            Estimated volume profile.
        """
        if date is None:
            date = datetime.now(UTC)

        # Check cache
        cache_key = f"{symbol}:{date.date()}"
        if cache_key in self.cached_curves:
            return self.cached_curves[cache_key]

        # Get historical data
        historical = await self._fetch_historical_volumes(symbol, date)

        # Estimate curve based on historical patterns
        if historical and historical.average_profile:
            profile = await self._adjust_for_conditions(
                historical.average_profile, date, special_events
            )
        else:
            # Generate default U-shaped curve
            profile = self._generate_default_curve(symbol, date)

        # Cache result
        self.cached_curves[cache_key] = profile

        return profile

    async def _fetch_historical_volumes(
        self, symbol: str, target_date: datetime
    ) -> HistoricalVolumeData | None:
        """Fetch historical volume data.

        Args:
            symbol: Trading symbol.
            target_date: Target date for analysis.

        Returns:
            Historical volume data or None.
        """
        # In production, this would fetch from database or market data provider
        # For now, generate synthetic historical data

        start_date = target_date - timedelta(days=self.lookback_days)
        daily_profiles = []

        for i in range(self.lookback_days):
            date = start_date + timedelta(days=i)
            if date.weekday() < 5:  # Weekdays only
                profile = self._generate_daily_profile(symbol, date)
                daily_profiles.append(profile)

        if not daily_profiles:
            return None

        # Calculate average profile
        average_profile = self._calculate_average_profile(symbol, daily_profiles)

        # Calculate volatility profile
        volatility_profile = self._calculate_volatility_profile(
            daily_profiles, average_profile
        )

        historical = HistoricalVolumeData(
            symbol=symbol,
            start_date=start_date,
            end_date=target_date,
            daily_profiles=daily_profiles,
            average_profile=average_profile,
            volatility_profile=volatility_profile,
        )

        self.historical_data[symbol] = historical

        return historical

    def _generate_daily_profile(self, symbol: str, date: datetime) -> VolumeProfile:
        """Generate synthetic daily volume profile.

        Args:
            symbol: Trading symbol.
            date: Trading date.

        Returns:
            Daily volume profile.
        """
        intervals = []
        volumes = []

        # Generate intervals for the day
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)

        for i in range(self.intervals_per_day):
            interval_time = start_time + timedelta(minutes=i * self.interval_minutes)
            intervals.append(interval_time)

            # Generate U-shaped volume with some randomness
            hour = interval_time.hour
            base_volume = self._u_shaped_volume(hour)

            # Add random variation
            variation = np.random.normal(1.0, 0.1)
            volume = Decimal(
                str(base_volume * variation * 1000000)
            )  # Scale to millions
            volumes.append(volume)

        total_volume = sum(volumes)
        normalized = [v / total_volume for v in volumes]

        return VolumeProfile(
            symbol=symbol,
            intervals=intervals,
            volumes=volumes,
            normalized_volumes=normalized,
            total_volume=total_volume,
            date=date,
        )

    def _u_shaped_volume(self, hour: int) -> float:
        """Generate U-shaped volume distribution.

        Args:
            hour: Hour of day (0-23).

        Returns:
            Relative volume level.
        """
        # U-shaped curve: high at open/close, low mid-day
        if hour < 1:  # Market open
            return 3.0
        elif hour < 3:
            return 2.5
        elif hour < 6:
            return 1.5
        elif hour < 10:
            return 1.0
        elif hour < 14:
            return 0.8
        elif hour < 18:
            return 1.0
        elif hour < 21:
            return 1.5
        elif hour < 23:
            return 2.5
        else:  # Market close
            return 3.0

    def _calculate_average_profile(
        self, symbol: str, daily_profiles: List[VolumeProfile]
    ) -> VolumeProfile:
        """Calculate average volume profile from daily profiles.

        Args:
            symbol: Trading symbol.
            daily_profiles: List of daily profiles.

        Returns:
            Average volume profile.
        """
        if not daily_profiles:
            return None

        # Initialize arrays
        num_intervals = self.intervals_per_day
        avg_volumes = [Decimal("0")] * num_intervals

        # Sum volumes across days
        for profile in daily_profiles:
            for i, vol in enumerate(profile.normalized_volumes):
                if i < num_intervals:
                    avg_volumes[i] += vol

        # Calculate average
        num_days = len(daily_profiles)
        avg_volumes = [v / num_days for v in avg_volumes]

        # Renormalize to sum to 1
        total = sum(avg_volumes)
        if total > 0:
            avg_volumes = [v / total for v in avg_volumes]

        # Use first profile's intervals as template
        template = daily_profiles[0]

        return VolumeProfile(
            symbol=symbol,
            intervals=template.intervals,
            volumes=[v * Decimal("1000000") for v in avg_volumes],  # Scale
            normalized_volumes=avg_volumes,
            total_volume=Decimal("1000000"),
            date=datetime.now(UTC),
        )

    def _calculate_volatility_profile(
        self, daily_profiles: List[VolumeProfile], average_profile: VolumeProfile
    ) -> List[Decimal]:
        """Calculate volatility of volume across intervals.

        Args:
            daily_profiles: Daily volume profiles.
            average_profile: Average profile.

        Returns:
            Volatility by interval.
        """
        if not daily_profiles or not average_profile:
            return []

        volatilities = []
        num_intervals = len(average_profile.normalized_volumes)

        for i in range(num_intervals):
            # Collect volumes for this interval across days
            interval_volumes = []
            avg_volume = average_profile.normalized_volumes[i]

            for profile in daily_profiles:
                if i < len(profile.normalized_volumes):
                    interval_volumes.append(profile.normalized_volumes[i])

            if interval_volumes:
                # Calculate standard deviation
                volumes_array = np.array([float(v) for v in interval_volumes])
                std_dev = np.std(volumes_array)
                volatility = Decimal(str(std_dev))
            else:
                volatility = Decimal("0")

            volatilities.append(volatility)

        return volatilities

    async def _adjust_for_conditions(
        self,
        base_profile: VolumeProfile,
        date: datetime,
        special_events: Optional[List[str]] = None,
    ) -> VolumeProfile:
        """Adjust volume profile for special conditions.

        Args:
            base_profile: Base volume profile.
            date: Target date.
            special_events: Special events affecting volume.

        Returns:
            Adjusted volume profile.
        """
        adjusted_volumes = base_profile.normalized_volumes.copy()

        # Adjust for day of week
        day_of_week = date.weekday()
        if day_of_week == 0:  # Monday
            # Typically higher volume at open
            for i in range(min(4, len(adjusted_volumes))):
                adjusted_volumes[i] *= Decimal("1.2")
        elif day_of_week == 4:  # Friday
            # Higher volume at close
            for i in range(max(0, len(adjusted_volumes) - 4), len(adjusted_volumes)):
                adjusted_volumes[i] *= Decimal("1.15")

        # Adjust for special events
        if special_events:
            for event in special_events:
                if "earnings" in event.lower():
                    # Increase volume around announcement time
                    for i in range(len(adjusted_volumes)):
                        adjusted_volumes[i] *= Decimal("1.5")
                elif "fed" in event.lower():
                    # Increase volume in afternoon
                    for i in range(len(adjusted_volumes) // 2, len(adjusted_volumes)):
                        adjusted_volumes[i] *= Decimal("1.3")

        # Renormalize
        total = sum(adjusted_volumes)
        if total > 0:
            adjusted_volumes = [v / total for v in adjusted_volumes]

        return VolumeProfile(
            symbol=base_profile.symbol,
            intervals=base_profile.intervals,
            volumes=[v * base_profile.total_volume for v in adjusted_volumes],
            normalized_volumes=adjusted_volumes,
            total_volume=base_profile.total_volume,
            date=date,
            metadata={"adjusted": True, "special_events": special_events},
        )

    def _generate_default_curve(self, symbol: str, date: datetime) -> VolumeProfile:
        """Generate default U-shaped volume curve.

        Args:
            symbol: Trading symbol.
            date: Target date.

        Returns:
            Default volume profile.
        """
        intervals = []
        normalized_volumes = []

        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)

        for i in range(self.intervals_per_day):
            interval_time = start_time + timedelta(minutes=i * self.interval_minutes)
            intervals.append(interval_time)

            # Generate U-shaped distribution
            hour = interval_time.hour
            volume = Decimal(str(self._u_shaped_volume(hour)))
            normalized_volumes.append(volume)

        # Normalize to sum to 1
        total = sum(normalized_volumes)
        normalized_volumes = [v / total for v in normalized_volumes]

        total_volume = Decimal("1000000")  # Default 1M volume
        volumes = [v * total_volume for v in normalized_volumes]

        return VolumeProfile(
            symbol=symbol,
            intervals=intervals,
            volumes=volumes,
            normalized_volumes=normalized_volumes,
            total_volume=total_volume,
            date=date,
            metadata={"default": True},
        )

    def get_current_interval_volume(
        self, profile: VolumeProfile, current_time: datetime
    ) -> Tuple[Decimal, int]:
        """Get expected volume for current time interval.

        Args:
            profile: Volume profile.
            current_time: Current time.

        Returns:
            Tuple of (expected volume, interval index).
        """
        # Find current interval
        for i, interval_time in enumerate(profile.intervals):
            next_interval = interval_time + timedelta(minutes=self.interval_minutes)
            if interval_time <= current_time < next_interval:
                return profile.normalized_volumes[i], i

        # If not found, return last interval
        if profile.normalized_volumes:
            return profile.normalized_volumes[-1], len(profile.normalized_volumes) - 1

        return Decimal("0"), 0

    async def update_with_realtime_data(
        self, symbol: str, current_volume: Decimal, current_time: datetime
    ) -> None:
        """Update volume curve with real-time data.

        Args:
            symbol: Trading symbol.
            current_volume: Current observed volume.
            current_time: Current time.
        """
        cache_key = f"{symbol}:{current_time.date()}"
        profile = self.cached_curves.get(cache_key)

        if not profile:
            return

        # Find current interval
        expected_volume, interval_idx = self.get_current_interval_volume(
            profile, current_time
        )

        if expected_volume > 0:
            # Calculate deviation
            deviation = (current_volume - expected_volume) / expected_volume

            # Adjust future intervals if deviation is significant
            if abs(deviation) > Decimal("0.2"):  # 20% threshold
                logger.info(
                    "Adjusting volume curve based on real-time data",
                    symbol=symbol,
                    deviation=float(deviation),
                    interval=interval_idx,
                )

                # Adjust remaining intervals proportionally
                adjustment_factor = Decimal("1") + (
                    deviation * Decimal("0.5")
                )  # Partial adjustment

                for i in range(interval_idx + 1, len(profile.normalized_volumes)):
                    profile.normalized_volumes[i] *= adjustment_factor
                    profile.volumes[i] *= adjustment_factor

                # Renormalize
                total = sum(profile.normalized_volumes)
                if total > 0:
                    profile.normalized_volumes = [
                        v / total for v in profile.normalized_volumes
                    ]
