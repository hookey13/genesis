"""
Spread Tracking System

Manages real-time spread tracking across multiple trading pairs with
historical pattern analysis and aggregation.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from genesis.analytics.spread_analyzer import SpreadAnalyzer, SpreadMetrics
from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus
from typing import Optional

logger = structlog.get_logger(__name__)


@dataclass
class SpreadPattern:
    """Spread pattern data for time-based analysis"""

    symbol: str
    hour_of_day: int
    day_of_week: int
    mean_spread: Decimal
    median_spread: Decimal
    std_deviation: Decimal
    sample_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AggregatedSpread:
    """Aggregated spread data for a time period"""

    symbol: str
    period_start: datetime
    period_end: datetime
    mean_spread: Decimal
    median_spread: Decimal
    min_spread: Decimal
    max_spread: Decimal
    std_deviation: Decimal
    sample_count: int


class SpreadTracker:
    """
    Manages spread tracking for multiple trading pairs with
    real-time updates and historical analysis
    """

    def __init__(
        self,
        spread_analyzer: SpreadAnalyzer,
        event_bus: Optional[EventBus] = None,
        aggregation_interval_seconds: int = 3600,
    ):
        """
        Initialize spread tracker

        Args:
            spread_analyzer: SpreadAnalyzer instance
            event_bus: Optional event bus for publishing events
            aggregation_interval_seconds: Interval for aggregating spreads
        """
        self.analyzer = spread_analyzer
        self.event_bus = event_bus
        self.aggregation_interval = aggregation_interval_seconds

        # Historical data storage
        self._raw_spreads: dict[str, list[tuple[datetime, Decimal]]] = defaultdict(list)
        self._hourly_aggregates: dict[str, list[AggregatedSpread]] = defaultdict(list)
        self._daily_aggregates: dict[str, list[AggregatedSpread]] = defaultdict(list)
        self._spread_patterns: dict[str, dict[tuple[int, int], SpreadPattern]] = (
            defaultdict(dict)
        )

        # Rolling window settings
        self.max_raw_history_hours = 24
        self.max_aggregate_days = 30

        self._logger = logger.bind(component="SpreadTracker")
        self._running = False
        self._aggregation_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start spread tracking system"""
        if self._running:
            return

        self._running = True
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._logger.info("Spread tracker started")

    async def stop(self) -> None:
        """Stop spread tracking system"""
        self._running = False
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Spread tracker stopped")

    async def track_pair_spread(self, symbol: str, orderbook: dict) -> SpreadMetrics:
        """
        Track spread for a trading pair from orderbook data

        Args:
            symbol: Trading pair symbol
            orderbook: Order book data

        Returns:
            SpreadMetrics for the pair
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            raise ValueError(f"Invalid orderbook data for {symbol}")

        # Extract best bid and ask
        best_bid = Decimal(str(bids[0][0]))
        best_ask = Decimal(str(asks[0][0]))
        bid_volume = Decimal(str(bids[0][1]))
        ask_volume = Decimal(str(asks[0][1]))

        # Track spread
        metrics = self.analyzer.track_spread(symbol, best_bid, best_ask)

        # Update volumes
        metrics.bid_volume = bid_volume
        metrics.ask_volume = ask_volume

        # Store raw spread data
        now = datetime.now(UTC)
        self._raw_spreads[symbol].append((now, metrics.spread_bps))

        # Trim old data
        cutoff_time = now - timedelta(hours=self.max_raw_history_hours)
        self._raw_spreads[symbol] = [
            (ts, spread) for ts, spread in self._raw_spreads[symbol] if ts > cutoff_time
        ]

        # Check for spread compression
        compression_event = self.analyzer.detect_spread_compression(symbol)
        if compression_event and self.event_bus:
            event = Event(
                type=EventType.SPREAD_COMPRESSION,
                data={"symbol": symbol, "compression_event": compression_event},
            )
            await self.event_bus.publish(event)

        # Calculate order imbalance
        imbalance = self.analyzer.calculate_order_imbalance(orderbook)
        if imbalance.is_significant and self.event_bus:
            event = Event(
                type=EventType.ORDER_IMBALANCE,
                data={"symbol": symbol, "imbalance": imbalance},
            )
            await self.event_bus.publish(event)

        self._logger.debug(
            "Spread tracked",
            symbol=symbol,
            spread_bps=float(metrics.spread_bps),
            bid_volume=float(bid_volume),
            ask_volume=float(ask_volume),
        )

        return metrics

    def get_spread_history(
        self, symbol: str, period: str = "raw"
    ) -> list[SpreadMetrics | AggregatedSpread]:
        """
        Get spread history for a symbol

        Args:
            symbol: Trading pair symbol
            period: Period type ('raw', 'hourly', 'daily')

        Returns:
            List of spread data
        """
        if period == "raw":
            # Return raw spread data as SpreadMetrics
            raw_data = self._raw_spreads.get(symbol, [])
            metrics_list = []
            for timestamp, spread_bps in raw_data:
                # Get cached metrics or create minimal one
                cached = self.analyzer.get_spread_metrics(symbol)
                if cached:
                    metrics = SpreadMetrics(
                        symbol=symbol,
                        current_spread=spread_bps,
                        avg_spread=cached.avg_spread,
                        volatility=cached.volatility,
                        bid_price=cached.bid_price,
                        ask_price=cached.ask_price,
                        bid_volume=cached.bid_volume,
                        ask_volume=cached.ask_volume,
                        timestamp=timestamp,
                    )
                    metrics_list.append(metrics)
            return metrics_list

        elif period == "hourly":
            return self._hourly_aggregates.get(symbol, [])

        elif period == "daily":
            return self._daily_aggregates.get(symbol, [])

        else:
            raise ValueError(f"Invalid period: {period}")

    def identify_spread_patterns(
        self, symbol: str
    ) -> dict[tuple[int, int], SpreadPattern]:
        """
        Identify spread patterns by time of day and day of week

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary of (hour, day) to SpreadPattern
        """
        raw_data = self._raw_spreads.get(symbol, [])
        if not raw_data:
            return {}

        # Group by hour and day
        patterns_data = defaultdict(list)
        for timestamp, spread in raw_data:
            hour = timestamp.hour
            day = timestamp.weekday()
            patterns_data[(hour, day)].append(spread)

        # Calculate statistics for each pattern
        patterns = {}
        for (hour, day), spreads in patterns_data.items():
            if len(spreads) < 3:  # Need minimum samples
                continue

            spreads_sorted = sorted(spreads)
            mean_spread = sum(spreads) / len(spreads)
            median_spread = spreads_sorted[len(spreads) // 2]

            # Calculate standard deviation
            variance = sum((x - mean_spread) ** 2 for x in spreads) / len(spreads)
            std_dev = variance.sqrt()

            pattern = SpreadPattern(
                symbol=symbol,
                hour_of_day=hour,
                day_of_week=day,
                mean_spread=mean_spread,
                median_spread=median_spread,
                std_deviation=std_dev,
                sample_count=len(spreads),
            )
            patterns[(hour, day)] = pattern

        # Cache patterns
        self._spread_patterns[symbol] = patterns

        return patterns

    async def _aggregation_loop(self) -> None:
        """Background task for aggregating spread data"""
        while self._running:
            try:
                await asyncio.sleep(self.aggregation_interval)
                await self._aggregate_spreads()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Aggregation error", error=str(e))

    async def _aggregate_spreads(self) -> None:
        """Aggregate spread data into hourly and daily summaries"""
        now = datetime.now(UTC)

        for symbol in list(self._raw_spreads.keys()):
            raw_data = self._raw_spreads[symbol]
            if not raw_data:
                continue

            # Hourly aggregation
            hour_ago = now - timedelta(hours=1)
            hour_data = [(ts, spread) for ts, spread in raw_data if ts >= hour_ago]

            if hour_data:
                spreads = [spread for _, spread in hour_data]
                spreads_sorted = sorted(spreads)

                hourly_agg = AggregatedSpread(
                    symbol=symbol,
                    period_start=hour_ago,
                    period_end=now,
                    mean_spread=sum(spreads) / len(spreads),
                    median_spread=spreads_sorted[len(spreads) // 2],
                    min_spread=min(spreads),
                    max_spread=max(spreads),
                    std_deviation=self.analyzer.calculate_spread_volatility(spreads),
                    sample_count=len(spreads),
                )
                self._hourly_aggregates[symbol].append(hourly_agg)

                # Trim old hourly data
                cutoff = now - timedelta(days=7)
                self._hourly_aggregates[symbol] = [
                    agg
                    for agg in self._hourly_aggregates[symbol]
                    if agg.period_end > cutoff
                ]

            # Daily aggregation
            day_ago = now - timedelta(days=1)
            day_data = [(ts, spread) for ts, spread in raw_data if ts >= day_ago]

            if day_data:
                spreads = [spread for _, spread in day_data]
                spreads_sorted = sorted(spreads)

                daily_agg = AggregatedSpread(
                    symbol=symbol,
                    period_start=day_ago,
                    period_end=now,
                    mean_spread=sum(spreads) / len(spreads),
                    median_spread=spreads_sorted[len(spreads) // 2],
                    min_spread=min(spreads),
                    max_spread=max(spreads),
                    std_deviation=self.analyzer.calculate_spread_volatility(spreads),
                    sample_count=len(spreads),
                )
                self._daily_aggregates[symbol].append(daily_agg)

                # Trim old daily data
                cutoff = now - timedelta(days=self.max_aggregate_days)
                self._daily_aggregates[symbol] = [
                    agg
                    for agg in self._daily_aggregates[symbol]
                    if agg.period_end > cutoff
                ]

        self._logger.debug(
            "Spreads aggregated",
            symbols=len(self._raw_spreads),
            hourly_aggregates=sum(len(h) for h in self._hourly_aggregates.values()),
            daily_aggregates=sum(len(d) for d in self._daily_aggregates.values()),
        )

    def get_best_spread_times(
        self, symbol: str, top_n: int = 5
    ) -> list[tuple[int, int, Decimal]]:
        """
        Get best times for tight spreads

        Args:
            symbol: Trading pair symbol
            top_n: Number of top results to return

        Returns:
            List of (hour, day, mean_spread) tuples
        """
        patterns = self._spread_patterns.get(symbol, {})
        if not patterns:
            patterns = self.identify_spread_patterns(symbol)

        # Sort by mean spread
        sorted_patterns = sorted(
            [(k[0], k[1], p.mean_spread) for k, p in patterns.items()],
            key=lambda x: x[2],
        )

        return sorted_patterns[:top_n]

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear spread history

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol:
            if symbol in self._raw_spreads:
                del self._raw_spreads[symbol]
            if symbol in self._hourly_aggregates:
                del self._hourly_aggregates[symbol]
            if symbol in self._daily_aggregates:
                del self._daily_aggregates[symbol]
            if symbol in self._spread_patterns:
                del self._spread_patterns[symbol]
        else:
            self._raw_spreads.clear()
            self._hourly_aggregates.clear()
            self._daily_aggregates.clear()
            self._spread_patterns.clear()

        self.analyzer.clear_history(symbol)
        self._logger.info("Spread history cleared", symbol=symbol)
