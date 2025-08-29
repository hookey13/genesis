
"""Liquidity scanner for pair discovery and analysis."""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from uuid import uuid4

import aiohttp
import structlog

from genesis.core.exceptions import MarketDataError

logger = structlog.get_logger(__name__)


class LiquidityTier(Enum):
    """Liquidity tier categorization based on daily volume."""

    LOW = "LOW"  # <$100k daily volume
    MEDIUM = "MEDIUM"  # $100k-$1M daily volume
    HIGH = "HIGH"  # >$1M daily volume


class HealthStatus(Enum):
    """Pair health status for monitoring."""

    HEALTHY = "HEALTHY"
    DEGRADING = "DEGRADING"
    UNHEALTHY = "UNHEALTHY"
    BLACKLISTED = "BLACKLISTED"


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a trading pair."""

    symbol: str
    volume_24h: Decimal
    spread_bps: int
    bid_depth_10: Decimal
    ask_depth_10: Decimal
    tier: LiquidityTier
    depth_score: Decimal
    timestamp: datetime


@dataclass
class TierAlert:
    """Alert for tier graduation eligibility."""

    current_tier: str
    recommended_tier: str
    current_capital: Decimal
    message: str


@dataclass
class OrderBook:
    """Order book snapshot."""

    symbol: str
    bids: list[tuple[Decimal, Decimal]]  # (price, quantity)
    asks: list[tuple[Decimal, Decimal]]  # (price, quantity)
    timestamp: datetime


class LiquidityScanner:
    """Main liquidity scanner for pair discovery and analysis."""

    def __init__(self, session: aiohttp.ClientSession | None = None):
        """Initialize liquidity scanner.

        Args:
            session: Optional aiohttp session for API calls
        """
        self.session = session
        self._owned_session = False
        if not self.session:
            self.session = aiohttp.ClientSession()
            self._owned_session = True

        self.logger = logger.bind(component="liquidity_scanner")
        self._rate_limit_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._owned_session and self.session:
            await self.session.close()

    async def scan_all_pairs(self) -> dict[str, LiquidityMetrics]:
        """Scan all trading pairs for liquidity metrics.

        Returns:
            Dictionary mapping symbol to liquidity metrics
        """
        try:
            self.logger.info("starting_liquidity_scan")

            # Fetch all trading pairs and their 24hr data
            async with self._rate_limit_semaphore:
                async with self.session.get(
                    "https://api.binance.com/api/v3/ticker/24hr"
                ) as response:
                    if response.status != 200:
                        raise MarketDataError(f"API error: {response.status}")
                    ticker_data = await response.json()

            results = {}
            tasks = []

            for ticker in ticker_data:
                symbol = ticker["symbol"]
                if not symbol.endswith("USDT"):
                    continue  # Only scan USDT pairs

                volume_24h = Decimal(ticker["quoteVolume"])
                if volume_24h < Decimal("10000"):
                    continue  # Skip very low volume pairs

                # Create task for depth analysis
                task = self._analyze_pair_liquidity(symbol, volume_24h, ticker)
                tasks.append(task)

            # Process in batches to respect rate limits
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error("pair_analysis_failed", error=str(result))
                    elif result:
                        results[result.symbol] = result

            self.logger.info("liquidity_scan_complete", pairs_scanned=len(results))
            return results

        except Exception as e:
            self.logger.error("liquidity_scan_failed", error=str(e))
            raise MarketDataError(f"Liquidity scan failed: {e}")

    async def _analyze_pair_liquidity(
        self, symbol: str, volume_24h: Decimal, ticker: dict
    ) -> LiquidityMetrics | None:
        """Analyze liquidity for a single pair.

        Args:
            symbol: Trading pair symbol
            volume_24h: 24-hour volume in USDT
            ticker: Ticker data from API

        Returns:
            Liquidity metrics or None if analysis fails
        """
        try:
            async with self._rate_limit_semaphore:
                # Fetch order book depth
                depth_data = await self.analyze_order_book_depth(symbol, levels=10)

                if not depth_data:
                    return None

                # Calculate spread
                best_bid = Decimal(ticker.get("bidPrice", "0"))
                best_ask = Decimal(ticker.get("askPrice", "0"))

                if best_bid == 0 or best_ask == 0:
                    return None

                spread_bps = int((best_ask - best_bid) / best_ask * Decimal("10000"))

                # Categorize by volume
                tier = self.categorize_by_volume(volume_24h)

                # Calculate depth score (0-100)
                total_depth = depth_data["bid_depth"] + depth_data["ask_depth"]
                depth_score = min(
                    Decimal("100"),
                    (
                        (total_depth / volume_24h * Decimal("100"))
                        if volume_24h > 0
                        else Decimal("0")
                    ),
                )

                return LiquidityMetrics(
                    symbol=symbol,
                    volume_24h=volume_24h,
                    spread_bps=spread_bps,
                    bid_depth_10=depth_data["bid_depth"],
                    ask_depth_10=depth_data["ask_depth"],
                    tier=tier,
                    depth_score=depth_score,
                    timestamp=datetime.now(UTC),
                )

        except Exception as e:
            self.logger.error("pair_analysis_error", symbol=symbol, error=str(e))
            return None

    def categorize_by_volume(self, volume_24h: Decimal) -> LiquidityTier:
        """Categorize pair by 24-hour volume.

        Args:
            volume_24h: 24-hour volume in USDT

        Returns:
            Liquidity tier
        """
        if volume_24h < Decimal("100000"):
            return LiquidityTier.LOW
        elif volume_24h < Decimal("1000000"):
            return LiquidityTier.MEDIUM
        else:
            return LiquidityTier.HIGH

    async def analyze_order_book_depth(
        self, symbol: str, levels: int = 10
    ) -> dict | None:
        """Analyze order book depth at specified levels.

        Args:
            symbol: Trading pair symbol
            levels: Number of order book levels to analyze

        Returns:
            Dictionary with depth analysis or None if error
        """
        try:
            async with self.session.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": symbol, "limit": levels},
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                # Calculate total bid and ask depth
                bid_depth = Decimal("0")
                ask_depth = Decimal("0")

                for bid in data.get("bids", []):
                    price = Decimal(bid[0])
                    qty = Decimal(bid[1])
                    bid_depth += price * qty

                for ask in data.get("asks", []):
                    price = Decimal(ask[0])
                    qty = Decimal(ask[1])
                    ask_depth += price * qty

                return {
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "levels": levels,
                    "timestamp": datetime.now(UTC),
                }

        except Exception as e:
            self.logger.error("order_book_analysis_error", symbol=symbol, error=str(e))
            return None


class SpreadPersistenceTracker:
    """Track spread persistence over time."""

    def __init__(self, window_hours: int = 24):
        """Initialize spread persistence tracker.

        Args:
            window_hours: Rolling window size in hours
        """
        self.window_hours = window_hours
        self.spread_history: dict[str, deque] = {}
        self.logger = logger.bind(component="spread_tracker")

    def record_spread(self, symbol: str, spread_bps: int, timestamp: datetime):
        """Record spread observation.

        Args:
            symbol: Trading pair symbol
            spread_bps: Spread in basis points
            timestamp: Observation timestamp
        """
        if symbol not in self.spread_history:
            self.spread_history[symbol] = deque()

        # Add new observation
        self.spread_history[symbol].append((timestamp, spread_bps))

        # Remove old observations outside window
        cutoff = timestamp - timedelta(hours=self.window_hours)
        while (
            self.spread_history[symbol] and self.spread_history[symbol][0][0] < cutoff
        ):
            self.spread_history[symbol].popleft()

    def calculate_spread_persistence_score(self, symbol: str) -> Decimal:
        """Calculate spread persistence score (0-100).

        Higher score means spreads remain profitable longer.

        Args:
            symbol: Trading pair symbol

        Returns:
            Persistence score from 0 to 100
        """
        if symbol not in self.spread_history or len(self.spread_history[symbol]) < 2:
            return Decimal("50")  # Default neutral score

        spreads = [spread for _, spread in self.spread_history[symbol]]

        # Calculate persistence metrics
        profitable_threshold = 10  # 10 bps considered profitable
        profitable_periods = sum(1 for s in spreads if s >= profitable_threshold)
        total_periods = len(spreads)

        if total_periods == 0:
            return Decimal("50")

        # Calculate score based on profitable period ratio
        persistence_ratio = Decimal(profitable_periods) / Decimal(total_periods)
        score = persistence_ratio * Decimal("100")

        # Adjust for consistency (lower variance is better)
        if len(spreads) > 10:
            avg_spread = sum(spreads) / len(spreads)
            variance = sum((s - avg_spread) ** 2 for s in spreads) / len(spreads)
            consistency_factor = max(
                Decimal("0.5"), Decimal("1") - Decimal(str(variance)) / Decimal("1000")
            )
            score *= consistency_factor

        return min(Decimal("100"), max(Decimal("0"), score))


class PairRecommendationEngine:
    """Recommend trading pairs based on tier and capital."""

    def __init__(self):
        """Initialize recommendation engine."""
        self.logger = logger.bind(component="recommendation_engine")

    def recommend_pairs_for_tier(
        self, tier: str, capital: Decimal, liquidity_data: dict[str, LiquidityMetrics]
    ) -> list[str]:
        """Recommend pairs appropriate for tier and capital.

        Args:
            tier: Current trading tier (SNIPER, HUNTER, STRATEGIST)
            capital: Current capital in USDT
            liquidity_data: Dictionary of liquidity metrics by symbol

        Returns:
            List of recommended symbols
        """
        # Map tier to appropriate liquidity levels
        tier_liquidity_map = {
            "SNIPER": LiquidityTier.LOW,  # <$2k capital → <$100k volume
            "HUNTER": LiquidityTier.MEDIUM,  # $2k-$10k → $100k-$1M volume
            "STRATEGIST": LiquidityTier.HIGH,  # $10k+ → >$1M volume
        }

        target_liquidity = tier_liquidity_map.get(tier, LiquidityTier.LOW)

        # Filter pairs by liquidity tier
        eligible_pairs = [
            (symbol, metrics)
            for symbol, metrics in liquidity_data.items()
            if metrics.tier == target_liquidity
        ]

        # Sort by composite score
        scored_pairs = []
        for symbol, metrics in eligible_pairs:
            # Score based on spread, depth, and volume
            spread_score = max(
                Decimal("0"), Decimal("100") - Decimal(metrics.spread_bps)
            )
            depth_score = metrics.depth_score
            volume_score = min(
                Decimal("100"),
                (metrics.volume_24h / Decimal("1000000")) * Decimal("100"),
            )

            composite_score = (
                spread_score * Decimal("0.4")
                + depth_score * Decimal("0.3")
                + volume_score * Decimal("0.3")
            )

            scored_pairs.append((symbol, composite_score))

        # Sort by score and return top recommendations
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        recommendations = [symbol for symbol, _ in scored_pairs[:10]]

        self.logger.info(
            "pairs_recommended",
            tier=tier,
            capital=float(capital),
            count=len(recommendations),
        )

        return recommendations

    def check_graduation_eligibility(
        self, current_capital: Decimal, current_tier: str
    ) -> TierAlert | None:
        """Check if capital qualifies for tier graduation.

        Args:
            current_capital: Current capital in USDT
            current_tier: Current trading tier

        Returns:
            Tier alert if graduation eligible, None otherwise
        """
        tier_thresholds = {
            "SNIPER": (Decimal("500"), Decimal("2000")),
            "HUNTER": (Decimal("2000"), Decimal("10000")),
            "STRATEGIST": (Decimal("10000"), None),
        }

        current_range = tier_thresholds.get(current_tier)
        if not current_range:
            return None

        _, upper_limit = current_range

        # Check if capital exceeds tier upper limit
        if upper_limit and current_capital >= upper_limit:
            next_tier = (
                "HUNTER"
                if current_tier == "SNIPER"
                else "STRATEGIST" if current_tier == "HUNTER" else None
            )

            if next_tier:
                return TierAlert(
                    current_tier=current_tier,
                    recommended_tier=next_tier,
                    current_capital=current_capital,
                    message=f"Capital of ${current_capital} qualifies for {next_tier} tier",
                )

        return None


class PairHealthMonitor:
    """Monitor pair health and maintain blacklist."""

    def __init__(self):
        """Initialize pair health monitor."""
        self.logger = logger.bind(component="health_monitor")
        self.blacklist: dict[str, dict] = {}
        self.health_history: dict[str, list[HealthStatus]] = {}

    def monitor_pair_health(
        self,
        symbol: str,
        current_metrics: LiquidityMetrics,
        historical_metrics: list[LiquidityMetrics],
    ) -> HealthStatus:
        """Monitor pair health status.

        Args:
            symbol: Trading pair symbol
            current_metrics: Current liquidity metrics
            historical_metrics: Historical metrics for comparison

        Returns:
            Health status
        """
        if self.is_blacklisted(symbol):
            return HealthStatus.BLACKLISTED

        if len(historical_metrics) < 5:
            return HealthStatus.HEALTHY  # Not enough data

        # Calculate degradation indicators
        avg_volume = sum(m.volume_24h for m in historical_metrics) / len(
            historical_metrics
        )
        volume_decline = (avg_volume - current_metrics.volume_24h) / avg_volume

        avg_spread = sum(m.spread_bps for m in historical_metrics) / len(
            historical_metrics
        )
        spread_widening = (
            (current_metrics.spread_bps - avg_spread) / avg_spread
            if avg_spread > 0
            else Decimal("0")
        )

        avg_depth = sum(m.depth_score for m in historical_metrics) / len(
            historical_metrics
        )
        depth_reduction = (avg_depth - current_metrics.depth_score) / avg_depth

        # Determine health status
        if (
            volume_decline > Decimal("0.5")
            or spread_widening > Decimal("0.5")
            or depth_reduction > Decimal("0.5")
        ):
            status = HealthStatus.UNHEALTHY
        elif (
            volume_decline > Decimal("0.25")
            or spread_widening > Decimal("0.25")
            or depth_reduction > Decimal("0.25")
        ):
            status = HealthStatus.DEGRADING
        else:
            status = HealthStatus.HEALTHY

        # Track health history
        if symbol not in self.health_history:
            self.health_history[symbol] = []
        self.health_history[symbol].append(status)

        # Keep only last 5 days
        self.health_history[symbol] = self.health_history[symbol][-5:]

        # Check for auto-blacklisting (5 consecutive unhealthy days)
        if len(self.health_history[symbol]) >= 5 and all(
            s == HealthStatus.UNHEALTHY for s in self.health_history[symbol]
        ):
            self.add_to_blacklist(symbol, "5 consecutive days of unhealthy status", 5)
            return HealthStatus.BLACKLISTED

        return status

    def add_to_blacklist(self, symbol: str, reason: str, consecutive_losses: int):
        """Add pair to blacklist.

        Args:
            symbol: Trading pair symbol
            reason: Blacklist reason
            consecutive_losses: Number of consecutive losing days
        """
        self.blacklist[symbol] = {
            "blacklist_id": str(uuid4()),
            "reason": reason,
            "consecutive_losses": consecutive_losses,
            "blacklisted_at": datetime.now(UTC),
            "expires_at": datetime.now(UTC) + timedelta(days=30),  # 30-day blacklist
        }

        self.logger.warning(
            "pair_blacklisted",
            symbol=symbol,
            reason=reason,
            losses=consecutive_losses,
        )

    def is_blacklisted(self, symbol: str) -> bool:
        """Check if pair is blacklisted.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if blacklisted and not expired
        """
        if symbol not in self.blacklist:
            return False

        entry = self.blacklist[symbol]
        if entry["expires_at"] and datetime.now(UTC) > entry["expires_at"]:
            # Expired, remove from blacklist
            del self.blacklist[symbol]
            return False

        return True


class LiquidityScannerJob:
    """Scheduled job for periodic liquidity scanning."""

    def __init__(
        self,
        scanner: LiquidityScanner,
        persistence_tracker: SpreadPersistenceTracker,
        health_monitor: PairHealthMonitor,
    ):
        """Initialize scanner job.

        Args:
            scanner: Liquidity scanner instance
            persistence_tracker: Spread persistence tracker
            health_monitor: Pair health monitor
        """
        self.scanner = scanner
        self.persistence_tracker = persistence_tracker
        self.health_monitor = health_monitor
        self.logger = logger.bind(component="scanner_job")
        self._running = False
        self._task = None

    async def start(self, scan_interval_hours: int = 24):
        """Start periodic scanning.

        Args:
            scan_interval_hours: Hours between full scans
        """
        if self._running:
            self.logger.warning("scanner_job_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_periodic(scan_interval_hours))
        self.logger.info("scanner_job_started", interval_hours=scan_interval_hours)

    async def stop(self):
        """Stop periodic scanning."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("scanner_job_stopped")

    async def _run_periodic(self, interval_hours: int):
        """Run periodic scanning loop.

        Args:
            interval_hours: Hours between scans
        """
        while self._running:
            try:
                await self.run_scan()
                await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("scan_job_error", error=str(e))
                await asyncio.sleep(60)  # Retry after 1 minute on error

    async def run_scan(self) -> dict[str, LiquidityMetrics]:
        """Execute a full liquidity scan.

        Returns:
            Scan results
        """
        self.logger.info("executing_liquidity_scan")

        try:
            # Run the scan
            results = await self.scanner.scan_all_pairs()

            # Update spread persistence
            for symbol, metrics in results.items():
                self.persistence_tracker.record_spread(
                    symbol, metrics.spread_bps, metrics.timestamp
                )

            self.logger.info("liquidity_scan_completed", pairs_count=len(results))
            return results

        except Exception as e:
            self.logger.error("liquidity_scan_failed", error=str(e))
            raise

    async def trigger_manual_scan(self) -> dict[str, LiquidityMetrics]:
        """Trigger a manual scan outside of schedule.

        Returns:
            Scan results
        """
        self.logger.info("manual_scan_triggered")
        return await self.run_scan()
