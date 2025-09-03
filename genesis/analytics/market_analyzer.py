"""
Market Analyzer Core Implementation

Real-time order book analysis and arbitrage opportunity detection system.
Processes market data efficiently for opportunity detection with <10ms latency per pair.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from threading import Lock
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OpportunityType(Enum):
    """Types of trading opportunities"""
    DIRECT_ARBITRAGE = "direct_arbitrage"
    TRIANGULAR_ARBITRAGE = "triangular_arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    SPREAD_CAPTURE = "spread_capture"
    LIQUIDITY_IMBALANCE = "liquidity_imbalance"


@dataclass
class OrderBookLevel:
    """Single level in an order book"""
    price: Decimal
    quantity: Decimal
    exchange: str
    timestamp: float


@dataclass
class OrderBook:
    """Complete order book for a trading pair"""
    symbol: str
    exchange: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: float
    last_update: float = field(default_factory=time.time)

    def get_best_bid(self) -> OrderBookLevel | None:
        """Get best bid price"""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> OrderBookLevel | None:
        """Get best ask price"""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> Decimal | None:
        """Calculate bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    def get_midpoint(self) -> Decimal | None:
        """Calculate midpoint price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / Decimal('2')
        return None

    def get_depth_at_level(self, level: int = 5) -> tuple[Decimal, Decimal]:
        """Get cumulative depth at specified level"""
        bid_depth = sum(b.quantity for b in self.bids[:level])
        ask_depth = sum(a.quantity for a in self.asks[:level])
        return bid_depth, ask_depth


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity"""
    opportunity_id: str
    opportunity_type: OpportunityType
    symbol: str
    exchanges: list[str]
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    quantity: Decimal
    profit_pct: Decimal
    profit_usd: Decimal
    confidence: float
    detected_at: float
    expires_at: float
    execution_path: list[dict[str, Any]] = field(default_factory=list)
    liquidity_score: float = 0.0
    slippage_estimate: Decimal = field(default_factory=lambda: Decimal('0'))
    risk_score: float = 0.0
    statistical_significance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketAnalyzerConfig:
    """Configuration for MarketAnalyzer"""
    min_profit_pct: Decimal = Decimal('0.3')
    min_confidence: float = 0.7
    max_latency_ms: int = 100
    orderbook_ttl_seconds: int = 5
    max_opportunities: int = 10
    spread_history_size: int = 1000
    max_cache_size_mb: int = 100
    enable_triangular: bool = True
    enable_statistical: bool = True
    z_score_threshold: float = 2.0
    min_liquidity_usd: Decimal = Decimal('100')


class MarketAnalyzer:
    """
    Core market analysis engine for real-time opportunity detection.
    
    Processes order book data to identify arbitrage opportunities,
    spread anomalies, and liquidity imbalances with latency optimization.
    """

    def __init__(self, config: MarketAnalyzerConfig | None = None):
        """
        Initialize MarketAnalyzer with configuration.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or MarketAnalyzerConfig()

        # Thread-safe data structures
        self._lock = Lock()
        self._orderbook_cache: dict[tuple[str, str], OrderBook] = {}
        self._spread_history: dict[str, deque] = {}
        self._opportunity_cache: dict[str, ArbitrageOpportunity] = {}

        # Performance tracking
        self._latency_tracker: deque = deque(maxlen=1000)
        self._cache_hits = 0
        self._cache_misses = 0
        self._opportunities_detected = 0

        # Statistical baselines
        self._spread_baselines: dict[str, dict[str, float]] = {}
        self._correlation_matrix: np.ndarray | None = None
        self._last_cleanup = time.time()

        # Memory management
        self._cache_size_bytes = 0
        self._max_cache_size_bytes = self.config.max_cache_size_mb * 1024 * 1024

        logger.info(
            "MarketAnalyzer initialized",
            min_profit_pct=str(self.config.min_profit_pct),
            min_confidence=self.config.min_confidence,
            max_latency_ms=self.config.max_latency_ms,
            orderbook_ttl_seconds=self.config.orderbook_ttl_seconds
        )

    def _update_orderbook_cache(
        self,
        symbol: str,
        exchange: str,
        orderbook: OrderBook
    ) -> None:
        """
        Update order book cache with TTL and memory management.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            orderbook: Order book data
        """
        with self._lock:
            key = (symbol, exchange)

            # Check cache size before adding
            if self._cache_size_bytes >= self._max_cache_size_bytes:
                self._evict_oldest_entries()

            # Update cache
            self._orderbook_cache[key] = orderbook

            # Update spread history
            if symbol not in self._spread_history:
                self._spread_history[symbol] = deque(
                    maxlen=self.config.spread_history_size
                )

            spread = orderbook.get_spread()
            if spread:
                self._spread_history[symbol].append({
                    'spread': float(spread),
                    'timestamp': orderbook.timestamp,
                    'exchange': exchange
                })

            # Clean up expired entries periodically
            if time.time() - self._last_cleanup > 60:
                self._cleanup_expired_entries()
                self._last_cleanup = time.time()

    def _evict_oldest_entries(self) -> None:
        """Evict oldest cache entries to free memory"""
        if not self._orderbook_cache:
            return

        # Sort by last_update and remove oldest 10%
        sorted_items = sorted(
            self._orderbook_cache.items(),
            key=lambda x: x[1].last_update
        )

        num_to_remove = max(1, len(sorted_items) // 10)
        for key, _ in sorted_items[:num_to_remove]:
            del self._orderbook_cache[key]

        logger.debug(f"Evicted {num_to_remove} cache entries")

    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = []

        for key, orderbook in self._orderbook_cache.items():
            if current_time - orderbook.last_update > self.config.orderbook_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._orderbook_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _calculate_spread_baseline(self, symbol: str) -> dict[str, float]:
        """
        Calculate statistical baseline for spread.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with mean, std, percentiles
        """
        if symbol not in self._spread_history or len(self._spread_history[symbol]) < 100:
            return {'mean': 0, 'std': 0, 'p25': 0, 'p50': 0, 'p75': 0}

        spreads = [s['spread'] for s in self._spread_history[symbol]]

        return {
            'mean': np.mean(spreads),
            'std': np.std(spreads),
            'p25': np.percentile(spreads, 25),
            'p50': np.percentile(spreads, 50),
            'p75': np.percentile(spreads, 75)
        }

    async def analyze_market_data(
        self,
        symbol: str,
        exchange: str,
        orderbook_data: dict[str, Any]
    ) -> list[ArbitrageOpportunity]:
        """
        Main entry point for market data analysis.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            orderbook_data: Raw order book data
            
        Returns:
            List of detected opportunities
        """
        start_time = time.time()

        try:
            # Validate and parse order book
            orderbook = self._parse_orderbook(symbol, exchange, orderbook_data)
            if not orderbook:
                return []

            # Update cache
            self._update_orderbook_cache(symbol, exchange, orderbook)

            # Detect opportunities
            opportunities = []

            # Direct arbitrage
            direct_opps = await self._find_direct_arbitrage(symbol)
            opportunities.extend(direct_opps)

            # Triangular arbitrage
            if self.config.enable_triangular:
                triangular_opps = await self._find_triangular_arbitrage(symbol)
                opportunities.extend(triangular_opps)

            # Statistical arbitrage
            if self.config.enable_statistical:
                stat_opps = await self._find_statistical_arbitrage(symbol)
                opportunities.extend(stat_opps)

            # Filter and rank opportunities
            opportunities = self._filter_opportunities(opportunities)
            opportunities = self._rank_opportunities(opportunities)

            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self._latency_tracker.append(latency_ms)

            if latency_ms > self.config.max_latency_ms:
                logger.warning(
                    "High latency detected",
                    symbol=symbol,
                    latency_ms=latency_ms,
                    threshold_ms=self.config.max_latency_ms
                )

            # Update metrics
            self._opportunities_detected += len(opportunities)

            if opportunities:
                logger.info(
                    "Opportunities detected",
                    symbol=symbol,
                    count=len(opportunities),
                    latency_ms=f"{latency_ms:.2f}"
                )

            return opportunities[:self.config.max_opportunities]

        except Exception as e:
            logger.error(
                "Error analyzing market data",
                symbol=symbol,
                exchange=exchange,
                error=str(e)
            )
            return []

    def _parse_orderbook(
        self,
        symbol: str,
        exchange: str,
        data: dict[str, Any]
    ) -> OrderBook | None:
        """
        Parse raw order book data into OrderBook object.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            data: Raw order book data
            
        Returns:
            Parsed OrderBook or None if invalid
        """
        try:
            bids = []
            asks = []

            # Parse bids
            for bid in data.get('bids', []):
                bids.append(OrderBookLevel(
                    price=Decimal(str(bid[0])),
                    quantity=Decimal(str(bid[1])),
                    exchange=exchange,
                    timestamp=time.time()
                ))

            # Parse asks
            for ask in data.get('asks', []):
                asks.append(OrderBookLevel(
                    price=Decimal(str(ask[0])),
                    quantity=Decimal(str(ask[1])),
                    exchange=exchange,
                    timestamp=time.time()
                ))

            if not bids or not asks:
                return None

            return OrderBook(
                symbol=symbol,
                exchange=exchange,
                bids=bids,
                asks=asks,
                timestamp=data.get('timestamp', time.time())
            )

        except Exception as e:
            logger.error(f"Failed to parse orderbook: {e}")
            return None

    async def _find_direct_arbitrage(
        self,
        symbol: str
    ) -> list[ArbitrageOpportunity]:
        """
        Find direct arbitrage opportunities across exchanges.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of direct arbitrage opportunities
        """
        opportunities = []

        with self._lock:
            # Get all orderbooks for this symbol
            symbol_books = {
                exchange: book
                for (sym, exchange), book in self._orderbook_cache.items()
                if sym == symbol
            }

        if len(symbol_books) < 2:
            return opportunities

        # Compare all exchange pairs
        exchanges = list(symbol_books.keys())
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                ex1, ex2 = exchanges[i], exchanges[j]
                book1, book2 = symbol_books[ex1], symbol_books[ex2]

                # Check buy on ex1, sell on ex2
                opp1 = self._check_arbitrage_pair(
                    symbol, ex1, ex2, book1, book2
                )
                if opp1:
                    opportunities.append(opp1)

                # Check buy on ex2, sell on ex1
                opp2 = self._check_arbitrage_pair(
                    symbol, ex2, ex1, book2, book1
                )
                if opp2:
                    opportunities.append(opp2)

        return opportunities

    def _check_arbitrage_pair(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_book: OrderBook,
        sell_book: OrderBook
    ) -> ArbitrageOpportunity | None:
        """
        Check for arbitrage opportunity between two exchanges.
        
        Args:
            symbol: Trading pair symbol
            buy_exchange: Exchange to buy from
            sell_exchange: Exchange to sell on
            buy_book: Buy exchange order book
            sell_book: Sell exchange order book
            
        Returns:
            ArbitrageOpportunity if profitable, None otherwise
        """
        buy_ask = buy_book.get_best_ask()
        sell_bid = sell_book.get_best_bid()

        if not buy_ask or not sell_bid:
            return None

        # Calculate profit
        spread = sell_bid.price - buy_ask.price
        profit_pct = (spread / buy_ask.price) * Decimal('100')

        if profit_pct < self.config.min_profit_pct:
            return None

        # Calculate executable quantity
        quantity = min(buy_ask.quantity, sell_bid.quantity)
        profit_usd = spread * quantity

        # Check minimum liquidity
        if profit_usd < self.config.min_liquidity_usd:
            return None

        # Calculate confidence based on order book depth
        buy_depth, _ = buy_book.get_depth_at_level(5)
        _, sell_depth = sell_book.get_depth_at_level(5)
        depth_ratio = float(min(buy_depth, sell_depth) / max(buy_depth, sell_depth))
        confidence = min(0.95, 0.5 + depth_ratio * 0.45)

        if confidence < self.config.min_confidence:
            return None

        # Create opportunity
        opportunity_id = f"{symbol}_{buy_exchange}_{sell_exchange}_{int(time.time()*1000)}"

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            opportunity_type=OpportunityType.DIRECT_ARBITRAGE,
            symbol=symbol,
            exchanges=[buy_exchange, sell_exchange],
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_ask.price,
            sell_price=sell_bid.price,
            quantity=quantity,
            profit_pct=profit_pct,
            profit_usd=profit_usd,
            confidence=confidence,
            detected_at=time.time(),
            expires_at=time.time() + 5,  # 5 second expiry
            liquidity_score=float(depth_ratio),
            execution_path=[
                {'action': 'buy', 'exchange': buy_exchange, 'price': str(buy_ask.price)},
                {'action': 'sell', 'exchange': sell_exchange, 'price': str(sell_bid.price)}
            ]
        )

    async def _find_triangular_arbitrage(
        self,
        symbol: str
    ) -> list[ArbitrageOpportunity]:
        """
        Find triangular arbitrage opportunities.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of triangular arbitrage opportunities
        """
        opportunities = []

        # Parse symbol to get base and quote currencies
        # Assuming format like BTC/USDT
        parts = symbol.split('/')
        if len(parts) != 2:
            return opportunities

        base, quote = parts

        with self._lock:
            # Find related pairs for triangular paths
            related_pairs = {}
            for (sym, exchange), book in self._orderbook_cache.items():
                if base in sym or quote in sym:
                    if sym not in related_pairs:
                        related_pairs[sym] = {}
                    related_pairs[sym][exchange] = book

        # Check triangular paths
        # Example: BTC/USDT -> ETH/BTC -> ETH/USDT
        for pair1 in related_pairs:
            for pair2 in related_pairs:
                if pair1 == pair2 or pair1 == symbol:
                    continue

                # Check if we can form a triangle
                path = self._find_triangular_path(symbol, pair1, pair2)
                if path:
                    opp = self._calculate_triangular_profit(
                        path,
                        related_pairs
                    )
                    if opp:
                        opportunities.append(opp)

        return opportunities

    def _find_triangular_path(
        self,
        pair1: str,
        pair2: str,
        pair3: str
    ) -> list[str] | None:
        """
        Validate if three pairs form a valid triangular path.
        
        Args:
            pair1: First trading pair
            pair2: Second trading pair
            pair3: Third trading pair
            
        Returns:
            Valid path or None
        """
        # Extract currencies from pairs
        def get_currencies(pair):
            parts = pair.split('/')
            return parts if len(parts) == 2 else None

        c1 = get_currencies(pair1)
        c2 = get_currencies(pair2)
        c3 = get_currencies(pair3)

        if not all([c1, c2, c3]):
            return None

        # Check if currencies form a closed loop
        all_currencies = set()
        all_currencies.update(c1)
        all_currencies.update(c2)
        all_currencies.update(c3)

        # Should have exactly 3 unique currencies for a triangle
        if len(all_currencies) == 3:
            # Verify each currency appears exactly twice
            currency_count = {}
            for c in c1 + c2 + c3:
                currency_count[c] = currency_count.get(c, 0) + 1

            if all(count == 2 for count in currency_count.values()):
                return [pair1, pair2, pair3]

        return None

    def _calculate_triangular_profit(
        self,
        path: list[str],
        related_pairs: dict[str, dict[str, OrderBook]]
    ) -> ArbitrageOpportunity | None:
        """
        Calculate profit for triangular arbitrage path.
        
        Args:
            path: List of trading pairs in the path
            related_pairs: Available order books
            
        Returns:
            ArbitrageOpportunity if profitable
        """
        if len(path) != 3:
            return None
        
        # Get best exchange for each pair in the path
        best_books = []
        for pair in path:
            if pair not in related_pairs or not related_pairs[pair]:
                return None
            
            # Find exchange with best price for this pair
            best_book = None
            best_exchange = None
            for exchange, book in related_pairs[pair].items():
                if book and book.get_best_bid() and book.get_best_ask():
                    if not best_book or book.get_spread() < best_book.get_spread():
                        best_book = book
                        best_exchange = exchange
            
            if not best_book:
                return None
            
            best_books.append((best_exchange, best_book))
        
        # Calculate triangular arbitrage profit
        # Example path: BTC/USDT -> ETH/BTC -> ETH/USDT
        # Start with 1 unit of base currency
        initial_amount = Decimal('1000')  # Start with $1000 USDT
        
        # Leg 1: Buy BTC with USDT
        ex1, book1 = best_books[0]
        ask1 = book1.get_best_ask()
        if not ask1:
            return None
        btc_amount = initial_amount / ask1.price
        
        # Leg 2: Trade BTC for ETH (or other intermediate)
        ex2, book2 = best_books[1]
        # Determine if we're buying or selling based on pair structure
        if 'BTC' in path[1].split('/')[1]:  # BTC is quote currency
            ask2 = book2.get_best_ask()
            if not ask2:
                return None
            intermediate_amount = btc_amount / ask2.price
        else:  # BTC is base currency
            bid2 = book2.get_best_bid()
            if not bid2:
                return None
            intermediate_amount = btc_amount * bid2.price
        
        # Leg 3: Sell intermediate for USDT
        ex3, book3 = best_books[2]
        bid3 = book3.get_best_bid()
        if not bid3:
            return None
        final_amount = intermediate_amount * bid3.price
        
        # Calculate profit
        profit_amount = final_amount - initial_amount
        profit_pct = (profit_amount / initial_amount) * Decimal('100')
        
        # Check if profitable after considering fees (approximate 0.1% per trade)
        total_fees = initial_amount * Decimal('0.003')  # 0.1% * 3 trades
        net_profit = profit_amount - total_fees
        net_profit_pct = (net_profit / initial_amount) * Decimal('100')
        
        if net_profit_pct < self.config.min_profit_pct:
            return None
        
        # Calculate executable quantity based on available liquidity
        min_liquidity = min(
            ask1.quantity * ask1.price,  # Leg 1 liquidity in USDT
            intermediate_amount,  # Leg 2 liquidity
            bid3.quantity * bid3.price  # Leg 3 liquidity in USDT
        )
        
        executable_amount = min(initial_amount, min_liquidity)
        actual_profit = (executable_amount / initial_amount) * net_profit
        
        # Calculate confidence based on liquidity and spread tightness
        liquidity_score = float(min(1.0, min_liquidity / Decimal('10000')))  # Score based on $10k benchmark
        spread_scores = []
        for _, book in best_books:
            spread = book.get_spread()
            midpoint = book.get_midpoint()
            if spread and midpoint:
                spread_pct = (spread / midpoint) * Decimal('100')
                spread_score = max(0, 1 - float(spread_pct) / 2)  # Lower spread = higher score
                spread_scores.append(spread_score)
        
        avg_spread_score = sum(spread_scores) / len(spread_scores) if spread_scores else 0.5
        confidence = (liquidity_score * 0.6 + avg_spread_score * 0.4)
        
        if confidence < self.config.min_confidence:
            return None
        
        # Create opportunity
        opportunity_id = f"tri_{'_'.join(path)}_{int(time.time()*1000)}"
        exchanges_involved = [ex1, ex2, ex3]
        
        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            opportunity_type=OpportunityType.TRIANGULAR_ARBITRAGE,
            symbol='_'.join(path),
            exchanges=list(set(exchanges_involved)),
            buy_exchange=ex1,
            sell_exchange=ex3,
            buy_price=ask1.price,
            sell_price=bid3.price,
            quantity=executable_amount,
            profit_pct=net_profit_pct,
            profit_usd=actual_profit,
            confidence=confidence,
            detected_at=time.time(),
            expires_at=time.time() + 3,  # 3 second expiry for triangular
            liquidity_score=liquidity_score,
            execution_path=[
                {'action': 'buy', 'pair': path[0], 'exchange': ex1, 'price': str(ask1.price)},
                {'action': 'trade', 'pair': path[1], 'exchange': ex2},
                {'action': 'sell', 'pair': path[2], 'exchange': ex3, 'price': str(bid3.price)}
            ],
            metadata={
                'path': path,
                'initial_amount': str(initial_amount),
                'final_amount': str(final_amount),
                'total_fees': str(total_fees)
            }
        )

    async def _find_statistical_arbitrage(
        self,
        symbol: str
    ) -> list[ArbitrageOpportunity]:
        """
        Find statistical arbitrage opportunities based on historical patterns.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            List of statistical arbitrage opportunities
        """
        opportunities = []

        # Calculate spread baseline
        baseline = self._calculate_spread_baseline(symbol)
        if baseline['std'] == 0:
            return opportunities

        with self._lock:
            # Get current spreads across exchanges
            current_spreads = {}
            for (sym, exchange), book in self._orderbook_cache.items():
                if sym == symbol:
                    spread = book.get_spread()
                    if spread:
                        current_spreads[exchange] = float(spread)

        # Check for statistical anomalies
        for exchange, spread in current_spreads.items():
            z_score = (spread - baseline['mean']) / baseline['std']

            if abs(z_score) > self.config.z_score_threshold:
                # Create statistical arbitrage opportunity
                opp = self._create_statistical_opportunity(
                    symbol,
                    exchange,
                    spread,
                    z_score,
                    baseline
                )
                if opp:
                    opportunities.append(opp)

        return opportunities

    def _create_statistical_opportunity(
        self,
        symbol: str,
        exchange: str,
        spread: float,
        z_score: float,
        baseline: dict[str, float]
    ) -> ArbitrageOpportunity | None:
        """
        Create statistical arbitrage opportunity.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            spread: Current spread
            z_score: Statistical z-score
            baseline: Historical baseline statistics
            
        Returns:
            ArbitrageOpportunity or None
        """
        # Estimate profit based on mean reversion
        expected_spread = baseline['mean']
        spread_diff = abs(spread - expected_spread)

        # Get order book for sizing
        with self._lock:
            book = self._orderbook_cache.get((symbol, exchange))
            if not book:
                return None

        best_bid = book.get_best_bid()
        best_ask = book.get_best_ask()

        if not best_bid or not best_ask:
            return None

        # Calculate potential profit
        midpoint = book.get_midpoint()
        profit_pct = (Decimal(str(spread_diff)) / midpoint) * Decimal('100')

        if profit_pct < self.config.min_profit_pct:
            return None

        # Estimate quantity based on liquidity
        quantity = min(best_bid.quantity, best_ask.quantity)
        profit_usd = Decimal(str(spread_diff)) * quantity

        # Calculate confidence based on z-score magnitude
        confidence = min(0.95, abs(z_score) / 5.0)

        opportunity_id = f"stat_{symbol}_{exchange}_{int(time.time()*1000)}"

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            opportunity_type=OpportunityType.STATISTICAL_ARBITRAGE,
            symbol=symbol,
            exchanges=[exchange],
            buy_exchange=exchange,
            sell_exchange=exchange,
            buy_price=best_ask.price,
            sell_price=best_bid.price,
            quantity=quantity,
            profit_pct=profit_pct,
            profit_usd=profit_usd,
            confidence=confidence,
            detected_at=time.time(),
            expires_at=time.time() + 10,  # 10 second expiry for stat arb
            statistical_significance=abs(z_score),
            metadata={
                'z_score': z_score,
                'baseline_mean': baseline['mean'],
                'baseline_std': baseline['std'],
                'current_spread': spread
            }
        )

    def _calculate_liquidity_score(
        self,
        orderbook: OrderBook,
        quantity: Decimal
    ) -> float:
        """
        Calculate liquidity score for a given quantity.
        
        Args:
            orderbook: Order book data
            quantity: Desired quantity
            
        Returns:
            Liquidity score between 0 and 1
        """
        # Calculate how much of the order book we'd consume
        bid_depth, ask_depth = orderbook.get_depth_at_level(10)
        total_depth = bid_depth + ask_depth

        if total_depth == 0:
            return 0.0

        consumption_ratio = float(quantity / total_depth)

        # Score decreases as we consume more of the book
        liquidity_score = max(0.0, 1.0 - consumption_ratio)

        return liquidity_score

    def _estimate_slippage(
        self,
        orderbook: OrderBook,
        quantity: Decimal,
        is_buy: bool
    ) -> Decimal:
        """
        Estimate slippage for a given order size.
        
        Args:
            orderbook: Order book data
            quantity: Order quantity
            is_buy: True for buy orders, False for sell
            
        Returns:
            Estimated slippage in price units
        """
        levels = orderbook.asks if is_buy else orderbook.bids

        if not levels:
            return Decimal('0')

        remaining_qty = quantity
        weighted_price = Decimal('0')
        total_qty = Decimal('0')

        for level in levels:
            if remaining_qty <= 0:
                break

            fill_qty = min(remaining_qty, level.quantity)
            weighted_price += level.price * fill_qty
            total_qty += fill_qty
            remaining_qty -= fill_qty

        if total_qty == 0:
            return Decimal('0')

        avg_price = weighted_price / total_qty
        best_price = levels[0].price

        return abs(avg_price - best_price)

    def _filter_opportunities(
        self,
        opportunities: list[ArbitrageOpportunity]
    ) -> list[ArbitrageOpportunity]:
        """
        Filter opportunities based on criteria.
        
        Args:
            opportunities: List of detected opportunities
            
        Returns:
            Filtered list of opportunities
        """
        filtered = []

        for opp in opportunities:
            # Check minimum profit
            if opp.profit_pct < self.config.min_profit_pct:
                continue

            # Check minimum confidence
            if opp.confidence < self.config.min_confidence:
                continue

            # Check minimum liquidity
            if opp.profit_usd < self.config.min_liquidity_usd:
                continue

            # Check if not expired
            if time.time() > opp.expires_at:
                continue

            # Add liquidity assessment
            with self._lock:
                for exchange in opp.exchanges:
                    book = self._orderbook_cache.get((opp.symbol, exchange))
                    if book:
                        liquidity_score = self._calculate_liquidity_score(
                            book, opp.quantity
                        )
                        opp.liquidity_score = min(opp.liquidity_score or 1.0, liquidity_score)

                        # Estimate slippage
                        is_buy = exchange == opp.buy_exchange
                        slippage = self._estimate_slippage(book, opp.quantity, is_buy)
                        opp.slippage_estimate += slippage

            filtered.append(opp)

        return filtered

    def _rank_opportunities(
        self,
        opportunities: list[ArbitrageOpportunity]
    ) -> list[ArbitrageOpportunity]:
        """
        Rank opportunities by risk-adjusted profit potential.
        
        Args:
            opportunities: List of opportunities to rank
            
        Returns:
            Sorted list of opportunities
        """
        for opp in opportunities:
            # Calculate risk-adjusted score
            # Higher profit, higher confidence, lower slippage = better
            profit_score = float(opp.profit_pct)
            confidence_score = opp.confidence * 100
            liquidity_score = opp.liquidity_score * 50
            slippage_penalty = float(opp.slippage_estimate) * 10

            # Statistical significance bonus
            stat_bonus = 0
            if opp.statistical_significance:
                stat_bonus = min(20, opp.statistical_significance * 5)

            # Calculate final score
            opp.risk_score = (
                profit_score +
                confidence_score +
                liquidity_score +
                stat_bonus -
                slippage_penalty
            ) / 100

        # Sort by risk score (descending)
        return sorted(opportunities, key=lambda x: x.risk_score, reverse=True)

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get performance metrics for monitoring.
        
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            avg_latency = np.mean(list(self._latency_tracker)) if self._latency_tracker else 0
            p95_latency = np.percentile(list(self._latency_tracker), 95) if self._latency_tracker else 0

            cache_hit_rate = 0
            if self._cache_hits + self._cache_misses > 0:
                cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)

            return {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'cache_size': len(self._orderbook_cache),
                'cache_hit_rate': cache_hit_rate,
                'opportunities_detected': self._opportunities_detected,
                'cache_size_mb': self._cache_size_bytes / (1024 * 1024)
            }

    def cleanup(self) -> None:
        """Clean up resources and clear caches"""
        with self._lock:
            self._orderbook_cache.clear()
            self._spread_history.clear()
            self._opportunity_cache.clear()
            self._latency_tracker.clear()
            self._cache_size_bytes = 0

        logger.info("MarketAnalyzer cleanup completed")
