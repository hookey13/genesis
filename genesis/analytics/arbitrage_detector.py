"""Arbitrage detection engine for identifying profitable trading opportunities."""
import asyncio
import time
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid

import numpy as np
from scipy import stats
import structlog

from genesis.analytics.opportunity_models import (
    ArbitrageOpportunity,
    DirectArbitrageOpportunity,
    TriangularArbitrageOpportunity,
    StatisticalArbitrageOpportunity,
    OpportunityType,
    OpportunityStatus,
    ExchangePair,
    ExecutionPath
)
from genesis.core.exceptions import ValidationError
from genesis.core.constants import DECIMAL_PRECISION


logger = structlog.get_logger(__name__)


class ArbitrageDetector:
    """Main arbitrage detection engine."""
    
    def __init__(
        self,
        min_profit_pct: float = 0.3,
        min_confidence: float = 0.6,
        max_path_length: int = 4,
        stat_arb_window: int = 100,
        opportunity_ttl: int = 5,
        max_opportunities: int = 50,
        fee_structures: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """Initialize arbitrage detector.
        
        Args:
            min_profit_pct: Minimum profit percentage to consider
            min_confidence: Minimum confidence score for opportunities
            max_path_length: Maximum hops for triangular arbitrage
            stat_arb_window: Lookback window for statistical arbitrage
            opportunity_ttl: Time to live for opportunities in seconds
            max_opportunities: Maximum opportunities to track
            fee_structures: Exchange fee structures
        """
        self.min_profit_pct = Decimal(str(min_profit_pct))
        self.min_confidence = min_confidence
        self.max_path_length = max_path_length
        self.stat_arb_window = stat_arb_window
        self.opportunity_ttl = opportunity_ttl
        self.max_opportunities = max_opportunities
        
        self.fee_structures = fee_structures or {
            "binance": {"maker": 0.001, "taker": 0.001},
            "coinbase": {"maker": 0.005, "taker": 0.005},
            "kraken": {"maker": 0.0016, "taker": 0.0026}
        }
        
        self.opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = defaultdict(list)
        self.pair_graph: Dict[str, Dict[str, ExchangePair]] = {}
        self._lock = asyncio.Lock()
        
        logger.info(
            "arbitrage_detector_initialized",
            min_profit_pct=float(self.min_profit_pct),
            min_confidence=self.min_confidence,
            max_path_length=self.max_path_length
        )
    
    async def find_direct_arbitrage(
        self,
        market_data: Dict[str, List[ExchangePair]]
    ) -> List[DirectArbitrageOpportunity]:
        """Find direct arbitrage opportunities between exchanges.
        
        Args:
            market_data: Market data by symbol and exchange
            
        Returns:
            List of direct arbitrage opportunities
        """
        opportunities = []
        
        for symbol, exchange_pairs in market_data.items():
            if len(exchange_pairs) < 2:
                continue
                
            for i, pair_a in enumerate(exchange_pairs):
                for pair_b in exchange_pairs[i + 1:]:
                    opportunity = self._check_direct_arbitrage(pair_a, pair_b, symbol)
                    if opportunity:
                        opportunities.append(opportunity)
        
        logger.info(
            "direct_arbitrage_scan_complete",
            opportunities_found=len(opportunities)
        )
        
        return opportunities
    
    def _check_direct_arbitrage(
        self,
        pair_a: ExchangePair,
        pair_b: ExchangePair,
        symbol: str
    ) -> Optional[DirectArbitrageOpportunity]:
        """Check for direct arbitrage between two exchange pairs.
        
        Args:
            pair_a: First exchange pair
            pair_b: Second exchange pair
            symbol: Trading symbol
            
        Returns:
            DirectArbitrageOpportunity if profitable, None otherwise
        """
        buy_exchange = None
        sell_exchange = None
        buy_price = None
        sell_price = None
        
        if pair_a.ask_price < pair_b.bid_price:
            buy_exchange = pair_a.exchange
            sell_exchange = pair_b.exchange
            buy_price = pair_a.ask_price
            sell_price = pair_b.bid_price
            max_volume = min(pair_a.ask_volume, pair_b.bid_volume)
        elif pair_b.ask_price < pair_a.bid_price:
            buy_exchange = pair_b.exchange
            sell_exchange = pair_a.exchange
            buy_price = pair_b.ask_price
            sell_price = pair_a.bid_price
            max_volume = min(pair_b.ask_volume, pair_a.bid_volume)
        else:
            return None
        
        buy_fee = Decimal(str(self.fee_structures.get(buy_exchange, {"taker": 0.001})["taker"]))
        sell_fee = Decimal(str(self.fee_structures.get(sell_exchange, {"taker": 0.001})["taker"]))
        
        profit_before_fees = (sell_price - buy_price) / buy_price * 100
        total_fees = (buy_fee + sell_fee) * 100
        net_profit_pct = profit_before_fees - total_fees
        
        if net_profit_pct < self.min_profit_pct:
            return None
        
        profit_amount = (sell_price * (1 - sell_fee) - buy_price * (1 + buy_fee)) * max_volume
        confidence = self._calculate_direct_confidence(pair_a, pair_b, max_volume)
        
        return DirectArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=profit_before_fees,
            profit_amount=profit_amount,
            confidence_score=confidence,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.opportunity_ttl),
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            symbol=symbol,
            buy_price=buy_price,
            sell_price=sell_price,
            max_volume=max_volume,
            buy_fee=buy_fee * 100,
            sell_fee=sell_fee * 100,
            net_profit_pct=net_profit_pct
        )
    
    async def find_triangular_arbitrage(
        self,
        exchange: str,
        market_data: Dict[str, ExchangePair]
    ) -> List[TriangularArbitrageOpportunity]:
        """Find triangular arbitrage opportunities.
        
        Args:
            exchange: Exchange name
            market_data: Market data for all pairs on exchange
            
        Returns:
            List of triangular arbitrage opportunities
        """
        self._build_pair_graph(market_data)
        opportunities = []
        
        currencies = self._extract_currencies(market_data)
        
        for start_currency in currencies:
            cycles = self._find_profitable_cycles(
                start_currency,
                exchange,
                max_depth=min(self.max_path_length, 4)
            )
            
            for cycle in cycles:
                opportunity = self._create_triangular_opportunity(
                    cycle,
                    exchange,
                    start_currency
                )
                if opportunity:
                    opportunities.append(opportunity)
        
        logger.info(
            "triangular_arbitrage_scan_complete",
            exchange=exchange,
            opportunities_found=len(opportunities)
        )
        
        return opportunities
    
    def _build_pair_graph(self, market_data: Dict[str, ExchangePair]) -> None:
        """Build graph representation of currency pairs.
        
        Args:
            market_data: Market data dictionary with symbol keys
        """
        self.pair_graph.clear()
        
        for symbol, pair in market_data.items():
            if '/' not in symbol:
                logger.warning("invalid_symbol_format", symbol=symbol)
                continue
                
            base, quote = symbol.split('/', 1)  # Handle edge case of multiple slashes
            
            if base not in self.pair_graph:
                self.pair_graph[base] = {}
            self.pair_graph[base][quote] = pair
            
            if quote not in self.pair_graph:
                self.pair_graph[quote] = {}
    
    def _extract_currencies(self, market_data: Dict[str, ExchangePair]) -> Set[str]:
        """Extract unique currencies from market data.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Set of unique currency symbols
        """
        currencies = set()
        for symbol in market_data.keys():
            if '/' not in symbol:
                continue
            base, quote = symbol.split('/', 1)  # Handle edge case of multiple slashes
            currencies.add(base)
            currencies.add(quote)
        return currencies
    
    def _find_profitable_cycles(
        self,
        start: str,
        exchange: str,
        max_depth: int
    ) -> List[List[Tuple[str, str, ExchangePair]]]:
        """Find profitable arbitrage cycles using modified Bellman-Ford."""
        profitable_cycles = []
        
        def dfs(current: str, path: List[Tuple[str, str, ExchangePair]], product: Decimal):
            if len(path) >= 3 and current == start:
                fee_product = Decimal('1')
                for _, _, pair in path:
                    fee_product *= (Decimal('1') - pair.fee_rate)
                
                net_return = product * fee_product
                if net_return > Decimal('1') + self.min_profit_pct / 100:
                    profitable_cycles.append(path.copy())
                return
            
            if len(path) >= max_depth:
                return
            
            if current in self.pair_graph:
                for next_currency, pair in self.pair_graph[current].items():
                    if pair.exchange != exchange:
                        continue
                    
                    if not path or next_currency != path[-1][0]:
                        rate = pair.bid_price if pair.bid_volume > 0 else Decimal('0')
                        if rate > 0:
                            path.append((current, next_currency, pair))
                            dfs(next_currency, path, product * rate)
                            path.pop()
        
        dfs(start, [], Decimal('1'))
        return profitable_cycles
    
    def _create_triangular_opportunity(
        self,
        cycle: List[Tuple[str, str, ExchangePair]],
        exchange: str,
        start_currency: str
    ) -> Optional[TriangularArbitrageOpportunity]:
        """Create triangular arbitrage opportunity from cycle."""
        path = [pair for _, _, pair in cycle]
        path_description = " -> ".join([curr for curr, _, _ in cycle] + [start_currency])
        
        cumulative_fees = Decimal('0')
        rate_product = Decimal('1')
        
        for _, _, pair in cycle:
            cumulative_fees += pair.fee_rate
            rate_product *= pair.bid_price
        
        fee_product = Decimal('1')
        for _, _, pair in cycle:
            fee_product *= (Decimal('1') - pair.fee_rate)
        
        net_return = rate_product * fee_product
        profit_pct = (net_return - Decimal('1')) * 100
        
        if profit_pct < self.min_profit_pct:
            return None
        
        execution_order = []
        for i, (from_curr, to_curr, pair) in enumerate(cycle):
            execution_order.append({
                "step": i + 1,
                "action": "exchange",
                "from": from_curr,
                "to": to_curr,
                "rate": float(pair.bid_price),
                "fee": float(pair.fee_rate)
            })
        
        confidence = self._calculate_triangular_confidence(path)
        
        return TriangularArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.TRIANGULAR,
            profit_pct=profit_pct,
            profit_amount=Decimal('0'),
            confidence_score=confidence,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.opportunity_ttl),
            path=path,
            exchange=exchange,
            start_currency=start_currency,
            end_currency=start_currency,
            path_description=path_description,
            cumulative_fees=cumulative_fees * 100,
            execution_order=execution_order
        )
    
    async def find_statistical_arbitrage(
        self,
        pair_a: ExchangePair,
        pair_b: ExchangePair,
        historical_data: Optional[Dict[str, List[Tuple[datetime, Decimal]]]] = None
    ) -> Optional[StatisticalArbitrageOpportunity]:
        """Find statistical arbitrage opportunities between correlated pairs.
        
        Args:
            pair_a: First trading pair
            pair_b: Second trading pair
            historical_data: Historical price data
            
        Returns:
            Statistical arbitrage opportunity if found
        """
        if historical_data:
            self.price_history.update(historical_data)
        
        key_a = f"{pair_a.exchange}:{pair_a.symbol}"
        key_b = f"{pair_b.exchange}:{pair_b.symbol}"
        
        history_a = self.price_history.get(key_a, [])
        history_b = self.price_history.get(key_b, [])
        
        if len(history_a) < self.stat_arb_window or len(history_b) < self.stat_arb_window:
            logger.debug(
                "insufficient_historical_data",
                key_a=key_a,
                key_b=key_b,
                history_a_len=len(history_a),
                history_b_len=len(history_b),
                required=self.stat_arb_window
            )
            return None
        
        try:
            prices_a = np.array([float(price) for _, price in history_a[-self.stat_arb_window:]])
            prices_b = np.array([float(price) for _, price in history_b[-self.stat_arb_window:]])
        except (ValueError, TypeError) as e:
            logger.error("price_conversion_error", error=str(e))
            return None
        
        correlation = np.corrcoef(prices_a, prices_b)[0, 1]
        
        if abs(correlation) < 0.7:
            return None
        
        spreads = prices_a - prices_b
        mean_spread = Decimal(str(np.mean(spreads)))
        std_spread = Decimal(str(np.std(spreads)))
        
        if std_spread == 0:
            return None
        
        current_spread = pair_a.bid_price - pair_b.bid_price
        z_score = float((current_spread - mean_spread) / std_spread)
        
        if abs(z_score) < 2:
            return None
        
        mean_reversion_prob = self._calculate_mean_reversion_probability(z_score)
        
        expected_profit = abs(current_spread - mean_spread)
        total_fees = pair_a.fee_rate + pair_b.fee_rate
        net_profit = expected_profit - (current_spread * total_fees)
        
        profit_pct = (net_profit / current_spread) * 100 if current_spread != 0 else Decimal('0')
        
        if profit_pct < self.min_profit_pct:
            return None
        
        confidence = self._calculate_statistical_confidence(
            correlation,
            z_score,
            mean_reversion_prob
        )
        
        return StatisticalArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.STATISTICAL,
            profit_pct=profit_pct,
            profit_amount=net_profit,
            confidence_score=confidence,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.opportunity_ttl * 2),
            pair_a=pair_a,
            pair_b=pair_b,
            correlation=correlation,
            z_score=z_score,
            mean_spread=mean_spread,
            current_spread=current_spread,
            std_spread=std_spread,
            mean_reversion_probability=mean_reversion_prob,
            historical_success_rate=0.0,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
    
    def _calculate_mean_reversion_probability(self, z_score: float) -> float:
        """Calculate probability of mean reversion.
        
        Args:
            z_score: Z-score of current spread
            
        Returns:
            Probability between 0 and 1
        """
        return 1 - (2 * stats.norm.cdf(-abs(z_score)))
    
    def calculate_net_profit(
        self,
        opportunity: ArbitrageOpportunity,
        size: Decimal,
        include_slippage: bool = True
    ) -> Decimal:
        """Calculate net profit for an opportunity.
        
        Args:
            opportunity: Arbitrage opportunity
            size: Trade size
            include_slippage: Whether to include slippage estimate
            
        Returns:
            Net profit amount
        """
        if isinstance(opportunity, DirectArbitrageOpportunity):
            buy_cost = size * opportunity.buy_price * (Decimal('1') + opportunity.buy_fee / 100)
            sell_revenue = size * opportunity.sell_price * (Decimal('1') - opportunity.sell_fee / 100)
            
            if include_slippage:
                slippage_factor = Decimal('0.001')
                sell_revenue *= (Decimal('1') - slippage_factor)
            
            return sell_revenue - buy_cost
            
        elif isinstance(opportunity, TriangularArbitrageOpportunity):
            remaining = size
            for step in opportunity.execution_order:
                fee = Decimal(str(step['fee']))
                rate = Decimal(str(step['rate']))
                remaining = remaining * rate * (Decimal('1') - fee)
            
            if include_slippage:
                slippage_per_hop = Decimal('0.0005')
                slippage_factor = (Decimal('1') - slippage_per_hop) ** len(opportunity.path)
                remaining *= slippage_factor
            
            return remaining - size
            
        elif isinstance(opportunity, StatisticalArbitrageOpportunity):
            expected_move = abs(opportunity.current_spread - opportunity.mean_spread)
            total_fees = opportunity.pair_a.fee_rate + opportunity.pair_b.fee_rate
            
            gross_profit = size * expected_move
            fee_cost = size * opportunity.current_spread * total_fees
            
            if include_slippage:
                slippage_factor = Decimal('0.002')
                gross_profit *= (Decimal('1') - slippage_factor)
            
            return gross_profit - fee_cost
        
        return Decimal('0')
    
    def calculate_confidence(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate confidence score for an opportunity.
        
        Args:
            opportunity: Arbitrage opportunity
            
        Returns:
            Confidence score between 0 and 1
        """
        if isinstance(opportunity, DirectArbitrageOpportunity):
            return self._calculate_direct_confidence(
                None, None, opportunity.max_volume
            )
        elif isinstance(opportunity, TriangularArbitrageOpportunity):
            return self._calculate_triangular_confidence(opportunity.path)
        elif isinstance(opportunity, StatisticalArbitrageOpportunity):
            return self._calculate_statistical_confidence(
                opportunity.correlation,
                opportunity.z_score,
                opportunity.mean_reversion_probability
            )
        return 0.0
    
    def _calculate_direct_confidence(
        self,
        pair_a: Optional[ExchangePair],
        pair_b: Optional[ExchangePair],
        volume: Decimal
    ) -> float:
        """Calculate confidence for direct arbitrage."""
        base_confidence = 0.7
        
        if volume < Decimal('100'):
            volume_factor = 0.5
        elif volume < Decimal('1000'):
            volume_factor = 0.7
        else:
            volume_factor = 0.9
        
        spread_stability_factor = 0.8
        
        return min(1.0, base_confidence * volume_factor * spread_stability_factor)
    
    def _calculate_triangular_confidence(self, path: List[ExchangePair]) -> float:
        """Calculate confidence for triangular arbitrage."""
        base_confidence = 0.6
        
        path_length_penalty = 0.9 ** (len(path) - 3)
        
        min_volume = min(pair.bid_volume for pair in path)
        if min_volume < Decimal('50'):
            volume_factor = 0.4
        elif min_volume < Decimal('500'):
            volume_factor = 0.6
        else:
            volume_factor = 0.8
        
        return min(1.0, base_confidence * path_length_penalty * volume_factor)
    
    def _calculate_statistical_confidence(
        self,
        correlation: float,
        z_score: float,
        mean_reversion_prob: float
    ) -> float:
        """Calculate confidence for statistical arbitrage."""
        correlation_factor = abs(correlation)
        
        z_score_factor = min(1.0, abs(z_score) / 4.0)
        
        return min(1.0, correlation_factor * z_score_factor * mean_reversion_prob)
    
    async def update_opportunity(
        self,
        opportunity_id: str,
        market_data: Dict[str, Any]
    ) -> Optional[ArbitrageOpportunity]:
        """Update an existing opportunity with new market data.
        
        Args:
            opportunity_id: ID of opportunity to update
            market_data: New market data
            
        Returns:
            Updated opportunity or None if invalidated
        """
        async with self._lock:
            if opportunity_id not in self.opportunities:
                return None
            
            opportunity = self.opportunities[opportunity_id]
            
            if datetime.now() > opportunity.expires_at:
                opportunity.status = OpportunityStatus.EXPIRED
                del self.opportunities[opportunity_id]
                return None
            
            if isinstance(opportunity, DirectArbitrageOpportunity):
                updated = self._update_direct_opportunity(opportunity, market_data)
            elif isinstance(opportunity, TriangularArbitrageOpportunity):
                updated = self._update_triangular_opportunity(opportunity, market_data)
            elif isinstance(opportunity, StatisticalArbitrageOpportunity):
                updated = self._update_statistical_opportunity(opportunity, market_data)
            else:
                updated = opportunity
            
            if updated and updated.profit_pct >= self.min_profit_pct:
                self.opportunities[opportunity_id] = updated
                return updated
            else:
                opportunity.status = OpportunityStatus.INVALIDATED
                del self.opportunities[opportunity_id]
                return None
    
    def _update_direct_opportunity(
        self,
        opportunity: DirectArbitrageOpportunity,
        market_data: Dict[str, Any]
    ) -> Optional[DirectArbitrageOpportunity]:
        """Update direct arbitrage opportunity."""
        symbol_key = opportunity.symbol
        if symbol_key not in market_data:
            return None
        
        exchanges_data = market_data[symbol_key]
        
        buy_data = None
        sell_data = None
        
        for exchange_pair in exchanges_data:
            if exchange_pair.exchange == opportunity.buy_exchange:
                buy_data = exchange_pair
            elif exchange_pair.exchange == opportunity.sell_exchange:
                sell_data = exchange_pair
        
        if not buy_data or not sell_data:
            return None
        
        opportunity.buy_price = buy_data.ask_price
        opportunity.sell_price = sell_data.bid_price
        opportunity.max_volume = min(buy_data.ask_volume, sell_data.bid_volume)
        
        profit_before_fees = (opportunity.sell_price - opportunity.buy_price) / opportunity.buy_price * 100
        opportunity.net_profit_pct = profit_before_fees - (opportunity.buy_fee + opportunity.sell_fee)
        
        return opportunity if opportunity.net_profit_pct >= self.min_profit_pct else None
    
    def _update_triangular_opportunity(
        self,
        opportunity: TriangularArbitrageOpportunity,
        market_data: Dict[str, Any]
    ) -> Optional[TriangularArbitrageOpportunity]:
        """Update triangular arbitrage opportunity."""
        return opportunity
    
    def _update_statistical_opportunity(
        self,
        opportunity: StatisticalArbitrageOpportunity,
        market_data: Dict[str, Any]
    ) -> Optional[StatisticalArbitrageOpportunity]:
        """Update statistical arbitrage opportunity."""
        return opportunity
    
    def optimize_execution_path(
        self,
        opportunity: ArbitrageOpportunity,
        latencies: Dict[str, float]
    ) -> ExecutionPath:
        """Optimize execution path for an opportunity.
        
        Args:
            opportunity: Arbitrage opportunity
            latencies: Exchange latencies in ms
            
        Returns:
            Optimized execution path
        """
        if isinstance(opportunity, DirectArbitrageOpportunity):
            steps = [
                {
                    "action": "buy",
                    "exchange": opportunity.buy_exchange,
                    "symbol": opportunity.symbol,
                    "price": float(opportunity.buy_price),
                    "size": float(opportunity.max_volume)
                },
                {
                    "action": "sell",
                    "exchange": opportunity.sell_exchange,
                    "symbol": opportunity.symbol,
                    "price": float(opportunity.sell_price),
                    "size": float(opportunity.max_volume)
                }
            ]
            
            buy_latency = latencies.get(opportunity.buy_exchange, 50)
            sell_latency = latencies.get(opportunity.sell_exchange, 50)
            estimated_time = int(buy_latency + sell_latency)
            
        elif isinstance(opportunity, TriangularArbitrageOpportunity):
            steps = []
            for order in opportunity.execution_order:
                steps.append({
                    "action": "exchange",
                    "from": order["from"],
                    "to": order["to"],
                    "rate": order["rate"],
                    "exchange": opportunity.exchange
                })
            
            exchange_latency = latencies.get(opportunity.exchange, 50)
            estimated_time = int(exchange_latency * len(steps))
            
        else:
            steps = []
            estimated_time = 100
        
        estimated_slippage = Decimal('0.001') * len(steps)
        
        fallback_paths = []
        
        risk_score = 1.0 - opportunity.confidence_score
        
        return ExecutionPath(
            opportunity_id=opportunity.id,
            steps=steps,
            estimated_time_ms=estimated_time,
            estimated_slippage=estimated_slippage,
            fallback_paths=fallback_paths,
            risk_score=risk_score
        )
    
    def filter_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity],
        min_profit: Optional[Decimal] = None,
        min_confidence: Optional[float] = None,
        opportunity_types: Optional[List[OpportunityType]] = None
    ) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on criteria.
        
        Args:
            opportunities: List of opportunities to filter
            min_profit: Minimum profit percentage
            min_confidence: Minimum confidence score
            opportunity_types: Types to include
            
        Returns:
            Filtered list of opportunities
        """
        filtered = []
        
        min_profit = min_profit or self.min_profit_pct
        min_confidence = min_confidence or self.min_confidence
        
        for opp in opportunities:
            if opp.status != OpportunityStatus.ACTIVE:
                continue
            
            if datetime.now() > opp.expires_at:
                opp.status = OpportunityStatus.EXPIRED
                continue
            
            if opp.profit_pct < min_profit:
                continue
            
            if opp.confidence_score < min_confidence:
                continue
            
            if opportunity_types and opp.type not in opportunity_types:
                continue
            
            filtered.append(opp)
        
        if len(filtered) > self.max_opportunities:
            filtered = sorted(
                filtered,
                key=lambda x: x.profit_pct * Decimal(str(x.confidence_score)),
                reverse=True
            )[:self.max_opportunities]
        
        return filtered
    
    def rank_opportunities(
        self,
        opportunities: List[ArbitrageOpportunity],
        risk_free_rate: float = 0.02
    ) -> List[Tuple[ArbitrageOpportunity, float]]:
        """Rank opportunities by risk-adjusted returns.
        
        Args:
            opportunities: List of opportunities
            risk_free_rate: Risk-free rate for Sharpe ratio
            
        Returns:
            List of (opportunity, score) tuples sorted by score
        """
        ranked = []
        
        for opp in opportunities:
            expected_return = float(opp.profit_pct) / 100
            
            if opp.type == OpportunityType.DIRECT:
                volatility = 0.05
            elif opp.type == OpportunityType.TRIANGULAR:
                volatility = 0.08
            else:
                volatility = 0.10
            
            sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            risk_adjusted_score = sharpe_ratio * opp.confidence_score
            
            success_probability = opp.confidence_score
            expected_value = expected_return * success_probability
            
            final_score = (risk_adjusted_score * 0.6 + expected_value * 0.4)
            
            ranked.append((opp, final_score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    async def update_opportunities(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, List[ArbitrageOpportunity]]:
        """Update all opportunities with new market data.
        
        Args:
            market_data: New market data
            
        Returns:
            Dictionary of opportunity lists by type
        """
        async with self._lock:
            opportunity_ids = list(self.opportunities.keys())
            
            for opp_id in opportunity_ids:
                await self.update_opportunity(opp_id, market_data)
            
            direct_opps = await self.find_direct_arbitrage(market_data)
            
            for opp in direct_opps:
                if opp.id not in self.opportunities:
                    self.opportunities[opp.id] = opp
            
            all_opportunities = list(self.opportunities.values())
            filtered = self.filter_opportunities(all_opportunities)
            
            result = {
                "direct": [],
                "triangular": [],
                "statistical": []
            }
            
            for opp in filtered:
                if opp.type == OpportunityType.DIRECT:
                    result["direct"].append(opp)
                elif opp.type == OpportunityType.TRIANGULAR:
                    result["triangular"].append(opp)
                elif opp.type == OpportunityType.STATISTICAL:
                    result["statistical"].append(opp)
            
            logger.info(
                "opportunities_updated",
                direct=len(result["direct"]),
                triangular=len(result["triangular"]),
                statistical=len(result["statistical"])
            )
            
            return result