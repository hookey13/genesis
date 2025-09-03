"""Unit tests for the ArbitrageDetector class."""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List
import uuid

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from genesis.analytics.arbitrage_detector import ArbitrageDetector
from genesis.analytics.opportunity_models import (
    ExchangePair,
    OpportunityType,
    OpportunityStatus,
    DirectArbitrageOpportunity,
    TriangularArbitrageOpportunity,
    StatisticalArbitrageOpportunity
)


@pytest.fixture
def detector():
    """Create an ArbitrageDetector instance for testing."""
    return ArbitrageDetector(
        min_profit_pct=0.3,
        min_confidence=0.6,
        max_path_length=4,
        stat_arb_window=100,
        opportunity_ttl=5,
        max_opportunities=50
    )


@pytest.fixture
def sample_exchange_pairs():
    """Create sample exchange pairs for testing."""
    return {
        "BTC/USDT": [
            ExchangePair(
                exchange="binance",
                symbol="BTC/USDT",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10"),
                timestamp=datetime.utcnow(),
                fee_rate=Decimal("0.001")
            ),
            ExchangePair(
                exchange="coinbase",
                symbol="BTC/USDT",
                bid_price=Decimal("50100"),
                ask_price=Decimal("50110"),
                bid_volume=Decimal("8"),
                ask_volume=Decimal("8"),
                timestamp=datetime.utcnow(),
                fee_rate=Decimal("0.005")
            )
        ],
        "ETH/USDT": [
            ExchangePair(
                exchange="binance",
                symbol="ETH/USDT",
                bid_price=Decimal("3000"),
                ask_price=Decimal("3001"),
                bid_volume=Decimal("50"),
                ask_volume=Decimal("50"),
                timestamp=datetime.utcnow(),
                fee_rate=Decimal("0.001")
            ),
            ExchangePair(
                exchange="kraken",
                symbol="ETH/USDT",
                bid_price=Decimal("3010"),
                ask_price=Decimal("3011"),
                bid_volume=Decimal("40"),
                ask_volume=Decimal("40"),
                timestamp=datetime.utcnow(),
                fee_rate=Decimal("0.0026")
            )
        ]
    }


@pytest.fixture
def triangular_market_data():
    """Create market data for triangular arbitrage testing."""
    return {
        "BTC/USDT": ExchangePair(
            exchange="binance",
            symbol="BTC/USDT",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.001")
        ),
        "ETH/USDT": ExchangePair(
            exchange="binance",
            symbol="ETH/USDT",
            bid_price=Decimal("3000"),
            ask_price=Decimal("3001"),
            bid_volume=Decimal("50"),
            ask_volume=Decimal("50"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.001")
        ),
        "ETH/BTC": ExchangePair(
            exchange="binance",
            symbol="ETH/BTC",
            bid_price=Decimal("0.0601"),
            ask_price=Decimal("0.0602"),
            bid_volume=Decimal("100"),
            ask_volume=Decimal("100"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.001")
        )
    }


class TestArbitrageDetector:
    """Test cases for ArbitrageDetector."""
    
    def test_initialization(self, detector):
        """Test proper initialization of ArbitrageDetector."""
        assert detector.min_profit_pct == Decimal("0.3")
        assert detector.min_confidence == 0.6
        assert detector.max_path_length == 4
        assert detector.stat_arb_window == 100
        assert detector.opportunity_ttl == 5
        assert detector.max_opportunities == 50
        assert len(detector.opportunities) == 0
        assert len(detector.price_history) == 0
    
    @pytest.mark.asyncio
    async def test_find_direct_arbitrage(self, detector, sample_exchange_pairs):
        """Test direct arbitrage detection between exchanges."""
        opportunities = await detector.find_direct_arbitrage(sample_exchange_pairs)
        
        assert len(opportunities) > 0
        
        for opp in opportunities:
            assert isinstance(opp, DirectArbitrageOpportunity)
            assert opp.type == OpportunityType.DIRECT
            assert opp.net_profit_pct >= detector.min_profit_pct
            assert opp.confidence_score >= 0
            assert opp.confidence_score <= 1
            assert opp.buy_exchange in ["binance", "coinbase", "kraken"]
            assert opp.sell_exchange in ["binance", "coinbase", "kraken"]
            assert opp.buy_exchange != opp.sell_exchange
    
    @pytest.mark.asyncio
    async def test_find_direct_arbitrage_no_opportunity(self, detector):
        """Test direct arbitrage with no profitable opportunities."""
        market_data = {
            "BTC/USDT": [
                ExchangePair(
                    exchange="binance",
                    symbol="BTC/USDT",
                    bid_price=Decimal("50000"),
                    ask_price=Decimal("50010"),
                    bid_volume=Decimal("10"),
                    ask_volume=Decimal("10"),
                    timestamp=datetime.utcnow(),
                    fee_rate=Decimal("0.001")
                ),
                ExchangePair(
                    exchange="coinbase",
                    symbol="BTC/USDT",
                    bid_price=Decimal("50005"),
                    ask_price=Decimal("50015"),
                    bid_volume=Decimal("8"),
                    ask_volume=Decimal("8"),
                    timestamp=datetime.utcnow(),
                    fee_rate=Decimal("0.005")
                )
            ]
        }
        
        opportunities = await detector.find_direct_arbitrage(market_data)
        assert len(opportunities) == 0
    
    @pytest.mark.asyncio
    async def test_find_triangular_arbitrage(self, detector, triangular_market_data):
        """Test triangular arbitrage detection."""
        opportunities = await detector.find_triangular_arbitrage(
            "binance",
            triangular_market_data
        )
        
        for opp in opportunities:
            assert isinstance(opp, TriangularArbitrageOpportunity)
            assert opp.type == OpportunityType.TRIANGULAR
            assert opp.exchange == "binance"
            assert len(opp.path) >= 3
            assert len(opp.path) <= detector.max_path_length
            assert opp.start_currency == opp.end_currency
    
    @pytest.mark.asyncio
    async def test_find_statistical_arbitrage(self, detector):
        """Test statistical arbitrage detection."""
        pair_a = ExchangePair(
            exchange="binance",
            symbol="BTC/USDT",
            bid_price=Decimal("50000"),
            ask_price=Decimal("50010"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("10"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.001")
        )
        
        pair_b = ExchangePair(
            exchange="coinbase",
            symbol="BTC/USDT",
            bid_price=Decimal("49800"),
            ask_price=Decimal("49810"),
            bid_volume=Decimal("8"),
            ask_volume=Decimal("8"),
            timestamp=datetime.utcnow(),
            fee_rate=Decimal("0.005")
        )
        
        # Create correlated historical data
        historical_data = {}
        prices_a = []
        prices_b = []
        base_price = 50000
        
        for i in range(100):
            noise_a = np.random.normal(0, 10)
            noise_b = np.random.normal(0, 10)
            price_a = base_price + noise_a
            price_b = base_price + noise_b * 0.9  # Correlated
            
            timestamp = datetime.utcnow() - timedelta(minutes=100-i)
            prices_a.append((timestamp, Decimal(str(price_a))))
            prices_b.append((timestamp, Decimal(str(price_b))))
        
        historical_data["binance:BTC/USDT"] = prices_a
        historical_data["coinbase:BTC/USDT"] = prices_b
        
        opportunity = await detector.find_statistical_arbitrage(
            pair_a,
            pair_b,
            historical_data
        )
        
        if opportunity:
            assert isinstance(opportunity, StatisticalArbitrageOpportunity)
            assert opportunity.type == OpportunityType.STATISTICAL
            assert abs(opportunity.correlation) >= 0.7
            assert abs(opportunity.z_score) >= 2.0
    
    def test_calculate_net_profit_direct(self, detector):
        """Test net profit calculation for direct arbitrage."""
        opportunity = DirectArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("1.0"),
            profit_amount=Decimal("100"),
            confidence_score=0.8,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=5),
            buy_exchange="binance",
            sell_exchange="coinbase",
            symbol="BTC/USDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50500"),
            max_volume=Decimal("1"),
            buy_fee=Decimal("0.1"),
            sell_fee=Decimal("0.5"),
            net_profit_pct=Decimal("0.4")
        )
        
        size = Decimal("1")
        profit = detector.calculate_net_profit(opportunity, size, include_slippage=False)
        
        buy_cost = size * Decimal("50000") * Decimal("1.001")
        sell_revenue = size * Decimal("50500") * Decimal("0.995")
        expected_profit = sell_revenue - buy_cost
        
        assert abs(profit - expected_profit) < Decimal("0.01")
    
    def test_calculate_net_profit_triangular(self, detector):
        """Test net profit calculation for triangular arbitrage."""
        path = [
            ExchangePair(
                exchange="binance",
                symbol="BTC/USDT",
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("10"),
                ask_volume=Decimal("10"),
                timestamp=datetime.utcnow(),
                fee_rate=Decimal("0.001")
            )
        ] * 3
        
        opportunity = TriangularArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.TRIANGULAR,
            profit_pct=Decimal("0.5"),
            profit_amount=Decimal("50"),
            confidence_score=0.7,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=5),
            path=path,
            exchange="binance",
            start_currency="USDT",
            end_currency="USDT",
            path_description="USDT -> BTC -> ETH -> USDT",
            cumulative_fees=Decimal("0.3"),
            execution_order=[
                {"step": 1, "action": "exchange", "from": "USDT", "to": "BTC", "rate": 50000, "fee": 0.001},
                {"step": 2, "action": "exchange", "from": "BTC", "to": "ETH", "rate": 0.06, "fee": 0.001},
                {"step": 3, "action": "exchange", "from": "ETH", "to": "USDT", "rate": 3000, "fee": 0.001}
            ]
        )
        
        size = Decimal("1000")
        profit = detector.calculate_net_profit(opportunity, size, include_slippage=False)
        
        assert isinstance(profit, Decimal)
    
    def test_calculate_confidence_scores(self, detector):
        """Test confidence score calculation for different opportunity types."""
        # Direct arbitrage confidence
        direct_conf = detector._calculate_direct_confidence(None, None, Decimal("1000"))
        assert 0 <= direct_conf <= 1
        
        # Triangular arbitrage confidence
        path = [Mock(bid_volume=Decimal("100"))] * 3
        triangular_conf = detector._calculate_triangular_confidence(path)
        assert 0 <= triangular_conf <= 1
        
        # Statistical arbitrage confidence
        stat_conf = detector._calculate_statistical_confidence(0.9, 3.0, 0.95)
        assert 0 <= stat_conf <= 1
    
    @pytest.mark.asyncio
    async def test_update_opportunity(self, detector):
        """Test opportunity update with new market data."""
        opportunity = DirectArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.5"),
            profit_amount=Decimal("50"),
            confidence_score=0.7,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=10),
            buy_exchange="binance",
            sell_exchange="coinbase",
            symbol="BTC/USDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50300"),
            max_volume=Decimal("1"),
            buy_fee=Decimal("0.1"),
            sell_fee=Decimal("0.5"),
            net_profit_pct=Decimal("0.4")
        )
        
        detector.opportunities[opportunity.id] = opportunity
        
        new_market_data = {
            "BTC/USDT": [
                ExchangePair(
                    exchange="binance",
                    symbol="BTC/USDT",
                    bid_price=Decimal("50100"),
                    ask_price=Decimal("50110"),
                    bid_volume=Decimal("10"),
                    ask_volume=Decimal("10"),
                    timestamp=datetime.utcnow(),
                    fee_rate=Decimal("0.001")
                ),
                ExchangePair(
                    exchange="coinbase",
                    symbol="BTC/USDT",
                    bid_price=Decimal("50400"),
                    ask_price=Decimal("50410"),
                    bid_volume=Decimal("8"),
                    ask_volume=Decimal("8"),
                    timestamp=datetime.utcnow(),
                    fee_rate=Decimal("0.005")
                )
            ]
        }
        
        updated_opp = await detector.update_opportunity(opportunity.id, new_market_data)
        
        assert updated_opp is not None
        assert updated_opp.buy_price == Decimal("50110")
        assert updated_opp.sell_price == Decimal("50400")
    
    @pytest.mark.asyncio
    async def test_update_opportunity_expired(self, detector):
        """Test that expired opportunities are removed."""
        opportunity = DirectArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.5"),
            profit_amount=Decimal("50"),
            confidence_score=0.7,
            created_at=datetime.utcnow() - timedelta(seconds=10),
            expires_at=datetime.utcnow() - timedelta(seconds=1),  # Already expired
            buy_exchange="binance",
            sell_exchange="coinbase",
            symbol="BTC/USDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50300"),
            max_volume=Decimal("1"),
            buy_fee=Decimal("0.1"),
            sell_fee=Decimal("0.5"),
            net_profit_pct=Decimal("0.4")
        )
        
        detector.opportunities[opportunity.id] = opportunity
        
        updated_opp = await detector.update_opportunity(opportunity.id, {})
        
        assert updated_opp is None
        assert opportunity.id not in detector.opportunities
    
    def test_optimize_execution_path_direct(self, detector):
        """Test execution path optimization for direct arbitrage."""
        opportunity = DirectArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.5"),
            profit_amount=Decimal("50"),
            confidence_score=0.7,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=5),
            buy_exchange="binance",
            sell_exchange="coinbase",
            symbol="BTC/USDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50300"),
            max_volume=Decimal("1"),
            buy_fee=Decimal("0.1"),
            sell_fee=Decimal("0.5"),
            net_profit_pct=Decimal("0.4")
        )
        
        latencies = {"binance": 30, "coinbase": 40}
        path = detector.optimize_execution_path(opportunity, latencies)
        
        assert path.opportunity_id == opportunity.id
        assert len(path.steps) == 2
        assert path.steps[0]["action"] == "buy"
        assert path.steps[1]["action"] == "sell"
        assert path.estimated_time_ms == 70
        assert path.risk_score == 1.0 - opportunity.confidence_score
    
    def test_filter_opportunities(self, detector):
        """Test opportunity filtering."""
        opportunities = []
        
        # Create mix of opportunities
        for i in range(10):
            profit = Decimal(str(0.1 + i * 0.1))
            confidence = 0.5 + i * 0.05
            
            opp = DirectArbitrageOpportunity(
                id=str(uuid.uuid4()),
                type=OpportunityType.DIRECT,
                profit_pct=profit,
                profit_amount=profit * 100,
                confidence_score=confidence,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=10),
                buy_exchange="binance",
                sell_exchange="coinbase",
                symbol="BTC/USDT",
                buy_price=Decimal("50000"),
                sell_price=Decimal("50000") + profit * 1000,
                max_volume=Decimal("1"),
                buy_fee=Decimal("0.1"),
                sell_fee=Decimal("0.5"),
                net_profit_pct=profit
            )
            opportunities.append(opp)
        
        filtered = detector.filter_opportunities(
            opportunities,
            min_profit=Decimal("0.3"),
            min_confidence=0.6
        )
        
        assert all(opp.profit_pct >= Decimal("0.3") for opp in filtered)
        assert all(opp.confidence_score >= 0.6 for opp in filtered)
    
    def test_filter_opportunities_max_limit(self, detector):
        """Test that filtering respects max_opportunities limit."""
        detector.max_opportunities = 5
        
        opportunities = []
        for i in range(20):
            opp = DirectArbitrageOpportunity(
                id=str(uuid.uuid4()),
                type=OpportunityType.DIRECT,
                profit_pct=Decimal(str(1.0 + i * 0.1)),
                profit_amount=Decimal("100"),
                confidence_score=0.8,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=10),
                buy_exchange="binance",
                sell_exchange="coinbase",
                symbol="BTC/USDT",
                buy_price=Decimal("50000"),
                sell_price=Decimal("50500"),
                max_volume=Decimal("1"),
                buy_fee=Decimal("0.1"),
                sell_fee=Decimal("0.5"),
                net_profit_pct=Decimal("0.4")
            )
            opportunities.append(opp)
        
        filtered = detector.filter_opportunities(opportunities)
        
        assert len(filtered) == 5
        # Should keep the highest profit opportunities
        assert filtered[0].profit_pct >= filtered[-1].profit_pct
    
    def test_rank_opportunities(self, detector):
        """Test opportunity ranking by risk-adjusted returns."""
        opportunities = []
        
        # Create opportunities with different profiles
        for i in range(5):
            opp = DirectArbitrageOpportunity(
                id=str(uuid.uuid4()),
                type=OpportunityType.DIRECT,
                profit_pct=Decimal(str(0.5 + i * 0.2)),
                profit_amount=Decimal("100"),
                confidence_score=0.9 - i * 0.1,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=10),
                buy_exchange="binance",
                sell_exchange="coinbase",
                symbol="BTC/USDT",
                buy_price=Decimal("50000"),
                sell_price=Decimal("50500"),
                max_volume=Decimal("1"),
                buy_fee=Decimal("0.1"),
                sell_fee=Decimal("0.5"),
                net_profit_pct=Decimal("0.4")
            )
            opportunities.append(opp)
        
        ranked = detector.rank_opportunities(opportunities)
        
        assert len(ranked) == 5
        assert all(isinstance(score, float) for _, score in ranked)
        # Check that ranking produces scores in descending order
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_update_opportunities_comprehensive(self, detector, sample_exchange_pairs):
        """Test comprehensive opportunity update with new market data."""
        # Add some existing opportunities
        opp1 = DirectArbitrageOpportunity(
            id=str(uuid.uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.5"),
            profit_amount=Decimal("50"),
            confidence_score=0.7,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(seconds=10),
            buy_exchange="binance",
            sell_exchange="coinbase",
            symbol="BTC/USDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50300"),
            max_volume=Decimal("1"),
            buy_fee=Decimal("0.1"),
            sell_fee=Decimal("0.5"),
            net_profit_pct=Decimal("0.4")
        )
        detector.opportunities[opp1.id] = opp1
        
        result = await detector.update_opportunities(sample_exchange_pairs)
        
        assert "direct" in result
        assert "triangular" in result
        assert "statistical" in result
        assert isinstance(result["direct"], list)
        assert isinstance(result["triangular"], list)
        assert isinstance(result["statistical"], list)
    
    def test_mean_reversion_probability(self, detector):
        """Test mean reversion probability calculation."""
        z_score = 2.5
        prob = detector._calculate_mean_reversion_probability(z_score)
        
        assert 0 <= prob <= 1
        # Higher z-score should give higher probability
        prob2 = detector._calculate_mean_reversion_probability(3.0)
        assert prob2 > prob
    
    def test_build_pair_graph(self, detector, triangular_market_data):
        """Test building the pair graph for triangular arbitrage."""
        detector._build_pair_graph(triangular_market_data)
        
        assert len(detector.pair_graph) > 0
        assert "BTC" in detector.pair_graph
        assert "USDT" in detector.pair_graph
        assert "ETH" in detector.pair_graph
    
    def test_extract_currencies(self, detector, triangular_market_data):
        """Test currency extraction from market data."""
        currencies = detector._extract_currencies(triangular_market_data)
        
        assert "BTC" in currencies
        assert "USDT" in currencies
        assert "ETH" in currencies
        assert len(currencies) == 3