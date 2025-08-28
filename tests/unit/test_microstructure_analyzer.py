"""Unit tests for Microstructure Analyzer."""

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from genesis.analytics.microstructure_analyzer import (
    ExecutionOptimizer,
    MarketMakerAnalyzer,
    MarketRegime,
    MicrostructureAnalyzer,
    ToxicityScorer,
)


class TestMicrostructureAnalyzer:
    """Test suite for MicrostructureAnalyzer."""

    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def analyzer(self, event_bus):
        """Create MicrostructureAnalyzer instance."""
        return MicrostructureAnalyzer(event_bus)

    @pytest.mark.asyncio
    async def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.event_bus is not None
        assert analyzer.execution_optimizer is not None
        assert analyzer.market_maker_analyzer is not None
        assert analyzer.toxicity_scorer is not None
        assert analyzer.current_regime == MarketRegime.NORMAL

    @pytest.mark.asyncio
    async def test_analyze_microstructure(self, analyzer, event_bus):
        """Test comprehensive microstructure analysis."""
        # Mock component results
        analyzer.execution_optimizer.analyze_timing = AsyncMock(return_value={
            "optimal_time": "09:30",
            "liquidity_score": Decimal("0.8")
        })
        analyzer.market_maker_analyzer.analyze_behavior = AsyncMock(return_value={
            "mm_present": True,
            "spread_control": Decimal("0.7")
        })
        analyzer.toxicity_scorer.calculate_toxicity = AsyncMock(return_value={
            "toxicity_score": Decimal("0.3"),
            "adverse_selection": Decimal("0.2")
        })

        result = await analyzer.analyze_microstructure(
            symbol="BTC/USDT",
            order_flow_metrics={"ofi": Decimal("0.6")},
            large_trades=[],
            manipulation_events=[]
        )

        assert "regime" in result
        assert "execution_analysis" in result
        assert "market_maker_analysis" in result
        assert "toxicity_analysis" in result
        assert "recommendations" in result

        # Check event was published
        event_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_detect_regime_normal(self, analyzer):
        """Test normal regime detection."""
        regime = await analyzer.detect_regime(
            flow_imbalance=Decimal("0.1"),
            volatility=Decimal("0.02"),
            manipulation_count=0,
            whale_activity=False
        )

        assert regime["regime"] == MarketRegime.NORMAL
        assert regime["confidence"] >= Decimal("0.7")

    @pytest.mark.asyncio
    async def test_detect_regime_stressed(self, analyzer):
        """Test stressed regime detection."""
        regime = await analyzer.detect_regime(
            flow_imbalance=Decimal("0.8"),
            volatility=Decimal("0.05"),
            manipulation_count=2,
            whale_activity=True
        )

        assert regime["regime"] == MarketRegime.STRESSED
        assert regime["transition_probability"] is not None

    @pytest.mark.asyncio
    async def test_detect_regime_trending(self, analyzer):
        """Test trending regime detection."""
        # Setup trending conditions
        analyzer.flow_history = [Decimal("0.6")] * 10  # Consistent positive flow

        regime = await analyzer.detect_regime(
            flow_imbalance=Decimal("0.6"),
            volatility=Decimal("0.03"),
            manipulation_count=0,
            whale_activity=False
        )

        assert regime["regime"] in [MarketRegime.TRENDING, MarketRegime.NORMAL]

    @pytest.mark.asyncio
    async def test_detect_regime_toxic(self, analyzer):
        """Test toxic regime detection."""
        regime = await analyzer.detect_regime(
            flow_imbalance=Decimal("0.9"),
            volatility=Decimal("0.1"),
            manipulation_count=5,
            whale_activity=True
        )

        assert regime["regime"] == MarketRegime.TOXIC
        assert regime["confidence"] >= Decimal("0.5")

    @pytest.mark.asyncio
    async def test_integrate_metrics(self, analyzer):
        """Test metric integration."""
        integrated = await analyzer.integrate_metrics(
            order_flow={"ofi": Decimal("0.5"), "pressure": Decimal("0.3")},
            large_trader={"whale_detected": True, "cluster_count": 2},
            manipulation={"spoofing_detected": False, "layering_score": Decimal("0.1")},
            price_impact={"total_impact": Decimal("0.02")}
        )

        assert "flow_strength" in integrated
        assert "institutional_presence" in integrated
        assert "manipulation_risk" in integrated
        assert "execution_difficulty" in integrated

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, analyzer):
        """Test recommendation generation."""
        # Normal conditions
        recommendations = await analyzer.generate_recommendations(
            regime=MarketRegime.NORMAL,
            metrics={
                "flow_strength": Decimal("0.3"),
                "institutional_presence": Decimal("0.2"),
                "manipulation_risk": Decimal("0.1"),
                "toxicity_score": Decimal("0.2")
            }
        )

        assert "execution_strategy" in recommendations
        assert "position_sizing" in recommendations
        assert "risk_controls" in recommendations
        assert len(recommendations["warnings"]) == 0

    @pytest.mark.asyncio
    async def test_generate_recommendations_toxic(self, analyzer):
        """Test recommendations in toxic conditions."""
        recommendations = await analyzer.generate_recommendations(
            regime=MarketRegime.TOXIC,
            metrics={
                "flow_strength": Decimal("0.9"),
                "institutional_presence": Decimal("0.8"),
                "manipulation_risk": Decimal("0.7"),
                "toxicity_score": Decimal("0.9")
            }
        )

        assert recommendations["execution_strategy"] == "AVOID"
        assert recommendations["position_sizing"] == "ZERO"
        assert len(recommendations["warnings"]) > 0
        assert "TOXIC" in recommendations["warnings"][0]


class TestExecutionOptimizer:
    """Test suite for ExecutionOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create ExecutionOptimizer instance."""
        return ExecutionOptimizer()

    @pytest.mark.asyncio
    async def test_analyze_timing(self, optimizer):
        """Test timing analysis."""
        analysis = await optimizer.analyze_timing(
            symbol="BTC/USDT",
            historical_volumes=[
                {"hour": 9, "volume": Decimal("1000")},
                {"hour": 10, "volume": Decimal("1500")},
                {"hour": 14, "volume": Decimal("2000")},
                {"hour": 15, "volume": Decimal("1800")}
            ],
            current_spread=Decimal("10"),
            volatility=Decimal("0.02")
        )

        assert "optimal_hours" in analysis
        assert "liquidity_pattern" in analysis
        assert "participation_rate" in analysis
        assert len(analysis["optimal_hours"]) > 0

    @pytest.mark.asyncio
    async def test_calculate_participation_rate(self, optimizer):
        """Test participation rate calculation."""
        rate = await optimizer.calculate_participation_rate(
            order_size=Decimal("100"),
            avg_volume=Decimal("10000"),
            urgency="NORMAL",
            market_impact_tolerance=Decimal("0.01")
        )

        assert rate > 0
        assert rate <= 0.1  # Should not exceed 10% for normal orders

    @pytest.mark.asyncio
    async def test_create_execution_schedule(self, optimizer):
        """Test execution schedule creation."""
        schedule = await optimizer.create_execution_schedule(
            total_size=Decimal("1000"),
            time_horizon=3600,  # 1 hour
            participation_rate=Decimal("0.05"),
            volume_profile=[Decimal("100")] * 12  # 5-minute bins
        )

        assert len(schedule) > 0
        assert all("time_offset" in s for s in schedule)
        assert all("size" in s for s in schedule)
        assert all("target_participation" in s for s in schedule)

        # Total should match
        total_scheduled = sum(s["size"] for s in schedule)
        assert abs(total_scheduled - Decimal("1000")) < Decimal("1")


class TestMarketMakerAnalyzer:
    """Test suite for MarketMakerAnalyzer."""

    @pytest.fixture
    def mm_analyzer(self):
        """Create MarketMakerAnalyzer instance."""
        return MarketMakerAnalyzer()

    @pytest.mark.asyncio
    async def test_analyze_behavior(self, mm_analyzer):
        """Test market maker behavior analysis."""
        order_book = {
            "bids": [
                {"price": Decimal("49990"), "quantity": Decimal("100")},
                {"price": Decimal("49980"), "quantity": Decimal("100")},
            ],
            "asks": [
                {"price": Decimal("50010"), "quantity": Decimal("100")},
                {"price": Decimal("50020"), "quantity": Decimal("100")},
            ]
        }

        analysis = await mm_analyzer.analyze_behavior(
            symbol="BTC/USDT",
            order_book_history=[order_book] * 5,
            trade_history=[]
        )

        assert "mm_present" in analysis
        assert "quote_stability" in analysis
        assert "spread_control" in analysis
        assert "inventory_pattern" in analysis

    @pytest.mark.asyncio
    async def test_detect_mm_patterns(self, mm_analyzer):
        """Test market maker pattern detection."""
        # Symmetric quotes suggest MM presence
        patterns = await mm_analyzer.detect_patterns(
            bid_sizes=[Decimal("100"), Decimal("100"), Decimal("100")],
            ask_sizes=[Decimal("100"), Decimal("100"), Decimal("100")],
            spread_history=[Decimal("20"), Decimal("20"), Decimal("20")]
        )

        assert patterns["symmetric_quotes"] == True
        assert patterns["stable_spread"] == True
        assert patterns["mm_confidence"] >= Decimal("0.6")

    @pytest.mark.asyncio
    async def test_predict_liquidity_withdrawal(self, mm_analyzer):
        """Test liquidity withdrawal prediction."""
        prediction = await mm_analyzer.predict_liquidity_withdrawal(
            volatility_spike=Decimal("0.1"),
            order_cancellation_rate=Decimal("0.8"),
            spread_widening=Decimal("2.0")
        )

        assert prediction["withdrawal_probability"] > Decimal("0.7")
        assert prediction["expected_impact"] == "HIGH"
        assert "recommended_action" in prediction


class TestToxicityScorer:
    """Test suite for ToxicityScorer."""

    @pytest.fixture
    def toxicity_scorer(self):
        """Create ToxicityScorer instance."""
        return ToxicityScorer()

    @pytest.mark.asyncio
    async def test_calculate_toxicity(self, toxicity_scorer):
        """Test toxicity calculation."""
        toxicity = await toxicity_scorer.calculate_toxicity(
            symbol="BTC/USDT",
            adverse_selection=Decimal("0.3"),
            manipulation_frequency=Decimal("0.1"),
            spread_volatility=Decimal("0.2"),
            order_rejection_rate=Decimal("0.05")
        )

        assert "toxicity_score" in toxicity
        assert "components" in toxicity
        assert "trading_recommendation" in toxicity
        assert 0 <= toxicity["toxicity_score"] <= 100

    @pytest.mark.asyncio
    async def test_calculate_pin(self, toxicity_scorer):
        """Test PIN (Probability of Informed Trading) calculation."""
        trades = [
            {"size": Decimal("10"), "side": "BUY"},
            {"size": Decimal("5"), "side": "SELL"},
            {"size": Decimal("15"), "side": "BUY"},
            {"size": Decimal("3"), "side": "SELL"},
        ]

        pin = await toxicity_scorer.calculate_pin(
            trades=trades,
            time_window=300  # 5 minutes
        )

        assert 0 <= pin <= 1
        assert isinstance(pin, Decimal)

    @pytest.mark.asyncio
    async def test_measure_adverse_selection(self, toxicity_scorer):
        """Test adverse selection measurement."""
        executions = [
            {
                "price": Decimal("50000"),
                "post_trade_price": Decimal("50050"),
                "side": "BUY"
            },
            {
                "price": Decimal("50100"),
                "post_trade_price": Decimal("50080"),
                "side": "SELL"
            }
        ]

        adverse_selection = await toxicity_scorer.measure_adverse_selection(
            executions=executions,
            time_horizon=60  # 1 minute
        )

        assert "realized_spread" in adverse_selection
        assert "effective_spread" in adverse_selection
        assert "adverse_selection_component" in adverse_selection

    @pytest.mark.asyncio
    async def test_toxicity_classification(self, toxicity_scorer):
        """Test toxicity level classification."""
        # Low toxicity
        result = await toxicity_scorer.calculate_toxicity(
            symbol="SAFE/USDT",
            adverse_selection=Decimal("0.05"),
            manipulation_frequency=Decimal("0.01"),
            spread_volatility=Decimal("0.1"),
            order_rejection_rate=Decimal("0.01")
        )
        assert result["trading_recommendation"] == "SAFE"

        # High toxicity
        result = await toxicity_scorer.calculate_toxicity(
            symbol="TOXIC/USDT",
            adverse_selection=Decimal("0.8"),
            manipulation_frequency=Decimal("0.5"),
            spread_volatility=Decimal("0.6"),
            order_rejection_rate=Decimal("0.3")
        )
        assert result["trading_recommendation"] == "AVOID"
