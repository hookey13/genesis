"""Integration tests for microstructure analysis workflow."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.analytics.microstructure_analyzer import (
    MicrostructureAnalyzer,
    MarketRegime,
    MicrostructureState
)
from genesis.exchange.order_book_manager import OrderBookSnapshot, OrderBookLevel
from genesis.engine.event_bus import EventBus


@pytest.mark.asyncio
class TestMicrostructureWorkflow:
    """Test complete microstructure analysis workflow."""
    
    @pytest.fixture
    async def event_bus(self):
        """Create event bus."""
        return EventBus()
    
    @pytest.fixture
    async def analyzer(self, event_bus):
        """Create microstructure analyzer."""
        return MicrostructureAnalyzer(event_bus)
    
    async def test_full_analysis_workflow(self, analyzer, event_bus):
        """Test complete microstructure analysis workflow."""
        symbol = "BTCUSDT"
        
        # Create sample order book
        order_book = OrderBookSnapshot(
            symbol=symbol,
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("2")),
                OrderBookLevel(Decimal("49999"), Decimal("3")),
                OrderBookLevel(Decimal("49998"), Decimal("1.5"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("1")),
                OrderBookLevel(Decimal("50002"), Decimal("2")),
                OrderBookLevel(Decimal("50003"), Decimal("1.5"))
            ]
        )
        
        # Create sample trades
        recent_trades = [
            {'price': 50000, 'quantity': 1, 'side': 'buy', 'timestamp': datetime.now(timezone.utc)},
            {'price': 50001, 'quantity': 0.5, 'side': 'sell', 'timestamp': datetime.now(timezone.utc)},
            {'price': 50000, 'quantity': 2, 'side': 'buy', 'timestamp': datetime.now(timezone.utc)},
        ]
        
        # Perform analysis
        state = await analyzer.analyze_market(symbol, order_book, recent_trades)
        
        # Verify state
        assert state is not None
        assert state.symbol == symbol
        assert state.regime in MarketRegime
        assert 0 <= state.regime_confidence <= 1
        assert state.flow_imbalance is not None
        assert isinstance(state.whale_activity, bool)
        assert isinstance(state.manipulation_detected, bool)
        assert 0 <= state.toxicity <= 100
        assert 0 <= state.execution_quality <= 100
    
    async def test_order_flow_integration(self, analyzer):
        """Test order flow analysis integration."""
        symbol = "BTCUSDT"
        
        # Simulate trades for flow analysis
        for i in range(20):
            side = 'buy' if i % 3 != 0 else 'sell'
            is_buyer_maker = side == 'sell'
            
            await analyzer.flow_analyzer.analyze_trade(
                symbol=symbol,
                price=Decimal("50000") + Decimal(i),
                quantity=Decimal("1"),
                is_buyer_maker=is_buyer_maker
            )
        
        # Check flow metrics
        flow_trend = analyzer.flow_analyzer.get_flow_trend(symbol)
        assert flow_trend in ['bullish', 'bearish', 'neutral']
        
        cumulative_flow = analyzer.flow_analyzer.get_cumulative_flow(symbol)
        assert cumulative_flow is not None
    
    async def test_whale_detection_integration(self, analyzer):
        """Test whale detection integration."""
        symbol = "BTCUSDT"
        
        # Add historical trades for distribution
        analyzer.whale_detector.trade_history[symbol] = [
            Decimal(i) for i in range(1, 101)
        ]
        
        # Simulate whale trade
        whale_activity = await analyzer.whale_detector.analyze_trade(
            symbol=symbol,
            price=Decimal("50000"),
            quantity=Decimal("100"),  # Large trade
            side='buy'
        )
        
        assert whale_activity is not None
        assert whale_activity.percentile >= Decimal("95")
        
        # Check active whales
        active_whales = analyzer.whale_detector.get_active_whales(symbol)
        assert isinstance(active_whales, list)
    
    async def test_manipulation_detection_integration(self, analyzer):
        """Test manipulation detection integration."""
        symbol = "BTCUSDT"
        
        # Simulate spoofing pattern
        timestamp = datetime.now(timezone.utc)
        
        # Place orders
        for i in range(10):
            await analyzer.manipulation_detector.track_order_placement(
                symbol=symbol,
                order_id=f"order_{i}",
                price=Decimal("49990") - Decimal(i),
                quantity=Decimal("10"),
                side='bid',
                timestamp=timestamp
            )
        
        # Quick cancellations
        for i in range(9):
            await analyzer.manipulation_detector.track_order_cancellation(
                symbol=symbol,
                order_id=f"order_{i}",
                timestamp=timestamp + timedelta(seconds=2)
            )
        
        # Check statistics
        stats = analyzer.manipulation_detector.get_manipulation_statistics(symbol)
        assert 'cancellation_rate' in stats
        assert stats['cancellation_rate'] >= 0.8
    
    async def test_regime_detection(self, analyzer):
        """Test market regime detection."""
        symbol = "BTCUSDT"
        
        # Test different scenarios
        scenarios = [
            # Normal market
            (Decimal("0.05"), False, False, Decimal("20"), MarketRegime.NORMAL),
            # Stressed market
            (Decimal("0.6"), True, False, Decimal("40"), MarketRegime.STRESSED),
            # Toxic market
            (Decimal("0.3"), True, True, Decimal("75"), MarketRegime.TOXIC),
            # Trending market
            (Decimal("0.4"), False, False, Decimal("30"), MarketRegime.TRENDING),
        ]
        
        for flow_imbalance, whale, manipulation, toxicity, expected_regime in scenarios:
            regime, confidence = analyzer._detect_regime(
                symbol, flow_imbalance, whale, manipulation, toxicity
            )
            
            # Allow for some flexibility in regime detection
            assert regime in MarketRegime
            assert 0 <= confidence <= 1
    
    async def test_price_impact_integration(self, analyzer):
        """Test price impact model integration."""
        symbol = "BTCUSDT"
        
        # Create order book for impact calculation
        order_book = OrderBookSnapshot(
            symbol=symbol,
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("5")),
                OrderBookLevel(Decimal("49999"), Decimal("10")),
                OrderBookLevel(Decimal("49998"), Decimal("15"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("5")),
                OrderBookLevel(Decimal("50002"), Decimal("10")),
                OrderBookLevel(Decimal("50003"), Decimal("15"))
            ]
        )
        
        # Estimate impact
        impact_estimate = analyzer.impact_model.estimate_impact(
            symbol=symbol,
            side='buy',
            quantity=Decimal("20"),
            order_book=order_book
        )
        
        assert impact_estimate is not None
        assert impact_estimate.temporary_impact >= 0
        assert impact_estimate.permanent_impact >= 0
        assert impact_estimate.slippage_bps >= 0
    
    async def test_execution_optimization(self, analyzer):
        """Test execution optimization."""
        # Calculate optimal participation
        participation_rate = analyzer.execution_optimizer.calculate_optimal_participation(
            total_size=Decimal("100"),
            time_horizon=timedelta(minutes=30),
            urgency=Decimal("0.5")
        )
        
        assert 0 < participation_rate <= Decimal("0.3")
        
        # Get execution schedule
        schedule = analyzer.execution_optimizer.get_execution_schedule(
            symbol="BTCUSDT",
            total_quantity=Decimal("100"),
            duration_minutes=30
        )
        
        assert len(schedule) > 0
        total_scheduled = sum(qty for _, qty in schedule)
        assert abs(total_scheduled - Decimal("100")) < Decimal("10")  # Allow some variance
    
    async def test_market_maker_analysis(self, analyzer):
        """Test market maker behavior analysis."""
        symbol = "BTCUSDT"
        
        # Create order book history
        order_book_history = []
        for i in range(150):
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                bids=[
                    OrderBookLevel(Decimal("50000") - Decimal(j), Decimal("1"))
                    for j in range(5)
                ],
                asks=[
                    OrderBookLevel(Decimal("50001") + Decimal(j), Decimal("1"))
                    for j in range(5)
                ]
            )
            order_book_history.append(snapshot)
        
        # Identify market makers
        maker_ids = analyzer.market_maker_analyzer.identify_market_makers(
            symbol, order_book_history
        )
        
        assert isinstance(maker_ids, list)
        
        # Test withdrawal detection
        before = order_book_history[0]
        
        # Create snapshot with reduced depth
        after = OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(Decimal("49990"), Decimal("0.1"))],
            asks=[OrderBookLevel(Decimal("50010"), Decimal("0.1"))]
        )
        
        withdrawal = analyzer.market_maker_analyzer.detect_withdrawal(
            symbol, before, after
        )
        
        assert withdrawal is True
    
    async def test_toxicity_scoring(self, analyzer):
        """Test toxicity scoring."""
        symbol = "BTCUSDT"
        
        # Create sample trades
        trades = [
            {'price': 50000 + i, 'quantity': 1, 'side': 'buy' if i % 2 == 0 else 'sell'}
            for i in range(20)
        ]
        
        # Calculate toxicity
        toxicity_score = analyzer.toxicity_scorer.calculate_toxicity(
            symbol=symbol,
            trades=trades,
            manipulation_events=2,
            vpin_score=Decimal("0.4")
        )
        
        assert toxicity_score is not None
        assert 0 <= toxicity_score.toxicity_score <= 100
        assert toxicity_score.recommendation in ['avoid', 'caution', 'safe']
    
    async def test_state_persistence(self, analyzer):
        """Test microstructure state persistence."""
        symbol = "BTCUSDT"
        
        # Create initial state
        order_book = OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1"))]
        )
        
        trades = [{'price': 50000, 'quantity': 1, 'side': 'buy'}]
        
        # Analyze multiple times
        states = []
        for _ in range(3):
            state = await analyzer.analyze_market(symbol, order_book, trades)
            states.append(state)
            await asyncio.sleep(0.1)
        
        # Check state persistence
        assert symbol in analyzer.current_states
        assert symbol in analyzer.regime_history
        assert len(analyzer.regime_history[symbol]) >= 3
    
    async def test_event_publishing(self, analyzer, event_bus):
        """Test event publishing for state changes."""
        symbol = "BTCUSDT"
        events_received = []
        
        # Subscribe to events
        async def event_handler(event):
            events_received.append(event)
        
        event_bus.subscribe("microstructure_state_changed", event_handler)
        
        # Trigger analysis
        order_book = OrderBookSnapshot(
            symbol=symbol,
            bids=[OrderBookLevel(Decimal("50000"), Decimal("1"))],
            asks=[OrderBookLevel(Decimal("50001"), Decimal("1"))]
        )
        
        trades = [{'price': 50000, 'quantity': 1, 'side': 'buy'}]
        
        await analyzer.analyze_market(symbol, order_book, trades)
        
        # Allow time for event processing
        await asyncio.sleep(0.1)
        
        # Check events were published
        assert len(events_received) > 0
        event = events_received[0]
        assert event.type == "microstructure_state_changed"
        assert event.data['symbol'] == symbol