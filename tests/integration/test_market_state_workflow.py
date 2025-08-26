"""
Integration tests for Market State Management workflow.

Tests the complete workflow of market state classification, database persistence,
event publishing, position sizing adjustments, and strategy activation.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from genesis.analytics.market_state_classifier import (
    MarketState,
    MarketStateClassifier,
    StateTransitionManager,
    MaintenanceMonitor,
    GlobalMarketStateClassifier,
    GlobalMarketState,
    PositionSizeAdjuster,
    StrategyStateManager
)
from genesis.analytics.volatility_calculator import VolatilityCalculator
from genesis.data.market_data_service import MarketDataService, Candle, OrderBook
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus
from genesis.exchange.gateway import BinanceGateway
from genesis.exchange.websocket_manager import WebSocketManager


@pytest.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    repo = SQLiteRepository(db_path)
    await repo.initialize()
    
    yield repo
    
    await repo.shutdown()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_gateway():
    """Create mock Binance gateway."""
    gateway = AsyncMock(spec=BinanceGateway)
    
    # Mock klines data
    gateway.get_klines.return_value = [
        [1640000000000, "50000", "51000", "49000", "50500", "1000", 1640003600000, "50000000", 1000],
        [1640003600000, "50500", "51500", "49500", "51000", "1100", 1640007200000, "55000000", 1100],
        # Add more sample data...
    ] * 20
    
    # Mock 24h ticker
    gateway.get_24h_ticker.return_value = {
        "volume": "10000",
        "quoteVolume": "500000000",
        "priceChangePercent": "2.5",
        "highPrice": "52000",
        "lowPrice": "49000",
        "weightedAvgPrice": "50500"
    }
    
    # Mock order book
    gateway.get_order_book.return_value = {
        "bids": [["50000", "10"], ["49999", "20"], ["49998", "30"]],
        "asks": [["50001", "10"], ["50002", "20"], ["50003", "30"]]
    }
    
    return gateway


@pytest.fixture
async def market_data_service(event_bus, mock_gateway):
    """Create market data service."""
    ws_manager = WebSocketManager(mock_gateway)
    service = MarketDataService(
        websocket_manager=ws_manager,
        gateway=mock_gateway,
        event_bus=event_bus
    )
    
    await service.start()
    yield service
    await service.stop()


class TestMarketStateWorkflow:
    """Test complete market state management workflow."""
    
    @pytest.mark.asyncio
    async def test_full_classification_workflow(
        self,
        event_bus,
        temp_db,
        market_data_service
    ):
        """Test full market state classification workflow."""
        # Create components
        classifier = MarketStateClassifier(event_bus=event_bus)
        volatility_calc = VolatilityCalculator()
        state_manager = StateTransitionManager(event_bus=event_bus)
        
        symbol = "BTCUSDT"
        
        # Get market data
        volatility_data = await market_data_service.get_volatility_data(symbol)
        ticker_data = await market_data_service.get_24h_ticker(symbol)
        order_book = await market_data_service.get_order_book_snapshot(symbol)
        
        # Calculate spread
        spread_bps = 0
        if order_book:
            spread = order_book.spread_basis_points()
            if spread:
                spread_bps = spread
        
        # Update market state
        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=volatility_data.get("atr", Decimal("500")),
            realized_volatility=volatility_data.get("realized_volatility", Decimal("0.5")),
            volume_24h=ticker_data.get("volume", Decimal("10000")),
            spread_bps=spread_bps,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        assert context.symbol == symbol
        assert context.current_state in MarketState
        assert context.volatility_atr > 0
        
        # Save to database
        await temp_db.save_market_state({
            "symbol": context.symbol,
            "state": context.current_state.value,
            "volatility_atr": context.volatility_atr,
            "spread_basis_points": context.spread_basis_points,
            "volume_24h": context.volume_24h,
            "liquidity_score": context.liquidity_score,
            "detected_at": context.detected_at,
            "state_duration_seconds": context.state_duration_seconds
        })
        
        # Verify database persistence
        saved_state = await temp_db.get_latest_market_state(symbol)
        assert saved_state is not None
        assert saved_state["symbol"] == symbol
        assert saved_state["state"] == context.current_state.value
    
    @pytest.mark.asyncio
    async def test_state_transition_with_events(self, event_bus):
        """Test state transitions with event publishing."""
        classifier = MarketStateClassifier(event_bus=event_bus)
        events_received = []
        
        # Subscribe to events
        async def event_handler(event_type, data):
            events_received.append((event_type, data))
        
        event_bus.subscribe("MarketStateChangeEvent", event_handler)
        
        symbol = "BTCUSDT"
        
        # First update - establish baseline
        await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("500"),
            realized_volatility=Decimal("0.3"),
            volume_24h=Decimal("10000000"),
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        # Second update - trigger state change to VOLATILE
        await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("2000"),  # High ATR
            realized_volatility=Decimal("1.5"),  # High volatility
            volume_24h=Decimal("50000000"),  # High volume
            spread_bps=50,
            liquidity_score=Decimal("80"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        # Allow events to propagate
        await asyncio.sleep(0.1)
        
        # Check events were published
        assert len(events_received) > 0
        event_type, event_data = events_received[-1]
        assert event_type == "MarketStateChangeEvent"
        assert event_data["symbol"] == symbol
    
    @pytest.mark.asyncio
    async def test_position_sizing_integration(self, event_bus):
        """Test position sizing adjustments based on market state."""
        classifier = MarketStateClassifier(event_bus=event_bus)
        adjuster = PositionSizeAdjuster(event_bus=event_bus)
        
        symbol = "BTCUSDT"
        base_size = Decimal("1000")
        
        # Update to VOLATILE state
        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("2000"),
            realized_volatility=Decimal("1.5"),
            volume_24h=Decimal("50000000"),
            spread_bps=50,
            liquidity_score=Decimal("80"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        # Calculate adjusted position size
        adjusted_size, reason = await adjuster.calculate_position_size(
            symbol=symbol,
            base_size=base_size,
            market_state=context.current_state,
            volatility_percentile=context.volatility_percentile
        )
        
        # Should be reduced due to volatility
        assert adjusted_size < base_size
        assert "VOLATILE" in reason or "Volatility" in reason
    
    @pytest.mark.asyncio
    async def test_strategy_activation_workflow(self, event_bus):
        """Test strategy activation based on market states."""
        classifier = MarketStateClassifier(event_bus=event_bus)
        strategy_manager = StrategyStateManager(event_bus=event_bus)
        
        symbol = "BTCUSDT"
        
        # Normal market conditions
        await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("500"),
            realized_volatility=Decimal("0.3"),
            volume_24h=Decimal("10000000"),
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        states = await strategy_manager.update_strategy_states(MarketState.NORMAL)
        assert states["grid_trading"] is True
        assert states["mean_reversion"] is True
        
        # Volatile market conditions
        await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("2000"),
            realized_volatility=Decimal("1.5"),
            volume_24h=Decimal("50000000"),
            spread_bps=50,
            liquidity_score=Decimal("80"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        states = await strategy_manager.update_strategy_states(MarketState.VOLATILE)
        assert states["grid_trading"] is False  # Disabled in volatile
        assert states["momentum"] is True  # Enabled in volatile
    
    @pytest.mark.asyncio
    async def test_maintenance_detection_workflow(self, event_bus):
        """Test maintenance detection and handling workflow."""
        classifier = MarketStateClassifier(event_bus=event_bus)
        monitor = MaintenanceMonitor(classifier)
        
        symbol = "BTCUSDT"
        
        # Schedule maintenance
        maintenance_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        monitor.schedule_maintenance(symbol, maintenance_time)
        
        # Check if approaching maintenance (within buffer)
        is_approaching = await classifier.schedule_maintenance_check(
            symbol, maintenance_time, buffer_minutes=30
        )
        assert is_approaching
        
        # Update state with maintenance detected
        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("500"),
            realized_volatility=Decimal("0.3"),
            volume_24h=Decimal("10000000"),
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=True
        )
        
        assert context.current_state == MarketState.MAINTENANCE
    
    @pytest.mark.asyncio
    async def test_global_market_state_integration(self, event_bus, temp_db):
        """Test global market state classification and persistence."""
        global_classifier = GlobalMarketStateClassifier(event_bus=event_bus)
        
        btc_price = Decimal("45000")  # Down from baseline
        major_pairs = [
            {"symbol": "ETHUSDT", "change_percent": Decimal("-8")},
            {"symbol": "BNBUSDT", "change_percent": Decimal("-7")},
            {"symbol": "ADAUSDT", "change_percent": Decimal("-9")},
        ]
        
        # Add price history for comparison
        for i in range(25):
            global_classifier._btc_price_history.append(
                Decimal(str(50000 - i * 100))  # Declining prices
            )
        
        # Classify global state
        global_state = await global_classifier.classify_global_state(
            btc_price, major_pairs, fear_greed_index=20  # Extreme fear
        )
        
        # Should detect bearish/crash conditions
        assert global_state in [GlobalMarketState.BEAR, GlobalMarketState.CRASH]
        
        # Save to database
        await temp_db.save_global_market_state({
            "btc_price": btc_price,
            "total_market_cap": Decimal("2000000000000"),
            "fear_greed_index": 20,
            "correlation_spike": True,
            "state": global_state.value,
            "vix_crypto": Decimal("80"),
            "detected_at": datetime.now(timezone.utc)
        })
        
        # Verify persistence
        saved_state = await temp_db.get_latest_global_market_state()
        assert saved_state is not None
        assert saved_state["state"] == global_state.value
        assert saved_state["fear_greed_index"] == 20
    
    @pytest.mark.asyncio
    async def test_volume_anomaly_detection_workflow(self, event_bus):
        """Test volume anomaly detection and state updates."""
        classifier = MarketStateClassifier(event_bus=event_bus)
        symbol = "BTCUSDT"
        
        # Build volume profile with normal volumes
        base_volume = Decimal("1000000")
        for hour in range(24):
            timestamp = datetime.now(timezone.utc).replace(hour=hour)
            classifier.update_volume_profile(symbol, base_volume, timestamp)
        
        # Test with normal volume
        is_anomaly, reason = classifier.detect_volume_pattern_anomaly(
            symbol, base_volume * Decimal("1.1"), datetime.now(timezone.utc)
        )
        assert not is_anomaly
        
        # Test with anomalous volume spike
        spike_volume = base_volume * Decimal("5")
        is_anomaly, reason = classifier.detect_volume_pattern_anomaly(
            symbol, spike_volume, datetime.now(timezone.utc)
        )
        assert is_anomaly
        assert "deviates" in reason
        
        # Update state with volume anomaly
        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=Decimal("500"),
            realized_volatility=Decimal("0.3"),
            volume_24h=spike_volume,  # Anomalous volume
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        # Volume anomaly should be considered in state classification
        assert context.volume_zscore > 2  # High Z-score indicates anomaly
    
    @pytest.mark.asyncio
    async def test_emergency_workflow(self, event_bus):
        """Test emergency procedures and strategy shutdown."""
        strategy_manager = StrategyStateManager(event_bus=event_bus)
        events_received = []
        
        # Subscribe to emergency events
        async def event_handler(event_type, data):
            events_received.append((event_type, data))
        
        event_bus.subscribe("EmergencyStrategyStopEvent", event_handler)
        
        # Enable some strategies
        await strategy_manager.update_strategy_states(MarketState.NORMAL)
        assert len(strategy_manager.get_enabled_strategies()) > 0
        
        # Trigger emergency shutdown
        await strategy_manager.emergency_disable_all("Critical system failure")
        
        # All strategies should be disabled
        assert len(strategy_manager.get_enabled_strategies()) == 0
        
        # Emergency event should be published
        await asyncio.sleep(0.1)
        assert len(events_received) > 0
        event_type, event_data = events_received[0]
        assert event_type == "EmergencyStrategyStopEvent"
        assert "Critical system failure" in event_data["reason"]
    
    @pytest.mark.asyncio
    async def test_state_history_persistence(self, temp_db):
        """Test market state history storage and retrieval."""
        symbol = "BTCUSDT"
        
        # Save multiple states
        states = [MarketState.NORMAL, MarketState.VOLATILE, MarketState.PANIC]
        for i, state in enumerate(states):
            await temp_db.save_market_state({
                "symbol": symbol,
                "state": state.value,
                "volatility_atr": Decimal(str(500 + i * 500)),
                "spread_basis_points": 10 + i * 10,
                "volume_24h": Decimal(str(1000000 * (i + 1))),
                "liquidity_score": Decimal(str(95 - i * 5)),
                "detected_at": datetime.now(timezone.utc) - timedelta(hours=len(states)-i),
                "state_duration_seconds": 3600
            })
        
        # Retrieve history
        history = await temp_db.get_market_state_history(symbol, hours_back=24)
        
        assert len(history) == len(states)
        # Should be ordered by time (most recent first)
        assert history[0]["state"] == MarketState.PANIC.value
        assert history[-1]["state"] == MarketState.NORMAL.value
    
    @pytest.mark.asyncio
    async def test_complete_market_analysis_pipeline(
        self,
        event_bus,
        temp_db,
        market_data_service
    ):
        """Test complete market analysis pipeline from data to decision."""
        # Create all components
        classifier = MarketStateClassifier(event_bus=event_bus)
        global_classifier = GlobalMarketStateClassifier(event_bus=event_bus)
        volatility_calc = VolatilityCalculator()
        position_adjuster = PositionSizeAdjuster(event_bus=event_bus)
        strategy_manager = StrategyStateManager(event_bus=event_bus)
        
        symbol = "BTCUSDT"
        
        # Step 1: Fetch market data
        candles = await market_data_service.get_historical_candles(symbol, "1h", 100)
        ticker = await market_data_service.get_24h_ticker(symbol)
        volatility_data = await market_data_service.get_volatility_data(symbol)
        
        # Step 2: Classify market state
        context = await classifier.update_state(
            symbol=symbol,
            volatility_atr=volatility_data.get("atr", Decimal("500")),
            realized_volatility=volatility_data.get("realized_volatility", Decimal("0.5")),
            volume_24h=ticker.get("volume", Decimal("10000")),
            spread_bps=10,
            liquidity_score=Decimal("95"),
            correlation_spike=False,
            maintenance_detected=False
        )
        
        # Step 3: Classify global state
        global_state = await global_classifier.classify_global_state(
            btc_price=Decimal("50000"),
            major_pairs=[
                {"symbol": "ETHUSDT", "change_percent": Decimal("2")},
                {"symbol": "BNBUSDT", "change_percent": Decimal("1.5")},
            ],
            fear_greed_index=50
        )
        
        # Step 4: Adjust position sizing
        base_size = Decimal("1000")
        adjusted_size, reason = await position_adjuster.calculate_position_size(
            symbol=symbol,
            base_size=base_size,
            market_state=context.current_state,
            global_state=global_state,
            volatility_percentile=context.volatility_percentile
        )
        
        # Step 5: Update strategy states
        strategy_states = await strategy_manager.update_strategy_states(
            context.current_state, symbol
        )
        
        # Step 6: Persist everything
        await temp_db.save_market_state({
            "symbol": context.symbol,
            "state": context.current_state.value,
            "volatility_atr": context.volatility_atr,
            "spread_basis_points": context.spread_basis_points,
            "volume_24h": context.volume_24h,
            "liquidity_score": context.liquidity_score,
            "detected_at": context.detected_at,
            "state_duration_seconds": context.state_duration_seconds
        })
        
        await temp_db.save_global_market_state({
            "btc_price": Decimal("50000"),
            "total_market_cap": Decimal("2000000000000"),
            "fear_greed_index": 50,
            "correlation_spike": False,
            "state": global_state.value,
            "vix_crypto": Decimal("30"),
            "detected_at": datetime.now(timezone.utc)
        })
        
        # Verify complete workflow
        assert context.current_state in MarketState
        assert global_state in GlobalMarketState
        assert adjusted_size > 0
        assert len(strategy_states) > 0
        
        # Verify persistence
        saved_market_state = await temp_db.get_latest_market_state(symbol)
        saved_global_state = await temp_db.get_latest_global_market_state()
        
        assert saved_market_state is not None
        assert saved_global_state is not None