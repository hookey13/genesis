"""Unit tests for market making strategy components."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from genesis.core.models import Order, OrderSide, OrderType, OrderStatus, Signal, SignalType
from genesis.execution.quote_generator import (
    QuoteGenerator,
    QuoteLevel,
    QuoteParameters,
    QuoteSet,
)
from genesis.execution.spread_model import MarketConditions, SpreadModel
from genesis.strategies.strategist.inventory_manager import (
    InventoryManager,
    InventoryLimits,
    InventoryZone,
)
from genesis.strategies.strategist.market_maker import (
    AdverseSelectionTracker,
    MarketMakerConfig,
    MarketMakingStrategy,
)


class TestSpreadModel:
    """Test spread calculation model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = SpreadModel(
            base_spread_bps=Decimal("10"),
            min_spread_bps=Decimal("5"),
            max_spread_bps=Decimal("50")
        )
    
    def test_basic_spread_calculation(self):
        """Test basic spread calculation without adjustments."""
        conditions = MarketConditions(
            current_price=Decimal("50000"),
            bid_price=Decimal("49950"),
            ask_price=Decimal("50050"),
            volatility=Decimal("0.005"),  # 0.5%
            volume_24h=Decimal("1000000"),
            order_book_depth={"bids": [], "asks": []},
            recent_trades=[]
        )
        
        spread_bps, factors = self.model.calculate_spread(conditions)
        
        assert spread_bps >= self.model.min_spread_bps
        assert spread_bps <= self.model.max_spread_bps
        assert factors.base_spread == Decimal("10")
    
    def test_volatility_adjustment(self):
        """Test spread adjustment based on volatility."""
        # Low volatility
        low_vol_conditions = MarketConditions(
            current_price=Decimal("50000"),
            bid_price=Decimal("49950"),
            ask_price=Decimal("50050"),
            volatility=Decimal("0.0005"),  # Very low
            volume_24h=Decimal("1000000"),
            order_book_depth={"bids": [], "asks": []},
            recent_trades=[]
        )
        
        low_spread, low_factors = self.model.calculate_spread(low_vol_conditions)
        
        # High volatility
        high_vol_conditions = MarketConditions(
            current_price=Decimal("50000"),
            bid_price=Decimal("49950"),
            ask_price=Decimal("50050"),
            volatility=Decimal("0.02"),  # High
            volume_24h=Decimal("1000000"),
            order_book_depth={"bids": [], "asks": []},
            recent_trades=[]
        )
        
        high_spread, high_factors = self.model.calculate_spread(high_vol_conditions)
        
        # High volatility should result in wider spreads
        assert high_spread > low_spread
        assert high_factors.volatility_factor > low_factors.volatility_factor
    
    def test_inventory_skew_adjustment(self):
        """Test spread adjustment based on inventory skew."""
        conditions = MarketConditions(
            current_price=Decimal("50000"),
            bid_price=Decimal("49950"),
            ask_price=Decimal("50050"),
            volatility=Decimal("0.005"),
            volume_24h=Decimal("1000000"),
            order_book_depth={"bids": [], "asks": []},
            recent_trades=[]
        )
        
        # High inventory skew
        high_skew_spread, high_skew_factors = self.model.calculate_spread(
            conditions,
            inventory_skew=Decimal("0.8")
        )
        
        # No skew
        no_skew_spread, no_skew_factors = self.model.calculate_spread(
            conditions,
            inventory_skew=Decimal("0")
        )
        
        # High skew should result in wider spreads
        assert high_skew_spread >= no_skew_spread
        assert high_skew_factors.inventory_factor >= no_skew_factors.inventory_factor
    
    def test_toxic_flow_adjustment(self):
        """Test spread adjustment for toxic flow."""
        conditions = MarketConditions(
            current_price=Decimal("50000"),
            bid_price=Decimal("49950"),
            ask_price=Decimal("50050"),
            volatility=Decimal("0.005"),
            volume_24h=Decimal("1000000"),
            order_book_depth={"bids": [], "asks": []},
            recent_trades=[]
        )
        
        # With toxic flow
        toxic_spread, toxic_factors = self.model.calculate_spread(
            conditions,
            toxic_flow_detected=True
        )
        
        # Without toxic flow
        normal_spread, normal_factors = self.model.calculate_spread(
            conditions,
            toxic_flow_detected=False
        )
        
        # Toxic flow should double the spread
        assert toxic_spread > normal_spread
        assert toxic_factors.adverse_selection_factor == Decimal("2.0")
        assert normal_factors.adverse_selection_factor == Decimal("1.0")
    
    def test_quote_price_calculation(self):
        """Test bid and ask price calculation."""
        mid_price = Decimal("50000")
        spread_bps = Decimal("20")
        
        # No skew
        bid, ask = self.model.calculate_quote_prices(mid_price, spread_bps)
        
        assert bid < mid_price
        assert ask > mid_price
        # Convert to float for pytest.approx comparison
        actual_spread = float((ask - bid) / bid * Decimal("10000"))
        expected_spread = float(spread_bps)
        assert actual_spread == pytest.approx(expected_spread, rel=0.01)
        
        # With positive skew (want to sell more)
        skew_bps = Decimal("10")
        bid_skewed, ask_skewed = self.model.calculate_quote_prices(
            mid_price, spread_bps, skew_bps
        )
        
        assert bid_skewed < bid  # Lower bids
        assert ask_skewed < ask  # Lower asks (but still above mid)


class TestInventoryManager:
    """Test inventory management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.limits = InventoryLimits(
            max_position_size_usdt=Decimal("10000"),
            max_total_inventory_usdt=Decimal("50000"),
            max_concentration_pct=Decimal("1.0")  # Allow 100% for testing
        )
        self.manager = InventoryManager(self.limits)
    
    def test_position_update(self):
        """Test updating inventory positions."""
        # Buy position
        pos = self.manager.update_position(
            "BTCUSDT",
            Decimal("1"),  # Buy 1 BTC
            Decimal("50000")
        )
        
        assert pos.quantity == Decimal("1")
        assert pos.average_price == Decimal("50000")
        assert pos.dollar_value == Decimal("50000")
        assert pos.is_long
        
        # Add to position
        pos = self.manager.update_position(
            "BTCUSDT",
            Decimal("0.5"),  # Buy 0.5 more
            Decimal("51000")
        )
        
        assert pos.quantity == Decimal("1.5")
        # Average price should be weighted
        expected_avg = (Decimal("50000") + Decimal("25500")) / Decimal("1.5")
        assert pos.average_price == expected_avg
        
        # Reduce position
        pos = self.manager.update_position(
            "BTCUSDT",
            Decimal("-0.5"),  # Sell 0.5
            Decimal("52000")
        )
        
        assert pos.quantity == Decimal("1")
        # PnL should be calculated
        assert self.manager.realized_pnl > 0
    
    def test_inventory_zones(self):
        """Test inventory zone calculation."""
        # Start in green zone
        assert self.manager.get_inventory_zone() == InventoryZone.GREEN
        
        # Add position to reach yellow zone
        self.manager.update_position(
            "BTCUSDT",
            Decimal("0.4"),  # 40% of max
            Decimal("50000")
        )
        
        assert self.manager.get_inventory_zone() == InventoryZone.YELLOW
        
        # Add more to reach red zone
        self.manager.update_position(
            "BTCUSDT",
            Decimal("0.4"),  # Now 80% of max
            Decimal("50000")
        )
        
        assert self.manager.get_inventory_zone() == InventoryZone.RED
    
    def test_skew_calculation(self):
        """Test inventory skew calculation."""
        # Balanced positions
        self.manager.update_position("BTCUSDT", Decimal("1"), Decimal("50000"))
        self.manager.update_position("ETHUSDT", Decimal("-20"), Decimal("2500"))  # Short
        
        skew = self.manager.calculate_skew()
        assert skew == Decimal("0")  # Balanced long and short
        
        # Long skewed
        self.manager.update_position("BTCUSDT", Decimal("1"), Decimal("50000"))
        
        skew = self.manager.calculate_skew()
        assert skew > 0  # Long skewed
    
    def test_order_acceptance(self):
        """Test order acceptance based on limits."""
        # Should accept small order
        should_accept, reason = self.manager.should_accept_order(
            "BTCUSDT",
            "BUY",
            Decimal("0.1"),
            Decimal("50000")
        )
        
        assert should_accept
        assert reason == "OK"
        
        # Should reject order exceeding position limit
        should_accept, reason = self.manager.should_accept_order(
            "BTCUSDT",
            "BUY",
            Decimal("0.5"),  # Would be $25k > $10k limit
            Decimal("50000")
        )
        
        assert not should_accept
        assert "Would exceed position limit" in reason or "position limit" in reason
    
    def test_exit_signals(self):
        """Test generation of exit signals."""
        # Add old position
        past_time = datetime.now(UTC) - timedelta(hours=25)
        self.manager.update_position(
            "BTCUSDT",
            Decimal("1"),
            Decimal("50000"),
            past_time
        )
        
        signals = self.manager.get_exit_signals()
        
        assert len(signals) > 0
        assert signals[0]["symbol"] == "BTCUSDT"
        assert signals[0]["side"] == "SELL"  # Sell to exit long
        assert "Age" in signals[0]["reason"]


class TestQuoteGenerator:
    """Test quote generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = QuoteParameters(
            num_layers=3,
            base_quote_size_usdt=Decimal("1000")
        )
        self.generator = QuoteGenerator(self.params)
    
    def test_basic_quote_generation(self):
        """Test basic quote generation."""
        quote_set = self.generator.generate_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("10")
        )
        
        assert quote_set.symbol == "BTCUSDT"
        assert len(quote_set.bid_quotes) == 3
        assert len(quote_set.ask_quotes) == 3
        
        # Check bid prices are below mid
        for quote in quote_set.bid_quotes:
            assert quote.price < quote_set.mid_price
        
        # Check ask prices are above mid
        for quote in quote_set.ask_quotes:
            assert quote.price > quote_set.mid_price
        
        # Check layers have increasing spreads
        for i in range(1, len(quote_set.bid_quotes)):
            assert quote_set.bid_quotes[i].spread_bps > quote_set.bid_quotes[i-1].spread_bps
    
    def test_quote_generation_with_skew(self):
        """Test quote generation with inventory skew."""
        # Positive skew (long inventory, want to sell)
        quote_set = self.generator.generate_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("10"),
            inventory_skew=Decimal("0.5")  # 50% long
        )
        
        # With positive skew, both bids and asks should be lower
        bid_price = quote_set.bid_quotes[0].price
        ask_price = quote_set.ask_quotes[0].price
        
        # Generate without skew for comparison
        normal_quotes = self.generator.generate_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("10"),
            inventory_skew=Decimal("0")
        )
        
        assert bid_price < normal_quotes.bid_quotes[0].price
        assert ask_price < normal_quotes.ask_quotes[0].price
    
    def test_size_adjustments(self):
        """Test quote size adjustments."""
        quote_set = self.generator.generate_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("10"),
            size_multiplier=Decimal("0.5"),
            bid_size_adjustment=Decimal("2"),
            ask_size_adjustment=Decimal("0.5")
        )
        
        # Bid sizes should be larger than ask sizes
        total_bid_value = sum(q.dollar_value for q in quote_set.bid_quotes)
        total_ask_value = sum(q.dollar_value for q in quote_set.ask_quotes)
        
        assert total_bid_value > total_ask_value
    
    def test_aggressive_quotes(self):
        """Test aggressive quote generation for position reduction."""
        quotes = self.generator.generate_aggressive_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            side=OrderSide.SELL,
            target_size_usdt=Decimal("5000")
        )
        
        assert len(quotes) > 0
        
        # Check prices are aggressive (below mid for sells)
        for quote in quotes:
            assert quote.side == OrderSide.SELL
            assert quote.price < Decimal("50000")
        
        # Check increasing aggressiveness
        for i in range(1, len(quotes)):
            assert quotes[i].price < quotes[i-1].price
    
    def test_quote_validation(self):
        """Test quote validation."""
        # Create valid quotes
        valid_quotes = self.generator.generate_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("10")
        )
        
        is_valid, errors = self.generator.validate_quotes(valid_quotes)
        assert is_valid
        assert len(errors) == 0
        
        # Create invalid quotes (crossed)
        invalid_quotes = QuoteSet(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            bid_quotes=[QuoteLevel(
                price=Decimal("50100"),  # Above mid
                quantity=Decimal("1"),
                side=OrderSide.BUY,
                layer=0,
                spread_bps=Decimal("10"),
                distance_bps=Decimal("20")
            )],
            ask_quotes=[QuoteLevel(
                price=Decimal("49900"),  # Below mid
                quantity=Decimal("1"),
                side=OrderSide.SELL,
                layer=0,
                spread_bps=Decimal("10"),
                distance_bps=Decimal("20")
            )],
            total_bid_value=Decimal("50100"),
            total_ask_value=Decimal("49900")
        )
        
        is_valid, errors = self.generator.validate_quotes(invalid_quotes)
        assert not is_valid
        assert "Crossed quotes" in errors[0]


class TestAdverseSelectionTracker:
    """Test adverse selection tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = AdverseSelectionTracker()
    
    def test_fill_tracking(self):
        """Test tracking of fill statistics."""
        # Add some fills
        for _ in range(8):
            self.tracker.update_fill(OrderSide.BUY)
        
        for _ in range(2):
            self.tracker.update_fill(OrderSide.SELL)
        
        assert self.tracker.total_fills == 10
        assert self.tracker.buy_fills == 8
        assert self.tracker.sell_fills == 2
    
    def test_toxic_flow_detection(self):
        """Test detection of toxic flow."""
        # Balanced flow - not toxic
        for _ in range(5):
            self.tracker.update_fill(OrderSide.BUY)
            self.tracker.update_fill(OrderSide.SELL)
        
        assert not self.tracker.check_toxic_flow(Decimal("0.80"))
        
        # Reset for clean toxic flow test
        self.tracker.reset()
        
        # One-sided flow - toxic (16 buys out of 20 = 80%)
        for _ in range(16):
            self.tracker.update_fill(OrderSide.BUY)
        for _ in range(4):
            self.tracker.update_fill(OrderSide.SELL)
        
        assert self.tracker.check_toxic_flow(Decimal("0.80"))
        assert self.tracker.toxic_flow_detected
    
    def test_recovery_from_toxic_flow(self):
        """Test recovery mechanism."""
        # Create toxic flow
        for _ in range(10):
            self.tracker.update_fill(OrderSide.BUY)
        
        self.tracker.check_toxic_flow(Decimal("0.80"))
        assert self.tracker.toxic_flow_detected
        
        # Add recovery fills
        for _ in range(100):
            self.tracker.update_fill(OrderSide.SELL)
            if self.tracker.should_recover(100):
                break
        
        assert not self.tracker.toxic_flow_detected
        assert self.tracker.recovery_fills == 0


@pytest.mark.asyncio
class TestMarketMakingStrategy:
    """Test market making strategy integration."""
    
    def setup_method(self, method):
        """Set up test fixtures."""
        self.config = MarketMakerConfig(
            name="TestMarketMaker",
            symbol="BTCUSDT",
            max_position_usdt=Decimal("10000"),
            base_spread_bps=Decimal("10")
        )
        self.strategy = MarketMakingStrategy(self.config)
    
    @pytest.mark.asyncio
    async def test_signal_generation(self):
        """Test generation of market making signals."""
        # Set up market data
        self.strategy.current_bid = Decimal("49950")
        self.strategy.current_ask = Decimal("50050")
        self.strategy.current_mid_price = Decimal("50000")
        
        # Force refresh by setting old refresh time
        self.strategy.last_refresh_time = datetime.now(UTC) - timedelta(seconds=10)
        
        signals = await self.strategy.generate_signals()
        
        # Should generate buy and sell signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        assert len(buy_signals) > 0
        assert len(sell_signals) > 0
        
        # Check signal properties
        for signal in buy_signals:
            assert signal.price_target < self.strategy.current_mid_price
            assert signal.metadata.get("post_only") is True
        
        for signal in sell_signals:
            assert signal.price_target > self.strategy.current_mid_price
            assert signal.metadata.get("post_only") is True
    
    @pytest.mark.asyncio
    async def test_inventory_management(self):
        """Test inventory position management."""
        # Simulate order fills
        buy_order = Order(
            order_id=str(uuid4()),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            price=Decimal("49900"),
            quantity=Decimal("0.1"),
            status=OrderStatus.FILLED
        )
        
        await self.strategy.on_order_filled(buy_order)
        
        assert self.strategy.inventory.current_position == Decimal("0.1")
        assert self.strategy.inventory.zone == "GREEN"
        
        # Check that position was tracked
        assert self.strategy.inventory.current_position > 0
    
    @pytest.mark.asyncio
    async def test_refresh_logic(self):
        """Test quote refresh logic."""
        self.strategy.last_refresh_time = datetime.now(UTC) - timedelta(seconds=10)
        self.strategy.current_mid_price = Decimal("50000")
        self.strategy.last_mid_price = Decimal("49900")
        
        # Should refresh due to time
        assert self.strategy._should_refresh_quotes()
        
        # Reset and test price-based refresh
        self.strategy.last_refresh_time = datetime.now(UTC)
        self.strategy.last_mid_price = Decimal("49000")  # Big move
        
        assert self.strategy._should_refresh_quotes()
    
    @pytest.mark.asyncio
    async def test_adverse_selection_response(self):
        """Test response to adverse selection."""
        # Simulate toxic flow
        for _ in range(10):
            self.strategy.adverse_tracker.update_fill(OrderSide.BUY, immediate_loss=True)
        
        self.strategy.adverse_tracker.check_toxic_flow(Decimal("0.80"))
        
        # Generate signals with toxic flow
        self.strategy.current_bid = Decimal("49950")
        self.strategy.current_ask = Decimal("50050")
        self.strategy.current_mid_price = Decimal("50000")
        
        signals = await self.strategy.generate_signals()
        
        # Should still generate signals but with adjusted parameters
        assert len(signals) > 0
        
        # Spread should be wider due to toxic flow
        spread = self.strategy._calculate_effective_spread()
        assert spread > self.config.base_spread_bps