"""Integration tests for market making strategy."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.models import Order, OrderSide, OrderType, Position, Signal, SignalType
from genesis.execution.quote_generator import QuoteGenerator, QuoteParameters
from genesis.execution.spread_model import MarketConditions, SpreadModel
from genesis.strategies.strategist.inventory_manager import InventoryManager, InventoryLimits
from genesis.strategies.strategist.market_maker import MarketMakerConfig, MarketMakingStrategy


class MockMarketData:
    """Mock market data provider for testing."""
    
    def __init__(self, initial_price: Decimal = Decimal("50000")):
        self.mid_price = initial_price
        self.bid_price = initial_price * Decimal("0.999")
        self.ask_price = initial_price * Decimal("1.001")
        self.volatility = Decimal("0.005")
        self.tick_count = 0
        
        # Order book depth
        self.bids = []
        self.asks = []
        self._generate_order_book()
    
    def _generate_order_book(self):
        """Generate mock order book."""
        # Generate 10 levels of bids and asks
        for i in range(10):
            bid_price = self.bid_price * (Decimal("1") - Decimal(i) * Decimal("0.0001"))
            ask_price = self.ask_price * (Decimal("1") + Decimal(i) * Decimal("0.0001"))
            
            self.bids.append([float(bid_price), float(Decimal("10") - i)])
            self.asks.append([float(ask_price), float(Decimal("10") - i)])
    
    def tick(self, price_change_pct: Decimal = Decimal("0")):
        """Simulate a market tick."""
        self.tick_count += 1
        
        # Apply price change
        if price_change_pct != 0:
            self.mid_price *= (Decimal("1") + price_change_pct)
            self.bid_price = self.mid_price * Decimal("0.999")
            self.ask_price = self.mid_price * Decimal("1.001")
            self._generate_order_book()
        
        # Random walk for realistic simulation
        import random
        if random.random() > 0.5:
            change = Decimal(str(random.uniform(-0.001, 0.001)))
            self.mid_price *= (Decimal("1") + change)
            self.bid_price = self.mid_price * Decimal("0.999")
            self.ask_price = self.mid_price * Decimal("1.001")
    
    def get_market_data(self) -> Dict:
        """Get current market data."""
        return {
            "symbol": "BTCUSDT",
            "bid": self.bid_price,
            "ask": self.ask_price,
            "mid": self.mid_price,
            "bids": self.bids,
            "asks": self.asks,
            "volume_24h": Decimal("1000000"),
            "volatility": self.volatility,
            "timestamp": datetime.now(UTC)
        }


class MockOrderExecutor:
    """Mock order executor for testing."""
    
    def __init__(self, fill_probability: float = 0.3):
        self.orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.cancelled_orders: List[Order] = []
        self.fill_probability = fill_probability
        self.total_volume = Decimal("0")
        self.total_fees = Decimal("0")
    
    async def submit_order(self, order: Order) -> Order:
        """Submit an order."""
        order.status = "NEW"
        self.orders.append(order)
        return order
    
    async def cancel_order(self, order_id) -> bool:
        """Cancel an order."""
        for order in self.orders:
            if order.order_id == order_id:
                order.status = "CANCELLED"
                self.cancelled_orders.append(order)
                self.orders.remove(order)
                return True
        return False
    
    def simulate_fills(self, market_data: Dict) -> List[Order]:
        """Simulate order fills based on market conditions."""
        filled = []
        mid_price = market_data["mid"]
        
        import random
        for order in self.orders[:]:  # Copy list to avoid modification during iteration
            should_fill = False
            
            if order.side == OrderSide.BUY:
                # Buy orders fill if price drops below order price
                if market_data["bid"] <= order.price or random.random() < self.fill_probability:
                    should_fill = True
            else:
                # Sell orders fill if price rises above order price
                if market_data["ask"] >= order.price or random.random() < self.fill_probability:
                    should_fill = True
            
            if should_fill:
                order.status = "FILLED"
                order.filled_quantity = order.quantity
                order.filled_price = order.price
                
                self.filled_orders.append(order)
                self.orders.remove(order)
                filled.append(order)
                
                # Track volume and fees
                self.total_volume += order.quantity * order.price
                # Assume maker fee of -0.025%
                self.total_fees -= order.quantity * order.price * Decimal("0.00025")
        
        return filled


@pytest.mark.asyncio
class TestMarketMakingIntegration:
    """Integration tests for complete market making flow."""
    
    async def test_full_market_making_cycle(self):
        """Test a complete market making cycle with order flow."""
        # Initialize components
        config = MarketMakerConfig(
            name="IntegrationTest",
            symbol="BTCUSDT",
            max_position_usdt=Decimal("10000"),
            base_spread_bps=Decimal("10"),
            quote_layers=3,
            quote_refresh_seconds=5
        )
        
        strategy = MarketMakingStrategy(config)
        market_data = MockMarketData(Decimal("50000"))
        executor = MockOrderExecutor(fill_probability=0.3)
        
        # Track performance
        pnl_history = []
        inventory_history = []
        spread_history = []
        
        # Run for 100 ticks
        for tick in range(100):
            # Get market data
            data = market_data.get_market_data()
            
            # Update strategy with market data
            await strategy.analyze(data)
            
            # Generate signals
            signals = await strategy.generate_signals()
            
            # Convert signals to orders and submit
            for signal in signals:
                if signal.signal_type == SignalType.BUY:
                    order = Order(
                        order_id=signal.metadata.get("order_id"),
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        price=signal.entry_price,
                        quantity=signal.position_size,
                        status="NEW"
                    )
                    await executor.submit_order(order)
                    
                elif signal.signal_type == SignalType.SELL:
                    order = Order(
                        order_id=signal.metadata.get("order_id"),
                        symbol=signal.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        price=signal.entry_price,
                        quantity=signal.position_size,
                        status="NEW"
                    )
                    await executor.submit_order(order)
                    
                elif signal.signal_type == SignalType.CANCEL:
                    order_id = signal.metadata.get("order_id")
                    if order_id:
                        await executor.cancel_order(order_id)
            
            # Simulate order fills
            filled_orders = executor.simulate_fills(data)
            
            # Update strategy with fills
            for order in filled_orders:
                await strategy.on_order_filled(order)
            
            # Track metrics
            if tick % 10 == 0:
                pnl_history.append(float(strategy.state.pnl_usdt))
                inventory_history.append(float(strategy.inventory.current_position))
                spread_history.append(float(strategy._calculate_effective_spread()))
            
            # Simulate market movement
            market_data.tick()
            
            # Wait to simulate time passing
            await asyncio.sleep(0.01)
        
        # Verify results
        assert len(executor.filled_orders) > 0, "Should have filled some orders"
        assert executor.total_volume > 0, "Should have traded volume"
        assert len(pnl_history) > 0, "Should have PnL history"
        
        # Check inventory management
        final_inventory = abs(strategy.inventory.current_position)
        max_allowed = config.max_position_usdt / market_data.mid_price
        assert final_inventory <= max_allowed, "Inventory should be within limits"
        
        # Check spread adjustments occurred
        assert min(spread_history) != max(spread_history), "Spread should vary"
    
    async def test_inventory_risk_management(self):
        """Test inventory risk limits and position reduction."""
        config = MarketMakerConfig(
            name="RiskTest",
            symbol="BTCUSDT",
            max_position_usdt=Decimal("5000"),
            base_spread_bps=Decimal("10")
        )
        
        strategy = MarketMakingStrategy(config)
        
        # Force large position
        strategy.inventory.current_position = Decimal("0.15")  # ~$7500 at $50k
        strategy.inventory.max_position = config.max_position_usdt
        
        # Update inventory zone
        strategy.inventory.update_zone(config.inventory_zones)
        
        # Should be in RED zone
        assert strategy.inventory.zone == "RED"
        
        # Generate signals - should only reduce position
        strategy.current_mid_price = Decimal("50000")
        strategy.current_bid = Decimal("49950")
        strategy.current_ask = Decimal("50050")
        
        signals = await strategy.manage_positions()
        
        # Should generate exit signals
        exit_signals = [s for s in signals if s.signal_type in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT]]
        assert len(exit_signals) > 0, "Should generate position reduction signals"
    
    async def test_adverse_selection_handling(self):
        """Test response to adverse selection / toxic flow."""
        config = MarketMakerConfig(
            name="ToxicTest",
            symbol="BTCUSDT",
            base_spread_bps=Decimal("10"),
            toxic_flow_threshold=Decimal("0.80"),
            adverse_spread_multiplier=Decimal("2.0")
        )
        
        strategy = MarketMakingStrategy(config)
        market_data = MockMarketData()
        
        # Simulate toxic flow (one-sided fills)
        for _ in range(10):
            buy_order = Order(
                order_id=None,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("49900"),
                quantity=Decimal("0.1"),
                status="FILLED"
            )
            await strategy.on_order_filled(buy_order)
        
        # Check toxic flow detected
        strategy.adverse_tracker.check_toxic_flow(config.toxic_flow_threshold)
        assert strategy.adverse_tracker.toxic_flow_detected
        
        # Calculate spread - should be wider
        normal_spread = config.base_spread_bps
        toxic_spread = strategy._calculate_effective_spread()
        
        assert toxic_spread > normal_spread, "Spread should widen with toxic flow"
        assert toxic_spread >= normal_spread * config.adverse_spread_multiplier
    
    async def test_multi_layer_quoting(self):
        """Test multi-layer quote generation and management."""
        config = MarketMakerConfig(
            name="LayerTest",
            symbol="BTCUSDT",
            base_spread_bps=Decimal("10"),
            quote_layers=3,
            layer_spacing_multiplier=Decimal("2")
        )
        
        generator = QuoteGenerator(QuoteParameters(
            num_layers=config.quote_layers,
            layer_spacing_multiplier=config.layer_spacing_multiplier
        ))
        
        # Generate quotes
        quote_set = generator.generate_quotes(
            symbol="BTCUSDT",
            mid_price=Decimal("50000"),
            spread_bps=Decimal("10")
        )
        
        # Verify layer structure
        assert len(quote_set.bid_quotes) == 3
        assert len(quote_set.ask_quotes) == 3
        
        # Check layer spacing
        for i in range(1, len(quote_set.bid_quotes)):
            prev_spread = quote_set.bid_quotes[i-1].spread_bps
            curr_spread = quote_set.bid_quotes[i].spread_bps
            
            # Each layer should have wider spread
            assert curr_spread > prev_spread
            # Should follow multiplier pattern
            assert curr_spread == prev_spread * config.layer_spacing_multiplier
    
    async def test_performance_tracking(self):
        """Test performance metrics and PnL tracking."""
        config = MarketMakerConfig(
            name="PerfTest",
            symbol="BTCUSDT",
            max_position_usdt=Decimal("10000")
        )
        
        strategy = MarketMakingStrategy(config)
        
        # Simulate profitable trades
        buy_price = Decimal("49900")
        sell_price = Decimal("50100")
        quantity = Decimal("0.1")
        
        # Buy
        buy_order = Order(
            order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=buy_price,
            quantity=quantity,
            status="FILLED"
        )
        await strategy.on_order_filled(buy_order)
        
        # Sell at higher price
        sell_order = Order(
            order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=sell_price,
            quantity=quantity,
            status="FILLED"
        )
        await strategy.on_order_filled(sell_order)
        
        # Check PnL tracking
        # Profit = (sell_price - buy_price) * quantity
        expected_gross = (sell_price - buy_price) * quantity
        # Add maker rebates (simplified)
        maker_rebate = (buy_price + sell_price) * quantity * abs(config.maker_fee_bps) / Decimal("10000")
        
        assert strategy.state.pnl_usdt > 0, "Should have positive PnL"
        assert strategy.state.trades_count > 0, "Should track trade count"
    
    async def test_emergency_position_reduction(self):
        """Test emergency position reduction in extreme scenarios."""
        config = MarketMakerConfig(
            name="EmergencyTest",
            symbol="BTCUSDT",
            max_position_usdt=Decimal("5000"),
            max_daily_loss_pct=Decimal("0.01")  # 1% max loss
        )
        
        strategy = MarketMakingStrategy(config)
        
        # Simulate large loss
        strategy.daily_pnl = -config.max_position_usdt * Decimal("0.015")  # 1.5% loss
        
        # Create mock position
        position = MagicMock()
        position.realized_pnl = -Decimal("100")
        
        # Trigger position close event
        await strategy.on_position_closed(position)
        
        # Strategy should pause
        assert strategy.state.status == "PAUSED", "Should pause after hitting loss limit"
    
    async def test_spread_optimization(self):
        """Test dynamic spread optimization based on market conditions."""
        model = SpreadModel(
            base_spread_bps=Decimal("10"),
            min_spread_bps=Decimal("5"),
            max_spread_bps=Decimal("50")
        )
        
        # Test different market conditions
        conditions = [
            # Low volatility - should tighten spreads
            MarketConditions(
                current_price=Decimal("50000"),
                bid_price=Decimal("49950"),
                ask_price=Decimal("50050"),
                volatility=Decimal("0.0005"),  # Very low
                volume_24h=Decimal("1000000"),
                order_book_depth={"bids": [], "asks": []},
                recent_trades=[]
            ),
            # High volatility - should widen spreads
            MarketConditions(
                current_price=Decimal("50000"),
                bid_price=Decimal("49900"),
                ask_price=Decimal("50100"),
                volatility=Decimal("0.02"),  # High
                volume_24h=Decimal("1000000"),
                order_book_depth={"bids": [], "asks": []},
                recent_trades=[]
            ),
            # Normal volatility
            MarketConditions(
                current_price=Decimal("50000"),
                bid_price=Decimal("49975"),
                ask_price=Decimal("50025"),
                volatility=Decimal("0.005"),  # Normal
                volume_24h=Decimal("1000000"),
                order_book_depth={"bids": [], "asks": []},
                recent_trades=[]
            )
        ]
        
        spreads = []
        for condition in conditions:
            spread, _ = model.calculate_spread(condition)
            spreads.append(spread)
        
        # Verify optimization
        assert spreads[0] < spreads[2], "Low volatility should have tighter spreads"
        assert spreads[1] > spreads[2], "High volatility should have wider spreads"
        
        # All should be within bounds
        for spread in spreads:
            assert model.min_spread_bps <= spread <= model.max_spread_bps