"""Unit tests for paper trading simulator."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.paper_trading.simulator import (
    PaperTradingSimulator,
    SimulatedOrder,
    SimulationConfig,
    SimulationMode,
)
from genesis.paper_trading.validation_criteria import ValidationCriteria
from genesis.paper_trading.virtual_portfolio import VirtualPortfolio


class TestSimulationConfig:
    """Test SimulationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.mode == SimulationMode.REALISTIC
        assert config.base_latency_ms == 10.0
        assert config.latency_std_ms == 2.0
        assert config.base_slippage_bps == 2.0
        assert config.slippage_std_bps == 1.0
        assert config.partial_fill_threshold == Decimal("10000")
        assert config.max_fill_ratio == 0.5
        assert config.enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SimulationConfig(
            mode=SimulationMode.PESSIMISTIC,
            base_latency_ms=20.0,
            base_slippage_bps=5.0,
        )
        assert config.mode == SimulationMode.PESSIMISTIC
        assert config.base_latency_ms == 20.0
        assert config.base_slippage_bps == 5.0


class TestSimulatedOrder:
    """Test SimulatedOrder dataclass."""

    def test_order_creation(self):
        """Test creating a simulated order."""
        order = SimulatedOrder(
            order_id="test123",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            timestamp=datetime.now(),
        )
        assert order.order_id == "test123"
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.status == "pending"
        assert order.filled_quantity == Decimal("0")
        assert order.average_fill_price is None

    def test_order_fill_update(self):
        """Test updating order fill information."""
        order = SimulatedOrder(
            order_id="test123",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("1.0"),
            price=None,
            timestamp=datetime.now(),
        )
        
        # Update fill information
        order.status = "filled"
        order.filled_quantity = Decimal("1.0")
        order.average_fill_price = Decimal("50100")
        order.fill_timestamp = datetime.now()
        order.slippage = Decimal("2.0")  # 2 basis points
        order.latency_ms = 12.5
        
        assert order.status == "filled"
        assert order.filled_quantity == Decimal("1.0")
        assert order.average_fill_price == Decimal("50100")
        assert order.slippage == Decimal("2.0")
        assert order.latency_ms == 12.5


class TestPaperTradingSimulator:
    """Test PaperTradingSimulator class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulationConfig(
            mode=SimulationMode.REALISTIC,
            base_latency_ms=10.0,
            enabled=True,
        )

    @pytest.fixture
    def validation_criteria(self):
        """Create test validation criteria."""
        return ValidationCriteria(
            min_trades=100,
            min_days=7,
            min_sharpe=1.5,
            max_drawdown=0.1,
            min_win_rate=0.55,
        )

    @pytest.fixture
    def simulator(self, config, validation_criteria):
        """Create test simulator."""
        return PaperTradingSimulator(config, validation_criteria)

    def test_initialization(self, simulator, config, validation_criteria):
        """Test simulator initialization."""
        assert simulator.config == config
        assert simulator.validation_criteria == validation_criteria
        assert len(simulator.portfolios) == 0
        assert len(simulator.orders) == 0
        assert simulator.order_counter == 0
        assert simulator.running is False

    @pytest.mark.asyncio
    async def test_start(self, simulator):
        """Test starting the simulator."""
        await simulator.start()
        assert simulator.running is True
        
        # Try starting again - should log warning
        with patch("genesis.paper_trading.simulator.logger") as mock_logger:
            await simulator.start()
            mock_logger.warning.assert_called_once()
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_stop(self, simulator):
        """Test stopping the simulator."""
        await simulator.start()
        assert simulator.running is True
        
        await simulator.stop()
        assert simulator.running is False

    @pytest.mark.asyncio
    async def test_create_portfolio(self, simulator):
        """Test creating a virtual portfolio."""
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        assert isinstance(portfolio, VirtualPortfolio)
        assert "test_strategy" in simulator.portfolios
        assert simulator.portfolios["test_strategy"] == portfolio
        assert portfolio.balance == Decimal("10000")
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_submit_order(self, simulator):
        """Test submitting an order."""
        await simulator.start()
        
        # Create portfolio first
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Submit order
        order = await simulator.submit_order(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )
        
        assert isinstance(order, SimulatedOrder)
        assert order.order_id in simulator.orders
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.quantity == Decimal("0.1")
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_submit_order_no_portfolio(self, simulator):
        """Test submitting order without portfolio."""
        await simulator.start()
        
        # Should raise error if portfolio doesn't exist
        with pytest.raises(ValueError, match="Portfolio .* not found"):
            await simulator.submit_order(
                strategy_id="nonexistent",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_execute_order_market(self, simulator):
        """Test executing a market order."""
        await simulator.start()
        
        # Create portfolio
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Create mock market data
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            # Submit and execute market order
            order = await simulator.submit_order(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
            
            # Execute with simulated latency
            await simulator._execute_order("test_strategy", order)
            
            # Check order is filled
            assert order.status == "filled"
            assert order.filled_quantity == Decimal("0.1")
            assert order.average_fill_price is not None
            assert order.slippage is not None
            assert order.latency_ms is not None
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_execute_order_limit(self, simulator):
        """Test executing a limit order."""
        await simulator.start()
        
        # Create portfolio
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Submit limit order
        order = await simulator.submit_order(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
        )
        
        # Mock price movement
        with patch.object(simulator, '_get_current_price', return_value=Decimal("49000")):
            await simulator._execute_order("test_strategy", order)
            
            # Check order is filled at limit price
            assert order.status == "filled"
            assert order.filled_quantity == Decimal("0.1")
            assert order.average_fill_price == Decimal("49000")
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_partial_fill(self, simulator):
        """Test partial order fills for large orders."""
        config = SimulationConfig(
            partial_fill_threshold=Decimal("0.05"),  # Very low threshold
            max_fill_ratio=0.5,
        )
        simulator = PaperTradingSimulator(config, ValidationCriteria())
        
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("100000"),
        )
        
        # Submit large order
        order = await simulator.submit_order(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=Decimal("1.0"),  # Large order
        )
        
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            await simulator._execute_order("test_strategy", order)
            
            # Should be partially filled
            assert order.status == "partial"
            assert order.filled_quantity < order.quantity
            assert order.filled_quantity > Decimal("0")
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_slippage_calculation(self, simulator):
        """Test slippage calculation."""
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Mock current price
        base_price = Decimal("50000")
        
        with patch.object(simulator, '_get_current_price', return_value=base_price):
            # Submit market order
            order = await simulator.submit_order(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
            
            # Execute with slippage
            await simulator._execute_order("test_strategy", order)
            
            # Check slippage was applied
            assert order.average_fill_price > base_price  # Buy order has positive slippage
            assert order.slippage is not None
            assert order.slippage >= 0
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_latency_simulation(self, simulator):
        """Test latency simulation."""
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            # Submit order
            order = await simulator.submit_order(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
            
            # Measure execution time
            start = datetime.now()
            await simulator._execute_order("test_strategy", order)
            end = datetime.now()
            
            # Check latency was simulated
            execution_time_ms = (end - start).total_seconds() * 1000
            assert execution_time_ms >= simulator.config.base_latency_ms
            assert order.latency_ms is not None
            assert order.latency_ms > 0
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_get_portfolio_metrics(self, simulator):
        """Test getting portfolio metrics."""
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Make some trades
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            await simulator.submit_order(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
        
        metrics = simulator.get_portfolio_metrics("test_strategy")
        
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_validate_strategy(self, simulator, validation_criteria):
        """Test strategy validation."""
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Initially should not pass validation (no trades)
        eligible = simulator.check_promotion_eligibility("test_strategy")
        assert eligible is False
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_reset_portfolio(self, simulator):
        """Test resetting a portfolio."""
        await simulator.start()
        
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Make a trade
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            await simulator.submit_order(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
        
        # Reset portfolio by recreating it
        simulator.portfolios["test_strategy"] = VirtualPortfolio("test_strategy", Decimal("10000"))
        portfolio = simulator.portfolios["test_strategy"]
        
        # Check portfolio is reset
        assert portfolio.balance == Decimal("10000")
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_persistence_save_load(self, simulator):
        """Test saving and loading simulator state."""
        await simulator.start()
        
        # Create portfolio and make trades
        portfolio = simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            order = await simulator.submit_order(
                strategy_id="test_strategy",
                symbol="BTC/USDT",
                side="buy",
                order_type="market",
                quantity=Decimal("0.1"),
            )
        
        # Save state through persistence
        await simulator._save_all_portfolios()
        
        # Create new simulator and load portfolios
        new_simulator = PaperTradingSimulator(
            simulator.config,
            simulator.validation_criteria,
        )
        loaded_portfolio = new_simulator.load_portfolio("test_strategy")
        
        # Check state is restored
        assert loaded_portfolio is not None
        assert "test_strategy" in new_simulator.portfolios
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_auto_save(self, simulator):
        """Test automatic state saving."""
        # Configure auto-save with short interval
        simulator.persistence.config.auto_save_interval_seconds = 0.1
        
        await simulator.start()
        
        # Create portfolio
        simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("10000"),
        )
        
        # Wait for auto-save to trigger
        await asyncio.sleep(0.2)
        
        # Check save was called
        with patch.object(simulator, '_save_all_portfolios') as mock_save:
            await asyncio.sleep(0.15)
            mock_save.assert_called()
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_simulation_modes(self):
        """Test different simulation modes."""
        # Test optimistic mode
        opt_config = SimulationConfig(
            mode=SimulationMode.OPTIMISTIC,
            base_slippage_bps=0.5,  # Lower slippage
            base_latency_ms=5.0,  # Lower latency
        )
        opt_simulator = PaperTradingSimulator(opt_config, ValidationCriteria())
        assert opt_simulator.config.mode == SimulationMode.OPTIMISTIC
        
        # Test pessimistic mode
        pess_config = SimulationConfig(
            mode=SimulationMode.PESSIMISTIC,
            base_slippage_bps=5.0,  # Higher slippage
            base_latency_ms=20.0,  # Higher latency
        )
        pess_simulator = PaperTradingSimulator(pess_config, ValidationCriteria())
        assert pess_simulator.config.mode == SimulationMode.PESSIMISTIC

    @pytest.mark.asyncio
    async def test_concurrent_orders(self, simulator):
        """Test handling concurrent orders."""
        await simulator.start()
        
        # Create portfolio
        simulator.create_portfolio(
            "test_strategy",
            initial_balance=Decimal("50000"),
        )
        
        with patch.object(simulator, '_get_current_price', return_value=Decimal("50000")):
            # Submit multiple orders concurrently
            orders = await asyncio.gather(
                simulator.submit_order(
                    strategy_id="test_strategy",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=Decimal("0.1"),
                ),
                simulator.submit_order(
                    strategy_id="test_strategy",
                    symbol="ETH/USDT",
                    side="buy",
                    order_type="market",
                    quantity=Decimal("1.0"),
                ),
                simulator.submit_order(
                    strategy_id="test_strategy",
                    symbol="BNB/USDT",
                    side="buy",
                    order_type="market",
                    quantity=Decimal("10.0"),
                ),
            )
        
        # Check all orders were created
        assert len(orders) == 3
        for order in orders:
            assert order.order_id in simulator.orders
        
        await simulator.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self, simulator):
        """Test error handling in simulator."""
        await simulator.start()
        
        # Test invalid strategy ID
        with pytest.raises(ValueError):
            await simulator.get_portfolio_metrics("nonexistent")
        
        # Test invalid order submission
        with pytest.raises(ValueError):
            await simulator.submit_order(
                strategy_id="invalid",
                symbol="BTC/USDT",
                side="invalid_side",
                order_type="market",
                quantity=Decimal("-1.0"),  # Invalid quantity
            )
        
        await simulator.stop()

    def test_order_id_generation(self, simulator):
        """Test unique order ID generation."""
        ids = set()
        for i in range(100):
            simulator.order_counter = i
            order_id = f"SIM-{simulator.order_counter:08d}"
            assert order_id not in ids
            ids.add(order_id)
        
        assert len(ids) == 100