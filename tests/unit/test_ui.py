"""Unit tests for Genesis UI module."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.core.models import Account, Position, PositionSide, TradingTier
from genesis.ui.app import GenesisApp
from genesis.ui.commands import CommandParser
from genesis.ui.dashboard import DashboardScreen
from genesis.ui.integration import UIIntegration
from genesis.ui.themes.zen_garden import ZEN_COLORS, get_pnl_color, get_status_color
from genesis.ui.widgets.pnl import PnLWidget
from genesis.ui.widgets.positions import PositionWidget


class TestPnLWidget:
    """Test P&L display widget."""

    def test_init(self):
        """Test widget initialization."""
        widget = PnLWidget()
        assert widget.current_pnl == Decimal("0.00")
        assert widget.daily_pnl == Decimal("0.00")
        assert widget.daily_pnl_pct == Decimal("0.00")
        assert widget.account_balance == Decimal("0.00")

    def test_render_no_pnl(self):
        """Test rendering with zero P&L."""
        widget = PnLWidget()
        output = widget.render()

        assert "P&L Dashboard" in output
        assert "$0.00" in output

    def test_render_profit(self):
        """Test rendering with profit."""
        widget = PnLWidget()
        widget.current_pnl = Decimal("150.50")
        widget.daily_pnl = Decimal("250.75")
        widget.account_balance = Decimal("5000.00")
        widget.daily_pnl_pct = Decimal("5.015")

        output = widget.render()

        assert "$150.50" in output
        assert "$250.75" in output
        assert "+5.02%" in output  # Properly rounded from 5.015
        assert "green" in output  # Profit color

    def test_render_loss(self):
        """Test rendering with loss."""
        widget = PnLWidget()
        widget.current_pnl = Decimal("-50.25")
        widget.daily_pnl = Decimal("-75.00")
        widget.account_balance = Decimal("5000.00")
        widget.daily_pnl_pct = Decimal("-1.50")

        output = widget.render()

        assert "$-50.25" in output
        assert "$-75.00" in output
        assert "-1.50%" in output
        assert "grey50" in output  # Loss color (not red)
        assert "red" not in output  # Never use red

    def test_set_mock_data(self):
        """Test setting mock data."""
        widget = PnLWidget()
        widget.set_mock_data(
            current=Decimal("100.00"),
            daily=Decimal("200.00"),
            balance=Decimal("1000.00")
        )

        assert widget.current_pnl == Decimal("100.00")
        assert widget.daily_pnl == Decimal("200.00")
        assert widget.account_balance == Decimal("1000.00")
        assert widget.daily_pnl_pct == Decimal("20.00")  # 200/1000 * 100


class TestPositionWidget:
    """Test position display widget."""

    def test_init(self):
        """Test widget initialization."""
        widget = PositionWidget()
        assert not widget.has_position
        assert widget.symbol == "BTC/USDT"
        assert widget.side == "NONE"
        assert widget.quantity == Decimal("0.00000000")
        assert not widget.show_details

    def test_render_no_position(self):
        """Test rendering with no position."""
        widget = PositionWidget()
        output = widget.render()

        assert "No Open Position" in output
        assert "Ready to trade" in output

    def test_render_long_position_profit(self):
        """Test rendering long position with profit."""
        widget = PositionWidget()
        widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("40000"),
            current=Decimal("41000"),
            stop_loss=Decimal("39000")
        )

        output = widget.render()

        assert "BTC/USDT" in output
        assert "LONG" in output
        assert "0.1" in output
        assert "40,000" in output  # Entry price
        assert "41,000" in output  # Current price
        assert "100" in output  # Unrealized P&L
        assert "green" in output  # Profit color
        assert "39,000" in output  # Stop loss

    def test_render_short_position_loss(self):
        """Test rendering short position with loss."""
        widget = PositionWidget()
        widget.set_mock_position(
            symbol="BTC/USDT",
            side="SHORT",
            qty=Decimal("0.1"),
            entry=Decimal("40000"),
            current=Decimal("41000")
        )

        output = widget.render()

        assert "SHORT" in output
        assert "-100" in output  # Loss on short
        assert "grey50" in output  # Loss color (not red)
        assert "red" not in output

    def test_toggle_details(self):
        """Test toggling position details."""
        widget = PositionWidget()
        widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("40000"),
            current=Decimal("40000")
        )

        assert not widget.show_details
        widget.toggle_details()
        assert widget.show_details
        widget.toggle_details()
        assert not widget.show_details

    def test_render_with_details(self):
        """Test rendering with details enabled."""
        widget = PositionWidget()
        widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("40000"),
            current=Decimal("40000"),
            stop_loss=Decimal("39000")
        )
        widget.show_details = True

        output = widget.render()

        assert "Position Value:" in output
        assert "Risk:" in output
        assert "Status: Active" in output


class TestCommandParser:
    """Test command parsing system."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return CommandParser()

    @pytest.mark.asyncio
    async def test_parse_buy_shorthand(self, parser):
        """Test parsing buy shorthand command."""
        result = await parser.parse("b100u")

        assert result.success
        assert "Buy order placed: $100.00" in result.message
        assert result.command_type == "buy"
        assert result.params["amount"] == Decimal("100")

    @pytest.mark.asyncio
    async def test_parse_sell_shorthand(self, parser):
        """Test parsing sell shorthand command."""
        result = await parser.parse("s50.5u")

        assert result.success
        assert "Sell order placed: $50.50" in result.message
        assert result.command_type == "sell"
        assert result.params["amount"] == Decimal("50.5")

    @pytest.mark.asyncio
    async def test_parse_buy_full(self, parser):
        """Test parsing full buy command."""
        result = await parser.parse("buy 1000")

        assert result.success
        assert "Buy order placed: $1000.00" in result.message

    @pytest.mark.asyncio
    async def test_parse_invalid_amount(self, parser):
        """Test parsing with invalid amount."""
        result = await parser.parse("buy abc")

        assert not result.success
        assert "Invalid" in result.message or "amount" in result.message

    @pytest.mark.asyncio
    async def test_parse_minimum_order(self, parser):
        """Test minimum order size validation."""
        result = await parser.parse("b5u")

        assert not result.success
        assert "Minimum order size is $10" in result.message

    @pytest.mark.asyncio
    async def test_parse_cancel(self, parser):
        """Test parsing cancel command."""
        result = await parser.parse("cancel")

        assert result.success
        assert "Cancelling all orders" in result.message
        assert result.command_type == "cancel"

    @pytest.mark.asyncio
    async def test_parse_status(self, parser):
        """Test parsing status command."""
        result = await parser.parse("status")

        assert result.success
        assert "System Status" in result.message
        assert result.command_type == "status"

    @pytest.mark.asyncio
    async def test_parse_help(self, parser):
        """Test parsing help command."""
        result = await parser.parse("help")

        assert result.success
        assert "Available Commands" in result.message
        assert "b100u" in result.message

    @pytest.mark.asyncio
    async def test_parse_unknown(self, parser):
        """Test parsing unknown command."""
        result = await parser.parse("unknown")

        assert not result.success
        assert "Unknown command" in result.message


class TestUIIntegration:
    """Test UI integration layer."""

    @pytest.fixture
    def integration(self):
        """Create integration instance with mocks."""
        integration = UIIntegration()

        # Mock components
        integration.account_manager = MagicMock()
        integration.risk_engine = MagicMock()
        integration.order_executor = AsyncMock()
        integration.gateway = AsyncMock()

        # Set up account
        integration.account_manager.account = Account(
            balance_usdt=Decimal("5000"),
            tier=TradingTier.SNIPER
        )

        # Set up session as mock
        integration.risk_engine.session = MagicMock()
        integration.risk_engine.session.realized_pnl = Decimal("100")
        integration.risk_engine.session.trade_count = 0

        return integration

    @pytest.mark.asyncio
    async def test_update_pnl_data(self, integration):
        """Test updating P&L widget data."""
        widget = PnLWidget()
        integration.pnl_widget = widget

        # Set up position
        integration.risk_engine.position = Position(
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            dollar_value=Decimal("4000")
        )
        integration.risk_engine.calculate_unrealized_pnl.return_value = Decimal("50")

        await integration.update_pnl_data()

        assert widget.account_balance == Decimal("5000")
        assert widget.daily_pnl == Decimal("100")
        assert widget.current_pnl == Decimal("50")

    @pytest.mark.asyncio
    async def test_update_position_data(self, integration):
        """Test updating position widget data."""
        widget = PositionWidget()
        integration.position_widget = widget

        # Set up position
        position = Position(
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            dollar_value=Decimal("4000"),
            stop_loss=Decimal("39000")
        )
        integration.risk_engine.position = position
        integration.risk_engine.calculate_unrealized_pnl.return_value = Decimal("100")

        # Mock ticker
        integration.gateway.get_ticker.return_value = {"last": 41000}

        await integration.update_position_data()

        assert widget.has_position
        assert widget.symbol == "BTC/USDT"
        assert widget.side == "LONG"
        assert widget.quantity == Decimal("0.1")
        assert widget.entry_price == Decimal("40000")
        assert widget.current_price == Decimal("41000")
        assert widget.unrealized_pnl == Decimal("100")
        assert widget.stop_loss == Decimal("39000")

    @pytest.mark.asyncio
    async def test_execute_buy_command(self, integration):
        """Test executing buy command."""
        integration.risk_engine.calculate_position_size.return_value = Decimal("0.1")
        integration.order_executor.execute_market_order.return_value = {
            "id": "123",
            "status": "FILLED"
        }

        result = await integration.execute_buy_command(Decimal("100"))

        assert result["success"]
        assert "Buy order executed" in result["message"]
        integration.order_executor.execute_market_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_sell_command(self, integration):
        """Test executing sell command."""
        # Set up position
        integration.risk_engine.position = Position(
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.1"),
            entry_price=Decimal("40000"),
            dollar_value=Decimal("4000")
        )

        integration.order_executor.execute_market_order.return_value = {
            "id": "124",
            "status": "FILLED"
        }

        result = await integration.execute_sell_command(Decimal("100"))

        assert result["success"]
        assert "Sell order executed" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, integration):
        """Test cancelling all orders."""
        integration.order_executor.cancel_all_orders.return_value = 3

        result = await integration.cancel_all_orders()

        assert result["success"]
        assert "Cancelled 3 orders" in result["message"]

    def test_get_system_status(self, integration):
        """Test getting system status."""
        integration.gateway.connected = True
        integration.risk_engine.session.trade_count = 5

        status = integration.get_system_status()

        assert status["exchange"] == "Connected"
        assert status["trading"] == "Active"
        assert status["tier"] == "SNIPER"
        assert status["balance"] == "$5000.00"
        assert status["daily_pnl"] == "$100.00"
        assert status["trades_today"] == 5


class TestZenGardenTheme:
    """Test Zen Garden color theme."""

    def test_colors_no_red(self):
        """Test that no red colors are used."""
        for color_name, color_value in ZEN_COLORS.items():
            assert "red" not in color_name.lower()
            # Check hex colors don't contain pure red
            if color_value.startswith("#"):
                # Pure red would have high R and low G,B
                r = int(color_value[1:3], 16)
                g = int(color_value[3:5], 16)
                b = int(color_value[5:7], 16)
                # Ensure it's not pure red (high red, low green/blue)
                if r > 200:
                    assert g > 100 or b > 100, f"{color_name} looks too red"

    def test_get_pnl_color(self):
        """Test P&L color selection."""
        assert get_pnl_color(100) == ZEN_COLORS["profit"]
        assert get_pnl_color(-100) == ZEN_COLORS["loss"]
        assert get_pnl_color(0) == ZEN_COLORS["neutral"]

    def test_get_status_color(self):
        """Test status color selection."""
        assert get_status_color("success") == ZEN_COLORS["success"]
        assert get_status_color("warning") == ZEN_COLORS["warning"]
        assert get_status_color("error") == ZEN_COLORS["warning"]  # Not red
        assert get_status_color("info") == ZEN_COLORS["info"]


class TestDashboardScreen:
    """Test dashboard screen functionality."""

    @pytest.mark.asyncio
    async def test_dashboard_init(self):
        """Test dashboard initialization."""
        screen = DashboardScreen()

        assert screen.pnl_widget is None
        assert screen.position_widget is None
        assert screen.command_input is None
        assert screen.integration is not None

    @pytest.mark.asyncio
    async def test_show_status_fade(self):
        """Test status message with fade."""
        screen = DashboardScreen()
        screen.status_message = MagicMock()

        screen.show_status("Test message", "success")

        screen.status_message.update.assert_called_with("[green]Test message[/green]")
        assert screen.status_timer is not None

    @pytest.mark.asyncio
    async def test_emergency_cancel(self):
        """Test emergency cancel orders."""
        screen = DashboardScreen()
        screen.integration = MagicMock()
        screen.integration.cancel_all_orders = AsyncMock(
            return_value={"success": True, "message": "Cancelled"}
        )
        screen.show_status = MagicMock()

        screen.emergency_cancel_orders()

        screen.show_status.assert_called_with(
            "EMERGENCY: Cancelling all orders...", "warning"
        )


class TestGenesisApp:
    """Test main Genesis application."""

    @pytest.mark.asyncio
    async def test_app_init(self):
        """Test app initialization."""
        app = GenesisApp()

        assert app.TITLE == "Genesis Trading Terminal"
        assert app.SUB_TITLE == "Zen Mode Active"
        assert app.UPDATE_INTERVAL == 0.1
        assert app.dashboard_screen is None

    @pytest.mark.asyncio
    async def test_app_bindings(self):
        """Test keyboard bindings are configured."""
        app = GenesisApp()

        # Check bindings exist
        binding_keys = [b.key for b in app.BINDINGS]
        assert "ctrl+q" in binding_keys
        assert "ctrl+h" in binding_keys
        assert "ctrl+p" in binding_keys
        assert "ctrl+c" in binding_keys


class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    def test_command_input_history(self):
        """Test command input history navigation."""
        from genesis.ui.commands import CommandInput

        input_widget = CommandInput()

        # Add commands to history
        input_widget.command_history = ["buy 100", "sell 50", "status"]

        # Test history navigation
        assert input_widget.history_index == -1
        assert len(input_widget.command_history) == 3

    @pytest.mark.asyncio
    async def test_command_parser_edge_cases(self):
        """Test edge cases in command parsing."""
        parser = CommandParser()

        # Test empty command
        result = await parser.parse("")
        assert not result.success
        assert "Empty command" in result.message

        # Test zero amount
        result = await parser.parse("b0u")
        assert not result.success
        assert "positive" in result.message.lower()

        # Test negative amount
        result = await parser.parse("buy -100")
        assert not result.success

    def test_pnl_widget_edge_cases(self):
        """Test P&L widget edge cases."""
        widget = PnLWidget()

        # Test with zero balance
        widget.set_mock_data(
            current=Decimal("0"),
            daily=Decimal("0"),
            balance=Decimal("0")
        )
        assert widget.daily_pnl_pct == Decimal("0.00")

        # Test negative values
        widget.set_mock_data(
            current=Decimal("-100"),
            daily=Decimal("-200"),
            balance=Decimal("1000")
        )
        assert widget.daily_pnl_pct == Decimal("-20.00")

    def test_position_widget_no_stop_loss(self):
        """Test position widget without stop loss."""
        widget = PositionWidget()
        widget.set_mock_position(
            symbol="ETH/USDT",
            side="LONG",
            qty=Decimal("1.0"),
            entry=Decimal("2000"),
            current=Decimal("2100"),
            stop_loss=None
        )

        output = widget.render()
        assert "ETH/USDT" in output
        assert "Stop Loss" not in output  # No stop loss displayed

    @pytest.mark.asyncio
    async def test_ui_integration_no_components(self):
        """Test UI integration without connected components."""
        integration = UIIntegration()

        # Test with no components connected
        result = await integration.execute_buy_command(Decimal("100"))
        assert not result["success"]
        assert "not connected" in result["message"]

        result = await integration.execute_sell_command(Decimal("50"))
        assert not result["success"]
        assert "not connected" in result["message"]

        result = await integration.cancel_all_orders()
        assert not result["success"]

        status = integration.get_connection_status()
        assert status == "Disconnected"

        system_status = integration.get_system_status()
        assert system_status["exchange"] == "Disconnected"
        assert system_status["trading"] == "Inactive"

    def test_zen_garden_theme_edge_cases(self):
        """Test theme edge cases."""
        from genesis.ui.themes.zen_garden import get_pnl_color, get_status_color

        # Test with very small values
        assert get_pnl_color(Decimal("0.01")) == ZEN_COLORS["profit"]
        assert get_pnl_color(Decimal("-0.01")) == ZEN_COLORS["loss"]

        # Test unknown status - returns default text color
        assert get_status_color("unknown") == ZEN_COLORS["text"]

    @pytest.mark.asyncio
    async def test_dashboard_update_cycle(self):
        """Test dashboard update cycle."""
        screen = DashboardScreen()
        screen.integration = MagicMock()
        screen.integration.update_pnl_data = AsyncMock()
        screen.integration.update_position_data = AsyncMock()

        # Mock widgets
        screen.pnl_widget = MagicMock()
        screen.position_widget = MagicMock()

        # Test update cycle
        await screen.update_widgets()

        screen.integration.update_pnl_data.assert_called_once()
        screen.integration.update_position_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_shortcuts(self):
        """Test command shortcuts processing."""
        parser = CommandParser()

        # Test various shorthand formats
        result = await parser.parse("b1000.50u")
        assert result.success
        assert result.params["amount"] == Decimal("1000.50")

        result = await parser.parse("s999.99u")
        assert result.success
        assert result.params["amount"] == Decimal("999.99")

    def test_position_widget_percentage_calculation(self):
        """Test position widget percentage calculations."""
        widget = PositionWidget()

        # Test with zero entry price
        widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.1"),
            entry=Decimal("0"),  # Edge case
            current=Decimal("40000")
        )

        output = widget.render()
        assert "+0.00%" in output  # Should handle division by zero

    @pytest.mark.asyncio
    async def test_integration_error_handling(self):
        """Test error handling in integration layer."""
        integration = UIIntegration()
        integration.account_manager = MagicMock()
        integration.risk_engine = MagicMock()
        integration.order_executor = AsyncMock()

        # Simulate exception in order execution
        integration.order_executor.execute_market_order.side_effect = Exception("Network error")

        result = await integration.execute_buy_command(Decimal("100"))
        assert not result["success"]
        assert "Network error" in result["message"]
