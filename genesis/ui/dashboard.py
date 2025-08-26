"""Dashboard screen with three-panel layout for Genesis trading terminal."""

import asyncio

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Static

from genesis.engine.state_machine import TierStateMachine
from genesis.ui.commands import CommandInput, CommandParser
from genesis.ui.integration import UIIntegration
from genesis.ui.widgets.iceberg_status import IcebergStatusWidget
from genesis.ui.widgets.pnl import PnLWidget
from genesis.ui.widgets.positions import PositionWidget
from genesis.ui.widgets.tier_gate_progress import TierGateProgressWidget


class DashboardScreen(Screen):
    """Main trading dashboard with three-panel layout."""

    DEFAULT_CSS = """
    DashboardScreen {
        layout: vertical;
    }
    
    #pnl-container {
        height: 25%;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    #position-container {
        height: 50%;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    #command-container {
        height: 25%;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    #status-area {
        dock: bottom;
        height: 1;
        margin: 0 1;
    }
    """

    def __init__(self, integration: UIIntegration | None = None, **kwargs):
        """Initialize the dashboard screen."""
        super().__init__(**kwargs)
        self.pnl_widget: PnLWidget | None = None
        self.position_widget: PositionWidget | None = None
        self.iceberg_widget: IcebergStatusWidget | None = None
        self.tier_gate_widget: TierGateProgressWidget | None = None
        self.command_input: CommandInput | None = None
        self.command_parser = CommandParser()
        self.status_message: Static | None = None
        self.status_timer: asyncio.Task | None = None
        self.integration = integration or UIIntegration()
        self.state_machine = TierStateMachine()  # Initialize tier state machine

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        # P&L Panel (Top)
        with Container(id="pnl-container"):
            self.pnl_widget = PnLWidget()
            yield self.pnl_widget

        # Position Panel (Middle-Top)
        with Container(id="position-container"):
            self.position_widget = PositionWidget()
            yield self.position_widget

        # Iceberg Status Panel (Middle-Bottom)
        with Container(id="iceberg-container"):
            self.iceberg_widget = IcebergStatusWidget()
            yield self.iceberg_widget

        # Tier Gate Progress Panel
        with Container(id="tier-gate-container"):
            self.tier_gate_widget = TierGateProgressWidget(self.state_machine)
            yield self.tier_gate_widget

        # Command Panel (Bottom)
        with Container(id="command-container"):
            self.command_input = CommandInput()
            yield self.command_input

        # Status message area
        self.status_message = Static("", id="status-area")
        yield self.status_message

    async def on_mount(self) -> None:
        """Handle screen mount."""
        # Connect widgets to integration
        if self.integration:
            self.integration.connect_widgets(self.pnl_widget, self.position_widget)

        # Focus on command input by default
        if self.command_input:
            self.command_input.focus()

    async def update_widgets(self) -> None:
        """Update all widgets with latest data."""
        # This method is called every 100ms from the app
        if self.integration:
            await self.integration.update_pnl_data()
            await self.integration.update_position_data()
        else:
            # Direct widget updates if no integration
            if self.pnl_widget:
                await self.pnl_widget.update_data()
            if self.position_widget:
                await self.position_widget.update_data()

    def show_status(self, message: str, level: str = "info") -> None:
        """Show a status message with 3-second fade."""
        if self.status_message:
            # Cancel existing timer
            if self.status_timer:
                self.status_timer.cancel()

            # Set message with appropriate styling
            if level == "success":
                self.status_message.update(f"[green]{message}[/green]")
            elif level == "warning":
                self.status_message.update(f"[yellow]{message}[/yellow]")
            else:
                self.status_message.update(message)

            # Start fade timer
            self.status_timer = asyncio.create_task(self._fade_status())

    async def _fade_status(self) -> None:
        """Fade out status message after 3 seconds."""
        await asyncio.sleep(3)
        if self.status_message:
            self.status_message.update("")

    def show_help(self) -> None:
        """Display help information."""
        help_text = """[bold]Keyboard Shortcuts:[/bold]
Ctrl+C - Emergency Cancel All Orders
Ctrl+P - Toggle Position Details
Ctrl+H - Show This Help
Ctrl+Q - Quit Application

[bold]Commands:[/bold]
b100u - Buy $100 USDT worth
s50u - Sell $50 USDT worth
cancel - Cancel all orders
status - Show current status"""
        self.show_status(help_text, "info")

    def toggle_position_details(self) -> None:
        """Toggle detailed position view."""
        if self.position_widget:
            self.position_widget.toggle_details()
            self.show_status("Position details toggled", "info")

    def emergency_cancel_orders(self) -> None:
        """Emergency cancel all orders."""
        self.show_status("EMERGENCY: Cancelling all orders...", "warning")

        if self.integration:
            asyncio.create_task(self._execute_cancel_all())

    async def _execute_cancel_all(self) -> None:
        """Execute cancel all orders through integration."""
        result = await self.integration.cancel_all_orders()
        if result["success"]:
            self.show_status(result["message"], "success")
        else:
            self.show_status(result["message"], "warning")

    async def handle_command(self, command: str) -> None:
        """Process a command from the input."""
        result = await self.command_parser.parse(command)

        if result.success:
            self.show_status(result.message, "success")
        else:
            self.show_status(result.message, "warning")
