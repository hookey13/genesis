
"""Main Textual application for Genesis trading system."""

import asyncio

from rich.console import Console
from textual.app import App
from textual.binding import Binding

from genesis.ui.dashboard import DashboardScreen
from genesis.ui.themes.zen_garden import apply_zen_theme_css


class GenesisApp(App):
    """Main Genesis trading terminal application."""

    CSS = apply_zen_theme_css()  # Apply Zen Garden theme
    TITLE = "Genesis Trading Terminal"
    SUB_TITLE = "Zen Mode Active"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+h", "help", "Help"),
        Binding("ctrl+p", "toggle_position", "Position Details"),
        Binding("ctrl+c", "emergency_cancel", "Cancel All Orders", priority=True),
    ]

    # Update interval in milliseconds
    UPDATE_INTERVAL = 0.1  # 100ms

    def __init__(self, **kwargs):
        """Initialize the Genesis application."""
        super().__init__(**kwargs)
        self.console = Console()
        self.update_task: asyncio.Task | None = None
        self.dashboard_screen: DashboardScreen | None = None

        # Event loop integration
        self.event_loop = asyncio.get_event_loop()

    async def on_mount(self) -> None:
        """Handle application startup."""
        # Push the dashboard screen
        self.dashboard_screen = DashboardScreen()
        await self.push_screen(self.dashboard_screen)

        # Start the update loop
        self.update_task = asyncio.create_task(self._update_loop())

    async def on_unmount(self) -> None:
        """Handle application shutdown."""
        # Cancel update task
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

    async def _update_loop(self) -> None:
        """Main update loop running at 100ms intervals."""
        while True:
            try:
                await asyncio.sleep(self.UPDATE_INTERVAL)

                # Trigger dashboard update
                if self.dashboard_screen:
                    await self.dashboard_screen.update_widgets()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but don't crash on update errors
                self.console.print(f"Update error: {e}")

    def action_quit(self) -> None:
        """Quit the application with confirmation."""
        # TODO: Add confirmation dialog in future
        self.exit()

    def action_help(self) -> None:
        """Show help information."""
        if self.dashboard_screen:
            self.dashboard_screen.show_help()

    def action_toggle_position(self) -> None:
        """Toggle position details display."""
        if self.dashboard_screen:
            self.dashboard_screen.toggle_position_details()

    def action_emergency_cancel(self) -> None:
        """Emergency cancel all orders."""
        if self.dashboard_screen:
            self.dashboard_screen.emergency_cancel_orders()

    def on_resize(self, event) -> None:
        """Handle terminal resize events."""
        # Textual handles most resize automatically
        # This is for any custom resize logic
        pass


def run_app(**kwargs) -> None:
    """Run the Genesis terminal application."""
    app = GenesisApp(**kwargs)
    app.run()
