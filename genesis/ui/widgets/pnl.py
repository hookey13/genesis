"""P&L display widget for Genesis trading terminal."""

from decimal import Decimal

from textual.reactive import reactive
from textual.widgets import Static


class PnLWidget(Static):
    """Widget for displaying P&L information with color coding."""

    # Reactive values that trigger updates
    current_pnl = reactive(Decimal("0.00"))
    daily_pnl = reactive(Decimal("0.00"))
    daily_pnl_pct = reactive(Decimal("0.00"))

    DEFAULT_CSS = """
    PnLWidget {
        content-align: center middle;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the P&L widget."""
        super().__init__("Loading P&L...", **kwargs)
        self.account_balance = Decimal("0.00")

    def render(self) -> str:
        """Render the P&L display."""
        # Color coding: green for profit, gray for loss (no red)
        current_color = "green" if self.current_pnl >= 0 else "grey50"
        daily_color = "green" if self.daily_pnl >= 0 else "grey50"

        # Format values to 2 decimal places
        current_str = f"${self.current_pnl:,.2f}"
        daily_str = f"${self.daily_pnl:,.2f}"
        daily_pct_str = f"{self.daily_pnl_pct:+.2f}%"

        # Build display
        lines = [
            "[bold]═══ P&L Dashboard ═══[/bold]",
            "",
            f"[bold]Current P&L:[/bold] [{current_color}]{current_str}[/{current_color}]",
            "",
            f"[bold]Daily P&L:[/bold] [{daily_color}]{daily_str} ({daily_pct_str})[/{daily_color}]",
            "",
            f"[dim]Account Balance: ${self.account_balance:,.2f}[/dim]"
        ]

        return "\n".join(lines)

    async def update_data(self) -> None:
        """Update P&L data from connected components."""
        # TODO: Connect to AccountManager and RiskEngine
        # For now, using mock data to demonstrate
        pass

    def watch_current_pnl(self, value: Decimal) -> None:
        """React to current P&L changes."""
        self.update(self.render())

    def watch_daily_pnl(self, value: Decimal) -> None:
        """React to daily P&L changes."""
        self.update(self.render())

    def watch_daily_pnl_pct(self, value: Decimal) -> None:
        """React to daily P&L percentage changes."""
        self.update(self.render())

    def set_mock_data(self, current: Decimal, daily: Decimal, balance: Decimal) -> None:
        """Set mock data for testing."""
        self.current_pnl = current
        self.daily_pnl = daily
        self.account_balance = balance

        # Calculate percentage
        if balance > 0:
            self.daily_pnl_pct = (daily / balance) * 100
        else:
            self.daily_pnl_pct = Decimal("0.00")
