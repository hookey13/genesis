
"""Position display widget for Genesis trading terminal."""

from decimal import Decimal

from textual.reactive import reactive
from textual.widgets import Static


class PositionWidget(Static):
    """Widget for displaying current position information."""

    # Reactive position data
    has_position = reactive(False)
    symbol = reactive("BTC/USDT")
    side = reactive("NONE")
    quantity = reactive(Decimal("0.00000000"))
    entry_price = reactive(Decimal("0.00"))
    current_price = reactive(Decimal("0.00"))
    unrealized_pnl = reactive(Decimal("0.00"))
    stop_loss: Decimal | None = reactive(None)

    # Display mode
    show_details = reactive(False)

    DEFAULT_CSS = """
    PositionWidget {
        content-align: center middle;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the position widget."""
        super().__init__("No Position", **kwargs)

    def render(self) -> str:
        """Render the position display."""
        if not self.has_position:
            return self._render_no_position()

        # Color coding for P&L
        pnl_color = "green" if self.unrealized_pnl >= 0 else "grey50"
        side_color = "cyan" if self.side == "LONG" else "magenta"

        # Format values
        qty_str = f"{self.quantity:.8f}".rstrip("0").rstrip(".")
        entry_str = f"${self.entry_price:,.2f}"
        current_str = f"${self.current_price:,.2f}"
        pnl_str = f"${self.unrealized_pnl:,.2f}"

        # Calculate percentage change
        if self.entry_price > 0:
            pct_change = (
                (self.current_price - self.entry_price) / self.entry_price
            ) * 100
            pct_str = f"{pct_change:+.2f}%"
        else:
            pct_str = "+0.00%"

        # Build display
        lines = [
            "[bold]═══ Current Position ═══[/bold]",
            "",
            f"[bold]{self.symbol}[/bold] - [{side_color}]{self.side}[/{side_color}]",
            f"Quantity: {qty_str}",
            "",
            f"Entry: {entry_str}",
            f"Current: {current_str} ({pct_str})",
            "",
            f"[bold]Unrealized P&L:[/bold] [{pnl_color}]{pnl_str}[/{pnl_color}]",
        ]

        # Add stop loss if set
        if self.stop_loss:
            sl_str = f"${self.stop_loss:,.2f}"
            lines.append(f"[yellow]Stop Loss: {sl_str}[/yellow]")

        # Add details if enabled
        if self.show_details:
            lines.extend(["", "[dim]─────────────────────[/dim]", self._get_details()])

        return "\n".join(lines)

    def _render_no_position(self) -> str:
        """Render when no position is open."""
        return """[bold]═══ Current Position ═══[/bold]

[dim]No Open Position[/dim]

Ready to trade
Use command input below to place orders"""

    def _get_details(self) -> str:
        """Get detailed position information."""
        # Calculate position value
        position_value = self.quantity * self.current_price

        # Calculate risk if stop loss is set
        if self.stop_loss:
            risk_per_unit = abs(self.entry_price - self.stop_loss)
            total_risk = risk_per_unit * self.quantity
            risk_str = f"Risk: ${total_risk:,.2f}"
        else:
            risk_str = "Risk: No Stop Loss"

        return f"""[dim]Position Value: ${position_value:,.2f}
{risk_str}
Status: Active[/dim]"""

    async def update_data(self) -> None:
        """Update position data from connected components."""
        # TODO: Connect to RiskEngine for position data
        # For now, method is ready for integration
        pass

    def toggle_details(self) -> None:
        """Toggle detailed position view."""
        self.show_details = not self.show_details

    def watch_has_position(self, value: bool) -> None:
        """React to position status changes."""
        self.update(self.render())

    def watch_unrealized_pnl(self, value: Decimal) -> None:
        """React to P&L changes."""
        self.update(self.render())

    def watch_current_price(self, value: Decimal) -> None:
        """React to price changes."""
        self.update(self.render())

    def watch_show_details(self, value: bool) -> None:
        """React to detail toggle."""
        self.update(self.render())

    def set_mock_position(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        entry: Decimal,
        current: Decimal,
        stop_loss: Decimal | None = None,
    ) -> None:
        """Set mock position data for testing."""
        self.has_position = True
        self.symbol = symbol
        self.side = side
        self.quantity = qty
        self.entry_price = entry
        self.current_price = current
        self.stop_loss = stop_loss

        # Calculate unrealized P&L
        if side == "LONG":
            self.unrealized_pnl = (current - entry) * qty
        else:  # SHORT
            self.unrealized_pnl = (entry - current) * qty
