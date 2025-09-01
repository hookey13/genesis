
"""Position display widget for Genesis trading terminal with enhanced risk metrics."""

from datetime import UTC, datetime
from decimal import Decimal

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
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
    stop_loss = reactive(None)  # Optional[Decimal]

    # Paper trading mode
    paper_trading_mode = reactive(False)
    paper_session_id = reactive("")

    # Display mode
    show_details = reactive(False)
    
    # Risk metrics
    risk_reward_ratio = reactive(Decimal("0.00"))
    position_risk_pct = reactive(Decimal("0.00"))
    position_risk_percentage = reactive(Decimal("0.00"))  # Alias for compatibility
    account_risk_percentage = reactive(Decimal("0.00"))
    time_in_position = reactive(0)  # seconds
    max_profit = reactive(Decimal("0.00"))
    max_loss = reactive(Decimal("0.00"))
    
    # Market metrics
    bid_ask_spread = reactive(Decimal("0.00"))
    market_volatility = reactive(Decimal("0.00"))
    volume_24h = reactive(Decimal("0.00"))

    DEFAULT_CSS = """
    PositionWidget {
        content-align: center middle;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the position widget."""
        super().__init__("No Position", **kwargs)
        self.entry_time = None
        self.position_history = []  # Track position P&L over time

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

        # Build display header
        if self.paper_trading_mode:
            header = "[bold yellow]═══ Paper Trading Position ═══[/bold yellow]"
        else:
            header = "[bold]═══ Current Position ═══[/bold]"

        # Build display
        lines = [
            header,
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
            
        # Add risk metrics section
        lines.extend(self._render_risk_metrics())

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

    def _render_risk_metrics(self) -> list[str]:
        """Render risk metrics section."""
        lines = ["", "[bold]Risk Metrics:[/bold]"]
        
        # Risk/Reward ratio
        if self.risk_reward_ratio != 0:
            rr_color = "green" if self.risk_reward_ratio >= 2 else "yellow" if self.risk_reward_ratio >= 1 else "grey50"
            lines.append(f"R:R Ratio: [{rr_color}]{self.risk_reward_ratio:.2f}:1[/{rr_color}]")
        
        # Position risk percentage
        if self.position_risk_pct != 0:
            risk_color = "green" if self.position_risk_pct <= 1 else "yellow" if self.position_risk_pct <= 2 else "red"
            lines.append(f"Position Risk: [{risk_color}]{self.position_risk_pct:.2f}%[/{risk_color}]")
        
        # Time in position
        if self.time_in_position > 0:
            hours = self.time_in_position // 3600
            minutes = (self.time_in_position % 3600) // 60
            time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            lines.append(f"Duration: {time_str}")
        
        # Max profit/loss
        if self.max_profit != 0 or self.max_loss != 0:
            lines.append(f"Max Profit: [green]${self.max_profit:,.2f}[/green]")
            lines.append(f"Max Loss: [grey50]${abs(self.max_loss):,.2f}[/grey50]")
        
        # Market metrics
        if self.show_details and (self.bid_ask_spread != 0 or self.market_volatility != 0):
            lines.extend([
                "",
                "[bold]Market Conditions:[/bold]",
                f"Spread: {self.bid_ask_spread:.4f}%",
                f"Volatility: {self.market_volatility:.2f}%",
                f"24h Volume: ${self.volume_24h:,.0f}",
            ])
        
        return lines if len(lines) > 2 else []

    def calculate_risk_metrics(self) -> None:
        """Calculate risk metrics based on current position."""
        if not self.has_position:
            return
            
        # Calculate risk/reward ratio if stop loss is set
        if self.stop_loss:
            risk = abs(self.entry_price - self.stop_loss) * self.quantity
            
            # Assume take profit at 2x risk for calculation
            potential_reward = risk * 2
            self.risk_reward_ratio = potential_reward / risk if risk > 0 else Decimal("0")
            
            # Calculate position risk as percentage of account (needs account balance)
            # This would be connected to account manager in production
            
        # Update time in position
        if self.entry_time:
            self.time_in_position = int((datetime.now(UTC) - self.entry_time).total_seconds())
            
        # Track max profit/loss
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl

    def set_mock_position(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
        entry: Decimal,
        current: Decimal,
        stop_loss=None,  # Optional[Decimal]
    ) -> None:
        """Set mock position data for testing."""
        self.has_position = True
        self.symbol = symbol
        self.side = side
        self.quantity = qty
        self.entry_price = entry
        self.current_price = current
        self.stop_loss = stop_loss
        self.entry_time = datetime.now(UTC)

        # Calculate unrealized P&L
        if side == "LONG":
            self.unrealized_pnl = (current - entry) * qty
        else:  # SHORT
            self.unrealized_pnl = (entry - current) * qty
            
        # Calculate risk metrics
        self.calculate_risk_metrics()
