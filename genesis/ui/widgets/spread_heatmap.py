"""
Spread Heatmap Widget

Terminal UI widget for visualizing spread data as an interactive heatmap
with color gradients and sorting capabilities.
"""

from decimal import Decimal
from typing import Literal, Optional

import structlog
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Static

from genesis.analytics.spread_analyzer import SpreadMetrics

logger = structlog.get_logger(__name__)

SortMode = Literal["spread", "volume", "volatility", "symbol"]


class SpreadHeatmap(Widget):
    """
    Interactive heatmap widget for spread visualization
    with real-time updates and keyboard navigation
    """

    # Reactive attributes for state management
    sort_mode: reactive[SortMode] = reactive("spread")
    selected_symbol: reactive[Optional[str]] = reactive(None)
    refresh_interval: reactive[int] = reactive(1)  # seconds

    # CSS for styling
    DEFAULT_CSS = """
    SpreadHeatmap {
        height: 100%;
        width: 100%;
        background: $surface;
        border: solid $primary;
    }

    .heatmap-container {
        padding: 1;
    }

    .control-bar {
        height: 3;
        margin-bottom: 1;
    }

    .spread-table {
        height: 1fr;
    }
    """

    def __init__(
        self,
        spread_data: Optional[dict[str, SpreadMetrics]] = None,
        update_callback: Optional[callable] = None,
        **kwargs,
    ):
        """
        Initialize spread heatmap widget

        Args:
            spread_data: Initial spread metrics data
            update_callback: Callback for fetching updated data
        """
        super().__init__(**kwargs)
        self.spread_data = spread_data or {}
        self.update_callback = update_callback
        self._logger = logger.bind(component="SpreadHeatmap")

    def compose(self) -> ComposeResult:
        """Compose widget layout"""
        with Vertical(classes="heatmap-container"):
            # Control bar
            with Horizontal(classes="control-bar"):
                yield Label("Sort by: ")
                yield Button("Spread", id="sort-spread", variant="primary")
                yield Button("Volume", id="sort-volume")
                yield Button("Volatility", id="sort-volatility")
                yield Button("Symbol", id="sort-symbol")
                yield Label(f"  Refresh: {self.refresh_interval}s", id="refresh-label")

            # Main heatmap display
            yield Static(
                self._render_heatmap(), classes="spread-table", id="heatmap-display"
            )

    def _render_heatmap(self) -> RenderableType:
        """
        Render the spread heatmap table

        Returns:
            Rich renderable for the heatmap
        """
        if not self.spread_data:
            return Panel("[dim]No spread data available[/dim]", title="Spread Heatmap")

        # Create table
        table = Table(
            title="Spread Heatmap",
            show_header=True,
            header_style="bold magenta",
            show_lines=True,
            expand=True,
        )

        # Add columns
        table.add_column("Symbol", style="cyan", width=12)
        table.add_column("Spread (bps)", justify="right", width=12)
        table.add_column("Bid", justify="right", width=12)
        table.add_column("Ask", justify="right", width=12)
        table.add_column("Volatility", justify="right", width=10)
        table.add_column("Status", justify="center", width=10)

        # Sort data
        sorted_data = self._sort_spread_data()

        # Add rows with color coding
        for symbol, metrics in sorted_data:
            spread_color = self._get_spread_color(metrics.spread_bps)
            volatility_color = self._get_volatility_color(metrics.volatility)
            status = self._get_spread_status(metrics.spread_bps)

            # Highlight selected symbol
            row_style = "bold yellow" if symbol == self.selected_symbol else ""

            table.add_row(
                Text(symbol, style=row_style or "cyan"),
                Text(f"{metrics.spread_bps:.2f}", style=spread_color),
                Text(f"{metrics.bid_price:.4f}", style=row_style),
                Text(f"{metrics.ask_price:.4f}", style=row_style),
                Text(f"{metrics.volatility:.2f}", style=volatility_color),
                Text(status[0], style=status[1]),
            )

        return table

    def _sort_spread_data(self) -> list[tuple]:
        """
        Sort spread data based on current sort mode

        Returns:
            Sorted list of (symbol, metrics) tuples
        """
        items = list(self.spread_data.items())

        if self.sort_mode == "spread":
            items.sort(key=lambda x: x[1].spread_bps)
        elif self.sort_mode == "volume":
            items.sort(key=lambda x: x[1].bid_volume + x[1].ask_volume, reverse=True)
        elif self.sort_mode == "volatility":
            items.sort(key=lambda x: x[1].volatility, reverse=True)
        else:  # symbol
            items.sort(key=lambda x: x[0])

        return items

    def _get_spread_color(self, spread_bps: Decimal) -> str:
        """
        Get color for spread value (green=tight, red=wide)

        Args:
            spread_bps: Spread in basis points

        Returns:
            Rich color style string
        """
        if spread_bps < Decimal("5"):
            return "green"
        elif spread_bps < Decimal("10"):
            return "bright_green"
        elif spread_bps < Decimal("20"):
            return "yellow"
        elif spread_bps < Decimal("30"):
            return "bright_yellow"
        elif spread_bps < Decimal("50"):
            return "red"
        else:
            return "bright_red"

    def _get_volatility_color(self, volatility: Decimal) -> str:
        """
        Get color for volatility value

        Args:
            volatility: Volatility value

        Returns:
            Rich color style string
        """
        if volatility < Decimal("1"):
            return "blue"
        elif volatility < Decimal("3"):
            return "cyan"
        elif volatility < Decimal("5"):
            return "yellow"
        elif volatility < Decimal("10"):
            return "magenta"
        else:
            return "red"

    def _get_spread_status(self, spread_bps: Decimal) -> tuple:
        """
        Get status indicator for spread

        Args:
            spread_bps: Spread in basis points

        Returns:
            Tuple of (status_text, color_style)
        """
        if spread_bps < Decimal("5"):
            return ("TIGHT", "green")
        elif spread_bps < Decimal("10"):
            return ("GOOD", "bright_green")
        elif spread_bps < Decimal("20"):
            return ("NORMAL", "yellow")
        elif spread_bps < Decimal("30"):
            return ("WIDE", "bright_yellow")
        else:
            return ("EXTREME", "bright_red")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        button_id = event.button.id

        # Update sort mode
        if button_id == "sort-spread":
            self.sort_mode = "spread"
        elif button_id == "sort-volume":
            self.sort_mode = "volume"
        elif button_id == "sort-volatility":
            self.sort_mode = "volatility"
        elif button_id == "sort-symbol":
            self.sort_mode = "symbol"

        # Update button styles
        for btn_id in ["sort-spread", "sort-volume", "sort-volatility", "sort-symbol"]:
            btn = self.query_one(f"#{btn_id}", Button)
            btn.variant = "primary" if btn_id == button_id else "default"

        # Refresh display
        self._refresh_display()

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard events"""
        key = event.key

        if key == "s":
            # Cycle through sort modes
            modes = ["spread", "volume", "volatility", "symbol"]
            current_idx = modes.index(self.sort_mode)
            self.sort_mode = modes[(current_idx + 1) % len(modes)]
            self._refresh_display()

        elif key == "r":
            # Force refresh
            self._fetch_updates()
            self._refresh_display()

        elif key in ["up", "down"]:
            # Navigate selection
            sorted_data = self._sort_spread_data()
            if sorted_data:
                symbols = [s for s, _ in sorted_data]

                if self.selected_symbol is None:
                    self.selected_symbol = symbols[0]
                else:
                    current_idx = symbols.index(self.selected_symbol)
                    if key == "up":
                        new_idx = max(0, current_idx - 1)
                    else:
                        new_idx = min(len(symbols) - 1, current_idx + 1)
                    self.selected_symbol = symbols[new_idx]

                self._refresh_display()

        elif key in ["+", "-"]:
            # Adjust refresh interval
            if key == "+":
                self.refresh_interval = min(10, self.refresh_interval + 1)
            else:
                self.refresh_interval = max(1, self.refresh_interval - 1)

            # Update label
            label = self.query_one("#refresh-label", Label)
            label.update(f"  Refresh: {self.refresh_interval}s")

    def _refresh_display(self) -> None:
        """Refresh the heatmap display"""
        display = self.query_one("#heatmap-display", Static)
        display.update(self._render_heatmap())

    def _fetch_updates(self) -> None:
        """Fetch updated spread data"""
        if self.update_callback:
            try:
                new_data = self.update_callback()
                if new_data:
                    self.spread_data = new_data
                    self._logger.debug("Spread data updated", symbols=len(new_data))
            except Exception as e:
                self._logger.error("Failed to fetch spread updates", error=str(e))

    def update_spread_data(self, spread_data: dict[str, SpreadMetrics]) -> None:
        """
        Update spread data externally

        Args:
            spread_data: New spread metrics data
        """
        self.spread_data = spread_data
        self._refresh_display()

    async def on_mount(self) -> None:
        """Set up refresh timer when widget is mounted"""
        self.set_interval(self.refresh_interval, self._fetch_and_refresh)

    def _fetch_and_refresh(self) -> None:
        """Fetch updates and refresh display"""
        self._fetch_updates()
        self._refresh_display()

    def get_selected_metrics(self) -> Optional[SpreadMetrics]:
        """
        Get metrics for selected symbol

        Returns:
            SpreadMetrics or None if no selection
        """
        if self.selected_symbol:
            return self.spread_data.get(self.selected_symbol)
        return None
