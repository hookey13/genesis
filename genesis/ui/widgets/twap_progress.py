"""
TWAP progress monitoring widget for Project GENESIS.

This module provides real-time visualization of TWAP execution progress,
including slice execution status, participation rates, and price tracking.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from rich.align import Align
from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

logger = structlog.get_logger(__name__)


class TwapProgressBar(ProgressColumn):
    """Custom progress bar for TWAP execution."""

    def render(self, task: Task) -> Text:
        """Render the progress bar."""
        completed = int(task.completed)
        total = int(task.total) if task.total else 100

        if total == 0:
            return Text("")

        percentage = (completed / total) * 100
        filled = int(percentage / 2)  # 50 character wide bar

        # Color based on progress
        if percentage < 25:
            color = "red"
        elif percentage < 50:
            color = "yellow"
        elif percentage < 75:
            color = "blue"
        else:
            color = "green"

        bar = f"[{color}]{'█' * filled}{'░' * (50 - filled)}[/{color}]"
        return Text.from_markup(f"{bar} {percentage:.1f}%")


class TwapProgressWidget(Widget):
    """
    Widget for monitoring TWAP execution progress.

    Displays:
    - Overall execution progress
    - Time remaining
    - Current TWAP price vs market price
    - Participation rate and volume metrics
    - Individual slice execution history
    """

    # Reactive properties for live updates
    execution_id: reactive[Optional[str]] = reactive(None)
    symbol: reactive[str] = reactive("")
    side: reactive[str] = reactive("")
    total_quantity: reactive[Decimal] = reactive(Decimal("0"))
    executed_quantity: reactive[Decimal] = reactive(Decimal("0"))
    remaining_quantity: reactive[Decimal] = reactive(Decimal("0"))
    slice_count: reactive[int] = reactive(0)
    completed_slices: reactive[int] = reactive(0)
    arrival_price: reactive[Decimal] = reactive(Decimal("0"))
    current_price: reactive[Decimal] = reactive(Decimal("0"))
    twap_price: reactive[Decimal] = reactive(Decimal("0"))
    avg_participation_rate: reactive[Decimal] = reactive(Decimal("0"))
    implementation_shortfall: reactive[Decimal] = reactive(Decimal("0"))
    status: reactive[str] = reactive("INACTIVE")
    time_remaining_seconds: reactive[int] = reactive(0)
    started_at: reactive[Optional[datetime]] = reactive(None)

    # Slice history
    slice_history: reactive[list[dict]] = reactive([])

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Container(
            Vertical(
                Static(id="twap-header"),
                Static(id="twap-progress"),
                Static(id="twap-metrics"),
                Static(id="twap-slices"),
                id="twap-container"
            )
        )

    def on_mount(self) -> None:
        """Initialize the widget when mounted."""
        self.update_display()
        # Set up periodic updates
        self.set_interval(1.0, self.update_time_remaining)
        self.set_interval(2.0, self.update_display)

    def update_execution(self, execution_data: dict) -> None:
        """
        Update the widget with new execution data.

        Args:
            execution_data: Dictionary containing execution details
        """
        self.execution_id = execution_data.get("execution_id")
        self.symbol = execution_data.get("symbol", "")
        self.side = execution_data.get("side", "")
        self.total_quantity = Decimal(str(execution_data.get("total_quantity", 0)))
        self.executed_quantity = Decimal(str(execution_data.get("executed_quantity", 0)))
        self.remaining_quantity = Decimal(str(execution_data.get("remaining_quantity", 0)))
        self.slice_count = execution_data.get("slice_count", 0)
        self.completed_slices = execution_data.get("completed_slices", 0)
        self.arrival_price = Decimal(str(execution_data.get("arrival_price", 0)))
        self.current_price = Decimal(str(execution_data.get("current_price", 0)))
        self.twap_price = Decimal(str(execution_data.get("twap_price", 0)))
        self.avg_participation_rate = Decimal(str(execution_data.get("participation_rate", 0)))
        self.implementation_shortfall = Decimal(str(execution_data.get("implementation_shortfall", 0)))
        self.status = execution_data.get("status", "INACTIVE")
        self.started_at = execution_data.get("started_at")

        # Update slice history
        if "slice_history" in execution_data:
            self.slice_history = execution_data["slice_history"]

        self.update_display()

    def add_slice_execution(self, slice_data: dict) -> None:
        """
        Add a new slice execution to the history.

        Args:
            slice_data: Dictionary containing slice execution details
        """
        self.slice_history.append(slice_data)
        self.completed_slices += 1
        self.executed_quantity += Decimal(str(slice_data.get("executed_quantity", 0)))
        self.remaining_quantity = self.total_quantity - self.executed_quantity

        # Recalculate TWAP price
        if self.slice_history:
            total_value = Decimal("0")
            total_qty = Decimal("0")
            for slice in self.slice_history:
                qty = Decimal(str(slice.get("executed_quantity", 0)))
                price = Decimal(str(slice.get("execution_price", 0)))
                total_value += qty * price
                total_qty += qty

            if total_qty > 0:
                self.twap_price = total_value / total_qty

        self.update_display()

    def update_time_remaining(self) -> None:
        """Update the time remaining display."""
        if self.started_at and self.status == "ACTIVE":
            elapsed = (datetime.now() - self.started_at).total_seconds()
            # Estimate based on progress
            if self.completed_slices > 0 and self.slice_count > 0:
                avg_time_per_slice = elapsed / self.completed_slices
                remaining_slices = self.slice_count - self.completed_slices
                self.time_remaining_seconds = int(avg_time_per_slice * remaining_slices)
            self.update_display()

    def update_display(self) -> None:
        """Update the entire display."""
        # Update header
        header_widget = self.query_one("#twap-header", Static)
        header_widget.update(self._create_header())

        # Update progress
        progress_widget = self.query_one("#twap-progress", Static)
        progress_widget.update(self._create_progress_display())

        # Update metrics
        metrics_widget = self.query_one("#twap-metrics", Static)
        metrics_widget.update(self._create_metrics_table())

        # Update slice history
        slices_widget = self.query_one("#twap-slices", Static)
        slices_widget.update(self._create_slices_table())

    def _create_header(self) -> RenderableType:
        """Create the header panel."""
        if not self.execution_id:
            return Panel(
                Text("No active TWAP execution", style="dim"),
                title="TWAP Progress Monitor",
                border_style="blue"
            )

        # Status color
        status_colors = {
            "ACTIVE": "green",
            "PAUSED": "yellow",
            "COMPLETED": "blue",
            "FAILED": "red",
            "CANCELLED": "red"
        }
        status_color = status_colors.get(self.status, "white")

        header_text = Text()
        header_text.append("Execution: ", style="bold")
        header_text.append(f"{self.execution_id[:8]}...\n", style="cyan")
        header_text.append("Symbol: ", style="bold")
        header_text.append(f"{self.symbol} ", style="white")
        header_text.append(f"[{self.side}]\n", style="yellow" if self.side == "BUY" else "red")
        header_text.append("Status: ", style="bold")
        header_text.append(f"{self.status}", style=status_color)

        return Panel(
            header_text,
            title="TWAP Execution Monitor",
            border_style="blue"
        )

    def _create_progress_display(self) -> RenderableType:
        """Create the progress bar display."""
        if not self.execution_id or self.slice_count == 0:
            return Panel(Text("", style="dim"))

        # Create progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
        )

        # Slice progress
        slice_task = progress.add_task(
            "Slices",
            total=self.slice_count,
            completed=self.completed_slices
        )

        # Quantity progress
        qty_task = progress.add_task(
            "Quantity",
            total=float(self.total_quantity),
            completed=float(self.executed_quantity)
        )

        # Time remaining
        if self.time_remaining_seconds > 0:
            minutes = self.time_remaining_seconds // 60
            seconds = self.time_remaining_seconds % 60
            time_text = f"Time Remaining: {minutes:02d}:{seconds:02d}"
        else:
            time_text = "Time Remaining: --:--"

        progress_panel = Panel(
            Align.center(
                Vertical(
                    progress,
                    Text(time_text, style="dim", justify="center")
                ),
                vertical="middle"
            ),
            title="Execution Progress",
            border_style="green" if self.status == "ACTIVE" else "dim"
        )

        return progress_panel

    def _create_metrics_table(self) -> RenderableType:
        """Create the metrics table."""
        table = Table(title="Execution Metrics", show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        # Price metrics
        if self.arrival_price > 0:
            price_diff = self.twap_price - self.arrival_price
            price_color = "green" if price_diff <= 0 else "red"
            table.add_row(
                "Arrival Price",
                f"{self.arrival_price:,.2f}",
                ""
            )

        if self.twap_price > 0:
            table.add_row(
                "TWAP Price",
                f"{self.twap_price:,.2f}",
                "✓" if self.twap_price else ""
            )

        if self.current_price > 0:
            market_diff = ((self.current_price - self.twap_price) / self.twap_price * 100) if self.twap_price > 0 else 0
            market_color = "green" if abs(market_diff) < 0.5 else "yellow" if abs(market_diff) < 1 else "red"
            table.add_row(
                "Market Price",
                f"{self.current_price:,.2f}",
                f"[{market_color}]{market_diff:+.2f}%[/{market_color}]"
            )

        # Volume metrics
        if self.avg_participation_rate > 0:
            part_color = "green" if self.avg_participation_rate < 8 else "yellow" if self.avg_participation_rate < 10 else "red"
            table.add_row(
                "Avg Participation",
                f"{self.avg_participation_rate:.2f}%",
                f"[{part_color}]{'✓' if self.avg_participation_rate < 10 else '⚠'}[/{part_color}]"
            )

        # Implementation shortfall
        if self.implementation_shortfall != 0:
            shortfall_color = "green" if self.implementation_shortfall < 0 else "yellow" if abs(self.implementation_shortfall) < 0.2 else "red"
            table.add_row(
                "Implementation Shortfall",
                f"{self.implementation_shortfall:+.4f}%",
                f"[{shortfall_color}]{'✓' if abs(self.implementation_shortfall) < 0.2 else '⚠'}[/{shortfall_color}]"
            )

        # Quantity metrics
        table.add_row(
            "Executed / Total",
            f"{self.executed_quantity:.4f} / {self.total_quantity:.4f}",
            f"{(self.executed_quantity / self.total_quantity * 100):.1f}%"
        )

        return Panel(table, border_style="blue")

    def _create_slices_table(self) -> RenderableType:
        """Create the slice history table."""
        table = Table(title="Slice Execution History", show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Time", style="cyan")
        table.add_column("Quantity", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Participation", justify="right")
        table.add_column("Slippage", justify="right")
        table.add_column("Status", justify="center")

        # Show last 10 slices
        display_slices = self.slice_history[-10:] if len(self.slice_history) > 10 else self.slice_history

        for slice_data in display_slices:
            slice_num = slice_data.get("slice_number", 0)
            executed_at = slice_data.get("executed_at", "")
            if executed_at:
                try:
                    time_str = datetime.fromisoformat(executed_at).strftime("%H:%M:%S")
                except:
                    time_str = "--:--:--"
            else:
                time_str = "--:--:--"

            quantity = Decimal(str(slice_data.get("executed_quantity", 0)))
            price = Decimal(str(slice_data.get("execution_price", 0)))
            participation = Decimal(str(slice_data.get("participation_rate", 0)))
            slippage_bps = Decimal(str(slice_data.get("slippage_bps", 0)))
            status = slice_data.get("status", "PENDING")

            # Status icon
            status_icons = {
                "EXECUTED": "[green]✓[/green]",
                "FAILED": "[red]✗[/red]",
                "SKIPPED": "[yellow]-[/yellow]",
                "PENDING": "[dim]⋯[/dim]"
            }
            status_icon = status_icons.get(status, "?")

            # Slippage color
            slippage_color = "green" if abs(slippage_bps) < 10 else "yellow" if abs(slippage_bps) < 20 else "red"

            table.add_row(
                str(slice_num),
                time_str,
                f"{quantity:.4f}",
                f"{price:,.2f}",
                f"{participation:.1f}%",
                f"[{slippage_color}]{slippage_bps:+.1f}[/{slippage_color}]",
                status_icon
            )

        # Add summary row
        if self.slice_history:
            table.add_row(
                "",
                "[bold]Total[/bold]",
                f"[bold]{self.executed_quantity:.4f}[/bold]",
                f"[bold]{self.twap_price:,.2f}[/bold]",
                f"[bold]{self.avg_participation_rate:.1f}%[/bold]",
                "",
                f"[bold]{self.completed_slices}/{self.slice_count}[/bold]"
            )

        return Panel(
            table,
            border_style="green" if self.status == "ACTIVE" else "dim"
        )

    def pause_execution(self) -> None:
        """Update display when execution is paused."""
        self.status = "PAUSED"
        self.update_display()

    def resume_execution(self) -> None:
        """Update display when execution is resumed."""
        self.status = "ACTIVE"
        self.update_display()

    def complete_execution(self, final_metrics: dict) -> None:
        """
        Update display when execution completes.

        Args:
            final_metrics: Final execution metrics
        """
        self.status = "COMPLETED"
        self.twap_price = Decimal(str(final_metrics.get("twap_price", 0)))
        self.implementation_shortfall = Decimal(str(final_metrics.get("implementation_shortfall", 0)))
        self.update_display()

    def clear_execution(self) -> None:
        """Clear the current execution display."""
        self.execution_id = None
        self.symbol = ""
        self.side = ""
        self.total_quantity = Decimal("0")
        self.executed_quantity = Decimal("0")
        self.remaining_quantity = Decimal("0")
        self.slice_count = 0
        self.completed_slices = 0
        self.slice_history = []
        self.status = "INACTIVE"
        self.update_display()
