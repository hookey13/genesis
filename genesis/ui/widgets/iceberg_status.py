"""Widget for displaying iceberg order execution status."""

from decimal import Decimal

from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from textual.widget import Widget


class IcebergStatusWidget(Widget):
    """Widget displaying current iceberg execution status."""

    def __init__(self):
        """Initialize the iceberg status widget."""
        super().__init__()
        self.active_executions: dict[str, dict] = {}

    def update_execution(
        self,
        execution_id: str,
        symbol: str,
        completed_slices: int,
        total_slices: int,
        status: str,
        current_slippage: Decimal | None = None,
        cumulative_impact: Decimal | None = None
    ):
        """Update execution status."""
        self.active_executions[execution_id] = {
            "symbol": symbol,
            "completed_slices": completed_slices,
            "total_slices": total_slices,
            "status": status,
            "slippage": current_slippage,
            "impact": cumulative_impact,
            "progress": (completed_slices / total_slices * 100) if total_slices > 0 else 0
        }
        self.refresh()

    def remove_execution(self, execution_id: str):
        """Remove completed or aborted execution."""
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]
            self.refresh()

    def render(self) -> RenderableType:
        """Render the iceberg status widget."""
        if not self.active_executions:
            return Panel("No active iceberg executions", title="🧊 Iceberg Status")

        table = Table(title="🧊 Active Iceberg Executions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Progress", width=20)
        table.add_column("Slices", style="yellow")
        table.add_column("Slippage", style="red")
        table.add_column("Impact", style="magenta")
        table.add_column("Status", style="green")

        for exec_id, data in self.active_executions.items():
            # Create progress bar
            progress = Progress(
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                BarColumn(bar_width=10),
                expand=False
            )
            task = progress.add_task("", total=100, completed=data["progress"])

            # Format slices
            slices = f"{data['completed_slices']}/{data['total_slices']}"

            # Format slippage and impact
            slippage = f"{data['slippage']:.2f}%" if data['slippage'] else "—"
            impact = f"{data['impact']:.2f}%" if data['impact'] else "—"

            # Status with emoji
            status_emoji = {
                "IN_PROGRESS": "⏳",
                "COMPLETED": "✅",
                "ABORTED": "❌",
                "FAILED": "⚠️"
            }.get(data['status'], "❓")

            table.add_row(
                data['symbol'],
                progress,
                slices,
                slippage,
                impact,
                f"{status_emoji} {data['status']}"
            )

        return Panel(table, title="🧊 Iceberg Executions")
