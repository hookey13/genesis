"""
Behavioral dashboard widget for baseline visualization.

This module provides the UI component for displaying behavioral
metrics and baseline comparisons.
"""

from datetime import UTC, datetime

import structlog
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, Static

logger = structlog.get_logger(__name__)


class BehavioralDashboard(Widget):
    """Dashboard widget for behavioral metrics visualization."""

    # Reactive properties
    profile_id = reactive("")
    baseline_status = reactive("Not Established")
    current_context = reactive("normal")

    def __init__(
        self,
        profile_manager=None,
        **kwargs
    ):
        """
        Initialize the behavioral dashboard.

        Args:
            profile_manager: Profile manager instance
        """
        super().__init__(**kwargs)
        self.profile_manager = profile_manager
        self.current_metrics = {}
        self.baseline_ranges = {}
        self.time_patterns = {}

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        with Container(id="behavioral-dashboard"):
            # Header section
            with Horizontal(id="dashboard-header"):
                yield Label("Behavioral Baseline Dashboard", id="dashboard-title")
                yield Label(f"Context: {self.current_context}", id="context-label")
                yield Button("Reset Baseline", id="reset-button", variant="warning")

            # Main content area
            with Vertical(id="dashboard-content"):
                # Baseline status panel
                yield Static(id="baseline-status-panel")

                # Metrics comparison panel
                yield Static(id="metrics-comparison-panel")

                # Time patterns panel
                yield Static(id="time-patterns-panel")

                # Profile controls
                with Horizontal(id="profile-controls"):
                    yield Button("Export Data", id="export-button")
                    yield Button("Switch Context", id="switch-context-button")
                    yield Button("Refresh", id="refresh-button", variant="primary")

    def on_mount(self) -> None:
        """Handle mount event."""
        self.update_dashboard()
        # Set up periodic refresh
        self.set_interval(30, self.update_dashboard)

    def update_dashboard(self) -> None:
        """Update dashboard with latest data."""
        if not self.profile_manager or not self.profile_id:
            return

        # Update baseline status
        self._update_baseline_status()

        # Update metrics comparison
        self._update_metrics_comparison()

        # Update time patterns
        self._update_time_patterns()

    def _update_baseline_status(self) -> None:
        """Update baseline status panel."""
        status_panel = self.query_one("#baseline-status-panel", Static)

        # Create status table
        table = Table(title="Baseline Status", expand=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        # Get profile data
        if self.profile_manager and self.profile_id:
            profile = self.profile_manager.active_profiles.get(self.profile_id)
            if profile:
                table.add_row("Profile ID", self.profile_id[:8] + "...")
                table.add_row("Learning Started", profile.learning_start_date.strftime("%Y-%m-%d %H:%M"))
                table.add_row("Maturity Status", "✓ Mature" if profile.is_mature else "⚠ Learning")
                table.add_row("Total Samples", str(profile.total_samples))
                table.add_row("Context", profile.context.capitalize())

                if profile.last_updated:
                    time_ago = datetime.now(UTC) - profile.last_updated
                    table.add_row("Last Updated", f"{int(time_ago.total_seconds() / 60)} min ago")
            else:
                table.add_row("Status", "No profile loaded")
        else:
            table.add_row("Status", "Not initialized")

        status_panel.update(Panel(table, title="📊 Baseline Status"))

    def _update_metrics_comparison(self) -> None:
        """Update metrics comparison panel."""
        comparison_panel = self.query_one("#metrics-comparison-panel", Static)

        # Create comparison table
        table = Table(title="Current vs Baseline", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="white")
        table.add_column("Baseline Range", style="green")
        table.add_column("Status", style="white")

        # Define metrics to display
        metrics_info = {
            "click_speed": ("Click Speed", "ms"),
            "order_frequency": ("Orders/Hour", ""),
            "position_size_variance": ("Size Variance", "%"),
            "cancel_rate": ("Cancel Rate", "%")
        }

        for metric_type, (display_name, unit) in metrics_info.items():
            current_value = self.current_metrics.get(metric_type, "N/A")
            baseline_range = self.baseline_ranges.get(metric_type, {})

            if baseline_range:
                lower = baseline_range.get("lower_bound", 0)
                upper = baseline_range.get("upper_bound", 0)
                range_str = f"{lower:.2f} - {upper:.2f} {unit}"

                # Determine status
                if current_value != "N/A":
                    if lower <= current_value <= upper:
                        status = Text("✓ Normal", style="green")
                    else:
                        status = Text("⚠ Deviation", style="yellow")
                else:
                    status = Text("-", style="dim")
            else:
                range_str = "Not established"
                status = Text("-", style="dim")

            # Format current value
            if current_value != "N/A":
                current_str = f"{current_value:.2f} {unit}"
            else:
                current_str = "N/A"

            table.add_row(display_name, current_str, range_str, status)

        comparison_panel.update(Panel(table, title="📈 Metrics Comparison"))

    def _update_time_patterns(self) -> None:
        """Update time patterns panel."""
        patterns_panel = self.query_one("#time-patterns-panel", Static)

        # Create time patterns visualization
        current_hour = datetime.now(UTC).hour

        # Create hourly activity chart
        chart = self._create_hourly_chart(current_hour)

        patterns_panel.update(Panel(chart, title="🕐 Time-of-Day Patterns"))

    def _create_hourly_chart(self, current_hour: int) -> RenderableType:
        """
        Create hourly activity chart.

        Args:
            current_hour: Current hour for highlighting

        Returns:
            Renderable chart
        """
        # Create progress bars for each metric
        progress_group = []

        # Mock data for demonstration - would come from actual patterns
        hourly_data = {
            "00-06": 20,
            "06-12": 60,
            "12-18": 80,
            "18-24": 40
        }

        for time_range, activity_level in hourly_data.items():
            progress = Progress(
                TextColumn(f"[cyan]{time_range}[/cyan]"),
                BarColumn(bar_width=20),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                expand=False
            )

            task = progress.add_task(time_range, total=100, completed=activity_level)
            progress_group.append(progress)

        return Group(*progress_group)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id

        if button_id == "reset-button":
            await self._handle_reset_baseline()
        elif button_id == "export-button":
            await self._handle_export_data()
        elif button_id == "switch-context-button":
            await self._handle_switch_context()
        elif button_id == "refresh-button":
            self.update_dashboard()

    async def _handle_reset_baseline(self) -> None:
        """Handle baseline reset."""
        if self.profile_manager and self.profile_id:
            # Reset baseline
            await self.profile_manager.reset_baseline(self.profile_id)

            # Update display
            self.baseline_status = "Reset - Learning"
            self.update_dashboard()

            logger.info("Baseline reset", profile_id=self.profile_id)

    async def _handle_export_data(self) -> None:
        """Handle data export."""
        if self.profile_manager and self.profile_id:
            # Export baseline data
            export_data = await self.profile_manager.repository.export_baseline_data(
                self.profile_id
            )

            # Save to file (simplified for demo)
            export_path = f"baseline_export_{self.profile_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info("Baseline data exported", path=export_path)

    async def _handle_switch_context(self) -> None:
        """Handle context switch."""
        # Cycle through contexts
        contexts = ["normal", "tired", "alert", "stressed"]
        current_idx = contexts.index(self.current_context)
        new_idx = (current_idx + 1) % len(contexts)
        new_context = contexts[new_idx]

        if self.profile_manager and self.profile_id:
            # Switch context
            await self.profile_manager.switch_profile_context(
                self.profile_id,
                new_context
            )

            # Update display
            self.current_context = new_context
            context_label = self.query_one("#context-label", Label)
            context_label.update(f"Context: {new_context.capitalize()}")

            logger.info("Context switched", new_context=new_context)

    def set_profile(self, profile_id: str) -> None:
        """
        Set the profile to display.

        Args:
            profile_id: Profile ID
        """
        self.profile_id = profile_id
        self.update_dashboard()

    def update_current_metrics(self, metrics: dict) -> None:
        """
        Update current metrics display.

        Args:
            metrics: Current metric values
        """
        self.current_metrics = metrics
        self._update_metrics_comparison()

    def update_baseline_ranges(self, ranges: dict) -> None:
        """
        Update baseline range values.

        Args:
            ranges: Baseline ranges for each metric
        """
        self.baseline_ranges = ranges
        self._update_metrics_comparison()
