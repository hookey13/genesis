"""Interactive validation dashboard using Textual framework."""

import json
from datetime import datetime
from pathlib import Path

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Footer, Header, Static

from genesis.validation import ValidationOrchestrator


class ValidationStatus(Static):
    """Widget to display validation status."""

    def __init__(self, validator_name: str, **kwargs):
        super().__init__(**kwargs)
        self.validator_name = validator_name
        self.status = "pending"
        self.score = 0.0

    def update_status(self, passed: bool, score: float, details: list[str]):
        """Update the validation status."""
        self.status = "passed" if passed else "failed"
        self.score = score

        # Create status indicator
        if self.status == "pending":
            icon = "⏳"
            color = "yellow"
        elif self.status == "passed":
            icon = "✅"
            color = "green"
        else:
            icon = "❌"
            color = "red"

        # Build status text
        status_text = Text()
        status_text.append(f"{icon} ", style=f"bold {color}")
        status_text.append(f"{self.validator_name}: ", style="bold")
        status_text.append(f"{score:.1f}%", style=color)

        if details:
            status_text.append("\n")
            for detail in details[:3]:  # Show first 3 details
                status_text.append(f"  • {detail}\n", style="dim")

        self.update(Panel(status_text, title=self.validator_name.replace("_", " ").title()))


class ValidationDashboard(App):
    """Interactive dashboard for validation status monitoring."""

    CSS = """
    ValidationDashboard {
        background: $surface;
    }
    
    .header {
        background: $primary;
        height: 3;
        padding: 1;
    }
    
    .validation-container {
        layout: grid;
        grid-size: 2 4;
        grid-gutter: 1;
        padding: 1;
    }
    
    .validation-status {
        height: 8;
        border: solid $primary;
    }
    
    .summary-panel {
        height: 10;
        border: solid $secondary;
        padding: 1;
    }
    
    .control-panel {
        dock: bottom;
        height: 5;
        padding: 1;
    }
    
    Button {
        margin: 0 1;
    }
    
    .passed {
        color: $success;
    }
    
    .failed {
        color: $error;
    }
    
    .pending {
        color: $warning;
    }
    """

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("c", "run_critical", "Critical Only"),
        ("a", "run_all", "Run All"),
        ("e", "export", "Export Report"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, genesis_root: Path | None = None):
        super().__init__()
        self.genesis_root = genesis_root or Path.cwd()
        self.orchestrator = ValidationOrchestrator(self.genesis_root)
        self.validators = {}
        self.last_results = None
        self.auto_refresh = True
        self.refresh_interval = 30  # seconds

    def compose(self) -> ComposeResult:
        """Create dashboard layout."""
        yield Header(show_clock=True)

        with Container(classes="main-container"):
            # Summary panel at top
            yield Static(self._get_summary_text(), classes="summary-panel", id="summary")

            # Validation status grid
            with Container(classes="validation-container"):
                for name in self.orchestrator.validators.keys():
                    status_widget = ValidationStatus(name, classes="validation-status")
                    self.validators[name] = status_widget
                    yield status_widget

            # Control panel
            with Horizontal(classes="control-panel"):
                yield Button("Run All", id="run-all", variant="primary")
                yield Button("Critical Only", id="run-critical", variant="warning")
                yield Button("Refresh", id="refresh", variant="default")
                yield Button("Export", id="export", variant="success")
                yield Button("Auto: ON", id="auto-refresh", variant="default")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize dashboard on mount."""
        # Start with a refresh
        await self.action_refresh()

        # Start auto-refresh timer
        self.set_interval(self.refresh_interval, self._auto_refresh)

    def _get_summary_text(self) -> RenderableType:
        """Generate summary panel content."""
        if not self.last_results:
            return Panel(
                "[yellow]No validation results yet. Press 'r' to refresh or 'a' to run all validators.[/yellow]",
                title="Validation Summary",
                border_style="yellow"
            )

        # Build summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        overall_passed = self.last_results.get("overall_passed", False)
        overall_score = self.last_results.get("overall_score", 0)
        timestamp = self.last_results.get("timestamp", "")
        duration = self.last_results.get("duration_seconds", 0)

        # Status icon and color
        if overall_passed:
            status = "[green]✅ PASSED[/green]"
            border_style = "green"
        else:
            status = "[red]❌ FAILED[/red]"
            border_style = "red"

        table.add_row("Status", status)
        table.add_row("Overall Score", f"{overall_score:.1f}%")
        table.add_row("Last Run", timestamp[:19] if timestamp else "N/A")
        table.add_row("Duration", f"{duration:.1f}s")

        # Count passed/failed validators
        if "validators" in self.last_results:
            validators = self.last_results["validators"]
            passed = sum(1 for v in validators.values() if v.get("passed"))
            total = len(validators)
            table.add_row("Validators", f"{passed}/{total} passed")

        return Panel(table, title="Validation Summary", border_style=border_style)

    async def _run_validation(self, critical_only: bool = False) -> None:
        """Run validation and update display."""
        # Show progress
        self.query_one("#summary", Static).update(
            Panel("[yellow]⏳ Running validation...[/yellow]", title="Validation in Progress")
        )

        # Clear existing statuses
        for validator in self.validators.values():
            validator.update_status(False, 0, ["Running..."])

        # Run validation
        if critical_only:
            self.last_results = await self.orchestrator.run_critical_validators()
        else:
            self.last_results = await self.orchestrator.run_all_validators(parallel=True)

        # Update displays
        await self._update_display()

    async def _update_display(self) -> None:
        """Update all display elements with latest results."""
        if not self.last_results:
            return

        # Update summary
        self.query_one("#summary", Static).update(self._get_summary_text())

        # Update individual validators
        if "validators" in self.last_results:
            for name, result in self.last_results["validators"].items():
                if name in self.validators:
                    passed = result.get("passed", False)
                    score = result.get("score", 0)

                    # Extract details from checks or details field
                    details = []
                    if "checks" in result:
                        for check_name, check_data in result["checks"].items():
                            if isinstance(check_data, dict) and "details" in check_data:
                                details.extend(check_data["details"][:1])  # First detail from each check
                    elif "details" in result:
                        if isinstance(result["details"], list):
                            details = result["details"][:3]

                    self.validators[name].update_status(passed, score, details)

    async def _auto_refresh(self) -> None:
        """Auto-refresh if enabled."""
        if self.auto_refresh:
            await self.action_refresh()

    async def action_refresh(self) -> None:
        """Refresh validation status."""
        # Check for existing results without re-running
        reports_dir = self.genesis_root / "docs" / "reports"
        if reports_dir.exists():
            # Find most recent report
            reports = list(reports_dir.glob("validation_report_*.json"))
            if reports:
                latest_report = max(reports, key=lambda p: p.stat().st_mtime)

                # Check if recent (less than 5 minutes old)
                age_seconds = (datetime.now().timestamp() - latest_report.stat().st_mtime)
                if age_seconds < 300:
                    with open(latest_report) as f:
                        self.last_results = json.load(f)
                    await self._update_display()
                    return

        # No recent results, inform user
        self.query_one("#summary", Static).update(
            Panel(
                "[yellow]No recent validation results. Run validation to see current status.[/yellow]",
                title="Validation Summary"
            )
        )

    async def action_run_all(self) -> None:
        """Run all validators."""
        await self._run_validation(critical_only=False)

    async def action_run_critical(self) -> None:
        """Run critical validators only."""
        await self._run_validation(critical_only=True)

    async def action_export(self) -> None:
        """Export validation report."""
        if not self.last_results:
            self.notify("No results to export", severity="warning")
            return

        # Save report
        report_path = await self.orchestrator.save_report(self.last_results)

        # Also create HTML report
        html_path = report_path.with_suffix(".html")
        await self._export_html_report(html_path)

        self.notify(f"Report exported to {report_path.name}", severity="information")

    async def _export_html_report(self, output_path: Path) -> None:
        """Export results as HTML report."""
        if not self.last_results:
            return

        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis Validation Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .passed { color: #27ae60; font-weight: bold; }
        .failed { color: #e74c3c; font-weight: bold; }
        .validator {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .validator.passed { border-left-color: #27ae60; }
        .validator.failed { border-left-color: #e74c3c; }
        .score { float: right; font-size: 1.2em; font-weight: bold; }
        .details { margin-top: 10px; padding-left: 20px; color: #666; }
        .details li { margin: 5px 0; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        .timestamp { color: #666; font-size: 0.9em; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Genesis Validation Report</h1>
"""

        # Add summary
        overall_passed = self.last_results.get("overall_passed", False)
        overall_score = self.last_results.get("overall_score", 0)
        status_class = "passed" if overall_passed else "failed"
        status_text = "PASSED" if overall_passed else "FAILED"

        html += f"""
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Status</th><td class="{status_class}">{status_text}</td></tr>
            <tr><th>Overall Score</th><td>{overall_score:.1f}%</td></tr>
            <tr><th>Timestamp</th><td>{self.last_results.get('timestamp', 'N/A')}</td></tr>
            <tr><th>Duration</th><td>{self.last_results.get('duration_seconds', 0):.1f} seconds</td></tr>
        </table>
    </div>
"""

        # Add validator results
        html += "<h2>Validator Results</h2>"

        if "validators" in self.last_results:
            for name, result in self.last_results["validators"].items():
                passed = result.get("passed", False)
                score = result.get("score", 0)
                validator_class = "validator passed" if passed else "validator failed"

                html += f"""
    <div class="{validator_class}">
        <span class="score">{score:.1f}%</span>
        <h3>{name.replace('_', ' ').title()}</h3>
"""

                # Add details if available
                if "checks" in result:
                    html += "<ul class='details'>"
                    for check_name, check_data in result["checks"].items():
                        if isinstance(check_data, dict) and "details" in check_data:
                            for detail in check_data["details"][:2]:
                                html += f"<li>{detail}</li>"
                    html += "</ul>"

                html += "</div>"

        # Add footer
        html += f"""
    <div class="timestamp">
        Report generated: {datetime.utcnow().isoformat()}
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "run-all":
            await self.action_run_all()
        elif button_id == "run-critical":
            await self.action_run_critical()
        elif button_id == "refresh":
            await self.action_refresh()
        elif button_id == "export":
            await self.action_export()
        elif button_id == "auto-refresh":
            # Toggle auto-refresh
            self.auto_refresh = not self.auto_refresh
            button_text = "Auto: ON" if self.auto_refresh else "Auto: OFF"
            event.button.label = button_text
            self.notify(f"Auto-refresh {'enabled' if self.auto_refresh else 'disabled'}")


def main():
    """Run the validation dashboard."""
    app = ValidationDashboard()
    app.run()


if __name__ == "__main__":
    main()
