#!/usr/bin/env python3
"""Master go-live dashboard for production readiness assessment."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from decimal import Decimal

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.production_readiness import ProductionReadinessValidator

console = Console()


class GoLiveDashboard:
    """Master dashboard aggregating all validation results."""
    
    def __init__(self):
        self.validator = ProductionReadinessValidator()
        self.results = {}
        self.status_colors = {
            "GO": "green",
            "NO-GO": "red",
            "PENDING": "yellow",
        }
        
    async def run(self):
        """Run the go-live dashboard."""
        console.print(Panel.fit(
            "[bold cyan]🚀 Genesis Go-Live Dashboard[/bold cyan]\n"
            "Comprehensive Production Readiness Assessment",
            box=box.DOUBLE,
        ))
        
        # Run validation with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running validation suite...", total=None)
            self.results = await self.validator.run_validation()
            progress.remove_task(task)
        
        # Display dashboard
        self._display_dashboard()
        
        # Generate detailed report
        self._generate_html_report()
        
        # Return appropriate exit code
        recommendation = self.results["assessment"]["recommendation"]
        if recommendation == "GO":
            console.print("\n[bold green]✅ System is ready for production![/bold green]")
            return 0
        elif recommendation == "NO-GO":
            console.print("\n[bold red]❌ System is NOT ready for production.[/bold red]")
            return 1
        else:
            console.print("\n[bold yellow]⚠️ Manual verification required before go-live.[/bold yellow]")
            return 2
    
    def _display_dashboard(self):
        """Display the dashboard in the terminal."""
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5),
        )
        
        # Header with overall status
        assessment = self.results["assessment"]
        recommendation = assessment["recommendation"]
        color = self.status_colors[recommendation]
        
        header_text = Text()
        header_text.append(f"Status: {recommendation}", style=f"bold {color}")
        if assessment["readiness_score"] is not None:
            header_text.append(f" | Readiness: {assessment['readiness_score']:.1f}%")
        header_text.append(f" | {assessment['reason']}")
        
        layout["header"].update(Panel(header_text, title="Overall Assessment"))
        
        # Main content with checklist table
        main_table = self._create_checklist_table()
        layout["main"].update(Panel(main_table, title="Validation Checklist"))
        
        # Footer with key metrics
        footer_content = self._create_metrics_summary()
        layout["footer"].update(Panel(footer_content, title="Key Metrics"))
        
        console.print(layout)
    
    def _create_checklist_table(self) -> Table:
        """Create the checklist results table."""
        table = Table(box=box.ROUNDED)
        table.add_column("✓", justify="center", width=3)
        table.add_column("Criteria", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")
        
        for item in self.results["checklist"]:
            # Status icon
            if item["passed"] is True:
                check = "✅"
                status = "[green]PASS[/green]"
            elif item["passed"] is False:
                check = "❌"
                status = "[red]FAIL[/red]"
            else:
                check = "⚠️"
                status = "[yellow]MANUAL[/yellow]"
            
            # Format details
            details = []
            if isinstance(item["details"], dict):
                for key, value in item["details"].items():
                    if key != "note":
                        if isinstance(value, (int, float)):
                            details.append(f"{key}: {value:.1f}")
                        else:
                            details.append(f"{key}: {value}")
                    else:
                        details.append(value)
            
            table.add_row(
                check,
                item["description"],
                status,
                "\n".join(details[:2]),  # Show first 2 details
            )
        
        return table
    
    def _create_metrics_summary(self) -> Text:
        """Create metrics summary text."""
        text = Text()
        
        # Get key metrics from validation results
        checklist = self.results["checklist"]
        
        # Extract specific metrics
        metrics = []
        for item in checklist:
            if item["id"] == "AC1" and item["details"]:
                coverage = item["details"].get("unit_coverage", 0)
                metrics.append(f"Test Coverage: {coverage:.1f}%")
            elif item["id"] == "AC3" and item["details"]:
                hours = item["details"].get("hours_stable", 0)
                metrics.append(f"Stability Test: {hours:.1f} hours")
            elif item["id"] == "AC5" and item["details"]:
                latency = item["details"].get("p99_latency_ms", 0)
                metrics.append(f"P99 Latency: {latency:.1f}ms")
            elif item["id"] == "AC8" and item["details"]:
                profit = item["details"].get("total_profit", 0)
                metrics.append(f"Paper Trading Profit: ${profit:,.2f}")
        
        text.append(" | ".join(metrics))
        
        return text
    
    def _generate_html_report(self):
        """Generate comprehensive HTML report."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        assessment = self.results["assessment"]
        recommendation = assessment["recommendation"]
        
        # Determine styling based on recommendation
        if recommendation == "GO":
            theme_color = "#00ff00"
            status_class = "go"
        elif recommendation == "NO-GO":
            theme_color = "#ff0000"
            status_class = "no-go"
        else:
            theme_color = "#ffaa00"
            status_class = "pending"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Genesis Go-Live Dashboard - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #e0e0e0;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #00d4ff;
            border-bottom: 3px solid #00d4ff;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        .status-banner {{
            background: #2a2a2a;
            border-left: 5px solid {theme_color};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .status-banner.go {{ background: rgba(0, 255, 0, 0.1); }}
        .status-banner.no-go {{ background: rgba(255, 0, 0, 0.1); }}
        .status-banner.pending {{ background: rgba(255, 170, 0, 0.1); }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #333;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #555;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .metric-label {{
            color: #aaa;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #2a2a2a;
        }}
        th {{
            background: #333;
            padding: 15px;
            text-align: left;
            border: 1px solid #555;
            color: #00d4ff;
        }}
        td {{
            padding: 12px;
            border: 1px solid #555;
        }}
        tr:nth-child(even) {{ background: rgba(255, 255, 255, 0.02); }}
        .status-pass {{ color: #00ff00; font-weight: bold; }}
        .status-fail {{ color: #ff0000; font-weight: bold; }}
        .status-pending {{ color: #ffaa00; font-weight: bold; }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #333;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #000;
            font-weight: bold;
        }}
        .recommendations {{
            background: #333;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .recommendations ul {{
            list-style-type: none;
            padding: 0;
        }}
        .recommendations li {{
            padding: 8px 0;
            border-bottom: 1px solid #555;
        }}
        .recommendations li:before {{
            content: "→ ";
            color: #00d4ff;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>🚀 Genesis Go-Live Dashboard</h1>
        
        <div class="status-banner {status_class}">
            <h2 style="color: {theme_color}; margin-top: 0;">
                Recommendation: {recommendation}
            </h2>
            <p style="font-size: 1.2em;">{assessment['reason']}</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {assessment['readiness_score'] or 0:.1f}%">
                    {assessment['readiness_score'] or 0:.1f}% Ready
                </div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{assessment['passed_count']}/{assessment['total_required']}</div>
                <div class="metric-label">Criteria Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{assessment['failed_count']}</div>
                <div class="metric-label">Failed Items</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{assessment['manual_count']}</div>
                <div class="metric-label">Manual Verification</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{assessment['readiness_score'] or 0:.1f}%</div>
                <div class="metric-label">Overall Readiness</div>
            </div>
        </div>
        
        <h2>Validation Checklist</h2>
        <table>
            <tr>
                <th width="80">ID</th>
                <th width="100">Status</th>
                <th>Criteria</th>
                <th>Details</th>
                <th>Required</th>
            </tr>
"""
        
        # Add checklist items
        for item in self.results["checklist"]:
            if item["passed"] is True:
                status_text = "PASS"
                status_class = "status-pass"
            elif item["passed"] is False:
                status_text = "FAIL"
                status_class = "status-fail"
            else:
                status_text = "MANUAL"
                status_class = "status-pending"
            
            # Format details
            details_html = []
            if isinstance(item["details"], dict):
                for key, value in item["details"].items():
                    if isinstance(value, (int, float)):
                        details_html.append(f"{key}: {value:.1f}")
                    else:
                        details_html.append(f"{key}: {value}")
            
            html_content += f"""
            <tr>
                <td>{item['id']}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{item['description']}</td>
                <td>{', '.join(details_html[:3])}</td>
                <td>{'Yes' if item['required'] else 'No'}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <div class="recommendations">
            <h2>Next Steps</h2>
            <ul>
"""
        
        # Add recommendations
        if recommendation == "GO":
            html_content += """
                <li>Review go-live protocol in architecture documentation</li>
                <li>Start with $100 (not full $500) for first 24 hours</li>
                <li>Trade smallest positions possible ($10)</li>
                <li>Monitor continuously for first week</li>
                <li>Keep manual kill switch ready</li>
                <li>Perform daily reconciliation</li>
"""
        elif recommendation == "NO-GO":
            # Add specific failure recommendations
            for item in self.results["checklist"]:
                if item["passed"] is False and item["required"]:
                    html_content += f"<li>Fix: {item['description']}</li>"
        else:
            # Manual verification required
            for item in self.results["checklist"]:
                if item["passed"] is None and item["required"]:
                    html_content += f"<li>Verify: {item['description']}</li>"
        
        html_content += f"""
            </ul>
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #555; color: #888;">
            <p>Generated: {timestamp}</p>
            <p>Report Location: {self.results.get('report', 'N/A')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        report_dir = Path("docs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"go_live_dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        report_file.write_text(html_content)
        
        # Also save as latest
        latest_file = report_dir / "go_live_dashboard_latest.html"
        latest_file.write_text(html_content)
        
        console.print(f"\n📊 Dashboard saved to: {report_file}")
        console.print(f"📊 Latest dashboard: {latest_file}")


async def main():
    """Main entry point."""
    dashboard = GoLiveDashboard()
    
    try:
        exit_code = await dashboard.run()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Dashboard failed: {e}[/bold red]")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())