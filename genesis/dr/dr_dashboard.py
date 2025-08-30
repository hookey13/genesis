"""Disaster recovery dashboard for monitoring and control."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

logger = structlog.get_logger(__name__)


class DRDashboard:
    """Real-time disaster recovery dashboard."""
    
    def __init__(self, dr_orchestrator):
        """Initialize DR dashboard.
        
        Args:
            dr_orchestrator: DR orchestrator instance
        """
        self.dr_orchestrator = dr_orchestrator
        self.console = Console()
        self.layout = self._create_layout()
        self.refresh_interval = 5  # seconds
        
    def _create_layout(self) -> Layout:
        """Create dashboard layout.
        
        Returns:
            Dashboard layout
        """
        layout = Layout(name="root")
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split(
            Layout(name="readiness"),
            Layout(name="backup_status")
        )
        
        layout["right"].split(
            Layout(name="failover_status"),
            Layout(name="recent_events")
        )
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create header panel.
        
        Returns:
            Header panel
        """
        status = self.dr_orchestrator.get_status()
        
        if status["dr_active"]:
            header_text = Text(
                f"🚨 DR ACTIVE - {status['current_scenario']} 🚨",
                style="bold red"
            )
        else:
            header_text = Text(
                "DR Dashboard - System Normal",
                style="bold green"
            )
        
        return Panel(
            header_text,
            title=f"Genesis DR System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="cyan"
        )
    
    def _create_readiness_panel(self) -> Panel:
        """Create readiness score panel.
        
        Returns:
            Readiness panel
        """
        readiness = self.dr_orchestrator.calculate_readiness_score()
        
        # Create readiness table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status")
        
        # Overall score
        score_color = self._get_score_color(readiness["overall_score"])
        table.add_row(
            "Overall Readiness",
            f"{readiness['overall_score']:.1%}",
            Text(readiness["grade"], style=score_color)
        )
        
        table.add_row("", "", "")  # Spacer
        
        # Component scores
        for component, score in readiness["components"].items():
            status_icon = "✓" if score >= 0.8 else "⚠" if score >= 0.5 else "✗"
            status_color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
            
            table.add_row(
                component.replace("_", " ").title(),
                f"{score:.1%}",
                Text(status_icon, style=status_color)
            )
        
        # Add recommendations
        recommendations = "\n".join(f"• {rec}" for rec in readiness["recommendations"][:3])
        
        content = Layout()
        content.split(
            Layout(table),
            Layout(Text(f"\nRecommendations:\n{recommendations}", style="dim"), size=6)
        )
        
        return Panel(
            content.renderable,
            title="DR Readiness Score",
            border_style="blue"
        )
    
    def _create_backup_status_panel(self) -> Panel:
        """Create backup status panel.
        
        Returns:
            Backup status panel
        """
        backup_status = self.dr_orchestrator.backup_manager.get_backup_status()
        replication_status = self.dr_orchestrator.replication_manager.get_replication_status()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        # Last backup
        if backup_status["last_full_backup"]:
            last_backup = datetime.fromisoformat(backup_status["last_full_backup"])
            age = (datetime.utcnow() - last_backup).total_seconds() / 3600
            age_color = "green" if age < 4 else "yellow" if age < 8 else "red"
            
            table.add_row(
                "Last Full Backup",
                Text(f"{age:.1f} hours ago", style=age_color)
            )
        else:
            table.add_row("Last Full Backup", Text("Never", style="red"))
        
        # Backup count
        table.add_row("Total Backups", str(backup_status["backup_count"]))
        
        # Next backup
        if backup_status["next_full_backup"]:
            next_backup = datetime.fromisoformat(backup_status["next_full_backup"])
            time_until = (next_backup - datetime.now()).total_seconds() / 60
            table.add_row("Next Backup", f"{time_until:.0f} min")
        
        # Replication lag
        lag = replication_status["replication_lag_seconds"]
        lag_color = "green" if lag < 300 else "yellow" if lag < 900 else "red"
        table.add_row(
            "Replication Lag",
            Text(f"{lag:.0f} seconds", style=lag_color)
        )
        
        # Queue size
        table.add_row(
            "Replication Queue",
            str(replication_status["queue_size"])
        )
        
        return Panel(
            table,
            title="Backup & Replication",
            border_style="green"
        )
    
    def _create_failover_status_panel(self) -> Panel:
        """Create failover status panel.
        
        Returns:
            Failover status panel
        """
        failover_status = self.dr_orchestrator.failover_coordinator.get_status()
        
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        # State
        state_color = "green" if failover_status["state"] == "normal" else "yellow"
        table.add_row(
            "State",
            Text(failover_status["state"], style=state_color)
        )
        
        # Monitoring
        monitoring_icon = "✓" if failover_status["monitoring"] else "✗"
        monitoring_color = "green" if failover_status["monitoring"] else "red"
        table.add_row(
            "Monitoring",
            Text(monitoring_icon, style=monitoring_color)
        )
        
        # Health checks
        if failover_status["health_status"]:
            health = failover_status["health_status"]
            healthy = health["healthy_checks"]
            total = health["total_checks"]
            health_color = "green" if healthy == total else "yellow" if healthy > 0 else "red"
            
            table.add_row(
                "Health Checks",
                Text(f"{healthy}/{total}", style=health_color)
            )
        
        # Failover count
        table.add_row(
            "Failover Count",
            str(failover_status["failover_count"])
        )
        
        # Last failover
        if failover_status["last_failover"]:
            last_failover = datetime.fromisoformat(failover_status["last_failover"])
            days_ago = (datetime.now() - last_failover).days
            table.add_row("Last Failover", f"{days_ago} days ago")
        else:
            table.add_row("Last Failover", "Never")
        
        return Panel(
            table,
            title="Failover Status",
            border_style="yellow"
        )
    
    def _create_recent_events_panel(self) -> Panel:
        """Create recent events panel.
        
        Returns:
            Recent events panel
        """
        table = Table(show_header=True, box=None)
        table.add_column("Time", style="cyan")
        table.add_column("Event")
        table.add_column("Status")
        
        # Get recent DR events
        if self.dr_orchestrator.dr_history:
            for event in self.dr_orchestrator.dr_history[-5:]:
                event_time = datetime.fromisoformat(event["start_time"])
                time_str = event_time.strftime("%H:%M")
                
                status_icon = "✓" if event["success"] else "✗"
                status_color = "green" if event["success"] else "red"
                
                table.add_row(
                    time_str,
                    event["scenario"],
                    Text(status_icon, style=status_color)
                )
        else:
            table.add_row("--:--", "No recent events", "")
        
        return Panel(
            table,
            title="Recent DR Events",
            border_style="magenta"
        )
    
    def _create_footer(self) -> Panel:
        """Create footer panel.
        
        Returns:
            Footer panel
        """
        rpo_result = self.dr_orchestrator._verify_rpo()
        
        # Create metrics text
        metrics = []
        
        # RTO
        rto_text = f"RTO Target: {self.dr_orchestrator.RTO_TARGET_MINUTES} min"
        metrics.append(Text(rto_text, style="cyan"))
        
        # RPO
        rpo_met = "✓" if rpo_result["met"] else "✗"
        rpo_color = "green" if rpo_result["met"] else "red"
        rpo_text = f"RPO: {rpo_result['data_loss_minutes']:.1f} min {rpo_met}"
        metrics.append(Text(rpo_text, style=rpo_color))
        
        # Combine metrics
        footer_content = Text(" | ").join(metrics)
        
        return Panel(
            footer_content,
            border_style="dim"
        )
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on score.
        
        Args:
            score: Score value (0-1)
            
        Returns:
            Color name
        """
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "yellow"
        else:
            return "red"
    
    def update(self) -> None:
        """Update dashboard display."""
        self.layout["header"].update(self._create_header())
        self.layout["readiness"].update(self._create_readiness_panel())
        self.layout["backup_status"].update(self._create_backup_status_panel())
        self.layout["failover_status"].update(self._create_failover_status_panel())
        self.layout["recent_events"].update(self._create_recent_events_panel())
        self.layout["footer"].update(self._create_footer())
    
    async def run(self) -> None:
        """Run dashboard with live updates."""
        with Live(self.layout, refresh_per_second=1, console=self.console) as live:
            while True:
                self.update()
                await asyncio.sleep(self.refresh_interval)
    
    def print_summary(self) -> None:
        """Print DR summary to console."""
        self.console.print("\n[bold cyan]Disaster Recovery Summary[/bold cyan]\n")
        
        # Get status
        status = self.dr_orchestrator.get_status()
        readiness = self.dr_orchestrator.calculate_readiness_score()
        
        # Print readiness
        score_color = self._get_score_color(readiness["overall_score"])
        self.console.print(
            f"Readiness Score: [{score_color}]{readiness['overall_score']:.1%}[/{score_color}] "
            f"(Grade: {readiness['grade']})"
        )
        
        # Print RTO/RPO
        self.console.print(
            f"RTO Target: {status['rto_target_minutes']} minutes | "
            f"RPO Target: {status['rpo_target_minutes']} minutes"
        )
        
        # Print recommendations
        self.console.print("\n[yellow]Recommendations:[/yellow]")
        for rec in readiness["recommendations"]:
            self.console.print(f"  • {rec}")
        
        # Print recent events
        if status["last_dr_event"]:
            self.console.print("\n[cyan]Last DR Event:[/cyan]")
            event = status["last_dr_event"]
            self.console.print(f"  Scenario: {event['scenario']}")
            self.console.print(f"  Time: {event['start_time']}")
            self.console.print(f"  Success: {event['success']}")
            self.console.print(f"  Recovery Time: {event.get('recovery_time_minutes', 'N/A')} min")


import asyncio  # Add at the top of the file