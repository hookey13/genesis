"""Alert management UI widget for Genesis trading terminal."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertState(Enum):
    """Alert states."""
    FIRING = "firing"
    PENDING = "pending"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert information."""
    id: str
    name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    source: str
    timestamp: datetime
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    firing_duration: Optional[timedelta] = None
    occurrence_count: int = 1


class AlertManagerWidget(Static):
    """Widget for managing and displaying alerts."""
    
    # Alert lists
    active_alerts = reactive(list)  # List[Alert]
    pending_alerts = reactive(list)  # List[Alert]
    resolved_alerts = reactive(list)  # List[Alert]
    
    # Summary metrics
    critical_count = reactive(0)
    high_count = reactive(0)
    medium_count = reactive(0)
    unacknowledged_count = reactive(0)
    
    # Alert rules
    suppression_rules = reactive(list)  # List of suppression patterns
    routing_rules = reactive(dict)  # Severity -> notification channels
    
    # UI state
    show_resolved = reactive(False)
    filter_severity = reactive(None)  # Optional[AlertSeverity]
    
    DEFAULT_CSS = """
    AlertManagerWidget {
        content-align: center middle;
        padding: 1;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the alert manager widget."""
        super().__init__("Loading alerts...", **kwargs)
        self.last_update = datetime.now(UTC)
        self.active_alerts = []
        self.pending_alerts = []
        self.resolved_alerts = []
        self.acknowledgment_callbacks = []
    
    def render(self) -> str:
        """Render the alert management display."""
        lines = [
            "[bold]═══ Alert Management ═══[/bold]",
            "",
            self._render_summary(),
            "",
            self._render_critical_alerts(),
            "",
            self._render_active_alerts_table(),
            "",
        ]
        
        if self.pending_alerts:
            lines.extend([
                self._render_pending_alerts(),
                "",
            ])
        
        if self.show_resolved and self.resolved_alerts:
            lines.extend([
                self._render_resolved_alerts(),
                "",
            ])
        
        lines.extend([
            self._render_alert_statistics(),
            "",
            f"[dim]Last update: {self.last_update.strftime('%H:%M:%S')}[/dim]",
        ])
        
        return "\n".join(lines)
    
    def _render_summary(self) -> str:
        """Render alert summary."""
        # Determine overall status
        if self.critical_count > 0:
            status_color = "red"
            status_icon = "🔴"
            status_text = "CRITICAL"
        elif self.high_count > 0:
            status_color = "orange"
            status_icon = "🟠"
            status_text = "WARNING"
        elif self.medium_count > 0:
            status_color = "yellow"
            status_icon = "🟡"
            status_text = "CAUTION"
        else:
            status_color = "green"
            status_icon = "🟢"
            status_text = "NORMAL"
        
        lines = [
            f"[{status_color}]{status_icon} {status_text}[/{status_color}]",
            "",
            f"Active Alerts: {len(self.active_alerts)} | Unacknowledged: {self.unacknowledged_count}",
        ]
        
        # Severity breakdown
        severity_line = "Severity: "
        if self.critical_count > 0:
            severity_line += f"[red]{self.critical_count} Critical[/red] "
        if self.high_count > 0:
            severity_line += f"[orange]{self.high_count} High[/orange] "
        if self.medium_count > 0:
            severity_line += f"[yellow]{self.medium_count} Medium[/yellow]"
        
        if severity_line != "Severity: ":
            lines.append(severity_line)
        
        return "\n".join(lines)
    
    def _render_critical_alerts(self) -> str:
        """Render critical alerts that need immediate attention."""
        critical = [a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]
        
        if not critical:
            return ""
        
        lines = ["[bold red]⚠ CRITICAL ALERTS:[/bold red]"]
        
        for alert in critical[:3]:  # Show top 3
            ack_status = " [UNACK]" if not alert.acknowledged_by else ""
            duration = self._format_duration(alert.firing_duration) if alert.firing_duration else "just now"
            lines.append(
                f"  • {alert.name}: {alert.message[:50]}... "
                f"[dim]({duration})[/dim]{ack_status}"
            )
        
        if len(critical) > 3:
            lines.append(f"  [dim]... and {len(critical) - 3} more critical alerts[/dim]")
        
        return "\n".join(lines)
    
    def _render_active_alerts_table(self) -> str:
        """Render table of active alerts."""
        if not self.active_alerts:
            return "[dim]No active alerts[/dim]"
        
        lines = ["[bold]Active Alerts:[/bold]", ""]
        
        # Filter alerts if needed
        alerts_to_show = self.active_alerts
        if self.filter_severity:
            alerts_to_show = [a for a in alerts_to_show if a.severity == self.filter_severity]
        
        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }
        alerts_to_show.sort(key=lambda a: (severity_order[a.severity], a.timestamp))
        
        for alert in alerts_to_show[:10]:  # Show top 10
            # Severity icon and color
            sev_icon, sev_color = self._get_severity_display(alert.severity)
            
            # Acknowledgment status
            if alert.acknowledged_by:
                ack_icon = "✓"
                ack_color = "green"
            else:
                ack_icon = "!"
                ack_color = "yellow"
            
            # Duration
            duration = self._format_duration(alert.firing_duration) if alert.firing_duration else "now"
            
            # Format line
            line = (f"  [{sev_color}]{sev_icon}[/{sev_color}] "
                   f"[{ack_color}]{ack_icon}[/{ack_color}] "
                   f"{alert.name[:20]:20} {alert.message[:40]:40} "
                   f"[dim]{duration:>8}[/dim]")
            
            if alert.occurrence_count > 1:
                line += f" [dim](x{alert.occurrence_count})[/dim]"
            
            lines.append(line)
        
        if len(alerts_to_show) > 10:
            lines.append(f"  [dim]... and {len(alerts_to_show) - 10} more alerts[/dim]")
        
        return "\n".join(lines)
    
    def _render_pending_alerts(self) -> str:
        """Render pending alerts."""
        lines = ["[bold]Pending Alerts:[/bold]"]
        
        for alert in self.pending_alerts[:5]:
            sev_icon, sev_color = self._get_severity_display(alert.severity)
            lines.append(
                f"  [{sev_color}]{sev_icon}[/{sev_color}] "
                f"{alert.name}: {alert.message[:50]}..."
            )
        
        return "\n".join(lines)
    
    def _render_resolved_alerts(self) -> str:
        """Render recently resolved alerts."""
        lines = ["[bold]Recently Resolved:[/bold]"]
        
        for alert in self.resolved_alerts[:5]:
            resolved_ago = datetime.now(UTC) - alert.resolved_at if alert.resolved_at else timedelta(0)
            time_str = self._format_duration(resolved_ago)
            
            lines.append(
                f"  [green]✓[/green] {alert.name} "
                f"[dim](resolved {time_str} ago)[/dim]"
            )
        
        return "\n".join(lines)
    
    def _render_alert_statistics(self) -> str:
        """Render alert statistics."""
        lines = ["[bold]Statistics (24h):[/bold]"]
        
        # Count alerts by state in last 24h
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=24)
        
        recent_alerts = [
            a for a in (self.active_alerts + self.resolved_alerts)
            if a.timestamp > cutoff
        ]
        
        if recent_alerts:
            total = len(recent_alerts)
            resolved = len([a for a in recent_alerts if a.state == AlertState.RESOLVED])
            acknowledged = len([a for a in recent_alerts if a.acknowledged_by])
            
            resolution_rate = (resolved / total * 100) if total > 0 else 0
            ack_rate = (acknowledged / total * 100) if total > 0 else 0
            
            lines.append(f"  Total Alerts: {total}")
            lines.append(f"  Resolution Rate: {resolution_rate:.1f}%")
            lines.append(f"  Acknowledgment Rate: {ack_rate:.1f}%")
            
            # Mean time to acknowledge
            ack_times = []
            for alert in recent_alerts:
                if alert.acknowledged_at and alert.timestamp:
                    ack_time = (alert.acknowledged_at - alert.timestamp).total_seconds()
                    ack_times.append(ack_time)
            
            if ack_times:
                mean_ack_time = sum(ack_times) / len(ack_times)
                lines.append(f"  Mean Time to Ack: {self._format_duration(timedelta(seconds=mean_ack_time))}")
        
        return "\n".join(lines)
    
    def _get_severity_display(self, severity: AlertSeverity) -> tuple[str, str]:
        """Get display icon and color for severity."""
        displays = {
            AlertSeverity.CRITICAL: ("⛔", "red"),
            AlertSeverity.HIGH: ("⚠", "orange"),
            AlertSeverity.MEDIUM: ("!", "yellow"),
            AlertSeverity.LOW: ("ℹ", "blue"),
            AlertSeverity.INFO: ("·", "dim"),
        }
        return displays.get(severity, ("?", "white"))
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for display."""
        if not duration:
            return "now"
        
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h"
        else:
            return f"{total_seconds // 86400}d"
    
    def add_alert(self, alert: Alert) -> None:
        """Add a new alert."""
        # Check if alert already exists
        for existing in self.active_alerts:
            if existing.id == alert.id:
                # Update occurrence count
                existing.occurrence_count += 1
                existing.timestamp = alert.timestamp
                self._recalculate_counts()
                return
        
        # Add new alert
        if alert.state == AlertState.FIRING:
            self.active_alerts.append(alert)
        elif alert.state == AlertState.PENDING:
            self.pending_alerts.append(alert)
        elif alert.state == AlertState.RESOLVED:
            self.resolved_alerts.append(alert)
        
        self._recalculate_counts()
        self.last_update = datetime.now(UTC)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(UTC)
                alert.state = AlertState.ACKNOWLEDGED
                self._recalculate_counts()
                
                # Call acknowledgment callbacks
                for callback in self.acknowledgment_callbacks:
                    callback(alert)
                
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for i, alert in enumerate(self.active_alerts):
            if alert.id == alert_id:
                alert.state = AlertState.RESOLVED
                alert.resolved_at = datetime.now(UTC)
                
                # Move to resolved list
                self.resolved_alerts.insert(0, alert)
                del self.active_alerts[i]
                
                # Keep only recent resolved alerts
                self.resolved_alerts = self.resolved_alerts[:20]
                
                self._recalculate_counts()
                return True
        return False
    
    def _recalculate_counts(self) -> None:
        """Recalculate summary counts."""
        self.critical_count = sum(1 for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL)
        self.high_count = sum(1 for a in self.active_alerts if a.severity == AlertSeverity.HIGH)
        self.medium_count = sum(1 for a in self.active_alerts if a.severity == AlertSeverity.MEDIUM)
        self.unacknowledged_count = sum(1 for a in self.active_alerts if not a.acknowledged_by)
    
    def add_acknowledgment_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for when alerts are acknowledged."""
        self.acknowledgment_callbacks.append(callback)
    
    def set_severity_filter(self, severity: Optional[AlertSeverity]) -> None:
        """Set severity filter for display."""
        self.filter_severity = severity
    
    def toggle_resolved_display(self) -> None:
        """Toggle display of resolved alerts."""
        self.show_resolved = not self.show_resolved
    
    def set_mock_data(self) -> None:
        """Set mock data for testing."""
        now = datetime.now(UTC)
        
        # Add active alerts
        self.active_alerts = [
            Alert(
                id="alert_001",
                name="High CPU Usage",
                severity=AlertSeverity.HIGH,
                state=AlertState.FIRING,
                message="CPU usage above 90% for 5 minutes",
                source="system_monitor",
                timestamp=now - timedelta(minutes=15),
                firing_duration=timedelta(minutes=15),
                labels={"component": "trading_engine", "host": "prod-01"},
                occurrence_count=3,
            ),
            Alert(
                id="alert_002",
                name="Order Execution Failed",
                severity=AlertSeverity.CRITICAL,
                state=AlertState.FIRING,
                message="Failed to execute market order - exchange timeout",
                source="order_executor",
                timestamp=now - timedelta(minutes=2),
                firing_duration=timedelta(minutes=2),
                labels={"exchange": "binance", "symbol": "BTC/USDT"},
                acknowledged_by="operator1",
                acknowledged_at=now - timedelta(minutes=1),
            ),
            Alert(
                id="alert_003",
                name="Rate Limit Warning",
                severity=AlertSeverity.MEDIUM,
                state=AlertState.FIRING,
                message="API rate limit at 85% capacity",
                source="rate_limiter",
                timestamp=now - timedelta(minutes=30),
                firing_duration=timedelta(minutes=30),
                labels={"endpoint": "/api/v3/order"},
            ),
            Alert(
                id="alert_004",
                name="Websocket Reconnect",
                severity=AlertSeverity.LOW,
                state=AlertState.FIRING,
                message="WebSocket disconnected and reconnecting",
                source="websocket_manager",
                timestamp=now - timedelta(seconds=45),
                firing_duration=timedelta(seconds=45),
            ),
        ]
        
        # Add pending alerts
        self.pending_alerts = [
            Alert(
                id="alert_005",
                name="Memory Usage Rising",
                severity=AlertSeverity.MEDIUM,
                state=AlertState.PENDING,
                message="Memory usage trending upward - 75% used",
                source="system_monitor",
                timestamp=now - timedelta(minutes=5),
            ),
        ]
        
        # Add resolved alerts
        self.resolved_alerts = [
            Alert(
                id="alert_006",
                name="Database Connection",
                severity=AlertSeverity.HIGH,
                state=AlertState.RESOLVED,
                message="Database connection restored",
                source="database_monitor",
                timestamp=now - timedelta(hours=1),
                resolved_at=now - timedelta(minutes=45),
                acknowledged_by="operator2",
            ),
            Alert(
                id="alert_007",
                name="Disk Space Low",
                severity=AlertSeverity.MEDIUM,
                state=AlertState.RESOLVED,
                message="Disk space recovered after log rotation",
                source="system_monitor",
                timestamp=now - timedelta(hours=2),
                resolved_at=now - timedelta(hours=1, minutes=30),
            ),
        ]
        
        self._recalculate_counts()
    
    def watch_critical_count(self, value: int) -> None:
        """React to critical alert count changes."""
        self.update(self.render())
    
    def watch_unacknowledged_count(self, value: int) -> None:
        """React to unacknowledged count changes."""
        self.update(self.render())