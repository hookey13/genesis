"""Correlation heatmap widget for terminal UI."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
import numpy as np

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from rich.style import Style
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from genesis.core.models import Position
from genesis.analytics.correlation import CorrelationMonitor
from genesis.core.events import Event, EventType, EventPriority
from genesis.engine.event_bus import EventBus


class CorrelationHeatmap(Widget):
    """Real-time correlation heatmap widget."""
    
    DEFAULT_CSS = """
    CorrelationHeatmap {
        height: auto;
        width: 100%;
        border: solid $accent;
        padding: 1;
    }
    """
    
    correlation_matrix = reactive(np.array([]))
    positions = reactive([])
    last_update = reactive(datetime.now(timezone.utc))
    
    def __init__(
        self,
        correlation_monitor: Optional[CorrelationMonitor] = None,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        """Initialize correlation heatmap widget.
        
        Args:
            correlation_monitor: Correlation monitor instance
            event_bus: Event bus for subscribing to updates
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.correlation_monitor = correlation_monitor or CorrelationMonitor()
        self.event_bus = event_bus
        self.update_task = None
        self.update_interval = 5  # Update every 5 seconds
        
        # Color thresholds
        self.color_levels = {
            "very_high": (0.8, "red"),
            "high": (0.6, "bright_red"),
            "medium": (0.4, "yellow"),
            "low": (0.2, "green"),
            "very_low": (0.0, "bright_green")
        }
        
    async def on_mount(self) -> None:
        """Handle mount event."""
        # Subscribe to position updates if event bus available
        if self.event_bus:
            await self._subscribe_to_events()
            
        # Start periodic updates
        self.update_task = asyncio.create_task(self._periodic_update())
        
    async def on_unmount(self) -> None:
        """Handle unmount event."""
        # Cancel update task
        if self.update_task:
            self.update_task.cancel()
            
    async def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events from event bus."""
        # Subscribe to position updates
        async def handle_position_update(event: Event):
            if event.type in [EventType.POSITION_OPENED, EventType.POSITION_UPDATED, EventType.POSITION_CLOSED]:
                await self.update_correlation()
                
        # Would register handler with event bus
        # self.event_bus.subscribe(EventType.POSITION_OPENED, handle_position_update)
        
    async def _periodic_update(self) -> None:
        """Periodically update correlation data."""
        while True:
            try:
                await asyncio.sleep(self.update_interval)
                await self.update_correlation()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error(f"Error in periodic update: {e}")
                
    async def update_correlation(self, positions: Optional[List[Position]] = None) -> None:
        """Update correlation matrix with new position data.
        
        Args:
            positions: List of positions to analyze
        """
        if positions:
            self.positions = positions
            
        if len(self.positions) < 2:
            self.correlation_matrix = np.array([])
            self.last_update = datetime.now(timezone.utc)
            self.refresh()
            return
            
        try:
            # Calculate correlation matrix
            matrix = await self.correlation_monitor.calculate_correlation_matrix(self.positions)
            self.correlation_matrix = matrix
            self.last_update = datetime.now(timezone.utc)
            self.refresh()
        except Exception as e:
            self.log.error(f"Failed to update correlation: {e}")
            
    def render(self) -> RenderableType:
        """Render the correlation heatmap."""
        if self.correlation_matrix.size == 0:
            return self._render_empty()
            
        return self._render_heatmap()
        
    def _render_empty(self) -> RenderableType:
        """Render empty state."""
        return Text(
            "No correlation data available (need at least 2 positions)",
            style="dim italic"
        )
        
    def _render_heatmap(self) -> RenderableType:
        """Render the correlation heatmap table."""
        table = Table(
            title="Portfolio Correlation Heatmap",
            show_header=True,
            header_style="bold",
            title_style="bold cyan",
            border_style="bright_black"
        )
        
        # Add symbol column
        table.add_column("Symbol", style="bold", no_wrap=True)
        
        # Add columns for each position
        for position in self.positions:
            symbol = position.symbol.split('/')[0]  # Get base currency
            table.add_column(symbol, justify="center", width=8)
            
        # Add rows
        for i, position in enumerate(self.positions):
            row = [position.symbol]
            
            for j in range(len(self.positions)):
                correlation = self.correlation_matrix[i, j]
                
                # Format cell
                if i == j:
                    # Diagonal - always 1
                    cell = Text("1.00", style="dim")
                else:
                    # Get color based on correlation level
                    color = self._get_correlation_color(abs(correlation))
                    
                    # Format value
                    if correlation >= 0:
                        cell = Text(f"+{correlation:.2f}", style=color)
                    else:
                        cell = Text(f"{correlation:.2f}", style=color)
                        
                row.append(cell)
                
            table.add_row(*row)
            
        # Add footer with metadata
        footer = Text()
        footer.append(f"Last Update: {self.last_update.strftime('%H:%M:%S')} | ", style="dim")
        footer.append(f"Positions: {len(self.positions)} | ", style="dim")
        
        # Calculate average correlation
        if self.correlation_matrix.size > 1:
            avg_corr = self._calculate_average_correlation()
            avg_color = self._get_correlation_color(avg_corr)
            footer.append("Avg Correlation: ", style="dim")
            footer.append(f"{avg_corr:.2%}", style=avg_color)
        
        # Create container with table and footer
        from rich.console import Group
        return Group(table, footer)
        
    def _get_correlation_color(self, correlation: float) -> str:
        """Get color style for correlation value.
        
        Args:
            correlation: Correlation value (0-1)
            
        Returns:
            Color style string
        """
        for level_name, (threshold, color) in self.color_levels.items():
            if abs(correlation) >= threshold:
                return color
        return "white"
        
    def _calculate_average_correlation(self) -> float:
        """Calculate average correlation excluding diagonal."""
        if self.correlation_matrix.size <= 1:
            return 0.0
            
        n = len(self.correlation_matrix)
        if n == 1:
            return 0.0
            
        # Get upper triangle excluding diagonal
        upper_triangle = np.triu(self.correlation_matrix, k=1)
        non_zero_count = n * (n - 1) / 2  # Number of pairs
        
        if non_zero_count == 0:
            return 0.0
            
        return float(np.sum(np.abs(upper_triangle)) / non_zero_count)
        
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get correlation summary statistics.
        
        Returns:
            Dictionary with correlation statistics
        """
        if self.correlation_matrix.size == 0:
            return {
                "positions": 0,
                "avg_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "high_correlation_pairs": []
            }
            
        # Find high correlation pairs
        high_corr_pairs = []
        n = len(self.positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(self.correlation_matrix[i, j])
                if corr > 0.6:  # High correlation threshold
                    high_corr_pairs.append({
                        "pair": f"{self.positions[i].symbol}-{self.positions[j].symbol}",
                        "correlation": float(corr)
                    })
                    
        # Sort by correlation
        high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
        
        # Calculate statistics
        upper_triangle = np.triu(self.correlation_matrix, k=1)
        non_diagonal = upper_triangle[upper_triangle != 0]
        
        return {
            "positions": len(self.positions),
            "avg_correlation": self._calculate_average_correlation(),
            "max_correlation": float(np.max(np.abs(non_diagonal))) if non_diagonal.size > 0 else 0.0,
            "min_correlation": float(np.min(np.abs(non_diagonal))) if non_diagonal.size > 0 else 0.0,
            "high_correlation_pairs": high_corr_pairs[:5]  # Top 5 pairs
        }


class CorrelationSummaryWidget(Static):
    """Compact correlation summary widget."""
    
    def __init__(
        self,
        correlation_monitor: Optional[CorrelationMonitor] = None,
        **kwargs
    ):
        """Initialize correlation summary widget.
        
        Args:
            correlation_monitor: Correlation monitor instance
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.correlation_monitor = correlation_monitor or CorrelationMonitor()
        self.summary_data = {}
        
    def update_summary(self, summary: Dict[str, Any]) -> None:
        """Update summary data.
        
        Args:
            summary: Summary data from heatmap widget
        """
        self.summary_data = summary
        self.update(self._render_summary())
        
    def _render_summary(self) -> Text:
        """Render compact summary."""
        if not self.summary_data:
            return Text("No correlation data", style="dim")
            
        text = Text()
        
        # Average correlation with color
        avg_corr = self.summary_data.get("avg_correlation", 0.0)
        if avg_corr > 0.8:
            color = "red"
            icon = "⚠️"
        elif avg_corr > 0.6:
            color = "yellow"
            icon = "⚡"
        else:
            color = "green"
            icon = "✓"
            
        text.append(f"{icon} Portfolio Correlation: ", style="bold")
        text.append(f"{avg_corr:.1%}", style=f"bold {color}")
        
        # High correlation warning
        high_pairs = self.summary_data.get("high_correlation_pairs", [])
        if high_pairs:
            text.append(f" | ⚠️ {len(high_pairs)} high correlation pair(s)", style="yellow")
            
        return text