"""Real-time validation status dashboard UI components."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, DataTable, ProgressBar, Label
from textual.widget import Widget

import structlog

logger = structlog.get_logger(__name__)


class ValidationOverview(Static):
    """Overview widget showing validation summary."""
    
    overall_score = reactive(0.0)
    ready_status = reactive(False)
    validators_passed = reactive(0)
    validators_total = reactive(0)
    
    def render(self) -> RenderableType:
        """Render the validation overview."""
        # Create summary table
        table = Table(title="Validation Overview", expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold")
        
        # Ready status
        status_icon = "✅" if self.ready_status else "❌"
        status_text = "GO" if self.ready_status else "NO-GO"
        status_style = "green" if self.ready_status else "red"
        table.add_row("Status", f"[{status_style}]{status_icon} {status_text}[/{status_style}]")
        
        # Overall score
        score_color = "green" if self.overall_score >= 95 else "yellow" if self.overall_score >= 80 else "red"
        table.add_row("Overall Score", f"[{score_color}]{self.overall_score:.1f}%[/{score_color}]")
        
        # Validators passed
        validators_text = f"{self.validators_passed}/{self.validators_total}"
        validators_color = "green" if self.validators_passed == self.validators_total else "yellow"
        table.add_row("Validators", f"[{validators_color}]{validators_text}[/{validators_color}]")
        
        return Panel(table, title="System Status", border_style="blue")
    
    def update_status(self, score: float, ready: bool, passed: int, total: int):
        """Update the overview status.
        
        Args:
            score: Overall validation score
            ready: Go/no-go readiness status
            passed: Number of validators passed
            total: Total number of validators
        """
        self.overall_score = score
        self.ready_status = ready
        self.validators_passed = passed
        self.validators_total = total


class ValidationProgress(Static):
    """Progress widget showing real-time validation progress."""
    
    def __init__(self, **kwargs):
        """Initialize the progress widget."""
        super().__init__(**kwargs)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            expand=True
        )
        self.tasks = {}
        
    def render(self) -> RenderableType:
        """Render the progress bars."""
        return Panel(self.progress, title="Validation Progress", border_style="green")
    
    def add_validator(self, validator_id: str, description: str, total: int = 100):
        """Add a validator to track.
        
        Args:
            validator_id: Unique validator identifier
            description: Validator description
            total: Total steps for the validator
        """
        if validator_id not in self.tasks:
            task_id = self.progress.add_task(description, total=total)
            self.tasks[validator_id] = task_id
    
    def update_validator(self, validator_id: str, completed: int):
        """Update validator progress.
        
        Args:
            validator_id: Validator identifier
            completed: Number of completed steps
        """
        if validator_id in self.tasks:
            self.progress.update(self.tasks[validator_id], completed=completed)
    
    def complete_validator(self, validator_id: str):
        """Mark a validator as complete.
        
        Args:
            validator_id: Validator identifier
        """
        if validator_id in self.tasks:
            task_id = self.tasks[validator_id]
            total = self.progress.tasks[task_id].total
            self.progress.update(task_id, completed=total)


class ValidationTree(Static):
    """Tree widget showing validation results hierarchy."""
    
    def __init__(self, **kwargs):
        """Initialize the validation tree."""
        super().__init__(**kwargs)
        self.results_tree = Tree("📋 Validation Results")
        self.category_nodes = {}
        
    def render(self) -> RenderableType:
        """Render the validation tree."""
        return Panel(self.results_tree, title="Results Tree", border_style="yellow")
    
    def add_category(self, category: str) -> Any:
        """Add a category node to the tree.
        
        Args:
            category: Category name
            
        Returns:
            Category tree node
        """
        if category not in self.category_nodes:
            icon = self._get_category_icon(category)
            node = self.results_tree.add(f"{icon} {category.capitalize()}")
            self.category_nodes[category] = node
        return self.category_nodes[category]
    
    def add_validator_result(
        self, 
        category: str, 
        validator_name: str, 
        passed: bool, 
        score: float,
        errors: List[str] = None,
        warnings: List[str] = None
    ):
        """Add a validator result to the tree.
        
        Args:
            category: Validator category
            validator_name: Validator name
            passed: Whether validation passed
            score: Validation score
            errors: List of errors
            warnings: List of warnings
        """
        category_node = self.add_category(category)
        
        # Format validator node
        status_icon = "✅" if passed else "❌"
        score_color = "green" if score >= 90 else "yellow" if score >= 70 else "red"
        validator_text = f"{status_icon} {validator_name}: [{score_color}]{score:.1f}%[/{score_color}]"
        
        validator_node = category_node.add(validator_text)
        
        # Add errors
        if errors:
            errors_node = validator_node.add("[red]⚠️ Errors[/red]")
            for error in errors[:5]:  # Limit to 5 errors in tree
                errors_node.add(f"[red]• {error}[/red]")
                
        # Add warnings
        if warnings:
            warnings_node = validator_node.add("[yellow]⚠️ Warnings[/yellow]")
            for warning in warnings[:5]:  # Limit to 5 warnings
                warnings_node.add(f"[yellow]• {warning}[/yellow]")
    
    def _get_category_icon(self, category: str) -> str:
        """Get icon for category.
        
        Args:
            category: Category name
            
        Returns:
            Category icon
        """
        icons = {
            "technical": "🔧",
            "security": "🔒",
            "operational": "⚙️",
            "business": "💼"
        }
        return icons.get(category, "📁")
    
    def clear(self):
        """Clear the tree."""
        self.results_tree = Tree("📋 Validation Results")
        self.category_nodes = {}


class ValidationDetails(Static):
    """Detailed view of selected validation result."""
    
    def __init__(self, **kwargs):
        """Initialize the details widget."""
        super().__init__(**kwargs)
        self.current_validator = None
        self.current_details = {}
        
    def render(self) -> RenderableType:
        """Render the validation details."""
        if not self.current_validator:
            return Panel(
                Text("Select a validator to view details", style="dim"),
                title="Validation Details",
                border_style="cyan"
            )
        
        # Create details table
        table = Table(expand=True, show_header=False)
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value")
        
        # Add validator info
        table.add_row("Validator", self.current_validator)
        
        if self.current_details:
            # Status
            passed = self.current_details.get("passed", False)
            status_text = "PASSED" if passed else "FAILED"
            status_style = "green" if passed else "red"
            table.add_row("Status", f"[{status_style}]{status_text}[/{status_style}]")
            
            # Score
            score = self.current_details.get("score", 0)
            score_color = "green" if score >= 90 else "yellow" if score >= 70 else "red"
            table.add_row("Score", f"[{score_color}]{score:.1f}%[/{score_color}]")
            
            # Duration
            duration = self.current_details.get("duration_seconds", 0)
            table.add_row("Duration", f"{duration:.2f}s")
            
            # Timestamp
            timestamp = self.current_details.get("timestamp", "")
            table.add_row("Timestamp", timestamp)
            
            # Checks summary
            checks = self.current_details.get("checks", [])
            if checks:
                passed_checks = sum(1 for c in checks if c.get("passed", False))
                table.add_row("Checks", f"{passed_checks}/{len(checks)} passed")
            
            # Errors
            errors = self.current_details.get("errors", [])
            if errors:
                table.add_row("Errors", f"{len(errors)} error(s)")
                for i, error in enumerate(errors[:3], 1):
                    table.add_row(f"  Error {i}", f"[red]{error}[/red]")
            
            # Warnings
            warnings = self.current_details.get("warnings", [])
            if warnings:
                table.add_row("Warnings", f"{len(warnings)} warning(s)")
                for i, warning in enumerate(warnings[:3], 1):
                    table.add_row(f"  Warning {i}", f"[yellow]{warning}[/yellow]")
        
        return Panel(table, title=f"Details: {self.current_validator}", border_style="cyan")
    
    def show_validator_details(self, validator_name: str, details: Dict[str, Any]):
        """Show details for a specific validator.
        
        Args:
            validator_name: Name of the validator
            details: Validator result details
        """
        self.current_validator = validator_name
        self.current_details = details


class ValidationDashboard(Container):
    """Main validation dashboard combining all widgets."""
    
    def __init__(self, **kwargs):
        """Initialize the validation dashboard."""
        super().__init__(**kwargs)
        self.overview = ValidationOverview()
        self.progress = ValidationProgress()
        self.tree = ValidationTree()
        self.details = ValidationDetails()
        self.validation_results = {}
        
    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Header(show_clock=True)
        
        with Horizontal():
            with Vertical(classes="column"):
                yield self.overview
                yield self.progress
            
            with Vertical(classes="column"):
                yield self.tree
        
        yield self.details
        yield Footer()
    
    async def update_validation_status(self, validator_id: str, status: str, details: Dict[str, Any] = None):
        """Update status for a specific validator.
        
        Args:
            validator_id: Validator identifier
            status: Current status (pending, running, completed, failed)
            details: Optional validation details
        """
        logger.info(f"Updating validator {validator_id} status to {status}")
        
        if status == "pending":
            self.progress.add_validator(validator_id, f"Pending: {validator_id}")
            
        elif status == "running":
            self.progress.add_validator(validator_id, f"Running: {validator_id}")
            self.progress.update_validator(validator_id, 50)
            
        elif status in ["completed", "failed"]:
            self.progress.complete_validator(validator_id)
            
            if details:
                # Store results
                self.validation_results[validator_id] = details
                
                # Update tree
                category = details.get("category", "unknown")
                passed = details.get("passed", False)
                score = details.get("score", 0)
                errors = details.get("errors", [])
                warnings = details.get("warnings", [])
                
                self.tree.add_validator_result(
                    category=category,
                    validator_name=validator_id,
                    passed=passed,
                    score=score,
                    errors=errors,
                    warnings=warnings
                )
                
                # Update overview
                self._update_overview()
    
    def _update_overview(self):
        """Update the overview widget based on current results."""
        if not self.validation_results:
            return
            
        # Calculate metrics
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results.values() if r.get("passed", False))
        total_score = sum(r.get("score", 0) for r in self.validation_results.values())
        overall_score = total_score / total if total > 0 else 0
        
        # Determine readiness (simplified logic)
        ready = overall_score >= 95 and passed == total
        
        self.overview.update_status(
            score=overall_score,
            ready=ready,
            passed=passed,
            total=total
        )
    
    def show_validator_details(self, validator_id: str):
        """Show detailed view for a validator.
        
        Args:
            validator_id: Validator identifier
        """
        if validator_id in self.validation_results:
            self.details.show_validator_details(
                validator_id,
                self.validation_results[validator_id]
            )
    
    def clear_results(self):
        """Clear all validation results."""
        self.validation_results = {}
        self.tree.clear()
        self.overview.update_status(0, False, 0, 0)
        
    async def run_validation_pipeline(self, pipeline_name: str = "standard"):
        """Run a validation pipeline and update UI.
        
        Args:
            pipeline_name: Name of the pipeline to run
        """
        from genesis.validation.orchestrator import ValidationOrchestrator
        
        logger.info(f"Starting validation pipeline: {pipeline_name}")
        self.clear_results()
        
        try:
            orchestrator = ValidationOrchestrator()
            
            # Start validation
            report = await orchestrator.run_pipeline(pipeline_name)
            
            # Update UI with results
            for result in report.results:
                await self.update_validation_status(
                    validator_id=result.validator_name,
                    status="completed" if result.passed else "failed",
                    details=result.to_dict()
                )
            
            # Final overview update
            self.overview.update_status(
                score=report.overall_score,
                ready=report.ready,
                passed=sum(1 for r in report.results if r.passed),
                total=len(report.results)
            )
            
            logger.info(f"Pipeline {pipeline_name} completed", ready=report.ready)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise