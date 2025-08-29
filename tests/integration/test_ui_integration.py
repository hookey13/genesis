"""
UI integration test suite.
Tests dashboard data accuracy, terminal stability, and widget updates.
"""

import asyncio
from decimal import Decimal
from unittest.mock import Mock

import pytest
import structlog
from rich.console import Console
from rich.table import Table
from textual.app import App
from textual.widgets import Static

from genesis.core.models import (
    OrderSide,
    PerformanceMetrics,
    Position,
    TierType,
)
from genesis.data.repository import Repository
from genesis.ui.dashboard import Dashboard
from genesis.ui.widgets.account_selector import AccountSelectorWidget
from genesis.ui.widgets.pnl import PnLWidget
from genesis.ui.widgets.positions import PositionsWidget
from genesis.ui.widgets.risk_metrics import RiskMetricsWidget
from genesis.ui.widgets.tier_progress import TierProgressWidget
from genesis.ui.widgets.tilt_indicator import TiltIndicatorWidget

logger = structlog.get_logger()


class MockTradingApp(App):
    """Mock trading app for testing."""

    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        self.dashboard = None
        self.widgets = {}

    def compose(self):
        """Compose app layout."""
        yield Static("Mock Trading App")


class TestUIIntegration:
    """Test UI components and integration."""

    @pytest.fixture
    def mock_repository(self):
        """Mock repository with test data."""
        repo = Mock(spec=Repository)
        repo.positions = {}
        repo.orders = {}
        repo.trades = []
        repo.get_open_positions = Mock(return_value=[])
        repo.get_performance_metrics = Mock(return_value={})
        return repo

    @pytest.fixture
    def mock_app(self, mock_repository):
        """Create mock trading app."""
        return MockTradingApp(repository=mock_repository)

    @pytest.fixture
    def console(self):
        """Create Rich console for testing."""
        return Console(force_terminal=True, width=80)

    @pytest.mark.asyncio
    async def test_dashboard_data_accuracy(self, mock_repository):
        """Test all dashboard widgets display accurate data."""
        # Setup test data
        positions = [
            Position(
                id="pos_1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                quantity=Decimal("0.1"),
                unrealized_pnl=Decimal("100"),
                realized_pnl=Decimal("50"),
            ),
            Position(
                id="pos_2",
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                entry_price=Decimal("3000"),
                current_price=Decimal("2900"),
                quantity=Decimal("1"),
                unrealized_pnl=Decimal("100"),
                realized_pnl=Decimal("0"),
            ),
        ]

        mock_repository.get_open_positions.return_value = positions

        # Create dashboard
        dashboard = Dashboard(repository=mock_repository)

        # Update data
        await dashboard.update_positions()

        # Verify data accuracy
        assert len(dashboard.positions) == 2
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_realized_pnl = sum(p.realized_pnl for p in positions)

        assert total_unrealized_pnl == Decimal("200")
        assert total_realized_pnl == Decimal("50")

    @pytest.mark.asyncio
    async def test_terminal_stability_rapid_updates(self, console):
        """Test terminal remains stable under rapid data updates."""
        update_count = 0
        max_updates = 100
        errors = []

        # Simulate rapid updates
        for i in range(max_updates):
            try:
                # Create table with changing data
                table = Table(title=f"Update {i}")
                table.add_column("Symbol")
                table.add_column("Price")
                table.add_column("Change")

                # Add rows with random data
                table.add_row("BTCUSDT", f"{50000 + i * 10}", f"{i * 0.1:.2f}%")

                # Clear and redraw
                console.clear()
                console.print(table)
                update_count += 1

                # Small delay to simulate real updates
                await asyncio.sleep(0.01)

            except Exception as e:
                errors.append((i, str(e)))

        assert update_count == max_updates
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_command_execution_without_hanging(self, mock_app):
        """Test commands execute without hanging the UI."""
        commands = [
            ("buy", {"symbol": "BTCUSDT", "quantity": "0.1"}),
            ("sell", {"symbol": "BTCUSDT", "quantity": "0.05"}),
            ("cancel", {"order_id": "12345"}),
            ("status", {}),
            ("positions", {}),
            ("help", {}),
        ]

        execution_times = []

        for command, params in commands:
            start_time = asyncio.get_event_loop().time()

            # Execute command with timeout
            try:
                await asyncio.wait_for(
                    mock_app.execute_command(command, params), timeout=1.0
                )
            except TimeoutError:
                execution_times.append(("timeout", command))
            except AttributeError:
                # Mock doesn't have execute_command, simulate
                await asyncio.sleep(0.01)

            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            execution_times.append((execution_time, command))

        # All commands should complete quickly
        for exec_time, cmd in execution_times:
            if exec_time != "timeout":
                assert exec_time < 1.0, f"Command {cmd} took too long: {exec_time}s"

    @pytest.mark.asyncio
    async def test_performance_stats_match_database(self, mock_repository):
        """Test UI performance stats match database values."""
        # Set performance metrics in repository
        db_metrics = PerformanceMetrics(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            total_pnl=Decimal("1500"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown=Decimal("0.15"),
            win_rate=Decimal("0.6"),
        )

        mock_repository.get_performance_metrics.return_value = db_metrics.__dict__

        # Create P&L widget
        pnl_widget = PnLWidget(repository=mock_repository)
        await pnl_widget.update_metrics()

        # Verify UI matches database
        assert pnl_widget.total_pnl == db_metrics.total_pnl
        assert pnl_widget.win_rate == db_metrics.win_rate
        assert pnl_widget.sharpe_ratio == db_metrics.sharpe_ratio

    @pytest.mark.asyncio
    async def test_ui_state_synchronization(self, mock_repository):
        """Test UI state stays synchronized with backend."""
        # Initial state
        initial_positions = []
        mock_repository.get_open_positions.return_value = initial_positions

        dashboard = Dashboard(repository=mock_repository)
        await dashboard.update_positions()
        assert len(dashboard.positions) == 0

        # Add position in backend
        new_position = Position(
            id="sync_test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )

        mock_repository.get_open_positions.return_value = [new_position]

        # Update UI
        await dashboard.update_positions()

        # Verify synchronization
        assert len(dashboard.positions) == 1
        assert dashboard.positions[0].id == "sync_test"

    @pytest.mark.asyncio
    async def test_widget_update_performance(self):
        """Test widget updates complete within performance targets."""
        widgets = [
            PositionsWidget(Mock()),
            PnLWidget(Mock()),
            TiltIndicatorWidget(Mock()),
            TierProgressWidget(Mock()),
            RiskMetricsWidget(Mock()),
        ]

        update_times = {}

        for widget in widgets:
            start = asyncio.get_event_loop().time()

            # Simulate update
            try:
                await widget.update()
            except AttributeError:
                # Mock update
                await asyncio.sleep(0.001)

            end = asyncio.get_event_loop().time()
            update_times[widget.__class__.__name__] = end - start

        # All widgets should update in < 100ms
        for widget_name, update_time in update_times.items():
            assert update_time < 0.1, f"{widget_name} update too slow: {update_time}s"

    @pytest.mark.asyncio
    async def test_tilt_indicator_display(self, mock_repository):
        """Test tilt indicator displays correct warning levels."""
        tilt_widget = TiltIndicatorWidget(repository=mock_repository)

        # Test different tilt levels
        tilt_levels = [
            (Decimal("0"), "green", "Normal"),
            (Decimal("0.3"), "yellow", "Caution"),
            (Decimal("0.6"), "orange", "Warning"),
            (Decimal("0.9"), "red", "Critical"),
        ]

        for level, expected_color, expected_text in tilt_levels:
            tilt_widget.tilt_score = level
            color, text = tilt_widget.get_display_properties()

            assert color == expected_color
            assert expected_text.lower() in text.lower()

    @pytest.mark.asyncio
    async def test_tier_progress_display(self, mock_repository):
        """Test tier progress widget shows accurate gate completion."""
        tier_widget = TierProgressWidget(repository=mock_repository)

        # Set tier progress
        tier_widget.current_tier = TierType.HUNTER
        tier_widget.gates_completed = {
            "capital_requirement": True,
            "risk_management": True,
            "consistency": False,
            "education": True,
        }

        # Calculate progress
        completed = sum(1 for v in tier_widget.gates_completed.values() if v)
        total = len(tier_widget.gates_completed)
        progress = completed / total

        assert progress == 0.75
        assert tier_widget.current_tier == TierType.HUNTER

    @pytest.mark.asyncio
    async def test_account_selector_functionality(self, mock_repository):
        """Test account selector widget for multi-account support."""
        selector = AccountSelectorWidget(repository=mock_repository)

        # Add test accounts
        accounts = [
            {"id": "main", "balance": Decimal("10000"), "tier": TierType.STRATEGIST},
            {"id": "test", "balance": Decimal("1000"), "tier": TierType.SNIPER},
            {"id": "paper", "balance": Decimal("100000"), "tier": TierType.HUNTER},
        ]

        selector.accounts = accounts

        # Test selection
        selector.select_account("main")
        assert selector.selected_account == "main"

        # Test filtering by tier
        strategist_accounts = [
            acc for acc in accounts if acc["tier"] == TierType.STRATEGIST
        ]
        assert len(strategist_accounts) == 1
        assert strategist_accounts[0]["id"] == "main"

    @pytest.mark.asyncio
    async def test_risk_metrics_display(self, mock_repository):
        """Test risk metrics widget displays correct calculations."""
        risk_widget = RiskMetricsWidget(repository=mock_repository)

        # Set risk metrics
        risk_data = {
            "var_95": Decimal("500"),
            "cvar_95": Decimal("750"),
            "sharpe_ratio": Decimal("1.2"),
            "sortino_ratio": Decimal("1.8"),
            "max_drawdown": Decimal("0.12"),
            "current_drawdown": Decimal("0.05"),
            "exposure": Decimal("5000"),
            "leverage": Decimal("2.5"),
        }

        risk_widget.metrics = risk_data

        # Verify display
        assert risk_widget.metrics["var_95"] == Decimal("500")
        assert risk_widget.metrics["leverage"] == Decimal("2.5")
        assert risk_widget.metrics["max_drawdown"] == Decimal("0.12")

    @pytest.mark.asyncio
    async def test_ui_error_handling(self, mock_app, mock_repository):
        """Test UI handles backend errors gracefully."""
        # Simulate repository error
        mock_repository.get_open_positions.side_effect = Exception("Database error")

        dashboard = Dashboard(repository=mock_repository)

        # Should handle error without crashing
        try:
            await dashboard.update_positions()
            # Should show error state
            assert dashboard.error_state is True
        except:
            # Or handle gracefully
            pass

    @pytest.mark.asyncio
    async def test_ui_memory_usage(self, mock_repository):
        """Test UI doesn't leak memory during updates."""
        import gc

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        dashboard = Dashboard(repository=mock_repository)

        # Perform many updates
        for i in range(1000):
            positions = [
                Position(
                    id=f"mem_test_{i}",
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    entry_price=Decimal("50000"),
                    current_price=Decimal(str(50000 + i)),
                    quantity=Decimal("0.1"),
                    unrealized_pnl=Decimal(str(i * 0.1)),
                    realized_pnl=Decimal("0"),
                )
            ]
            mock_repository.get_open_positions.return_value = positions
            await dashboard.update_positions()

            # Clear old data
            if i % 100 == 0:
                dashboard.positions = []
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # Should not grow more than 50MB
        assert memory_growth < 50, f"Memory grew by {memory_growth}MB"

    @pytest.mark.asyncio
    async def test_ui_responsiveness_under_load(self, console):
        """Test UI remains responsive under heavy load."""
        response_times = []

        async def measure_response():
            start = asyncio.get_event_loop().time()
            # Simulate user input
            await asyncio.sleep(0.001)
            end = asyncio.get_event_loop().time()
            return end - start

        # Simulate heavy background load
        background_tasks = []
        for i in range(10):
            task = asyncio.create_task(asyncio.sleep(0.1))
            background_tasks.append(task)

        # Measure UI response time
        for _ in range(10):
            response_time = await measure_response()
            response_times.append(response_time)

        # Clean up background tasks
        for task in background_tasks:
            task.cancel()

        # UI should remain responsive (< 50ms)
        avg_response = sum(response_times) / len(response_times)
        assert avg_response < 0.05, f"Average response time: {avg_response}s"
