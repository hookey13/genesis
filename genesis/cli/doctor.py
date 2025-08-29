"""
Doctor command for preflight system checks.

Runs comprehensive health checks to ensure the system
is properly configured and ready for operation.
"""

import asyncio
import sys
from pathlib import Path

import aiohttp
import click
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine, text

from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import ValidationError, get_settings, validate_configuration
from genesis.utils.time_sync import check_clock_drift_ms

console = Console()


class HealthCheck:
    """Individual health check result."""

    def __init__(self, name: str, passed: bool, message: str, severity: str = "error"):
        self.name = name
        self.passed = passed
        self.message = message
        self.severity = severity  # "error", "warning", "info"


class DoctorRunner:
    """Runs system health checks."""

    def __init__(self):
        self.checks: list[HealthCheck] = []
        self.settings = None

    async def run_all_checks(self) -> bool:
        """Run all health checks in sequence."""
        console.print(
            "\n[bold cyan]🏥 Running Genesis Doctor Health Checks...[/bold cyan]\n"
        )

        # 1. Configuration validation
        self._check_configuration()

        # Only continue if config loaded successfully
        if self.settings:
            # 2. Database connectivity
            self._check_database()

            # 3. Database migrations
            self._check_migrations()

            # 4. REST API connectivity
            await self._check_rest_api()

            # 5. Clock drift
            await self._check_clock_drift()

            # 6. WebSocket connectivity
            await self._check_websocket()

        # Display results
        self._display_results()

        # Return True if all critical checks passed
        critical_checks_passed = all(
            check.passed for check in self.checks if check.severity == "error"
        )

        return critical_checks_passed

    def _check_configuration(self):
        """Check configuration validity."""
        try:
            self.settings = get_settings()

            # Run validation
            report = validate_configuration()

            if report["valid"]:
                self.checks.append(
                    HealthCheck(
                        "Configuration",
                        True,
                        f"Valid ({report['tier']} tier, {report['environment']} env)",
                    )
                )

                # Add warnings as separate checks
                for warning in report["warnings"]:
                    self.checks.append(
                        HealthCheck(
                            "Config Warning", False, warning, severity="warning"
                        )
                    )
            else:
                self.checks.append(
                    HealthCheck(
                        "Configuration",
                        False,
                        report.get("error", "Invalid configuration"),
                    )
                )

        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                errors.append(f"{field}: {error['msg']}")

            self.checks.append(
                HealthCheck(
                    "Configuration", False, "; ".join(errors[:3])  # Show first 3 errors
                )
            )
            self.settings = None

        except FileNotFoundError:
            self.checks.append(
                HealthCheck(
                    "Configuration",
                    False,
                    ".env file not found - copy .env.example to .env",
                )
            )
            self.settings = None

        except Exception as e:
            self.checks.append(HealthCheck("Configuration", False, str(e)))
            self.settings = None

    def _check_database(self):
        """Check database connectivity."""
        if not self.settings:
            return

        try:
            engine = create_engine(self.settings.database.database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            engine.dispose()

            self.checks.append(
                HealthCheck(
                    "Database Connection",
                    True,
                    f"Connected to {self.settings.database.database_url.split('://')[0]}",
                )
            )
        except Exception as e:
            self.checks.append(HealthCheck("Database Connection", False, str(e)[:100]))

    def _check_migrations(self):
        """Check if database migrations are up to date."""
        if not self.settings:
            return

        try:
            engine = create_engine(self.settings.database.database_url)

            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()

            # Get latest revision from Alembic
            alembic_cfg = AlembicConfig(str(project_root / "alembic.ini"))
            script = ScriptDirectory.from_config(alembic_cfg)
            head_rev = script.get_current_head()

            if current_rev == head_rev:
                self.checks.append(
                    HealthCheck(
                        "Database Migrations",
                        True,
                        f"Up to date (revision: {current_rev or 'initial'})",
                    )
                )
            else:
                self.checks.append(
                    HealthCheck(
                        "Database Migrations",
                        False,
                        f"Behind (current: {current_rev or 'none'}, latest: {head_rev})",
                        severity="warning",
                    )
                )

            engine.dispose()

        except Exception as e:
            self.checks.append(
                HealthCheck(
                    "Database Migrations", False, str(e)[:100], severity="warning"
                )
            )

    async def _check_rest_api(self):
        """Check REST API connectivity to exchange."""
        if not self.settings:
            return

        try:
            async with aiohttp.ClientSession() as session:
                url = "https://testnet.binance.vision/api/v3/ping"
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.checks.append(
                            HealthCheck(
                                "REST API (Binance Testnet)",
                                True,
                                "Connected successfully",
                            )
                        )
                    else:
                        self.checks.append(
                            HealthCheck(
                                "REST API (Binance Testnet)",
                                False,
                                f"HTTP {response.status}",
                            )
                        )
        except Exception as e:
            self.checks.append(
                HealthCheck("REST API (Binance Testnet)", False, str(e)[:100])
            )

    async def _check_clock_drift(self):
        """Check system clock synchronization."""
        if not self.settings:
            return

        try:
            result = await check_clock_drift_ms(
                self.settings.time_sync.max_clock_drift_ms
            )

            if result.is_acceptable:
                self.checks.append(
                    HealthCheck(
                        "Clock Synchronization",
                        True,
                        f"Drift: {result.drift_ms}ms (max: {self.settings.time_sync.max_clock_drift_ms}ms)",
                    )
                )
            else:
                self.checks.append(
                    HealthCheck(
                        "Clock Synchronization",
                        False,
                        f"Drift too high: {result.drift_ms}ms (max: {self.settings.time_sync.max_clock_drift_ms}ms)",
                    )
                )
        except Exception as e:
            self.checks.append(
                HealthCheck("Clock Synchronization", False, str(e)[:100])
            )

    async def _check_websocket(self):
        """Check WebSocket connectivity to exchange."""
        if not self.settings:
            return

        try:
            import websockets

            ws_url = "wss://testnet.binance.vision/ws"
            async with websockets.connect(
                ws_url, timeout=10, close_timeout=1
            ) as websocket:
                await websocket.ping()

                self.checks.append(
                    HealthCheck(
                        "WebSocket (Binance Testnet)",
                        True,
                        "Connected successfully",
                        severity="warning",  # Not critical
                    )
                )
        except Exception as e:
            self.checks.append(
                HealthCheck(
                    "WebSocket (Binance Testnet)",
                    False,
                    str(e)[:100],
                    severity="warning",  # Not critical
                )
            )

    def _display_results(self):
        """Display check results in a formatted table."""
        table = Table(title="Health Check Results", show_lines=True)
        table.add_column("Check", style="cyan", width=30)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Details", style="dim", width=60)

        error_count = 0
        warning_count = 0

        for check in self.checks:
            if check.passed:
                status = "[green]✅ PASS[/green]"
            elif check.severity == "warning":
                status = "[yellow]⚠️  WARN[/yellow]"
                warning_count += 1
            else:
                status = "[red]❌ FAIL[/red]"
                error_count += 1

            table.add_row(check.name, status, check.message)

        console.print(table)

        # Summary
        console.print("\n[bold]Summary:[/bold]")

        if error_count == 0 and warning_count == 0:
            console.print("[green]✅ All checks passed - system is healthy![/green]")
        elif error_count == 0:
            console.print(
                f"[yellow]⚠️  System operational with {warning_count} warning(s)[/yellow]"
            )
        else:
            console.print(f"[red]❌ {error_count} critical issue(s) found[/red]")
            if warning_count > 0:
                console.print(f"[yellow]   Plus {warning_count} warning(s)[/yellow]")

        console.print()


@click.command()
def doctor():
    """Run preflight system health checks."""
    runner = DoctorRunner()

    # Run async checks
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        all_passed = loop.run_until_complete(runner.run_all_checks())

        if not all_passed:
            console.print(
                "[red]Please fix critical issues before starting Genesis.[/red]"
            )
            sys.exit(1)
        else:
            console.print("[green]Ready to run Genesis![/green]")
            sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Health check interrupted[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    doctor()
