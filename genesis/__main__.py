"""
Main entry point for Project GENESIS.

This module serves as the entry point when running the application
with `python -m genesis`.

Features:
    - Configuration validation on startup
    - Structured logging initialization
    - Graceful error handling and shutdown
    - Signal handling for clean termination
"""

import asyncio
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import aiohttp
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, validate_configuration
from genesis.utils.logger import LoggerType, get_logger, setup_logging
from genesis.utils.time_sync import check_clock_drift_ms


class GenesisApplication:
    """
    Main application class for Project GENESIS.

    Handles initialization, configuration, and graceful shutdown
    of the trading system.
    """

    def __init__(self):
        """Initialize the application."""
        self.logger: Optional[Any] = None
        self.settings = None
        self.running = False

    def setup_signal_handlers(self) -> None:
        """Configure signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            """Handle shutdown signals."""
            if self.logger:
                self.logger.info("shutdown_signal_received", signal=signum)
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize(self) -> bool:
        """
        Initialize the application with full bootstrap sequence.

        Bootstrap order:
        1. Load & validate settings
        2. Initialize structured logging
        3. Database connection test & migrations
        4. REST API ping (exchange connectivity)
        5. Clock drift check (hard fail if exceeded)
        6. WebSocket probe (warn only)

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Step 1: Load configuration
            print("📋 Loading configuration...")
            self.settings = get_settings()
            print("✅ Configuration loaded successfully")

            # Step 2: Setup logging
            setup_logging(
                log_level=self.settings.logging.log_level.value,
                log_dir=self.settings.logging.log_file_path.parent,
                enable_json=self.settings.deployment.deployment_env != "development",
                max_bytes=self.settings.logging.log_max_bytes,
                backup_count=self.settings.logging.log_backup_count,
            )
            self.logger = get_logger(__name__, LoggerType.SYSTEM)
            print("✅ Logging initialized")

            # Log redacted configuration
            self.logger.info(
                "configuration_loaded", config=self.settings.redacted_dict()
            )

            # Step 3: Database smoke test & migrations
            print("🗄️  Testing database connection...")
            if not self._test_database_connection():
                return False
            print("✅ Database connection successful")

            print("🔄 Running database migrations...")
            if not self._run_migrations():
                return False
            print("✅ Database migrations complete")

            # Step 4: REST API ping
            print("🌐 Testing exchange REST connectivity...")
            if not asyncio.run(self._test_rest_connectivity()):
                return False
            print("✅ REST API connection successful")

            # Step 5: Clock drift check (hard fail)
            print("⏰ Checking clock synchronization...")
            drift_result = asyncio.run(
                check_clock_drift_ms(self.settings.time_sync.max_clock_drift_ms)
            )
            if not drift_result.is_acceptable:
                self.logger.error(
                    "clock_drift_exceeded",
                    drift_ms=drift_result.drift_ms,
                    max_ms=self.settings.time_sync.max_clock_drift_ms,
                    source=drift_result.source,
                )
                print(f"❌ Clock drift too high: {drift_result.drift_ms}ms")
                print(
                    f"   Maximum allowed: {self.settings.time_sync.max_clock_drift_ms}ms"
                )
                print("   Please sync your system time and try again.")
                return False
            print(f"✅ Clock synchronized (drift: {drift_result.drift_ms}ms)")

            # Step 6: WebSocket probe (warn only)
            print("🔌 Testing WebSocket connectivity...")
            ws_result = asyncio.run(self._test_websocket_connectivity())
            if not ws_result:
                self.logger.warning("websocket_connectivity_degraded")
                print("⚠️  WebSocket connection failed - service degraded")
                print("   Live data feeds may not be available")
            else:
                print("✅ WebSocket connection successful")

            # Validate configuration
            validation_report = validate_configuration()
            if not validation_report["valid"]:
                self.logger.error(
                    "configuration_invalid", error=validation_report.get("error")
                )
                return False

            # Log warnings
            for warning in validation_report["warnings"]:
                self.logger.warning("configuration_warning", message=warning)
                print(f"⚠️  {warning}")

            # Setup signal handlers
            self.setup_signal_handlers()

            self.logger.info(
                "startup.complete",
                version="1.0.0",
                tier=self.settings.trading.trading_tier,
                environment=self.settings.deployment.deployment_env,
                testnet=self.settings.exchange.binance_testnet,
            )
            print("\n✅ All preflight checks passed - system ready")
            return True

        except FileNotFoundError as e:
            print(f"\n❌ Configuration file not found: {e}")
            print("📝 Copy .env.example to .env and configure your settings")
            return False

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "initialization_failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            else:
                print(f"\n❌ Initialization failed: {e}")
                traceback.print_exc()
            return False

    def run(self) -> int:
        """
        Run the main application loop.

        Returns:
            int: Exit code (0 for success, non-zero for error)
        """
        if not self.initialize():
            return 1

        try:
            self.running = True

            print("\n" + "=" * 60)
            print("🚀 PROJECT GENESIS - TRADING SYSTEM")
            print("=" * 60)
            print(f"📊 Tier: {self.settings.trading.trading_tier.upper()}")
            print(f"🌍 Environment: {self.settings.deployment.deployment_env}")
            print(
                f"🔧 Testnet: {'Enabled' if self.settings.exchange.binance_testnet else 'Disabled'}"
            )
            print(f"💰 Max Position: ${self.settings.trading.max_position_size_usdt}")
            print(f"🛑 Daily Loss Limit: ${self.settings.trading.max_daily_loss_usdt}")
            print("=" * 60)

            self.logger.info(
                "genesis_started", trading_pairs=self.settings.trading.trading_pairs
            )

            # TODO: Initialize and start trading engine
            print("\n⚠️  Trading engine not yet implemented")
            print("📚 This is the infrastructure setup phase")
            print("\n💡 Next steps:")
            print("  1. Implement core trading engine")
            print("  2. Add exchange connectivity")
            print("  3. Implement trading strategies")
            print("  4. Add tilt detection system")

            # For now, just indicate successful startup
            self.logger.info("genesis_ready", status="awaiting_implementation")

            return 0

        except KeyboardInterrupt:
            self.logger.info("keyboard_interrupt_received")
            return 0

        except Exception as e:
            self.logger.error(
                "runtime_error", error=str(e), traceback=traceback.format_exc()
            )
            return 1

        finally:
            self.shutdown()

    def _test_database_connection(self) -> bool:
        """Test database connectivity."""
        try:
            engine = create_engine(self.settings.database.database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            engine.dispose()
            return True
        except Exception as e:
            if self.logger:
                self.logger.error("database_connection_failed", error=str(e))
            else:
                print(f"❌ Database connection failed: {e}")
            return False

    def _run_migrations(self) -> bool:
        """Run Alembic migrations programmatically."""
        try:
            alembic_cfg = AlembicConfig(str(project_root / "alembic.ini"))
            alembic_cfg.set_main_option(
                "sqlalchemy.url", self.settings.database.database_url
            )
            command.upgrade(alembic_cfg, "head")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error("migration_failed", error=str(e))
            else:
                print(f"❌ Migration failed: {e}")
            return False

    async def _test_rest_connectivity(self) -> bool:
        """Test REST API connectivity to exchange."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://testnet.binance.vision/api/v3/ping"
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            if self.logger:
                self.logger.error("rest_connectivity_failed", error=str(e))
            else:
                print(f"❌ REST connectivity failed: {e}")
            return False

    async def _test_websocket_connectivity(self) -> bool:
        """Test WebSocket connectivity to exchange."""
        try:
            import websockets

            ws_url = "wss://testnet.binance.vision/ws"
            async with websockets.connect(
                ws_url, timeout=10, close_timeout=1
            ) as websocket:
                # Send ping
                await websocket.ping()
                return True
        except Exception as e:
            if self.logger:
                self.logger.warning("websocket_connectivity_failed", error=str(e))
            return False

    def shutdown(self) -> None:
        """Perform graceful shutdown."""
        if not self.running:
            return

        self.running = False

        if self.logger:
            self.logger.info("genesis_shutting_down")

        try:
            # TODO: Add cleanup for:
            # - Open positions
            # - Active connections
            # - Pending orders
            # - Save state

            if self.logger:
                self.logger.info("genesis_shutdown_complete")

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "shutdown_error", error=str(e), traceback=traceback.format_exc()
                )


def main() -> int:
    """
    Main entry point for the application.

    Returns:
        int: Exit code
    """
    app = GenesisApplication()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
