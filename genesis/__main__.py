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

import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, validate_configuration
from genesis.utils.logger import LoggerType, get_logger, setup_logging


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
        Initialize the application.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load configuration first
            print("📋 Loading configuration...")
            self.settings = get_settings()

            # Setup logging based on configuration
            setup_logging(
                log_level=self.settings.logging.log_level.value,
                log_dir=self.settings.logging.log_file_path.parent,
                enable_json=self.settings.deployment.deployment_env != "development",
                max_bytes=self.settings.logging.log_max_bytes,
                backup_count=self.settings.logging.log_backup_count,
            )

            # Get logger after setup
            self.logger = get_logger(__name__, LoggerType.SYSTEM)
            self.logger.info(
                "genesis_initializing",
                version="1.0.0",
                tier=self.settings.trading.trading_tier,
                environment=self.settings.deployment.deployment_env,
                testnet=self.settings.exchange.binance_testnet,
            )

            # Validate configuration
            print("✓ Configuration loaded successfully")
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

            self.logger.info("genesis_initialized_successfully")
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
