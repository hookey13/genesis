"""
Runtime configuration management with hot-reload capability.

Enables dynamic configuration updates without system restart.
"""

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import yaml
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = structlog.get_logger(__name__)


class ConfigSnapshot:
    """Immutable configuration snapshot for rollback capability."""

    def __init__(self, config: dict[str, Any], timestamp: datetime, checksum: str):
        self.config = config.copy()
        self.timestamp = timestamp
        self.checksum = checksum
        self.applied = False
        self.rollback_reason: str | None = None

    def __repr__(self) -> str:
        return f"ConfigSnapshot(timestamp={self.timestamp}, checksum={self.checksum[:8]}...)"


class ConfigValidator:
    """Validate configuration changes before applying."""

    @staticmethod
    def validate_trading_rules(config: dict[str, Any]) -> list[str]:
        """Validate trading rules configuration."""
        errors = []

        # Check tier limits are present
        if "tiers" not in config:
            errors.append("Missing 'tiers' section in trading rules")
            return errors

        # Validate each tier
        required_tier_fields = [
            "daily_loss_limit",
            "position_risk_percent",
            "max_positions",
            "stop_loss_percent",
        ]

        for tier_name, tier_config in config.get("tiers", {}).items():
            for field in required_tier_fields:
                if field not in tier_config:
                    errors.append(f"Missing '{field}' in tier {tier_name}")
                elif not isinstance(tier_config[field], (int, float)):
                    errors.append(f"Invalid type for '{field}' in tier {tier_name}")

        # Validate risk parameters
        if "global" in config:
            global_config = config["global"]
            if "minimum_position_size" in global_config:
                if global_config["minimum_position_size"] <= 0:
                    errors.append("Minimum position size must be positive")
            if "emergency_stop_loss" in global_config:
                if not 0 < global_config["emergency_stop_loss"] <= 100:
                    errors.append("Emergency stop loss must be between 0 and 100")

        return errors

    @staticmethod
    def validate_config(config_type: str, config: dict[str, Any]) -> list[str]:
        """Validate configuration based on type."""
        if config_type == "trading_rules":
            return ConfigValidator.validate_trading_rules(config)
        return []


class RuntimeConfigManager:
    """
    Manages runtime configuration with hot-reload capability.

    Features:
    - File system monitoring for configuration changes
    - Validation before applying changes
    - Rollback capability for failed configurations
    - Configuration history with snapshots
    - Subscriber notification for configuration updates
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, list[ConfigSnapshot]] = {}
        self.subscribers: dict[str, set[Callable]] = {}
        self.observer: Observer | None = None
        self.file_checksums: dict[str, str] = {}
        self.max_snapshots = 10  # Keep last 10 snapshots per config
        self.running = False

        # Configuration file mappings
        self.config_files = {
            "trading_rules.yaml": "trading_rules",
            "tier_gates.yaml": "tier_gates",
            "settings.py": "settings",
        }

    async def start(self) -> None:
        """Start configuration monitoring."""
        logger.info(
            "Starting runtime configuration manager", config_dir=str(self.config_dir)
        )

        # Load initial configurations
        await self.load_all_configs()

        # Start file system monitoring
        self.start_file_monitoring()

        self.running = True
        logger.info("Runtime configuration manager started")

    async def stop(self) -> None:
        """Stop configuration monitoring."""
        logger.info("Stopping runtime configuration manager")

        self.running = False

        if self.observer:
            self.observer.stop()
            self.observer.join()

        logger.info("Runtime configuration manager stopped")

    async def load_all_configs(self) -> None:
        """Load all configuration files."""
        for filename, config_type in self.config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                await self.load_config(file_path, config_type)

    async def load_config(self, file_path: Path, config_type: str) -> bool:
        """
        Load and validate a configuration file.

        Args:
            file_path: Path to configuration file
            config_type: Type of configuration

        Returns:
            True if configuration loaded successfully
        """
        try:
            # Read configuration file
            if file_path.suffix == ".yaml":
                with open(file_path) as f:
                    config = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                with open(file_path) as f:
                    config = json.load(f)
            else:
                # For Python files, we'll import them dynamically
                return False

            # Calculate checksum
            with open(file_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            # Check if configuration changed
            if config_type in self.file_checksums:
                if self.file_checksums[config_type] == checksum:
                    return True  # No changes

            # Validate configuration
            errors = ConfigValidator.validate_config(config_type, config)
            if errors:
                logger.error(
                    "Configuration validation failed",
                    config_type=config_type,
                    errors=errors,
                )
                return False

            # Create snapshot before applying
            snapshot = ConfigSnapshot(
                config=config, timestamp=datetime.now(), checksum=checksum
            )

            # Store snapshot
            if config_type not in self.snapshots:
                self.snapshots[config_type] = []
            self.snapshots[config_type].append(snapshot)

            # Trim old snapshots
            if len(self.snapshots[config_type]) > self.max_snapshots:
                self.snapshots[config_type] = self.snapshots[config_type][
                    -self.max_snapshots :
                ]

            # Apply configuration
            old_config = self.configs.get(config_type, {})
            self.configs[config_type] = config
            self.file_checksums[config_type] = checksum
            snapshot.applied = True

            # Notify subscribers
            await self.notify_subscribers(config_type, old_config, config)

            logger.info(
                "Configuration loaded successfully",
                config_type=config_type,
                checksum=checksum[:8],
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to load configuration", config_type=config_type, error=str(e)
            )
            return False

    def get_config(self, config_type: str, path: str | None = None) -> Any:
        """
        Get configuration value.

        Args:
            config_type: Type of configuration
            path: Dot-separated path to value (e.g., "tiers.SNIPER.stop_loss_percent")

        Returns:
            Configuration value or entire config if path not specified
        """
        if config_type not in self.configs:
            return None

        config = self.configs[config_type]

        if path:
            # Navigate to requested value
            parts = path.split(".")
            value = config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value

        return config

    def update_config(self, config_type: str, path: str, value: Any) -> bool:
        """
        Update configuration value at runtime.

        Args:
            config_type: Type of configuration
            path: Dot-separated path to value
            value: New value

        Returns:
            True if update successful
        """
        if config_type not in self.configs:
            return False

        # Create snapshot before modification
        snapshot = ConfigSnapshot(
            config=self.configs[config_type].copy(),
            timestamp=datetime.now(),
            checksum=hashlib.sha256(
                json.dumps(self.configs[config_type], sort_keys=True).encode()
            ).hexdigest(),
        )

        try:
            # Navigate to parent of target value
            parts = path.split(".")
            config = self.configs[config_type]

            for i, part in enumerate(parts[:-1]):
                if part not in config:
                    config[part] = {}
                config = config[part]

            # Store old value for rollback
            old_value = config.get(parts[-1])

            # Update value
            config[parts[-1]] = value

            # Validate new configuration
            errors = ConfigValidator.validate_config(
                config_type, self.configs[config_type]
            )
            if errors:
                # Rollback
                config[parts[-1]] = old_value
                logger.error(
                    "Configuration update failed validation",
                    config_type=config_type,
                    path=path,
                    errors=errors,
                )
                return False

            # Store snapshot
            if config_type not in self.snapshots:
                self.snapshots[config_type] = []
            self.snapshots[config_type].append(snapshot)

            # Notify subscribers
            asyncio.create_task(
                self.notify_subscribers(config_type, {path: old_value}, {path: value})
            )

            logger.info(
                "Configuration updated",
                config_type=config_type,
                path=path,
                old_value=old_value,
                new_value=value,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update configuration",
                config_type=config_type,
                path=path,
                error=str(e),
            )
            return False

    def rollback_config(self, config_type: str, reason: str) -> bool:
        """
        Rollback to previous configuration snapshot.

        Args:
            config_type: Type of configuration to rollback
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        if config_type not in self.snapshots or len(self.snapshots[config_type]) < 2:
            logger.warning(
                "No previous snapshot available for rollback", config_type=config_type
            )
            return False

        try:
            # Get previous snapshot (second to last)
            previous_snapshot = self.snapshots[config_type][-2]
            current_snapshot = self.snapshots[config_type][-1]

            # Mark current snapshot as rolled back
            current_snapshot.rollback_reason = reason
            current_snapshot.applied = False

            # Restore previous configuration
            self.configs[config_type] = previous_snapshot.config.copy()

            # Notify subscribers
            asyncio.create_task(
                self.notify_subscribers(
                    config_type, current_snapshot.config, previous_snapshot.config
                )
            )

            logger.info(
                "Configuration rolled back",
                config_type=config_type,
                reason=reason,
                restored_timestamp=previous_snapshot.timestamp,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to rollback configuration",
                config_type=config_type,
                error=str(e),
            )
            return False

    def subscribe(self, config_type: str, callback: Callable) -> None:
        """
        Subscribe to configuration changes.

        Args:
            config_type: Type of configuration to monitor
            callback: Async function to call on changes
        """
        if config_type not in self.subscribers:
            self.subscribers[config_type] = set()
        self.subscribers[config_type].add(callback)
        logger.debug(
            "Configuration subscriber added",
            config_type=config_type,
            callback=callback.__name__,
        )

    def unsubscribe(self, config_type: str, callback: Callable) -> None:
        """
        Unsubscribe from configuration changes.

        Args:
            config_type: Type of configuration
            callback: Callback to remove
        """
        if config_type in self.subscribers:
            self.subscribers[config_type].discard(callback)

    async def notify_subscribers(
        self, config_type: str, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        """Notify subscribers of configuration changes."""
        if config_type not in self.subscribers:
            return

        for callback in self.subscribers[config_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(config_type, old_config, new_config)
                else:
                    callback(config_type, old_config, new_config)
            except Exception as e:
                logger.error(
                    "Failed to notify configuration subscriber",
                    config_type=config_type,
                    callback=callback.__name__,
                    error=str(e),
                )

    def start_file_monitoring(self) -> None:
        """Start monitoring configuration files for changes."""

        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, manager: RuntimeConfigManager):
                self.manager = manager

            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent):
                    file_path = Path(event.src_path)
                    if file_path.name in self.manager.config_files:
                        config_type = self.manager.config_files[file_path.name]
                        logger.info(
                            "Configuration file modified",
                            file=file_path.name,
                            config_type=config_type,
                        )
                        # Schedule reload
                        asyncio.create_task(
                            self.manager.load_config(file_path, config_type)
                        )

        self.observer = Observer()
        handler = ConfigFileHandler(self)
        self.observer.schedule(handler, str(self.config_dir), recursive=False)
        self.observer.start()
        logger.info("File system monitoring started", directory=str(self.config_dir))

    def get_snapshot_history(self, config_type: str) -> list[ConfigSnapshot]:
        """Get configuration snapshot history."""
        return self.snapshots.get(config_type, [])

    def export_config(self, config_type: str, file_path: str) -> bool:
        """Export current configuration to file."""
        if config_type not in self.configs:
            return False

        try:
            path = Path(file_path)
            if path.suffix == ".yaml":
                with open(path, "w") as f:
                    yaml.dump(self.configs[config_type], f, default_flow_style=False)
            elif path.suffix == ".json":
                with open(path, "w") as f:
                    json.dump(self.configs[config_type], f, indent=2)
            else:
                return False

            logger.info(
                "Configuration exported", config_type=config_type, file_path=file_path
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to export configuration", config_type=config_type, error=str(e)
            )
            return False
