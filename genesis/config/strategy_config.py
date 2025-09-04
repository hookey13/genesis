"""Strategy Configuration Management Module.

This module handles loading, validation, hot-reload, and versioning of strategy
configurations from YAML/JSON files. Supports A/B testing, environment overrides,
and comprehensive audit logging.
"""

import asyncio
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field, ValidationError, validator

logger = structlog.get_logger(__name__)


class Environment(str, Enum):
    """Environment types for configuration overrides."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class ConfigVersion(BaseModel):
    """Represents a version of a configuration."""

    version_id: str
    timestamp: datetime
    config_data: dict[str, Any]
    checksum: str
    source: str  # file_change, api, manual, rollback
    user: str | None = None
    changes: dict[str, Any] | None = None  # diff from previous version


class StrategyParameters(BaseModel):
    """Strategy parameter validation model."""

    min_profit_pct: Decimal = Field(ge=0, le=100)
    max_position_pct: Decimal = Field(gt=0, le=100)
    stop_loss_pct: Decimal = Field(gt=0, le=100)
    take_profit_pct: Decimal = Field(gt=0, le=100)
    min_order_size: Decimal = Field(gt=0)
    max_order_size: Decimal = Field(gt=0)

    @validator("max_order_size")
    def validate_order_sizes(cls, v, values):
        if "min_order_size" in values and v < values["min_order_size"]:
            raise ValueError("max_order_size must be >= min_order_size")
        return v


class RiskLimits(BaseModel):
    """Risk limit configuration."""

    max_positions: int = Field(ge=1)
    max_daily_loss_pct: Decimal = Field(gt=0, le=100)
    max_correlation: Decimal = Field(ge=0, le=1)


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    order_type: str = Field(default="market")
    time_in_force: str = Field(default="IOC")
    retry_attempts: int = Field(ge=0, default=3)
    retry_delay_ms: int = Field(ge=0, default=100)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    log_level: str = Field(default="INFO")
    metrics_interval_seconds: int = Field(gt=0, default=60)
    alert_on_loss: bool = Field(default=True)


class StrategyConfig(BaseModel):
    """Complete strategy configuration model."""

    strategy: dict[str, Any]
    parameters: StrategyParameters
    risk_limits: RiskLimits
    execution: ExecutionConfig
    monitoring: MonitoringConfig
    overrides: dict[str, dict[str, Any]] | None = None


@dataclass
class ConfigChange:
    """Represents a configuration change for audit logging."""

    timestamp: datetime
    strategy_name: str
    field_path: str
    old_value: Any
    new_value: Any
    source: str
    user: str | None = None


class StrategyConfigManager:
    """Manages strategy configurations with hot-reload and versioning."""

    MAX_VERSION_HISTORY = 100

    def __init__(
        self,
        config_path: str,
        environment: Environment = Environment.PROD,
        enable_hot_reload: bool = True,
        enable_ab_testing: bool = False,
    ):
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration directory
            environment: Current environment (dev/staging/prod)
            enable_hot_reload: Enable automatic configuration reload
            enable_ab_testing: Enable A/B testing support
        """
        self.config_path = Path(config_path)
        self.environment = environment
        self.enable_hot_reload = enable_hot_reload
        self.enable_ab_testing = enable_ab_testing

        # Configuration storage
        self.configs: dict[str, StrategyConfig] = {}
        self.raw_configs: dict[str, dict[str, Any]] = {}

        # Version history (per strategy)
        self.version_history: dict[str, deque] = {}

        # A/B testing variants
        self.ab_variants: dict[str, list[StrategyConfig]] = {}
        self.active_variants: dict[str, str] = {}  # strategy -> active variant ID

        # File modification tracking
        self.file_mtimes: dict[str, float] = {}

        # Audit log
        self.audit_log: list[ConfigChange] = []

        # Callbacks for configuration changes
        self.change_callbacks: list = []

        # Lock for thread-safe updates
        self._lock = asyncio.Lock()

        logger.info(
            "StrategyConfigManager initialized",
            config_path=str(config_path),
            environment=environment.value,
            hot_reload_enabled=enable_hot_reload,
            ab_testing_enabled=enable_ab_testing,
        )

    async def initialize(self) -> None:
        """Initialize the configuration manager and load configs."""
        await self.load_all_configs()

        if self.enable_hot_reload:
            # File watcher will be initialized separately
            logger.info("Hot-reload enabled, file watcher will monitor changes")

    async def load_all_configs(self) -> None:
        """Load all strategy configuration files."""
        if not self.config_path.exists():
            logger.warning(f"Config path does not exist: {self.config_path}")
            return

        # Find all YAML and JSON files
        config_files = list(self.config_path.glob("*.yaml"))
        config_files.extend(self.config_path.glob("*.yml"))
        config_files.extend(self.config_path.glob("*.json"))

        logger.info(f"Found {len(config_files)} configuration files")

        for config_file in config_files:
            try:
                await self.load_config_file(config_file)
            except Exception as e:
                logger.error(
                    "Failed to load config file", file=str(config_file), error=str(e)
                )

    async def load_config_file(self, file_path: Path) -> StrategyConfig | None:
        """Load and validate a single configuration file.

        Args:
            file_path: Path to configuration file

        Returns:
            Loaded configuration or None if failed
        """
        async with self._lock:
            try:
                # Read file content
                with open(file_path) as f:
                    if file_path.suffix in [".yaml", ".yml"]:
                        raw_config = yaml.safe_load(f)
                    else:  # JSON
                        raw_config = json.load(f)

                # Apply environment overrides
                if (
                    raw_config.get("overrides")
                    and self.environment.value in raw_config["overrides"]
                ):
                    env_overrides = raw_config["overrides"][self.environment.value]
                    self._apply_overrides(raw_config, env_overrides)

                # Validate configuration
                config = StrategyConfig(**raw_config)

                # Extract strategy name
                strategy_name = config.strategy.get("name")
                if not strategy_name:
                    logger.error(f"No strategy name in config: {file_path}")
                    return None

                # Check for changes
                old_config = self.configs.get(strategy_name)
                if old_config:
                    changes = self._compute_changes(
                        self.raw_configs[strategy_name], raw_config
                    )
                    if changes:
                        await self._log_config_change(
                            strategy_name, changes, "file_change"
                        )

                # Store configuration
                self.configs[strategy_name] = config
                self.raw_configs[strategy_name] = raw_config

                # Update file modification time
                self.file_mtimes[str(file_path)] = file_path.stat().st_mtime

                # Add to version history
                await self._add_version(strategy_name, raw_config, "file_change")

                # Handle A/B testing variants
                if self.enable_ab_testing and "variants" in raw_config:
                    await self._load_ab_variants(strategy_name, raw_config["variants"])

                logger.info(
                    "Configuration loaded successfully",
                    strategy=strategy_name,
                    file=str(file_path),
                )

                # Notify callbacks
                await self._notify_callbacks(strategy_name, config)

                return config

            except ValidationError as e:
                logger.error(
                    "Configuration validation failed",
                    file=str(file_path),
                    errors=e.errors(),
                )
                return None
            except Exception as e:
                logger.error(
                    "Failed to load configuration", file=str(file_path), error=str(e)
                )
                return None

    def _apply_overrides(
        self, config: dict[str, Any], overrides: dict[str, Any]
    ) -> None:
        """Apply environment-specific overrides to configuration.

        Args:
            config: Base configuration dictionary
            overrides: Override values to apply
        """
        for section, values in overrides.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

    def _compute_changes(
        self, old_config: dict[str, Any], new_config: dict[str, Any], path: str = ""
    ) -> list[tuple[str, Any, Any]]:
        """Compute differences between two configurations.

        Args:
            old_config: Previous configuration
            new_config: New configuration
            path: Current path in configuration tree

        Returns:
            List of (field_path, old_value, new_value) tuples
        """
        changes = []

        # Check for modified and removed keys
        for key in old_config:
            current_path = f"{path}.{key}" if path else key

            if key not in new_config:
                changes.append((current_path, old_config[key], None))
            elif isinstance(old_config[key], dict) and isinstance(
                new_config[key], dict
            ):
                # Recursive comparison for nested dictionaries
                changes.extend(
                    self._compute_changes(
                        old_config[key], new_config[key], current_path
                    )
                )
            elif old_config[key] != new_config[key]:
                changes.append((current_path, old_config[key], new_config[key]))

        # Check for added keys
        for key in new_config:
            if key not in old_config:
                current_path = f"{path}.{key}" if path else key
                changes.append((current_path, None, new_config[key]))

        return changes

    async def _log_config_change(
        self,
        strategy_name: str,
        changes: list[tuple[str, Any, Any]],
        source: str,
        user: str | None = None,
    ) -> None:
        """Log configuration changes for audit.

        Args:
            strategy_name: Name of the strategy
            changes: List of changes
            source: Source of change
            user: User who made the change
        """
        timestamp = datetime.now(UTC)

        for field_path, old_value, new_value in changes:
            # Skip logging sensitive fields
            if any(
                sensitive in field_path.lower()
                for sensitive in ["key", "secret", "password", "token"]
            ):
                old_value = "***REDACTED***"
                new_value = "***REDACTED***"

            change = ConfigChange(
                timestamp=timestamp,
                strategy_name=strategy_name,
                field_path=field_path,
                old_value=old_value,
                new_value=new_value,
                source=source,
                user=user,
            )

            self.audit_log.append(change)

            logger.info(
                "Configuration changed",
                strategy=strategy_name,
                field=field_path,
                old_value=old_value,
                new_value=new_value,
                source=source,
                user=user,
            )

    async def _add_version(
        self,
        strategy_name: str,
        config_data: dict[str, Any],
        source: str,
        user: str | None = None,
    ) -> None:
        """Add configuration to version history.

        Args:
            strategy_name: Name of the strategy
            config_data: Configuration data
            source: Source of the configuration
            user: User who created the version
        """
        # Initialize history deque if needed
        if strategy_name not in self.version_history:
            self.version_history[strategy_name] = deque(maxlen=self.MAX_VERSION_HISTORY)

        # Compute checksum
        config_str = json.dumps(config_data, sort_keys=True)
        checksum = hashlib.sha256(config_str.encode()).hexdigest()

        # Compute changes from previous version
        changes = None
        if self.version_history[strategy_name]:
            prev_version = self.version_history[strategy_name][-1]
            changes = self._compute_changes(prev_version.config_data, config_data)

        # Create version
        version = ConfigVersion(
            version_id=f"{strategy_name}_{datetime.now(UTC).isoformat()}",
            timestamp=datetime.now(UTC),
            config_data=config_data,
            checksum=checksum,
            source=source,
            user=user,
            changes={str(k): (v1, v2) for k, v1, v2 in changes} if changes else None,
        )

        self.version_history[strategy_name].append(version)

        logger.info(
            "Configuration version saved",
            strategy=strategy_name,
            version_id=version.version_id,
            checksum=checksum[:8],
            source=source,
        )

    async def _load_ab_variants(
        self, strategy_name: str, variants: list[dict[str, Any]]
    ) -> None:
        """Load A/B testing variants for a strategy.

        Args:
            strategy_name: Name of the strategy
            variants: List of variant configurations
        """
        self.ab_variants[strategy_name] = []

        for i, variant_data in enumerate(variants):
            try:
                # Start with base config
                base_config = self.raw_configs[strategy_name].copy()
                
                # Extract variant name (not part of config)
                variant_name = variant_data.get("name", f"variant_{i}")
                
                # Merge variant overrides (deep merge for nested sections)
                for section in ["parameters", "risk_limits", "execution", "monitoring"]:
                    if section in variant_data:
                        if section not in base_config:
                            base_config[section] = {}
                        base_config[section].update(variant_data[section])

                # Validate variant
                variant_config = StrategyConfig(**base_config)
                self.ab_variants[strategy_name].append(variant_config)

                logger.info(
                    "A/B variant loaded",
                    strategy=strategy_name,
                    variant_index=i,
                    variant_name=variant_name,
                )

            except ValidationError as e:
                logger.error(
                    "Failed to load A/B variant",
                    strategy=strategy_name,
                    variant_index=i,
                    errors=e.errors(),
                )

    async def _notify_callbacks(
        self, strategy_name: str, config: StrategyConfig
    ) -> None:
        """Notify registered callbacks of configuration change.

        Args:
            strategy_name: Name of the strategy
            config: New configuration
        """
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(strategy_name, config)
                else:
                    callback(strategy_name, config)
            except Exception as e:
                logger.error(
                    "Callback execution failed",
                    callback=callback.__name__,
                    error=str(e),
                )

    def get_config(self, strategy_name: str) -> StrategyConfig | None:
        """Get configuration for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Strategy configuration or None if not found
        """
        # Handle A/B testing
        if self.enable_ab_testing and strategy_name in self.ab_variants:
            return self._get_ab_variant(strategy_name)

        return self.configs.get(strategy_name)

    def _get_ab_variant(self, strategy_name: str) -> StrategyConfig | None:
        """Get A/B testing variant for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Selected variant configuration
        """
        variants = self.ab_variants.get(strategy_name, [])
        if not variants:
            return self.configs.get(strategy_name)

        # Check if variant already selected
        if strategy_name in self.active_variants:
            variant_index = int(self.active_variants[strategy_name].split("_")[-1])
            if variant_index < len(variants):
                return variants[variant_index]

        # Select random variant
        variant_index = random.randint(0, len(variants) - 1)
        self.active_variants[strategy_name] = f"variant_{variant_index}"

        logger.info(
            "A/B variant selected",
            strategy=strategy_name,
            variant=self.active_variants[strategy_name],
        )

        return variants[variant_index]

    async def reload_config(self, strategy_name: str) -> bool:
        """Manually reload configuration for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            True if reload successful, False otherwise
        """
        # Find config file for strategy
        config_files = list(self.config_path.glob(f"*{strategy_name}*.yaml"))
        config_files.extend(self.config_path.glob(f"*{strategy_name}*.yml"))
        config_files.extend(self.config_path.glob(f"*{strategy_name}*.json"))

        if not config_files:
            logger.error(f"No config file found for strategy: {strategy_name}")
            return False

        # Reload first matching file
        result = await self.load_config_file(config_files[0])
        return result is not None

    async def rollback_config(
        self, strategy_name: str, version_id: str | None = None
    ) -> bool:
        """Rollback configuration to a previous version.

        Args:
            strategy_name: Name of the strategy
            version_id: Specific version to rollback to (or latest if None)

        Returns:
            True if rollback successful, False otherwise
        """
        if strategy_name not in self.version_history:
            logger.error(f"No version history for strategy: {strategy_name}")
            return False

        history = self.version_history[strategy_name]
        if len(history) < 2:
            logger.error(f"Insufficient version history for rollback: {strategy_name}")
            return False

        async with self._lock:
            try:
                # Find target version
                target_version = None
                if version_id:
                    for version in history:
                        if version.version_id == version_id:
                            target_version = version
                            break
                else:
                    # Rollback to previous version
                    target_version = history[-2]

                if not target_version:
                    logger.error(f"Version not found: {version_id}")
                    return False

                # Validate configuration
                config = StrategyConfig(**target_version.config_data)

                # Log rollback
                changes = self._compute_changes(
                    self.raw_configs[strategy_name], target_version.config_data
                )
                await self._log_config_change(strategy_name, changes, "rollback")

                # Apply rollback
                self.configs[strategy_name] = config
                self.raw_configs[strategy_name] = target_version.config_data

                # Add rollback as new version
                await self._add_version(
                    strategy_name, target_version.config_data, "rollback"
                )

                # Notify callbacks
                await self._notify_callbacks(strategy_name, config)

                logger.info(
                    "Configuration rolled back",
                    strategy=strategy_name,
                    version_id=target_version.version_id,
                )

                return True

            except Exception as e:
                logger.error("Rollback failed", strategy=strategy_name, error=str(e))
                return False

    def get_version_history(
        self, strategy_name: str, limit: int | None = None
    ) -> list[ConfigVersion]:
        """Get version history for a strategy.

        Args:
            strategy_name: Name of the strategy
            limit: Maximum number of versions to return

        Returns:
            List of configuration versions
        """
        if strategy_name not in self.version_history:
            return []

        history = list(self.version_history[strategy_name])
        if limit:
            history = history[-limit:]

        return history

    def get_audit_log(
        self,
        strategy_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[ConfigChange]:
        """Get audit log entries.

        Args:
            strategy_name: Filter by strategy name
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of entries to return

        Returns:
            List of audit log entries
        """
        logs = self.audit_log

        # Apply filters
        if strategy_name:
            logs = [l for l in logs if l.strategy_name == strategy_name]

        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]

        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]

        # Sort by timestamp (newest first)
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)

        if limit:
            logs = logs[:limit]

        return logs

    def register_change_callback(self, callback) -> None:
        """Register a callback for configuration changes.

        Args:
            callback: Function to call on configuration change
        """
        self.change_callbacks.append(callback)
        logger.info(f"Registered change callback: {callback.__name__}")

    def unregister_change_callback(self, callback) -> None:
        """Unregister a change callback.

        Args:
            callback: Callback to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            logger.info(f"Unregistered change callback: {callback.__name__}")

    async def check_file_changes(self) -> set[str]:
        """Check for modified configuration files.

        Returns:
            Set of modified file paths
        """
        modified_files = set()

        for file_path, old_mtime in self.file_mtimes.items():
            path = Path(file_path)
            if path.exists():
                current_mtime = path.stat().st_mtime
                if current_mtime > old_mtime:
                    modified_files.add(file_path)

        return modified_files

    async def export_config(self, strategy_name: str, format: str = "yaml") -> str:
        """Export configuration to string format.

        Args:
            strategy_name: Name of the strategy
            format: Export format (yaml or json)

        Returns:
            Configuration as string
        """
        if strategy_name not in self.raw_configs:
            raise ValueError(f"Strategy not found: {strategy_name}")

        config = self.raw_configs[strategy_name]

        if format == "json":
            return json.dumps(config, indent=2, default=str)
        else:  # yaml
            return yaml.dump(config, default_flow_style=False)

    def get_config_stats(self) -> dict[str, Any]:
        """Get configuration statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_strategies": len(self.configs),
            "environment": self.environment.value,
            "hot_reload_enabled": self.enable_hot_reload,
            "ab_testing_enabled": self.enable_ab_testing,
            "active_ab_tests": len(self.ab_variants),
            "total_versions": sum(len(h) for h in self.version_history.values()),
            "audit_log_entries": len(self.audit_log),
            "registered_callbacks": len(self.change_callbacks),
        }
