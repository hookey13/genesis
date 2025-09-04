"""Unit tests for strategy configuration management.

Tests configuration loading, validation, hot-reload, versioning, and A/B testing.
"""

import asyncio
import json
import shutil
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from genesis.config.config_validator import (
    ConfigValidator,
    FieldConstraint,
    SchemaDefinition,
)
from genesis.config.file_watcher import ConfigFileWatcher, FileWatcher
from genesis.config.strategy_config import (
    Environment,
    StrategyConfigManager,
)

# Test fixtures


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configs."""
    temp_dir = tempfile.mkdtemp(prefix="test_config_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample valid configuration."""
    return {
        "strategy": {
            "name": "TestStrategy",
            "version": "1.0.0",
            "tier": "sniper",
            "enabled": True,
        },
        "parameters": {
            "min_profit_pct": 0.3,
            "max_position_pct": 0.02,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 0.5,
            "min_order_size": 10.0,
            "max_order_size": 100.0,
        },
        "risk_limits": {
            "max_positions": 1,
            "max_daily_loss_pct": 5.0,
            "max_correlation": 0.7,
        },
        "execution": {
            "order_type": "market",
            "time_in_force": "IOC",
            "retry_attempts": 3,
            "retry_delay_ms": 100,
        },
        "monitoring": {
            "log_level": "INFO",
            "metrics_interval_seconds": 60,
            "alert_on_loss": True,
        },
    }


@pytest.fixture
def sample_config_with_overrides(sample_config):
    """Sample configuration with environment overrides."""
    config = sample_config.copy()
    config["overrides"] = {
        "dev": {
            "parameters": {"min_profit_pct": 0.1},
            "monitoring": {"log_level": "DEBUG"},
        },
        "prod": {"parameters": {"min_profit_pct": 0.5}},
    }
    return config


@pytest.fixture
def sample_config_with_variants(sample_config):
    """Sample configuration with A/B testing variants."""
    config = sample_config.copy()
    config["variants"] = [
        {
            "name": "conservative",
            "parameters": {"min_profit_pct": 0.5, "stop_loss_pct": 0.5},
        },
        {
            "name": "aggressive",
            "parameters": {"min_profit_pct": 0.2, "max_position_pct": 0.03},
        },
    ]
    return config


@pytest.fixture
async def config_manager(temp_config_dir):
    """Create a config manager instance."""
    manager = StrategyConfigManager(
        config_path=str(temp_config_dir),
        environment=Environment.DEV,
        enable_hot_reload=True,
        enable_ab_testing=True,
    )
    await manager.initialize()
    return manager


# StrategyConfigManager Tests


class TestStrategyConfigManager:
    """Test StrategyConfigManager functionality."""

    async def test_initialization(self, temp_config_dir):
        """Test config manager initialization."""
        manager = StrategyConfigManager(
            config_path=str(temp_config_dir), environment=Environment.PROD
        )

        assert manager.config_path == temp_config_dir
        assert manager.environment == Environment.PROD
        assert manager.enable_hot_reload is True
        assert manager.enable_ab_testing is False
        assert len(manager.configs) == 0
        assert len(manager.version_history) == 0

    async def test_load_yaml_config(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test loading YAML configuration file."""
        # Write config file
        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Load config
        result = await config_manager.load_config_file(config_file)

        assert result is not None
        assert "TestStrategy" in config_manager.configs
        config = config_manager.configs["TestStrategy"]
        assert config.strategy["name"] == "TestStrategy"
        assert config.parameters.min_profit_pct == Decimal("0.3")

    async def test_load_json_config(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test loading JSON configuration file."""
        # Write config file
        config_file = temp_config_dir / "test_strategy.json"
        with open(config_file, "w") as f:
            json.dump(sample_config, f)

        # Load config
        result = await config_manager.load_config_file(config_file)

        assert result is not None
        assert "TestStrategy" in config_manager.configs
        config = config_manager.configs["TestStrategy"]
        assert config.strategy["name"] == "TestStrategy"

    async def test_environment_overrides(
        self, temp_config_dir, sample_config_with_overrides
    ):
        """Test environment-specific overrides."""
        # Test DEV environment
        manager_dev = StrategyConfigManager(
            config_path=str(temp_config_dir), environment=Environment.DEV
        )

        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_with_overrides, f)

        await manager_dev.load_config_file(config_file)
        config = manager_dev.configs["TestStrategy"]

        # Check DEV overrides applied
        assert config.parameters.min_profit_pct == Decimal("0.1")
        assert config.monitoring.log_level == "DEBUG"

        # Test PROD environment
        manager_prod = StrategyConfigManager(
            config_path=str(temp_config_dir), environment=Environment.PROD
        )

        await manager_prod.load_config_file(config_file)
        config = manager_prod.configs["TestStrategy"]

        # Check PROD overrides applied
        assert config.parameters.min_profit_pct == Decimal("0.5")
        assert config.monitoring.log_level == "INFO"  # No override, uses default

    async def test_config_validation_error(self, config_manager, temp_config_dir):
        """Test handling of invalid configuration."""
        # Invalid config (missing required fields)
        invalid_config = {
            "strategy": {
                "name": "InvalidStrategy"
                # Missing version, tier, etc.
            }
        }

        config_file = temp_config_dir / "invalid_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        # Load should fail
        result = await config_manager.load_config_file(config_file)
        assert result is None
        assert "InvalidStrategy" not in config_manager.configs

    async def test_version_history(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test configuration version history."""
        config_file = temp_config_dir / "test_strategy.yaml"

        # Load initial version
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        # Modify and reload
        sample_config["parameters"]["min_profit_pct"] = 0.5
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        # Check version history
        history = config_manager.get_version_history("TestStrategy")
        assert len(history) == 2

        # Check versions have different checksums
        assert history[0].checksum != history[1].checksum

        # Check changes recorded
        assert history[1].changes is not None

    async def test_config_rollback(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test configuration rollback."""
        config_file = temp_config_dir / "test_strategy.yaml"

        # Load initial version
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        initial_profit_pct = config_manager.configs[
            "TestStrategy"
        ].parameters.min_profit_pct

        # Modify and reload
        sample_config["parameters"]["min_profit_pct"] = 0.9
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        assert config_manager.configs[
            "TestStrategy"
        ].parameters.min_profit_pct == Decimal("0.9")

        # Rollback
        success = await config_manager.rollback_config("TestStrategy")
        assert success is True

        # Check rollback successful
        assert (
            config_manager.configs["TestStrategy"].parameters.min_profit_pct
            == initial_profit_pct
        )

        # Check rollback in version history
        history = config_manager.get_version_history("TestStrategy")
        assert history[-1].source == "rollback"

    async def test_ab_testing_variants(
        self, temp_config_dir, sample_config_with_variants
    ):
        """Test A/B testing variant loading and selection."""
        manager = StrategyConfigManager(
            config_path=str(temp_config_dir),
            environment=Environment.PROD,
            enable_ab_testing=True,
        )

        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_with_variants, f)

        await manager.load_config_file(config_file)

        # Check variants loaded
        assert "TestStrategy" in manager.ab_variants
        assert len(manager.ab_variants["TestStrategy"]) == 2

        # Get config multiple times - should get same variant
        config1 = manager.get_config("TestStrategy")
        config2 = manager.get_config("TestStrategy")

        assert config1 is config2  # Same variant selected
        assert "TestStrategy" in manager.active_variants

    async def test_audit_logging(self, config_manager, temp_config_dir, sample_config):
        """Test audit logging of configuration changes."""
        config_file = temp_config_dir / "test_strategy.yaml"

        # Load initial version
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        # Modify and reload
        sample_config["parameters"]["min_profit_pct"] = 0.7
        sample_config["parameters"]["max_position_pct"] = 0.05
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        # Check audit log
        audit_log = config_manager.get_audit_log(strategy_name="TestStrategy")
        assert len(audit_log) > 0

        # Find min_profit_pct change
        profit_changes = [
            log for log in audit_log if "min_profit_pct" in log.field_path
        ]
        assert len(profit_changes) == 1
        assert profit_changes[0].old_value == 0.3
        assert profit_changes[0].new_value == 0.7
        assert profit_changes[0].source == "file_change"

    async def test_change_callbacks(
        self, config_manager, temp_config_dir, sample_config
    ):
        """Test configuration change callbacks."""
        callback_called = False
        callback_strategy = None
        callback_config = None

        def test_callback(strategy_name, config):
            nonlocal callback_called, callback_strategy, callback_config
            callback_called = True
            callback_strategy = strategy_name
            callback_config = config

        config_manager.register_change_callback(test_callback)

        # Load config
        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        # Check callback was called
        assert callback_called is True
        assert callback_strategy == "TestStrategy"
        assert callback_config is not None

    async def test_export_config(self, config_manager, temp_config_dir, sample_config):
        """Test configuration export."""
        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        # Export as YAML
        yaml_export = await config_manager.export_config("TestStrategy", format="yaml")
        assert "TestStrategy" in yaml_export
        assert "min_profit_pct" in yaml_export

        # Export as JSON
        json_export = await config_manager.export_config("TestStrategy", format="json")
        exported_dict = json.loads(json_export)
        assert exported_dict["strategy"]["name"] == "TestStrategy"

    async def test_config_stats(self, config_manager, temp_config_dir, sample_config):
        """Test configuration statistics."""
        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)
        await config_manager.load_config_file(config_file)

        stats = config_manager.get_config_stats()

        assert stats["total_strategies"] == 1
        assert stats["environment"] == "dev"
        assert stats["hot_reload_enabled"] is True
        assert stats["ab_testing_enabled"] is True
        assert stats["total_versions"] >= 1


# ConfigValidator Tests


class TestConfigValidator:
    """Test ConfigValidator functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ConfigValidator()

        assert validator.tier_constraints is not None
        assert "sniper" in validator.tier_constraints
        assert "hunter" in validator.tier_constraints
        assert "strategist" in validator.tier_constraints

    def test_validate_valid_config(self, sample_config):
        """Test validation of valid configuration."""
        validator = ConfigValidator()
        result = validator.validate_config(sample_config)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.validated_config is not None

    def test_validate_missing_section(self):
        """Test validation with missing required section."""
        validator = ConfigValidator()
        config = {
            "strategy": {"name": "TestStrategy", "version": "1.0.0", "tier": "sniper"}
            # Missing other required sections
        }

        result = validator.validate_config(config)
        assert result.is_valid is False
        assert "Missing required section" in str(result.errors)

    def test_validate_invalid_field_type(self, sample_config):
        """Test validation with invalid field type."""
        validator = ConfigValidator()
        sample_config["parameters"]["min_profit_pct"] = "not_a_number"

        result = validator.validate_config(sample_config)
        assert result.is_valid is False

    def test_validate_tier_constraints(self, sample_config):
        """Test tier-specific constraint validation."""
        validator = ConfigValidator()

        # Exceed sniper tier limits
        sample_config["risk_limits"]["max_positions"] = 10  # Sniper max is 1
        sample_config["parameters"]["max_order_size"] = 1000  # Sniper max is 100

        result = validator.validate_config(sample_config, tier="sniper")
        assert result.is_valid is False
        assert any("exceeds tier limit" in error for error in result.errors)

    def test_validate_cross_field_constraints(self, sample_config):
        """Test cross-field constraint validation."""
        validator = ConfigValidator()

        # Min > Max order size
        sample_config["parameters"]["min_order_size"] = 200
        sample_config["parameters"]["max_order_size"] = 100

        result = validator.validate_config(sample_config)
        assert result.is_valid is False
        assert any(
            "Min order size" in error and "max order size" in error
            for error in result.errors
        )

    def test_validate_partial_config(self):
        """Test partial configuration validation."""
        validator = ConfigValidator()

        partial_config = {
            "min_profit_pct": 0.3,
            "max_position_pct": 0.02,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 0.5,
            "min_order_size": 10.0,
            "max_order_size": 100.0,
        }

        result = validator.validate_partial_config(partial_config, "parameters")
        assert result.is_valid is True
        assert result.validated_config is not None

    def test_strict_mode_validation(self, sample_config):
        """Test strict mode validation."""
        validator = ConfigValidator()

        # Add unknown field (would be warning normally)
        sample_config["parameters"]["unknown_field"] = 123

        # Normal mode - should pass with warning
        result = validator.validate_config(sample_config, strict_mode=False)
        assert result.is_valid is True
        assert len(result.warnings) > 0

        # Strict mode - warnings become errors
        result = validator.validate_config(sample_config, strict_mode=True)
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_custom_schema(self):
        """Test adding and using custom schema."""
        validator = ConfigValidator()

        # Define custom schema
        custom_schema = SchemaDefinition(
            fields={
                "custom_field": FieldConstraint(
                    field_type="int", min_value=0, max_value=100, required=True
                )
            }
        )

        validator.add_custom_schema("custom_section", custom_schema)

        # Validate with custom schema
        config = {"custom_field": 50}
        result = validator.validate_partial_config(config, "custom_section")
        assert result.is_valid is True

        # Invalid value
        config = {"custom_field": 200}
        result = validator.validate_partial_config(config, "custom_section")
        assert result.is_valid is False

    def test_get_schema_info(self):
        """Test getting schema information."""
        validator = ConfigValidator()

        # Get specific schema info
        info = validator.get_schema_info("parameters")
        assert "fields" in info
        assert "min_profit_pct" in info["fields"]

        # Get all schemas
        all_info = validator.get_schema_info()
        assert "available_schemas" in all_info
        assert "parameters" in all_info["available_schemas"]
        assert "tier_constraints" in all_info


# FileWatcher Tests


class TestFileWatcher:
    """Test FileWatcher functionality."""

    async def test_file_watcher_initialization(self, temp_config_dir):
        """Test file watcher initialization."""
        watcher = FileWatcher(
            watch_paths=[temp_config_dir], poll_interval=0.5, debounce_seconds=0.1
        )

        assert watcher.poll_interval == 0.5
        assert watcher.debounce_seconds == 0.1
        assert not watcher.is_running()

    async def test_file_watcher_start_stop(self, temp_config_dir):
        """Test starting and stopping file watcher."""
        watcher = FileWatcher(watch_paths=[temp_config_dir])

        # Start watcher
        await watcher.start()
        assert watcher.is_running()

        # Stop watcher
        await watcher.stop()
        assert not watcher.is_running()

    async def test_file_change_detection(self, temp_config_dir):
        """Test file change detection."""
        watcher = FileWatcher(
            watch_paths=[temp_config_dir], poll_interval=0.1, debounce_seconds=0.05
        )

        changes_detected = []

        async def change_callback(files):
            changes_detected.extend(files)

        watcher.register_change_callback(change_callback)

        # Start watcher
        await watcher.start()

        # Create a file
        test_file = temp_config_dir / "test.yaml"
        with open(test_file, "w") as f:
            f.write("test: 123")

        # Wait for detection
        await asyncio.sleep(0.3)

        # Check change detected
        assert len(changes_detected) > 0
        assert str(test_file) in changes_detected

        # Stop watcher
        await watcher.stop()

    async def test_file_modification_detection(self, temp_config_dir):
        """Test file modification detection."""
        # Create initial file
        test_file = temp_config_dir / "test.yaml"
        with open(test_file, "w") as f:
            f.write("initial: content")

        watcher = FileWatcher(
            watch_paths=[temp_config_dir], poll_interval=0.1, debounce_seconds=0.05
        )

        changes_detected = []

        async def change_callback(files):
            changes_detected.extend(files)

        watcher.register_change_callback(change_callback)

        # Start watcher
        await watcher.start()
        await asyncio.sleep(0.2)  # Let initial scan complete

        # Modify file
        with open(test_file, "w") as f:
            f.write("modified: content")

        # Wait for detection
        await asyncio.sleep(0.3)

        # Check modification detected
        assert len(changes_detected) > 0
        assert str(test_file) in changes_detected

        # Stop watcher
        await watcher.stop()

    async def test_file_deletion_detection(self, temp_config_dir):
        """Test file deletion detection."""
        # Create initial file
        test_file = temp_config_dir / "test.yaml"
        with open(test_file, "w") as f:
            f.write("test: content")

        watcher = FileWatcher(
            watch_paths=[temp_config_dir], poll_interval=0.1, debounce_seconds=0.05
        )

        changes_detected = []

        async def change_callback(files):
            changes_detected.extend(files)

        watcher.register_change_callback(change_callback)

        # Start watcher
        await watcher.start()
        await asyncio.sleep(0.2)  # Let initial scan complete

        # Delete file
        test_file.unlink()

        # Wait for detection
        await asyncio.sleep(0.3)

        # Check deletion detected
        assert len(changes_detected) > 0

        # Stop watcher
        await watcher.stop()

    async def test_config_file_watcher_integration(
        self, temp_config_dir, sample_config
    ):
        """Test ConfigFileWatcher with StrategyConfigManager."""
        # Create config manager
        manager = StrategyConfigManager(
            config_path=str(temp_config_dir), environment=Environment.DEV
        )

        # Create config file watcher
        watcher = ConfigFileWatcher(
            config_manager=manager,
            config_path=temp_config_dir,
            poll_interval=0.1,
            debounce_seconds=0.05,
        )

        # Start watcher
        await watcher.start()

        # Create config file
        config_file = temp_config_dir / "test_strategy.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Wait for auto-reload
        await asyncio.sleep(0.3)

        # Check config loaded
        assert "TestStrategy" in manager.configs

        # Modify config
        sample_config["parameters"]["min_profit_pct"] = 0.8
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Wait for reload
        await asyncio.sleep(0.3)

        # Check config updated
        assert manager.configs["TestStrategy"].parameters.min_profit_pct == Decimal(
            "0.8"
        )

        # Stop watcher
        await watcher.stop()

    def test_get_stats(self, temp_config_dir):
        """Test getting watcher statistics."""
        watcher = FileWatcher(watch_paths=[temp_config_dir])

        stats = watcher.get_stats()
        assert stats["is_running"] is False
        assert stats["poll_interval"] == 1.0
        assert stats["debounce_seconds"] == 0.5
        assert len(stats["watched_paths"]) == 1


# Integration Tests


class TestIntegration:
    """Integration tests for complete configuration system."""

    async def test_complete_workflow(self, temp_config_dir):
        """Test complete configuration management workflow."""
        # Create manager with all features enabled
        manager = StrategyConfigManager(
            config_path=str(temp_config_dir),
            environment=Environment.DEV,
            enable_hot_reload=True,
            enable_ab_testing=True,
        )

        # Create file watcher
        watcher = ConfigFileWatcher(
            config_manager=manager,
            config_path=temp_config_dir,
            poll_interval=0.1,
            debounce_seconds=0.05,
        )

        # Initialize and start
        await manager.initialize()
        await watcher.start()

        # Create initial config
        config = {
            "strategy": {
                "name": "IntegrationTest",
                "version": "1.0.0",
                "tier": "hunter",
                "enabled": True,
            },
            "parameters": {
                "min_profit_pct": 0.5,
                "max_position_pct": 0.05,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 1.0,
                "min_order_size": 50.0,
                "max_order_size": 500.0,
            },
            "risk_limits": {
                "max_positions": 5,
                "max_daily_loss_pct": 10.0,
                "max_correlation": 0.5,
            },
            "execution": {
                "order_type": "limit",
                "time_in_force": "GTC",
                "retry_attempts": 5,
                "retry_delay_ms": 200,
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_interval_seconds": 30,
                "alert_on_loss": True,
            },
            "overrides": {"dev": {"parameters": {"min_order_size": 10.0}}},
        }

        config_file = temp_config_dir / "integration_test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Wait for auto-load
        await asyncio.sleep(0.3)

        # Verify loaded with overrides
        assert "IntegrationTest" in manager.configs
        loaded_config = manager.configs["IntegrationTest"]
        assert loaded_config.parameters.min_order_size == Decimal(
            "10.0"
        )  # Dev override

        # Modify config
        config["parameters"]["stop_loss_pct"] = 3.0
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Wait for hot-reload
        await asyncio.sleep(0.3)

        # Verify updated
        assert manager.configs["IntegrationTest"].parameters.stop_loss_pct == Decimal(
            "3.0"
        )

        # Check version history
        history = manager.get_version_history("IntegrationTest")
        assert len(history) >= 2

        # Rollback
        await manager.rollback_config("IntegrationTest")
        assert manager.configs["IntegrationTest"].parameters.stop_loss_pct == Decimal(
            "2.0"
        )

        # Check audit log
        audit_log = manager.get_audit_log(strategy_name="IntegrationTest")
        assert len(audit_log) > 0

        # Get stats
        stats = manager.get_config_stats()
        assert stats["total_strategies"] == 1
        assert stats["total_versions"] >= 3

        # Stop watcher
        await watcher.stop()

    async def test_performance_with_multiple_configs(self, temp_config_dir):
        """Test performance with multiple configuration files."""
        manager = StrategyConfigManager(
            config_path=str(temp_config_dir), environment=Environment.PROD
        )

        # Create multiple config files
        base_config = {
            "strategy": {"version": "1.0.0", "tier": "sniper", "enabled": True},
            "parameters": {
                "min_profit_pct": 0.3,
                "max_position_pct": 0.02,
                "stop_loss_pct": 1.0,
                "take_profit_pct": 0.5,
                "min_order_size": 10.0,
                "max_order_size": 100.0,
            },
            "risk_limits": {
                "max_positions": 1,
                "max_daily_loss_pct": 5.0,
                "max_correlation": 0.7,
            },
            "execution": {
                "order_type": "market",
                "time_in_force": "IOC",
                "retry_attempts": 3,
                "retry_delay_ms": 100,
            },
            "monitoring": {
                "log_level": "INFO",
                "metrics_interval_seconds": 60,
                "alert_on_loss": True,
            },
        }

        # Create 10 config files
        for i in range(10):
            config = base_config.copy()
            config["strategy"]["name"] = f"Strategy{i}"

            config_file = temp_config_dir / f"strategy_{i}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f)

        # Load all configs
        start_time = datetime.now(UTC)
        await manager.load_all_configs()
        load_time = (datetime.now(UTC) - start_time).total_seconds()

        # Check all loaded
        assert len(manager.configs) == 10

        # Performance check - should be fast
        assert load_time < 1.0  # Should load 10 configs in under 1 second

        # Test retrieval performance
        start_time = datetime.now(UTC)
        for i in range(100):
            config = manager.get_config(f"Strategy{i % 10}")
            assert config is not None
        retrieval_time = (datetime.now(UTC) - start_time).total_seconds()

        assert retrieval_time < 0.1  # 100 retrievals in under 100ms
