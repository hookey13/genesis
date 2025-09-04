"""Integration tests for hot-reload configuration functionality.

Tests the complete hot-reload flow including file watching, validation, and atomic updates.
"""

import asyncio
import shutil
import tempfile
import time
from pathlib import Path

import pytest
import yaml

from genesis.config.strategy_config import Environment, StrategyConfigManager


@pytest.fixture
async def hot_reload_manager():
    """Create a config manager with hot-reload enabled."""
    temp_dir = tempfile.mkdtemp(prefix="test_hot_reload_")
    config_path = Path(temp_dir)
    
    manager = StrategyConfigManager(
        config_path=str(config_path),
        environment=Environment.DEV,
        enable_hot_reload=True,
        poll_interval=0.1,  # Fast polling for tests
    )
    
    await manager.initialize()
    
    yield manager, config_path
    
    # Cleanup
    await manager.stop_watching()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestHotReloadIntegration:
    """Integration tests for hot-reload functionality."""

    async def test_hot_reload_detects_file_changes(self, hot_reload_manager):
        """Test that file changes are detected and reloaded."""
        manager, config_path = hot_reload_manager
        
        # Create initial config
        config_file = config_path / "test_strategy.yaml"
        initial_config = {
            "strategy": {
                "name": "TestStrategy",
                "version": "1.0.0",
                "tier": "sniper",
                "enabled": True,
            },
            "parameters": {
                "min_profit_pct": 0.3,
                "max_position_pct": 0.02,
            },
        }
        
        with open(config_file, "w") as f:
            yaml.dump(initial_config, f)
        
        # Load initial config
        await manager.load_all_configs()
        assert "TestStrategy" in manager.configs
        assert manager.configs["TestStrategy"]["parameters"]["min_profit_pct"] == 0.3
        
        # Start watching
        await manager.start_watching()
        
        # Wait for watcher to initialize
        await asyncio.sleep(0.2)
        
        # Modify config file
        initial_config["parameters"]["min_profit_pct"] = 0.5
        with open(config_file, "w") as f:
            yaml.dump(initial_config, f)
        
        # Wait for hot-reload to process change
        await asyncio.sleep(0.5)
        
        # Verify config was reloaded
        assert manager.configs["TestStrategy"]["parameters"]["min_profit_pct"] == 0.5

    async def test_hot_reload_validation_prevents_invalid_updates(self, hot_reload_manager):
        """Test that invalid configs are rejected during hot-reload."""
        manager, config_path = hot_reload_manager
        
        # Create valid initial config
        config_file = config_path / "test_strategy.yaml"
        valid_config = {
            "strategy": {
                "name": "TestStrategy",
                "version": "1.0.0",
                "tier": "sniper",
                "enabled": True,
            },
            "parameters": {
                "min_profit_pct": 0.3,
                "max_position_pct": 0.02,
                "min_order_size": 10.0,
                "max_order_size": 100.0,
            },
        }
        
        with open(config_file, "w") as f:
            yaml.dump(valid_config, f)
        
        # Load initial config
        await manager.load_all_configs()
        initial_min_profit = manager.configs["TestStrategy"]["parameters"]["min_profit_pct"]
        
        # Start watching
        await manager.start_watching()
        await asyncio.sleep(0.2)
        
        # Write invalid config (min > max order size)
        invalid_config = valid_config.copy()
        invalid_config["parameters"]["min_order_size"] = 200.0
        invalid_config["parameters"]["max_order_size"] = 100.0
        
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)
        
        # Wait for hot-reload attempt
        await asyncio.sleep(0.5)
        
        # Verify config was NOT updated (kept valid version)
        assert manager.configs["TestStrategy"]["parameters"]["min_profit_pct"] == initial_min_profit
        assert manager.configs["TestStrategy"]["parameters"]["min_order_size"] == 10.0

    async def test_hot_reload_concurrent_updates(self, hot_reload_manager):
        """Test that concurrent config updates are handled safely."""
        manager, config_path = hot_reload_manager
        
        # Create multiple config files
        configs = {}
        for i in range(3):
            config_file = config_path / f"strategy_{i}.yaml"
            config = {
                "strategy": {
                    "name": f"Strategy{i}",
                    "version": "1.0.0",
                    "tier": "sniper",
                    "enabled": True,
                },
                "parameters": {
                    "min_profit_pct": 0.1 * (i + 1),
                },
            }
            configs[f"Strategy{i}"] = config
            with open(config_file, "w") as f:
                yaml.dump(config, f)
        
        # Load initial configs
        await manager.load_all_configs()
        
        # Start watching
        await manager.start_watching()
        await asyncio.sleep(0.2)
        
        # Update all configs concurrently
        async def update_config(name, path, value):
            config = configs[name].copy()
            config["parameters"]["min_profit_pct"] = value
            config_file = path / f"{name.lower().replace('strategy', 'strategy_')}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config, f)
        
        tasks = [
            update_config("Strategy0", config_path, 0.15),
            update_config("Strategy1", config_path, 0.25),
            update_config("Strategy2", config_path, 0.35),
        ]
        
        await asyncio.gather(*tasks)
        
        # Wait for all hot-reloads to process
        await asyncio.sleep(1.0)
        
        # Verify all configs were updated
        assert manager.configs["Strategy0"]["parameters"]["min_profit_pct"] == 0.15
        assert manager.configs["Strategy1"]["parameters"]["min_profit_pct"] == 0.25
        assert manager.configs["Strategy2"]["parameters"]["min_profit_pct"] == 0.35

    async def test_hot_reload_rollback_on_error(self, hot_reload_manager):
        """Test that configs can be rolled back if hot-reload fails."""
        manager, config_path = hot_reload_manager
        
        # Create initial config
        config_file = config_path / "test_strategy.yaml"
        initial_config = {
            "strategy": {
                "name": "TestStrategy",
                "version": "1.0.0",
                "tier": "sniper",
                "enabled": True,
            },
            "parameters": {
                "min_profit_pct": 0.3,
            },
        }
        
        with open(config_file, "w") as f:
            yaml.dump(initial_config, f)
        
        # Load and track version
        await manager.load_all_configs()
        initial_version_count = len(manager.version_history.get("TestStrategy", []))
        
        # Start watching
        await manager.start_watching()
        await asyncio.sleep(0.2)
        
        # Update to valid config first
        initial_config["parameters"]["min_profit_pct"] = 0.4
        with open(config_file, "w") as f:
            yaml.dump(initial_config, f)
        
        await asyncio.sleep(0.5)
        assert manager.configs["TestStrategy"]["parameters"]["min_profit_pct"] == 0.4
        
        # Write malformed YAML
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")
        
        await asyncio.sleep(0.5)
        
        # Config should remain at last valid state
        assert manager.configs["TestStrategy"]["parameters"]["min_profit_pct"] == 0.4
        
        # Can manually rollback to initial version
        await manager.rollback_config("TestStrategy", version_index=0)
        assert manager.configs["TestStrategy"]["parameters"]["min_profit_pct"] == 0.3

    async def test_hot_reload_performance(self, hot_reload_manager):
        """Test hot-reload performance meets requirements."""
        manager, config_path = hot_reload_manager
        
        # Create config file
        config_file = config_path / "perf_test.yaml"
        config = {
            "strategy": {
                "name": "PerfTest",
                "version": "1.0.0",
                "tier": "sniper",
                "enabled": True,
            },
            "parameters": {
                "min_profit_pct": 0.3,
            },
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Load initial config
        await manager.load_all_configs()
        
        # Start watching
        await manager.start_watching()
        await asyncio.sleep(0.2)
        
        # Measure reload time
        config["parameters"]["min_profit_pct"] = 0.5
        
        start_time = time.time()
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Poll until change detected
        max_wait = 1.0  # 1 second max
        while (
            manager.configs["PerfTest"]["parameters"]["min_profit_pct"] == 0.3
            and time.time() - start_time < max_wait
        ):
            await asyncio.sleep(0.01)
        
        reload_time = time.time() - start_time
        
        # Verify performance requirement (<500ms)
        assert reload_time < 0.5, f"Hot-reload took {reload_time}s, exceeding 500ms requirement"
        assert manager.configs["PerfTest"]["parameters"]["min_profit_pct"] == 0.5

    async def test_hot_reload_file_deletion_handling(self, hot_reload_manager):
        """Test that file deletion is handled gracefully."""
        manager, config_path = hot_reload_manager
        
        # Create two config files
        config_file1 = config_path / "strategy1.yaml"
        config_file2 = config_path / "strategy2.yaml"
        
        for i, config_file in enumerate([config_file1, config_file2], 1):
            config = {
                "strategy": {
                    "name": f"Strategy{i}",
                    "version": "1.0.0",
                    "tier": "sniper",
                    "enabled": True,
                },
                "parameters": {
                    "min_profit_pct": 0.3,
                },
            }
            with open(config_file, "w") as f:
                yaml.dump(config, f)
        
        # Load configs
        await manager.load_all_configs()
        assert len(manager.configs) == 2
        
        # Start watching
        await manager.start_watching()
        await asyncio.sleep(0.2)
        
        # Delete one config file
        config_file1.unlink()
        
        # Wait for change detection
        await asyncio.sleep(0.5)
        
        # Strategy1 should be removed, Strategy2 should remain
        assert "Strategy1" not in manager.configs
        assert "Strategy2" in manager.configs

    async def test_hot_reload_with_environment_overrides(self, hot_reload_manager):
        """Test hot-reload with environment-specific overrides."""
        manager, config_path = hot_reload_manager
        
        # Create config with overrides
        config_file = config_path / "env_test.yaml"
        config = {
            "strategy": {
                "name": "EnvTest",
                "version": "1.0.0",
                "tier": "sniper",
                "enabled": True,
            },
            "parameters": {
                "min_profit_pct": 0.3,
            },
            "overrides": {
                "dev": {
                    "parameters": {
                        "min_profit_pct": 0.1,
                    },
                },
            },
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Load config (manager is in DEV environment)
        await manager.load_all_configs()
        
        # Should have dev override applied
        assert manager.configs["EnvTest"]["parameters"]["min_profit_pct"] == 0.1
        
        # Start watching
        await manager.start_watching()
        await asyncio.sleep(0.2)
        
        # Update dev override
        config["overrides"]["dev"]["parameters"]["min_profit_pct"] = 0.15
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        await asyncio.sleep(0.5)
        
        # Verify override was reloaded
        assert manager.configs["EnvTest"]["parameters"]["min_profit_pct"] == 0.15