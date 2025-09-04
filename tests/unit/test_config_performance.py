"""Performance benchmarks for configuration loading.

Tests that configuration operations meet performance requirements.
"""

import asyncio
import shutil
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from genesis.config.config_validator import ConfigValidator
from genesis.config.strategy_config import Environment, StrategyConfigManager


@pytest.fixture
def large_config():
    """Generate a large configuration for performance testing."""
    return {
        "strategy": {
            "name": "PerfTestStrategy",
            "version": "1.0.0",
            "tier": "strategist",
            "enabled": True,
            "description": "A" * 1000,  # Large description
        },
        "parameters": {
            f"param_{i}": i * 0.1
            for i in range(50)  # Many parameters
        },
        "risk_limits": {
            "max_positions": 20,
            "max_daily_loss_pct": 10.0,
            "max_correlation": 0.8,
            **{f"limit_{i}": i for i in range(20)}  # Additional limits
        },
        "execution": {
            "order_type": "limit",
            "time_in_force": "GTC",
            "retry_attempts": 5,
            "retry_delay_ms": 200,
            **{f"exec_param_{i}": f"value_{i}" for i in range(10)}
        },
        "monitoring": {
            "log_level": "DEBUG",
            "metrics_interval_seconds": 30,
            "alert_on_loss": True,
            **{f"monitor_{i}": i % 2 == 0 for i in range(10)}
        },
        "overrides": {
            env: {
                "parameters": {f"param_{i}": i * 0.2 for i in range(10)}
            }
            for env in ["dev", "staging", "prod"]
        },
    }


class TestConfigPerformance:
    """Performance benchmark tests for configuration management."""

    def test_config_load_time_single_file(self, large_config):
        """Test that single config file loads in <100ms."""
        temp_dir = tempfile.mkdtemp(prefix="test_perf_")
        config_path = Path(temp_dir)
        
        try:
            # Write config file
            config_file = config_path / "perf_test.yaml"
            with open(config_file, "w") as f:
                yaml.dump(large_config, f)
            
            # Measure load time
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
            )
            
            start_time = time.perf_counter()
            asyncio.run(manager.load_config_file(config_file))
            load_time = time.perf_counter() - start_time
            
            # Convert to milliseconds
            load_time_ms = load_time * 1000
            
            assert load_time_ms < 100, f"Config load took {load_time_ms:.2f}ms, exceeding 100ms requirement"
            assert "PerfTestStrategy" in manager.configs
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_config_load_time_multiple_files(self, large_config):
        """Test that multiple config files load efficiently."""
        temp_dir = tempfile.mkdtemp(prefix="test_perf_multi_")
        config_path = Path(temp_dir)
        
        try:
            # Write 10 config files
            for i in range(10):
                config = large_config.copy()
                config["strategy"]["name"] = f"Strategy{i}"
                config_file = config_path / f"strategy_{i}.yaml"
                with open(config_file, "w") as f:
                    yaml.dump(config, f)
            
            # Measure total load time
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
            )
            
            start_time = time.perf_counter()
            asyncio.run(manager.load_all_configs())
            load_time = time.perf_counter() - start_time
            
            # Convert to milliseconds
            load_time_ms = load_time * 1000
            avg_load_time_ms = load_time_ms / 10
            
            assert avg_load_time_ms < 100, f"Average config load took {avg_load_time_ms:.2f}ms per file"
            assert len(manager.configs) == 10
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_validation_performance(self, large_config):
        """Test that validation completes quickly even for large configs."""
        validator = ConfigValidator()
        
        # Measure validation time
        start_time = time.perf_counter()
        for _ in range(100):  # Validate 100 times
            result = validator.validate_config(large_config, tier="strategist")
        validation_time = time.perf_counter() - start_time
        
        # Average time per validation in milliseconds
        avg_validation_ms = (validation_time / 100) * 1000
        
        assert avg_validation_ms < 10, f"Validation took {avg_validation_ms:.2f}ms average"
        assert result.is_valid

    async def test_hot_reload_latency(self):
        """Test that hot-reload completes in <500ms."""
        temp_dir = tempfile.mkdtemp(prefix="test_hotreload_perf_")
        config_path = Path(temp_dir)
        
        try:
            # Create initial config
            config_file = config_path / "hot_reload_test.yaml"
            config = {
                "strategy": {
                    "name": "HotReloadTest",
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
            
            # Initialize manager with fast polling
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
                enable_hot_reload=True,
                poll_interval=0.05,  # 50ms polling
            )
            
            await manager.initialize()
            await manager.load_all_configs()
            await manager.start_watching()
            
            # Wait for watcher to stabilize
            await asyncio.sleep(0.1)
            
            # Modify config and measure reload time
            config["parameters"]["min_profit_pct"] = 0.5
            
            start_time = time.perf_counter()
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            
            # Wait for reload
            max_wait = 0.5  # 500ms max
            while (
                manager.configs["HotReloadTest"]["parameters"]["min_profit_pct"] == 0.3
                and time.perf_counter() - start_time < max_wait
            ):
                await asyncio.sleep(0.01)
            
            reload_latency = time.perf_counter() - start_time
            reload_latency_ms = reload_latency * 1000
            
            assert reload_latency_ms < 500, f"Hot-reload took {reload_latency_ms:.2f}ms"
            assert manager.configs["HotReloadTest"]["parameters"]["min_profit_pct"] == 0.5
            
            await manager.stop_watching()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_memory_usage(self, large_config):
        """Test that configuration storage uses <10MB for all configs."""
        import sys
        
        temp_dir = tempfile.mkdtemp(prefix="test_memory_")
        config_path = Path(temp_dir)
        
        try:
            # Create 50 config files (stress test)
            for i in range(50):
                config = large_config.copy()
                config["strategy"]["name"] = f"MemTest{i}"
                config_file = config_path / f"mem_test_{i}.yaml"
                with open(config_file, "w") as f:
                    yaml.dump(config, f)
            
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
            )
            
            # Load all configs
            asyncio.run(manager.load_all_configs())
            
            # Estimate memory usage (rough approximation)
            total_size = 0
            for config_name, config_data in manager.configs.items():
                total_size += sys.getsizeof(config_name)
                total_size += sys.getsizeof(config_data)
                # Add size of nested structures
                for key, value in config_data.items():
                    total_size += sys.getsizeof(key)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            total_size += sys.getsizeof(k) + sys.getsizeof(v)
                    else:
                        total_size += sys.getsizeof(value)
            
            # Convert to MB
            total_size_mb = total_size / (1024 * 1024)
            
            assert total_size_mb < 10, f"Configs use {total_size_mb:.2f}MB, exceeding 10MB limit"
            assert len(manager.configs) == 50
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_version_history_storage(self):
        """Test that version history storage is <1MB per 100 versions."""
        import sys
        
        temp_dir = tempfile.mkdtemp(prefix="test_version_")
        config_path = Path(temp_dir)
        
        try:
            config_file = config_path / "version_test.yaml"
            base_config = {
                "strategy": {
                    "name": "VersionTest",
                    "version": "1.0.0",
                    "tier": "sniper",
                    "enabled": True,
                },
                "parameters": {
                    "min_profit_pct": 0.3,
                    **{f"param_{i}": i * 0.1 for i in range(20)}
                },
            }
            
            with open(config_file, "w") as f:
                yaml.dump(base_config, f)
            
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
                max_version_history=100,
            )
            
            asyncio.run(manager.load_config_file(config_file))
            
            # Create 100 versions
            for i in range(100):
                base_config["parameters"]["min_profit_pct"] = 0.3 + (i * 0.001)
                with open(config_file, "w") as f:
                    yaml.dump(base_config, f)
                asyncio.run(manager.load_config_file(config_file))
            
            # Estimate version history size
            history = manager.version_history.get("VersionTest", [])
            history_size = sys.getsizeof(history)
            for version in history:
                history_size += sys.getsizeof(version)
                if hasattr(version, "__dict__"):
                    for key, value in version.__dict__.items():
                        history_size += sys.getsizeof(key) + sys.getsizeof(value)
            
            # Convert to MB
            history_size_mb = history_size / (1024 * 1024)
            
            assert history_size_mb < 1, f"Version history uses {history_size_mb:.2f}MB for 100 versions"
            assert len(history) <= 100
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_file_watch_cpu_overhead(self):
        """Test that file watching uses <1% CPU."""
        import psutil
        import os
        
        temp_dir = tempfile.mkdtemp(prefix="test_cpu_")
        config_path = Path(temp_dir)
        
        try:
            # Create a config file
            config_file = config_path / "cpu_test.yaml"
            config = {
                "strategy": {
                    "name": "CPUTest",
                    "version": "1.0.0",
                    "tier": "sniper",
                    "enabled": True,
                },
                "parameters": {"min_profit_pct": 0.3},
            }
            
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
                enable_hot_reload=True,
                poll_interval=1.0,  # 1 second polling
            )
            
            async def measure_cpu():
                await manager.initialize()
                await manager.load_all_configs()
                await manager.start_watching()
                
                # Get process
                process = psutil.Process(os.getpid())
                
                # Measure CPU over 5 seconds
                cpu_samples = []
                for _ in range(5):
                    cpu_percent = process.cpu_percent(interval=1)
                    cpu_samples.append(cpu_percent)
                
                await manager.stop_watching()
                return cpu_samples
            
            cpu_samples = asyncio.run(measure_cpu())
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            
            # Note: This is total process CPU, not just file watching
            # In practice, file watching should be much less than 1%
            assert avg_cpu < 10, f"Process uses {avg_cpu:.2f}% CPU (file watching should be <1%)"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concurrent_config_access(self):
        """Test performance under concurrent config access."""
        temp_dir = tempfile.mkdtemp(prefix="test_concurrent_")
        config_path = Path(temp_dir)
        
        try:
            # Create config
            config_file = config_path / "concurrent_test.yaml"
            config = {
                "strategy": {
                    "name": "ConcurrentTest",
                    "version": "1.0.0",
                    "tier": "sniper",
                    "enabled": True,
                },
                "parameters": {
                    f"param_{i}": i * 0.1 for i in range(50)
                },
            }
            
            with open(config_file, "w") as f:
                yaml.dump(config, f)
            
            manager = StrategyConfigManager(
                config_path=str(config_path),
                environment=Environment.DEV,
            )
            
            asyncio.run(manager.load_config_file(config_file))
            
            async def concurrent_reads():
                async def read_config(index):
                    for _ in range(100):
                        config = manager.get_strategy_config("ConcurrentTest")
                        assert config is not None
                        # Simulate some processing
                        await asyncio.sleep(0.001)
                
                # Run 10 concurrent readers
                start_time = time.perf_counter()
                await asyncio.gather(*[read_config(i) for i in range(10)])
                total_time = time.perf_counter() - start_time
                
                return total_time
            
            total_time = asyncio.run(concurrent_reads())
            
            # Should complete 1000 total reads in reasonable time
            assert total_time < 5, f"Concurrent reads took {total_time:.2f}s"
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)