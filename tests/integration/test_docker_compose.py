"""
Integration tests for Docker Compose orchestration.
Tests health checks, restart policies, and service dependencies.
"""

import os
import subprocess
import time
import json
import yaml
from pathlib import Path
from decimal import Decimal
import pytest
import structlog
import asyncio

logger = structlog.get_logger(__name__)


class TestDockerCompose:
    """Test Docker Compose orchestration configurations."""
    
    @pytest.fixture
    def compose_dir(self):
        """Provide Docker Compose directory path."""
        return Path(__file__).parent.parent.parent / "docker"
    
    @pytest.fixture
    def dev_compose_file(self, compose_dir):
        """Provide development compose file path."""
        return compose_dir / "docker-compose.yml"
    
    @pytest.fixture
    def prod_compose_file(self, compose_dir):
        """Provide production compose file path."""
        return compose_dir / "docker-compose.prod.yml"
    
    def test_compose_files_exist(self, dev_compose_file, prod_compose_file):
        """Test that Docker Compose files exist."""
        assert dev_compose_file.exists(), f"Development compose file not found at {dev_compose_file}"
        assert prod_compose_file.exists(), f"Production compose file not found at {prod_compose_file}"
    
    def test_dev_compose_syntax(self, dev_compose_file):
        """Test development compose file syntax."""
        with open(dev_compose_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "Empty compose file"
                assert 'version' in config, "Missing version field"
                assert 'services' in config, "Missing services field"
                assert 'genesis' in config['services'], "Missing genesis service"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax: {e}")
    
    def test_prod_compose_syntax(self, prod_compose_file):
        """Test production compose file syntax."""
        with open(prod_compose_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "Empty compose file"
                assert 'version' in config, "Missing version field"
                assert 'services' in config, "Missing services field"
                assert 'genesis' in config['services'], "Missing genesis service"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax: {e}")
    
    def test_health_checks_configured(self, dev_compose_file, prod_compose_file):
        """Test that health checks are properly configured."""
        for compose_file in [dev_compose_file, prod_compose_file]:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check genesis service health check
                genesis = config['services']['genesis']
                assert 'healthcheck' in genesis, f"Missing healthcheck in {compose_file.name}"
                
                healthcheck = genesis['healthcheck']
                assert 'test' in healthcheck, "Missing health check test command"
                assert 'interval' in healthcheck, "Missing health check interval"
                assert 'timeout' in healthcheck, "Missing health check timeout"
                assert 'retries' in healthcheck, "Missing health check retries"
                
                # Verify health check command
                test_cmd = healthcheck['test']
                assert isinstance(test_cmd, list), "Health check test should be a list"
                assert "genesis.api.health" in str(test_cmd), "Should use health module"
    
    def test_restart_policies(self, dev_compose_file, prod_compose_file):
        """Test restart policies are configured correctly."""
        with open(dev_compose_file, 'r') as f:
            dev_config = yaml.safe_load(f)
            genesis_dev = dev_config['services']['genesis']
            assert genesis_dev.get('restart') == 'unless-stopped', "Dev should restart unless-stopped"
        
        with open(prod_compose_file, 'r') as f:
            prod_config = yaml.safe_load(f)
            genesis_prod = prod_config['services']['genesis']
            assert genesis_prod.get('restart') == 'always', "Production should always restart"
    
    def test_resource_limits(self, dev_compose_file, prod_compose_file):
        """Test that resource limits are properly configured."""
        for compose_file in [dev_compose_file, prod_compose_file]:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                genesis = config['services']['genesis']
                
                # Check for deploy configuration
                assert 'deploy' in genesis, f"Missing deploy config in {compose_file.name}"
                deploy = genesis['deploy']
                
                # Check resource limits
                assert 'resources' in deploy, "Missing resources configuration"
                resources = deploy['resources']
                
                assert 'limits' in resources, "Missing resource limits"
                limits = resources['limits']
                assert 'cpus' in limits, "Missing CPU limit"
                assert 'memory' in limits, "Missing memory limit"
                
                # Check resource reservations
                assert 'reservations' in resources, "Missing resource reservations"
                reservations = resources['reservations']
                assert 'cpus' in reservations, "Missing CPU reservation"
                assert 'memory' in reservations, "Missing memory reservation"
    
    def test_volume_configuration(self, dev_compose_file, prod_compose_file):
        """Test volume configuration for data persistence."""
        required_volumes = [
            'genesis-data',
            'genesis-logs', 
            'genesis-state',
            'genesis-backups'
        ]
        
        for compose_file in [dev_compose_file, prod_compose_file]:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check volumes are defined
                assert 'volumes' in config, f"Missing volumes section in {compose_file.name}"
                volumes = config['volumes']
                
                for vol in required_volumes:
                    assert vol in volumes, f"Missing {vol} volume definition"
                
                # Check volumes are mounted in service
                genesis = config['services']['genesis']
                assert 'volumes' in genesis, "Genesis service missing volume mounts"
                
                service_volumes = genesis['volumes']
                for vol in required_volumes:
                    vol_mounted = any(vol in str(v) for v in service_volumes)
                    assert vol_mounted, f"{vol} not mounted in genesis service"
    
    def test_network_configuration(self, dev_compose_file, prod_compose_file):
        """Test network configuration and isolation."""
        for compose_file in [dev_compose_file, prod_compose_file]:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # Check network definition
                assert 'networks' in config, f"Missing networks section in {compose_file.name}"
                networks = config['networks']
                assert 'genesis-network' in networks, "Missing genesis-network definition"
                
                network_config = networks['genesis-network']
                assert network_config.get('driver') == 'bridge', "Should use bridge driver"
                
                # Check IPAM configuration
                if 'ipam' in network_config:
                    ipam = network_config['ipam']
                    if 'config' in ipam:
                        subnet_config = ipam['config'][0]
                        assert 'subnet' in subnet_config, "Missing subnet configuration"
    
    def test_logging_configuration(self, dev_compose_file, prod_compose_file):
        """Test logging is properly configured."""
        for compose_file in [dev_compose_file, prod_compose_file]:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                genesis = config['services']['genesis']
                
                # Check logging configuration
                assert 'logging' in genesis, f"Missing logging config in {compose_file.name}"
                logging = genesis['logging']
                
                assert logging.get('driver') == 'json-file', "Should use json-file driver"
                assert 'options' in logging, "Missing logging options"
                
                options = logging['options']
                assert 'max-size' in options, "Missing max-size option"
                assert 'max-file' in options, "Missing max-file option"
    
    def test_environment_variables(self, dev_compose_file, prod_compose_file):
        """Test environment variables are properly configured."""
        with open(dev_compose_file, 'r') as f:
            dev_config = yaml.safe_load(f)
            dev_env = dev_config['services']['genesis'].get('environment', [])
            
            # Check development environment
            env_dict = {}
            for item in dev_env:
                if '=' in item:
                    key, value = item.split('=', 1)
                    env_dict[key] = value
            
            assert env_dict.get('DEPLOYMENT_ENV') == 'development'
            assert env_dict.get('DEBUG') == 'true'
            assert env_dict.get('BINANCE_TESTNET') == 'true'
        
        with open(prod_compose_file, 'r') as f:
            prod_config = yaml.safe_load(f)
            prod_env = prod_config['services']['genesis'].get('environment', [])
            
            # Check production environment
            env_dict = {}
            for item in prod_env:
                if '=' in item:
                    key, value = item.split('=', 1)
                    env_dict[key] = value
            
            assert env_dict.get('DEPLOYMENT_ENV') == 'production'
            assert env_dict.get('DEBUG') == 'false'
            assert env_dict.get('BINANCE_TESTNET') == 'false'
    
    def test_service_dependencies(self, prod_compose_file):
        """Test service dependencies in production compose."""
        with open(prod_compose_file, 'r') as f:
            config = yaml.safe_load(f)
            genesis = config['services']['genesis']
            
            # Check depends_on configuration
            if 'depends_on' in genesis:
                depends = genesis['depends_on']
                
                # Check Redis dependency
                if 'redis' in depends:
                    redis_dep = depends['redis']
                    assert redis_dep.get('condition') == 'service_healthy', \
                        "Genesis should wait for Redis to be healthy"
                
                # Check PostgreSQL dependency
                if 'postgres' in depends:
                    postgres_dep = depends['postgres']
                    assert postgres_dep.get('condition') == 'service_healthy', \
                        "Genesis should wait for PostgreSQL to be healthy"
    
    def test_build_args_support(self, dev_compose_file, prod_compose_file):
        """Test that build args are properly configured."""
        for compose_file in [dev_compose_file, prod_compose_file]:
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                genesis = config['services']['genesis']
                
                if 'build' in genesis:
                    build = genesis['build']
                    
                    # Check build context
                    assert 'context' in build, "Missing build context"
                    assert build['context'] == '..', "Build context should be parent dir"
                    
                    # Check dockerfile path
                    assert 'dockerfile' in build, "Missing dockerfile path"
                    assert 'docker/Dockerfile' in build['dockerfile'], \
                        "Should reference docker/Dockerfile"
                    
                    # Check build args
                    if 'args' in build:
                        args = build['args']
                        assert 'BUILD_DATE' in args or 'VERSION' in args or 'VCS_REF' in args, \
                            "Should support build arguments"
    
    def test_port_exposure(self, dev_compose_file, prod_compose_file):
        """Test port exposure configuration."""
        with open(dev_compose_file, 'r') as f:
            dev_config = yaml.safe_load(f)
            dev_genesis = dev_config['services']['genesis']
            
            # Development should expose ports
            if 'ports' in dev_genesis:
                ports = dev_genesis['ports']
                assert any('8000' in str(p) for p in ports), \
                    "Development should expose port 8000"
        
        with open(prod_compose_file, 'r') as f:
            prod_config = yaml.safe_load(f)
            prod_genesis = prod_config['services']['genesis']
            
            # Production should bind to localhost only
            if 'ports' in prod_genesis:
                ports = prod_genesis['ports']
                for port in ports:
                    if '8000' in str(port):
                        assert '127.0.0.1' in str(port), \
                            "Production should bind to localhost only"
    
    @pytest.mark.skipif(
        not Path("/usr/bin/docker-compose").exists() and 
        not Path("/usr/local/bin/docker-compose").exists() and
        not Path("/usr/bin/docker").exists(),
        reason="Docker Compose not installed"
    )
    def test_compose_config_validation(self, compose_dir):
        """Validate Docker Compose configuration (requires Docker Compose)."""
        try:
            # Test development compose
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.yml", "config"],
                cwd=compose_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, f"Dev compose validation failed: {result.stderr}"
            
            # Test production compose
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.prod.yml", "config"],
                cwd=compose_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, f"Prod compose validation failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Docker Compose validation timed out")
        except FileNotFoundError:
            pytest.skip("Docker Compose command not found")