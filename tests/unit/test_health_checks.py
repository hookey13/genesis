"""
Unit tests for health check endpoints and doctor command.

Tests verify that health checks properly detect system status
and return appropriate exit codes for Docker health checks.
"""

import asyncio
import os
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genesis.cli.doctor import DoctorRunner, HealthCheck


class TestHealthCheck:
    """Test suite for HealthCheck class."""
    
    def test_health_check_creation(self):
        """Test creating a health check result."""
        check = HealthCheck("Test Check", True, "All good")
        
        assert check.name == "Test Check"
        assert check.passed is True
        assert check.message == "All good"
        assert check.severity == "error"  # Default severity
    
    def test_health_check_with_severity(self):
        """Test creating a health check with custom severity."""
        check = HealthCheck("Warning Check", False, "Minor issue", severity="warning")
        
        assert check.name == "Warning Check"
        assert check.passed is False
        assert check.message == "Minor issue"
        assert check.severity == "warning"


class TestDoctorRunner:
    """Test suite for DoctorRunner health checks."""
    
    @pytest.fixture
    def runner(self):
        """Create a DoctorRunner instance."""
        return DoctorRunner()
    
    @patch('genesis.cli.doctor.get_settings')
    @patch('genesis.cli.doctor.validate_configuration')
    def test_check_configuration_valid(self, mock_validate, mock_get_settings, runner):
        """Test configuration check when valid."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = {
            "valid": True,
            "tier": "sniper",
            "environment": "development",
            "warnings": []
        }
        
        runner._check_configuration()
        
        assert runner.settings == mock_settings
        assert len(runner.checks) == 1
        assert runner.checks[0].passed is True
        assert "sniper tier" in runner.checks[0].message
    
    @patch('genesis.cli.doctor.get_settings')
    @patch('genesis.cli.doctor.validate_configuration')
    def test_check_configuration_with_warnings(self, mock_validate, mock_get_settings, runner):
        """Test configuration check with warnings."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = {
            "valid": True,
            "tier": "sniper",
            "environment": "development",
            "warnings": ["Warning 1", "Warning 2"]
        }
        
        runner._check_configuration()
        
        assert runner.settings == mock_settings
        assert len(runner.checks) == 3  # 1 pass + 2 warnings
        assert runner.checks[0].passed is True
        assert runner.checks[1].passed is False
        assert runner.checks[1].severity == "warning"
    
    @patch('genesis.cli.doctor.create_engine')
    def test_check_database_connection(self, mock_create_engine, runner):
        """Test database connectivity check."""
        runner.settings = Mock()
        runner.settings.database.database_url = "sqlite:///test.db"
        
        # Mock successful connection
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_create_engine.return_value = mock_engine
        
        runner._check_database()
        
        assert len(runner.checks) == 1
        assert runner.checks[0].passed is True
        assert "sqlite" in runner.checks[0].message
    
    @patch('genesis.cli.doctor.create_engine')
    def test_check_database_connection_failure(self, mock_create_engine, runner):
        """Test database connectivity check failure."""
        runner.settings = Mock()
        runner.settings.database.database_url = "sqlite:///test.db"
        
        # Mock connection failure
        mock_create_engine.side_effect = Exception("Connection failed")
        
        runner._check_database()
        
        assert len(runner.checks) == 1
        assert runner.checks[0].passed is False
        assert "Connection failed" in runner.checks[0].message
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_check_rest_api(self, mock_session_class, runner):
        """Test REST API connectivity check."""
        runner.settings = Mock()
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status = 200
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        await runner._check_rest_api()
        
        assert len(runner.checks) == 1
        assert runner.checks[0].passed is True
        assert "Connected successfully" in runner.checks[0].message
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession')
    async def test_check_rest_api_failure(self, mock_session_class, runner):
        """Test REST API connectivity check failure."""
        runner.settings = Mock()
        
        # Mock API failure
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Connection timeout")
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        await runner._check_rest_api()
        
        assert len(runner.checks) == 1
        assert runner.checks[0].passed is False
        assert "Connection timeout" in runner.checks[0].message
    
    @pytest.mark.asyncio
    async def test_check_redis_not_configured(self, runner):
        """Test Redis check when not configured."""
        runner.settings = Mock()
        
        with patch.dict(os.environ, {}, clear=True):
            await runner._check_redis()
        
        # No checks should be added if Redis is not configured
        assert len(runner.checks) == 0
    
    @pytest.mark.asyncio
    @patch('redis.asyncio.from_url')
    async def test_check_redis_success(self, mock_redis_from_url, runner):
        """Test Redis connectivity check success."""
        runner.settings = Mock()
        
        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_from_url.return_value = mock_client
        
        with patch.dict(os.environ, {'REDIS_URL': 'redis://localhost:6379'}):
            await runner._check_redis()
        
        assert len(runner.checks) == 1
        assert runner.checks[0].passed is True
        assert "Connected successfully" in runner.checks[0].message
        assert runner.checks[0].severity == "warning"  # Redis is not critical for MVP
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_check_system_resources(self, mock_cpu_percent, mock_virtual_memory, 
                                   mock_process_class, runner):
        """Test system resource usage check."""
        runner.settings = Mock()
        
        # Mock process stats
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.0
        mock_process.cpu_percent.return_value = 25.0
        mock_process.open_files.return_value = [Mock()] * 10  # 10 open files
        mock_process_class.return_value = mock_process
        
        # Mock system stats
        mock_virtual_memory.return_value = Mock(percent=50.0)
        mock_cpu_percent.return_value = 30.0
        
        runner._check_system_resources()
        
        # Should have checks for memory, CPU, and open files
        assert len(runner.checks) >= 3
        
        # Find specific checks
        memory_check = next((c for c in runner.checks if c.name == "Memory Usage"), None)
        cpu_check = next((c for c in runner.checks if c.name == "CPU Usage"), None)
        files_check = next((c for c in runner.checks if c.name == "Open Files"), None)
        
        assert memory_check is not None and memory_check.passed is True
        assert cpu_check is not None and cpu_check.passed is True
        assert files_check is not None and files_check.passed is True
    
    @patch('psutil.process_iter')
    def test_check_background_tasks(self, mock_process_iter, runner):
        """Test background task checking."""
        runner.settings = Mock()
        
        # Mock process list
        mock_genesis_proc = Mock()
        mock_genesis_proc.info = {
            'pid': 1234,
            'name': 'python',
            'cmdline': ['python', '-m', 'genesis']
        }
        
        mock_supervisor_proc = Mock()
        mock_supervisor_proc.info = {
            'pid': 5678,
            'name': 'supervisord',
            'cmdline': ['/usr/bin/supervisord']
        }
        
        mock_process_iter.return_value = [mock_genesis_proc, mock_supervisor_proc]
        
        runner._check_background_tasks()
        
        # Should detect both processes
        genesis_check = next((c for c in runner.checks if c.name == "Genesis Process"), None)
        supervisor_check = next((c for c in runner.checks if c.name == "Supervisor Process"), None)
        
        assert genesis_check is not None and genesis_check.passed is True
        assert supervisor_check is not None and supervisor_check.passed is True
    
    @pytest.mark.asyncio
    @patch('genesis.cli.doctor.get_settings')
    async def test_run_all_checks_critical_failure(self, mock_get_settings, runner):
        """Test that critical failures return False."""
        # Mock configuration failure (critical)
        mock_get_settings.side_effect = Exception("Config error")
        
        result = await runner.run_all_checks()
        
        assert result is False  # Critical check failed
        assert len(runner.checks) > 0
        assert any(not c.passed and c.severity == "error" for c in runner.checks)
    
    @pytest.mark.asyncio
    @patch('genesis.cli.doctor.get_settings')
    @patch('genesis.cli.doctor.validate_configuration')
    async def test_run_all_checks_warnings_only(self, mock_validate, mock_get_settings, runner):
        """Test that warnings don't cause failure."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        mock_validate.return_value = {
            "valid": True,
            "tier": "sniper",
            "environment": "development",
            "warnings": ["Minor warning"]
        }
        
        # Mock other checks to pass
        with patch.object(runner, '_check_database'):
            with patch.object(runner, '_check_migrations'):
                with patch.object(runner, '_check_rest_api', new_callable=AsyncMock):
                    with patch.object(runner, '_check_clock_drift', new_callable=AsyncMock):
                        with patch.object(runner, '_check_websocket', new_callable=AsyncMock):
                            with patch.object(runner, '_check_redis', new_callable=AsyncMock):
                                with patch.object(runner, '_check_system_resources'):
                                    with patch.object(runner, '_check_background_tasks'):
                                        result = await runner.run_all_checks()
        
        # Should pass even with warnings
        assert result is True


class TestDoctorCommand:
    """Test suite for the doctor CLI command."""
    
    @patch('genesis.cli.doctor.DoctorRunner')
    @patch('genesis.cli.doctor.asyncio.new_event_loop')
    def test_doctor_command_success(self, mock_event_loop, mock_runner_class):
        """Test doctor command with all checks passing."""
        # Mock runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Mock event loop
        mock_loop = Mock()
        mock_loop.run_until_complete.return_value = True  # All checks pass
        mock_event_loop.return_value = mock_loop
        
        with pytest.raises(SystemExit) as exc_info:
            from genesis.cli.doctor import doctor
            doctor()
        
        assert exc_info.value.code == 0  # Success exit code
    
    @patch('genesis.cli.doctor.DoctorRunner')
    @patch('genesis.cli.doctor.asyncio.new_event_loop')
    def test_doctor_command_failure(self, mock_event_loop, mock_runner_class):
        """Test doctor command with critical failures."""
        # Mock runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Mock event loop
        mock_loop = Mock()
        mock_loop.run_until_complete.return_value = False  # Some checks failed
        mock_event_loop.return_value = mock_loop
        
        with pytest.raises(SystemExit) as exc_info:
            from genesis.cli.doctor import doctor
            doctor()
        
        assert exc_info.value.code == 1  # Failure exit code