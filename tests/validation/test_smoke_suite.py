"""Unit tests for smoke test suite."""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.smoke_tests import SmokeTestSuite, SmokeTestResult


class TestSmokeTestResult:
    """Test SmokeTestResult class."""
    
    def test_smoke_test_result_creation(self):
        """Test SmokeTestResult creation."""
        result = SmokeTestResult(
            name="Test",
            passed=True,
            duration=1.5,
            message="Success",
            details={"key": "value"}
        )
        
        assert result.name == "Test"
        assert result.passed is True
        assert result.duration == 1.5
        assert result.message == "Success"
        assert result.details["key"] == "value"
        assert result.timestamp is not None
    
    def test_smoke_test_result_to_dict(self):
        """Test SmokeTestResult serialization."""
        result = SmokeTestResult(
            name="Test",
            passed=False,
            duration=0.5,
            message="Failed"
        )
        
        data = result.to_dict()
        assert data["name"] == "Test"
        assert data["passed"] is False
        assert data["duration"] == 0.5
        assert data["message"] == "Failed"
        assert "timestamp" in data


class TestSmokeTestSuite:
    """Test smoke test suite."""
    
    def test_smoke_test_suite_initialization(self):
        """Test SmokeTestSuite initialization."""
        suite = SmokeTestSuite()
        assert suite.results == []
        assert suite.start_time is None
        assert suite.end_time is None
    
    @pytest.mark.asyncio
    async def test_python_version_test(self):
        """Test Python version check."""
        suite = SmokeTestSuite()
        
        # Test with correct version
        with patch("sys.version_info") as mock_version:
            mock_version.major = 3
            mock_version.minor = 11
            
            result = await suite.test_python_version()
            assert result is True
        
        # Test with wrong version
        with patch("sys.version_info") as mock_version:
            mock_version.major = 3
            mock_version.minor = 10
            
            result = await suite.test_python_version()
            assert isinstance(result, str)
            assert "3.10" in result
    
    @pytest.mark.asyncio
    async def test_required_directories(self):
        """Test required directories check."""
        suite = SmokeTestSuite()
        
        with patch("pathlib.Path.exists") as mock_exists:
            # All directories exist
            mock_exists.return_value = True
            result = await suite.test_required_directories()
            assert result is True
            
            # Some directories missing
            mock_exists.side_effect = [True, False, True, True, True]
            result = await suite.test_required_directories()
            assert isinstance(result, str)
            assert "Missing directories" in result
    
    @pytest.mark.asyncio
    async def test_configuration_files(self):
        """Test configuration files check."""
        suite = SmokeTestSuite()
        
        with patch("pathlib.Path.exists") as mock_exists:
            # All config files exist
            mock_exists.return_value = True
            result = await suite.test_configuration_files()
            assert result is True
            
            # Some files missing
            mock_exists.side_effect = [True, False, True]
            result = await suite.test_configuration_files()
            assert isinstance(result, str)
            assert "Missing config files" in result
    
    @pytest.mark.asyncio
    async def test_database_connection(self):
        """Test database connection check."""
        suite = SmokeTestSuite()
        
        with patch("sqlite3.connect") as mock_connect:
            # Successful connection
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ["3.39.0"]
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            result = await suite.test_database_connection()
            assert result is True
            
            # Connection failure
            mock_connect.side_effect = Exception("Connection failed")
            result = await suite.test_database_connection()
            assert isinstance(result, str)
            assert "Database error" in result
    
    @pytest.mark.asyncio
    async def test_import_core_modules(self):
        """Test core module imports."""
        suite = SmokeTestSuite()
        
        with patch("builtins.__import__") as mock_import:
            # All imports successful
            mock_import.return_value = MagicMock()
            result = await suite.test_import_core_modules()
            assert result is True
            
            # Some imports fail
            def import_side_effect(name):
                if "genesis.engine" in name:
                    raise ImportError(f"Module {name} not found")
                return MagicMock()
            
            mock_import.side_effect = import_side_effect
            result = await suite.test_import_core_modules()
            assert isinstance(result, str)
            assert "Import errors" in result
    
    @pytest.mark.asyncio
    async def test_environment_variables(self):
        """Test environment variables check."""
        suite = SmokeTestSuite()
        
        with patch("pathlib.Path.exists") as mock_exists:
            # .env file exists
            mock_exists.return_value = True
            result = await suite.test_environment_variables()
            assert result is True
            
            # No .env file, but env vars set
            mock_exists.return_value = False
            with patch("os.getenv") as mock_getenv:
                mock_getenv.return_value = "test_value"
                result = await suite.test_environment_variables()
                assert result is True
                
                # Env vars not set
                mock_getenv.return_value = None
                result = await suite.test_environment_variables()
                assert isinstance(result, str)
                assert "Missing env vars" in result
    
    @pytest.mark.asyncio
    async def test_log_directory(self):
        """Test log directory check."""
        suite = SmokeTestSuite()
        
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            with patch("pathlib.Path.write_text") as mock_write:
                with patch("pathlib.Path.unlink") as mock_unlink:
                    result = await suite.test_log_directory()
                    assert result is True
                    
                    # Write fails
                    mock_write.side_effect = Exception("Permission denied")
                    result = await suite.test_log_directory()
                    assert isinstance(result, str)
                    assert "not writable" in result
    
    @pytest.mark.asyncio
    async def test_network_connectivity(self):
        """Test network connectivity check."""
        suite = SmokeTestSuite()
        
        with patch("socket.gethostbyname") as mock_gethostbyname:
            # DNS resolution successful
            mock_gethostbyname.return_value = "1.2.3.4"
            result = await suite.test_network_connectivity()
            assert result is True
            
            # DNS resolution fails
            import socket
            mock_gethostbyname.side_effect = socket.gaierror("DNS failed")
            result = await suite.test_network_connectivity()
            assert isinstance(result, str)
            assert "Cannot resolve" in result
    
    @pytest.mark.asyncio
    async def test_critical_dependencies(self):
        """Test critical dependencies check."""
        suite = SmokeTestSuite()
        
        with patch("builtins.__import__") as mock_import:
            # All packages available
            mock_import.return_value = MagicMock()
            result = await suite.test_critical_dependencies()
            assert result is True
            
            # Some packages missing
            def import_side_effect(name):
                if name == "ccxt":
                    raise ImportError(f"Module {name} not found")
                return MagicMock()
            
            mock_import.side_effect = import_side_effect
            result = await suite.test_critical_dependencies()
            assert isinstance(result, str)
            assert "Missing packages" in result
    
    @pytest.mark.asyncio
    async def test_run_test_method(self):
        """Test _run_test method."""
        suite = SmokeTestSuite()
        
        # Test with passing test
        async def passing_test():
            return True
        
        await suite._run_test(passing_test)
        assert len(suite.results) == 1
        assert suite.results[0].passed is True
        
        # Test with failing test
        async def failing_test():
            return "Test failed"
        
        await suite._run_test(failing_test)
        assert len(suite.results) == 2
        assert suite.results[1].passed is False
        assert suite.results[1].message == "Test failed"
        
        # Test with exception
        async def error_test():
            raise Exception("Test error")
        
        await suite._run_test(error_test)
        assert len(suite.results) == 3
        assert suite.results[2].passed is False
        assert "Test error" in suite.results[2].message
    
    @pytest.mark.asyncio
    async def test_full_smoke_test_run(self):
        """Test full smoke test run."""
        suite = SmokeTestSuite()
        
        # Mock all tests to pass quickly
        with patch.object(suite, "test_python_version", return_value=True):
            with patch.object(suite, "test_required_directories", return_value=True):
                with patch.object(suite, "test_configuration_files", return_value=True):
                    with patch.object(suite, "test_database_connection", return_value=True):
                        with patch.object(suite, "test_import_core_modules", return_value=True):
                            with patch.object(suite, "test_environment_variables", return_value=True):
                                with patch.object(suite, "test_log_directory", return_value=True):
                                    with patch.object(suite, "test_backup_directory", return_value=True):
                                        with patch.object(suite, "test_network_connectivity", return_value=True):
                                            with patch.object(suite, "test_critical_dependencies", return_value=True):
                                                result = await suite.run()
                                                
                                                assert result is True
                                                assert len(suite.results) == 10
                                                assert all(r.passed for r in suite.results)
                                                assert suite.start_time is not None
                                                assert suite.end_time is not None
    
    def test_save_results(self):
        """Test saving results to file."""
        suite = SmokeTestSuite()
        
        # Add some test results
        suite.results.append(SmokeTestResult("Test1", True, 1.0))
        suite.results.append(SmokeTestResult("Test2", False, 2.0, "Failed"))
        suite.start_time = 1000.0
        suite.end_time = 1010.0
        
        with patch("builtins.open", create=True) as mock_open:
            with patch("json.dump") as mock_dump:
                suite.save_results("test_output.json")
                
                mock_open.assert_called_once()
                mock_dump.assert_called_once()
                
                # Check the data being saved
                saved_data = mock_dump.call_args[0][0]
                assert saved_data["total_tests"] == 2
                assert saved_data["passed"] == 1
                assert saved_data["failed"] == 1
                assert saved_data["duration"] == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])