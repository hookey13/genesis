"""Unit tests for Python environment configuration."""

import sys
import os
import subprocess
from pathlib import Path
import pytest


class TestPythonEnvironment:
    """Test Python version and environment setup."""
    
    def test_python_version(self):
        """Verify Python version is exactly 3.11.8."""
        version_info = sys.version_info
        assert version_info.major == 3, "Python major version must be 3"
        assert version_info.minor == 11, "Python minor version must be 11"
        assert version_info.micro == 8, f"Python micro version must be 8, got {version_info.micro}"
        
    def test_python_version_string(self):
        """Verify Python version string format."""
        version = sys.version
        assert version.startswith("3.11.8"), f"Python version must be 3.11.8, got {version}"
        
    def test_pyenv_version_file_exists(self):
        """Verify .python-version file exists."""
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / ".python-version"
        assert version_file.exists(), ".python-version file must exist"
        
    def test_pyenv_version_content(self):
        """Verify .python-version file contains correct version."""
        project_root = Path(__file__).parent.parent.parent
        version_file = project_root / ".python-version"
        
        if version_file.exists():
            content = version_file.read_text().strip()
            assert content == "3.11.8", f".python-version must contain '3.11.8', got '{content}'"
            
    def test_virtual_environment_detection(self):
        """Check if running in a virtual environment."""
        # Check for virtual environment indicators
        in_venv = (
            hasattr(sys, 'real_prefix') or  # virtualenv
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
            os.environ.get('VIRTUAL_ENV') is not None  # environment variable
        )
        
        # This test is informational - it doesn't fail if not in venv
        if not in_venv:
            pytest.skip("Not running in virtual environment (informational)")
            
    def test_activation_scripts_exist(self):
        """Verify activation scripts exist."""
        project_root = Path(__file__).parent.parent.parent
        scripts_dir = project_root / "scripts"
        
        # Check Linux/Mac activation script
        bash_script = scripts_dir / "activate_env.sh"
        assert bash_script.exists(), "activate_env.sh must exist"
        
        # Check Windows activation script  
        ps1_script = scripts_dir / "activate_env.ps1"
        assert ps1_script.exists(), "activate_env.ps1 must exist"
        
    def test_activation_scripts_executable(self):
        """Verify bash activation script is executable on Unix systems."""
        if sys.platform == "win32":
            pytest.skip("Skipping Unix executable test on Windows")
            
        project_root = Path(__file__).parent.parent.parent
        bash_script = project_root / "scripts" / "activate_env.sh"
        
        if bash_script.exists():
            # Check if file has shebang
            with open(bash_script, 'r') as f:
                first_line = f.readline()
                assert first_line.startswith("#!/"), "Bash script must have shebang"
                
    def test_pip_tools_available(self):
        """Check if pip-tools is available for dependency management."""
        try:
            import piptools
            assert piptools is not None
        except ImportError:
            # pip-tools might not be installed in test environment
            pytest.skip("pip-tools not installed (expected in dev environment)")
            
    def test_project_root_env_variable(self):
        """Check if GENESIS_ROOT can be set."""
        project_root = Path(__file__).parent.parent.parent
        expected_root = str(project_root.resolve())
        
        # Set and verify environment variable
        os.environ['GENESIS_ROOT'] = expected_root
        assert os.environ.get('GENESIS_ROOT') == expected_root
        
    def test_pythonpath_configuration(self):
        """Verify project root can be added to PYTHONPATH."""
        project_root = Path(__file__).parent.parent.parent
        root_str = str(project_root.resolve())
        
        # Check if project root is in sys.path
        # It should be there for imports to work correctly
        assert any(root_str in path for path in sys.path), \
            f"Project root should be in Python path for imports"
            
    def test_tier_environment_variable(self):
        """Test tier environment variable handling."""
        # Test default tier
        tier = os.environ.get('TIER', 'sniper')
        assert tier in ['sniper', 'hunter', 'strategist'], \
            f"TIER must be one of: sniper, hunter, strategist. Got: {tier}"
            
    def test_requirements_directory_structure(self):
        """Verify requirements directory structure exists."""
        project_root = Path(__file__).parent.parent.parent
        req_dir = project_root / "requirements"
        
        assert req_dir.exists(), "requirements directory must exist"
        assert req_dir.is_dir(), "requirements must be a directory"
        
        # Check for tier-specific requirements files
        expected_files = ['base.txt', 'sniper.txt', 'hunter.txt', 'strategist.txt', 'dev.txt']
        for filename in expected_files:
            file_path = req_dir / filename
            assert file_path.exists(), f"requirements/{filename} must exist"
            
    def test_python_stdlib_features(self):
        """Test that required Python 3.11 stdlib features are available."""
        # Test asyncio features
        import asyncio
        assert hasattr(asyncio, 'TaskGroup'), "Python 3.11 TaskGroup must be available"
        
        # Test typing features
        import typing
        assert hasattr(typing, 'Self'), "Python 3.11 typing.Self must be available"
        assert hasattr(typing, 'TypeVarTuple'), "Python 3.11 TypeVarTuple must be available"
        
        # Test exception groups (new in 3.11)
        try:
            raise ExceptionGroup("test", [ValueError("test")])
        except ExceptionGroup:
            pass  # Feature is available
        except NameError:
            pytest.fail("Python 3.11 ExceptionGroup not available")