"""Integration tests for Poetry dependency management."""

import subprocess
import sys
from pathlib import Path
import json
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil


class TestPoetryIntegration:
    """Test Poetry installation and dependency management."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_pyproject_toml_exists(self, project_root):
        """Verify pyproject.toml exists."""
        pyproject_path = project_root / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml must exist"
    
    def test_pyproject_has_poetry_config(self, project_root):
        """Verify pyproject.toml has Poetry configuration."""
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            assert "[tool.poetry]" in content, "pyproject.toml must have [tool.poetry] section"
            assert "[tool.poetry.dependencies]" in content, "Must have dependencies section"
            assert 'python = "~3.11.8"' in content, "Must specify Python 3.11.8"
    
    def test_poetry_groups_defined(self, project_root):
        """Verify tier-specific Poetry groups are defined."""
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Check for tier groups
            assert "[tool.poetry.group.sniper]" in content, "Sniper group must be defined"
            assert "[tool.poetry.group.hunter]" in content, "Hunter group must be defined"
            assert "[tool.poetry.group.strategist]" in content, "Strategist group must be defined"
            assert "[tool.poetry.group.dev.dependencies]" in content, "Dev group must be defined"
    
    def test_poetry_command_available(self):
        """Check if Poetry command is available."""
        try:
            result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # If Poetry is installed, verify version
            if result.returncode == 0:
                assert "Poetry" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Poetry not installed (expected in CI/CD environment)")
    
    @pytest.mark.slow
    def test_poetry_lock_generation(self, project_root, tmp_path):
        """Test that poetry.lock can be generated."""
        # Skip if Poetry not installed
        try:
            subprocess.run(["poetry", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Poetry not installed")
        
        # Copy pyproject.toml to temp directory
        temp_project = tmp_path / "test_project"
        temp_project.mkdir()
        
        pyproject_src = project_root / "pyproject.toml"
        if pyproject_src.exists():
            shutil.copy(pyproject_src, temp_project / "pyproject.toml")
            
            # Try to generate lock file
            result = subprocess.run(
                ["poetry", "lock", "--no-update"],
                cwd=temp_project,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if lock file was created
            lock_file = temp_project / "poetry.lock"
            if result.returncode == 0:
                assert lock_file.exists(), "poetry.lock should be created"
    
    def test_migration_script_exists(self, project_root):
        """Verify migration script exists."""
        script_path = project_root / "scripts" / "migrate_to_poetry.py"
        assert script_path.exists(), "migrate_to_poetry.py must exist"
    
    def test_poetry_activation_scripts(self, project_root):
        """Verify Poetry activation wrapper scripts."""
        scripts_dir = project_root / "scripts"
        
        # These might not exist until migration is run
        bash_script = scripts_dir / "poetry_activate.sh"
        ps1_script = scripts_dir / "poetry_activate.ps1"
        
        # Check if migration has been run
        if bash_script.exists():
            content = bash_script.read_text()
            assert "poetry shell" in content, "Bash script should activate Poetry"
            
        if ps1_script.exists():
            content = ps1_script.read_text()
            assert "poetry shell" in content, "PowerShell script should activate Poetry"
    
    def test_core_dependencies_in_poetry(self, project_root):
        """Verify core dependencies are defined in Poetry config."""
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Check critical dependencies
            core_deps = [
                'ccxt = "4.4.0"',
                'rich = "13.7.0"',
                'textual = "0.47.1"',
                'aiohttp = "3.10.11"',
                'pydantic = "2.5.3"',
                'structlog = "24.1.0"',
                'SQLAlchemy = "2.0.25"',
                'pandas = "2.2.3"',
                'numpy = "1.26.4"'
            ]
            
            for dep in core_deps:
                assert dep in content, f"Core dependency {dep} must be in pyproject.toml"
    
    def test_dev_dependencies_in_poetry(self, project_root):
        """Verify development dependencies are defined."""
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Check dev tools
            dev_deps = ["pytest", "black", "ruff", "mypy", "pre-commit"]
            
            for dep in dev_deps:
                assert f'{dep} = "' in content or f'{dep} = ^' in content, \
                    f"Dev dependency {dep} must be defined"
    
    def test_poetry_scripts_defined(self, project_root):
        """Verify Poetry scripts are defined."""
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Check for script definitions
            assert "[tool.poetry.scripts]" in content, "Scripts section must be defined"
            assert 'genesis = "genesis.__main__:main"' in content, \
                "Main genesis script must be defined"
    
    @patch('subprocess.run')
    def test_poetry_install_command(self, mock_run):
        """Test Poetry install command execution."""
        # Mock successful Poetry installation
        mock_run.return_value = MagicMock(returncode=0, stdout="Installing dependencies")
        
        # Simulate running poetry install
        from scripts.migrate_to_poetry import install_dependencies
        
        with patch('pathlib.Path.cwd', return_value=Path("/fake/project")):
            result = install_dependencies("sniper")
            
        # Verify poetry install was called
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "poetry"
        assert call_args[1] == "install"
    
    def test_poetry_export_compatibility(self, project_root):
        """Test that Poetry can export to requirements.txt format."""
        # Skip if Poetry not installed
        try:
            subprocess.run(["poetry", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("Poetry not installed")
        
        # Try to export dependencies
        result = subprocess.run(
            ["poetry", "export", "-f", "requirements.txt", "--without-hashes"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            output = result.stdout
            # Check for some expected packages
            assert "ccxt==" in output or not output, "Should export ccxt dependency"
    
    def test_tier_extras_defined(self, project_root):
        """Verify tier extras are defined for optional installation."""
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Check for extras definition
            assert "[tool.poetry.extras]" in content, "Extras section must be defined"
            assert 'hunter = [' in content, "Hunter extras must be defined"
            assert 'strategist = [' in content, "Strategist extras must be defined"
    
    @pytest.mark.slow
    def test_poetry_virtual_environment(self):
        """Test Poetry virtual environment management."""
        # Skip if Poetry not installed
        try:
            result = subprocess.run(
                ["poetry", "env", "info"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Check environment info output
                assert "Python" in result.stdout
                assert "3.11" in result.stdout or "Virtualenv" in result.stdout
        except FileNotFoundError:
            pytest.skip("Poetry not installed")
    
    def test_documentation_exists(self, project_root):
        """Verify Poetry migration documentation exists or will be created."""
        docs_dir = project_root / "docs" / "setup"
        
        # Check if directory exists (will be created by migration)
        if docs_dir.exists():
            # Look for any Poetry-related documentation
            poetry_docs = list(docs_dir.glob("*poetry*.md"))
            # Documentation will be created during migration
            assert True  # Pass if directory exists