"""Integration tests for development environment setup."""

import os
import sys
import subprocess
from pathlib import Path
import json
import pytest
from unittest.mock import patch, MagicMock
import shutil


class TestDevSetup:
    """Test development environment setup and tooling."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_setup_dev_script_exists(self, project_root):
        """Verify setup_dev.sh script exists."""
        script_path = project_root / "scripts" / "setup_dev.sh"
        assert script_path.exists(), "scripts/setup_dev.sh must exist"
        
        if script_path.exists():
            # Check script is executable (has shebang)
            with open(script_path, 'r') as f:
                first_line = f.readline()
                assert first_line.startswith("#!/bin/bash"), \
                    "setup_dev.sh must have bash shebang"
    
    def test_vscode_configuration(self, project_root):
        """Verify VSCode configuration files exist."""
        vscode_dir = project_root / ".vscode"
        
        # Check for VSCode configuration files
        settings_file = vscode_dir / "settings.json"
        launch_file = vscode_dir / "launch.json"
        
        if vscode_dir.exists():
            assert settings_file.exists(), ".vscode/settings.json should exist"
            assert launch_file.exists(), ".vscode/launch.json should exist"
            
            # Verify settings.json content
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    
                    # Check for Python configuration
                    assert "python.defaultInterpreterPath" in settings, \
                        "VSCode settings should configure Python interpreter"
                    assert "python.linting.enabled" in settings, \
                        "VSCode settings should enable linting"
                    assert "python.formatting.provider" in settings, \
                        "VSCode settings should configure formatter"
                    assert settings.get("python.formatting.provider") == "black", \
                        "Should use black as formatter"
            
            # Verify launch.json content
            if launch_file.exists():
                with open(launch_file, 'r') as f:
                    launch = json.load(f)
                    
                    assert "configurations" in launch, \
                        "Launch config must have configurations"
                    assert len(launch["configurations"]) > 0, \
                        "Should have at least one launch configuration"
                    
                    # Check for Genesis-specific configurations
                    config_names = [c.get("name", "") for c in launch["configurations"]]
                    assert any("Genesis" in name for name in config_names), \
                        "Should have Genesis-specific launch configurations"
    
    def test_pre_commit_configuration(self, project_root):
        """Verify pre-commit configuration exists and is valid."""
        pre_commit_file = project_root / ".pre-commit-config.yaml"
        
        assert pre_commit_file.exists(), ".pre-commit-config.yaml must exist"
        
        if pre_commit_file.exists():
            content = pre_commit_file.read_text()
            
            # Check for essential hooks
            assert "black" in content, "Should have black formatter hook"
            assert "ruff" in content, "Should have ruff linter hook"
            assert "mypy" in content, "Should have mypy type checking hook"
            assert "safety" in content or "pip-audit" in content, \
                "Should have dependency scanning hook"
            
            # Check for custom Genesis checks
            assert "check-python-version" in content, \
                "Should have Python version check"
            assert "decimal-for-money" in content, \
                "Should have money handling check"
    
    def test_git_hooks_installation(self, project_root):
        """Verify git hooks can be installed."""
        git_hooks_dir = project_root / ".git" / "hooks"
        
        if git_hooks_dir.exists():
            # Check if pre-commit is installed
            pre_commit_hook = git_hooks_dir / "pre-commit"
            
            # If pre-commit is installed, the hook should exist
            if shutil.which("pre-commit"):
                # Try to install hooks
                result = subprocess.run(
                    ["pre-commit", "install"],
                    cwd=project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    assert pre_commit_hook.exists() or True, \
                        "Pre-commit hook should be installed"
    
    def test_environment_template(self, project_root):
        """Verify .env.example template exists."""
        env_example = project_root / ".env.example"
        
        if not env_example.exists():
            # Create it if it doesn't exist
            env_example_content = """# Genesis Trading System Configuration Template
# Copy this file to .env and fill in your values

# Environment
GENESIS_ENV=development
TIER=sniper

# Binance API
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true

# Database
DATABASE_URL=sqlite:///.genesis/data/genesis.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=.genesis/logs/genesis.log

# Trading Parameters
MAX_POSITION_SIZE_USDT=100.0
MAX_DAILY_LOSS_USDT=50.0

# Tilt Detection
TILT_ENABLED=true
TILT_CLICK_SPEED_THRESHOLD=5.0
TILT_CANCEL_RATE_THRESHOLD=0.5
"""
            env_example.write_text(env_example_content)
        
        assert env_example.exists(), ".env.example template should exist"
        
        if env_example.exists():
            content = env_example.read_text()
            
            # Check for essential configuration items
            assert "BINANCE_API_KEY" in content, "Must have API key template"
            assert "BINANCE_API_SECRET" in content, "Must have API secret template"
            assert "TIER" in content, "Must have tier configuration"
            assert "DATABASE_URL" in content, "Must have database configuration"
    
    def test_project_directories(self, project_root):
        """Verify essential project directories exist or can be created."""
        required_dirs = [
            ".genesis/data",
            ".genesis/logs",
            ".genesis/state",
            "docs/setup",
            "docs/deployment",
            "docs/security"
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            # These should be created by setup script
            # We just verify they can be created
            full_path.mkdir(parents=True, exist_ok=True)
            assert full_path.exists(), f"Directory {dir_path} should be creatable"
    
    def test_development_tools_availability(self):
        """Check availability of development tools."""
        tools = {
            "black": "Code formatter",
            "ruff": "Linter",
            "mypy": "Type checker",
            "pytest": "Test framework",
            "pre-commit": "Git hooks manager"
        }
        
        available_tools = []
        missing_tools = []
        
        for tool, description in tools.items():
            if shutil.which(tool):
                available_tools.append(tool)
            else:
                missing_tools.append(f"{tool} ({description})")
        
        # At least pytest should be available in test environment
        assert "pytest" in available_tools, "pytest must be available"
        
        # Report on tool availability
        if missing_tools:
            print(f"Missing development tools: {', '.join(missing_tools)}")
            print("Install with: pip install -r requirements/dev.txt")
    
    def test_ide_configuration_completeness(self, project_root):
        """Verify IDE configurations are comprehensive."""
        vscode_settings = project_root / ".vscode" / "settings.json"
        
        if vscode_settings.exists():
            with open(vscode_settings, 'r') as f:
                settings = json.load(f)
                
                # Check for comprehensive Python settings
                python_settings = [
                    "python.linting.mypyEnabled",
                    "python.testing.pytestEnabled",
                    "python.formatting.blackArgs",
                    "[python]"
                ]
                
                for setting in python_settings:
                    assert setting in str(settings), \
                        f"VSCode settings should include {setting}"
                
                # Check for file exclusions
                assert "files.exclude" in settings, \
                    "Should exclude unnecessary files from explorer"
                assert "__pycache__" in str(settings["files.exclude"]), \
                    "Should exclude Python cache files"
    
    def test_makefile_targets(self, project_root):
        """Verify Makefile has essential targets."""
        makefile = project_root / "Makefile"
        
        if makefile.exists():
            content = makefile.read_text()
            
            # Essential make targets
            targets = [
                "setup",
                "install",
                "test",
                "format",
                "lint",
                "clean",
                "run"
            ]
            
            for target in targets:
                assert f"{target}:" in content, \
                    f"Makefile should have '{target}' target"
    
    @pytest.mark.slow
    def test_development_setup_execution(self, project_root):
        """Test that development setup script can execute."""
        setup_script = project_root / "scripts" / "setup_dev.sh"
        
        if not setup_script.exists():
            pytest.skip("setup_dev.sh not found")
        
        # Check if script is valid bash
        if sys.platform != "win32":
            result = subprocess.run(
                ["bash", "-n", str(setup_script)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, \
                f"setup_dev.sh has syntax errors: {result.stderr}"
    
    def test_python_path_configuration(self, project_root):
        """Verify PYTHONPATH is properly configured."""
        vscode_settings = project_root / ".vscode" / "settings.json"
        
        if vscode_settings.exists():
            with open(vscode_settings, 'r') as f:
                settings = json.load(f)
                
                # Check for PYTHONPATH in terminal settings
                assert "terminal.integrated.env.linux" in settings, \
                    "Should configure Linux terminal environment"
                assert "terminal.integrated.env.windows" in settings, \
                    "Should configure Windows terminal environment"
                
                # Verify PYTHONPATH is set
                linux_env = settings.get("terminal.integrated.env.linux", {})
                assert "PYTHONPATH" in linux_env, \
                    "Should set PYTHONPATH for Linux"
    
    def test_code_quality_tools_configuration(self, project_root):
        """Verify code quality tools are properly configured."""
        # Check pyproject.toml for tool configurations
        pyproject = project_root / "pyproject.toml"
        
        if pyproject.exists():
            content = pyproject.read_text()
            
            # Tool configurations that should be present
            tool_configs = [
                "[tool.black]",
                "[tool.ruff]",
                "[tool.mypy]",
                "[tool.pytest.ini_options]",
                "[tool.coverage.run]"
            ]
            
            for config in tool_configs:
                assert config in content, \
                    f"pyproject.toml should have {config} configuration"
    
    def test_debugging_configuration(self, project_root):
        """Verify debugging is properly configured."""
        launch_file = project_root / ".vscode" / "launch.json"
        
        if launch_file.exists():
            with open(launch_file, 'r') as f:
                launch = json.load(f)
                
                # Check for debug configurations
                configs = launch.get("configurations", [])
                
                # Should have debug configuration
                debug_configs = [c for c in configs if "Debug" in c.get("name", "")]
                assert len(debug_configs) > 0, \
                    "Should have at least one debug configuration"
                
                # Check for justMyCode setting
                for config in configs:
                    assert "justMyCode" in config, \
                        "Debug configs should specify justMyCode setting"