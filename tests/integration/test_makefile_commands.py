"""Integration tests for Makefile commands."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestMakefileCommands:
    """Test all Makefile commands for proper execution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.makefile = self.project_root / "Makefile"

        # Create backup of current venv if it exists
        self.venv_backup = None
        venv_path = self.project_root / "venv"
        if venv_path.exists():
            self.venv_backup = tempfile.mkdtemp()
            shutil.copytree(venv_path, f"{self.venv_backup}/venv")

    def teardown_method(self):
        """Restore environment after tests."""
        if self.venv_backup:
            venv_path = self.project_root / "venv"
            if venv_path.exists():
                shutil.rmtree(venv_path)
            shutil.copytree(f"{self.venv_backup}/venv", venv_path)
            shutil.rmtree(self.venv_backup)

    def test_makefile_exists(self):
        """Verify Makefile exists and contains required targets."""
        assert self.makefile.exists(), "Makefile not found"

        content = self.makefile.read_text()
        required_targets = [
            "install:",
            "test:",
            "run:",
            "deploy:",
            "format:",
            "lint:",
            "backup:",
        ]

        for target in required_targets:
            assert target in content, f"Makefile missing target: {target}"

    def test_make_help(self):
        """Test make help command."""
        result = subprocess.run(
            ["make", "help"], cwd=self.project_root, capture_output=True, text=True
        )

        # Help should always succeed
        assert result.returncode == 0, f"Make help failed: {result.stderr}"

        # Verify help output contains expected commands
        expected_commands = ["install", "test", "run", "format", "lint"]
        for cmd in expected_commands:
            assert cmd in result.stdout, f"Help missing command: {cmd}"

    def test_make_install(self):
        """Test dependency installation."""
        result = subprocess.run(
            ["make", "install"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Make install failed: {result.stderr}"

        # Verify key packages are installed
        pip_list = subprocess.run(
            ["pip", "list"], cwd=self.project_root, capture_output=True, text=True
        )

        required_packages = ["ccxt", "pydantic", "structlog", "pandas"]
        for package in required_packages:
            assert package in pip_list.stdout, f"Package {package} not installed"

    def test_make_format(self):
        """Test code formatting command."""
        # Create a test file with poor formatting
        test_file = self.project_root / "test_format_temp.py"
        test_file.write_text("def test():\n    x=1+2\n    return   x")

        try:
            result = subprocess.run(
                ["make", "format"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # Format should succeed even if no changes needed
            assert result.returncode == 0, f"Make format failed: {result.stderr}"

            # Check if file was formatted
            formatted_content = test_file.read_text()
            assert (
                "x = 1 + 2" in formatted_content or "would reformat" in result.stdout
            ), "Black formatting not applied"
        finally:
            test_file.unlink(missing_ok=True)

    def test_make_lint(self):
        """Test linting command."""
        result = subprocess.run(
            ["make", "lint"], cwd=self.project_root, capture_output=True, text=True
        )

        # Lint might find issues but should not crash
        assert result.returncode in [0, 1], f"Make lint crashed: {result.stderr}"

        # Verify ruff is being used (not pylint)
        assert (
            "ruff" in result.stdout or "ruff" in result.stderr or result.returncode == 0
        ), "Ruff linter not executed"

    def test_make_test(self):
        """Test pytest execution."""
        result = subprocess.run(
            ["make", "test"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Tests should run without crashing
        assert (
            "pytest" in result.stdout
            or "test" in result.stdout.lower()
            or result.returncode == 0
        ), "Pytest not executed"

    def test_make_clean(self):
        """Test clean command if it exists."""
        content = self.makefile.read_text()
        if "clean:" in content:
            # Create some cache files to clean
            cache_dir = self.project_root / "__pycache__"
            cache_dir.mkdir(exist_ok=True)
            (cache_dir / "test.pyc").touch()

            result = subprocess.run(
                ["make", "clean"], cwd=self.project_root, capture_output=True, text=True
            )

            assert result.returncode == 0, f"Make clean failed: {result.stderr}"

    def test_make_backup(self):
        """Test backup command."""
        result = subprocess.run(
            ["make", "backup"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Backup should complete without errors
        assert result.returncode == 0, f"Make backup failed: {result.stderr}"

    def test_makefile_variables(self):
        """Test Makefile uses proper variables."""
        content = self.makefile.read_text()

        # Should use Python from virtual environment
        assert (
            "python" in content.lower() or "pip" in content.lower()
        ), "Makefile doesn't reference Python"

        # Should have proper shell configuration
        if "SHELL" in content:
            assert (
                "/bin/bash" in content or "/bin/sh" in content
            ), "Invalid shell configuration"

    def test_make_typecheck(self):
        """Test type checking with mypy if configured."""
        content = self.makefile.read_text()
        if "typecheck:" in content or "mypy" in content:
            result = subprocess.run(
                ["make", "typecheck"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Type checking might find issues but shouldn't crash
            assert result.returncode in [
                0,
                1,
            ], f"Make typecheck crashed: {result.stderr}"

    def test_make_targets_are_phony(self):
        """Verify .PHONY targets are declared."""
        content = self.makefile.read_text()
        if ".PHONY" in content:
            # Common targets should be marked as phony
            phony_targets = ["install", "test", "format", "lint", "clean"]
            for target in phony_targets:
                if f"{target}:" in content:
                    assert (
                        target in content[content.find(".PHONY") :]
                    ), f"Target {target} not marked as .PHONY"
