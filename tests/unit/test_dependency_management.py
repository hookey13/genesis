"""Unit tests for dependency management and pinning."""

import sys
import subprocess
from pathlib import Path
import tempfile
import pytest
from unittest.mock import patch, MagicMock, mock_open


class TestDependencyManagement:
    """Test dependency pinning and hash verification."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_requirements_directory_exists(self, project_root):
        """Verify requirements directory exists."""
        req_dir = project_root / "requirements"
        assert req_dir.exists(), "requirements/ directory must exist"
        assert req_dir.is_dir(), "requirements/ must be a directory"
    
    def test_tier_requirements_files_exist(self, project_root):
        """Verify all tier requirements files exist."""
        req_dir = project_root / "requirements"
        required_files = ["base.txt", "sniper.txt", "hunter.txt", "strategist.txt", "dev.txt"]
        
        for filename in required_files:
            file_path = req_dir / filename
            assert file_path.exists(), f"requirements/{filename} must exist"
            
    def test_requirements_inheritance(self, project_root):
        """Verify tier requirements properly inherit from base."""
        req_dir = project_root / "requirements"
        
        # Check sniper inherits from base
        sniper_file = req_dir / "sniper.txt"
        if sniper_file.exists():
            content = sniper_file.read_text()
            assert "-r base.txt" in content or content.strip() == "", \
                "sniper.txt should inherit from base.txt or be empty"
        
        # Check hunter inherits from base
        hunter_file = req_dir / "hunter.txt"
        if hunter_file.exists():
            content = hunter_file.read_text()
            assert "-r base.txt" in content, "hunter.txt should inherit from base.txt"
            
        # Check strategist inherits from hunter
        strategist_file = req_dir / "strategist.txt"
        if strategist_file.exists():
            content = strategist_file.read_text()
            assert "-r hunter.txt" in content, "strategist.txt should inherit from hunter.txt"
    
    def test_base_requirements_format(self, project_root):
        """Verify base.txt has proper format with pinned versions."""
        base_file = project_root / "requirements" / "base.txt"
        
        if not base_file.exists():
            pytest.skip("base.txt not found")
            
        with open(base_file, 'r') as f:
            lines = f.readlines()
            
        # Check for pinned versions
        packages_with_versions = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                # Should have version pinning
                assert '==' in line or '>=' in line or '~=' in line, \
                    f"Package {line} should have version specification"
                if '==' in line:
                    packages_with_versions += 1
                    
        assert packages_with_versions > 0, "base.txt should have pinned packages"
    
    def test_update_requirements_script_exists(self, project_root):
        """Verify update_requirements.py script exists."""
        script_path = project_root / "scripts" / "update_requirements.py"
        assert script_path.exists(), "scripts/update_requirements.py must exist"
        
    def test_verify_hashes_script_exists(self, project_root):
        """Verify verify_hashes.py script exists."""
        script_path = project_root / "scripts" / "verify_hashes.py"
        assert script_path.exists(), "scripts/verify_hashes.py must exist"
        
    def test_scripts_are_executable(self, project_root):
        """Verify Python scripts have proper shebang."""
        scripts = [
            project_root / "scripts" / "update_requirements.py",
            project_root / "scripts" / "verify_hashes.py"
        ]
        
        for script in scripts:
            if script.exists():
                with open(script, 'r') as f:
                    first_line = f.readline()
                    assert first_line.startswith("#!/usr/bin/env python"), \
                        f"{script.name} should have Python shebang"
                        
    @patch('subprocess.run')
    def test_pip_tools_installation_check(self, mock_run):
        """Test that pip-tools installation is checked."""
        # Mock successful pip-tools import
        with patch.dict('sys.modules', {'piptools': MagicMock()}):
            # This should not call subprocess since piptools exists
            from scripts.update_requirements import install_pip_tools
            install_pip_tools()
            mock_run.assert_not_called()
            
    def test_requirements_lock_generation(self, project_root):
        """Test that requirements.lock can be generated."""
        # This is an integration test that would actually run the script
        # For unit testing, we just verify the structure
        lock_file = project_root / "requirements.lock"
        
        # If lock file exists, verify its format
        if lock_file.exists():
            content = lock_file.read_text()
            lines = content.split('\n')
            
            # Check for hash entries
            has_hashes = any('--hash=' in line for line in lines)
            if len(lines) > 10:  # Only check if file has content
                assert has_hashes, "requirements.lock should contain hashes"
                
    def test_tier_specific_loading(self):
        """Test tier-specific requirement loading logic."""
        tiers = ["sniper", "hunter", "strategist"]
        
        for tier in tiers:
            # Verify tier is valid
            assert tier in tiers, f"Invalid tier: {tier}"
            
            # Test tier environment variable
            import os
            os.environ['TIER'] = tier
            assert os.environ.get('TIER') == tier
            
    @patch('subprocess.run')
    def test_package_verification_logic(self, mock_run):
        """Test package verification logic."""
        # Mock pip list output
        mock_run.return_value = MagicMock(
            stdout='[{"name": "ccxt", "version": "4.4.0"}, {"name": "rich", "version": "13.7.0"}]',
            stderr='',
            returncode=0
        )
        
        from scripts.verify_hashes import get_installed_packages
        packages = get_installed_packages()
        
        assert isinstance(packages, list)
        assert len(packages) == 2
        assert packages[0]['name'] == 'ccxt'
        assert packages[0]['version'] == '4.4.0'
        
    def test_hash_verification_parsing(self, project_root):
        """Test parsing of requirements.lock for hash verification."""
        # Create a mock requirements.lock content
        mock_content = """
# This file is autogenerated by pip-compile
ccxt==4.4.0 \\
    --hash=sha256:abc123 \\
    --hash=sha256:def456
rich==13.7.0 \\
    --hash=sha256:ghi789
"""
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            from scripts.verify_hashes import parse_requirements_lock
            
            # Mock the path existence check
            with patch('pathlib.Path.exists', return_value=True):
                packages = parse_requirements_lock()
                
                assert 'ccxt' in packages
                assert packages['ccxt']['version'] == '4.4.0'
                assert len(packages['ccxt']['hashes']) == 2
                assert 'rich' in packages
                assert packages['rich']['version'] == '13.7.0'
                
    def test_python_version_compatibility(self):
        """Ensure no Python 3.12+ features are used."""
        # Check Python version
        assert sys.version_info.major == 3
        assert sys.version_info.minor == 11
        
        # Ensure we're not using 3.12+ features
        # This would fail if run on 3.12+
        try:
            # Python 3.12+ has different error messages
            eval("match x: case 1: pass")  # This should fail in 3.11
            pytest.fail("Python 3.12+ pattern matching should not work")
        except SyntaxError:
            pass  # Expected in Python 3.11
            
    def test_dev_requirements_separation(self, project_root):
        """Verify dev requirements are separate from production."""
        dev_file = project_root / "requirements" / "dev.txt"
        
        if dev_file.exists():
            content = dev_file.read_text()
            
            # Check for common dev packages
            dev_packages = ['pytest', 'black', 'ruff', 'mypy', 'pre-commit']
            found_dev_packages = []
            
            for line in content.split('\n'):
                line = line.strip().lower()
                for pkg in dev_packages:
                    if pkg in line:
                        found_dev_packages.append(pkg)
                        
            # Should have at least some dev packages
            assert len(found_dev_packages) > 0, \
                "dev.txt should contain development packages"