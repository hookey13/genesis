"""Integration tests for automated dependency update system."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, call
import json
import subprocess

from genesis.operations.dependency_updater import (
    DependencyUpdater,
    DependencyVersion,
    SecurityScan,
    UpdateStrategy,
    RollbackManager,
    TestEnvironment
)


class TestDependencyVersion:
    """Test dependency version management."""
    
    def test_version_parsing(self):
        """Test parsing of version strings."""
        version = DependencyVersion("numpy==1.26.3")
        
        assert version.package == "numpy"
        assert version.version == "1.26.3"
        assert version.major == 1
        assert version.minor == 26
        assert version.patch == 3
    
    def test_version_comparison(self):
        """Test comparing dependency versions."""
        v1 = DependencyVersion("pandas==2.1.0")
        v2 = DependencyVersion("pandas==2.2.0")
        v3 = DependencyVersion("pandas==2.1.0")
        
        assert v2 > v1
        assert v1 < v2
        assert v1 == v3
        assert v1 != v2
    
    def test_version_constraints(self):
        """Test version constraint checking."""
        version = DependencyVersion("ccxt>=4.0.0,<5.0.0")
        
        assert version.satisfies("4.2.25")
        assert not version.satisfies("3.9.9")
        assert not version.satisfies("5.0.0")
    
    def test_semantic_versioning(self):
        """Test semantic version bumps."""
        version = DependencyVersion("structlog==24.1.0")
        
        # Patch bump
        patch_bump = version.bump_patch()
        assert patch_bump.version == "24.1.1"
        
        # Minor bump
        minor_bump = version.bump_minor()
        assert minor_bump.version == "24.2.0"
        
        # Major bump (should be avoided in automation)
        major_bump = version.bump_major()
        assert major_bump.version == "25.0.0"


class TestSecurityScan:
    """Test security vulnerability scanning."""
    
    @pytest.mark.asyncio
    async def test_safety_scan(self):
        """Test scanning with safety tool."""
        scanner = SecurityScan()
        
        with patch('subprocess.run') as mock_run:
            # Mock safety output with vulnerability
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout=json.dumps([{
                    "package": "requests",
                    "installed_version": "2.25.0",
                    "vulnerability": "CVE-2023-32681",
                    "severity": "high"
                }])
            )
            
            vulnerabilities = await scanner.scan_with_safety("requirements.txt")
            
            assert len(vulnerabilities) == 1
            assert vulnerabilities[0]["package"] == "requests"
            assert vulnerabilities[0]["severity"] == "high"
    
    @pytest.mark.asyncio
    async def test_bandit_scan(self):
        """Test scanning code with bandit."""
        scanner = SecurityScan()
        
        with patch('subprocess.run') as mock_run:
            # Mock bandit output
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "results": [{
                        "test_id": "B101",
                        "severity": "LOW",
                        "confidence": "HIGH",
                        "filename": "test.py",
                        "line_number": 10
                    }]
                })
            )
            
            issues = await scanner.scan_with_bandit("genesis/")
            
            assert len(issues) == 1
            assert issues[0]["severity"] == "LOW"
    
    @pytest.mark.asyncio
    async def test_severity_filtering(self):
        """Test filtering vulnerabilities by severity."""
        scanner = SecurityScan()
        
        vulnerabilities = [
            {"severity": "critical", "package": "pkg1"},
            {"severity": "high", "package": "pkg2"},
            {"severity": "medium", "package": "pkg3"},
            {"severity": "low", "package": "pkg4"}
        ]
        
        critical_only = scanner.filter_by_severity(vulnerabilities, "critical")
        assert len(critical_only) == 1
        
        high_and_above = scanner.filter_by_severity(vulnerabilities, ["critical", "high"])
        assert len(high_and_above) == 2


class TestUpdateStrategy:
    """Test different update strategies."""
    
    def test_conservative_strategy(self):
        """Test conservative update strategy (patch only)."""
        strategy = UpdateStrategy.CONSERVATIVE
        
        current = DependencyVersion("numpy==1.26.0")
        available = [
            DependencyVersion("numpy==1.26.1"),  # Patch
            DependencyVersion("numpy==1.27.0"),  # Minor
            DependencyVersion("numpy==2.0.0")    # Major
        ]
        
        selected = strategy.select_version(current, available)
        assert selected.version == "1.26.1"  # Only patch update
    
    def test_moderate_strategy(self):
        """Test moderate update strategy (minor and patch)."""
        strategy = UpdateStrategy.MODERATE
        
        current = DependencyVersion("pandas==2.1.0")
        available = [
            DependencyVersion("pandas==2.1.4"),  # Patch
            DependencyVersion("pandas==2.2.0"),  # Minor
            DependencyVersion("pandas==3.0.0")   # Major
        ]
        
        selected = strategy.select_version(current, available)
        assert selected.version == "2.2.0"  # Minor update acceptable
    
    def test_aggressive_strategy(self):
        """Test aggressive update strategy (all updates)."""
        strategy = UpdateStrategy.AGGRESSIVE
        
        current = DependencyVersion("aiohttp==3.8.0")
        available = [
            DependencyVersion("aiohttp==3.8.6"),  # Patch
            DependencyVersion("aiohttp==3.9.3"),  # Minor
            DependencyVersion("aiohttp==4.0.0")   # Major (hypothetical)
        ]
        
        selected = strategy.select_version(current, available)
        # Should select latest stable, but avoid pre-release
        assert selected.version in ["3.9.3", "4.0.0"]


class TestTestEnvironment:
    """Test staging environment for update testing."""
    
    @pytest.mark.asyncio
    async def test_environment_creation(self):
        """Test creating isolated test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = TestEnvironment(base_path=tmpdir)
            
            await env.create()
            
            assert env.venv_path.exists()
            assert (env.venv_path / "bin" / "python").exists()
            assert env.is_active
    
    @pytest.mark.asyncio
    async def test_dependency_installation(self):
        """Test installing dependencies in test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = TestEnvironment(base_path=tmpdir)
            await env.create()
            
            requirements = ["pytest==8.0.0", "black==24.1.1"]
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                
                success = await env.install_dependencies(requirements)
                assert success
                
                # Verify pip install was called
                mock_run.assert_called()
                call_args = mock_run.call_args[0][0]
                assert "pip" in call_args
                assert "install" in call_args
    
    @pytest.mark.asyncio
    async def test_test_execution(self):
        """Test running tests in isolated environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = TestEnvironment(base_path=tmpdir)
            await env.create()
            
            with patch('subprocess.run') as mock_run:
                # Mock successful test run
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="All tests passed"
                )
                
                result = await env.run_tests()
                assert result.success
                assert "All tests passed" in result.output
    
    @pytest.mark.asyncio
    async def test_environment_cleanup(self):
        """Test cleaning up test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = TestEnvironment(base_path=tmpdir)
            await env.create()
            
            assert env.venv_path.exists()
            
            await env.cleanup()
            
            assert not env.venv_path.exists()
            assert not env.is_active


class TestRollbackManager:
    """Test rollback functionality for failed updates."""
    
    @pytest.mark.asyncio
    async def test_backup_creation(self):
        """Test creating backups before updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=tmpdir)
            
            # Create fake requirements file
            req_path = Path(tmpdir) / "requirements.txt"
            req_path.write_text("numpy==1.26.3\npandas==2.2.0\n")
            
            backup_id = await manager.create_backup(req_path)
            
            assert backup_id is not None
            backup_path = Path(tmpdir) / f"backup_{backup_id}.txt"
            assert backup_path.exists()
            assert backup_path.read_text() == req_path.read_text()
    
    @pytest.mark.asyncio
    async def test_rollback_execution(self):
        """Test rolling back to previous version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=tmpdir)
            
            # Original requirements
            req_path = Path(tmpdir) / "requirements.txt"
            original_content = "numpy==1.26.3\npandas==2.2.0\n"
            req_path.write_text(original_content)
            
            # Create backup
            backup_id = await manager.create_backup(req_path)
            
            # Modify requirements (simulate failed update)
            req_path.write_text("numpy==1.27.0\npandas==2.3.0\n")
            
            # Rollback
            success = await manager.rollback(backup_id, req_path)
            assert success
            assert req_path.read_text() == original_content
    
    @pytest.mark.asyncio
    async def test_rollback_history(self):
        """Test maintaining rollback history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RollbackManager(backup_dir=tmpdir)
            
            # Create multiple backups
            backup_ids = []
            for i in range(5):
                req_path = Path(tmpdir) / "requirements.txt"
                req_path.write_text(f"version=={i}.0.0\n")
                backup_id = await manager.create_backup(req_path)
                backup_ids.append(backup_id)
                await asyncio.sleep(0.1)  # Ensure different timestamps
            
            history = await manager.get_history()
            assert len(history) == 5
            
            # History should be in chronological order
            for i in range(1, len(history)):
                assert history[i].timestamp > history[i-1].timestamp


class TestDependencyUpdater:
    """Test main dependency updater system."""
    
    @pytest.mark.asyncio
    async def test_updater_initialization(self):
        """Test dependency updater initialization."""
        with patch('genesis.operations.dependency_updater.load_config'):
            updater = DependencyUpdater()
            
            assert updater.strategy == UpdateStrategy.MODERATE
            assert updater.auto_update_enabled is False  # Safe default
            assert updater.update_cycle_days == 30
            assert updater.max_rollback_attempts == 3
    
    @pytest.mark.asyncio
    async def test_check_for_updates(self):
        """Test checking for available updates."""
        updater = DependencyUpdater()
        
        with patch('subprocess.run') as mock_run:
            # Mock pip list --outdated
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps([
                    {"name": "numpy", "version": "1.26.0", "latest_version": "1.26.3"},
                    {"name": "pandas", "version": "2.1.0", "latest_version": "2.2.0"}
                ])
            )
            
            updates = await updater.check_for_updates()
            
            assert len(updates) == 2
            assert updates[0]["name"] == "numpy"
            assert updates[0]["latest_version"] == "1.26.3"
    
    @pytest.mark.asyncio
    async def test_security_check_before_update(self):
        """Test security scanning before applying updates."""
        updater = DependencyUpdater()
        
        with patch.object(updater.security_scanner, 'scan_with_safety') as mock_scan:
            mock_scan.return_value = [
                {"package": "requests", "severity": "critical"}
            ]
            
            # Should block update with critical vulnerability
            can_proceed = await updater.check_security_before_update(["requests==2.31.0"])
            assert not can_proceed
            
            # Should allow if no critical issues
            mock_scan.return_value = []
            can_proceed = await updater.check_security_before_update(["requests==2.31.0"])
            assert can_proceed
    
    @pytest.mark.asyncio
    async def test_staged_update_process(self):
        """Test full staged update process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = DependencyUpdater()
            
            # Setup requirements file
            req_path = Path(tmpdir) / "requirements.txt"
            req_path.write_text("numpy==1.26.0\n")
            
            with patch.object(updater, 'check_for_updates') as mock_check:
                mock_check.return_value = [
                    {"name": "numpy", "version": "1.26.0", "latest_version": "1.26.3"}
                ]
                
                with patch.object(updater.test_env, 'run_tests') as mock_tests:
                    mock_tests.return_value = MagicMock(success=True)
                    
                    # Run staged update
                    result = await updater.run_staged_update(req_path)
                    
                    assert result.success
                    assert result.updated_packages == ["numpy"]
                    assert result.test_passed
    
    @pytest.mark.asyncio
    async def test_automatic_rollback_on_failure(self):
        """Test automatic rollback when tests fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            updater = DependencyUpdater()
            
            req_path = Path(tmpdir) / "requirements.txt"
            original = "numpy==1.26.0\n"
            req_path.write_text(original)
            
            with patch.object(updater.test_env, 'run_tests') as mock_tests:
                mock_tests.return_value = MagicMock(success=False, output="Tests failed")
                
                with patch.object(updater.rollback_manager, 'rollback') as mock_rollback:
                    mock_rollback.return_value = True
                    
                    result = await updater.run_staged_update(req_path)
                    
                    assert not result.success
                    assert result.rollback_performed
                    mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_scheduling(self):
        """Test scheduled update cycles."""
        updater = DependencyUpdater()
        updater.update_cycle_days = 0.00001  # Very short for testing
        
        with patch.object(updater, 'run_update_cycle') as mock_update:
            with patch('asyncio.sleep', side_effect=asyncio.CancelledError):
                with pytest.raises(asyncio.CancelledError):
                    await updater.start_scheduled_updates()
                
                mock_update.assert_called()
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self):
        """Test pip-tools dependency resolution."""
        updater = DependencyUpdater()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create requirements.in file
            req_in = Path(tmpdir) / "requirements.in"
            req_in.write_text("numpy\npandas\n")
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="numpy==1.26.3\npandas==2.2.0\n"
                )
                
                resolved = await updater.resolve_dependencies(req_in)
                
                assert "numpy==1.26.3" in resolved
                assert "pandas==2.2.0" in resolved
                
                # Verify pip-compile was called
                call_args = mock_run.call_args[0][0]
                assert "pip-compile" in call_args
    
    @pytest.mark.asyncio
    async def test_update_notification(self):
        """Test notifications for update status."""
        updater = DependencyUpdater()
        
        with patch('genesis.operations.dependency_updater.send_notification') as mock_notify:
            # Successful update
            await updater.notify_update_status(
                success=True,
                updated=["numpy", "pandas"],
                failed=[]
            )
            
            mock_notify.assert_called_once()
            notification = mock_notify.call_args[0][0]
            assert "success" in notification.lower()
            
            # Failed update
            mock_notify.reset_mock()
            await updater.notify_update_status(
                success=False,
                updated=[],
                failed=["requests"],
                error="Security vulnerability detected"
            )
            
            mock_notify.assert_called_once()
            notification = mock_notify.call_args[0][0]
            assert "failed" in notification.lower()
            assert "security" in notification.lower()


class TestDependencyIntegration:
    """Integration tests for complete dependency update workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_update_workflow(self):
        """Test complete update workflow from check to deployment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            req_path = Path(tmpdir) / "requirements.txt"
            req_path.write_text("numpy==1.26.0\npandas==2.1.0\nrequests==2.28.0\n")
            
            updater = DependencyUpdater()
            
            # Mock external calls
            with patch('subprocess.run') as mock_run:
                # Mock pip list --outdated
                mock_run.side_effect = [
                    MagicMock(  # First call: check for updates
                        returncode=0,
                        stdout=json.dumps([
                            {"name": "numpy", "version": "1.26.0", "latest_version": "1.26.3"},
                            {"name": "pandas", "version": "2.1.0", "latest_version": "2.2.0"}
                        ])
                    ),
                    MagicMock(  # Security scan
                        returncode=0,
                        stdout="[]"  # No vulnerabilities
                    ),
                    MagicMock(  # Test execution
                        returncode=0,
                        stdout="All tests passed"
                    )
                ]
                
                # Phase 1: Check for updates
                updates = await updater.check_for_updates()
                assert len(updates) == 2
                
                # Phase 2: Security validation
                can_proceed = await updater.check_security_before_update(
                    ["numpy==1.26.3", "pandas==2.2.0"]
                )
                assert can_proceed
                
                # Phase 3: Staged testing
                with patch.object(updater.test_env, 'create') as mock_create:
                    with patch.object(updater.test_env, 'install_dependencies') as mock_install:
                        mock_create.return_value = None
                        mock_install.return_value = True
                        
                        result = await updater.run_staged_update(req_path)
                        
                        # Verify success
                        assert result.success
                        assert "numpy" in result.updated_packages
                        assert "pandas" in result.updated_packages
                        assert result.test_passed
                        assert not result.rollback_performed