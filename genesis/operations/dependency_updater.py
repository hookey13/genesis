"""Automated Dependency Update System."""

import asyncio
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from genesis.utils.logger import get_logger, LoggerType


class DependencyConfig(BaseModel):
    """Configuration for dependency updates."""
    
    requirements_files: List[str] = Field(
        default_factory=lambda: ["requirements/base.txt"],
        description="Requirements files to update"
    )
    update_interval_days: int = Field(30, description="Days between update cycles")
    enable_security_scan: bool = Field(True, description="Enable security scanning")
    enable_staging_test: bool = Field(True, description="Test in staging first")
    auto_rollback: bool = Field(True, description="Rollback on test failure")
    test_command: str = Field("pytest", description="Test command to validate updates")


class DependencyUpdater:
    """Manages automated dependency updates with testing."""
    
    def __init__(self, config: DependencyConfig):
        self.config = config
        self.logger = get_logger(__name__, LoggerType.SYSTEM)
    
    async def check_outdated_packages(self) -> List[Dict[str, str]]:
        """Check for outdated packages."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                import json
                outdated = json.loads(result.stdout)
                
                self.logger.info(
                    "outdated_packages_found",
                    count=len(outdated)
                )
                
                return outdated
            
            return []
            
        except Exception as e:
            self.logger.error("check_outdated_failed", error=str(e))
            return []
    
    async def security_scan(self) -> List[Dict[str, Any]]:
        """Run security scan on dependencies."""
        if not self.config.enable_security_scan:
            return []
        
        vulnerabilities = []
        
        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                import json
                vulnerabilities = json.loads(result.stdout)
                
                if vulnerabilities:
                    self.logger.warning(
                        "security_vulnerabilities_found",
                        count=len(vulnerabilities)
                    )
            
            # Run bandit for code security
            result = subprocess.run(
                ["bandit", "-r", "genesis/", "-f", "json"],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                import json
                bandit_results = json.loads(result.stdout)
                if bandit_results.get("results"):
                    self.logger.warning(
                        "code_security_issues_found",
                        count=len(bandit_results["results"])
                    )
            
        except Exception as e:
            self.logger.error("security_scan_failed", error=str(e))
        
        return vulnerabilities
    
    async def update_dependencies(self, staging_dir: Optional[Path] = None) -> bool:
        """Update dependencies with pip-tools."""
        try:
            # Use pip-tools for dependency resolution
            for req_file in self.config.requirements_files:
                # Compile new requirements
                result = subprocess.run(
                    ["pip-compile", "--upgrade", req_file],
                    capture_output=True,
                    text=True,
                    cwd=staging_dir
                )
                
                if result.returncode != 0:
                    self.logger.error(
                        "pip_compile_failed",
                        file=req_file,
                        error=result.stderr
                    )
                    return False
            
            # Sync dependencies
            result = subprocess.run(
                ["pip-sync"] + self.config.requirements_files,
                capture_output=True,
                text=True,
                cwd=staging_dir
            )
            
            if result.returncode == 0:
                self.logger.info("dependencies_updated")
                return True
            else:
                self.logger.error(
                    "pip_sync_failed",
                    error=result.stderr
                )
                return False
                
        except Exception as e:
            self.logger.error("update_failed", error=str(e))
            return False
    
    async def test_updates(self, staging_dir: Path) -> bool:
        """Test updated dependencies."""
        try:
            result = subprocess.run(
                self.config.test_command.split(),
                capture_output=True,
                text=True,
                cwd=staging_dir
            )
            
            if result.returncode == 0:
                self.logger.info("tests_passed")
                return True
            else:
                self.logger.error(
                    "tests_failed",
                    error=result.stdout + result.stderr
                )
                return False
                
        except Exception as e:
            self.logger.error("test_execution_failed", error=str(e))
            return False
    
    async def create_staging_environment(self) -> Path:
        """Create staging environment for testing."""
        staging_dir = Path(tempfile.mkdtemp(prefix="genesis_staging_"))
        
        # Copy project to staging
        import shutil
        shutil.copytree(".", staging_dir, ignore=shutil.ignore_patterns(
            "*.pyc", "__pycache__", ".git", ".pytest_cache", "venv", ".venv"
        ))
        
        self.logger.info("staging_environment_created", path=str(staging_dir))
        
        return staging_dir
    
    async def rollback_updates(self) -> None:
        """Rollback to previous dependency versions."""
        try:
            # Git restore requirements files
            for req_file in self.config.requirements_files:
                subprocess.run(
                    ["git", "checkout", "--", req_file],
                    check=True
                )
            
            # Reinstall dependencies
            subprocess.run(
                ["pip-sync"] + self.config.requirements_files,
                check=True
            )
            
            self.logger.info("dependencies_rolled_back")
            
        except Exception as e:
            self.logger.error("rollback_failed", error=str(e))
    
    async def update_cycle(self) -> bool:
        """Run complete update cycle."""
        self.logger.info("update_cycle_started")
        
        # Check for outdated packages
        outdated = await self.check_outdated_packages()
        if not outdated:
            self.logger.info("no_updates_available")
            return True
        
        # Security scan
        vulnerabilities = await self.security_scan()
        
        if self.config.enable_staging_test:
            # Create staging environment
            staging_dir = await self.create_staging_environment()
            
            try:
                # Update in staging
                if not await self.update_dependencies(staging_dir):
                    return False
                
                # Test in staging
                if not await self.test_updates(staging_dir):
                    if self.config.auto_rollback:
                        await self.rollback_updates()
                    return False
                
                # Apply to production
                if await self.update_dependencies():
                    self.logger.info("update_cycle_completed")
                    return True
                    
            finally:
                # Cleanup staging
                import shutil
                shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            # Direct update
            if await self.update_dependencies():
                if await self.test_updates(Path(".")):
                    return True
                elif self.config.auto_rollback:
                    await self.rollback_updates()
        
        return False
    
    async def scheduled_updates(self) -> None:
        """Run scheduled dependency updates."""
        while True:
            try:
                await asyncio.sleep(self.config.update_interval_days * 86400)
                await self.update_cycle()
                
            except Exception as e:
                self.logger.error("scheduled_update_error", error=str(e))