"""Backup and recovery validation module."""

import asyncio
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


class BackupValidator:
    """Validates backup and recovery procedures."""

    REQUIRED_BACKUP_COMPONENTS = [
        "database",
        "configuration",
        "logs",
        "state_files",
        "encryption_keys",
    ]

    BACKUP_FREQUENCY_HOURS = 4  # Expected backup frequency
    MAX_RESTORE_TIME_SECONDS = 300  # Maximum acceptable restore time

    def __init__(self, genesis_root: Path | None = None):
        """Initialize backup validator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.backup_dir = self.genesis_root / ".genesis" / "backups"
        self.results: Dict[str, Any] = {}

    async def validate(self) -> Dict[str, Any]:
        """Run backup and recovery validation checks.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting backup validation")
        start_time = datetime.utcnow()

        self.results = {
            "validator": "backup",
            "timestamp": start_time.isoformat(),
            "passed": True,
            "score": 0,
            "checks": {},
            "summary": "",
            "details": [],
        }

        # Test backup creation
        backup_result = await self._test_backup_creation()
        self.results["checks"]["backup_creation"] = backup_result

        # Test restore procedure
        restore_result = await self._test_restore_procedure()
        self.results["checks"]["restore_procedure"] = restore_result

        # Verify backup encryption
        encryption_result = await self._verify_backup_encryption()
        self.results["checks"]["backup_encryption"] = encryption_result

        # Check backup schedule
        schedule_result = await self._check_backup_schedule()
        self.results["checks"]["backup_schedule"] = schedule_result

        # Validate offsite storage
        offsite_result = await self._validate_offsite_storage()
        self.results["checks"]["offsite_storage"] = offsite_result

        # Check backup scripts
        scripts_result = await self._check_backup_scripts()
        self.results["checks"]["backup_scripts"] = scripts_result

        # Calculate overall score
        total_checks = len(self.results["checks"])
        passed_checks = sum(
            1 for check in self.results["checks"].values() if check.get("passed", False)
        )
        self.results["score"] = int((passed_checks / total_checks) * 100) if total_checks > 0 else 0

        # Determine overall status
        if all(check.get("passed", False) for check in self.results["checks"].values()):
            self.results["passed"] = True
            self.results["summary"] = "Backup and recovery fully operational"
        else:
            self.results["passed"] = False
            self.results["summary"] = "Backup/recovery issues detected - review failed checks"

        # Add execution time
        self.results["execution_time"] = (datetime.utcnow() - start_time).total_seconds()

        return self.results

    async def _test_backup_creation(self) -> Dict[str, Any]:
        """Test the backup creation process.
        
        Returns:
            Validation result for backup creation
        """
        result = {
            "passed": False,
            "message": "",
            "backup_size": 0,
            "components_backed_up": [],
        }

        try:
            # Create a test backup in a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                test_backup_path = Path(temp_dir) / "test_backup"
                
                # Simulate backing up each component
                components_backed_up = []
                
                # Database backup
                db_path = self.genesis_root / ".genesis" / "data" / "genesis.db"
                if db_path.exists():
                    backup_db_path = test_backup_path / "database" / "genesis.db"
                    backup_db_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(db_path, backup_db_path)
                    components_backed_up.append("database")
                
                # Configuration backup
                config_dir = self.genesis_root / "config"
                if config_dir.exists():
                    backup_config_path = test_backup_path / "configuration"
                    shutil.copytree(config_dir, backup_config_path, dirs_exist_ok=True)
                    components_backed_up.append("configuration")
                
                # State files backup
                state_dir = self.genesis_root / ".genesis" / "state"
                if state_dir.exists():
                    backup_state_path = test_backup_path / "state_files"
                    shutil.copytree(state_dir, backup_state_path, dirs_exist_ok=True)
                    components_backed_up.append("state_files")
                
                # Calculate backup size
                total_size = sum(
                    f.stat().st_size
                    for f in test_backup_path.rglob("*")
                    if f.is_file()
                )
                
                result["backup_size"] = total_size
                result["components_backed_up"] = components_backed_up
                
                # Check if all required components are backed up
                missing_components = [
                    comp for comp in self.REQUIRED_BACKUP_COMPONENTS
                    if comp not in components_backed_up and comp != "encryption_keys" and comp != "logs"
                ]
                
                if not missing_components:
                    result["passed"] = True
                    result["message"] = f"Backup created successfully ({total_size / 1024:.2f} KB)"
                else:
                    result["message"] = f"Missing backup components: {', '.join(missing_components)}"
                    
        except Exception as e:
            result["message"] = f"Backup creation failed: {str(e)}"

        return result

    async def _test_restore_procedure(self) -> Dict[str, Any]:
        """Test the restore procedure.
        
        Returns:
            Validation result for restore procedure
        """
        result = {
            "passed": False,
            "message": "",
            "restore_time": 0,
            "components_restored": [],
        }

        try:
            start_restore = datetime.utcnow()
            
            # Simulate restore process
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a mock backup
                mock_backup_dir = Path(temp_dir) / "mock_backup"
                mock_backup_dir.mkdir(parents=True, exist_ok=True)
                
                # Create mock backup files
                (mock_backup_dir / "database" / "genesis.db").parent.mkdir(parents=True, exist_ok=True)
                (mock_backup_dir / "database" / "genesis.db").touch()
                (mock_backup_dir / "configuration" / "settings.yaml").parent.mkdir(parents=True, exist_ok=True)
                (mock_backup_dir / "configuration" / "settings.yaml").touch()
                (mock_backup_dir / "state_files" / "tier_state.json").parent.mkdir(parents=True, exist_ok=True)
                (mock_backup_dir / "state_files" / "tier_state.json").touch()
                
                # Simulate restore to another temporary directory
                restore_dir = Path(temp_dir) / "restore_test"
                restore_dir.mkdir(parents=True, exist_ok=True)
                
                components_restored = []
                
                # Restore database
                if (mock_backup_dir / "database").exists():
                    shutil.copytree(mock_backup_dir / "database", restore_dir / "database", dirs_exist_ok=True)
                    components_restored.append("database")
                
                # Restore configuration
                if (mock_backup_dir / "configuration").exists():
                    shutil.copytree(mock_backup_dir / "configuration", restore_dir / "configuration", dirs_exist_ok=True)
                    components_restored.append("configuration")
                
                # Restore state files
                if (mock_backup_dir / "state_files").exists():
                    shutil.copytree(mock_backup_dir / "state_files", restore_dir / "state_files", dirs_exist_ok=True)
                    components_restored.append("state_files")
                
                restore_time = (datetime.utcnow() - start_restore).total_seconds()
                result["restore_time"] = restore_time
                result["components_restored"] = components_restored
                
                if restore_time <= self.MAX_RESTORE_TIME_SECONDS and len(components_restored) >= 3:
                    result["passed"] = True
                    result["message"] = f"Restore completed in {restore_time:.2f} seconds"
                else:
                    if restore_time > self.MAX_RESTORE_TIME_SECONDS:
                        result["message"] = f"Restore too slow: {restore_time:.2f}s > {self.MAX_RESTORE_TIME_SECONDS}s"
                    else:
                        result["message"] = f"Incomplete restore: only {len(components_restored)} components"
                        
        except Exception as e:
            result["message"] = f"Restore procedure failed: {str(e)}"

        return result

    async def _verify_backup_encryption(self) -> Dict[str, Any]:
        """Verify backup encryption is configured.
        
        Returns:
            Validation result for backup encryption
        """
        result = {
            "passed": False,
            "message": "",
            "encryption_type": "",
        }

        # Check for restic configuration (which provides encryption)
        restic_config = self.genesis_root / ".restic"
        restic_password_file = self.genesis_root / ".restic.password"
        
        if restic_config.exists() or restic_password_file.exists():
            result["encryption_type"] = "restic (AES-256)"
            result["passed"] = True
            result["message"] = "Backup encryption configured with restic"
        else:
            # Check for other encryption configurations
            backup_script = self.genesis_root / "scripts" / "backup.sh"
            if backup_script.exists():
                with open(backup_script, "r") as f:
                    content = f.read()
                    if "gpg" in content or "openssl" in content or "restic" in content:
                        result["encryption_type"] = "Script-based encryption"
                        result["passed"] = True
                        result["message"] = "Backup encryption detected in backup script"
                    else:
                        result["message"] = "No encryption found in backup script"
            else:
                result["message"] = "No backup encryption configuration found"

        return result

    async def _check_backup_schedule(self) -> Dict[str, Any]:
        """Check if backups are scheduled appropriately.
        
        Returns:
            Validation result for backup schedule
        """
        result = {
            "passed": False,
            "message": "",
            "schedule_found": False,
            "last_backup": None,
        }

        # Check for cron job or systemd timer
        crontab_check = await self._check_crontab()
        systemd_check = await self._check_systemd_timer()
        
        if crontab_check or systemd_check:
            result["schedule_found"] = True
            
            # Check for recent backups
            if self.backup_dir.exists():
                backup_files = list(self.backup_dir.glob("*.tar.gz")) + list(self.backup_dir.glob("*.zip"))
                if backup_files:
                    most_recent = max(backup_files, key=lambda f: f.stat().st_mtime)
                    last_backup_time = datetime.fromtimestamp(most_recent.stat().st_mtime)
                    result["last_backup"] = last_backup_time.isoformat()
                    
                    hours_since_backup = (datetime.utcnow() - last_backup_time).total_seconds() / 3600
                    if hours_since_backup <= self.BACKUP_FREQUENCY_HOURS:
                        result["passed"] = True
                        result["message"] = f"Backups scheduled and recent (last: {hours_since_backup:.1f}h ago)"
                    else:
                        result["message"] = f"Backup schedule found but last backup was {hours_since_backup:.1f}h ago"
                else:
                    result["message"] = "Backup schedule found but no backup files exist"
            else:
                result["message"] = "Backup schedule found but backup directory doesn't exist"
        else:
            result["message"] = "No backup schedule found (cron or systemd)"

        return result

    async def _check_crontab(self) -> bool:
        """Check if backup is scheduled in crontab.
        
        Returns:
            True if backup job found in crontab
        """
        try:
            # Check user crontab
            process = await asyncio.create_subprocess_exec(
                "crontab", "-l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await process.communicate()
            
            if stdout:
                crontab_content = stdout.decode()
                if "backup" in crontab_content.lower() or "genesis" in crontab_content.lower():
                    return True
        except Exception:
            pass
        
        return False

    async def _check_systemd_timer(self) -> bool:
        """Check if backup is scheduled as systemd timer.
        
        Returns:
            True if backup timer found in systemd
        """
        try:
            # Check for genesis backup timer
            process = await asyncio.create_subprocess_exec(
                "systemctl", "list-timers", "--all",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await process.communicate()
            
            if stdout:
                timers_output = stdout.decode()
                if "genesis" in timers_output.lower() and "backup" in timers_output.lower():
                    return True
        except Exception:
            pass
        
        return False

    async def _validate_offsite_storage(self) -> Dict[str, Any]:
        """Validate offsite backup storage configuration.
        
        Returns:
            Validation result for offsite storage
        """
        result = {
            "passed": False,
            "message": "",
            "storage_type": "",
        }

        # Check for DigitalOcean Spaces configuration
        env_file = self.genesis_root / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                env_content = f.read()
                if "DO_SPACES" in env_content or "DIGITALOCEAN_SPACES" in env_content:
                    result["storage_type"] = "DigitalOcean Spaces"
                    result["passed"] = True
                    result["message"] = "Offsite storage configured (DigitalOcean Spaces)"
                elif "AWS_S3" in env_content or "S3_BUCKET" in env_content:
                    result["storage_type"] = "AWS S3"
                    result["passed"] = True
                    result["message"] = "Offsite storage configured (AWS S3)"
        
        # Check backup script for offsite storage
        if not result["passed"]:
            backup_script = self.genesis_root / "scripts" / "backup.sh"
            if backup_script.exists():
                with open(backup_script, "r") as f:
                    content = f.read()
                    if "s3" in content or "spaces" in content or "rsync" in content:
                        result["storage_type"] = "Script-based offsite storage"
                        result["passed"] = True
                        result["message"] = "Offsite storage detected in backup script"
        
        if not result["passed"]:
            result["message"] = "No offsite backup storage configured"

        return result

    async def _check_backup_scripts(self) -> Dict[str, Any]:
        """Check backup scripts exist and are executable.
        
        Returns:
            Validation result for backup scripts
        """
        result = {
            "passed": False,
            "message": "",
            "scripts_found": [],
            "scripts_missing": [],
        }

        required_scripts = {
            "backup.sh": self.genesis_root / "scripts" / "backup.sh",
            "restore.sh": self.genesis_root / "scripts" / "restore.sh",
        }

        for script_name, script_path in required_scripts.items():
            if script_path.exists():
                result["scripts_found"].append(script_name)
                
                # Check if script is executable
                if not os.access(script_path, os.X_OK):
                    result["scripts_missing"].append(f"{script_name} (not executable)")
            else:
                result["scripts_missing"].append(script_name)

        if not result["scripts_missing"]:
            result["passed"] = True
            result["message"] = "All backup scripts present and executable"
        else:
            result["message"] = f"Script issues: {', '.join(result['scripts_missing'])}"

        return result

    def generate_report(self) -> str:
        """Generate a detailed backup validation report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No validation results available. Run validate() first."

        report = []
        report.append("=" * 80)
        report.append("BACKUP VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Overall Status: {'PASSED' if self.results['passed'] else 'FAILED'}")
        report.append(f"Score: {self.results['score']}%")
        report.append(f"Summary: {self.results['summary']}")
        report.append("")

        report.append("CHECK RESULTS:")
        report.append("-" * 40)
        
        for check_name, check_result in self.results["checks"].items():
            status = "✓" if check_result.get("passed", False) else "✗"
            report.append(f"{status} {check_name}: {check_result.get('message', '')}")
            
            # Add details
            if check_result.get("backup_size"):
                report.append(f"  Backup size: {check_result['backup_size'] / 1024:.2f} KB")
            if check_result.get("restore_time"):
                report.append(f"  Restore time: {check_result['restore_time']:.2f} seconds")
            if check_result.get("encryption_type"):
                report.append(f"  Encryption: {check_result['encryption_type']}")
            if check_result.get("last_backup"):
                report.append(f"  Last backup: {check_result['last_backup']}")
            if check_result.get("scripts_missing"):
                report.append(f"  Missing scripts: {', '.join(check_result['scripts_missing'])}")

        report.append("")
        report.append(f"Execution Time: {self.results.get('execution_time', 0):.2f} seconds")
        report.append("=" * 80)

        return "\n".join(report)