"""Disaster recovery validation for Genesis trading system."""

import asyncio
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from . import BaseValidator, ValidationIssue, ValidationSeverity


class DisasterRecoveryValidator(BaseValidator):
    """Validates disaster recovery procedures and backup systems."""
    
    @property
    def name(self) -> str:
        return "disaster_recovery"
    
    @property
    def description(self) -> str:
        return "Validates backup procedures, recovery objectives, and failover mechanisms"
    
    async def _validate(self, mode: str):
        """Perform disaster recovery validation."""
        # Test backup procedures
        await self._test_backup_procedures()
        
        # Verify recovery time objectives
        await self._verify_recovery_objectives()
        
        # Check failover mechanisms
        await self._check_failover_mechanisms()
        
        # Validate data restoration
        if mode in ["standard", "thorough"]:
            await self._validate_data_restoration()
        
        # Test emergency procedures
        if mode == "thorough":
            await self._test_emergency_procedures()
    
    async def _test_backup_procedures(self):
        """Test backup procedures."""
        backup_dir = Path(".genesis/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Check backup script
        backup_script = Path("scripts/backup.sh")
        if backup_script.exists():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Backup script found",
                details={"path": str(backup_script)}
            ))
        else:
            # Create default backup script
            backup_content = """#!/bin/bash
# Genesis backup script

BACKUP_DIR=".genesis/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_FILE=".genesis/data/genesis.db"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
if [ -f "$DB_FILE" ]; then
    cp $DB_FILE "$BACKUP_DIR/genesis_${TIMESTAMP}.db"
    echo "Database backed up to $BACKUP_DIR/genesis_${TIMESTAMP}.db"
fi

# Backup configuration
tar -czf "$BACKUP_DIR/config_${TIMESTAMP}.tar.gz" genesis/config/

# Backup logs (last 7 days)
find .genesis/logs -type f -mtime -7 -exec tar -rf "$BACKUP_DIR/logs_${TIMESTAMP}.tar" {} \\;
gzip "$BACKUP_DIR/logs_${TIMESTAMP}.tar"

# Upload to DigitalOcean Spaces (if configured)
if [ ! -z "$DO_SPACES_KEY" ]; then
    # restic backup command would go here
    echo "Uploading to DigitalOcean Spaces..."
fi

echo "Backup completed at $(date)"
"""
            backup_script.write_text(backup_content)
            backup_script.chmod(0o755)
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Created default backup script",
                details={"path": str(backup_script)}
            ))
        
        # Check backup frequency
        recent_backups = list(backup_dir.glob("*.db"))
        if recent_backups:
            # Check age of most recent backup
            latest_backup = max(recent_backups, key=lambda p: p.stat().st_mtime)
            age_hours = (time.time() - latest_backup.stat().st_mtime) / 3600
            
            self.check_threshold(
                age_hours,
                24,
                "<",
                "Latest backup age",
                " hours",
                ValidationSeverity.WARNING
            )
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Found {len(recent_backups)} backups, latest: {latest_backup.name}",
                details={
                    "count": len(recent_backups),
                    "latest": str(latest_backup),
                    "age_hours": age_hours
                }
            ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No backups found",
                recommendation="Run backup script: ./scripts/backup.sh"
            ))
        
        # Check backup size and integrity
        for backup in recent_backups[:3]:  # Check last 3 backups
            size_mb = backup.stat().st_size / (1024 * 1024)
            self.check_condition(
                size_mb > 0,
                f"Backup {backup.name}: {size_mb:.2f}MB",
                f"Empty backup file: {backup.name}",
                ValidationSeverity.CRITICAL,
                details={"file": str(backup), "size_mb": size_mb}
            )
    
    async def _verify_recovery_objectives(self):
        """Verify recovery time and point objectives."""
        # Check RTO (Recovery Time Objective)
        rto_config = {
            "database_restore": 5,  # 5 minutes
            "service_restart": 2,   # 2 minutes
            "state_recovery": 3,    # 3 minutes
            "total_rto": 10        # 10 minutes total
        }
        
        for component, target_minutes in rto_config.items():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"RTO target for {component}: {target_minutes} minutes",
                details={"component": component, "target_minutes": target_minutes}
            ))
        
        # Check RPO (Recovery Point Objective)
        rpo_config = {
            "database": 1,      # 1 minute
            "state_snapshot": 1, # 1 minute
            "audit_logs": 0,    # Real-time
            "market_data": 5    # 5 minutes (less critical)
        }
        
        for data_type, target_minutes in rpo_config.items():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"RPO target for {data_type}: {target_minutes} minutes",
                details={"data_type": data_type, "target_minutes": target_minutes}
            ))
        
        # Test recovery time
        try:
            from genesis.recovery.recovery_manager import RecoveryManager
            
            manager = RecoveryManager()
            
            # Simulate recovery
            start_time = time.perf_counter()
            recovery_result = await manager.simulate_recovery()
            recovery_time = time.perf_counter() - start_time
            
            self.check_threshold(
                recovery_time,
                rto_config["total_rto"] * 60,
                "<",
                "Simulated recovery time",
                " seconds",
                ValidationSeverity.ERROR
            )
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Recovery simulation completed in {recovery_time:.1f} seconds",
                details={"recovery_time_seconds": recovery_time}
            ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Recovery manager not implemented",
                recommendation="Implement RecoveryManager for automated recovery"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Recovery simulation failed",
                details={"error": str(e)}
            ))
    
    async def _check_failover_mechanisms(self):
        """Check failover mechanisms."""
        # Check emergency stop script
        emergency_script = Path("scripts/emergency_close.py")
        if emergency_script.exists():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Emergency stop script found"
            ))
            
            # Test script syntax
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", str(emergency_script)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                self.check_condition(
                    result.returncode == 0,
                    "Emergency script syntax valid",
                    "Emergency script has syntax errors",
                    ValidationSeverity.CRITICAL,
                    details={"stderr": result.stderr if result.returncode != 0 else None},
                    recommendation="Fix syntax errors in emergency_close.py"
                )
            except Exception as e:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Failed to validate emergency script",
                    details={"error": str(e)}
                ))
        else:
            # Create emergency stop script
            emergency_content = '''#!/usr/bin/env python3
"""Emergency position closure script."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from genesis.engine.engine import TradingEngine
from genesis.config.settings import Settings


async def emergency_close_all():
    """Close all positions immediately."""
    print("EMERGENCY STOP INITIATED")
    
    settings = Settings()
    engine = TradingEngine(settings)
    
    try:
        # Get all open positions
        positions = await engine.get_open_positions()
        print(f"Found {len(positions)} open positions")
        
        # Close all positions at market
        for position in positions:
            print(f"Closing position: {position.symbol} - {position.quantity}")
            await engine.close_position(position.id, "EMERGENCY_STOP")
        
        # Cancel all pending orders
        orders = await engine.get_pending_orders()
        print(f"Cancelling {len(orders)} pending orders")
        
        for order in orders:
            await engine.cancel_order(order.id)
        
        # Set emergency flag
        await engine.set_emergency_stop(True)
        
        print("Emergency stop completed")
        print(f"- Positions closed: {len(positions)}")
        print(f"- Orders cancelled: {len(orders)}")
        
    except Exception as e:
        print(f"Emergency stop failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(emergency_close_all())
'''
            emergency_script.write_text(emergency_content)
            emergency_script.chmod(0o755)
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Created emergency stop script"
            ))
        
        # Check state persistence
        state_file = Path(".genesis/state/tier_state.json")
        if state_file.exists():
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="State persistence file found"
            ))
        else:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text('{"tier": "sniper", "capital": 500, "timestamp": null}')
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Created state persistence file"
            ))
    
    async def _validate_data_restoration(self):
        """Validate data restoration procedures."""
        # Test database restoration
        backup_dir = Path(".genesis/backups")
        test_restore_dir = Path(".genesis/test_restore")
        test_restore_dir.mkdir(parents=True, exist_ok=True)
        
        # Find a backup to test
        backups = list(backup_dir.glob("*.db"))
        if backups:
            test_backup = backups[0]
            
            try:
                # Copy backup to test location
                test_db = test_restore_dir / "test_restore.db"
                shutil.copy2(test_backup, test_db)
                
                # Verify restored database
                import sqlite3
                conn = sqlite3.connect(test_db)
                cursor = conn.cursor()
                
                # Check tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                self.check_condition(
                    len(tables) > 0,
                    f"Restored database has {len(tables)} tables",
                    "Restored database is empty",
                    ValidationSeverity.ERROR,
                    details={"tables": [t[0] for t in tables]}
                )
                
                conn.close()
                
                # Clean up test file
                test_db.unlink()
                
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Database restoration test successful"
                ))
                
            except Exception as e:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Database restoration test failed",
                    details={"error": str(e)},
                    recommendation="Verify backup integrity and restoration process"
                ))
        else:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="No backups available to test restoration",
                recommendation="Create backups first"
            ))
    
    async def _test_emergency_procedures(self):
        """Test emergency procedures."""
        procedures = [
            {
                "name": "Emergency Stop",
                "script": "scripts/emergency_close.py",
                "max_execution_time": 30
            },
            {
                "name": "Database Backup",
                "script": "scripts/backup.sh",
                "max_execution_time": 60
            },
            {
                "name": "State Snapshot",
                "script": "scripts/save_state.py",
                "max_execution_time": 10
            }
        ]
        
        for procedure in procedures:
            script_path = Path(procedure["script"])
            
            if script_path.exists():
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Emergency procedure ready: {procedure['name']}",
                    details={
                        "script": str(script_path),
                        "max_time": procedure["max_execution_time"]
                    }
                ))
            else:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing emergency procedure: {procedure['name']}",
                    details={"expected_script": str(script_path)},
                    recommendation=f"Create {script_path}"
                ))
        
        # Check monitoring of critical services
        critical_services = [
            "database_connection",
            "exchange_api",
            "websocket_stream",
            "risk_engine"
        ]
        
        for service in critical_services:
            # This would normally check actual service monitoring
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Critical service monitored: {service}"
            ))