"""Integration tests for disaster recovery procedures."""

import pytest
from pathlib import Path
from tests.dr.disaster_recovery import (
    DisasterRecoveryDrill,
    verify_backup_automation,
    test_point_in_time_recovery
)


@pytest.mark.asyncio
async def test_dr_drill_init():
    """Test DR drill initialization."""
    drill = DisasterRecoveryDrill()
    assert drill.backup_path == Path("backups")
    assert drill.data_path == Path(".genesis")
    assert drill.metrics["data_integrity"] is True


@pytest.mark.asyncio
async def test_backup_creation():
    """Test backup creation process."""
    drill = DisasterRecoveryDrill(
        backup_path="test_backups",
        data_path="test_data"
    )
    
    # Create test data
    drill.data_path.mkdir(exist_ok=True)
    (drill.data_path / "test.txt").write_text("test data")
    
    # Create backup
    await drill.create_backup()
    
    assert drill.metrics["backup_time"] > 0
    assert drill.metrics["recovery_point"] is not None
    
    # Cleanup
    import shutil
    shutil.rmtree("test_backups", ignore_errors=True)
    shutil.rmtree("test_data", ignore_errors=True)


@pytest.mark.asyncio
async def test_backup_automation_verification():
    """Test backup automation verification."""
    result = await verify_backup_automation()
    assert result is True


@pytest.mark.asyncio
async def test_point_in_time_recovery_capability():
    """Test point-in-time recovery."""
    result = await test_point_in_time_recovery()
    assert result is True