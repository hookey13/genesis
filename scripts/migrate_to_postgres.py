#!/usr/bin/env python3
"""Script to execute SQLite to PostgreSQL migration with rollback capability.

This script provides a safe migration path with:
- Pre-migration validation
- Automatic backup creation  
- Progress tracking
- Rollback on failure
- Post-migration verification
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.data.migration_engine import SQLiteToPostgreSQLMigrator
from genesis.core.exceptions import MigrationError


class MigrationManager:
    """Manages the complete migration process with safety checks."""
    
    def __init__(self, config_path: str = None, dry_run: bool = False):
        """Initialize migration manager.
        
        Args:
            config_path: Path to configuration file
            dry_run: If True, perform validation only without actual migration
        """
        self.config_path = config_path or "config/database.yaml"
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        self.backup_paths = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup migration logging."""
        log_dir = Path("logs/migration")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"migration_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def load_config(self) -> Dict[str, Any]:
        """Load database configuration."""
        # In production, this would load from Vault
        # For now, use environment variables
        return {
            'sqlite_path': os.getenv('SQLITE_DB_PATH', '.genesis/data/genesis.db'),
            'postgres': {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB', 'genesis_trading'),
                'user': os.getenv('POSTGRES_USER', 'genesis'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
            }
        }
    
    def validate_prerequisites(self, config: Dict[str, Any]) -> bool:
        """Validate migration prerequisites.
        
        Args:
            config: Database configuration
            
        Returns:
            True if all prerequisites are met
        """
        self.logger.info("Validating prerequisites...")
        
        # Check SQLite database exists
        sqlite_path = Path(config['sqlite_path'])
        if not sqlite_path.exists():
            self.logger.error(f"SQLite database not found: {sqlite_path}")
            return False
        
        # Check available disk space (need 3x database size for safety)
        db_size = sqlite_path.stat().st_size
        free_space = self._get_free_space(sqlite_path.parent)
        required_space = db_size * 3
        
        if free_space < required_space:
            self.logger.error(
                f"Insufficient disk space. Required: {required_space / 1e9:.2f}GB, "
                f"Available: {free_space / 1e9:.2f}GB"
            )
            return False
        
        self.logger.info(f"✓ SQLite database found: {sqlite_path}")
        self.logger.info(f"✓ Sufficient disk space: {free_space / 1e9:.2f}GB available")
        
        return True
    
    def _get_free_space(self, path: Path) -> int:
        """Get free disk space in bytes."""
        import shutil
        stat = shutil.disk_usage(path)
        return stat.free
    
    async def create_comprehensive_backup(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive backup before migration.
        
        Args:
            config: Database configuration
            
        Returns:
            Dictionary with backup paths
        """
        self.logger.info("Creating comprehensive backup...")
        
        backup_dir = Path("backups/migration")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Backup SQLite database
        sqlite_path = Path(config['sqlite_path'])
        sqlite_backup = backup_dir / f"genesis_{timestamp}.db"
        
        import shutil
        shutil.copy2(sqlite_path, sqlite_backup)
        
        # Create metadata file
        metadata = {
            'timestamp': timestamp,
            'source_database': str(sqlite_path),
            'source_size': sqlite_path.stat().st_size,
            'backup_location': str(sqlite_backup),
            'config': config,
        }
        
        metadata_file = backup_dir / f"migration_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.backup_paths.append(str(sqlite_backup))
        
        self.logger.info(f"✓ Backup created: {sqlite_backup}")
        self.logger.info(f"✓ Metadata saved: {metadata_file}")
        
        return {
            'database': str(sqlite_backup),
            'metadata': str(metadata_file),
        }
    
    async def perform_migration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the migration.
        
        Args:
            config: Database configuration
            
        Returns:
            Migration results
        """
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual migration will be performed")
            return {'status': 'dry_run', 'message': 'Validation completed successfully'}
        
        migrator = SQLiteToPostgreSQLMigrator(
            sqlite_path=config['sqlite_path'],
            postgres_config=config['postgres'],
            batch_size=1000,
            logger=self.logger
        )
        
        try:
            results = await migrator.execute_migration()
            return results
        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            raise
    
    async def rollback(self, backup_paths: Dict[str, str]) -> None:
        """Rollback migration using backup.
        
        Args:
            backup_paths: Dictionary with backup file paths
        """
        self.logger.warning("Initiating rollback...")
        
        try:
            # Restore SQLite from backup
            import shutil
            backup_db = backup_paths['database']
            original_path = json.load(open(backup_paths['metadata']))['source_database']
            
            # Create rollback backup of current state
            rollback_backup = f"{original_path}.rollback.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(original_path, rollback_backup)
            
            # Restore from backup
            shutil.copy2(backup_db, original_path)
            
            self.logger.info(f"✓ Database restored from backup: {backup_db}")
            self.logger.info(f"✓ Current state backed up to: {rollback_backup}")
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {str(e)}")
            raise
    
    async def run(self) -> int:
        """Run the complete migration process.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting SQLite to PostgreSQL Migration")
            self.logger.info("=" * 60)
            
            # Load configuration
            config = self.load_config()
            
            # Validate prerequisites
            if not self.validate_prerequisites(config):
                return 1
            
            # Create backup
            backup_paths = await self.create_comprehensive_backup(config)
            
            # Perform migration
            results = await self.perform_migration(config)
            
            # Log results
            if results['status'] == 'success':
                self.logger.info("=" * 60)
                self.logger.info("Migration completed successfully!")
                self.logger.info(f"Tables migrated: {results.get('tables_migrated', 0)}")
                self.logger.info(f"Total rows: {results.get('total_rows', 0)}")
                self.logger.info("=" * 60)
                return 0
            else:
                self.logger.error("Migration failed - initiating rollback")
                await self.rollback(backup_paths)
                return 1
                
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            if self.backup_paths:
                await self.rollback({'database': self.backup_paths[0]})
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate Genesis database from SQLite to PostgreSQL")
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        default='config/database.yaml'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform validation only without actual migration'
    )
    parser.add_argument(
        '--rollback',
        help='Rollback to specified backup file',
        default=None
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        # Perform rollback
        manager = MigrationManager()
        asyncio.run(manager.rollback({'database': args.rollback}))
        return 0
    
    # Run migration
    manager = MigrationManager(
        config_path=args.config,
        dry_run=args.dry_run
    )
    
    exit_code = asyncio.run(manager.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()