#!/usr/bin/env python3
"""Migration script to convert existing SHA256 password hashes to bcrypt.

This script should be run after database migration to mark existing users
for password migration on their next login.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from sqlalchemy import create_engine, select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from genesis.config.settings import settings
from genesis.data.models_db import User as UserDB
from genesis.security.password_manager import SecurePasswordManager

logger = structlog.get_logger(__name__)


class PasswordMigrator:
    """Handles migration of passwords from SHA256 to bcrypt."""
    
    def __init__(self, database_url: str):
        """Initialize migrator.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.password_manager = SecurePasswordManager()
        self.stats = {
            'total_users': 0,
            'sha256_users': 0,
            'bcrypt_users': 0,
            'migrated': 0,
            'errors': 0
        }
    
    async def migrate_all_users(self) -> dict:
        """Migrate all users with SHA256 passwords.
        
        Returns:
            Migration statistics
        """
        # Create async engine
        engine = create_async_engine(
            self.database_url,
            echo=False,
            future=True
        )
        
        async_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with async_session() as session:
            try:
                # Get all users
                result = await session.execute(
                    select(UserDB)
                )
                users = result.scalars().all()
                
                self.stats['total_users'] = len(users)
                
                for user in users:
                    await self._process_user(session, user)
                
                # Commit all changes
                await session.commit()
                
                logger.info(
                    "Password migration completed",
                    **self.stats
                )
                
                return self.stats
                
            except Exception as e:
                logger.error(
                    "Migration failed",
                    error=str(e)
                )
                await session.rollback()
                raise
            finally:
                await engine.dispose()
    
    async def _process_user(self, session: AsyncSession, user: UserDB) -> None:
        """Process a single user for migration.
        
        Args:
            session: Database session
            user: User to process
        """
        try:
            if not user.password_hash:
                logger.debug(
                    "User has no password",
                    user_id=user.id,
                    username=user.username
                )
                return
            
            # Check password hash length to determine type
            hash_length = len(user.password_hash)
            
            if hash_length == 64:
                # SHA256 hash - mark for migration
                self.stats['sha256_users'] += 1
                
                # Store old hash for migration
                user.old_sha256_hash = user.password_hash
                user.sha256_migrated = False
                
                logger.info(
                    "Marked user for SHA256 migration",
                    user_id=user.id,
                    username=user.username
                )
                
            elif hash_length == 60:
                # Bcrypt hash - already migrated
                self.stats['bcrypt_users'] += 1
                
                user.sha256_migrated = True
                user.old_sha256_hash = None
                
                logger.debug(
                    "User already using bcrypt",
                    user_id=user.id,
                    username=user.username
                )
                
            else:
                # Unknown hash format
                logger.warning(
                    "Unknown password hash format",
                    user_id=user.id,
                    username=user.username,
                    hash_length=hash_length
                )
                self.stats['errors'] += 1
                
        except Exception as e:
            logger.error(
                "Error processing user",
                user_id=user.id,
                username=user.username,
                error=str(e)
            )
            self.stats['errors'] += 1
    
    async def generate_migration_report(self) -> str:
        """Generate a migration report.
        
        Returns:
            Formatted migration report
        """
        report = [
            "=" * 60,
            "Password Migration Report",
            "=" * 60,
            f"Total Users: {self.stats['total_users']}",
            f"SHA256 Users (need migration): {self.stats['sha256_users']}",
            f"Bcrypt Users (already secure): {self.stats['bcrypt_users']}",
            f"Errors: {self.stats['errors']}",
            "=" * 60
        ]
        
        if self.stats['sha256_users'] > 0:
            report.append(
                f"\n⚠️  {self.stats['sha256_users']} users need to log in to complete migration"
            )
            report.append(
                "Their passwords will be automatically upgraded to bcrypt on next login."
            )
        
        if self.stats['bcrypt_users'] == self.stats['total_users']:
            report.append(
                "\n✅ All users are using secure bcrypt hashing!"
            )
        
        return "\n".join(report)


async def main():
    """Main migration function."""
    print("Starting password migration...")
    print("-" * 60)
    
    # Get database URL from settings
    database_url = settings.database_url
    
    if not database_url:
        print("❌ No database URL configured")
        return 1
    
    # Create migrator
    migrator = PasswordMigrator(database_url)
    
    try:
        # Run migration
        stats = await migrator.migrate_all_users()
        
        # Print report
        report = await migrator.generate_migration_report()
        print(report)
        
        # Return exit code based on errors
        return 1 if stats['errors'] > 0 else 0
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return 1


def run_migration():
    """Entry point for migration script."""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run_migration()