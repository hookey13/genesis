#!/usr/bin/env python3
"""Manual API key rotation script.

This script allows manual rotation of API keys with zero-downtime strategy.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import structlog

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.security.vault_client import VaultClient
from genesis.security.key_rotation import KeyRotationOrchestrator, RotationSchedule
from genesis.config.settings import settings

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def rotate_keys_manual(
    key_identifier: str,
    reason: str,
    force: bool = False,
    dry_run: bool = False
):
    """Manually rotate API keys.
    
    Args:
        key_identifier: Identifier for keys to rotate (exchange, exchange_read)
        reason: Reason for rotation
        force: Force rotation even if not due
        dry_run: Simulate rotation without making changes
    """
    logger.info("Starting manual key rotation",
               key_identifier=key_identifier,
               reason=reason,
               force=force,
               dry_run=dry_run)
    
    # Initialize Vault client
    vault_client = settings.get_vault_client()
    
    if not vault_client.is_connected() and settings.use_vault:
        logger.error("Failed to connect to Vault")
        return False
    
    # Create rotation orchestrator
    schedule = RotationSchedule(enabled=False)  # Disable automatic scheduling
    orchestrator = KeyRotationOrchestrator(
        vault_client=vault_client,
        schedule=schedule
    )
    
    # Initialize orchestrator
    await orchestrator.initialize()
    
    try:
        if dry_run:
            # Simulate rotation
            logger.info("DRY RUN: Simulating key rotation")
            
            # Check current keys
            current_keys = orchestrator.get_active_keys(key_identifier)
            if current_keys:
                logger.info("Current key information",
                          version=current_keys.version,
                          created_at=current_keys.created_at.isoformat(),
                          status=current_keys.status.value)
                
                # Check if rotation is due
                if current_keys.is_expired(30) or force:
                    logger.info("Key rotation would proceed",
                              expired=current_keys.is_expired(30),
                              forced=force)
                else:
                    logger.info("Key rotation not due",
                              age_days=(datetime.now() - current_keys.created_at).days,
                              max_age_days=30)
            else:
                logger.warning("No active keys found", key_identifier=key_identifier)
            
            return True
        
        # Perform actual rotation
        success, message = await orchestrator.rotate_keys(
            key_identifier=key_identifier,
            reason=reason,
            force=force
        )
        
        if success:
            logger.info("Key rotation completed successfully", message=message)
            
            # Display new status
            status = orchestrator.get_rotation_status()
            logger.info("Rotation status", status=status)
        else:
            logger.error("Key rotation failed", message=message)
        
        return success
        
    finally:
        # Cleanup
        await orchestrator.shutdown()


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Manually rotate API keys for Project GENESIS"
    )
    
    parser.add_argument(
        "--key-id",
        default="exchange",
        choices=["exchange", "exchange_read", "all"],
        help="Key identifier to rotate (default: exchange)"
    )
    
    parser.add_argument(
        "--reason",
        default="manual",
        help="Reason for rotation (default: manual)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rotation even if not due"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate rotation without making changes"
    )
    
    args = parser.parse_args()
    
    # Handle 'all' key rotation
    if args.key_id == "all":
        key_ids = ["exchange", "exchange_read"]
    else:
        key_ids = [args.key_id]
    
    # Run rotation for each key
    success = True
    for key_id in key_ids:
        logger.info(f"Processing key rotation for: {key_id}")
        result = asyncio.run(rotate_keys_manual(
            key_identifier=key_id,
            reason=args.reason,
            force=args.force,
            dry_run=args.dry_run
        ))
        success = success and result
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()