"""Security module for Project GENESIS.

Handles secrets management, API key rotation, permissions, audit logging,
and other security-critical functionality.
"""

from genesis.security.vault_client import VaultClient
from genesis.security.key_rotation import (
    KeyRotationOrchestrator,
    RotationSchedule,
    APIKeyVersion,
    KeyStatus
)

__all__ = [
    "VaultClient",
    "KeyRotationOrchestrator", 
    "RotationSchedule",
    "APIKeyVersion",
    "KeyStatus"
]