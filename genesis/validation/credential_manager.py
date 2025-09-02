"""Secure credential management for production deployment.

This module provides secure storage and retrieval of sensitive credentials
using environment variables, secure vaults, and encrypted storage.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)


@dataclass
class Credential:
    """Represents a stored credential."""
    name: str
    value: str
    created: datetime
    expires: datetime | None = None
    encrypted: bool = True
    metadata: dict[str, Any] | None = None

    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if not self.expires:
            return False
        return datetime.utcnow() > self.expires


class CredentialManager:
    """Manages secure storage and retrieval of credentials."""

    # Supported credential sources in priority order
    CREDENTIAL_SOURCES = [
        "vault",      # HashiCorp Vault or similar
        "env",        # Environment variables
        "encrypted",  # Local encrypted storage
        "config"      # Configuration files (least secure)
    ]

    def __init__(self, genesis_root: Path | None = None):
        """Initialize credential manager.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.credentials_dir = self.genesis_root / ".genesis" / "credentials"
        self.credentials_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption key
        self._init_encryption()

        # Cache for loaded credentials
        self._cache: dict[str, Credential] = {}

        # Vault connection (if available)
        self._vault_client = self._init_vault()

    def get_credential(self, name: str, required: bool = True) -> str | None:
        """Get a credential by name.
        
        Args:
            name: Credential name
            required: If True, raise error if not found
            
        Returns:
            Credential value or None if not found
            
        Raises:
            CredentialNotFoundError: If required and not found
        """
        # Check cache first
        if name in self._cache:
            cred = self._cache[name]
            if not cred.is_expired():
                logger.debug("Credential retrieved from cache", name=name)
                return cred.value
            else:
                logger.info("Cached credential expired", name=name)
                del self._cache[name]

        # Try each source in priority order
        for source in self.CREDENTIAL_SOURCES:
            value = self._get_from_source(name, source)
            if value:
                logger.info(
                    "Credential retrieved",
                    name=name,
                    source=source
                )

                # Cache the credential
                self._cache[name] = Credential(
                    name=name,
                    value=value,
                    created=datetime.utcnow(),
                    expires=datetime.utcnow() + timedelta(hours=1)
                )

                return value

        # Not found
        if required:
            logger.error("Required credential not found", name=name)
            raise CredentialNotFoundError(f"Credential '{name}' not found")

        logger.warning("Optional credential not found", name=name)
        return None

    def store_credential(
        self,
        name: str,
        value: str,
        source: str = "encrypted",
        expires_hours: int | None = None
    ) -> bool:
        """Store a credential securely.
        
        Args:
            name: Credential name
            value: Credential value
            source: Storage source
            expires_hours: Hours until expiration
            
        Returns:
            True if stored successfully
        """
        try:
            if source == "vault":
                return self._store_in_vault(name, value, expires_hours)
            elif source == "encrypted":
                return self._store_encrypted(name, value, expires_hours)
            elif source == "env":
                # Environment variables are set externally
                logger.warning(
                    "Cannot store in environment variables",
                    name=name
                )
                return False
            else:
                logger.error(
                    "Unsupported storage source",
                    name=name,
                    source=source
                )
                return False
        except Exception as e:
            logger.error(
                "Failed to store credential",
                name=name,
                source=source,
                error=str(e)
            )
            return False

    def rotate_credential(self, name: str, new_value: str) -> bool:
        """Rotate a credential to a new value.
        
        Args:
            name: Credential name
            new_value: New credential value
            
        Returns:
            True if rotated successfully
        """
        # Get current source
        current_source = self._find_credential_source(name)
        if not current_source:
            logger.error("Cannot rotate non-existent credential", name=name)
            return False

        # Store with same source
        success = self.store_credential(name, new_value, current_source)

        if success:
            # Clear cache
            if name in self._cache:
                del self._cache[name]

            logger.info(
                "Credential rotated",
                name=name,
                source=current_source
            )

            # Audit log rotation
            self._audit_rotation(name, current_source)

        return success

    def delete_credential(self, name: str) -> bool:
        """Delete a credential.
        
        Args:
            name: Credential name
            
        Returns:
            True if deleted successfully
        """
        # Find where it's stored
        source = self._find_credential_source(name)
        if not source:
            logger.warning("Credential not found for deletion", name=name)
            return False

        try:
            if source == "vault":
                return self._delete_from_vault(name)
            elif source == "encrypted":
                return self._delete_encrypted(name)
            else:
                logger.warning(
                    "Cannot delete from source",
                    name=name,
                    source=source
                )
                return False
        except Exception as e:
            logger.error(
                "Failed to delete credential",
                name=name,
                source=source,
                error=str(e)
            )
            return False
        finally:
            # Clear cache
            if name in self._cache:
                del self._cache[name]

    def list_credentials(self) -> list[str]:
        """List all available credential names.
        
        Returns:
            List of credential names
        """
        credentials = set()

        # From vault
        if self._vault_client:
            credentials.update(self._list_vault_credentials())

        # From environment
        prefix = "GENESIS_CRED_"
        for key in os.environ:
            if key.startswith(prefix):
                name = key[len(prefix):].lower()
                credentials.add(name)

        # From encrypted storage
        credentials.update(self._list_encrypted_credentials())

        return sorted(list(credentials))

    def _init_encryption(self):
        """Initialize encryption for local storage."""
        key_file = self.credentials_dir / ".key"

        if key_file.exists():
            # Load existing key
            with open(key_file, "rb") as f:
                self._encryption_key = f.read()
        else:
            # Generate new key
            self._encryption_key = Fernet.generate_key()

            # Save key (in production, this should be in a secure location)
            with open(key_file, "wb") as f:
                f.write(self._encryption_key)

            # Restrict permissions
            if os.name != 'nt':  # Unix-like systems
                os.chmod(key_file, 0o600)

            logger.info("Generated new encryption key")

        self._cipher = Fernet(self._encryption_key)

    def _init_vault(self):
        """Initialize connection to HashiCorp Vault if available."""
        try:
            # Check if vault is configured
            vault_addr = os.environ.get("VAULT_ADDR")
            vault_token = os.environ.get("VAULT_TOKEN")

            if not vault_addr or not vault_token:
                logger.debug("Vault not configured")
                return None

            # In production, use hvac library:
            # import hvac
            # client = hvac.Client(url=vault_addr, token=vault_token)
            # if client.is_authenticated():
            #     return client

            logger.info("Vault connection simulated for demo")
            return None  # Demo implementation

        except Exception as e:
            logger.warning("Failed to initialize vault", error=str(e))
            return None

    def _get_from_source(self, name: str, source: str) -> str | None:
        """Get credential from specific source.
        
        Args:
            name: Credential name
            source: Source to check
            
        Returns:
            Credential value or None
        """
        try:
            if source == "vault":
                return self._get_from_vault(name)
            elif source == "env":
                return self._get_from_env(name)
            elif source == "encrypted":
                return self._get_from_encrypted(name)
            elif source == "config":
                return self._get_from_config(name)
        except Exception as e:
            logger.debug(
                "Failed to get from source",
                name=name,
                source=source,
                error=str(e)
            )

        return None

    def _get_from_vault(self, name: str) -> str | None:
        """Get credential from vault.
        
        Args:
            name: Credential name
            
        Returns:
            Credential value or None
        """
        if not self._vault_client:
            return None

        # In production:
        # try:
        #     secret = self._vault_client.secrets.kv.v2.read_secret_version(
        #         path=f"genesis/{name}"
        #     )
        #     return secret["data"]["data"]["value"]
        # except Exception:
        #     return None

        return None  # Demo implementation

    def _get_from_env(self, name: str) -> str | None:
        """Get credential from environment variable.
        
        Args:
            name: Credential name
            
        Returns:
            Credential value or None
        """
        # Try different naming conventions
        env_names = [
            f"GENESIS_CRED_{name.upper()}",
            f"GENESIS_{name.upper()}",
            name.upper()
        ]

        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                return value

        return None

    def _get_from_encrypted(self, name: str) -> str | None:
        """Get credential from encrypted local storage.
        
        Args:
            name: Credential name
            
        Returns:
            Credential value or None
        """
        cred_file = self.credentials_dir / f"{name}.enc"

        if not cred_file.exists():
            return None

        try:
            with open(cred_file, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self._cipher.decrypt(encrypted_data)
            cred_data = json.loads(decrypted_data.decode())

            # Check expiration
            if cred_data.get("expires"):
                expires = datetime.fromisoformat(cred_data["expires"])
                if datetime.utcnow() > expires:
                    logger.info("Encrypted credential expired", name=name)
                    return None

            return cred_data["value"]

        except Exception as e:
            logger.error(
                "Failed to decrypt credential",
                name=name,
                error=str(e)
            )
            return None

    def _get_from_config(self, name: str) -> str | None:
        """Get credential from configuration file.
        
        WARNING: This is the least secure option.
        
        Args:
            name: Credential name
            
        Returns:
            Credential value or None
        """
        config_file = self.genesis_root / "config" / "credentials.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file) as f:
                config = json.load(f)

            if name in config:
                logger.warning(
                    "Using credential from config file - not secure",
                    name=name
                )
                return config[name]
        except Exception as e:
            logger.error(
                "Failed to read config file",
                error=str(e)
            )

        return None

    def _store_in_vault(
        self,
        name: str,
        value: str,
        expires_hours: int | None
    ) -> bool:
        """Store credential in vault.
        
        Args:
            name: Credential name
            value: Credential value
            expires_hours: Hours until expiration
            
        Returns:
            True if stored successfully
        """
        if not self._vault_client:
            return False

        # In production:
        # try:
        #     self._vault_client.secrets.kv.v2.create_or_update_secret(
        #         path=f"genesis/{name}",
        #         secret={"value": value},
        #         cas=0
        #     )
        #     return True
        # except Exception as e:
        #     logger.error("Failed to store in vault", error=str(e))
        #     return False

        return False  # Demo implementation

    def _store_encrypted(
        self,
        name: str,
        value: str,
        expires_hours: int | None
    ) -> bool:
        """Store credential in encrypted local storage.
        
        Args:
            name: Credential name
            value: Credential value
            expires_hours: Hours until expiration
            
        Returns:
            True if stored successfully
        """
        cred_file = self.credentials_dir / f"{name}.enc"

        try:
            cred_data = {
                "value": value,
                "created": datetime.utcnow().isoformat()
            }

            if expires_hours:
                expires = datetime.utcnow() + timedelta(hours=expires_hours)
                cred_data["expires"] = expires.isoformat()

            # Encrypt data
            json_data = json.dumps(cred_data).encode()
            encrypted_data = self._cipher.encrypt(json_data)

            # Write to file
            with open(cred_file, "wb") as f:
                f.write(encrypted_data)

            # Restrict permissions
            if os.name != 'nt':  # Unix-like systems
                os.chmod(cred_file, 0o600)

            logger.info("Credential stored encrypted", name=name)
            return True

        except Exception as e:
            logger.error(
                "Failed to store encrypted credential",
                name=name,
                error=str(e)
            )
            return False

    def _delete_from_vault(self, name: str) -> bool:
        """Delete credential from vault.
        
        Args:
            name: Credential name
            
        Returns:
            True if deleted successfully
        """
        if not self._vault_client:
            return False

        # In production:
        # try:
        #     self._vault_client.secrets.kv.v2.delete_metadata_and_all_versions(
        #         path=f"genesis/{name}"
        #     )
        #     return True
        # except Exception as e:
        #     logger.error("Failed to delete from vault", error=str(e))
        #     return False

        return False  # Demo implementation

    def _delete_encrypted(self, name: str) -> bool:
        """Delete credential from encrypted storage.
        
        Args:
            name: Credential name
            
        Returns:
            True if deleted successfully
        """
        cred_file = self.credentials_dir / f"{name}.enc"

        if cred_file.exists():
            try:
                cred_file.unlink()
                logger.info("Encrypted credential deleted", name=name)
                return True
            except Exception as e:
                logger.error(
                    "Failed to delete encrypted credential",
                    name=name,
                    error=str(e)
                )
                return False

        return False

    def _find_credential_source(self, name: str) -> str | None:
        """Find which source contains a credential.
        
        Args:
            name: Credential name
            
        Returns:
            Source name or None
        """
        for source in self.CREDENTIAL_SOURCES:
            if self._get_from_source(name, source):
                return source
        return None

    def _list_vault_credentials(self) -> list[str]:
        """List credentials in vault.
        
        Returns:
            List of credential names
        """
        if not self._vault_client:
            return []

        # In production:
        # try:
        #     result = self._vault_client.secrets.kv.v2.list_secrets(
        #         path="genesis"
        #     )
        #     return result["data"]["keys"]
        # except Exception:
        #     return []

        return []  # Demo implementation

    def _list_encrypted_credentials(self) -> list[str]:
        """List credentials in encrypted storage.
        
        Returns:
            List of credential names
        """
        credentials = []

        for cred_file in self.credentials_dir.glob("*.enc"):
            name = cred_file.stem
            credentials.append(name)

        return credentials

    def _audit_rotation(self, name: str, source: str):
        """Audit log credential rotation.
        
        Args:
            name: Credential name
            source: Storage source
        """
        audit_file = self.credentials_dir / "audit.log"

        try:
            with open(audit_file, "a") as f:
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "rotate",
                    "credential": name,
                    "source": source
                }
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to write audit log", error=str(e))


class CredentialNotFoundError(Exception):
    """Raised when a required credential is not found."""
    pass


# Singleton instance
_credential_manager: CredentialManager | None = None


def get_credential_manager(genesis_root: Path | None = None) -> CredentialManager:
    """Get or create the credential manager singleton.
    
    Args:
        genesis_root: Root directory of Genesis project
        
    Returns:
        Credential manager instance
    """
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager(genesis_root)
    return _credential_manager
