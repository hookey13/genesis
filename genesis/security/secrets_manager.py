"""
Secrets Manager with multi-backend support for enterprise-grade secret management.
Implements HashiCorp Vault, AWS Secrets Manager, and local encrypted storage.
"""

import os
import json
import base64
import asyncio
import structlog
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

import hvac
import boto3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from botocore.exceptions import ClientError

from genesis.core.exceptions import (
    GenesisException,
    SecurityError,
    VaultConnectionError,
    EncryptionError
)

logger = structlog.get_logger(__name__)


class SecretBackend(Enum):
    """Supported secret backend types."""
    VAULT = "vault"
    AWS = "aws"
    LOCAL = "local"
    ENVIRONMENT = "environment"


class SecretAccess:
    """Audit record for secret access."""
    
    def __init__(
        self,
        secret_path: str,
        accessor: str,
        action: str,
        timestamp: datetime,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.secret_path = secret_path
        self.accessor = accessor
        self.action = action
        self.timestamp = timestamp
        self.success = success
        self.metadata = metadata or {}


class SecretBackendInterface(ABC):
    """Abstract interface for secret backends."""
    
    @abstractmethod
    async def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from the backend."""
        pass
    
    @abstractmethod
    async def put_secret(self, path: str, secret: Dict[str, Any]) -> bool:
        """Store a secret in the backend."""
        pass
    
    @abstractmethod
    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from the backend."""
        pass
    
    @abstractmethod
    async def list_secrets(self, path: str) -> List[str]:
        """List secrets at a given path."""
        pass
    
    @abstractmethod
    async def rotate_secret(self, path: str, new_secret: Dict[str, Any]) -> bool:
        """Rotate a secret with versioning support."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and accessible."""
        pass


class VaultBackend(SecretBackendInterface):
    """HashiCorp Vault backend implementation."""
    
    def __init__(self, url: str, token: str, mount_point: str = "secret"):
        self.client = hvac.Client(url=url, token=token)
        self.mount_point = mount_point
        self.logger = structlog.get_logger(self.__class__.__name__)
        
    async def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from Vault."""
        try:
            response = await asyncio.to_thread(
                self.client.secrets.kv.v2.read_secret_version,
                path=path,
                mount_point=self.mount_point
            )
            return response.get("data", {}).get("data", {})
        except hvac.exceptions.InvalidPath:
            self.logger.warning(f"Secret not found at path: {path}")
            return None
        except Exception as e:
            raise VaultConnectionError(f"Failed to retrieve secret: {str(e)}")
    
    async def put_secret(self, path: str, secret: Dict[str, Any]) -> bool:
        """Store a secret in Vault."""
        try:
            await asyncio.to_thread(
                self.client.secrets.kv.v2.create_or_update_secret,
                path=path,
                secret=secret,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            raise VaultConnectionError(f"Failed to store secret: {str(e)}")
    
    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from Vault."""
        try:
            await asyncio.to_thread(
                self.client.secrets.kv.v2.delete_metadata_and_all_versions,
                path=path,
                mount_point=self.mount_point
            )
            return True
        except Exception as e:
            raise VaultConnectionError(f"Failed to delete secret: {str(e)}")
    
    async def list_secrets(self, path: str) -> List[str]:
        """List secrets at a given path in Vault."""
        try:
            response = await asyncio.to_thread(
                self.client.secrets.kv.v2.list_secrets,
                path=path,
                mount_point=self.mount_point
            )
            return response.get("data", {}).get("keys", [])
        except Exception as e:
            raise VaultConnectionError(f"Failed to list secrets: {str(e)}")
    
    async def rotate_secret(self, path: str, new_secret: Dict[str, Any]) -> bool:
        """Rotate a secret with versioning in Vault."""
        try:
            # Get current version
            current = await self.get_secret(path)
            if current:
                # Store old version with timestamp
                old_path = f"{path}_old_{datetime.utcnow().isoformat()}"
                await self.put_secret(old_path, current)
            
            # Update to new version
            await self.put_secret(path, new_secret)
            return True
        except Exception as e:
            raise VaultConnectionError(f"Failed to rotate secret: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check Vault health."""
        try:
            is_sealed = await asyncio.to_thread(self.client.sys.is_sealed)
            return not is_sealed
        except Exception:
            return False


class AWSSecretsManagerBackend(SecretBackendInterface):
    """AWS Secrets Manager backend implementation."""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client("secretsmanager", region_name=region_name)
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from AWS Secrets Manager."""
        try:
            response = await asyncio.to_thread(
                self.client.get_secret_value,
                SecretId=path
            )
            secret_string = response.get("SecretString", "{}")
            return json.loads(secret_string)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                self.logger.warning(f"Secret not found: {path}")
                return None
            raise SecurityError(f"Failed to retrieve secret: {str(e)}")
    
    async def put_secret(self, path: str, secret: Dict[str, Any]) -> bool:
        """Store a secret in AWS Secrets Manager."""
        try:
            secret_string = json.dumps(secret)
            
            # Try to update existing secret
            try:
                await asyncio.to_thread(
                    self.client.update_secret,
                    SecretId=path,
                    SecretString=secret_string
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    # Create new secret
                    await asyncio.to_thread(
                        self.client.create_secret,
                        Name=path,
                        SecretString=secret_string
                    )
            return True
        except Exception as e:
            raise SecurityError(f"Failed to store secret: {str(e)}")
    
    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from AWS Secrets Manager."""
        try:
            await asyncio.to_thread(
                self.client.delete_secret,
                SecretId=path,
                ForceDeleteWithoutRecovery=False
            )
            return True
        except Exception as e:
            raise SecurityError(f"Failed to delete secret: {str(e)}")
    
    async def list_secrets(self, path: str) -> List[str]:
        """List secrets with prefix in AWS Secrets Manager."""
        try:
            secrets = []
            paginator = self.client.get_paginator("list_secrets")
            
            # Execute pagination synchronously in thread pool
            pages = await asyncio.to_thread(
                lambda: list(paginator.paginate(
                    Filters=[{"Key": "name", "Values": [path]}]
                ))
            )
            
            for page in pages:
                for secret in page.get("SecretList", []):
                    secrets.append(secret["Name"])
            
            return secrets
        except Exception as e:
            raise SecurityError(f"Failed to list secrets: {str(e)}")
    
    async def rotate_secret(self, path: str, new_secret: Dict[str, Any]) -> bool:
        """Rotate a secret with versioning in AWS."""
        try:
            # AWS Secrets Manager handles versioning automatically
            return await self.put_secret(path, new_secret)
        except Exception as e:
            raise SecurityError(f"Failed to rotate secret: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check AWS Secrets Manager health."""
        try:
            await asyncio.to_thread(self.client.list_secrets, MaxResults=1)
            return True
        except Exception:
            return False


class LocalEncryptedBackend(SecretBackendInterface):
    """Local encrypted file storage backend using Fernet/AES-256."""
    
    def __init__(self, storage_path: Path, master_key: Optional[bytes] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Master key handling
        self.master_key_path = self.storage_path / "master.key"
        if master_key:
            self.cipher = Fernet(master_key)
        else:
            self.cipher = self._initialize_cipher()
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def _initialize_cipher(self) -> Fernet:
        """Initialize or load the master encryption key."""
        if self.master_key_path.exists():
            # Load existing master key
            with open(self.master_key_path, "rb") as f:
                key = f.read()
        else:
            # Generate new master key
            key = Fernet.generate_key()
            
            # Secure permissions (owner read/write only)
            with open(self.master_key_path, "wb") as f:
                f.write(key)
            os.chmod(self.master_key_path, 0o600)
            
            self.logger.info("Generated new master encryption key")
        
        return Fernet(key)
    
    def _get_secret_path(self, path: str) -> Path:
        """Convert logical path to filesystem path."""
        # Replace path separators with safe characters
        safe_path = path.replace("/", "_").replace("\\", "_")
        return self.storage_path / f"{safe_path}.encrypted"
    
    async def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt a secret from local storage."""
        secret_file = self._get_secret_path(path)
        
        if not secret_file.exists():
            return None
        
        try:
            with open(secret_file, "rb") as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt secret: {str(e)}")
    
    async def put_secret(self, path: str, secret: Dict[str, Any]) -> bool:
        """Encrypt and store a secret locally."""
        try:
            secret_file = self._get_secret_path(path)
            
            # Serialize and encrypt
            secret_json = json.dumps(secret)
            encrypted_data = self.cipher.encrypt(secret_json.encode())
            
            # Write with secure permissions
            with open(secret_file, "wb") as f:
                f.write(encrypted_data)
            os.chmod(secret_file, 0o600)
            
            return True
        except Exception as e:
            raise EncryptionError(f"Failed to encrypt secret: {str(e)}")
    
    async def delete_secret(self, path: str) -> bool:
        """Delete an encrypted secret from local storage."""
        secret_file = self._get_secret_path(path)
        
        if secret_file.exists():
            # Secure deletion - overwrite before removing
            with open(secret_file, "wb") as f:
                f.write(os.urandom(secret_file.stat().st_size))
            secret_file.unlink()
            return True
        return False
    
    async def list_secrets(self, path: str) -> List[str]:
        """List secrets matching a path prefix."""
        prefix = path.replace("/", "_").replace("\\", "_")
        secrets = []
        
        for file in self.storage_path.glob(f"{prefix}*.encrypted"):
            # Convert back to logical path
            logical_path = file.stem.replace("_", "/")
            secrets.append(logical_path)
        
        return secrets
    
    async def rotate_secret(self, path: str, new_secret: Dict[str, Any]) -> bool:
        """Rotate a secret with backup."""
        try:
            # Backup current secret
            current = await self.get_secret(path)
            if current:
                backup_path = f"{path}_backup_{datetime.utcnow().isoformat()}"
                await self.put_secret(backup_path, current)
            
            # Store new secret
            return await self.put_secret(path, new_secret)
        except Exception as e:
            raise EncryptionError(f"Failed to rotate secret: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check local storage health."""
        return self.storage_path.exists() and self.master_key_path.exists()


class SecretsManager:
    """
    Central secrets management system with multi-backend support.
    Provides unified interface for secret operations with audit logging.
    """
    
    def __init__(
        self,
        backend: SecretBackend = SecretBackend.LOCAL,
        config: Optional[Dict[str, Any]] = None
    ):
        self.backend_type = backend
        self.config = config or {}
        self.backend = self._initialize_backend()
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(hours=1)
        self.audit_log: List[SecretAccess] = []
        self.logger = structlog.get_logger(__name__)
    
    def _initialize_backend(self) -> SecretBackendInterface:
        """Initialize the appropriate backend based on configuration."""
        if self.backend_type == SecretBackend.VAULT:
            return VaultBackend(
                url=self.config.get("vault_url", "http://localhost:8200"),
                token=self.config.get("vault_token", ""),
                mount_point=self.config.get("mount_point", "secret")
            )
        elif self.backend_type == SecretBackend.AWS:
            return AWSSecretsManagerBackend(
                region_name=self.config.get("aws_region", "us-east-1")
            )
        elif self.backend_type == SecretBackend.LOCAL:
            storage_path = Path.home() / ".genesis" / ".secrets"
            return LocalEncryptedBackend(
                storage_path=storage_path,
                master_key=self.config.get("master_key")
            )
        else:
            # Fallback to environment variables
            return None
    
    def _log_access(
        self,
        path: str,
        action: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log secret access for audit trail."""
        access = SecretAccess(
            secret_path=path,
            accessor=os.getenv("USER", "unknown"),
            action=action,
            timestamp=datetime.utcnow(),
            success=success,
            metadata=metadata
        )
        self.audit_log.append(access)
        
        # Log to structured logger
        self.logger.info(
            "secret_access",
            path=path,
            action=action,
            success=success,
            accessor=access.accessor,
            **metadata or {}
        )
    
    async def get_secret(
        self,
        path: str,
        use_cache: bool = True,
        fallback_to_env: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret with caching and fallback support.
        
        Args:
            path: Secret path
            use_cache: Whether to use cached values
            fallback_to_env: Whether to fallback to environment variables
        
        Returns:
            Secret dictionary or None if not found
        """
        try:
            # Check cache first
            if use_cache and path in self.cache:
                cached_value, cached_time = self.cache[path]
                if datetime.utcnow() - cached_time < self.cache_ttl:
                    self._log_access(path, "get_cached", True)
                    return cached_value
            
            # Try backend
            if self.backend:
                secret = await self.backend.get_secret(path)
                if secret:
                    # Update cache
                    self.cache[path] = (secret, datetime.utcnow())
                    self._log_access(path, "get", True)
                    return secret
            
            # Fallback to environment variables
            if fallback_to_env:
                env_key = path.upper().replace("/", "_").replace("-", "_")
                env_value = os.getenv(env_key)
                if env_value:
                    try:
                        secret = json.loads(env_value)
                    except json.JSONDecodeError:
                        secret = {"value": env_value}
                    
                    self._log_access(path, "get_env", True)
                    return secret
            
            self._log_access(path, "get", False, {"reason": "not_found"})
            return None
            
        except Exception as e:
            self._log_access(path, "get", False, {"error": str(e)})
            raise
    
    async def put_secret(
        self,
        path: str,
        secret: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a secret in the backend.
        
        Args:
            path: Secret path
            secret: Secret data to store
            metadata: Additional metadata for audit
        
        Returns:
            True if successful
        """
        try:
            if not self.backend:
                raise SecurityError("No backend configured for storing secrets")
            
            success = await self.backend.put_secret(path, secret)
            
            if success:
                # Invalidate cache
                self.cache.pop(path, None)
                self._log_access(path, "put", True, metadata)
            
            return success
            
        except Exception as e:
            self._log_access(path, "put", False, {"error": str(e)})
            raise
    
    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from the backend."""
        try:
            if not self.backend:
                raise SecurityError("No backend configured")
            
            success = await self.backend.delete_secret(path)
            
            if success:
                # Invalidate cache
                self.cache.pop(path, None)
                self._log_access(path, "delete", True)
            
            return success
            
        except Exception as e:
            self._log_access(path, "delete", False, {"error": str(e)})
            raise
    
    async def rotate_api_keys(
        self,
        api_key_path: str = "/genesis/exchange/api-keys",
        grace_period: timedelta = timedelta(minutes=5)
    ) -> Dict[str, Any]:
        """
        Rotate API keys with dual-key strategy for zero downtime.
        
        Args:
            api_key_path: Path to API keys in backend
            grace_period: Time to maintain both old and new keys
        
        Returns:
            Dictionary with rotation status and new keys
        """
        try:
            # Get current keys
            current_keys = await self.get_secret(api_key_path, use_cache=False)
            if not current_keys:
                raise SecurityError("No existing API keys found")
            
            # Generate or fetch new keys (implementation specific)
            # This would integrate with exchange API for key generation
            new_keys = {
                "api_key": f"new_{current_keys.get('api_key', '')}",
                "api_secret": f"new_{current_keys.get('api_secret', '')}",
                "created_at": datetime.utcnow().isoformat(),
                "rotation_id": os.urandom(16).hex()
            }
            
            # Store both keys during grace period
            dual_keys = {
                "primary": new_keys,
                "secondary": current_keys,
                "grace_period_end": (datetime.utcnow() + grace_period).isoformat()
            }
            
            # Update backend
            await self.put_secret(api_key_path, dual_keys, {"action": "rotation_start"})
            
            # Schedule old key revocation after grace period
            # This would be handled by a background task in production
            
            self._log_access(
                api_key_path,
                "rotate",
                True,
                {"rotation_id": new_keys["rotation_id"]}
            )
            
            return {
                "status": "success",
                "rotation_id": new_keys["rotation_id"],
                "grace_period_end": dual_keys["grace_period_end"],
                "new_keys": new_keys
            }
            
        except Exception as e:
            self._log_access(api_key_path, "rotate", False, {"error": str(e)})
            raise
    
    async def generate_temporary_credential(
        self,
        purpose: str,
        ttl: timedelta = timedelta(hours=1),
        permissions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate temporary credentials for operations.
        
        Args:
            purpose: Purpose of the temporary credential
            ttl: Time to live for the credential
            permissions: List of allowed operations
        
        Returns:
            Temporary credential with expiration
        """
        try:
            temp_cred = {
                "credential_id": os.urandom(32).hex(),
                "purpose": purpose,
                "permissions": permissions or ["read"],
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + ttl).isoformat(),
                "token": base64.b64encode(os.urandom(64)).decode()
            }
            
            # Store temporary credential
            path = f"/genesis/temp/{temp_cred['credential_id']}"
            await self.put_secret(path, temp_cred, {"purpose": purpose})
            
            self._log_access(
                path,
                "generate_temp",
                True,
                {"purpose": purpose, "ttl": str(ttl)}
            )
            
            return temp_cred
            
        except Exception as e:
            self._log_access("temp_credential", "generate", False, {"error": str(e)})
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of the secrets management system."""
        health = {
            "backend": self.backend_type.value,
            "backend_healthy": False,
            "cache_size": len(self.cache),
            "audit_log_size": len(self.audit_log),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.backend:
            try:
                health["backend_healthy"] = await self.backend.health_check()
            except Exception as e:
                health["backend_error"] = str(e)
        
        return health
    
    def get_audit_log(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        path_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries with filtering.
        
        Args:
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            path_filter: Filter by secret path prefix
        
        Returns:
            List of audit log entries
        """
        filtered_log = self.audit_log
        
        if start_time:
            filtered_log = [a for a in filtered_log if a.timestamp >= start_time]
        
        if end_time:
            filtered_log = [a for a in filtered_log if a.timestamp <= end_time]
        
        if path_filter:
            filtered_log = [a for a in filtered_log if a.secret_path.startswith(path_filter)]
        
        return [
            {
                "path": access.secret_path,
                "accessor": access.accessor,
                "action": access.action,
                "timestamp": access.timestamp.isoformat(),
                "success": access.success,
                "metadata": access.metadata
            }
            for access in filtered_log
        ]