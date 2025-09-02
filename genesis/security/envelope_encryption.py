"""Envelope encryption for sensitive data using Vault's transit engine."""

import base64
import json
from typing import Any, Dict, Optional
from decimal import Decimal

import structlog
from cryptography.fernet import Fernet

from genesis.security.vault_manager import VaultManager
from genesis.core.exceptions import SecurityException


logger = structlog.get_logger(__name__)


class EnvelopeEncryption:
    """Provides envelope encryption for sensitive application data."""
    
    def __init__(self, vault_manager: VaultManager):
        """Initialize envelope encryption.
        
        Args:
            vault_manager: Initialized Vault manager instance
        """
        self.vault = vault_manager
        self._data_key_cache: Dict[str, bytes] = {}
        
    async def encrypt_data(
        self,
        data: Any,
        key_name: str = "genesis-data"
    ) -> Dict[str, str]:
        """Encrypt data using envelope encryption.
        
        Args:
            data: Data to encrypt (will be JSON serialized)
            key_name: Vault transit key name
            
        Returns:
            Dictionary containing encrypted data and encrypted DEK
        """
        try:
            # Generate data encryption key (DEK)
            dek = Fernet.generate_key()
            
            # Encrypt data with DEK
            fernet = Fernet(dek)
            
            # Serialize data (handle Decimal and datetime)
            json_data = json.dumps(data, default=self._json_serializer)
            encrypted_data = fernet.encrypt(json_data.encode())
            
            # Encrypt DEK with Vault (KEK)
            encrypted_dek = await self._encrypt_with_vault(
                dek.decode(),
                key_name
            )
            
            return {
                "ciphertext": base64.b64encode(encrypted_data).decode(),
                "encrypted_key": encrypted_dek,
                "key_name": key_name,
                "version": "1"
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityException(f"Failed to encrypt data: {e}")
    
    async def decrypt_data(
        self,
        encrypted_envelope: Dict[str, str]
    ) -> Any:
        """Decrypt data from envelope encryption.
        
        Args:
            encrypted_envelope: Encrypted envelope containing data and DEK
            
        Returns:
            Decrypted data
        """
        try:
            # Extract components
            ciphertext = base64.b64decode(encrypted_envelope["ciphertext"])
            encrypted_dek = encrypted_envelope["encrypted_key"]
            key_name = encrypted_envelope.get("key_name", "genesis-data")
            
            # Decrypt DEK with Vault
            dek = await self._decrypt_with_vault(encrypted_dek, key_name)
            
            # Decrypt data with DEK
            fernet = Fernet(dek.encode())
            decrypted_json = fernet.decrypt(ciphertext)
            
            # Deserialize data
            return json.loads(decrypted_json.decode())
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityException(f"Failed to decrypt data: {e}")
    
    async def _encrypt_with_vault(
        self,
        plaintext: str,
        key_name: str
    ) -> str:
        """Encrypt data using Vault's transit engine.
        
        Args:
            plaintext: Data to encrypt
            key_name: Transit key name
            
        Returns:
            Encrypted ciphertext
        """
        try:
            # Encode plaintext to base64 (Vault requirement)
            encoded = base64.b64encode(plaintext.encode()).decode()
            
            # Encrypt with Vault transit engine
            response = await self.vault._client.secrets.transit.encrypt_data(
                name=key_name,
                plaintext=encoded,
                mount_point=self.vault.config.transit_mount_point
            )
            
            return response["data"]["ciphertext"]
            
        except Exception as e:
            raise SecurityException(f"Vault encryption failed: {e}")
    
    async def _decrypt_with_vault(
        self,
        ciphertext: str,
        key_name: str
    ) -> str:
        """Decrypt data using Vault's transit engine.
        
        Args:
            ciphertext: Encrypted data from Vault
            key_name: Transit key name
            
        Returns:
            Decrypted plaintext
        """
        try:
            # Decrypt with Vault transit engine
            response = await self.vault._client.secrets.transit.decrypt_data(
                name=key_name,
                ciphertext=ciphertext,
                mount_point=self.vault.config.transit_mount_point
            )
            
            # Decode from base64
            encoded = response["data"]["plaintext"]
            return base64.b64decode(encoded).decode()
            
        except Exception as e:
            raise SecurityException(f"Vault decryption failed: {e}")
    
    async def encrypt_field(
        self,
        value: Any,
        field_name: str
    ) -> str:
        """Encrypt a single field value.
        
        Args:
            value: Field value to encrypt
            field_name: Field name for context
            
        Returns:
            Encrypted field value as string
        """
        try:
            # Use field-specific key if configured
            key_name = f"genesis-field-{field_name}"
            
            # Simple encryption for field-level data
            encrypted = await self.encrypt_data(
                {"value": value, "field": field_name},
                key_name
            )
            
            # Return as compact string
            return json.dumps(encrypted)
            
        except Exception as e:
            logger.error(f"Field encryption failed for {field_name}: {e}")
            raise SecurityException(f"Failed to encrypt field: {e}")
    
    async def decrypt_field(
        self,
        encrypted_value: str,
        field_name: str
    ) -> Any:
        """Decrypt a single field value.
        
        Args:
            encrypted_value: Encrypted field value
            field_name: Field name for context
            
        Returns:
            Decrypted field value
        """
        try:
            # Parse encrypted envelope
            envelope = json.loads(encrypted_value)
            
            # Decrypt data
            decrypted = await self.decrypt_data(envelope)
            
            # Extract field value
            return decrypted.get("value")
            
        except Exception as e:
            logger.error(f"Field decryption failed for {field_name}: {e}")
            raise SecurityException(f"Failed to decrypt field: {e}")
    
    async def rotate_encryption_key(
        self,
        key_name: str = "genesis-data"
    ) -> None:
        """Rotate encryption key in Vault.
        
        Args:
            key_name: Key name to rotate
        """
        try:
            # Rotate key in Vault
            await self.vault._client.secrets.transit.rotate_key(
                name=key_name,
                mount_point=self.vault.config.transit_mount_point
            )
            
            # Clear any cached data keys
            self._data_key_cache.clear()
            
            logger.info(f"Encryption key rotated: {key_name}")
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise SecurityException(f"Failed to rotate encryption key: {e}")
    
    async def rewrap_data(
        self,
        encrypted_envelope: Dict[str, str]
    ) -> Dict[str, str]:
        """Rewrap encrypted data with latest key version.
        
        Args:
            encrypted_envelope: Existing encrypted envelope
            
        Returns:
            Rewrapped envelope with latest key version
        """
        try:
            key_name = encrypted_envelope.get("key_name", "genesis-data")
            
            # Rewrap the encrypted DEK
            response = await self.vault._client.secrets.transit.rewrap_data(
                name=key_name,
                ciphertext=encrypted_envelope["encrypted_key"],
                mount_point=self.vault.config.transit_mount_point
            )
            
            # Update envelope with new encrypted DEK
            new_envelope = encrypted_envelope.copy()
            new_envelope["encrypted_key"] = response["data"]["ciphertext"]
            new_envelope["version"] = str(response["data"]["version"])
            
            return new_envelope
            
        except Exception as e:
            logger.error(f"Data rewrap failed: {e}")
            raise SecurityException(f"Failed to rewrap data: {e}")
    
    def _json_serializer(self, obj: Any) -> Any:
        """JSON serializer for special types."""
        if isinstance(obj, Decimal):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")