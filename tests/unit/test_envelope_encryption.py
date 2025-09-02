"""Unit tests for Envelope Encryption."""

import base64
import json
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from cryptography.fernet import Fernet

from genesis.security.envelope_encryption import EnvelopeEncryption
from genesis.security.vault_manager import VaultManager
from genesis.core.exceptions import SecurityException


@pytest.fixture
def mock_vault_manager():
    """Create mock Vault manager."""
    vault = MagicMock(spec=VaultManager)
    vault._client = MagicMock()
    vault.config = MagicMock()
    vault.config.transit_mount_point = "transit"
    return vault


@pytest.fixture
def envelope_encryption(mock_vault_manager):
    """Create EnvelopeEncryption instance."""
    return EnvelopeEncryption(mock_vault_manager)


@pytest.mark.asyncio
async def test_encrypt_data(envelope_encryption, mock_vault_manager):
    """Test data encryption with envelope encryption."""
    test_data = {"secret": "value", "number": 42}
    
    # Mock Vault transit encryption
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        return_value={"data": {"ciphertext": "vault:v1:encrypted_dek"}}
    )
    
    result = await envelope_encryption.encrypt_data(test_data)
    
    assert "ciphertext" in result
    assert "encrypted_key" in result
    assert result["encrypted_key"] == "vault:v1:encrypted_dek"
    assert result["key_name"] == "genesis-data"
    assert result["version"] == "1"
    
    # Verify data was encrypted (not plaintext)
    assert result["ciphertext"] != json.dumps(test_data)


@pytest.mark.asyncio
async def test_decrypt_data(envelope_encryption, mock_vault_manager):
    """Test data decryption from envelope encryption."""
    # Create encrypted envelope
    original_data = {"secret": "value", "number": 42}
    
    # Generate DEK and encrypt data
    dek = Fernet.generate_key()
    fernet = Fernet(dek)
    encrypted_data = fernet.encrypt(json.dumps(original_data).encode())
    
    encrypted_envelope = {
        "ciphertext": base64.b64encode(encrypted_data).decode(),
        "encrypted_key": "vault:v1:encrypted_dek",
        "key_name": "genesis-data",
        "version": "1"
    }
    
    # Mock Vault transit decryption
    encoded_dek = base64.b64encode(dek).decode()
    mock_vault_manager._client.secrets.transit.decrypt_data = AsyncMock(
        return_value={"data": {"plaintext": encoded_dek}}
    )
    
    result = await envelope_encryption.decrypt_data(encrypted_envelope)
    
    assert result == original_data


@pytest.mark.asyncio
async def test_encrypt_with_custom_key(envelope_encryption, mock_vault_manager):
    """Test encryption with custom key name."""
    test_data = {"api_key": "secret123"}
    
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        return_value={"data": {"ciphertext": "vault:v1:custom_encrypted"}}
    )
    
    result = await envelope_encryption.encrypt_data(test_data, "custom-key")
    
    assert result["key_name"] == "custom-key"
    mock_vault_manager._client.secrets.transit.encrypt_data.assert_called_once()
    call_args = mock_vault_manager._client.secrets.transit.encrypt_data.call_args
    assert call_args.kwargs["name"] == "custom-key"


@pytest.mark.asyncio
async def test_encrypt_field(envelope_encryption, mock_vault_manager):
    """Test field-level encryption."""
    field_value = "sensitive_api_key_123"
    field_name = "api_secret"
    
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        return_value={"data": {"ciphertext": "vault:v1:field_encrypted"}}
    )
    
    result = await envelope_encryption.encrypt_field(field_value, field_name)
    
    # Result should be JSON string
    parsed = json.loads(result)
    assert "ciphertext" in parsed
    assert "encrypted_key" in parsed
    assert parsed["key_name"] == f"genesis-field-{field_name}"


@pytest.mark.asyncio
async def test_decrypt_field(envelope_encryption, mock_vault_manager):
    """Test field-level decryption."""
    field_value = "sensitive_value"
    field_name = "api_secret"
    
    # Create encrypted field
    dek = Fernet.generate_key()
    fernet = Fernet(dek)
    field_data = {"value": field_value, "field": field_name}
    encrypted = fernet.encrypt(json.dumps(field_data).encode())
    
    encrypted_field = json.dumps({
        "ciphertext": base64.b64encode(encrypted).decode(),
        "encrypted_key": "vault:v1:field_dek",
        "key_name": f"genesis-field-{field_name}",
        "version": "1"
    })
    
    # Mock Vault decryption
    encoded_dek = base64.b64encode(dek).decode()
    mock_vault_manager._client.secrets.transit.decrypt_data = AsyncMock(
        return_value={"data": {"plaintext": encoded_dek}}
    )
    
    result = await envelope_encryption.decrypt_field(encrypted_field, field_name)
    
    assert result == field_value


@pytest.mark.asyncio
async def test_json_serializer_decimal(envelope_encryption, mock_vault_manager):
    """Test JSON serialization of Decimal values."""
    test_data = {"price": Decimal("123.456"), "quantity": Decimal("10")}
    
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        return_value={"data": {"ciphertext": "vault:v1:encrypted"}}
    )
    
    # Should not raise serialization error
    result = await envelope_encryption.encrypt_data(test_data)
    assert "ciphertext" in result


@pytest.mark.asyncio
async def test_json_serializer_datetime(envelope_encryption, mock_vault_manager):
    """Test JSON serialization of datetime values."""
    now = datetime.utcnow()
    test_data = {"timestamp": now, "created_at": now}
    
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        return_value={"data": {"ciphertext": "vault:v1:encrypted"}}
    )
    
    # Should not raise serialization error
    result = await envelope_encryption.encrypt_data(test_data)
    assert "ciphertext" in result


@pytest.mark.asyncio
async def test_rotate_encryption_key(envelope_encryption, mock_vault_manager):
    """Test encryption key rotation."""
    mock_vault_manager._client.secrets.transit.rotate_key = AsyncMock()
    
    await envelope_encryption.rotate_encryption_key("test-key")
    
    mock_vault_manager._client.secrets.transit.rotate_key.assert_called_once_with(
        name="test-key",
        mount_point="transit"
    )
    # Cache should be cleared
    assert len(envelope_encryption._data_key_cache) == 0


@pytest.mark.asyncio
async def test_rewrap_data(envelope_encryption, mock_vault_manager):
    """Test rewrapping data with new key version."""
    encrypted_envelope = {
        "ciphertext": "base64_encrypted_data",
        "encrypted_key": "vault:v1:old_encrypted_dek",
        "key_name": "test-key",
        "version": "1"
    }
    
    mock_vault_manager._client.secrets.transit.rewrap_data = AsyncMock(
        return_value={
            "data": {
                "ciphertext": "vault:v2:new_encrypted_dek",
                "version": 2
            }
        }
    )
    
    result = await envelope_encryption.rewrap_data(encrypted_envelope)
    
    assert result["encrypted_key"] == "vault:v2:new_encrypted_dek"
    assert result["version"] == "2"
    assert result["ciphertext"] == encrypted_envelope["ciphertext"]  # Data unchanged


@pytest.mark.asyncio
async def test_encryption_error_handling(envelope_encryption, mock_vault_manager):
    """Test error handling during encryption."""
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        side_effect=Exception("Vault error")
    )
    
    with pytest.raises(SecurityException) as exc_info:
        await envelope_encryption.encrypt_data({"data": "test"})
    
    assert "Failed to encrypt data" in str(exc_info.value)


@pytest.mark.asyncio
async def test_decryption_error_handling(envelope_encryption, mock_vault_manager):
    """Test error handling during decryption."""
    encrypted_envelope = {
        "ciphertext": "invalid_data",
        "encrypted_key": "vault:v1:encrypted",
        "key_name": "test-key"
    }
    
    mock_vault_manager._client.secrets.transit.decrypt_data = AsyncMock(
        side_effect=Exception("Vault error")
    )
    
    with pytest.raises(SecurityException) as exc_info:
        await envelope_encryption.decrypt_data(encrypted_envelope)
    
    assert "Failed to decrypt data" in str(exc_info.value)


@pytest.mark.asyncio
async def test_complex_data_encryption(envelope_encryption, mock_vault_manager):
    """Test encryption of complex nested data structures."""
    complex_data = {
        "user": {
            "id": 123,
            "balance": Decimal("1000.50"),
            "created": datetime.utcnow(),
            "settings": {
                "api_keys": ["key1", "key2"],
                "limits": {"daily": 100, "monthly": 3000}
            }
        },
        "metadata": None,
        "active": True
    }
    
    mock_vault_manager._client.secrets.transit.encrypt_data = AsyncMock(
        return_value={"data": {"ciphertext": "vault:v1:complex_encrypted"}}
    )
    
    result = await envelope_encryption.encrypt_data(complex_data)
    
    assert "ciphertext" in result
    assert result["encrypted_key"] == "vault:v1:complex_encrypted"