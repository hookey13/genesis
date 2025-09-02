"""
Unit tests for TOTP Manager.
Tests TOTP generation, verification, and backup code functionality.
"""

import pytest
import pyotp
import base64
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import hashlib

from genesis.security.totp_manager import TOTPManager


class TestTOTPManager:
    """Test suite for TOTP Manager."""
    
    @pytest.fixture
    def totp_manager(self):
        """Create TOTP manager instance."""
        return TOTPManager(vault_manager=None, issuer_name="Test Genesis")
    
    @pytest.fixture
    def mock_vault(self):
        """Create mock vault manager."""
        vault = AsyncMock()
        vault.store_secret = AsyncMock(return_value=True)
        vault.get_secret = AsyncMock()
        vault.delete_secret = AsyncMock(return_value=True)
        return vault
    
    def test_generate_secret(self, totp_manager):
        """Test TOTP secret generation."""
        secret = totp_manager.generate_secret()
        
        # Should be base32 encoded
        assert secret is not None
        assert len(secret) == 32
        assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567' for c in secret)
        
        # Should be different each time
        secret2 = totp_manager.generate_secret()
        assert secret != secret2
    
    def test_generate_provisioning_uri(self, totp_manager):
        """Test provisioning URI generation."""
        secret = "JBSWY3DPEHPK3PXP"
        username = "testuser"
        
        uri = totp_manager.generate_provisioning_uri(username, secret)
        
        # Should be otpauth URI
        assert uri.startswith("otpauth://totp/")
        assert "Test%20Genesis" in uri  # Issuer name
        assert username in uri
        assert f"secret={secret}" in uri
    
    def test_generate_qr_code(self, totp_manager):
        """Test QR code generation."""
        uri = "otpauth://totp/Test%20Genesis:testuser?secret=JBSWY3DPEHPK3PXP&issuer=Test%20Genesis"
        
        qr_code = totp_manager.generate_qr_code(uri)
        
        # Should be base64 encoded
        assert qr_code is not None
        assert isinstance(qr_code, str)
        
        # Should be valid base64
        try:
            decoded = base64.b64decode(qr_code)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("QR code is not valid base64")
    
    def test_verify_totp_code_valid(self, totp_manager):
        """Test TOTP code verification with valid code."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        current_code = totp.now()
        
        # Should verify current code
        assert totp_manager.verify_totp_code(secret, current_code) is True
    
    def test_verify_totp_code_invalid(self, totp_manager):
        """Test TOTP code verification with invalid code."""
        secret = pyotp.random_base32()
        
        # Invalid format codes
        assert totp_manager.verify_totp_code(secret, "") is False
        assert totp_manager.verify_totp_code(secret, "12345") is False  # Too short
        assert totp_manager.verify_totp_code(secret, "1234567") is False  # Too long
        assert totp_manager.verify_totp_code(secret, "abcdef") is False  # Not digits
        assert totp_manager.verify_totp_code(secret, "999999") is False  # Wrong code
    
    def test_verify_totp_code_with_window(self, totp_manager):
        """Test TOTP verification with time window tolerance."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        # Get code for 30 seconds ago (within window)
        past_time = int(time.time()) - 30
        past_code = totp.at(past_time)
        
        # Should verify with window tolerance
        assert totp_manager.verify_totp_code(secret, past_code) is True
    
    def test_generate_backup_codes(self, totp_manager):
        """Test backup code generation."""
        codes = totp_manager.generate_backup_codes()
        
        # Should generate correct number of codes
        assert len(codes) == totp_manager.backup_codes_count
        
        # Each code should be formatted correctly
        for code in codes:
            assert len(code) == 9  # XXXX-XXXX format
            assert code[4] == '-'
            assert all(c in '0123456789ABCDEF-' for c in code)
        
        # Codes should be unique
        assert len(set(codes)) == len(codes)
    
    def test_hash_backup_code(self, totp_manager):
        """Test backup code hashing."""
        code = "1234-5678"
        hashed = totp_manager.hash_backup_code(code)
        
        # Should return SHA256 hash
        assert len(hashed) == 64  # SHA256 hex length
        assert all(c in '0123456789abcdef' for c in hashed)
        
        # Should be consistent
        hashed2 = totp_manager.hash_backup_code(code)
        assert hashed == hashed2
        
        # Should handle formatting
        hashed3 = totp_manager.hash_backup_code("12345678")
        assert hashed == hashed3
    
    def test_verify_backup_code_valid(self, totp_manager):
        """Test backup code verification with valid code."""
        code = "1234-5678"
        hashed = totp_manager.hash_backup_code(code)
        hashed_codes = [hashed, "other_hash"]
        
        matched = totp_manager.verify_backup_code(code, hashed_codes)
        assert matched == hashed
    
    def test_verify_backup_code_invalid(self, totp_manager):
        """Test backup code verification with invalid code."""
        hashed_codes = ["hash1", "hash2"]
        
        matched = totp_manager.verify_backup_code("WRONG-CODE", hashed_codes)
        assert matched is None
    
    @pytest.mark.asyncio
    async def test_setup_2fa(self, mock_vault):
        """Test 2FA setup process."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        username = "testuser"
        
        result = await totp_manager.setup_2fa(user_id, username)
        
        # Should return setup data
        assert 'secret' in result
        assert 'qr_code' in result
        assert 'provisioning_uri' in result
        assert 'backup_codes' in result
        assert result['setup_complete'] is False
        
        # Should store in vault
        mock_vault.store_secret.assert_called_once()
        call_args = mock_vault.store_secret.call_args
        assert call_args[0][0] == f"genesis-secrets/2fa/{user_id}"
        
        vault_data = call_args[0][1]
        assert 'secret' in vault_data
        assert 'backup_codes' in vault_data
        assert vault_data['enabled'] is False
    
    @pytest.mark.asyncio
    async def test_enable_2fa_success(self, mock_vault):
        """Test enabling 2FA with valid code."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        secret = pyotp.random_base32()
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': False,
            'backup_codes': []
        }
        
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        
        result = await totp_manager.enable_2fa(user_id, valid_code)
        assert result is True
        
        # Should update vault
        assert mock_vault.store_secret.called
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert updated_data['enabled'] is True
        assert 'enabled_at' in updated_data
    
    @pytest.mark.asyncio
    async def test_enable_2fa_invalid_code(self, mock_vault):
        """Test enabling 2FA with invalid code."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': False,
            'backup_codes': []
        }
        
        result = await totp_manager.enable_2fa(user_id, "999999")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_disable_2fa(self, mock_vault):
        """Test disabling 2FA."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        
        result = await totp_manager.disable_2fa(user_id)
        assert result is True
        
        mock_vault.delete_secret.assert_called_once_with(
            f"genesis-secrets/2fa/{user_id}"
        )
    
    @pytest.mark.asyncio
    async def test_validate_2fa_totp(self, mock_vault):
        """Test 2FA validation with TOTP code."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        secret = pyotp.random_base32()
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': True,
            'backup_codes': [],
            'failure_count': 0
        }
        
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        
        is_valid, code_type = await totp_manager.validate_2fa(user_id, valid_code)
        assert is_valid is True
        assert code_type == "totp"
        
        # Should update last_used and reset failure count
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert 'last_used' in updated_data
        assert updated_data['failure_count'] == 0
    
    @pytest.mark.asyncio
    async def test_validate_2fa_backup_code(self, mock_vault):
        """Test 2FA validation with backup code."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        backup_code = "1234-5678"
        hashed_code = totp_manager.hash_backup_code(backup_code)
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': [hashed_code, "other_hash"],
            'failure_count': 0
        }
        
        is_valid, code_type = await totp_manager.validate_2fa(user_id, backup_code)
        assert is_valid is True
        assert code_type == "backup"
        
        # Should remove used backup code
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert hashed_code not in updated_data['backup_codes']
        assert len(updated_data['backup_codes']) == 1
    
    @pytest.mark.asyncio
    async def test_validate_2fa_invalid(self, mock_vault):
        """Test 2FA validation with invalid code."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': [],
            'failure_count': 2
        }
        
        is_valid, code_type = await totp_manager.validate_2fa(user_id, "999999")
        assert is_valid is False
        assert code_type == "invalid"
        
        # Should increment failure count
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert updated_data['failure_count'] == 3
    
    @pytest.mark.asyncio
    async def test_validate_2fa_not_enabled(self, mock_vault):
        """Test 2FA validation when not enabled."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        
        mock_vault.get_secret.return_value = {
            'enabled': False
        }
        
        is_valid, code_type = await totp_manager.validate_2fa(user_id, "123456")
        assert is_valid is False
        assert code_type == "not_enabled"
    
    @pytest.mark.asyncio
    async def test_regenerate_backup_codes(self, mock_vault):
        """Test regenerating backup codes."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        user_id = "user123"
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': ["old_hash"]
        }
        
        new_codes = await totp_manager.regenerate_backup_codes(user_id)
        
        # Should return new codes
        assert new_codes is not None
        assert len(new_codes) == totp_manager.backup_codes_count
        
        # Should update vault with hashed codes
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert len(updated_data['backup_codes']) == totp_manager.backup_codes_count
        assert 'backup_codes_regenerated_at' in updated_data
    
    def test_get_time_remaining(self, totp_manager):
        """Test getting time remaining in TOTP window."""
        remaining = totp_manager.get_time_remaining()
        
        # Should be between 0 and 30 seconds
        assert 0 < remaining <= 30
        
        # Should decrease over time
        time.sleep(1)
        remaining2 = totp_manager.get_time_remaining()
        assert remaining2 < remaining or remaining2 == 30