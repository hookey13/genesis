"""
Integration tests for 2FA authentication flow.
Tests complete 2FA setup, login, and management workflows.
"""

import pytest
import pyotp
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
import jwt
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import bcrypt

from genesis.data.models_db import Base, User, TwoFAAttempt
from genesis.security.totp_manager import TOTPManager
from genesis.security.vault_client import VaultClient
from genesis.api.auth.two_fa import router as two_fa_router
from genesis.api.auth.login import router as login_router


@pytest.fixture
async def test_db():
    """Create test database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.fixture
async def test_user(test_db):
    """Create test user."""
    password_hash = bcrypt.hashpw(b"TestPassword123!", bcrypt.gensalt()).decode()
    
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash=password_hash,
        role="user",
        two_fa_enabled=False,
        two_fa_required=False,
        created_at=datetime.now(timezone.utc)
    )
    
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    return user


@pytest.fixture
async def admin_user(test_db):
    """Create admin user."""
    password_hash = bcrypt.hashpw(b"AdminPassword123!", bcrypt.gensalt()).decode()
    
    user = User(
        username="admin",
        email="admin@example.com",
        password_hash=password_hash,
        role="admin",
        two_fa_enabled=False,
        two_fa_required=True,  # Admin requires 2FA
        created_at=datetime.now(timezone.utc)
    )
    
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    
    return user


@pytest.fixture
def mock_vault():
    """Create mock vault client."""
    vault = AsyncMock(spec=VaultClient)
    vault.store_secret = AsyncMock(return_value=True)
    vault.get_secret = AsyncMock()
    vault.delete_secret = AsyncMock(return_value=True)
    return vault


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    redis.hset = AsyncMock(return_value=True)
    redis.expire = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    return redis


class TestComplete2FAFlow:
    """Test complete 2FA authentication flow."""
    
    @pytest.mark.asyncio
    async def test_setup_2fa_flow(self, test_user, mock_vault):
        """Test complete 2FA setup flow."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        # Step 1: Initiate 2FA setup
        setup_data = await totp_manager.setup_2fa(
            user_id=str(test_user.id),
            username=test_user.username
        )
        
        assert setup_data['secret'] is not None
        assert setup_data['qr_code'] is not None
        assert len(setup_data['backup_codes']) == 10
        assert setup_data['setup_complete'] is False
        
        # Verify vault storage
        mock_vault.store_secret.assert_called_once()
        vault_call = mock_vault.store_secret.call_args[0]
        assert vault_call[0] == f"genesis-secrets/2fa/{test_user.id}"
        assert vault_call[1]['enabled'] is False
        
        # Step 2: Enable 2FA with verification
        secret = setup_data['secret']
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': False,
            'backup_codes': []
        }
        
        enabled = await totp_manager.enable_2fa(
            user_id=str(test_user.id),
            verification_code=valid_code
        )
        
        assert enabled is True
        
        # Verify vault update
        assert mock_vault.store_secret.call_count == 2
        updated_data = mock_vault.store_secret.call_args_list[1][0][1]
        assert updated_data['enabled'] is True
    
    @pytest.mark.asyncio
    async def test_login_with_2fa(self, test_user, mock_vault, mock_redis):
        """Test login flow with 2FA enabled."""
        # Setup 2FA for user
        secret = pyotp.random_base32()
        test_user.two_fa_enabled = True
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': True,
            'backup_codes': [],
            'failure_count': 0
        }
        
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        
        # Simulate login with 2FA
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        # Verify 2FA during login
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(test_user.id),
            code=valid_code
        )
        
        assert is_valid is True
        assert code_type == "totp"
        
        # Verify vault update for last_used
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert 'last_used' in updated_data
        assert updated_data['failure_count'] == 0
    
    @pytest.mark.asyncio
    async def test_backup_code_usage(self, test_user, mock_vault):
        """Test using backup code for authentication."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        # Generate backup codes
        backup_codes = totp_manager.generate_backup_codes()
        test_code = backup_codes[0]
        hashed_codes = [totp_manager.hash_backup_code(c) for c in backup_codes]
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': hashed_codes,
            'failure_count': 0
        }
        
        # Use backup code
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(test_user.id),
            code=test_code
        )
        
        assert is_valid is True
        assert code_type == "backup"
        
        # Verify used code is removed
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert len(updated_data['backup_codes']) == 9
        used_hash = totp_manager.hash_backup_code(test_code)
        assert used_hash not in updated_data['backup_codes']
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_user, test_db, mock_vault):
        """Test 2FA attempt rate limiting."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': [],
            'failure_count': 0
        }
        
        # Simulate multiple failed attempts
        for i in range(5):
            attempt = TwoFAAttempt(
                user_id=test_user.id,
                ip_address="192.168.1.1",
                code_used="999***",
                success=False,
                attempted_at=datetime.now(timezone.utc)
            )
            test_db.add(attempt)
        
        await test_db.commit()
        
        # Check rate limit
        from genesis.api.auth.two_fa import check_rate_limit
        
        is_allowed = await check_rate_limit(
            test_db,
            test_user.id,
            "192.168.1.1"
        )
        
        assert is_allowed is False
    
    @pytest.mark.asyncio
    async def test_admin_2fa_enforcement(self, admin_user, mock_vault):
        """Test that admin accounts require 2FA."""
        assert admin_user.two_fa_required is True
        
        # Admin should not be able to login without 2FA setup
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        # Setup 2FA for admin
        setup_data = await totp_manager.setup_2fa(
            user_id=str(admin_user.id),
            username=admin_user.username
        )
        
        secret = setup_data['secret']
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': False,
            'backup_codes': []
        }
        
        # Enable 2FA for admin
        enabled = await totp_manager.enable_2fa(
            user_id=str(admin_user.id),
            verification_code=valid_code
        )
        
        assert enabled is True
    
    @pytest.mark.asyncio
    async def test_disable_2fa(self, test_user, mock_vault):
        """Test disabling 2FA."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        # Setup and enable 2FA first
        secret = pyotp.random_base32()
        test_user.two_fa_enabled = True
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': True,
            'backup_codes': [],
            'failure_count': 0
        }
        
        # Disable 2FA
        disabled = await totp_manager.disable_2fa(user_id=str(test_user.id))
        
        assert disabled is True
        mock_vault.delete_secret.assert_called_once_with(
            f"genesis-secrets/2fa/{test_user.id}"
        )
    
    @pytest.mark.asyncio
    async def test_regenerate_backup_codes(self, test_user, mock_vault):
        """Test regenerating backup codes."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': ["old_hash1", "old_hash2"]
        }
        
        # Regenerate codes
        new_codes = await totp_manager.regenerate_backup_codes(
            user_id=str(test_user.id)
        )
        
        assert new_codes is not None
        assert len(new_codes) == 10
        
        # Verify vault update
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert len(updated_data['backup_codes']) == 10
        assert 'backup_codes_regenerated_at' in updated_data
        
        # Old codes should be replaced
        assert "old_hash1" not in updated_data['backup_codes']
        assert "old_hash2" not in updated_data['backup_codes']
    
    @pytest.mark.asyncio
    async def test_time_window_tolerance(self, test_user, mock_vault):
        """Test TOTP verification with clock skew tolerance."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        secret = pyotp.random_base32()
        
        mock_vault.get_secret.return_value = {
            'secret': secret,
            'enabled': True,
            'backup_codes': [],
            'failure_count': 0
        }
        
        totp = pyotp.TOTP(secret)
        
        # Get code for 30 seconds ago (within window)
        past_time = int(time.time()) - 30
        past_code = totp.at(past_time)
        
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(test_user.id),
            code=past_code
        )
        
        assert is_valid is True
        assert code_type == "totp"
    
    @pytest.mark.asyncio
    async def test_invalid_2fa_increments_failure_count(self, test_user, mock_vault):
        """Test that invalid 2FA attempts increment failure count."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        mock_vault.get_secret.return_value = {
            'secret': pyotp.random_base32(),
            'enabled': True,
            'backup_codes': [],
            'failure_count': 2
        }
        
        # Try invalid code
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(test_user.id),
            code="999999"
        )
        
        assert is_valid is False
        assert code_type == "invalid"
        
        # Verify failure count incremented
        updated_data = mock_vault.store_secret.call_args[0][1]
        assert updated_data['failure_count'] == 3
    
    @pytest.mark.asyncio
    async def test_2fa_not_enabled_response(self, test_user, mock_vault):
        """Test response when 2FA is not enabled."""
        totp_manager = TOTPManager(vault_manager=mock_vault)
        
        mock_vault.get_secret.return_value = {
            'enabled': False
        }
        
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(test_user.id),
            code="123456"
        )
        
        assert is_valid is False
        assert code_type == "not_enabled"