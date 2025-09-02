"""Unit tests for JWT Session Management System."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import jwt
import secrets
import redis.asyncio as redis
from genesis.security.jwt_manager import (
    JWTSessionManager,
    TokenClaims,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenRevokedError
)


class TestJWTSessionManager:
    """Test suite for JWT session management."""
    
    @pytest.fixture
    async def vault_mock(self):
        """Mock Vault client."""
        vault = AsyncMock()
        vault.get_secret = AsyncMock(return_value="test_signing_key_secret_123")
        return vault
    
    @pytest.fixture
    async def redis_mock(self):
        """Mock Redis client."""
        redis_client = AsyncMock(spec=redis.Redis)
        redis_client.hset = AsyncMock()
        redis_client.expire = AsyncMock()
        redis_client.sadd = AsyncMock()
        redis_client.hgetall = AsyncMock()
        redis_client.exists = AsyncMock(return_value=False)
        redis_client.setex = AsyncMock()
        redis_client.srem = AsyncMock()
        redis_client.smembers = AsyncMock(return_value=set())
        redis_client.delete = AsyncMock()
        return redis_client
    
    @pytest.fixture
    async def jwt_manager(self, vault_mock, redis_mock):
        """Create JWT manager instance."""
        return JWTSessionManager(vault_mock, redis_mock)
    
    async def test_token_generation(self, jwt_manager, vault_mock, redis_mock):
        """Test JWT token pair generation."""
        # Generate tokens
        tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Verify structure
        assert 'access_token' in tokens
        assert 'refresh_token' in tokens
        assert tokens['token_type'] == 'Bearer'
        assert tokens['expires_in'] == 3600
        assert 'session_id' in tokens
        
        # Verify tokens are different
        assert tokens['access_token'] != tokens['refresh_token']
        
        # Verify Vault was called for signing key
        vault_mock.get_secret.assert_called_once_with('genesis-secrets/jwt', 'signing_key')
        
        # Verify Redis storage was called
        assert redis_mock.hset.called
        assert redis_mock.expire.called
        assert redis_mock.sadd.called
        
        # Decode and verify token claims
        signing_key = "test_signing_key_secret_123"
        access_payload = jwt.decode(
            tokens['access_token'],
            signing_key,
            algorithms=['HS256']
        )
        
        assert access_payload['sub'] == "123"
        assert access_payload['username'] == "testuser"
        assert access_payload['role'] == "trader"
        assert access_payload['type'] == 'access'
        assert 'jti' in access_payload
        assert 'session_id' in access_payload
        
        refresh_payload = jwt.decode(
            tokens['refresh_token'],
            signing_key,
            algorithms=['HS256']
        )
        
        assert refresh_payload['sub'] == "123"
        assert refresh_payload['username'] == "testuser"
        assert refresh_payload['type'] == 'refresh'
        assert refresh_payload['session_id'] == access_payload['session_id']
    
    async def test_token_verification_success(self, jwt_manager, redis_mock):
        """Test successful token verification."""
        # Generate a token first
        tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Mock Redis responses for verification
        redis_mock.exists.side_effect = [False, True]  # Not blacklisted, session exists
        redis_mock.hgetall.return_value = {
            b'is_active': b'true',
            b'user_id': b'123',
            b'username': b'testuser'
        }
        
        # Verify access token
        claims = await jwt_manager.verify_token(tokens['access_token'])
        
        assert isinstance(claims, TokenClaims)
        assert claims.user_id == "123"
        assert claims.username == "testuser"
        assert claims.role == "trader"
        assert claims.token_id is not None
    
    async def test_token_verification_expired(self, jwt_manager):
        """Test verification of expired token."""
        # Create an expired token
        signing_key = await jwt_manager.get_signing_key()
        now = datetime.now(timezone.utc)
        expired_claims = {
            'sub': '123',
            'username': 'testuser',
            'role': 'trader',
            'session_id': 'test_session',
            'iat': now - timedelta(hours=2),
            'exp': now - timedelta(hours=1),  # Expired 1 hour ago
            'nbf': now - timedelta(hours=2),
            'jti': 'test_token_id',
            'type': 'access'
        }
        
        expired_token = jwt.encode(expired_claims, signing_key, algorithm='HS256')
        
        # Should raise TokenExpiredError
        with pytest.raises(TokenExpiredError) as exc_info:
            await jwt_manager.verify_token(expired_token)
        
        assert "Token has expired" in str(exc_info.value)
    
    async def test_token_verification_blacklisted(self, jwt_manager, redis_mock):
        """Test verification of blacklisted token."""
        # Generate a token
        tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Mock token as blacklisted
        redis_mock.exists.side_effect = [True]  # Token is blacklisted
        
        # Should raise TokenInvalidError (wrapped by exception handler)
        with pytest.raises(TokenInvalidError) as exc_info:
            await jwt_manager.verify_token(tokens['access_token'])
        
        assert "Token has been revoked" in str(exc_info.value)
    
    async def test_token_verification_invalid_type(self, jwt_manager, redis_mock):
        """Test verification with wrong token type."""
        # Generate tokens
        tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Mock Redis responses
        redis_mock.exists.return_value = False
        
        # Try to verify refresh token as access token
        with pytest.raises(TokenInvalidError) as exc_info:
            await jwt_manager.verify_token(tokens['refresh_token'], 'access')
        
        assert "Expected access token" in str(exc_info.value)
    
    async def test_token_refresh(self, jwt_manager, redis_mock):
        """Test token refresh mechanism."""
        # Generate initial tokens
        initial_tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Mock Redis for refresh token verification
        redis_mock.exists.side_effect = [False, True, False, True]  # For both verifications
        redis_mock.hgetall.return_value = {
            b'is_active': b'true',
            b'user_id': b'123',
            b'username': b'testuser',
            b'role': b'trader'
        }
        
        # Refresh tokens
        new_tokens = await jwt_manager.refresh_token(initial_tokens['refresh_token'])
        
        assert 'access_token' in new_tokens
        assert 'refresh_token' in new_tokens
        assert new_tokens['access_token'] != initial_tokens['access_token']
        assert new_tokens['refresh_token'] != initial_tokens['refresh_token']
        
        # Verify old refresh token was blacklisted
        assert redis_mock.setex.called
    
    async def test_session_invalidation(self, jwt_manager, redis_mock):
        """Test session invalidation."""
        session_id = "test_session_123"
        
        # Mock session data
        redis_mock.hgetall.return_value = {
            b'user_id': b'123',
            b'username': b'testuser',
            b'access_token_id': b'access_123',
            b'refresh_token_id': b'refresh_123'
        }
        
        # Invalidate session
        await jwt_manager.invalidate_session(session_id)
        
        # Verify blacklisting was called for both tokens
        blacklist_calls = redis_mock.setex.call_args_list
        assert len(blacklist_calls) >= 2
        
        # Verify session was marked inactive
        redis_mock.hset.assert_called_with(
            f"session:{session_id}",
            'is_active',
            'false'
        )
        
        # Verify session was removed from user sessions
        redis_mock.srem.assert_called()
    
    async def test_invalidate_all_user_sessions(self, jwt_manager, redis_mock):
        """Test invalidating all sessions for a user."""
        user_id = "123"
        session_ids = [b'session1', b'session2', b'session3']
        
        # Mock user sessions
        redis_mock.smembers.return_value = session_ids
        redis_mock.hgetall.return_value = {
            b'user_id': b'123',
            b'access_token_id': b'token_id',
            b'refresh_token_id': b'refresh_id'
        }
        
        # Invalidate all sessions
        await jwt_manager.invalidate_all_user_sessions(user_id)
        
        # Verify each session was invalidated
        assert redis_mock.hgetall.call_count >= len(session_ids)
        
        # Verify user sessions set was deleted
        redis_mock.delete.assert_called_with(f"user_sessions:{user_id}")
    
    async def test_get_active_sessions(self, jwt_manager, redis_mock):
        """Test retrieving active sessions for a user."""
        user_id = "123"
        session_ids = [b'session1', b'session2']
        
        # Mock user sessions
        redis_mock.smembers.return_value = session_ids
        
        # Mock session data
        session_data = {
            b'is_active': b'true',
            b'user_id': b'123',
            b'username': b'testuser',
            b'created_at': b'2024-01-01T00:00:00'
        }
        redis_mock.hgetall.return_value = session_data
        
        # Get active sessions
        sessions = await jwt_manager.get_active_sessions(user_id)
        
        assert len(sessions) == 2
        for session in sessions:
            assert session['user_id'] == '123'
            assert session['username'] == 'testuser'
            assert 'session_id' in session
            assert 'created_at' in session
    
    async def test_token_blacklist(self, jwt_manager, redis_mock):
        """Test adding token to blacklist."""
        token_id = "test_token_id_123"
        
        # Blacklist token
        await jwt_manager.blacklist_token(token_id)
        
        # Verify Redis setex was called
        redis_mock.setex.assert_called_once()
        call_args = redis_mock.setex.call_args
        
        # Verify correct key and TTL
        assert f"blacklist:{token_id}" in call_args[0]
        assert call_args[0][1] > 0  # TTL should be positive
        assert call_args[0][2] == "revoked"
    
    async def test_create_token_hash(self, jwt_manager):
        """Test token hash creation for logging."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
        
        # Create hash
        token_hash = jwt_manager.create_token_hash(token)
        
        # Verify hash properties
        assert len(token_hash) == 16  # Truncated to 16 chars
        assert token_hash.isalnum()  # Should be alphanumeric
        
        # Same token should produce same hash
        assert token_hash == jwt_manager.create_token_hash(token)
        
        # Different token should produce different hash
        different_token = "different.token.here"
        assert token_hash != jwt_manager.create_token_hash(different_token)
    
    async def test_session_no_longer_exists(self, jwt_manager, redis_mock):
        """Test verification when session no longer exists."""
        # Generate a token
        tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Mock session as not existing
        redis_mock.exists.side_effect = [False, False]  # Not blacklisted, session doesn't exist
        
        # Should raise TokenInvalidError
        with pytest.raises(TokenInvalidError) as exc_info:
            await jwt_manager.verify_token(tokens['access_token'])
        
        assert "Session no longer exists" in str(exc_info.value)
    
    async def test_session_inactive(self, jwt_manager, redis_mock):
        """Test verification when session is inactive."""
        # Generate a token
        tokens = await jwt_manager.generate_token_pair(
            "123", "testuser", "trader"
        )
        
        # Mock session as inactive
        redis_mock.exists.side_effect = [False, True]  # Not blacklisted, session exists
        redis_mock.hgetall.return_value = {
            b'is_active': b'false',  # Session is inactive
            b'user_id': b'123'
        }
        
        # Should raise TokenInvalidError
        with pytest.raises(TokenInvalidError) as exc_info:
            await jwt_manager.verify_token(tokens['access_token'])
        
        assert "Session is inactive" in str(exc_info.value)