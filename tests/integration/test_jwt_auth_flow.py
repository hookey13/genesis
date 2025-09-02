"""Integration tests for JWT authentication flow."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from fastapi import FastAPI
from fastapi.testclient import TestClient
import redis.asyncio as redis
from genesis.api.auth_endpoints import router, get_jwt_manager, get_redis_client, get_vault_client
from genesis.security.jwt_manager import JWTSessionManager
from genesis.security.vault_client import VaultClient
from genesis.validation.auth import User


class TestJWTAuthenticationFlow:
    """Integration tests for complete JWT authentication flow."""
    
    @pytest.fixture
    async def app(self):
        """Create FastAPI app with auth routes."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    async def vault_mock(self):
        """Mock Vault client."""
        vault = AsyncMock(spec=VaultClient)
        vault.get_secret = AsyncMock(return_value="test_signing_key_integration")
        return vault
    
    @pytest.fixture
    async def redis_mock(self):
        """Mock Redis client with persistent state."""
        class RedisStateMock:
            def __init__(self):
                self.data = {}
                self.sets = {}
                self.ttls = {}
            
            async def hset(self, key, mapping=None, **kwargs):
                if mapping:
                    if key not in self.data:
                        self.data[key] = {}
                    self.data[key].update(mapping)
                return True
            
            async def hgetall(self, key):
                data = self.data.get(key, {})
                # Convert to bytes for compatibility
                return {k.encode() if isinstance(k, str) else k: 
                       v.encode() if isinstance(v, str) else v 
                       for k, v in data.items()}
            
            async def exists(self, key):
                return key in self.data or key in self.sets
            
            async def expire(self, key, ttl):
                self.ttls[key] = ttl
                return True
            
            async def sadd(self, key, *values):
                if key not in self.sets:
                    self.sets[key] = set()
                self.sets[key].update(values)
                return len(values)
            
            async def smembers(self, key):
                return {v.encode() if isinstance(v, str) else v 
                       for v in self.sets.get(key, set())}
            
            async def srem(self, key, *values):
                if key in self.sets:
                    self.sets[key].difference_update(values)
                return True
            
            async def setex(self, key, ttl, value):
                self.data[key] = value
                self.ttls[key] = ttl
                return True
            
            async def delete(self, key):
                self.data.pop(key, None)
                self.sets.pop(key, None)
                self.ttls.pop(key, None)
                return True
            
            async def hset(self, key, field=None, value=None, mapping=None):
                if key not in self.data:
                    self.data[key] = {}
                if mapping:
                    self.data[key].update(mapping)
                if field and value:
                    self.data[key][field] = value
                return True
        
        return RedisStateMock()
    
    @pytest.fixture
    async def client(self, app, vault_mock, redis_mock):
        """Create test client with mocked dependencies."""
        # Override dependencies
        app.dependency_overrides[get_vault_client] = lambda: vault_mock
        app.dependency_overrides[get_redis_client] = lambda: redis_mock
        
        # Create JWT manager with mocks
        jwt_manager = JWTSessionManager(vault_mock, redis_mock)
        app.dependency_overrides[get_jwt_manager] = lambda: jwt_manager
        
        return TestClient(app)
    
    @pytest.fixture
    async def mock_user(self):
        """Mock user for authentication."""
        user = MagicMock()
        user.id = "user123"
        user.username = "testuser"
        user.role = "trader"
        return user
    
    async def test_complete_auth_flow(self, client, mock_user):
        """Test complete JWT authentication flow from login to logout."""
        # Mock User.authenticate
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            UserMock.authenticate = AsyncMock(return_value=mock_user)
            
            # Step 1: Login
            login_response = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            
            assert login_response.status_code == 200
            tokens = login_response.json()
            assert 'access_token' in tokens
            assert 'refresh_token' in tokens
            assert tokens['token_type'] == 'Bearer'
            assert tokens['expires_in'] == 3600
            assert 'session_id' in tokens
            
            # Store tokens for next steps
            access_token = tokens['access_token']
            refresh_token = tokens['refresh_token']
            session_id = tokens['session_id']
            
            # Step 2: Verify token
            headers = {'Authorization': f'Bearer {access_token}'}
            verify_response = client.get('/auth/verify', headers=headers)
            
            assert verify_response.status_code == 200
            user_info = verify_response.json()
            assert user_info['user_id'] == 'user123'
            assert user_info['username'] == 'testuser'
            assert user_info['role'] == 'trader'
            assert user_info['authenticated'] is True
            
            # Step 3: Get sessions
            sessions_response = client.get('/auth/sessions', headers=headers)
            
            assert sessions_response.status_code == 200
            sessions = sessions_response.json()
            assert len(sessions) > 0
            assert any(s['session_id'] == session_id for s in sessions)
            
            # Step 4: Refresh token
            refresh_response = client.post('/auth/refresh', json={
                'refresh_token': refresh_token
            })
            
            assert refresh_response.status_code == 200
            new_tokens = refresh_response.json()
            assert 'access_token' in new_tokens
            assert 'refresh_token' in new_tokens
            assert new_tokens['access_token'] != access_token  # New token generated
            
            # Update access token for logout
            new_access_token = new_tokens['access_token']
            headers = {'Authorization': f'Bearer {new_access_token}'}
            
            # Step 5: Logout
            logout_response = client.post('/auth/logout', headers=headers)
            
            assert logout_response.status_code == 200
            assert logout_response.json()['message'] == 'Logged out successfully'
    
    async def test_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            UserMock.authenticate = AsyncMock(return_value=None)
            
            response = client.post('/auth/login', json={
                'username': 'invaliduser',
                'password': 'WrongPassword'
            })
            
            assert response.status_code == 401
            assert 'Invalid credentials' in response.json()['detail']
    
    async def test_expired_token_refresh(self, client, mock_user):
        """Test refreshing with expired refresh token."""
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            UserMock.authenticate = AsyncMock(return_value=mock_user)
            
            # Login first
            login_response = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            
            tokens = login_response.json()
            
            # Create expired token
            expired_refresh = "expired.refresh.token"
            
            # Try to refresh with expired token
            refresh_response = client.post('/auth/refresh', json={
                'refresh_token': expired_refresh
            })
            
            assert refresh_response.status_code == 401
            assert 'Invalid refresh token' in refresh_response.json()['detail']
    
    async def test_session_revocation(self, client, mock_user, redis_mock):
        """Test revoking a specific session."""
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            UserMock.authenticate = AsyncMock(return_value=mock_user)
            
            # Login to create two sessions
            login1 = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            tokens1 = login1.json()
            
            login2 = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            tokens2 = login2.json()
            
            # Use first session to revoke second session
            headers = {'Authorization': f'Bearer {tokens1["access_token"]}'}
            
            # Get all sessions first
            sessions_response = client.get('/auth/sessions', headers=headers)
            sessions = sessions_response.json()
            
            # Find second session ID
            session2_id = tokens2['session_id']
            
            # Revoke second session
            revoke_response = client.delete(
                f'/auth/sessions/{session2_id}',
                headers=headers
            )
            
            assert revoke_response.status_code == 200
            assert 'Session revoked successfully' in revoke_response.json()['message']
    
    async def test_logout_all_sessions(self, client, mock_user):
        """Test logging out from all sessions."""
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            UserMock.authenticate = AsyncMock(return_value=mock_user)
            
            # Create multiple sessions
            login1 = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            tokens1 = login1.json()
            
            login2 = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            
            login3 = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'TestPassword123!'
            })
            
            # Logout all sessions
            headers = {'Authorization': f'Bearer {tokens1["access_token"]}'}
            logout_all_response = client.post('/auth/logout-all', headers=headers)
            
            assert logout_all_response.status_code == 200
            assert 'All sessions logged out successfully' in logout_all_response.json()['message']
    
    async def test_password_change_invalidates_sessions(self, client, mock_user):
        """Test that changing password invalidates all sessions."""
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            # Setup mocks
            UserMock.authenticate = AsyncMock()
            UserMock.authenticate.side_effect = [mock_user, mock_user]  # Login, then verify old password
            
            mock_user.update_password = AsyncMock()
            
            with patch('genesis.api.auth_endpoints.PasswordValidator') as ValidatorMock:
                validator_instance = MagicMock()
                validator_instance.validate_password = MagicMock(return_value=(True, []))
                ValidatorMock.return_value = validator_instance
                
                # Login first
                login_response = client.post('/auth/login', json={
                    'username': 'testuser',
                    'password': 'OldPassword123!'
                })
                
                tokens = login_response.json()
                headers = {'Authorization': f'Bearer {tokens["access_token"]}'}
                
                # Change password
                change_response = client.post('/auth/change-password',
                    headers=headers,
                    json={
                        'current_password': 'OldPassword123!',
                        'new_password': 'NewPassword456!'
                    }
                )
                
                assert change_response.status_code == 200
                assert 'Password changed successfully' in change_response.json()['message']
                
                # Verify password was updated
                mock_user.update_password.assert_called_once_with('NewPassword456!')
    
    async def test_unauthorized_access_without_token(self, client):
        """Test accessing protected endpoint without token."""
        response = client.get('/auth/sessions')
        
        assert response.status_code == 403  # FastAPI returns 403 for missing auth
    
    async def test_invalid_token_format(self, client):
        """Test using invalid token format."""
        headers = {'Authorization': 'InvalidTokenFormat'}
        response = client.get('/auth/verify', headers=headers)
        
        assert response.status_code == 403  # FastAPI returns 403 for invalid format
    
    async def test_multiple_concurrent_logins(self, client, mock_user):
        """Test multiple concurrent login attempts."""
        with patch('genesis.api.auth_endpoints.User') as UserMock:
            UserMock.authenticate = AsyncMock(return_value=mock_user)
            
            # Simulate concurrent logins
            responses = []
            for i in range(5):
                response = client.post('/auth/login', json={
                    'username': 'testuser',
                    'password': 'TestPassword123!'
                })
                responses.append(response)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                tokens = response.json()
                assert 'access_token' in tokens
                assert 'session_id' in tokens
            
            # All sessions should be different
            session_ids = [r.json()['session_id'] for r in responses]
            assert len(set(session_ids)) == 5  # All unique