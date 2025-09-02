"""Integration tests for authentication and password migration."""

import hashlib
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient
from fastapi import FastAPI

from genesis.api.auth_endpoints import router
from genesis.models.user import User
from genesis.security.password_manager import SecurePasswordManager


# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestAuthenticationEndpoints:
    """Integration tests for authentication endpoints."""
    
    @pytest.fixture
    def mock_user_db(self):
        """Mock user database operations."""
        with patch('genesis.models.user.User.find_by_username') as mock_find_username, \
             patch('genesis.models.user.User.find_by_email') as mock_find_email, \
             patch('genesis.models.user.User.save') as mock_save:
            
            mock_find_username.return_value = None
            mock_find_email.return_value = None
            mock_save.return_value = None
            
            yield {
                'find_username': mock_find_username,
                'find_email': mock_find_email,
                'save': mock_save
            }
    
    @pytest.mark.asyncio
    async def test_user_registration_success(self, mock_user_db):
        """Test successful user registration with bcrypt."""
        registration_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "SecurePassword123!"
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "password" not in data  # Password should not be returned
    
    @pytest.mark.asyncio
    async def test_user_registration_weak_password(self, mock_user_db):
        """Test registration with weak password."""
        registration_data = {
            "username": "weakuser",
            "email": "weak@example.com",
            "password": "weak"  # Too short, no complexity
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 400
        error = response.json()
        assert "at least 12 characters" in error["detail"]
    
    @pytest.mark.asyncio
    async def test_user_registration_duplicate_username(self, mock_user_db):
        """Test registration with existing username."""
        # Mock existing user
        existing_user = User(username="existing", email="other@example.com")
        mock_user_db['find_username'].return_value = existing_user
        
        registration_data = {
            "username": "existing",
            "email": "new@example.com",
            "password": "SecurePassword123!"
        }
        
        response = client.post("/auth/register", json=registration_data)
        
        assert response.status_code == 400
        assert "Username already exists" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_login_with_bcrypt_password(self, mock_user_db):
        """Test login with bcrypt hashed password."""
        # Create user with bcrypt password
        user = User(username="testuser", email="test@example.com")
        user.set_password("CorrectPassword123!")
        mock_user_db['find_username'].return_value = user
        
        # Attempt login
        response = client.post(
            "/auth/login",
            data={
                "username": "testuser",
                "password": "CorrectPassword123!"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_sha256_migration(self, mock_user_db):
        """Test automatic migration from SHA256 to bcrypt during login."""
        password = "MigrateMe123!"
        sha256_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Create user with SHA256 hash
        user = User(username="olduser", email="old@example.com")
        user.sha256_migrated = False
        user.old_sha256_hash = sha256_hash
        user.password_hash = ""
        mock_user_db['find_username'].return_value = user
        
        # Attempt login (should trigger migration)
        response = client.post(
            "/auth/login",
            data={
                "username": "olduser",
                "password": password
            }
        )
        
        assert response.status_code == 200
        
        # Verify user was migrated
        assert user.sha256_migrated is True
        assert user.password_hash.startswith('$2b$')
        assert user.old_sha256_hash is None
    
    @pytest.mark.asyncio
    async def test_login_failed_attempts_lockout(self, mock_user_db):
        """Test account lockout after multiple failed attempts."""
        user = User(username="locktest", email="lock@example.com")
        user.set_password("CorrectPassword123!")
        mock_user_db['find_username'].return_value = user
        
        # Attempt login with wrong password 5 times
        for i in range(5):
            response = client.post(
                "/auth/login",
                data={
                    "username": "locktest",
                    "password": "WrongPassword"
                }
            )
            assert response.status_code == 401
        
        # Account should be locked
        assert user.is_locked is True
        
        # Further login attempts should fail with locked message
        response = client.post(
            "/auth/login",
            data={
                "username": "locktest",
                "password": "CorrectPassword123!"  # Even with correct password
            }
        )
        assert response.status_code == 423  # Locked status
        assert "locked" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_password_strength_check(self):
        """Test password strength checking endpoint."""
        # Weak password
        response = client.get("/auth/check-password-strength", params={"password": "weak"})
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["strength"] == "very_weak"
        assert len(data["recommendations"]) > 0
        
        # Strong password
        response = client.get(
            "/auth/check-password-strength",
            params={"password": "VeryStr0ng!P@ssw0rd#2024"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["strength"] == "strong"
        assert "years" in data["estimated_crack_time"]
    
    @pytest.mark.asyncio
    async def test_secure_password_generation(self):
        """Test secure password generation endpoint."""
        response = client.post("/auth/generate-secure-password", params={"length": 20})
        assert response.status_code == 200
        data = response.json()
        
        assert "password" in data
        assert data["length"] == 20
        assert "crack_time" in data
        
        # Verify generated password meets complexity requirements
        pm = SecurePasswordManager()
        pm.validate_password_complexity(data["password"])  # Should not raise


class TestPasswordMigrationFlow:
    """Test complete password migration flow."""
    
    @pytest.mark.asyncio
    async def test_complete_migration_workflow(self):
        """Test complete workflow from SHA256 to bcrypt."""
        # Simulate database with SHA256 users
        users_db = []
        
        # Create users with different password types
        for i in range(5):
            user = User(username=f"user{i}", email=f"user{i}@example.com")
            if i < 3:
                # SHA256 users
                password = f"OldPassword{i}!"
                user.password_hash = hashlib.sha256(password.encode()).hexdigest()
                user.sha256_migrated = False
            else:
                # Already migrated bcrypt users
                user.set_password(f"NewPassword{i}!")
            users_db.append(user)
        
        # Run migration analysis
        sha256_count = sum(1 for u in users_db if not u.sha256_migrated)
        bcrypt_count = sum(1 for u in users_db if u.sha256_migrated)
        
        assert sha256_count == 3
        assert bcrypt_count == 2
        
        # Simulate user logins to trigger migration
        for i in range(3):
            user = users_db[i]
            password = f"OldPassword{i}!"
            
            # Store old hash for migration
            user.old_sha256_hash = user.password_hash
            
            # Trigger migration
            user.migrate_from_sha256(password)
            
            # Verify migration
            assert user.sha256_migrated is True
            assert user.password_hash.startswith('$2b$')
            assert user.verify_password(password)
        
        # All users should now be migrated
        assert all(u.sha256_migrated for u in users_db)


class TestAuthenticationSecurity:
    """Security-focused tests for authentication."""
    
    @pytest.mark.asyncio
    async def test_timing_attack_resistance(self, mock_user_db):
        """Test resistance to timing attacks."""
        import time
        
        # Non-existent user
        mock_user_db['find_username'].return_value = None
        
        times_nonexistent = []
        for _ in range(5):
            start = time.perf_counter()
            response = client.post(
                "/auth/login",
                data={
                    "username": "nonexistent",
                    "password": "TestPassword123!"
                }
            )
            elapsed = time.perf_counter() - start
            times_nonexistent.append(elapsed)
            assert response.status_code == 401
        
        # Existing user with wrong password
        user = User(username="existing", email="test@example.com")
        user.set_password("CorrectPassword123!")
        mock_user_db['find_username'].return_value = user
        
        times_wrong_password = []
        for _ in range(5):
            start = time.perf_counter()
            response = client.post(
                "/auth/login",
                data={
                    "username": "existing",
                    "password": "WrongPassword123!"
                }
            )
            elapsed = time.perf_counter() - start
            times_wrong_password.append(elapsed)
            assert response.status_code == 401
        
        # Compare average times
        avg_nonexistent = sum(times_nonexistent) / len(times_nonexistent)
        avg_wrong = sum(times_wrong_password) / len(times_wrong_password)
        
        # Times should be similar (prevents username enumeration)
        time_ratio = abs(avg_nonexistent - avg_wrong) / max(avg_nonexistent, avg_wrong)
        assert time_ratio < 0.5  # Less than 50% difference
    
    @pytest.mark.asyncio
    async def test_password_reset_security(self, mock_user_db):
        """Test password reset doesn't leak user existence."""
        # Request reset for non-existent email
        response1 = client.post("/auth/forgot-password", params={"email": "nonexistent@example.com"})
        assert response1.status_code == 200
        message1 = response1.json()["message"]
        
        # Request reset for existing email
        user = User(username="existing", email="existing@example.com")
        mock_user_db['find_email'].return_value = user
        
        response2 = client.post("/auth/forgot-password", params={"email": "existing@example.com"})
        assert response2.status_code == 200
        message2 = response2.json()["message"]
        
        # Messages should be identical (no information leakage)
        assert message1 == message2
    
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, mock_user_db):
        """Test that SQL injection attempts are handled safely."""
        injection_attempts = [
            "admin' OR '1'='1",
            "'; DROP TABLE users; --",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]
        
        for attempt in injection_attempts:
            response = client.post(
                "/auth/login",
                data={
                    "username": attempt,
                    "password": "password"
                }
            )
            # Should handle safely without crashes
            assert response.status_code in [400, 401]
    
    @pytest.mark.asyncio
    async def test_password_history_enforcement(self, mock_user_db):
        """Test password history prevents reuse."""
        user = User(username="historytest", email="history@example.com")
        
        # Set multiple passwords to build history
        passwords = [
            "FirstPassword123!",
            "SecondPassword123!",
            "ThirdPassword123!",
            "FourthPassword123!",
            "FifthPassword123!"
        ]
        
        for password in passwords:
            user.set_password(password)
        
        # Try to reuse an old password
        with pytest.raises(Exception) as exc:
            user.set_password("ThirdPassword123!")  # Was used 2 changes ago
        assert "recently used" in str(exc.value).lower()
        
        # New password should work
        user.set_password("BrandNewPassword123!")
        assert user.verify_password("BrandNewPassword123!")


class TestPerformanceAndScalability:
    """Test performance aspects of password security."""
    
    @pytest.mark.asyncio
    async def test_bcrypt_performance(self):
        """Test bcrypt hashing performance."""
        import time
        
        pm = SecurePasswordManager(cost_factor=12)
        password = "TestPassword123!"
        
        # Measure hashing time
        start = time.perf_counter()
        hashed = pm.hash_password(password)
        hash_time = time.perf_counter() - start
        
        # Should complete in reasonable time (< 500ms for cost factor 12)
        assert hash_time < 0.5
        
        # Measure verification time
        start = time.perf_counter()
        pm.verify_password(password, hashed)
        verify_time = time.perf_counter() - start
        
        # Verification should be similar to hashing time
        assert verify_time < 0.5
    
    @pytest.mark.asyncio
    async def test_concurrent_authentication(self, mock_user_db):
        """Test handling of concurrent authentication requests."""
        user = User(username="concurrent", email="concurrent@example.com")
        user.set_password("ConcurrentPass123!")
        mock_user_db['find_username'].return_value = user
        
        # Simulate concurrent login attempts
        async def login_attempt():
            response = client.post(
                "/auth/login",
                data={
                    "username": "concurrent",
                    "password": "ConcurrentPass123!"
                }
            )
            return response.status_code
        
        # Run multiple concurrent requests
        tasks = [login_attempt() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(status == 200 for status in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])