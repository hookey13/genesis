"""Unit tests for password security implementation."""

import time
import hashlib
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from genesis.security.password_manager import (
    SecurePasswordManager,
    PasswordComplexityError,
    PasswordReuseError
)
from genesis.models.user import User


class TestSecurePasswordManager:
    """Test suite for SecurePasswordManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pm = SecurePasswordManager(cost_factor=12, history_limit=5)
    
    def test_bcrypt_hashing(self):
        """Test bcrypt password hashing."""
        password = "SecurePassword123!"
        hashed = self.pm.hash_password(password)
        
        # Verify it's a bcrypt hash with correct cost factor
        assert hashed.startswith('$2b$12$')
        assert len(hashed) == 60  # Standard bcrypt hash length
        
        # Verify password verification works
        assert self.pm.verify_password(password, hashed) is True
        assert self.pm.verify_password("WrongPassword", hashed) is False
    
    def test_unique_salts(self):
        """Test that each hash uses a unique salt."""
        password = "SamePassword123!"
        
        # Hash the same password multiple times
        hash1 = self.pm.hash_password(password)
        hash2 = self.pm.hash_password(password)
        hash3 = self.pm.hash_password(password)
        
        # All hashes should be different due to unique salts
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3
        
        # But all should verify correctly
        assert self.pm.verify_password(password, hash1)
        assert self.pm.verify_password(password, hash2)
        assert self.pm.verify_password(password, hash3)
    
    def test_password_complexity_validation(self):
        """Test password complexity requirements."""
        # Valid passwords
        valid_passwords = [
            "SecurePassword123!",
            "MyVeryS3cure!Pass",
            "C0mplex&Password",
            "Test@Pass123Word",
            "Str0ng!Password#2024"
        ]
        
        for password in valid_passwords:
            # Should not raise exception
            self.pm.validate_password_complexity(password)
        
        # Invalid passwords with specific errors
        invalid_cases = [
            ("short", "at least 12 characters"),
            ("nouppercase123!", "uppercase letter"),
            ("NOLOWERCASE123!", "lowercase letter"),
            ("NoNumbers!", "number"),
            ("NoSpecialChars123", "special character"),
            ("password123!", "too common"),
            ("Abcd1234567!", "sequential characters"),
            ("AAA123!!!bbb", "repeated characters")
        ]
        
        for password, expected_error in invalid_cases:
            with pytest.raises(PasswordComplexityError) as exc:
                self.pm.validate_password_complexity(password)
            assert expected_error in str(exc.value).lower()
    
    def test_timing_safe_verification(self):
        """Test timing-safe password verification."""
        password = "TestPassword123!"
        hashed = self.pm.hash_password(password)
        
        # Measure timing for correct password
        times_correct = []
        for _ in range(10):
            start = time.perf_counter()
            result = self.pm.verify_password(password, hashed)
            elapsed = time.perf_counter() - start
            times_correct.append(elapsed)
            assert result is True
        
        # Measure timing for incorrect password
        times_incorrect = []
        for _ in range(10):
            start = time.perf_counter()
            result = self.pm.verify_password("WrongPassword123!", hashed)
            elapsed = time.perf_counter() - start
            times_incorrect.append(elapsed)
            assert result is False
        
        # Calculate average times
        avg_correct = sum(times_correct) / len(times_correct)
        avg_incorrect = sum(times_incorrect) / len(times_incorrect)
        
        # Timing difference should be minimal (< 50% difference)
        time_ratio = abs(avg_correct - avg_incorrect) / max(avg_correct, avg_incorrect)
        assert time_ratio < 0.5, f"Timing difference too large: {time_ratio:.2%}"
    
    def test_password_migration_from_sha256(self):
        """Test SHA256 to bcrypt migration."""
        password = "MigrationTest123!"
        sha256_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Test successful migration
        bcrypt_hash = self.pm.migrate_sha256_password(sha256_hash, password)
        
        # Verify new hash is bcrypt
        assert bcrypt_hash.startswith('$2b$12$')
        assert len(bcrypt_hash) == 60
        assert self.pm.verify_password(password, bcrypt_hash)
        
        # Test failed migration with wrong password
        with pytest.raises(ValueError) as exc:
            self.pm.migrate_sha256_password(sha256_hash, "WrongPassword")
        assert "doesn't match existing hash" in str(exc.value)
    
    def test_password_history_tracking(self):
        """Test password history to prevent reuse."""
        user_id = 1
        current_password = "CurrentPass123!"
        
        # Create password history
        password_history = [
            self.pm.hash_password("OldPassword1!"),
            self.pm.hash_password("OldPassword2!"),
            self.pm.hash_password("OldPassword3!"),
            self.pm.hash_password("OldPassword4!"),
            self.pm.hash_password(current_password)  # Current password in history
        ]
        
        # Should raise error when trying to reuse current password
        with pytest.raises(PasswordReuseError) as exc:
            self.pm.check_password_history(user_id, current_password, password_history)
        assert "used within last 5 changes" in str(exc.value)
        
        # Should not raise error for new password
        new_password = "BrandNewPass123!"
        self.pm.check_password_history(user_id, new_password, password_history)
    
    def test_secure_password_generation(self):
        """Test cryptographically secure password generation."""
        # Generate multiple passwords
        passwords = []
        for length in [12, 16, 20, 24]:
            password = self.pm.generate_secure_password(length)
            passwords.append(password)
            
            # Check length
            assert len(password) == length
            
            # Check complexity requirements
            self.pm.validate_password_complexity(password)
            
            # Verify it's not in common passwords
            assert password not in self.pm._common_passwords
        
        # All generated passwords should be unique
        assert len(passwords) == len(set(passwords))
    
    def test_password_strength_estimation(self):
        """Test password crack time estimation."""
        test_cases = [
            ("abc", "Less than 1 second"),  # Very weak
            ("Password1", "hours"),  # Weak - no special chars
            ("Pass@123", "days"),  # Moderate - short
            ("SecurePass123!", "years"),  # Strong
            ("MyVeryL0ng&SecureP@ssw0rd!", "years")  # Very strong
        ]
        
        for password, expected_strength in test_cases:
            crack_time = self.pm.estimate_crack_time(password)
            assert expected_strength in crack_time or "thousand" in crack_time or "Millions" in crack_time
    
    def test_malformed_hash_handling(self):
        """Test handling of malformed password hashes."""
        malformed_hashes = [
            "",
            "not-a-hash",
            "$2b$",  # Incomplete bcrypt
            "$2a$10$" + "x" * 50,  # Wrong bcrypt version
            "x" * 60,  # Right length, wrong format
        ]
        
        for bad_hash in malformed_hashes:
            result = self.pm.verify_password("TestPassword123!", bad_hash)
            assert result is False  # Should return False, not raise exception
    
    def test_cost_factor_configuration(self):
        """Test configurable bcrypt cost factor."""
        # Test with different cost factors
        for cost_factor in [10, 12, 14]:
            pm = SecurePasswordManager(cost_factor=cost_factor)
            password = "TestPassword123!"
            hashed = pm.hash_password(password)
            
            # Verify cost factor in hash
            assert hashed.startswith(f'$2b${cost_factor:02d}$')
            assert pm.verify_password(password, hashed)


class TestUserPasswordIntegration:
    """Test User model integration with password manager."""
    
    @pytest.fixture
    def user(self):
        """Create a test user."""
        return User(
            username="testuser",
            email="test@example.com"
        )
    
    def test_user_password_setting(self, user):
        """Test setting user password."""
        password = "UserPassword123!"
        user.set_password(password)
        
        # Verify password was hashed with bcrypt
        assert user.password_hash.startswith('$2b$')
        assert len(user.password_hash) == 60
        assert user.password_changed_at is not None
        assert user.sha256_migrated is True
        
        # Verify password verification
        assert user.verify_password(password) is True
        assert user.verify_password("WrongPassword") is False
    
    def test_user_password_history(self, user):
        """Test user password history tracking."""
        # Set initial password
        user.set_password("FirstPassword123!")
        first_hash = user.password_hash
        
        # Change password
        user.set_password("SecondPassword123!")
        
        # Verify history was updated
        assert len(user.password_history) == 1
        assert user.password_history[0] == first_hash
        
        # Try to reuse first password
        with pytest.raises(PasswordReuseError):
            user.set_password("FirstPassword123!")
    
    def test_user_sha256_migration(self, user):
        """Test user migration from SHA256."""
        password = "MigrateMe123!"
        sha256_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Set up user with SHA256 hash
        user.old_sha256_hash = sha256_hash
        user.sha256_migrated = False
        
        # Migrate password
        user.migrate_from_sha256(password)
        
        # Verify migration
        assert user.password_hash.startswith('$2b$')
        assert user.sha256_migrated is True
        assert user.old_sha256_hash is None
        assert user.verify_password(password) is True
    
    def test_user_account_locking(self, user):
        """Test account locking after failed attempts."""
        user.set_password("CorrectPassword123!")
        
        # Simulate failed login attempts
        for i in range(5):
            assert user.verify_password("WrongPassword") is False
            assert user.failed_login_attempts == i + 1
        
        # Account should be locked after 5 attempts
        assert user.is_locked is True
        
        # Unlock account
        user.unlock_account()
        assert user.is_locked is False
        assert user.failed_login_attempts == 0
    
    def test_user_password_expiry(self, user):
        """Test password expiry checking."""
        user.set_password("TestPassword123!")
        
        # Fresh password should not need changing
        assert user.needs_password_change(days=90) is False
        
        # Simulate old password
        from datetime import timedelta
        user.password_changed_at = datetime.now() - timedelta(days=100)
        assert user.needs_password_change(days=90) is True
    
    def test_user_email_validation(self):
        """Test email validation."""
        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.org"
        ]
        
        for email in valid_emails:
            user = User(username="test", email=email)
            assert user.email == email.lower()
        
        # Invalid emails should raise ValueError
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user@.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValueError):
                User(username="test", email=email)
    
    def test_user_username_validation(self):
        """Test username validation."""
        valid_usernames = [
            "user123",
            "test_user",
            "user-name",
            "User123"  # Should be lowercased
        ]
        
        for username in valid_usernames:
            user = User(username=username, email="test@example.com")
            assert user.username == username.lower()
        
        # Invalid usernames should raise ValueError
        invalid_usernames = [
            "ab",  # Too short
            "user@name",  # Invalid character
            "user name",  # Space
            "user.name"  # Dot
        ]
        
        for username in invalid_usernames:
            with pytest.raises(ValueError):
                User(username=username, email="test@example.com")


class TestPasswordComplexityEdgeCases:
    """Test edge cases in password complexity validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pm = SecurePasswordManager()
    
    def test_unicode_passwords(self):
        """Test handling of unicode characters in passwords."""
        # Unicode passwords should work
        unicode_password = "Pässwörd123!€"
        hashed = self.pm.hash_password(unicode_password)
        assert self.pm.verify_password(unicode_password, hashed)
    
    def test_very_long_passwords(self):
        """Test handling of very long passwords."""
        # bcrypt has a 72-byte limit
        long_password = "A" * 100 + "bcD123!"
        hashed = self.pm.hash_password(long_password)
        assert self.pm.verify_password(long_password, hashed)
    
    def test_sql_injection_attempts(self):
        """Test that SQL injection attempts are handled safely."""
        injection_attempts = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ]
        
        for attempt in injection_attempts:
            # Should handle as normal password (will fail complexity)
            with pytest.raises(PasswordComplexityError):
                self.pm.validate_password_complexity(attempt)
    
    def test_sequential_detection(self):
        """Test sequential character detection."""
        sequential_passwords = [
            "Password123!abc",  # abc sequence
            "Pass321!word",  # 321 sequence
            "zyxTest123!",  # zyx sequence
            "789Password!"  # 789 sequence
        ]
        
        for password in sequential_passwords:
            with pytest.raises(PasswordComplexityError) as exc:
                self.pm.validate_password_complexity(password)
            assert "sequential" in str(exc.value).lower()
    
    def test_repetition_detection(self):
        """Test repeated character detection."""
        repeated_passwords = [
            "Passsword123!",  # sss
            "Pass111word!",  # 111
            "Password!!!123"  # !!!
        ]
        
        for password in repeated_passwords:
            with pytest.raises(PasswordComplexityError) as exc:
                self.pm.validate_password_complexity(password)
            assert "repeated" in str(exc.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])