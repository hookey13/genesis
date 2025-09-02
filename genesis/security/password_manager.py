"""Cryptographically secure password management for Project GENESIS."""

import bcrypt
import secrets
import re
import hashlib
import hmac
from typing import List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


class PasswordComplexityError(Exception):
    """Raised when password doesn't meet complexity requirements."""
    pass


class PasswordReuseError(Exception):
    """Raised when password was recently used."""
    pass


class SecurePasswordManager:
    """Cryptographically secure password management with bcrypt."""
    
    def __init__(self, cost_factor: int = 12, history_limit: int = 5):
        """Initialize password manager.
        
        Args:
            cost_factor: Bcrypt cost factor (default 12 for good security/performance balance)
            history_limit: Number of previous passwords to check for reuse
        """
        self.cost_factor = cost_factor
        self.history_limit = history_limit
        self.min_length = 12
        
        # Compile regex patterns for efficiency
        self.complexity_patterns = {
            'uppercase': re.compile(r'[A-Z]'),
            'lowercase': re.compile(r'[a-z]'),
            'digit': re.compile(r'[0-9]'),
            'special': re.compile(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]')
        }
        
        # Common passwords list (would be loaded from file in production)
        self._common_passwords = {
            'password', 'password123', '123456', 'qwerty', 'admin',
            'letmein', 'welcome', 'monkey', 'password1', '123456789',
            '12345678', 'abc123', '111111', '1234567', 'sunshine',
            'master', 'iloveyou', 'princess', 'dragon', 'passw0rd',
            'baseball', 'football', 'michael', 'shadow', 'superman',
            'batman', 'trustno1', 'hello', 'charlie', 'biteme',
            'access', 'whatever', 'jordan', 'jessie', 'killer',
            'andrew', 'tigger', 'joshua', 'pepper', 'sophie',
            'starwars', 'genesis', 'trading', 'binance', 'crypto'
        }
        
        logger.info(
            "SecurePasswordManager initialized",
            cost_factor=self.cost_factor,
            history_limit=self.history_limit,
            min_length=self.min_length
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with proper salt rounds.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            Bcrypt hash string
            
        Raises:
            PasswordComplexityError: If password doesn't meet requirements
        """
        # Validate complexity first
        self.validate_password_complexity(password)
        
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=self.cost_factor)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        logger.info(
            "Password hashed successfully",
            hash_prefix=hashed[:10].decode('utf-8'),
            cost_factor=self.cost_factor
        )
        
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against bcrypt hash with timing safety.
        
        Args:
            password: Plain text password to verify
            hashed: Bcrypt hash to verify against
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # bcrypt.checkpw is timing-safe by design
            result = bcrypt.checkpw(
                password.encode('utf-8'),
                hashed.encode('utf-8')
            )
            
            if result:
                logger.info("Password verification successful")
            else:
                logger.warning("Password verification failed")
                
            return result
            
        except (ValueError, TypeError) as e:
            # Handle malformed hash or encoding errors
            logger.error(
                "Password verification error",
                error=str(e),
                hash_prefix=hashed[:10] if hashed else None
            )
            return False
    
    def validate_password_complexity(self, password: str) -> None:
        """Validate password meets complexity requirements.
        
        Args:
            password: Password to validate
            
        Raises:
            PasswordComplexityError: If validation fails
        """
        errors = []
        
        # Length check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")
        
        # Character type checks
        complexity_checks = {
            'uppercase': 'uppercase letter',
            'lowercase': 'lowercase letter',
            'digit': 'number',
            'special': 'special character (!@#$%^&*()_+-=[]{}|;:,.<>?)'
        }
        
        for check_name, pattern in self.complexity_patterns.items():
            if not pattern.search(password):
                errors.append(f"Password must contain at least one {complexity_checks[check_name]}")
        
        # Common password check
        if self._is_common_password(password):
            errors.append("Password is too common and easily guessable")
        
        # Sequential character check
        if self._has_sequential_characters(password):
            errors.append("Password contains too many sequential characters")
        
        # Repeated character check
        if self._has_excessive_repetition(password):
            errors.append("Password contains too many repeated characters")
        
        if errors:
            logger.warning(
                "Password complexity validation failed",
                error_count=len(errors),
                errors=errors
            )
            raise PasswordComplexityError("; ".join(errors))
        
        logger.debug("Password complexity validation passed")
    
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common password list.
        
        Args:
            password: Password to check
            
        Returns:
            True if password is common
        """
        # Check exact match and lowercase version
        return (
            password in self._common_passwords or
            password.lower() in self._common_passwords
        )
    
    def _has_sequential_characters(self, password: str) -> bool:
        """Check for sequential characters like 'abc' or '123'.
        
        Args:
            password: Password to check
            
        Returns:
            True if has 3+ sequential characters
        """
        for i in range(len(password) - 2):
            chars = password[i:i+3]
            
            # Check numeric sequences
            if chars.isdigit():
                nums = [int(c) for c in chars]
                if nums[1] == nums[0] + 1 and nums[2] == nums[1] + 1:
                    return True
                if nums[1] == nums[0] - 1 and nums[2] == nums[1] - 1:
                    return True
            
            # Check alphabetic sequences
            if chars.isalpha():
                codes = [ord(c.lower()) for c in chars]
                if codes[1] == codes[0] + 1 and codes[2] == codes[1] + 1:
                    return True
                if codes[1] == codes[0] - 1 and codes[2] == codes[1] - 1:
                    return True
        
        return False
    
    def _has_excessive_repetition(self, password: str) -> bool:
        """Check for excessive character repetition.
        
        Args:
            password: Password to check
            
        Returns:
            True if has 3+ repeated characters
        """
        for i in range(len(password) - 2):
            if password[i] == password[i+1] == password[i+2]:
                return True
        return False
    
    def migrate_sha256_password(self, sha256_hash: str, plaintext_password: str) -> str:
        """Migrate SHA256 hash to bcrypt when user logs in.
        
        Args:
            sha256_hash: Existing SHA256 hash
            plaintext_password: User's plaintext password
            
        Returns:
            New bcrypt hash
            
        Raises:
            ValueError: If password doesn't match SHA256 hash
        """
        # Verify the SHA256 hash matches (for migration only)
        expected_sha256 = hashlib.sha256(plaintext_password.encode()).hexdigest()
        
        if not hmac.compare_digest(sha256_hash, expected_sha256):
            logger.error(
                "SHA256 migration failed - password mismatch",
                provided_hash_prefix=sha256_hash[:10],
                expected_hash_prefix=expected_sha256[:10]
            )
            raise ValueError("Password doesn't match existing hash")
        
        # Create new bcrypt hash
        bcrypt_hash = self.hash_password(plaintext_password)
        
        logger.info(
            "Successfully migrated SHA256 to bcrypt",
            old_hash_prefix=sha256_hash[:10],
            new_hash_prefix=bcrypt_hash[:10]
        )
        
        return bcrypt_hash
    
    def check_password_history(
        self,
        user_id: int,
        new_password: str,
        password_history: List[str]
    ) -> None:
        """Check if password was recently used.
        
        Args:
            user_id: User ID for logging
            new_password: New password to check
            password_history: List of previous password hashes
            
        Raises:
            PasswordReuseError: If password was recently used
        """
        # Check against recent passwords
        recent_hashes = password_history[-self.history_limit:]
        
        for old_hash in recent_hashes:
            if self.verify_password(new_password, old_hash):
                logger.warning(
                    "Password reuse detected",
                    user_id=user_id,
                    history_limit=self.history_limit
                )
                raise PasswordReuseError(
                    f"Password was used within last {self.history_limit} changes"
                )
        
        logger.debug(
            "Password history check passed",
            user_id=user_id,
            history_count=len(recent_hashes)
        )
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate cryptographically secure password.
        
        Args:
            length: Desired password length (minimum 12)
            
        Returns:
            Secure random password meeting all complexity requirements
        """
        if length < self.min_length:
            length = self.min_length
        
        # Character sets
        uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        lowercase = 'abcdefghijklmnopqrstuvwxyz'
        digits = '0123456789'
        special = '!@#$%^&*()_+-=[]{}|;:,.<>?'
        
        # Ensure at least one character from each required set
        password = [
            secrets.choice(uppercase),
            secrets.choice(lowercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill remaining length with random characters
        all_chars = uppercase + lowercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Cryptographically secure shuffle
        secrets.SystemRandom().shuffle(password)
        
        generated = ''.join(password)
        
        logger.info(
            "Secure password generated",
            length=len(generated),
            complexity_met=True
        )
        
        return generated
    
    def estimate_crack_time(self, password: str) -> str:
        """Estimate time to crack password with current hardware.
        
        Args:
            password: Password to analyze
            
        Returns:
            Human-readable crack time estimate
        """
        # Calculate entropy
        entropy = 0
        char_space = 0
        
        if self.complexity_patterns['uppercase'].search(password):
            char_space += 26
        if self.complexity_patterns['lowercase'].search(password):
            char_space += 26
        if self.complexity_patterns['digit'].search(password):
            char_space += 10
        if self.complexity_patterns['special'].search(password):
            char_space += 32
        
        if char_space > 0:
            entropy = len(password) * (char_space ** 0.5)
        
        # Assume 10 billion guesses per second (modern GPU)
        guesses_per_second = 10_000_000_000
        seconds = (char_space ** len(password)) / guesses_per_second
        
        # Convert to human-readable time
        if seconds < 1:
            return "Less than 1 second"
        elif seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds / 3600)} hours"
        elif seconds < 31536000:
            return f"{int(seconds / 86400)} days"
        else:
            years = seconds / 31536000
            if years > 1_000_000:
                return "Millions of years"
            elif years > 1000:
                return f"{int(years / 1000)} thousand years"
            else:
                return f"{int(years)} years"