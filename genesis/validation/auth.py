"""Production-grade authentication module for deployment authorization.

This module provides secure authentication mechanisms for override operations.
It uses bcrypt for password hashing and includes rate limiting and account lockout.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

# For production, install: pip install bcrypt
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    # Fallback for demo environment

logger = structlog.get_logger(__name__)


@dataclass
class AuthAttempt:
    """Track authentication attempts for rate limiting."""
    username: str
    timestamp: datetime
    success: bool
    ip_address: str | None = None


@dataclass
class UserAccount:
    """User account for authentication."""
    username: str
    password_hash: str
    roles: list[str] = field(default_factory=list)
    locked: bool = False
    locked_until: datetime | None = None
    failed_attempts: int = 0
    last_attempt: datetime | None = None
    requires_mfa: bool = False

    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.locked:
            return False
        if self.locked_until and datetime.utcnow() > self.locked_until:
            # Auto-unlock after timeout
            self.locked = False
            self.locked_until = None
            self.failed_attempts = 0
            return False
        return True


class AuthenticationManager:
    """Manages authentication and authorization for critical operations."""

    # Security configuration
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    RATE_LIMIT_WINDOW_SECONDS = 60
    MAX_ATTEMPTS_PER_WINDOW = 10

    # Required roles for different operations
    REQUIRED_ROLES = {
        "override_deployment": ["admin", "lead_dev", "deployment_manager"],
        "emergency_override": ["admin"],
        "view_audit": ["admin", "lead_dev", "auditor"]
    }

    def __init__(self, genesis_root: Path | None = None):
        """Initialize authentication manager.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.auth_dir = self.genesis_root / ".genesis" / "auth"
        self.auth_dir.mkdir(parents=True, exist_ok=True)

        # Paths for persistence
        self.users_file = self.auth_dir / "users.json"
        self.attempts_file = self.auth_dir / "attempts.json"
        self.session_file = self.auth_dir / "sessions.json"

        # In-memory caches
        self.users: dict[str, UserAccount] = {}
        self.attempts: list[AuthAttempt] = []
        self.sessions: dict[str, dict[str, Any]] = {}

        # Load or initialize user data
        self._load_users()
        self._load_attempts()

        # Initialize with demo users if empty (ONLY for development)
        if not self.users and not self._is_production():
            self._initialize_demo_users()

    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str | None = None
    ) -> tuple[bool, str]:
        """Authenticate a user with rate limiting and account lockout.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address for rate limiting
            
        Returns:
            Tuple of (success, message)
        """
        # Check rate limiting
        if not self._check_rate_limit(username, ip_address):
            logger.warning("Rate limit exceeded", username=username, ip=ip_address)
            return False, "Rate limit exceeded. Please try again later."

        # Record attempt
        attempt = AuthAttempt(
            username=username,
            timestamp=datetime.utcnow(),
            success=False,
            ip_address=ip_address
        )
        self.attempts.append(attempt)

        # Check if user exists
        if username not in self.users:
            logger.warning("Authentication failed - unknown user", username=username)
            self._save_attempts()
            return False, "Invalid username or password"

        user = self.users[username]

        # Check if account is locked
        if user.is_locked():
            logger.warning("Authentication failed - account locked", username=username)
            self._save_attempts()
            return False, f"Account locked until {user.locked_until.isoformat() if user.locked_until else 'manually unlocked'}"

        # Verify password
        success = self._verify_password(password, user.password_hash)

        if success:
            # Reset failed attempts on success
            user.failed_attempts = 0
            user.last_attempt = datetime.utcnow()
            attempt.success = True
            self._save_users()
            self._save_attempts()

            logger.info("Authentication successful", username=username)

            # Create session
            session_token = self._create_session(username)

            return True, f"Authentication successful. Session: {session_token}"
        else:
            # Increment failed attempts
            user.failed_attempts += 1
            user.last_attempt = datetime.utcnow()

            # Lock account if threshold exceeded
            if user.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked = True
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.LOCKOUT_DURATION_MINUTES)
                logger.warning(
                    "Account locked due to failed attempts",
                    username=username,
                    attempts=user.failed_attempts
                )
                self._save_users()
                self._save_attempts()
                return False, f"Account locked due to {user.failed_attempts} failed attempts"

            self._save_users()
            self._save_attempts()

            remaining = self.MAX_FAILED_ATTEMPTS - user.failed_attempts
            logger.warning(
                "Authentication failed",
                username=username,
                attempts=user.failed_attempts,
                remaining=remaining
            )

            return False, f"Invalid username or password. {remaining} attempts remaining."

    def authorize(
        self,
        username: str,
        operation: str,
        session_token: str | None = None
    ) -> tuple[bool, str]:
        """Check if user is authorized for an operation.
        
        Args:
            username: Username
            operation: Operation to authorize
            session_token: Optional session token for validation
            
        Returns:
            Tuple of (authorized, message)
        """
        # Validate session if provided
        if session_token and not self._validate_session(username, session_token):
            return False, "Invalid or expired session"

        # Check if user exists
        if username not in self.users:
            return False, "User not found"

        user = self.users[username]

        # Check if account is locked
        if user.is_locked():
            return False, "Account is locked"

        # Check required roles
        required_roles = self.REQUIRED_ROLES.get(operation, [])
        if not required_roles:
            # No specific roles required
            return True, "Authorized"

        # Check if user has any required role
        user_roles = set(user.roles)
        if user_roles.intersection(required_roles):
            logger.info(
                "Authorization granted",
                username=username,
                operation=operation,
                roles=user.roles
            )
            return True, "Authorized"

        logger.warning(
            "Authorization denied",
            username=username,
            operation=operation,
            required=required_roles,
            user_roles=user.roles
        )

        return False, f"Insufficient privileges. Required roles: {required_roles}"

    def create_user(
        self,
        username: str,
        password: str,
        roles: list[str] | None = None,
        requires_mfa: bool = False
    ) -> tuple[bool, str]:
        """Create a new user account.
        
        Args:
            username: Username
            password: Password
            roles: User roles
            requires_mfa: Whether MFA is required
            
        Returns:
            Tuple of (success, message)
        """
        # Check if user already exists
        if username in self.users:
            return False, "User already exists"

        # Validate password strength
        is_strong, message = self._validate_password_strength(password)
        if not is_strong:
            return False, f"Weak password: {message}"

        # Hash password
        password_hash = self._hash_password(password)

        # Create user account
        user = UserAccount(
            username=username,
            password_hash=password_hash,
            roles=roles or [],
            requires_mfa=requires_mfa
        )

        self.users[username] = user
        self._save_users()

        logger.info("User created", username=username, roles=roles)

        return True, "User created successfully"

    def reset_password(
        self,
        username: str,
        new_password: str,
        admin_username: str | None = None
    ) -> tuple[bool, str]:
        """Reset user password.
        
        Args:
            username: Username to reset
            new_password: New password
            admin_username: Admin performing reset (if not self-reset)
            
        Returns:
            Tuple of (success, message)
        """
        # Check if user exists
        if username not in self.users:
            return False, "User not found"

        # If admin reset, verify admin privileges
        if admin_username and admin_username != username:
            is_authorized, msg = self.authorize(admin_username, "reset_password")
            if not is_authorized:
                return False, f"Admin authorization failed: {msg}"

        # Validate password strength
        is_strong, message = self._validate_password_strength(new_password)
        if not is_strong:
            return False, f"Weak password: {message}"

        # Update password
        user = self.users[username]
        user.password_hash = self._hash_password(new_password)
        user.failed_attempts = 0
        user.locked = False
        user.locked_until = None

        self._save_users()

        logger.info(
            "Password reset",
            username=username,
            admin=admin_username if admin_username != username else None
        )

        return True, "Password reset successfully"

    def unlock_account(self, username: str, admin_username: str) -> tuple[bool, str]:
        """Manually unlock a user account.
        
        Args:
            username: Username to unlock
            admin_username: Admin performing unlock
            
        Returns:
            Tuple of (success, message)
        """
        # Verify admin privileges
        is_authorized, msg = self.authorize(admin_username, "unlock_account")
        if not is_authorized:
            return False, f"Admin authorization failed: {msg}"

        # Check if user exists
        if username not in self.users:
            return False, "User not found"

        # Unlock account
        user = self.users[username]
        user.locked = False
        user.locked_until = None
        user.failed_attempts = 0

        self._save_users()

        logger.info("Account unlocked", username=username, admin=admin_username)

        return True, "Account unlocked successfully"

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt or fallback.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        if BCRYPT_AVAILABLE:
            # Use bcrypt for production
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        else:
            # Fallback for demo - NOT FOR PRODUCTION
            logger.warning("SECURITY: bcrypt not available - using demo hashing")
            return f"demo_hash_{hashlib.sha256(password.encode()).hexdigest()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password
            password_hash: Hashed password
            
        Returns:
            True if password matches
        """
        if BCRYPT_AVAILABLE and not password_hash.startswith("demo_hash_"):
            # Use bcrypt for production hashes
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        elif password_hash.startswith("demo_hash_"):
            # Demo fallback - NOT FOR PRODUCTION
            logger.warning("SECURITY: Using demo password verification")
            expected = f"demo_hash_{hashlib.sha256(password.encode()).hexdigest()}"
            return password_hash == expected
        else:
            # Hash type mismatch
            logger.error("SECURITY: Unable to verify password - hash type mismatch")
            return False

    def _validate_password_strength(self, password: str) -> tuple[bool, str]:
        """Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_strong, message)
        """
        if len(password) < 12:
            return False, "Password must be at least 12 characters"

        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in password)

        if not has_upper:
            return False, "Password must contain uppercase letters"
        if not has_lower:
            return False, "Password must contain lowercase letters"
        if not has_digit:
            return False, "Password must contain digits"
        if not has_special:
            return False, "Password must contain special characters"

        # Check for common patterns
        common_patterns = ["password", "12345", "qwerty", "admin", "genesis"]
        lower_password = password.lower()
        for pattern in common_patterns:
            if pattern in lower_password:
                return False, f"Password contains common pattern: {pattern}"

        return True, "Password is strong"

    def _check_rate_limit(self, username: str, ip_address: str | None) -> bool:
        """Check if rate limit is exceeded.
        
        Args:
            username: Username
            ip_address: Client IP address
            
        Returns:
            True if within rate limit
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.RATE_LIMIT_WINDOW_SECONDS)

        # Count recent attempts
        recent_attempts = [
            a for a in self.attempts
            if a.timestamp > window_start and
            (a.username == username or (ip_address and a.ip_address == ip_address))
        ]

        return len(recent_attempts) < self.MAX_ATTEMPTS_PER_WINDOW

    def _create_session(self, username: str) -> str:
        """Create a session for authenticated user.
        
        Args:
            username: Username
            
        Returns:
            Session token
        """
        session_token = hashlib.sha256(
            f"{username}:{datetime.utcnow().isoformat()}:{os.urandom(16).hex()}".encode()
        ).hexdigest()

        self.sessions[session_token] = {
            "username": username,
            "created": datetime.utcnow().isoformat(),
            "expires": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }

        self._save_sessions()

        return session_token

    def _validate_session(self, username: str, session_token: str) -> bool:
        """Validate a session token.
        
        Args:
            username: Username
            session_token: Session token
            
        Returns:
            True if session is valid
        """
        if session_token not in self.sessions:
            return False

        session = self.sessions[session_token]

        if session["username"] != username:
            return False

        expires = datetime.fromisoformat(session["expires"])
        if datetime.utcnow() > expires:
            # Session expired
            del self.sessions[session_token]
            self._save_sessions()
            return False

        return True

    def _is_production(self) -> bool:
        """Check if running in production environment.
        
        Returns:
            True if production environment
        """
        return os.environ.get("GENESIS_ENV", "development").lower() == "production"

    def _initialize_demo_users(self):
        """Initialize demo users for development environment."""
        logger.warning("SECURITY: Initializing demo users - not for production use")

        # Create demo admin user
        self.create_user(
            username="admin",
            password="DemoAdmin123!@#",
            roles=["admin", "deployment_manager"],
            requires_mfa=False
        )

        # Create demo lead developer
        self.create_user(
            username="lead_dev",
            password="DemoLeadDev456$%^",
            roles=["lead_dev", "deployment_manager"],
            requires_mfa=False
        )

        # Create demo auditor
        self.create_user(
            username="auditor",
            password="DemoAuditor789&*(",
            roles=["auditor"],
            requires_mfa=False
        )

    def _load_users(self):
        """Load users from file."""
        if self.users_file.exists():
            try:
                with open(self.users_file) as f:
                    data = json.load(f)
                    for username, user_data in data.items():
                        self.users[username] = UserAccount(
                            username=username,
                            password_hash=user_data["password_hash"],
                            roles=user_data.get("roles", []),
                            locked=user_data.get("locked", False),
                            locked_until=datetime.fromisoformat(user_data["locked_until"]) if user_data.get("locked_until") else None,
                            failed_attempts=user_data.get("failed_attempts", 0),
                            last_attempt=datetime.fromisoformat(user_data["last_attempt"]) if user_data.get("last_attempt") else None,
                            requires_mfa=user_data.get("requires_mfa", False)
                        )
            except Exception as e:
                logger.error("Failed to load users", error=str(e))

    def _save_users(self):
        """Save users to file."""
        try:
            data = {}
            for username, user in self.users.items():
                data[username] = {
                    "password_hash": user.password_hash,
                    "roles": user.roles,
                    "locked": user.locked,
                    "locked_until": user.locked_until.isoformat() if user.locked_until else None,
                    "failed_attempts": user.failed_attempts,
                    "last_attempt": user.last_attempt.isoformat() if user.last_attempt else None,
                    "requires_mfa": user.requires_mfa
                }

            with open(self.users_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save users", error=str(e))

    def _load_attempts(self):
        """Load authentication attempts from file."""
        if self.attempts_file.exists():
            try:
                with open(self.attempts_file) as f:
                    data = json.load(f)
                    for attempt_data in data:
                        self.attempts.append(AuthAttempt(
                            username=attempt_data["username"],
                            timestamp=datetime.fromisoformat(attempt_data["timestamp"]),
                            success=attempt_data["success"],
                            ip_address=attempt_data.get("ip_address")
                        ))

                    # Clean old attempts
                    cutoff = datetime.utcnow() - timedelta(days=7)
                    self.attempts = [a for a in self.attempts if a.timestamp > cutoff]
            except Exception as e:
                logger.error("Failed to load attempts", error=str(e))

    def _save_attempts(self):
        """Save authentication attempts to file."""
        try:
            # Keep only recent attempts (last 7 days)
            cutoff = datetime.utcnow() - timedelta(days=7)
            recent_attempts = [a for a in self.attempts if a.timestamp > cutoff]

            data = []
            for attempt in recent_attempts:
                data.append({
                    "username": attempt.username,
                    "timestamp": attempt.timestamp.isoformat(),
                    "success": attempt.success,
                    "ip_address": attempt.ip_address
                })

            with open(self.attempts_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save attempts", error=str(e))

    def _save_sessions(self):
        """Save sessions to file."""
        try:
            with open(self.session_file, "w") as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            logger.error("Failed to save sessions", error=str(e))


# Singleton instance
_auth_manager: AuthenticationManager | None = None


def get_auth_manager(genesis_root: Path | None = None) -> AuthenticationManager:
    """Get or create the authentication manager singleton.
    
    Args:
        genesis_root: Root directory of Genesis project
        
    Returns:
        Authentication manager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager(genesis_root)
    return _auth_manager
