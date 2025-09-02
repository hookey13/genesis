"""User model with secure password management for Project GENESIS."""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
import structlog

from genesis.security.password_manager import SecurePasswordManager

logger = structlog.get_logger(__name__)


class UserRole(str):
    """User role enumeration."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"


class User(BaseModel):
    """User domain model with secure password management."""
    
    user_id: str = Field(default_factory=lambda: str(uuid4()))
    username: str
    email: str
    password_hash: str = ""
    
    # Password security fields
    sha256_migrated: bool = True  # Set to False for existing SHA256 users
    old_sha256_hash: Optional[str] = None  # Temporary storage for migration
    password_changed_at: Optional[datetime] = None
    password_history: List[str] = Field(default_factory=list)
    
    # Account security
    is_active: bool = True
    is_locked: bool = False
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    last_successful_login: Optional[datetime] = None
    
    # Two-factor authentication
    totp_secret: Optional[str] = None
    totp_enabled: bool = False
    backup_codes: List[str] = Field(default_factory=list)
    
    # User metadata
    role: str = UserRole.VIEWER
    permissions: dict = Field(default_factory=dict)
    api_keys: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    # Password manager instance (not stored in DB)
    _password_manager: Optional[SecurePasswordManager] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """Initialize User with password manager."""
        super().__init__(**data)
        self._password_manager = SecurePasswordManager()
    
    @property
    def password_manager(self) -> SecurePasswordManager:
        """Get password manager instance."""
        if self._password_manager is None:
            self._password_manager = SecurePasswordManager()
        return self._password_manager
    
    def set_password(self, password: str) -> None:
        """Set user password with bcrypt hashing.
        
        Args:
            password: Plain text password to hash
            
        Raises:
            PasswordComplexityError: If password doesn't meet requirements
            PasswordReuseError: If password was recently used
        """
        # Check password history to prevent reuse
        if self.password_history:
            self.password_manager.check_password_history(
                int(self.user_id) if self.user_id.isdigit() else hash(self.user_id),
                password,
                self.password_history
            )
        
        # Hash the new password
        new_hash = self.password_manager.hash_password(password)
        
        # Update password history (keep last 5)
        if self.password_hash:
            self.password_history.append(self.password_hash)
            self.password_history = self.password_history[-5:]
        
        # Set new password
        self.password_hash = new_hash
        self.password_changed_at = datetime.now()
        self.sha256_migrated = True
        self.old_sha256_hash = None  # Clear any old SHA256 hash
        self.updated_at = datetime.now()
        
        logger.info(
            "Password updated for user",
            user_id=self.user_id,
            username=self.username
        )
    
    def verify_password(self, password: str) -> bool:
        """Verify user password.
        
        Args:
            password: Plain text password to verify
            
        Returns:
            True if password matches, False otherwise
        """
        result = self.password_manager.verify_password(password, self.password_hash)
        
        if result:
            # Reset failed login attempts on successful verification
            self.failed_login_attempts = 0
            self.last_successful_login = datetime.now()
            logger.info(
                "Password verification successful",
                user_id=self.user_id,
                username=self.username
            )
        else:
            # Track failed login attempts
            self.failed_login_attempts += 1
            self.last_failed_login = datetime.now()
            
            # Lock account after 5 failed attempts
            if self.failed_login_attempts >= 5:
                self.is_locked = True
                logger.warning(
                    "Account locked due to failed login attempts",
                    user_id=self.user_id,
                    username=self.username,
                    attempts=self.failed_login_attempts
                )
            else:
                logger.warning(
                    "Password verification failed",
                    user_id=self.user_id,
                    username=self.username,
                    attempts=self.failed_login_attempts
                )
        
        self.updated_at = datetime.now()
        return result
    
    def migrate_from_sha256(self, plaintext_password: str) -> None:
        """Migrate from SHA256 to bcrypt hash.
        
        This should only be called during login with correct password.
        
        Args:
            plaintext_password: User's plain text password
            
        Raises:
            ValueError: If password doesn't match SHA256 hash
        """
        if not self.old_sha256_hash:
            raise ValueError("No SHA256 hash to migrate from")
        
        # Migrate the password
        new_hash = self.password_manager.migrate_sha256_password(
            self.old_sha256_hash,
            plaintext_password
        )
        
        # Update password history
        if self.password_hash:
            self.password_history.append(self.password_hash)
            self.password_history = self.password_history[-5:]
        
        # Update user fields
        self.password_hash = new_hash
        self.password_changed_at = datetime.now()
        self.sha256_migrated = True
        self.old_sha256_hash = None  # Clear the old hash
        self.updated_at = datetime.now()
        
        logger.info(
            "Successfully migrated user from SHA256 to bcrypt",
            user_id=self.user_id,
            username=self.username
        )
    
    def needs_password_change(self, days: int = 90) -> bool:
        """Check if password needs to be changed.
        
        Args:
            days: Number of days before password expires
            
        Returns:
            True if password needs changing
        """
        if not self.password_changed_at:
            return True
        
        age = datetime.now() - self.password_changed_at
        return age.days >= days
    
    def unlock_account(self) -> None:
        """Unlock the user account."""
        self.is_locked = False
        self.failed_login_attempts = 0
        self.updated_at = datetime.now()
        
        logger.info(
            "Account unlocked",
            user_id=self.user_id,
            username=self.username
        )
    
    def generate_password_reset_token(self) -> str:
        """Generate secure password reset token.
        
        Returns:
            Secure random token
        """
        import secrets
        token = secrets.token_urlsafe(32)
        
        logger.info(
            "Password reset token generated",
            user_id=self.user_id,
            username=self.username
        )
        
        return token
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        return v.lower()
    
    def to_dict(self, exclude_sensitive: bool = True) -> dict:
        """Convert user to dictionary.
        
        Args:
            exclude_sensitive: Whether to exclude sensitive fields
            
        Returns:
            User data as dictionary
        """
        data = self.model_dump()
        
        if exclude_sensitive:
            # Remove sensitive fields
            sensitive_fields = [
                'password_hash',
                'old_sha256_hash',
                'password_history',
                'totp_secret',
                'backup_codes',
                'api_keys'
            ]
            for field in sensitive_fields:
                data.pop(field, None)
        
        # Remove internal fields
        data.pop('_password_manager', None)
        
        return data
    
    @classmethod
    async def find_by_username(cls, username: str) -> Optional["User"]:
        """Find user by username.
        
        This is a placeholder for database lookup.
        In production, this would query the database.
        
        Args:
            username: Username to search for
            
        Returns:
            User instance or None
        """
        # TODO: Implement database lookup
        logger.info("Finding user by username", username=username)
        return None
    
    @classmethod
    async def find_by_email(cls, email: str) -> Optional["User"]:
        """Find user by email.
        
        This is a placeholder for database lookup.
        In production, this would query the database.
        
        Args:
            email: Email to search for
            
        Returns:
            User instance or None
        """
        # TODO: Implement database lookup
        logger.info("Finding user by email", email=email)
        return None
    
    async def save(self) -> None:
        """Save user to database.
        
        This is a placeholder for database save operation.
        In production, this would persist to the database.
        """
        # TODO: Implement database save
        logger.info(
            "Saving user to database",
            user_id=self.user_id,
            username=self.username
        )