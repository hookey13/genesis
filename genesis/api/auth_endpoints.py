"""Authentication endpoints for Project GENESIS API."""

import hashlib
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import structlog

from genesis.models.user import User
from genesis.security.password_manager import (
    SecurePasswordManager,
    PasswordComplexityError,
    PasswordReuseError
)

logger = structlog.get_logger(__name__)

# Router for authentication endpoints
router = APIRouter(prefix="/auth", tags=["authentication"])

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


class UserRegistration(BaseModel):
    """User registration request model."""
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    """User login request model."""
    username: str
    password: str


class PasswordChange(BaseModel):
    """Password change request model."""
    current_password: str
    new_password: str


class PasswordReset(BaseModel):
    """Password reset request model."""
    token: str
    new_password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class UserResponse(BaseModel):
    """User response model."""
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime


@router.post("/register", response_model=UserResponse)
async def register(registration: UserRegistration):
    """Register a new user with secure password hashing.
    
    Args:
        registration: User registration data
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If registration fails
    """
    try:
        # Check if user already exists
        existing_user = await User.find_by_username(registration.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        existing_email = await User.find_by_email(registration.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = User(
            username=registration.username,
            email=registration.email
        )
        
        # Set password with bcrypt hashing
        user.set_password(registration.password)
        
        # Save user to database
        await user.save()
        
        logger.info(
            "User registered successfully",
            user_id=user.user_id,
            username=user.username
        )
        
        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at
        )
        
    except PasswordComplexityError as e:
        logger.warning(
            "Registration failed - password complexity",
            username=registration.username,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Registration failed",
            username=registration.username,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token.
    
    Handles SHA256 to bcrypt migration for existing users.
    
    Args:
        form_data: OAuth2 login form data
        
    Returns:
        Access token for authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Find user by username
        user = await User.find_by_username(form_data.username)
        if not user:
            logger.warning(
                "Login failed - user not found",
                username=form_data.username
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check if account is locked
        if user.is_locked:
            logger.warning(
                "Login failed - account locked",
                user_id=user.user_id,
                username=user.username
            )
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is locked due to too many failed attempts"
            )
        
        # Check if account is active
        if not user.is_active:
            logger.warning(
                "Login failed - account inactive",
                user_id=user.user_id,
                username=user.username
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Handle SHA256 migration if needed
        if not user.sha256_migrated and user.old_sha256_hash:
            try:
                # Verify password matches SHA256 hash
                expected_sha256 = hashlib.sha256(form_data.password.encode()).hexdigest()
                if expected_sha256 == user.old_sha256_hash:
                    # Migrate to bcrypt
                    user.migrate_from_sha256(form_data.password)
                    await user.save()
                    logger.info(
                        "User migrated from SHA256 to bcrypt",
                        user_id=user.user_id,
                        username=user.username
                    )
                else:
                    # Password doesn't match
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials",
                        headers={"WWW-Authenticate": "Bearer"}
                    )
            except ValueError as e:
                logger.error(
                    "SHA256 migration failed",
                    user_id=user.user_id,
                    username=user.username,
                    error=str(e)
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        else:
            # Normal bcrypt verification
            if not user.verify_password(form_data.password):
                await user.save()  # Save failed attempt count
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        
        # Update last login time
        user.last_successful_login = datetime.now()
        await user.save()
        
        # Generate access token (simplified - use JWT in production)
        import secrets
        access_token = secrets.token_urlsafe(32)
        
        # TODO: Store token in Redis with expiration
        
        logger.info(
            "User logged in successfully",
            user_id=user.user_id,
            username=user.username
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Login failed",
            username=form_data.username,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    token: str = Depends(oauth2_scheme)
):
    """Change user password with validation.
    
    Args:
        password_change: Password change data
        token: Authentication token
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If password change fails
    """
    try:
        # TODO: Validate token and get user from token
        # For now, this is a placeholder
        user = None  # Get from token
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication"
            )
        
        # Verify current password
        if not user.verify_password(password_change.current_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Set new password
        user.set_password(password_change.new_password)
        await user.save()
        
        logger.info(
            "Password changed successfully",
            user_id=user.user_id,
            username=user.username
        )
        
        return {"message": "Password changed successfully"}
        
    except PasswordComplexityError as e:
        logger.warning(
            "Password change failed - complexity",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PasswordReuseError as e:
        logger.warning(
            "Password change failed - reuse",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Password change failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/forgot-password")
async def forgot_password(email: str):
    """Request password reset token.
    
    Args:
        email: User email address
        
    Returns:
        Success message (always returns success for security)
    """
    # Find user by email
    user = await User.find_by_email(email)
    
    if user:
        # Generate reset token
        reset_token = user.generate_password_reset_token()
        
        # TODO: Send email with reset token
        # TODO: Store token in database with expiration
        
        logger.info(
            "Password reset requested",
            user_id=user.user_id,
            email=email
        )
    else:
        logger.warning(
            "Password reset requested for non-existent email",
            email=email
        )
    
    # Always return success for security (don't reveal if email exists)
    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/reset-password")
async def reset_password(password_reset: PasswordReset):
    """Reset password with token.
    
    Args:
        password_reset: Password reset data
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If reset fails
    """
    try:
        # TODO: Validate reset token and get user
        # For now, this is a placeholder
        user = None  # Get from token
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Set new password
        user.set_password(password_reset.new_password)
        
        # Unlock account if it was locked
        if user.is_locked:
            user.unlock_account()
        
        await user.save()
        
        logger.info(
            "Password reset successfully",
            user_id=user.user_id,
            username=user.username
        )
        
        return {"message": "Password reset successfully"}
        
    except PasswordComplexityError as e:
        logger.warning(
            "Password reset failed - complexity",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Password reset failed",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )


@router.get("/check-password-strength")
async def check_password_strength(password: str):
    """Check password strength and get recommendations.
    
    Args:
        password: Password to check
        
    Returns:
        Password strength analysis
    """
    manager = SecurePasswordManager()
    
    try:
        # Try to validate the password
        manager.validate_password_complexity(password)
        
        # Estimate crack time
        crack_time = manager.estimate_crack_time(password)
        
        return {
            "is_valid": True,
            "strength": "strong",
            "estimated_crack_time": crack_time,
            "recommendations": []
        }
        
    except PasswordComplexityError as e:
        # Parse error message for specific issues
        error_msg = str(e)
        recommendations = error_msg.split("; ")
        
        # Determine strength level
        if len(password) < 8:
            strength = "very_weak"
        elif len(password) < 12:
            strength = "weak"
        else:
            strength = "moderate"
        
        return {
            "is_valid": False,
            "strength": strength,
            "estimated_crack_time": "Less than 1 second",
            "recommendations": recommendations
        }


@router.post("/generate-secure-password")
async def generate_secure_password(length: int = 16):
    """Generate a cryptographically secure password.
    
    Args:
        length: Desired password length (minimum 12)
        
    Returns:
        Generated secure password
    """
    manager = SecurePasswordManager()
    
    # Ensure minimum length
    if length < 12:
        length = 12
    elif length > 128:
        length = 128
    
    password = manager.generate_secure_password(length)
    crack_time = manager.estimate_crack_time(password)
    
    return {
        "password": password,
        "length": len(password),
        "estimated_crack_time": crack_time
    }