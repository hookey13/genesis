"""
Login authentication flow with 2FA support.
Handles user login with optional two-factor authentication.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
import redis.asyncio as redis
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from genesis.api.auth import get_db_session, get_current_user, security
from genesis.config.settings import settings
from genesis.data.models_db import User
from genesis.security.audit_logger import AuditLogger
from genesis.security.totp_manager import TOTPManager

logger = structlog.get_logger(__name__)
audit_logger = AuditLogger()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    totp_code: str | None = Field(None, min_length=6, max_length=10, description="TOTP or backup code")


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 14400  # 4 hours in seconds
    requires_2fa: bool = False


class TwoFactorRequiredResponse(BaseModel):
    """Response when 2FA is required."""
    requires_2fa: bool = True
    session_token: str  # Temporary token for 2FA verification
    message: str = "2FA verification required"


async def get_redis_client() -> redis.Redis:
    """Get Redis client for session storage."""
    return await redis.from_url(
        settings.REDIS_URL or "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against bcrypt hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Bcrypt hashed password
        
    Returns:
        True if password matches
    """
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )


async def generate_tokens(user_id: int, username: str) -> dict[str, str]:
    """
    Generate JWT access and refresh tokens.
    
    Args:
        user_id: User ID
        username: Username
        
    Returns:
        Dictionary with access_token and refresh_token
    """
    # Access token (4 hours)
    access_payload = {
        "sub": str(user_id),
        "username": username,
        "type": "access",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(hours=4)
    }

    # Refresh token (30 days)
    refresh_payload = {
        "sub": str(user_id),
        "username": username,
        "type": "refresh",
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(days=30)
    }

    access_token = jwt.encode(
        access_payload,
        settings.JWT_SECRET_KEY,
        algorithm="HS256"
    )

    refresh_token = jwt.encode(
        refresh_payload,
        settings.JWT_SECRET_KEY,
        algorithm="HS256"
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token
    }


async def store_session(
    redis_client: redis.Redis,
    user_id: int,
    token: str,
    expires_in: int = 14400
) -> None:
    """
    Store session in Redis.
    
    Args:
        redis_client: Redis client
        user_id: User ID
        token: Session token
        expires_in: Expiration time in seconds
    """
    session_key = f"session:{user_id}:{token[:20]}"
    session_data = {
        "user_id": user_id,
        "created_at": datetime.now(UTC).isoformat(),
        "token": token
    }

    await redis_client.hset(session_key, mapping=session_data)
    await redis_client.expire(session_key, expires_in)


@router.post("/login", response_model=LoginResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    login_request: LoginRequest,
    db: AsyncSession = Depends(get_db_session)
) -> LoginResponse:
    """
    Authenticate user with optional 2FA.
    
    If 2FA is enabled, requires TOTP code or backup code.
    """
    ip_address = get_remote_address(request)

    try:
        # Find user by username
        result = await db.execute(
            select(User).where(User.username == login_request.username)
        )
        user = result.scalar_one_or_none()

        if not user:
            await audit_logger.log_event(
                event_type="LOGIN_FAILED",
                user_id=None,
                details={
                    "username": login_request.username,
                    "reason": "User not found",
                    "ip_address": ip_address
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Verify password
        if not await verify_password(login_request.password, user.password_hash):
            await audit_logger.log_event(
                event_type="LOGIN_FAILED",
                user_id=user.id,
                details={
                    "username": login_request.username,
                    "reason": "Invalid password",
                    "ip_address": ip_address
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Check if 2FA is required
        if user.two_fa_enabled or user.two_fa_required:
            # If no TOTP code provided, return 2FA required response
            if not login_request.totp_code:
                # Generate temporary session token for 2FA flow
                temp_payload = {
                    "sub": str(user.id),
                    "username": user.username,
                    "type": "2fa_pending",
                    "iat": datetime.now(UTC),
                    "exp": datetime.now(UTC) + timedelta(minutes=5)
                }

                temp_token = jwt.encode(
                    temp_payload,
                    settings.JWT_SECRET_KEY,
                    algorithm="HS256"
                )

                await audit_logger.log_event(
                    event_type="LOGIN_2FA_REQUIRED",
                    user_id=user.id,
                    details={
                        "username": user.username,
                        "ip_address": ip_address
                    }
                )

                return TwoFactorRequiredResponse(
                    requires_2fa=True,
                    session_token=temp_token,
                    message="Please provide your 2FA code"
                )

            # Verify 2FA code
            totp_manager = TOTPManager()
            is_valid, code_type = await totp_manager.validate_2fa(
                user_id=str(user.id),
                code=login_request.totp_code
            )

            if not is_valid:
                await audit_logger.log_event(
                    event_type="LOGIN_2FA_FAILED",
                    user_id=user.id,
                    details={
                        "username": user.username,
                        "ip_address": ip_address,
                        "code_type": code_type
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid 2FA code"
                )

            logger.info(
                "2FA verification successful during login",
                user_id=user.id,
                code_type=code_type
            )

        # Generate tokens
        tokens = await generate_tokens(user.id, user.username)

        # Store session in Redis
        redis_client = await get_redis_client()
        await store_session(
            redis_client,
            user.id,
            tokens["access_token"]
        )

        # Update last login
        user.last_login = datetime.now(UTC)
        await db.commit()

        # Log successful login
        await audit_logger.log_event(
            event_type="LOGIN_SUCCESS",
            user_id=user.id,
            details={
                "username": user.username,
                "ip_address": ip_address,
                "2fa_used": user.two_fa_enabled
            }
        )

        logger.info(
            "User logged in successfully",
            user_id=user.id,
            username=user.username,
            two_fa_used=user.two_fa_enabled
        )

        return LoginResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="bearer",
            expires_in=14400,
            requires_2fa=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Login error",
            error=str(e),
            username=login_request.username
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/verify-2fa", response_model=LoginResponse)
@limiter.limit("5/minute")
async def verify_2fa_login(
    request: Request,
    session_token: str = Field(..., description="Temporary session token"),
    totp_code: str = Field(..., min_length=6, max_length=10),
    db: AsyncSession = Depends(get_db_session)
) -> LoginResponse:
    """
    Complete login with 2FA verification.
    
    Uses temporary session token from initial login attempt.
    """
    ip_address = get_remote_address(request)

    try:
        # Decode temporary session token
        try:
            payload = jwt.decode(
                session_token,
                settings.JWT_SECRET_KEY,
                algorithms=["HS256"]
            )

            if payload.get("type") != "2fa_pending":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session token"
                )

            user_id = int(payload.get("sub"))

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )

        # Get user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        # Verify 2FA code
        totp_manager = TOTPManager()
        is_valid, code_type = await totp_manager.validate_2fa(
            user_id=str(user.id),
            code=totp_code
        )

        if not is_valid:
            await audit_logger.log_event(
                event_type="2FA_LOGIN_FAILED",
                user_id=user.id,
                details={
                    "username": user.username,
                    "ip_address": ip_address,
                    "code_type": code_type
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA code"
            )

        # Generate full tokens
        tokens = await generate_tokens(user.id, user.username)

        # Store session in Redis
        redis_client = await get_redis_client()
        await store_session(
            redis_client,
            user.id,
            tokens["access_token"]
        )

        # Update last login
        user.last_login = datetime.now(UTC)
        await db.commit()

        # Log successful login
        await audit_logger.log_event(
            event_type="2FA_LOGIN_SUCCESS",
            user_id=user.id,
            details={
                "username": user.username,
                "ip_address": ip_address,
                "code_type": code_type
            }
        )

        logger.info(
            "2FA login completed successfully",
            user_id=user.id,
            username=user.username,
            code_type=code_type
        )

        return LoginResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="bearer",
            expires_in=14400,
            requires_2fa=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "2FA verification error",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA verification failed"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    token: str = Depends(security)
) -> dict[str, str]:
    """
    Logout user and invalidate session.
    """
    try:
        # Remove session from Redis
        redis_client = await get_redis_client()
        session_key = f"session:{current_user.id}:{token[:20]}"
        await redis_client.delete(session_key)

        # Log logout
        await audit_logger.log_event(
            event_type="LOGOUT",
            user_id=current_user.id,
            details={"username": current_user.username}
        )

        logger.info(
            "User logged out",
            user_id=current_user.id,
            username=current_user.username
        )

        return {"message": "Logged out successfully"}

    except Exception as e:
        logger.error(
            "Logout error",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )
