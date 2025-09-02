"""JWT Authentication Endpoints for Genesis Trading Platform.

Provides login, logout, token refresh, and session management endpoints
using JWT tokens for secure authentication.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import structlog
import redis.asyncio as redis
from genesis.security.jwt_manager import (
    JWTSessionManager, 
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenRevokedError
)
from genesis.security.vault_client import VaultClient
from genesis.validation.auth import User, PasswordValidator
from genesis.config.settings import Settings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/jwt", tags=["jwt-authentication"])
security = HTTPBearer()


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    
    
class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str
    session_id: str
    

class RefreshRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str
    

class SessionResponse(BaseModel):
    """Session information response."""
    session_id: str
    created_at: str
    user_id: str
    username: str
    

class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


class PasswordChangeRequest(BaseModel):
    """Password change request model."""
    current_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8)


# Dependency injection
async def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    settings = Settings()
    # Use Redis connection from settings or default to localhost
    redis_host = getattr(settings, 'redis_host', 'localhost')
    redis_port = getattr(settings, 'redis_port', 6379)
    
    return redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=False
    )


async def get_vault_client() -> VaultClient:
    """Get Vault client instance."""
    settings = Settings()
    vault_url = getattr(settings, 'vault_url', 'http://localhost:8200')
    vault_token = getattr(settings, 'vault_token', 'dev-token')
    
    return VaultClient(
        vault_url=vault_url,
        vault_token=vault_token
    )


async def get_jwt_manager(
    vault_client: VaultClient = Depends(get_vault_client),
    redis_client: redis.Redis = Depends(get_redis_client)
) -> JWTSessionManager:
    """Get JWT session manager instance."""
    return JWTSessionManager(vault_client, redis_client)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    try:
        claims = await jwt_manager.verify_token(token, 'access')
        return {
            'user_id': claims.user_id,
            'username': claims.username,
            'role': claims.role,
            'session_id': claims.session_id
        }
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except TokenRevokedError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token revoked",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


# Authentication Endpoints
@router.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(
    request: LoginRequest,
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> LoginResponse:
    """Authenticate user and generate JWT token pair.
    
    Args:
        request: Login credentials
        jwt_manager: JWT session manager
        
    Returns:
        JWT token pair and session information
    """
    try:
        # Authenticate user credentials
        user = await User.authenticate(request.username, request.password)
        if not user:
            logger.warning(
                "login_failed",
                username=request.username,
                reason="invalid_credentials"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Generate JWT token pair
        token_data = await jwt_manager.generate_token_pair(
            str(user.id),
            user.username,
            user.role
        )
        
        # Log successful login
        logger.info(
            "user_login_success",
            user_id=user.id,
            username=user.username,
            session_id=token_data['session_id']
        )
        
        return LoginResponse(**token_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e), username=request.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def refresh_token(
    request: RefreshRequest,
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> LoginResponse:
    """Refresh access token using refresh token.
    
    Args:
        request: Refresh token
        jwt_manager: JWT session manager
        
    Returns:
        New JWT token pair
    """
    try:
        # Refresh token
        new_tokens = await jwt_manager.refresh_token(request.refresh_token)
        
        logger.info(
            "token_refreshed",
            session_id=new_tokens['session_id']
        )
        
        return LoginResponse(**new_tokens)
        
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except TokenRevokedError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> MessageResponse:
    """Logout user and invalidate current session.
    
    Args:
        current_user: Current authenticated user
        jwt_manager: JWT session manager
        
    Returns:
        Success message
    """
    try:
        session_id = current_user['session_id']
        
        # Invalidate session
        await jwt_manager.invalidate_session(session_id)
        
        logger.info(
            "user_logout",
            user_id=current_user['user_id'],
            username=current_user['username'],
            session_id=session_id
        )
        
        return MessageResponse(message="Logged out successfully")
        
    except Exception as e:
        logger.error("Logout error", error=str(e), user_id=current_user.get('user_id'))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/sessions", response_model=List[SessionResponse], status_code=status.HTTP_200_OK)
async def get_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> List[SessionResponse]:
    """Get all active sessions for current user.
    
    Args:
        current_user: Current authenticated user
        jwt_manager: JWT session manager
        
    Returns:
        List of active sessions
    """
    try:
        user_id = current_user['user_id']
        
        # Get active sessions
        sessions = await jwt_manager.get_active_sessions(user_id)
        
        logger.info(
            "sessions_retrieved",
            user_id=user_id,
            session_count=len(sessions)
        )
        
        return [SessionResponse(**session) for session in sessions]
        
    except Exception as e:
        logger.error("Get sessions error", error=str(e), user_id=current_user.get('user_id'))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@router.delete("/sessions/{session_id}", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def revoke_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> MessageResponse:
    """Revoke a specific session.
    
    Args:
        session_id: Session ID to revoke
        current_user: Current authenticated user
        jwt_manager: JWT session manager
        
    Returns:
        Success message
    """
    try:
        user_id = current_user['user_id']
        
        # Verify session belongs to user
        user_sessions = await jwt_manager.get_active_sessions(user_id)
        session_ids = [s['session_id'] for s in user_sessions]
        
        if session_id not in session_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Revoke session
        await jwt_manager.invalidate_session(session_id)
        
        logger.info(
            "session_revoked",
            user_id=user_id,
            revoked_session_id=session_id,
            current_session_id=current_user['session_id']
        )
        
        return MessageResponse(message="Session revoked successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Revoke session error", error=str(e), user_id=current_user.get('user_id'))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        )


@router.post("/logout-all", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def logout_all_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> MessageResponse:
    """Logout from all sessions (invalidate all user sessions).
    
    Args:
        current_user: Current authenticated user
        jwt_manager: JWT session manager
        
    Returns:
        Success message
    """
    try:
        user_id = current_user['user_id']
        
        # Invalidate all user sessions
        await jwt_manager.invalidate_all_user_sessions(user_id)
        
        logger.info(
            "all_sessions_logged_out",
            user_id=user_id,
            username=current_user['username']
        )
        
        return MessageResponse(message="All sessions logged out successfully")
        
    except Exception as e:
        logger.error("Logout all sessions error", error=str(e), user_id=current_user.get('user_id'))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout all sessions"
        )


@router.post("/change-password", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def change_password(
    request: PasswordChangeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    jwt_manager: JWTSessionManager = Depends(get_jwt_manager)
) -> MessageResponse:
    """Change user password and invalidate all sessions.
    
    Args:
        request: Password change request
        current_user: Current authenticated user
        jwt_manager: JWT session manager
        
    Returns:
        Success message
    """
    try:
        user_id = current_user['user_id']
        username = current_user['username']
        
        # Verify current password
        user = await User.authenticate(username, request.current_password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        validator = PasswordValidator()
        is_valid, errors = validator.validate_password(request.new_password, username)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid password: {', '.join(errors)}"
            )
        
        # Update password
        await user.update_password(request.new_password)
        
        # Invalidate all sessions on password change
        await jwt_manager.invalidate_all_user_sessions(user_id)
        
        logger.info(
            "password_changed",
            user_id=user_id,
            username=username,
            sessions_invalidated=True
        )
        
        return MessageResponse(message="Password changed successfully. Please login again.")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change error", error=str(e), user_id=current_user.get('user_id'))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


@router.get("/verify", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def verify_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Verify current JWT token and return user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information from token
    """
    return {
        "user_id": current_user['user_id'],
        "username": current_user['username'],
        "role": current_user['role'],
        "session_id": current_user['session_id'],
        "authenticated": True
    }


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Health check endpoint for JWT authentication service.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "jwt-authentication",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }