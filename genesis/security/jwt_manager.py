"""JWT Session Management System for Genesis Trading Platform.

Provides secure token generation, verification, and session management
with Redis backing for scalable, production-ready authentication.
"""

import jwt
import secrets
import redis.asyncio as redis
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import hashlib
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TokenClaims:
    """JWT token claims structure."""
    user_id: str
    username: str
    role: str
    session_id: str
    issued_at: datetime
    expires_at: datetime
    not_before: datetime
    token_id: str

    
class TokenError(Exception):
    """Base class for JWT token errors."""
    pass


class TokenExpiredError(TokenError):
    """Token has expired."""
    pass


class TokenInvalidError(TokenError):
    """Token is invalid or malformed."""
    pass


class TokenRevokedError(TokenError):
    """Token has been revoked."""
    pass


class JWTSessionManager:
    """Secure JWT session management with Redis backing."""
    
    def __init__(self, vault_manager, redis_client: redis.Redis):
        """Initialize JWT session manager.
        
        Args:
            vault_manager: Vault manager for secret retrieval
            redis_client: Redis client for session storage
        """
        self.vault = vault_manager
        self.redis = redis_client
        self.algorithm = 'HS256'
        self.token_ttl = timedelta(hours=1)
        self.refresh_ttl = timedelta(days=7)
        self.blacklist_ttl = timedelta(days=8)  # Longer than refresh TTL
        
        # Token prefixes for Redis keys
        self.session_prefix = 'session:'
        self.blacklist_prefix = 'blacklist:'
        self.user_sessions_prefix = 'user_sessions:'
        
    async def get_signing_key(self) -> str:
        """Get JWT signing key from Vault."""
        return await self.vault.get_secret('genesis-secrets/jwt', 'signing_key')
    
    async def generate_token_pair(self, user_id: str, username: str, 
                                role: str) -> Dict[str, Any]:
        """Generate access and refresh token pair.
        
        Args:
            user_id: User identifier
            username: Username
            role: User role
            
        Returns:
            Dictionary containing token pair and metadata
        """
        now = datetime.now(timezone.utc)
        session_id = secrets.token_urlsafe(32)
        access_token_id = secrets.token_urlsafe(16)
        refresh_token_id = secrets.token_urlsafe(16)
        
        # Access token claims
        access_claims = {
            'sub': user_id,
            'username': username,
            'role': role,
            'session_id': session_id,
            'iat': now,
            'exp': now + self.token_ttl,
            'nbf': now,
            'jti': access_token_id,
            'type': 'access'
        }
        
        # Refresh token claims  
        refresh_claims = {
            'sub': user_id,
            'username': username,
            'session_id': session_id,
            'iat': now,
            'exp': now + self.refresh_ttl,
            'nbf': now,
            'jti': refresh_token_id,
            'type': 'refresh'
        }
        
        # Sign tokens
        signing_key = await self.get_signing_key()
        
        access_token = jwt.encode(access_claims, signing_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_claims, signing_key, algorithm=self.algorithm)
        
        # Store session info in Redis
        session_data = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'created_at': now.isoformat(),
            'access_token_id': access_token_id,
            'refresh_token_id': refresh_token_id,
            'is_active': 'true'
        }
        
        await self.redis.hset(
            f"{self.session_prefix}{session_id}",
            mapping=session_data
        )
        await self.redis.expire(f"{self.session_prefix}{session_id}", int(self.refresh_ttl.total_seconds()))
        
        # Track user sessions
        await self.redis.sadd(f"{self.user_sessions_prefix}{user_id}", session_id)
        await self.redis.expire(f"{self.user_sessions_prefix}{user_id}", int(self.refresh_ttl.total_seconds()))
        
        logger.info(
            "jwt_token_generated",
            user_id=user_id,
            username=username,
            session_id=session_id,
            access_token_hash=self.create_token_hash(access_token)
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_in': int(self.token_ttl.total_seconds()),
            'token_type': 'Bearer',
            'session_id': session_id
        }
    
    async def verify_token(self, token: str, token_type: str = 'access') -> TokenClaims:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type ('access' or 'refresh')
            
        Returns:
            TokenClaims object with decoded claims
            
        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid
            TokenRevokedError: If token has been revoked
        """
        try:
            signing_key = await self.get_signing_key()
            
            # Decode and verify token
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=[self.algorithm],
                options={
                    'verify_exp': True,
                    'verify_nbf': True,
                    'verify_iat': True,
                    'require': ['sub', 'jti', 'exp', 'iat']
                }
            )
            
            # Verify token type
            if payload.get('type') != token_type:
                raise TokenInvalidError(f"Expected {token_type} token")
            
            # Check if token is blacklisted
            token_id = payload['jti']
            if await self.redis.exists(f"{self.blacklist_prefix}{token_id}"):
                raise TokenRevokedError("Token has been revoked")
            
            # Verify session still exists and is active
            session_id = payload['session_id']
            session_exists = await self.redis.exists(f"{self.session_prefix}{session_id}")
            if not session_exists:
                raise TokenInvalidError("Session no longer exists")
            
            session_data = await self.redis.hgetall(f"{self.session_prefix}{session_id}")
            if session_data.get(b'is_active') != b'true':
                raise TokenInvalidError("Session is inactive")
            
            # Return token claims
            return TokenClaims(
                user_id=payload['sub'],
                username=payload['username'],
                role=payload.get('role', 'user'),
                session_id=payload['session_id'],
                issued_at=datetime.fromtimestamp(payload['iat'], tz=timezone.utc),
                expires_at=datetime.fromtimestamp(payload['exp'], tz=timezone.utc),
                not_before=datetime.fromtimestamp(payload['nbf'], tz=timezone.utc),
                token_id=payload['jti']
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("jwt_token_expired", token_hash=self.create_token_hash(token))
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning("jwt_token_invalid", error=str(e), token_hash=self.create_token_hash(token))
            raise TokenInvalidError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error("jwt_verification_failed", error=str(e))
            raise TokenInvalidError(f"Token verification failed: {str(e)}")
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New token pair dictionary
        """
        # Verify refresh token
        claims = await self.verify_token(refresh_token, 'refresh')
        
        # Generate new access token
        new_token_pair = await self.generate_token_pair(
            claims.user_id,
            claims.username,
            claims.role
        )
        
        # Blacklist old refresh token
        await self.blacklist_token(claims.token_id)
        
        logger.info(
            "jwt_token_refreshed",
            user_id=claims.user_id,
            old_token_hash=self.create_token_hash(refresh_token),
            new_session_id=new_token_pair['session_id']
        )
        
        return new_token_pair
    
    async def blacklist_token(self, token_id: str):
        """Add token to blacklist.
        
        Args:
            token_id: JWT ID (jti) to blacklist
        """
        await self.redis.setex(
            f"{self.blacklist_prefix}{token_id}",
            int(self.blacklist_ttl.total_seconds()),
            "revoked"
        )
        logger.info("jwt_token_blacklisted", token_id=token_id)
    
    async def invalidate_session(self, session_id: str):
        """Invalidate a specific session.
        
        Args:
            session_id: Session ID to invalidate
        """
        # Get session data
        session_data = await self.redis.hgetall(f"{self.session_prefix}{session_id}")
        if not session_data:
            return
        
        # Blacklist associated tokens
        access_token_id = session_data.get(b'access_token_id')
        refresh_token_id = session_data.get(b'refresh_token_id')
        
        if access_token_id:
            await self.blacklist_token(access_token_id.decode('utf-8'))
        if refresh_token_id:
            await self.blacklist_token(refresh_token_id.decode('utf-8'))
        
        # Mark session as inactive
        await self.redis.hset(f"{self.session_prefix}{session_id}", 'is_active', 'false')
        
        # Remove from user sessions
        user_id = session_data.get(b'user_id')
        if user_id:
            await self.redis.srem(f"{self.user_sessions_prefix}{user_id.decode('utf-8')}", session_id)
        
        logger.info(
            "jwt_session_invalidated",
            session_id=session_id,
            user_id=user_id.decode('utf-8') if user_id else None
        )
    
    async def invalidate_all_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user (e.g., on password change).
        
        Args:
            user_id: User ID whose sessions to invalidate
        """
        # Get all user sessions
        session_ids = await self.redis.smembers(f"{self.user_sessions_prefix}{user_id}")
        
        # Invalidate each session
        for session_id in session_ids:
            await self.invalidate_session(session_id.decode('utf-8'))
        
        # Clear user sessions set
        await self.redis.delete(f"{self.user_sessions_prefix}{user_id}")
        
        logger.info(
            "jwt_all_sessions_invalidated",
            user_id=user_id,
            session_count=len(session_ids)
        )
    
    async def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for a user.
        
        Args:
            user_id: User ID to get sessions for
            
        Returns:
            List of active session dictionaries
        """
        session_ids = await self.redis.smembers(f"{self.user_sessions_prefix}{user_id}")
        sessions = []
        
        for session_id in session_ids:
            session_data = await self.redis.hgetall(f"{self.session_prefix}{session_id.decode('utf-8')}")
            if session_data and session_data.get(b'is_active') == b'true':
                sessions.append({
                    'session_id': session_id.decode('utf-8'),
                    'created_at': session_data.get(b'created_at', b'').decode('utf-8'),
                    'user_id': session_data.get(b'user_id', b'').decode('utf-8'),
                    'username': session_data.get(b'username', b'').decode('utf-8')
                })
        
        return sessions
    
    def create_token_hash(self, token: str) -> str:
        """Create hash of token for logging (never log full tokens).
        
        Args:
            token: JWT token to hash
            
        Returns:
            Truncated SHA256 hash of token
        """
        return hashlib.sha256(token.encode()).hexdigest()[:16]