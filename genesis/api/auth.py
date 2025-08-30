"""Authentication and authorization middleware for Project GENESIS API.

Implements permission checking and API key validation using the
principle of least privilege.
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
import hashlib
import hmac
import time
import structlog
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from genesis.security.permissions import (
    PermissionChecker,
    Role,
    Resource,
    Action,
    User,
    APIKeyPermissionManager
)
from genesis.security.vault_client import VaultClient
from genesis.config.settings import settings

logger = structlog.get_logger(__name__)

# Security scheme for FastAPI
security = HTTPBearer()


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    pass


class APIAuthenticator:
    """Handles API authentication and authorization."""
    
    def __init__(
        self,
        vault_client: Optional[VaultClient] = None,
        permission_checker: Optional[PermissionChecker] = None,
        user_repository: Optional[Any] = None
    ):
        """Initialize API authenticator.
        
        Args:
            vault_client: Vault client for API key validation
            permission_checker: Permission checker for authorization
            user_repository: Repository for user data
        """
        self.vault_client = vault_client or settings.get_vault_client()
        self.permission_checker = permission_checker or PermissionChecker(user_repository)
        self.user_repository = user_repository
        self._api_key_cache: Dict[str, Dict[str, Any]] = {}
    
    def validate_api_key(self, api_key: str, api_secret: str) -> Optional[Dict[str, Any]]:
        """Validate API key and secret.
        
        Args:
            api_key: API key
            api_secret: API secret
            
        Returns:
            API key metadata if valid, None otherwise
        """
        # Check cache first
        if api_key in self._api_key_cache:
            cached = self._api_key_cache[api_key]
            if cached["api_secret"] == api_secret:
                logger.debug("API key validated from cache", api_key=api_key[:8] + "...")
                return cached
        
        # Get keys from Vault
        trading_keys = self.vault_client.get_exchange_api_keys(read_only=False)
        read_only_keys = self.vault_client.get_exchange_api_keys(read_only=True)
        
        # Check if it's a trading key
        if trading_keys and api_key == trading_keys["api_key"]:
            if api_secret == trading_keys["api_secret"]:
                metadata = {
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "key_type": "trading",
                    "permissions": APIKeyPermissionManager.create_trading_key_permissions()
                }
                self._api_key_cache[api_key] = metadata
                logger.info("Trading API key validated", api_key=api_key[:8] + "...")
                return metadata
        
        # Check if it's a read-only key
        if read_only_keys and api_key == read_only_keys["api_key"]:
            if api_secret == read_only_keys["api_secret"]:
                metadata = {
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "key_type": "read_only",
                    "permissions": APIKeyPermissionManager.create_read_only_key_permissions()
                }
                self._api_key_cache[api_key] = metadata
                logger.info("Read-only API key validated", api_key=api_key[:8] + "...")
                return metadata
        
        logger.warning("Invalid API key", api_key=api_key[:8] + "...")
        return None
    
    def validate_hmac_signature(
        self,
        api_secret: str,
        timestamp: str,
        method: str,
        path: str,
        body: str,
        signature: str
    ) -> bool:
        """Validate HMAC signature for API request.
        
        Args:
            api_secret: API secret for HMAC
            timestamp: Request timestamp
            method: HTTP method
            path: Request path
            body: Request body
            signature: HMAC signature to validate
            
        Returns:
            True if signature is valid
        """
        # Check timestamp is recent (within 5 minutes)
        try:
            request_time = int(timestamp)
            current_time = int(time.time())
            if abs(current_time - request_time) > 300:  # 5 minutes
                logger.warning("Request timestamp too old", 
                             timestamp=timestamp,
                             current_time=current_time)
                return False
        except ValueError:
            logger.error("Invalid timestamp", timestamp=timestamp)
            return False
        
        # Create message to sign
        message = f"{timestamp}{method}{path}{body}"
        
        # Calculate expected signature
        expected_signature = hmac.new(
            api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        is_valid = hmac.compare_digest(expected_signature, signature)
        
        if not is_valid:
            logger.warning("Invalid HMAC signature")
        
        return is_valid
    
    async def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authenticate an API request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Authentication context
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Extract authentication headers
        api_key = request.headers.get("X-API-Key")
        signature = request.headers.get("X-Signature")
        timestamp = request.headers.get("X-Timestamp")
        
        if not all([api_key, signature, timestamp]):
            raise AuthenticationError("Missing authentication headers")
        
        # Get API key metadata
        key_metadata = self.validate_api_key(api_key, "")  # Need to get secret first
        if not key_metadata:
            raise AuthenticationError("Invalid API key")
        
        # Get request body
        body = await request.body()
        body_str = body.decode() if body else ""
        
        # Validate HMAC signature
        is_valid = self.validate_hmac_signature(
            api_secret=key_metadata["api_secret"],
            timestamp=timestamp,
            method=request.method,
            path=request.url.path,
            body=body_str,
            signature=signature
        )
        
        if not is_valid:
            raise AuthenticationError("Invalid signature")
        
        # Get user associated with API key
        user = await self._get_user_for_api_key(api_key)
        
        return {
            "user": user,
            "api_key": api_key,
            "key_type": key_metadata["key_type"],
            "key_permissions": key_metadata["permissions"]
        }
    
    async def _get_user_for_api_key(self, api_key: str) -> User:
        """Get user associated with an API key.
        
        Args:
            api_key: API key
            
        Returns:
            User object
        """
        # In production, look up user from database
        # For now, return a default user based on key type
        
        # Check if it's a trading or read-only key
        key_metadata = self._api_key_cache.get(api_key)
        if key_metadata:
            if key_metadata["key_type"] == "trading":
                return User(user_id="trader", role=Role.TRADER)
            else:
                return User(user_id="reader", role=Role.READ_ONLY)
        
        # Default to read-only
        return User(user_id="unknown", role=Role.READ_ONLY)
    
    async def authorize_request(
        self,
        auth_context: Dict[str, Any],
        resource: Resource,
        action: Action
    ) -> bool:
        """Authorize a request based on permissions.
        
        Args:
            auth_context: Authentication context from authenticate_request
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            True if authorized
            
        Raises:
            AuthorizationError: If authorization fails
        """
        user = auth_context["user"]
        api_key = auth_context["api_key"]
        
        # Check permission
        has_permission = await self.permission_checker.check_permission(
            user_id=user.user_id,
            resource=resource,
            action=action,
            api_key=api_key
        )
        
        if not has_permission:
            raise AuthorizationError(
                f"Permission denied: {resource.value}:{action.value}"
            )
        
        return True
    
    def clear_cache(self, api_key: Optional[str] = None):
        """Clear API key cache.
        
        Args:
            api_key: Specific key to clear, or None for all
        """
        if api_key:
            self._api_key_cache.pop(api_key, None)
        else:
            self._api_key_cache.clear()


# Global authenticator instance
_authenticator: Optional[APIAuthenticator] = None


def get_authenticator() -> APIAuthenticator:
    """Get or create global authenticator instance.
    
    Returns:
        APIAuthenticator instance
    """
    global _authenticator
    if _authenticator is None:
        _authenticator = APIAuthenticator()
    return _authenticator


async def authenticate(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> Dict[str, Any]:
    """FastAPI dependency for authentication.
    
    Args:
        credentials: Bearer token credentials
        request: FastAPI request
        
    Returns:
        Authentication context
        
    Raises:
        HTTPException: If authentication fails
    """
    authenticator = get_authenticator()
    
    try:
        auth_context = await authenticator.authenticate_request(request)
        return auth_context
    except AuthenticationError as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(status_code=401, detail=str(e))


def require_permission(resource: Resource, action: Action):
    """Decorator for FastAPI endpoints requiring specific permissions.
    
    Args:
        resource: Resource being accessed
        action: Action being performed
        
    Returns:
        FastAPI dependency
    """
    async def permission_dependency(
        auth_context: Dict[str, Any] = Depends(authenticate)
    ):
        """Check permission for the authenticated user.
        
        Args:
            auth_context: Authentication context
            
        Raises:
            HTTPException: If authorization fails
        """
        authenticator = get_authenticator()
        
        try:
            await authenticator.authorize_request(auth_context, resource, action)
        except AuthorizationError as e:
            logger.warning("Authorization failed", 
                         user_id=auth_context["user"].user_id,
                         resource=resource.value,
                         action=action.value,
                         error=str(e))
            raise HTTPException(status_code=403, detail=str(e))
    
    return Depends(permission_dependency)


# Convenience dependencies for common permissions
require_read_market_data = require_permission(Resource.MARKET_DATA, Action.READ)
require_read_account = require_permission(Resource.ACCOUNT_INFO, Action.READ)
require_write_orders = require_permission(Resource.ORDERS, Action.WRITE)
require_execute_trades = require_permission(Resource.TRADES, Action.EXECUTE)
require_admin = require_permission(Resource.CONFIG, Action.ADMIN)