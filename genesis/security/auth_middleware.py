"""Authentication Middleware for Genesis Trading Platform.

Provides automatic JWT token verification and request authentication
for all protected endpoints.
"""

from aiohttp import web
from aiohttp.web_request import Request
import structlog
from typing import Callable, Set
from .jwt_manager import JWTSessionManager, TokenExpiredError, TokenRevokedError, TokenInvalidError

logger = structlog.get_logger(__name__)


class AuthenticationMiddleware:
    """JWT authentication middleware for automatic request verification."""
    
    def __init__(self, jwt_manager: JWTSessionManager):
        """Initialize authentication middleware.
        
        Args:
            jwt_manager: JWT session manager instance
        """
        self.jwt_manager = jwt_manager
        
        # Paths that don't require authentication
        self.public_paths: Set[str] = {
            '/auth/login',
            '/auth/register',
            '/health',
            '/metrics',
            '/api/health',
            '/api/metrics'
        }
        
        # Paths that require specific roles
        self.role_requirements = {
            '/admin': ['admin'],
            '/api/admin': ['admin'],
            '/trading/strategies': ['trader', 'admin'],
            '/api/trading': ['trader', 'admin']
        }
    
    @web.middleware
    async def middleware(self, request: Request, handler: Callable):
        """Process request with JWT authentication.
        
        Args:
            request: Incoming HTTP request
            handler: Next handler in chain
            
        Returns:
            Response from handler or error response
        """
        # Skip auth for public paths
        if self._is_public_path(request.path):
            return await handler(request)
        
        # Extract Bearer token
        auth_header = request.headers.get('Authorization', '')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning(
                "auth_missing_header",
                path=request.path,
                method=request.method,
                remote=request.remote
            )
            return web.json_response(
                {'error': 'Missing or invalid authorization header'}, 
                status=401
            )
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            # Verify token
            claims = await self.jwt_manager.verify_token(token, 'access')
            
            # Check role requirements
            if not self._check_role_requirements(request.path, claims.role):
                logger.warning(
                    "auth_insufficient_role",
                    user_id=claims.user_id,
                    username=claims.username,
                    role=claims.role,
                    path=request.path,
                    required_roles=self._get_required_roles(request.path)
                )
                return web.json_response(
                    {'error': 'Insufficient permissions'}, 
                    status=403
                )
            
            # Add claims to request context
            request['user'] = {
                'user_id': claims.user_id,
                'username': claims.username,
                'role': claims.role,
                'session_id': claims.session_id
            }
            
            # Log successful authentication
            logger.info(
                "authenticated_request",
                user_id=claims.user_id,
                username=claims.username,
                path=request.path,
                method=request.method,
                token_hash=self.jwt_manager.create_token_hash(token)
            )
            
            # Call next handler
            response = await handler(request)
            
            # Add security headers to response
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            
            return response
            
        except TokenExpiredError:
            logger.warning(
                "auth_token_expired",
                path=request.path,
                token_hash=self.jwt_manager.create_token_hash(token)
            )
            return web.json_response({'error': 'Token expired'}, status=401)
            
        except TokenRevokedError:
            logger.warning(
                "auth_token_revoked",
                path=request.path,
                token_hash=self.jwt_manager.create_token_hash(token)
            )
            return web.json_response({'error': 'Token revoked'}, status=401)
            
        except TokenInvalidError as e:
            logger.warning(
                "auth_token_invalid",
                path=request.path,
                error=str(e),
                token_hash=self.jwt_manager.create_token_hash(token)
            )
            return web.json_response({'error': f'Invalid token: {str(e)}'}, status=401)
            
        except Exception as e:
            logger.error(
                "auth_error",
                path=request.path,
                error=str(e),
                token_hash=self.jwt_manager.create_token_hash(token)
            )
            return web.json_response({'error': 'Authentication failed'}, status=401)
    
    def _is_public_path(self, path: str) -> bool:
        """Check if path is public and doesn't require authentication.
        
        Args:
            path: Request path
            
        Returns:
            True if path is public
        """
        # Exact match
        if path in self.public_paths:
            return True
        
        # Check if path starts with any public prefix
        public_prefixes = ['/static/', '/docs/', '/.well-known/']
        return any(path.startswith(prefix) for prefix in public_prefixes)
    
    def _get_required_roles(self, path: str) -> list:
        """Get required roles for a path.
        
        Args:
            path: Request path
            
        Returns:
            List of required roles
        """
        # Check exact path match
        if path in self.role_requirements:
            return self.role_requirements[path]
        
        # Check prefix matches
        for prefix, roles in self.role_requirements.items():
            if path.startswith(prefix):
                return roles
        
        # Default: any authenticated user
        return []
    
    def _check_role_requirements(self, path: str, user_role: str) -> bool:
        """Check if user role meets path requirements.
        
        Args:
            path: Request path
            user_role: User's role
            
        Returns:
            True if role requirements are met
        """
        required_roles = self._get_required_roles(path)
        
        # No specific role required (any authenticated user)
        if not required_roles:
            return True
        
        # Admin can access everything
        if user_role == 'admin':
            return True
        
        # Check if user role is in required roles
        return user_role in required_roles
    
    def add_public_path(self, path: str):
        """Add a path to the public paths list.
        
        Args:
            path: Path to add
        """
        self.public_paths.add(path)
        logger.info("auth_public_path_added", path=path)
    
    def remove_public_path(self, path: str):
        """Remove a path from the public paths list.
        
        Args:
            path: Path to remove
        """
        self.public_paths.discard(path)
        logger.info("auth_public_path_removed", path=path)
    
    def set_role_requirement(self, path: str, roles: list):
        """Set role requirements for a path.
        
        Args:
            path: Path to set requirements for
            roles: List of allowed roles
        """
        self.role_requirements[path] = roles
        logger.info("auth_role_requirement_set", path=path, roles=roles)