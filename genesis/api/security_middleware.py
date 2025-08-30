"""Security middleware for API network protection.

Implements IP whitelisting, rate limiting, and security headers.
"""

import time
from typing import Optional, Dict, Any, Callable
from collections import defaultdict
import structlog
from fastapi import Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from genesis.security.ip_whitelist import (
    IPWhitelistManager,
    NetworkZone,
    NetworkSegmentation
)

logger = structlog.get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware for API protection."""
    
    def __init__(
        self,
        app,
        whitelist_manager: Optional[IPWhitelistManager] = None,
        rate_limit: int = 10,  # Requests per second
        enable_ip_whitelist: bool = True,
        enable_rate_limit: bool = True,
        enable_security_headers: bool = True
    ):
        """Initialize security middleware.
        
        Args:
            app: FastAPI application
            whitelist_manager: IP whitelist manager
            rate_limit: Maximum requests per second per IP
            enable_ip_whitelist: Enable IP whitelist checking
            enable_rate_limit: Enable rate limiting
            enable_security_headers: Enable security headers
        """
        super().__init__(app)
        self.whitelist_manager = whitelist_manager or IPWhitelistManager()
        self.network_segmentation = NetworkSegmentation(self.whitelist_manager)
        self.rate_limit = rate_limit
        self.enable_ip_whitelist = enable_ip_whitelist
        self.enable_rate_limit = enable_rate_limit
        self.enable_security_headers = enable_security_headers
        
        # Rate limiting state
        self._request_counts: Dict[str, list] = defaultdict(list)
        self._blocked_ips: Dict[str, float] = {}  # IP -> block expiry time
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request through security checks.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
            
        Returns:
            Response after security processing
        """
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            logger.warning("Blocked IP attempted access", ip=client_ip)
            raise HTTPException(status_code=403, detail="IP temporarily blocked")
        
        # IP whitelist check
        if self.enable_ip_whitelist:
            if not self._check_ip_whitelist(client_ip, request.url.path):
                logger.warning("IP not whitelisted", 
                             ip=client_ip, 
                             path=request.url.path)
                raise HTTPException(status_code=403, detail="IP not authorized")
        
        # Rate limiting
        if self.enable_rate_limit:
            if not self._check_rate_limit(client_ip):
                # Block IP temporarily
                self._block_ip(client_ip, duration=60)  # Block for 1 minute
                logger.warning("Rate limit exceeded, IP blocked", 
                             ip=client_ip,
                             limit=self.rate_limit)
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Log request
        start_time = time.time()
        logger.info("API request", 
                   ip=client_ip,
                   method=request.method,
                   path=request.url.path)
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        if self.enable_security_headers:
            self._add_security_headers(response)
        
        # Log response
        duration = time.time() - start_time
        logger.info("API response",
                   ip=client_ip,
                   method=request.method,
                   path=request.url.path,
                   status=response.status_code,
                   duration=duration)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (for proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        return request.client.host if request.client else "unknown"
    
    def _check_ip_whitelist(self, ip: str, path: str) -> bool:
        """Check if IP is whitelisted for the requested path.
        
        Args:
            ip: Client IP address
            path: Request path
            
        Returns:
            True if IP is allowed
        """
        # Determine required zone based on path
        zone = self._get_zone_for_path(path)
        
        # Check if IP is allowed in the zone
        return self.whitelist_manager.is_allowed(ip, zone)
    
    def _get_zone_for_path(self, path: str) -> NetworkZone:
        """Determine network zone for a path.
        
        Args:
            path: Request path
            
        Returns:
            Required network zone
        """
        # API path to zone mapping
        if path.startswith("/api/admin"):
            return NetworkZone.MANAGEMENT
        elif path.startswith("/api/internal"):
            return NetworkZone.PRIVATE
        elif path.startswith("/api/secrets"):
            return NetworkZone.RESTRICTED
        else:
            return NetworkZone.PUBLIC
    
    def _check_rate_limit(self, ip: str) -> bool:
        """Check if IP is within rate limit.
        
        Args:
            ip: Client IP address
            
        Returns:
            True if within rate limit
        """
        current_time = time.time()
        
        # Clean old requests (older than 1 second)
        if ip in self._request_counts:
            self._request_counts[ip] = [
                t for t in self._request_counts[ip]
                if current_time - t < 1.0
            ]
        
        # Check request count
        request_count = len(self._request_counts[ip])
        if request_count >= self.rate_limit:
            return False
        
        # Add current request
        self._request_counts[ip].append(current_time)
        return True
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is temporarily blocked.
        
        Args:
            ip: Client IP address
            
        Returns:
            True if IP is blocked
        """
        if ip in self._blocked_ips:
            if time.time() < self._blocked_ips[ip]:
                return True
            else:
                # Block expired, remove it
                del self._blocked_ips[ip]
        return False
    
    def _block_ip(self, ip: str, duration: int = 60):
        """Temporarily block an IP.
        
        Args:
            ip: IP address to block
            duration: Block duration in seconds
        """
        self._blocked_ips[ip] = time.time() + duration
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response.
        
        Args:
            response: FastAPI response
        """
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """Dedicated IP whitelist middleware."""
    
    def __init__(
        self,
        app,
        whitelist_manager: Optional[IPWhitelistManager] = None,
        strict_mode: bool = True
    ):
        """Initialize IP whitelist middleware.
        
        Args:
            app: FastAPI application
            whitelist_manager: IP whitelist manager
            strict_mode: Reject all non-whitelisted IPs
        """
        super().__init__(app)
        self.whitelist_manager = whitelist_manager or IPWhitelistManager()
        self.strict_mode = strict_mode
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Check IP whitelist.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
            
        Returns:
            Response if IP is allowed
        """
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Determine required zone
        zone = self._get_zone_for_path(request.url.path)
        
        # Check whitelist
        if not self.whitelist_manager.is_allowed(client_ip, zone):
            if self.strict_mode:
                logger.warning("IP rejected in strict mode",
                             ip=client_ip,
                             zone=zone.value,
                             path=request.url.path)
                raise HTTPException(
                    status_code=403,
                    detail="Access denied - IP not whitelisted"
                )
            else:
                # Log but allow in non-strict mode
                logger.warning("Non-whitelisted IP allowed (non-strict mode)",
                             ip=client_ip,
                             zone=zone.value)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_zone_for_path(self, path: str) -> NetworkZone:
        """Determine network zone for a path."""
        if "/admin" in path or "/management" in path:
            return NetworkZone.MANAGEMENT
        elif "/internal" in path:
            return NetworkZone.PRIVATE
        elif "/secrets" in path or "/vault" in path:
            return NetworkZone.RESTRICTED
        else:
            return NetworkZone.PUBLIC


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Dedicated rate limiting middleware."""
    
    def __init__(
        self,
        app,
        requests_per_second: int = 10,
        burst_size: int = 20,
        block_duration: int = 60
    ):
        """Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            requests_per_second: Sustained rate limit
            burst_size: Maximum burst size
            block_duration: How long to block violators (seconds)
        """
        super().__init__(app)
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.block_duration = block_duration
        
        # Token bucket implementation
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "tokens": burst_size,
                "last_update": time.time()
            }
        )
        self._blocked_until: Dict[str, float] = {}
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Apply rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
            
        Returns:
            Response if within rate limit
        """
        client_ip = self._get_client_ip(request)
        
        # Check if blocked
        if client_ip in self._blocked_until:
            if time.time() < self._blocked_until[client_ip]:
                remaining = int(self._blocked_until[client_ip] - time.time())
                logger.warning("Blocked IP attempted access",
                             ip=client_ip,
                             remaining_seconds=remaining)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Blocked for {remaining} seconds",
                    headers={"Retry-After": str(remaining)}
                )
            else:
                del self._blocked_until[client_ip]
        
        # Update token bucket
        if not self._consume_token(client_ip):
            # Block the IP
            self._blocked_until[client_ip] = time.time() + self.block_duration
            logger.warning("Rate limit exceeded, blocking IP",
                         ip=client_ip,
                         duration=self.block_duration)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Blocked for {self.block_duration} seconds",
                headers={"Retry-After": str(self.block_duration)}
            )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _consume_token(self, ip: str) -> bool:
        """Consume a token from the bucket.
        
        Args:
            ip: Client IP address
            
        Returns:
            True if token was available
        """
        bucket = self._buckets[ip]
        current_time = time.time()
        
        # Refill tokens based on time elapsed
        time_elapsed = current_time - bucket["last_update"]
        tokens_to_add = time_elapsed * self.requests_per_second
        
        bucket["tokens"] = min(
            self.burst_size,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = current_time
        
        # Try to consume a token
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False