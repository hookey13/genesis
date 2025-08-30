"""Role-based access control (RBAC) system for Project GENESIS.

Implements principle of least privilege with granular permissions
for different user roles and API operations.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Role(Enum):
    """User roles with increasing privileges."""

    READ_ONLY = "READ_ONLY"
    TRADER = "TRADER"
    ADMIN = "ADMIN"


class Resource(Enum):
    """Protected resources in the system."""

    # Market data resources
    MARKET_DATA = "market_data"
    ORDER_BOOK = "order_book"
    TRADE_HISTORY = "trade_history"

    # Account resources
    ACCOUNT_INFO = "account_info"
    BALANCE = "balance"
    POSITIONS = "positions"

    # Trading resources
    ORDERS = "orders"
    TRADES = "trades"
    STRATEGIES = "strategies"

    # System resources
    CONFIG = "config"
    LOGS = "logs"
    METRICS = "metrics"
    AUDIT = "audit"

    # Security resources
    API_KEYS = "api_keys"
    USERS = "users"
    PERMISSIONS = "permissions"


class Action(Enum):
    """Actions that can be performed on resources."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Permission:
    """Represents a permission to perform an action on a resource."""

    resource: Resource
    action: Action

    def __hash__(self):
        return hash((self.resource.value, self.action.value))

    def __eq__(self, other):
        if not isinstance(other, Permission):
            return False
        return self.resource == other.resource and self.action == other.action

    def __str__(self):
        return f"{self.resource.value}:{self.action.value}"


@dataclass
class User:
    """User with assigned role and permissions."""

    user_id: str
    role: Role
    custom_permissions: set[Permission] = field(default_factory=set)
    denied_permissions: set[Permission] = field(default_factory=set)
    api_key_permissions: dict[str, set[Permission]] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if user has the permission
        """
        # Check if explicitly denied
        if permission in self.denied_permissions:
            return False

        # Check custom permissions
        if permission in self.custom_permissions:
            return True

        # Check role-based permissions
        role_permissions = RolePermissionManager.get_role_permissions(self.role)
        return permission in role_permissions


class RolePermissionManager:
    """Manages role-based permissions mapping."""

    # Default permissions for each role
    _ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
        Role.READ_ONLY: {
            # Market data - read only
            Permission(Resource.MARKET_DATA, Action.READ),
            Permission(Resource.ORDER_BOOK, Action.READ),
            Permission(Resource.TRADE_HISTORY, Action.READ),

            # Account info - read only
            Permission(Resource.ACCOUNT_INFO, Action.READ),
            Permission(Resource.BALANCE, Action.READ),
            Permission(Resource.POSITIONS, Action.READ),

            # System info - read only
            Permission(Resource.METRICS, Action.READ),
        },

        Role.TRADER: {
            # Inherit all READ_ONLY permissions
            Permission(Resource.MARKET_DATA, Action.READ),
            Permission(Resource.ORDER_BOOK, Action.READ),
            Permission(Resource.TRADE_HISTORY, Action.READ),
            Permission(Resource.ACCOUNT_INFO, Action.READ),
            Permission(Resource.BALANCE, Action.READ),
            Permission(Resource.POSITIONS, Action.READ),
            Permission(Resource.METRICS, Action.READ),

            # Trading permissions
            Permission(Resource.ORDERS, Action.READ),
            Permission(Resource.ORDERS, Action.WRITE),
            Permission(Resource.ORDERS, Action.DELETE),
            Permission(Resource.TRADES, Action.READ),
            Permission(Resource.TRADES, Action.EXECUTE),
            Permission(Resource.STRATEGIES, Action.READ),
            Permission(Resource.STRATEGIES, Action.EXECUTE),

            # Limited system access
            Permission(Resource.LOGS, Action.READ),
        },

        Role.ADMIN: {
            # All permissions
            Permission(resource, action)
            for resource in Resource
            for action in Action
        }
    }

    @classmethod
    def get_role_permissions(cls, role: Role) -> set[Permission]:
        """Get permissions for a role.
        
        Args:
            role: User role
            
        Returns:
            Set of permissions for the role
        """
        return cls._ROLE_PERMISSIONS.get(role, set()).copy()

    @classmethod
    def add_role_permission(cls, role: Role, permission: Permission):
        """Add a permission to a role.
        
        Args:
            role: User role
            permission: Permission to add
        """
        if role not in cls._ROLE_PERMISSIONS:
            cls._ROLE_PERMISSIONS[role] = set()
        cls._ROLE_PERMISSIONS[role].add(permission)
        logger.info("Added permission to role",
                   role=role.value,
                   permission=str(permission))

    @classmethod
    def remove_role_permission(cls, role: Role, permission: Permission):
        """Remove a permission from a role.
        
        Args:
            role: User role
            permission: Permission to remove
        """
        if role in cls._ROLE_PERMISSIONS:
            cls._ROLE_PERMISSIONS[role].discard(permission)
            logger.info("Removed permission from role",
                       role=role.value,
                       permission=str(permission))


class PermissionChecker:
    """Checks permissions for users and API keys."""

    def __init__(self, user_repository: Any | None = None):
        """Initialize permission checker.
        
        Args:
            user_repository: Repository for user data
        """
        self.user_repository = user_repository
        self._user_cache: dict[str, User] = {}

    async def check_permission(
        self,
        user_id: str,
        resource: Resource,
        action: Action,
        api_key: str | None = None
    ) -> bool:
        """Check if a user has permission to perform an action.
        
        Args:
            user_id: User identifier
            resource: Resource to access
            action: Action to perform
            api_key: Optional API key with specific permissions
            
        Returns:
            True if permission is granted
        """
        # Get user
        user = await self._get_user(user_id)
        if not user:
            logger.warning("User not found", user_id=user_id)
            return False

        permission = Permission(resource, action)

        # Check API key specific permissions if provided
        if api_key and api_key in user.api_key_permissions:
            key_permissions = user.api_key_permissions[api_key]
            if permission not in key_permissions:
                logger.debug("Permission denied by API key restrictions",
                           user_id=user_id,
                           api_key=api_key[:8] + "...",
                           permission=str(permission))
                return False

        # Check user permissions
        has_permission = user.has_permission(permission)

        if not has_permission:
            logger.debug("Permission denied",
                       user_id=user_id,
                       role=user.role.value,
                       permission=str(permission))

        return has_permission

    async def _get_user(self, user_id: str) -> User | None:
        """Get user from cache or repository.
        
        Args:
            user_id: User identifier
            
        Returns:
            User object or None
        """
        # Check cache
        if user_id in self._user_cache:
            return self._user_cache[user_id]

        # Load from repository if available
        if self.user_repository:
            user_data = await self.user_repository.get_user(user_id)
            if user_data:
                user = User(
                    user_id=user_data["user_id"],
                    role=Role(user_data["role"]),
                    custom_permissions=set(
                        Permission(Resource(p["resource"]), Action(p["action"]))
                        for p in user_data.get("custom_permissions", [])
                    ),
                    denied_permissions=set(
                        Permission(Resource(p["resource"]), Action(p["action"]))
                        for p in user_data.get("denied_permissions", [])
                    )
                )
                self._user_cache[user_id] = user
                return user

        # Default to read-only for unknown users
        logger.warning("Unknown user, defaulting to READ_ONLY", user_id=user_id)
        user = User(user_id=user_id, role=Role.READ_ONLY)
        self._user_cache[user_id] = user
        return user

    def clear_cache(self, user_id: str | None = None):
        """Clear user cache.
        
        Args:
            user_id: Specific user to clear, or None for all
        """
        if user_id:
            self._user_cache.pop(user_id, None)
        else:
            self._user_cache.clear()


def requires_permission(resource: Resource, action: Action):
    """Decorator to check permissions before executing a function.
    
    Args:
        resource: Resource being accessed
        action: Action being performed
        
    Returns:
        Decorated function that checks permissions
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Extract user_id from self or kwargs
            user_id = getattr(self, 'user_id', None) or kwargs.get('user_id')
            if not user_id:
                logger.error("No user_id provided for permission check")
                raise PermissionError("Authentication required")

            # Get permission checker
            checker = getattr(self, 'permission_checker', None)
            if not checker:
                checker = PermissionChecker()

            # Check permission
            api_key = getattr(self, 'api_key', None) or kwargs.get('api_key')
            has_permission = await checker.check_permission(
                user_id=user_id,
                resource=resource,
                action=action,
                api_key=api_key
            )

            if not has_permission:
                raise PermissionError(
                    f"Permission denied: {resource.value}:{action.value}"
                )

            # Execute function
            return await func(self, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # For synchronous functions, run async check in event loop
            loop = asyncio.get_event_loop()

            # Extract user_id
            user_id = getattr(self, 'user_id', None) or kwargs.get('user_id')
            if not user_id:
                logger.error("No user_id provided for permission check")
                raise PermissionError("Authentication required")

            # Get permission checker
            checker = getattr(self, 'permission_checker', None)
            if not checker:
                checker = PermissionChecker()

            # Check permission
            api_key = getattr(self, 'api_key', None) or kwargs.get('api_key')
            has_permission = loop.run_until_complete(
                checker.check_permission(
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    api_key=api_key
                )
            )

            if not has_permission:
                raise PermissionError(
                    f"Permission denied: {resource.value}:{action.value}"
                )

            # Execute function
            return func(self, *args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class APIKeyPermissionManager:
    """Manages API key specific permissions."""

    @staticmethod
    def create_read_only_key_permissions() -> set[Permission]:
        """Create permissions for read-only API key.
        
        Returns:
            Set of read-only permissions
        """
        return {
            Permission(Resource.MARKET_DATA, Action.READ),
            Permission(Resource.ORDER_BOOK, Action.READ),
            Permission(Resource.TRADE_HISTORY, Action.READ),
            Permission(Resource.ACCOUNT_INFO, Action.READ),
            Permission(Resource.BALANCE, Action.READ),
            Permission(Resource.POSITIONS, Action.READ),
            Permission(Resource.ORDERS, Action.READ),
            Permission(Resource.TRADES, Action.READ),
        }

    @staticmethod
    def create_trading_key_permissions() -> set[Permission]:
        """Create permissions for trading API key.
        
        Returns:
            Set of trading permissions
        """
        read_permissions = APIKeyPermissionManager.create_read_only_key_permissions()
        trading_permissions = {
            Permission(Resource.ORDERS, Action.WRITE),
            Permission(Resource.ORDERS, Action.DELETE),
            Permission(Resource.TRADES, Action.EXECUTE),
            Permission(Resource.STRATEGIES, Action.EXECUTE),
        }
        return read_permissions.union(trading_permissions)

    @staticmethod
    def validate_api_key_permissions(
        api_key_type: str,
        requested_permission: Permission
    ) -> bool:
        """Validate if an API key type has a specific permission.
        
        Args:
            api_key_type: Type of API key (read_only, trading)
            requested_permission: Permission to check
            
        Returns:
            True if permission is valid for the key type
        """
        if api_key_type == "read_only":
            allowed = APIKeyPermissionManager.create_read_only_key_permissions()
        elif api_key_type == "trading":
            allowed = APIKeyPermissionManager.create_trading_key_permissions()
        else:
            logger.warning("Unknown API key type", key_type=api_key_type)
            return False

        return requested_permission in allowed
