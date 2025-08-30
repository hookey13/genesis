"""Unit tests for permissions and RBAC system."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from genesis.security.permissions import (
    Role,
    Resource,
    Action,
    Permission,
    User,
    RolePermissionManager,
    PermissionChecker,
    APIKeyPermissionManager,
    requires_permission
)


class TestPermission:
    """Test Permission class."""
    
    def test_permission_creation(self):
        """Test creating a permission."""
        perm = Permission(Resource.ORDERS, Action.WRITE)
        assert perm.resource == Resource.ORDERS
        assert perm.action == Action.WRITE
        assert str(perm) == "orders:write"
    
    def test_permission_equality(self):
        """Test permission equality comparison."""
        perm1 = Permission(Resource.ORDERS, Action.WRITE)
        perm2 = Permission(Resource.ORDERS, Action.WRITE)
        perm3 = Permission(Resource.ORDERS, Action.READ)
        
        assert perm1 == perm2
        assert perm1 != perm3
    
    def test_permission_hashing(self):
        """Test permission can be used in sets."""
        perm1 = Permission(Resource.ORDERS, Action.WRITE)
        perm2 = Permission(Resource.ORDERS, Action.WRITE)
        perm3 = Permission(Resource.TRADES, Action.EXECUTE)
        
        perm_set = {perm1, perm2, perm3}
        assert len(perm_set) == 2  # perm1 and perm2 are the same


class TestUser:
    """Test User class."""
    
    def test_user_creation(self):
        """Test creating a user."""
        user = User(user_id="test_user", role=Role.TRADER)
        assert user.user_id == "test_user"
        assert user.role == Role.TRADER
        assert len(user.custom_permissions) == 0
        assert len(user.denied_permissions) == 0
    
    def test_user_has_permission_from_role(self):
        """Test user has permissions from their role."""
        user = User(user_id="trader", role=Role.TRADER)
        
        # Should have trading permissions
        perm_read_orders = Permission(Resource.ORDERS, Action.READ)
        perm_write_orders = Permission(Resource.ORDERS, Action.WRITE)
        assert user.has_permission(perm_read_orders)
        assert user.has_permission(perm_write_orders)
        
        # Should not have admin permissions
        perm_admin = Permission(Resource.API_KEYS, Action.ADMIN)
        assert not user.has_permission(perm_admin)
    
    def test_user_custom_permissions(self):
        """Test user with custom permissions."""
        user = User(
            user_id="custom_user",
            role=Role.READ_ONLY,
            custom_permissions={
                Permission(Resource.ORDERS, Action.WRITE)
            }
        )
        
        # Should have custom permission despite READ_ONLY role
        perm_write = Permission(Resource.ORDERS, Action.WRITE)
        assert user.has_permission(perm_write)
    
    def test_user_denied_permissions(self):
        """Test user with denied permissions."""
        user = User(
            user_id="restricted_trader",
            role=Role.TRADER,
            denied_permissions={
                Permission(Resource.ORDERS, Action.DELETE)
            }
        )
        
        # Should not have denied permission despite TRADER role
        perm_delete = Permission(Resource.ORDERS, Action.DELETE)
        assert not user.has_permission(perm_delete)
        
        # Should still have other trader permissions
        perm_write = Permission(Resource.ORDERS, Action.WRITE)
        assert user.has_permission(perm_write)


class TestRolePermissionManager:
    """Test RolePermissionManager class."""
    
    def test_read_only_permissions(self):
        """Test READ_ONLY role has correct permissions."""
        perms = RolePermissionManager.get_role_permissions(Role.READ_ONLY)
        
        # Should have read permissions
        assert Permission(Resource.MARKET_DATA, Action.READ) in perms
        assert Permission(Resource.BALANCE, Action.READ) in perms
        
        # Should not have write permissions
        assert Permission(Resource.ORDERS, Action.WRITE) not in perms
        assert Permission(Resource.TRADES, Action.EXECUTE) not in perms
    
    def test_trader_permissions(self):
        """Test TRADER role has correct permissions."""
        perms = RolePermissionManager.get_role_permissions(Role.TRADER)
        
        # Should have read permissions
        assert Permission(Resource.MARKET_DATA, Action.READ) in perms
        
        # Should have trading permissions
        assert Permission(Resource.ORDERS, Action.WRITE) in perms
        assert Permission(Resource.ORDERS, Action.DELETE) in perms
        assert Permission(Resource.TRADES, Action.EXECUTE) in perms
        
        # Should not have admin permissions
        assert Permission(Resource.API_KEYS, Action.ADMIN) not in perms
    
    def test_admin_permissions(self):
        """Test ADMIN role has all permissions."""
        perms = RolePermissionManager.get_role_permissions(Role.ADMIN)
        
        # Should have all permissions
        for resource in Resource:
            for action in Action:
                assert Permission(resource, action) in perms
    
    def test_add_role_permission(self):
        """Test adding permission to a role."""
        # Add a new permission to READ_ONLY
        new_perm = Permission(Resource.STRATEGIES, Action.READ)
        RolePermissionManager.add_role_permission(Role.READ_ONLY, new_perm)
        
        perms = RolePermissionManager.get_role_permissions(Role.READ_ONLY)
        assert new_perm in perms
    
    def test_remove_role_permission(self):
        """Test removing permission from a role."""
        # Remove a permission from TRADER
        perm_to_remove = Permission(Resource.ORDERS, Action.DELETE)
        RolePermissionManager.remove_role_permission(Role.TRADER, perm_to_remove)
        
        perms = RolePermissionManager.get_role_permissions(Role.TRADER)
        assert perm_to_remove not in perms


@pytest.mark.asyncio
class TestPermissionChecker:
    """Test PermissionChecker class."""
    
    @pytest.fixture
    def mock_user_repository(self):
        """Create a mock user repository."""
        repo = AsyncMock()
        repo.get_user = AsyncMock()
        return repo
    
    @pytest.fixture
    def permission_checker(self, mock_user_repository):
        """Create a permission checker."""
        return PermissionChecker(user_repository=mock_user_repository)
    
    async def test_check_permission_success(self, permission_checker, mock_user_repository):
        """Test successful permission check."""
        # Mock user data
        mock_user_repository.get_user.return_value = {
            "user_id": "test_user",
            "role": "TRADER",
            "custom_permissions": [],
            "denied_permissions": []
        }
        
        # Check permission
        has_permission = await permission_checker.check_permission(
            user_id="test_user",
            resource=Resource.ORDERS,
            action=Action.WRITE
        )
        
        assert has_permission is True
    
    async def test_check_permission_denied(self, permission_checker, mock_user_repository):
        """Test denied permission check."""
        # Mock user data
        mock_user_repository.get_user.return_value = {
            "user_id": "read_user",
            "role": "READ_ONLY",
            "custom_permissions": [],
            "denied_permissions": []
        }
        
        # Check permission
        has_permission = await permission_checker.check_permission(
            user_id="read_user",
            resource=Resource.ORDERS,
            action=Action.WRITE
        )
        
        assert has_permission is False
    
    async def test_check_permission_with_api_key(self, permission_checker, mock_user_repository):
        """Test permission check with API key restrictions."""
        # Mock user with API key permissions
        mock_user_repository.get_user.return_value = {
            "user_id": "api_user",
            "role": "TRADER",
            "custom_permissions": [],
            "denied_permissions": []
        }
        
        # Create user with API key permissions
        user = await permission_checker._get_user("api_user")
        user.api_key_permissions["test_key"] = {
            Permission(Resource.MARKET_DATA, Action.READ),
            Permission(Resource.ORDERS, Action.READ)
        }
        
        # Should fail for permission not in API key set
        has_permission = await permission_checker.check_permission(
            user_id="api_user",
            resource=Resource.ORDERS,
            action=Action.WRITE,
            api_key="test_key"
        )
        
        assert has_permission is False
    
    async def test_cache_user(self, permission_checker, mock_user_repository):
        """Test user caching."""
        # Mock user data
        mock_user_repository.get_user.return_value = {
            "user_id": "cached_user",
            "role": "TRADER",
            "custom_permissions": [],
            "denied_permissions": []
        }
        
        # First call should fetch from repository
        await permission_checker._get_user("cached_user")
        assert mock_user_repository.get_user.call_count == 1
        
        # Second call should use cache
        await permission_checker._get_user("cached_user")
        assert mock_user_repository.get_user.call_count == 1
        
        # Clear cache and fetch again
        permission_checker.clear_cache("cached_user")
        await permission_checker._get_user("cached_user")
        assert mock_user_repository.get_user.call_count == 2


class TestAPIKeyPermissionManager:
    """Test APIKeyPermissionManager class."""
    
    def test_create_read_only_key_permissions(self):
        """Test creating read-only API key permissions."""
        perms = APIKeyPermissionManager.create_read_only_key_permissions()
        
        # Should have read permissions
        assert Permission(Resource.MARKET_DATA, Action.READ) in perms
        assert Permission(Resource.ORDERS, Action.READ) in perms
        
        # Should not have write permissions
        assert Permission(Resource.ORDERS, Action.WRITE) not in perms
        assert Permission(Resource.TRADES, Action.EXECUTE) not in perms
    
    def test_create_trading_key_permissions(self):
        """Test creating trading API key permissions."""
        perms = APIKeyPermissionManager.create_trading_key_permissions()
        
        # Should have read permissions
        assert Permission(Resource.MARKET_DATA, Action.READ) in perms
        
        # Should have trading permissions
        assert Permission(Resource.ORDERS, Action.WRITE) in perms
        assert Permission(Resource.ORDERS, Action.DELETE) in perms
        assert Permission(Resource.TRADES, Action.EXECUTE) in perms
    
    def test_validate_api_key_permissions(self):
        """Test validating API key permissions."""
        # Read-only key should not have write permission
        perm_write = Permission(Resource.ORDERS, Action.WRITE)
        assert not APIKeyPermissionManager.validate_api_key_permissions(
            "read_only", perm_write
        )
        
        # Trading key should have write permission
        assert APIKeyPermissionManager.validate_api_key_permissions(
            "trading", perm_write
        )
        
        # Unknown key type should return False
        assert not APIKeyPermissionManager.validate_api_key_permissions(
            "unknown", perm_write
        )


@pytest.mark.asyncio
class TestRequiresPermissionDecorator:
    """Test requires_permission decorator."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock service with permission decorator."""
        class MockService:
            def __init__(self):
                self.user_id = "test_user"
                self.permission_checker = PermissionChecker()
            
            @requires_permission(Resource.ORDERS, Action.WRITE)
            async def create_order(self):
                return "Order created"
            
            @requires_permission(Resource.MARKET_DATA, Action.READ)
            def get_market_data(self):
                return "Market data"
        
        return MockService()
    
    async def test_decorator_allows_with_permission(self, mock_service):
        """Test decorator allows action with permission."""
        # Mock permission check to return True
        with patch.object(mock_service.permission_checker, 'check_permission') as mock_check:
            mock_check.return_value = True
            
            result = await mock_service.create_order()
            assert result == "Order created"
            
            mock_check.assert_called_once_with(
                user_id="test_user",
                resource=Resource.ORDERS,
                action=Action.WRITE,
                api_key=None
            )
    
    async def test_decorator_denies_without_permission(self, mock_service):
        """Test decorator denies action without permission."""
        # Mock permission check to return False
        with patch.object(mock_service.permission_checker, 'check_permission') as mock_check:
            mock_check.return_value = False
            
            with pytest.raises(PermissionError) as exc_info:
                await mock_service.create_order()
            
            assert "Permission denied: orders:write" in str(exc_info.value)