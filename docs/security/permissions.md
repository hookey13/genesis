# Permissions and Access Control

This document describes the role-based access control (RBAC) system for Project GENESIS.

## Overview

Project GENESIS implements a principle of least privilege security model with:
- Role-based permissions (READ_ONLY, TRADER, ADMIN)
- Resource-based access control
- API key-specific restrictions
- Granular permission management

## Roles

### READ_ONLY
- View market data and account information
- No trading capabilities
- Suitable for monitoring and analytics

### TRADER
- All READ_ONLY permissions
- Execute trades and manage orders
- Access to strategies and execution
- View logs and metrics

### ADMIN
- All permissions
- Manage API keys and users
- Configure system settings
- Access audit logs

## Resources and Actions

### Resources
| Resource | Description | Actions |
|----------|-------------|---------|
| `market_data` | Market prices and data | read |
| `order_book` | Order book depth | read |
| `trade_history` | Historical trades | read |
| `account_info` | Account details | read |
| `balance` | Account balance | read |
| `positions` | Open positions | read |
| `orders` | Trading orders | read, write, delete |
| `trades` | Trade execution | read, execute |
| `strategies` | Trading strategies | read, execute |
| `config` | System configuration | read, write, admin |
| `logs` | System logs | read |
| `metrics` | Performance metrics | read |
| `audit` | Audit logs | read, write |
| `api_keys` | API key management | read, write, admin |
| `users` | User management | read, write, admin |
| `permissions` | Permission management | read, write, admin |

### Actions
- `read`: View resource data
- `write`: Create or modify resource
- `delete`: Remove resource
- `execute`: Run operations (trades, strategies)
- `admin`: Administrative operations

## Permission Matrix

### READ_ONLY Role
```
market_data:read
order_book:read
trade_history:read
account_info:read
balance:read
positions:read
metrics:read
```

### TRADER Role
```
[All READ_ONLY permissions plus:]
orders:read
orders:write
orders:delete
trades:read
trades:execute
strategies:read
strategies:execute
logs:read
```

### ADMIN Role
```
[All permissions on all resources]
```

## API Key Permissions

API keys can have restricted permissions independent of user roles:

### Read-Only API Key
- Limited to read operations only
- Cannot execute trades or modify orders
- Suitable for data collection and monitoring

### Trading API Key
- Full trading capabilities
- Can place, modify, and cancel orders
- Access to strategy execution

## Implementation

### Using Permissions in Code

#### 1. Permission Check in Service Methods
```python
from genesis.security.permissions import requires_permission, Resource, Action

class TradingService:
    @requires_permission(Resource.ORDERS, Action.WRITE)
    async def create_order(self, order_params):
        # Method will only execute if user has permission
        return await self.exchange.place_order(order_params)
```

#### 2. Manual Permission Check
```python
from genesis.security.permissions import PermissionChecker

checker = PermissionChecker()
has_permission = await checker.check_permission(
    user_id="user123",
    resource=Resource.ORDERS,
    action=Action.WRITE
)

if not has_permission:
    raise PermissionError("Cannot place orders")
```

#### 3. API Endpoint Protection
```python
from fastapi import FastAPI, Depends
from genesis.api.auth import require_permission, Resource, Action

app = FastAPI()

@app.post("/orders", dependencies=[require_permission(Resource.ORDERS, Action.WRITE)])
async def create_order(order: OrderRequest):
    # Endpoint protected by permission check
    return await place_order(order)
```

## Custom Permissions

### Adding Custom Permissions to User
```python
user = User(
    user_id="special_user",
    role=Role.READ_ONLY,
    custom_permissions={
        Permission(Resource.ORDERS, Action.WRITE),
        Permission(Resource.STRATEGIES, Action.EXECUTE)
    }
)
```

### Denying Specific Permissions
```python
user = User(
    user_id="restricted_trader",
    role=Role.TRADER,
    denied_permissions={
        Permission(Resource.ORDERS, Action.DELETE)
    }
)
```

## API Authentication

### Request Headers
All API requests must include:
- `X-API-Key`: Your API key
- `X-Timestamp`: Unix timestamp
- `X-Signature`: HMAC SHA256 signature

### Signature Generation
```python
import hmac
import hashlib
import time

timestamp = str(int(time.time()))
method = "POST"
path = "/api/orders"
body = '{"symbol": "BTC/USDT", "size": 0.001}'

message = f"{timestamp}{method}{path}{body}"
signature = hmac.new(
    api_secret.encode(),
    message.encode(),
    hashlib.sha256
).hexdigest()
```

## Database Schema

### User Roles Table
```sql
CREATE TABLE user_roles (
    id INTEGER PRIMARY KEY,
    user_id TEXT UNIQUE NOT NULL,
    role TEXT CHECK(role IN ('READ_ONLY', 'TRADER', 'ADMIN')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Permissions Table
```sql
CREATE TABLE permissions (
    id INTEGER PRIMARY KEY,
    role TEXT NOT NULL,
    resource TEXT NOT NULL,
    action TEXT NOT NULL,
    UNIQUE(role, resource, action)
);
```

### Custom User Permissions
```sql
CREATE TABLE user_permissions (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    action TEXT NOT NULL,
    granted BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user_roles(user_id),
    UNIQUE(user_id, resource, action)
);
```

## Security Best Practices

1. **Principle of Least Privilege**: Users should have minimum permissions needed
2. **Regular Audits**: Review permissions quarterly
3. **API Key Rotation**: Rotate keys every 30 days
4. **Separate Keys**: Use different keys for read vs trade operations
5. **IP Whitelisting**: Restrict API keys to specific IPs
6. **Rate Limiting**: Enforce per-endpoint rate limits
7. **Audit Logging**: Log all permission checks and access attempts

## Permission Escalation

To request additional permissions:
1. Submit request with business justification
2. Review by security team
3. Approval from admin
4. Audit trail maintained

## Troubleshooting

### Permission Denied Errors
1. Check user role: `SELECT role FROM user_roles WHERE user_id = ?`
2. Check custom permissions: `SELECT * FROM user_permissions WHERE user_id = ?`
3. Verify API key type matches operation
4. Check audit logs for denial reason

### Common Issues
- **403 Forbidden**: User lacks required permission
- **401 Unauthorized**: Invalid API key or signature
- **429 Too Many Requests**: Rate limit exceeded

## Examples

### Creating a Read-Only Monitoring User
```python
monitor_user = User(
    user_id="monitor",
    role=Role.READ_ONLY
)
```

### Creating a Trader with Limited Permissions
```python
limited_trader = User(
    user_id="limited_trader",
    role=Role.TRADER,
    denied_permissions={
        Permission(Resource.STRATEGIES, Action.EXECUTE),
        Permission(Resource.ORDERS, Action.DELETE)
    }
)
```

### API Key with Specific Permissions
```python
user.api_key_permissions["read_key_123"] = {
    Permission(Resource.MARKET_DATA, Action.READ),
    Permission(Resource.ACCOUNT_INFO, Action.READ)
}