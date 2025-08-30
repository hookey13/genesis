# Network Segmentation and IP Whitelisting

This document describes the network segmentation and IP-based access control for Project GENESIS.

## Overview

Project GENESIS implements network segmentation to isolate different components and enforce the principle of least privilege at the network level.

## Network Zones

### 1. Public Zone
- **Purpose**: Public API endpoints accessible from the internet
- **Services**: REST API, WebSocket feeds
- **Default Access**: Restricted to whitelisted IPs
- **Port Range**: 443 (HTTPS), 8080 (WebSocket)

### 2. Private Zone
- **Purpose**: Internal services and databases
- **Services**: PostgreSQL, Redis, internal APIs
- **Default Access**: Private network ranges only (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- **Port Range**: 5432 (PostgreSQL), 6379 (Redis)

### 3. Restricted Zone
- **Purpose**: Highly sensitive services
- **Services**: Vault, encryption keys, admin functions
- **Default Access**: Localhost only, requires SSH tunnel
- **Port Range**: 8200 (Vault)

### 4. Management Zone
- **Purpose**: Administration and monitoring
- **Services**: Grafana, Prometheus, admin API
- **Default Access**: Management subnet only (10.0.1.0/24)
- **Port Range**: 3000 (Grafana), 9090 (Prometheus)

## Firewall Configuration

### iptables Rules (Linux)

```bash
#!/bin/bash
# Network segmentation firewall rules

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Public Zone (API)
iptables -A INPUT -p tcp --dport 443 -s <WHITELIST_IP> -j ACCEPT
iptables -A INPUT -p tcp --dport 8080 -s <WHITELIST_IP> -j ACCEPT

# Private Zone (Database, Redis)
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT

# Restricted Zone (Vault) - localhost only
iptables -A INPUT -p tcp --dport 8200 -s 127.0.0.1 -j ACCEPT

# Management Zone
iptables -A INPUT -p tcp --dport 3000 -s 10.0.1.0/24 -j ACCEPT
iptables -A INPUT -p tcp --dport 9090 -s 10.0.1.0/24 -j ACCEPT

# SSH (jump box only)
iptables -A INPUT -p tcp --dport 22 -s <JUMP_BOX_IP> -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: "
```

### ufw Configuration (Ubuntu)

```bash
# Enable UFW
ufw enable

# Default policies
ufw default deny incoming
ufw default allow outgoing

# Public API
ufw allow from <WHITELIST_IP> to any port 443
ufw allow from <WHITELIST_IP> to any port 8080

# Private services from internal network
ufw allow from 10.0.0.0/8 to any port 5432
ufw allow from 10.0.0.0/8 to any port 6379

# Vault - localhost only
ufw allow from 127.0.0.1 to any port 8200

# Management network
ufw allow from 10.0.1.0/24 to any port 3000
ufw allow from 10.0.1.0/24 to any port 9090

# SSH from jump box
ufw allow from <JUMP_BOX_IP> to any port 22
```

## IP Whitelist Management

### Configuration File

Create `config/ip_whitelist.yaml`:

```yaml
public:
  - ip: "203.0.113.10"
    description: "Office network"
    enabled: true
  - ip: "198.51.100.0/24"
    description: "VPN subnet"
    enabled: true

private:
  - ip: "10.0.0.0/8"
    description: "Private network class A"
    enabled: true
  - ip: "172.16.0.0/12"
    description: "Private network class B"
    enabled: true
  - ip: "192.168.0.0/16"
    description: "Private network class C"
    enabled: true

restricted:
  - ip: "127.0.0.1"
    description: "Localhost only"
    enabled: true

management:
  - ip: "10.0.1.0/24"
    description: "Management subnet"
    enabled: true
  - ip: "203.0.113.100"
    description: "Admin workstation"
    enabled: true
```

### Python Implementation

```python
from genesis.security.ip_whitelist import (
    IPWhitelistManager,
    NetworkZone,
    IPRule
)

# Initialize whitelist manager
whitelist = IPWhitelistManager()

# Add custom rules
whitelist.add_rule(IPRule(
    ip_address="203.0.113.50",
    zone=NetworkZone.PUBLIC,
    description="Partner API server"
))

# Check access
ip = "203.0.113.50"
if whitelist.is_allowed(ip, NetworkZone.PUBLIC):
    print(f"IP {ip} is allowed in PUBLIC zone")

# List all rules for a zone
public_rules = whitelist.list_rules(NetworkZone.PUBLIC)
for rule in public_rules:
    print(f"{rule.ip_address}: {rule.description}")
```

## SSH Tunnel Configuration

For production access to restricted services:

### 1. Create SSH Tunnel

```bash
# Create tunnel to access Vault
ssh -L 8200:localhost:8200 user@production-server

# Create tunnel to access database
ssh -L 5432:localhost:5432 user@production-server

# Multiple tunnels in one command
ssh -L 8200:localhost:8200 \
    -L 5432:localhost:5432 \
    -L 6379:localhost:6379 \
    user@production-server
```

### 2. SSH Config

Create `~/.ssh/config`:

```
Host genesis-prod
    HostName production-server.example.com
    User genesis
    Port 22
    IdentityFile ~/.ssh/genesis_key
    LocalForward 8200 localhost:8200
    LocalForward 5432 localhost:5432
    LocalForward 6379 localhost:6379
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Connect with: `ssh genesis-prod`

## Docker Network Segmentation

### docker-compose.yml

```yaml
version: '3.8'

networks:
  public:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
  
  private:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.21.0.0/24
  
  restricted:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.22.0.0/24

services:
  api:
    image: genesis-api
    networks:
      - public
      - private
    environment:
      - NETWORK_ZONE=public
  
  database:
    image: postgres:16
    networks:
      - private
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
  
  vault:
    image: vault:latest
    networks:
      - restricted
    cap_add:
      - IPC_LOCK
  
  redis:
    image: redis:7
    networks:
      - private
```

## API Middleware Configuration

### FastAPI Application

```python
from fastapi import FastAPI
from genesis.api.security_middleware import (
    SecurityMiddleware,
    IPWhitelistMiddleware,
    RateLimitMiddleware
)
from genesis.security.ip_whitelist import IPWhitelistManager

app = FastAPI()

# Load IP whitelist
whitelist_manager = IPWhitelistManager()

# Add security middleware
app.add_middleware(
    SecurityMiddleware,
    whitelist_manager=whitelist_manager,
    rate_limit=10,  # 10 requests per second
    enable_ip_whitelist=True,
    enable_rate_limit=True,
    enable_security_headers=True
)

# Or use individual middleware
app.add_middleware(
    IPWhitelistMiddleware,
    whitelist_manager=whitelist_manager,
    strict_mode=True
)

app.add_middleware(
    RateLimitMiddleware,
    requests_per_second=10,
    burst_size=20,
    block_duration=60
)
```

## Security Headers

The middleware automatically adds these security headers:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

## Rate Limiting

### Token Bucket Algorithm

- **Sustained Rate**: 10 requests/second (configurable)
- **Burst Size**: 20 requests (configurable)
- **Block Duration**: 60 seconds for violations

### Response Headers

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
Content-Type: application/json

{
  "detail": "Rate limit exceeded. Blocked for 60 seconds"
}
```

## Monitoring and Alerts

### Access Logs

```python
# Log format
{
  "timestamp": "2024-01-15T10:30:00Z",
  "ip": "203.0.113.10",
  "method": "POST",
  "path": "/api/orders",
  "status": 200,
  "duration": 0.123,
  "zone": "public"
}
```

### Alert Rules

1. **Unauthorized Access Attempts**
   - Trigger: 5+ denied requests from same IP in 1 minute
   - Action: Email alert, consider permanent IP ban

2. **Rate Limit Violations**
   - Trigger: IP blocked 3+ times in 1 hour
   - Action: Investigate for potential attack

3. **Zone Violations**
   - Trigger: Attempt to access restricted zone from public IP
   - Action: Immediate alert, security review

## Testing Network Segmentation

### Test Script

```python
#!/usr/bin/env python3
"""Test network segmentation rules."""

import requests
from genesis.security.ip_whitelist import (
    IPWhitelistManager,
    NetworkSegmentation,
    NetworkZone
)

def test_segmentation():
    # Initialize
    whitelist = IPWhitelistManager()
    segmentation = NetworkSegmentation(whitelist)
    
    # Test cases
    test_cases = [
        ("127.0.0.1", "vault", True),  # Localhost can access vault
        ("10.0.0.5", "database", True),  # Private IP can access database
        ("203.0.113.10", "api", True),  # Whitelisted public IP can access API
        ("1.2.3.4", "vault", False),  # Random IP cannot access vault
    ]
    
    for ip, service, expected in test_cases:
        result = segmentation.can_access_service(ip, service)
        status = "✓" if result == expected else "✗"
        print(f"{status} IP {ip} -> {service}: {result}")

if __name__ == "__main__":
    test_segmentation()
```

## Best Practices

1. **Least Privilege**: Grant minimum network access required
2. **Defense in Depth**: Multiple layers of network security
3. **Regular Audits**: Review IP whitelist quarterly
4. **Fail Secure**: Default deny for unknown IPs
5. **Logging**: Log all access attempts for forensics
6. **Automation**: Use configuration management for firewall rules
7. **Testing**: Regular penetration testing of network boundaries
8. **Documentation**: Keep network diagrams up to date

## Troubleshooting

### Common Issues

1. **403 Forbidden - IP not authorized**
   - Check if IP is whitelisted
   - Verify correct network zone
   - Check firewall rules

2. **429 Too Many Requests**
   - Rate limit exceeded
   - Wait for retry-after period
   - Consider increasing limits if legitimate

3. **Connection Refused**
   - Service may be in restricted zone
   - SSH tunnel may be required
   - Check firewall rules

### Debug Commands

```bash
# Check current iptables rules
iptables -L -n -v

# Check UFW status
ufw status verbose

# Test connectivity
nc -zv production-server 8200

# Check listening ports
netstat -tlnp

# Trace network path
traceroute production-server
```