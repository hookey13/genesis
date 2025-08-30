# TLS Configuration for Internal Communication

This document describes the TLS implementation for secure internal communication in Project GENESIS.

## Overview

Project GENESIS uses TLS 1.3 for all internal service communication with:
- **Mutual TLS (mTLS)** for service authentication
- **Certificate rotation** every 90 days
- **Strong cipher suites** (AES-256-GCM, ChaCha20-Poly1305)
- **Certificate storage** in HashiCorp Vault

## Certificate Hierarchy

```
Root CA (5 years)
├── Server Certificate (90 days)
│   └── API Server
│   └── Database Server
│   └── Redis Server
└── Client Certificates (90 days)
    └── API Client
    └── Database Client
    └── Monitoring Client
```

## Certificate Generation

### Generate CA Certificate

```python
from genesis.security.tls_manager import TLSManager

tls_manager = TLSManager()

# Generate CA (first time only)
ca_cert, ca_key = tls_manager.generate_ca_certificate()
```

### Generate Server Certificate

```python
# Generate server certificate for API
server_cert, server_key = tls_manager.generate_server_certificate(
    hostname="api.genesis.internal",
    ip_addresses=["10.0.1.10", "127.0.0.1"]
)
```

### Generate Client Certificate

```python
# Generate client certificate for service
client_cert, client_key = tls_manager.generate_client_certificate(
    client_name="database-client"
)
```

## TLS Configuration

### Server Configuration

```python
import ssl
from genesis.security.tls_manager import TLSManager

# Create TLS manager
tls_manager = TLSManager()

# Create server SSL context
ssl_context = tls_manager.create_ssl_context(
    purpose=ssl.Purpose.CLIENT_AUTH,
    verify_mode=ssl.CERT_REQUIRED,  # Require client certificates
    check_hostname=False  # Internal network
)

# Use with FastAPI
from fastapi import FastAPI
import uvicorn

app = FastAPI()

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8443,
    ssl_keyfile=str(tls_manager.server_key_path),
    ssl_certfile=str(tls_manager.server_cert_path),
    ssl_ca_certs=str(tls_manager.ca_cert_path),
    ssl_cert_reqs=ssl.CERT_REQUIRED
)
```

### Client Configuration

```python
import httpx
from genesis.security.tls_manager import TLSManager

# Create TLS manager
tls_manager = TLSManager()

# Create client SSL context
ssl_context = tls_manager.create_ssl_context(
    purpose=ssl.Purpose.SERVER_AUTH,
    verify_mode=ssl.CERT_REQUIRED,
    check_hostname=True
)

# Use with httpx
async with httpx.AsyncClient(
    verify=ssl_context,
    cert=(
        str(tls_manager.client_cert_path),
        str(tls_manager.client_key_path)
    )
) as client:
    response = await client.get("https://api.genesis.internal:8443/health")
```

## Service-Specific Configuration

### PostgreSQL with TLS

```ini
# postgresql.conf
ssl = on
ssl_cert_file = '/genesis/certs/server.crt'
ssl_key_file = '/genesis/certs/server.key'
ssl_ca_file = '/genesis/certs/ca.crt'
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
ssl_prefer_server_ciphers = on
ssl_min_protocol_version = 'TLSv1.3'
```

Connection string:
```python
DATABASE_URL = "postgresql://user:pass@localhost:5432/genesis?sslmode=require&sslcert=client.crt&sslkey=client.key&sslrootcert=ca.crt"
```

### Redis with TLS

```conf
# redis.conf
tls-port 6380
port 0

tls-cert-file /genesis/certs/server.crt
tls-key-file /genesis/certs/server.key
tls-ca-cert-file /genesis/certs/ca.crt

tls-protocols "TLSv1.3"
tls-ciphers "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
tls-ciphersuites "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"

tls-replication yes
tls-cluster yes
```

Python connection:
```python
import redis

r = redis.Redis(
    host='localhost',
    port=6380,
    ssl=True,
    ssl_certfile='client.crt',
    ssl_keyfile='client.key',
    ssl_ca_certs='ca.crt',
    ssl_check_hostname=False
)
```

## Certificate Management

### Automatic Rotation

```python
import asyncio
from genesis.security.tls_manager import TLSManager

async def certificate_rotation_task():
    """Background task for certificate rotation."""
    tls_manager = TLSManager()
    
    while True:
        # Check and rotate certificates
        rotated = tls_manager.rotate_certificates()
        
        if rotated:
            logger.info("Certificates rotated, reloading services")
            # Trigger service reload
            await reload_services()
        
        # Check daily
        await asyncio.sleep(86400)
```

### Manual Rotation

```bash
# Rotate certificates manually
python scripts/rotate_certificates.py

# Verify certificates
python scripts/verify_certificates.py
```

### Certificate Verification

```python
# Verify certificate validity
info = tls_manager.verify_certificate(tls_manager.server_cert_path)

print(f"Valid: {info['valid']}")
print(f"Expires: {info['not_valid_after']}")
print(f"Days until expiry: {info['days_until_expiry']}")
print(f"Needs renewal: {info['needs_renewal']}")
```

## OpenSSL Commands

### Generate CA Certificate

```bash
# Generate CA private key
openssl genrsa -out ca.key 4096

# Generate CA certificate
openssl req -new -x509 -days 1825 -key ca.key -out ca.crt \
  -subj "/C=US/ST=CA/L=San Francisco/O=Project GENESIS/CN=GENESIS Internal CA"
```

### Generate Server Certificate

```bash
# Generate server private key
openssl genrsa -out server.key 2048

# Generate certificate signing request
openssl req -new -key server.key -out server.csr \
  -subj "/C=US/ST=CA/L=San Francisco/O=Project GENESIS/CN=api.genesis.internal"

# Sign with CA
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 90 \
  -extensions v3_req -extfile server.conf
```

### server.conf Extension File

```ini
[v3_req]
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = api.genesis.internal
DNS.2 = localhost
IP.1 = 10.0.1.10
IP.2 = 127.0.0.1
```

### Verify Certificate

```bash
# View certificate details
openssl x509 -in server.crt -text -noout

# Verify certificate chain
openssl verify -CAfile ca.crt server.crt

# Check certificate expiration
openssl x509 -in server.crt -noout -enddate

# Test TLS connection
openssl s_client -connect localhost:8443 \
  -cert client.crt -key client.key -CAfile ca.crt \
  -tls1_3
```

## Docker Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    image: genesis-api
    volumes:
      - ./certs:/genesis/certs:ro
    environment:
      - TLS_ENABLED=true
      - TLS_CERT=/genesis/certs/server.crt
      - TLS_KEY=/genesis/certs/server.key
      - TLS_CA=/genesis/certs/ca.crt
    ports:
      - "8443:8443"
  
  database:
    image: postgres:16
    volumes:
      - ./certs:/genesis/certs:ro
    environment:
      - POSTGRES_SSL=on
      - POSTGRES_SSL_CERT=/genesis/certs/server.crt
      - POSTGRES_SSL_KEY=/genesis/certs/server.key
      - POSTGRES_SSL_CA=/genesis/certs/ca.crt
    command: >
      postgres
      -c ssl=on
      -c ssl_cert_file=/genesis/certs/server.crt
      -c ssl_key_file=/genesis/certs/server.key
      -c ssl_ca_file=/genesis/certs/ca.crt
      -c ssl_min_protocol_version=TLSv1.3
```

## Security Headers

### nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name api.genesis.internal;
    
    # TLS Configuration
    ssl_certificate /genesis/certs/server.crt;
    ssl_certificate_key /genesis/certs/server.key;
    ssl_client_certificate /genesis/certs/ca.crt;
    ssl_verify_client on;
    
    # TLS 1.3 only
    ssl_protocols TLSv1.3;
    ssl_ciphers 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256';
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
}
```

## Monitoring

### Certificate Expiry Monitoring

```python
import prometheus_client

# Prometheus metric
cert_expiry_days = prometheus_client.Gauge(
    'tls_cert_expiry_days',
    'Days until certificate expiry',
    ['cert_type']
)

# Update metrics
for cert_type in ['server', 'client']:
    info = tls_manager.verify_certificate(cert_path)
    cert_expiry_days.labels(cert_type=cert_type).set(
        info['days_until_expiry']
    )
```

### Alerts

```yaml
# Prometheus alert rules
groups:
  - name: tls_alerts
    rules:
      - alert: CertificateExpiringSoon
        expr: tls_cert_expiry_days < 30
        for: 1h
        annotations:
          summary: "Certificate expiring in {{ $value }} days"
          
      - alert: CertificateExpired
        expr: tls_cert_expiry_days <= 0
        for: 1m
        annotations:
          summary: "Certificate has expired!"
```

## Troubleshooting

### Common Issues

#### 1. Certificate Verification Failed

```
ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution**: Ensure CA certificate is trusted and paths are correct

```python
# Debug certificate chain
import ssl
import socket

context = ssl.create_default_context()
with socket.create_connection(('localhost', 8443)) as sock:
    with context.wrap_socket(sock, server_hostname='localhost') as ssock:
        print(ssock.getpeercert())
```

#### 2. Cipher Suite Mismatch

```
ssl.SSLError: [SSL: NO_SHARED_CIPHER]
```

**Solution**: Ensure both client and server support TLS 1.3

```bash
# Check supported ciphers
openssl ciphers -v 'TLSv1.3'
```

#### 3. Certificate Not Yet Valid

```
ssl.SSLError: certificate is not yet valid
```

**Solution**: Check system time synchronization

```bash
# Sync time
timedatectl set-ntp true
```

## Best Practices

1. **Use TLS 1.3 exclusively** - Disable older versions
2. **Rotate certificates regularly** - Every 90 days maximum
3. **Use strong key sizes** - 4096 bits for CA, 2048 for others
4. **Implement mutual TLS** - Verify both client and server
5. **Store keys securely** - Use Vault or HSM
6. **Monitor expiration** - Alert 30 days before expiry
7. **Automate rotation** - Minimize manual intervention
8. **Test failover** - Ensure services handle rotation
9. **Log all TLS events** - For security audit
10. **Use certificate pinning** - For critical services