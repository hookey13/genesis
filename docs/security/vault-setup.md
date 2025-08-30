# HashiCorp Vault Setup Guide

This document describes how to set up and configure HashiCorp Vault for Project GENESIS secret management.

## Overview

Project GENESIS uses HashiCorp Vault (or AWS Secrets Manager) for secure storage and management of:
- Exchange API keys (trading and read-only)
- Database encryption keys
- TLS certificates
- Backup encryption keys

## Installation

### Option 1: HashiCorp Vault

1. **Install Vault**:
```bash
# Download and install Vault
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/
```

2. **Start Vault Server** (Development):
```bash
# Development mode (NOT for production)
vault server -dev
```

3. **Production Setup**:
```bash
# Create config file at /etc/vault/config.hcl
storage "file" {
  path = "/opt/vault/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_cert_file = "/opt/vault/tls/cert.pem"
  tls_key_file  = "/opt/vault/tls/key.pem"
}

api_addr = "https://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
ui = true
```

### Option 2: AWS Secrets Manager

If using AWS Secrets Manager instead of Vault, install AWS SDK:
```bash
pip install boto3
```

## Secret Structure

### 1. Exchange API Keys
Path: `/genesis/exchange/api-keys`

```json
{
  "api_key": "main_trading_api_key",
  "api_secret": "main_trading_api_secret",
  "api_key_read": "read_only_api_key",
  "api_secret_read": "read_only_api_secret"
}
```

### 2. Database Encryption Key
Path: `/genesis/database/encryption-key`

```json
{
  "key": "base64_encoded_encryption_key"
}
```

### 3. TLS Certificates
Path: `/genesis/tls/certificates`

```json
{
  "cert": "/path/to/server.crt",
  "key": "/path/to/server.key",
  "ca": "/path/to/ca.crt"
}
```

### 4. Backup Encryption Key
Path: `/genesis/backup/encryption-key`

```json
{
  "key": "backup_encryption_password"
}
```

## Initial Setup

### 1. Initialize Vault

```bash
# Initialize Vault (production)
vault operator init -key-shares=5 -key-threshold=3

# Save the unseal keys and root token securely!
```

### 2. Unseal Vault

```bash
# Unseal with 3 of the 5 keys
vault operator unseal <unseal-key-1>
vault operator unseal <unseal-key-2>
vault operator unseal <unseal-key-3>
```

### 3. Login to Vault

```bash
# Login with root token
vault login <root-token>
```

### 4. Enable KV Secrets Engine

```bash
# Enable KV v2 secrets engine
vault secrets enable -version=2 -path=secret kv
```

### 5. Create Secrets

```bash
# Store exchange API keys
vault kv put secret/genesis/exchange/api-keys \
  api_key="your_api_key" \
  api_secret="your_api_secret" \
  api_key_read="your_read_api_key" \
  api_secret_read="your_read_api_secret"

# Store database encryption key
vault kv put secret/genesis/database/encryption-key \
  key="$(openssl rand -base64 32)"

# Store backup encryption key
vault kv put secret/genesis/backup/encryption-key \
  key="$(openssl rand -base64 32)"
```

## Application Configuration

### Environment Variables

For production, set these environment variables:

```bash
# Vault configuration
export VAULT_URL="https://vault.example.com:8200"
export VAULT_TOKEN="your_app_token"
export GENESIS_USE_VAULT="true"
```

For development (without Vault):

```bash
# Direct credentials (development only)
export GENESIS_USE_VAULT="false"
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export DATABASE_ENCRYPTION_KEY="your_encryption_key"
```

## Access Control

### 1. Create Application Policy

```hcl
# File: genesis-app-policy.hcl
path "secret/data/genesis/*" {
  capabilities = ["read", "list"]
}

path "secret/metadata/genesis/*" {
  capabilities = ["read", "list"]
}
```

Apply the policy:
```bash
vault policy write genesis-app genesis-app-policy.hcl
```

### 2. Create Application Token

```bash
# Create token with specific policy
vault token create -policy=genesis-app -ttl=720h
```

Use this token in the application (not the root token).

## Key Rotation

### Manual Rotation

```bash
# Rotate API keys
vault kv put secret/genesis/exchange/api-keys \
  api_key="new_api_key" \
  api_secret="new_api_secret" \
  api_key_read="new_read_api_key" \
  api_secret_read="new_read_api_secret"
```

### Automated Rotation

Use the `scripts/rotate_keys.py` script:
```bash
python scripts/rotate_keys.py --path /genesis/exchange/api-keys
```

## Monitoring

### Health Check

```bash
# Check Vault health
vault status
```

### Audit Logging

Enable audit logging:
```bash
vault audit enable file file_path=/var/log/vault-audit.log
```

## Backup and Recovery

### Backup Vault Data

```bash
# Backup Vault data directory
tar -czf vault-backup-$(date +%Y%m%d).tar.gz /opt/vault/data
```

### Restore from Backup

```bash
# Stop Vault
systemctl stop vault

# Restore data
tar -xzf vault-backup-20240101.tar.gz -C /

# Start Vault
systemctl start vault

# Unseal
vault operator unseal <key-1>
vault operator unseal <key-2>
vault operator unseal <key-3>
```

## Security Best Practices

1. **Never use root token in production** - Create specific tokens with limited policies
2. **Enable TLS** - Always use HTTPS for Vault communication
3. **Rotate tokens regularly** - Application tokens should expire and be renewed
4. **Use least privilege** - Grant minimum required permissions
5. **Enable audit logging** - Track all access to secrets
6. **Backup regularly** - Maintain encrypted backups of Vault data
7. **Use auto-unseal** - Consider cloud KMS for auto-unsealing in production
8. **Monitor access** - Alert on unusual access patterns

## Troubleshooting

### Connection Issues

```bash
# Test connectivity
curl -k https://vault.example.com:8200/v1/sys/health

# Check environment
echo $VAULT_URL
echo $VAULT_TOKEN
```

### Authentication Failures

```bash
# Verify token is valid
vault token lookup

# Check token policies
vault token lookup -accessor <accessor>
```

### Secret Not Found

```bash
# List secrets
vault kv list secret/genesis

# Read specific secret
vault kv get secret/genesis/exchange/api-keys
```

## Migration from Environment Variables

To migrate from environment variables to Vault:

1. Extract current values from `.env` file
2. Store in Vault using paths above
3. Update application configuration to use Vault
4. Remove sensitive data from `.env` file
5. Test thoroughly in staging environment