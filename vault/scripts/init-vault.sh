#!/bin/bash
# HashiCorp Vault Initialization Script
# Initializes and configures Vault for Genesis Trading Platform

set -e

# Configuration
VAULT_ADDR=${VAULT_ADDR:-"http://127.0.0.1:8200"}
VAULT_INIT_FILE="/vault/data/init-keys.txt"
VAULT_TOKEN_FILE="/vault/data/.vault-token"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Wait for Vault to be ready
print_status "Waiting for Vault to start..."
until vault status 2>/dev/null; do
    sleep 2
done

# Check if Vault is already initialized
if vault status 2>/dev/null | grep -q "Initialized.*true"; then
    print_warning "Vault is already initialized"
else
    print_status "Initializing Vault..."
    
    # Initialize with 5 key shares and threshold of 3
    vault operator init \
        -key-shares=5 \
        -key-threshold=3 \
        -format=json > "$VAULT_INIT_FILE"
    
    if [ $? -eq 0 ]; then
        print_status "Vault initialized successfully"
        print_warning "IMPORTANT: Backup the keys in $VAULT_INIT_FILE immediately!"
        
        # Extract root token
        ROOT_TOKEN=$(cat "$VAULT_INIT_FILE" | jq -r '.root_token')
        echo "$ROOT_TOKEN" > "$VAULT_TOKEN_FILE"
        
        # Extract unseal keys
        for i in {0..2}; do
            KEY=$(cat "$VAULT_INIT_FILE" | jq -r ".unseal_keys_b64[$i]")
            print_status "Unsealing with key $((i+1))..."
            vault operator unseal "$KEY"
        done
        
        # Authenticate with root token
        export VAULT_TOKEN="$ROOT_TOKEN"
        
        print_status "Vault unsealed and ready"
    else
        print_error "Failed to initialize Vault"
        exit 1
    fi
fi

# Login with existing token if available
if [ -f "$VAULT_TOKEN_FILE" ]; then
    export VAULT_TOKEN=$(cat "$VAULT_TOKEN_FILE")
    print_status "Authenticated with existing token"
fi

# Enable audit logging
print_status "Configuring audit logging..."
if ! vault audit list 2>/dev/null | grep -q "file/"; then
    vault audit enable file \
        file_path=/vault/logs/audit.log \
        log_raw=false \
        hmac_accessor=true \
        mode=0600 \
        format=json \
        prefix="genesis-" || print_warning "Audit already enabled"
    print_status "Audit logging enabled"
fi

# Enable secret engines
print_status "Configuring secret engines..."

# KV-v2 Secret Engine
if ! vault secrets list -format=json | jq -r 'keys[]' | grep -q "genesis-secrets"; then
    vault secrets enable -path=genesis-secrets kv-v2
    vault kv enable-versioning genesis-secrets/
    vault write genesis-secrets/config \
        max_versions=10 \
        delete_version_after="720h" \
        cas_required=false
    print_status "KV-v2 secret engine enabled at genesis-secrets/"
fi

# Transit Engine for encryption
if ! vault secrets list -format=json | jq -r 'keys[]' | grep -q "genesis-transit"; then
    vault secrets enable -path=genesis-transit transit
    
    # Create encryption key with convergent encryption
    vault write -f genesis-transit/keys/genesis-key \
        convergent_encryption=true \
        derived=true \
        exportable=false \
        allow_plaintext_backup=false \
        type=aes256-gcm96
    
    # Configure key rotation policy
    vault write genesis-transit/keys/genesis-key/config \
        min_decryption_version=1 \
        min_encryption_version=1 \
        deletion_allowed=false \
        auto_rotate_period=720h
    
    print_status "Transit engine enabled at genesis-transit/"
fi

# Database Engine for PostgreSQL
if ! vault secrets list -format=json | jq -r 'keys[]' | grep -q "genesis-database"; then
    vault secrets enable -path=genesis-database database
    
    # Note: Database connection will be configured when PostgreSQL is available
    print_status "Database engine enabled at genesis-database/"
    print_warning "Database connection must be configured when PostgreSQL is available"
fi

# Create initial secret structure
print_status "Creating initial secret structure..."

# Application secrets structure
vault kv put genesis-secrets/app/config \
    environment="development" \
    log_level="INFO" \
    created_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

vault kv put genesis-secrets/app/api \
    service_name="genesis-trading" \
    version="1.0.0"

vault kv put genesis-secrets/app/features \
    two_factor_enabled="true" \
    session_timeout_minutes="30" \
    max_login_attempts="5"

# Trading configuration secrets
vault kv put genesis-secrets/trading/limits \
    max_position_size="10000" \
    max_daily_loss="500" \
    max_orders_per_minute="60"

vault kv put genesis-secrets/trading/risk \
    stop_loss_percentage="2" \
    take_profit_percentage="5" \
    max_leverage="3"

print_status "Initial secret structure created"

# Apply security policies
print_status "Applying security policies..."

# Create policies directory if not exists
mkdir -p /vault/policies

# Genesis App Policy
cat > /vault/policies/genesis-app-policy.hcl <<EOF
# Genesis Application Policy
# Read-only access to application secrets and credentials

# Read application configuration
path "genesis-secrets/data/app/*" {
  capabilities = ["read", "list"]
}

# Read trading configuration
path "genesis-secrets/data/trading/*" {
  capabilities = ["read", "list"]
}

# Get database credentials
path "genesis-database/creds/genesis-app" {
  capabilities = ["read"]
}

# Use transit encryption
path "genesis-transit/encrypt/genesis-key" {
  capabilities = ["update"]
}

path "genesis-transit/decrypt/genesis-key" {
  capabilities = ["update"]
}

path "genesis-transit/rewrap/genesis-key" {
  capabilities = ["update"]
}

# Read own token information
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

# Renew own token
path "auth/token/renew-self" {
  capabilities = ["update"]
}
EOF

vault policy write genesis-app /vault/policies/genesis-app-policy.hcl
print_status "Applied genesis-app policy"

# Genesis Admin Policy
cat > /vault/policies/genesis-admin-policy.hcl <<EOF
# Genesis Administrator Policy
# Full access to Genesis secrets and configuration

# Full access to secrets
path "genesis-secrets/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Manage database connections and roles
path "genesis-database/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Manage transit keys
path "genesis-transit/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# System health and information
path "sys/health" {
  capabilities = ["read"]
}

path "sys/mounts" {
  capabilities = ["read", "list"]
}

path "sys/policies/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Audit log management
path "sys/audit/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Token management
path "auth/token/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
EOF

vault policy write genesis-admin /vault/policies/genesis-admin-policy.hcl
print_status "Applied genesis-admin policy"

# Emergency Break-Glass Policy
cat > /vault/policies/genesis-emergency-policy.hcl <<EOF
# Genesis Emergency Break-Glass Policy
# Emergency access for critical incidents

# Root-like access to all Genesis paths
path "genesis-*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# System access for emergency operations
path "sys/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Auth management
path "auth/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}
EOF

vault policy write genesis-emergency /vault/policies/genesis-emergency-policy.hcl
print_status "Applied genesis-emergency policy"

# Create application tokens
print_status "Creating application tokens..."

# Create app token with 24h TTL
APP_TOKEN=$(vault token create \
    -policy=genesis-app \
    -ttl=24h \
    -renewable \
    -display-name="genesis-app-token" \
    -format=json | jq -r '.auth.client_token')

if [ ! -z "$APP_TOKEN" ]; then
    echo "$APP_TOKEN" > /vault/data/app-token.txt
    print_status "Application token created and saved"
fi

# Performance tuning
print_status "Applying performance tuning..."

# Configure connection limits
vault write sys/config/control-group \
    max_ttl="1h" \
    ttl="30m"

# Setup rate limiting
vault write sys/quotas/rate-limit/genesis-global \
    rate=1000 \
    interval=1s \
    path=""

print_status "Performance tuning applied"

# Final status
print_status "Vault initialization complete!"
print_warning "Remember to:"
print_warning "  1. Backup init keys from $VAULT_INIT_FILE"
print_warning "  2. Distribute unseal keys to key holders"
print_warning "  3. Store root token securely"
print_warning "  4. Configure database connection when PostgreSQL is ready"
print_warning "  5. Enable TLS in production"
print_warning "  6. Setup AWS KMS for auto-unseal in production"

# Display tokens for reference
echo ""
print_status "Generated Tokens:"
echo "  Root Token: Stored in $VAULT_TOKEN_FILE"
echo "  App Token: Stored in /vault/data/app-token.txt"
echo ""
print_status "Vault is ready for use at $VAULT_ADDR"