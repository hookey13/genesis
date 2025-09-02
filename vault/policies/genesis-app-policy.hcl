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

# Read exchange API keys
path "genesis-secrets/data/exchange/*" {
  capabilities = ["read", "list"]
}

# Get dynamic database credentials
path "genesis-database/creds/genesis-app" {
  capabilities = ["read"]
}

# Use transit encryption for sensitive data
path "genesis-transit/encrypt/genesis-key" {
  capabilities = ["update"]
}

path "genesis-transit/decrypt/genesis-key" {
  capabilities = ["update"]
}

path "genesis-transit/rewrap/genesis-key" {
  capabilities = ["update"]
}

# Read transit key information
path "genesis-transit/keys/genesis-key" {
  capabilities = ["read"]
}

# Read own token information
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

# Renew own token
path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Revoke own token
path "auth/token/revoke-self" {
  capabilities = ["update"]
}