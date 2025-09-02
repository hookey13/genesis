# Genesis Administrator Policy
# Full access to Genesis secrets and configuration

# Full access to application secrets
path "genesis-secrets/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Manage database connections and roles
path "genesis-database/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Manage transit encryption keys
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

path "sys/mounts/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Policy management
path "sys/policies/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "sys/policy/*" {
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

# Lease management
path "sys/leases/*" {
  capabilities = ["read", "update", "list"]
}

# Seal management (for emergency)
path "sys/seal" {
  capabilities = ["read", "sudo"]
}

# Key rotation
path "sys/rotate" {
  capabilities = ["update", "sudo"]
}

# Metrics and monitoring
path "sys/metrics" {
  capabilities = ["read"]
}

path "sys/pprof/*" {
  capabilities = ["read"]
}