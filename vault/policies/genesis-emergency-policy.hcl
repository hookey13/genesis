# Genesis Emergency Break-Glass Policy
# Emergency access for critical incidents
# Use only in documented emergency situations

# Root-like access to all Genesis paths
path "genesis-*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Full system access for emergency operations
path "sys/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Auth management for recovery
path "auth/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Identity management
path "identity/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Full secret access
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

path "kv/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Seal/unseal operations
path "sys/seal" {
  capabilities = ["update", "sudo"]
}

path "sys/unseal" {
  capabilities = ["update", "sudo"]
}

path "sys/seal-status" {
  capabilities = ["read"]
}

# Generate root token for recovery
path "sys/generate-root/*" {
  capabilities = ["update", "sudo"]
}

# Step down from leadership (HA clusters)
path "sys/step-down" {
  capabilities = ["update", "sudo"]
}

# Raw access (dangerous - use with extreme caution)
path "sys/raw/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Recovery operations
path "sys/rekey/*" {
  capabilities = ["create", "read", "update", "delete", "sudo"]
}

path "sys/rekey-recovery-key/*" {
  capabilities = ["create", "read", "update", "delete", "sudo"]
}