# HashiCorp Vault Configuration
# Production-ready configuration for Genesis Trading Platform

# Enable UI for administration
ui = true

# Cluster identification
cluster_name = "genesis-vault"

# Logging configuration
log_level = "Info"
log_format = "json"

# API listener configuration
listener "tcp" {
  address = "0.0.0.0:8200"
  
  # TLS Configuration - Disabled for initial setup, enable in production
  tls_disable = 1
  
  # When TLS is enabled, uncomment these:
  # tls_cert_file = "/vault/config/certs/vault.crt"
  # tls_key_file  = "/vault/config/certs/vault.key"
  # tls_min_version = "tls12"
  
  # Performance tuning
  max_request_size = 33554432
  max_request_duration = "90s"
  
  # CORS configuration for UI
  cors {
    enabled = true
    allowed_origins = ["http://localhost:8200"]
  }
}

# Cluster listener for HA
listener "tcp" {
  address = "0.0.0.0:8201"
  tls_disable = 1
  cluster_address = "0.0.0.0:8201"
}

# Storage backend configuration
storage "file" {
  path = "/vault/data"
  
  # Performance tuning
  max_parallel = 128
}

# For production with AWS KMS auto-unseal, uncomment:
# seal "awskms" {
#   region = "us-east-1"
#   kms_key_id = "alias/genesis-vault-unseal"
#   endpoint = ""  # Optional custom endpoint
# }

# High Availability configuration
# ha_storage "file" {
#   path = "/vault/data"
#   redirect_addr = "http://127.0.0.1:8200"
#   cluster_addr = "https://127.0.0.1:8201"
# }

# API and cluster addresses
api_addr = "http://127.0.0.1:8200"
cluster_addr = "http://127.0.0.1:8201"

# Performance and security settings
default_lease_ttl = "168h"  # 7 days
max_lease_ttl = "720h"      # 30 days
disable_mlock = false
disable_cache = false

# Telemetry configuration for monitoring
telemetry {
  statsite_address = "127.0.0.1:8125"
  statsd_address = "127.0.0.1:8125"
  prometheus_retention_time = "24h"
  disable_hostname = false
}

# Enable audit logging
# Note: This will be enabled after initialization
# audit {
#   enabled = true
#   path = "file"
#   options = {
#     file_path = "/vault/logs/audit.log"
#     log_raw = false
#     hmac_accessor = true
#     mode = "0600"
#     format = "json"
#   }
# }