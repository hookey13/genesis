"""HashiCorp Vault configuration and settings."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class VaultConfig(BaseSettings):
    """Vault configuration settings."""
    
    # Vault server configuration
    vault_url: str = Field(
        default="http://localhost:8200",
        env="VAULT_URL",
        description="Vault server URL"
    )
    vault_namespace: Optional[str] = Field(
        default=None,
        env="VAULT_NAMESPACE",
        description="Vault namespace for multi-tenant deployments"
    )
    
    # Authentication settings
    vault_auth_method: str = Field(
        default="token",
        env="VAULT_AUTH_METHOD",
        description="Authentication method: token, approle, aws, kubernetes"
    )
    vault_token: Optional[str] = Field(
        default=None,
        env="VAULT_TOKEN",
        description="Vault token for token auth"
    )
    vault_role_id: Optional[str] = Field(
        default=None,
        env="VAULT_ROLE_ID",
        description="AppRole ID for approle auth"
    )
    vault_secret_id: Optional[str] = Field(
        default=None,
        env="VAULT_SECRET_ID",
        description="AppRole secret ID"
    )
    
    # Secret paths
    kv_mount_point: str = Field(
        default="secret",
        env="VAULT_KV_MOUNT",
        description="KV v2 secrets engine mount point"
    )
    transit_mount_point: str = Field(
        default="transit",
        env="VAULT_TRANSIT_MOUNT",
        description="Transit encryption engine mount point"
    )
    database_mount_point: str = Field(
        default="database",
        env="VAULT_DATABASE_MOUNT",
        description="Database secrets engine mount point"
    )
    
    # Performance settings
    cache_ttl_seconds: int = Field(
        default=300,  # 5 minutes
        env="VAULT_CACHE_TTL",
        description="TTL for cached static secrets"
    )
    max_retries: int = Field(
        default=3,
        env="VAULT_MAX_RETRIES",
        description="Maximum retry attempts for Vault operations"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        env="VAULT_RETRY_DELAY",
        description="Initial retry delay (exponential backoff)"
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        env="VAULT_CIRCUIT_BREAKER_THRESHOLD",
        description="Consecutive failures before circuit breaker opens"
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        env="VAULT_CIRCUIT_BREAKER_TIMEOUT",
        description="Circuit breaker timeout in seconds"
    )
    
    # Token renewal settings
    token_renewal_threshold: int = Field(
        default=600,  # 10 minutes
        env="VAULT_TOKEN_RENEWAL_THRESHOLD",
        description="Renew token when TTL is below this threshold (seconds)"
    )
    
    # SSL/TLS settings
    verify_ssl: bool = Field(
        default=True,
        env="VAULT_VERIFY_SSL",
        description="Verify SSL certificates"
    )
    ca_cert_path: Optional[str] = Field(
        default=None,
        env="VAULT_CA_CERT",
        description="Path to CA certificate bundle"
    )
    
    # Emergency settings
    break_glass_enabled: bool = Field(
        default=True,
        env="VAULT_BREAK_GLASS_ENABLED",
        description="Enable break-glass emergency access"
    )
    break_glass_cache_file: str = Field(
        default="/tmp/.genesis_vault_cache",
        env="VAULT_BREAK_GLASS_CACHE",
        description="Encrypted cache file for break-glass scenarios"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"