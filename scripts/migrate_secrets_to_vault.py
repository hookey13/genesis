#!/usr/bin/env python
"""Migrate existing secrets to HashiCorp Vault.

This script helps migrate all existing secrets from environment variables,
configuration files, and hardcoded values to HashiCorp Vault.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from genesis.security.vault_manager import VaultManager
from genesis.security.vault_integration import VaultIntegration
from genesis.config.vault_config import VaultConfig


class SecretsMigrator:
    """Migrates existing secrets to Vault."""
    
    def __init__(self):
        """Initialize migrator."""
        self.vault_integration = None
        self.secrets_to_migrate = {}
        self.migration_report = {
            "migrated": [],
            "failed": [],
            "skipped": []
        }
        
    async def initialize(self):
        """Initialize Vault connection."""
        print("Initializing Vault connection...")
        self.vault_integration = await VaultIntegration.get_instance()
        print("✓ Vault connection established")
        
    async def scan_environment_variables(self):
        """Scan environment variables for secrets."""
        print("\nScanning environment variables...")
        
        secret_patterns = [
            "API_KEY", "API_SECRET", "PASSWORD", "TOKEN",
            "SECRET", "PRIVATE_KEY", "CLIENT_SECRET"
        ]
        
        for env_var, value in os.environ.items():
            # Check if this looks like a secret
            if any(pattern in env_var.upper() for pattern in secret_patterns):
                if env_var.startswith("VAULT_"):
                    # Skip Vault configuration itself
                    continue
                    
                self.secrets_to_migrate[f"env/{env_var}"] = {
                    "type": "environment",
                    "name": env_var,
                    "value": value,
                    "vault_path": f"environment/{env_var.lower()}"
                }
                print(f"  Found: {env_var}")
        
        print(f"✓ Found {len([s for s in self.secrets_to_migrate.values() if s['type'] == 'environment'])} environment secrets")
    
    async def scan_config_files(self):
        """Scan configuration files for secrets."""
        print("\nScanning configuration files...")
        
        config_paths = [
            Path(".env"),
            Path("config/settings.json"),
            Path("config/database.yml"),
            Path("config/api_keys.json")
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                print(f"  Checking {config_path}...")
                
                if config_path.suffix == ".env":
                    await self._scan_dotenv(config_path)
                elif config_path.suffix == ".json":
                    await self._scan_json(config_path)
                elif config_path.suffix in [".yml", ".yaml"]:
                    await self._scan_yaml(config_path)
        
        config_count = len([s for s in self.secrets_to_migrate.values() if s['type'] == 'config'])
        print(f"✓ Found {config_count} configuration secrets")
    
    async def _scan_dotenv(self, path: Path):
        """Scan .env file for secrets."""
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        
                        if any(pattern in key.upper() for pattern in 
                               ["API", "SECRET", "PASSWORD", "TOKEN", "KEY"]):
                            self.secrets_to_migrate[f"dotenv/{key}"] = {
                                "type": "config",
                                "name": key,
                                "value": value,
                                "vault_path": f"config/{key.lower()}"
                            }
        except Exception as e:
            print(f"    Error scanning {path}: {e}")
    
    async def _scan_json(self, path: Path):
        """Scan JSON file for secrets."""
        try:
            with open(path) as f:
                data = json.load(f)
                await self._extract_secrets_from_dict(data, "config", str(path))
        except Exception as e:
            print(f"    Error scanning {path}: {e}")
    
    async def _scan_yaml(self, path: Path):
        """Scan YAML file for secrets."""
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
                await self._extract_secrets_from_dict(data, "config", str(path))
        except ImportError:
            print("    PyYAML not installed, skipping YAML files")
        except Exception as e:
            print(f"    Error scanning {path}: {e}")
    
    async def _extract_secrets_from_dict(self, data: Dict, prefix: str, source: str):
        """Extract secrets from dictionary structure."""
        if not isinstance(data, dict):
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                await self._extract_secrets_from_dict(value, f"{prefix}/{key}", source)
            elif isinstance(value, str) and value:
                if any(pattern in key.upper() for pattern in 
                       ["API", "SECRET", "PASSWORD", "TOKEN", "KEY", "CREDENTIAL"]):
                    self.secrets_to_migrate[f"{source}/{key}"] = {
                        "type": "config",
                        "name": key,
                        "value": value,
                        "vault_path": f"{prefix}/{key.lower()}"
                    }
    
    async def migrate_to_vault(self):
        """Migrate all discovered secrets to Vault."""
        print(f"\nMigrating {len(self.secrets_to_migrate)} secrets to Vault...")
        
        for secret_id, secret_info in self.secrets_to_migrate.items():
            try:
                # Prepare secret data
                secret_data = {
                    "value": secret_info["value"],
                    "name": secret_info["name"],
                    "type": secret_info["type"],
                    "migrated_from": secret_id
                }
                
                # Write to Vault
                await self.vault_integration.vault_manager.write_secret(
                    secret_info["vault_path"],
                    secret_data
                )
                
                self.migration_report["migrated"].append({
                    "name": secret_info["name"],
                    "vault_path": secret_info["vault_path"]
                })
                
                print(f"  ✓ Migrated {secret_info['name']} -> {secret_info['vault_path']}")
                
            except Exception as e:
                self.migration_report["failed"].append({
                    "name": secret_info["name"],
                    "error": str(e)
                })
                print(f"  ✗ Failed to migrate {secret_info['name']}: {e}")
    
    async def setup_database_dynamic_secrets(self):
        """Configure dynamic database credential generation."""
        print("\nSetting up dynamic database credentials...")
        
        try:
            # Store static database config (host, port, database name)
            db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "database": os.getenv("DB_NAME", "genesis_trading")
            }
            
            await self.vault_integration.vault_manager.write_secret(
                "database/config",
                db_config
            )
            
            # Store static fallback credentials (encrypted)
            static_creds = {
                "username": os.getenv("DB_USER", "genesis"),
                "password": os.getenv("DB_PASSWORD", "")
            }
            
            if static_creds["password"]:
                await self.vault_integration.vault_manager.write_secret(
                    "database/static-creds",
                    static_creds
                )
                print("  ✓ Database configuration stored in Vault")
            else:
                print("  ⚠ No database password found in environment")
                
        except Exception as e:
            print(f"  ✗ Failed to setup database secrets: {e}")
    
    async def setup_jwt_keys(self):
        """Setup JWT signing keys in Vault."""
        print("\nSetting up JWT signing keys...")
        
        try:
            # Check if JWT key already exists
            existing_key = os.getenv("JWT_SECRET_KEY")
            
            if existing_key:
                # Migrate existing key
                await self.vault_integration.vault_manager.write_secret(
                    "jwt/signing-key",
                    {"key": existing_key, "created_at": "migrated"}
                )
                print("  ✓ Migrated existing JWT signing key")
            else:
                # Generate new key
                import secrets
                new_key = secrets.token_urlsafe(64)
                await self.vault_integration.vault_manager.write_secret(
                    "jwt/signing-key",
                    {"key": new_key, "created_at": "generated"}
                )
                print("  ✓ Generated new JWT signing key")
                
        except Exception as e:
            print(f"  ✗ Failed to setup JWT keys: {e}")
    
    async def setup_api_keys(self):
        """Setup API keys in Vault."""
        print("\nSetting up API keys...")
        
        # Common exchange API keys
        exchanges = ["binance", "coinbase", "kraken"]
        
        for exchange in exchanges:
            api_key = os.getenv(f"{exchange.upper()}_API_KEY")
            api_secret = os.getenv(f"{exchange.upper()}_API_SECRET")
            
            if api_key and api_secret:
                try:
                    await self.vault_integration.vault_manager.write_secret(
                        f"api-keys/{exchange}",
                        {
                            "api_key": api_key,
                            "api_secret": api_secret,
                            "exchange": exchange
                        }
                    )
                    print(f"  ✓ Migrated {exchange} API credentials")
                except Exception as e:
                    print(f"  ✗ Failed to migrate {exchange} credentials: {e}")
            else:
                print(f"  ⚠ No {exchange} credentials found")
    
    async def generate_migration_report(self):
        """Generate and save migration report."""
        print("\n" + "="*50)
        print("MIGRATION REPORT")
        print("="*50)
        
        print(f"\nSuccessfully migrated: {len(self.migration_report['migrated'])}")
        for item in self.migration_report["migrated"]:
            print(f"  ✓ {item['name']} -> {item['vault_path']}")
        
        if self.migration_report["failed"]:
            print(f"\nFailed migrations: {len(self.migration_report['failed'])}")
            for item in self.migration_report["failed"]:
                print(f"  ✗ {item['name']}: {item['error']}")
        
        # Save report to file
        report_path = Path("migration_report.json")
        with open(report_path, "w") as f:
            json.dump(self.migration_report, f, indent=2)
        
        print(f"\n✓ Migration report saved to {report_path}")
    
    async def cleanup_old_secrets(self):
        """Provide instructions for cleaning up old secrets."""
        print("\n" + "="*50)
        print("CLEANUP INSTRUCTIONS")
        print("="*50)
        
        print("\nTo complete the migration, perform these manual steps:")
        print("\n1. Update .env file:")
        print("   - Remove all migrated secrets")
        print("   - Keep only VAULT_URL and VAULT_TOKEN")
        
        print("\n2. Update application code:")
        print("   - Replace os.getenv() calls with Vault integration")
        print("   - Use await get_vault() to access secrets")
        
        print("\n3. Rotate all secrets:")
        print("   - Run: python -m genesis.security.vault_integration rotate")
        
        print("\n4. Test the application:")
        print("   - Verify all components can access secrets")
        print("   - Check that fallback mechanisms work")
        
        print("\n5. Secure the migration report:")
        print("   - Delete or encrypt migration_report.json")
        print("   - It contains sensitive information")
    
    async def run(self):
        """Run the complete migration process."""
        try:
            # Initialize Vault
            await self.initialize()
            
            # Scan for secrets
            await self.scan_environment_variables()
            await self.scan_config_files()
            
            # Migrate to Vault
            await self.migrate_to_vault()
            
            # Setup specific secret types
            await self.setup_database_dynamic_secrets()
            await self.setup_jwt_keys()
            await self.setup_api_keys()
            
            # Generate report
            await self.generate_migration_report()
            
            # Cleanup instructions
            await self.cleanup_old_secrets()
            
            print("\n✅ Migration completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            sys.exit(1)
        finally:
            if self.vault_integration:
                await self.vault_integration.shutdown()


async def main():
    """Main entry point."""
    print("="*50)
    print("GENESIS SECRETS MIGRATION TO VAULT")
    print("="*50)
    
    # Check if Vault is configured
    if not os.getenv("VAULT_URL"):
        print("\n❌ Error: VAULT_URL environment variable not set")
        print("Please configure Vault connection first:")
        print("  export VAULT_URL=http://localhost:8200")
        print("  export VAULT_TOKEN=your-token")
        sys.exit(1)
    
    # Confirm migration
    print("\n⚠️  WARNING: This will migrate all secrets to Vault")
    print("Make sure you have:")
    print("  1. Vault server running and accessible")
    print("  2. Proper Vault token with write permissions")
    print("  3. Backup of all current secrets")
    
    response = input("\nProceed with migration? (yes/no): ")
    if response.lower() != "yes":
        print("Migration cancelled")
        sys.exit(0)
    
    # Run migration
    migrator = SecretsMigrator()
    await migrator.run()


if __name__ == "__main__":
    asyncio.run(main())