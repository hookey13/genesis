# Database Encryption at Rest

This document describes the database encryption implementation for Project GENESIS.

## Overview

Project GENESIS implements encryption at rest for all sensitive data using:
- **SQLCipher** for transparent database encryption (preferred)
- **Application-level encryption** as fallback
- **AES-256** encryption for backups
- **Key management via HashiCorp Vault**

## Encryption Methods

### 1. SQLCipher (Preferred)

SQLCipher provides transparent, page-level encryption for SQLite databases.

#### Installation

```bash
# Install SQLCipher
apt-get install sqlcipher libsqlcipher-dev

# Install Python binding
pip install pysqlcipher3
```

#### Configuration

```python
from genesis.data.encrypted_sqlite_repo import EncryptedSQLiteRepository

# Initialize with SQLCipher
repo = EncryptedSQLiteRepository(
    db_path=".genesis/data/genesis_encrypted.db",
    use_sqlcipher=True
)

await repo.initialize()
```

### 2. Application-Level Encryption (Fallback)

When SQLCipher is not available, sensitive columns are encrypted at the application level.

#### Encrypted Columns

The following columns are automatically encrypted:
- `api_key`
- `api_secret`
- `password`
- `token`
- `private_key`
- `secret`
- `credential`

#### Implementation

```python
from genesis.data.encrypted_database import DatabaseEncryption

# Initialize encryption
encryption = DatabaseEncryption()

# Encrypt value
encrypted = encryption.encrypt_value("sensitive_data")

# Decrypt value
decrypted = encryption.decrypt_value(encrypted)
```

## Key Management

### Key Generation

Database encryption keys are automatically generated using:
- **Fernet** (symmetric encryption)
- **256-bit keys** (32 bytes)
- **Cryptographically secure random generation**

### Key Storage

#### Production (Vault)

Keys are stored in HashiCorp Vault at:
```
/genesis/database/encryption-key
```

#### Development (Local)

For development, keys are stored locally with restricted permissions:
```
.genesis/.keys/database.key  (chmod 600)
```

### Key Rotation

#### Manual Rotation

```bash
# Rotate database encryption key
python scripts/rotate_database_key.py
```

#### Programmatic Rotation

```python
repo = EncryptedSQLiteRepository()
success = await repo.rotate_encryption_key()
```

## Backup Encryption

### Creating Encrypted Backups

```python
# Create encrypted backup
backup_path = await repo.backup(backup_dir=".genesis/backups")

# Backup with custom encryption
encrypted_db.backup(
    backup_path="/backups/genesis_2024.db.enc",
    encrypt_backup=True
)
```

### Backup Format

- **SQLCipher databases**: Native encrypted format
- **Standard SQLite**: AES-256 encrypted file
- **Compression**: Optional gzip compression
- **Integrity**: SHA-256 checksum verification

### Restoring from Backup

```python
# Restore from encrypted backup
await repo.restore(backup_path="/backups/genesis_2024.db.enc")
```

## Configuration

### Environment Variables

```bash
# Enable database encryption
export GENESIS_DB_ENCRYPTION=true

# Use SQLCipher (if available)
export GENESIS_USE_SQLCIPHER=true

# Vault configuration for key storage
export VAULT_URL=https://vault.example.com:8200
export VAULT_TOKEN=your_vault_token
```

### Settings Configuration

```python
# genesis/config/settings.py

class DatabaseSettings(BaseSettings):
    db_encryption: bool = Field(default=True, description="Enable database encryption")
    use_sqlcipher: bool = Field(default=True, description="Use SQLCipher if available")
    db_path: str = Field(default=".genesis/data/genesis.db", description="Database path")
    encryption_key_path: str = Field(default="/genesis/database/encryption-key", description="Vault path for key")
```

## Verification

### Check Encryption Status

```python
# Get encryption status
status = await repo.get_encryption_status()
print(status)
# Output:
# {
#     "encrypted": true,
#     "method": "SQLCipher",
#     "key_source": "Vault",
#     "verified": true
# }
```

### Verify Database Encryption

```python
# Verify encryption
encrypted_db = EncryptedSQLiteDatabase(db_path)
is_encrypted = encrypted_db.verify_encryption()
```

### Command Line Verification

```bash
# Check if database is encrypted
file genesis.db
# SQLCipher: "SQLite 3.x database, encrypted"
# Standard: "SQLite 3.x database"

# Attempt to read encrypted database
sqlite3 genesis.db "SELECT count(*) FROM sqlite_master"
# SQLCipher: "Error: file is not a database"
# Standard: Returns count
```

## Migration

### From Unencrypted to Encrypted

```python
from genesis.data.migration import encrypt_existing_database

# Migrate existing database
await encrypt_existing_database(
    source_db=".genesis/data/genesis.db",
    target_db=".genesis/data/genesis_encrypted.db",
    encryption_key=None  # Auto-generate
)
```

### Script for Migration

```bash
#!/bin/bash
# migrate_to_encrypted.sh

# Backup existing database
cp .genesis/data/genesis.db .genesis/data/genesis_backup.db

# Run migration
python -c "
from genesis.data.migration import encrypt_existing_database
import asyncio

asyncio.run(encrypt_existing_database(
    '.genesis/data/genesis.db',
    '.genesis/data/genesis_encrypted.db'
))
"

# Verify migration
python -c "
from genesis.data.encrypted_database import EncryptedSQLiteDatabase
db = EncryptedSQLiteDatabase('.genesis/data/genesis_encrypted.db')
print('Encryption verified:', db.verify_encryption())
"
```

## Performance Considerations

### SQLCipher Performance

- **Overhead**: ~5-15% for most operations
- **Page size**: Use 4096 bytes (default)
- **Cache size**: Increase for better performance
- **KDF iterations**: Balance security vs speed (default: 256000)

```sql
-- Optimize SQLCipher performance
PRAGMA cipher_page_size = 4096;
PRAGMA cache_size = 10000;
PRAGMA kdf_iter = 256000;
```

### Application-Level Encryption

- **Overhead**: ~10-20% for encrypted columns
- **Batch operations**: Use transactions for bulk inserts
- **Indexing**: Encrypted columns cannot be indexed
- **Searching**: Limited to exact matches

## Security Best Practices

1. **Never hardcode encryption keys**
2. **Use Vault for production key storage**
3. **Rotate keys quarterly**
4. **Encrypt all backups**
5. **Restrict database file permissions (600)**
6. **Use SQLCipher when possible**
7. **Monitor for unauthorized access attempts**
8. **Test restoration procedures regularly**

## Troubleshooting

### Common Issues

#### 1. "file is not a database" Error

**Cause**: Attempting to open encrypted database without key
**Solution**: Ensure encryption key is provided

```python
# Correct
conn = sqlite3.connect(db_path)
conn.execute(f"PRAGMA key = '{encryption_key}'")

# Incorrect
conn = sqlite3.connect(db_path)  # Missing key
```

#### 2. Slow Query Performance

**Cause**: Encrypted columns in WHERE clause
**Solution**: Use unencrypted indexes or redesign schema

```sql
-- Slow (encrypted column)
SELECT * FROM users WHERE encrypted_email = ?

-- Fast (unencrypted index)
SELECT * FROM users WHERE user_id = ?
```

#### 3. Key Rotation Failures

**Cause**: Insufficient permissions or corrupted data
**Solution**: Create backup before rotation

```bash
# Safe rotation procedure
1. Create backup
2. Verify backup
3. Rotate key
4. Test new key
5. Remove old backup
```

## Compliance

### Regulatory Requirements

- **PCI DSS**: Encryption of cardholder data at rest
- **GDPR**: Protection of personal data
- **SOC 2**: Encryption controls for sensitive data

### Audit Trail

All encryption operations are logged:
- Key generation
- Key rotation
- Backup creation
- Restoration attempts
- Access failures

## Testing

### Unit Tests

```python
# tests/unit/test_encrypted_database.py

async def test_database_encryption():
    """Test that database is properly encrypted."""
    db = EncryptedSQLiteDatabase(":memory:")
    
    # Insert sensitive data
    await db.execute(
        "INSERT INTO api_keys (key, secret) VALUES (?, ?)",
        ("test_key", "test_secret")
    )
    
    # Verify encryption
    assert db.verify_encryption()
    
    # Verify data can be decrypted
    result = await db.execute("SELECT * FROM api_keys")
    assert result[0]["key"] == "test_key"
```

### Integration Tests

```python
# tests/integration/test_encrypted_repo.py

async def test_encrypted_repository():
    """Test encrypted repository operations."""
    repo = EncryptedSQLiteRepository()
    await repo.initialize()
    
    # Test CRUD operations
    order = await repo.create_order(...)
    assert order.id is not None
    
    # Test backup
    backup_path = await repo.backup()
    assert Path(backup_path).exists()
    
    # Test restoration
    await repo.restore(backup_path)
```

## Monitoring

### Metrics to Track

- Encryption/decryption operations per second
- Key rotation frequency
- Backup success rate
- Restoration time
- Failed access attempts

### Alerts

- Key rotation overdue (> 90 days)
- Backup failure
- Decryption errors
- Unauthorized access attempts