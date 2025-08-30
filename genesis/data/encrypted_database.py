"""Encrypted database implementation for secure data at rest.

Provides transparent encryption for SQLite databases using SQLCipher
or fallback to application-level encryption.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any

import structlog
from cryptography.fernet import Fernet

from genesis.config.settings import settings
from genesis.security.vault_client import VaultClient

logger = structlog.get_logger(__name__)


class DatabaseEncryption:
    """Handles database encryption key management and operations."""

    def __init__(self, vault_client: VaultClient | None = None):
        """Initialize database encryption.
        
        Args:
            vault_client: Vault client for key management
        """
        self.vault_client = vault_client or settings.get_vault_client()
        self._encryption_key = None
        self._cipher = None

    def get_or_create_key(self) -> str:
        """Get or create database encryption key.
        
        Returns:
            Base64-encoded encryption key
        """
        # Try to get from Vault
        key = self.vault_client.get_database_encryption_key()

        if not key:
            # Generate new key
            key = self._generate_key()

            # Store in Vault
            success = self.vault_client.store_secret(
                VaultClient.DATABASE_ENCRYPTION_KEY_PATH,
                {"key": key}
            )

            if success:
                logger.info("Generated and stored new database encryption key")
            else:
                logger.warning("Failed to store key in Vault, using local storage")
                # Store locally as fallback (less secure)
                self._store_key_locally(key)

        self._encryption_key = key
        self._cipher = self._create_cipher(key)
        return key

    def _generate_key(self) -> str:
        """Generate a new encryption key.
        
        Returns:
            Base64-encoded encryption key
        """
        return Fernet.generate_key().decode()

    def _create_cipher(self, key: str) -> Fernet:
        """Create cipher from key.
        
        Args:
            key: Base64-encoded encryption key
            
        Returns:
            Fernet cipher instance
        """
        return Fernet(key.encode())

    def _store_key_locally(self, key: str):
        """Store key locally (fallback for development).
        
        Args:
            key: Encryption key to store
        """
        key_file = Path(".genesis/.keys/database.key")
        key_file.parent.mkdir(parents=True, exist_ok=True)

        # Protect with file permissions
        key_file.write_text(key)
        os.chmod(key_file, 0o600)  # Read/write for owner only

        logger.warning("Database key stored locally - not recommended for production",
                      path=str(key_file))

    def encrypt_value(self, value: Any) -> bytes:
        """Encrypt a value.
        
        Args:
            value: Value to encrypt
            
        Returns:
            Encrypted bytes
        """
        if not self._cipher:
            self.get_or_create_key()

        # Convert to bytes if needed
        if isinstance(value, str):
            data = value.encode()
        elif isinstance(value, (int, float)):
            data = str(value).encode()
        elif isinstance(value, bytes):
            data = value
        else:
            data = str(value).encode()

        return self._cipher.encrypt(data)

    def decrypt_value(self, encrypted: bytes) -> str:
        """Decrypt a value.
        
        Args:
            encrypted: Encrypted bytes
            
        Returns:
            Decrypted string
        """
        if not self._cipher:
            self.get_or_create_key()

        decrypted = self._cipher.decrypt(encrypted)
        return decrypted.decode()

    def rotate_key(self, new_key: str | None = None) -> str:
        """Rotate the database encryption key.
        
        Args:
            new_key: New key to use, or generate if None
            
        Returns:
            New encryption key
        """
        old_key = self._encryption_key
        old_cipher = self._cipher

        # Generate or use provided key
        if new_key:
            self._encryption_key = new_key
        else:
            self._encryption_key = self._generate_key()

        self._cipher = self._create_cipher(self._encryption_key)

        # Store new key in Vault
        success = self.vault_client.rotate_secret(
            VaultClient.DATABASE_ENCRYPTION_KEY_PATH,
            "key",
            self._encryption_key
        )

        if success:
            logger.info("Database encryption key rotated successfully")
        else:
            # Rollback
            self._encryption_key = old_key
            self._cipher = old_cipher
            raise Exception("Failed to rotate database encryption key")

        return self._encryption_key


class EncryptedSQLiteDatabase:
    """SQLite database with transparent encryption.
    
    Provides encryption at rest for SQLite databases using either
    SQLCipher (if available) or application-level encryption.
    """

    def __init__(
        self,
        db_path: str,
        encryption: DatabaseEncryption | None = None,
        use_sqlcipher: bool = True
    ):
        """Initialize encrypted database.
        
        Args:
            db_path: Path to database file
            encryption: Database encryption instance
            use_sqlcipher: Try to use SQLCipher if available
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encryption = encryption or DatabaseEncryption()
        self.use_sqlcipher = use_sqlcipher
        self.connection = None

        # Check if SQLCipher is available
        self.has_sqlcipher = self._check_sqlcipher()

        if self.use_sqlcipher and not self.has_sqlcipher:
            logger.warning("SQLCipher not available, using application-level encryption")

    def _check_sqlcipher(self) -> bool:
        """Check if SQLCipher is available.
        
        Returns:
            True if SQLCipher is available
        """
        try:
            # Try to import pysqlcipher3
            import pysqlcipher3
            return True
        except ImportError:
            return False

    def connect(self) -> sqlite3.Connection:
        """Connect to the encrypted database.
        
        Returns:
            Database connection
        """
        if self.connection:
            return self.connection

        if self.has_sqlcipher and self.use_sqlcipher:
            self.connection = self._connect_sqlcipher()
        else:
            self.connection = self._connect_standard()

        return self.connection

    def _connect_sqlcipher(self) -> Any:
        """Connect using SQLCipher.
        
        Returns:
            SQLCipher connection
        """
        try:
            import pysqlcipher3.dbapi2 as sqlcipher

            # Get encryption key
            key = self.encryption.get_or_create_key()

            # Connect to database
            conn = sqlcipher.connect(str(self.db_path))

            # Set encryption key
            conn.execute(f"PRAGMA key = '{key}'")

            # Verify connection
            conn.execute("SELECT count(*) FROM sqlite_master")

            logger.info("Connected to SQLCipher encrypted database",
                       path=str(self.db_path))

            return conn

        except Exception as e:
            logger.error("Failed to connect with SQLCipher",
                        error=str(e))
            raise

    def _connect_standard(self) -> sqlite3.Connection:
        """Connect using standard SQLite with application-level encryption.
        
        Returns:
            Standard SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path))

        # Enable encryption adapter
        conn.row_factory = self._encrypted_row_factory

        logger.info("Connected to SQLite with application-level encryption",
                   path=str(self.db_path))

        return conn

    def _encrypted_row_factory(self, cursor, row):
        """Row factory that handles encryption/decryption.
        
        Args:
            cursor: Database cursor
            row: Row data
            
        Returns:
            Decrypted row data
        """
        # Get column names
        columns = [col[0] for col in cursor.description]

        # Decrypt sensitive columns
        decrypted_row = []
        for i, value in enumerate(row):
            col_name = columns[i]

            # Check if column should be encrypted
            if self._should_encrypt_column(col_name):
                if value and isinstance(value, bytes):
                    try:
                        decrypted_value = self.encryption.decrypt_value(value)
                        decrypted_row.append(decrypted_value)
                    except Exception:
                        # Not encrypted or corrupted
                        decrypted_row.append(value)
                else:
                    decrypted_row.append(value)
            else:
                decrypted_row.append(value)

        return decrypted_row

    def _should_encrypt_column(self, column_name: str) -> bool:
        """Check if a column should be encrypted.
        
        Args:
            column_name: Name of the column
            
        Returns:
            True if column should be encrypted
        """
        # Encrypt sensitive columns
        sensitive_columns = [
            'api_key', 'api_secret', 'password', 'token',
            'private_key', 'secret', 'credential'
        ]

        return any(
            sensitive in column_name.lower()
            for sensitive in sensitive_columns
        )

    def execute(self, query: str, params: tuple | None = None) -> sqlite3.Cursor:
        """Execute a query with encryption support.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Database cursor
        """
        if not self.connection:
            self.connect()

        # Encrypt parameters if needed
        if params and not self.has_sqlcipher:
            encrypted_params = self._encrypt_params(query, params)
            return self.connection.execute(query, encrypted_params)
        else:
            return self.connection.execute(query, params or ())

    def _encrypt_params(self, query: str, params: tuple) -> tuple:
        """Encrypt sensitive parameters.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Encrypted parameters
        """
        # Parse query to find which columns are being inserted/updated
        query_lower = query.lower()

        if 'insert' in query_lower or 'update' in query_lower:
            # Simple heuristic: encrypt string parameters that look sensitive
            encrypted_params = []
            for param in params:
                if isinstance(param, str) and self._looks_sensitive(param):
                    encrypted_params.append(self.encryption.encrypt_value(param))
                else:
                    encrypted_params.append(param)
            return tuple(encrypted_params)

        return params

    def _looks_sensitive(self, value: str) -> bool:
        """Check if a value looks sensitive.
        
        Args:
            value: Value to check
            
        Returns:
            True if value appears sensitive
        """
        # Check length (keys are usually long)
        if len(value) > 20:
            # Check for key-like patterns
            if any(c in value for c in ['=', '-', '_']) and value.isalnum():
                return True

        return False

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Closed encrypted database connection")

    def backup(self, backup_path: str, encrypt_backup: bool = True):
        """Create an encrypted backup of the database.
        
        Args:
            backup_path: Path for the backup
            encrypt_backup: Whether to encrypt the backup
        """
        if not self.connection:
            self.connect()

        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        if self.has_sqlcipher and self.use_sqlcipher:
            # SQLCipher backup
            self.connection.execute(f"ATTACH DATABASE '{backup_path}' AS backup KEY '{self.encryption.get_or_create_key()}'")
            self.connection.execute("SELECT sqlcipher_export('backup')")
            self.connection.execute("DETACH DATABASE backup")
        else:
            # Standard backup with optional encryption
            import shutil

            if encrypt_backup:
                # Create encrypted copy
                temp_backup = backup_path.with_suffix('.tmp')
                shutil.copy2(self.db_path, temp_backup)

                # Encrypt the backup file
                with open(temp_backup, 'rb') as f:
                    data = f.read()

                encrypted_data = self.encryption.encrypt_value(data)

                with open(backup_path, 'wb') as f:
                    f.write(encrypted_data)

                temp_backup.unlink()
            else:
                shutil.copy2(self.db_path, backup_path)

        logger.info("Database backup created",
                   backup_path=str(backup_path),
                   encrypted=encrypt_backup)

    def restore(self, backup_path: str):
        """Restore database from encrypted backup.
        
        Args:
            backup_path: Path to the backup file
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Close current connection
        self.close()

        if self.has_sqlcipher and self.use_sqlcipher:
            # SQLCipher restore
            import shutil
            shutil.copy2(backup_path, self.db_path)
        else:
            # Check if backup is encrypted
            with open(backup_path, 'rb') as f:
                data = f.read()

            try:
                # Try to decrypt
                decrypted_data = self.encryption.decrypt_value(data)
                with open(self.db_path, 'wb') as f:
                    f.write(decrypted_data.encode())
            except Exception:
                # Not encrypted, copy directly
                import shutil
                shutil.copy2(backup_path, self.db_path)

        logger.info("Database restored from backup",
                   backup_path=str(backup_path))

    def verify_encryption(self) -> bool:
        """Verify that the database is properly encrypted.
        
        Returns:
            True if database is encrypted
        """
        if self.has_sqlcipher and self.use_sqlcipher:
            # Check SQLCipher encryption
            try:
                if not self.connection:
                    self.connect()

                # Check cipher settings
                result = self.connection.execute("PRAGMA cipher_version").fetchone()
                if result:
                    logger.info("Database encrypted with SQLCipher",
                              version=result[0])
                    return True
            except Exception as e:
                logger.error("Failed to verify SQLCipher encryption",
                            error=str(e))
                return False
        else:
            # Check if sensitive data is encrypted at application level
            try:
                # Read raw database file
                with open(self.db_path, 'rb') as f:
                    data = f.read(1024)  # Read first KB

                # Check for plaintext sensitive strings
                sensitive_patterns = [
                    b'api_key', b'api_secret', b'password',
                    b'BEGIN PRIVATE KEY', b'token'
                ]

                for pattern in sensitive_patterns:
                    if pattern in data:
                        logger.warning("Found unencrypted sensitive data",
                                     pattern=pattern.decode())
                        return False

                return True

            except Exception as e:
                logger.error("Failed to verify encryption",
                            error=str(e))
                return False
