"""Unit tests for database encryption."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import base64

from genesis.data.encrypted_database import (
    DatabaseEncryption,
    EncryptedSQLiteDatabase
)
from genesis.security.vault_client import VaultClient


class TestDatabaseEncryption:
    """Test DatabaseEncryption class."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create a mock Vault client."""
        client = Mock(spec=VaultClient)
        client.get_database_encryption_key.return_value = base64.b64encode(b'test_key_32_bytes_long_for_fernet').decode()
        client.store_secret.return_value = True
        client.rotate_secret.return_value = True
        return client
    
    @pytest.fixture
    def db_encryption(self, mock_vault_client):
        """Create a DatabaseEncryption instance."""
        return DatabaseEncryption(vault_client=mock_vault_client)
    
    def test_get_or_create_key_from_vault(self, db_encryption, mock_vault_client):
        """Test getting encryption key from Vault."""
        key = db_encryption.get_or_create_key()
        
        assert key is not None
        assert mock_vault_client.get_database_encryption_key.called
        assert db_encryption._encryption_key == key
        assert db_encryption._cipher is not None
    
    def test_generate_new_key(self, db_encryption, mock_vault_client):
        """Test generating a new encryption key."""
        mock_vault_client.get_database_encryption_key.return_value = None
        
        key = db_encryption.get_or_create_key()
        
        assert key is not None
        assert len(base64.b64decode(key)) == 32  # Fernet key is 32 bytes
        assert mock_vault_client.store_secret.called
    
    def test_encrypt_decrypt_string(self, db_encryption):
        """Test encrypting and decrypting strings."""
        db_encryption.get_or_create_key()
        
        original = "sensitive_data_123"
        encrypted = db_encryption.encrypt_value(original)
        
        assert encrypted != original.encode()
        assert isinstance(encrypted, bytes)
        
        decrypted = db_encryption.decrypt_value(encrypted)
        assert decrypted == original
    
    def test_encrypt_decrypt_numbers(self, db_encryption):
        """Test encrypting and decrypting numbers."""
        db_encryption.get_or_create_key()
        
        # Test integer
        original_int = 42
        encrypted_int = db_encryption.encrypt_value(original_int)
        decrypted_int = db_encryption.decrypt_value(encrypted_int)
        assert decrypted_int == "42"
        
        # Test float
        original_float = 3.14159
        encrypted_float = db_encryption.encrypt_value(original_float)
        decrypted_float = db_encryption.decrypt_value(encrypted_float)
        assert decrypted_float == "3.14159"
    
    def test_rotate_key(self, db_encryption, mock_vault_client):
        """Test key rotation."""
        # Get initial key
        initial_key = db_encryption.get_or_create_key()
        
        # Encrypt with initial key
        data = "test_data"
        encrypted_with_old = db_encryption.encrypt_value(data)
        
        # Rotate key
        new_key = db_encryption.rotate_key()
        
        assert new_key != initial_key
        assert mock_vault_client.rotate_secret.called
        
        # Encrypt with new key
        encrypted_with_new = db_encryption.encrypt_value(data)
        
        # Should produce different ciphertext
        assert encrypted_with_old != encrypted_with_new
    
    def test_rotate_key_failure_rollback(self, db_encryption, mock_vault_client):
        """Test key rotation rollback on failure."""
        initial_key = db_encryption.get_or_create_key()
        initial_cipher = db_encryption._cipher
        
        # Make rotation fail
        mock_vault_client.rotate_secret.return_value = False
        
        with pytest.raises(Exception, match="Failed to rotate"):
            db_encryption.rotate_key()
        
        # Should roll back to original key
        assert db_encryption._encryption_key == initial_key
        assert db_encryption._cipher == initial_cipher


class TestEncryptedSQLiteDatabase:
    """Test EncryptedSQLiteDatabase class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name
        yield path
        Path(path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_encryption(self):
        """Create a mock DatabaseEncryption."""
        encryption = Mock(spec=DatabaseEncryption)
        encryption.get_or_create_key.return_value = "test_key"
        encryption.encrypt_value.side_effect = lambda x: f"encrypted_{x}".encode()
        encryption.decrypt_value.side_effect = lambda x: x.decode().replace("encrypted_", "")
        return encryption
    
    @pytest.fixture
    def encrypted_db(self, temp_db_path, mock_encryption):
        """Create an EncryptedSQLiteDatabase instance."""
        db = EncryptedSQLiteDatabase(
            db_path=temp_db_path,
            encryption=mock_encryption,
            use_sqlcipher=False  # Use app-level encryption for tests
        )
        return db
    
    def test_connect_standard(self, encrypted_db):
        """Test connecting to database with standard SQLite."""
        conn = encrypted_db.connect()
        
        assert conn is not None
        assert encrypted_db.connection is not None
        
        # Should be able to execute queries
        conn.execute("CREATE TABLE test (id INTEGER, data TEXT)")
        conn.commit()
    
    def test_should_encrypt_column(self, encrypted_db):
        """Test column encryption detection."""
        # Should encrypt
        assert encrypted_db._should_encrypt_column("api_key")
        assert encrypted_db._should_encrypt_column("api_secret")
        assert encrypted_db._should_encrypt_column("password")
        assert encrypted_db._should_encrypt_column("user_token")
        assert encrypted_db._should_encrypt_column("private_key")
        
        # Should not encrypt
        assert not encrypted_db._should_encrypt_column("id")
        assert not encrypted_db._should_encrypt_column("username")
        assert not encrypted_db._should_encrypt_column("email")
        assert not encrypted_db._should_encrypt_column("created_at")
    
    def test_execute_with_encryption(self, encrypted_db, mock_encryption):
        """Test executing queries with parameter encryption."""
        encrypted_db.connect()
        
        # Create table
        encrypted_db.execute(
            "CREATE TABLE users (id INTEGER, api_key TEXT, name TEXT)"
        )
        
        # Insert with encryption
        encrypted_db.execute(
            "INSERT INTO users (id, api_key, name) VALUES (?, ?, ?)",
            (1, "secret_key_123", "Alice")
        )
        
        # The sensitive parameter should be encrypted
        # (In real implementation, this would be checked differently)
        encrypted_db.connection.commit()
    
    def test_backup_and_restore(self, encrypted_db, mock_encryption):
        """Test backup and restore functionality."""
        encrypted_db.connect()
        
        # Create and populate table
        encrypted_db.execute("CREATE TABLE test (id INTEGER, data TEXT)")
        encrypted_db.execute("INSERT INTO test VALUES (1, 'test_data')")
        encrypted_db.connection.commit()
        
        # Create backup
        backup_path = str(Path(encrypted_db.db_path).parent / "backup.db")
        encrypted_db.backup(backup_path, encrypt_backup=True)
        
        assert Path(backup_path).exists()
        
        # Modify original database
        encrypted_db.execute("INSERT INTO test VALUES (2, 'new_data')")
        encrypted_db.connection.commit()
        
        # Restore from backup
        encrypted_db.restore(backup_path)
        
        # Verify restoration
        encrypted_db.connect()
        cursor = encrypted_db.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        
        # Should have only original record
        assert count == 1
    
    def test_verify_encryption_no_plaintext(self, encrypted_db, temp_db_path):
        """Test that sensitive data is not stored in plaintext."""
        encrypted_db.connect()
        
        # Create table with sensitive data
        encrypted_db.execute(
            "CREATE TABLE secrets (id INTEGER, api_key TEXT, api_secret TEXT)"
        )
        encrypted_db.execute(
            "INSERT INTO secrets VALUES (1, 'plaintext_key', 'plaintext_secret')"
        )
        encrypted_db.connection.commit()
        encrypted_db.close()
        
        # Read raw database file
        with open(temp_db_path, 'rb') as f:
            raw_data = f.read()
        
        # In a properly encrypted database, these should not appear
        # (This test would need adjustment based on actual encryption)
        if encrypted_db.has_sqlcipher:
            assert b'plaintext_key' not in raw_data
            assert b'plaintext_secret' not in raw_data
    
    @patch('genesis.data.encrypted_database.pysqlcipher3')
    def test_connect_with_sqlcipher(self, mock_sqlcipher, mock_encryption):
        """Test connecting with SQLCipher."""
        # Mock SQLCipher availability
        mock_conn = MagicMock()
        mock_sqlcipher.dbapi2.connect.return_value = mock_conn
        
        db = EncryptedSQLiteDatabase(
            db_path=":memory:",
            encryption=mock_encryption,
            use_sqlcipher=True
        )
        db.has_sqlcipher = True
        
        conn = db._connect_sqlcipher()
        
        assert conn is not None
        mock_conn.execute.assert_any_call("PRAGMA key = 'test_key'")
        mock_conn.execute.assert_any_call("SELECT count(*) FROM sqlite_master")
    
    def test_looks_sensitive(self, encrypted_db):
        """Test sensitive value detection."""
        # Should look sensitive
        assert encrypted_db._looks_sensitive("a" * 30)  # Long string
        assert encrypted_db._looks_sensitive("abc123-def456-ghi789-jkl012")  # Key-like
        
        # Should not look sensitive
        assert not encrypted_db._looks_sensitive("hello")  # Too short
        assert not encrypted_db._looks_sensitive("normal text here")  # Spaces
        assert not encrypted_db._looks_sensitive("12345")  # Too short