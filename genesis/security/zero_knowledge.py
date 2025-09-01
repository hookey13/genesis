"""
Zero-Knowledge Architecture for Genesis.
Implements client-side encryption, secure multi-party computation,
and Shamir's Secret Sharing for sensitive operations.
"""

import os
import json
import hashlib
import hmac
import base64
import asyncio
import structlog
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

from genesis.core.exceptions import SecurityError

logger = structlog.get_logger(__name__)


class EncryptionMode(Enum):
    """Encryption modes for different security levels."""
    CLIENT_SIDE = "client_side"
    END_TO_END = "end_to_end"
    HOMOMORPHIC = "homomorphic"
    THRESHOLD = "threshold"


@dataclass
class EncryptedPayload:
    """Container for encrypted data with metadata."""
    ciphertext: bytes
    nonce: bytes
    tag: Optional[bytes] = None
    algorithm: str = "AES-256-GCM"
    timestamp: datetime = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if payload has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "tag": base64.b64encode(self.tag).decode() if self.tag else None,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedPayload":
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            tag=base64.b64decode(data["tag"]) if data.get("tag") else None,
            algorithm=data.get("algorithm", "AES-256-GCM"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        )


class ShamirSecret:
    """
    Shamir's Secret Sharing implementation.
    Splits secrets into n shares where k shares are required to reconstruct.
    """
    
    PRIME = 2**521 - 1  # Large prime for finite field
    
    def __init__(self, threshold: int, total_shares: int):
        """
        Initialize Shamir's Secret Sharing.
        
        Args:
            threshold: Minimum shares needed to reconstruct secret (k)
            total_shares: Total number of shares to generate (n)
        """
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
        
        self.threshold = threshold
        self.total_shares = total_shares
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def _polynomial(self, secret: int, degree: int) -> List[int]:
        """Generate random polynomial with secret as constant term."""
        coefficients = [secret]
        for _ in range(degree):
            coefficients.append(int.from_bytes(os.urandom(64), 'big') % self.PRIME)
        return coefficients
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x."""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, self.PRIME)) % self.PRIME
        return result
    
    def split_secret(self, secret: bytes) -> List[Tuple[int, int]]:
        """
        Split secret into shares using Shamir's Secret Sharing.
        
        Args:
            secret: Secret data to split
        
        Returns:
            List of (x, y) shares
        """
        # Convert secret to integer
        secret_int = int.from_bytes(secret, 'big')
        if secret_int >= self.PRIME:
            raise ValueError("Secret too large for field")
        
        # Generate polynomial
        coefficients = self._polynomial(secret_int, self.threshold - 1)
        
        # Generate shares
        shares = []
        for i in range(1, self.total_shares + 1):
            x = i
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))
        
        self.logger.info(
            "Secret split into shares",
            threshold=self.threshold,
            total_shares=self.total_shares
        )
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> bytes:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (x, y) shares
        
        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares to reconstruct")
        
        # Use only required number of shares
        shares = shares[:self.threshold]
        
        # Lagrange interpolation
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.PRIME
                    denominator = (denominator * (xi - xj)) % self.PRIME
            
            # Modular inverse of denominator
            inv_denominator = pow(denominator, self.PRIME - 2, self.PRIME)
            lagrange = (numerator * inv_denominator) % self.PRIME
            
            secret = (secret + yi * lagrange) % self.PRIME
        
        # Convert back to bytes
        byte_length = (secret.bit_length() + 7) // 8
        secret_bytes = secret.to_bytes(byte_length, 'big')
        
        self.logger.info("Secret reconstructed from shares", shares_used=len(shares))
        
        return secret_bytes


class ClientSideEncryption:
    """
    Client-side encryption for sensitive data.
    Ensures server never sees plaintext.
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize client-side encryption.
        
        Args:
            master_key: Master encryption key (32 bytes)
        """
        if master_key:
            if len(master_key) != 32:
                raise ValueError("Master key must be 32 bytes")
            self.master_key = master_key
        else:
            self.master_key = os.urandom(32)
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def derive_key(self, salt: bytes, info: bytes = b"") -> bytes:
        """
        Derive encryption key from master key.
        
        Args:
            salt: Salt for key derivation
            info: Optional context info
        
        Returns:
            Derived key (32 bytes)
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.master_key + info)
    
    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> EncryptedPayload:
        """
        Encrypt data using AES-256-GCM.
        
        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data
        
        Returns:
            Encrypted payload
        """
        # Generate nonce
        nonce = os.urandom(12)
        
        # Derive key
        salt = os.urandom(16)
        key = self.derive_key(salt)
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Combine salt with ciphertext
        combined_ciphertext = salt + ciphertext
        
        return EncryptedPayload(
            ciphertext=combined_ciphertext,
            nonce=nonce,
            tag=encryptor.tag
        )
    
    def decrypt(
        self,
        payload: EncryptedPayload,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt data using AES-256-GCM.
        
        Args:
            payload: Encrypted payload
            associated_data: Additional authenticated data
        
        Returns:
            Decrypted plaintext
        """
        if payload.is_expired():
            raise SecurityError("Encrypted payload has expired")
        
        # Extract salt and ciphertext
        salt = payload.ciphertext[:16]
        ciphertext = payload.ciphertext[16:]
        
        # Derive key
        key = self.derive_key(salt)
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(payload.nonce, payload.tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext


class EndToEndEncryption:
    """
    End-to-end encryption for API communications.
    Uses RSA for key exchange and AES for data encryption.
    """
    
    def __init__(self):
        """Initialize E2E encryption with key pair generation."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.peer_public_keys: Dict[str, rsa.RSAPublicKey] = {}
        self.session_keys: Dict[str, bytes] = {}
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def get_public_key_pem(self) -> bytes:
        """Get public key in PEM format."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def add_peer_public_key(self, peer_id: str, public_key_pem: bytes):
        """
        Add peer's public key for encryption.
        
        Args:
            peer_id: Identifier for the peer
            public_key_pem: Public key in PEM format
        """
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        self.peer_public_keys[peer_id] = public_key
        self.logger.info("Added peer public key", peer_id=peer_id)
    
    def establish_session(self, peer_id: str) -> bytes:
        """
        Establish encrypted session with peer.
        
        Args:
            peer_id: Identifier for the peer
        
        Returns:
            Encrypted session key for peer
        """
        if peer_id not in self.peer_public_keys:
            raise SecurityError(f"No public key for peer {peer_id}")
        
        # Generate session key
        session_key = os.urandom(32)
        self.session_keys[peer_id] = session_key
        
        # Encrypt session key with peer's public key
        peer_public_key = self.peer_public_keys[peer_id]
        encrypted_session_key = peer_public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        self.logger.info("Session established", peer_id=peer_id)
        
        return encrypted_session_key
    
    def decrypt_session_key(self, encrypted_session_key: bytes) -> bytes:
        """
        Decrypt session key from peer.
        
        Args:
            encrypted_session_key: Encrypted session key
        
        Returns:
            Decrypted session key
        """
        session_key = self.private_key.decrypt(
            encrypted_session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return session_key
    
    def encrypt_message(self, peer_id: str, message: bytes) -> EncryptedPayload:
        """
        Encrypt message for peer using session key.
        
        Args:
            peer_id: Identifier for the peer
            message: Message to encrypt
        
        Returns:
            Encrypted payload
        """
        if peer_id not in self.session_keys:
            raise SecurityError(f"No session established with peer {peer_id}")
        
        session_key = self.session_keys[peer_id]
        
        # Use AES-GCM for message encryption
        nonce = os.urandom(12)
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(message) + encryptor.finalize()
        
        return EncryptedPayload(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=encryptor.tag
        )
    
    def decrypt_message(self, peer_id: str, payload: EncryptedPayload) -> bytes:
        """
        Decrypt message from peer using session key.
        
        Args:
            peer_id: Identifier for the peer
            payload: Encrypted payload
        
        Returns:
            Decrypted message
        """
        if peer_id not in self.session_keys:
            raise SecurityError(f"No session established with peer {peer_id}")
        
        session_key = self.session_keys[peer_id]
        
        cipher = Cipher(
            algorithms.AES(session_key),
            modes.GCM(payload.nonce, payload.tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(payload.ciphertext) + decryptor.finalize()
        
        return plaintext


class HomomorphicOperation:
    """
    Simulated homomorphic encryption for calculations on encrypted data.
    Note: This is a simplified implementation for demonstration.
    Real homomorphic encryption requires specialized libraries like SEAL or HElib.
    """
    
    def __init__(self, modulus: int = 2**64):
        """
        Initialize homomorphic operations.
        
        Args:
            modulus: Modulus for operations
        """
        self.modulus = modulus
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def encrypt_value(self, value: int, key: int) -> Tuple[int, int]:
        """
        Encrypt value for homomorphic operations.
        
        Args:
            value: Value to encrypt
            key: Encryption key
        
        Returns:
            (encrypted_value, noise) tuple
        """
        noise = int.from_bytes(os.urandom(8), 'big') % 1000
        encrypted = (value + key + noise) % self.modulus
        
        return encrypted, noise
    
    def add_encrypted(self, encrypted1: int, encrypted2: int) -> int:
        """
        Add two encrypted values.
        
        Args:
            encrypted1: First encrypted value
            encrypted2: Second encrypted value
        
        Returns:
            Sum of encrypted values
        """
        return (encrypted1 + encrypted2) % self.modulus
    
    def multiply_encrypted(self, encrypted: int, scalar: int) -> int:
        """
        Multiply encrypted value by scalar.
        
        Args:
            encrypted: Encrypted value
            scalar: Scalar multiplier
        
        Returns:
            Product
        """
        return (encrypted * scalar) % self.modulus
    
    def decrypt_value(self, encrypted: int, key: int, noise: int) -> int:
        """
        Decrypt value from homomorphic encryption.
        
        Args:
            encrypted: Encrypted value
            key: Encryption key
            noise: Noise value
        
        Returns:
            Decrypted value
        """
        return (encrypted - key - noise) % self.modulus


class ZeroKnowledgeManager:
    """
    Manager for zero-knowledge operations in Genesis.
    Coordinates encryption, secret sharing, and secure computations.
    """
    
    def __init__(
        self,
        encryption_mode: EncryptionMode = EncryptionMode.CLIENT_SIDE,
        threshold: int = 3,
        total_shares: int = 5
    ):
        """
        Initialize zero-knowledge manager.
        
        Args:
            encryption_mode: Default encryption mode
            threshold: Threshold for secret sharing
            total_shares: Total shares for secret sharing
        """
        self.encryption_mode = encryption_mode
        
        # Initialize components
        self.client_encryption = ClientSideEncryption()
        self.e2e_encryption = EndToEndEncryption()
        self.shamir = ShamirSecret(threshold, total_shares)
        self.homomorphic = HomomorphicOperation()
        
        self.logger = structlog.get_logger(__name__)
    
    async def encrypt_api_credentials(
        self,
        api_key: str,
        api_secret: str
    ) -> Dict[str, Any]:
        """
        Encrypt API credentials with zero-knowledge approach.
        
        Args:
            api_key: API key to encrypt
            api_secret: API secret to encrypt
        
        Returns:
            Encrypted credentials with shares
        """
        # Combine credentials
        credentials = json.dumps({
            "api_key": api_key,
            "api_secret": api_secret,
            "timestamp": datetime.utcnow().isoformat()
        }).encode()
        
        # Client-side encryption
        encrypted_payload = self.client_encryption.encrypt(credentials)
        
        # Split master key into shares
        key_shares = self.shamir.split_secret(self.client_encryption.master_key)
        
        self.logger.info(
            "API credentials encrypted with zero-knowledge",
            shares_created=len(key_shares)
        )
        
        return {
            "encrypted_payload": encrypted_payload.to_dict(),
            "key_shares": [(x, y) for x, y in key_shares],
            "threshold": self.shamir.threshold,
            "encryption_mode": self.encryption_mode.value
        }
    
    async def decrypt_with_shares(
        self,
        encrypted_data: Dict[str, Any],
        shares: List[Tuple[int, int]]
    ) -> Dict[str, str]:
        """
        Decrypt data using secret shares.
        
        Args:
            encrypted_data: Encrypted payload data
            shares: Secret shares for reconstruction
        
        Returns:
            Decrypted credentials
        """
        # Reconstruct master key
        master_key = self.shamir.reconstruct_secret(shares)
        
        # Create encryption instance with reconstructed key
        encryption = ClientSideEncryption(master_key)
        
        # Decrypt payload
        payload = EncryptedPayload.from_dict(encrypted_data["encrypted_payload"])
        plaintext = encryption.decrypt(payload)
        
        # Parse credentials
        credentials = json.loads(plaintext.decode())
        
        self.logger.info("Credentials decrypted with shares", shares_used=len(shares))
        
        return credentials
    
    async def secure_calculation(
        self,
        value1: int,
        value2: int,
        operation: str = "add"
    ) -> int:
        """
        Perform calculation on encrypted values.
        
        Args:
            value1: First value
            value2: Second value
            operation: Operation to perform (add, multiply)
        
        Returns:
            Result of calculation
        """
        # Generate encryption key
        key = int.from_bytes(os.urandom(8), 'big')
        
        # Encrypt values
        enc1, noise1 = self.homomorphic.encrypt_value(value1, key)
        enc2, noise2 = self.homomorphic.encrypt_value(value2, key)
        
        # Perform operation on encrypted values
        if operation == "add":
            encrypted_result = self.homomorphic.add_encrypted(enc1, enc2)
            combined_noise = noise1 + noise2
        elif operation == "multiply":
            # Note: Real homomorphic multiplication is more complex
            encrypted_result = self.homomorphic.multiply_encrypted(enc1, value2)
            combined_noise = noise1 * value2
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Decrypt result
        result = self.homomorphic.decrypt_value(encrypted_result, key * 2, combined_noise)
        
        self.logger.info(
            "Secure calculation performed",
            operation=operation,
            encrypted=True
        )
        
        return result
    
    async def establish_secure_channel(self, peer_id: str, peer_public_key: bytes) -> bytes:
        """
        Establish secure E2E channel with peer.
        
        Args:
            peer_id: Peer identifier
            peer_public_key: Peer's public key in PEM format
        
        Returns:
            Encrypted session key for peer
        """
        # Add peer's public key
        self.e2e_encryption.add_peer_public_key(peer_id, peer_public_key)
        
        # Establish session
        encrypted_session_key = self.e2e_encryption.establish_session(peer_id)
        
        self.logger.info("Secure channel established", peer_id=peer_id)
        
        return encrypted_session_key
    
    async def send_secure_message(
        self,
        peer_id: str,
        message: Dict[str, Any]
    ) -> EncryptedPayload:
        """
        Send encrypted message to peer.
        
        Args:
            peer_id: Peer identifier
            message: Message to send
        
        Returns:
            Encrypted payload
        """
        message_bytes = json.dumps(message).encode()
        encrypted = self.e2e_encryption.encrypt_message(peer_id, message_bytes)
        
        self.logger.info("Secure message sent", peer_id=peer_id)
        
        return encrypted
    
    def get_status(self) -> Dict[str, Any]:
        """Get zero-knowledge system status."""
        return {
            "encryption_mode": self.encryption_mode.value,
            "shamir_threshold": self.shamir.threshold,
            "shamir_total_shares": self.shamir.total_shares,
            "e2e_peers": list(self.e2e_encryption.peer_public_keys.keys()),
            "active_sessions": list(self.e2e_encryption.session_keys.keys())
        }