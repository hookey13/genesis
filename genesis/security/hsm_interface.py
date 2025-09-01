"""
Hardware Security Module (HSM) interface for Genesis.
Provides abstraction for HSM operations with PKCS#11 support.
"""

import os
import asyncio
import structlog
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from genesis.core.exceptions import SecurityError

logger = structlog.get_logger(__name__)


class HSMType(Enum):
    """Supported HSM types."""
    SOFTHSM = "softhsm"  # Software HSM for testing
    LUNA = "luna"  # Thales Luna HSM
    YUBIHSM = "yubihsm"  # YubiHSM
    AWS_CLOUDHSM = "aws_cloudhsm"  # AWS CloudHSM
    SIMULATOR = "simulator"  # Mock HSM for testing


class KeyType(Enum):
    """Cryptographic key types."""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"


@dataclass
class HSMKey:
    """Representation of a key stored in HSM."""
    key_id: str
    key_type: KeyType
    label: str
    created_at: datetime
    attributes: Dict[str, Any]
    handle: Optional[int] = None  # PKCS#11 handle


class HSMInterface(ABC):
    """Abstract interface for HSM operations."""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize HSM connection."""
        pass
    
    @abstractmethod
    async def generate_key(
        self,
        key_type: KeyType,
        label: str,
        extractable: bool = False
    ) -> HSMKey:
        """Generate a new key in the HSM."""
        pass
    
    @abstractmethod
    async def import_key(
        self,
        key_material: bytes,
        key_type: KeyType,
        label: str
    ) -> HSMKey:
        """Import an existing key into the HSM."""
        pass
    
    @abstractmethod
    async def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt data using HSM key."""
        pass
    
    @abstractmethod
    async def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt data using HSM key."""
        pass
    
    @abstractmethod
    async def sign(self, key_id: str, data: bytes) -> bytes:
        """Sign data using HSM key."""
        pass
    
    @abstractmethod
    async def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """Verify signature using HSM key."""
        pass
    
    @abstractmethod
    async def delete_key(self, key_id: str) -> bool:
        """Delete a key from the HSM."""
        pass
    
    @abstractmethod
    async def list_keys(self) -> List[HSMKey]:
        """List all keys in the HSM."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check HSM health and connectivity."""
        pass


class PKCS11HSM(HSMInterface):
    """PKCS#11 compliant HSM implementation."""
    
    def __init__(self, library_path: str):
        """
        Initialize PKCS#11 HSM.
        
        Args:
            library_path: Path to PKCS#11 library
        """
        self.library_path = library_path
        self.session = None
        self.slot = None
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Import PyKCS11 if available
        try:
            import PyKCS11
            self.pkcs11 = PyKCS11.PyKCS11Lib()
            self.pkcs11.load(library_path)
        except ImportError:
            logger.warning("PyKCS11 not installed, HSM functionality limited")
            self.pkcs11 = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize PKCS#11 connection."""
        if not self.pkcs11:
            return False
        
        try:
            # Get slot
            slots = self.pkcs11.getSlotList(tokenPresent=True)
            if not slots:
                raise SecurityError("No HSM slots available")
            
            self.slot = slots[0]  # Use first available slot
            
            # Open session
            self.session = self.pkcs11.openSession(
                self.slot,
                PyKCS11.CKF_SERIAL_SESSION | PyKCS11.CKF_RW_SESSION
            )
            
            # Login if PIN provided
            if "pin" in config:
                self.session.login(config["pin"])
            
            self.logger.info("PKCS#11 HSM initialized", slot=self.slot)
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize PKCS#11", error=str(e))
            return False
    
    async def generate_key(
        self,
        key_type: KeyType,
        label: str,
        extractable: bool = False
    ) -> HSMKey:
        """Generate key using PKCS#11."""
        if not self.session:
            raise SecurityError("HSM not initialized")
        
        try:
            # Map key type to PKCS#11 mechanism
            mechanism = self._get_mechanism(key_type)
            
            # Generate key pair or symmetric key
            if key_type in [KeyType.RSA_2048, KeyType.RSA_4096, KeyType.ECDSA_P256]:
                # Generate key pair
                public_key, private_key = await asyncio.to_thread(
                    self._generate_key_pair,
                    mechanism,
                    label,
                    extractable
                )
                
                return HSMKey(
                    key_id=f"hsm_{label}_{os.urandom(8).hex()}",
                    key_type=key_type,
                    label=label,
                    created_at=datetime.utcnow(),
                    attributes={"extractable": extractable},
                    handle=private_key
                )
            else:
                # Generate symmetric key
                key_handle = await asyncio.to_thread(
                    self._generate_symmetric_key,
                    mechanism,
                    label,
                    extractable
                )
                
                return HSMKey(
                    key_id=f"hsm_{label}_{os.urandom(8).hex()}",
                    key_type=key_type,
                    label=label,
                    created_at=datetime.utcnow(),
                    attributes={"extractable": extractable},
                    handle=key_handle
                )
                
        except Exception as e:
            raise SecurityError(f"Failed to generate key: {str(e)}")
    
    async def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt using PKCS#11."""
        if not self.session:
            raise SecurityError("HSM not initialized")
        
        # Implementation would use PKCS#11 encryption
        # This is a placeholder
        return b"encrypted_" + plaintext
    
    async def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt using PKCS#11."""
        if not self.session:
            raise SecurityError("HSM not initialized")
        
        # Implementation would use PKCS#11 decryption
        # This is a placeholder
        if ciphertext.startswith(b"encrypted_"):
            return ciphertext[10:]
        return ciphertext
    
    async def sign(self, key_id: str, data: bytes) -> bytes:
        """Sign using PKCS#11."""
        if not self.session:
            raise SecurityError("HSM not initialized")
        
        # Implementation would use PKCS#11 signing
        # This is a placeholder
        import hashlib
        return hashlib.sha256(data).digest()
    
    async def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """Verify using PKCS#11."""
        if not self.session:
            raise SecurityError("HSM not initialized")
        
        # Implementation would use PKCS#11 verification
        # This is a placeholder
        import hashlib
        return signature == hashlib.sha256(data).digest()
    
    async def import_key(
        self,
        key_material: bytes,
        key_type: KeyType,
        label: str
    ) -> HSMKey:
        """Import key using PKCS#11."""
        # Implementation would import key material
        return HSMKey(
            key_id=f"imported_{label}_{os.urandom(8).hex()}",
            key_type=key_type,
            label=label,
            created_at=datetime.utcnow(),
            attributes={"imported": True}
        )
    
    async def delete_key(self, key_id: str) -> bool:
        """Delete key using PKCS#11."""
        # Implementation would delete key from HSM
        return True
    
    async def list_keys(self) -> List[HSMKey]:
        """List keys using PKCS#11."""
        # Implementation would list all keys
        return []
    
    async def health_check(self) -> bool:
        """Check PKCS#11 connection health."""
        return self.session is not None
    
    def _get_mechanism(self, key_type: KeyType):
        """Map key type to PKCS#11 mechanism."""
        # This would map to actual PKCS#11 mechanisms
        return None
    
    def _generate_key_pair(self, mechanism, label, extractable):
        """Generate asymmetric key pair."""
        # Implementation would use PKCS#11 to generate key pair
        return (None, None)
    
    def _generate_symmetric_key(self, mechanism, label, extractable):
        """Generate symmetric key."""
        # Implementation would use PKCS#11 to generate symmetric key
        return None


class HSMSimulator(HSMInterface):
    """HSM simulator for testing without hardware."""
    
    def __init__(self):
        """Initialize HSM simulator."""
        self.keys: Dict[str, HSMKey] = {}
        self.key_material: Dict[str, bytes] = {}
        self.initialized = False
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize simulator."""
        self.initialized = True
        self.logger.info("HSM simulator initialized")
        return True
    
    async def generate_key(
        self,
        key_type: KeyType,
        label: str,
        extractable: bool = False
    ) -> HSMKey:
        """Generate simulated key."""
        key_id = f"sim_{label}_{os.urandom(8).hex()}"
        
        # Generate random key material
        if key_type == KeyType.AES_256:
            key_material = os.urandom(32)
        elif key_type in [KeyType.RSA_2048, KeyType.RSA_4096]:
            # Simulate RSA key generation
            key_material = os.urandom(256 if key_type == KeyType.RSA_2048 else 512)
        else:
            key_material = os.urandom(64)
        
        key = HSMKey(
            key_id=key_id,
            key_type=key_type,
            label=label,
            created_at=datetime.utcnow(),
            attributes={"extractable": extractable, "simulated": True}
        )
        
        self.keys[key_id] = key
        self.key_material[key_id] = key_material
        
        self.logger.info("Generated simulated key", key_id=key_id, type=key_type.value)
        return key
    
    async def import_key(
        self,
        key_material: bytes,
        key_type: KeyType,
        label: str
    ) -> HSMKey:
        """Import key into simulator."""
        key_id = f"imported_{label}_{os.urandom(8).hex()}"
        
        key = HSMKey(
            key_id=key_id,
            key_type=key_type,
            label=label,
            created_at=datetime.utcnow(),
            attributes={"imported": True, "simulated": True}
        )
        
        self.keys[key_id] = key
        self.key_material[key_id] = key_material
        
        return key
    
    async def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Simulate encryption."""
        if key_id not in self.keys:
            raise SecurityError(f"Key {key_id} not found")
        
        # Simple XOR encryption for simulation
        key_material = self.key_material[key_id]
        encrypted = bytes(a ^ b for a, b in zip(plaintext, key_material * (len(plaintext) // len(key_material) + 1)))
        return encrypted
    
    async def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Simulate decryption."""
        # XOR is reversible
        return await self.encrypt(key_id, ciphertext)
    
    async def sign(self, key_id: str, data: bytes) -> bytes:
        """Simulate signing."""
        if key_id not in self.keys:
            raise SecurityError(f"Key {key_id} not found")
        
        import hashlib
        import hmac
        
        key_material = self.key_material[key_id]
        signature = hmac.new(key_material, data, hashlib.sha256).digest()
        return signature
    
    async def verify(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """Simulate verification."""
        expected_signature = await self.sign(key_id, data)
        return signature == expected_signature
    
    async def delete_key(self, key_id: str) -> bool:
        """Delete key from simulator."""
        if key_id in self.keys:
            del self.keys[key_id]
            del self.key_material[key_id]
            return True
        return False
    
    async def list_keys(self) -> List[HSMKey]:
        """List simulated keys."""
        return list(self.keys.values())
    
    async def health_check(self) -> bool:
        """Check simulator health."""
        return self.initialized


class HSMManager:
    """Manager for HSM operations with fallback support."""
    
    def __init__(
        self,
        hsm_type: HSMType = HSMType.SIMULATOR,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HSM manager.
        
        Args:
            hsm_type: Type of HSM to use
            config: HSM configuration
        """
        self.hsm_type = hsm_type
        self.config = config or {}
        self.hsm: Optional[HSMInterface] = None
        self.logger = structlog.get_logger(__name__)
        
        self._initialize_hsm()
    
    def _initialize_hsm(self):
        """Initialize the appropriate HSM implementation."""
        if self.hsm_type == HSMType.SIMULATOR:
            self.hsm = HSMSimulator()
        elif self.hsm_type == HSMType.SOFTHSM:
            library_path = self.config.get("library_path", "/usr/lib/softhsm/libsofthsm2.so")
            self.hsm = PKCS11HSM(library_path)
        elif self.hsm_type == HSMType.YUBIHSM:
            library_path = self.config.get("library_path", "/usr/lib/yubihsm_pkcs11.so")
            self.hsm = PKCS11HSM(library_path)
        else:
            self.logger.warning(f"HSM type {self.hsm_type} not implemented, using simulator")
            self.hsm = HSMSimulator()
    
    async def initialize(self) -> bool:
        """Initialize HSM connection."""
        if not self.hsm:
            return False
        
        success = await self.hsm.initialize(self.config)
        if success:
            self.logger.info("HSM manager initialized", type=self.hsm_type.value)
        else:
            self.logger.error("Failed to initialize HSM", type=self.hsm_type.value)
        
        return success
    
    async def create_master_key(self, label: str = "genesis_master") -> str:
        """Create master encryption key in HSM."""
        if not self.hsm:
            raise SecurityError("HSM not initialized")
        
        key = await self.hsm.generate_key(
            key_type=KeyType.AES_256,
            label=label,
            extractable=False
        )
        
        self.logger.info("Master key created in HSM", key_id=key.key_id)
        return key.key_id
    
    async def encrypt_with_master(self, key_id: str, data: bytes) -> bytes:
        """Encrypt data with master key."""
        if not self.hsm:
            raise SecurityError("HSM not initialized")
        
        return await self.hsm.encrypt(key_id, data)
    
    async def decrypt_with_master(self, key_id: str, data: bytes) -> bytes:
        """Decrypt data with master key."""
        if not self.hsm:
            raise SecurityError("HSM not initialized")
        
        return await self.hsm.decrypt(key_id, data)
    
    def get_status(self) -> Dict[str, Any]:
        """Get HSM status."""
        return {
            "type": self.hsm_type.value,
            "initialized": self.hsm is not None,
            "health": asyncio.run(self.hsm.health_check()) if self.hsm else False
        }