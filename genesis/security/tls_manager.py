"""TLS certificate management for secure internal communication.

Handles certificate generation, validation, rotation, and storage
for TLS 1.3 encrypted communication between internal services.
"""

import os
import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from genesis.config.settings import settings
from genesis.security.vault_client import VaultClient

logger = structlog.get_logger(__name__)


class TLSManager:
    """Manages TLS certificates for internal communication."""

    # TLS configuration
    TLS_VERSION = ssl.TLSVersion.TLSv1_3
    CIPHER_SUITES = [
        'TLS_AES_256_GCM_SHA384',
        'TLS_CHACHA20_POLY1305_SHA256',
        'TLS_AES_128_GCM_SHA256'
    ]

    def __init__(
        self,
        vault_client: VaultClient | None = None,
        cert_dir: str = ".genesis/certs",
        validity_days: int = 90,
        auto_renewal_days: int = 30
    ):
        """Initialize TLS manager.
        
        Args:
            vault_client: Vault client for certificate storage
            cert_dir: Directory for certificate files
            validity_days: Certificate validity period
            auto_renewal_days: Days before expiry to auto-renew
        """
        self.vault_client = vault_client or settings.get_vault_client()
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.validity_days = validity_days
        self.auto_renewal_days = auto_renewal_days

        # Certificate paths
        self.ca_cert_path = self.cert_dir / "ca.crt"
        self.ca_key_path = self.cert_dir / "ca.key"
        self.server_cert_path = self.cert_dir / "server.crt"
        self.server_key_path = self.cert_dir / "server.key"
        self.client_cert_path = self.cert_dir / "client.crt"
        self.client_key_path = self.cert_dir / "client.key"

    def generate_ca_certificate(self) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate a Certificate Authority certificate.
        
        Returns:
            Tuple of (CA certificate, CA private key)
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )

        # Certificate details
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Project GENESIS"),
            x509.NameAttribute(NameOID.COMMON_NAME, "GENESIS Internal CA"),
        ])

        # Generate certificate
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365 * 5)  # 5 years for CA
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        ).sign(private_key, hashes.SHA256(), backend=default_backend())

        # Save CA certificate and key
        self._save_certificate(cert, self.ca_cert_path)
        self._save_private_key(private_key, self.ca_key_path)

        # Store in Vault
        self._store_in_vault("ca", cert, private_key)

        logger.info("Generated CA certificate",
                   subject=subject.rfc4514_string(),
                   valid_until=cert.not_valid_after)

        return cert, private_key

    def generate_server_certificate(
        self,
        hostname: str = "localhost",
        ip_addresses: list | None = None
    ) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate a server certificate signed by CA.
        
        Args:
            hostname: Server hostname
            ip_addresses: List of IP addresses for the server
            
        Returns:
            Tuple of (server certificate, server private key)
        """
        # Load CA certificate and key
        ca_cert, ca_key = self._load_ca()

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Certificate details
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Project GENESIS"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])

        # Build certificate
        builder = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            ca_cert.issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=self.validity_days)
        )

        # Add SAN extension
        san_list = [x509.DNSName(hostname)]
        if ip_addresses:
            for ip in ip_addresses:
                san_list.append(x509.IPAddress(ipaddress.ip_address(ip)))

        builder = builder.add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False
        )

        # Add key usage extensions
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH
            ]),
            critical=True
        )

        # Sign with CA key
        cert = builder.sign(ca_key, hashes.SHA256(), backend=default_backend())

        # Save server certificate and key
        self._save_certificate(cert, self.server_cert_path)
        self._save_private_key(private_key, self.server_key_path)

        # Store in Vault
        self._store_in_vault("server", cert, private_key)

        logger.info("Generated server certificate",
                   hostname=hostname,
                   valid_until=cert.not_valid_after)

        return cert, private_key

    def generate_client_certificate(
        self,
        client_name: str = "genesis-client"
    ) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Generate a client certificate for mutual TLS.
        
        Args:
            client_name: Client identifier
            
        Returns:
            Tuple of (client certificate, client private key)
        """
        # Load CA certificate and key
        ca_cert, ca_key = self._load_ca()

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Certificate details
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Project GENESIS"),
            x509.NameAttribute(NameOID.COMMON_NAME, client_name),
        ])

        # Build certificate
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            ca_cert.issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=self.validity_days)
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH
            ]),
            critical=True
        ).sign(ca_key, hashes.SHA256(), backend=default_backend())

        # Save client certificate and key
        self._save_certificate(cert, self.client_cert_path)
        self._save_private_key(private_key, self.client_key_path)

        # Store in Vault
        self._store_in_vault("client", cert, private_key)

        logger.info("Generated client certificate",
                   client_name=client_name,
                   valid_until=cert.not_valid_after)

        return cert, private_key

    def create_ssl_context(
        self,
        purpose: ssl.Purpose = ssl.Purpose.CLIENT_AUTH,
        verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED,
        check_hostname: bool = True
    ) -> ssl.SSLContext:
        """Create an SSL context for TLS communication.
        
        Args:
            purpose: SSL purpose (CLIENT_AUTH or SERVER_AUTH)
            verify_mode: Certificate verification mode
            check_hostname: Whether to check hostname
            
        Returns:
            Configured SSL context
        """
        # Create context with TLS 1.3
        context = ssl.create_default_context(purpose)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Set cipher suites (TLS 1.3 uses different configuration)
        # context.set_ciphers(':'.join(self.CIPHER_SUITES))

        # Load CA certificate
        context.load_verify_locations(str(self.ca_cert_path))

        # Configure verification
        context.verify_mode = verify_mode
        context.check_hostname = check_hostname

        # Load appropriate certificate
        if purpose == ssl.Purpose.CLIENT_AUTH:
            # Server context
            context.load_cert_chain(
                str(self.server_cert_path),
                str(self.server_key_path)
            )
        else:
            # Client context
            if self.client_cert_path.exists():
                context.load_cert_chain(
                    str(self.client_cert_path),
                    str(self.client_key_path)
                )

        logger.info("Created SSL context",
                   purpose=purpose.name,
                   verify_mode=verify_mode.name)

        return context

    def verify_certificate(self, cert_path: Path) -> dict[str, Any]:
        """Verify a certificate.
        
        Args:
            cert_path: Path to certificate file
            
        Returns:
            Certificate information and validation status
        """
        try:
            # Load certificate
            with open(cert_path, 'rb') as f:
                cert_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data, default_backend())

            # Check expiration
            now = datetime.utcnow()
            is_expired = now > cert.not_valid_after
            is_not_yet_valid = now < cert.not_valid_before
            days_until_expiry = (cert.not_valid_after - now).days

            # Get subject and issuer
            subject = cert.subject.rfc4514_string()
            issuer = cert.issuer.rfc4514_string()

            # Check if self-signed
            is_self_signed = subject == issuer

            return {
                "valid": not is_expired and not is_not_yet_valid,
                "expired": is_expired,
                "not_yet_valid": is_not_yet_valid,
                "days_until_expiry": days_until_expiry,
                "subject": subject,
                "issuer": issuer,
                "serial_number": str(cert.serial_number),
                "not_valid_before": cert.not_valid_before.isoformat(),
                "not_valid_after": cert.not_valid_after.isoformat(),
                "is_self_signed": is_self_signed,
                "needs_renewal": days_until_expiry <= self.auto_renewal_days
            }

        except Exception as e:
            logger.error("Failed to verify certificate",
                        path=str(cert_path),
                        error=str(e))
            return {
                "valid": False,
                "error": str(e)
            }

    def rotate_certificates(self) -> bool:
        """Rotate all certificates that are near expiry.
        
        Returns:
            True if any certificates were rotated
        """
        rotated = False

        # Check server certificate
        if self.server_cert_path.exists():
            info = self.verify_certificate(self.server_cert_path)
            if info.get("needs_renewal", False):
                logger.info("Rotating server certificate",
                           days_until_expiry=info.get("days_until_expiry"))
                self.generate_server_certificate()
                rotated = True

        # Check client certificate
        if self.client_cert_path.exists():
            info = self.verify_certificate(self.client_cert_path)
            if info.get("needs_renewal", False):
                logger.info("Rotating client certificate",
                           days_until_expiry=info.get("days_until_expiry"))
                self.generate_client_certificate()
                rotated = True

        return rotated

    def _load_ca(self) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
        """Load CA certificate and key.
        
        Returns:
            Tuple of (CA certificate, CA private key)
        """
        # Try to load from Vault first
        cert_data = self.vault_client.get_tls_certificates()

        if cert_data and "ca_cert" in cert_data and "ca_key" in cert_data:
            cert = x509.load_pem_x509_certificate(
                cert_data["ca_cert"].encode(),
                default_backend()
            )
            key = serialization.load_pem_private_key(
                cert_data["ca_key"].encode(),
                password=None,
                backend=default_backend()
            )
            return cert, key

        # Load from files
        if not self.ca_cert_path.exists() or not self.ca_key_path.exists():
            # Generate new CA
            return self.generate_ca_certificate()

        with open(self.ca_cert_path, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        with open(self.ca_key_path, 'rb') as f:
            key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )

        return cert, key

    def _save_certificate(self, cert: x509.Certificate, path: Path):
        """Save certificate to file.
        
        Args:
            cert: Certificate to save
            path: File path
        """
        with open(path, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Set restrictive permissions
        os.chmod(path, 0o644)

    def _save_private_key(self, key: rsa.RSAPrivateKey, path: Path):
        """Save private key to file.
        
        Args:
            key: Private key to save
            path: File path
        """
        with open(path, 'wb') as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Set restrictive permissions
        os.chmod(path, 0o600)

    def _store_in_vault(
        self,
        cert_type: str,
        cert: x509.Certificate,
        key: rsa.RSAPrivateKey
    ):
        """Store certificate and key in Vault.
        
        Args:
            cert_type: Type of certificate (ca, server, client)
            cert: Certificate to store
            key: Private key to store
        """
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()

        data = {
            f"{cert_type}_cert": cert_pem,
            f"{cert_type}_key": key_pem,
            "created_at": datetime.utcnow().isoformat(),
            "valid_until": cert.not_valid_after.isoformat()
        }

        success = self.vault_client.store_secret(
            VaultClient.TLS_CERTIFICATES_PATH,
            data
        )

        if success:
            logger.info("Stored certificate in Vault", cert_type=cert_type)
        else:
            logger.warning("Failed to store certificate in Vault",
                         cert_type=cert_type)


# Import ipaddress for IP address handling
import ipaddress
