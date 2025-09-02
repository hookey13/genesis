"""
Time-based One-Time Password (TOTP) management for 2FA implementation.
Provides RFC 6238 compliant TOTP generation and verification.
"""

import base64
import hashlib
import io
import secrets
import time
from datetime import UTC, datetime
from typing import Any

import pyotp
import qrcode
import structlog

logger = structlog.get_logger(__name__)


class TOTPManager:
    """Time-based One-Time Password management for two-factor authentication."""

    def __init__(self, vault_manager=None, issuer_name: str = "Genesis Trading"):
        """
        Initialize TOTP manager.
        
        Args:
            vault_manager: Vault client for secret storage
            issuer_name: Name to display in authenticator apps
        """
        self.vault = vault_manager
        self.issuer_name = issuer_name
        self.window = 1  # Allow 1 time step tolerance (30 seconds)
        self.backup_codes_count = 10
        self.code_length = 6
        self.time_step = 30  # Standard TOTP time step in seconds

    def generate_secret(self) -> str:
        """
        Generate cryptographically secure TOTP secret.
        
        Returns:
            Base32 encoded secret string
        """
        return pyotp.random_base32()

    def generate_provisioning_uri(self, username: str, secret: str) -> str:
        """
        Generate provisioning URI for QR code scanning.
        
        Args:
            username: User identifier for authenticator app
            secret: TOTP secret
            
        Returns:
            otpauth:// URI for authenticator apps
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=username,
            issuer_name=self.issuer_name
        )

    def generate_qr_code(self, provisioning_uri: str) -> str:
        """
        Generate QR code as base64 encoded PNG image.
        
        Args:
            provisioning_uri: otpauth:// URI
            
        Returns:
            Base64 encoded PNG image string
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=5
        )
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def verify_totp_code(self, secret: str, code: str, user_id: str | None = None) -> bool:
        """
        Verify TOTP code with window tolerance for clock skew.
        
        Args:
            secret: TOTP secret
            code: 6-digit code from authenticator
            user_id: Optional user ID for logging
            
        Returns:
            True if code is valid, False otherwise
        """
        if not code or not code.isdigit() or len(code) != self.code_length:
            logger.warning(
                "Invalid TOTP code format",
                user_id=user_id,
                code_length=len(code) if code else 0
            )
            return False

        totp = pyotp.TOTP(secret)
        is_valid = totp.verify(code, valid_window=self.window)

        if is_valid:
            logger.info(
                "TOTP verification successful",
                user_id=user_id
            )
        else:
            logger.warning(
                "TOTP verification failed",
                user_id=user_id
            )

        return is_valid

    def generate_backup_codes(self) -> list[str]:
        """
        Generate backup recovery codes for account access.
        
        Returns:
            List of formatted backup codes
        """
        codes = []
        for _ in range(self.backup_codes_count):
            # Generate 8-character alphanumeric codes
            code = secrets.token_hex(4).upper()
            # Format as XXXX-XXXX for readability
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes

    def hash_backup_code(self, code: str) -> str:
        """
        Hash backup code for secure storage.
        
        Args:
            code: Plain backup code
            
        Returns:
            SHA256 hash of the code
        """
        # Remove formatting if present
        clean_code = code.replace("-", "").upper()
        return hashlib.sha256(clean_code.encode()).hexdigest()

    def verify_backup_code(self, code: str, hashed_codes: list[str]) -> str | None:
        """
        Verify backup code against stored hashes.
        
        Args:
            code: Provided backup code
            hashed_codes: List of hashed backup codes
            
        Returns:
            Matched hash if valid, None otherwise
        """
        clean_code = code.replace("-", "").upper()
        code_hash = hashlib.sha256(clean_code.encode()).hexdigest()

        if code_hash in hashed_codes:
            return code_hash
        return None

    async def setup_2fa(self, user_id: str, username: str) -> dict[str, Any]:
        """
        Setup 2FA for user account.
        
        Args:
            user_id: Unique user identifier
            username: Username for display
            
        Returns:
            Dictionary with setup details including QR code
        """
        secret = self.generate_secret()
        backup_codes = self.generate_backup_codes()

        # Hash backup codes for storage
        hashed_backup_codes = [self.hash_backup_code(code) for code in backup_codes]

        # Store in Vault if available
        if self.vault:
            try:
                await self.vault.store_secret(
                    f"genesis-secrets/2fa/{user_id}",
                    {
                        'secret': secret,
                        'backup_codes': hashed_backup_codes,
                        'enabled': False,
                        'setup_at': datetime.now(UTC).isoformat(),
                        'last_used': None,
                        'failure_count': 0
                    }
                )
                logger.info(
                    "2FA secrets stored in Vault",
                    user_id=user_id
                )
            except Exception as e:
                logger.error(
                    "Failed to store 2FA secrets in Vault",
                    user_id=user_id,
                    error=str(e)
                )
                raise

        # Generate QR code
        provisioning_uri = self.generate_provisioning_uri(username, secret)
        qr_code = self.generate_qr_code(provisioning_uri)

        logger.info(
            "2FA setup initiated",
            user_id=user_id,
            username=username
        )

        return {
            'secret': secret,
            'qr_code': qr_code,
            'provisioning_uri': provisioning_uri,
            'backup_codes': backup_codes,  # Return plain codes to user
            'setup_complete': False
        }

    async def enable_2fa(self, user_id: str, verification_code: str) -> bool:
        """
        Enable 2FA after successful verification.
        
        Args:
            user_id: User identifier
            verification_code: Code from authenticator app
            
        Returns:
            True if enabled successfully
        """
        if not self.vault:
            logger.error("Vault not configured for 2FA enablement")
            return False

        try:
            # Retrieve secret from Vault
            secret_data = await self.vault.get_secret(f"genesis-secrets/2fa/{user_id}")
            if not secret_data:
                logger.error("2FA secret not found", user_id=user_id)
                return False

            secret = secret_data.get('secret')
            if not secret:
                logger.error("Invalid 2FA secret data", user_id=user_id)
                return False

            # Verify the code
            if not self.verify_totp_code(secret, verification_code, user_id):
                return False

            # Update Vault to enable 2FA
            secret_data['enabled'] = True
            secret_data['enabled_at'] = datetime.now(UTC).isoformat()

            await self.vault.store_secret(
                f"genesis-secrets/2fa/{user_id}",
                secret_data
            )

            logger.info(
                "2FA enabled successfully",
                user_id=user_id
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to enable 2FA",
                user_id=user_id,
                error=str(e)
            )
            return False

    async def disable_2fa(self, user_id: str) -> bool:
        """
        Disable 2FA for user account.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if disabled successfully
        """
        if not self.vault:
            logger.error("Vault not configured for 2FA management")
            return False

        try:
            # Delete 2FA secret from Vault
            await self.vault.delete_secret(f"genesis-secrets/2fa/{user_id}")

            logger.info(
                "2FA disabled successfully",
                user_id=user_id
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to disable 2FA",
                user_id=user_id,
                error=str(e)
            )
            return False

    async def validate_2fa(self, user_id: str, code: str) -> tuple[bool, str]:
        """
        Validate 2FA code (TOTP or backup).
        
        Args:
            user_id: User identifier
            code: TOTP or backup code
            
        Returns:
            Tuple of (is_valid, code_type)
        """
        if not self.vault:
            logger.error("Vault not configured for 2FA validation")
            return False, "error"

        try:
            # Retrieve secret from Vault
            secret_data = await self.vault.get_secret(f"genesis-secrets/2fa/{user_id}")
            if not secret_data or not secret_data.get('enabled'):
                return False, "not_enabled"

            # Try TOTP verification first
            secret = secret_data.get('secret')
            if secret and self.verify_totp_code(secret, code, user_id):
                # Update last used timestamp
                secret_data['last_used'] = datetime.now(UTC).isoformat()
                secret_data['failure_count'] = 0
                await self.vault.store_secret(
                    f"genesis-secrets/2fa/{user_id}",
                    secret_data
                )
                return True, "totp"

            # Try backup code verification
            backup_codes = secret_data.get('backup_codes', [])
            matched_hash = self.verify_backup_code(code, backup_codes)
            if matched_hash:
                # Remove used backup code
                backup_codes.remove(matched_hash)
                secret_data['backup_codes'] = backup_codes
                secret_data['last_used'] = datetime.now(UTC).isoformat()
                secret_data['failure_count'] = 0

                await self.vault.store_secret(
                    f"genesis-secrets/2fa/{user_id}",
                    secret_data
                )

                logger.info(
                    "Backup code used",
                    user_id=user_id,
                    remaining_codes=len(backup_codes)
                )
                return True, "backup"

            # Increment failure count
            secret_data['failure_count'] = secret_data.get('failure_count', 0) + 1
            await self.vault.store_secret(
                f"genesis-secrets/2fa/{user_id}",
                secret_data
            )

            return False, "invalid"

        except Exception as e:
            logger.error(
                "2FA validation error",
                user_id=user_id,
                error=str(e)
            )
            return False, "error"

    async def regenerate_backup_codes(self, user_id: str) -> list[str] | None:
        """
        Generate new set of backup codes.
        
        Args:
            user_id: User identifier
            
        Returns:
            New backup codes or None on error
        """
        if not self.vault:
            logger.error("Vault not configured for backup code regeneration")
            return None

        try:
            # Retrieve existing secret data
            secret_data = await self.vault.get_secret(f"genesis-secrets/2fa/{user_id}")
            if not secret_data:
                logger.error("2FA not setup for user", user_id=user_id)
                return None

            # Generate new backup codes
            new_codes = self.generate_backup_codes()
            hashed_codes = [self.hash_backup_code(code) for code in new_codes]

            # Update Vault
            secret_data['backup_codes'] = hashed_codes
            secret_data['backup_codes_regenerated_at'] = datetime.now(UTC).isoformat()

            await self.vault.store_secret(
                f"genesis-secrets/2fa/{user_id}",
                secret_data
            )

            logger.info(
                "Backup codes regenerated",
                user_id=user_id
            )

            return new_codes

        except Exception as e:
            logger.error(
                "Failed to regenerate backup codes",
                user_id=user_id,
                error=str(e)
            )
            return None

    def get_time_remaining(self) -> int:
        """
        Get seconds remaining in current TOTP window.
        
        Returns:
            Seconds until next code
        """
        return self.time_step - (int(time.time()) % self.time_step)
