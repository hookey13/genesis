"""Certificate Management System for Project GENESIS.

Provides automated Let's Encrypt certificate renewal and management.
"""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from genesis.utils.logger import LoggerType, get_logger


class CertificateConfig(BaseModel):
    """Certificate management configuration."""

    domain: str = Field(description="Domain name for certificate")
    email: str = Field(description="Email for Let's Encrypt notifications")
    cert_dir: Path = Field(Path("/etc/letsencrypt"), description="Certificate directory")
    renewal_days_before_expiry: int = Field(30, description="Days before expiry to renew")
    enable_auto_renewal: bool = Field(True, description="Enable automatic renewal")
    reload_services: list[str] = Field(
        default_factory=lambda: ["nginx", "supervisor"],
        description="Services to reload after renewal"
    )
    use_staging: bool = Field(False, description="Use Let's Encrypt staging server")
    fallback_to_self_signed: bool = Field(True, description="Create self-signed if renewal fails")


class CertManager:
    """Manages SSL/TLS certificates with Let's Encrypt integration."""

    def __init__(self, config: CertificateConfig):
        self.config = config
        self.logger = get_logger(__name__, LoggerType.SYSTEM)

    async def check_certificate_expiry(self) -> datetime | None:
        """Check when the current certificate expires."""
        cert_path = self.config.cert_dir / "live" / self.config.domain / "cert.pem"

        if not cert_path.exists():
            self.logger.warning("certificate_not_found", path=str(cert_path))
            return None

        try:
            # Use openssl to check certificate expiry
            result = subprocess.run(
                ["openssl", "x509", "-enddate", "-noout", "-in", str(cert_path)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Parse expiry date from output
                expiry_str = result.stdout.strip().replace("notAfter=", "")
                expiry = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")

                days_until_expiry = (expiry - datetime.utcnow()).days

                self.logger.info(
                    "certificate_expiry_checked",
                    domain=self.config.domain,
                    expiry=expiry.isoformat(),
                    days_until_expiry=days_until_expiry
                )

                # Send warnings at 30, 7, and 1 day marks
                if days_until_expiry <= 1:
                    self.logger.critical("certificate_expires_tomorrow", domain=self.config.domain)
                elif days_until_expiry <= 7:
                    self.logger.error("certificate_expires_soon", domain=self.config.domain, days=days_until_expiry)
                elif days_until_expiry <= 30:
                    self.logger.warning("certificate_expiring", domain=self.config.domain, days=days_until_expiry)

                return expiry

        except Exception as e:
            self.logger.error("failed_to_check_expiry", error=str(e))
            return None

    async def renew_certificate(self) -> bool:
        """Renew certificate using certbot."""
        try:
            self.logger.info("certificate_renewal_started", domain=self.config.domain)

            # Build certbot command
            cmd = [
                "certbot", "renew",
                "--non-interactive",
                "--agree-tos",
                "--email", self.config.email,
            ]

            if self.config.use_staging:
                cmd.append("--staging")

            # Run certbot
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(
                    "certificate_renewed",
                    domain=self.config.domain,
                    output=result.stdout
                )

                # Reload services
                await self._reload_services()
                return True
            else:
                self.logger.error(
                    "certificate_renewal_failed",
                    domain=self.config.domain,
                    error=result.stderr
                )

                # Try fallback to self-signed
                if self.config.fallback_to_self_signed:
                    return await self._create_self_signed_certificate()

                return False

        except Exception as e:
            self.logger.error("renewal_exception", error=str(e))
            return False

    async def _create_self_signed_certificate(self) -> bool:
        """Create self-signed certificate as fallback."""
        try:
            self.logger.warning("creating_self_signed_certificate", domain=self.config.domain)

            cert_dir = self.config.cert_dir / "self-signed"
            cert_dir.mkdir(parents=True, exist_ok=True)

            key_path = cert_dir / f"{self.config.domain}.key"
            cert_path = cert_dir / f"{self.config.domain}.crt"

            # Generate self-signed certificate
            cmd = [
                "openssl", "req", "-x509", "-nodes",
                "-days", "365",
                "-newkey", "rsa:2048",
                "-keyout", str(key_path),
                "-out", str(cert_path),
                "-subj", f"/CN={self.config.domain}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("self_signed_certificate_created", domain=self.config.domain)
                return True
            else:
                self.logger.error("self_signed_creation_failed", error=result.stderr)
                return False

        except Exception as e:
            self.logger.error("self_signed_exception", error=str(e))
            return False

    async def _reload_services(self) -> None:
        """Reload services after certificate renewal."""
        for service in self.config.reload_services:
            try:
                result = subprocess.run(
                    ["systemctl", "reload", service],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    self.logger.info("service_reloaded", service=service)
                else:
                    self.logger.error("service_reload_failed", service=service, error=result.stderr)

            except Exception as e:
                self.logger.error("reload_exception", service=service, error=str(e))

    async def setup_auto_renewal(self) -> None:
        """Setup automatic certificate renewal."""
        if not self.config.enable_auto_renewal:
            return

        async def renewal_check():
            while True:
                try:
                    # Check every day
                    await asyncio.sleep(86400)

                    expiry = await self.check_certificate_expiry()
                    if expiry:
                        days_until = (expiry - datetime.utcnow()).days
                        if days_until <= self.config.renewal_days_before_expiry:
                            await self.renew_certificate()

                except Exception as e:
                    self.logger.error("auto_renewal_error", error=str(e))

        asyncio.create_task(renewal_check())
