"""Encryption and data protection validation."""

import re
from pathlib import Path
from typing import Any

import structlog

from genesis.validation.base import Validator

logger = structlog.get_logger(__name__)


class EncryptionValidator(Validator):
    """Validates encryption for data at rest and in transit."""

    ENCRYPTION_STANDARDS = {
        "tls_version": "1.3",  # Minimum TLS version
        "cipher_suites": [
            "TLS_AES_256_GCM_SHA384",
            "TLS_AES_128_GCM_SHA256",
            "TLS_CHACHA20_POLY1305_SHA256",
        ],
        "key_sizes": {
            "rsa": 2048,  # Minimum RSA key size
            "aes": 256,   # AES key size
            "ecdsa": 256, # ECDSA key size
        },
        "hash_algorithms": ["sha256", "sha384", "sha512"],
        "kdf_iterations": 100000,  # Key derivation iterations
    }

    def __init__(self):
        """Initialize encryption validator."""
        super().__init__(
            validator_id="SEC-004",
            name="Encryption Validator",
            description="Validates encryption for data at rest and in transit"
        )
        self.encryption_issues = []
        self.validated_files = 0
        self.is_critical = True
        self.timeout_seconds = 90

    async def validate(self) -> dict[str, Any]:
        """Run comprehensive encryption validation.
        
        Returns:
            Encryption validation results
        """
        logger.info("Starting encryption validation")

        # Validate API key encryption
        api_key_results = await self._validate_api_key_encryption()

        # Validate TLS/SSL configuration
        tls_results = await self._validate_tls_configuration()

        # Validate data at rest encryption
        data_at_rest_results = await self._validate_data_at_rest()

        # Validate secure communication channels
        comm_channel_results = await self._validate_communication_channels()

        # Validate key management
        key_mgmt_results = await self._validate_key_management()

        # Validate password storage
        password_results = await self._validate_password_storage()

        # Calculate overall score
        all_checks = [
            api_key_results,
            tls_results,
            data_at_rest_results,
            comm_channel_results,
            key_mgmt_results,
            password_results,
        ]

        total_passed = sum(r["passed"] for r in all_checks)
        total_checks_count = sum(r["total_checks"] for r in all_checks)

        compliance_score = (total_passed / total_checks_count * 100) if total_checks_count > 0 else 0
        passed = compliance_score >= 90  # 90% encryption compliance required

        return {
            "passed": passed,
            "compliance_score": compliance_score,
            "summary": {
                "total_checks": total_checks_count,
                "passed_checks": total_passed,
                "failed_checks": total_checks_count - total_passed,
                "api_keys_encrypted": api_key_results["encrypted"],
                "tls_compliant": tls_results["compliant"],
                "data_at_rest_encrypted": data_at_rest_results["encrypted"],
                "secure_channels": comm_channel_results["secure"],
            },
            "details": {
                "api_key_encryption": api_key_results,
                "tls_configuration": tls_results,
                "data_at_rest": data_at_rest_results,
                "communication_channels": comm_channel_results,
                "key_management": key_mgmt_results,
                "password_storage": password_results,
            },
            "issues": self.encryption_issues,
            "recommendations": self._generate_recommendations(all_checks),
        }

    async def _validate_api_key_encryption(self) -> dict[str, Any]:
        """Validate API key encryption in storage.
        
        Returns:
            API key encryption validation results
        """
        checks = {
            "vault_integration": False,
            "encrypted_storage": False,
            "secure_retrieval": False,
            "key_rotation": False,
            "access_control": False,
        }

        # Check for vault client
        vault_client = Path("genesis/security/vault_client.py")
        if vault_client.exists():
            checks["vault_integration"] = True
            content = vault_client.read_text()

            # Check for encryption methods
            checks["encrypted_storage"] = "encrypt" in content.lower()
            checks["secure_retrieval"] = "decrypt" in content.lower() or "get_secret" in content.lower()
            checks["key_rotation"] = "rotate" in content.lower()
            checks["access_control"] = "authenticate" in content.lower() or "authorize" in content.lower()

        # Check for alternative encryption
        if not checks["vault_integration"]:
            # Check for Fernet encryption
            encryption_files = list(Path("genesis").rglob("*encrypt*.py"))
            if encryption_files:
                for enc_file in encryption_files:
                    content = enc_file.read_text()
                    if "Fernet" in content or "cryptography" in content:
                        checks["encrypted_storage"] = True
                        break

        # Check environment variable usage
        env_usage = await self._check_env_var_usage()
        if env_usage["uses_env_vars"] and not env_usage["hardcoded_keys"]:
            checks["secure_retrieval"] = True

        passed = sum(checks.values())
        total = len(checks)

        return {
            "encrypted": passed >= 3,  # At least 3 of 5 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "env_var_usage": env_usage,
        }

    async def _check_env_var_usage(self) -> dict[str, Any]:
        """Check environment variable usage for secrets.
        
        Returns:
            Environment variable usage analysis
        """
        env_patterns = {
            "getenv": r"os\.getenv\([\"']([A-Z_]+)[\"']\)",
            "environ": r"os\.environ\[[\"']([A-Z_]+)[\"']\]",
            "config": r"config\.get\([\"']([A-Z_]+)[\"']\)",
        }

        hardcoded_patterns = {
            "api_key": r"api_key\s*=\s*[\"']([A-Za-z0-9]{20,})[\"']",
            "secret": r"secret\s*=\s*[\"']([A-Za-z0-9]{20,})[\"']",
        }

        uses_env_vars = False
        hardcoded_keys = []
        env_vars_found = set()

        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for environment variable usage
                for pattern_name, pattern in env_patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        uses_env_vars = True
                        env_vars_found.update(matches)

                # Check for hardcoded keys
                for pattern_name, pattern in hardcoded_patterns.items():
                    if re.search(pattern, content):
                        hardcoded_keys.append(str(py_file))

            except Exception as e:
                logger.error(f"Error checking file {py_file}: {e}")

        return {
            "uses_env_vars": uses_env_vars,
            "hardcoded_keys": hardcoded_keys,
            "env_vars_found": list(env_vars_found),
        }

    async def _validate_tls_configuration(self) -> dict[str, Any]:
        """Validate TLS/SSL configuration.
        
        Returns:
            TLS configuration validation results
        """
        checks = {
            "min_tls_version": False,
            "strong_ciphers": False,
            "certificate_validation": False,
            "hsts_enabled": False,
            "ssl_redirect": False,
        }

        # Check for TLS configuration files
        tls_configs = []
        for config_pattern in ["*.conf", "*.yaml", "*.yml", "*.toml"]:
            tls_configs.extend(Path(".").rglob(config_pattern))

        for config_file in tls_configs:
            try:
                content = config_file.read_text()

                # Check TLS version
                if "TLSv1.3" in content or "TLS1.3" in content or "tls_version: 1.3" in content:
                    checks["min_tls_version"] = True

                # Check cipher suites
                for cipher in self.ENCRYPTION_STANDARDS["cipher_suites"]:
                    if cipher in content:
                        checks["strong_ciphers"] = True
                        break

                # Check certificate validation
                if "verify_mode" in content or "ca_cert" in content or "verify_ssl" in content:
                    checks["certificate_validation"] = True

                # Check HSTS
                if "Strict-Transport-Security" in content or "hsts" in content.lower():
                    checks["hsts_enabled"] = True

                # Check SSL redirect
                if "ssl_redirect" in content or "force_ssl" in content or "https_only" in content:
                    checks["ssl_redirect"] = True

            except Exception as e:
                logger.error(f"Error checking config {config_file}: {e}")

        # Check Python code for TLS settings
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for SSL context creation
                if "ssl.create_default_context" in content:
                    checks["certificate_validation"] = True

                # Check for TLS version setting
                if "ssl.TLSVersion.TLSv1_3" in content or "PROTOCOL_TLS" in content:
                    checks["min_tls_version"] = True

            except Exception as e:
                logger.error(f"Error checking file {py_file}: {e}")

        passed = sum(checks.values())
        total = len(checks)

        return {
            "compliant": passed >= 3,  # At least 3 of 5 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "min_version": self.ENCRYPTION_STANDARDS["tls_version"],
        }

    async def _validate_data_at_rest(self) -> dict[str, Any]:
        """Validate data at rest encryption.
        
        Returns:
            Data at rest encryption validation results
        """
        checks = {
            "database_encryption": False,
            "file_encryption": False,
            "backup_encryption": False,
            "log_encryption": False,
            "temp_file_encryption": False,
        }

        # Check database encryption
        db_files = list(Path("genesis/data").rglob("*.py")) if Path("genesis/data").exists() else []
        for db_file in db_files:
            try:
                content = db_file.read_text()
                if "encrypt" in content.lower() or "cipher" in content.lower():
                    checks["database_encryption"] = True
                    break
            except Exception:
                pass

        # Check file encryption utilities
        if Path("genesis/security/encryption.py").exists() or any(Path("genesis").rglob("*crypt*.py")):
            checks["file_encryption"] = True

        # Check backup encryption
        backup_script = Path("scripts/backup.sh")
        if backup_script.exists():
            content = backup_script.read_text()
            if "gpg" in content or "openssl" in content or "encrypt" in content:
                checks["backup_encryption"] = True

        # Check log encryption (optional for most cases)
        log_config = Path("genesis/utils/logger.py")
        if log_config.exists():
            content = log_config.read_text()
            if "encrypt" in content.lower():
                checks["log_encryption"] = True

        # Check temp file handling
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()
                if "tempfile" in content and ("encrypt" in content or "secure" in content):
                    checks["temp_file_encryption"] = True
                    break
            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "encrypted": passed >= 3,  # At least 3 of 5 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
        }

    async def _validate_communication_channels(self) -> dict[str, Any]:
        """Validate secure communication channels.
        
        Returns:
            Communication channel validation results
        """
        checks = {
            "https_only": False,
            "websocket_secure": False,
            "api_encryption": False,
            "internal_comms_secure": False,
            "third_party_secure": False,
        }

        # Check for HTTPS enforcement
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for HTTPS URLs
                if re.search(r"https://", content):
                    checks["https_only"] = True

                # Check for secure websockets
                if "wss://" in content or ("websocket" in content.lower() and "ssl" in content):
                    checks["websocket_secure"] = True

                # Check API encryption
                if "api" in str(py_file).lower():
                    if "https" in content or "ssl" in content or "tls" in content:
                        checks["api_encryption"] = True

                # Check internal communications
                if ("grpc" in content and "credentials" in content) or ("rabbitmq" in content and "ssl" in content):
                    checks["internal_comms_secure"] = True

                # Check third-party integrations
                if "binance" in content.lower() or "exchange" in content.lower():
                    if "https" in content or "signature" in content:
                        checks["third_party_secure"] = True

            except Exception as e:
                logger.error(f"Error checking file {py_file}: {e}")

        passed = sum(checks.values())
        total = len(checks)

        return {
            "secure": passed >= 3,  # At least 3 of 5 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
        }

    async def _validate_key_management(self) -> dict[str, Any]:
        """Validate encryption key management.
        
        Returns:
            Key management validation results
        """
        checks = {
            "key_generation": False,
            "key_storage": False,
            "key_rotation": False,
            "key_derivation": False,
            "key_destruction": False,
        }

        # Check for key management implementation
        key_mgmt_files = [
            Path("genesis/security/key_management.py"),
            Path("genesis/security/kms.py"),
            Path("genesis/security/vault_client.py"),
        ]

        for key_file in key_mgmt_files:
            if key_file.exists():
                try:
                    content = key_file.read_text()

                    # Check key generation
                    if "generate_key" in content or "keygen" in content or "Fernet.generate_key" in content:
                        checks["key_generation"] = True

                    # Check key storage
                    if "store_key" in content or "save_key" in content or "vault" in content.lower():
                        checks["key_storage"] = True

                    # Check key rotation
                    if "rotate" in content or "key_rotation" in content:
                        checks["key_rotation"] = True

                    # Check key derivation
                    if "pbkdf" in content.lower() or "kdf" in content or "derive_key" in content:
                        checks["key_derivation"] = True

                    # Check key destruction
                    if "destroy_key" in content or "delete_key" in content or "wipe" in content:
                        checks["key_destruction"] = True

                except Exception as e:
                    logger.error(f"Error checking key file {key_file}: {e}")

        # Check for cryptography usage
        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for proper key derivation
                if "PBKDF2" in content or "scrypt" in content or "bcrypt" in content:
                    checks["key_derivation"] = True

                # Check for key generation
                if "os.urandom" in content or "secrets.token" in content:
                    checks["key_generation"] = True

            except Exception:
                pass

        passed = sum(checks.values())
        total = len(checks)

        return {
            "secure": passed >= 3,  # At least 3 of 5 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "standards": {
                "min_key_sizes": self.ENCRYPTION_STANDARDS["key_sizes"],
                "kdf_iterations": self.ENCRYPTION_STANDARDS["kdf_iterations"],
            },
        }

    async def _validate_password_storage(self) -> dict[str, Any]:
        """Validate password storage security.
        
        Returns:
            Password storage validation results
        """
        checks = {
            "hashed_storage": False,
            "salt_usage": False,
            "strong_algorithm": False,
            "no_plaintext": True,  # Start with True, set False if found
            "secure_comparison": False,
        }

        plaintext_issues = []

        for py_file in Path("genesis").rglob("*.py"):
            try:
                content = py_file.read_text()

                # Check for password hashing
                if any(algo in content for algo in ["bcrypt", "scrypt", "argon2", "pbkdf2"]):
                    checks["hashed_storage"] = True
                    checks["strong_algorithm"] = True

                # Check for salt usage
                if "salt" in content.lower() or "bcrypt" in content or "scrypt" in content:
                    checks["salt_usage"] = True

                # Check for plaintext passwords
                plaintext_patterns = [
                    r"password\s*=\s*[\"'][^\"']+[\"']",
                    r"pwd\s*=\s*[\"'][^\"']+[\"']",
                ]

                for pattern in plaintext_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Check if it's not a placeholder
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match and not any(placeholder in match.group(0) for placeholder in ["example", "test", "demo", "$"]):
                            checks["no_plaintext"] = False
                            plaintext_issues.append(str(py_file))

                # Check for secure comparison
                if "hmac.compare_digest" in content or "secrets.compare_digest" in content or "bcrypt.check" in content or "verify_password" in content:
                    checks["secure_comparison"] = True

            except Exception as e:
                logger.error(f"Error checking file {py_file}: {e}")

        if plaintext_issues:
            self.encryption_issues.append({
                "type": "plaintext_passwords",
                "files": plaintext_issues,
                "severity": "critical",
            })

        passed = sum(checks.values())
        total = len(checks)

        return {
            "secure": passed >= 4,  # At least 4 of 5 checks
            "passed": passed,
            "total_checks": total,
            "checks": checks,
            "plaintext_issues": plaintext_issues,
        }

    def _generate_recommendations(self, results: list[dict[str, Any]]) -> list[str]:
        """Generate encryption recommendations.
        
        Args:
            results: List of validation results
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # API key recommendations
        api_results = results[0]
        if not api_results["encrypted"]:
            missing = [k for k, v in api_results["checks"].items() if not v]
            recommendations.append(f"Implement API key encryption: {', '.join(missing)}")

        # TLS recommendations
        tls_results = results[1]
        if not tls_results["compliant"]:
            recommendations.append(f"Upgrade to TLS {self.ENCRYPTION_STANDARDS['tls_version']} minimum")
            if not tls_results["checks"]["strong_ciphers"]:
                recommendations.append("Configure strong cipher suites")

        # Data at rest recommendations
        dar_results = results[2]
        if not dar_results["encrypted"]:
            if not dar_results["checks"]["database_encryption"]:
                recommendations.append("Implement database encryption (column-level or transparent)")
            if not dar_results["checks"]["backup_encryption"]:
                recommendations.append("Encrypt all backups using GPG or similar")

        # Communication channel recommendations
        comm_results = results[3]
        if not comm_results["secure"]:
            if not comm_results["checks"]["https_only"]:
                recommendations.append("Enforce HTTPS for all external communications")
            if not comm_results["checks"]["websocket_secure"]:
                recommendations.append("Use WSS (WebSocket Secure) for real-time connections")

        # Key management recommendations
        key_results = results[4]
        if not key_results["secure"]:
            if not key_results["checks"]["key_rotation"]:
                recommendations.append("Implement key rotation policy (90 days recommended)")
            if not key_results["checks"]["key_derivation"]:
                recommendations.append(f"Use PBKDF2 with {self.ENCRYPTION_STANDARDS['kdf_iterations']} iterations")

        # Password storage recommendations
        pwd_results = results[5]
        if not pwd_results["secure"]:
            if not pwd_results["checks"]["hashed_storage"]:
                recommendations.append("Use bcrypt, scrypt, or argon2 for password hashing")
            if pwd_results["plaintext_issues"]:
                recommendations.append(f"Remove plaintext passwords from {len(pwd_results['plaintext_issues'])} files")

        # General recommendations
        if self.encryption_issues:
            recommendations.append(f"Address {len(self.encryption_issues)} critical encryption issues")

        if not recommendations:
            recommendations.append("Maintain current encryption standards with regular audits")

        return recommendations
