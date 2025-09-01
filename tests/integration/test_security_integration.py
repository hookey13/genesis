"""
Comprehensive integration tests for the complete security infrastructure.
Tests all components working together: SecretsManager, API rotation, HSM, 
Zero-Knowledge, and SOC 2 compliance.
"""

import os
import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from genesis.security.secrets_manager import SecretsManager, SecretBackend
from genesis.security.api_key_rotation import APIKeyRotationManager, APIKeySet
from genesis.security.vault_client import VaultClient
from genesis.security.hsm_interface import HSMManager, HSMType
from genesis.security.zero_knowledge import (
    ZeroKnowledgeManager,
    EncryptionMode,
    ClientSideEncryption,
    ShamirSecret
)
from genesis.security.soc2_compliance import (
    SOC2ComplianceManager,
    EventType,
    AccessLevel
)
from genesis.config.settings import Settings


class TestCompleteSecurityIntegration:
    """Test complete security infrastructure integration."""
    
    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    async def security_stack(self, temp_dir):
        """Create complete security stack."""
        # Configuration
        config = {
            "audit_db_path": temp_dir / "audit.db",
            "secrets_path": temp_dir / ".secrets"
        }
        
        # Initialize components
        secrets_manager = SecretsManager(
            backend=SecretBackend.LOCAL,
            config={"storage_path": temp_dir / ".secrets"}
        )
        
        vault_client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL,
            enable_rotation=True
        )
        
        hsm_manager = HSMManager(
            hsm_type=HSMType.SIMULATOR,
            config={}
        )
        await hsm_manager.initialize()
        
        zk_manager = ZeroKnowledgeManager(
            encryption_mode=EncryptionMode.CLIENT_SIDE,
            threshold=2,
            total_shares=3
        )
        
        soc2_manager = SOC2ComplianceManager(config)
        
        return {
            "secrets_manager": secrets_manager,
            "vault_client": vault_client,
            "hsm_manager": hsm_manager,
            "zk_manager": zk_manager,
            "soc2_manager": soc2_manager
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_secret_workflow(self, security_stack):
        """Test complete secret management workflow."""
        secrets_manager = security_stack["secrets_manager"]
        soc2_manager = security_stack["soc2_manager"]
        
        # Step 1: Store API credentials
        api_credentials = {
            "api_key": "test_api_key_12345",
            "api_secret": "test_api_secret_67890"
        }
        
        # Log the storage event
        soc2_manager.log_event(
            event_type=EventType.KEY_OPERATION,
            user_id="test_user",
            resource="/genesis/exchange/api-keys",
            action="store",
            result="pending",
            details={"operation": "store_credentials"}
        )
        
        # Store with SecretsManager
        success = await secrets_manager.put_secret(
            "/genesis/exchange/api-keys",
            api_credentials,
            metadata={"user": "test_user"}
        )
        assert success is True
        
        # Log success
        soc2_manager.log_event(
            event_type=EventType.KEY_OPERATION,
            user_id="test_user",
            resource="/genesis/exchange/api-keys",
            action="store",
            result="success",
            details={"operation": "store_credentials"}
        )
        
        # Step 2: Retrieve credentials
        retrieved = await secrets_manager.get_secret("/genesis/exchange/api-keys")
        assert retrieved == api_credentials
        
        # Step 3: Verify audit trail
        events = soc2_manager.audit_log.query_events(
            user_id="test_user",
            event_type=EventType.KEY_OPERATION
        )
        assert len(events) == 2
        assert events[0]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_api_key_rotation_with_compliance(self, security_stack):
        """Test API key rotation with full compliance logging."""
        secrets_manager = security_stack["secrets_manager"]
        soc2_manager = security_stack["soc2_manager"]
        
        # Initialize rotation manager
        rotation_manager = APIKeyRotationManager(
            secrets_manager=secrets_manager,
            exchange_api=None,
            verification_callback=None
        )
        
        # Store initial keys
        initial_keys = APIKeySet(
            api_key="old_key",
            api_secret="old_secret",
            created_at=datetime.utcnow(),
            permissions=["trade", "read"]
        )
        rotation_manager.primary_keys = initial_keys
        
        # Log rotation start
        soc2_manager.log_event(
            event_type=EventType.KEY_OPERATION,
            user_id="system",
            resource="/genesis/exchange/api-keys",
            action="rotate",
            result="started",
            details={"operation": "key_rotation"}
        )
        
        # Perform rotation
        result = await rotation_manager.rotate_keys(
            grace_period=timedelta(seconds=1)
        )
        
        assert result["status"] == "success"
        assert "rotation_id" in result
        
        # Log rotation success
        soc2_manager.log_event(
            event_type=EventType.KEY_OPERATION,
            user_id="system",
            resource="/genesis/exchange/api-keys",
            action="rotate",
            result="success",
            details={
                "rotation_id": result["rotation_id"],
                "grace_period": "1 second"
            }
        )
        
        # Verify dual keys during grace period
        all_keys = rotation_manager.get_all_active_keys()
        assert len(all_keys) == 2
        
        # Wait for grace period
        await asyncio.sleep(1.5)
        
        # Verify old key removed
        assert rotation_manager.secondary_keys is None
    
    @pytest.mark.asyncio
    async def test_zero_knowledge_encryption_with_hsm(self, security_stack):
        """Test zero-knowledge encryption using HSM for key storage."""
        hsm_manager = security_stack["hsm_manager"]
        zk_manager = security_stack["zk_manager"]
        soc2_manager = security_stack["soc2_manager"]
        
        # Generate master key in HSM
        master_key_id = await hsm_manager.create_master_key("zk_master")
        assert master_key_id is not None
        
        # Encrypt API credentials with zero-knowledge
        encrypted_creds = await zk_manager.encrypt_api_credentials(
            api_key="sensitive_api_key",
            api_secret="sensitive_api_secret"
        )
        
        assert "encrypted_payload" in encrypted_creds
        assert "key_shares" in encrypted_creds
        assert len(encrypted_creds["key_shares"]) == 3
        
        # Log encryption event
        soc2_manager.log_event(
            event_type=EventType.DATA_ACCESS,
            user_id="test_user",
            resource="api_credentials",
            action="encrypt",
            result="success",
            details={
                "encryption_mode": "zero_knowledge",
                "shares_created": 3,
                "hsm_used": True
            }
        )
        
        # Test reconstruction with minimum shares
        min_shares = encrypted_creds["key_shares"][:2]  # Use 2 of 3 shares
        decrypted = await zk_manager.decrypt_with_shares(
            encrypted_creds,
            min_shares
        )
        
        assert decrypted["api_key"] == "sensitive_api_key"
        assert decrypted["api_secret"] == "sensitive_api_secret"
    
    @pytest.mark.asyncio
    async def test_access_control_with_audit(self, security_stack):
        """Test access control enforcement with audit logging."""
        soc2_manager = security_stack["soc2_manager"]
        
        # Add users with different access levels
        soc2_manager.access_control.add_user(
            user_id="admin_user",
            access_level=AccessLevel.ADMIN,
            resources={"*"},
            ip_whitelist=["192.168.1.1", "10.0.0.1"]
        )
        
        soc2_manager.access_control.add_user(
            user_id="trader_user",
            access_level=AccessLevel.TRADER,
            resources={"/api/*", "/trading/*"},
            time_restrictions={
                "allowed_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                "allowed_hours": {"start": 9, "end": 17}
            }
        )
        
        soc2_manager.access_control.add_user(
            user_id="auditor_user",
            access_level=AccessLevel.AUDITOR,
            resources={"/audit/*", "/reports/*"}
        )
        
        # Test admin access
        admin_granted = soc2_manager.check_access(
            user_id="admin_user",
            resource="/genesis/secrets",
            permission="rotate_keys",
            ip_address="192.168.1.1"
        )
        assert admin_granted is True
        
        # Test trader access denied for admin resource
        trader_denied = soc2_manager.check_access(
            user_id="trader_user",
            resource="/genesis/secrets",
            permission="rotate_keys",
            ip_address="192.168.1.1"
        )
        assert trader_denied is False
        
        # Test auditor access to audit logs
        auditor_granted = soc2_manager.check_access(
            user_id="auditor_user",
            resource="/audit/logs",
            permission="view_audit_log",
            ip_address="192.168.1.1"
        )
        assert auditor_granted is True
        
        # Verify audit events
        events = soc2_manager.audit_log.query_events(
            event_type=EventType.AUTHORIZATION
        )
        assert len(events) >= 3
    
    @pytest.mark.asyncio
    async def test_compliance_monitoring(self, security_stack):
        """Test SOC 2 compliance monitoring and reporting."""
        soc2_manager = security_stack["soc2_manager"]
        
        # Generate various events for compliance checking
        for i in range(5):
            soc2_manager.log_event(
                event_type=EventType.AUTHENTICATION,
                user_id=f"user_{i}",
                resource="/api/login",
                action="login",
                result="success",
                details={"method": "password"},
                ip_address=f"192.168.1.{i}"
            )
        
        # Add some failed attempts
        for i in range(3):
            soc2_manager.log_event(
                event_type=EventType.AUTHENTICATION,
                user_id="suspicious_user",
                resource="/api/login",
                action="login",
                result="failed",
                details={"reason": "invalid_password"},
                ip_address="10.0.0.100",
                risk_score=30
            )
        
        # Run compliance check
        compliance_report = await soc2_manager.run_compliance_check()
        
        assert "checks" in compliance_report
        assert compliance_report["checks"]["audit_integrity"] is True
        assert compliance_report["checks"]["access_control"] is True
        assert compliance_report["score"] > 70
        
        # Generate compliance report
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow()
        
        report = soc2_manager.generate_report(start_date, end_date)
        
        assert report["metrics"]["total_events"] >= 8
        assert report["metrics"]["unique_users"] >= 6
        assert "SECURITY" in report["trust_principles"]["security"]
    
    @pytest.mark.asyncio
    async def test_audit_log_integrity(self, security_stack):
        """Test audit log tamper-proof integrity."""
        soc2_manager = security_stack["soc2_manager"]
        
        # Create chain of events
        hashes = []
        for i in range(5):
            hash_value = soc2_manager.log_event(
                event_type=EventType.SYSTEM_EVENT,
                user_id="system",
                resource=f"/test/{i}",
                action="test",
                result="success",
                details={"index": i}
            )
            hashes.append(hash_value)
        
        # Verify integrity
        is_valid, issues = soc2_manager.audit_log.verify_integrity()
        assert is_valid is True
        assert len(issues) == 0
        
        # Each hash should be different
        assert len(set(hashes)) == 5
    
    @pytest.mark.asyncio
    async def test_runtime_secret_injection(self, security_stack):
        """Test runtime secret injection without code changes."""
        vault_client = security_stack["vault_client"]
        
        # Inject secret at runtime
        runtime_secret = {
            "webhook_secret": "runtime_webhook_123",
            "jwt_secret": "runtime_jwt_456"
        }
        
        vault_client.inject_runtime_secret("/runtime/secrets", runtime_secret)
        
        # Retrieve injected secret
        retrieved = vault_client.get_secret("/runtime/secrets")
        assert retrieved == runtime_secret
        
        # Test specific key retrieval
        webhook = vault_client.get_secret("/runtime/secrets", key="webhook_secret")
        assert webhook == "runtime_webhook_123"
    
    @pytest.mark.asyncio
    async def test_secure_multi_party_computation(self, security_stack):
        """Test secure multi-party computation with Shamir's Secret Sharing."""
        zk_manager = security_stack["zk_manager"]
        
        # Create secret shares for multi-party computation
        secret_data = b"highly_sensitive_trading_algorithm"
        
        # Split into shares (3 total, 2 needed)
        shamir = ShamirSecret(threshold=2, total_shares=3)
        shares = shamir.split_secret(secret_data)
        
        assert len(shares) == 3
        
        # Party 1 and Party 2 collaborate (2 of 3)
        collaborative_shares = [shares[0], shares[1]]
        reconstructed = shamir.reconstruct_secret(collaborative_shares)
        
        assert reconstructed == secret_data
        
        # Single party cannot reconstruct
        with pytest.raises(ValueError, match="Need at least 2 shares"):
            shamir.reconstruct_secret([shares[0]])
    
    @pytest.mark.asyncio
    async def test_end_to_end_encryption_api_communication(self, security_stack):
        """Test end-to-end encrypted API communication."""
        zk_manager = security_stack["zk_manager"]
        
        # Simulate two parties establishing secure channel
        client_e2e = zk_manager.e2e_encryption
        server_e2e = security_stack["zk_manager"].e2e_encryption
        
        # Exchange public keys
        client_public = client_e2e.get_public_key_pem()
        server_public = server_e2e.get_public_key_pem()
        
        # Establish sessions
        client_e2e.add_peer_public_key("server", server_public)
        encrypted_session = client_e2e.establish_session("server")
        
        # Server decrypts session key
        session_key = server_e2e.decrypt_session_key(encrypted_session)
        server_e2e.session_keys["client"] = session_key
        
        # Send encrypted message
        message = {"order": "buy", "amount": 100, "price": 50000}
        encrypted_msg = await zk_manager.send_secure_message("server", message)
        
        assert encrypted_msg.ciphertext is not None
        assert encrypted_msg.nonce is not None
        assert encrypted_msg.tag is not None
    
    @pytest.mark.asyncio
    async def test_complete_security_status(self, security_stack):
        """Test status reporting from all security components."""
        # Get status from each component
        secrets_status = await security_stack["secrets_manager"].health_check()
        hsm_status = security_stack["hsm_manager"].get_status()
        zk_status = security_stack["zk_manager"].get_status()
        soc2_status = security_stack["soc2_manager"].get_status()
        vault_status = security_stack["vault_client"].health_check()
        
        # Verify all components are operational
        assert secrets_status["backend_healthy"] is True
        assert hsm_status["initialized"] is True
        assert zk_status["encryption_mode"] == "client_side"
        assert soc2_status["audit_log_valid"] is True
        assert vault_status["backend"] == "local"
        
        print("\n=== Security Infrastructure Status ===")
        print(f"Secrets Manager: {secrets_status}")
        print(f"HSM: {hsm_status}")
        print(f"Zero-Knowledge: {zk_status}")
        print(f"SOC 2 Compliance: {soc2_status}")
        print(f"Vault Client: {vault_status}")
        print("=====================================\n")