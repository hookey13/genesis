"""Unit tests for certificate management."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.operations.cert_manager import CertManager, CertificateConfig


@pytest.fixture
def cert_config():
    """Create test certificate configuration."""
    return CertificateConfig(
        domain="test.example.com",
        email="test@example.com",
        cert_dir=Path("/tmp/test_certs"),
        renewal_days_before_expiry=30,
        enable_auto_renewal=True,
        reload_services=["nginx"],
        use_staging=True,
        fallback_to_self_signed=True
    )


@pytest.fixture
def cert_manager(cert_config):
    """Create certificate manager instance."""
    return CertManager(cert_config)


class TestCertManager:
    """Test cases for CertManager."""
    
    @pytest.mark.asyncio
    async def test_check_certificate_expiry(self, cert_manager):
        """Test certificate expiry checking."""
        with patch('subprocess.run') as mock_run:
            # Mock successful openssl output
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="notAfter=Dec 31 23:59:59 2024 GMT"
            )
            
            expiry = await cert_manager.check_certificate_expiry()
            
            assert expiry is not None
            assert expiry.year == 2024
            assert expiry.month == 12
            assert expiry.day == 31
    
    @pytest.mark.asyncio
    async def test_renew_certificate_success(self, cert_manager):
        """Test successful certificate renewal."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Certificate renewed successfully"
            )
            
            result = await cert_manager.renew_certificate()
            
            assert result is True
            # Check certbot was called with correct arguments
            call_args = mock_run.call_args[0][0]
            assert "certbot" in call_args
            assert "renew" in call_args
            assert "--staging" in call_args  # Because use_staging=True
    
    @pytest.mark.asyncio
    async def test_renew_certificate_fallback_to_self_signed(self, cert_manager):
        """Test fallback to self-signed certificate."""
        with patch('subprocess.run') as mock_run:
            # First call to certbot fails
            mock_run.side_effect = [
                MagicMock(returncode=1, stderr="Renewal failed"),
                MagicMock(returncode=0, stdout="Self-signed created")
            ]
            
            result = await cert_manager.renew_certificate()
            
            assert result is True
            # Check openssl was called for self-signed
            second_call = mock_run.call_args_list[1][0][0]
            assert "openssl" in second_call
            assert "req" in second_call
            assert "-x509" in second_call
    
    @pytest.mark.asyncio
    async def test_reload_services(self, cert_manager):
        """Test service reload after renewal."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            await cert_manager._reload_services()
            
            # Check systemctl reload was called
            call_args = mock_run.call_args[0][0]
            assert "systemctl" in call_args
            assert "reload" in call_args
            assert "nginx" in call_args