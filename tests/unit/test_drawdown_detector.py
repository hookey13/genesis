"""Unit tests for drawdown detection and monitoring."""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from genesis.analytics.drawdown_detector import DrawdownDetector
from genesis.core.events import EventType
from genesis.core.models import Account, TradingTier


class TestDrawdownDetector:
    """Test suite for DrawdownDetector class."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository."""
        repo = Mock()
        repo.get_account = Mock()
        repo.get_peak_balance = Mock()
        repo.update_peak_balance = Mock()
        repo.get_tilt_profile = Mock()
        return repo
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        bus = Mock()
        bus.publish = Mock()
        return bus
    
    @pytest.fixture
    def detector(self, mock_repository, mock_event_bus):
        """Create a DrawdownDetector instance."""
        return DrawdownDetector(mock_repository, mock_event_bus)
    
    def test_calculate_drawdown_normal(self, detector):
        """Test normal drawdown calculation."""
        balance = Decimal("900")
        peak = Decimal("1000")
        
        drawdown = detector.calculate_drawdown(balance, peak)
        
        assert drawdown == Decimal("0.10")  # 10% drawdown
    
    def test_calculate_drawdown_zero_peak(self, detector):
        """Test drawdown calculation with zero peak."""
        balance = Decimal("1000")
        peak = Decimal("0")
        
        drawdown = detector.calculate_drawdown(balance, peak)
        
        assert drawdown == Decimal("0")
    
    def test_calculate_drawdown_negative(self, detector):
        """Test drawdown calculation when balance exceeds peak."""
        balance = Decimal("1100")
        peak = Decimal("1000")
        
        drawdown = detector.calculate_drawdown(balance, peak)
        
        assert drawdown == Decimal("0")  # No drawdown when above peak
    
    def test_detect_drawdown_breach_threshold_exceeded(self, detector, mock_repository, mock_event_bus):
        """Test drawdown breach detection when threshold exceeded."""
        account = Mock()
        account.balance = Decimal("850")
        account.tier = TradingTier.HUNTER
        
        mock_repository.get_account.return_value = account
        mock_repository.get_peak_balance.return_value = Decimal("1000")
        
        result = detector.detect_drawdown_breach("test_account", Decimal("0.10"))
        
        assert result is True
        mock_event_bus.publish.assert_called_once()
        event_data = mock_event_bus.publish.call_args[0][0]
        assert event_data["type"] == EventType.DRAWDOWN_DETECTED
        assert event_data["drawdown_pct"] == Decimal("0.15")  # 15% drawdown
    
    def test_detect_drawdown_breach_threshold_not_exceeded(self, detector, mock_repository):
        """Test drawdown breach detection when threshold not exceeded."""
        account = Mock()
        account.balance = Decimal("950")
        account.tier = TradingTier.HUNTER
        
        mock_repository.get_account.return_value = account
        mock_repository.get_peak_balance.return_value = Decimal("1000")
        
        result = detector.detect_drawdown_breach("test_account", Decimal("0.10"))
        
        assert result is False
    
    def test_detect_drawdown_breach_account_not_found(self, detector, mock_repository):
        """Test drawdown breach detection when account not found."""
        mock_repository.get_account.return_value = None
        
        result = detector.detect_drawdown_breach("invalid_account")
        
        assert result is False
    
    def test_get_tier_threshold(self, detector, mock_repository):
        """Test tier-based threshold retrieval."""
        account_sniper = Mock()
        account_sniper.tier = TradingTier.SNIPER
        
        account_hunter = Mock()
        account_hunter.tier = TradingTier.HUNTER
        
        account_strategist = Mock()
        account_strategist.tier = TradingTier.STRATEGIST
        
        mock_repository.get_account.side_effect = [
            account_sniper,
            account_hunter,
            account_strategist
        ]
        
        # Test each tier
        assert detector._get_tier_threshold("sniper") == Decimal("0.05")
        assert detector._get_tier_threshold("hunter") == Decimal("0.10")
        assert detector._get_tier_threshold("strategist") == Decimal("0.15")
    
    def test_update_balance_tracking(self, detector, mock_repository):
        """Test balance tracking update."""
        mock_repository.get_peak_balance.return_value = Decimal("1000")
        
        # Update with lower balance
        peak, drawdown = detector.update_balance_tracking("test_account", Decimal("900"))
        
        assert peak == Decimal("1000")
        assert drawdown == Decimal("0.10")
        
        # Update with higher balance
        mock_repository.get_peak_balance.return_value = None
        peak, drawdown = detector.update_balance_tracking("test_account", Decimal("1100"))
        
        assert peak == Decimal("1100")
        assert drawdown == Decimal("0")
        mock_repository.update_peak_balance.assert_called_with("test_account", Decimal("1100"))
    
    def test_get_drawdown_stats(self, detector, mock_repository):
        """Test comprehensive drawdown statistics retrieval."""
        account = Mock()
        account.balance = Decimal("800")
        account.tier = TradingTier.HUNTER
        
        mock_repository.get_account.return_value = account
        mock_repository.get_peak_balance.return_value = Decimal("1000")
        
        stats = detector.get_drawdown_stats("test_account")
        
        assert stats["account_id"] == "test_account"
        assert stats["current_balance"] == Decimal("800")
        assert stats["peak_balance"] == Decimal("1000")
        assert stats["drawdown_pct"] == Decimal("0.20")
        assert stats["drawdown_amount"] == Decimal("200")
        assert stats["threshold"] == Decimal("0.10")
        assert stats["threshold_breached"] is True
        assert stats["recovery_needed"] == Decimal("200")
    
    def test_reset_peak_balance(self, detector, mock_repository):
        """Test peak balance reset."""
        account = Mock()
        account.balance = Decimal("500")
        
        mock_repository.get_account.return_value = account
        
        detector.reset_peak_balance("test_account")
        
        mock_repository.update_peak_balance.assert_called_once_with(
            "test_account",
            Decimal("500")
        )
        assert detector._peak_balances["test_account"] == Decimal("500")
    
    def test_edge_case_exact_threshold(self, detector, mock_repository):
        """Test detection at exactly the threshold."""
        account = Mock()
        account.balance = Decimal("900")  # Exactly 10% down from 1000
        account.tier = TradingTier.HUNTER
        
        mock_repository.get_account.return_value = account
        mock_repository.get_peak_balance.return_value = Decimal("1000")
        
        result = detector.detect_drawdown_breach("test_account", Decimal("0.10"))
        
        assert result is True  # Should trigger at exactly 10%
    
    def test_concurrent_balance_updates(self, detector, mock_repository):
        """Test concurrent balance updates maintain consistency."""
        mock_repository.get_peak_balance.return_value = Decimal("1000")
        
        # Simulate concurrent updates
        detector.update_balance_tracking("account1", Decimal("950"))
        detector.update_balance_tracking("account2", Decimal("800"))
        detector.update_balance_tracking("account1", Decimal("900"))
        
        assert "account1" in detector._peak_balances
        assert "account2" in detector._peak_balances
        assert detector._peak_balances["account1"] == Decimal("1000")
        assert detector._peak_balances["account2"] == Decimal("1000")