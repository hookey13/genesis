"""Unit tests for ReconciliationEngine - Critical for financial accuracy."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from genesis.analytics.reconciliation import (
    BalanceDiscrepancy,
    ReconciliationEngine,
    ReconciliationReport,
    ReconciliationStatus,
)
from genesis.core.models import Account, AccountType, Position, Tier
from genesis.data.repository import Repository


class TestReconciliationEngine:
    """Test suite for automated reconciliation system."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = Mock(spec=Repository)
        repo.get_account_balance = AsyncMock()
        repo.get_positions_by_account = AsyncMock()
        repo.get_exchange_balances = AsyncMock()
        repo.get_exchange_positions = AsyncMock()
        repo.save_reconciliation_report = AsyncMock()
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange gateway."""
        exchange = Mock()
        exchange.fetch_balance = AsyncMock()
        exchange.fetch_positions = AsyncMock()
        return exchange

    @pytest.fixture
    def sample_account(self):
        """Create test account."""
        return Account(
            account_id="test_account_123",
            tier=Tier.STRATEGIST,
            account_type=AccountType.MASTER,
            balance_usdt=Decimal("50000.00"),
            created_at=datetime.now(UTC),
        )

    @pytest.fixture
    def reconciliation_engine(self, mock_repo, mock_exchange):
        """Create ReconciliationEngine instance."""
        return ReconciliationEngine(repository=mock_repo, exchange=mock_exchange)

    @pytest.mark.asyncio
    async def test_reconcile_balances_match(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test balance reconciliation when balances match."""
        # Setup
        db_balance = Decimal("50000.00")
        exchange_balance = Decimal("50000.00")

        mock_repo.get_account_balance.return_value = db_balance
        mock_exchange.fetch_balance.return_value = {
            "USDT": {
                "free": float(exchange_balance),
                "used": 0,
                "total": float(exchange_balance),
            }
        }

        # Execute
        report = await reconciliation_engine.reconcile_balances(
            sample_account.account_id
        )

        # Verify
        assert report.status == ReconciliationStatus.MATCHED
        assert len(report.balance_discrepancies) == 0
        assert report.total_discrepancy == Decimal("0.00")
        mock_repo.save_reconciliation_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_balances_discrepancy(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test balance reconciliation with discrepancy detection."""
        # Setup
        db_balance = Decimal("50000.00")
        exchange_balance = Decimal("49995.50")  # $4.50 discrepancy

        mock_repo.get_account_balance.return_value = db_balance
        mock_exchange.fetch_balance.return_value = {
            "USDT": {
                "free": float(exchange_balance),
                "used": 0,
                "total": float(exchange_balance),
            }
        }

        # Execute
        report = await reconciliation_engine.reconcile_balances(
            sample_account.account_id
        )

        # Verify
        assert report.status == ReconciliationStatus.DISCREPANCY
        assert len(report.balance_discrepancies) == 1
        assert report.balance_discrepancies[0].database_value == db_balance
        assert report.balance_discrepancies[0].exchange_value == exchange_balance
        assert report.balance_discrepancies[0].difference == Decimal("4.50")
        assert report.total_discrepancy == Decimal("4.50")

    @pytest.mark.asyncio
    async def test_reconcile_positions_match(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test position reconciliation when positions match."""
        # Setup
        db_positions = [
            Position(
                position_id="pos_1",
                account_id=sample_account.account_id,
                symbol="BTC/USDT",
                side="long",
                size=Decimal("0.5"),
                entry_price=Decimal("45000.00"),
            ),
            Position(
                position_id="pos_2",
                account_id=sample_account.account_id,
                symbol="ETH/USDT",
                side="long",
                size=Decimal("10.0"),
                entry_price=Decimal("3000.00"),
            ),
        ]

        exchange_positions = [
            {"symbol": "BTC/USDT", "contracts": 0.5, "side": "long"},
            {"symbol": "ETH/USDT", "contracts": 10.0, "side": "long"},
        ]

        mock_repo.get_positions_by_account.return_value = db_positions
        mock_exchange.fetch_positions.return_value = exchange_positions

        # Execute
        report = await reconciliation_engine.reconcile_positions(
            sample_account.account_id
        )

        # Verify
        assert report.status == ReconciliationStatus.MATCHED
        assert len(report.position_discrepancies) == 0

    @pytest.mark.asyncio
    async def test_reconcile_positions_missing_in_exchange(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test detection of positions missing in exchange."""
        # Setup
        db_positions = [
            Position(
                position_id="pos_1",
                account_id=sample_account.account_id,
                symbol="BTC/USDT",
                side="long",
                size=Decimal("0.5"),
                entry_price=Decimal("45000.00"),
            )
        ]

        exchange_positions = []  # Position missing in exchange

        mock_repo.get_positions_by_account.return_value = db_positions
        mock_exchange.fetch_positions.return_value = exchange_positions

        # Execute
        report = await reconciliation_engine.reconcile_positions(
            sample_account.account_id
        )

        # Verify
        assert report.status == ReconciliationStatus.DISCREPANCY
        assert len(report.position_discrepancies) == 1
        assert report.position_discrepancies[0].symbol == "BTC/USDT"
        assert report.position_discrepancies[0].database_size == Decimal("0.5")
        assert report.position_discrepancies[0].exchange_size == Decimal("0")
        assert (
            report.position_discrepancies[0].discrepancy_type == "MISSING_IN_EXCHANGE"
        )

    @pytest.mark.asyncio
    async def test_reconcile_positions_size_mismatch(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test detection of position size mismatches."""
        # Setup
        db_positions = [
            Position(
                position_id="pos_1",
                account_id=sample_account.account_id,
                symbol="BTC/USDT",
                side="long",
                size=Decimal("0.5"),
                entry_price=Decimal("45000.00"),
            )
        ]

        exchange_positions = [
            {"symbol": "BTC/USDT", "contracts": 0.45, "side": "long"}  # Size mismatch
        ]

        mock_repo.get_positions_by_account.return_value = db_positions
        mock_exchange.fetch_positions.return_value = exchange_positions

        # Execute
        report = await reconciliation_engine.reconcile_positions(
            sample_account.account_id
        )

        # Verify
        assert report.status == ReconciliationStatus.DISCREPANCY
        assert len(report.position_discrepancies) == 1
        assert report.position_discrepancies[0].database_size == Decimal("0.5")
        assert report.position_discrepancies[0].exchange_size == Decimal("0.45")
        assert report.position_discrepancies[0].size_difference == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_month_end_reconciliation(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test automated month-end reconciliation process."""
        # Setup
        mock_repo.get_all_accounts.return_value = [sample_account]
        mock_repo.get_account_balance.return_value = Decimal("50000.00")
        mock_exchange.fetch_balance.return_value = {
            "USDT": {"free": 50000.0, "used": 0, "total": 50000.0}
        }
        mock_repo.get_positions_by_account.return_value = []
        mock_exchange.fetch_positions.return_value = []

        # Execute
        reports = await reconciliation_engine.run_month_end_reconciliation()

        # Verify
        assert len(reports) == 1
        assert reports[0].account_id == sample_account.account_id
        assert reports[0].status == ReconciliationStatus.MATCHED
        assert reports[0].reconciliation_type == "MONTH_END"

    @pytest.mark.asyncio
    async def test_reconciliation_with_retry_on_failure(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test retry mechanism for reconciliation failures."""
        # Setup - First call fails, second succeeds
        mock_exchange.fetch_balance.side_effect = [
            Exception("Network error"),
            {"USDT": {"free": 50000.0, "used": 0, "total": 50000.0}},
        ]
        mock_repo.get_account_balance.return_value = Decimal("50000.00")

        # Execute
        report = await reconciliation_engine.reconcile_balances(
            sample_account.account_id, retry_attempts=2
        )

        # Verify
        assert report.status == ReconciliationStatus.MATCHED
        assert mock_exchange.fetch_balance.call_count == 2

    @pytest.mark.asyncio
    async def test_reconciliation_alert_on_critical_discrepancy(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test alert generation for critical discrepancies."""
        # Setup - Large discrepancy
        db_balance = Decimal("50000.00")
        exchange_balance = Decimal("45000.00")  # $5000 discrepancy

        mock_repo.get_account_balance.return_value = db_balance
        mock_exchange.fetch_balance.return_value = {
            "USDT": {
                "free": float(exchange_balance),
                "used": 0,
                "total": float(exchange_balance),
            }
        }

        with patch.object(reconciliation_engine, "_send_alert") as mock_alert:
            # Execute
            report = await reconciliation_engine.reconcile_balances(
                sample_account.account_id
            )

            # Verify
            assert report.status == ReconciliationStatus.CRITICAL_DISCREPANCY
            assert report.total_discrepancy == Decimal("5000.00")
            mock_alert.assert_called_once_with(
                severity="CRITICAL",
                message=f"Critical balance discrepancy detected for account {sample_account.account_id}: $5000.00",
            )

    def test_reconciliation_report_generation(self, reconciliation_engine):
        """Test reconciliation report formatting and generation."""
        # Setup
        report = ReconciliationReport(
            account_id="test_account",
            timestamp=datetime.now(UTC),
            status=ReconciliationStatus.DISCREPANCY,
            balance_discrepancies=[
                BalanceDiscrepancy(
                    currency="USDT",
                    database_value=Decimal("50000.00"),
                    exchange_value=Decimal("49995.50"),
                    difference=Decimal("4.50"),
                )
            ],
            position_discrepancies=[],
            total_discrepancy=Decimal("4.50"),
            reconciliation_type="DAILY",
        )

        # Execute
        formatted_report = reconciliation_engine.format_report(report)

        # Verify
        assert "account_id" in formatted_report
        assert formatted_report["status"] == "DISCREPANCY"
        assert len(formatted_report["balance_discrepancies"]) == 1
        assert formatted_report["total_discrepancy"] == "4.50"

    @pytest.mark.asyncio
    async def test_decimal_precision_preservation(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test that Decimal precision is preserved throughout reconciliation."""
        # Setup with precise decimal values
        db_balance = Decimal("12345.678901234567890")
        exchange_balance_float = 12345.678901234567890

        mock_repo.get_account_balance.return_value = db_balance
        mock_exchange.fetch_balance.return_value = {
            "USDT": {
                "free": exchange_balance_float,
                "used": 0,
                "total": exchange_balance_float,
            }
        }

        # Execute
        report = await reconciliation_engine.reconcile_balances(
            sample_account.account_id
        )

        # Verify - Check that precision is maintained
        assert isinstance(report.balance_discrepancies[0].database_value, Decimal)
        assert isinstance(report.balance_discrepancies[0].exchange_value, Decimal)
        assert isinstance(report.balance_discrepancies[0].difference, Decimal)
        # Should detect tiny discrepancy due to float conversion
        assert report.status == ReconciliationStatus.MATCHED  # Within tolerance

    @pytest.mark.asyncio
    async def test_multi_currency_reconciliation(
        self, reconciliation_engine, mock_repo, mock_exchange, sample_account
    ):
        """Test reconciliation with multiple currencies."""
        # Setup
        db_balances = {
            "USDT": Decimal("50000.00"),
            "BTC": Decimal("1.5"),
            "ETH": Decimal("20.0"),
        }

        exchange_balances = {
            "USDT": {"free": 50000.0, "used": 0, "total": 50000.0},
            "BTC": {"free": 1.5, "used": 0, "total": 1.5},
            "ETH": {"free": 19.95, "used": 0, "total": 19.95},  # Discrepancy
        }

        mock_repo.get_account_balances.return_value = db_balances
        mock_exchange.fetch_balance.return_value = exchange_balances

        # Execute
        report = await reconciliation_engine.reconcile_all_balances(
            sample_account.account_id
        )

        # Verify
        assert report.status == ReconciliationStatus.DISCREPANCY
        assert len(report.balance_discrepancies) == 1  # Only ETH has discrepancy
        assert report.balance_discrepancies[0].currency == "ETH"
        assert report.balance_discrepancies[0].difference == Decimal("0.05")
