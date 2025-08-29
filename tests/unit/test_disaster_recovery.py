"""Unit tests for DisasterRecovery - Recovery procedure validation."""

import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from genesis.core.models import Account, Order, OrderSide, OrderType, Position, Tier
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.utils.disaster_recovery import (
    DisasterRecovery,
    RecoveryStatus,
    SystemSnapshot,
)


class TestDisasterRecovery:
    """Test suite for disaster recovery procedures."""

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = Mock(spec=Repository)
        repo.get_all_positions = AsyncMock()
        repo.get_all_orders = AsyncMock()
        repo.get_all_accounts = AsyncMock()
        repo.get_events_since = AsyncMock()
        repo.save_snapshot = AsyncMock()
        repo.load_snapshot = AsyncMock()
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange gateway."""
        exchange = Mock()
        exchange.cancel_all_orders = AsyncMock()
        exchange.close_position = AsyncMock()
        exchange.get_open_orders = AsyncMock()
        exchange.get_positions = AsyncMock()
        return exchange

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        event_bus = Mock(spec=EventBus)
        event_bus.publish = AsyncMock()
        event_bus.replay_events = AsyncMock()
        return event_bus

    @pytest.fixture
    def disaster_recovery(self, mock_repo, mock_exchange, mock_event_bus):
        """Create DisasterRecovery instance."""
        return DisasterRecovery(
            repository=mock_repo, exchange=mock_exchange, event_bus=mock_event_bus
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing."""
        return [
            Position(
                position_id="pos_1",
                account_id="acc_1",
                symbol="BTC/USDT",
                side="long",
                size=Decimal("0.5"),
                entry_price=Decimal("45000.00"),
                current_price=Decimal("44500.00"),
                unrealized_pnl=Decimal("-250.00"),
            ),
            Position(
                position_id="pos_2",
                account_id="acc_1",
                symbol="ETH/USDT",
                side="long",
                size=Decimal("10.0"),
                entry_price=Decimal("3000.00"),
                current_price=Decimal("3100.00"),
                unrealized_pnl=Decimal("1000.00"),
            ),
        ]

    @pytest.fixture
    def sample_orders(self):
        """Create sample orders for testing."""
        return [
            Order(
                order_id="ord_1",
                client_order_id="client_1",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.5"),
                price=Decimal("46000.00"),
                status="OPEN",
            ),
            Order(
                order_id="ord_2",
                client_order_id="client_2",
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("5.0"),
                price=Decimal("2900.00"),
                status="OPEN",
            ),
        ]

    @pytest.mark.asyncio
    async def test_emergency_position_close_all(
        self, disaster_recovery, mock_exchange, sample_positions
    ):
        """Test emergency closure of all positions."""
        # Setup
        mock_exchange.get_positions.return_value = sample_positions
        mock_exchange.close_position.return_value = {"status": "FILLED"}

        # Execute
        results = await disaster_recovery.emergency_close_all_positions()

        # Verify
        assert len(results) == 2
        assert mock_exchange.close_position.call_count == 2
        assert all(r["status"] == "FILLED" for r in results)
        # Verify positions closed at market
        for call in mock_exchange.close_position.call_args_list:
            assert call[1]["order_type"] == OrderType.MARKET

    @pytest.mark.asyncio
    async def test_emergency_cancel_all_orders(
        self, disaster_recovery, mock_exchange, sample_orders
    ):
        """Test emergency cancellation of all open orders."""
        # Setup
        mock_exchange.get_open_orders.return_value = sample_orders
        mock_exchange.cancel_all_orders.return_value = {
            "cancelled": 2,
            "failed": 0,
            "orders": ["ord_1", "ord_2"],
        }

        # Execute
        result = await disaster_recovery.emergency_cancel_all_orders()

        # Verify
        assert result["cancelled"] == 2
        assert result["failed"] == 0
        assert len(result["orders"]) == 2
        mock_exchange.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_snapshot_creation(
        self, disaster_recovery, mock_repo, sample_positions, sample_orders
    ):
        """Test creation of system state snapshot."""
        # Setup
        mock_repo.get_all_positions.return_value = sample_positions
        mock_repo.get_all_orders.return_value = sample_orders
        mock_repo.get_all_accounts.return_value = [
            Account(
                account_id="acc_1",
                tier=Tier.STRATEGIST,
                balance_usdt=Decimal("100000.00"),
            )
        ]

        # Execute
        snapshot = await disaster_recovery.create_system_snapshot()

        # Verify
        assert snapshot.timestamp is not None
        assert len(snapshot.positions) == 2
        assert len(snapshot.orders) == 2
        assert len(snapshot.accounts) == 1
        assert snapshot.version == "1.0"
        assert snapshot.checksum is not None
        mock_repo.save_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_restore_from_snapshot(self, disaster_recovery, mock_repo):
        """Test system restoration from snapshot."""
        # Setup
        snapshot = SystemSnapshot(
            timestamp=datetime.now(UTC),
            positions=[],
            orders=[],
            accounts=[],
            tier_states={},
            version="1.0",
            checksum="abc123",
        )
        mock_repo.load_snapshot.return_value = snapshot

        # Execute
        restore_result = await disaster_recovery.restore_from_snapshot(
            snapshot_id="snapshot_001"
        )

        # Verify
        assert restore_result.status == RecoveryStatus.SUCCESS
        assert restore_result.snapshot_id == "snapshot_001"
        assert restore_result.items_restored >= 0
        mock_repo.load_snapshot.assert_called_with("snapshot_001")

    @pytest.mark.asyncio
    async def test_backup_verification(self, disaster_recovery):
        """Test automated backup verification."""
        # Setup
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_file = Path(tmpdir) / "backup_2024_01_01.tar.gz"
            backup_file.write_bytes(b"test backup data")

            # Execute
            verification = await disaster_recovery.verify_backup(str(backup_file))

            # Verify
            assert verification.file_exists is True
            assert verification.file_size > 0
            assert verification.is_corrupted is False
            assert verification.checksum is not None

    @pytest.mark.asyncio
    async def test_position_recovery_from_events(
        self, disaster_recovery, mock_repo, mock_event_bus
    ):
        """Test position recovery from event store."""
        # Setup
        events = [
            {
                "event_type": "POSITION_OPENED",
                "data": {"position_id": "pos_1", "size": "0.5"},
            },
            {
                "event_type": "POSITION_UPDATED",
                "data": {"position_id": "pos_1", "size": "0.7"},
            },
            {"event_type": "POSITION_CLOSED", "data": {"position_id": "pos_2"}},
        ]
        mock_repo.get_events_since.return_value = events

        # Execute
        recovered_positions = await disaster_recovery.recover_positions_from_events(
            since=datetime.now(UTC) - timedelta(days=1)
        )

        # Verify
        assert len(recovered_positions) >= 0
        mock_repo.get_events_since.assert_called_once()
        mock_event_bus.replay_events.assert_called_once_with(events)

    @pytest.mark.asyncio
    async def test_incremental_backup(self, disaster_recovery, mock_repo):
        """Test incremental backup creation."""
        # Setup
        last_backup_time = datetime.now(UTC) - timedelta(hours=6)
        changes = {
            "positions": [{"position_id": "pos_1", "action": "UPDATE"}],
            "orders": [{"order_id": "ord_1", "action": "CREATE"}],
            "accounts": [],
        }

        with patch.object(disaster_recovery, "get_changes_since", return_value=changes):
            # Execute
            incremental = await disaster_recovery.create_incremental_backup(
                last_backup_time
            )

            # Verify
            assert incremental.backup_type == "INCREMENTAL"
            assert incremental.changes_count == 2
            assert incremental.since_timestamp == last_backup_time

    @pytest.mark.asyncio
    async def test_emergency_mode_activation(self, disaster_recovery, mock_exchange):
        """Test activation of emergency trading mode."""
        # Setup
        mock_exchange.set_read_only_mode = AsyncMock(return_value=True)

        # Execute
        await disaster_recovery.activate_emergency_mode()

        # Verify
        assert disaster_recovery.is_emergency_mode is True
        mock_exchange.set_read_only_mode.assert_called_once_with(True)
        # Verify no new orders can be placed
        with pytest.raises(Exception, match="Emergency mode"):
            await disaster_recovery.place_order(Mock())

    @pytest.mark.asyncio
    async def test_connection_recovery(self, disaster_recovery, mock_exchange):
        """Test automatic connection recovery procedures."""
        # Setup
        mock_exchange.is_connected = Mock(side_effect=[False, False, True])
        mock_exchange.reconnect = AsyncMock(return_value=True)

        # Execute
        recovery_result = await disaster_recovery.recover_exchange_connection(
            max_retries=3
        )

        # Verify
        assert recovery_result is True
        assert mock_exchange.reconnect.call_count <= 3

    @pytest.mark.asyncio
    async def test_data_integrity_check(
        self, disaster_recovery, mock_repo, mock_exchange
    ):
        """Test data integrity verification between database and exchange."""
        # Setup
        db_positions = [
            {"position_id": "pos_1", "symbol": "BTC/USDT", "size": Decimal("0.5")}
        ]
        exchange_positions = [{"symbol": "BTC/USDT", "contracts": 0.5}]

        mock_repo.get_all_positions.return_value = db_positions
        mock_exchange.get_positions.return_value = exchange_positions

        # Execute
        integrity_report = await disaster_recovery.verify_data_integrity()

        # Verify
        assert integrity_report.has_discrepancies is False
        assert integrity_report.checked_items > 0

    @pytest.mark.asyncio
    async def test_automated_recovery_workflow(self, disaster_recovery):
        """Test complete automated recovery workflow."""
        # Setup
        with patch.object(disaster_recovery, "detect_failure", return_value=True):
            with patch.object(
                disaster_recovery, "emergency_cancel_all_orders"
            ) as mock_cancel:
                with patch.object(
                    disaster_recovery, "create_system_snapshot"
                ) as mock_snapshot:
                    with patch.object(
                        disaster_recovery, "notify_operators"
                    ) as mock_notify:
                        mock_cancel.return_value = {"cancelled": 5}
                        mock_snapshot.return_value = Mock(snapshot_id="snap_001")

                        # Execute
                        recovery = await disaster_recovery.execute_automated_recovery()

                        # Verify
                        assert recovery.status == RecoveryStatus.COMPLETED
                        mock_cancel.assert_called_once()
                        mock_snapshot.assert_called_once()
                        mock_notify.assert_called()

    @pytest.mark.asyncio
    async def test_rollback_capability(self, disaster_recovery, mock_repo):
        """Test rollback to previous known good state."""
        # Setup
        checkpoints = [
            {
                "checkpoint_id": "cp_1",
                "timestamp": datetime.now(UTC) - timedelta(hours=2),
            },
            {
                "checkpoint_id": "cp_2",
                "timestamp": datetime.now(UTC) - timedelta(hours=1),
            },
        ]

        mock_repo.get_checkpoints.return_value = checkpoints

        # Execute
        rollback_result = await disaster_recovery.rollback_to_checkpoint("cp_1")

        # Verify
        assert rollback_result.status == RecoveryStatus.SUCCESS
        assert rollback_result.checkpoint_id == "cp_1"
        mock_repo.restore_checkpoint.assert_called_with("cp_1")

    def test_recovery_configuration_validation(self, disaster_recovery):
        """Test validation of recovery configuration."""
        # Setup
        config = {
            "max_recovery_time": 300,  # 5 minutes
            "backup_retention_days": 30,
            "snapshot_interval": 3600,  # 1 hour
            "emergency_contacts": ["admin@genesis.com"],
            "recovery_modes": ["AUTOMATIC", "SEMI_AUTOMATIC", "MANUAL"],
        }

        # Execute
        is_valid, errors = disaster_recovery.validate_configuration(config)

        # Verify
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_decimal_precision_in_recovery(self, disaster_recovery, mock_repo):
        """Test that Decimal precision is maintained during recovery."""
        # Setup
        position = Position(
            position_id="pos_1",
            size=Decimal("0.123456789012345678"),
            entry_price=Decimal("45678.987654321098765"),
            unrealized_pnl=Decimal("-1234.567890123456789"),
        )

        snapshot = SystemSnapshot(
            positions=[position],
            orders=[],
            accounts=[],
            timestamp=datetime.now(UTC),
        )

        mock_repo.load_snapshot.return_value = snapshot

        # Execute
        restored = await disaster_recovery.restore_from_snapshot("snap_001")

        # Verify
        restored_pos = restored.positions[0]
        assert isinstance(restored_pos.size, Decimal)
        assert isinstance(restored_pos.entry_price, Decimal)
        assert isinstance(restored_pos.unrealized_pnl, Decimal)
        assert restored_pos.size == position.size
        assert restored_pos.entry_price == position.entry_price

    @pytest.mark.asyncio
    async def test_emergency_notification_system(self, disaster_recovery):
        """Test emergency notification to operators."""
        # Setup
        with patch("smtplib.SMTP") as mock_smtp:
            with patch("twilio.rest.Client") as mock_twilio:
                # Execute
                notifications = await disaster_recovery.send_emergency_notifications(
                    message="Critical system failure detected", severity="CRITICAL"
                )

                # Verify
                assert notifications.email_sent is True
                assert notifications.sms_sent is True
                assert notifications.slack_sent is True
