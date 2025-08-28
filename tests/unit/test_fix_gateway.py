"""Unit tests for FIXGateway - Protocol validation for future FIX integration."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from genesis.exchange.fix_gateway import (
    FIXGateway,
    FIXMessage,
    FIXMessageType,
    FIXField,
    FIXSession,
    FIXOrderStatus,
    FIXExecutionReport,
)
from genesis.core.models import Order, OrderType, OrderSide, Position


class TestFIXGateway:
    """Test suite for FIX protocol gateway implementation."""

    @pytest.fixture
    def fix_config(self):
        """Create FIX configuration."""
        return {
            "sender_comp_id": "GENESIS",
            "target_comp_id": "BROKER",
            "fix_version": "FIX.4.4",
            "heartbeat_interval": 30,
            "socket_host": "fix.broker.com",
            "socket_port": 9876,
        }

    @pytest.fixture
    def fix_gateway(self, fix_config):
        """Create FIXGateway instance."""
        return FIXGateway(config=fix_config)

    @pytest.fixture
    def sample_order(self):
        """Create sample order for testing."""
        return Order(
            order_id="ORD_123456",
            client_order_id="CLIENT_789",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("45000.00"),
            time_in_force="GTC",
            account_id="ACC_001",
        )

    def test_fix_message_creation(self, fix_gateway):
        """Test FIX message creation with proper fields."""
        # Execute
        message = fix_gateway.create_message(
            msg_type=FIXMessageType.NEW_ORDER_SINGLE,
            fields={
                FIXField.SYMBOL: "BTC/USD",
                FIXField.ORDER_QTY: "0.5",
                FIXField.PRICE: "45000.00",
            },
        )

        # Verify
        assert message.msg_type == FIXMessageType.NEW_ORDER_SINGLE
        assert message.get_field(FIXField.SYMBOL) == "BTC/USD"
        assert message.get_field(FIXField.ORDER_QTY) == "0.5"
        assert message.get_field(FIXField.PRICE) == "45000.00"
        assert FIXField.MSG_SEQ_NUM in message.fields

    def test_fix_message_parsing(self, fix_gateway):
        """Test parsing of incoming FIX messages."""
        # Setup
        raw_message = "8=FIX.4.4|9=178|35=D|49=GENESIS|56=BROKER|34=123|52=20240828-10:30:45|55=BTC/USD|54=1|38=0.5|40=2|44=45000.00|59=0|10=089|"

        # Execute
        message = fix_gateway.parse_message(raw_message.replace("|", "\x01"))

        # Verify
        assert message.msg_type == "D"  # New Order Single
        assert message.get_field("55") == "BTC/USD"  # Symbol
        assert message.get_field("38") == "0.5"  # Order Qty
        assert message.get_field("44") == "45000.00"  # Price
        assert message.get_field("49") == "GENESIS"  # Sender

    def test_order_to_fix_conversion(self, fix_gateway, sample_order):
        """Test conversion of internal order to FIX message."""
        # Execute
        fix_message = fix_gateway.order_to_fix(sample_order)

        # Verify
        assert fix_message.msg_type == FIXMessageType.NEW_ORDER_SINGLE
        assert fix_message.get_field(FIXField.CL_ORD_ID) == sample_order.client_order_id
        assert fix_message.get_field(FIXField.SYMBOL) == "BTC/USD"
        assert fix_message.get_field(FIXField.SIDE) == "1"  # Buy
        assert fix_message.get_field(FIXField.ORDER_QTY) == "0.5"
        assert fix_message.get_field(FIXField.PRICE) == "45000.00"
        assert fix_message.get_field(FIXField.ORD_TYPE) == "2"  # Limit

    def test_execution_report_parsing(self, fix_gateway):
        """Test parsing of execution reports."""
        # Setup
        exec_report = FIXMessage(
            msg_type=FIXMessageType.EXECUTION_REPORT,
            fields={
                FIXField.CL_ORD_ID: "CLIENT_789",
                FIXField.ORDER_ID: "BROKER_456",
                FIXField.EXEC_ID: "EXEC_001",
                FIXField.ORD_STATUS: "2",  # Filled
                FIXField.SYMBOL: "BTC/USD",
                FIXField.SIDE: "1",
                FIXField.CUM_QTY: "0.5",
                FIXField.AVG_PX: "44998.50",
                FIXField.LAST_QTY: "0.5",
                FIXField.LAST_PX: "44998.50",
            },
        )

        # Execute
        execution = fix_gateway.parse_execution_report(exec_report)

        # Verify
        assert execution.client_order_id == "CLIENT_789"
        assert execution.exchange_order_id == "BROKER_456"
        assert execution.status == FIXOrderStatus.FILLED
        assert execution.executed_quantity == Decimal("0.5")
        assert execution.average_price == Decimal("44998.50")

    def test_heartbeat_message_generation(self, fix_gateway):
        """Test heartbeat message generation."""
        # Execute
        heartbeat = fix_gateway.create_heartbeat()

        # Verify
        assert heartbeat.msg_type == FIXMessageType.HEARTBEAT
        assert FIXField.SENDING_TIME in heartbeat.fields

    def test_logon_message_creation(self, fix_gateway, fix_config):
        """Test FIX session logon message."""
        # Execute
        logon = fix_gateway.create_logon_message()

        # Verify
        assert logon.msg_type == FIXMessageType.LOGON
        assert logon.get_field(FIXField.SENDER_COMP_ID) == fix_config["sender_comp_id"]
        assert logon.get_field(FIXField.TARGET_COMP_ID) == fix_config["target_comp_id"]
        assert logon.get_field(FIXField.HEARTBT_INT) == str(
            fix_config["heartbeat_interval"]
        )

    def test_order_cancel_request(self, fix_gateway, sample_order):
        """Test order cancellation request message."""
        # Execute
        cancel_message = fix_gateway.create_cancel_request(
            original_client_order_id=sample_order.client_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
        )

        # Verify
        assert cancel_message.msg_type == FIXMessageType.ORDER_CANCEL_REQUEST
        assert (
            cancel_message.get_field(FIXField.ORIG_CL_ORD_ID)
            == sample_order.client_order_id
        )
        assert cancel_message.get_field(FIXField.SYMBOL) == sample_order.symbol
        assert FIXField.CL_ORD_ID in cancel_message.fields  # New cancel order ID

    def test_order_replace_request(self, fix_gateway, sample_order):
        """Test order modification/replace request."""
        # Execute
        replace_message = fix_gateway.create_replace_request(
            original_client_order_id=sample_order.client_order_id,
            symbol=sample_order.symbol,
            side=sample_order.side,
            new_quantity=Decimal("0.7"),
            new_price=Decimal("44500.00"),
        )

        # Verify
        assert replace_message.msg_type == FIXMessageType.ORDER_CANCEL_REPLACE_REQUEST
        assert (
            replace_message.get_field(FIXField.ORIG_CL_ORD_ID)
            == sample_order.client_order_id
        )
        assert replace_message.get_field(FIXField.ORDER_QTY) == "0.7"
        assert replace_message.get_field(FIXField.PRICE) == "44500.00"

    def test_fix_session_sequence_management(self, fix_gateway):
        """Test FIX session sequence number management."""
        # Setup
        session = fix_gateway.session

        # Execute
        initial_seq = session.next_outgoing_seq_num()
        next_seq = session.next_outgoing_seq_num()

        # Verify
        assert next_seq == initial_seq + 1
        assert session.get_outgoing_seq_num() == next_seq

    def test_reject_message_handling(self, fix_gateway):
        """Test handling of FIX reject messages."""
        # Setup
        reject = FIXMessage(
            msg_type=FIXMessageType.REJECT,
            fields={
                FIXField.REF_SEQ_NUM: "123",
                FIXField.TEXT: "Invalid symbol",
                FIXField.REF_MSG_TYPE: FIXMessageType.NEW_ORDER_SINGLE,
            },
        )

        # Execute
        rejection = fix_gateway.handle_reject(reject)

        # Verify
        assert rejection.ref_seq_num == 123
        assert rejection.reason == "Invalid symbol"
        assert rejection.ref_msg_type == FIXMessageType.NEW_ORDER_SINGLE

    def test_message_validation(self, fix_gateway):
        """Test FIX message validation."""
        # Setup - Invalid message (missing required fields)
        invalid_message = FIXMessage(
            msg_type=FIXMessageType.NEW_ORDER_SINGLE,
            fields={
                FIXField.SYMBOL: "BTC/USD"
                # Missing required fields like OrderQty, Side
            },
        )

        # Execute
        is_valid, errors = fix_gateway.validate_message(invalid_message)

        # Verify
        assert is_valid is False
        assert len(errors) > 0
        assert any("OrderQty" in error for error in errors)
        assert any("Side" in error for error in errors)

    def test_checksum_calculation(self, fix_gateway):
        """Test FIX message checksum calculation."""
        # Setup
        message_body = "8=FIX.4.4\x019=178\x0135=D\x0149=GENESIS\x0156=BROKER\x01"

        # Execute
        checksum = fix_gateway.calculate_checksum(message_body)

        # Verify
        assert len(checksum) == 3
        assert checksum.isdigit()

    def test_decimal_precision_in_fix_messages(self, fix_gateway):
        """Test that Decimal values are properly formatted in FIX messages."""
        # Setup
        order = Order(
            order_id="ORD_123",
            client_order_id="CLIENT_123",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("45678.987654321"),
        )

        # Execute
        fix_message = fix_gateway.order_to_fix(order)

        # Verify
        assert fix_message.get_field(FIXField.ORDER_QTY) == "0.123456789"
        assert fix_message.get_field(FIXField.PRICE) == "45678.987654321"

    def test_connection_state_management(self, fix_gateway):
        """Test FIX connection state transitions."""
        # Initial state
        assert fix_gateway.session.state == "DISCONNECTED"

        # Simulate connection
        fix_gateway.session.transition_to("CONNECTING")
        assert fix_gateway.session.state == "CONNECTING"

        # Simulate logon
        fix_gateway.session.transition_to("LOGGED_ON")
        assert fix_gateway.session.state == "LOGGED_ON"
        assert fix_gateway.session.is_active()

        # Simulate logout
        fix_gateway.session.transition_to("LOGGED_OUT")
        assert fix_gateway.session.state == "LOGGED_OUT"
        assert not fix_gateway.session.is_active()

    def test_resend_request_handling(self, fix_gateway):
        """Test handling of resend requests."""
        # Setup
        resend_request = FIXMessage(
            msg_type=FIXMessageType.RESEND_REQUEST,
            fields={FIXField.BEGIN_SEQ_NO: "100", FIXField.END_SEQ_NO: "105"},
        )

        # Execute
        messages_to_resend = fix_gateway.handle_resend_request(resend_request)

        # Verify
        assert messages_to_resend.begin_seq == 100
        assert messages_to_resend.end_seq == 105
        assert messages_to_resend.count == 6

    def test_market_data_request(self, fix_gateway):
        """Test market data subscription request."""
        # Execute
        md_request = fix_gateway.create_market_data_request(
            symbols=["BTC/USD", "ETH/USD"], subscription_type="SNAPSHOT_PLUS_UPDATES"
        )

        # Verify
        assert md_request.msg_type == FIXMessageType.MARKET_DATA_REQUEST
        assert "BTC/USD" in md_request.get_field(FIXField.SYMBOL)
        assert "ETH/USD" in md_request.get_field(FIXField.SYMBOL)
