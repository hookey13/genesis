"""
Integration tests for complete data flow.

Tests order lifecycle, trade processing, position updates, and PnL tracking
with real database transactions.
"""

import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from genesis.data.models import Base, configure_sqlite_pragmas
from genesis.data.repositories import (
    CandleRepository,
    InstrumentRepository,
    OrderRepository,
    SessionRepository,
    TradeRepository,
)
from genesis.data.services import PositionService


@pytest.fixture
def temp_database():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def db_engine(temp_database):
    """Create database engine with test database."""
    engine = create_engine(temp_database)

    # Configure SQLite for production settings
    with engine.connect() as conn:
        configure_sqlite_pragmas(conn)

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Create database session for testing."""
    Session = sessionmaker(bind=db_engine)
    session = Session()

    yield session

    session.close()


@pytest.mark.integration
class TestCompleteOrderFlow:
    """Test complete order lifecycle from creation to PnL realization."""

    def test_order_create_fill_position_update(self, db_session):
        """Test full flow: create order → record trades → update position → track PnL."""
        # Initialize repositories and services
        session_repo = SessionRepository(db_session)
        order_repo = OrderRepository(db_session)
        trade_repo = TradeRepository(db_session)
        position_service = PositionService(db_session)
        instrument_repo = InstrumentRepository(db_session)

        # Step 1: Start trading session
        trading_session_id = session_repo.start("Integration test session")
        assert trading_session_id is not None

        # Step 2: Setup instrument
        instrument_repo.upsert(
            symbol="BTC/USDT",
            base="BTC",
            quote="USDT",
            tick_size=Decimal("0.01"),
            lot_step=Decimal("0.001"),
            min_notional=Decimal("10.0"),
        )

        # Step 3: Create buy order
        order_data = {
            "client_order_id": "test-order-001",
            "session_id": trading_session_id,
            "symbol": "BTC/USDT",
            "side": "buy",
            "type": "limit",
            "qty": Decimal("1.5"),
            "price": Decimal("50000"),
        }
        order = order_repo.create(order_data)
        assert order.status == "new"
        assert order.filled_qty == Decimal("0")

        # Step 4: Record first partial fill
        trade1_data = {
            "exchange_trade_id": "binance-trade-001",
            "order_id": order.id,
            "symbol": "BTC/USDT",
            "side": "buy",
            "qty": Decimal("0.5"),
            "price": Decimal("49900"),
            "fee_amount": Decimal("10"),
            "trade_time": datetime.now(UTC),
        }
        trade1 = trade_repo.record_trade(trade1_data)
        assert trade1 is not None

        # Apply trade to order and position
        order_repo.append_fill(order.client_order_id, trade1)
        position, pnl = position_service.apply_trade(trade1)

        # Verify order partially filled
        updated_order = order_repo.get_by_client_id("test-order-001")
        assert updated_order.status == "partially_filled"
        assert updated_order.filled_qty == Decimal("0.5")
        assert updated_order.avg_price == Decimal("49900")

        # Verify position opened
        assert position.qty == Decimal("0.5")
        assert position.avg_entry_price == Decimal("49900")
        assert pnl == Decimal("-10")  # Just the fee

        # Step 5: Record second fill completing the order
        trade2_data = {
            "exchange_trade_id": "binance-trade-002",
            "order_id": order.id,
            "symbol": "BTC/USDT",
            "side": "buy",
            "qty": Decimal("1.0"),
            "price": Decimal("50050"),
            "fee_amount": Decimal("20"),
            "trade_time": datetime.now(UTC),
        }
        trade2 = trade_repo.record_trade(trade2_data)

        order_repo.append_fill(order.client_order_id, trade2)
        position, pnl = position_service.apply_trade(trade2)

        # Verify order fully filled
        updated_order = order_repo.get_by_client_id("test-order-001")
        assert updated_order.status == "filled"
        assert updated_order.filled_qty == Decimal("1.5")
        # Weighted avg: (0.5 * 49900 + 1.0 * 50050) / 1.5 = 50000
        assert updated_order.avg_price == Decimal("50000")

        # Verify position increased
        assert position.qty == Decimal("1.5")
        assert position.avg_entry_price == Decimal("50000")

        # Step 6: Create sell order to realize PnL
        sell_order_data = {
            "client_order_id": "test-order-002",
            "session_id": trading_session_id,
            "symbol": "BTC/USDT",
            "side": "sell",
            "type": "limit",
            "qty": Decimal("1.5"),
            "price": Decimal("52000"),
        }
        sell_order = order_repo.create(sell_order_data)

        # Record sell trade
        sell_trade_data = {
            "exchange_trade_id": "binance-trade-003",
            "order_id": sell_order.id,
            "symbol": "BTC/USDT",
            "side": "sell",
            "qty": Decimal("1.5"),
            "price": Decimal("52000"),
            "fee_amount": Decimal("30"),
            "trade_time": datetime.now(UTC),
        }
        sell_trade = trade_repo.record_trade(sell_trade_data)

        order_repo.append_fill(sell_order.client_order_id, sell_trade)
        position, realized_pnl = position_service.apply_trade(sell_trade)

        # Verify position closed
        assert position.qty == Decimal("0")
        assert position.avg_entry_price == Decimal("0")

        # Calculate expected PnL: 1.5 * (52000 - 50000) - 30 = 2970
        assert realized_pnl == Decimal("2970")

        # Step 7: End session and verify
        session_repo.end(trading_session_id, "shutdown", "Test complete")

        # Verify all data persisted
        from genesis.data.models import Order

        all_orders = db_session.query(Order).count()
        assert all_orders == 2

        all_trades = trade_repo.list_for_order(order.id)
        assert len(all_trades) == 2

    def test_idempotent_trade_ingestion(self, db_session):
        """Test that duplicate trades are ignored (idempotency)."""
        trade_repo = TradeRepository(db_session)

        # Record trade first time
        trade_data = {
            "exchange_trade_id": "duplicate-test-001",
            "symbol": "ETH/USDT",
            "side": "buy",
            "qty": Decimal("5.0"),
            "price": Decimal("3000"),
            "fee_amount": Decimal("5"),
            "trade_time": datetime.now(UTC),
        }

        trade1 = trade_repo.record_trade(trade_data)
        assert trade1 is not None

        # Try to record same trade again
        trade2 = trade_repo.record_trade(trade_data)
        assert trade2 is None  # Should return None for duplicate

        # Verify only one trade in database
        from genesis.data.models import Trade

        trades = (
            db_session.query(Trade)
            .filter_by(exchange_trade_id="duplicate-test-001")
            .all()
        )
        assert len(trades) == 1

    def test_candle_upsert(self, db_session):
        """Test candle bulk upsert with conflict resolution."""
        candle_repo = CandleRepository(db_session)

        # Initial candles
        candles_batch1 = [
            {
                "open_time": datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                "open": Decimal("40000"),
                "high": Decimal("40500"),
                "low": Decimal("39500"),
                "close": Decimal("40200"),
                "volume": Decimal("100"),
            },
            {
                "open_time": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                "open": Decimal("40200"),
                "high": Decimal("40300"),
                "low": Decimal("40100"),
                "close": Decimal("40250"),
                "volume": Decimal("50"),
            },
        ]

        candle_repo.upsert_bulk("BTC/USDT", "1m", candles_batch1)

        # Get latest candle
        latest = candle_repo.latest("BTC/USDT", "1m")
        assert latest.close == Decimal("40250")

        # Update with overlapping candles (should update existing)
        candles_batch2 = [
            {
                "open_time": datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                "open": Decimal("40200"),
                "high": Decimal("40350"),  # Updated high
                "low": Decimal("40100"),
                "close": Decimal("40300"),  # Updated close
                "volume": Decimal("75"),  # Updated volume
            },
            {
                "open_time": datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                "open": Decimal("40300"),
                "high": Decimal("40400"),
                "low": Decimal("40250"),
                "close": Decimal("40350"),
                "volume": Decimal("60"),
            },
        ]

        candle_repo.upsert_bulk("BTC/USDT", "1m", candles_batch2)

        # Verify update
        from genesis.data.models import Candle

        all_candles = (
            db_session.query(Candle)
            .filter_by(symbol="BTC/USDT", timeframe="1m")
            .order_by(Candle.open_time)
            .all()
        )

        assert len(all_candles) == 3  # 2 original + 1 new
        assert all_candles[1].high == Decimal("40350")  # Updated
        assert all_candles[1].close == Decimal("40300")  # Updated
        assert all_candles[1].volume == Decimal("75")  # Updated

    def test_position_reversal_flow(self, db_session):
        """Test position reversal from long to short."""
        position_service = PositionService(db_session)
        instrument_repo = InstrumentRepository(db_session)

        # Setup instrument
        instrument_repo.upsert(
            symbol="ETH/USDT",
            base="ETH",
            quote="USDT",
            tick_size=Decimal("0.01"),
            lot_step=Decimal("0.01"),
            min_notional=Decimal("10.0"),
        )

        # Create mock trade helper
        def create_trade(side, qty, price, fee="1"):
            from genesis.data.models import Trade

            trade = Trade(
                id="test-" + str(datetime.now().timestamp()),
                exchange_trade_id="ex-" + str(datetime.now().timestamp()),
                symbol="ETH/USDT",
                side=side,
                qty=Decimal(str(qty)),
                price=Decimal(str(price)),
                fee_ccy="USDT",
                fee_amount=Decimal(str(fee)),
                trade_time=datetime.now(UTC),
            )
            return trade

        # Open long position
        trade1 = create_trade("buy", 10, 3000, 3)
        position, pnl = position_service.apply_trade(trade1)
        assert position.qty == Decimal("10")
        assert pnl == Decimal("-3")  # Fee

        # Reverse to short (sell 15 ETH)
        trade2 = create_trade("sell", 15, 3100, 5)
        position, pnl = position_service.apply_trade(trade2)

        # Should close 10 long (profit) and open 5 short
        assert position.qty == Decimal("-5")  # Short 5 ETH
        assert position.avg_entry_price == Decimal("3100")
        # PnL: 10 * (3100 - 3000) - 5 = 995
        assert pnl == Decimal("995")

        # Cover short with profit
        trade3 = create_trade("buy", 5, 3050, 2)
        position, pnl = position_service.apply_trade(trade3)

        assert position.qty == Decimal("0")  # Flat
        # PnL: 5 * (3100 - 3050) - 2 = 248
        assert pnl == Decimal("248")

        # Total realized PnL
        assert position.realised_pnl == Decimal("1240")  # -3 + 995 + 248

    def test_migration_and_schema(self, db_engine):
        """Test that all expected tables and constraints exist."""
        # Get table names
        with db_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            )
            tables = [row[0] for row in result]

        expected_tables = [
            "candles",
            "inbox_events",
            "instruments",
            "journal",
            "orders",
            "outbox_events",
            "pnl_ledger",
            "positions",
            "sessions",
            "trades",
        ]

        for table in expected_tables:
            assert table in tables

        # Verify some key constraints exist (check_constraints in SQLite)
        with db_engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'"
                )
            )
            create_sql = result.fetchone()[0]

            # Check for key constraints in CREATE TABLE statement
            assert "CHECK" in create_sql  # Has check constraints
            assert "symbol" in create_sql.lower()
            assert "status" in create_sql.lower()
