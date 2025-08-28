"""
Repository implementations for data access.

All repositories are synchronous and transaction-aware.
Financial values use Decimal throughout for precision.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import IntegrityError

from genesis.data.models import (
    Session,
    Order,
    Trade,
    Position,
    PnLLedger,
    Instrument,
    Candle,
    Journal,
    OutboxEvent,
    InboxEvent,
)
from genesis.data import metrics

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations."""

    def __init__(self, session: DBSession):
        self.session = session

    def save(self, entity):
        """Save entity to database."""
        self.session.add(entity)
        return entity

    def commit(self):
        """Commit current transaction."""
        self.session.commit()

    def rollback(self):
        """Rollback current transaction."""
        self.session.rollback()


class SessionRepository(BaseRepository):
    """Repository for trading sessions."""

    def start(self, notes: Optional[str] = None) -> str:
        """Start a new trading session."""
        session = Session(
            id=str(uuid4()),
            started_at=datetime.now(timezone.utc),
            state="running",
            notes=notes,
        )
        self.save(session)
        self.commit()
        logger.info(f"Started session {session.id}")
        return session.id

    def end(self, session_id: str, state: str, reason: Optional[str] = None):
        """End a trading session."""
        session = self.session.query(Session).filter_by(id=session_id).first()
        if session:
            session.ended_at = datetime.now(timezone.utc)
            session.state = state
            session.reason = reason
            self.commit()
            logger.info(f"Ended session {session_id} with state={state}")

    def get_active(self) -> Optional[Session]:
        """Get current active session."""
        return (
            self.session.query(Session)
            .filter_by(state="running")
            .order_by(Session.started_at.desc())
            .first()
        )


class InstrumentRepository(BaseRepository):
    """Repository for instrument configuration."""

    def upsert(
        self,
        symbol: str,
        base: str,
        quote: str,
        tick_size: Decimal,
        lot_step: Decimal,
        min_notional: Decimal,
    ):
        """Insert or update instrument configuration."""
        instrument = self.session.query(Instrument).filter_by(symbol=symbol).first()

        if instrument:
            instrument.base = base
            instrument.quote = quote
            instrument.tick_size = tick_size
            instrument.lot_step = lot_step
            instrument.min_notional = min_notional
            instrument.updated_at = datetime.now(timezone.utc)
        else:
            instrument = Instrument(
                symbol=symbol,
                base=base,
                quote=quote,
                tick_size=tick_size,
                lot_step=lot_step,
                min_notional=min_notional,
            )
            self.save(instrument)

        self.commit()
        return instrument

    def get(self, symbol: str) -> Optional[Instrument]:
        """Get instrument by symbol."""
        return self.session.query(Instrument).filter_by(symbol=symbol).first()


class OrderRepository(BaseRepository):
    """Repository for order management."""

    @metrics.track_db_operation("create_order")
    def create(self, order_data: Dict) -> Order:
        """Create new order with NEW status."""
        order = Order(
            id=str(uuid4()),
            client_order_id=order_data["client_order_id"],
            session_id=order_data["session_id"],
            symbol=order_data["symbol"],
            side=order_data["side"],
            type=order_data["type"],
            qty=Decimal(str(order_data["qty"])),
            filled_qty=Decimal("0"),
            price=(
                Decimal(str(order_data.get("price", 0)))
                if order_data.get("price")
                else None
            ),
            time_in_force=order_data.get("time_in_force"),
            status="new",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.save(order)
        self.commit()

        # Track metrics
        metrics.track_order_created(order.symbol, order.side, order.type)

        logger.info(f"Created order {order.client_order_id}")
        return order

    def set_status(self, client_order_id: str, status: str, **fields):
        """Update order status and optional fields."""
        order = (
            self.session.query(Order).filter_by(client_order_id=client_order_id).first()
        )

        if not order:
            logger.warning(f"Order {client_order_id} not found")
            return None

        order.status = status
        order.updated_at = datetime.now(timezone.utc)

        # Update optional fields
        if "avg_price" in fields:
            order.avg_price = Decimal(str(fields["avg_price"]))
        if "exchange_order_id" in fields:
            order.exchange_order_id = fields["exchange_order_id"]
        if "last_error" in fields:
            order.last_error = fields["last_error"]
        if "filled_qty" in fields:
            order.filled_qty = Decimal(str(fields["filled_qty"]))

        self.commit()
        logger.info(f"Updated order {client_order_id} status to {status}")
        return order

    def append_fill(self, client_order_id: str, trade: Trade) -> Order:
        """Append trade fill to order and update filled quantity."""
        order = (
            self.session.query(Order)
            .filter_by(client_order_id=client_order_id)
            .with_for_update()
            .first()
        )

        if not order:
            raise ValueError(f"Order {client_order_id} not found")

        # Add trade
        trade.order_id = order.id
        self.save(trade)

        # Update order filled quantity
        order.filled_qty += trade.qty

        # Calculate weighted average price
        if order.avg_price:
            old_value = order.avg_price * (order.filled_qty - trade.qty)
            new_value = trade.price * trade.qty
            order.avg_price = (old_value + new_value) / order.filled_qty
        else:
            order.avg_price = trade.price

        # Update status
        if order.filled_qty >= order.qty:
            order.status = "filled"
            metrics.track_order_filled(order.symbol, order.side)
        else:
            order.status = "partially_filled"

        order.updated_at = datetime.now(timezone.utc)

        # Write to PnL ledger
        pnl_entry = PnLLedger(
            id=str(uuid4()),
            session_id=order.session_id,
            event_type="trade",
            symbol=order.symbol,
            amount_quote=trade.qty * trade.price * (-1 if order.side == "buy" else 1),
            at=trade.trade_time,
            ref_type="trade",
            ref_id=trade.id,
        )
        self.save(pnl_entry)

        # Fee entry
        if trade.fee_amount > 0:
            fee_entry = PnLLedger(
                id=str(uuid4()),
                session_id=order.session_id,
                event_type="fee",
                symbol=order.symbol,
                amount_quote=-trade.fee_amount,
                at=trade.trade_time,
                ref_type="trade",
                ref_id=trade.id,
            )
            self.save(fee_entry)

        self.commit()
        logger.info(
            f"Appended fill to order {client_order_id}: {trade.qty} @ {trade.price}"
        )
        return order

    def get_open_by_symbol(self, symbol: str) -> List[Order]:
        """Get all open orders for a symbol."""
        return (
            self.session.query(Order)
            .filter(
                and_(
                    Order.symbol == symbol,
                    Order.status.in_(["new", "partially_filled"]),
                )
            )
            .all()
        )

    def get_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        return (
            self.session.query(Order).filter_by(client_order_id=client_order_id).first()
        )


class TradeRepository(BaseRepository):
    """Repository for trade records."""

    @metrics.track_db_operation("record_trade")
    def record_trade(self, trade_data: Dict) -> Optional[Trade]:
        """Record trade with idempotency on exchange_trade_id."""
        # Check for duplicate
        existing = (
            self.session.query(Trade)
            .filter_by(exchange_trade_id=trade_data["exchange_trade_id"])
            .first()
        )

        if existing:
            metrics.track_trade_ingested(
                trade_data["symbol"], trade_data["side"], 0, is_duplicate=True
            )
            logger.debug(f"Trade {trade_data['exchange_trade_id']} already exists")
            return None

        trade = Trade(
            id=str(uuid4()),
            exchange_trade_id=trade_data["exchange_trade_id"],
            order_id=trade_data.get("order_id"),
            symbol=trade_data["symbol"],
            side=trade_data["side"],
            qty=Decimal(str(trade_data["qty"])),
            price=Decimal(str(trade_data["price"])),
            fee_ccy=trade_data.get("fee_ccy", "USDT"),
            fee_amount=Decimal(str(trade_data.get("fee_amount", 0))),
            trade_time=trade_data.get("trade_time", datetime.now(timezone.utc)),
        )

        self.save(trade)
        self.commit()

        # Track metrics
        volume_quote = float(trade.qty * trade.price)
        metrics.track_trade_ingested(trade.symbol, trade.side, volume_quote)

        logger.info(f"Recorded trade {trade.exchange_trade_id}")
        return trade

    def list_for_order(self, order_id: str) -> List[Trade]:
        """List all trades for an order."""
        return self.session.query(Trade).filter_by(order_id=order_id).all()


class PositionRepository(BaseRepository):
    """Repository for position tracking."""

    def get_or_create(self, symbol: str) -> Position:
        """Get existing position or create new one."""
        position = self.session.query(Position).filter_by(symbol=symbol).first()

        if not position:
            position = Position(
                symbol=symbol,
                qty=Decimal("0"),
                avg_entry_price=Decimal("0"),
                realised_pnl=Decimal("0"),
                updated_at=datetime.now(timezone.utc),
            )
            self.save(position)

        return position

    def update(
        self,
        symbol: str,
        qty: Decimal,
        avg_entry_price: Decimal,
        realised_pnl_delta: Decimal = Decimal("0"),
    ):
        """Update position state."""
        position = self.get_or_create(symbol)

        position.qty = qty
        position.avg_entry_price = avg_entry_price
        position.realised_pnl += realised_pnl_delta
        position.updated_at = datetime.now(timezone.utc)

        self.commit()
        logger.info(f"Updated position {symbol}: qty={qty}, avg={avg_entry_price}")
        return position


class CandleRepository(BaseRepository):
    """Repository for OHLCV candle data."""

    @metrics.track_db_operation("upsert_candles")
    def upsert_bulk(self, symbol: str, timeframe: str, candles: List[Dict]):
        """Bulk upsert candle data with conflict resolution."""
        for candle_data in candles:
            existing = (
                self.session.query(Candle)
                .filter_by(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=candle_data["open_time"],
                )
                .first()
            )

            if existing:
                # Update existing candle
                existing.open = Decimal(str(candle_data["open"]))
                existing.high = Decimal(str(candle_data["high"]))
                existing.low = Decimal(str(candle_data["low"]))
                existing.close = Decimal(str(candle_data["close"]))
                existing.volume = Decimal(str(candle_data["volume"]))
            else:
                # Insert new candle
                candle = Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    open_time=candle_data["open_time"],
                    open=Decimal(str(candle_data["open"])),
                    high=Decimal(str(candle_data["high"])),
                    low=Decimal(str(candle_data["low"])),
                    close=Decimal(str(candle_data["close"])),
                    volume=Decimal(str(candle_data["volume"])),
                )
                self.save(candle)

        self.commit()

        # Track metrics
        metrics.track_candles_upserted(symbol, timeframe, len(candles))

        logger.info(f"Upserted {len(candles)} candles for {symbol} {timeframe}")

    def latest(self, symbol: str, timeframe: str) -> Optional[Candle]:
        """Get most recent candle."""
        return (
            self.session.query(Candle)
            .filter_by(symbol=symbol, timeframe=timeframe)
            .order_by(Candle.open_time.desc())
            .first()
        )

    def get_range(
        self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime
    ) -> List[Candle]:
        """Get candles in time range."""
        return (
            self.session.query(Candle)
            .filter(
                and_(
                    Candle.symbol == symbol,
                    Candle.timeframe == timeframe,
                    Candle.open_time >= start_time,
                    Candle.open_time <= end_time,
                )
            )
            .order_by(Candle.open_time)
            .all()
        )


class JournalRepository(BaseRepository):
    """Repository for system journal."""

    def log(self, level: str, category: str, message: str, ctx: Optional[Dict] = None):
        """Write journal entry."""
        entry = Journal(
            id=str(uuid4()),
            at=datetime.now(timezone.utc),
            level=level,
            category=category,
            message=message,
            ctx=ctx,
        )
        self.save(entry)
        self.commit()


class EventRepository(BaseRepository):
    """Repository for event inbox/outbox patterns."""

    def publish_event(self, event_type: str, payload: Dict) -> str:
        """Publish event to outbox."""
        event = OutboxEvent(
            id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload,
        )
        self.save(event)
        self.commit()

        # Track metrics
        metrics.track_event_published(event_type)

        return event.id

    def consume_event(self, dedupe_key: str, event_type: str, payload: Dict) -> bool:
        """Consume event with idempotency check."""
        # Check if already processed
        existing = (
            self.session.query(InboxEvent).filter_by(dedupe_key=dedupe_key).first()
        )

        if existing:
            metrics.track_event_consumed(event_type, is_duplicate=True)
            return False  # Already processed

        # Record consumption
        event = InboxEvent(
            id=str(uuid4()),
            dedupe_key=dedupe_key,
            created_at=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload,
            consumed_at=datetime.now(timezone.utc),
        )
        self.save(event)
        self.commit()

        # Track metrics
        metrics.track_event_consumed(event_type, is_duplicate=False)

        return True

    def get_unprocessed_events(self, limit: int = 100) -> List[OutboxEvent]:
        """Get unprocessed outbox events."""
        return (
            self.session.query(OutboxEvent)
            .filter(OutboxEvent.processed_at.is_(None))
            .order_by(OutboxEvent.created_at)
            .limit(limit)
            .all()
        )

    def mark_processed(self, event_id: str):
        """Mark outbox event as processed."""
        event = self.session.query(OutboxEvent).filter_by(id=event_id).first()
        if event:
            event.processed_at = datetime.now(timezone.utc)
            self.commit()
