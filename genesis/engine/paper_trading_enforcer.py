"""Paper trading enforcement for new execution methods.

Forces traders to practice new execution methods in simulation mode
before allowing live trading with real capital. Critical for preventing
costly mistakes during tier transitions.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.exceptions import StateError, ValidationError
from genesis.data.models_db import (
    PaperTradingSession,
    Session,
    TierTransition,
    get_session,
)

logger = structlog.get_logger(__name__)


class SessionStatus(Enum):
    """Paper trading session status."""
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Minimum requirements for paper trading by strategy
PAPER_TRADING_REQUIREMENTS = {
    'iceberg_orders': {
        'min_duration_hours': 24,
        'min_trades': 20,
        'min_success_rate': 0.70,
        'min_profitable_ratio': 0.55
    },
    'vwap_execution': {
        'min_duration_hours': 48,
        'min_trades': 30,
        'min_success_rate': 0.75,
        'min_profitable_ratio': 0.60
    },
    'market_making': {
        'min_duration_hours': 72,
        'min_trades': 50,
        'min_success_rate': 0.80,
        'min_profitable_ratio': 0.65
    },
    'statistical_arbitrage': {
        'min_duration_hours': 48,
        'min_trades': 40,
        'min_success_rate': 0.75,
        'min_profitable_ratio': 0.60
    }
}


@dataclass
class PaperTrade:
    """Represents a single paper trade."""
    trade_id: str
    session_id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal | None = None
    pnl: Decimal | None = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    execution_method: str = ""
    notes: str = ""

    @property
    def is_profitable(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl is not None and self.pnl > 0

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None and self.closed_at is not None


@dataclass
class SessionMetrics:
    """Metrics for a paper trading session."""
    session_id: str
    total_trades: int
    profitable_trades: int
    total_pnl: Decimal
    success_rate: Decimal
    profitable_ratio: Decimal
    average_pnl: Decimal
    max_drawdown: Decimal
    duration_hours: Decimal
    trades_per_hour: Decimal

    @property
    def meets_requirements(self) -> dict[str, bool]:
        """Check which requirements are met."""
        return {
            'trades': self.total_trades >= 20,
            'success_rate': self.success_rate >= Decimal('0.70'),
            'profitable_ratio': self.profitable_ratio >= Decimal('0.55'),
            'duration': self.duration_hours >= 24
        }


class PaperTradingEnforcer:
    """Enforces paper trading requirements before live trading."""

    def __init__(self, session: Session | None = None):
        """Initialize paper trading enforcer.
        
        Args:
            session: Optional database session
        """
        self.session = session or get_session()
        self._active_sessions: dict[str, list[PaperTrade]] = {}
        self._session_tasks: dict[str, asyncio.Task] = {}

    async def require_paper_trading(
        self,
        account_id: str,
        strategy: str,
        duration_hours: int,
        transition_id: str | None = None
    ) -> str:
        """Create and enforce paper trading requirement.
        
        Args:
            account_id: Account required to paper trade
            strategy: Strategy/execution method to practice
            duration_hours: Minimum duration required
            transition_id: Optional tier transition ID
            
        Returns:
            Session ID for tracking
            
        Raises:
            ValidationError: If requirements invalid
        """
        # Validate strategy
        if strategy not in PAPER_TRADING_REQUIREMENTS:
            raise ValidationError(f"Unknown strategy for paper trading: {strategy}")

        # Get requirements
        requirements = PAPER_TRADING_REQUIREMENTS[strategy]
        actual_duration = max(duration_hours, requirements['min_duration_hours'])

        # Create paper trading session
        session_id = str(uuid.uuid4())
        paper_session = PaperTradingSession(
            session_id=session_id,
            account_id=account_id,
            transition_id=transition_id,
            strategy_name=strategy,
            required_duration_hours=actual_duration,
            status=SessionStatus.ACTIVE.value,
            started_at=datetime.utcnow()
        )

        try:
            self.session.add(paper_session)
            self.session.commit()

            # Initialize in-memory tracking
            self._active_sessions[session_id] = []

            # Start monitoring task
            task = asyncio.create_task(
                self._monitor_session(session_id, actual_duration)
            )
            self._session_tasks[session_id] = task

            logger.info(
                "Paper trading session created",
                session_id=session_id,
                account_id=account_id,
                strategy=strategy,
                duration_hours=actual_duration
            )

            return session_id

        except Exception as e:
            logger.error(
                "Failed to create paper trading session",
                account_id=account_id,
                strategy=strategy,
                error=str(e)
            )
            self.session.rollback()
            raise

    async def record_paper_trade(
        self,
        session_id: str,
        trade: PaperTrade
    ) -> None:
        """Record a paper trade in the session.
        
        Args:
            session_id: Paper trading session ID
            trade: Paper trade to record
        """
        if session_id not in self._active_sessions:
            raise ValidationError(f"Session not found or inactive: {session_id}")

        # Add to in-memory tracking
        self._active_sessions[session_id].append(trade)

        # Update database metrics
        await self._update_session_metrics(session_id)

        logger.debug(
            "Paper trade recorded",
            session_id=session_id,
            trade_id=trade.trade_id,
            pnl=float(trade.pnl) if trade.pnl else None
        )

    async def close_paper_trade(
        self,
        session_id: str,
        trade_id: str,
        exit_price: Decimal,
        closed_at: datetime | None = None
    ) -> PaperTrade:
        """Close a paper trade and calculate P&L.
        
        Args:
            session_id: Session ID
            trade_id: Trade to close
            exit_price: Exit price
            closed_at: Optional close timestamp
            
        Returns:
            Updated paper trade
        """
        if session_id not in self._active_sessions:
            raise ValidationError(f"Session not found: {session_id}")

        # Find trade
        trade = None
        for t in self._active_sessions[session_id]:
            if t.trade_id == trade_id:
                trade = t
                break

        if not trade:
            raise ValidationError(f"Trade not found: {trade_id}")

        if trade.is_closed:
            raise StateError(f"Trade already closed: {trade_id}")

        # Calculate P&L
        trade.exit_price = exit_price
        trade.closed_at = closed_at or datetime.utcnow()

        if trade.side == 'BUY':
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SELL
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        # Update metrics
        await self._update_session_metrics(session_id)

        logger.info(
            "Paper trade closed",
            session_id=session_id,
            trade_id=trade_id,
            pnl=float(trade.pnl),
            is_profitable=trade.is_profitable
        )

        return trade

    async def get_session_metrics(self, session_id: str) -> SessionMetrics:
        """Get current metrics for a paper trading session.
        
        Args:
            session_id: Session to get metrics for
            
        Returns:
            SessionMetrics with current performance
        """
        # Get session from database
        paper_session = self.session.query(PaperTradingSession).filter_by(
            session_id=session_id
        ).first()

        if not paper_session:
            raise ValidationError(f"Session not found: {session_id}")

        # Get trades
        trades = self._active_sessions.get(session_id, [])
        closed_trades = [t for t in trades if t.is_closed]

        if not closed_trades:
            # No closed trades yet
            duration = (datetime.utcnow() - paper_session.started_at).total_seconds() / 3600
            return SessionMetrics(
                session_id=session_id,
                total_trades=0,
                profitable_trades=0,
                total_pnl=Decimal('0'),
                success_rate=Decimal('0'),
                profitable_ratio=Decimal('0'),
                average_pnl=Decimal('0'),
                max_drawdown=Decimal('0'),
                duration_hours=Decimal(str(duration)),
                trades_per_hour=Decimal('0')
            )

        # Calculate metrics
        total_trades = len(closed_trades)
        profitable_trades = len([t for t in closed_trades if t.is_profitable])
        total_pnl = sum(t.pnl for t in closed_trades if t.pnl)

        success_rate = Decimal(profitable_trades) / Decimal(total_trades)
        profitable_ratio = success_rate  # Same as success rate for closed trades
        average_pnl = total_pnl / Decimal(total_trades)

        # Calculate max drawdown
        cumulative_pnl = Decimal('0')
        peak = Decimal('0')
        max_drawdown = Decimal('0')

        for trade in sorted(closed_trades, key=lambda t: t.closed_at):
            cumulative_pnl += trade.pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate duration
        duration = (datetime.utcnow() - paper_session.started_at).total_seconds() / 3600
        trades_per_hour = Decimal(total_trades) / Decimal(str(duration)) if duration > 0 else Decimal('0')

        return SessionMetrics(
            session_id=session_id,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            total_pnl=total_pnl,
            success_rate=success_rate,
            profitable_ratio=profitable_ratio,
            average_pnl=average_pnl,
            max_drawdown=max_drawdown,
            duration_hours=Decimal(str(duration)),
            trades_per_hour=trades_per_hour
        )

    async def check_session_completion(self, session_id: str) -> tuple[bool, list[str]]:
        """Check if paper trading session meets completion requirements.
        
        Args:
            session_id: Session to check
            
        Returns:
            Tuple of (is_complete, failure_reasons)
        """
        # Get session
        paper_session = self.session.query(PaperTradingSession).filter_by(
            session_id=session_id
        ).first()

        if not paper_session:
            raise ValidationError(f"Session not found: {session_id}")

        # Get requirements
        requirements = PAPER_TRADING_REQUIREMENTS.get(
            paper_session.strategy_name, {}
        )

        # Get metrics
        metrics = await self.get_session_metrics(session_id)

        failure_reasons = []

        # Check duration
        if metrics.duration_hours < requirements.get('min_duration_hours', 0):
            failure_reasons.append(
                f"Insufficient duration: {float(metrics.duration_hours):.1f} hours "
                f"(required: {requirements['min_duration_hours']})"
            )

        # Check trade count
        if metrics.total_trades < requirements.get('min_trades', 0):
            failure_reasons.append(
                f"Insufficient trades: {metrics.total_trades} "
                f"(required: {requirements['min_trades']})"
            )

        # Check success rate
        min_success = Decimal(str(requirements.get('min_success_rate', 0)))
        if metrics.success_rate < min_success:
            failure_reasons.append(
                f"Success rate too low: {float(metrics.success_rate):.2%} "
                f"(required: {float(min_success):.2%})"
            )

        # Check profitability ratio
        min_profitable = Decimal(str(requirements.get('min_profitable_ratio', 0)))
        if metrics.profitable_ratio < min_profitable:
            failure_reasons.append(
                f"Profitability ratio too low: {float(metrics.profitable_ratio):.2%} "
                f"(required: {float(min_profitable):.2%})"
            )

        is_complete = len(failure_reasons) == 0

        return is_complete, failure_reasons

    async def complete_session(
        self,
        session_id: str,
        force: bool = False
    ) -> SessionMetrics:
        """Complete a paper trading session.
        
        Args:
            session_id: Session to complete
            force: Force completion even if requirements not met
            
        Returns:
            Final session metrics
            
        Raises:
            ValidationError: If requirements not met and not forced
        """
        # Check completion requirements
        is_complete, failure_reasons = await self.check_session_completion(session_id)

        if not is_complete and not force:
            raise ValidationError(
                f"Session requirements not met: {'; '.join(failure_reasons)}"
            )

        # Get final metrics
        metrics = await self.get_session_metrics(session_id)

        # Update database
        paper_session = self.session.query(PaperTradingSession).filter_by(
            session_id=session_id
        ).first()

        if paper_session:
            paper_session.completed_at = datetime.utcnow()
            paper_session.actual_duration_hours = metrics.duration_hours
            paper_session.success_rate = metrics.success_rate
            paper_session.total_trades = metrics.total_trades
            paper_session.profitable_trades = metrics.profitable_trades
            paper_session.status = SessionStatus.COMPLETED.value if is_complete else SessionStatus.FAILED.value

            # Update transition if linked
            if paper_session.transition_id:
                transition = self.session.query(TierTransition).filter_by(
                    transition_id=paper_session.transition_id
                ).first()

                if transition and is_complete:
                    transition.paper_trading_completed = True
                    transition.updated_at = datetime.utcnow()

            self.session.commit()

            logger.info(
                "Paper trading session completed",
                session_id=session_id,
                success=is_complete,
                total_trades=metrics.total_trades,
                success_rate=float(metrics.success_rate)
            )

        # Clean up in-memory tracking
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        # Cancel monitoring task
        if session_id in self._session_tasks:
            self._session_tasks[session_id].cancel()
            del self._session_tasks[session_id]

        return metrics

    async def cancel_session(self, session_id: str, reason: str) -> None:
        """Cancel a paper trading session.
        
        Args:
            session_id: Session to cancel
            reason: Reason for cancellation
        """
        paper_session = self.session.query(PaperTradingSession).filter_by(
            session_id=session_id
        ).first()

        if paper_session:
            paper_session.status = SessionStatus.CANCELLED.value
            paper_session.completed_at = datetime.utcnow()
            self.session.commit()

        # Clean up
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        if session_id in self._session_tasks:
            self._session_tasks[session_id].cancel()
            del self._session_tasks[session_id]

        logger.info(
            "Paper trading session cancelled",
            session_id=session_id,
            reason=reason
        )

    async def _monitor_session(
        self,
        session_id: str,
        duration_hours: int
    ) -> None:
        """Monitor paper trading session for timeout.
        
        Args:
            session_id: Session to monitor
            duration_hours: Maximum duration
        """
        try:
            # Wait for duration
            await asyncio.sleep(duration_hours * 3600)

            # Check if still active
            if session_id in self._active_sessions:
                # Auto-complete session
                try:
                    await self.complete_session(session_id, force=False)
                except ValidationError:
                    # Requirements not met, mark as failed
                    await self.complete_session(session_id, force=True)

        except asyncio.CancelledError:
            # Session completed or cancelled early
            pass
        except Exception as e:
            logger.error(
                "Error monitoring paper trading session",
                session_id=session_id,
                error=str(e)
            )

    async def _update_session_metrics(self, session_id: str) -> None:
        """Update session metrics in database.
        
        Args:
            session_id: Session to update
        """
        try:
            metrics = await self.get_session_metrics(session_id)

            paper_session = self.session.query(PaperTradingSession).filter_by(
                session_id=session_id
            ).first()

            if paper_session:
                paper_session.total_trades = metrics.total_trades
                paper_session.profitable_trades = metrics.profitable_trades
                paper_session.success_rate = metrics.success_rate
                paper_session.actual_duration_hours = metrics.duration_hours

                self.session.commit()

        except Exception as e:
            logger.error(
                "Failed to update session metrics",
                session_id=session_id,
                error=str(e)
            )
            self.session.rollback()

    def get_active_sessions(self, account_id: str | None = None) -> list[str]:
        """Get list of active paper trading sessions.
        
        Args:
            account_id: Optional filter by account
            
        Returns:
            List of active session IDs
        """
        query = self.session.query(PaperTradingSession).filter_by(
            status=SessionStatus.ACTIVE.value
        )

        if account_id:
            query = query.filter_by(account_id=account_id)

        sessions = query.all()
        return [s.session_id for s in sessions]

    async def cleanup(self) -> None:
        """Clean up all active sessions and tasks."""
        # Cancel all monitoring tasks
        for task in self._session_tasks.values():
            task.cancel()

        # Clear tracking
        self._active_sessions.clear()
        self._session_tasks.clear()
