"""Paper trading profit validation for production readiness."""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ValidationResult:
    """Standardized validation result."""
    
    def __init__(
        self,
        check_id: str,
        status: str,
        message: str,
        evidence: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.check_id = check_id
        self.status = status
        self.message = message
        self.evidence = evidence
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "evidence": self.evidence,
            "metadata": self.metadata
        }


class CheckStatus:
    """Validation check status constants."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class PaperTradingValidator:
    """Validates paper trading performance requirements."""
    
    PROFIT_REQUIREMENT = Decimal("10000.00")  # $10k USD
    MIN_TRADING_DAYS = 3  # Consecutive profitable days
    MIN_TRADES = 100  # Minimum trades to be statistically significant
    MAX_DRAWDOWN = Decimal("0.20")  # 20% maximum drawdown
    MIN_WIN_RATE = Decimal("0.55")  # 55% minimum win rate
    
    def __init__(self):
        """Initialize paper trading validator."""
        self.trading_log = Path(".genesis/logs/paper_trading.json")
        self.trades_db = Path(".genesis/data/genesis.db")
        
    async def validate(self) -> Dict[str, Any]:
        """Run validation following existing validator pattern."""
        try:
            # Load paper trading history
            trading_history = await self._load_trading_history()
            
            if not trading_history:
                return ValidationResult(
                    check_id="BIZ-001",
                    status=CheckStatus.FAILED,
                    message="No paper trading history found",
                    evidence={"trades": 0},
                    metadata={"requirement": "Complete paper trading test"}
                ).to_dict()
            
            # Calculate total P&L
            total_pnl = sum(Decimal(str(trade.get("pnl", 0))) for trade in trading_history)
            
            # Check profit requirement
            if total_pnl < self.PROFIT_REQUIREMENT:
                return ValidationResult(
                    check_id="BIZ-001",
                    status=CheckStatus.FAILED,
                    message=f"Paper trading profit ${total_pnl:.2f} < required ${self.PROFIT_REQUIREMENT:.2f}",
                    evidence={"total_pnl": str(total_pnl), "trades": len(trading_history)},
                    metadata={"requirement": str(self.PROFIT_REQUIREMENT)}
                ).to_dict()
            
            # Check consecutive profitable days
            profitable_days = await self._check_consecutive_profitable_days(trading_history)
            if profitable_days < self.MIN_TRADING_DAYS:
                return ValidationResult(
                    check_id="BIZ-001",
                    status=CheckStatus.FAILED,
                    message=f"Only {profitable_days} consecutive profitable days, need {self.MIN_TRADING_DAYS}",
                    evidence={"profitable_days": profitable_days},
                    metadata={"requirement": self.MIN_TRADING_DAYS}
                ).to_dict()
            
            # Validate trade count
            if len(trading_history) < self.MIN_TRADES:
                return ValidationResult(
                    check_id="BIZ-001",
                    status=CheckStatus.WARNING,
                    message=f"Only {len(trading_history)} trades, recommend {self.MIN_TRADES}+ for significance",
                    evidence={"trade_count": len(trading_history)},
                    metadata={"recommended": self.MIN_TRADES}
                ).to_dict()
            
            # Calculate and validate metrics
            metrics = await self._calculate_metrics(trading_history)
            
            # Check win rate
            if metrics["win_rate"] < self.MIN_WIN_RATE:
                return ValidationResult(
                    check_id="BIZ-001",
                    status=CheckStatus.FAILED,
                    message=f"Win rate {metrics['win_rate']:.1%} < required {self.MIN_WIN_RATE:.1%}",
                    evidence={"win_rate": float(metrics["win_rate"])},
                    metadata={"requirement": float(self.MIN_WIN_RATE)}
                ).to_dict()
            
            # Check drawdown
            if metrics["max_drawdown"] > self.MAX_DRAWDOWN:
                return ValidationResult(
                    check_id="BIZ-001",
                    status=CheckStatus.FAILED,
                    message=f"Max drawdown {metrics['max_drawdown']:.1%} > limit {self.MAX_DRAWDOWN:.1%}",
                    evidence={"max_drawdown": float(metrics["max_drawdown"])},
                    metadata={"limit": float(self.MAX_DRAWDOWN)}
                ).to_dict()
            
            # Generate performance report
            report = await self._generate_performance_report(trading_history, metrics)
            
            return ValidationResult(
                check_id="BIZ-001",
                status=CheckStatus.PASSED,
                message=f"Paper trading validation passed: ${total_pnl:.2f} profit",
                evidence={
                    "total_pnl": str(total_pnl),
                    "trades": len(trading_history),
                    "profitable_days": profitable_days,
                    "win_rate": float(metrics["win_rate"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "sharpe_ratio": float(metrics.get("sharpe_ratio", 0))
                },
                metadata={"report": report}
            ).to_dict()
            
        except Exception as e:
            logger.error("Paper trading validation failed", error=str(e))
            return ValidationResult(
                check_id="BIZ-001",
                status=CheckStatus.FAILED,
                message=f"Validation error: {str(e)}",
                evidence={"error": str(e)},
                metadata={}
            ).to_dict()
    
    async def _load_trading_history(self) -> List[Dict[str, Any]]:
        """Load paper trading history from logs or database."""
        trades = []
        
        # Try loading from JSON log
        if self.trading_log.exists():
            try:
                with open(self.trading_log) as f:
                    data = json.load(f)
                    trades = data.get("trades", [])
            except Exception as e:
                logger.error("Failed to load paper trading log", error=str(e))
        
        # If no JSON log, try database
        if not trades and self.trades_db.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(self.trades_db)
                cursor = conn.cursor()
                
                # Check if paper_trades table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='paper_trades'
                """)
                
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT symbol, side, quantity, entry_price, exit_price, 
                               pnl, timestamp, duration_seconds
                        FROM paper_trades
                        ORDER BY timestamp
                    """)
                    
                    for row in cursor.fetchall():
                        trades.append({
                            "symbol": row[0],
                            "side": row[1],
                            "quantity": row[2],
                            "entry_price": row[3],
                            "exit_price": row[4],
                            "pnl": row[5],
                            "timestamp": row[6],
                            "duration": row[7]
                        })
                
                conn.close()
            except Exception as e:
                logger.error("Failed to load from database", error=str(e))
        
        return trades
    
    async def _check_consecutive_profitable_days(self, trades: List[Dict[str, Any]]) -> int:
        """Check number of consecutive profitable trading days."""
        if not trades:
            return 0
        
        # Group trades by day
        daily_pnl = {}
        for trade in trades:
            # Parse timestamp
            if isinstance(trade.get("timestamp"), str):
                trade_date = datetime.fromisoformat(trade["timestamp"]).date()
            else:
                trade_date = datetime.fromtimestamp(trade.get("timestamp", 0)).date()
            
            if trade_date not in daily_pnl:
                daily_pnl[trade_date] = Decimal("0")
            
            daily_pnl[trade_date] += Decimal(str(trade.get("pnl", 0)))
        
        # Find consecutive profitable days
        sorted_days = sorted(daily_pnl.keys())
        max_consecutive = 0
        current_consecutive = 0
        
        for i, day in enumerate(sorted_days):
            if daily_pnl[day] > 0:
                if i == 0 or (day - sorted_days[i-1]).days == 1:
                    current_consecutive += 1
                else:
                    current_consecutive = 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    async def _calculate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trading metrics."""
        if not trades:
            return {
                "win_rate": Decimal("0"),
                "max_drawdown": Decimal("0"),
                "sharpe_ratio": Decimal("0"),
                "profit_factor": Decimal("0")
            }
        
        # Calculate win rate
        winning_trades = sum(1 for t in trades if Decimal(str(t.get("pnl", 0))) > 0)
        win_rate = Decimal(str(winning_trades)) / Decimal(str(len(trades)))
        
        # Calculate max drawdown
        cumulative_pnl = []
        running_total = Decimal("0")
        peak = Decimal("0")
        max_drawdown = Decimal("0")
        
        for trade in trades:
            running_total += Decimal(str(trade.get("pnl", 0)))
            cumulative_pnl.append(running_total)
            
            if running_total > peak:
                peak = running_total
            elif peak > 0:
                drawdown = (peak - running_total) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            pnls = [Decimal(str(t.get("pnl", 0))) for t in trades]
            avg_pnl = sum(pnls) / len(pnls)
            
            # Calculate standard deviation
            variance = sum((pnl - avg_pnl) ** 2 for pnl in pnls) / len(pnls)
            std_dev = variance.sqrt() if variance > 0 else Decimal("1")
            
            # Sharpe ratio (annualized, assuming 252 trading days)
            if std_dev > 0:
                sharpe_ratio = (avg_pnl / std_dev) * Decimal("15.87")  # sqrt(252)
            else:
                sharpe_ratio = Decimal("0")
        else:
            sharpe_ratio = Decimal("0")
        
        # Calculate profit factor
        gross_profit = sum(Decimal(str(t.get("pnl", 0))) for t in trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(Decimal(str(t.get("pnl", 0))) for t in trades if t.get("pnl", 0) < 0))
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = gross_profit if gross_profit > 0 else Decimal("0")
        
        return {
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor
        }
    
    async def _generate_performance_report(
        self, 
        trades: List[Dict[str, Any]], 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed performance report."""
        if not trades:
            return {}
        
        # Calculate additional statistics
        pnls = [Decimal(str(t.get("pnl", 0))) for t in trades]
        
        # Find streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        is_winning = None
        
        for pnl in pnls:
            if pnl > 0:
                if is_winning:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning = True
                max_win_streak = max(max_win_streak, current_streak)
            elif pnl < 0:
                if not is_winning:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning = False
                max_loss_streak = max(max_loss_streak, current_streak)
        
        return {
            "total_trades": len(trades),
            "winning_trades": sum(1 for p in pnls if p > 0),
            "losing_trades": sum(1 for p in pnls if p < 0),
            "best_trade": float(max(pnls)),
            "worst_trade": float(min(pnls)),
            "average_win": float(sum(p for p in pnls if p > 0) / max(1, sum(1 for p in pnls if p > 0))),
            "average_loss": float(sum(p for p in pnls if p < 0) / max(1, sum(1 for p in pnls if p < 0))),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "metrics": {
                "win_rate": float(metrics["win_rate"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "sharpe_ratio": float(metrics["sharpe_ratio"]),
                "profit_factor": float(metrics["profit_factor"])
            }
        }