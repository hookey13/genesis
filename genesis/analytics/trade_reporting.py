"""Trade reporting system for regulatory compliance."""
import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)

ReportFormat = Literal["CSV", "JSON", "FIX"]


@dataclass
class TradeReport:
    """Trade report data structure for regulatory reporting."""

    trade_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: Decimal
    price: Decimal
    executed_at: datetime
    account_id: str
    order_type: str
    fee: Decimal
    fee_currency: str
    venue: str = "BINANCE"
    settlement_date: date | None = None
    counterparty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with string representations."""
        data = asdict(self)
        data['quantity'] = str(self.quantity)
        data['price'] = str(self.price)
        data['fee'] = str(self.fee)
        data['executed_at'] = self.executed_at.isoformat()
        if self.settlement_date:
            data['settlement_date'] = self.settlement_date.isoformat()
        return data

    def to_fix_format(self) -> str:
        """Convert to FIX protocol format (simplified)."""
        fix_fields = [
            "35=8",  # ExecutionReport
            f"49={self.venue}",  # SenderCompID
            f"56={self.account_id}",  # TargetCompID
            f"55={self.symbol}",  # Symbol
            f"54={'1' if self.side == 'BUY' else '2'}",  # Side
            f"38={self.quantity}",  # OrderQty
            f"44={self.price}",  # Price
            f"60={self.executed_at.strftime('%Y%m%d-%H:%M:%S')}",  # TransactTime
            f"17={self.trade_id}",  # ExecID
            "150=F",  # ExecType (Fill)
            "39=2",  # OrdStatus (Filled)
        ]
        return chr(1).join(fix_fields)


class TradeReportingSystem:
    """System for generating regulatory trade reports."""

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)

    def aggregate_trades_by_date(
        self,
        trades: list[TradeReport],
        start_date: date,
        end_date: date
    ) -> dict[date, list[TradeReport]]:
        """Aggregate trades by date."""
        aggregated = {}

        for trade in trades:
            trade_date = trade.executed_at.date()
            if start_date <= trade_date <= end_date:
                if trade_date not in aggregated:
                    aggregated[trade_date] = []
                aggregated[trade_date].append(trade)

        self.logger.info(
            "trades_aggregated_by_date",
            start_date=str(start_date),
            end_date=str(end_date),
            days_with_trades=len(aggregated)
        )

        return aggregated

    def aggregate_trades_by_symbol(
        self,
        trades: list[TradeReport]
    ) -> dict[str, list[TradeReport]]:
        """Aggregate trades by symbol."""
        aggregated = {}

        for trade in trades:
            if trade.symbol not in aggregated:
                aggregated[trade.symbol] = []
            aggregated[trade.symbol].append(trade)

        self.logger.info(
            "trades_aggregated_by_symbol",
            unique_symbols=len(aggregated),
            total_trades=len(trades)
        )

        return aggregated

    def aggregate_trades_by_account(
        self,
        trades: list[TradeReport]
    ) -> dict[str, list[TradeReport]]:
        """Aggregate trades by account."""
        aggregated = {}

        for trade in trades:
            if trade.account_id not in aggregated:
                aggregated[trade.account_id] = []
            aggregated[trade.account_id].append(trade)

        self.logger.info(
            "trades_aggregated_by_account",
            unique_accounts=len(aggregated),
            total_trades=len(trades)
        )

        return aggregated

    def generate_csv_report(
        self,
        trades: list[TradeReport],
        output_path: str
    ) -> None:
        """Generate CSV format trade report."""
        if not trades:
            self.logger.warning("no_trades_to_export", format="CSV")
            return

        fieldnames = [
            'trade_id', 'symbol', 'side', 'quantity', 'price',
            'executed_at', 'account_id', 'order_type', 'fee',
            'fee_currency', 'venue', 'settlement_date', 'counterparty'
        ]

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for trade in trades:
                writer.writerow(trade.to_dict())

        self.logger.info(
            "csv_report_generated",
            path=output_path,
            trade_count=len(trades)
        )

    def generate_json_report(
        self,
        trades: list[TradeReport],
        output_path: str
    ) -> None:
        """Generate JSON format trade report."""
        if not trades:
            self.logger.warning("no_trades_to_export", format="JSON")
            return

        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "total_trades": len(trades),
            "trades": [trade.to_dict() for trade in trades]
        }

        with open(output_path, 'w') as jsonfile:
            json.dump(report_data, jsonfile, indent=2)

        self.logger.info(
            "json_report_generated",
            path=output_path,
            trade_count=len(trades)
        )

    def generate_fix_report(
        self,
        trades: list[TradeReport],
        output_path: str
    ) -> None:
        """Generate FIX protocol format trade report."""
        if not trades:
            self.logger.warning("no_trades_to_export", format="FIX")
            return

        with open(output_path, 'w') as fixfile:
            for trade in trades:
                fixfile.write(trade.to_fix_format() + '\n')

        self.logger.info(
            "fix_report_generated",
            path=output_path,
            trade_count=len(trades)
        )

    def generate_report(
        self,
        trades: list[TradeReport],
        format: ReportFormat,
        output_path: str
    ) -> None:
        """Generate trade report in specified format."""
        if format == "CSV":
            self.generate_csv_report(trades, output_path)
        elif format == "JSON":
            self.generate_json_report(trades, output_path)
        elif format == "FIX":
            self.generate_fix_report(trades, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def calculate_summary_statistics(
        self,
        trades: list[TradeReport]
    ) -> dict[str, Any]:
        """Calculate summary statistics for trades."""
        if not trades:
            return {
                "total_trades": 0,
                "total_volume": "0",
                "total_fees": "0",
                "unique_symbols": 0,
                "buy_trades": 0,
                "sell_trades": 0
            }

        total_volume = Decimal("0")
        total_fees = Decimal("0")
        unique_symbols = set()
        buy_trades = 0
        sell_trades = 0

        for trade in trades:
            total_volume += trade.quantity * trade.price
            total_fees += trade.fee
            unique_symbols.add(trade.symbol)

            if trade.side == "BUY":
                buy_trades += 1
            else:
                sell_trades += 1

        return {
            "total_trades": len(trades),
            "total_volume": str(total_volume),
            "total_fees": str(total_fees),
            "unique_symbols": len(unique_symbols),
            "buy_trades": buy_trades,
            "sell_trades": sell_trades
        }
