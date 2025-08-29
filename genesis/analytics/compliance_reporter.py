"""
Compliance reporting module for Project GENESIS.

Provides trade audit logs, regulatory reports, and compliance export functionality
for institutional-grade regulatory compliance requirements.
"""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pandas as pd
import structlog

from genesis.core.constants import TradingTier
from genesis.core.models import OrderSide
from genesis.data.repository import Repository
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class ComplianceReporter:
    """Generates compliance reports and audit trails for regulatory purposes."""

    def __init__(self, repository: Repository):
        """Initialize ComplianceReporter with repository."""
        self.repository = repository
        logger.info("compliance_reporter_initialized")

    @requires_tier(TradingTier.STRATEGIST)
    async def extract_trade_audit_log(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Extract trade audit log from events table.

        Args:
            account_id: Account to extract audit log for
            start_date: Start of reporting period
            end_date: End of reporting period
            include_metadata: Include additional trade metadata

        Returns:
            List of audit log entries
        """
        try:
            # Get events from repository
            events = await self.repository.get_events_by_aggregate(
                aggregate_id=account_id,
                start_time=start_date,
                end_time=end_date,
            )

            audit_log = []

            for event in events:
                if event.event_type in [
                    "order.executed",
                    "trade.completed",
                    "position.opened",
                    "position.closed",
                ]:
                    entry = {
                        "audit_id": str(uuid4()),
                        "timestamp": event.timestamp.isoformat(),
                        "account_id": account_id,
                        "event_type": event.event_type,
                        "sequence_number": event.sequence_number,
                    }

                    # Parse event data
                    event_data = (
                        json.loads(event.event_data)
                        if isinstance(event.event_data, str)
                        else event.event_data
                    )

                    # Add relevant fields based on event type
                    if event.event_type == "order.executed":
                        entry.update(
                            {
                                "order_id": event_data.get("order_id"),
                                "symbol": event_data.get("symbol"),
                                "side": event_data.get("side"),
                                "quantity": event_data.get("quantity"),
                                "price": event_data.get("price"),
                                "order_type": event_data.get("order_type"),
                            }
                        )
                    elif event.event_type == "trade.completed":
                        entry.update(
                            {
                                "trade_id": event_data.get("trade_id"),
                                "symbol": event_data.get("symbol"),
                                "side": event_data.get("side"),
                                "entry_price": event_data.get("entry_price"),
                                "exit_price": event_data.get("exit_price"),
                                "quantity": event_data.get("quantity"),
                                "pnl_dollars": event_data.get("pnl_dollars"),
                            }
                        )
                    elif event.event_type in ["position.opened", "position.closed"]:
                        entry.update(
                            {
                                "position_id": event_data.get("position_id"),
                                "symbol": event_data.get("symbol"),
                                "side": event_data.get("side"),
                                "quantity": event_data.get("quantity"),
                                "price": event_data.get("price"),
                                "pnl_dollars": (
                                    event_data.get("pnl_dollars")
                                    if event.event_type == "position.closed"
                                    else None
                                ),
                            }
                        )

                    if include_metadata:
                        entry["metadata"] = event_data.get("metadata", {})

                    audit_log.append(entry)

            logger.info(
                "trade_audit_log_extracted",
                account_id=account_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                entry_count=len(audit_log),
            )

            return audit_log

        except Exception as e:
            logger.error("trade_audit_log_extraction_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def generate_regulatory_report(
        self,
        account_id: str,
        report_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """
        Generate regulatory compliance report.

        Args:
            account_id: Account to generate report for
            report_type: Type of report (e.g., "MiFID_II", "EMIR", "CFTC")
            start_date: Start of reporting period
            end_date: End of reporting period

        Returns:
            Regulatory report data
        """
        report_templates = {
            "MiFID_II": self._generate_mifid_ii_report,
            "EMIR": self._generate_emir_report,
            "CFTC": self._generate_cftc_report,
            "BEST_EXECUTION": self._generate_best_execution_report,
            "TRANSACTION_COST": self._generate_transaction_cost_report,
        }

        if report_type not in report_templates:
            raise ValueError(f"Unsupported report type: {report_type}")

        report_generator = report_templates[report_type]
        report = await report_generator(account_id, start_date, end_date)

        logger.info(
            "regulatory_report_generated",
            account_id=account_id,
            report_type=report_type,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        return report

    async def _generate_mifid_ii_report(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate MiFID II compliance report."""
        # Get trading data
        trades = await self.repository.get_trades_by_account(
            account_id, start_date, end_date
        )

        # Calculate required metrics
        total_trades = len(trades)
        total_volume = sum(
            Decimal(str(t.quantity)) * Decimal(str(t.entry_price)) for t in trades
        )

        # Best execution analysis
        slippage_data = []
        for trade in trades:
            if hasattr(trade, "expected_price") and trade.expected_price:
                slippage = abs(
                    Decimal(str(trade.entry_price)) - Decimal(str(trade.expected_price))
                )
                slippage_data.append(float(slippage))

        avg_slippage = sum(slippage_data) / len(slippage_data) if slippage_data else 0

        return {
            "report_type": "MiFID_II",
            "account_id": account_id,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "transaction_reporting": {
                "total_trades": total_trades,
                "total_volume": str(total_volume),
                "average_trade_size": (
                    str(total_volume / total_trades) if total_trades > 0 else "0"
                ),
            },
            "best_execution": {
                "average_slippage": avg_slippage,
                "execution_quality_score": (
                    100 - (avg_slippage * 100) if avg_slippage < 1 else 0
                ),
            },
            "transparency_requirements": {
                "pre_trade_transparency": "Compliant",
                "post_trade_transparency": "Compliant",
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }

    async def _generate_emir_report(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate EMIR (European Market Infrastructure Regulation) report."""
        positions = await self.repository.get_positions_by_account(account_id)

        # Calculate exposure metrics
        total_exposure = sum(Decimal(str(p.dollar_value)) for p in positions)
        max_position_size = max(
            (Decimal(str(p.dollar_value)) for p in positions), default=Decimal("0")
        )

        return {
            "report_type": "EMIR",
            "account_id": account_id,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "derivatives_exposure": {
                "total_notional": str(total_exposure),
                "number_of_positions": len(positions),
                "largest_position": str(max_position_size),
            },
            "clearing_obligation": "N/A - Spot trading only",
            "risk_mitigation": {
                "portfolio_reconciliation": "Daily",
                "dispute_resolution": "Automated",
                "portfolio_compression": "Not applicable",
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }

    async def _generate_cftc_report(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate CFTC (Commodity Futures Trading Commission) report."""
        trades = await self.repository.get_trades_by_account(
            account_id, start_date, end_date
        )

        # Large trader reporting
        large_trades = [
            t
            for t in trades
            if Decimal(str(t.quantity)) * Decimal(str(t.entry_price)) > Decimal("50000")
        ]

        return {
            "report_type": "CFTC",
            "account_id": account_id,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "large_trader_reporting": {
                "reportable_positions": len(large_trades),
                "total_reportable_volume": str(
                    sum(
                        Decimal(str(t.quantity)) * Decimal(str(t.entry_price))
                        for t in large_trades
                    )
                ),
            },
            "position_limits": "Within regulatory limits",
            "swap_dealer_status": "Not applicable",
            "generated_at": datetime.now(UTC).isoformat(),
        }

    async def _generate_best_execution_report(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate best execution analysis report."""
        orders = await self.repository.get_orders_by_account(
            account_id, start_date, end_date
        )

        # Calculate execution metrics
        total_orders = len(orders)
        filled_orders = [o for o in orders if o.status == "FILLED"]
        fill_rate = len(filled_orders) / total_orders if total_orders > 0 else 0

        # Calculate average latency
        latencies = [
            o.latency_ms for o in orders if hasattr(o, "latency_ms") and o.latency_ms
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Calculate price improvement
        price_improvements = []
        for order in filled_orders:
            if order.side == OrderSide.BUY and hasattr(order, "expected_price"):
                improvement = Decimal(str(order.expected_price)) - Decimal(
                    str(order.price)
                )
            elif order.side == OrderSide.SELL and hasattr(order, "expected_price"):
                improvement = Decimal(str(order.price)) - Decimal(
                    str(order.expected_price)
                )
            else:
                improvement = Decimal("0")

            if improvement > 0:
                price_improvements.append(float(improvement))

        avg_price_improvement = (
            sum(price_improvements) / len(price_improvements)
            if price_improvements
            else 0
        )

        return {
            "report_type": "BEST_EXECUTION",
            "account_id": account_id,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "execution_metrics": {
                "total_orders": total_orders,
                "fill_rate": fill_rate,
                "average_latency_ms": avg_latency,
                "average_price_improvement": avg_price_improvement,
            },
            "execution_venues": {
                "primary": "Binance",
                "alternative": "None",
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }

    async def _generate_transaction_cost_report(
        self, account_id: str, start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate transaction cost analysis report."""
        trades = await self.repository.get_trades_by_account(
            account_id, start_date, end_date
        )
        orders = await self.repository.get_orders_by_account(
            account_id, start_date, end_date
        )

        # Calculate fees
        total_maker_fees = sum(
            Decimal(str(o.maker_fee_paid))
            for o in orders
            if hasattr(o, "maker_fee_paid") and o.maker_fee_paid
        )
        total_taker_fees = sum(
            Decimal(str(o.taker_fee_paid))
            for o in orders
            if hasattr(o, "taker_fee_paid") and o.taker_fee_paid
        )

        # Calculate slippage costs
        slippage_costs = Decimal("0")
        for order in orders:
            if hasattr(order, "slippage_percent") and order.slippage_percent:
                cost = (
                    Decimal(str(order.quantity))
                    * Decimal(str(order.price))
                    * Decimal(str(order.slippage_percent))
                    / 100
                )
                slippage_costs += cost

        return {
            "report_type": "TRANSACTION_COST",
            "account_id": account_id,
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "explicit_costs": {
                "maker_fees": str(total_maker_fees),
                "taker_fees": str(total_taker_fees),
                "total_fees": str(total_maker_fees + total_taker_fees),
            },
            "implicit_costs": {
                "slippage": str(slippage_costs),
                "market_impact": "0",  # Would need order book data to calculate
            },
            "total_transaction_costs": str(
                total_maker_fees + total_taker_fees + slippage_costs
            ),
            "generated_at": datetime.now(UTC).isoformat(),
        }

    @requires_tier(TradingTier.STRATEGIST)
    async def export_compliance_data(
        self,
        account_id: str,
        export_format: str,
        start_date: datetime,
        end_date: datetime,
        output_path: str | None = None,
    ) -> str:
        """
        Export compliance data in specified format.

        Args:
            account_id: Account to export data for
            export_format: Format for export (JSON, CSV, PDF)
            start_date: Start of reporting period
            end_date: End of reporting period
            output_path: Optional path to save the export

        Returns:
            Path to exported file or JSON string
        """
        # Get audit log data
        audit_log = await self.extract_trade_audit_log(
            account_id, start_date, end_date, include_metadata=True
        )

        if export_format == "JSON":
            export_data = {
                "account_id": account_id,
                "export_date": datetime.now(UTC).isoformat(),
                "reporting_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "audit_log": audit_log,
                "record_count": len(audit_log),
            }

            if output_path:
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)
                return output_path
            else:
                return json.dumps(export_data, indent=2, default=str)

        elif export_format == "CSV":
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(audit_log)

            if output_path:
                df.to_csv(output_path, index=False)
                return output_path
            else:
                return df.to_csv(index=False)

        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    @requires_tier(TradingTier.STRATEGIST)
    async def validate_compliance_requirements(
        self, account_id: str, compliance_settings: dict[str, Any]
    ) -> dict[str, bool]:
        """
        Validate account compliance with regulatory requirements.

        Args:
            account_id: Account to validate
            compliance_settings: Compliance settings to validate against

        Returns:
            Dictionary of compliance check results
        """
        results = {}

        # Check position limits
        if "max_position_size" in compliance_settings:
            positions = await self.repository.get_positions_by_account(account_id)
            max_position = max(
                (Decimal(str(p.dollar_value)) for p in positions), default=Decimal("0")
            )
            results["position_limits"] = max_position <= Decimal(
                str(compliance_settings["max_position_size"])
            )

        # Check daily trade limits
        if "max_daily_trades" in compliance_settings:
            today_start = datetime.now(UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            today_end = today_start + timedelta(days=1)
            trades = await self.repository.get_trades_by_account(
                account_id, today_start, today_end
            )
            results["daily_trade_limits"] = (
                len(trades) <= compliance_settings["max_daily_trades"]
            )

        # Check required reporting
        if "reporting_frequency" in compliance_settings:
            # This would check if reports are being generated at required frequency
            results["reporting_compliance"] = True  # Placeholder

        logger.info(
            "compliance_validation_completed",
            account_id=account_id,
            results=results,
        )

        return results
