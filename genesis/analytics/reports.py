"""P&L and financial reporting system."""
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import structlog

# Optional imports for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Optional Jinja2 for HTML templates
try:
    from jinja2 import Environment, FileSystemLoader, Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

logger = structlog.get_logger(__name__)


@dataclass
class PnLEntry:
    """Individual P&L entry."""

    date: date
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    gross_pnl: Decimal
    fees: Decimal
    net_pnl: Decimal
    position_type: str  # LONG or SHORT
    holding_period_days: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "gross_pnl": str(self.gross_pnl),
            "fees": str(self.fees),
            "net_pnl": str(self.net_pnl),
            "position_type": self.position_type,
            "holding_period_days": self.holding_period_days
        }


@dataclass
class MonthlyPnLSummary:
    """Monthly P&L summary."""

    year: int
    month: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    gross_profit: Decimal
    gross_loss: Decimal
    total_fees: Decimal
    net_pnl: Decimal
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    max_drawdown: Decimal
    best_trade: PnLEntry | None = None
    worst_trade: PnLEntry | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "year": self.year,
            "month": self.month,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "gross_profit": str(self.gross_profit),
            "gross_loss": str(self.gross_loss),
            "total_fees": str(self.total_fees),
            "net_pnl": str(self.net_pnl),
            "win_rate": str(self.win_rate),
            "average_win": str(self.average_win),
            "average_loss": str(self.average_loss),
            "profit_factor": str(self.profit_factor),
            "max_drawdown": str(self.max_drawdown)
        }

        if self.best_trade:
            data["best_trade"] = self.best_trade.to_dict()
        if self.worst_trade:
            data["worst_trade"] = self.worst_trade.to_dict()

        return data


class PnLReportGenerator:
    """P&L statement and report generator."""

    def __init__(self, template_dir: str | None = None):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.template_dir = template_dir

        if HAS_JINJA2 and template_dir:
            self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        else:
            self.jinja_env = None

    def calculate_monthly_summary(
        self,
        trades: list[PnLEntry],
        year: int,
        month: int
    ) -> MonthlyPnLSummary:
        """Calculate monthly P&L summary from trades."""
        month_trades = [
            t for t in trades
            if t.date.year == year and t.date.month == month
        ]

        if not month_trades:
            return MonthlyPnLSummary(
                year=year,
                month=month,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                gross_profit=Decimal("0"),
                gross_loss=Decimal("0"),
                total_fees=Decimal("0"),
                net_pnl=Decimal("0"),
                win_rate=Decimal("0"),
                average_win=Decimal("0"),
                average_loss=Decimal("0"),
                profit_factor=Decimal("0"),
                max_drawdown=Decimal("0")
            )

        winning_trades = [t for t in month_trades if t.net_pnl > 0]
        losing_trades = [t for t in month_trades if t.net_pnl < 0]

        gross_profit = sum(t.gross_pnl for t in winning_trades)
        gross_loss = abs(sum(t.gross_pnl for t in losing_trades))
        total_fees = sum(t.fees for t in month_trades)
        net_pnl = sum(t.net_pnl for t in month_trades)

        win_rate = (
            Decimal(len(winning_trades)) / Decimal(len(month_trades)) * 100
            if month_trades else Decimal("0")
        )

        average_win = (
            gross_profit / Decimal(len(winning_trades))
            if winning_trades else Decimal("0")
        )

        average_loss = (
            gross_loss / Decimal(len(losing_trades))
            if losing_trades else Decimal("0")
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0 else Decimal("999")
        )

        # Calculate max drawdown
        cumulative_pnl = Decimal("0")
        peak = Decimal("0")
        max_drawdown = Decimal("0")

        for trade in sorted(month_trades, key=lambda x: x.date):
            cumulative_pnl += trade.net_pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Find best and worst trades
        best_trade = max(month_trades, key=lambda x: x.net_pnl) if month_trades else None
        worst_trade = min(month_trades, key=lambda x: x.net_pnl) if month_trades else None

        return MonthlyPnLSummary(
            year=year,
            month=month,
            total_trades=len(month_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            total_fees=total_fees,
            net_pnl=net_pnl,
            win_rate=win_rate,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            best_trade=best_trade,
            worst_trade=worst_trade
        )

    def generate_json_report(
        self,
        summary: MonthlyPnLSummary,
        output_path: str
    ) -> None:
        """Generate JSON format P&L report."""
        report_data = {
            "report_type": "monthly_pnl",
            "generated_at": datetime.now().isoformat(),
            "summary": summary.to_dict()
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(
            "json_pnl_report_generated",
            path=output_path,
            year=summary.year,
            month=summary.month
        )

    def generate_html_report(
        self,
        summary: MonthlyPnLSummary,
        output_path: str,
        template_name: str | None = None
    ) -> None:
        """Generate HTML format P&L report."""
        if not HAS_JINJA2:
            self.logger.warning("jinja2_not_available", format="HTML")
            return

        if template_name and self.jinja_env:
            # Use custom template
            template = self.jinja_env.get_template(template_name)
            html_content = template.render(summary=summary.to_dict())
        else:
            # Use default template
            html_content = self._generate_default_html(summary)

        with open(output_path, 'w') as f:
            f.write(html_content)

        self.logger.info(
            "html_pnl_report_generated",
            path=output_path,
            year=summary.year,
            month=summary.month
        )

    def _generate_default_html(self, summary: MonthlyPnLSummary) -> str:
        """Generate default HTML report without template."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Monthly P&L Report - {summary.year}/{summary.month:02d}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Monthly P&L Report</h1>
    <h2>{summary.year} - {summary.month:02d}</h2>
    
    <div class="summary-grid">
        <div>
            <div class="metric">
                <span class="metric-label">Total Trades:</span> {summary.total_trades}
            </div>
            <div class="metric">
                <span class="metric-label">Winning Trades:</span> {summary.winning_trades}
            </div>
            <div class="metric">
                <span class="metric-label">Losing Trades:</span> {summary.losing_trades}
            </div>
            <div class="metric">
                <span class="metric-label">Win Rate:</span> {summary.win_rate:.2f}%
            </div>
        </div>
        <div>
            <div class="metric">
                <span class="metric-label">Net P&L:</span> 
                <span class="{'positive' if summary.net_pnl >= 0 else 'negative'}">${summary.net_pnl:,.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Gross Profit:</span> ${summary.gross_profit:,.2f}
            </div>
            <div class="metric">
                <span class="metric-label">Gross Loss:</span> ${summary.gross_loss:,.2f}
            </div>
            <div class="metric">
                <span class="metric-label">Total Fees:</span> ${summary.total_fees:,.2f}
            </div>
        </div>
    </div>
    
    <h3>Performance Metrics</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Average Win</td>
            <td>${summary.average_win:,.2f}</td>
        </tr>
        <tr>
            <td>Average Loss</td>
            <td>${summary.average_loss:,.2f}</td>
        </tr>
        <tr>
            <td>Profit Factor</td>
            <td>{summary.profit_factor:.2f}</td>
        </tr>
        <tr>
            <td>Max Drawdown</td>
            <td>${summary.max_drawdown:,.2f}</td>
        </tr>
    </table>
    
    <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
</body>
</html>
"""
        return html

    def generate_pdf_report(
        self,
        summary: MonthlyPnLSummary,
        output_path: str
    ) -> None:
        """Generate PDF format P&L report."""
        if not HAS_REPORTLAB:
            self.logger.warning("reportlab_not_available", format="PDF")
            return

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        # Container for the 'Flowable' objects
        elements = []

        # Define styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']

        # Title
        elements.append(Paragraph(
            f"Monthly P&L Report - {summary.year}/{summary.month:02d}",
            title_style
        ))
        elements.append(Spacer(1, 12))

        # Summary section
        elements.append(Paragraph("Summary", heading_style))
        elements.append(Spacer(1, 6))

        # Create summary table
        summary_data = [
            ["Metric", "Value"],
            ["Total Trades", str(summary.total_trades)],
            ["Winning Trades", str(summary.winning_trades)],
            ["Losing Trades", str(summary.losing_trades)],
            ["Win Rate", f"{summary.win_rate:.2f}%"],
            ["Net P&L", f"${summary.net_pnl:,.2f}"],
            ["Gross Profit", f"${summary.gross_profit:,.2f}"],
            ["Gross Loss", f"${summary.gross_loss:,.2f}"],
            ["Total Fees", f"${summary.total_fees:,.2f}"],
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Performance metrics section
        elements.append(Paragraph("Performance Metrics", heading_style))
        elements.append(Spacer(1, 6))

        metrics_data = [
            ["Metric", "Value"],
            ["Average Win", f"${summary.average_win:,.2f}"],
            ["Average Loss", f"${summary.average_loss:,.2f}"],
            ["Profit Factor", f"{summary.profit_factor:.2f}"],
            ["Max Drawdown", f"${summary.max_drawdown:,.2f}"],
        ]

        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(metrics_table)
        elements.append(Spacer(1, 20))

        # Footer
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style
        ))

        # Build PDF
        doc.build(elements)

        self.logger.info(
            "pdf_pnl_report_generated",
            path=output_path,
            year=summary.year,
            month=summary.month
        )

    def generate_quarterly_report(
        self,
        trades: list[PnLEntry],
        year: int,
        quarter: int,
        output_format: str = "JSON"
    ) -> dict[str, Any]:
        """Generate quarterly P&L report."""
        quarter_months = {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12]
        }

        if quarter not in quarter_months:
            raise ValueError(f"Invalid quarter: {quarter}")

        months = quarter_months[quarter]
        monthly_summaries = []

        for month in months:
            summary = self.calculate_monthly_summary(trades, year, month)
            monthly_summaries.append(summary)

        # Calculate quarterly totals
        total_trades = sum(s.total_trades for s in monthly_summaries)
        total_winning = sum(s.winning_trades for s in monthly_summaries)
        total_losing = sum(s.losing_trades for s in monthly_summaries)
        total_net_pnl = sum(s.net_pnl for s in monthly_summaries)
        total_fees = sum(s.total_fees for s in monthly_summaries)

        quarterly_report = {
            "year": year,
            "quarter": quarter,
            "months": [s.to_dict() for s in monthly_summaries],
            "quarterly_summary": {
                "total_trades": total_trades,
                "winning_trades": total_winning,
                "losing_trades": total_losing,
                "net_pnl": str(total_net_pnl),
                "total_fees": str(total_fees),
                "average_monthly_pnl": str(total_net_pnl / Decimal("3"))
            }
        }

        self.logger.info(
            "quarterly_report_generated",
            year=year,
            quarter=quarter,
            total_trades=total_trades,
            net_pnl=str(total_net_pnl)
        )

        return quarterly_report

    def generate_annual_report(
        self,
        trades: list[PnLEntry],
        year: int
    ) -> dict[str, Any]:
        """Generate annual P&L report."""
        monthly_summaries = []

        for month in range(1, 13):
            summary = self.calculate_monthly_summary(trades, year, month)
            if summary.total_trades > 0:
                monthly_summaries.append(summary)

        # Calculate annual totals
        total_trades = sum(s.total_trades for s in monthly_summaries)
        total_winning = sum(s.winning_trades for s in monthly_summaries)
        total_losing = sum(s.losing_trades for s in monthly_summaries)
        total_net_pnl = sum(s.net_pnl for s in monthly_summaries)
        total_fees = sum(s.total_fees for s in monthly_summaries)

        annual_report = {
            "year": year,
            "months_traded": len(monthly_summaries),
            "monthly_summaries": [s.to_dict() for s in monthly_summaries],
            "annual_summary": {
                "total_trades": total_trades,
                "winning_trades": total_winning,
                "losing_trades": total_losing,
                "net_pnl": str(total_net_pnl),
                "total_fees": str(total_fees),
                "average_monthly_pnl": str(
                    total_net_pnl / Decimal(len(monthly_summaries))
                    if monthly_summaries else Decimal("0")
                ),
                "best_month": max(
                    monthly_summaries, key=lambda x: x.net_pnl
                ).month if monthly_summaries else None,
                "worst_month": min(
                    monthly_summaries, key=lambda x: x.net_pnl
                ).month if monthly_summaries else None
            }
        }

        self.logger.info(
            "annual_report_generated",
            year=year,
            total_trades=total_trades,
            net_pnl=str(total_net_pnl)
        )

        return annual_report
