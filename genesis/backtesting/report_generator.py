"""Report generation for backtest results.

Generates comprehensive HTML and text reports with performance metrics,
charts, and trade logs for strategy analysis.
"""

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

from .performance_metrics import PerformanceMetrics, TradeStatistics

logger = structlog.get_logger(__name__)


class BacktestReportGenerator:
    """Generate comprehensive backtest reports with metrics and visualizations."""
    
    def __init__(
        self,
        output_dir: Path = Path(".genesis/reports"),
        template_dir: Optional[Path] = None
    ):
        """Initialize report generator.
        
        Args:
            output_dir: Directory for report output
            template_dir: Directory containing HTML templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = template_dir
        
        logger.info(
            "Report generator initialized",
            output_dir=str(self.output_dir)
        )
    
    def generate(
        self,
        metrics: PerformanceMetrics,
        portfolio_data: Dict[str, Any],
        strategy_name: str,
        backtest_params: Dict[str, Any],
        trades: Optional[List[Dict]] = None
    ) -> Path:
        """Generate comprehensive backtest report.
        
        Args:
            metrics: Calculated performance metrics
            portfolio_data: Portfolio equity curve and positions
            strategy_name: Name of the strategy tested
            backtest_params: Backtest configuration parameters
            trades: Optional list of executed trades
            
        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{strategy_name}_{timestamp}"
        
        # Generate HTML report
        html_path = self._generate_html_report(
            metrics, portfolio_data, strategy_name, 
            backtest_params, trades, report_name
        )
        
        # Generate JSON report for programmatic access
        json_path = self._generate_json_report(
            metrics, portfolio_data, strategy_name,
            backtest_params, trades, report_name
        )
        
        # Generate text summary
        text_path = self._generate_text_summary(
            metrics, strategy_name, backtest_params, report_name
        )
        
        logger.info(
            "Backtest reports generated",
            html_report=str(html_path),
            json_report=str(json_path),
            text_summary=str(text_path)
        )
        
        return html_path
    
    def _generate_html_report(
        self,
        metrics: PerformanceMetrics,
        portfolio_data: Dict[str, Any],
        strategy_name: str,
        backtest_params: Dict[str, Any],
        trades: Optional[List[Dict]],
        report_name: str
    ) -> Path:
        """Generate HTML report with charts and tables."""
        html_content = self._render_html(
            metrics, portfolio_data, strategy_name,
            backtest_params, trades
        )
        
        html_path = self.output_dir / f"{report_name}.html"
        html_path.write_text(html_content)
        
        return html_path
    
    def _render_html(
        self,
        metrics: PerformanceMetrics,
        portfolio_data: Dict[str, Any],
        strategy_name: str,
        backtest_params: Dict[str, Any],
        trades: Optional[List[Dict]]
    ) -> str:
        """Render HTML report content."""
        # Create sections
        header = self._create_header(strategy_name, backtest_params)
        summary = self._create_summary_section(metrics)
        performance = self._create_performance_section(metrics)
        risk = self._create_risk_section(metrics)
        trades_section = self._create_trades_section(metrics.trade_stats, trades)
        charts = self._create_charts_section(portfolio_data, metrics)
        
        # Combine into HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {strategy_name}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {header}
        {summary}
        <div class="metrics-grid">
            {performance}
            {risk}
        </div>
        {trades_section}
        {charts}
        <div class="footer">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Project GENESIS Backtest Engine v1.0
        </div>
    </div>
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>"""
        return html
    
    def _create_header(self, strategy_name: str, params: Dict[str, Any]) -> str:
        """Create report header section."""
        return f"""
        <div class="header">
            <h1>Backtest Report: {strategy_name}</h1>
            <div class="params">
                <h3>Backtest Parameters</h3>
                <table class="params-table">
                    <tr><td>Start Date:</td><td>{params.get('start_date', 'N/A')}</td></tr>
                    <tr><td>End Date:</td><td>{params.get('end_date', 'N/A')}</td></tr>
                    <tr><td>Initial Capital:</td><td>${params.get('initial_capital', 10000):,.2f}</td></tr>
                    <tr><td>Data Frequency:</td><td>{params.get('frequency', '1m')}</td></tr>
                    <tr><td>Commission:</td><td>{params.get('commission', 0.001):.3%}</td></tr>
                </table>
            </div>
        </div>
        """
    
    def _create_summary_section(self, metrics: PerformanceMetrics) -> str:
        """Create executive summary section."""
        profit_class = "positive" if metrics.total_return >= 0 else "negative"
        
        return f"""
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Total Return</h4>
                    <p class="{profit_class}">{metrics.total_return:.2f}%</p>
                </div>
                <div class="summary-card">
                    <h4>Sharpe Ratio</h4>
                    <p>{metrics.sharpe_ratio:.2f}</p>
                </div>
                <div class="summary-card">
                    <h4>Max Drawdown</h4>
                    <p class="negative">{metrics.max_drawdown:.2f}%</p>
                </div>
                <div class="summary-card">
                    <h4>Win Rate</h4>
                    <p>{metrics.trade_stats.win_rate:.1%}</p>
                </div>
            </div>
        </div>
        """ if metrics.trade_stats else self._create_summary_no_trades(metrics)
    
    def _create_summary_no_trades(self, metrics: PerformanceMetrics) -> str:
        """Create summary when no trades available."""
        profit_class = "positive" if metrics.total_return >= 0 else "negative"
        
        return f"""
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Total Return</h4>
                    <p class="{profit_class}">{metrics.total_return:.2f}%</p>
                </div>
                <div class="summary-card">
                    <h4>Annualized Return</h4>
                    <p>{metrics.annualized_return:.2%}</p>
                </div>
                <div class="summary-card">
                    <h4>Volatility</h4>
                    <p>{metrics.volatility:.2%}</p>
                </div>
                <div class="summary-card">
                    <h4>Max Drawdown</h4>
                    <p class="negative">{metrics.max_drawdown:.2f}%</p>
                </div>
            </div>
        </div>
        """
    
    def _create_performance_section(self, metrics: PerformanceMetrics) -> str:
        """Create performance metrics section."""
        return f"""
        <div class="metrics-section">
            <h3>Performance Metrics</h3>
            <table class="metrics-table">
                <tr><td>Total Return:</td><td class="{'positive' if metrics.total_return >= 0 else 'negative'}">{metrics.total_return:.2f}%</td></tr>
                <tr><td>Annualized Return:</td><td>{metrics.annualized_return:.2%}</td></tr>
                <tr><td>CAGR:</td><td>{metrics.compound_annual_growth_rate:.2%}</td></tr>
                <tr class="separator"><td colspan="2"></td></tr>
                <tr><td>Sharpe Ratio:</td><td>{metrics.sharpe_ratio:.3f}</td></tr>
                <tr><td>Sortino Ratio:</td><td>{metrics.sortino_ratio:.3f}</td></tr>
                <tr><td>Calmar Ratio:</td><td>{metrics.calmar_ratio:.3f}</td></tr>
                <tr><td>Information Ratio:</td><td>{metrics.information_ratio:.3f}</td></tr>
                <tr class="separator"><td colspan="2"></td></tr>
                <tr><td>Best Day:</td><td class="positive">{metrics.best_day:.2f}%</td></tr>
                <tr><td>Worst Day:</td><td class="negative">{metrics.worst_day:.2f}%</td></tr>
                <tr><td>Positive Days:</td><td>{metrics.positive_days}</td></tr>
                <tr><td>Negative Days:</td><td>{metrics.negative_days}</td></tr>
            </table>
        </div>
        """
    
    def _create_risk_section(self, metrics: PerformanceMetrics) -> str:
        """Create risk metrics section."""
        # Format optional values with defaults
        ulcer_index = f"{metrics.ulcer_index:.3f}" if metrics.ulcer_index is not None else "N/A"
        recovery_factor = f"{metrics.recovery_factor:.2f}" if metrics.recovery_factor is not None else "N/A"
        kelly_criterion = f"{metrics.kelly_criterion:.1%}" if metrics.kelly_criterion is not None else "N/A"
        
        return f"""
        <div class="metrics-section">
            <h3>Risk Metrics</h3>
            <table class="metrics-table">
                <tr><td>Volatility (Annual):</td><td>{metrics.volatility:.2%}</td></tr>
                <tr><td>Downside Volatility:</td><td>{metrics.downside_volatility:.2%}</td></tr>
                <tr><td>Max Drawdown:</td><td class="negative">{metrics.max_drawdown:.2f}%</td></tr>
                <tr><td>Max DD Duration:</td><td>{self._format_duration(metrics.max_drawdown_duration)}</td></tr>
                <tr class="separator"><td colspan="2"></td></tr>
                <tr><td>VaR (95%):</td><td>{metrics.var_95:.2%}</td></tr>
                <tr><td>CVaR (95%):</td><td>{metrics.cvar_95:.2%}</td></tr>
                <tr><td>Ulcer Index:</td><td>{ulcer_index}</td></tr>
                <tr><td>Recovery Factor:</td><td>{recovery_factor}</td></tr>
                <tr class="separator"><td colspan="2"></td></tr>
                <tr><td>Kelly Criterion:</td><td>{kelly_criterion}</td></tr>
            </table>
        </div>
        """
    
    def _create_trades_section(
        self,
        trade_stats: Optional[TradeStatistics],
        trades: Optional[List[Dict]]
    ) -> str:
        """Create trade statistics section."""
        if not trade_stats:
            return "<div class='trades-section'><h3>Trade Statistics</h3><p>No trades executed</p></div>"
        
        return f"""
        <div class="trades-section">
            <h3>Trade Statistics</h3>
            <div class="trade-stats-grid">
                <table class="metrics-table">
                    <tr><td>Total Trades:</td><td>{trade_stats.total_trades}</td></tr>
                    <tr><td>Winning Trades:</td><td class="positive">{trade_stats.winning_trades}</td></tr>
                    <tr><td>Losing Trades:</td><td class="negative">{trade_stats.losing_trades}</td></tr>
                    <tr><td>Win Rate:</td><td>{trade_stats.win_rate:.1%}</td></tr>
                    <tr><td>Profit Factor:</td><td>{trade_stats.profit_factor:.2f}</td></tr>
                </table>
                <table class="metrics-table">
                    <tr><td>Average Win:</td><td class="positive">${trade_stats.avg_win:.2f}</td></tr>
                    <tr><td>Average Loss:</td><td class="negative">${trade_stats.avg_loss:.2f}</td></tr>
                    <tr><td>Largest Win:</td><td class="positive">${trade_stats.largest_win:.2f}</td></tr>
                    <tr><td>Largest Loss:</td><td class="negative">${trade_stats.largest_loss:.2f}</td></tr>
                    <tr><td>Expectancy:</td><td>${trade_stats.expectancy:.2f}</td></tr>
                </table>
                <table class="metrics-table">
                    <tr><td>Avg Trade Duration:</td><td>{self._format_duration(trade_stats.avg_trade_duration)}</td></tr>
                    <tr><td>Max Consecutive Wins:</td><td>{trade_stats.max_consecutive_wins}</td></tr>
                    <tr><td>Max Consecutive Losses:</td><td>{trade_stats.max_consecutive_losses}</td></tr>
                    <tr><td>Current Streak:</td><td>{trade_stats.current_streak}</td></tr>
                    <tr><td>Payoff Ratio:</td><td>{trade_stats.payoff_ratio:.2f}</td></tr>
                </table>
            </div>
            {self._create_trade_log(trades) if trades else ""}
        </div>
        """
    
    def _create_trade_log(self, trades: List[Dict]) -> str:
        """Create detailed trade log table."""
        if not trades or len(trades) > 100:
            # Don't show log for too many trades
            return ""
        
        rows = []
        for i, trade in enumerate(trades[:50], 1):  # Limit to first 50
            pnl = Decimal(str(trade.get('pnl', 0)))
            pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""
            
            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{trade.get('entry_time', '')}</td>
                    <td>{trade.get('exit_time', '')}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('side', '')}</td>
                    <td>{trade.get('quantity', 0):.4f}</td>
                    <td>${trade.get('entry_price', 0):.4f}</td>
                    <td>${trade.get('exit_price', 0):.4f}</td>
                    <td class="{pnl_class}">${pnl:.2f}</td>
                </tr>
            """)
        
        return f"""
        <div class="trade-log">
            <h4>Recent Trades</h4>
            <table class="trade-log-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """
    
    def _create_charts_section(
        self,
        portfolio_data: Dict[str, Any],
        metrics: PerformanceMetrics
    ) -> str:
        """Create charts section with embedded SVG visualizations."""
        # Generate SVG charts inline for better portability
        equity_chart = self._generate_equity_curve_svg(portfolio_data)
        drawdown_chart = self._generate_drawdown_svg(portfolio_data)
        returns_chart = self._generate_returns_distribution_svg(portfolio_data)
        rolling_chart = self._generate_rolling_sharpe_svg(metrics)
        
        return f"""
        <div class="charts-section">
            <h3>Performance Charts</h3>
            <div class="charts-grid">
                <div class="chart-container">
                    <h4>Equity Curve</h4>
                    <div class="chart-content">
                        {equity_chart}
                    </div>
                </div>
                <div class="chart-container">
                    <h4>Drawdown</h4>
                    <div class="chart-content">
                        {drawdown_chart}
                    </div>
                </div>
                <div class="chart-container">
                    <h4>Returns Distribution</h4>
                    <div class="chart-content">
                        {returns_chart}
                    </div>
                </div>
                <div class="chart-container">
                    <h4>Rolling Sharpe Ratio</h4>
                    <div class="chart-content">
                        {rolling_chart}
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0 0 20px 0;
            color: #2c3e50;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h3 {
            color: #34495e;
            margin-top: 30px;
        }
        .params-table {
            width: 100%;
            max-width: 400px;
        }
        .params-table td {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }
        .params-table td:first-child {
            font-weight: 600;
            color: #7f8c8d;
        }
        .summary {
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .summary-card {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .summary-card h4 {
            margin: 0 0 10px 0;
            color: #7f8c8d;
            font-size: 14px;
            text-transform: uppercase;
        }
        .summary-card p {
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .metrics-section {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics-table td {
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }
        .metrics-table td:first-child {
            font-weight: 500;
            color: #7f8c8d;
        }
        .metrics-table td:last-child {
            text-align: right;
            font-weight: 600;
        }
        .metrics-table tr.separator td {
            border-bottom: 2px solid #ecf0f1;
            padding: 5px;
        }
        .positive {
            color: #27ae60;
        }
        .negative {
            color: #e74c3c;
        }
        .trades-section {
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .trade-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .trade-log {
            margin-top: 30px;
        }
        .trade-log-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .trade-log-table th {
            background: #f8f9fa;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            color: #7f8c8d;
            border-bottom: 2px solid #dee2e6;
        }
        .trade-log-table td {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }
        .charts-section {
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .chart-container {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .chart-container h4 {
            margin: 0 0 15px 0;
            color: #34495e;
        }
        .chart-content {
            height: 300px;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            overflow: hidden;
        }
        .chart-content svg {
            width: 100%;
            height: 100%;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 14px;
        }
        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features."""
        return """
        // Add any interactive features here
        console.log('Backtest report loaded');
        """
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format timedelta for display."""
        if not duration:
            return "N/A"
        
        days = duration.days
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        
        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _generate_json_report(
        self,
        metrics: PerformanceMetrics,
        portfolio_data: Dict[str, Any],
        strategy_name: str,
        backtest_params: Dict[str, Any],
        trades: Optional[List[Dict]],
        report_name: str
    ) -> Path:
        """Generate JSON report for programmatic access."""
        report_data = {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": backtest_params,
            "metrics": self._metrics_to_dict(metrics),
            "portfolio": portfolio_data,
            "trades": trades or []
        }
        
        json_path = self.output_dir / f"{report_name}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return json_path
    
    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        result = {
            "returns": {
                "total_return": float(metrics.total_return),
                "annualized_return": metrics.annualized_return,
                "cagr": metrics.compound_annual_growth_rate
            },
            "risk": {
                "volatility": metrics.volatility,
                "downside_volatility": metrics.downside_volatility,
                "max_drawdown": float(metrics.max_drawdown),
                "max_drawdown_duration": str(metrics.max_drawdown_duration),
                "var_95": metrics.var_95,
                "cvar_95": metrics.cvar_95
            },
            "risk_adjusted": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "calmar_ratio": metrics.calmar_ratio,
                "information_ratio": metrics.information_ratio
            },
            "period_performance": {
                "best_day": float(metrics.best_day),
                "worst_day": float(metrics.worst_day),
                "best_month": float(metrics.best_month),
                "worst_month": float(metrics.worst_month),
                "positive_days": metrics.positive_days,
                "negative_days": metrics.negative_days
            }
        }
        
        if metrics.trade_stats:
            result["trades"] = {
                "total_trades": metrics.trade_stats.total_trades,
                "winning_trades": metrics.trade_stats.winning_trades,
                "losing_trades": metrics.trade_stats.losing_trades,
                "win_rate": metrics.trade_stats.win_rate,
                "profit_factor": metrics.trade_stats.profit_factor,
                "avg_win": float(metrics.trade_stats.avg_win),
                "avg_loss": float(metrics.trade_stats.avg_loss),
                "expectancy": float(metrics.trade_stats.expectancy),
                "payoff_ratio": metrics.trade_stats.payoff_ratio
            }
        
        if metrics.beta is not None:
            result["benchmark"] = {
                "beta": metrics.beta,
                "alpha": metrics.alpha,
                "correlation": metrics.correlation,
                "tracking_error": metrics.tracking_error
            }
        
        return result
    
    def _generate_text_summary(
        self,
        metrics: PerformanceMetrics,
        strategy_name: str,
        backtest_params: Dict[str, Any],
        report_name: str
    ) -> Path:
        """Generate text summary report."""
        lines = [
            f"BACKTEST REPORT - {strategy_name}",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PARAMETERS",
            "-" * 40,
            f"Start Date: {backtest_params.get('start_date', 'N/A')}",
            f"End Date: {backtest_params.get('end_date', 'N/A')}",
            f"Initial Capital: ${backtest_params.get('initial_capital', 10000):,.2f}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 40,
            f"Total Return: {metrics.total_return:.2f}%",
            f"Annualized Return: {metrics.annualized_return:.2%}",
            f"CAGR: {metrics.compound_annual_growth_rate:.2%}",
            "",
            "RISK METRICS",
            "-" * 40,
            f"Volatility: {metrics.volatility:.2%}",
            f"Max Drawdown: {metrics.max_drawdown:.2f}%",
            f"Max DD Duration: {metrics.max_drawdown_duration}",
            f"VaR (95%): {metrics.var_95:.2%}",
            f"CVaR (95%): {metrics.cvar_95:.2%}",
            "",
            "RISK-ADJUSTED RETURNS",
            "-" * 40,
            f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}",
            f"Sortino Ratio: {metrics.sortino_ratio:.3f}",
            f"Calmar Ratio: {metrics.calmar_ratio:.3f}",
            f"Information Ratio: {metrics.information_ratio:.3f}",
            ""
        ]
        
        if metrics.trade_stats:
            lines.extend([
                "TRADE STATISTICS",
                "-" * 40,
                f"Total Trades: {metrics.trade_stats.total_trades}",
                f"Win Rate: {metrics.trade_stats.win_rate:.1%}",
                f"Profit Factor: {metrics.trade_stats.profit_factor:.2f}",
                f"Average Win: ${metrics.trade_stats.avg_win:.2f}",
                f"Average Loss: ${metrics.trade_stats.avg_loss:.2f}",
                f"Expectancy: ${metrics.trade_stats.expectancy:.2f}",
                f"Max Consecutive Wins: {metrics.trade_stats.max_consecutive_wins}",
                f"Max Consecutive Losses: {metrics.trade_stats.max_consecutive_losses}",
                ""
            ])
        
        lines.extend([
            "PERIOD PERFORMANCE",
            "-" * 40,
            f"Best Day: {metrics.best_day:.2f}%",
            f"Worst Day: {metrics.worst_day:.2f}%",
            f"Positive Days: {metrics.positive_days}",
            f"Negative Days: {metrics.negative_days}",
            "",
            "=" * 60,
            "Project GENESIS Backtest Engine v1.0"
        ])
        
        text_path = self.output_dir / f"{report_name}.txt"
        text_path.write_text("\n".join(lines))
        
        return text_path
    
    def _generate_equity_curve_svg(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate SVG equity curve chart."""
        if 'equity_curve' not in portfolio_data or not portfolio_data['equity_curve']:
            return self._generate_empty_chart("No equity data available")
        
        equity = portfolio_data['equity_curve']
        width, height = 480, 280
        margin = 40
        
        # Normalize data to fit in chart area
        min_val = min(equity)
        max_val = max(equity)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Generate path data
        points = []
        for i, value in enumerate(equity):
            x = margin + (i * (width - 2 * margin) / max(len(equity) - 1, 1))
            y = height - margin - ((value - min_val) / range_val) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        
        path_data = "M" + " L".join(points)
        
        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid lines -->
            <g stroke="#e0e0e0" stroke-width="0.5">
                {"".join([f'<line x1="{margin}" y1="{margin + i * (height - 2 * margin) / 4}" x2="{width - margin}" y2="{margin + i * (height - 2 * margin) / 4}"/>' for i in range(5)])}
                {"".join([f'<line x1="{margin + i * (width - 2 * margin) / 4}" y1="{margin}" x2="{margin + i * (width - 2 * margin) / 4}" y2="{height - margin}"/>' for i in range(5)])}
            </g>
            
            <!-- Axes -->
            <g stroke="#333" stroke-width="2">
                <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"/>
                <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}"/>
            </g>
            
            <!-- Equity curve -->
            <path d="{path_data}" fill="none" stroke="#3498db" stroke-width="2"/>
            
            <!-- Labels -->
            <text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">Portfolio Value</text>
            <text x="{margin - 10}" y="{margin}" text-anchor="end" font-size="10">${max_val:.0f}</text>
            <text x="{margin - 10}" y="{height - margin}" text-anchor="end" font-size="10">${min_val:.0f}</text>
        </svg>
        """
    
    def _generate_drawdown_svg(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate SVG drawdown chart."""
        if 'equity_curve' not in portfolio_data or not portfolio_data['equity_curve']:
            return self._generate_empty_chart("No drawdown data available")
        
        equity = portfolio_data['equity_curve']
        width, height = 480, 280
        margin = 40
        
        # Calculate drawdown series
        drawdowns = []
        running_max = equity[0]
        for value in equity:
            running_max = max(running_max, value)
            dd = ((value - running_max) / running_max * 100) if running_max > 0 else 0
            drawdowns.append(dd)
        
        # Generate area chart
        points = []
        for i, dd in enumerate(drawdowns):
            x = margin + (i * (width - 2 * margin) / max(len(drawdowns) - 1, 1))
            y = margin + (abs(dd) / max(abs(min(drawdowns)), 0.01)) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        
        # Create filled area path
        area_path = "M" + f"{margin},{margin} L" + " L".join(points) + f" L{width - margin},{margin} Z"
        
        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid lines -->
            <g stroke="#e0e0e0" stroke-width="0.5">
                {"".join([f'<line x1="{margin}" y1="{margin + i * (height - 2 * margin) / 4}" x2="{width - margin}" y2="{margin + i * (height - 2 * margin) / 4}"/>' for i in range(5)])}
            </g>
            
            <!-- Axes -->
            <g stroke="#333" stroke-width="2">
                <line x1="{margin}" y1="{margin}" x2="{width - margin}" y2="{margin}"/>
                <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}"/>
            </g>
            
            <!-- Drawdown area -->
            <path d="{area_path}" fill="#e74c3c" fill-opacity="0.3"/>
            <path d="M{" L".join(points)}" fill="none" stroke="#e74c3c" stroke-width="2"/>
            
            <!-- Labels -->
            <text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">Drawdown %</text>
            <text x="{margin - 10}" y="{margin}" text-anchor="end" font-size="10">0%</text>
            <text x="{margin - 10}" y="{height - margin}" text-anchor="end" font-size="10">{min(drawdowns):.1f}%</text>
        </svg>
        """
    
    def _generate_returns_distribution_svg(self, portfolio_data: Dict[str, Any]) -> str:
        """Generate SVG returns distribution histogram."""
        if 'equity_curve' not in portfolio_data or len(portfolio_data['equity_curve']) < 2:
            return self._generate_empty_chart("Insufficient data for distribution")
        
        equity = portfolio_data['equity_curve']
        width, height = 480, 280
        margin = 40
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity)):
            if equity[i-1] != 0:
                ret = (equity[i] - equity[i-1]) / equity[i-1] * 100
                returns.append(ret)
        
        if not returns:
            return self._generate_empty_chart("No returns data")
        
        # Create histogram bins
        num_bins = min(20, max(5, len(returns) // 5))
        min_ret = min(returns)
        max_ret = max(returns)
        bin_width = (max_ret - min_ret) / num_bins if max_ret != min_ret else 1
        
        bins = [0] * num_bins
        for ret in returns:
            bin_idx = min(int((ret - min_ret) / bin_width), num_bins - 1)
            if bin_idx >= 0:
                bins[bin_idx] += 1
        
        max_count = max(bins) if bins else 1
        bar_width = (width - 2 * margin) / num_bins
        
        # Generate bars
        bars = []
        for i, count in enumerate(bins):
            x = margin + i * bar_width
            bar_height = (count / max_count) * (height - 2 * margin) if max_count > 0 else 0
            y = height - margin - bar_height
            color = "#27ae60" if (min_ret + (i + 0.5) * float(bin_width)) >= 0 else "#e74c3c"
            bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.8:.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.7"/>')
        
        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Axes -->
            <g stroke="#333" stroke-width="2">
                <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"/>
                <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}"/>
            </g>
            
            <!-- Histogram bars -->
            {"".join(bars)}
            
            <!-- Zero line -->
            <line x1="{margin - min_ret * (width - 2 * margin) / (max_ret - min_ret)}" y1="{margin}" 
                  x2="{margin - min_ret * (width - 2 * margin) / (max_ret - min_ret)}" y2="{height - margin}" 
                  stroke="#333" stroke-width="1" stroke-dasharray="5,5"/>
            
            <!-- Labels -->
            <text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">Returns Distribution</text>
            <text x="{margin}" y="{height - margin + 20}" text-anchor="start" font-size="10">{min_ret:.1f}%</text>
            <text x="{width - margin}" y="{height - margin + 20}" text-anchor="end" font-size="10">{max_ret:.1f}%</text>
        </svg>
        """
    
    def _generate_rolling_sharpe_svg(self, metrics: PerformanceMetrics) -> str:
        """Generate SVG rolling Sharpe ratio chart."""
        if not metrics.rolling_metrics or not metrics.rolling_metrics.sharpe_ratios:
            return self._generate_empty_chart("No rolling metrics data")
        
        sharpe_values = metrics.rolling_metrics.sharpe_ratios
        width, height = 480, 280
        margin = 40
        
        # Normalize data
        min_val = min(sharpe_values)
        max_val = max(sharpe_values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Generate path
        points = []
        for i, value in enumerate(sharpe_values):
            x = margin + (i * (width - 2 * margin) / max(len(sharpe_values) - 1, 1))
            y = height - margin - ((value - min_val) / range_val) * (height - 2 * margin)
            points.append(f"{x:.1f},{y:.1f}")
        
        path_data = "M" + " L".join(points)
        
        # Zero line position
        zero_y = height - margin - ((0 - min_val) / range_val) * (height - 2 * margin) if min_val <= 0 <= max_val else 0
        
        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <!-- Grid lines -->
            <g stroke="#e0e0e0" stroke-width="0.5">
                {"".join([f'<line x1="{margin}" y1="{margin + i * (height - 2 * margin) / 4}" x2="{width - margin}" y2="{margin + i * (height - 2 * margin) / 4}"/>' for i in range(5)])}
            </g>
            
            <!-- Axes -->
            <g stroke="#333" stroke-width="2">
                <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"/>
                <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}"/>
            </g>
            
            <!-- Zero line -->
            {f'<line x1="{margin}" y1="{zero_y}" x2="{width - margin}" y2="{zero_y}" stroke="#666" stroke-width="1" stroke-dasharray="5,5"/>' if min_val <= 0 <= max_val else ''}
            
            <!-- Sharpe ratio line -->
            <path d="{path_data}" fill="none" stroke="#9b59b6" stroke-width="2"/>
            
            <!-- Labels -->
            <text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">Rolling Sharpe Ratio</text>
            <text x="{margin - 10}" y="{margin}" text-anchor="end" font-size="10">{max_val:.2f}</text>
            <text x="{margin - 10}" y="{height - margin}" text-anchor="end" font-size="10">{min_val:.2f}</text>
        </svg>
        """
    
    def _generate_empty_chart(self, message: str) -> str:
        """Generate empty chart SVG with message."""
        width, height = 480, 280
        
        return f"""
        <svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="{width}" height="{height}" fill="#f8f9fa"/>
            <text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-size="14" fill="#6c757d">{message}</text>
        </svg>
        """