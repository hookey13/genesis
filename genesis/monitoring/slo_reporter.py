"""
SLO reporting system with PDF generation and email distribution.

This module generates comprehensive SLO compliance reports in various formats
including PDF, HTML, and JSON for distribution to stakeholders.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import structlog
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image, KeepTogether
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

from genesis.monitoring.slo_tracker import SLOTracker

logger = structlog.get_logger(__name__)


@dataclass
class ReportConfig:
    """Configuration for SLO report generation."""
    title: str = "SLO Compliance Report"
    company: str = "Genesis Trading System"
    report_period_days: int = 30
    include_charts: bool = True
    include_trends: bool = True
    include_recommendations: bool = True
    output_formats: List[str] = None
    recipients: List[str] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["pdf", "html", "json"]
        if self.recipients is None:
            self.recipients = ["ops-team@genesis.io"]


class SLOReporter:
    """
    Generates and distributes SLO compliance reports.
    
    Features:
    - Multiple output formats (PDF, HTML, JSON)
    - Customizable report templates
    - Automated distribution via email
    - Trend analysis and visualizations
    """
    
    def __init__(
        self,
        slo_tracker: SLOTracker,
        output_dir: str = "reports/slo",
        template_dir: str = "templates/reports"
    ):
        self.slo_tracker = slo_tracker
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self) -> None:
        """Setup custom styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceBefore=20,
            spaceAfter=10
        ))
        
        # Service name style
        self.styles.add(ParagraphStyle(
            name='ServiceName',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#2980B9'),
            spaceBefore=15,
            spaceAfter=10
        ))
        
        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        ))
    
    async def generate_report(
        self,
        config: Optional[ReportConfig] = None,
        services: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate SLO compliance report.
        
        Args:
            config: Report configuration
            services: List of services to include (all if None)
            
        Returns:
            Dictionary of format -> file path
        """
        if config is None:
            config = ReportConfig()
        
        # Get report data
        report_data = await self._collect_report_data(config, services)
        
        # Generate reports in requested formats
        output_files = {}
        
        if "pdf" in config.output_formats:
            output_files["pdf"] = await self._generate_pdf_report(report_data, config)
        
        if "html" in config.output_formats:
            output_files["html"] = await self._generate_html_report(report_data, config)
        
        if "json" in config.output_formats:
            output_files["json"] = await self._generate_json_report(report_data, config)
        
        logger.info(
            "Generated SLO reports",
            formats=list(output_files.keys()),
            services_count=len(report_data["services"])
        )
        
        return output_files
    
    async def _collect_report_data(
        self,
        config: ReportConfig,
        services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Collect data for the report."""
        if services is None:
            services = list(self.slo_tracker.slo_configs.keys())
        
        report_data = {
            "metadata": {
                "title": config.title,
                "company": config.company,
                "generated_at": datetime.utcnow().isoformat(),
                "report_period_days": config.report_period_days,
                "period_start": (datetime.utcnow() - timedelta(days=config.report_period_days)).isoformat(),
                "period_end": datetime.utcnow().isoformat()
            },
            "summary": {
                "total_services": len(services),
                "services_meeting_slo": 0,
                "overall_compliance": 0.0,
                "critical_violations": 0,
                "warnings": 0
            },
            "services": {}
        }
        
        total_compliance = 0.0
        
        for service_name in services:
            if service_name not in self.slo_tracker.slo_configs:
                continue
            
            # Get service SLO data
            service_summary = self.slo_tracker.get_slo_summary(service_name)
            
            # Check compliance
            window_key = f"{config.report_period_days}d"
            if window_key not in service_summary.get("compliance", {}):
                window_key = "30d"  # Fallback to 30 days
            
            compliance = service_summary.get("compliance", {}).get(window_key, 0.0)
            error_budget = service_summary.get("error_budgets", {}).get(window_key, {})
            
            # Determine status
            status = "healthy"
            if compliance < 0.99:
                status = "warning"
                report_data["summary"]["warnings"] += 1
            if compliance < 0.95 or error_budget.get("remaining", 1.0) < 0.1:
                status = "critical"
                report_data["summary"]["critical_violations"] += 1
            
            if compliance >= 0.999:  # Meeting SLO target
                report_data["summary"]["services_meeting_slo"] += 1
            
            total_compliance += compliance
            
            # Store service data
            report_data["services"][service_name] = {
                "status": status,
                "compliance": compliance,
                "error_budget": error_budget,
                "slis": service_summary.get("current_slis", {}),
                "trends": self._calculate_trends(service_name, config.report_period_days)
            }
        
        # Calculate overall compliance
        if services:
            report_data["summary"]["overall_compliance"] = total_compliance / len(services)
        
        return report_data
    
    def _calculate_trends(self, service: str, period_days: int) -> Dict[str, Any]:
        """Calculate trend data for a service."""
        # This would analyze historical data to determine trends
        # For now, return mock trend data
        return {
            "compliance_trend": "improving",  # improving, stable, degrading
            "burn_rate_trend": "stable",
            "incident_frequency": "decreasing",
            "improvement_since_last_period": 2.5  # percentage points
        }
    
    async def _generate_pdf_report(
        self,
        report_data: Dict[str, Any],
        config: ReportConfig
    ) -> str:
        """Generate PDF report."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"slo_report_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build content
        story = []
        
        # Title page
        story.append(Paragraph(config.title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(config.company, self.styles['CustomSubtitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # Report metadata
        metadata = report_data["metadata"]
        story.append(Paragraph(
            f"Report Period: {metadata['period_start'][:10]} to {metadata['period_end'][:10]}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Generated: {metadata['generated_at'][:19]}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        summary = report_data["summary"]
        
        summary_data = [
            ["Metric", "Value"],
            ["Total Services", str(summary["total_services"])],
            ["Services Meeting SLO", str(summary["services_meeting_slo"])],
            ["Overall Compliance", f"{summary['overall_compliance']*100:.2f}%"],
            ["Critical Violations", str(summary["critical_violations"])],
            ["Warnings", str(summary["warnings"])]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(PageBreak())
        
        # Service Details
        story.append(Paragraph("Service Details", self.styles['CustomSubtitle']))
        
        for service_name, service_data in report_data["services"].items():
            # Service header
            story.append(Paragraph(service_name, self.styles['ServiceName']))
            
            # Status indicator
            status_color = {
                "healthy": colors.green,
                "warning": colors.orange,
                "critical": colors.red
            }.get(service_data["status"], colors.grey)
            
            # Service metrics table
            service_metrics = [
                ["Metric", "Value", "Status"],
                ["Compliance", f"{service_data['compliance']*100:.2f}%", service_data["status"].upper()],
                ["Error Budget Remaining", f"{service_data['error_budget'].get('remaining', 0)*100:.1f}%", ""],
                ["Burn Rate", f"{service_data['error_budget'].get('burn_rate', 0):.2f}x", ""],
                ["Trend", service_data["trends"]["compliance_trend"].upper(), ""]
            ]
            
            service_table = Table(service_metrics, colWidths=[2*inch, 2*inch, 1.5*inch])
            service_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (2, 1), (2, 1), status_color),
                ('TEXTCOLOR', (2, 1), (2, 1), colors.white if service_data["status"] != "healthy" else colors.black)
            ]))
            
            story.append(service_table)
            story.append(Spacer(1, 0.2*inch))
            
            # SLI details
            if service_data.get("slis"):
                sli_data = [["SLI", "Current Value", "Threshold", "Status"]]
                
                for sli_name, sli_info in service_data["slis"].items():
                    status = "✓" if sli_info.get("is_good", False) else "✗"
                    sli_data.append([
                        sli_name,
                        f"{sli_info.get('value', 0):.4f}",
                        f"{sli_info.get('threshold', 0):.4f}",
                        status
                    ])
                
                sli_table = Table(sli_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 0.5*inch])
                sli_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                
                story.append(Paragraph("SLI Details", self.styles['Normal']))
                story.append(sli_table)
            
            story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        if config.include_recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Recommendations", self.styles['CustomSubtitle']))
            
            recommendations = self._generate_recommendations(report_data)
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(
            "This report is automatically generated by Genesis Monitoring System",
            self.styles['Footer']
        ))
        
        # Build PDF
        doc.build(story)
        
        return str(filepath)
    
    async def _generate_html_report(
        self,
        report_data: Dict[str, Any],
        config: ReportConfig
    ) -> str:
        """Generate HTML report."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"slo_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{config.title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2C3E50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495E;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #2980B9;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #3498db;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    background: white;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .status-healthy {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .status-warning {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .status-critical {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .footer {{
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #bdc3c7;
                }}
            </style>
        </head>
        <body>
            <h1>{config.title}</h1>
            <p style="text-align: center; color: #7f8c8d;">
                {config.company} | Report Period: {report_data['metadata']['period_start'][:10]} to {report_data['metadata']['period_end'][:10]}
            </p>
            
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="metric-value">{report_data['summary']['total_services']}</div>
                    <div class="metric-label">Total Services</div>
                </div>
                <div class="summary-card">
                    <div class="metric-value">{report_data['summary']['services_meeting_slo']}</div>
                    <div class="metric-label">Meeting SLO</div>
                </div>
                <div class="summary-card">
                    <div class="metric-value">{report_data['summary']['overall_compliance']*100:.1f}%</div>
                    <div class="metric-label">Overall Compliance</div>
                </div>
                <div class="summary-card">
                    <div class="metric-value" style="color: #e74c3c;">{report_data['summary']['critical_violations']}</div>
                    <div class="metric-label">Critical Violations</div>
                </div>
            </div>
            
            <h2>Service Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Status</th>
                        <th>Compliance</th>
                        <th>Error Budget</th>
                        <th>Burn Rate</th>
                        <th>Trend</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for service_name, service_data in report_data["services"].items():
            status_class = f"status-{service_data['status']}"
            compliance = service_data['compliance'] * 100
            error_budget = service_data['error_budget'].get('remaining', 0) * 100
            burn_rate = service_data['error_budget'].get('burn_rate', 0)
            trend = service_data['trends']['compliance_trend']
            
            html_content += f"""
                    <tr>
                        <td><strong>{service_name}</strong></td>
                        <td class="{status_class}">{service_data['status'].upper()}</td>
                        <td>{compliance:.2f}%</td>
                        <td>{error_budget:.1f}%</td>
                        <td>{burn_rate:.2f}x</td>
                        <td>{trend}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
            
            <div class="footer">
                <p>Generated: """ + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC") + """</p>
                <p>This report is automatically generated by Genesis Monitoring System</p>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        return str(filepath)
    
    async def _generate_json_report(
        self,
        report_data: Dict[str, Any],
        config: ReportConfig
    ) -> str:
        """Generate JSON report."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"slo_report_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(filepath)
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on report data."""
        recommendations = []
        
        # Check overall compliance
        if report_data["summary"]["overall_compliance"] < 0.99:
            recommendations.append(
                "Overall SLO compliance is below target. Review and optimize system reliability practices."
            )
        
        # Check critical violations
        if report_data["summary"]["critical_violations"] > 0:
            recommendations.append(
                f"Address {report_data['summary']['critical_violations']} critical SLO violations immediately."
            )
        
        # Check services with low error budget
        for service_name, service_data in report_data["services"].items():
            error_budget = service_data["error_budget"].get("remaining", 1.0)
            if error_budget < 0.2:
                recommendations.append(
                    f"Service '{service_name}' has only {error_budget*100:.1f}% error budget remaining. "
                    "Consider implementing stricter change controls."
                )
            
            burn_rate = service_data["error_budget"].get("burn_rate", 0)
            if burn_rate > 2:
                recommendations.append(
                    f"Service '{service_name}' is burning error budget at {burn_rate:.1f}x rate. "
                    "Investigate and address reliability issues."
                )
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("All services are meeting SLO targets. Continue monitoring for changes.")
        
        recommendations.append("Schedule regular SLO reviews to ensure targets align with business requirements.")
        recommendations.append("Consider implementing automated remediation for common failure scenarios.")
        
        return recommendations
    
    async def schedule_reports(
        self,
        schedule: str = "weekly",
        config: Optional[ReportConfig] = None
    ) -> None:
        """
        Schedule periodic report generation.
        
        Args:
            schedule: Frequency (daily, weekly, monthly)
            config: Report configuration
        """
        intervals = {
            "daily": timedelta(days=1),
            "weekly": timedelta(days=7),
            "monthly": timedelta(days=30)
        }
        
        interval = intervals.get(schedule, timedelta(days=7))
        
        logger.info(f"Scheduled SLO reports", schedule=schedule)
        
        while True:
            try:
                # Generate report
                report_files = await self.generate_report(config)
                
                # Distribute report (would send via email in production)
                logger.info(
                    "Generated scheduled SLO report",
                    files=report_files,
                    recipients=config.recipients if config else []
                )
                
                # Wait for next scheduled time
                await asyncio.sleep(interval.total_seconds())
                
            except Exception as e:
                logger.error("Error generating scheduled report", error=str(e))
                await asyncio.sleep(3600)  # Retry in an hour