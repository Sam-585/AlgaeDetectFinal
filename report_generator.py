"""
Report Generation Module
Handles PDF and CSV report generation for algae analysis results
"""

import io
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Try to import reportlab for better PDF generation, fallback to matplotlib
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class ReportGenerator:
    """Generate comprehensive reports for algae bloom analysis"""
    
    def __init__(self):
        """Initialize report generator"""
        self.styles = self._create_styles()
    
    def _create_styles(self):
        """Create consistent styling for reports"""
        if REPORTLAB_AVAILABLE:
            styles = getSampleStyleSheet()
            
            # Custom styles
            styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.HexColor('#1f77b4')
            ))
            
            styles.add(ParagraphStyle(
                name='CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.HexColor('#2c3e50')
            ))
            
            styles.add(ParagraphStyle(
                name='CustomBody',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6
            ))
            
            return styles
        else:
            return {}
    
    def generate_pdf_report(self, analysis_results: Dict[str, Any]) -> bytes:
        """
        Generate comprehensive PDF report
        
        Args:
            analysis_results: Complete analysis results dictionary
            
        Returns:
            PDF report as bytes
        """
        
        if REPORTLAB_AVAILABLE:
            return self._generate_reportlab_pdf(analysis_results)
        else:
            return self._generate_matplotlib_pdf(analysis_results)
    
    def _generate_reportlab_pdf(self, results: Dict[str, Any]) -> bytes:
        """Generate PDF using ReportLab"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        
        # Build story (content)
        story = []
        
        # Title and header
        story.append(Paragraph("Algae Bloom Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Analysis summary table
        summary_data = [
            ['Parameter', 'Value'],
            ['Waterbody', results.get('waterbody', 'Unknown')],
            ['Analysis Type', results.get('type', 'Unknown')],
            ['Analysis Date', results.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))],
            ['Risk Level', results['risk_assessment']['risk_level']],
            ['Risk Score', f"{results['risk_assessment']['risk_score']:.3f}"],
            ['Algae Coverage', f"{results['risk_assessment']['algae_coverage_percent']:.1f}%"],
        ]
        
        if 'growth_rate' in results['risk_assessment']:
            summary_data.append(['Growth Rate', f"{results['risk_assessment']['growth_rate']:+.2f}%/week"])
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Spectral Indices Section
        story.append(Paragraph("Spectral Indices Analysis", self.styles['CustomHeading']))
        
        indices_data = [['Index', 'Value', 'Interpretation']]
        for index, value in results['indices'].items():
            if isinstance(value, dict):
                val_str = f"{value.get('mean', 0):.3f}"
                interp = value.get('interpretation', 'No interpretation')
            else:
                val_str = f"{value:.3f}"
                interp = self._interpret_index_value(index, value)
            
            indices_data.append([index, val_str, interp])
        
        indices_table = Table(indices_data, colWidths=[1.5*inch, 1*inch, 3*inch])
        indices_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        story.append(indices_table)
        story.append(Spacer(1, 20))
        
        # Environmental Impact Section
        story.append(Paragraph("Environmental Impact Assessment", self.styles['CustomHeading']))
        
        impact = results['environmental_impact']
        
        impact_text = f"""
        <b>Water Quality Score:</b> {impact['water_quality_score']:.1f}/10<br/>
        <b>Dissolved Oxygen Reduction:</b> {impact['dissolved_oxygen_reduction']:.1f}%<br/>
        <b>Fish Mortality Risk:</b> {impact['fish_mortality_risk']}<br/><br/>
        
        <b>Water Usability Assessment:</b><br/>
        """
        
        for use, status in impact['water_usability'].items():
            icon = "✓" if status == "Safe" else "⚠" if status == "Caution" else "✗"
            impact_text += f"• {use}: {icon} {status}<br/>"
        
        story.append(Paragraph(impact_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Mitigation Recommendations", self.styles['CustomHeading']))
        
        # This would be populated from mitigation_strategies.py
        rec_text = "<b>Recommended Actions:</b><br/>"
        rec_text += "• Monitor water quality regularly<br/>"
        rec_text += "• Reduce nutrient inputs to waterbody<br/>"
        rec_text += "• Consider professional water testing<br/>"
        rec_text += "• Implement appropriate treatment measures<br/>"
        
        story.append(Paragraph(rec_text, self.styles['CustomBody']))
        story.append(Spacer(1, 20))
        
        # Footer
        footer_text = f"""
        <br/><br/>
        <i>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i><br/>
        <i>Algae Bloom Monitoring System - Uttarakhand Water Quality Initiative</i>
        """
        
        story.append(Paragraph(footer_text, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _generate_matplotlib_pdf(self, results: Dict[str, Any]) -> bytes:
        """Generate PDF using matplotlib as fallback"""
        
        buffer = io.BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Page 1: Summary and Charts
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8))
            fig.suptitle('Algae Bloom Analysis Report', fontsize=16, fontweight='bold')
            
            # Risk assessment pie chart
            risk_dist = results['environmental_impact'].get('risk_distribution', {
                'Low Risk': 30, 'Medium Risk': 45, 'High Risk': 25
            })
            
            ax1.pie(risk_dist.values(), labels=risk_dist.keys(), autopct='%1.1f%%', 
                   colors=['green', 'yellow', 'red'])
            ax1.set_title('Risk Distribution')
            
            # Spectral indices bar chart
            indices_names = []
            indices_values = []
            
            for name, value in results['indices'].items():
                if isinstance(value, dict):
                    val = value.get('mean', 0)
                else:
                    val = value
                
                indices_names.append(name[:10])  # Truncate long names
                indices_values.append(val)
            
            ax2.bar(range(len(indices_names)), indices_values)
            ax2.set_xticks(range(len(indices_names)))
            ax2.set_xticklabels(indices_names, rotation=45, ha='right')
            ax2.set_title('Spectral Indices')
            ax2.set_ylabel('Value')
            
            # Temporal trend (if available)
            if 'temporal_data' in results:
                temporal_df = pd.DataFrame(results['temporal_data'])
                ax3.plot(pd.to_datetime(temporal_df['date']), temporal_df['algae_coverage'])
                ax3.set_title('Algae Coverage Trend')
                ax3.set_ylabel('Coverage %')
                ax3.tick_params(axis='x', rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No temporal data available', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Temporal Analysis')
            
            # Summary text
            ax4.axis('off')
            summary_text = f"""
            Waterbody: {results.get('waterbody', 'Unknown')}
            Analysis Date: {results.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}
            
            Risk Level: {results['risk_assessment']['risk_level']}
            Risk Score: {results['risk_assessment']['risk_score']:.3f}
            Algae Coverage: {results['risk_assessment']['algae_coverage_percent']:.1f}%
            
            Water Quality Score: {results['environmental_impact']['water_quality_score']:.1f}/10
            DO Reduction: {results['environmental_impact']['dissolved_oxygen_reduction']:.1f}%
            Fish Risk: {results['environmental_impact']['fish_mortality_risk']}
            """
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Detailed Data Tables
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Create table data
            table_data = []
            table_data.append(['Parameter', 'Value', 'Status'])
            
            # Add indices data
            for name, value in results['indices'].items():
                if isinstance(value, dict):
                    val_str = f"{value.get('mean', 0):.4f}"
                    status = value.get('interpretation', 'Normal')
                else:
                    val_str = f"{value:.4f}"
                    status = self._interpret_index_value(name, value)
                
                table_data.append([name, val_str, status])
            
            # Add environmental data
            impact = results['environmental_impact']
            table_data.append(['Water Quality Score', f"{impact['water_quality_score']:.1f}/10", 
                             'Good' if impact['water_quality_score'] > 7 else 'Poor'])
            
            # Create table
            table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                           colWidths=[0.3, 0.2, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Style header row
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#1f77b4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax.set_title('Detailed Analysis Results', fontsize=14, fontweight='bold', pad=20)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_csv_export(self, results: Dict[str, Any]) -> str:
        """
        Generate CSV export of analysis data
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            CSV data as string
        """
        
        export_data = []
        
        # Metadata
        export_data.extend([
            ['Section', 'Parameter', 'Value', 'Unit', 'Notes'],
            ['Metadata', 'Waterbody', results.get('waterbody', ''), '', ''],
            ['Metadata', 'Analysis Type', results.get('type', ''), '', ''],
            ['Metadata', 'Analysis Date', results.get('analysis_date', datetime.now().strftime('%Y-%m-%d')), '', ''],
            ['', '', '', '', '']  # Empty row
        ])
        
        # Risk Assessment
        risk = results['risk_assessment']
        export_data.extend([
            ['Risk Assessment', 'Risk Level', risk['risk_level'], '', ''],
            ['Risk Assessment', 'Risk Score', f"{risk['risk_score']:.4f}", '0-1 scale', ''],
            ['Risk Assessment', 'Algae Coverage', f"{risk['algae_coverage_percent']:.2f}", '%', ''],
        ])
        
        if 'growth_rate' in risk:
            export_data.append(['Risk Assessment', 'Growth Rate', f"{risk['growth_rate']:.2f}", '%/week', ''])
        
        export_data.append(['', '', '', '', ''])  # Empty row
        
        # Spectral Indices
        for index, value in results['indices'].items():
            if isinstance(value, dict):
                val = value.get('mean', 0)
                unit = value.get('unit', '')
                interp = value.get('interpretation', '')
            else:
                val = value
                unit = self._get_index_unit(index)
                interp = self._interpret_index_value(index, val)
            
            export_data.append(['Spectral Indices', index, f"{val:.4f}", unit, interp])
        
        export_data.append(['', '', '', '', ''])  # Empty row
        
        # Environmental Impact
        impact = results['environmental_impact']
        export_data.extend([
            ['Environmental Impact', 'Water Quality Score', f"{impact['water_quality_score']:.2f}", '0-10 scale', ''],
            ['Environmental Impact', 'DO Reduction', f"{impact['dissolved_oxygen_reduction']:.2f}", '%', ''],
            ['Environmental Impact', 'Fish Mortality Risk', impact['fish_mortality_risk'], 'Category', ''],
        ])
        
        # Water Usability
        for use, status in impact['water_usability'].items():
            export_data.append(['Water Usability', use, status, 'Category', ''])
        
        # Temporal data if available
        if 'temporal_data' in results:
            export_data.append(['', '', '', '', ''])  # Empty row
            export_data.append(['Temporal Data', 'Date', 'Algae Coverage', 'Risk Score', ''])
            
            for entry in results['temporal_data']:
                export_data.append([
                    'Temporal Data',
                    entry['date'],
                    f"{entry['algae_coverage']:.2f}",
                    f"{entry['risk_score']:.4f}",
                    ''
                ])
        
        # Convert to DataFrame and then CSV
        df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, header=False)
        
        return csv_buffer.getvalue()
    
    def _interpret_index_value(self, index_name: str, value: float) -> str:
        """Interpret index values for reporting"""
        
        if 'NDVI' in index_name:
            if value > 0.6:
                return "Dense vegetation/algae"
            elif value > 0.3:
                return "Moderate vegetation"
            elif value > 0.1:
                return "Sparse vegetation"
            else:
                return "Water/non-vegetated"
                
        elif 'NDWI' in index_name:
            if value > 0.3:
                return "Open water"
            elif value > 0:
                return "Wet areas"
            else:
                return "Dry/vegetated areas"
                
        elif 'Chlorophyll' in index_name:
            if value > 30:
                return "Very high - severe bloom"
            elif value > 15:
                return "High - bloom present"
            elif value > 8:
                return "Moderate levels"
            else:
                return "Low levels"
                
        elif 'Turbidity' in index_name:
            if value > 40:
                return "Very turbid"
            elif value > 20:
                return "Turbid"
            elif value > 10:
                return "Slightly turbid"
            else:
                return "Clear"
                
        elif 'FAI' in index_name:
            if value > 0.015:
                return "Dense floating algae"
            elif value > 0.005:
                return "Moderate floating algae"
            elif value > 0:
                return "Light algae presence"
            else:
                return "No floating algae"
        
        return "Normal range"
    
    def _get_index_unit(self, index_name: str) -> str:
        """Get units for different indices"""
        
        if 'Chlorophyll' in index_name:
            return 'μg/L'
        elif 'Turbidity' in index_name:
            return 'NTU'
        elif any(x in index_name for x in ['NDVI', 'NDWI', 'FAI']):
            return 'index'
        else:
            return ''
    
    def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary text
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Executive summary as formatted string
        """
        
        risk_level = results['risk_assessment']['risk_level']
        coverage = results['risk_assessment']['algae_coverage_percent']
        waterbody = results.get('waterbody', 'the analyzed waterbody')
        
        summary = f"""
        EXECUTIVE SUMMARY - ALGAE BLOOM ANALYSIS
        
        Waterbody: {waterbody}
        Analysis Date: {results.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))}
        
        KEY FINDINGS:
        • Risk Level: {risk_level}
        • Algae Coverage: {coverage:.1f}%
        • Water Quality Score: {results['environmental_impact']['water_quality_score']:.1f}/10
        
        ENVIRONMENTAL IMPACT:
        • Dissolved Oxygen Reduction: {results['environmental_impact']['dissolved_oxygen_reduction']:.1f}%
        • Fish Mortality Risk: {results['environmental_impact']['fish_mortality_risk']}
        
        IMMEDIATE ACTIONS REQUIRED:
        """
        
        # Add risk-specific recommendations
        if risk_level == "High":
            summary += """
        • URGENT: Restrict water contact and usage
        • Implement emergency treatment measures
        • Notify relevant authorities and communities
        • Begin intensive monitoring program
            """
        elif risk_level == "Medium":
            summary += """
        • Increase monitoring frequency
        • Implement preventive measures
        • Reduce nutrient inputs to waterbody
        • Consider water treatment options
            """
        else:
            summary += """
        • Continue regular monitoring
        • Maintain current management practices
        • Monitor for seasonal changes
            """
        
        summary += f"""
        
        This analysis was conducted using satellite imagery and spectral analysis techniques.
        For detailed results and technical data, please refer to the complete report.
        
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return summary

