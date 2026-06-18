# Transcript export helpers for TXT and PDF downloads.
import io
import logging
from datetime import datetime
from typing import Dict
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

logger = logging.getLogger(__name__)

# Export formats supported by the current UI.
EXPORT_CONFIG = {
    'txt': {
        'extension': '.txt',
        'mime_type': 'text/plain',
        'encoding': 'utf-8'
    },
    'pdf': {
        'extension': '.pdf',
        'mime_type': 'application/pdf',
        'page_size': A4,
        'margins': 0.75 * inch,
        'include_toc': False,
        'include_metadata': True
    }
}

PDF_STYLES = {
    'title_color': colors.HexColor('#1a365d'),
    'speaker_color': colors.HexColor('#22c55e'),
    'timestamp_color': colors.HexColor('#06b6d4'),
    'text_color': colors.HexColor('#1f2937'),
}


class TranscriptExporter:
    # Format merged transcript data into user-downloadable files.
    def __init__(self):
        logger.info("TranscriptExporter initialized")

    def export(self, transcript_data: Dict, format: str = 'txt', metadata: Dict = None) -> str or bytes:
        # Dispatch to the requested export format.
        logger.info(f"Exporting transcript as {format.upper()}")

        if format.lower() == 'txt':
            return self.export_txt(transcript_data, metadata)
        if format.lower() == 'pdf':
            return self.export_pdf(transcript_data, metadata)
        raise ValueError(f"Unsupported format: {format}. Supported: txt, pdf")

    def export_txt(self, transcript_data: Dict, metadata: Dict = None) -> str:
        # Create a plain-text transcript with optional meeting metadata.
        logger.info("Generating TXT export")

        try:
            lines = []

            lines.append("═" * 70)
            lines.append("Meeting Transcript")
            lines.append("═" * 70)
            lines.append("")

            if metadata:
                lines.append("Meeting Information:")
                lines.append("─" * 70)
                lines.append(f"Title: {metadata.get('title', 'Untitled Meeting')}")
                lines.append(f"Date: {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}")

                duration = transcript_data.get('total_duration', 0)
                duration_str = self._format_duration(duration)
                lines.append(f"Duration: {duration_str}")

                num_speakers = transcript_data.get('num_speakers', 0)
                lines.append(f"Participants: {num_speakers} speaker(s)")

                lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("")

            lines.append("═" * 70)
            lines.append("TRANSCRIPT")
            lines.append("═" * 70)
            lines.append("")

            entries = transcript_data.get('transcript_entries', [])
            for entry in entries:
                timestamp_start = entry.get('timestamp_start', '00:00:00')
                timestamp_end = entry.get('timestamp_end', '00:00:00')
                speaker = entry.get('speaker', 'Unknown')
                text = entry.get('text', '')

                lines.append(f"[{timestamp_start} - {timestamp_end}]")
                lines.append(f"{speaker}:")
                lines.append(text)
                lines.append("")

            lines.append("═" * 70)
            lines.append("TRANSCRIPT STATISTICS")
            lines.append("═" * 70)

            speaker_counts = transcript_data.get('speaker_turn_count', {})
            lines.append(f"Total speaker turns: {len(entries)}")
            lines.append(f"Number of unique speakers: {len(speaker_counts)}")
            lines.append("")

            if speaker_counts:
                lines.append("Speaker Turn Count:")
                for speaker, count in sorted(speaker_counts.items()):
                    lines.append(f"  {speaker}: {count} turn(s)")

            lines.append("")
            lines.append("─" * 70)
            lines.append("End of Transcript")
            lines.append("─" * 70)

            content = '\n'.join(lines)
            logger.info(f"✓ TXT export generated ({len(content)} bytes)")
            return content

        except Exception as exc:
            logger.error(f"TXT export failed: {str(exc)}")
            raise

    def export_pdf(self, transcript_data: Dict, metadata: Dict = None) -> bytes:
        # Create a PDF transcript using ReportLab flowables.
        logger.info("Generating PDF export")

        try:
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=EXPORT_CONFIG['pdf']['page_size'],
                rightMargin=EXPORT_CONFIG['pdf']['margins'],
                leftMargin=EXPORT_CONFIG['pdf']['margins'],
                topMargin=EXPORT_CONFIG['pdf']['margins'],
                bottomMargin=EXPORT_CONFIG['pdf']['margins']
            )

            # ReportLab builds the PDF from a linear story of paragraphs, tables, and spacers.
            story = []
            styles = self._get_pdf_styles()

            title = metadata.get('title', 'Meeting Transcript') if metadata else 'Meeting Transcript'
            subtitle = "Generated transcript export"
            story.append(Paragraph(escape(title), styles['title']))
            story.append(Paragraph(escape(subtitle), styles['subtitle']))
            story.append(Spacer(1, 0.2 * inch))

            if metadata:
                story.append(self._create_metadata_table(
                    metadata,
                    transcript_data,
                    styles
                ))
                story.append(Spacer(1, 0.2 * inch))

            story.append(PageBreak())
            entries = transcript_data.get('transcript_entries', [])
            for entry in entries:
                timestamp_start = entry.get('timestamp_start', '00:00:00')
                timestamp_end = entry.get('timestamp_end', '00:00:00')
                speaker = entry.get('speaker', 'Unknown')
                text = entry.get('text', '')
                confidence = entry.get('confidence', 0)

                timestamp_text = escape(f"[{timestamp_start} - {timestamp_end}]")
                speaker_text = escape(speaker)
                header_text = f"<font color='#06b6d4'>{timestamp_text}</font> <font color='#22c55e'><b>{speaker_text}</b></font>"
                story.append(Paragraph(header_text, styles['transcript_header']))
                story.append(Paragraph(escape(text), styles['transcript_text']))

                if confidence < 1.0:
                    conf_text = f"<font size='8' color='#9ca3af'>(Confidence: {confidence:.0%})</font>"
                    story.append(Paragraph(conf_text, styles['confidence']))

                story.append(Spacer(1, 0.1 * inch))

            story.append(Spacer(1, 0.3 * inch))
            footer_text = (
                f"<b>Transcript Statistics:</b><br/>"
                f"Total entries: {len(entries)}<br/>"
                f"Unique speakers: {transcript_data.get('num_speakers', 0)}<br/>"
                f"Total duration: {self._format_duration(transcript_data.get('total_duration', 0))}<br/>"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            story.append(Paragraph(footer_text, styles['footer']))

            doc.build(story)
            pdf_bytes = pdf_buffer.getvalue()
            logger.info(f"✓ PDF export generated ({len(pdf_bytes)} bytes)")
            return pdf_bytes

        except Exception as exc:
            logger.error(f"PDF export failed: {str(exc)}")
            raise

    @staticmethod
    def _get_pdf_styles() -> Dict:
        # Define the paragraph styles used by the PDF export.
        base_styles = getSampleStyleSheet()
        styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=base_styles['Heading1'],
                fontSize=24,
                textColor=PDF_STYLES['title_color'],
                spaceAfter=6,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ),

            'subtitle': ParagraphStyle(
                'CustomSubtitle',
                parent=base_styles['Heading2'],
                fontSize=12,
                textColor=colors.grey,
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica'
            ),

            'transcript_header': ParagraphStyle(
                'TranscriptHeader',
                parent=base_styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                spaceBefore=6,
                fontName='Helvetica-Bold'
            ),

            'transcript_text': ParagraphStyle(
                'TranscriptText',
                parent=base_styles['Normal'],
                fontSize=10,
                leftIndent=0.25 * inch,
                spaceAfter=6,
                alignment=TA_JUSTIFY,
                fontName='Helvetica'
            ),

            'confidence': ParagraphStyle(
                'Confidence',
                parent=base_styles['Normal'],
                fontSize=8,
                leftIndent=0.25 * inch,
                spaceAfter=4
            ),

            'footer': ParagraphStyle(
                'Footer',
                parent=base_styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                alignment=TA_CENTER
            ),

            'metadata_label': ParagraphStyle(
                'MetadataLabel',
                parent=base_styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold'
            ),

            'metadata_value': ParagraphStyle(
                'MetadataValue',
                parent=base_styles['Normal'],
                fontSize=10
            )
        }
        return styles

    @staticmethod
    def _create_metadata_table(metadata: Dict, transcript_data: Dict, _styles: Dict) -> Table:
        # Build the summary table shown at the top of exported PDFs.
        rows = [
            ["Meeting Information", ""],
            [
                "Title:",
                metadata.get('title', 'Untitled Meeting')
            ],
            [
                "Date:",
                metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
            ],
            [
                "Duration:",
                TranscriptExporter._format_duration(
                    transcript_data.get('total_duration', 0)
                )
            ],
            [
                "Speakers:",
                f"{transcript_data.get('num_speakers', 0)} participants"
            ],
            [
                "Segments:",
                f"{len(transcript_data.get('transcript_entries', []))} entries"
            ]
        ]

        table = Table(rows, colWidths=[1.5 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1a365d')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
        ]))

        return table

    @staticmethod
    def _format_duration(seconds: float) -> str:
        # Format a numeric duration into a short human-readable phrase.
        if seconds < 60:
            return f"{int(seconds)} second{'s' if seconds != 1 else ''}"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"


def export_transcript(transcript_data: Dict, format: str = 'txt', metadata: Dict = None) -> str or bytes:
    # Convenience function used by Flask routes to export transcript data.
    try:
        exporter = TranscriptExporter()
        return exporter.export(transcript_data, format, metadata)
    except Exception as exc:
        logger.error(f"Export error: {str(exc)}")
        raise
