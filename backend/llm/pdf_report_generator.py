"""
PDF Report Generator Module
============================
Generates professional PDF reports in two formats:
1. Doctor/Medical Professional Version (technical, detailed)
2. Patient/Family Friendly Version (simple, easy to understand)

Includes patient name/phone, nutrient recommendations, images, and formatting.
"""

import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image as PILImage


class PDFReportGenerator:
    """Generates professional PDF reports for ultrasound diagnosis."""

    def __init__(self):
        """Initialize PDF generator with styles."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Header style
        self.styles.add(ParagraphStyle(
            name='CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Subheader style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#0d47a1'),
            spaceAfter=4,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=9,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
            leading=13
        ))

        # Patient-friendly style
        self.styles.add(ParagraphStyle(
            name='PatientFriendly',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=6,
            leading=15,
            textColor=colors.HexColor('#212121')
        ))

        # Nutrient header style
        self.styles.add(ParagraphStyle(
            name='NutrientHeader',
            parent=self.styles['Heading3'],
            fontSize=10,
            textColor=colors.HexColor('#1b5e20'),
            spaceAfter=3,
            fontName='Helvetica-Bold'
        ))

        # Small disclaimer style
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['BodyText'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#757575'),
            spaceAfter=4,
            leading=11
        ))

    def _process_image_for_pdf(self, image_path, max_size_inches=2.5):
        """
        Process and prepare image for embedding in PDF.
        Converts to grayscale with CLAHE enhancement.
        Returns path to temp PNG file, or None on failure.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # Convert to grayscale and enhance contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            pil_image = PILImage.fromarray(enhanced)

            # Resize proportionally
            size_px = int(max_size_inches * 96)
            pil_image.thumbnail((size_px, size_px), PILImage.LANCZOS)

            # Save to a named temp file
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            pil_image.save(tmp.name, 'PNG')
            tmp.close()
            return tmp.name

        except Exception as e:
            print(f"Error processing image for PDF: {e}")
            return None

    def _build_patient_info_table(self, diagnosis_data):
        """Build a styled patient info table."""
        patient_name = diagnosis_data.get('patient_name', 'N/A')
        patient_phone = diagnosis_data.get('patient_phone', 'N/A')
        report_id = diagnosis_data.get('report_id', 'N/A')
        timestamp = diagnosis_data.get('timestamp', datetime.now().isoformat())

        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_date = dt.strftime('%d %B %Y, %H:%M:%S')
        except Exception:
            formatted_date = timestamp

        data = [
            ['Patient Name:', patient_name, 'Report ID:', report_id],
            ['Contact Phone:', patient_phone, 'Report Date:', formatted_date],
        ]

        table = Table(data, colWidths=[1.2 * inch, 2.5 * inch, 1.2 * inch, 2.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e3f2fd')),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#e3f2fd')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90caf9')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return table

    def _build_nutrient_table(self, nutrients_list):
        """Build a formatted nutrient table."""
        if not nutrients_list:
            return None

        data = [['#', 'Nutrient', 'Recommended Dose', 'Key Benefit']]

        for i, nutrient in enumerate(nutrients_list, 1):
            data.append([
                str(i),
                nutrient.get('name', ''),
                nutrient.get('dose', ''),
                nutrient.get('benefit', '')
            ])

        col_widths = [0.3 * inch, 1.6 * inch, 1.8 * inch, 3.7 * inch]
        table = Table(data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1b5e20')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f8e9')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a5d6a7')),
        ]))
        return table

    def generate_doctor_report(self, diagnosis_data, image_path, gradcam_path, output_pdf_path):
        """
        Generate professional medical report for doctors.

        Args:
            diagnosis_data: Dictionary with diagnosis information
            image_path: Path to ultrasound image
            gradcam_path: Path to Grad-CAM visualization
            output_pdf_path: Path to save PDF

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            doc = SimpleDocTemplate(
                output_pdf_path,
                pagesize=letter,
                rightMargin=0.6 * inch,
                leftMargin=0.6 * inch,
                topMargin=0.6 * inch,
                bottomMargin=0.6 * inch
            )

            story = []

            # ── HEADER ──────────────────────────────────────────────────────
            story.append(Paragraph(
                "🏥 Deep Learning Based Fetal Ultrasound Analysis",
                self.styles['CustomHeader']
            ))
            story.append(Paragraph(
                "For Early Detection of Fetal Growth Abnormalities — Medical Report",
                self.styles['CustomSubHeader']
            ))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a237e')))
            story.append(Spacer(1, 0.1 * inch))

            # ── PATIENT INFO TABLE ────────────────────────────────────────────
            story.append(Paragraph("PATIENT INFORMATION", self.styles['CustomSubHeader']))
            story.append(self._build_patient_info_table(diagnosis_data))
            story.append(Spacer(1, 0.15 * inch))

            # ── CLINICAL FINDINGS ─────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
            story.append(Paragraph("📋 CLINICAL FINDINGS", self.styles['CustomSubHeader']))

            findings_data = [
                ['Primary Diagnosis:', diagnosis_data['prediction']['primary_diagnosis']],
                ['Diagnostic Confidence:', f"{diagnosis_data['prediction']['confidence_percentage']:.1f}%"],
            ]

            # Risk level
            conf = diagnosis_data['prediction']['confidence_percentage']
            primary_diag = diagnosis_data['prediction']['primary_diagnosis'].lower()
            
            if 'normal' in primary_diag:
                risk_level = "LOW RISK"
            elif conf > 65:
                risk_level = "HIGH RISK — Urgent"
            else:
                risk_level = "MODERATE RISK"
                
            findings_data.append(['Risk Level:', risk_level])

            for condition, prob in diagnosis_data['prediction']['all_probabilities'].items():
                findings_data.append([f"  • {condition}:", f"{prob:.1f}%"])

            findings_table = Table(findings_data, colWidths=[2 * inch, 5.4 * inch])
            findings_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
                ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#e8eaf6')),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#c5cae9')),
            ]))
            story.append(findings_table)
            story.append(Spacer(1, 0.1 * inch))

            # ── DIAGNOSTIC VISUALIZATIONS ──────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
            story.append(Paragraph("🔍 DIAGNOSTIC VISUALIZATIONS", self.styles['CustomSubHeader']))

            images_row = []
            labels_row = []
            temp_files = []

            if image_path and Path(str(image_path)).exists():
                tmp_path = self._process_image_for_pdf(image_path, max_size_inches=2.3)
                if tmp_path:
                    images_row.append(Image(tmp_path, width=2.3 * inch, height=2.0 * inch))
                    labels_row.append(Paragraph('<b>Original Ultrasound</b><br/><small>(AI-processed B&W)</small>', self.styles['Normal']))
                    temp_files.append(tmp_path)

            if gradcam_path and Path(str(gradcam_path)).exists():
                tmp_path = self._process_image_for_pdf(gradcam_path, max_size_inches=2.3)
                if tmp_path:
                    images_row.append(Image(tmp_path, width=2.3 * inch, height=2.0 * inch))
                    labels_row.append(Paragraph('<b>Model Attention Map</b><br/><small>(Grad-CAM Heatmap)</small>', self.styles['Normal']))
                    temp_files.append(tmp_path)

            if images_row:
                num_cols = len(images_row)
                col_w = 7.4 / num_cols * inch
                img_table = Table([images_row, labels_row], colWidths=[col_w] * num_cols)
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#90caf9')),
                    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#e8eaf6')),
                ]))
                story.append(img_table)
                story.append(Spacer(1, 0.05 * inch))
                story.append(Paragraph(
                    "⚠️ Note: Visualizations are AI-generated analysis aids. "
                    "These are NOT photographs of the fetus. Clinical correlation recommended.",
                    self.styles['Disclaimer']
                ))
            story.append(Spacer(1, 0.1 * inch))

            # ── CONDITION ANALYSIS ────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
            story.append(Paragraph("👶 FETAL CONDITION ANALYSIS", self.styles['CustomSubHeader']))
            condition_text = diagnosis_data.get('condition_description', 'Analysis completed.')
            story.append(Paragraph(condition_text, self.styles['CustomBody']))
            story.append(Spacer(1, 0.1 * inch))

            # ── DETAILED REPORT TEXT ──────────────────────────────────────────
            if diagnosis_data.get('report_text'):
                story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
                story.append(Paragraph("📄 COMPLETE DIAGNOSTIC REPORT", self.styles['CustomSubHeader']))
                # Render as preformatted
                lines = diagnosis_data['report_text'].split('\n')
                for line in lines:
                    if line.strip():
                        story.append(Paragraph(
                            line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'),
                            self.styles['CustomBody']
                        ))

            # ── NUTRIENTS SECTION ─────────────────────────────────────────────
            story.append(PageBreak())
            story.append(Paragraph("🥗 NUTRITIONAL RECOMMENDATIONS", self.styles['CustomSubHeader']))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1b5e20')))
            story.append(Spacer(1, 0.1 * inch))

            nutrients_content = diagnosis_data.get('nutrients', '')
            if nutrients_content:
                story.append(Paragraph(
                    "The following nutritional recommendations are tailored to the detected fetal condition. "
                    "Please consult a registered dietitian for a personalized plan.",
                    self.styles['CustomBody']
                ))
                story.append(Spacer(1, 0.08 * inch))
                lines = nutrients_content.split('\n')
                for line in lines:
                    if line.strip():
                        safe_line = (line
                                     .replace('&', '&amp;')
                                     .replace('<', '&lt;')
                                     .replace('>', '&gt;'))
                        story.append(Paragraph(safe_line, self.styles['CustomBody']))

            # ── DISCLAIMER ────────────────────────────────────────────────────
            story.append(Spacer(1, 0.2 * inch))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#c5cae9')))
            story.append(Paragraph(
                "📌 MEDICAL DISCLAIMER: This AI-generated report is intended to assist qualified healthcare "
                "professionals and should not replace comprehensive clinical evaluation, patient history, "
                "and specialist consultation. All clinical decisions must be made by licensed medical personnel.",
                self.styles['Disclaimer']
            ))

            # Build PDF
            doc.build(story)

            # Clean up temp files
            for tf in temp_files:
                try:
                    os.unlink(tf)
                except Exception:
                    pass

            return True

        except Exception as e:
            print(f"Error generating doctor report: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_patient_report(self, diagnosis_data, image_path, gradcam_path, output_pdf_path):
        """
        Generate patient/family-friendly report with nutrients.

        Args:
            diagnosis_data: Dictionary with diagnosis information
            image_path: Path to ultrasound image
            gradcam_path: Path to Grad-CAM visualization
            output_pdf_path: Path to save PDF

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            doc = SimpleDocTemplate(
                output_pdf_path,
                pagesize=letter,
                rightMargin=0.6 * inch,
                leftMargin=0.6 * inch,
                topMargin=0.6 * inch,
                bottomMargin=0.6 * inch
            )

            story = []

            # ── HEADER ──────────────────────────────────────────────────────
            story.append(Paragraph(
                "👶 Fetal Ultrasound Analysis — Patient Report",
                self.styles['CustomHeader']
            ))
            story.append(Paragraph(
                "Deep Learning Based Early Detection of Fetal Growth Abnormalities",
                self.styles['CustomSubHeader']
            ))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a237e')))
            story.append(Spacer(1, 0.1 * inch))

            # ── PATIENT INFO ─────────────────────────────────────────────────
            patient_name = diagnosis_data.get('patient_name', 'N/A')
            patient_phone = diagnosis_data.get('patient_phone', 'N/A')
            report_id = diagnosis_data.get('report_id', 'N/A')

            story.append(Paragraph("YOUR INFORMATION", self.styles['CustomSubHeader']))
            story.append(self._build_patient_info_table(diagnosis_data))
            story.append(Spacer(1, 0.15 * inch))

            # ── IMPORTANT NOTICE ──────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#ff8f00')))
            notice = (
                "<b>⚠️ IMPORTANT NOTICE:</b><br/>"
                "This report is a <b>preliminary AI-assisted analysis</b> for <b>{name}</b>. "
                "It is <b>NOT a replacement for professional medical advice.</b> "
                "Please discuss these results with your doctor immediately.<br/>"
                "A summary has been sent to: <b>{phone}</b>"
            ).format(name=patient_name, phone=patient_phone)
            story.append(Paragraph(notice, self.styles['PatientFriendly']))
            story.append(Spacer(1, 0.1 * inch))

            # ── DIAGNOSIS RESULT ──────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
            story.append(Paragraph("🔬 What We Found:", self.styles['CustomSubHeader']))
            diagnosis = diagnosis_data['prediction']['primary_diagnosis']
            confidence = diagnosis_data['prediction']['confidence_percentage']
            diag_lower = diagnosis.lower()

            if 'normal' in diag_lower:
                simple_text = (
                    '✅ <b>GOOD NEWS!</b> The ultrasound analysis shows normal fetal development. '
                    'Your baby appears to be growing well. Continue regular prenatal check-ups.'
                )
            elif 'fgr' in diag_lower or 'restriction' in diag_lower:
                simple_text = (
                    '⚠️ <b>ATTENTION:</b> The analysis shows signs of Fetal Growth Restriction (FGR) — '
                    'your baby may be growing slower than expected. '
                    'Please consult your doctor immediately for further evaluation.'
                )
            else:
                simple_text = (
                    '🔴 <b>IMPORTANT:</b> The analysis detected some unusual findings. '
                    'Your doctor will discuss next steps and additional tests with you. '
                    'Early detection allows for better care — please see your doctor right away.'
                )

            story.append(Paragraph(simple_text, self.styles['PatientFriendly']))

            confidence_text = (
                f"<b>AI Confidence Level:</b> {confidence:.1f}% — "
                f"{'High confidence in this result.' if confidence > 65 else 'Moderate confidence — additional clinical tests recommended.'}"
            )
            story.append(Paragraph(confidence_text, self.styles['PatientFriendly']))
            story.append(Spacer(1, 0.1 * inch))

            # ── ULTRASOUND IMAGE ──────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
            story.append(Paragraph("🖼️ Your Ultrasound Image:", self.styles['CustomSubHeader']))

            temp_files = []

            if image_path and Path(str(image_path)).exists():
                tmp_path = self._process_image_for_pdf(image_path, max_size_inches=2.5)
                if tmp_path:
                    story.append(Spacer(1, 0.05 * inch))
                    img = Image(tmp_path, width=2.5 * inch, height=2.2 * inch)
                    story.append(Table([[img]], colWidths=[7.4 * inch]))
                    temp_files.append(tmp_path)

            story.append(Paragraph(
                "⚠️ Note: This is a processed black-and-white analysis image. "
                "<b>This is NOT an actual photograph of your baby.</b> "
                "It is an AI representation used to highlight areas of interest.",
                self.styles['Disclaimer']
            ))
            story.append(Spacer(1, 0.1 * inch))

            # ── WHAT TO DO NEXT ──────────────────────────────────────────────
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#90caf9')))
            story.append(Paragraph("📋 What Should You Do Next?", self.styles['CustomSubHeader']))

            is_normal = 'normal' in diag_lower
            if not is_normal and confidence > 65:
                next_steps = [
                    ("<b>1. See Your Doctor Immediately</b>", "Your results show concerns. Please schedule an appointment with your OB/GYN right away."),
                    ("<b>2. Bring This Report</b>", "Show this report to your doctor for detailed discussion."),
                    ("<b>3. Follow Medical Advice</b>", "Your doctor may recommend more frequent ultrasounds or additional tests."),
                    ("<b>4. Follow the Nutrition Plan Below</b>", "See the nutrition section for dietary recommendations specific to your condition."),
                    ("<b>5. Stay Calm</b>", "Early detection helps doctors provide better care for you and your baby."),
                ]
            else:
                next_steps = [
                    ("<b>1. Schedule Your Next Check-up</b>", "Share these results in your next regular appointment."),
                    ("<b>2. Continue Regular Prenatal Care</b>", "Keep all your prenatal appointments as scheduled."),
                    ("<b>3. Follow the Nutrition Guide Below</b>", "Good nutrition is essential for your baby's development."),
                    ("<b>4. Stay Active & Rest Well</b>", "Light exercise and adequate rest support healthy pregnancy."),
                    ("<b>5. Stay Positive</b>", "Most pregnancies develop normally. Stay positive!"),
                ]

            for title, detail in next_steps:
                story.append(Paragraph(f"{title}<br/>{detail}", self.styles['PatientFriendly']))

            # ── NUTRIENTS ─────────────────────────────────────────────────────
            story.append(PageBreak())
            story.append(Paragraph("🥗 Your Nutrition Guide", self.styles['CustomSubHeader']))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1b5e20')))
            story.append(Spacer(1, 0.08 * inch))

            nutrients_content = diagnosis_data.get('nutrients', '')
            if nutrients_content:
                story.append(Paragraph(
                    "Based on your analysis result, here are the key nutrients your body needs. "
                    "Please consult a dietitian for a personalized meal plan.",
                    self.styles['PatientFriendly']
                ))
                story.append(Spacer(1, 0.05 * inch))
                lines = nutrients_content.split('\n')
                for line in lines:
                    if line.strip():
                        safe_line = (line
                                     .replace('&', '&amp;')
                                     .replace('<', '&lt;')
                                     .replace('>', '&gt;'))
                        story.append(Paragraph(safe_line, self.styles['PatientFriendly']))
            else:
                # Fallback general nutrition advice
                general = [
                    "• <b>Folate/Folic Acid</b>: 600 mcg/day — leafy greens, legumes, fortified cereals",
                    "• <b>Iron</b>: 27 mg/day — meat, beans, spinach",
                    "• <b>Calcium</b>: 1000 mg/day — dairy, almonds, broccoli",
                    "• <b>Omega-3 (DHA)</b>: 200–300 mg/day — salmon, walnuts, fish oil",
                    "• <b>Protein</b>: 71 g/day — eggs, poultry, legumes, tofu",
                    "• <b>Vitamin D</b>: 600 IU/day — sunlight, fortified milk, egg yolks",
                ]
                for item in general:
                    story.append(Paragraph(item, self.styles['PatientFriendly']))

            # ── DISCLAIMER ─────────────────────────────────────────────────────
            story.append(Spacer(1, 0.2 * inch))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#ffcc02')))
            story.append(Paragraph(
                f"📌 This report was prepared for {patient_name} (Phone: {patient_phone}). "
                "AI analysis is for informational purposes only. It does not replace professional "
                "medical diagnosis or treatment. Always consult with a qualified healthcare provider. "
                f"Report ID: {report_id}",
                self.styles['Disclaimer']
            ))

            # Build PDF
            doc.build(story)

            # Clean up temp files
            for tf in temp_files:
                try:
                    os.unlink(tf)
                except Exception:
                    pass

            return True

        except Exception as e:
            print(f"Error generating patient report: {e}")
            import traceback
            traceback.print_exc()
            return False
