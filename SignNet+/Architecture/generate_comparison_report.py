"""
SignNet Training Comparison Report Generator
Requirements: pip install reportlab
Usage: python generate_comparison_report.py
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime


def create_comparison_report():
    doc = SimpleDocTemplate(
        "SignNet_Training_Comparison_Report.pdf",
        pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'],
                                 fontSize=24, spaceAfter=30, alignment=TA_CENTER, textColor=colors.HexColor('#1a365d'))
    subtitle_style = ParagraphStyle('CustomSubtitle', parent=styles['Normal'],
                                    fontSize=12, spaceAfter=20, alignment=TA_CENTER,
                                    textColor=colors.HexColor('#4a5568'))
    heading1_style = ParagraphStyle('CustomHeading1', parent=styles['Heading1'],
                                    fontSize=16, spaceBefore=20, spaceAfter=12, textColor=colors.HexColor('#2c5282'))
    heading2_style = ParagraphStyle('CustomHeading2', parent=styles['Heading2'],
                                    fontSize=13, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2d3748'))
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'],
                                fontSize=10, spaceAfter=8, alignment=TA_JUSTIFY)

    content = []

    # TITLE PAGE
    content.append(Spacer(1, 3 * cm))
    content.append(Paragraph("SignNet", title_style))
    content.append(Paragraph("Training Comparison Report", ParagraphStyle(
        'SubTitle', parent=title_style, fontSize=18, spaceAfter=20)))
    content.append(Spacer(1, 1 * cm))
    content.append(Paragraph("Versuch 1 vs Versuch 2", subtitle_style))
    content.append(Spacer(1, 2 * cm))

    meta_data = [
        ['Datum:', datetime.now().strftime('%d. November %Y')],
        ['Autoren:', 'Andrei Chirila, Roman Schläpfer'],
        ['Institution:', 'OST - Ostschweizer Fachhochschule'],
        ['Projekt:', 'Machine Learning Abschlussprojekt'],
        ['Dataset:', 'RWTH-PHOENIX-2014'],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 10 * cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(meta_table)
    content.append(PageBreak())

    # EXECUTIVE SUMMARY
    content.append(Paragraph("1. Executive Summary", heading1_style))
    content.append(Paragraph("""
    Dieser Report vergleicht zwei Trainingsläufe des SignNet-Modells für die Erkennung 
    von Gebärdensprache. Beide Versuche verwenden die Top-200 Klassen des RWTH-PHOENIX-2014 
    Datasets und die gleiche Multi-Stream GCN + Transformer Architektur. Der Hauptunterschied 
    liegt im Learning Rate Scheduler.
    """.strip(), body_style))
    content.append(Spacer(1, 0.5 * cm))

    content.append(Paragraph("<b>Wichtigste Erkenntnisse:</b>", body_style))
    for finding in [
        "Test WER: Versuch 1 (68.8%) marginal besser als Versuch 2 (69.2%)",
        "Test Loss: Versuch 2 (10.13) deutlich besser als Versuch 1 (12.23)",
        "Overfitting: Versuch 2 zeigt früheres Overfitting (ab Epoch 20)",
        "Trainingszeit: Versuch 2 konvergiert schneller (57 vs 76 Epochs)",
        "Scheduler: Cosine Decay (V1) scheint stabiler als Linear Decay (V2)"
    ]:
        content.append(Paragraph(f"• {finding}", body_style))
    content.append(Spacer(1, 1 * cm))

    # RESULTS COMPARISON
    content.append(Paragraph("2. Ergebnisvergleich", heading1_style))
    content.append(Paragraph("2.1 Hauptmetriken", heading2_style))

    results_data = [
        ['Metrik', 'Versuch 1', 'Versuch 2', 'Differenz', 'Gewinner'],
        ['Test WER', '68.8%', '69.2%', '+0.4%', 'V1 ✓'],
        ['Best Val WER', '70.0%', '70.0%', '±0.0%', 'Gleich'],
        ['Test Loss', '12.23', '10.13', '-2.10', 'V2 ✓'],
        ['Best Epoch', '61', '42', '-19', 'V2 ✓'],
        ['Early Stop Epoch', '76', '57', '-19', 'V2 ✓'],
        ['Train Loss (final)', '1.02', '~2.0', '+0.98', 'V1 ✓'],
    ]
    results_table = Table(results_data, colWidths=[3.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f7')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(results_table)
    content.append(Spacer(1, 0.8 * cm))

    content.append(Paragraph("2.2 Training Verlauf", heading2_style))
    training_data = [
        ['Aspekt', 'Versuch 1', 'Versuch 2'],
        ['Trainingszeit', '~55 Minuten', '~40 Minuten'],
        ['Konvergenz', 'Langsam, stabil', 'Schneller, früh gestoppt'],
        ['Overfitting Start', '~Epoch 50+', '~Epoch 20'],
        ['Val Loss Minimum', '~12.23', '~7.5 (Epoch 15-20)'],
        ['Stabilität', 'Hoch', 'Mittel'],
    ]
    training_table = Table(training_data, colWidths=[4 * cm, 4.5 * cm, 4.5 * cm])
    training_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f7')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(training_table)
    content.append(PageBreak())

    # HYPERPARAMETERS
    content.append(Paragraph("3. Hyperparameter Vergleich", heading1_style))
    content.append(Paragraph("3.1 Training Konfiguration", heading2_style))

    hyperparam_data = [
        ['Parameter', 'Versuch 1', 'Versuch 2', 'Geändert?'],
        ['Optimizer', 'AdamW', 'AdamW', '✗'],
        ['Learning Rate', '1e-4', '1e-4', '✗'],
        ['Weight Decay', '0.01', '0.01', '✗'],
        ['Scheduler', 'Warmup + Cosine Decay', 'Warmup + Linear Decay', '✓'],
        ['Min Learning Rate', '1e-6', '~2e-5', '✓'],
        ['Batch Size', '8', '8', '✗'],
        ['Max Epochs', '100', '100', '✗'],
        ['Early Stopping Patience', '15', '15', '✗'],
        ['Mixed Precision', 'FP16', 'FP16', '✗'],
        ['Gradient Clipping', '1.0', '1.0', '✗'],
    ]
    hyperparam_table = Table(hyperparam_data, colWidths=[4.5 * cm, 3.5 * cm, 3.5 * cm, 2 * cm])
    hyperparam_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f7')]),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#fef3c7')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#fef3c7')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    content.append(hyperparam_table)
    content.append(Spacer(1, 0.8 * cm))

    content.append(Paragraph("3.2 Modell Architektur (Unverändert)", heading2_style))
    arch_data = [
        ['Parameter', 'Wert'],
        ['Total Parameters', '21,719,476'],
        ['GCN Hidden Dims', '[64, 128, 256]'],
        ['Transformer d_model', '512'],
        ['Transformer Layers', '6'],
        ['Transformer Heads', '8'],
        ['Feed-Forward Dim', '2048'],
        ['Model Size', '82.9 MB'],
    ]
    arch_table = Table(arch_data, colWidths=[5 * cm, 5 * cm])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#48bb78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fff4')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    content.append(arch_table)
    content.append(PageBreak())

    # ANALYSIS
    content.append(Paragraph("4. Analyse der Unterschiede", heading1_style))
    content.append(Paragraph("4.1 Scheduler-Vergleich", heading2_style))
    content.append(Paragraph("""
    <b>Versuch 1 - Cosine Decay:</b> Der Learning Rate sinkt sanft in einer Kosinuskurve 
    von 1e-4 auf 1e-6. Dies ermöglicht feines Tuning am Ende des Trainings.
    <br/><br/>
    <b>Versuch 2 - Linear Decay:</b> Der Learning Rate sinkt linear von 1e-4 auf ~2e-5. 
    Der höhere Endwert führt zu weniger feinem Tuning.
    """, body_style))

    content.append(Paragraph("4.2 Overfitting-Analyse", heading2_style))
    content.append(Paragraph("""
    <b>Versuch 2 zeigt früheres Overfitting:</b>
    <br/>• Validation Loss sinkt bis Epoch 15-20 auf ~7.5
    <br/>• Danach steigt Val Loss wieder auf ~11
    <br/>• Modell memorisiert Trainingsdaten
    <br/><br/>
    <b>Versuch 1 war stabiler:</b>
    <br/>• Overfitting begann erst ab Epoch 50+
    <br/>• Cosine Decay regularisiert besser
    """, body_style))
    content.append(Spacer(1, 0.8 * cm))

    # CONFUSION MATRIX
    content.append(Paragraph("5. Confusion Matrix Analyse", heading1_style))
    content.append(Paragraph("5.1 Gut erkannte Glosses", heading2_style))
    good_data = [
        ['Gloss', 'Erkennungsrate'],
        ['REGEN', '~45%'], ['WOLKE', '~35%'], ['MORGEN', '~30%'],
        ['SONNE', '~25%'], ['HEUTE', '~25%'],
    ]
    good_table = Table(good_data, colWidths=[4 * cm, 4 * cm])
    good_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#48bb78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fff4')]),
    ]))
    content.append(good_table)
    content.append(Spacer(1, 0.8 * cm))

    content.append(Paragraph("5.2 Häufige Verwechslungen", heading2_style))
    confusion_data = [
        ['True', 'Predicted als', 'Grund'],
        ['WEHEN', 'REGEN', 'Ähnliche Handbewegung'],
        ['KOMMEN', 'REGEN', 'Ähnliche Bewegungsrichtung'],
        ['BISSCHEN', '__ON__', 'Schwer zu unterscheiden'],
    ]
    confusion_table = Table(confusion_data, colWidths=[3 * cm, 3 * cm, 6 * cm])
    confusion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e53e3e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff5f5')]),
    ]))
    content.append(confusion_table)
    content.append(PageBreak())

    # RECOMMENDATIONS
    content.append(Paragraph("6. Empfehlungen für Versuch 3", heading1_style))
    content.append(Paragraph("""
    <b>6.1 Scheduler:</b> Zurück zu Cosine Decay (eta_min=1e-6)
    <br/><br/>
    <b>6.2 Regularisierung:</b>
    <br/>• Dropout: 0.4 (statt 0.3)
    <br/>• Weight Decay: 0.05 (statt 0.01)
    <br/>• Label Smoothing: 0.1
    <br/><br/>
    <b>6.3 Data Augmentation:</b>
    <br/>• occlusion_prob: 20% (statt 15%)
    <br/>• frame_dropout: 15% (statt 10%)
    <br/>• Mixup: alpha=0.2
    <br/><br/>
    <b>6.4 Erwartete Verbesserung:</b> Test WER 60-65%
    """, body_style))
    content.append(Spacer(1, 1 * cm))

    # CONCLUSION
    content.append(Paragraph("7. Fazit", heading1_style))
    eval_data = [
        ['Aspekt', 'Gewinner', 'Begründung'],
        ['Test WER', 'Versuch 1', '68.8% vs 69.2%'],
        ['Test Loss', 'Versuch 2', '10.13 vs 12.23'],
        ['Trainingszeit', 'Versuch 2', '~40 min vs ~55 min'],
        ['Stabilität', 'Versuch 1', 'Weniger Overfitting'],
    ]
    eval_table = Table(eval_data, colWidths=[3.5 * cm, 3 * cm, 6 * cm])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#edf2f7')]),
    ]))
    content.append(eval_table)
    content.append(Spacer(1, 0.8 * cm))

    content.append(Paragraph("""
    <b>Schlussfolgerung:</b> Der Wechsel von Cosine Decay zu Linear Decay hat keine 
    Verbesserung gebracht. Versuch 2 zeigt früheres Overfitting und marginal schlechtere 
    Test WER. Empfehlung für Versuch 3: Cosine Decay beibehalten + stärkere Regularisierung.
    """, body_style))

    doc.build(content)
    print("✅ PDF erstellt: SignNet_Training_Comparison_Report.pdf")


if __name__ == "__main__":
    create_comparison_report()