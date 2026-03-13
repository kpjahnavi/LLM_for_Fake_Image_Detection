import os
import json
import warnings
import io
from contextlib import redirect_stdout, redirect_stderr
from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import logging
import html as html_module

# Reduce noisy HF/Transformers logs before model modules import.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub.*",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from pipeline.run_pipeline import predict_image

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except Exception:
    pass

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "outputs/reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

latest_report = {}


@app.route("/", methods=["GET", "POST"])
def home():
    image_path = None
    prediction = None
    confidence = None
    heatmap_path = None
    explanation_text = None

    if request.method == "POST":
        logger.info("Received POST request for image analysis")
        image = request.files.get("image")

        if image:
            logger.info(f"Processing image: {image.filename}")
            save_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(save_path)

            try:
                logger.info("Starting prediction pipeline...")
                (
                    prediction,
                    confidence,
                    heatmap_path,
                    explanation_text,
                ) = predict_image(save_path)

                logger.info(f"Prediction: {prediction}, Confidence: {confidence}")

                # -------- Save TXT Report --------
                txt_path = os.path.join(REPORT_FOLDER, "report.txt")
                with open(txt_path, "w", encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("VISIONXPLAIN – FAKE IMAGE DETECTION REPORT\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write("ANALYSIS METADATA\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Image File: {os.path.basename(save_path)}\n")
                    f.write(f"Analysis Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("CLASSIFICATION RESULT\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Prediction: {prediction.upper()}\n")
                    f.write(f"Confidence Score: {confidence*100:.2f}%\n\n")
                    
                    f.write("AI-GENERATED EXPLANATION\n")
                    f.write("-" * 80 + "\n")
                    # Wrap text properly
                    from textwrap import fill
                    wrapped_explanation = fill(explanation_text, width=80)
                    f.write(wrapped_explanation + "\n\n")
                    
                    f.write("ANALYSIS DETAILS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Original Image Path: {save_path}\n")
                    f.write(f"Heatmap Path: {heatmap_path}\n")
                    f.write(f"Model: CLIP ViT-L/14 + Custom Classifier\n")
                    f.write(f"Feature Dimension: 768\n\n")
                    
                    f.write("=" * 80 + "\n")
                    f.write("END OF REPORT\n")
                    f.write("=" * 80 + "\n")

                # -------- Save PDF Report with Images --------
                from reportlab.lib.pagesizes import A4, letter
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
                from reportlab.lib import colors
                from reportlab.lib.enums import TA_CENTER, TA_LEFT

                pdf_path = os.path.join(REPORT_FOLDER, "report.pdf")
                
                # Create PDF document
                doc = SimpleDocTemplate(
                    pdf_path,
                    pagesize=letter,
                    rightMargin=0.5*inch,
                    leftMargin=0.5*inch,
                    topMargin=0.5*inch,
                    bottomMargin=0.5*inch
                )
                
                # Container for PDF elements
                elements = []
                styles = getSampleStyleSheet()
                
                # Custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#0066CC'),
                    spaceAfter=12,
                    alignment=TA_CENTER,
                    fontName='Helvetica-Bold'
                )
                
                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=14,
                    textColor=colors.HexColor('#0066CC'),
                    spaceAfter=12,
                    spaceBefore=12,
                    fontName='Helvetica-Bold',
                    borderPadding=8,
                    borderWidth=1,
                    borderColor=colors.HexColor('#CCCCCC'),
                    leftIndent=0,
                    rightIndent=0,
                    alignment=TA_LEFT
                )
                
                body_style = ParagraphStyle(
                    'CustomBody',
                    parent=styles['BodyText'],
                    fontSize=11,
                    leading=16,
                    alignment=TA_LEFT,
                    spaceAfter=10
                )
                
                # Title
                elements.append(Paragraph("VisionXplain – Fake Image Detection Report", title_style))
                elements.append(Spacer(1, 0.2*inch))
                
                # Metadata table
                metadata_data = [
                    ['Image File', os.path.basename(save_path)],
                    ['Analysis Date', __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ]
                metadata_table = Table(metadata_data, colWidths=[1.8*inch, 4.2*inch])
                metadata_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F2FF')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(metadata_table)
                elements.append(Spacer(1, 0.2*inch))
                
                # Classification Results - wrapped in table for proper width
                heading_table = Table([
                    [Paragraph("Classification Result", heading_style)]
                ], colWidths=[6*inch])
                heading_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                elements.append(heading_table)
                elements.append(Spacer(1, 0.1*inch))
                
                result_color = colors.HexColor('#FFE6E6') if prediction.lower() == 'fake' else colors.HexColor('#E6FFE6')
                result_text_color = colors.HexColor('#CC0000') if prediction.lower() == 'fake' else colors.HexColor('#00AA00')
                
                result_data = [
                    ['Prediction', f'{prediction.upper()}'],
                    ['Confidence Score', f'{confidence*100:.2f}%'],
                ]
                result_table = Table(result_data, colWidths=[1.8*inch, 4.2*inch])
                result_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F2FF')),
                    ('BACKGROUND', (1, 0), (1, 0), result_color),
                    ('TEXTCOLOR', (1, 0), (1, 0), result_text_color),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (1, 0), (1, 0), 14),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTSIZE', (0, 0), (-1, -1), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(result_table)
                elements.append(Spacer(1, 0.25*inch))
                
                # Images section - wrapped in table for proper width
                img_heading_table = Table([
                    [Paragraph("Analysis Images", heading_style)]
                ], colWidths=[6*inch])
                img_heading_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                elements.append(img_heading_table)
                elements.append(Spacer(1, 0.12*inch))
                
                # Create image table for side-by-side display
                image_elements = []
                
                # Original Image
                original_img_cell = None
                try:
                    if os.path.exists(save_path):
                        img = Image(save_path, width=2.8*inch, height=2.8*inch)
                        img_style = ParagraphStyle(
                            'ImageLabel',
                            parent=styles['Normal'],
                            fontSize=11,
                            alignment=TA_CENTER,
                            fontName='Helvetica-Bold',
                            textColor=colors.HexColor('#0066CC'),
                            spaceAfter=8
                        )
                        original_img_cell = [
                            Paragraph("Original Image", img_style),
                            img
                        ]
                except Exception as e:
                    logger.warning(f"Could not add original image to PDF: {e}")
                
                # Heatmap Image
                heatmap_img_cell = None
                try:
                    if os.path.exists(heatmap_path):
                        heatmap_img = Image(heatmap_path, width=2.8*inch, height=2.8*inch)
                        img_style = ParagraphStyle(
                            'ImageLabel',
                            parent=styles['Normal'],
                            fontSize=11,
                            alignment=TA_CENTER,
                            fontName='Helvetica-Bold',
                            textColor=colors.HexColor('#0066CC'),
                            spaceAfter=8
                        )
                        heatmap_img_cell = [
                            Paragraph("Attention Heatmap", img_style),
                            heatmap_img
                        ]
                except Exception as e:
                    logger.warning(f"Could not add heatmap to PDF: {e}")
                
                # Add images side by side
                if original_img_cell or heatmap_img_cell:
                    left_cell = original_img_cell if original_img_cell else []
                    right_cell = heatmap_img_cell if heatmap_img_cell else []
                    
                    img_table = Table([
                        [left_cell if left_cell else "", right_cell if right_cell else ""]
                    ], colWidths=[3.2*inch, 3.2*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    elements.append(img_table)
                
                elements.append(Spacer(1, 0.2*inch))
                
                elements.append(PageBreak())
                
                # Explanation - wrapped in table for proper width
                exp_heading_table = Table([
                    [Paragraph("AI-Generated Explanation", heading_style)]
                ], colWidths=[6*inch])
                exp_heading_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                elements.append(exp_heading_table)
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph(explanation_text, body_style))
                elements.append(Spacer(1, 0.25*inch))
                
                # Technical Details - wrapped in table for proper width
                tech_heading_table = Table([
                    [Paragraph("Technical Details", heading_style)]
                ], colWidths=[6*inch])
                tech_heading_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                elements.append(tech_heading_table)
                elements.append(Spacer(1, 0.1*inch))
                tech_data = [
                    ['Feature Extractor', 'CLIP ViT-L/14'],
                    ['Feature Dimension', '768'],
                    ['Classifier Type', 'Neural Network + Attention'],
                    ['Classification Type', 'Binary (Real/Fake)'],
                    ['Explainability Method', 'Attention Heatmaps + Vision-Language Reasoning'],
                ]
                tech_table = Table(tech_data, colWidths=[2.2*inch, 3.8*inch])
                tech_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F2FF')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(tech_table)
                
                # Build PDF
                doc.build(elements)

                latest_report["txt"] = txt_path
                latest_report["pdf"] = pdf_path

                # Escape HTML special characters in the data
                safe_prediction = html_module.escape(str(prediction))
                safe_explanation = html_module.escape(str(explanation_text))
                
                # Convert file paths to URLs that can be served via frontend
                # Images are served via the Node.js server at /api/images/ and /api/heatmaps/
                image_filename = os.path.basename(save_path)
                heatmap_filename = os.path.basename(heatmap_path)
                
                safe_image_path = f"/api/images/{image_filename}"
                safe_heatmap_path = f"/api/heatmaps/{heatmap_filename}"
                
                logger.info(f"Image path: {safe_image_path}")
                logger.info(f"Heatmap path: {safe_heatmap_path}")
                
                # Return HTML response with properly escaped data attributes
                html_response = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body>
<div data-prediction="{safe_prediction}" 
     data-confidence="{confidence}" 
     data-image-path="{safe_image_path}" 
     data-heatmap-path="{safe_heatmap_path}" 
     data-explanation="{safe_explanation}"></div>
</body>
</html>"""
                logger.info("Prediction complete, returning results")
                return html_response

            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}", exc_info=True)
                # Return error response as JSON
                return jsonify({"error": str(e), "type": type(e).__name__}), 500
        else:
            logger.error("No image file provided")
            return jsonify({"error": "No image file provided"}), 400

    return render_template(
        "index.html",
        image_path=image_path,
        prediction=prediction,
        confidence=confidence,
        heatmap_path=heatmap_path,
        explanation_text=explanation_text,
    )


@app.route("/download/<filetype>")
def download(filetype):
    if filetype not in latest_report:
        return jsonify({"error": "Report not found"}), 404
    return send_file(latest_report[filetype], as_attachment=True)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint that returns JSON instead of HTML"""
    image = request.files.get("image")

    if not image:
        return jsonify({"error": "No image provided"}), 400

    try:
        save_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(save_path)

        (
            prediction,
            confidence,
            heatmap_path,
            explanation_text,
        ) = predict_image(save_path)

        # -------- Save Reports --------
        txt_path = os.path.join(REPORT_FOLDER, "report.txt")
        with open(txt_path, "w") as f:
            f.write("VisionXplain – Fake Image Detection Report\n\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Confidence: {confidence:.2f}\n\n")
            f.write("Explanation:\n")
            f.write(explanation_text)

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        pdf_path = os.path.join(REPORT_FOLDER, "report.pdf")
        c = canvas.Canvas(pdf_path, pagesize=A4)
        text = c.beginText(40, 800)
        text.textLine("VisionXplain – Fake Image Detection Report")
        text.textLine("")
        text.textLine(f"Prediction: {prediction}")
        text.textLine(f"Confidence: {confidence:.2f}")
        text.textLine("")
        text.textLine("Explanation:")
        text.textLine(explanation_text)
        c.drawText(text)
        c.save()

        latest_report["txt"] = txt_path
        latest_report["pdf"] = pdf_path

        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence),
            "image_path": save_path,
            "heatmap_path": heatmap_path,
            "explanation": explanation_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
