"""
Flask Application - Main Backend
==================================
Orchestrates the complete diagnostic pipeline:
1. Image upload (JPEG, PNG, DICOM only)
2. CNN inference
3. RAG retrieval
4. Report generation (with patient info + nutrients)
5. Visualization
6. SMS notification to patient phone
"""

import sys
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import json
import traceback
import tempfile
from datetime import datetime
from pathlib import Path

# Import custom modules
from model.inference import InferencePipeline
from model.gradcam import GradCAM, ClinicalVisualization
from rag.retriever import MedicalKnowledgeRetriever
from llm.report_generator import RadiologyReportGenerator
from llm.pdf_report_generator import PDFReportGenerator
from db import db


from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
# ONLY allow medical imaging formats — NO GIF, NO arbitrary files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'dcm'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize pipeline components
print("[STARTUP] Initializing AI components...")
try:
    inference_pipeline = InferencePipeline(model_name='resnet50', device='cpu')
    print("[OK] CNN model loaded")

    rag_retriever = MedicalKnowledgeRetriever()
    print("[OK] RAG retriever initialized")

    report_generator = RadiologyReportGenerator()
    print("[OK] Report generator initialized")

    pdf_generator = PDFReportGenerator()
    print("[OK] PDF generator initialized")

    print("[STARTUP] All components initialized successfully!\n")
except Exception as e:
    print(f"[ERROR] Failed to initialize components: {e}")
    traceback.print_exc()

# Track server start time for uptime
SERVER_START_TIME = datetime.now()


def allowed_file(filename):
    """Check if uploaded file has allowed extension (PNG, JPG, JPEG, DICOM only)."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in app.config['ALLOWED_EXTENSIONS']


def get_file_type_label(filename):
    """Return human-readable file type label."""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    labels = {
        'png': 'PNG Image',
        'jpg': 'JPEG Image',
        'jpeg': 'JPEG Image',
        'dcm': 'DICOM Medical Image'
    }
    return labels.get(ext, 'Unknown')


def handle_dicom_file(filepath):
    """
    Convert DICOM file to PNG for processing.
    Returns the path to the converted PNG, or original path if not DICOM.
    """
    if not filepath.lower().endswith('.dcm'):
        return filepath

    try:
        import pydicom
        import numpy as np
        from PIL import Image as PILImage

        ds = pydicom.dcmread(filepath)
        pixel_array = ds.pixel_array.astype(np.float32)

        # Normalize to 0-255
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)

        # Handle grayscale vs RGB
        if len(pixel_array.shape) == 2:
            img = PILImage.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            img = PILImage.fromarray(pixel_array)

        # Save as PNG
        png_path = filepath.replace('.dcm', '_converted.png')
        img.save(png_path)
        print(f"  [OK] DICOM converted to PNG: {png_path}")
        return png_path

    except ImportError:
        print("  [WARN] pydicom not installed — treating DICOM as raw file")
        return filepath
    except Exception as e:
        print(f"  [WARN] DICOM conversion failed: {e}")
        return filepath


def validate_ultrasound_image(filepath):
    """
    Validate that the image is a grayscale/B&W fetal ultrasound image.
    Rejects colorful photos, selfies, and non-medical images.

    Returns:
        (is_valid: bool, error_message: str)
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(filepath))
        if img is None:
            return False, "Could not read the image file. Please upload a valid image."

        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return False, "Image resolution is too low for analysis. Please upload a higher-quality ultrasound image."

        # ── CHECK 1: Grayscale / B&W validation ─────────────────────────────
        # Ultrasound images are almost always grayscale — R, G, B channels are nearly equal.
        b, g, r = cv2.split(img)
        rg_diff = float(np.mean(np.abs(r.astype(np.int32) - g.astype(np.int32))))
        rb_diff = float(np.mean(np.abs(r.astype(np.int32) - b.astype(np.int32))))
        gb_diff = float(np.mean(np.abs(g.astype(np.int32) - b.astype(np.int32))))
        avg_color_diff = (rg_diff + rb_diff + gb_diff) / 3.0

        # If average inter-channel difference > 18, image is colourful (not ultrasound)
        if avg_color_diff > 18.0:
            return (
                False,
                "The inputted image cannot be scanned. "
                "Fetal ultrasound images must be in black & white / grayscale format. "
                "This appears to be a colour photograph or non-medical image. "
                "Please upload a valid fetal ultrasound image."
            )

        # ── CHECK 2: Brightness sanity ──────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        # Ultrasounds: mostly dark background (mean 20-180).
        # Extremely bright images are likely X-rays stored inverted, blank pages, or photos.
        if mean_brightness > 210:
            return (
                False,
                "The inputted image cannot be scanned. "
                "The image appears too bright to be a valid fetal ultrasound. "
                "Please upload a genuine grayscale ultrasound image."
            )

        # ── CHECK 3: Texture / noise check ─────────────────────────────────
        # Ultrasound images have characteristic speckle noise — measured by local std deviation.
        # A completely uniform (blank) image has near-zero std.
        local_std = float(np.std(gray))
        if local_std < 8.0:
            return (
                False,
                "The inputted image cannot be scanned. "
                "The image appears to be blank or has insufficient texture. "
                "Please upload a valid fetal ultrasound image with visible scan data."
            )

        # ── CHECK 4: Aspect ratio sanity ───────────────────────────────────
        aspect = max(h, w) / min(h, w)
        if aspect > 5.0:
            return (
                False,
                "The inputted image cannot be scanned. "
                "Image dimensions appear invalid for an ultrasound scan."
            )

        return True, "Valid ultrasound image"

    except Exception as e:
        print(f"  [WARN] Validation error: {e}")
        # On unexpected error, allow through (do not block valid scans due to code issues)
        return True, "Validation skipped due to error"


def enhance_ultrasound_image(filepath, output_path):
    """
    Enhance a blurry/low-contrast ultrasound image using:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Unsharp masking (sharpening)
    - Denoising

    Returns:
        str: path to the enhanced image (or original if enhancement fails)
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(filepath))
        if img is None:
            return filepath

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: CLAHE — adaptive contrast enhancement (clipLimit=3 for stronger effect)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        # Step 2: Bilateral filter — removes noise while preserving edges
        denoised = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 3: Unsharp masking — sharpens structural details
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(denoised, 1.6, gaussian, -0.6, 0)

        # Step 4: Final CLAHE pass — boost fine details
        clahe2 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        final = clahe2.apply(sharpened)

        cv2.imwrite(str(output_path), final)
        print(f"  [OK] Enhanced image saved: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"  [WARN] Enhancement failed: {e}")
        return filepath


def upload_pdf_to_cloud(filepath):
    """
    Uploads a file to catbox.moe for temporary public hosting (free API, no auth).
    Returns the public URL on success, or None on failure.
    """
    try:
        import requests
        print(f"  [CLOUD] Uploading PDF to cloud storage...")
        url = "https://catbox.moe/user/api.php"
        data = {"reqtype": "fileupload"}
        with open(filepath, 'rb') as f:
            files = {"fileToUpload": f}
            response = requests.post(url, data=data, files=files, timeout=30)
        
        if response.status_code == 200:
            public_url = response.text.strip()
            print(f"  [OK] PDF uploaded: {public_url}")
            return public_url
        else:
            print(f"  [WARN] Cloud upload failed with status {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"  [WARN] Cloud upload exception: {e}")
        return None



@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with real-time server info."""
    uptime_seconds = (datetime.now() - SERVER_START_TIME).total_seconds()
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    seconds = int(uptime_seconds % 60)

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'server_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'uptime': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        'version': '2.0',
        'components': {
            'cnn_model': 'ResNet50',
            'rag_database': 'FAISS + Sentence Transformers',
            'report_generator': 'Template-based LLM',
            'allowed_formats': list(app.config['ALLOWED_EXTENSIONS'])
        }
    })


@app.route('/uploads/<filename>', methods=['GET'])
def serve_uploads(filename):
    """Serve files from the uploads folder."""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Security check
        if not os.path.abspath(filepath).startswith(os.path.abspath(app.config['UPLOAD_FOLDER'])):
            return jsonify({'error': 'Invalid file path'}), 403

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        if filename.endswith('.pdf'):
            return send_file(filepath, mimetype='application/pdf')
        elif filename.endswith('.png'):
            return send_file(filepath, mimetype='image/png')
        else:
            return send_file(filepath, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """
    Main diagnosis endpoint.
    Accepts ultrasound image (PNG/JPEG/DICOM), patient name, and phone number.
    Returns complete diagnostic report with nutrients and sends SMS.
    """
    try:
        # ── VALIDATE IMAGE ──────────────────────────────────────────────────
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': (
                    f'Invalid file format: "{file.filename}". '
                    'Only JPEG, PNG, and DICOM (.dcm) images are accepted. '
                    'Please upload a valid medical ultrasound image.'
                )
            }), 400

        # ── VALIDATE PATIENT INFO ────────────────────────────────────────────
        patient_name = request.form.get('patient_name', '').strip()
        patient_phone = request.form.get('patient_phone', '').strip()

        if not patient_name:
            return jsonify({'error': 'Patient name is required'}), 400

        if not patient_phone:
            return jsonify({'error': 'Patient phone number is required'}), 400

        # Basic phone validation (10 digits for India, or E.164 format)
        import re
        phone_clean = re.sub(r'[\s\-\(\)]', '', patient_phone)
        if not re.match(r'^(\+?\d{10,15})$', phone_clean):
            return jsonify({'error': 'Invalid phone number. Please enter a valid 10-digit phone number.'}), 400

        # ── SAVE UPLOADED FILE ────────────────────────────────────────────────
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = re.sub(r'[^\w\.-]', '_', file.filename)
        filename = f"ultrasound_{timestamp}_{safe_name}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        file_type = get_file_type_label(file.filename)
        print(f"\n[DIAGNOSIS] Processing {file_type}: {filename}")
        print(f"  Patient: {patient_name} | Phone: {patient_phone}")

        # Handle DICOM conversion
        processing_path = handle_dicom_file(filepath)

        # ── VALIDATE ULTRASOUND IMAGE ─────────────────────────────────────────
        print("  [0/4] Validating ultrasound image...")
        is_valid_us, validation_msg = validate_ultrasound_image(processing_path)
        if not is_valid_us:
            # Delete the uploaded file to avoid storing invalid images
            try:
                os.remove(filepath)
                if processing_path != filepath:
                    os.remove(processing_path)
            except Exception:
                pass
            return jsonify({
                'error': validation_msg,
                'error_type': 'invalid_ultrasound'
            }), 422
        print(f"  [OK] Image validation passed")

        # ── ENHANCE ULTRASOUND IMAGE ──────────────────────────────────────────
        print("  [0b/4] Enhancing image quality...")
        enhanced_filename = f"enhanced_{timestamp}.png"
        enhanced_filepath = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
        enhance_ultrasound_image(processing_path, enhanced_filepath)
        enhanced_image_path = f"/uploads/{enhanced_filename}" if os.path.exists(enhanced_filepath) else None

        # ── STEP 1: CNN INFERENCE ─────────────────────────────────────────────
        print("  [1/4] Running CNN inference...")
        cnn_result = inference_pipeline.run_inference(processing_path)

        if cnn_result['status'] != 'success':
            return jsonify({
                'error': f"CNN inference failed: {cnn_result['message']}"
            }), 500

        cnn_prediction = cnn_result['cnn_prediction']
        print(f"  [OK] Prediction: {cnn_prediction['class_label']}")
        print(f"  [OK] Confidence: {cnn_prediction['confidence_percentage']:.1f}%")

        # ── STEP 2: RAG RETRIEVAL ─────────────────────────────────────────────
        print("  [2/4] Retrieving medical context (RAG)...")
        rag_context = rag_retriever.retrieve_combined_context(
            predicted_label=cnn_prediction['class_label'],
            confidence=cnn_prediction['confidence_percentage'],
            image_embedding=cnn_result['cnn_prediction']['confidence_percentage'],
            top_k=3
        )
        print("  [OK] Medical context retrieved")

        # ── STEP 3: REPORT GENERATION ─────────────────────────────────────────
        print("  [3/4] Generating diagnostic report...")
        report = report_generator.generate_report(
            cnn_prediction,
            rag_context,
            patient_name=patient_name,
            patient_phone=patient_phone
        )
        report_text = report_generator.format_report_for_display(report)
        print("  [OK] Report generated")

        # ── STEP 4: VISUALIZATION (GRAD-CAM) ─────────────────────────────────
        print("  [4/4] Generating attention visualization...")
        visualizations_available = False
        gradcam_path = None

        try:
            grad_cam = GradCAM(inference_pipeline.model.model, device='cpu')
            viz_output_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                f"gradcam_{timestamp}.png"
            )
            try:
                grad_cam.visualize(
                    processing_path,
                    output_path=viz_output_path,
                    heatmap_intensity=0.4
                )
                if os.path.exists(viz_output_path):
                    print("  [OK] Attention map generated successfully")
                    visualizations_available = True
                    gradcam_path = f"/uploads/gradcam_{timestamp}.png"
                else:
                    print("  [WARN] Grad-CAM file not created")
                    visualizations_available = False

            except Exception as grad_error:
                print(f"  [WARN] Grad-CAM generation failed: {str(grad_error)}")
        except Exception as viz_error:
            print(f"  [WARN] Visualization initialization failed: {str(viz_error)}")
            traceback.print_exc()

        # ── PDF REPORT GENERATION ─────────────────────────────────────────────
        print("  [PDF] Generating professional reports...")
        pdf_doctor_path = None
        pdf_patient_path = None

        try:
            pdf_diagnosis_data = {
                'report_id': report.get('report_id', f"REPORT-{timestamp}"),
                'timestamp': datetime.now().isoformat(),
                'patient_name': patient_name,
                'patient_phone': patient_phone,
                'prediction': {
                    'primary_diagnosis': cnn_prediction['class_label'],
                    'confidence_percentage': cnn_prediction['confidence_percentage'],
                    'all_probabilities': cnn_prediction['all_probabilities']
                },
                'condition_description': cnn_prediction['condition_description'],
                'report_text': report_text,
                'nutrients': report.get('section_10_nutrients', {}).get('content', '')
            }

            # Generate doctor report
            doctor_pdf_filename = f"report_doctor_{timestamp}.pdf"
            doctor_pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], doctor_pdf_filename)

            if pdf_generator.generate_doctor_report(
                pdf_diagnosis_data,
                processing_path,
                os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{timestamp}.png") if visualizations_available else None,
                doctor_pdf_filepath
            ):
                pdf_doctor_path = f"/uploads/{doctor_pdf_filename}"
                print("  [OK] Doctor report generated")

            # Generate patient report
            patient_pdf_filename = f"report_patient_{timestamp}.pdf"
            patient_pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], patient_pdf_filename)

            if pdf_generator.generate_patient_report(
                pdf_diagnosis_data,
                processing_path,
                os.path.join(app.config['UPLOAD_FOLDER'], f"gradcam_{timestamp}.png") if visualizations_available else None,
                patient_pdf_filepath
            ):
                pdf_patient_path = f"/uploads/{patient_pdf_filename}"
                print("  [OK] Patient report generated")

        except Exception as pdf_error:
            print(f"  [WARN] PDF generation failed: {str(pdf_error)}")
            traceback.print_exc()

        # ── BUILD RESPONSE ────────────────────────────────────────────────────
        response = {
            'status': 'success',
            'diagnosis': {
                'timestamp': datetime.now().isoformat(),
                'patient_name': patient_name,
                'patient_phone': patient_phone,
                'image_file': filename,
                'image_path': f"/uploads/{filename}",
                'file_type': file_type,
                'prediction': {
                    'primary_diagnosis': cnn_prediction['class_label'],
                    'confidence_percentage': cnn_prediction['confidence_percentage'],
                    'all_probabilities': cnn_prediction['all_probabilities'],
                    'condition_description': cnn_prediction['condition_description']
                },
                'retrieved_context': {
                    'related_conditions': rag_context['retrieved_conditions']['related_fetal_conditions'],
                    'maternal_risk_factors': rag_context.get('maternal_risk_factors', []),
                    'risk_stratification': rag_context.get('applicable_risk_stratification', {})
                },
                'report': {
                    'full_text': report_text,
                    'structured': report,
                    'report_id': report.get('report_id'),
                    'sections': {
                        'findings': report.get('section_3_findings', {}).get('content'),
                        'condition_analysis': report.get('section_4_fetal_condition_analysis', {}).get('content'),
                        'risk_assessment': report.get('section_6_risk_assessment', {}).get('content'),
                        'recommendations': report.get('section_8_recommendations', {}).get('content'),
                        'nutrients': report.get('section_10_nutrients', {}).get('content')
                    }
                },
                'visualizations': {
                    'available': visualizations_available,
                    'gradcam_path': gradcam_path if visualizations_available else None,
                    'enhanced_image_path': enhanced_image_path
                },
                'pdf_reports': {
                    'doctor_report': pdf_doctor_path,
                    'patient_report': pdf_patient_path
                },
                'sms_sent': False
            },
            'metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'model_version': '2.0',
                'cnn_model': 'ResNet50',
                'rag_system': 'FAISS + Sentence Transformers',
                'llm_type': 'Template-based Generator'
            }
        }

        # ── SAVE TO DATABASE ──────────────────────────────────────────────────
        print("  [DB] Saving diagnosis record to database...")
        db.save_diagnosis(response['diagnosis'])

        print("  [OK] Diagnosis complete\n")
        return jsonify(response)

    except Exception as e:
        print(f"\n[ERROR] Diagnosis failed: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': f'Diagnosis processing failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Fetch recent diagnostic records from the database."""
    try:
        limit = int(request.args.get('limit', 10))
        records = db.get_recent_diagnoses(limit=limit)
        return jsonify({
            'status': 'success',
            'count': len(records),
            'records': records
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-diagnose', methods=['POST'])
def batch_diagnose():
    """Batch processing endpoint for multiple images."""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        files = request.files.getlist('images')
        results = []

        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_{timestamp}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processing_path = handle_dicom_file(filepath)
            cnn_result = inference_pipeline.run_inference(processing_path)

            if cnn_result['status'] == 'success':
                results.append({
                    'filename': filename,
                    'prediction': cnn_result['cnn_prediction']
                })

        return jsonify({
            'status': 'success',
            'processed_images': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models."""
    try:
        info = inference_pipeline.model.get_model_info()
        return jsonify({
            'status': 'success',
            'model': info,
            'rag_knowledge_base': {
                'fetal_conditions': len(rag_retriever.knowledge_base.get('fetal_conditions', [])),
                'maternal_conditions': len(rag_retriever.knowledge_base.get('maternal_conditions', [])),
                'doppler_parameters': len(rag_retriever.knowledge_base.get('doppler_parameters', [])),
                'biometric_measurements': len(rag_retriever.knowledge_base.get('biometric_measurements', []))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum size: 50MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print(" DEEP LEARNING BASED ULTRASOUND IMAGE ANALYSIS")
    print(" FOR EARLY DETECTION OF FETAL GROWTH ABNORMALITIES")
    print("=" * 80)
    print("\nStarting Flask application...")
    print("Open browser at: http://localhost:5000")
    print("Allowed image formats: JPEG, PNG, DICOM")
    print("=" * 80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
