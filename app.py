import os
import cv2
import numpy as np
import pandas as pd
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import logging
from parking_detector import ParkingDetector
from parking_analyzer import ParkingAnalyzer
from utils import create_temp_directory, cleanup_temp_files

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "parking-detection-secret")

# Create temporary directory for uploads
UPLOAD_FOLDER = create_temp_directory()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Clear any previous session data
    if 'result_data' in session:
        del session['result_data']
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image with our parking detector
            detector = ParkingDetector()
            analyzer = ParkingAnalyzer()
            
            # Detect slots and analyze parking spaces
            slots_data = detector.process_image(filepath)
            result = analyzer.analyze_parking_data(slots_data, filepath)
            
            # Generate the CSV output
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'parking_status.csv')
            df = pd.DataFrame({
                'Total Slots': [result['total_slots']],
                'Occupied Slots': [result['occupied_slots']],
                'Available Slots': [result['available_slots']]
            })
            df.to_csv(csv_path, index=False)
            
            # Convert result image to base64 for display
            _, buffer = cv2.imencode('.png', result['annotated_image'])
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # Store results in session
            session['result_data'] = {
                'total_slots': result['total_slots'],
                'occupied_slots': result['occupied_slots'],
                'available_slots': result['available_slots'],
                'image_data': img_str,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return redirect(url_for('results'))
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload an image (PNG, JPG, JPEG).', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'result_data' not in session:
        flash('No analysis results found. Please upload an image first.', 'warning')
        return redirect(url_for('index'))
    
    return render_template('results.html', result=session['result_data'])

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image
            detector = ParkingDetector()
            analyzer = ParkingAnalyzer()
            
            slots_data = detector.process_image(filepath)
            result = analyzer.analyze_parking_data(slots_data, filepath)
            
            # Generate CSV
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'parking_status.csv')
            df = pd.DataFrame({
                'Total Slots': [result['total_slots']],
                'Occupied Slots': [result['occupied_slots']],
                'Available Slots': [result['available_slots']]
            })
            df.to_csv(csv_path, index=False)
            
            return jsonify({
                'total_slots': result['total_slots'],
                'occupied_slots': result['occupied_slots'],
                'available_slots': result['available_slots'],
                'success': True
            })
            
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

# Clean up temporary files when the app exits
import atexit
atexit.register(lambda: cleanup_temp_files(app.config['UPLOAD_FOLDER']))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
