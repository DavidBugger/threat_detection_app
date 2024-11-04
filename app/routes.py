from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
from datetime import datetime
from app.models.threat_detector import ThreatDetector

main = Blueprint('main', __name__)
detector = ThreatDetector()

# Load the pre-trained model
model_path = 'path_to_your_saved_model.joblib'
if os.path.exists(model_path):
    detector.load_model(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/results')
def results():
    return render_template('results.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Preprocess the data
        X, processed_df = detector.preprocess_data(df)
        
        # Make predictions
        predictions = detector.predict(X)
        
        # Prepare results
        results = processed_df[['Target Username', 'DateTime', 'IP']].copy()
        results['Prediction'] = predictions
        results['DateTime'] = results['DateTime'].astype(str)
        
        return jsonify({
            'success': True,
            'results': results.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/realtime', methods=['POST'])
def realtime_analysis():
    try:
        data = request.get_json()
        
        # Create DataFrame with required columns
        current_time = datetime.now()
        df = pd.DataFrame([{
            'Target Username': data['Target Username'],
            'IP': data['IP'],
            'Event ID': data['Event ID'],
            'Date': current_time.date(),
            'Time': current_time.time(),
            'Authentication Package': 'Negotiate',  # Default value
            'Logon Type': 3,  # Default value
        }])
        
        # Preprocess the data
        X, processed_df = detector.preprocess_data(df)
        
        # Make prediction
        prediction = detector.predict(X)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500