"""
Flask Web Application for URL Phishing Detection
Using Deep Learning Neural Network Model
"""
import os
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
from urllib.parse import urlparse

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# ============ LOAD MODELS ============
print("🧠 Loading Deep Learning Model...")

try:
    with open('dl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded: dl_model.pkl")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    with open('feature_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Vectorizer loaded: feature_vectorizer.pkl")
except Exception as e:
    print(f"❌ Error loading vectorizer: {e}")
    vectorizer = None

try:
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded: feature_scaler.pkl")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

try:
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print("✓ Metadata loaded: model_metadata.pkl")
except Exception as e:
    print(f"❌ Error loading metadata: {e}")
    metadata = {}

# ============ FEATURE EXTRACTION FUNCTIONS ============
def extract_url_features(url):
    """Extract features from URL using TF-IDF vectorizer"""
    try:
        features = vectorizer.transform([url]).toarray()
        return features
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

def analyze_url(url):
    """
    Analyze URL for phishing indicators
    Returns: {'prediction': 0/1, 'confidence': float, 'is_phishing': bool}
    """
    try:
        if not model or not vectorizer or not scaler:
            return {'error': 'Model not loaded'}
        
        # Validate URL
        if not url or not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Extract features
        features = extract_url_features(url)
        if features is None:
            return {'error': 'Feature extraction failed'}
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0]
        
        phishing_prob = confidence[1]
        legitimate_prob = confidence[0]
        
        return {
            'prediction': int(prediction),
            'phishing_probability': float(phishing_prob),
            'legitimate_probability': float(legitimate_prob),
            'is_phishing': bool(prediction == 1),
            'confidence': max(phishing_prob, legitimate_prob)
        }
    
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return {'error': str(e)}

# ============ FLASK ROUTES ============
@app.route('/')
def index():
    """Home page"""
    return render_template('dashboard.html', model_info=metadata)

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    """Scan URL for phishing"""
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        
        if not url:
            return render_template('dashboard.html', 
                                   error='Please enter a URL',
                                   model_info=metadata)
        
        # Analyze URL
        result = analyze_url(url)
        
        # Prepare display result
        if 'error' in result:
            return render_template('dashboard.html',
                                   error=result['error'],
                                   model_info=metadata)
        
        display_result = {
            'url': url,
            'is_phishing': result['is_phishing'],
            'confidence': result['confidence'],
            'phishing_prob': result['phishing_probability'],
            'legitimate_prob': result['legitimate_probability'],
            'status': '🚨 PHISHING DETECTED' if result['is_phishing'] else '✅ SAFE',
            'model_accuracy': metadata.get('accuracy', 0),
            'model_type': metadata.get('model_type', 'Neural Network')
        }
        
        return render_template('result.html', result=display_result, model_info=metadata)
    
    return render_template('dashboard.html', model_info=metadata)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for URL prediction"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL required'}), 400
        
        result = analyze_url(url)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': metadata.get('model_type', 'Unknown'),
        'model_accuracy': metadata.get('accuracy', 'Unknown')
    })

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def page_not_found(e):
    return render_template('dashboard.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('dashboard.html', error='Internal server error'), 500

# ============ MAIN ============
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 STARTING FLASK APP - PHISHING DETECTION WITH DEEP LEARNING")
    print("=" * 80)
    
    if model and vectorizer and scaler:
        print("✅ All models loaded successfully!")
        print(f"📊 Model Accuracy: {metadata.get('accuracy', 'N/A'):.2%}")
        print(f"🏗️  Model Type: {metadata.get('model_type', 'N/A')}")
        print("\n🌐 Web Interface: http://127.0.0.1:5000/")
        print("📡 API Endpoint: http://127.0.0.1:5000/api/predict")
        print("💚 Health Check: http://127.0.0.1:5000/health")
        print("=" * 80 + "\n")
        
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("❌ Failed to load models. Please run train_dl_simple.py first.")
