from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("doctore_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the model - use absolute path for systemd service
try:
    # Get the absolute directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'doctore_model.pkl')
    logger.info(f"Attempting to load model from: {model_path}")
    
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Process input data
        input_df = pd.DataFrame(data['features'])
        
        if model is None:
            logger.error("Model not loaded, cannot make prediction")
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
        
        # Make prediction
        predictions = model.predict(input_df)
        
        logger.info(f"Prediction successful: {predictions[:5]}")
        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist()
        })
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    logger.info(f"Health check called. Model status: {model_status}")
    return jsonify({
        'status': 'healthy', 
        'model_status': model_status
    })

if __name__ == '__main__':
    logger.info("Starting Doctore API in standalone mode")
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    logger.info("Doctore API imported as module (for Gunicorn)")
    