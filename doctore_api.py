http://127.0.0.1:5000/predict.
from flask import Flask, request, jsonify
import joblib
import os
import torch
import numpy as np

app = Flask(__name__)

# Load your trained model (adjust the path as needed)
MODEL_PATH = 'models/doctore_model.pkl'

def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            return model
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is running", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            raise Exception("Model not loaded. Please check your model path.")

        data = request.json
        input_data = np.array(data['features'])

        # Make prediction
        prediction = model.predict(input_data.reshape(1, -1))

        response = {
            'prediction': prediction.tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)