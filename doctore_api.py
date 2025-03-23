gunicorn --workers 4 --bind 0.0.0.0:5000 doctore_api:app
from flask import Flask, request, jsonify
import torch
import joblib
import pandas as pd
from hybrid_model import HybridModel

app = Flask(__name__)

# Load Hybrid Model
hybrid_model = HybridModel()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        # Make predictions using the Hybrid Model
        predictions = hybrid_model.predict(df)
        
        response = {
            "Hybrid_Predictions": predictions.tolist()
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    