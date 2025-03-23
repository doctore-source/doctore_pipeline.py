from flask import Flask, request, jsonify
import joblib
import torch
import numpy as np

app = Flask(__name__)

# Load your trained model (adjust the path as needed)
model = joblib.load('path_to_your_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Assuming you have a model that takes JSON input
        input_data = np.array(data['features'])
        
        # Make prediction (adjust based on your model’s requirements)
        prediction = model.predict(input_data.reshape(1, -1))
        
        response = {
            'prediction': prediction.tolist()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)