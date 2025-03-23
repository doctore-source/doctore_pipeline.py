from flask import Flask, request, jsonify
import torch
import joblib
import pandas as pd
from gnn_model import GNNModel
from lstm_model import LSTMModel

app = Flask(__name__)

# Load models and scaler
gnn_model = GNNModel(num_node_features=16, hidden_channels=32)
gnn_model.load_state_dict(torch.load('gnn_model.pth'))
gnn_model.eval()

lstm_model = LSTMModel(input_size=16, hidden_size=128, num_layers=2, output_size=1)
lstm_model.load_state_dict(torch.load('lstm_model.pth'))
lstm_model.eval()

scaler = joblib.load('scaler.pkl')

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)
        
        # Scale data
        features = scaler.transform(df)
        features_tensor = torch.tensor(features).float()
        
        # GNN Prediction
        gnn_output = gnn_model(features_tensor).detach().numpy()
        
        # LSTM Prediction
        lstm_output = lstm_model(features_tensor.unsqueeze(1)).detach().numpy()
        
        response = {
            "GNN_Prediction": gnn_output.tolist(),
            "LSTM_Prediction": lstm_output.tolist()
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)