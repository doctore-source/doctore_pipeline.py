import torch
import joblib
import numpy as np
from gnn_model import GNNModel
from lstm_model import LSTMModel
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class HybridModel:
    def __init__(self):
        # Load Models
        self.gnn_model = GNNModel(num_node_features=16, hidden_channels=32)
        self.gnn_model.load_state_dict(torch.load('gnn_model.pth'))
        self.gnn_model.eval()

        self.lstm_model = LSTMModel(input_size=16, hidden_size=128, num_layers=2, output_size=1)
        self.lstm_model.load_state_dict(torch.load('lstm_model.pth'))
        self.lstm_model.eval()
        
        self.ensemble_model = joblib.load('enhanced_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def predict(self, features):
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.tensor(features_scaled).float()

        # GNN Prediction
        gnn_output = self.gnn_model(features_tensor).detach().numpy().flatten()
        
        # LSTM Prediction
        lstm_output = self.lstm_model(features_tensor.unsqueeze(1)).detach().numpy().flatten()
        
        # Ensemble Model Prediction
        ensemble_output = self.ensemble_model.predict_proba(features_scaled)[:, 1]
        
        # Combine all predictions (Weighted Average)
        hybrid_prediction = (0.4 * gnn_output) + (0.3 * lstm_output) + (0.3 * ensemble_output)
        
        return hybrid_prediction