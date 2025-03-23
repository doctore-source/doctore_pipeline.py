sudo systemctl status doctore_api.service
@app.route('/health', methods=['GET'])
def health():
    status = {
        "status": "API is running",
        "model_loaded": model is not None
    }
    return jsonify(status), 200
    pip install torch
pip install torch-geometric
pip install flask
pip install streamlit
