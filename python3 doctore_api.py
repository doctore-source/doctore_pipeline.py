python3 doctore_api.py
gunicorn --workers 4 --bind 0.0.0.0:5000 doctore_api:app
import logging

# Configure Logging
logging.basicConfig(filename='doctore_api.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.before_request
def log_request():
    logging.info(f"Incoming request: {request.method} {request.url}")

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server Error: {error}")
    return "Server Error", 500
    