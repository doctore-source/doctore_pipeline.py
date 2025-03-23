scp -i "YOUR_KEY.pem" *.py predictions.db requirements.txt ubuntu@YOUR_SERVER_IP:/home/ubuntu/
python3 generate_reports.py
