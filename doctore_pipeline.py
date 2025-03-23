scp -i "YOUR_KEY.pem" *.py predictions.db requirements.txt ubuntu@YOUR_SERVER_IP:/home/ubuntu/
import schedule
import time

def run_pipeline():
    predictions = automated_prediction_pipeline()
    predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved successfully.")

# Schedule the pipeline to run every hour
schedule.every(1).hours.do(run_pipeline)

while True:
    schedule.run_pending()
    time.sleep(60)
    