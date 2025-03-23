scp -i "YOUR_KEY.pem" *.py predictions.db requirements.txt ubuntu@YOUR_SERVER_IP:/home/ubuntu/
import pandas as pd
import numpy as np
import requests
import schedule
import time
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# ------------------ STEP 1: Fetching Data from APIs ------------------

def fetch_real_nba_data():
    """
    Fetches live NBA game data from the SportsDataIO API.
    Returns a DataFrame with processed data.
    """
    url = "https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/2025-MAR-23"
    headers = {'Ocp-Apim-Subscription-Key': 'YOUR_SPORTSDATAIO_API_KEY'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        
        nba_data = []
        for game in data:
            nba_data.append({
                "Team_A": game['HomeTeam'],
                "Team_B": game['AwayTeam'],
                "Team_A_Points": game['HomeTeamScore'],
                "Team_B_Points": game['AwayTeamScore']
            })
        
        return pd.DataFrame(nba_data)
    else:
        print(f"Failed to fetch NBA data. Status Code: {response.status_code}")
        return pd.DataFrame()


def fetch_real_betting_odds():
    """
    Fetches live betting odds from TheOddsAPI.
    Returns a DataFrame with processed data.
    """
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        'apiKey': 'YOUR_ODDS_API_KEY',
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'decimal'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        
        odds_data = []
        for event in data:
            odds_data.append({
                "Team_A": event['home_team'],
                "Team_B": event['away_team'],
                "Bookmaker_Odds_Team_A": event['bookmakers'][0]['markets'][0]['outcomes'][0]['price'],
                "Bookmaker_Odds_Team_B": event['bookmakers'][0]['markets'][0]['outcomes'][1]['price']
            })
        
        return pd.DataFrame(odds_data)
    else:
        print(f"Failed to fetch odds data. Status Code: {response.status_code}")
        return pd.DataFrame()

# ------------------ STEP 2: Prediction Pipeline ------------------

def doctore_nba_odds_calculator(merged_df):
    """
    Generates predictions using your 'Doctore' model and identifies value bets.
    """
    # Assuming you already have a trained model (Random Forest)
    # Loading your trained model and scaler
    scaler = StandardScaler()
    pca = PCA(n_components=8)
    rf_model_embedded = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Preprocess and scale data
    features = merged_df.iloc[:, 2:].values
    features = scaler.fit_transform(features)
    features_embedded = pca.fit_transform(features)
    
    # Generate predictions
    predictions = rf_model_embedded.predict(features_embedded)
    prediction_probs = rf_model_embedded.predict_proba(features_embedded)[:, 1]
    
    # Add predictions to DataFrame
    merged_df['Predicted_Winner'] = predictions
    merged_df['Win_Probability'] = prediction_probs
    merged_df['Implied_Prob_Team_A'] = 1 / merged_df['Bookmaker_Odds_Team_A']
    merged_df['Implied_Prob_Team_B'] = 1 / merged_df['Bookmaker_Odds_Team_B']
    merged_df['Value_Bet_A'] = merged_df['Win_Probability'] > merged_df['Implied_Prob_Team_A']
    merged_df['Value_Bet_B'] = (1 - merged_df['Win_Probability']) > merged_df['Implied_Prob_Team_B']
    
    return merged_df

# ------------------ STEP 3: Saving Results ------------------

def save_to_database(predictions_df, db_name="predictions.db"):
    """
    Saves predictions to an SQLite database.
    """
    conn = sqlite3.connect(db_name)
    predictions_df.to_sql("predictions", conn, if_exists="append", index=False)
    conn.close()

# ------------------ STEP 4: Full Pipeline Execution ------------------

def full_prediction_pipeline():
    """
    Runs the full prediction pipeline and saves results to a database.
    """
    nba_data = fetch_real_nba_data()
    betting_odds = fetch_real_betting_odds()
    
    if not nba_data.empty and not betting_odds.empty:
        merged_df = nba_data.merge(betting_odds, on=['Team_A', 'Team_B'])
        predictions = doctore_nba_odds_calculator(merged_df)
        
        save_to_database(predictions)
        print("Predictions saved successfully.")
    else:
        print("Failed to fetch data.")

# ------------------ STEP 5: Automating the Process ------------------

def run_scheduled_pipeline():
    """
    Runs the prediction pipeline at regular intervals.
    """
    schedule.every().hour.do(full_prediction_pipeline)

    while True:
        schedule.run_pending()
        time.sleep(60)
        def save_actual_results(game_results, db_name="predictions.db"):
    """
    Save actual game results to the database for comparison with predictions.
    """
    conn = sqlite3.connect(db_name)
    game_results.to_sql("actual_results", conn, if_exists="append", index=False)
    conn.close()
    