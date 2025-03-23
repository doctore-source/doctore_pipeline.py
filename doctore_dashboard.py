import streamlit as st
from retrain_monitor import generate_performance_report

st.title("📊 Doctore NBA Model Performance Monitoring")

if st.button("Generate Performance Report"):
    generate_performance_report()
    
    with open("performance_report.pdf", "rb") as file:
        st.download_button(
            label="Download Performance Report",
            data=file,
            file_name="performance_report.pdf",
            mime="application/pdf"
        )import requests
import pandas as pd

def fetch_real_nba_data():
    """
    Fetches live NBA game data from the SportsDataIO API (Replace with your API Key).
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
    Fetches live betting odds from TheOddsAPI (Replace with your API Key).
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
        