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
        