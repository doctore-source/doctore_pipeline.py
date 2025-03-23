scp -i "YOUR_KEY.pem" *.py predictions.db requirements.txt ubuntu@YOUR_SERVER_IP:/home/ubuntu/
def full_prediction_pipeline():
    """
    Combines data fetching, prediction generation, and saving results to a file/database.
    """
    # Fetch live NBA game data
    nba_data = fetch_real_nba_data()
    
    # Fetch live betting odds
    betting_odds = fetch_real_betting_odds()
    
    if not nba_data.empty and not betting_odds.empty:
        # Merge dataframes
        merged_df = nba_data.merge(betting_odds, on=['Team_A', 'Team_B'])
        
        # Generate predictions
        predictions = doctore_nba_odds_calculator(merged_df)
        
        # Save predictions to CSV and Database
        save_to_csv(predictions)
        save_to_database(predictions)
        
        print("Predictions saved successfully.")
    else:
        print("Failed to fetch data. Skipping prediction.")
      