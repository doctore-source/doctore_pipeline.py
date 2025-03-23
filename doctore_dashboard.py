import streamlit as st
import pandas as pd
import sqlite3
import altair as alt

# ------------------ Function to Load Data from Database ------------------

def load_data(db_name="predictions.db"):
    """
    Loads predictions from the SQLite database.
    """
    conn = sqlite3.connect(db_name)
    predictions = pd.read_sql("SELECT * FROM predictions", conn)
    actuals = pd.read_sql("SELECT * FROM actual_results", conn)
    conn.close()
    return predictions, actuals

# ------------------ Calculate Success Rate & Profit Calculation ------------------

def calculate_performance(predictions, actuals):
    """
    Calculates the success rate and profit/loss from predictions.
    """
    merged = predictions.merge(actuals, on=["Team_A", "Team_B"], how="inner")
    
    merged['Correct_Prediction'] = merged['Predicted_Winner'] == merged['Actual_Winner']
    total_bets = len(merged)
    correct_bets = merged['Correct_Prediction'].sum()
    
    # Calculate Success Rate
    success_rate = correct_bets / total_bets if total_bets > 0 else 0

    # Profit Calculation (Assuming 100€ per bet)
    merged['Profit/Loss'] = merged.apply(
        lambda row: 100 * (row['Bookmaker_Odds_Team_A'] - 1) if row['Correct_Prediction'] and row['Predicted_Winner'] == row['Team_A'] else
                    100 * (row['Bookmaker_Odds_Team_B'] - 1) if row['Correct_Prediction'] and row['Predicted_Winner'] == row['Team_B'] else -100,
        axis=1
    )
    
    total_profit = merged['Profit/Loss'].sum()
    return success_rate, total_profit, merged

# ------------------ Streamlit Dashboard Interface ------------------

st.set_page_config(page_title="Doctore NBA Odds Calculator Dashboard", layout="wide")

# Title
st.title("📊 Doctore NBA Odds Calculator Dashboard")

# Load data from the database
predictions, actuals = load_data()

if not predictions.empty:
    # Show the raw predictions data
    st.subheader("🔍 Predictions Data")
    st.write(predictions)
    
    # Show actual results data
    if not actuals.empty:
        st.subheader("🏀 Actual Results Data")
        st.write(actuals)
        
        # Calculate performance metrics
        success_rate, total_profit, merged_df = calculate_performance(predictions, actuals)
        
        # Display performance metrics
        st.subheader("📊 Performance Overview")
        st.write(f"✅ Success Rate: {success_rate * 100:.2f}%")
        st.write(f"💰 Total Profit/Loss: {total_profit} €")
        
        # Show Value Bets Only
        st.subheader("💡 Value Bets")
        value_bets = merged_df[(merged_df['Value_Bet_A'] == True) | (merged_df['Value_Bet_B'] == True)]
        st.write(value_bets)
        
        # Plot: Profit/Loss per Game
        st.subheader("📈 Profit/Loss Analysis")
        profit_chart = alt.Chart(merged_df).mark_bar().encode(
            x='Team_A:N',
            y='Profit/Loss',
            color='Correct_Prediction:N',
            tooltip=['Team_A', 'Team_B', 'Profit/Loss']
        ).interactive()
        
        st.altair_chart(profit_chart, use_container_width=True)
        
else:
    st.write("No predictions available. Please run the prediction pipeline first.")

# ------------------ Refresh Button ------------------

if st.button("🔄 Refresh Data"):
    st.experimental_rerun()
    