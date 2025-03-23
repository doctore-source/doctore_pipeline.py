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
    query = "SELECT * FROM predictions"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# ------------------ Streamlit Dashboard Interface ------------------

st.set_page_config(page_title="Doctore NBA Odds Calculator Dashboard", layout="wide")

# Title
st.title("📊 Doctore NBA Odds Calculator Dashboard")

# Load data from the database
df = load_data()

if not df.empty:
    # Show the raw data
    st.subheader("🔍 Predictions Data")
    st.write(df)
    
    # Display Value Bets Only
    st.subheader("💡 Value Bets")
    value_bets = df[(df['Value_Bet_A'] == True) | (df['Value_Bet_B'] == True)]
    st.write(value_bets)
    
    # Plotting Predictions vs. Bookmaker Odds
    st.subheader("📈 Prediction Analysis")
    
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='Win_Probability',
        y='Implied_Prob_Team_A',
        color='Predicted_Winner:N',
        tooltip=['Team_A', 'Team_B', 'Win_Probability', 'Implied_Prob_Team_A']
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    # Performance Overview
    st.subheader("📊 Performance Overview")
    
    # Plot: Value Bet Distribution
    value_bet_chart = alt.Chart(value_bets).mark_bar().encode(
        x='Predicted_Winner:N',
        y='count()',
        color='Predicted_Winner:N'
    ).interactive()

    st.altair_chart(value_bet_chart, use_container_width=True)

else:
    st.write("No predictions available. Please run the prediction pipeline first.")

# ------------------ Refresh Button ------------------

if st.button("🔄 Refresh Data"):
    df = load_data()
    st.experimental_rerun()
    