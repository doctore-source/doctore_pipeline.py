scp -i "YOUR_KEY.pem" *.py predictions.db requirements.txt ubuntu@YOUR_SERVER_IP:/home/ubuntu/
 streamlit as st
import pandas as pd

st.title("Doctore: NBA Betting Analysis")

# Example: show a sample DataFrame
df = pd.DataFrame({
    "Teams": ["Team A", "Team B", "Team C"],
    "Odds": [1.8, 2.1, 1.95]
})
st.write("Here are some sample odds data:")
st.dataframe(df)
 streamlit as st
import pandas as pd

st.title("Doctore: NBA Betting Analysis")

# Example: show

import pandas as pd

st.title("Doctore:
 streamlit as st

st.title("Welcome to Doctore!")
st.write("This is where your custom content, data, and visuals will go.")
