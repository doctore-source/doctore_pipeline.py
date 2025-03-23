import requests
import streamlit as st

url = "http://YOUR_SERVER_IP:5000/predict"
data = {"features": [1, 2, 3, 4, 5]}
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    st.write("Prediction:", result['prediction'])
else:
    st.write("Error:", response.text)