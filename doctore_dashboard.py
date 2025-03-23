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
        )
        