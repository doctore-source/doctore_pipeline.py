python3 doctore_api.py
import sqlite3

try:
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    print("Tables in the database:", tables)
except Exception as e:
    print("Error connecting to the database:", e)
    git add .
git commit -m "Fixed dependencies and updated requirements.txt"
git push
