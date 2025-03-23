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