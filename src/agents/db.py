import csv
import sqlite3
conn = sqlite3.connect('artifacts/langgraph_listings.db')
cur = conn.cursor()
print(cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print(cur.execute('SELECT id, created_at, title, predicted_price_tnd FROM listings ORDER BY created_at DESC LIMIT 5').fetchall())

with open('artifacts/langgraph_listings.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'created_at', 'title', 'predicted_price_tnd'])
    for row in cur.execute('SELECT id, created_at, title, predicted_price_tnd FROM listings ORDER BY created_at DESC'):
        writer.writerow(row)
        
conn.close()