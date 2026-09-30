import sqlite3
import os

db_path = 'data/alerts.db'
print('DB exists:', os.path.exists(db_path))
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Check journal mode
cur.execute('PRAGMA journal_mode;')
print('Journal mode:', cur.fetchone()[0])

# Check tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in cur.fetchall()]
print('Tables:', tables)

# Check indexes
cur.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index';")
indexes = cur.fetchall()
print('Indexes count:', len(indexes))
for idx in indexes:
    print('  Index:', idx)

# Check count in tables
for t in ['alerts', 'alert_events', 'alert_subscriptions', 'in_app_notifications']:
    if t in tables:
        cur.execute(f"SELECT count(*) FROM {t};")
        print(f"Count in {t}:", cur.fetchone()[0])
conn.close()
