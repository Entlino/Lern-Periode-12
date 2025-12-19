import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/garmin.db")

def init_db():
    """Erstellt Tabellen für Stats UND Settings."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Die bekannte Stats-Tabelle
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            steps INTEGER,
            calories INTEGER,
            rest_hr REAL,
            sleep_minutes INTEGER,
            source TEXT
        )
    ''')

    # 2. NEU: Die Settings-Tabelle (Key-Value Speicher)
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    # Standard-Werte setzen (falls noch nicht da)
    defaults = [
        ("step_goal", "10000"),
        ("ai_persona", "coach"), # coach, drill, yogi, nerd
    ]
    for k, v in defaults:
        c.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (k, v))

    conn.commit()
    conn.close()

# --- Neue Funktionen für Settings ---

def get_setting(key, default=None):
    conn = sqlite3.connect(DB_PATH)
    res = conn.execute('SELECT value FROM settings WHERE key = ?', (key,)).fetchone()
    conn.close()
    return res[0] if res else default

def update_setting(key, value):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(value)))
    conn.commit()
    conn.close()

def save_dataframe(df):
    """Speichert den Pandas DataFrame in die SQLite DB (Upsert)."""
    conn = sqlite3.connect(DB_PATH)
    # Pandas to_sql ist super, aber 'replace' löscht die Tabelle. 
    # Für Updates nutzen wir hier einen einfachen Loop oder 'append' mit Logik.
    # Für den Anfang (einfach):
    for _, row in df.iterrows():
        sql = '''
            INSERT OR REPLACE INTO daily_stats (date, steps, calories, rest_hr, sleep_minutes, source)
            VALUES (?, ?, ?, ?, ?, 'csv_import')
        '''
        conn.execute(sql, (
            str(row['date']), 
            row.get('steps'), 
            row.get('calories'), 
            row.get('rest_hr'), 
            row.get('sleep_minutes')
        ))
    conn.commit()
    conn.close()

def get_data(start_date=None, end_date=None):
    """Lädt Daten als DataFrame zurück."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM daily_stats ORDER BY date ASC"
    # Hier könnten später WHERE klauseln hinzukommen
    df = pd.read_sql(query, conn)
    conn.close()
    return df