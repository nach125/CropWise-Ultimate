import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "cropwise.db"

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            N REAL, P REAL, K REAL, temperature REAL, humidity REAL, ph REAL, rainfall REAL,
            predicted_crop TEXT NOT NULL, model_name TEXT, notes TEXT
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, best_model TEXT, rows_used INTEGER,
            model_count INTEGER, metrics_json TEXT
        )""")

    with connect() as con:  # lightweight migration for older databases
        cols = [r["name"] for r in con.execute("PRAGMA table_info(predictions)")]
        if "notes" not in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN notes TEXT")

def save_prediction(values, crop, model_name, notes=None):
    with connect() as con:
        con.execute("""INSERT INTO predictions
        (created_at,N,P,K,temperature,humidity,ph,rainfall,predicted_crop,model_name,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [datetime.now().isoformat(timespec="seconds"), values["N"], values["P"], values["K"],
         values["temperature"], values["humidity"], values["ph"], values["rainfall"], crop, model_name, notes])

def get_predictions(limit=500):
    with connect() as con:
        return con.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

def save_training(best_model, rows_used, model_count, metrics_json):
    with connect() as con:
        con.execute("INSERT INTO training_runs(created_at,best_model,rows_used,model_count,metrics_json) VALUES(?,?,?,?,?)",
        [datetime.now().isoformat(timespec="seconds"), best_model, rows_used, model_count, metrics_json])

def get_training_runs():
    with connect() as con:
        return con.execute("SELECT * FROM training_runs ORDER BY id DESC").fetchall()
