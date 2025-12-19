import io
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify

# --- Eigene Module importieren ---
import database as db
from ai_engine import GarminAI

# --- Konfiguration ---
APP_TITLE = "Garmin Vision Dashboard"
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
# Wir behalten DATA_DIR für Backups/Sync-CSVs, auch wenn die DB die Hauptquelle ist
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Verzeichnisse erstellen
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# KI-Coach initialisieren (Vision Modell)
coach = GarminAI(model="llama3.2-vision")

# Datenbank beim Start initialisieren
try:
    db.init_db()
    print("Datenbank initialisiert.")
except Exception as e:
    print(f"Warnung bei DB-Init: {e}")


# --- Konstanten für UI ---
DISPLAY_NAMES = {
    "steps": "Schritte",
    "calories": "Kalorien",
    "rest_hr": "Ruhepuls",
    "sleep_minutes": "Schlaf (Minuten)",
}

# (Optional) Kann auch in Config ausgelagert werden
TREND_THRESHOLDS = {
    "steps": 250, "calories": 80, "rest_hr": 0.2, "sleep_minutes": 10,
}
METRIC_GOALS = {
    "steps": "higher", "calories": "higher", "sleep_minutes": "higher", "rest_hr": "lower",
}
KPI_FIELDS = {
    "steps": "avg_steps", "calories": "avg_cal", "rest_hr": "avg_rhr", "sleep_minutes": "avg_sleep",
}

# --- Hilfsfunktionen für Datenverarbeitung ---

def _parse_csv_to_df(file_path: Path) -> pd.DataFrame:
    """Liest eine CSV ein und normalisiert sie (für Uploads)."""
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path)
    
    # Minimales Normalisieren (Spalten umbenennen etc.)
    # Hier nutzen wir eine vereinfachte Version deiner alten Logik
    # Wichtig: Wir mappen auf die DB-Spaltennamen
    column_map = {
        "Steps": "steps", "TotalSteps": "steps",
        "Calories": "calories", "TotalCalories": "calories",
        "RestingHeartRate": "rest_hr", "restingHR": "rest_hr",
        "TotalSleepMinutes": "sleep_minutes", "SleepMinutes": "sleep_minutes",
        "Date": "date", "ActivityDate": "date"
    }
    df = df.rename(columns=column_map)
    
    # Datum standardisieren
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])
    
    return df

def _filter_by_dates(df: pd.DataFrame, start_str: Optional[str], end_str: Optional[str]) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    # Sicherstellen, dass 'date' Spalte datetime.date Objekte enthält (falls aus DB als String kommt)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    if start_str:
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
            df = df[df["date"] >= start]
        except Exception: pass
    if end_str:
        try:
            end = datetime.strptime(end_str, "%Y-%m-%d").date()
            df = df[df["date"] <= end]
        except Exception: pass
    return df

# --- Plotting Engine (Ausgelagert für Vision Support) ---

def create_plot_image(df: pd.DataFrame, metric: str) -> Optional[io.BytesIO]:
    """Erzeugt ein Diagramm und gibt es als BytesIO zurück (für Web & KI)."""
    if df.empty or metric not in df.columns:
        return None

    # Letzte 14 Tage für den Plot
    plot_df = df.sort_values("date").tail(14).copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric, "date"])

    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = plot_df["date"]
    y = plot_df[metric]
    
    # Sleep in Stunden umrechnen für Anzeige
    if metric == "sleep_minutes":
        y = y / 60.0
        ylabel = "Stunden"
    else:
        ylabel = DISPLAY_NAMES.get(metric, metric)

    ax.plot(x, y, marker='o', linestyle='-', linewidth=2, color='#2563eb', markersize=5)
    ax.fill_between(x, y, color='#2563eb', alpha=0.1)
    
    ax.set_title(f"Verlauf: {DISPLAY_NAMES.get(metric, metric)}", fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# --- UI Helper (KPIs, Formatting) ---
# (Hier haben wir deine bestehenden Funktionen etwas gestrafft übernommen)

def _kpi(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    return {
        "days": df["date"].nunique(),
        "avg_steps": df["steps"].mean() if "steps" in df else 0,
        "avg_cal": df["calories"].mean() if "calories" in df else 0,
        "avg_rhr": df["rest_hr"].mean() if "rest_hr" in df else 0,
        "avg_sleep": df["sleep_minutes"].mean() if "sleep_minutes" in df else 0,
    }

def _display_value(value, metric):
    if pd.isna(value) or value is None: return "-"
    if metric == "sleep_minutes":
        h, m = divmod(int(value), 60)
        return f"{h}h {m}m"
    if metric == "rest_hr": return f"{value:.1f}"
    return f"{value:,.0f}".replace(",", ".")

# --- Routes ---

# In app.py

@app.route("/", methods=["GET"])
def index():
    # 1. Daten holen
    df = db.get_data()
    
    # 2. Filter anwenden
    start = request.args.get("start") or ""
    end = request.args.get("end") or ""
    filtered = _filter_by_dates(df, start, end)
    current_goal = int(db.get_setting("step_goal", 10000))
    current_persona = db.get_setting("ai_persona", "coach")
    # 3. KPI berechnen
    kpi = _kpi(filtered)
    
    # 4. CHART DATEN VORBEREITEN (Das hat gefehlt/war fehlerhaft)
    # Wir müssen sicherstellen, dass es Listen sind, keine Pandas-Objekte
    if not filtered.empty:
        chart_df = filtered.sort_values("date")
        chart_data = {
            "dates": chart_df["date"].astype(str).tolist(),
            "steps": chart_df["steps"].fillna(0).tolist(),
            "calories": chart_df["calories"].fillna(0).tolist(),
            "rest_hr": chart_df["rest_hr"].fillna(0).tolist(),
            "sleep": (chart_df["sleep_minutes"] / 60).fillna(0).tolist()
        }
    else:
        # Fallback für leere Daten
        chart_data = {"dates": [], "steps": [], "calories": [], "rest_hr": [], "sleep": []}

    # Restliche UI Daten
    table_rows = []
    if not filtered.empty:
        table_rows = filtered.sort_values("date", ascending=False).head(14).to_dict(orient="records")

    return render_template(
        "index.html",
        title=APP_TITLE,
        kpi=kpi,
        chart_data=chart_data,  # <--- WICHTIG: Das muss hier stehen!
        table_rows=table_rows,
        start=start, end=end,
        has_data=not filtered.empty,
        ai_enabled=True,
        step_goal=current_goal,
        ai_persona=current_persona
    )

@app.route("/settings", methods=["POST"])
def update_settings():
    goal = request.form.get("step_goal")
    persona = request.form.get("ai_persona")
    
    if goal:
        db.update_setting("step_goal", goal)
    if persona:
        db.update_setting("ai_persona", persona)
        
    flash("Einstellungen gespeichert!")
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".csv"):
        flash("Bitte eine gültige CSV-Datei wählen.")
        return redirect(url_for("index"))

    # 1. Speichern (Backup)
    save_path = Path(app.config["UPLOAD_FOLDER"]) / file.filename
    file.save(save_path)
    
    # 2. Parsen & in DB speichern
    try:
        df = _parse_csv_to_df(save_path)
        db.save_dataframe(df) # UPSERT in die Datenbank
        flash(f"Datei importiert: {len(df)} Zeilen in Datenbank übernommen.")
    except Exception as e:
        flash(f"Fehler beim Import: {e}")
        
    return redirect(url_for("index"))

@app.route("/sync", methods=["POST"])
def sync_now():
    days = int(request.form.get("days", "30"))
    try:
        # Importiert jetzt unsere neue Funktion
        from sync_garmin_cli import sync_to_dataframe
        
        # 1. Daten holen
        df_new = sync_to_dataframe(days)
        
        if not df_new.empty:
            # 2. In DB speichern (Upsert)
            db.save_dataframe(df_new)
            flash(f"Sync erfolgreich: {len(df_new)} Tage aktualisiert.")
        else:
            flash("Sync lief durch, aber keine neuen Daten gefunden.")
            
    except Exception as e:
        flash(f"Sync Fehler: {e}")

    return redirect(url_for("index"))

@app.route("/plot/<metric>")
def plot(metric: str):
    """Liefert das Diagramm-Bild an den Browser."""
    df = db.get_data()
    df = _filter_by_dates(df, request.args.get("start"), request.args.get("end"))
    
    img_buf = create_plot_image(df, metric)
    
    if img_buf:
        return send_file(img_buf, mimetype="image/png")
    else:
        # Leeres Bild oder Platzhalter
        return "Keine Daten", 404

@app.route("/ai/coach", methods=["POST"])
def ai_coach():
    """Die neue smarte AI-Route mit Vision Support."""
    payload = request.get_json(silent=True) or {}
    metric = payload.get("metric")
    user_prompt = payload.get("prompt") or "Analysiere diese Daten."
    active_persona = db.get_setting("ai_persona", "coach")
    
    # 1. Daten laden
    df = db.get_data()
    # Filter anwenden (damit KI nur den angezeigten Zeitraum sieht)
    df = _filter_by_dates(df, payload.get("start"), payload.get("end"))
    
    if df.empty:
        return jsonify({"message": "Keine Daten für diesen Zeitraum verfügbar."})

    # 2. Bild generieren (für das Vision Modell!)
    chart_image = None
    if metric:
        chart_image = create_plot_image(df, metric)
    
    # 3. An KI Engine senden
    # Hier passiert die Magie: Wir senden Dataframe UND Bild
    try:
        answer = coach.generate_response(
            df_context=df,       # Engine baut daraus den Smart Context
            user_prompt=user_prompt,
            persona=active_persona  # <--- HIER ÜBERGEBEN
        )
        return jsonify({"message": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download.csv")
def download_csv():
    """Download der aktuell gefilterten Daten als CSV (aus der DB)."""
    # 1. Daten aus der DB laden
    df = db.get_data()
    
    # 2. Filtern (damit man nur den angezeigten Zeitraum herunterlädt)
    start = request.args.get("start")
    end = request.args.get("end")
    df = _filter_by_dates(df, start, end)
    
    if df.empty:
        # Leere CSV mit Headern, falls keine Daten da sind
        df = pd.DataFrame(columns=["date", "steps", "calories", "rest_hr", "sleep_minutes"])

    # 3. CSV im Speicher erstellen
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    
    # 4. In Bytes umwandeln für den Download
    mem = io.BytesIO()
    mem.write(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="garmin_daten_export.csv"
    )


if __name__ == "__main__":
    app.run(debug=True)