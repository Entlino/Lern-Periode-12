# -*- coding: utf-8 -*-
import io
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend (server-friendly)
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

APP_TITLE = "Garmin Mini-Dashboard"
UPLOAD_DIR = "uploads"
DATA_DIR = "data"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Ensure dirs exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Column name aliases we try to normalize from Garmin CSVs
COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["Date", "date", "ActivityDate", "CalendarDate", "day", "StartDate", "Datum"],
    "steps": ["Steps", "steps", "TotalSteps", "dailySteps", "Step Count", "StepCount", "step_count", "Schritte"],
    "calories": ["Calories", "calories", "TotalCalories", "ActiveCalories", "Active Energy", "Kilocalories", "Kalorien", "totalKilocalories"],
    "rest_hr": ["RestingHeartRate", "restingHR", "Resting Hr", "Resting HR", "rest_hr", "RestingHeartRateInBeatsPerMinute", "RHR", "Ruhepuls", "restingHeartRate"],
    "sleep_minutes": ["SleepMinutes", "Sleep Duration (minutes)", "TotalSleepMinutes", "sleep_minutes", "Sleep time (min)", "Total Sleep Minutes", "Schlaf (Min.)", "totalSleepSeconds", "sleepDurationInSeconds"]
}

DISPLAY_NAMES = {
    "steps": "Schritte",
    "calories": "Kalorien",
    "rest_hr": "Ruhepuls",
    "sleep_minutes": "Schlaf (Minuten)",
}

def _first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "steps", "calories", "rest_hr", "sleep_minutes"])

    # Rename columns to our canonical names if possible
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        present = _first_present(df.columns, aliases)
        if present:
            rename_map[present] = canonical
    df = df.rename(columns=rename_map)

    keep = [c for c in ["date", "steps", "calories", "rest_hr", "sleep_minutes"] if c in df.columns]
    if not keep:
        # special: if sleep seconds present, convert to minutes
        if "totalSleepSeconds" in df.columns:
            df["sleep_minutes"] = pd.to_numeric(df["totalSleepSeconds"], errors="coerce") / 60.0
            keep = ["sleep_minutes"]
        else:
            return pd.DataFrame(columns=["date", "steps", "calories", "rest_hr", "sleep_minutes"])

    df = df[keep].copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Convert numerics (support comma decimals)
    for col in ["steps", "calories", "rest_hr", "sleep_minutes"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("\u00A0", "", regex=False)  # non-breaking space
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If sleep was provided in seconds, convert
    if "sleep_minutes" in df.columns and df["sleep_minutes"].notna().any() and (df["sleep_minutes"].max() > 600):
        # looks like seconds
        df["sleep_minutes"] = df["sleep_minutes"] / 60.0

    # Drop rows without date
    if "date" in df.columns:
        df = df.dropna(subset=["date"])

    # Aggregate by date (if multiple rows per day from different files)
    if "date" in df.columns and len(df) > 0:
        agg_map = {c: "mean" for c in df.columns if c != "date"}
        df = df.groupby("date", as_index=False).agg(agg_map).sort_values("date")

    return df

def _parse_csv(file_path: str) -> pd.DataFrame:
    # Try auto-detected separator to support comma/semicolon
    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path)
    return _normalize_dataframe(df)

def _load_all_data() -> Tuple[pd.DataFrame, List[str]]:
    found_files: List[str] = []
    frames: List[pd.DataFrame] = []

    def maybe_add(path: str):
        if os.path.isfile(path):
            try:
                frames.append(_parse_csv(path))
                found_files.append(os.path.basename(path))
            except Exception as e:
                print(f"Fehler beim Einlesen von {path}: {e}")

    # Single combined CSV
    for name in os.listdir(DATA_DIR):
        if name.lower().endswith(".csv"):
            maybe_add(os.path.join(DATA_DIR, name))

    # Uploaded files
    for name in os.listdir(UPLOAD_DIR):
        if name.lower().endswith(".csv"):
            maybe_add(os.path.join(UPLOAD_DIR, name))

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = _normalize_dataframe(df)
        return df, found_files
    else:
        return pd.DataFrame(columns=["date", "steps", "calories", "rest_hr", "sleep_minutes"]), found_files

def _filter_by_dates(df: pd.DataFrame, start_str: Optional[str], end_str: Optional[str]) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    if start_str:
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
            df = df[df["date"] >= start]
        except Exception:
            pass
    if end_str:
        try:
            end = datetime.strptime(end_str, "%Y-%m-%d").date()
            df = df[df["date"] <= end]
        except Exception:
            pass
    return df

def _kpi(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if df.empty:
        return {"days": 0, "avg_steps": None, "avg_cal": None, "avg_rhr": None, "avg_sleep": None}
    out = {"days": int(df["date"].nunique())}
    out["avg_steps"] = round(df["steps"].mean(), 0) if "steps" in df.columns else None
    out["avg_cal"] = round(df["calories"].mean(), 0) if "calories" in df.columns else None
    out["avg_rhr"] = round(df["rest_hr"].mean(), 1) if "rest_hr" in df.columns else None
    out["avg_sleep"] = round(df["sleep_minutes"].mean(), 0) if "sleep_minutes" in df.columns else None
    return out

def _metric_present(df: pd.DataFrame, col: str) -> bool:
    return (col in df.columns) and df[col].notna().any()

# --- Variant 2: Optional Auto-Sync über inoffizielle garminconnect-Bibliothek ---
def _extract_sleep_minutes(sleep) -> Optional[float]:
    try:
        if isinstance(sleep, dict):
            for key in ("totalSleepSeconds", "sleepDurationInSeconds", "durationInSeconds"):
                v = sleep.get(key)
                if isinstance(v, (int, float)):
                    return float(v) / 60.0
            for sub in ("sleepSummary", "dailySleepDTO"):
                v = sleep.get(sub)
                if isinstance(v, dict):
                    for key in ("totalSleepSeconds", "sleepDurationInSeconds", "durationInSeconds"):
                        vv = v.get(key)
                        if isinstance(vv, (int, float)):
                            return float(vv) / 60.0
        if isinstance(sleep, list) and sleep:
            secs = 0.0
            for item in sleep:
                for key in ("totalSleepSeconds", "sleepDurationInSeconds", "durationInSeconds"):
                    vv = item.get(key) if isinstance(item, dict) else None
                    if isinstance(vv, (int, float)):
                        secs += float(vv)
            if secs:
                return secs / 60.0
    except Exception:
        pass
    return None

def _sync_from_garmin(days: int = 30) -> Tuple[int, Optional[str]]:
    """Fetch last N days from Garmin Connect into data/garmin_sync.csv.
    Returns (rows_written, error_message)."""
    try:
        from garminconnect import Garmin  # type: ignore
    except Exception as e:
        return 0, "garminconnect nicht installiert. Bitte 'pip install garminconnect' ausführen."

    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")
    if not email or not password:
        return 0, "Bitte Umgebungsvariablen GARMIN_EMAIL und GARMIN_PASSWORD setzen."

    try:
        client = Garmin(email, password)
        client.login()  # MFA beim ersten Mal möglich
    except Exception as e:
        return 0, f"Login fehlgeschlagen: {e}"

    today = date.today()
    start = today - timedelta(days=max(1, days) - 1)

    rows = []
    d = start
    while d <= today:
        ds = d.isoformat()
        # defensiv: jede Abfrage einzeln fangen
        try:
            stats = client.get_stats(ds) or {}
        except Exception:
            stats = {}
        try:
            hr = client.get_heart_rates(ds) or {}
        except Exception:
            hr = {}
        try:
            sleep = client.get_sleep_data(ds) or {}
        except Exception:
            sleep = {}

        # Schätze Felder robust
        steps = None
        for k in ("totalSteps", "steps"):
            if isinstance(stats, dict) and stats.get(k) is not None:
                steps = stats.get(k); break
        calories = None
        for k in ("totalKilocalories", "totalCalories", "calories"):
            if isinstance(stats, dict) and stats.get(k) is not None:
                calories = stats.get(k); break
        rhr = hr.get("restingHeartRate") if isinstance(hr, dict) else None
        sleep_minutes = _extract_sleep_minutes(sleep)

        rows.append({
            "Date": ds,
            "Steps": steps,
            "Calories": calories,
            "RestingHeartRate": rhr,
            "TotalSleepMinutes": sleep_minutes
        })
        d += timedelta(days=1)

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "garmin_sync.csv")
    try:
        df.to_csv(out, index=False)
        return len(df), None
    except Exception as e:
        return 0, f"Fehler beim Schreiben der CSV: {e}"

@app.route("/", methods=["GET"])
def index():
    df, files = _load_all_data()
    start = request.args.get("start") or ""
    end = request.args.get("end") or ""
    filtered = _filter_by_dates(df, start, end)

    kpi = _kpi(filtered)
    # Table: latest 14 days (or fewer)
    table_rows = []
    if not filtered.empty:
        latest = filtered.sort_values("date", ascending=False).head(14)
        table_rows = latest.to_dict(orient="records")

    # Which plots to show
    metrics = []
    for m in ["steps", "calories", "rest_hr", "sleep_minutes"]:
        if _metric_present(filtered, m):
            metrics.append(m)

    # Sync status hints
    garmin_env_ok = bool(os.environ.get("GARMIN_EMAIL") and os.environ.get("GARMIN_PASSWORD"))
    try:
        import garminconnect  # noqa: F401
        gc_available = True
    except Exception:
        gc_available = False

    return render_template(
        "index.html",
        title=APP_TITLE,
        files=files,
        kpi=kpi,
        metrics=metrics,
        start=start,
        end=end,
        has_data=(not filtered.empty),
        table_rows=table_rows,
        display_names=DISPLAY_NAMES,
        query_string=request.query_string.decode() if request.query_string else "",
        gc_available=gc_available,
        garmin_env_ok=garmin_env_ok
    )

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Bitte wählen Sie eine CSV-Datei aus.")
        return redirect(url_for("index"))
    if not file.filename.lower().endswith(".csv"):
        flash("Nur CSV-Dateien sind erlaubt.")
        return redirect(url_for("index"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    flash(f"Datei hochgeladen: {file.filename}")
    return redirect(url_for("index"))

@app.route("/sync", methods=["POST"])
def sync_now():
    # Auto-Sync über inoffizielle Bibliothek
    try:
        days = int(request.form.get("days", "30"))
    except Exception:
        days = 30
    rows, err = _sync_from_garmin(days=days)
    if err:
        flash(f"Sync fehlgeschlagen: {err}")
    else:
        flash(f"Sync ok: {rows} Tageszeilen aktualisiert (data/garmin_sync.csv).")
    return redirect(url_for("index"))

@app.route("/plot/<metric>")
def plot(metric: str):
    if metric not in ["steps", "calories", "rest_hr", "sleep_minutes"]:
        return "Unbekannte Metrik", 404
    df, _ = _load_all_data()
    df = _filter_by_dates(df, request.args.get("start"), request.args.get("end"))

    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        if df.empty or (metric not in df.columns) or (not df[metric].notna().any()):
            ax.text(0.5, 0.5, f"Keine Daten für {DISPLAY_NAMES.get(metric, metric)}", ha="center", va="center")
            ax.set_axis_off()
        else:
            # Robust gegen NaNs + sichere Datumsachse
            plot_df = df[["date", metric]].dropna()
            x = pd.to_datetime(plot_df["date"])
            y = pd.to_numeric(plot_df[metric], errors="coerce")
            mask = y.notna()
            x, y = x[mask], y[mask]

            if len(x) == 0:
                ax.text(0.5, 0.5, f"Keine Daten für {DISPLAY_NAMES.get(metric, metric)}", ha="center", va="center")
                ax.set_axis_off()
            else:
                ax.plot(x, y, marker="o", linewidth=1.5)
                ax.set_title(DISPLAY_NAMES.get(metric, metric))
                ax.set_xlabel("Datum")
                ax.set_ylabel(DISPLAY_NAMES.get(metric, metric))
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                fig.autofmt_xdate()

        buf = io.BytesIO()
        fig.tight_layout()
        plt.savefig(buf, format="png", dpi=140)
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        plt.close(fig)
        return f"Fehler beim Erstellen des Diagramms: {e}", 500

@app.route("/download.csv")
def download_csv():
    """Download der aktuell gefilterten, normalisierten Daten als CSV."""
    df, _ = _load_all_data()
    df = _filter_by_dates(df, request.args.get("start"), request.args.get("end"))
    if df.empty:
        df = pd.DataFrame(columns=["date", "steps", "calories", "rest_hr", "sleep_minutes"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="garmin_normalisiert.csv"
    )

if __name__ == "__main__":
    # For local dev, run: python app.py
    app.run(debug=True)
