# -*- coding: utf-8 -*-
import io
import json
import os
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend (server-friendly)
import matplotlib.pyplot as plt
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify

APP_TITLE = "Garmin Mini-Dashboard"
UPLOAD_DIR = "uploads"
DATA_DIR = "data"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "400"))
ENABLE_LOCAL_AI = os.environ.get("ENABLE_LOCAL_AI", "1").lower() not in {"0", "false", "off"}

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

TREND_THRESHOLDS = {
    "steps": 250,
    "calories": 80,
    "rest_hr": 0.2,
    "sleep_minutes": 10,
}

METRIC_GOALS = {
    "steps": "higher",
    "calories": "higher",
    "sleep_minutes": "higher",
    "rest_hr": "lower",
}

METRIC_AI_HINTS = {
    "steps": "Beurteile meine Schrittaktivität und gib Tipps, wie ich sie steigern kann.",
    "calories": "Analysiere meinen Kalorienverbrauch und gib Hinweise zu Aktivitätsmix und Balance.",
    "rest_hr": "Bewerte den Verlauf meines Ruhepulses und was er über meine Fitness aussagt.",
    "sleep_minutes": "Welchen Eindruck gewinnt man von meinem Schlaf? Bitte Empfehlungen geben.",
}

KPI_FIELDS = {
    "steps": "avg_steps",
    "calories": "avg_cal",
    "rest_hr": "avg_rhr",
    "sleep_minutes": "avg_sleep",
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

    if "sleep_minutes" in df.columns and df["sleep_minutes"].notna().any():
        non_na = df["sleep_minutes"].dropna()
        max_val = non_na.max()
        # looks like seconds
        if max_val > 600:
            df["sleep_minutes"] = df["sleep_minutes"] / 60.0
        else:
            q95 = non_na.quantile(0.95)
            # values probably in hours if most entries below 24
            if q95 <= 24:
                df["sleep_minutes"] = df["sleep_minutes"] * 60.0

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

def _latest_metric_value(df: pd.DataFrame, col: str) -> Tuple[Optional[float], Optional[date]]:
    if df.empty or col not in df.columns:
        return None, None
    subset = df.dropna(subset=[col]).sort_values("date")
    if subset.empty:
        return None, None
    last = subset.iloc[-1]
    return float(last[col]), last["date"]

def _format_metric(value: Optional[float], metric: str, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return ""
    if metric == "rest_hr":
        formatted = f"{value:.1f}"
    else:
        formatted = f"{value:,.0f}".replace(",", ".")
    if signed and value > 0:
        formatted = f"+{formatted}"
    return formatted

def _format_duration(minutes: Optional[float]) -> str:
    if minutes is None or pd.isna(minutes):
        return ""
    total = int(round(minutes))
    hours, mins = divmod(total, 60)
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins}min"

def _display_value(value: Optional[float], metric: str) -> str:
    if value is None or pd.isna(value):
        return ""
    if metric == "sleep_minutes":
        return _format_duration(value)
    if metric == "rest_hr":
        return f"{value:.1f} bpm"
    if metric == "calories":
        return f"{value:,.0f}".replace(",", ".") + " kcal"
    return f"{value:,.0f}".replace(",", ".")

def _build_trend_cards(df: pd.DataFrame, window: int = 7) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    if df.empty or "date" not in df.columns:
        return cards
    sorted_df = df.sort_values("date")
    for metric in ["steps", "calories", "rest_hr", "sleep_minutes"]:
        if metric not in sorted_df.columns:
            continue
        subset = sorted_df[["date", metric]].dropna().copy()
        subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
        subset = subset.dropna(subset=[metric])
        if len(subset) < window:
            continue
        current = subset.tail(window)
        current_mean = current[metric].mean()
        prev_mean = None
        if len(subset) >= window * 2:
            prev = subset.iloc[-2 * window:-window]
            if not prev.empty:
                prev_mean = prev[metric].mean()
                if pd.isna(prev_mean):
                    prev_mean = None
        delta = current_mean - prev_mean if prev_mean is not None else None
        threshold = TREND_THRESHOLDS.get(metric, 1)
        change = "flat"
        tone = "neutral"
        preferred = METRIC_GOALS.get(metric, "higher")
        if delta is not None:
            if delta > threshold:
                change = "up"
                tone = "positive" if preferred == "higher" else "negative"
            elif delta < -threshold:
                change = "down"
                tone = "positive" if preferred == "lower" else "negative"
        cards.append({
            "metric": metric,
            "label": DISPLAY_NAMES.get(metric, metric),
            "current": _format_metric(current_mean, metric),
            "delta": _format_metric(delta, metric, signed=True) if delta is not None else None,
            "change": change,
            "tone": tone,
            "caption": "Ø letzte 7 Tage" + (" vs. Vorwoche" if delta is not None else "")
        })
    return cards

def _weekly_overview(df: pd.DataFrame, limit: int = 6) -> List[Dict[str, str]]:
    if df.empty or "date" not in df.columns:
        return []
    temp = df.dropna(subset=["date"]).copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = temp.dropna(subset=["date"])
    if temp.empty:
        return []
    temp["week_start"] = temp["date"] - pd.to_timedelta(temp["date"].dt.weekday, unit="D")
    agg_map: Dict[str, str] = {}
    if "steps" in temp.columns:
        agg_map["steps"] = "sum"
    if "calories" in temp.columns:
        agg_map["calories"] = "sum"
    if "sleep_minutes" in temp.columns:
        agg_map["sleep_minutes"] = "mean"
    if "rest_hr" in temp.columns:
        agg_map["rest_hr"] = "mean"
    if not agg_map:
        return []
    grouped = temp.groupby("week_start", as_index=False).agg(agg_map)
    counts = temp.groupby("week_start")["date"].nunique().reset_index(name="days")
    grouped = grouped.merge(counts, on="week_start", how="left")
    grouped = grouped.sort_values("week_start", ascending=False).head(limit)
    overview: List[Dict[str, str]] = []
    for _, row in grouped.iterrows():
        start_dt = row["week_start"].date()
        end_dt = start_dt + timedelta(days=6)
        iso = pd.Timestamp(row["week_start"]).isocalendar()
        label = f"KW {int(iso.week)}"
        range_text = f"{start_dt.strftime('%d.%m.')}–{end_dt.strftime('%d.%m.')}"
        overview.append({
            "label": label,
            "range": range_text,
            "steps": _format_metric(row.get("steps"), "steps"),
            "calories": _format_metric(row.get("calories"), "calories"),
            "sleep": _format_metric(row.get("sleep_minutes"), "sleep_minutes"),
            "rest_hr": _format_metric(row.get("rest_hr"), "rest_hr"),
            "days": str(int(row.get("days", 0)))
        })
    return overview

def _build_insights(df: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    if df.empty or "date" not in df.columns:
        return insights
    sorted_df = df.sort_values("date")

    if _metric_present(sorted_df, "steps"):
        steps_df = sorted_df.dropna(subset=["steps"])
        if not steps_df.empty:
            best_row = steps_df.loc[steps_df["steps"].idxmax()]
            best_value = _format_metric(best_row.get("steps"), "steps")
            if best_value:
                insights.append(f"Aktivster Tag: {best_row['date'].strftime('%d.%m.%Y')} mit {best_value} Schritten.")
            goal_days = int((steps_df["steps"] >= 10000).sum())
            total_days = int(steps_df["steps"].count())
            if total_days:
                pct = (goal_days / total_days) * 100
                insights.append(f"10k-Schritte-Ziel an {goal_days} von {total_days} Tagen erreicht ({pct:.0f}%).")

    if _metric_present(sorted_df, "sleep_minutes"):
        sleep_df = sorted_df.dropna(subset=["sleep_minutes"])
        if not sleep_df.empty:
            avg_last_week = sleep_df.tail(7)["sleep_minutes"].mean()
            insights.append(f"Ø Schlaf der letzten 7 Tage: {_format_duration(avg_last_week)}.")
            rested_nights = int((sleep_df["sleep_minutes"] >= 7 * 60).sum())
            total_nights = int(sleep_df["sleep_minutes"].count())
            if total_nights:
                pct = (rested_nights / total_nights) * 100
                insights.append(f"{rested_nights} von {total_nights} Nächten ≥ 7h ({pct:.0f}%).")

    if _metric_present(sorted_df, "rest_hr"):
        hr_df = sorted_df.dropna(subset=["rest_hr"])
        if len(hr_df) >= 2:
            last = hr_df.tail(7)["rest_hr"].mean()
            first = hr_df.head(7)["rest_hr"].mean()
            if first and last:
                diff = last - first
                if abs(diff) >= 0.2:
                    if diff < 0:
                        insights.append(f"Ruhepuls zuletzt gesunken um {abs(diff):.1f} bpm – gutes Zeichen für Erholung.")
                    else:
                        insights.append(f"Ruhepuls zuletzt gestiegen um {abs(diff):.1f} bpm – ggf. Belastung oder Stress prüfen.")

    if not insights:
        insights.append("Sobald mehr Daten vorhanden sind, erscheinen hier automatisch kurze Auswertungen.")

    return insights

def _build_ai_context(df: pd.DataFrame, focus_metric: Optional[str] = None) -> str:
    if df.empty or "date" not in df.columns:
        return "Es liegen aktuell keine normalisierten Garmin-Daten vor."

    metrics = ["steps", "calories", "rest_hr", "sleep_minutes"]
    if focus_metric in metrics:
        metrics = [focus_metric]

    kpi = _kpi(df)
    trend_lookup = {c["metric"]: c for c in _build_trend_cards(df)}
    weekly = _weekly_overview(df)

    period_line = f"Datenzeitraum: {df['date'].min()} bis {df['date'].max()} ({kpi.get('days', 0)} Tage)."
    lines = [period_line]

    weekly_key_map = {
        "steps": "steps",
        "calories": "calories",
        "sleep_minutes": "sleep",
        "rest_hr": "rest_hr",
    }

    for metric in metrics:
        if not _metric_present(df, metric):
            continue
        label = DISPLAY_NAMES.get(metric, metric)
        latest_value, latest_date = _latest_metric_value(df, metric)
        latest_text = ""
        if latest_value is not None and latest_date:
            latest_text = f"Aktueller Wert { _display_value(latest_value, metric) } am {latest_date.strftime('%d.%m.%Y')}."

        avg_key = KPI_FIELDS.get(metric)
        avg_text = ""
        if avg_key and kpi.get(avg_key):
            avg_text = f"Ø { _display_value(kpi[avg_key], metric) } pro Tag."

        trend = trend_lookup.get(metric)
        trend_text = ""
        if trend and trend.get("delta"):
            direction_word = "gestiegen" if trend["change"] == "up" else "gefallen" if trend["change"] == "down" else "stabil"
            trend_text = f"7-Tage-Durchschnitt {direction_word} um {trend['delta']}."

        lines.append(f"{label}: {latest_text} {avg_text} {trend_text}".strip())

        if weekly and metric in weekly_key_map:
            week_lines = []
            for w in weekly[-3:]:
                val = w.get(weekly_key_map[metric], "")
                if val:
                    week_lines.append(f"{w['label']} ({w['range']}): {val}")
            if week_lines:
                lines.append("Wochenwerte " + "; ".join(week_lines) + ".")

    return "\n".join(lines)

def _call_local_ai(user_prompt: str, context: str, metric: Optional[str] = None) -> str:
    base_prompt = (
        "Du bist ein freundlicher deutschsprachiger Fitness-Coach. "
        "Du erhältst normalisierte Garmin-Daten (Schritte, Kalorien, Ruhepuls, Schlaf in Minuten). "
        "Gib maximal drei kurze Erkenntnisse oder Tipps, ohne medizinische Ratschläge zu ersetzen."
    )
    if metric:
        base_prompt += f" Fokussiere dich auf den Bereich '{DISPLAY_NAMES.get(metric, metric)}'."
    prompt = (
        f"{base_prompt}\n\n"
        f"Daten:\n{context}\n\n"
        f"Aufgabe:\n{user_prompt or 'Gib mir eine kurze Zusammenfassung.'}\n"
        "Antwort:\n"
    )
    url = OLLAMA_URL.rstrip("/") + "/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.4, "num_predict": OLLAMA_MAX_TOKENS},
    }
    try:
        resp = requests.post(url, json=payload, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama-Anfrage fehlgeschlagen: {exc}") from exc

    chunks: List[str] = []
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                chunks.append(data["response"])
            if data.get("done"):
                break
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama-Antwort konnte nicht gelesen werden: {exc}") from exc

    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Ollama meldet keine Antwort.")
    return text

# --- Variant 2: Optional Auto-Sync über inoffizielle garminconnect-Bibliothek ---
SLEEP_SECOND_KEYS = {
    "totalSleepSeconds",
    "sleepDurationInSeconds",
    "durationInSeconds",
    "sleepTimeSeconds",
    "sleepTimeInSeconds",
    "overallSleepSeconds",
}
SLEEP_MINUTE_KEYS = {
    "totalSleepMinutes",
    "sleepDurationInMinutes",
    "sleepTimeMinutes",
    "overallSleepMinutes",
}
SLEEP_NESTED_KEYS = ("sleepSummary", "dailySleepDTO", "sleepProfile")


def _extract_sleep_minutes(sleep) -> Optional[float]:
    """Garmin liefert Schlafdaten in wechselnden Strukturen – wir extrahieren robust Minuten."""
    minute_candidates: List[float] = []
    second_totals: List[float] = []
    segment_seconds: List[float] = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (int, float)):
                    if key in SLEEP_MINUTE_KEYS:
                        minute_candidates.append(float(value))
                    elif key in SLEEP_SECOND_KEYS:
                        second_totals.append(float(value))
                elif isinstance(value, dict):
                    if key in SLEEP_NESTED_KEYS:
                        walk(value)
                    else:
                        walk(value)
                elif isinstance(value, list):
                    if key in ("sleepLevels", "sleepLevelsMap", "sleepScores"):
                        secs = 0.0
                        for item in value:
                            if isinstance(item, dict):
                                seg = item.get("durationInSeconds") or item.get("seconds")
                                if isinstance(seg, (int, float)):
                                    secs += float(seg)
                        if secs:
                            segment_seconds.append(secs)
                    walk(value)
        elif isinstance(obj, list):
            for entry in obj:
                walk(entry)

    try:
        walk(sleep)
    except Exception:
        return None

    if minute_candidates:
        return max(minute_candidates)
    if second_totals:
        return max(second_totals) / 60.0
    if segment_seconds:
        return max(segment_seconds) / 60.0
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

    trend_cards = _build_trend_cards(filtered)
    trend_lookup = {card["metric"]: card for card in trend_cards}
    weekly_overview = _weekly_overview(filtered)
    insights = _build_insights(filtered)
    metric_cards: List[Dict[str, Any]] = []
    for m in metrics:
        latest_value, latest_date = _latest_metric_value(filtered, m)
        avg_key = KPI_FIELDS.get(m)
        avg_value = kpi.get(avg_key) if avg_key else None
        card_info = {
            "metric": m,
            "label": DISPLAY_NAMES.get(m, m),
            "latest": _display_value(latest_value, m),
            "latest_date": latest_date.strftime("%d.%m.%Y") if latest_date else "",
            "avg": _display_value(avg_value, m),
            "trend": trend_lookup.get(m),
            "hint": METRIC_AI_HINTS.get(m, ""),
        }
        metric_cards.append(card_info)
    chart_metrics = metrics[:2]

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
        metric_cards=metric_cards,
        chart_metrics=chart_metrics,
        weekly_overview=weekly_overview,
        insights=insights,
        ai_enabled=ENABLE_LOCAL_AI,
        ollama_model=OLLAMA_MODEL,
        query_string=request.query_string.decode() if request.query_string else "",
        gc_available=gc_available,
        garmin_env_ok=garmin_env_ok
    )

@app.route("/ai/coach", methods=["POST"])
def ai_coach():
    if not ENABLE_LOCAL_AI:
        return jsonify({"error": "Lokales KI-Feature ist deaktiviert."}), 400
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    start = payload.get("start") or None
    end = payload.get("end") or None
    metric = payload.get("metric")
    if metric not in DISPLAY_NAMES:
        metric = None
    if not prompt:
        if metric and metric in METRIC_AI_HINTS:
            prompt = METRIC_AI_HINTS[metric]
        else:
            prompt = "Gib mir eine kurze Zusammenfassung und Handlungsempfehlungen."

    df, _ = _load_all_data()
    df = _filter_by_dates(df, start, end)
    if df.empty:
        return jsonify({"error": "Keine Daten im aktuellen Filter."}), 400

    context = _build_ai_context(df, metric)
    try:
        ai_text = _call_local_ai(prompt, context, metric)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"message": ai_text})

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
                ax.set_facecolor("#f8fafc")
                ax.plot(x, y, marker="o", linewidth=1.5, color="#2563eb", label="Tageswerte", alpha=0.85)
                if len(y) >= 5:
                    rolling = pd.Series(y).rolling(window=7, min_periods=3).mean()
                    ax.plot(x, rolling, linewidth=2.2, color="#0f172a", label="7-Tage Ø")
                ax.legend(loc="upper left", frameon=False)
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
