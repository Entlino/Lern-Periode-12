import os
import pandas as pd
from datetime import date, timedelta
from garminconnect import Garmin

def sync_to_dataframe(days=30):
    """
    Loggt sich bei Garmin ein, holt die Daten der letzten N Tage 
    und gibt sie als Pandas DataFrame zurück (passend für unsere DB).
    """
    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")
    
    if not email or not password:
        raise ValueError("Umgebungsvariablen GARMIN_EMAIL und GARMIN_PASSWORD fehlen.")

    try:
        client = Garmin(email, password)
        client.login()
    except Exception as e:
        raise ConnectionError(f"Garmin Login fehlgeschlagen: {e}")

    # Zeitraum definieren
    today = date.today()
    start_date = today - timedelta(days=days)
    
    rows = []
    current_date = start_date
    
    print(f"Starte Sync für {days} Tage...")

    while current_date <= today:
        ds = current_date.isoformat()
        try:
            # Daten von Garmin holen
            stats = client.get_stats(ds) or {}
            hr_data = client.get_heart_rates(ds) or {}
            sleep_data = client.get_sleep_data(ds) or {}
            
            # 1. Schritte & Kalorien aus 'stats'
            steps = stats.get("totalSteps")
            calories = stats.get("totalKilocalories")
            
            # 2. Ruhepuls aus 'heart_rates'
            rhr = hr_data.get("restingHeartRate")
            
            # 3. Schlaf (Minuten) aus 'sleep_data' extrahieren
            sleep_min = None
            if sleep_data.get("dailySleepDTO"):
                sleep_min = sleep_data["dailySleepDTO"].get("sleepTimeSeconds", 0) / 60
            elif sleep_data.get("dailySleepDTO") == {}:
                 # Fallback, manchmal sind Sleep-Daten anders strukturiert
                 pass

            # Nur hinzufügen, wenn wir wenigstens irgendwelche Daten haben
            if any(x is not None for x in [steps, calories, rhr, sleep_min]):
                rows.append({
                    "date": ds,
                    "steps": steps,
                    "calories": calories,
                    "rest_hr": rhr,
                    "sleep_minutes": sleep_min
                })
                
        except Exception as e:
            print(f"Fehler bei Tag {ds}: {e}")
        
        current_date += timedelta(days=1)

    # DataFrame erstellen
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    # Datentypen erzwingen, damit die DB nicht meckert
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce")
    df["rest_hr"] = pd.to_numeric(df["rest_hr"], errors="coerce")
    df["sleep_minutes"] = pd.to_numeric(df["sleep_minutes"], errors="coerce")
    
    return df