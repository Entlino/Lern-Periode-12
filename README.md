# Garmin Mini-Dashboard – v2 (CSV + optional Auto‑Sync)

Einfaches Flask‑Backend + HTML/CSS‑Frontend. Liest **CSV‑Exporte** aus Garmin Connect oder holt (inoffiziell) Daten direkt über `garminconnect` ab. Zeigt Kennzahlen & Diagramme (Schritte, Kalorien, Ruhepuls, Schlaf‑Minuten).
> Hinweis: Der Auto‑Sync (Variante 2) nutzt eine inoffizielle Library – nur für den Eigengebrauch, ohne Garantie. Beachte die Garmin‑Nutzungsbedingungen.

## Schnellstart
```bash
cd garmin_simple_app_v2
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Dann im Browser `http://127.0.0.1:5000` öffnen.

## CSV‑Import
- In Garmin Connect einen Bericht (Schritte/Kalorien/Schlaf etc.) öffnen → **Export CSV**.
- CSV in `data/` legen oder im UI hochladen. Mehrere CSVs sind ok (Tagesdaten werden zusammengeführt).

## Auto‑Sync (Variante 2 – inoffiziell)
1. Optionales Paket installieren: `pip install garminconnect`
2. Umgebungsvariablen setzen:
   - `GARMIN_EMAIL`, `GARMIN_PASSWORD`
3. Im UI auf **„Jetzt synchronisieren“** klicken (Standard: letzte 30 Tage).  
   Ergebnis wird als `data/garmin_sync.csv` abgelegt und automatisch angezeigt.

## Bekannte Stolpersteine
- **MFA/2FA**: Beim ersten Login kann eine Bestätigung nötig sein; Folge den Konsolenhinweisen.
- **Diagramme**: In v2 werden Datumsachsen explizit normalisiert – falls zuvor nichts angezeigt wurde, sollte es jetzt funktionieren.
- **Matplotlib**: Der serverseitige Backend „Agg“ ist aktiv; es wird kein GUI‑Stack benötigt.
