<<<<<<< HEAD
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
=======
# Lern-Periode-12

## Grob-Planung:
In dieser Lernperiode möchte ich ein Tool erstellen welches man mit seiner Sportuhr verbinden kann und dann werden automatisch die Daten daraus in dieses Tool gezogen und dieses Tool analysiert und wertet diese Daten dann aus. Somit sollte man einen guten überblick erhalten. Aktuell denke ich an Daten wie, Schlafdaten, Herzferquenz, Lauf/Rad/Schwimmdaten, jedoch füge ich über die Zeit weitere hinzu.

### 17.10.2025 
Heute habe ich damit begonnen mir ein Projekt auszusuchen und mir einen überblick über mein vorhaben zu verschaffen, zudem liess ich mir einen Prototypen generieren damit ich einen visuellen Standpunkt habe wie das ganze Projekt aussehen könnte:
<img width="1418" height="849" alt="grafik" src="https://github.com/user-attachments/assets/aa99fbc2-03fc-4f3b-9353-6c1ba79d9855" />
Das Projekt sollte am schluss in ca diesem Design enden da mich dies sehr anspricht, inhalt ist aber noch unklar ob es in die selbe richtung gehen sollte.
>>>>>>> 1f85dbff504cfa6452e6a5b6f5c8843a4c4c2dff
