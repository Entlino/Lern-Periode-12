import os
from garminconnect import Garmin

email = os.environ.get("GARMIN_EMAIL")
password = os.environ.get("GARMIN_PASSWORD")
if not email or not password:
    raise SystemExit("Bitte GARMIN_EMAIL und GARMIN_PASSWORD setzen.")

g = Garmin(email, password)
print("Anmeldung...")
g.login()
print("Login ok. Falls MFA aktiv war, ist der Login nun initialisiert.")
