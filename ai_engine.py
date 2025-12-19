import requests
import json
import pandas as pd

class GarminAI:
    def __init__(self, model="llama3.2:1b", url="http://127.0.0.1:11434"):
        self.model = model
        self.url = f"{url.rstrip('/')}/api/generate"

    def build_smart_context(self, df):
        """Erstellt aus einem DataFrame einen intelligenten Text-Kontext."""
        if df.empty:
            return "Keine Daten verfügbar."
        
        df = df.sort_values("date")
        latest = df.iloc[-1]
        summary_lines = []
        metrics = ["steps", "calories", "rest_hr", "sleep_minutes"]
        
        for m in metrics:
            if m in df.columns and pd.notna(latest.get(m)):
                val = latest[m]
                line = f"- {m}: {val:.0f}"
                if len(df) >= 7:
                    recent = df[m].tail(7).mean()
                    prev = df[m].iloc[-14:-7].mean() if len(df) >= 14 else recent
                    if pd.notna(recent) and pd.notna(prev) and prev > 0:
                        change = ((recent - prev) / prev) * 100
                        trend = "steigend" if change > 2 else "fallend" if change < -2 else "stabil"
                        line += f" (Trend: {trend}, {change:+.1f}%)"
                summary_lines.append(line)

        return "\n".join(summary_lines)

# in ai_engine.py

    def generate_response(self, df_context, user_prompt, image=None, persona="coach"):
        # 1. Kontext bauen (bleibt gleich)
        if isinstance(df_context, pd.DataFrame):
            context_text = self.build_smart_context(df_context)
        else:
            context_text = str(df_context)

        # 2. Die Persönlichkeiten definieren
        personas = {
            "coach": "Du bist ein professioneller, freundlicher Fitness-Coach. Sei motivierend.",
            "drill": "Du bist ein harter Drill-Sergeant! Schreie den Nutzer an (in Großbuchstaben), wenn er faul war. Keine Ausreden!",
            "yogi": "Du bist ein spiritueller Yogi. Sprich von Energie, Balance und Achtsamkeit. Sei sehr entspannt.",
            "nerd": "Du bist ein Sportwissenschaftler. Nutze Fachbegriffe, zitiere Statistiken und sei extrem präzise und nüchtern."
        }
        
        # Wähle den System-Prompt (Fallback auf 'coach')
        system_role = personas.get(persona, personas["coach"])
        
        base_prompt = (
            f"{system_role}\n"
            "Analysiere die Daten. Erkenne Trends. Gib 3 kurze Tipps."
        )

        full_prompt = f"{base_prompt}\n\n[DATEN]:\n{context_text}\n\n[FRAGE]: {user_prompt}\n\n[ANTWORT]:"

        # ... (Rest der Funktion mit payload und request bleibt gleich) ...
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.7 if persona == "yogi" else 0.3} # Yogi darf kreativer sein
        }
        # ...

    
        try:
            # Timeout auf 60s runter, da das kleine Modell viel schneller ist
            resp = requests.post(self.url, json=payload, timeout=300) 
            resp.raise_for_status()
            return resp.json().get("response", "Keine Antwort.").strip()
        except Exception as e:
            return f"KI-Fehler: {str(e)}"