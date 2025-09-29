from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ===== Load data & model =====
df = pd.read_csv("matches.csv")
df['team'] = df['team'].str.strip()
df['opponent'] = df['opponent'].str.strip()

model = joblib.load("xgb_model.pkl")
le_team = joblib.load("le_team.pkl")
le_result = joblib.load("le_result.pkl")

def prediksi_klub(home, away):
    home = home.strip()
    away = away.strip()
    teams_lower = {t.lower(): t for t in le_team.classes_}

    if home.lower() not in teams_lower or away.lower() not in teams_lower:
        return {"error": "❌ Salah satu klub tidak ditemukan. Pastikan nama sesuai dataset."}

    home_name = teams_lower[home.lower()]
    away_name = teams_lower[away.lower()]

    df_home = df[df['team'] == home_name]
    df_away = df[df['team'] == away_name]

    avg_gf_home = df_home[df_home['venue']=='Home']['gf'].mean()
    avg_ga_home = df_home[df_home['venue']=='Home']['ga'].mean()
    win_home = df_home[(df_home['venue']=='Home') & (df_home['result']=='W')].shape[0]
    total_home = df_home[df_home['venue']=='Home'].shape[0]
    win_pct_home = (win_home / total_home * 100) if total_home > 0 else 0

    avg_gf_away = df_away[df_away['venue']=='Away']['gf'].mean()
    avg_ga_away = df_away[df_away['venue']=='Away']['ga'].mean()
    win_away = df_away[(df_away['venue']=='Away') & (df_away['result']=='W')].shape[0]
    total_away = df_away[df_away['venue']=='Away'].shape[0]
    win_pct_away = (win_away / total_away * 100) if total_away > 0 else 0

    fitur_input = pd.DataFrame([[ 
        avg_gf_home, avg_ga_home, df_home['xg'].mean(), df_home['xga'].mean(),
        df_home['poss'].mean(), df_home['sh'].mean(), df_home['sot'].mean(),
        avg_gf_away, avg_ga_away, df_away['xg'].mean(), df_away['xga'].mean(),
        df_away['poss'].mean(), df_away['sh'].mean(), df_away['sot'].mean()
    ]], columns=model.get_booster().feature_names)

    proba = model.predict_proba(fitur_input)[0]
    pred_idx = np.argmax(proba)
    hasil_pred = le_result.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx]

    return {
        "home_name": home_name,
        "away_name": away_name,
        "stats": {
            "home": {
                "avg_gf": round(avg_gf_home, 2),
                "avg_ga": round(avg_ga_home, 2),
                "win_pct": round(win_pct_home, 1)
            },
            "away": {
                "avg_gf": round(avg_gf_away, 2),
                "avg_ga": round(avg_ga_away, 2),
                "win_pct": round(win_pct_away, 1)
            }
        },
        "prediction": {
            "result": hasil_pred,
            "confidence": round(confidence, 2)
        }
    }

# ===== Route =====
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        home_team = request.form.get("home_team")
        away_team = request.form.get("away_team")
        result = prediksi_klub(home_team, away_team)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
