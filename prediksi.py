from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ----- Load & prepare data (same logic as your script) -----
DF_PATH = "matches.csv"  # pastikan file CSV berada di folder yang sama

def prepare_data(df):
    df = df.copy()
    df['team'] = df['team'].str.strip()
    df['opponent'] = df['opponent'].str.strip()

    def encode_result(row):
        if row['venue'] == 'Home':
            if row['result'] == 'W':
                return 'Home Win'
            elif row['result'] == 'L':
                return 'Away Win'
            else:
                return 'Draw'
        else:
            if row['result'] == 'W':
                return 'Away Win'
            elif row['result'] == 'L':
                return 'Home Win'
            else:
                return 'Draw'

    df['match_outcome'] = df.apply(encode_result, axis=1)

    # LabelEncoder for teams
    le_team = LabelEncoder()
    all_teams = pd.concat([df['team'], df['opponent']]).unique()
    le_team.fit(all_teams)

    # Encode match outcome
    le_result = LabelEncoder()
    df['result_encoded'] = le_result.fit_transform(df['match_outcome'])

    # Feature construction function (vectorized via apply later)
    def create_features(row):
        home_team = row['team']
        away_team = row['opponent']

        df_home = df[df['team'] == home_team]
        avg_gf_home = df_home[df_home['venue']=='Home']['gf'].mean()
        avg_ga_home = df_home[df_home['venue']=='Home']['ga'].mean()
        avg_xg_home = df_home['xg'].mean()
        avg_xga_home = df_home['xga'].mean()
        avg_poss_home = df_home['poss'].mean()
        avg_sh_home = df_home['sh'].mean()
        avg_sot_home = df_home['sot'].mean()

        df_away = df[df['team'] == away_team]
        avg_gf_away = df_away[df_away['venue']=='Away']['gf'].mean()
        avg_ga_away = df_away[df_away['venue']=='Away']['ga'].mean()
        avg_xg_away = df_away['xg'].mean()
        avg_xga_away = df_away['xga'].mean()
        avg_poss_away = df_away['poss'].mean()
        avg_sh_away = df_away['sh'].mean()
        avg_sot_away = df_away['sot'].mean()

        return pd.Series([
            avg_gf_home, avg_ga_home, avg_xg_home, avg_xga_home, avg_poss_home, avg_sh_home, avg_sot_home,
            avg_gf_away, avg_ga_away, avg_xg_away, avg_xga_away, avg_poss_away, avg_sh_away, avg_sot_away
        ])

    X = df.apply(create_features, axis=1)
    y = df['result_encoded']

    return df, le_team, le_result, X, y

# ----- Train model at startup (may take some time) -----
print("[app] Loading data and training model... (training happens once on startup)")
raw_df = pd.read_csv(DF_PATH)
raw_df, le_team, le_result, X, y = prepare_data(raw_df)

# In case of missing values (teams with few matches), fill with zeros
X = X.fillna(0)

# simple split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
print("[app] Model trained successfully.")

# ----- Prediction helper (same logic as your prediksi_klub) -----

def prediksi_klub(home, away):
    home = home.strip()
    away = away.strip()
    teams_lower = {t.lower(): t for t in le_team.classes_}

    if home.lower() not in teams_lower or away.lower() not in teams_lower:
        return {"error": "Salah satu klub tidak ditemukan. Pastikan nama sesuai dataset."}

    home_name = teams_lower[home.lower()]
    away_name = teams_lower[away.lower()]

    df_home = raw_df[raw_df['team'] == home_name]
    df_away = raw_df[raw_df['team'] == away_name]

    avg_gf_home = df_home[df_home['venue']=='Home']['gf'].mean() or 0
    avg_ga_home = df_home[df_home['venue']=='Home']['ga'].mean() or 0
    win_home = df_home[(df_home['venue']=='Home') & (df_home['match_outcome']=='Home Win')].shape[0]
    total_home = df_home[df_home['venue']=='Home'].shape[0]
    win_pct_home = (win_home / total_home * 100) if total_home > 0 else 0

    avg_gf_away = df_away[df_away['venue']=='Away']['gf'].mean() or 0
    avg_ga_away = df_away[df_away['venue']=='Away']['ga'].mean() or 0
    win_away = df_away[(df_away['venue']=='Away') & (df_away['match_outcome']=='Away Win')].shape[0]
    total_away = df_away[df_away['venue']=='Away'].shape[0]
    win_pct_away = (win_away / total_away * 100) if total_away > 0 else 0

    fitur_input = pd.DataFrame([[
        avg_gf_home, avg_ga_home, df_home['xg'].mean() or 0, df_home['xga'].mean() or 0, df_home['poss'].mean() or 0, df_home['sh'].mean() or 0, df_home['sot'].mean() or 0,
        avg_gf_away, avg_ga_away, df_away['xg'].mean() or 0, df_away['xga'].mean() or 0, df_away['poss'].mean() or 0, df_away['sh'].mean() or 0, df_away['sot'].mean() or 0
    ]], columns=X.columns)

    proba = model.predict_proba(fitur_input.fillna(0))[0]
    pred_idx = np.argmax(proba)
    hasil_pred = le_result.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx]

    return {
        "home": home_name,
        "away": away_name,
        "stats": {
            "home": {
                "avg_gf_home": float(avg_gf_home),
                "avg_ga_home": float(avg_ga_home),
                "win_pct_home": float(round(win_pct_home, 1))
            },
            "away": {
                "avg_gf_away": float(avg_gf_away),
                "avg_ga_away": float(avg_ga_away),
                "win_pct_away": float(round(win_pct_away, 1))
            }
        },
        "prediction": {
            "result": hasil_pred,
            "confidence": float(round(confidence, 2))
        }
    }

# ----- Flask App -----
app = Flask(__name__)

HTML_TMPL = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Football Match Predictor</title>
    <style>
      body { font-family: Arial, sans-serif; max-width:800px; margin:30px auto; padding:0 16px; }
      .card { border-radius:10px; box-shadow:0 3px 10px rgba(0,0,0,0.1); padding:20px; }
      input, button { padding:8px 10px; font-size:1rem; }
      pre { background:#f6f6f6; padding:12px; border-radius:6px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>Prediksi Menang/Kalah/Draw dengan Statistik</h2>
      <form method="post">
        <label>Masukkan nama klub Home: <input type="text" name="home" required></label><br><br>
        <label>Masukkan nama klub Away: <input type="text" name="away" required></label><br><br>
        <button type="submit">Prediksi</button>
      </form>

      {% if result %}
      <hr>
      <h3>==== Statistik Klub ====</h3>
      <pre>
{{ result_text }}
      </pre>
      {% endif %}

    </div>
  </body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    result_text = ''
    if request.method == 'POST':
        home = request.form.get('home', '')
        away = request.form.get('away', '')
        out = prediksi_klub(home, away)
        if 'error' in out:
            result_text = out['error']
        else:
            result_text = f"{out['home']} (Home):\n- Rata-rata gol kandang: {out['stats']['home']['avg_gf_home']:.2f}\n- Rata-rata kebobolan kandang: {out['stats']['home']['avg_ga_home']:.2f}\n- Persentase menang kandang: {out['stats']['home']['win_pct_home']:.1f}%\n\n{out['away']} (Away):\n- Rata-rata gol tandang: {out['stats']['away']['avg_gf_away']:.2f}\n- Rata-rata kebobolan tandang: {out['stats']['away']['avg_ga_away']:.2f}\n- Persentase menang tandang: {out['stats']['away']['win_pct_away']:.1f}%\n\n📊 Prediksi: {out['home']} vs {out['away']} → {out['prediction']['result']} (confidence: {out['prediction']['confidence']:.2f})"
            result = True
    return render_template_string(HTML_TMPL, result=result, result_text=result_text)

# Simple JSON endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json or {}
    home = data.get('home', '')
    away = data.get('away', '')
    if not home or not away:
        return jsonify({"error": "provide 'home' and 'away' in JSON body"}), 400
    out = prediksi_klub(home, away)
    if 'error' in out:
        return jsonify(out), 400
    return jsonify(out)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
