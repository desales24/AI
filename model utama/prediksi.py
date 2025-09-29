import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ===== Load Dataset =====
df = pd.read_csv("matches.csv")

# Bersihkan nama tim
df['team'] = df['team'].str.strip()
df['opponent'] = df['opponent'].str.strip()

# Buat kolom hasil: Home Win / Away Win / Draw
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

# ===== Encode Nama Tim =====
le_team = LabelEncoder()
all_teams = pd.concat([df['team'], df['opponent']]).unique()
le_team.fit(all_teams)

df['home_team_encoded'] = le_team.transform(df['team'])
df['away_team_encoded'] = le_team.transform(df['opponent'])

# Encode hasil pertandingan
le_result = LabelEncoder()
df['result_encoded'] = le_result.fit_transform(df['match_outcome'])

# ===== Fitur & Label =====
# Statistik tim home + statistik lawan
def create_features(row):
    home_team = row['team']
    away_team = row['opponent']

    # Home team stats
    df_home = df[df['team'] == home_team]
    avg_gf_home = df_home[df_home['venue']=='Home']['gf'].mean()
    avg_ga_home = df_home[df_home['venue']=='Home']['ga'].mean()
    avg_xg_home = df_home['xg'].mean()
    avg_xga_home = df_home['xga'].mean()
    avg_poss_home = df_home['poss'].mean()
    avg_sh_home = df_home['sh'].mean()
    avg_sot_home = df_home['sot'].mean()

    # Away team stats
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

feature_cols = [
    'gf_home','ga_home','xg_home','xga_home','poss_home','sh_home','sot_home',
    'gf_away','ga_away','xg_away','xga_away','poss_away','sh_away','sot_away'
]

X = df.apply(create_features, axis=1)
y = df['result_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Latih Model XGBoost =====
model = XGBClassifier(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# ===== Fungsi Prediksi + Statistik =====
def prediksi_klub(home, away):
    home = home.strip()
    away = away.strip()
    teams_lower = {t.lower(): t for t in le_team.classes_}

    if home.lower() not in teams_lower or away.lower() not in teams_lower:
        return "❌ Salah satu klub tidak ditemukan. Pastikan nama sesuai dataset."

    home_name = teams_lower[home.lower()]
    away_name = teams_lower[away.lower()]

    # Ambil data historis
    df_home = df[df['team'] == home_name]
    df_away = df[df['team'] == away_name]

    # Statistik Home
    avg_gf_home = df_home[df_home['venue']=='Home']['gf'].mean()
    avg_ga_home = df_home[df_home['venue']=='Home']['ga'].mean()
    win_home = df_home[(df_home['venue']=='Home') & (df_home['match_outcome']=='Home Win')].shape[0]
    total_home = df_home[df_home['venue']=='Home'].shape[0]
    win_pct_home = (win_home / total_home * 100) if total_home > 0 else 0

    # Statistik Away
    avg_gf_away = df_away[df_away['venue']=='Away']['gf'].mean()
    avg_ga_away = df_away[df_away['venue']=='Away']['ga'].mean()
    win_away = df_away[(df_away['venue']=='Away') & (df_away['match_outcome']=='Away Win')].shape[0]
    total_away = df_away[df_away['venue']=='Away'].shape[0]
    win_pct_away = (win_away / total_away * 100) if total_away > 0 else 0

    # Fitur input untuk prediksi
    fitur_input = pd.DataFrame([[
        avg_gf_home, avg_ga_home, df_home['xg'].mean(), df_home['xga'].mean(), df_home['poss'].mean(), df_home['sh'].mean(), df_home['sot'].mean(),
        avg_gf_away, avg_ga_away, df_away['xg'].mean(), df_away['xga'].mean(), df_away['poss'].mean(), df_away['sh'].mean(), df_away['sot'].mean()
    ]], columns=X.columns)

    proba = model.predict_proba(fitur_input)[0]
    pred_idx = np.argmax(proba)
    hasil_pred = le_result.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx]

    output = f"""
==== Statistik Klub ====
{home_name} (Home):
- Rata-rata gol kandang: {avg_gf_home:.2f}
- Rata-rata kebobolan kandang: {avg_ga_home:.2f}
- Persentase menang kandang: {win_pct_home:.1f}%

{away_name} (Away):
- Rata-rata gol tandang: {avg_gf_away:.2f}
- Rata-rata kebobolan tandang: {avg_ga_away:.2f}
- Persentase menang tandang: {win_pct_away:.1f}%

📊 Prediksi: {home_name} vs {away_name} → {hasil_pred} (confidence: {confidence:.2f})
"""
    return output

# ===== Interaksi Pengguna =====
print("==== Prediksi Menang/Kalah/Draw dengan Statistik ====")
home_team_input = input("Masukkan nama klub Home: ")
away_team_input = input("Masukkan nama klub Away: ")
print(prediksi_klub(home_team_input, away_team_input))
