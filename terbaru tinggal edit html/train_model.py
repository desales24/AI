import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# ===== Load Dataset =====
df = pd.read_csv("matches.csv")

# Bersihkan nama tim
df['team'] = df['team'].str.strip()
df['opponent'] = df['opponent'].str.strip()

# Buat kolom hasil
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

# Encode nama tim
le_team = LabelEncoder()
all_teams = pd.concat([df['team'], df['opponent']]).unique()
le_team.fit(all_teams)

df['home_team_encoded'] = le_team.transform(df['team'])
df['away_team_encoded'] = le_team.transform(df['opponent'])

# Encode hasil pertandingan
le_result = LabelEncoder()
df['result_encoded'] = le_result.fit_transform(df['match_outcome'])

# Buat fitur
def create_features(row):
    home_team = row['team']
    away_team = row['opponent']

    df_home = df[df['team'] == home_team]
    df_away = df[df['team'] == away_team]

    avg_gf_home = df_home[df_home['venue']=='Home']['gf'].mean()
    avg_ga_home = df_home[df_home['venue']=='Home']['ga'].mean()
    avg_xg_home = df_home['xg'].mean()
    avg_xga_home = df_home['xga'].mean()
    avg_poss_home = df_home['poss'].mean()
    avg_sh_home = df_home['sh'].mean()
    avg_sot_home = df_home['sot'].mean()

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = XGBClassifier(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.1,
    objective='multi:softprob',
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# Simpan model dan encoder
joblib.dump(model, "xgb_model.pkl")
joblib.dump(le_team, "le_team.pkl")
joblib.dump(le_result, "le_result.pkl")

print("✅ Model dan encoder berhasil disimpan!")
