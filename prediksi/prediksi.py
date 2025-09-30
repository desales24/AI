import pandas as pd
import joblib
import sys
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load('model_xgb.pkl')
df = pd.read_csv('/root/semester5/coba_AI/persiapandata/football_data_cleaned.csv')

# Siapkan label encoder dari data training
categorical_cols = ['venue', 'comp', 'captain']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col])
    label_encoders[col] = le

# Fitur yang digunakan
features = ['gf','ga','xg','xga','poss','sh','sot','dist','fk','pk','pkatt','goal_difference','venue','comp','captain']

# Fungsi prediksi kemenangan
# input: dict berisi nilai fitur
# output: prediksi (0=Lose, 1=Draw, 2=Win) dan probabilitas


labels = ['Kalah', 'Seri', 'Menang']

# Ambil statistik 5 pertandingan terakhir
def get_last5_stats(team_name):
    import re
    def normalize(s):
        return re.sub(r'[^a-z]', '', str(s).lower())
    team_df = df[df['team'].apply(normalize) == normalize(team_name)].sort_values('date', ascending=False).head(5)
    win = (team_df['result'] == 'W').sum()
    draw = (team_df['result'] == 'D').sum()
    lose = (team_df['result'] == 'L').sum()
    avg_gf = team_df['gf'].mean()
    avg_ga = team_df['ga'].mean()
    return {'last5_win': win, 'last5_draw': draw, 'last5_lose': lose, 'last5_avg_gf': avg_gf, 'last5_avg_ga': avg_ga}

# Ambil statistik head-to-head
def get_head2head_stats(team1, team2):
    import re
    def normalize(s):
        return re.sub(r'[^a-z]', '', str(s).lower())
    h2h_df = df[(df['team'].apply(normalize) == normalize(team1)) & (df['opponent'].apply(normalize) == normalize(team2))].sort_values('date', ascending=False).head(5)
    win = (h2h_df['result'] == 'W').sum()
    draw = (h2h_df['result'] == 'D').sum()
    lose = (h2h_df['result'] == 'L').sum()
    avg_gf = h2h_df['gf'].mean()
    avg_ga = h2h_df['ga'].mean()
    return {'h2h_win': win, 'h2h_draw': draw, 'h2h_lose': lose, 'h2h_avg_gf': avg_gf, 'h2h_avg_ga': avg_ga}

def get_team_stats(team_name, venue_label=1):
    # Normalisasi nama klub untuk pencarian
    import re
    def normalize(s):
        return re.sub(r'[^a-z]', '', str(s).lower())
    team_df = df[df['team'].apply(normalize) == normalize(team_name)]
    if team_df.empty:
        raise ValueError(f"Klub '{team_name}' tidak ditemukan di data.")
    stats = {}
    # Fitur numerik: ambil mean
    num_cols = [c for c in features if c not in categorical_cols]
    for col in num_cols:
        stats[col] = team_df[col].mean()
    # Fitur kategori: ambil mode, lalu label encoding
    for col in categorical_cols:
        mode_val = team_df[col].mode()[0]
        stats[col] = label_encoders[col].transform([mode_val])[0]
    stats['venue'] = label_encoders['venue'].transform([team_df['venue'].mode()[0]])[0]
    return stats

def prediksi_kemenangan(input_dict):
    # Gabungkan fitur tambahan jika ada
    ext_features = [k for k in input_dict.keys() if k not in features]
    X_new = pd.DataFrame([input_dict], columns=features + ext_features)
    probs = model.predict_proba(X_new[features])  # Model hanya pakai fitur utama
    pred = probs[0].argmax()
    return pred, (probs[0] * 100).round(2)

# Contoh penggunaan


# Bandingkan dua tim
tim1 = input("Masukkan nama klub pertama: ")
tim2 = input("Masukkan nama klub kedua: ")
try:
    input_dict1 = get_team_stats(tim1, venue_label=1)  # Home
    input_dict1 = get_team_stats(tim1, venue_label=1)
    input_dict2 = get_team_stats(tim2, venue_label=0)
    # Tambahkan form 5 pertandingan terakhir
    input_dict1.update(get_last5_stats(tim1))
    input_dict2.update(get_last5_stats(tim2))
    # Tambahkan head-to-head
    input_dict1.update(get_head2head_stats(tim1, tim2))
    input_dict2.update(get_head2head_stats(tim2, tim1))
    _, prob1 = prediksi_kemenangan(input_dict1)
    _, prob2 = prediksi_kemenangan(input_dict2)
    avg_prob = (prob1 + prob2) / 2
    # Normalisasi agar total 100% setelah pembulatan
    avg_prob = avg_prob / avg_prob.sum() * 100
    avg_prob = avg_prob.round(2)
    # Koreksi jika total tidak pas 100.00 karena pembulatan
    diff = 100.00 - avg_prob.sum()
    if abs(diff) > 0.01:
        idx = avg_prob.argmax()
        avg_prob[idx] = (avg_prob[idx] + diff).round(2)
    print("Akumulasi probabilitas hasil pertandingan:")
    for i, label in enumerate(labels):
        print(f"  {label}: {avg_prob[i]:.2f}%")
    # Normalisasi agar total 100 persen
    klub1_prob = avg_prob[2]
    klub2_prob = avg_prob[0]
    total = klub1_prob + klub2_prob
    if total != 0:
        klub1_prob = round(klub1_prob / total * 100, 2)
        klub2_prob = round(klub2_prob / total * 100, 2)
        diff = 100.00 - (klub1_prob + klub2_prob)
        if abs(diff) > 0.01:
            if klub1_prob >= klub2_prob:
                klub1_prob = round(klub1_prob + diff, 2)
            else:
                klub2_prob = round(klub2_prob + diff, 2)
    print(f"{tim1}: {klub1_prob:.2f}%")
    print(f"{tim2}: {klub2_prob:.2f}%")
except Exception as e:
    print(e)
