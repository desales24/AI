from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
import re
import numpy as np
from scipy.stats import poisson
from datetime import datetime

app = Flask(__name__)

class PoissonScorePredictor:
    def __init__(self):
        self.team_stats = {}
    
    def normalize_team_name(self, team_name):
        return re.sub(r'[^a-z\s]', '', str(team_name).lower()).strip()
    
    def calculate_current_stats(self, df, team_name, is_home=True):
        """Hitung statistik dengan mempertimbangkan home/away performance"""
        team_norm = self.normalize_team_name(team_name)
        
        if is_home:
            team_data = df[(df['team'].apply(self.normalize_team_name) == team_norm) & 
                          (df['venue'] == 'Home')].sort_values('date', ascending=False).head(5)
        else:
            team_data = df[(df['team'].apply(self.normalize_team_name) == team_norm) & 
                          (df['venue'] == 'Away')].sort_values('date', ascending=False).head(5)
        
        # Fallback jika tidak ada data home/away spesifik
        if len(team_data) == 0:
            team_data = df[df['team'].apply(self.normalize_team_name) == team_norm].sort_values('date', ascending=False).head(5)
        
        if len(team_data) == 0:
            return None
            
        return {
            'attack_gf': team_data['gf'].mean(),
            'defense_ga': team_data['ga'].mean(),
            'attack_xg': team_data['xg'].mean(),
            'defense_xga': team_data['xga'].mean(),
            'matches_played': len(team_data)
        }
    
    def predict_score(self, home_team, away_team, df):
        home_stats = self.calculate_current_stats(df, home_team, is_home=True)
        away_stats = self.calculate_current_stats(df, away_team, is_home=False)
        
        if home_stats is None or away_stats is None:
            return (1, 1), 0.1, [((1, 1), 0.1), ((0, 0), 0.05), ((2, 1), 0.05)], 1.0, 1.0
        
        # Calculate expected goals dengan home advantage yang lebih realistis
        home_attack = (home_stats['attack_gf'] * 0.6 + home_stats['attack_xg'] * 0.4)
        home_defense = (home_stats['defense_ga'] * 0.6 + home_stats['defense_xga'] * 0.4)
        away_attack = (away_stats['attack_gf'] * 0.6 + away_stats['attack_xg'] * 0.4)
        away_defense = (away_stats['defense_ga'] * 0.6 + away_stats['defense_xga'] * 0.4)
        
        # Expected goals dengan home advantage yang lebih moderat
        home_expected = (home_attack + away_defense) / 2 * 1.15  # Reduced from 1.2
        away_expected = (away_attack + home_defense) / 2 * 0.92  # Increased from 0.9
        
        # Ensure reasonable values
        home_expected = max(0.1, min(4.0, home_expected))
        away_expected = max(0.1, min(3.0, away_expected))
        
        # Prediksi skor menggunakan Poisson
        max_goals = 5
        score_probs = []
        best_score = (0, 0)
        best_prob = 0
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = poisson.pmf(i, home_expected) * poisson.pmf(j, away_expected)
                score_probs.append(((i, j), prob))
                if prob > best_prob:
                    best_prob = prob
                    best_score = (i, j)
        
        top_scores = sorted(score_probs, key=lambda x: x[1], reverse=True)[:3]
        
        return best_score, best_prob, top_scores, home_expected, away_expected

# Load model dan components
try:
    model = joblib.load('model_xgb.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    features = joblib.load('features.pkl')
    
    poisson_predictor = PoissonScorePredictor()
    
    DATA_PATH = '/root/semester5/coba_AI/persiapandata/football_data_cleaned.csv'
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)
    
    all_teams = sorted(df['team'].unique().tolist())
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

def normalize_team_name(team_name):
    return re.sub(r'[^a-z\s]', '', str(team_name).lower()).strip()

def get_team_form_stats(team_name, is_home=True):
    """Hitung statistik form dengan mempertimbangkan home/away"""
    team_norm = normalize_team_name(team_name)
    
    if is_home:
        venue_filter = 'Home'
    else:
        venue_filter = 'Away'
    
    team_df = df[(df['team'].apply(normalize_team_name) == team_norm) & 
                (df['venue'] == venue_filter)].sort_values('date', ascending=False).head(5)
    
    # Fallback ke semua pertandingan jika tidak ada data home/away
    if team_df.empty:
        team_df = df[df['team'].apply(normalize_team_name) == team_norm].sort_values('date', ascending=False).head(5)
    
    if team_df.empty:
        return {'avg_gf': 0, 'avg_ga': 0, 'form': 0}
    
    avg_gf = team_df['gf'].mean()
    avg_ga = team_df['ga'].mean()
    form_points = team_df['points'].sum() if 'points' in team_df.columns else 0
    
    return {'avg_gf': round(avg_gf, 2), 'avg_ga': round(avg_ga, 2), 'form': form_points}

def get_head2head_stats(home_team, away_team):
    """Statistik head-to-head yang lebih komprehensif"""
    home_norm = normalize_team_name(home_team)
    away_norm = normalize_team_name(away_team)
    
    # Pertandingan dimana home_team vs away_team
    h2h_home = df[
        (df['team'].apply(normalize_team_name) == home_norm) & 
        (df['opponent'].apply(normalize_team_name) == away_norm)
    ].sort_values('date', ascending=False).head(10)
    
    # Pertandingan dimana away_team vs home_team (untuk balance)
    h2h_away = df[
        (df['team'].apply(normalize_team_name) == away_norm) & 
        (df['opponent'].apply(normalize_team_name) == home_norm)
    ].sort_values('date', ascending=False).head(10)
    
    if h2h_home.empty and h2h_away.empty:
        return {'h2h_avg_gf': 0, 'h2h_avg_ga': 0, 'h2h_wins': 0, 'total_matches': 0}
    
    # Gabungkan statistik dari kedua perspektif
    total_matches = len(h2h_home) + len(h2h_away)
    total_gf_home = h2h_home['gf'].sum() + h2h_away['ga'].sum()  # Gol yang dicetak home_team
    total_ga_home = h2h_home['ga'].sum() + h2h_away['gf'].sum()  # Gol yang kemasukan home_team
    
    wins_home = len(h2h_home[h2h_home['result'] == 'W']) + len(h2h_away[h2h_away['result'] == 'L'])
    
    return {
        'h2h_avg_gf': round(total_gf_home / total_matches, 2) if total_matches > 0 else 0,
        'h2h_avg_ga': round(total_ga_home / total_matches, 2) if total_matches > 0 else 0,
        'h2h_wins': wins_home,
        'total_matches': total_matches
    }

def prepare_prediction_data(home_team, away_team):
    """Siapkan data prediksi dengan pendekatan yang lebih seimbang"""
    # Get team stats dengan home/away context
    home_stats = get_team_form_stats(home_team, is_home=True)
    away_stats = get_team_form_stats(away_team, is_home=False)
    h2h_stats = get_head2head_stats(home_team, away_team)
    
    # Siapkan input untuk model - gunakan pendekatan yang lebih netral
    input_data = {}
    
    # Numerical features dengan weighting yang lebih seimbang
    numerical_features = {
        'last5_avg_gf': (home_stats['avg_gf'] * 0.6 + away_stats['avg_ga'] * 0.4),
        'last5_avg_ga': (home_stats['avg_ga'] * 0.6 + away_stats['avg_gf'] * 0.4),
        'h2h_avg_gf': h2h_stats['h2h_avg_gf'],
        'h2h_avg_ga': h2h_stats['h2h_avg_ga'],
        'form': home_stats['form']
    }
    
    for feature, value in numerical_features.items():
        if feature in features:
            input_data[feature] = value
    
    # Categorical features
    for col in label_encoders.keys():
        if col in features:
            if col == 'venue':
                input_data[col] = 1  # Home venue
            else:
                # Gunakan nilai dari home team
                team_data = df[df['team'].apply(normalize_team_name) == normalize_team_name(home_team)]
                if not team_data.empty:
                    mode_val = team_data[col].mode()[0] if len(team_data[col].mode()) > 0 else team_data[col].iloc[0]
                    input_data[col] = label_encoders[col].transform([mode_val])[0]
                else:
                    input_data[col] = 0
    
    return input_data, home_stats, away_stats, h2h_stats

def calculate_balanced_probabilities(home_probs, away_probs):
    """Hitung probabilitas yang lebih seimbang antara home dan away"""
    # Konversi ke numpy array untuk perhitungan
    home_probs = np.array(home_probs)
    away_probs = np.array(away_probs)
    
    # Pendekatan baru: rata-rata tertimbang dengan mempertimbangkan kedua perspektif
    # Berikan bobot lebih pada performa aktual daripada home advantage
    home_weight = 0.55  # Sedikit home advantage
    away_weight = 0.45
    
    weighted_probs = (home_probs * home_weight + away_probs * away_weight)
    
    # Normalisasi
    total = np.sum(weighted_probs)
    if total > 0:
        weighted_probs = (weighted_probs / total) * 100
    
    # Probabilitas akhir dengan pembagian seri yang lebih fair
    prob_home_win = weighted_probs[2]
    prob_draw = weighted_probs[1]
    prob_away_win = weighted_probs[0]
    
    # Hitung probabilitas menang dengan mempertimbangkan home advantage yang lebih moderat
    home_win_prob = prob_home_win + (prob_draw * 0.45)  # Reduced home advantage in draws
    away_win_prob = prob_away_win + (prob_draw * 0.45)
    # 10% dari draw tetap sebagai draw murni
    
    # Normalisasi final untuk memastikan total 100%
    total_final = home_win_prob + away_win_prob
    if total_final > 0:
        home_win_prob = (home_win_prob / total_final) * 100
        away_win_prob = (away_win_prob / total_final) * 100
    
    return round(home_win_prob, 2), round(away_win_prob, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    error = None
    
    if request.method == 'POST':
        tim1 = request.form.get('tim1')
        tim2 = request.form.get('tim2')
        
        try:
            if not tim1 or not tim2:
                raise ValueError("Nama kedua klub harus diisi")
            
            if normalize_team_name(tim1) == normalize_team_name(tim2):
                raise ValueError("Nama klub tidak boleh sama")
            
            # Validasi tim ada di database
            if tim1 not in all_teams:
                raise ValueError(f"Klub '{tim1}' tidak ditemukan dalam database")
            if tim2 not in all_teams:
                raise ValueError(f"Klub '{tim2}' tidak ditemukan dalam database")
            
            # Siapkan data prediksi
            input_data, home_stats, away_stats, h2h_stats = prepare_prediction_data(tim1, tim2)
            
            # Prediksi dari kedua perspektif
            X_input = pd.DataFrame([input_data], columns=features)
            
            # Prediksi untuk tim1 sebagai home
            home_probs = model.predict_proba(X_input)[0] * 100
            
            # Prediksi untuk tim2 sebagai home (membalik venue)
            away_input = input_data.copy()
            away_input['venue'] = 0  # Away venue
            X_away = pd.DataFrame([away_input], columns=features)
            away_probs = model.predict_proba(X_away)[0] * 100
            
            # Hitung probabilitas menang yang lebih seimbang
            win_prob1, win_prob2 = calculate_balanced_probabilities(home_probs, away_probs)
            
            # Prediksi skor
            pred_score, score_prob, top_scores, home_expected, away_expected = poisson_predictor.predict_score(tim1, tim2, df)
            
            hasil = {
                'tim1': tim1,
                'tim2': tim2,
                'win_prob1': win_prob1,
                'win_prob2': win_prob2,
                'predicted_score': pred_score,
                'score_probability': round(score_prob * 100, 2),
                'top_scores': top_scores,
                'home_stats': home_stats,
                'away_stats': away_stats,
                'h2h_stats': h2h_stats,
                'home_expected': round(home_expected, 2),
                'away_expected': round(away_expected, 2),
                'note': 'Prediksi sudah mempertimbangkan performa home/away dan head-to-head'
            }
            
        except Exception as e:
            error = str(e)
    
    return render_template('index.html', hasil=hasil, error=error, teams=all_teams)

@app.route('/search_teams')
def search_teams():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    matched_teams = [team for team in all_teams if query in team.lower()]
    return jsonify(matched_teams[:10])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)