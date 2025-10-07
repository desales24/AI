from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

class FootballPredictorWithScore:
    def __init__(self, model_path='football_predictor_with_score.joblib'):
        """Load trained model package with score prediction"""
        try:
            self.model_package = joblib.load(model_path)
            self.result_model = self.model_package['result_model']
            self.home_goals_model = self.model_package['home_goals_model']
            self.away_goals_model = self.model_package['away_goals_model']
            self.label_encoders = self.model_package['label_encoders']
            self.target_encoder = self.model_package['target_encoder']
            self.features = self.model_package['features']
            self.model_info = self.model_package.get('model_info', {})
            
            print("✅ Model dengan prediksi skor loaded successfully!")
            print(f"📊 Model Info: {self.model_info.get('accuracy', 'N/A')} accuracy")
            print(f"📊 MAE Skor: {self.model_info.get('mae_home', 'N/A')} (Home), {self.model_info.get('mae_away', 'N/A')} (Away)")
            
        except FileNotFoundError:
            print("❌ Model file not found. Please train the model first.")
            raise
    
    def get_available_teams(self):
        """Get list of teams available in the model"""
        home_teams = list(self.label_encoders['HomeTeam'].classes_)
        away_teams = list(self.label_encoders['AwayTeam'].classes_)
        
        return {
            'home_teams': home_teams,
            'away_teams': away_teams
        }
    
    def predict_match(self, home_team, away_team):
        """Predict match outcome and score between two teams"""
        
        # Default stats (sesuai dengan kode asli)
        default_stats = {
            'HTH Goals': 0.8, 'HTA Goals': 0.5,
            'H Shots': 12.5, 'A Shots': 10.5,
            'H SOT': 4.5, 'A SOT': 3.8,
            'H Fouls': 11.2, 'A Fouls': 11.8,
            'H Corners': 5.2, 'A Corners': 4.8,
            'H Yellow': 1.8, 'A Yellow': 2.1,
            'H Red': 0.08, 'A Red': 0.07
        }
        
        # Prepare feature vector
        features_dict = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            **default_stats
        }
        
        # Calculate engineered features
        features_dict['Shot_Ratio_HT'] = features_dict['HTH Goals'] / (features_dict['HTA Goals'] + 1)
        features_dict['Is_Home_Leading_HT'] = 1 if features_dict['HTH Goals'] > features_dict['HTA Goals'] else 0
        features_dict['Is_Away_Leading_HT'] = 1 if features_dict['HTH Goals'] < features_dict['HTA Goals'] else 0
        features_dict['Is_Draw_HT'] = 1 if features_dict['HTH Goals'] == features_dict['HTA Goals'] else 0
        
        # Create DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Encode categorical variables
        for col in ['HomeTeam', 'AwayTeam']:
            if col in self.label_encoders:
                try:
                    input_df[col] = self.label_encoders[col].transform([features_dict[col]])[0]
                except ValueError:
                    print(f"⚠️  {col} '{features_dict[col]}' not in training data, using default encoding")
                    input_df[col] = 0
        
        # Ensure correct feature order
        input_df = input_df[self.features]
        
        # Make predictions
        # 1. Prediksi hasil (H/D/A)
        result_prediction = self.result_model.predict(input_df)[0]
        result_probabilities = self.result_model.predict_proba(input_df)[0]
        
        # 2. Prediksi skor
        home_goals_pred = self.home_goals_model.predict(input_df)[0]
        away_goals_pred = self.away_goals_model.predict(input_df)[0]
        
        # Round scores to integers
        home_goals_rounded = max(0, round(home_goals_pred))
        away_goals_rounded = max(0, round(away_goals_pred))
        
        # Adjust scores based on result prediction
        if result_prediction == 0:  # Home Win
            if away_goals_rounded >= home_goals_rounded:
                home_goals_rounded = max(home_goals_rounded, away_goals_rounded + 1)
        elif result_prediction == 1:  # Draw
            if home_goals_rounded != away_goals_rounded:
                avg_goals = round((home_goals_rounded + away_goals_rounded) / 2)
                home_goals_rounded = avg_goals
                away_goals_rounded = avg_goals
        else:  # Away Win
            if home_goals_rounded >= away_goals_rounded:
                away_goals_rounded = max(away_goals_rounded, home_goals_rounded + 1)
        
        # Decode result prediction
        result = self.target_encoder.inverse_transform([result_prediction])[0]
        prob_dict = {self.target_encoder.classes_[i]: round(prob * 100, 1) for i, prob in enumerate(result_probabilities)}
        
        return {
            'result': result,
            'probabilities': prob_dict,
            'home_team': home_team,
            'away_team': away_team,
            'score_prediction': {
                'home_goals': home_goals_rounded,
                'away_goals': away_goals_rounded,
                'home_goals_raw': round(home_goals_pred, 2),
                'away_goals_raw': round(away_goals_pred, 2)
            },
            'confidence': max(prob_dict.values()),
            'features_used': features_dict
        }

# Inisialisasi predictor
try:
    predictor = FootballPredictorWithScore('football_predictor_with_score.joblib')
    teams_data = predictor.get_available_teams()
    print("✅ Predictor initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing predictor: {e}")
    predictor = None
    teams_data = {'home_teams': [], 'away_teams': []}

@app.route('/')
def index():
    return render_template('index.html', teams=teams_data['home_teams'])

@app.route('/get_teams')
def get_teams():
    return jsonify(teams_data)

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model tidak tersedia'}), 500
    
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    
    if not home_team or not away_team:
        return jsonify({'error': 'Tim tuan rumah dan tamu harus dipilih'}), 400
    
    try:
        prediction = predictor.predict_match(home_team, away_team)
        
        # Format response untuk frontend
        result = prediction['result']
        home_goals = prediction['score_prediction']['home_goals']
        away_goals = prediction['score_prediction']['away_goals']
        
        # Determine result text
        if result == 'H':
            result_text = f"{home_team} diprediksi MENANG {home_goals}-{away_goals}"
            result_type = "home"
        elif result == 'A':
            result_text = f"{away_team} diprediksi MENANG {away_goals}-{home_goals}"
            result_type = "away"
        else:
            result_text = f"Kedua tim diprediksi SERI {home_goals}-{away_goals}"
            result_type = "draw"
        
        # Match insights
        total_goals = home_goals + away_goals
        goal_diff = abs(home_goals - away_goals)
        
        # Scoring type
        if total_goals >= 4:
            scoring_type = "HIGH-SCORING"
            scoring_icon = "⚡"
        elif total_goals <= 1:
            scoring_type = "LOW-SCORING"
            scoring_icon = "🛡️"
        else:
            scoring_type = "MEDIUM-SCORING"
            scoring_icon = "⚖️"
        
        # Match type
        if goal_diff >= 3:
            match_type = f"KEMENANGAN BESAR"
            match_icon = "💥"
            match_desc = f"Selisih {goal_diff} gol menunjukkan dominasi"
        elif goal_diff == 0:
            match_type = f"SERI KETAT"
            match_icon = "🤝"
            match_desc = "Kedua tim seimbang sepanjang pertandingan"
        elif goal_diff == 1:
            match_type = f"KEMENANGAN TIPIS"
            match_icon = "🎯"
            match_desc = f"Selisih {goal_diff} gol menunjukkan pertandingan sengit"
        else:
            match_type = f"KEMENANGAN NYAMAN"
            match_icon = "😎"
            match_desc = f"Selisih {goal_diff} gol menunjukkan keunggulan"
        
        # Additional insights based on probabilities
        max_prob = max(prediction['probabilities'].values())
        if max_prob > 75:
            confidence_insight = "Prediksi sangat kuat berdasarkan data historis"
            confidence_icon = "🎯"
        elif max_prob > 60:
            confidence_insight = "Prediksi cukup kuat dengan beberapa ketidakpastian"
            confidence_icon = "📊"
        else:
            confidence_insight = "Pertandingan sulit diprediksi, hasil bisa beragam"
            confidence_icon = "⚠️"
        
        # Goal scoring pattern
        if home_goals >= 3 or away_goals >= 3:
            goal_pattern = "Salah satu tim menunjukkan kekuatan serangan tinggi"
            goal_icon = "🔥"
        elif home_goals == 0 and away_goals == 0:
            goal_pattern = "Pertahanan kedua tim sangat solid"
            goal_icon = "🛡️"
        else:
            goal_pattern = "Pola gol normal dengan distribusi seimbang"
            goal_icon = "⚽"
        
        response = {
            'success': True,
            'prediction': {
                'score': f"{home_goals} - {away_goals}",
                'result': result_text,
                'result_type': result_type,
                'probabilities': prediction['probabilities'],
                'stats_used': {
                    'home': {
                        'ht_goals': prediction['features_used']['HTH Goals'],
                        'shots': prediction['features_used']['H Shots'],
                        'shots_on_target': prediction['features_used']['H SOT'],
                        'fouls': prediction['features_used']['H Fouls'],
                        'corners': prediction['features_used']['H Corners']
                    },
                    'away': {
                        'ht_goals': prediction['features_used']['HTA Goals'],
                        'shots': prediction['features_used']['A Shots'],
                        'shots_on_target': prediction['features_used']['A SOT'],
                        'fouls': prediction['features_used']['A Fouls'],
                        'corners': prediction['features_used']['A Corners']
                    }
                },
                'insights': {
                    'scoring': {
                        'type': scoring_type,
                        'icon': scoring_icon,
                        'total_goals': total_goals,
                        'description': f"Pertandingan diprediksi {scoring_type.lower()} dengan {total_goals} gol"
                    },
                    'match_type': {
                        'type': match_type,
                        'icon': match_icon,
                        'goal_diff': goal_diff,
                        'description': match_desc
                    },
                    'confidence': {
                        'insight': confidence_insight,
                        'icon': confidence_icon,
                        'max_prob': max_prob
                    },
                    'goal_pattern': {
                        'insight': goal_pattern,
                        'icon': goal_icon
                    }
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error dalam prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)