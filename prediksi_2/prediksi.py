import pandas as pd
import numpy as np
import joblib

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
    
    def predict_match(self, home_team, away_team, additional_stats=None):
        """Predict match outcome and score between two teams"""
        
        # Default stats
        default_stats = {
            'HTH Goals': 0.8, 'HTA Goals': 0.5,
            'H Shots': 12.5, 'A Shots': 10.5,
            'H SOT': 4.5, 'A SOT': 3.8,
            'H Fouls': 11.2, 'A Fouls': 11.8,
            'H Corners': 5.2, 'A Corners': 4.8,
            'H Yellow': 1.8, 'A Yellow': 2.1,
            'H Red': 0.08, 'A Red': 0.07
        }
        
        # Update with user provided stats
        if additional_stats:
            default_stats.update(additional_stats)
        
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
        
        # Round scores to integers (tapi simpan juga yang asli untuk probabilitas)
        home_goals_rounded = max(0, round(home_goals_pred))
        away_goals_rounded = max(0, round(away_goals_pred))
        
        # Adjust scores based on result prediction
        # Jika prediksi Home Win tapi skor away > home, adjust
        if result_prediction == 0:  # Home Win
            if away_goals_rounded >= home_goals_rounded:
                home_goals_rounded = max(home_goals_rounded, away_goals_rounded + 1)
        elif result_prediction == 1:  # Draw
            if home_goals_rounded != away_goals_rounded:
                # Set to average of both, atau sesuaikan
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

def display_prediction(result_dict):
    """Display prediction results with score in a nice format"""
    result = result_dict['result']
    home_team = result_dict['home_team']
    away_team = result_dict['away_team']
    probabilities = result_dict['probabilities']
    confidence = result_dict['confidence']
    score = result_dict['score_prediction']
    
    print("\n" + "=" * 70)
    print(f"🏆 PREDIKSI PERTANDINGAN: {home_team} vs {away_team}")
    print("=" * 70)
    
    # Display score prediction
    print(f"📅 PREDIKSI SKOR: {score['home_goals']} - {score['away_goals']}")
    
    # Result interpretation
    if result == 'H':
        print(f"🎯 HASIL PREDIKSI: HOME WIN")
        print(f"   🏆 {home_team} diprediksi MENANG {score['home_goals']}-{score['away_goals']}")
    elif result == 'A':
        print(f"🎯 HASIL PREDIKSI: AWAY WIN") 
        print(f"   🏆 {away_team} diprediksi MENANG {score['away_goals']}-{score['home_goals']}")
    else:
        print(f"🎯 HASIL PREDIKSI: DRAW")
        print(f"   🤝 Kedua tim diprediksi SERI {score['home_goals']}-{score['away_goals']}")
    
    print(f"\n📊 PROBABILITAS HASIL:")
    print(f"   🟢 {home_team} menang: {probabilities['H']}%")
    print(f"   🟡 Draw: {probabilities['D']}%")
    print(f"   🔴 {away_team} menang: {probabilities['A']}%")
    
    # Score confidence (berdasarkan selisih probabilitas)
    max_prob = max(probabilities.values())
    if max_prob > 70:
        confidence_text = "TINGGI"
        emoji = "🎯"
    elif max_prob > 60:
        confidence_text = "SEDANG"
        emoji = "📊"
    else:
        confidence_text = "RENDAH"
        emoji = "⚠️"
    
    print(f"\n🎲 TINGKAT KEYAKINAN: {confidence_text} {emoji} ({confidence}%)")
    
    # Show key features used
    features = result_dict['features_used']
    print(f"\n📈 STATISTIK YANG DIGUNAKAN:")
    print(f"   ⚽ Gol HT: {features['HTH Goals']} - {features['HTA Goals']}")
    print(f"   🎯 Shots: {features['H Shots']} - {features['A Shots']}")
    print(f"   🎪 Shots on Target: {features['H SOT']} - {features['A SOT']}")
    
    # Additional score insights
    print(f"\n💡 INSIGHT SKOR:")
    goal_diff = abs(score['home_goals'] - score['away_goals'])
    total_goals = score['home_goals'] + score['away_goals']
    
    if total_goals >= 4:
        print(f"   ⚡ Pertandingan diprediksi HIGH-SCORING ({total_goals} gol)")
    elif total_goals <= 1:
        print(f"   🛡️  Pertandingan diprediksi LOW-SCORING ({total_goals} gol)")
    else:
        print(f"   ⚖️  Pertandingan diprediksi MEDIUM-SCORING ({total_goals} gol)")
    
    if goal_diff >= 3:
        print(f"   💥 Diprediksi KEMENANGAN BESAR (selisih {goal_diff} gol)")
    elif goal_diff == 0:
        print(f"   🤝 Diprediksi SERI")
    else:
        print(f"   🎪 Diprediksi pertandingan KETAT (selisih {goal_diff} gol)")

def get_custom_stats():
    """Get custom stats from user input"""
    print("\n📊 Masukkan statistik tambahan (tekan Enter untuk skip):")
    
    stats = {}
    stat_mapping = {
        'Gol Home Team di HT': 'HTH Goals',
        'Gol Away Team di HT': 'HTA Goals', 
        'Shots Home Team': 'H Shots',
        'Shots Away Team': 'A Shots',
        'Shots on Target Home': 'H SOT',
        'Shots on Target Away': 'A SOT',
        'Fouls Home Team': 'H Fouls',
        'Fouls Away Team': 'A Fouls',
        'Corners Home Team': 'H Corners',
        'Corners Away Team': 'A Corners'
    }
    
    for display_name, feature_name in stat_mapping.items():
        default_value = "0.8" if "Gol Home" in display_name else "12.5" if "Shots Home" in display_name else "10.5" if "Shots Away" in display_name else "4.5" if "Target Home" in display_name else "3.8" if "Target Away" in display_name else "11.2" if "Fouls Home" in display_name else "11.8" if "Fouls Away" in display_name else "5.2" if "Corners Home" in display_name else "4.8"
        
        user_input = input(f"   {display_name} (default {default_value}): ").strip()
        if user_input:
            try:
                stats[feature_name] = float(user_input)
            except ValueError:
                print(f"     ❌ Input tidak valid, menggunakan default")
    
    return stats

def main():
    # Load predictor
    try:
        predictor = FootballPredictorWithScore('football_predictor_with_score.joblib')
    except Exception as e:
        print("Tidak dapat memuat model. Pastikan file 'football_predictor_with_score.joblib' ada.")
        print("Jalankan training model terlebih dahulu.")
        return
    
    print("\n🎯 PREDIKSI PERTANDINGAN SEPAK BOLA + SKOR")
    print("=" * 50)
    
    while True:
        print("\n" + "-" * 40)
        print("Masukkan detail pertandingan:")
        home_team = input("Home Team: ").strip()
        away_team = input("Away Team: ").strip()
        
        if not home_team or not away_team:
            print("❌ Nama tim harus diisi!")
            continue
        
        # Ask for custom stats
        use_custom = input("\nGunakan statistik custom? (y/n): ").lower().strip()
        additional_stats = {}
        if use_custom == 'y':
            additional_stats = get_custom_stats()
        
        # Make prediction
        try:
            prediction = predictor.predict_match(home_team, away_team, additional_stats)
            display_prediction(prediction)
            
        except Exception as e:
            print(f"❌ Error dalam prediksi: {e}")
        
        # Ask to continue
        continue_pred = input("\n🔁 Lakukan prediksi lagi? (y/n): ").lower().strip()
        if continue_pred != 'y':
            print("👋 Terima kasih telah menggunakan prediksi sepak bola!")
            break

if __name__ == "__main__":
    main()