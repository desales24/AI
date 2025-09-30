# training_model_fixed.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy.stats import poisson
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define PoissonScorePredictor di luar function agar bisa di-pickle
class PoissonScorePredictor:
    def __init__(self):
        self.team_stats = {}
        
    def calculate_stats(self, df):
        """Hitung statistik untuk setiap tim"""
        # Normalize team names
        df = df.copy()
        df['team_normalized'] = df['team'].str.lower().str.replace(r'[^a-z\s]', '', regex=True).str.strip()
        
        for team in df['team_normalized'].unique():
            team_data = df[df['team_normalized'] == team]
            if len(team_data) > 0:
                original_name = team_data['team'].iloc[0]  # Keep original name for reference
                self.team_stats[team] = {
                    'original_name': original_name,
                    'attack_gf': team_data['gf'].mean(),
                    'defense_ga': team_data['ga'].mean(),
                    'attack_xg': team_data['xg'].mean(),
                    'defense_xga': team_data['xga'].mean(),
                    'matches_played': len(team_data)
                }
        print(f"Calculated stats for {len(self.team_stats)} teams")
    
    def normalize_team_name(self, team_name):
        """Normalize team name untuk konsistensi"""
        return str(team_name).lower().replace('[^a-z\s]', '').strip()
    
    def predict_score(self, home_team, away_team):
        """Prediksi skor menggunakan distribusi Poisson"""
        home_norm = self.normalize_team_name(home_team)
        away_norm = self.normalize_team_name(away_team)
        
        if home_norm not in self.team_stats or away_norm not in self.team_stats:
            # Fallback: return average scores
            return (1, 1), 0.1, [((1, 1), 0.1), ((0, 0), 0.05), ((2, 1), 0.05)]
        
        home_stats = self.team_stats[home_norm]
        away_stats = self.team_stats[away_norm]
        
        # Calculate expected goals dengan weighting
        home_attack = (home_stats['attack_gf'] * 0.6 + home_stats['attack_xg'] * 0.4)
        home_defense = (home_stats['defense_ga'] * 0.6 + home_stats['defense_xga'] * 0.4)
        away_attack = (away_stats['attack_gf'] * 0.6 + away_stats['attack_xg'] * 0.4)
        away_defense = (away_stats['defense_ga'] * 0.6 + away_stats['defense_xga'] * 0.4)
        
        # Expected goals dengan home advantage
        home_expected = (home_attack + away_defense) / 2 * 1.2
        away_expected = (away_attack + home_defense) / 2 * 0.9
        
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
        
        # Get top 3 scores
        score_probs.sort(key=lambda x: x[1], reverse=True)
        top_scores = score_probs[:3]
        
        return best_score, best_prob, top_scores
    
    def evaluate(self, df, sample_size=100):
        """Evaluasi model Poisson"""
        print("Evaluating Poisson model...")
        matches_evaluated = 0
        total_mae = 0
        
        # Sample random matches for evaluation
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        for _, match in sample_df.iterrows():
            home_team = match['team']
            away_team = match['opponent']
            actual_score = (match['gf'], match['ga'])
            
            pred_score, _, _ = self.predict_score(home_team, away_team)
            
            # Calculate MAE
            mae = (abs(actual_score[0] - pred_score[0]) + abs(actual_score[1] - pred_score[1])) / 2
            total_mae += mae
            matches_evaluated += 1
        
        avg_mae = total_mae / matches_evaluated if matches_evaluated > 0 else 0
        print(f"Poisson Model MAE: {avg_mae:.3f} (based on {matches_evaluated} matches)")
        return avg_mae

def train_complete_model():
    # Load data
    df = pd.read_csv('/root/semester5/coba_AI/persiapandata/football_data_cleaned.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Check for potential data leakage
    print("\nChecking for data leakage...")
    print(f"Unique teams: {df['team'].nunique()}")
    print(f"Unique opponents: {df['opponent'].nunique()}")
    
    # 1. Train XGBoost Model untuk Prediksi Hasil
    print("\n" + "="*50)
    print("TRAINING XGBOOST MODEL")
    print("="*50)
    
    # Preprocessing
    categorical_cols = ['venue', 'comp', 'captain']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Map result to numerical values
    result_mapping = {'W': 2, 'D': 1, 'L': 0}
    df['result_encoded'] = df['result'].map(result_mapping)
    
    # Select features - HAPUS goal_difference untuk hindari data leakage
    features = ['gf','ga','xg','xga','poss','sh','sot','dist','fk','pk','pkatt','venue','comp','captain']
    
    # Remove rows with missing values
    df_clean = df.dropna(subset=features + ['result_encoded'])
    print(f"Data after cleaning: {df_clean.shape}")
    
    # Split data by season atau time-based split untuk hindari leakage
    df_clean = df_clean.sort_values('date')
    split_idx = int(len(df_clean) * 0.8)
    
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['result_encoded']
    X_test = test_df[features]
    y_test = test_df['result_encoded']
    
    print(f"Time-based split:")
    print(f"Training set: {X_train.shape} ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test set: {X_test.shape} ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Train XGBoost model dengan parameter lebih konservatif
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=200,  # Kurangi untuk hindari overfitting
        learning_rate=0.05,  # Learning rate lebih kecil
        max_depth=4,  # Depth lebih kecil
        random_state=42,
        eval_metric='mlogloss',
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Fit model
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate model
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nXGBoost Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Kalah', 'Seri', 'Menang']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # 2. Train Poisson Model untuk Prediksi Skor
    print("\n" + "="*50)
    print("TRAINING POISSON SCORE PREDICTOR")
    print("="*50)
    
    # Train Poisson model
    poisson_predictor = PoissonScorePredictor()
    poisson_predictor.calculate_stats(df)
    
    # Evaluate Poisson model
    poisson_mae = poisson_predictor.evaluate(df)
    
    # 3. Save semua model
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    
    # Save models
    joblib.dump(xgb_model, 'model_xgb.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(result_mapping, 'result_mapping.pkl')
    joblib.dump(poisson_predictor, 'poisson_predictor.pkl')
    joblib.dump(features, 'features.pkl')
    
    print("✅ All models saved successfully!")
    print("📁 Files created:")
    print("   - model_xgb.pkl (XGBoost model)")
    print("   - label_encoders.pkl (Label encoders)") 
    print("   - result_mapping.pkl (Result mapping)")
    print("   - poisson_predictor.pkl (Poisson score predictor)")
    print("   - features.pkl (Feature list)")
    
    print(f"\n📊 Model Performance Summary:")
    print(f"   XGBoost Accuracy: {accuracy:.4f}")
    print(f"   Poisson MAE: {poisson_mae:.3f}")
    
    # Test prediction example
    print(f"\n🧪 Example Prediction:")
    if len(df_clean) > 0:
        sample_teams = df_clean['team'].unique()[:2]
        if len(sample_teams) >= 2:
            team1, team2 = sample_teams[0], sample_teams[1]
            pred_score, prob, top_scores = poisson_predictor.predict_score(team1, team2)
            print(f"   {team1} vs {team2}")
            print(f"   Predicted score: {pred_score[0]}-{pred_score[1]} (prob: {prob:.3f})")
            print(f"   Top 3 scores: {[f'{s[0]}-{s[1]} (p:{p:.3f})' for s, p in top_scores]}")

if __name__ == "__main__":
    train_complete_model()