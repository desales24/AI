import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import joblib

def train_and_save_model():
    """Train model for result prediction AND score prediction"""
    print("Loading data...")
    df = pd.read_csv('EnglandCSV_cleaned.csv')
    
    # Features untuk prediksi hasil
    features = ['HomeTeam', 'AwayTeam', 'HTH Goals', 'HTA Goals', 'H Shots', 'A Shots', 
               'H SOT', 'A SOT', 'H Fouls', 'A Fouls', 'H Corners', 'A Corners', 
               'H Yellow', 'A Yellow', 'H Red', 'A Red']

    # Feature engineering
    df['Shot_Ratio_HT'] = df['HTH Goals'] / (df['HTA Goals'] + 1)
    df['Is_Home_Leading_HT'] = (df['HTH Goals'] > df['HTA Goals']).astype(int)
    df['Is_Away_Leading_HT'] = (df['HTH Goals'] < df['HTA Goals']).astype(int)
    df['Is_Draw_HT'] = (df['HTH Goals'] == df['HTA Goals']).astype(int)

    features.extend(['Shot_Ratio_HT', 'Is_Home_Leading_HT', 'Is_Away_Leading_HT', 'Is_Draw_HT'])

    # Preprocessing
    label_encoders = {}
    for column in ['HomeTeam', 'AwayTeam']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Encode target untuk hasil (H/D/A)
    target_encoder = LabelEncoder()
    df['FT Result_encoded'] = target_encoder.fit_transform(df['FT Result'])

    X = df[features]
    y_result = df['FT Result_encoded']  # Target untuk hasil
    y_home_goals = df['FTH Goals']      # Target untuk gol home
    y_away_goals = df['FTA Goals']      # Target untuk gol away

    # Split data
    X_train, X_test, y_train_result, y_test_result, y_train_home, y_test_home, y_train_away, y_test_away = train_test_split(
        X, y_result, y_home_goals, y_away_goals, test_size=0.2, random_state=42, stratify=y_result
    )

    # Train model untuk prediksi HASIL
    print("Training Random Forest untuk prediksi hasil...")
    rf_result_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_result_model.fit(X_train, y_train_result)

    # Train model untuk prediksi SKOR HOME
    print("Training Random Forest untuk prediksi skor home...")
    rf_home_goals_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_home_goals_model.fit(X_train, y_train_home)

    # Train model untuk prediksi SKOR AWAY
    print("Training Random Forest untuk prediksi skor away...")
    rf_away_goals_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_away_goals_model.fit(X_train, y_train_away)

    # Evaluate models
    y_pred_result = rf_result_model.predict(X_test)
    accuracy = accuracy_score(y_test_result, y_pred_result)
    
    y_pred_home = rf_home_goals_model.predict(X_test)
    y_pred_away = rf_away_goals_model.predict(X_test)
    
    mae_home = mean_absolute_error(y_test_home, y_pred_home)
    mae_away = mean_absolute_error(y_test_away, y_pred_away)

    print(f"✅ Model Accuracy (Hasil): {accuracy:.4f}")
    print(f"✅ MAE Skor Home: {mae_home:.4f}")
    print(f"✅ MAE Skor Away: {mae_away:.4f}")

    # Save model package dengan joblib
    print("Saving models with joblib...")
    
    model_package = {
        'result_model': rf_result_model,
        'home_goals_model': rf_home_goals_model,
        'away_goals_model': rf_away_goals_model,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'features': features,
        'model_info': {
            'accuracy': accuracy,
            'mae_home': mae_home,
            'mae_away': mae_away,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'Model dengan prediksi hasil dan skor'
        }
    }
    
    joblib.dump(model_package, 'football_predictor_with_score.joblib')
    print("✅ Model dengan prediksi skor saved successfully!")
    
    return model_package

if __name__ == "__main__":
    train_and_save_model()