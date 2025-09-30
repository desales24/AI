import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# 1. Load dataset
csv_path = '/root/semester5/coba_AI/persiapandata/football_data_cleaned.csv'
df = pd.read_csv(csv_path)

# 2. Preprocessing
# Ubah label hasil pertandingan ke numerik (0=Lose, 1=Draw, 2=Win)
df['result_numeric'] = df['result_numeric'].replace({3: 2})

categorical_cols = ['venue', 'comp', 'captain']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = ['gf','ga','xg','xga','poss','sh','sot','dist','fk','pk','pkatt','goal_difference','venue','comp','captain']
X = df[features]
y = df['result_numeric']

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# 5. Save model
joblib.dump(model, 'model_xgb.pkl')

# 6. Print accuracy

print(f"{model.score(X_test, y_test)*100:.2f}%")

# 7. Prediksi probabilitas pada data uji (dalam persen)
probs = model.predict_proba(X_test)
labels = ['Lose', 'Draw', 'Win']

