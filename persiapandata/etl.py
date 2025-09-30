import pandas as pd

# 1️⃣ EXTRACT
file_path = "matches.csv"
df = pd.read_csv(file_path)

# 2️⃣ TRANSFORM
df['date'] = pd.to_datetime(df['date'], errors='coerce')
numeric_cols = ['gf','ga','xg','xga','poss','sh','sot','dist','fk','pk','pkatt']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
df['goal_difference'] = abs(df['gf'] - df['ga'])
df = df.drop(columns=['attendance', 'notes', 'match report'], errors='ignore')
df['result_numeric'] = df['result'].map({'W': 3, 'D': 1, 'L': 0})

# 3️⃣ LOAD
df.to_csv("football_data_cleaned.csv", index=False)
df.to_excel("football_data_cleaned.xlsx", index=False)

print(df.head())
