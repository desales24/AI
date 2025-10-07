import pandas as pd

# Extract: Membaca file CSV
df = pd.read_csv('/root/semester5/coba_AI/data2/EnglandCSV.csv')

# Transform: Membersihkan data
# 1. Menghapus baris dengan data kosong
df_clean = df.dropna()

# 2. Menghapus data dengan Season = 2024/25
df_clean = df_clean[df_clean['Season'] != '2024/25']

# Load: Menyimpan hasil ETL ke file baru
df_clean.to_csv('EnglandCSV_cleaned.csv', index=False)

print("ETL selesai. Data yang sudah dibersihkan disimpan sebagai 'EnglandCSV_cleaned.csv'")
print(f"Jumlah baris sebelum ETL: {len(df)}")
print(f"Jumlah baris setelah ETL: {len(df_clean)}")