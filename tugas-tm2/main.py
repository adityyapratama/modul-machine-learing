import pandas as pd

# Memuat dataset
data = pd.read_csv('StressLevelDataset.csv')


print("5 baris pertama dari dataset:")
print(data.head())
print("-" * 40) # Memberi garis pemisah agar rapi


print("Jumlah missing value per kolom:")
print(data.isnull().sum())
print("-" * 80)

# ✅ BENAR: Cara memilih beberapa kolom (gunakan kurung siku ganda)
multi_kolom = data[['mental_health_history', 'depression']]

# Menampilkan 5 baris terakhir dari dua kolom yang dipilih
# print("5 baris terakhir dari kolom 'mental_health_history' dan 'depression':")
# print(multi_kolom.tail())