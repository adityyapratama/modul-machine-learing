import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Langkah 1: Memuat Dataset
df = pd.read_csv('Netflix Dataset.csv')

# Langkah 2: Memeriksa Missing Values
print("\nJumlah Missing Values per Kolom Sebelum Penanganan:")
print(df.isna().sum())

# Visualisasi missing values sebelum penanganan
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap Missing Values Sebelum Penanganan')
plt.savefig('netflix_missing_values_before.png')
plt.close()

# Langkah 3: Mengganti Missing Values dengan Modus
df_cleaned = df.copy()

# Mengganti missing values untuk kolom kategorikal
for col in ['Cast', 'Rating']:
    if df_cleaned[col].isna().sum() > 0:  # Hanya kolom dengan missing values
        modus = df_cleaned[col].mode()[0]
        df_cleaned[col].fillna(modus, inplace=True)
        print(f"\nModus untuk {col}: {modus}")

# Mengganti missing values untuk Release_Date (menggunakan Release_Year)
df_cleaned['Release_Date'] = pd.to_datetime(df_cleaned['Release_Date'], errors='coerce')
df_cleaned['Release_Year'] = df_cleaned['Release_Date'].dt.year
if df_cleaned['Release_Year'].isna().sum() > 0:
    year_modus = df_cleaned['Release_Year'].mode()[0]
    df_cleaned['Release_Year'].fillna(year_modus, inplace=True)
    print(f"\nModus untuk Release_Year: {year_modus}")

# Langkah 4: Verifikasi Missing Values Setelah Penanganan
print("\nJumlah Missing Values per Kolom Setelah Penanganan:")
print(df_cleaned.isna().sum())

# Visualisasi missing values setelah penanganan
plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap Missing Values Setelah Penanganan')
plt.savefig('netflix_missing_values_after.png')
plt.close()

# Langkah 5: Simpan Dataset yang Telah Diproses
df_cleaned.to_csv('netflix_cleaned_modus.csv', index=False)