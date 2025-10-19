import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# memanggil dataset
df=pd.read_csv('Netflix Dataset.csv')


# menampikan semua informasi di dataset yang telah di panggil
print("informasi data netlix tersebut adalah")
print(df)

# Menampilkan informasi dasar dataset
print("Informasi Dataset:")
print(df.info())
print("\n5 Baris Pertama Dataset:")
print(df.head())
print("\nStatistik Deskriptif:")
print(df.describe(include='all'))




# memerika missing value
missing_value= df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_value)

# menghitung missing value dalam bentuk presentasi
missing_value_percentage=(missing_value/len(df))*100
print("\nMissing Value Percentage in Each Column:")
print(missing_value_percentage)


# visualisasi missing value
plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap Missing Values dalam Dataset Netflix')
plt.savefig('netflix_missing_values.png')
plt.close()

print("jadi baris country adalah")
print(df['Country'].iloc[10:50])



# chek missing value di country
missing_country=df['Country'].isnull().sum()
print(f"Jumlah missing value di kolom 'country': {missing_country}")
# menggaanti missing value di country dengan modus
df_cleaned = df.copy()
# lihat modus di country
country_modus=df_cleaned['Country'].mode()[0]
print("\nModus Kolom Country:", country_modus)
# mengisi missing value di country dengan modus
df_cleaned['Country'] = df_cleaned['Country'].fillna(country_modus)


# Mengecek missing value setelah penggantian
missing_value_after_replace = df_cleaned.isna().sum()
print("\nJumlah Missing Values per Kolom Setelah Penggantian Country:")
print(missing_value_after_replace)





# # chek missing value di director

# chek missing value di director
missing_Director=df['Director'].isnull().sum()
print(f"Jumlah missing value di kolom 'Director': {missing_Director}")
# menggaanti missing value di director dengan modus
df_cleaned = df.copy()
# lihat modus di director
Director_modus=df_cleaned['Director'].mode()[0]
print("\nModus Kolom Country:", Director_modus)
# mengisi missing value di director dengan modus
df_cleaned['Director'] = df_cleaned['Country'].fillna(Director_modus)


# Mengecek missing value setelah penggantian
missing_value_after_replace2 = df_cleaned.isna().sum()
print("\nJumlah Missing Values per Kolom Setelah Penggantian Director:")
print(missing_value_after_replace2)