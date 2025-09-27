import pandas as pd

# Baca data
df_all_data = pd.read_csv('StressLevelDataset.csv')

# Daftar kolom yang akan ditangani outlier-nya
columns_to_handle = ['noise_level', 'living_conditions', 'study_load']

# Buat salinan DataFrame untuk menyimpan data yang dimodifikasi
df_modified = df_all_data.copy()

# Iterasi melalui setiap kolom untuk menangani outlier
for column in columns_to_handle:
    Q1 = df_modified[column].quantile(0.25)
    Q3 = df_modified[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_before = df_modified[(df_modified[column] < lower_bound) | (df_modified[column] > upper_bound)].shape[0]
    print(f"Kolom: {column}, Jumlah outlier sebelum capping: {outliers_before}")

    # Terapkan capping (Winsorization)
    df_modified[column] = df_modified[column].clip(lower=lower_bound, upper=upper_bound)

    outliers_after = df_modified[(df_modified[column] < lower_bound) | (df_modified[column] > upper_bound)].shape[0]
    print(f"Kolom: {column}, Jumlah outlier setelah capping: {outliers_after}")
    print("-" * 50)

# Tampilkan beberapa baris pertama dari DataFrame yang telah dimodifikasi
print("\n5 baris pertama dari DataFrame setelah penanganan outlier:")
print(df_modified[columns_to_handle].head().to_string(index=False))