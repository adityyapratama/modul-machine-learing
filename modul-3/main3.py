import pandas as pd
import matplotlib.pyplot as plt

# Baca data
df = pd.read_csv('StressLevelDataset_capped.csv')

# chek missing value dulu boskuh
print(df.isnull().sum())


# Daftar kolom numerik (ganti jika ada kolom bertipe kategori)
kolom_numerik = [
    'anxiety_level', 'self_esteem', 'depression', 'headache', 'blood_pressure',
    'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions',
    'safety', 'basic_needs', 'academic_performance', 'study_load',
    'teacher_student_relationship', 'future_career_concerns', 'social_support',
    'peer_pressure', 'extracurricular_activities', 'bullying', 'stress_level'
]

for kolom in kolom_numerik:
    if kolom in df.columns:
        Q1 = df[kolom].quantile(0.25)
        Q3 = df[kolom].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        outlier = df[(df[kolom] < batas_bawah) | (df[kolom] > batas_atas)]
        print(f"Kolom: {kolom}, Jumlah outlier: {outlier.shape[0]}")
        plt.figure()
        plt.boxplot(df[kolom].dropna())
        plt.title(f'Boxplot {kolom}')
        plt.show()
        
 # Menyimpan data outlier kolom study_load ke file CSV
outlier_study_load = df[(df['study_load'] < df['study_load'].quantile(0.25) - 1.5 * (df['study_load'].quantile(0.75) - df['study_load'].quantile(0.25))) |
                        (df['study_load'] > df['study_load'].quantile(0.75) + 1.5 * (df['study_load'].quantile(0.75) - df['study_load'].quantile(0.25)))]
print("Outlier pada study_load telah disimpan di outlier_study_load.csv")       