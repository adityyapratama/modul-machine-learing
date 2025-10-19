import pandas as pd

df = pd.read_csv('StressLevelDataset_capped.csv')

kolom_numerik =[
    
    'anxiety_level', 'self_esteem', 'depression', 'headache', 'blood_pressure',
    'sleep_quality', 'breathing_problem', 'noise_level', 'living_conditions',
    'safety', 'basic_needs', 'academic_performance', 'study_load',
    'teacher_student_relationship', 'future_career_concerns', 'social_support',
    'peer_pressure', 'extracurricular_activities', 'bullying', 'stress_level'
    
]

# type of simple feature scaling
for col in kolom_numerik:
    df[col + '_simple'] = df[col]/df[col].max()

# type of Min-Max Normalization
for col in kolom_numerik:
    df[col+'_minmax'] = (df[col]-df[col].min()/df[col].max()-df[col].min())


# type of Z-Score Standardization
for col in kolom_numerik:
    df[col+'_zscore'] = (df[col]-df[col].mean())/df[col].std()

    
#  print hasil
print('Simple Feature Scaling:')
print(df[[col + '_simple' for col in kolom_numerik]].head())
print('\nMin-Max Normalization:')
print(df[[col + '_minmax' for col in kolom_numerik]].head())
print('\nZ-Score Standardization:')
print(df[[col + '_zscore' for col in kolom_numerik]].head())


# Menyimpan Simple Feature Scaling ke exel
df_simple = df[[col for col in kolom_numerik] + [col + '_simple' for col in kolom_numerik]]
df_simple.to_csv('StressLevelDataset_simple.csv', index=False)

# Menyimpan Min-Max Normalization ke exel
df_minmax = df[[col for col in kolom_numerik] + [col + '_minmax' for col in kolom_numerik]]
df_minmax.to_csv('StressLevelDataset_minmax.csv', index=False)

# Menyimpan Z-Score Standardization ke exel
df_zscore = df[[col for col in kolom_numerik] + [col + '_zscore' for col in kolom_numerik]]
df_zscore.to_csv('StressLevelDataset_zscore.csv', index=False)