import pandas as pd
from scipy.stats import chi2_contingency


df =pd.read_csv('StressLevelDatasetFreeOutlier.csv')

# target dan fitur yg ingin saya test
target_variable = 'stress_level'
feature_to_test ='anxiety_level'

print("menguji hubungan antara  '{feature_to_test}' dan '{target_variable}'....")
print("-" * 50)
contingency_table =pd.crosstab(df[feature_to_test],df[target_variable])

print("table kontigensi:\n")
print(contingency_table)
print("-" *50)

# 2. Melakukan uji chi-square
chi2, p ,dof, expected = chi2_contingency(contingency_table)

#hasil
print(f"Nilai Chi-Square: {chi2}")
print(f"P-Value: {p}")


# kondisi untuk menentukan hubungan
alpha = 0.05
if p < alpha:
    print(f"\nHasil: Terdapat hubungan yang signifikan antara {feature_to_test} dan {target_variable}.")
else:
    print(f"\nHasil: Tidak terdapat hubungan yang signifikan antara {feature_to_test} dan {target_variable}.")

contingency_table.to_csv('hasil_seleksi_Chi_Square.csv')
print("File 'hasil_kontingensi.csv' berhasil disimpan!")


