import pandas as pd
from scipy.stats import f_oneway

df = pd.read_csv('StressLevelDatasetFreeOutlier.csv')

# memisahkan fitur dari target
X = df.drop('stress_level', axis=1)
y = df['stress_level']

significant_features = []
alpha =0.05

# loop untuk setiap kolom fitur
for feature in X.columns:
    groups =[df[feature][y==level] for level in y.unique()]
    
    f_stats, p_value = f_oneway(*groups)
    
    print(f"menguji {feature}...P-value : {p_value:.4f}")
     
    
    
    if p_value < alpha :
        significant_features.append(feature)
        print(f"-> '{feature}' adalah fitur yang signifikan.\n")
        

print("\n" + "="*50)
print("Fitur signifikan berdasarkan ANOVA:")
print(significant_features)

df_anova_selection = df[significant_features + ['stress_level']]
df_anova_selection.to_csv('hasil_seleksi_anova.csv', index=False)
print("\nFile 'hasil_seleksi_anova.csv' berhasil disimpan!")
