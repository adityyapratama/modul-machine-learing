import pandas as pd
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Memuat dataset dari file CSV
df = pd.read_csv("StressLevelDatasetFreeOutlier.csv")
 
x = df.drop(columns=['stress_level'])
y = df['stress_level']


print("--- Sebelum SMOTE ---")
print(y.value_counts())

smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(x,y)

print("\n--- Setelah SMOTE ---")
print(pd.Series(y_resampled).value_counts())

sns.countplot(x=y_resampled,palette='viridis').set_title('setelah smote')
plt.show()

