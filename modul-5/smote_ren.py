import pandas as pd
from imblearn.combine import SMOTEENN
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('StressLevelDatasetFreeOutlier.csv')

x= df.drop(columns=['stress_level'])
y = df['stress_level']

print("--- Sebelum SMOTE-ENN ---")
print(y.value_counts())


smote_enn = SMOTEENN(random_state = 42)
x_resampled, y_resampled= smote_enn.fit_resample(x,y)

print("\n--- Setelah SMOTE-ENN ---")
print(pd.Series(y_resampled).value_counts())


sns.countplot(x=y_resampled, palette='viridis').set_title("setelah SMOTE-ENN")
plt.show()
