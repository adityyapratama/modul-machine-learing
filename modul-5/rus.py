import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('StressLevelDatasetFreeOutlier.csv')

x = df.drop(columns=['stress_level'])
y = df['stress_level']


print("--- Sebelum Random Undersampling ---")
print(y.value_counts())

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled =rus.fit_resample(x,y)

print("\n--- Setelah Random Undersampling ---")
print(pd.Series(y_resampled).value_counts())


sns.countplot(x=y_resampled,palette='viridis').set_title("setelah random sampling")
plt.show()


