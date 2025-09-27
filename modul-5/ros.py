import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('StressLevelDatasetFreeOutlier.csv')

x = df.drop(columns=['stress_level'])
y =df['stress_level']

print("--- Sebelum Random Oversampling ---")
print(y.value_counts())


ros = RandomOverSampler(random_state=42)
x_resampld, y_resampled = ros.fit_resample(x,y)

print("\n--- Setelah Random Oversampling ---")
print(pd.Series(y_resampled).value_counts())    

sns.countplot(x=y_resampled,palette='viridis').set_title('setelah random oversamling')
plt.show()
