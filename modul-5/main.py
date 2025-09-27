import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("StressLevelDatasetFreeOutlier.csv")

target_coloumn = 'stress_level'

print("jumlah data untuk setiap colomn stress level")
value_counts = df[target_coloumn].value_counts()
print(value_counts)


plt.figure(figsize=(8, 6))
sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
plt.title('Distribusi Kelas Stress Level')
plt.xlabel('Stress Level (0=Rendah, 1=Sedang, 2=Tinggi)')
plt.ylabel('Jumlah Responden')
plt.show()