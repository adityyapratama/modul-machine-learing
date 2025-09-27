import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Memuat dataset
data = pd.read_csv('test.csv')

print("jumlah missing value di semua data")
print(data.isnull().sum())

data_age= 'age' 

plt.figure(figsize=(10, 6)) # Mengatur ukuran gambar
sns.boxplot(x=data[data_age])
plt.title(f'Box Plot untuk Kolom {data_age}')
plt.xlabel('Nilai')
plt.show()