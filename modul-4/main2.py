import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df =pd.read_csv('StressLevelDataset.csv')

# memisahkan fitur yg ingin digunakan sebagai target
x = df.drop('stress_level', axis=1)
y=df['stress_level']

# Melakukan PCA
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Melakukan PCA
# Kita akan meringkas semua fitur menjadi 2 komponen utama (fitur baru)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled) 
# bikin dataframe baru dari hasil PCA

df_pca=pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])

df_pca['target'] = y 
print("data setelah di PCA  (UNTUK 5 BARIS PERTAMA):\n")
print(df_pca.head())


print("n" +"-" *50)

print("Explained Variance Ratio (seberapa banyak info yang ditangkap per komponen):")
print(pca.explained_variance_ratio_)

total_variance =sum(pca.explained_variance_ratio_)
print(f"\nTotal Varians yang dijelaskan oleh 2 komponen: {total_variance:.2%}")




# visualisasi data
plt.figure(figsize=(10,8))
# Membuat scatter plot
# x-axis: Principal Component 1
# y-axis: Principal Component 2
# c=df_pca['target']: Memberi warna titik berdasarkan nilai target (stress_level)
# cmap='coolwarm': Skema warna yang digunakan (biru-merah)

scatter = plt.scatter(df_pca['Principal Component 1'], df_pca['Principal Component 2'], c=df_pca['target'], cmap='coolwarm', alpha=0.8)

# menambahkan color bar (legenda warna)
plt.colorbar(scatter, label='Stress Level')

plt.show()


df_pca.to_csv('hasil_ekstraksi_pca_2_komponen.csv', index=False)
print("File 'hasil_ekstraksi_pca_2_komponen.csv' berhasil disimpan!")