import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('StressLevelDataset.csv')

X=  df.drop('stress_level',axis=1)
y=df['stress_level']

# standarisasi data
scaler=StandardScaler()
x_scaled = scaler.fit_transform(X)

# Buat objek PCA dengan target varians 95%
# Ini akan secara otomatis memilih jumlah komponen yang diperlukan
pca_optimal =PCA(n_components=0.95)

# menerapkan PCA
X_pca_optimal =pca_optimal.fit_transform(x_scaled)

# melihat banyak components yg di pilih
num_components=pca_optimal.n_components_
print(f"PCA Optimal memilih {num_components} komponen untuk menangkap 95% varians.")

# Buat DataFrame dari hasil PCA optimal dan simpan
df_pca_optimal =pd.DataFrame(data=X_pca_optimal, columns=[f'PC_{i+1}' for i in range(num_components)])
df_pca_optimal['target']=y.values


df_pca_optimal.to_csv('hasil_PCA.csv', index=False)
print("File 'hasil_ekstraksi_pca_2_komponen.csv' berhasil disimpan!")