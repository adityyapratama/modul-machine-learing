import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB



col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car_evaluation.csv',header=None, names=col_names)
print(df.head())


# encoding data ke numerik
encoders = {}
df_encoded = df.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le 
    
print("hasil encoding:")    
print(df_encoded.head())

# ekstrasi fitur
x = df_encoded.drop('class',axis=1)
y= df_encoded['class']


# split data latih dan data uji
X_train,X_test,Y_train, Y_test =train_test_split(x,y,test_size=0.2, random_state=42)

# 1. Menggabungkan X_train dan Y_train untuk ekspor
train_data_encoded = X_train.copy()
train_data_encoded['class'] = Y_train

# 2. Mengembalikan data latih ke label aslinya (agar mudah dibaca di Excel)
train_data_text = train_data_encoded.copy()
for col in train_data_text.columns:
    train_data_text[col] = encoders[col].inverse_transform(train_data_text[col])

# 3. Simpan ke file Excel baru
train_data_text.to_excel("data_latih_1382.xlsx", index=False)

print("\n--- INFO UNTUK EXCEL ---")
print("File 'data_latih_1382.xlsx' berhasil dibuat. Gunakan file ini untuk perhitungan manual!")

# 4. Cari tahu label teks untuk Data Uji ke-1 (Index 599)
x_coba_encoded = X_test.iloc[[0]] # Ini adalah data Index 599
x_coba_text = x_coba_encoded.copy()
for col in x_coba_text.columns:
    x_coba_text[col] = encoders[col].inverse_transform(x_coba_text[col])
    
print("\nData Uji yang akan dihitung manual (Index 599):")
print(x_coba_text.iloc[0])
print("--------------------------\n")

print(f"Ukuran data latih (X_train): {X_train.shape}")
print(f"Ukuran data uji (X_test): {X_test.shape}")

model = CategoricalNB()

model.fit(X_train, Y_train)

print("\nModel Naive Bayes berhasil dilatih")


x_coba = X_test.iloc[0:5]
y_asli = Y_test.iloc[0:5]
print("\nData uji:")
print(x_coba)
print(f"\nKelas Asli (y_asli): {y_asli}")


# melakukan prediksi
prediksi_encode = model.predict(x_coba)
probabilitas =model.predict_proba(x_coba)

hasil_prediksi_label = encoders['class'].inverse_transform(prediksi_encode)
label_asli_teks = encoders['class'].inverse_transform(y_asli)

print("\n--- HASIL PREDIKSI PROGRAM (untuk 5 data uji) ---")

# Loop sebanyak 5 kali (sesuai jumlah data di x_coba)
for idx in range(len(x_coba)): 
    pred_label = hasil_prediksi_label[idx]
    asli_label = label_asli_teks[idx]
    
    print(f"\n--- Data Uji ke-{idx+1} ---")
    print(f"Prediksi: '{pred_label}' | Asli: '{asli_label}'")
    
    if pred_label == asli_label:
        print("Status: BENAR ✅")
    else:
        print("Status: SALAH ❌")
    
    print("Probabilitas:")
    # Tampilkan probabilitas untuk data ke-idx
    for i, class_label in enumerate(encoders['class'].classes_):
        print(f"  - {class_label}: {probabilitas[idx][i]*100:.2f}%")