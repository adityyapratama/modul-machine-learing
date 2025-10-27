import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

# Abaikan warning
# warnings.filterwarnings('ignore')


df = pd.read_csv("bank.csv", sep=';')
X = df.drop('y', axis=1)
y = df['y']


X = pd.get_dummies(X, drop_first=True)


y = y.map({'no': 0, 'yes': 1})


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


kernels = ['linear', 'poly', 'rbf']
hasil_evaluasi = [] #simpan di list agar bisa dibuat tabel

print("Memulai evaluasi...\n")

for k in kernels:
    # --- PERBAIKAN 3: Tambahkan 'class_weight='balanced'' ---
    # Ini adalah bagian PALING PENTING.
    # Ini memberi tahu SVM untuk "lebih memperhatikan" data 'yes'
    # yang jumlahnya sedikit. Tanpa ini, model akan malas
    # dan cenderung menebak 'no' terus.
    svm = SVC(kernel=k, class_weight='balanced', random_state=42, gamma='auto')
    
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    # mengumpulkan hasil evaluasi
    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print(f"\n=== Kernel: {k.upper()} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Accuracy : {akurasi:.4f}")
    print(f"Precision: {presisi:.4f} (untuk kelas 'yes')")
    print(f"Recall   : {recall:.4f} (untuk kelas 'yes')")
    print(f"F1-Score : {f1:.4f} (untuk kelas 'yes')")
    
    hasil_evaluasi.append({
        "Kernel": k,
        "Akurasi": akurasi,
        "Presisi": presisi,
        "Recall": recall,
        "F1-Score": f1
    })

# Tabel Perbandingan 
print("     TABEL PERBANDINGAN HASIL")
print("=" * 40)
hasil_df = pd.DataFrame(hasil_evaluasi)
print(hasil_df.to_string(index=False))