import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

try:
    # --- Memuat dan Menyiapkan Data ---
    dataframe = pd.read_excel('BlaBla.xlsx') 
    kolom_fitur = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    kolom_target = 'N'
    X = dataframe[kolom_fitur]
    y = dataframe[kolom_target]
    print("="*25, "DATA AWAL", "="*25)
    print(dataframe.head())

    # --- Membagi Data Training dan Testing ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print("\nData berhasil dibagi menjadi 80% Training dan 20% Testing.")

    # ---  Membuat dan Melatih Model Decision Tree ---
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.fit(X_train, y_train)
    print("\nModel Decision Tree berhasil dilatih!")

    # --- MENYIMPAN MODEL ---
    # TARUH KODENYA DI SINI
    joblib.dump(decision_tree, 'model_decision_tree.pkl')
    print("Model berhasil disimpan sebagai 'model_decision_tree.pkl'")
    # -------------------------

    # --- Membuat Prediksi ---
    y_pred = decision_tree.predict(X_test)

    # --- mengevaluasi Performa Model ---
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "="*20, "HASIL EVALUASI MODEL", "="*20)
    print(f"Akurasi Model: {accuracy * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Visualisasi Decision Tree ---
    print("\nMembuat visualisasi pohon keputusan...")
    plt.figure(figsize=(20,10))
    plot_tree(decision_tree, 
              feature_names=kolom_fitur, 
              class_names=[str(c) for c in decision_tree.classes_], 
              filled=True, 
              rounded=True,
              fontsize=8)
    plt.title("Visualisasi Decision Tree")
    plt.show()

except FileNotFoundError:
    print("\nERROR: File 'BlaBla.xlsx' tidak ditemukan!")
    print("Pastikan nama file sudah benar dan berada di folder yang sama dengan skrip Python ini.")