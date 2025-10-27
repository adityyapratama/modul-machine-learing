import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')



@st.cache_data # Cache data mentah
def load_and_process_data(file_path):
    """Memuat, membersihkan, dan membagi data"""
    data = pd.read_csv(file_path, sep=';')
    X = pd.get_dummies(data.drop('y', axis=1), drop_first=True)
    y = data['y'].map({'no': 0, 'yes': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

@st.cache_data # Cache hasil model
def jalankan_evaluasi_model(kernel, X_train, y_train, X_test, y_test):
    """Melatih 1 model SVM dan mengembalikan metriknya"""
    model = SVC(kernel=kernel, class_weight='balanced', random_state=42, gamma='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    akurasi = accuracy_score(y_test, y_pred)
    presisi = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "Kernel": kernel,
        "Akurasi": akurasi,
        "Presisi": presisi,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": cm
    }

# --- TAMPILAN UTAMA APLIKASI ---
st.set_page_config(layout="wide")
st.title("Aplikasi SVM dengan Grafik Perbandingan ")
st.write("Menggunakan dataset `bank.csv`")

# Muat data (hanya sekali)
try:
    X_train, X_test, y_train, y_test = load_and_process_data('bank.csv')
except FileNotFoundError:
    st.error("Error: File 'bank.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    st.stop()


# --- SIDEBAR: PILIHAN PENGGUNA ---
st.sidebar.header("Opsi Tampilan")

# --- Evaluasi Tunggal ---
st.sidebar.subheader("Evaluasi Kernel Tunggal")
kernel_pilihan = st.sidebar.selectbox(
    "Pilih Kernel:",
    ('linear', 'rbf', 'poly')
)
if st.sidebar.button("Jalankan Evaluasi Tunggal"):
    st.header(f"Hasil untuk Kernel: `{kernel_pilihan}`")
    
    with st.spinner(f"Sedang melatih kernel {kernel_pilihan}..."):
        hasil = jalankan_evaluasi_model(kernel_pilihan, X_train, y_train, X_test, y_test)

    # Tampilkan Metrik
    col1, col2 = st.columns(2)
    col1.metric("Akurasi", f"{hasil['Akurasi']:.4f}")
    col1.metric("Presisi ('yes')", f"{hasil['Presisi']:.4f}")
    col2.metric("Recall ('yes')", f"{hasil['Recall']:.4f}")
    col2.metric("F1-Score ('yes')", f"{hasil['F1-Score']:.4f}")

    # Tampilkan Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(hasil['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prediksi No', 'Prediksi Yes'], 
                yticklabels=['Aktual No', 'Aktual Yes'])
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)

st.sidebar.divider()

# --- Grafik Perbandingan ---
st.sidebar.subheader("Perbandingan Semua Kernel")
if st.sidebar.button("📊 Tampilkan Grafik Perbandingan"):
    
    st.header("Perbandingan 3 Kernel SVM")
    
    with st.spinner("Mengevaluasi semua 3 kernel... (Poly mungkin lambat)"):
        # Jalankan evaluasi untuk semua kernel
        hasil_linear = jalankan_evaluasi_model('linear', X_train, y_train, X_test, y_test)
        hasil_poly = jalankan_evaluasi_model('poly', X_train, y_train, X_test, y_test)
        hasil_rbf = jalankan_evaluasi_model('rbf', X_train, y_train, X_test, y_test)
        
        # Kumpulkan hasil (tanpa confusion matrix)
        data_perbandingan = [
            {k: v for k, v in hasil.items() if k != 'Confusion Matrix'}
            for hasil in [hasil_linear, hasil_rbf, hasil_poly] # RBF ditengah agar grafiknya bagus
        ]
        
        # Buat DataFrame
        df_hasil = pd.DataFrame(data_perbandingan).set_index('Kernel')
    
    st.success("Evaluasi selesai!")
    
    # 1. Tampilkan Tabel Perbandingan
    st.subheader("Tabel Perbandingan Hasil (Fokus ke 'yes')")
    st.dataframe(df_hasil.style.highlight_max(axis=0, color='lightgreen'))
    
    # 2. Tampilkan Grafik Batang
    st.subheader("Grafik Perbandingan Metrik")
    st.bar_chart(df_hasil)
    
    st.markdown(f"""
    **Analisis Grafik:**
    * **Akurasi:** Kernel `{df_hasil['Akurasi'].idxmax()}` memiliki akurasi tertinggi.
    * **Presisi:** Kernel `{df_hasil['Presisi'].idxmax()}` paling akurat saat menebak 'yes'.
    * **Recall:** Kernel `{df_hasil['Recall'].idxmax()}` paling baik menemukan semua 'yes' yang asli.
    * **F1-Score:** Kernel `{df_hasil['F1-Score'].idxmax()}` memiliki performa paling seimbang.
    """)

# Pesan default di halaman utama
if not st.sidebar.button:
    st.info("Silakan pilih salah satu opsi di sidebar kiri.")