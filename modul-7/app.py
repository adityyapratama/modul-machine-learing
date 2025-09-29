import streamlit as st
import joblib
import pandas as pd

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Decision Tree",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Memuat Model ---
try:
    model = joblib.load('model_decision_tree.pkl')
except FileNotFoundError:
    st.error("File model 'model_decision_tree.pkl' tidak ditemukan.")
    st.stop()

# --- Header Aplikasi ---
st.title('🔬 Prediksi Klasifikasi Menggunakan Decision Tree')
st.markdown("---")
st.write("Aplikasi ini menggunakan model Decision Tree untuk memprediksi apakah seorang pasien **Positif** atau **Negatif** berdasarkan fitur-fitur yang ada.")

# --- Container untuk Input Fitur ---
st.header('📋 Input Fitur-Fitur')

def user_input_features():
    # Membuat 3 kolom untuk input yang lebih rapi
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        A = st.selectbox('Fitur A', (0, 1), key='A')
        C = st.selectbox('Fitur C', (0, 1), key='C')
        E = st.selectbox('Fitur E', (0, 1), key='E')
    
    with col2:
        B = st.selectbox('Fitur B', (0, 1), key='B')
        D = st.selectbox('Fitur D', (0, 1), key='D')
        F = st.selectbox('Fitur F', (0, 1), key='F')
    
    with col3:
        G = st.selectbox('Fitur G', (0, 1), key='G')
        I = st.selectbox('Fitur I', (0, 1), key='I')
        K = st.selectbox('Fitur K', (0, 1), key='K')
    
    with col4:
        H = st.selectbox('Fitur H', (0, 1), key='H')
        J = st.selectbox('Fitur J', (0, 1), key='J')
        L = st.selectbox('Fitur L', (0, 1), key='L')
    
    with col5:
        M = st.selectbox('Fitur M', (0, 1), key='M')
        # Tombol prediksi di kolom terakhir
        st.write("")  # Spacer
        predict_button = st.button('🔍 Lakukan Prediksi', type="primary", use_container_width=True)
    
    data = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 
            'G': G, 'H': H, 'I': I, 'J': J, 'K': K, 'L': L, 'M': M}
    
    features = pd.DataFrame(data, index=[0])
    return features, predict_button

# Mendapatkan input user dan status tombol
input_df, predict_button = user_input_features()

st.markdown("---")

# --- Layout Hasil ---
# Membuat 2 kolom utama
left_col, right_col = st.columns([1, 2])

# Kolom kiri untuk menampilkan input
with left_col:
    st.subheader('📊 Fitur yang Diinput')
    
    # Menampilkan input dalam format yang lebih menarik
    input_display = input_df.T.rename(columns={0: 'Nilai'})
    
    # Styling untuk tabel
    st.dataframe(
        input_display,
        use_container_width=True,
        height=400
    )
    
    # Menampilkan ringkasan
    total_positive = sum(input_df.iloc[0])
    st.info(f"📈 Total fitur bernilai 1: **{total_positive}/13**")

# Kolom kanan untuk hasil prediksi
with right_col:
    if predict_button:
        # Melakukan prediksi dari data input
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.subheader('🎯 Hasil Prediksi')
        
        # Menentukan hasil dan warna
        result = 'Positif' if prediction[0] == 1 else 'Negatif'
        
        # Container untuk hasil utama
        result_container = st.container()
        with result_container:
            if result == 'Positif':
                st.error(f"🔴 Pasien diprediksi: **{result}**")
            else:
                st.success(f"🟢 Pasien diprediksi: **{result}**")
        
        st.markdown("---")
        
        # Probabilitas dalam format yang lebih menarik
        st.subheader('📈 Probabilitas Prediksi')
        
        prob_negatif = prediction_proba[0][0] * 100
        prob_positif = prediction_proba[0][1] * 100
        
        # Progress bars untuk probabilitas
        st.write("**Probabilitas Negatif:**")
        st.progress(prob_negatif / 100)
        st.write(f"{prob_negatif:.2f}%")
        
        st.write("**Probabilitas Positif:**")
        st.progress(prob_positif / 100)
        st.write(f"{prob_positif:.2f}%")
        
        # Metrics dalam 2 kolom
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="🟢 Negatif", 
                value=f"{prob_negatif:.1f}%",
                delta=f"{prob_negatif - 50:.1f}%" if prob_negatif != 50 else None
            )
        with metric_col2:
            st.metric(
                label="🔴 Positif", 
                value=f"{prob_positif:.1f}%",
                delta=f"{prob_positif - 50:.1f}%" if prob_positif != 50 else None
            )
        
        # Interpretasi hasil
        st.markdown("---")
        st.subheader('💡 Interpretasi')
        
        confidence = max(prob_negatif, prob_positif)
        if confidence > 80:
            confidence_level = "Sangat Tinggi"
            confidence_color = "🔵"
        elif confidence > 60:
            confidence_level = "Tinggi"
            confidence_color = "🟡"
        else:
            confidence_level = "Rendah"
            confidence_color = "🟠"
        
        st.info(f"{confidence_color} Tingkat kepercayaan prediksi: **{confidence_level}** ({confidence:.1f}%)")
        
    else:
        # Tampilan default ketika belum ada prediksi
        st.subheader('🎯 Hasil Prediksi')
        st.info("👆 Silakan atur nilai fitur di atas dan klik tombol **'Lakukan Prediksi'** untuk melihat hasil.")
        
        # Placeholder untuk grafik atau informasi tambahan
        st.markdown("---")
        st.subheader('ℹ️ Informasi Model')
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Total Fitur", "13")
            st.metric("Model Type", "Decision Tree")
        
        with info_col2:
            st.metric("Output Classes", "2")
            st.metric("Binary Features", "0 atau 1")


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666;'>
        <p>🤖 Aplikasi Machine Learning - Decision Tree Classifier</p>
    </div>
    """, 
    unsafe_allow_html=True
)