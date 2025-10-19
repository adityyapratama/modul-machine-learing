import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
import tkinter as tk
from tkinter import ttk, messagebox

# --- LANGKAH 1: Latih Model (Sama seperti main.py) ---
# Kita perlu melatih model lagi di file ini agar UI-nya "pintar"

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car_evaluation.csv', header=None, names=col_names)

# Encoding data ke numerik
encoders = {}
df_encoded = df.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le  # Simpan encoder

# Ekstraksi fitur
x = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# Split data latih dan data uji (Gunakan data latih yang sama)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Latih model
model = CategoricalNB()
model.fit(X_train, Y_train)

print("Model dan Encoders berhasil dimuat untuk UI.")

# --- LANGKAH 2: Fungsi untuk Prediksi ---

def lakukan_prediksi():
    try:
        # 1. Ambil semua nilai teks dari dropdown
        input_teks = [
            combo_buying.get(),
            combo_maint.get(),
            combo_doors.get(),
            combo_persons.get(),
            combo_lug_boot.get(),
            combo_safety.get()
        ]

        # 2. Cek apakah semua sudah diisi
        if "" in input_teks:
            messagebox.showwarning("Input Tidak Lengkap", "Harap isi semua kolom!")
            return

        # 3. Ubah nilai teks menjadi angka menggunakan encoder
        input_angka = []
        input_angka.append(encoders['buying'].transform([input_teks[0]])[0])
        input_angka.append(encoders['maint'].transform([input_teks[1]])[0])
        input_angka.append(encoders['doors'].transform([input_teks[2]])[0])
        input_angka.append(encoders['persons'].transform([input_teks[3]])[0])
        input_angka.append(encoders['lug_boot'].transform([input_teks[4]])[0])
        input_angka.append(encoders['safety'].transform([input_teks[5]])[0])

        # 4. Lakukan prediksi
        # model.predict() butuh 2D array, jadi kita masukkan ke [ ]
        prediksi_angka = model.predict([input_angka])
        probabilitas = model.predict_proba([input_angka])

        # 5. Ubah hasil prediksi angka ke teks
        hasil_prediksi = encoders['class'].inverse_transform(prediksi_angka)[0]

        # 6. Format hasil probabilitas
        hasil_prob_teks = ""
        for i, class_label in enumerate(encoders['class'].classes_):
            hasil_prob_teks += f"- {class_label}: {probabilitas[0][i]*100:.2f}%\n"

        # 7. Tampilkan hasil di label
        label_hasil.config(text=f"Hasil Prediksi:\n{hasil_prediksi.upper()} ✅\n\nProbabilitas:\n{hasil_prob_teks}")

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")


# --- LANGKAH 3: Buat Jendela UI ---

root = tk.Tk()
root.title("Kalkulator Naive Bayes (Car Evaluation)")

# Buat frame utama
frame = ttk.Frame(root, padding="20")
frame.grid(row=0, column=0)

# Judul di dalam jendela
ttk.Label(frame, text="Masukkan Atribut Mobil:", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

# --- Membuat 6 Dropdown ---

# 1. Buying
ttk.Label(frame, text="Buying:").grid(row=1, column=0, sticky=tk.W, pady=5)
combo_buying = ttk.Combobox(frame, values=list(encoders['buying'].classes_))
combo_buying.grid(row=1, column=1)

# 2. Maint
ttk.Label(frame, text="Maint:").grid(row=2, column=0, sticky=tk.W, pady=5)
combo_maint = ttk.Combobox(frame, values=list(encoders['maint'].classes_))
combo_maint.grid(row=2, column=1)

# 3. Doors
ttk.Label(frame, text="Doors:").grid(row=3, column=0, sticky=tk.W, pady=5)
combo_doors = ttk.Combobox(frame, values=list(encoders['doors'].classes_))
combo_doors.grid(row=3, column=1)

# 4. Persons
ttk.Label(frame, text="Persons:").grid(row=4, column=0, sticky=tk.W, pady=5)
combo_persons = ttk.Combobox(frame, values=list(encoders['persons'].classes_))
combo_persons.grid(row=4, column=1)

# 5. Lug Boot
ttk.Label(frame, text="Lug Boot:").grid(row=5, column=0, sticky=tk.W, pady=5)
combo_lug_boot = ttk.Combobox(frame, values=list(encoders['lug_boot'].classes_))
combo_lug_boot.grid(row=5, column=1)

# 6. Safety
ttk.Label(frame, text="Safety:").grid(row=6, column=0, sticky=tk.W, pady=5)
combo_safety = ttk.Combobox(frame, values=list(encoders['safety'].classes_))
combo_safety.grid(row=6, column=1)


# --- Tombol Prediksi ---
tombol_prediksi = ttk.Button(frame, text="Prediksi Sekarang", command=lakukan_prediksi)
tombol_prediksi.grid(row=7, column=0, columnspan=2, pady=20)

# --- Label Hasil ---
label_hasil = ttk.Label(frame, text="Hasil akan muncul di sini", font=("Helvetica", 12))
label_hasil.grid(row=8, column=0, columnspan=2)

# --- Menjalankan UI ---
root.mainloop()