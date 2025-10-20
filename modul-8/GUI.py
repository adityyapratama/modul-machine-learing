import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# --- Model Training ---
df = pd.read_csv("emails.csv")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.text)
X_train, X_test, y_train, y_test = train_test_split(X, df.spam, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("predictions :", y_pred)
print("accuracy :", accuracy_score(y_test, y_pred))


def cek_spam():
    input_text = entry_text.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Peringatan", "Teks tidak boleh kosong!")
        return

    input_vector = vectorizer.transform([input_text])
    hasil = model.predict(input_vector)[0]

    if hasil == 1:
        label_hasil.config(text="🚨 Ini adalah SPAM", foreground="#e74c3c")  # Merah
    else:
        label_hasil.config(text="✅ Ini BUKAN Spam", foreground="#2ecc71")  # Hijau


# --- GUI Setup ---
root = tk.Tk()
root.title("Detektor Spam Email")
root.geometry("500x350")
root.configure(bg="#2c3e50")
root.resizable(False, False)

# Style
style = ttk.Style()
style.theme_use('clam')

# Configure styles
style.configure("TFrame", background="#2c3e50")
style.configure("TLabel", background="#2c3e50", foreground="#ecf0f1", font=("Segoe UI", 10))
style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
style.configure("Result.TLabel", font=("Segoe UI", 14, "bold"))
style.configure("TButton", font=("Segoe UI", 12, "bold"), borderwidth=0)
style.map("TButton",
          foreground=[('active', '#ecf0f1')],
          background=[('active', '#455a64'), ('!disabled', '#3498db')])

# Main frame
main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.pack(expand=True, fill="both")

# Widgets
label_judul = ttk.Label(main_frame, text="Detektor Spam Email", style="Title.TLabel")
label_judul.pack(pady=(0, 20))

entry_text = tk.Text(main_frame, height=6, width=50,
                     font=("Segoe UI", 10),
                     bg="#34495e", fg="#ecf0f1",
                     relief="flat",
                     insertbackground="#ecf0f1",
                     borderwidth=2,
                     highlightthickness=1,
                     highlightbackground="#3498db")
entry_text.pack(pady=10, padx=5)

btn_cek = ttk.Button(main_frame, text="Cek Teks", command=cek_spam, style="TButton", width=15)
btn_cek.pack(pady=15)

label_hasil = ttk.Label(main_frame, text="", style="Result.TLabel")
label_hasil.pack(pady=(10, 0))

root.mainloop()

